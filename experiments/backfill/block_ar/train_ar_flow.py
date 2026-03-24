#!/usr/bin/env python
"""
H2 Stage 2: Conditional AR Flow Matching

Autoregressive flow matching for multi-horizon IV surface forecasting.
Each frame is generated via CFM conditioned on GRU encoder output + previous frame.

Architecture:
  - Frozen GRU encoder (from 146b) → 128-dim condition
  - ConditionalVelocityMLP: (x_t, t, condition, prev_frame) → velocity
  - AR loop: 30 frames, 8-step Euler ODE per frame
  - Training: per-frame CFM loss on (history, future) windows

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_ar_flow.py \
        --encoder_path models/backfill/afcrps_146b/best_model.pt \
        --epochs 100 --batch_size 64 --lr 1e-3 \
        --output_dir models/backfill/flow_152b --device cuda
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig, normalize_iv, denormalize_iv,
)
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


class ConditionalVelocityMLP(nn.Module):
    """Velocity network conditioned on encoder output + previous frame.

    Input: (x_t [25], t_emb [64], condition [128], prev_frame [25]) = 242-dim
    Output: velocity [25]
    """

    def __init__(self, frame_dim=25, cond_dim=128, hidden=256, n_layers=4):
        super().__init__()
        self.time_dim = 64
        self.frame_dim = frame_dim

        in_dim = frame_dim + self.time_dim + cond_dim + frame_dim  # x_t + t + cond + prev
        layers = []
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else frame_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        # Zero-init output
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def time_embed(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_t, t, condition, prev_frame):
        """
        Args:
            x_t: (B, 25) noisy frame at time t
            t: (B,) time in [0, 1]
            condition: (B, 128) encoder output
            prev_frame: (B, 25) previous frame (standardized)
        Returns:
            v: (B, 25) predicted velocity
        """
        t_emb = self.time_embed(t)
        inp = torch.cat([x_t, t_emb, condition, prev_frame], dim=-1)
        return self.net(inp)


class ARFlowMatchingModel(nn.Module):
    """Autoregressive flow matching for IV surface forecasting.

    Uses a frozen GRU encoder for conditioning and a ConditionalVelocityMLP
    for per-frame generation via ODE.
    """

    def __init__(self, encoder, velocity_net, frame_mean, frame_std,
                 n_steps=8, future_len=30):
        super().__init__()
        self.encoder = encoder
        self.velocity_net = velocity_net
        self.register_buffer("frame_mean", torch.from_numpy(frame_mean).float())
        self.register_buffer("frame_std", torch.from_numpy(frame_std).float())
        self.n_steps = n_steps
        self.future_len = future_len

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def standardize(self, x):
        """Standardize IV frames to zero mean, unit variance."""
        return (x - self.frame_mean) / self.frame_std

    def destandardize(self, x):
        """Reverse standardization."""
        return x * self.frame_std + self.frame_mean

    def cfm_loss(self, history, future):
        """Compute per-frame CFM loss over all future frames.

        Args:
            history: (B, H, 5, 5) normalized [-1,1]
            future: (B, T, 5, 5) normalized [-1,1]
        Returns:
            loss: scalar
        """
        B, T = future.shape[0], future.shape[1]
        device = history.device

        # Encode history
        with torch.no_grad():
            condition = self.encoder(history)  # (B, 128)

        # Denormalize future to [0,1] then standardize for CFM
        future_iv = denormalize_iv(future)  # (B, T, 5, 5) in [0,1]
        future_flat = future_iv.reshape(B, T, 25)  # (B, T, 25)
        future_std = self.standardize(future_flat)  # standardized

        # Previous frames: history[-1] for t=0, then future[t-1] for t>0
        history_iv = denormalize_iv(history)
        prev_frames = torch.cat([
            history_iv[:, -1:].reshape(B, 1, 25),  # last history frame
            future_flat[:, :-1],  # future[0..T-2]
        ], dim=1)  # (B, T, 25)
        prev_std = self.standardize(prev_frames)

        # Sample random time for each (batch, frame)
        t = torch.rand(B, T, device=device)

        # CFM: x0 ~ N(0,I), x1 = future_std
        x0 = torch.randn_like(future_std)
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * future_std  # (B, T, 25)
        u_t = future_std - x0  # target velocity

        # Flatten batch and time for velocity network
        x_t_flat = x_t.reshape(B * T, 25)
        t_flat = t.reshape(B * T)
        cond_flat = condition.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        prev_flat = prev_std.reshape(B * T, 25)

        # Predict velocity
        v_t = self.velocity_net(x_t_flat, t_flat, cond_flat, prev_flat)
        u_t_flat = u_t.reshape(B * T, 25)

        return F.mse_loss(v_t, u_t_flat)

    @torch.no_grad()
    def sample(self, history, n_samples=50, noise_sigma=0.0):
        """Generate n_samples future paths via AR ODE.

        Args:
            history: (B, H, 5, 5) normalized [-1,1]
            n_samples: number of ensemble members
            noise_sigma: post-ODE noise (in standardized space). 0=deterministic ODE.
        Returns:
            samples: (B, n_samples, T, 5, 5) in [0,1]
        """
        B = history.shape[0]
        device = history.device
        dt = 1.0 / self.n_steps

        # Encode — detect zero history (unconditional) and use null_embedding
        is_zero = (history.abs().sum(dim=(1, 2, 3)) < 1e-6)  # (B,)
        condition = self.encoder(history)  # (B, 128)
        if is_zero.any() and hasattr(self.encoder, 'null_embedding'):
            null_cond = self.encoder.null_embedding.expand(B, -1)
            condition = torch.where(is_zero.unsqueeze(-1), null_cond, condition)

        # Expand for n_samples: (B*S, ...)
        BS = B * n_samples
        cond = condition.unsqueeze(1).expand(B, n_samples, -1).reshape(BS, -1)

        # Initial previous frame: last history frame
        history_iv = denormalize_iv(history)  # (B, H, 5, 5)
        prev = history_iv[:, -1].reshape(B, 25)  # (B, 25)
        prev = prev.unsqueeze(1).expand(B, n_samples, 25).reshape(BS, 25)
        prev_std = self.standardize(prev)

        all_frames = []

        for frame_idx in range(self.future_len):
            # ODE: integrate from t=0 (noise) to t=1 (data)
            x = torch.randn(BS, 25, device=device)

            for step in range(self.n_steps):
                t = torch.full((BS,), step * dt, device=device)
                v = self.velocity_net(x, t, cond, prev_std)
                x = x + v * dt

            # Add post-ODE noise for ensemble spread (SDE-like)
            if noise_sigma > 0:
                x = x + noise_sigma * torch.randn_like(x)

            # x is now in standardized space → destandardize → clamp to [0,1]
            frame_iv = self.destandardize(x).clamp(0, 1)  # (BS, 25)
            all_frames.append(frame_iv)

            # Update previous frame
            prev_std = self.standardize(frame_iv)

        # Stack: (BS, T, 25) → (B, S, T, 5, 5)
        frames = torch.stack(all_frames, dim=1)  # (BS, T, 25)
        frames = frames.reshape(B, n_samples, self.future_len, 5, 5)

        return frames

    def sample_batched(self, history, n_samples=50, noise_sigma=0.0, **kwargs):
        """Compatible with test_block_ar_requirements_v2.py interface."""
        return self.sample(history, n_samples=n_samples, noise_sigma=noise_sigma)


def main():
    parser = argparse.ArgumentParser(description="H2 Stage 2: Conditional AR Flow Matching")
    parser.add_argument("--encoder_path", type=str, required=True,
                        help="Path to model with GRU encoder (e.g., 146b)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future_len", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H2 Stage 2: Conditional AR Flow Matching")
    print("=" * 60)

    # Load encoder from existing model
    print(f"\nLoading encoder from {args.encoder_path}...")
    ckpt = torch.load(args.encoder_path, weights_only=False, map_location=device)
    sp_cfg = {k: v for k, v in ckpt["config"].items()
              if k in SinglePassConfig.__dataclass_fields__}
    sp_config = SinglePassConfig(**sp_cfg)
    base_model = SinglePassBlockAR(sp_config)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    encoder = base_model.encoder
    encoder.eval()
    cond_dim = sp_config.bottleneck_dim
    print(f"  Encoder: bottleneck_dim={cond_dim}")

    # Load data
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]

    # Compute standardization stats from training frames
    train_frames = surfaces[:config.test_start].reshape(-1, 25).astype(np.float32)
    frame_mean = train_frames.mean(axis=0, keepdims=True)  # (1, 25)
    frame_std = train_frames.std(axis=0, keepdims=True) + 1e-6

    # Create model
    velocity_net = ConditionalVelocityMLP(
        frame_dim=25, cond_dim=cond_dim,
        hidden=args.hidden, n_layers=args.n_layers,
    )
    model = ARFlowMatchingModel(
        encoder=encoder, velocity_net=velocity_net,
        frame_mean=frame_mean, frame_std=frame_std,
        n_steps=args.n_steps, future_len=args.future_len,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Velocity net params: {n_trainable:,} trainable / {n_total:,} total")

    # Data loaders
    returns = data["ret"] if "ret" in data else None
    extra_features = getattr(sp_config, "extra_features", 0)
    return_scale = getattr(sp_config, "return_scale", 0.05)
    model_returns = returns if extra_features > 0 else None

    # Train: indices 0 to 4040, Val: 4040 to 4540 (same split as afCRPS)
    train_ds = VolSurfaceDataset(
        surfaces, config.history_len, args.future_len,
        start_idx=0, end_idx=4040,
        returns=model_returns, return_scale=return_scale,
    )
    val_ds = VolSurfaceDataset(
        surfaces, config.history_len, args.future_len,
        start_idx=4040, end_idx=4540,
        returns=model_returns, return_scale=return_scale,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"  Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    optimizer = torch.optim.Adam(velocity_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        velocity_net.train()
        encoder.eval()  # always eval

        epoch_loss = 0
        n_batches = 0
        for batch in train_loader:
            hist = batch["history"].to(device)
            future = batch["future"].to(device)

            loss = model.cfm_loss(hist, future)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(velocity_net.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                hist = batch["history"].to(device)
                future = batch["future"].to(device)
                loss = model.cfm_loss(hist, future)
                val_loss += loss.item() * hist.shape[0]
                n_val += hist.shape[0]
        val_loss = val_loss / max(n_val, 1)

        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "velocity_state_dict": velocity_net.state_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "frame_mean": frame_mean,
                "frame_std": frame_std,
                "config": {
                    "hidden": args.hidden, "n_layers": args.n_layers,
                    "n_steps": args.n_steps, "frame_dim": 25,
                    "cond_dim": cond_dim, "future_len": args.future_len,
                    "encoder_path": args.encoder_path,
                },
                "encoder_config": ckpt["config"],
            }, f"{args.output_dir}/best_model.pt")

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  ({elapsed:.1f}s)"
              + (f"  *best" if val_loss <= best_val_loss else ""))

    # Save final
    torch.save({
        "velocity_state_dict": velocity_net.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "epoch": args.epochs,
        "val_loss": val_loss,
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "config": {
            "hidden": args.hidden, "n_layers": args.n_layers,
            "n_steps": args.n_steps, "frame_dim": 25,
            "cond_dim": cond_dim, "future_len": args.future_len,
            "encoder_path": args.encoder_path,
        },
        "encoder_config": ckpt["config"],
    }, f"{args.output_dir}/final_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
