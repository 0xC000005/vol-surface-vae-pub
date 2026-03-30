#!/usr/bin/env python
"""
155c: ConditionalLayerNorm in Velocity Net + afCRPS (RC17-H3)

Adds CLN to the existing 153a factored velocity transformer. CLN modulates
LayerNorm activations via (scale(z)+1)*LN(x)+bias(z). Same noise z across
all 8 ODE steps per member. Train with afCRPS on K ensemble members.

The hypothesis: CLN noise in the velocity net produces diverse velocity fields
that integrate to calibrated spread through the ODE, while factored attention
preserves cross-cell correlation.

Literature: AIFS-CRPS (2412.15832), Anemoi source ConditionalLayerNorm

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_155c_cln_velocity.py \
        --base_model models/backfill/flow_153a/final_model.pt \
        --epochs 200 --batch_size 16 --n_members 4 --noise_dim 32 \
        --lr 5e-4 --output_dir models/backfill/flow_155c --device cuda
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
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp, kurtosis

import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    load_encoder, normalize_iv, make_serializable
)


class ConditionalLayerNorm(nn.Module):
    """CLN: y = (scale(z) + 1) * LayerNorm(x) + bias(z).

    Zero-init: at start, scale(z)≈0 and bias(z)≈0, so y≈LN(x).
    This is the exact AIFS-CRPS / Anemoi implementation.
    """

    def __init__(self, d_model, noise_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.scale_proj = nn.Linear(noise_dim, d_model)
        self.bias_proj = nn.Linear(noise_dim, d_model)
        # Zero-init so CLN starts as regular LayerNorm
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, x, z):
        """
        Args:
            x: (..., d_model) input
            z: (B, noise_dim) noise vector — will be broadcast
        Returns:
            (..., d_model) modulated output
        """
        h = self.norm(x)
        scale = self.scale_proj(z)  # (B, d_model)
        bias = self.bias_proj(z)    # (B, d_model)

        # Broadcast z to match x shape
        # x could be (B*C, T, d_model) or (B*T, C, d_model)
        # z is (B, d_model) — needs reshaping
        while scale.dim() < h.dim():
            scale = scale.unsqueeze(-2)
            bias = bias.unsqueeze(-2)

        return (scale + 1.0) * h + bias


class CLNFactoredVelocityTransformer(nn.Module):
    """Velocity net with factored attention + ConditionalLayerNorm.

    Same as ConditionalFactoredVelocityTransformer but each LayerNorm
    is replaced with CLN modulated by a global noise vector z.
    """

    def __init__(self, n_frames=30, n_cells=25, d_model=128, n_heads=4,
                 n_layers=4, time_dim=64, cond_dim=128, noise_dim=32):
        super().__init__()
        self.n_frames = n_frames
        self.n_cells = n_cells
        self.d_model = d_model
        self.time_dim = time_dim
        self.noise_dim = noise_dim

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.temporal_pos = nn.Parameter(torch.randn(1, n_frames, 1, d_model) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_cells, d_model) * 0.02)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        # Noise projection: noise_dim → noise_embed_dim for CLN
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.SiLU(), nn.Linear(d_model, noise_dim),
        )

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'temp_cln': ConditionalLayerNorm(d_model, noise_dim),
                'temp_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'temp_ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'temp_ff_cln': ConditionalLayerNorm(d_model, noise_dim),
                'spat_cln': ConditionalLayerNorm(d_model, noise_dim),
                'spat_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'spat_ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'spat_ff_cln': ConditionalLayerNorm(d_model, noise_dim),
            }))

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def time_embed(self, t):
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_t, t, cond=None, noise_z=None):
        """
        Args:
            x_t: (B, T*C) noisy state
            t: (B,) time in [0, 1]
            cond: (B, cond_dim) condition from encoder
            noise_z: (B, noise_dim) noise for CLN modulation
        """
        B = x_t.shape[0]
        T, C = self.n_frames, self.n_cells

        h = x_t.reshape(B, T, C, 1)
        h = self.input_proj(h)
        h = h + self.temporal_pos + self.spatial_pos
        t_emb = self.time_proj(self.time_embed(t))
        h = h + t_emb[:, None, None, :]

        if cond is not None:
            c_emb = self.cond_proj(cond)
            h = h + c_emb[:, None, None, :]

        # Project noise for CLN
        if noise_z is not None:
            z = self.noise_proj(noise_z)  # (B, noise_dim)
        else:
            z = torch.zeros(B, self.noise_dim, device=x_t.device)

        for layer in self.layers:
            # Temporal attention with CLN
            h_temp = h.permute(0, 2, 1, 3).reshape(B * C, T, -1)
            # Expand z for temporal: (B, noise_dim) → (B*C, noise_dim)
            z_temp = z.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
            h_norm = layer['temp_cln'](h_temp, z_temp)
            attn_out, _ = layer['temp_attn'](h_norm, h_norm, h_norm)
            h_temp = h_temp + attn_out
            h_temp = h_temp + layer['temp_ff'](layer['temp_ff_cln'](h_temp, z_temp))
            h = h_temp.reshape(B, C, T, -1).permute(0, 2, 1, 3)

            # Spatial attention with CLN
            h_spat = h.reshape(B * T, C, -1)
            z_spat = z.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            h_norm = layer['spat_cln'](h_spat, z_spat)
            attn_out, _ = layer['spat_attn'](h_norm, h_norm, h_norm)
            h_spat = h_spat + attn_out
            h_spat = h_spat + layer['spat_ff'](layer['spat_ff_cln'](h_spat, z_spat))
            h = h_spat.reshape(B, T, C, -1)

        h = self.output_norm(h)
        v = self.output_proj(h).squeeze(-1)
        return v.reshape(B, T * C)


def load_pretrained_into_cln(cln_model, pretrained_path, device):
    """Load 153a weights into CLN model, ignoring CLN-specific params."""
    ckpt = torch.load(pretrained_path, weights_only=False, map_location=device)
    pretrained_sd = ckpt['model_state_dict']

    # Map pretrained LayerNorm weights to CLN's internal norm
    mapping = {}
    for key in pretrained_sd:
        if 'temp_norm' in key:
            new_key = key.replace('temp_norm', 'temp_cln.norm')
            mapping[new_key] = pretrained_sd[key]
        elif 'temp_ff_norm' in key:
            new_key = key.replace('temp_ff_norm', 'temp_ff_cln.norm')
            mapping[new_key] = pretrained_sd[key]
        elif 'spat_norm' in key:
            new_key = key.replace('spat_norm', 'spat_cln.norm')
            mapping[new_key] = pretrained_sd[key]
        elif 'spat_ff_norm' in key:
            new_key = key.replace('spat_ff_norm', 'spat_ff_cln.norm')
            mapping[new_key] = pretrained_sd[key]
        else:
            mapping[key] = pretrained_sd[key]

    # Load what we can, skip CLN projection weights (they're zero-init)
    model_sd = cln_model.state_dict()
    loaded = 0
    for key, val in mapping.items():
        if key in model_sd and model_sd[key].shape == val.shape:
            model_sd[key] = val
            loaded += 1

    cln_model.load_state_dict(model_sd)
    total = len(model_sd)
    print(f"  Loaded {loaded}/{total} params from pretrained ({total - loaded} new CLN params)")
    return ckpt


def afcrps_loss(samples, gt, alpha=0.95):
    """afCRPS for (B, K, D) tensors."""
    K = samples.shape[1]
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)
    mae = (samples - gt.unsqueeze(1)).abs().mean()
    spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()
    fcrps = mae - 0.5 * spread
    loss = alpha * fcrps + (1 - alpha) * mae
    return loss, mae, spread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="models/backfill/flow_153a/final_model.pt")
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--n_members", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load encoder
    encoder, cond_dim = load_encoder(args.encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False

    # Create CLN velocity model and load pretrained weights
    ckpt = torch.load(args.base_model, weights_only=False, map_location=device)
    cfg = ckpt['config']
    model = CLNFactoredVelocityTransformer(
        n_frames=cfg['n_frames'], n_cells=cfg['n_cells'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
        cond_dim=cfg.get('cond_dim', 128), noise_dim=args.noise_dim,
    ).to(device)

    ckpt = load_pretrained_into_cln(model, args.base_model, device)
    train_mean = torch.from_numpy(ckpt['train_mean']).float().to(device)
    train_std = torch.from_numpy(ckpt['train_std']).float().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_cln = sum(p.numel() for p in model.parameters() if any(
        n in name for name, p2 in model.named_parameters()
        if p2 is p for n in ['scale_proj', 'bias_proj', 'noise_proj']))
    print(f"\n{'='*60}")
    print(f"155c: CLN Velocity Transformer + afCRPS (RC17-H3)")
    print(f"{'='*60}")
    print(f"  Total params: {n_params:,}")
    print(f"  K members: {args.n_members}, noise_dim: {args.noise_dim}")
    print(f"  ODE steps: {args.n_steps}")

    # Data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, T, DIM = 30, 30, 750

    # Training data: sliding windows on validation set
    train_start, train_end = 4040, 4540
    val_start, val_end = 3540, 4040
    train_windows = [(surfaces[i:i+H], surfaces[i+H:i+H+T])
                     for i in range(train_start, train_end - H - T + 1)]
    val_windows = [(surfaces[i:i+H], surfaces[i+H:i+H+T])
                   for i in range(val_start, val_end - H - T + 1)]

    print(f"  Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []
    dt = 1.0 / args.n_steps

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0; epoch_mae = 0; epoch_spread = 0; nb = 0

        # Shuffle training windows
        indices = np.random.permutation(len(train_windows))

        for batch_start in range(0, len(indices), args.batch_size):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            B = len(batch_idx)
            K = args.n_members

            # Load history and future
            hist_list = [train_windows[i][0] for i in batch_idx]
            fut_list = [train_windows[i][1] for i in batch_idx]
            hist = torch.from_numpy(np.array(hist_list, dtype=np.float32)).to(device)
            future = torch.from_numpy(np.array(fut_list, dtype=np.float32)).to(device)

            # Encode condition
            with torch.no_grad():
                cond = encoder(normalize_iv(hist))  # (B, 128)

            # Generate K members via ODE with different noise vectors
            gt_flat = future.reshape(B, DIM)  # (B, 750)

            # Fold K into batch dim for efficient ODE
            cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            noise_z = torch.randn(B * K, args.noise_dim, device=device)

            # ODE integration with gradient checkpointing
            x = torch.randn(B * K, DIM, device=device)
            for step in range(args.n_steps):
                t_step = torch.full((B * K,), step * dt, device=device)
                def ode_step(x_in, t_in, c_in, z_in):
                    return x_in + model(x_in, t_in, cond=c_in, noise_z=z_in) * dt
                x = torch.utils.checkpoint.checkpoint(
                    ode_step, x, t_step, cond_K, noise_z, use_reentrant=False)

            # Denormalize
            predictions = (x * train_std + train_mean).clamp(0, 1)  # (B*K, 750)
            predictions = predictions.reshape(B, K, DIM)

            # afCRPS loss
            crps, mae, spread = afcrps_loss(predictions, gt_flat, alpha=args.alpha)
            loss = crps

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += mae.item()
            epoch_spread += spread.item()
            nb += 1

        scheduler.step()
        train_loss = epoch_loss / nb
        train_mae = epoch_mae / nb
        train_spread = epoch_spread / nb
        elapsed = time.time() - t0

        # Validation
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for vi in range(0, len(val_windows), args.batch_size):
                vb = val_windows[vi:vi + args.batch_size]
                B = len(vb)
                K = args.n_members
                hist = torch.from_numpy(np.array([w[0] for w in vb], dtype=np.float32)).to(device)
                future = torch.from_numpy(np.array([w[1] for w in vb], dtype=np.float32)).to(device)
                cond = encoder(normalize_iv(hist))
                gt_flat = future.reshape(B, DIM)

                cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
                noise_z = torch.randn(B * K, args.noise_dim, device=device)
                x = torch.randn(B * K, DIM, device=device)
                for step in range(args.n_steps):
                    t_step = torch.full((B * K,), step * dt, device=device)
                    x = x + model(x, t_step, cond=cond_K, noise_z=noise_z) * dt
                preds = (x * train_std + train_mean).clamp(0, 1).reshape(B, K, DIM)
                crps, _, _ = afcrps_loss(preds, gt_flat, alpha=args.alpha)
                vl += crps.item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "train_mean": ckpt['train_mean'],
                "train_std": ckpt['train_std'],
                "config": {
                    "n_frames": cfg['n_frames'], "n_cells": cfg['n_cells'],
                    "d_model": cfg['d_model'], "n_heads": cfg['n_heads'],
                    "n_layers": cfg['n_layers'], "cond_dim": cfg.get('cond_dim', 128),
                    "noise_dim": args.noise_dim, "n_steps": args.n_steps,
                    "type": "cln_velocity_afcrps",
                },
            }, f"{args.output_dir}/best_model.pt")

        # Full eval every 40 epochs
        if epoch % 40 == 0 or epoch == 1 or epoch == args.epochs:
            # Quick eval: CI, KS, corr on val windows
            model.eval()
            n_eval = min(160, len(val_windows))
            ns = 50
            all_samps = []; all_gt = []
            with torch.no_grad():
                for ei in range(n_eval):
                    h = torch.from_numpy(val_windows[ei][0][None].astype(np.float32)).to(device)
                    c = encoder(normalize_iv(h))  # (1, 128)
                    c_K = c.expand(ns, -1)
                    nz = torch.randn(ns, args.noise_dim, device=device)
                    x = torch.randn(ns, DIM, device=device)
                    for step in range(args.n_steps):
                        t_step = torch.full((ns,), step * dt, device=device)
                        x = x + model(x, t_step, cond=c_K, noise_z=nz) * dt
                    pred = (x * train_std + train_mean).clamp(0, 1).cpu().numpy()
                    all_samps.append(pred.reshape(ns, T, 25))
                    all_gt.append(val_windows[ei][1].reshape(T, 25))
            samps = np.array(all_samps)  # (N, K, T, C)
            gts = np.array(all_gt)       # (N, T, C)
            N, C = len(samps), 25

            # CI worst cell
            worst_ci = 1.0
            for c in range(C):
                lo = np.percentile(samps[:, :, :, c], 5, axis=1)
                hi = np.percentile(samps[:, :, :, c], 95, axis=1)
                cov = ((gts[:, :, c] >= lo) & (gts[:, :, c] <= hi)).mean()
                worst_ci = min(worst_ci, cov)

            # KS, kurtosis, correlation
            gch = np.diff(samps[:, 0], axis=1).reshape(-1, C)
            gtch = np.diff(gts, axis=1).reshape(-1, C)
            ks = sum(1 for c2 in range(C) if ks_2samp(gch[:, c2], gtch[:, c2])[0] < 0.15)
            kr = kurtosis(gch.flatten()) / (kurtosis(gtch.flatten()) + 1e-6)
            gc = np.corrcoef(gch.T); gtc = np.corrcoef(gtch.T)
            corr = np.abs(gc).mean() / (np.abs(gtc).mean() + 1e-6)
            ss = samps.std(axis=1).mean() / (np.abs(samps.mean(axis=1) - gts).mean() + 1e-8)

            # CLN scale analysis
            scale_norms = []
            for layer in model.layers:
                for name in ['temp_cln', 'temp_ff_cln', 'spat_cln', 'spat_ff_cln']:
                    w = layer[name].scale_proj.weight.detach()
                    scale_norms.append(w.norm().item())
            avg_scale = np.mean(scale_norms)

            print(f"Ep {epoch:3d}  loss={train_loss:.4f}  val={val_loss:.4f}  "
                  f"mae={train_mae:.4f}  spread={train_spread:.4f}  "
                  f"({elapsed:.1f}s)")
            print(f"  -> CI worst={worst_ci:.3f}  KS={ks}/25  kurt={kr:.3f}  "
                  f"corr={corr:.3f}  SS={ss:.3f}  CLN_scale={avg_scale:.4f}")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, "ci_worst": float(worst_ci),
                           "ks": ks, "kurt": float(kr), "corr": float(corr),
                           "ss": float(ss), "cln_scale": float(avg_scale)})
        else:
            print(f"Ep {epoch:3d}  loss={train_loss:.4f}  val={val_loss:.4f}  "
                  f"mae={train_mae:.4f}  spread={train_spread:.4f}  "
                  f"({elapsed:.1f}s)")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss})

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss, "train_mean": ckpt['train_mean'],
        "train_std": ckpt['train_std'],
        "config": {
            "n_frames": cfg['n_frames'], "n_cells": cfg['n_cells'],
            "d_model": cfg['d_model'], "n_heads": cfg['n_heads'],
            "n_layers": cfg['n_layers'], "cond_dim": cfg.get('cond_dim', 128),
            "noise_dim": args.noise_dim, "n_steps": args.n_steps,
            "type": "cln_velocity_afcrps",
        },
    }, f"{args.output_dir}/final_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
