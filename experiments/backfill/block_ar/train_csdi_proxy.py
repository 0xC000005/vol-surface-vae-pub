"""
Train CSDI-style 2D attention denoiser with DDPM loss (Exp 118a).

Minimal training loop: add noise to GT future, predict noise, MSE loss.
Uses frozen GRU encoder for conditioning. Samples via DDIM.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_csdi_proxy.py \
        --encoder_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --epochs 30 --batch_size 16 --device cuda \
        --output_dir models/backfill/csdi_proxy_118a
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

from diffusion.block_ar.attention_denoiser import AttentionDenoiser
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from diffusion.block_ar.single_pass_ar import SinglePassConfig


def cosine_beta_schedule(T: int, s: float = 0.008):
    """Cosine noise schedule from Improved DDPM."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0.0001, 0.999).float()


def ddim_sample(model, encoder, history_norm, n_samples=50, n_steps=20,
                T=200, device='cuda'):
    """DDIM sampling from the attention denoiser."""
    B = history_norm.shape[0]
    K, L = 25, 30

    # Get condition from frozen encoder
    with torch.no_grad():
        condition = encoder(history_norm)  # (B, 128)

    # Noise schedule
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1 - betas
    alpha_bar = alphas.cumprod(0)

    # DDIM timestep subsequence
    step_size = T // n_steps
    timesteps = list(range(0, T, step_size))[::-1]

    all_samples = []
    for _ in range(n_samples):
        # Start from pure noise
        x = torch.randn(B, K, L, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = model(x, condition, t_batch)

            # DDIM update
            ab_t = alpha_bar[t]
            if i < len(timesteps) - 1:
                ab_prev = alpha_bar[timesteps[i + 1]]
            else:
                ab_prev = torch.tensor(1.0, device=device)

            x0_pred = (x - (1 - ab_t).sqrt() * noise_pred) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)  # clip to valid range

            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * noise_pred

        # Denormalize: [-1, 1] → [0, 1]
        sample = (x + 1) / 2
        sample = sample.clamp(0.001, 1.0).reshape(B, L, 5, 5)
        all_samples.append(sample)

    return torch.stack(all_samples, dim=1)  # (B, n_samples, L, 5, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=200, help="Diffusion steps")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)

    # Build training windows
    history_len, future_len = 30, 30
    train_end = 4040
    train_hist, train_fut = [], []
    for i in range(0, train_end - history_len - future_len + 1):
        train_hist.append(surfaces[i:i+history_len])
        train_fut.append(surfaces[i+history_len:i+history_len+future_len])

    train_hist = torch.tensor(np.array(train_hist), dtype=torch.float32)  # (N, 30, 5, 5)
    train_fut = torch.tensor(np.array(train_fut), dtype=torch.float32)   # (N, 30, 5, 5)
    print(f"Training windows: {train_hist.shape[0]}")

    # Normalize to [-1, 1]
    train_hist_norm = train_hist * 2 - 1
    train_fut_norm = train_fut * 2 - 1
    train_fut_flat = train_fut_norm.reshape(-1, 25, 30)  # (N, K=25, L=30)

    dataset = TensorDataset(train_hist_norm, train_fut_flat)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Load frozen encoder
    enc_ckpt = torch.load(args.encoder_path, weights_only=False)
    enc_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1
    )
    encoder = GRUEncoder(enc_config).to(device)
    # Load encoder weights from pretrained checkpoint
    enc_state = {k.replace("encoder.", ""): v for k, v in enc_ckpt["model_state_dict"].items()
                 if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Encoder loaded: {sum(p.numel() for p in encoder.parameters())} params (frozen)")

    # Build denoiser
    model = AttentionDenoiser(
        n_cells=25, n_steps=30, channels=args.channels,
        n_layers=args.n_layers, n_heads=args.n_heads,
        cond_dim=128, n_diffusion_steps=args.T,
    ).to(device)
    print(f"Denoiser: {sum(p.numel() for p in model.parameters()):,} params")

    # Noise schedule
    betas = cosine_beta_schedule(args.T).to(device)
    alphas = 1 - betas
    alpha_bar = alphas.cumprod(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for hist_batch, fut_batch in loader:
            hist_batch = hist_batch.to(device)
            fut_batch = fut_batch.to(device)  # (B, 25, 30)

            B = hist_batch.shape[0]

            # Get condition
            with torch.no_grad():
                condition = encoder(hist_batch)  # (B, 128)

            # Sample random timestep
            t = torch.randint(0, args.T, (B,), device=device)

            # Add noise
            ab = alpha_bar[t].view(B, 1, 1)  # (B, 1, 1)
            noise = torch.randn_like(fut_batch)
            noisy = ab.sqrt() * fut_batch + (1 - ab).sqrt() * noise

            # Predict noise
            noise_pred = model(noisy, condition, t)

            # MSE loss
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.6f}  ({elapsed:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': {
                    'channels': args.channels, 'n_layers': args.n_layers,
                    'n_heads': args.n_heads, 'T': args.T,
                },
            }, f"{args.output_dir}/best_model.pt")
            print(f"  → Saved best model (loss={best_loss:.6f})")

    print(f"\nTraining complete. Best loss={best_loss:.6f}")
    print(f"Model saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
