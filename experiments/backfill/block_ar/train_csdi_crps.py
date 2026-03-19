"""
Train CSDI-style attention denoiser with afCRPS fine-tuning (Exp 118b).

Phase 1: Load pretrained DDPM attention denoiser (118a)
Phase 2: Fine-tune with afCRPS on single-step x0 predictions

Each step: sample K noise vectors at t=T-1, predict x0 for each → K members.
Compute afCRPS between K members and GT. Backprop through single denoising step.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_csdi_crps.py \
        --pretrained models/backfill/csdi_proxy_118a/best_model.pt \
        --encoder_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --epochs 30 --batch_size 8 --n_members 8 --device cuda \
        --output_dir models/backfill/csdi_crps_118b
"""

import argparse
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
from diffusion.block_ar.single_pass_ar import afcrps_loss
from experiments.backfill.block_ar.train_csdi_proxy import cosine_beta_schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True, help="Path to 118a checkpoint")
    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_members", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--denoise_t", type=int, default=199, help="Timestep for x0 prediction (T-1 = near-pure noise)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    history_len, future_len = 30, 30
    train_end = 4040
    train_hist, train_fut = [], []
    for i in range(0, train_end - history_len - future_len + 1):
        train_hist.append(surfaces[i:i+history_len])
        train_fut.append(surfaces[i+history_len:i+history_len+future_len])

    train_hist = torch.tensor(np.array(train_hist), dtype=torch.float32)
    train_fut = torch.tensor(np.array(train_fut), dtype=torch.float32)
    # Normalize
    train_hist_norm = train_hist * 2 - 1
    # Keep GT in IV space [0, 1] for afCRPS
    train_fut_iv = train_fut  # (N, 30, 5, 5)
    # Also need normalized for denoiser input
    train_fut_norm = train_fut * 2 - 1
    train_fut_flat = train_fut_norm.reshape(-1, 25, 30)  # (N, K=25, L=30)

    dataset = TensorDataset(train_hist_norm, train_fut_iv, train_fut_flat)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Training windows: {len(dataset)}")

    # Load frozen encoder
    enc_ckpt = torch.load(args.encoder_path, weights_only=False)
    enc_config = EncoderConfig(input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1)
    encoder = GRUEncoder(enc_config).to(device).eval()
    enc_state = {k.replace("encoder.", ""): v for k, v in enc_ckpt["model_state_dict"].items()
                 if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())} params (frozen)")

    # Load pretrained denoiser
    ckpt = torch.load(args.pretrained, weights_only=False)
    model = AttentionDenoiser(
        n_cells=25, n_steps=30, channels=ckpt['config']['channels'],
        n_layers=ckpt['config']['n_layers'], n_heads=ckpt['config']['n_heads'],
        cond_dim=128, n_diffusion_steps=args.T,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train()
    print(f"Denoiser: {sum(p.numel() for p in model.parameters()):,} params (pretrained from 118a)")

    # Noise schedule
    betas = cosine_beta_schedule(args.T).to(device)
    alphas = 1 - betas
    alpha_bar = alphas.cumprod(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    t_denoise = args.denoise_t

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_crps = 0
        epoch_mae = 0
        epoch_spread = 0
        t0 = time.time()

        for hist_batch, gt_iv_batch, gt_flat_batch in loader:
            hist_batch = hist_batch.to(device)
            gt_iv_batch = gt_iv_batch.to(device)  # (B, 30, 5, 5) in [0, 1]
            B = hist_batch.shape[0]

            # Get condition
            with torch.no_grad():
                condition = encoder(hist_batch)  # (B, 128)

            # Generate K members via single-step x0 prediction
            ab = alpha_bar[t_denoise]
            t_batch = torch.full((B,), t_denoise, device=device, dtype=torch.long)

            members = []
            for _ in range(args.n_members):
                # Pure noise at t=T-1
                noise = torch.randn(B, 25, 30, device=device)
                # x_t = sqrt(ab) * x_0 + sqrt(1-ab) * noise ≈ noise when ab ≈ 0
                # For t=199 with cosine schedule, ab is very small → x_t ≈ noise
                x_t = ab.sqrt() * gt_flat_batch.to(device) + (1 - ab).sqrt() * noise

                # Predict noise → get x0
                noise_pred = model(x_t, condition, t_batch)
                x0_pred = (x_t - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
                x0_pred = x0_pred.clamp(-1, 1)

                # Denormalize to IV space [0, 1]
                iv_pred = (x0_pred + 1) / 2  # (B, 25, 30)
                iv_pred = iv_pred.clamp(0.001, 1.0)

                # Reshape to (B, 30, 5, 5)
                iv_pred = iv_pred.reshape(B, 5, 5, 30).permute(0, 3, 1, 2)
                members.append(iv_pred)

            iv_samples = torch.stack(members, dim=1)  # (B, K, 30, 5, 5)

            # Compute afCRPS
            crps, mae, spread = afcrps_loss(iv_samples, gt_iv_batch, alpha=0.95, reduction="frame_sum")
            loss = crps

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_crps += crps.item()
            epoch_mae += mae.item()
            epoch_spread += spread.item()

        scheduler.step()
        n_batches = len(loader)
        avg_loss = epoch_loss / n_batches
        avg_mae = epoch_mae / n_batches
        avg_spread = epoch_spread / n_batches
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  mae={avg_mae:.4f}  "
              f"spread={avg_spread:.4f}  s/m={avg_spread/max(avg_mae,1e-8):.4f}  ({elapsed:.1f}s)")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': ckpt['config'],
                'denoise_t': t_denoise,
            }, f"{args.output_dir}/best_model.pt")
            print(f"  → Saved best model (loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
