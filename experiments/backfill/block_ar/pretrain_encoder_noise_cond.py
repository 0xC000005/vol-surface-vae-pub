"""Pretrain GRU encoder with noise-conditioned prediction.

The prediction head receives BOTH condition AND noise z, forcing the encoder
to leave room for noise in the prediction. A diversity loss penalizes the
system if it ignores z.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/pretrain_encoder_noise_cond.py \
        --epochs 30 --output_dir models/backfill/gru_encoder_noise_cond --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig


class NextFrameDataset(Dataset):
    """Fixed 30-frame history → next frame."""

    def __init__(self, surfaces, start_idx=0, end_idx=None):
        end_idx = end_idx or len(surfaces)
        self.surfaces = surfaces[start_idx:end_idx]
        self.history_len = 30
        self.n_samples = len(self.surfaces) - self.history_len
        print(f"NextFrameDataset: {self.n_samples} samples from indices {start_idx}:{end_idx}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        history = self.surfaces[idx:idx + self.history_len]
        target = self.surfaces[idx + self.history_len]
        return (
            torch.tensor(history, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def main():
    parser = argparse.ArgumentParser(description="Pretrain encoder: noise-conditioned prediction")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1,
                        help="Diversity loss weight (negative MSE between two noise samples)")
    parser.add_argument("--output_dir", type=str, default="models/backfill/gru_encoder_noise_cond")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)

    train_ds = NextFrameDataset(surfaces, start_idx=0, end_idx=4040)
    val_ds = NextFrameDataset(surfaces, start_idx=4040, end_idx=4540)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build encoder + noise-conditioned prediction head
    enc_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=args.gru_hidden,
        bottleneck_dim=args.bottleneck_dim,
        dropout=args.dropout,
    )
    encoder = GRUEncoder(enc_config).to(device)
    # Pred head takes condition (128) + noise (16) = 144
    pred_head = nn.Sequential(
        nn.Linear(args.bottleneck_dim + args.noise_dim, 128),
        nn.SiLU(),
        nn.Linear(128, 25),
    ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in pred_head.parameters())
    print(f"Encoder: {n_enc:,} params, Pred head: {n_head:,} params")

    all_params = list(encoder.parameters()) + list(pred_head.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    history = []
    best_val = float("inf")

    print(f"\n{'='*70}")
    print(f"Pretraining GRU encoder: noise-conditioned prediction")
    print(f"  gru_hidden={args.gru_hidden}, bottleneck={args.bottleneck_dim}, noise_dim={args.noise_dim}")
    print(f"  lambda_div={args.lambda_div}, dropout={args.dropout}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        encoder.train()
        pred_head.train()
        total_mse = 0.0
        total_div = 0.0
        n_batches = 0

        for hist_batch, target_batch in train_loader:
            hist_batch = hist_batch.to(device)
            target_batch = target_batch.to(device)
            B = hist_batch.shape[0]
            target_flat = target_batch.reshape(B, 25)

            cond = encoder(hist_batch)  # (B, 128)

            # First prediction with noise z1
            z1 = torch.randn(B, args.noise_dim, device=device)
            combined1 = torch.cat([cond, z1], dim=-1)
            pred1 = pred_head(combined1)
            mse_loss = F.mse_loss(pred1, target_flat)

            # Second prediction with different noise z2
            z2 = torch.randn(B, args.noise_dim, device=device)
            combined2 = torch.cat([cond, z2], dim=-1)
            pred2 = pred_head(combined2)

            # Diversity loss: maximize difference between predictions
            div_loss = -F.mse_loss(pred1, pred2)

            loss = mse_loss + args.lambda_div * div_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_mse += mse_loss.item()
            total_div += div_loss.item()
            n_batches += 1

        avg_mse = total_mse / n_batches
        avg_div = total_div / n_batches

        # Validate
        encoder.eval()
        pred_head.eval()
        val_mse = 0.0
        val_div = 0.0
        n_val = 0
        with torch.no_grad():
            for hist_batch, target_batch in val_loader:
                hist_batch = hist_batch.to(device)
                target_batch = target_batch.to(device)
                B = hist_batch.shape[0]
                target_flat = target_batch.reshape(B, 25)

                cond = encoder(hist_batch)
                z1 = torch.randn(B, args.noise_dim, device=device)
                pred1 = pred_head(torch.cat([cond, z1], dim=-1))
                val_mse += F.mse_loss(pred1, target_flat).item()

                z2 = torch.randn(B, args.noise_dim, device=device)
                pred2 = pred_head(torch.cat([cond, z2], dim=-1))
                val_div += -F.mse_loss(pred1, pred2).item()
                n_val += 1
        val_mse /= n_val
        val_div /= n_val

        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"mse={avg_mse:.6f}  div={avg_div:.6f}  "
                  f"val_mse={val_mse:.6f}  val_div={val_div:.6f}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_mse": avg_mse, "train_div": avg_div,
            "val_mse": val_mse, "val_div": val_div,
        })

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "encoder_config": {
                    "input_dim": 25,
                    "gru_hidden_dim": args.gru_hidden,
                    "bottleneck_dim": args.bottleneck_dim,
                    "dropout": args.dropout,
                },
                "epoch": epoch,
                "val_mse": val_mse,
            }, f"{args.output_dir}/encoder.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Check how much noise matters: compare pred variance across noise samples
    encoder.eval()
    pred_head.eval()
    noise_vars = []
    with torch.no_grad():
        for hist_batch, _ in val_loader:
            hist_batch = hist_batch.to(device)
            B = hist_batch.shape[0]
            cond = encoder(hist_batch)
            preds = []
            for _ in range(20):
                z = torch.randn(B, args.noise_dim, device=device)
                p = pred_head(torch.cat([cond, z], dim=-1))
                preds.append(p)
            preds = torch.stack(preds)  # (20, B, 25)
            noise_vars.append(preds.var(dim=0).mean().item())
    avg_noise_var = np.mean(noise_vars)
    print(f"\nNoise-driven prediction variance (20 samples): {avg_noise_var:.6f}")
    print(f"  (Higher = noise matters more in prediction)")

    best_epoch = [h for h in history if h["val_mse"] == best_val][0]["epoch"]
    print(f"\nBest val MSE: {best_val:.6f} at epoch {best_epoch}")
    print(f"Encoder saved to {args.output_dir}/encoder.pt")


if __name__ == "__main__":
    main()
