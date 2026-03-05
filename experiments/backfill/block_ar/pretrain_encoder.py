"""Pretrain GRU encoder with next-frame MSE prediction.

Trains the same GRUEncoder architecture used by SinglePassBlockAR,
but with a simple MSE objective: predict frame T+1 from frames 1..T.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/pretrain_encoder.py \
        --epochs 100 --output_dir models/backfill/gru_encoder_mse --device cuda
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
    """Variable-length history → next frame prediction."""

    def __init__(self, surfaces: np.ndarray, min_history: int = 10, max_history: int = 60,
                 start_idx: int = 0, end_idx: int | None = None):
        end_idx = end_idx or len(surfaces)
        self.surfaces = surfaces[start_idx:end_idx]
        self.max_history = max_history
        self.min_history = min_history
        # Valid starts: need at least min_history + 1 frames
        self.n_samples = len(self.surfaces) - max_history
        print(f"NextFrameDataset: {self.n_samples} samples from indices {start_idx}:{end_idx}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random history length for each sample (robustness)
        hist_len = np.random.randint(self.min_history, self.max_history + 1)
        history = self.surfaces[idx:idx + hist_len]  # (H, 5, 5)
        target = self.surfaces[idx + hist_len]  # (5, 5)
        return torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def collate_fn(batch):
    """Pad variable-length histories to max length in batch."""
    histories, targets = zip(*batch)
    max_len = max(h.shape[0] for h in histories)
    padded = torch.zeros(len(histories), max_len, 5, 5)
    masks = torch.zeros(len(histories), max_len)  # 1 = valid, 0 = pad
    for i, h in enumerate(histories):
        padded[i, :h.shape[0]] = h
        masks[i, :h.shape[0]] = 1.0
    targets = torch.stack(targets)
    return padded, masks, targets


def main():
    parser = argparse.ArgumentParser(description="Pretrain GRU encoder with next-frame MSE")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="models/backfill/gru_encoder_mse")
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Build encoder + prediction head
    enc_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=args.gru_hidden,
        bottleneck_dim=args.bottleneck_dim,
        dropout=args.dropout,
    )
    encoder = GRUEncoder(enc_config).to(device)
    pred_head = nn.Linear(args.bottleneck_dim, 25).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in pred_head.parameters())
    print(f"Encoder: {n_enc:,} params, Pred head: {n_head:,} params")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(pred_head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = []
    best_val = float("inf")

    print(f"\n{'='*60}")
    print(f"Pretraining GRU encoder with next-frame MSE")
    print(f"  gru_hidden={args.gru_hidden}, bottleneck={args.bottleneck_dim}, dropout={args.dropout}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        encoder.train()
        pred_head.train()
        train_loss = 0.0
        n_batches = 0
        for padded, masks, targets in train_loader:
            padded, targets = padded.to(device), targets.to(device)
            cond = encoder(padded)  # (B, bottleneck_dim)
            pred = pred_head(cond)  # (B, 25)
            loss = F.mse_loss(pred, targets.reshape(-1, 25))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(pred_head.parameters()), 1.0
            )
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        # Validate
        encoder.eval()
        pred_head.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for padded, masks, targets in val_loader:
                padded, targets = padded.to(device), targets.to(device)
                cond = encoder(padded)
                pred = pred_head(cond)
                val_loss += F.mse_loss(pred, targets.reshape(-1, 25)).item()
                n_val += 1
        val_loss /= n_val

        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  ({elapsed:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "encoder_config": {
                    "input_dim": 25,
                    "gru_hidden_dim": args.gru_hidden,
                    "bottleneck_dim": args.bottleneck_dim,
                    "dropout": args.dropout,
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }, f"{args.output_dir}/encoder.pt")

    # Save training history
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Per-cell prediction error at final epoch
    encoder.eval()
    pred_head.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for padded, masks, targets in val_loader:
            padded, targets = padded.to(device), targets.to(device)
            cond = encoder(padded)
            pred = pred_head(cond).reshape(-1, 5, 5)
            all_preds.append(pred.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    per_cell_mse = ((all_preds - all_targets) ** 2).mean(dim=0)

    print(f"\nBest val MSE: {best_val:.6f} at epoch {history[[h['val_loss'] for h in history].index(best_val)]['epoch']}")
    print(f"\nPer-cell MSE (×1000) on validation:")
    labels_k = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
    labels_t = ["1M", "3M", "6M", "1Y", "2Y"]
    print(f"{'':>8s}", end="")
    for k in labels_k:
        print(f"  {k:>8s}", end="")
    print()
    for i, t in enumerate(labels_t):
        print(f"{t:>8s}", end="")
        for j in range(5):
            print(f"  {per_cell_mse[i,j].item()*1000:8.3f}", end="")
        print()

    print(f"\nEncoder saved to {args.output_dir}/encoder.pt")


if __name__ == "__main__":
    main()
