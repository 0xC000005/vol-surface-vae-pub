"""Pretrain GRU encoder with next-frame MSE + supervised contrastive loss.

Combines:
1. Next-frame prediction (MSE) — learn useful features
2. Supervised contrastive (SupCon) — push calm/turb representations apart

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/pretrain_encoder_contrastive.py \
        --epochs 30 --output_dir models/backfill/gru_encoder_contrastive --device cuda
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


def compute_regime_labels(surfaces, history_len=30):
    """Precompute regime labels for all valid windows.

    For each window starting at idx, history = surfaces[idx:idx+history_len].
    Compute vol-of-vol from daily changes of mean IV.
    Top 20% = turb (2), bottom 20% = calm (0), middle 60% = neutral (1).
    """
    n_windows = len(surfaces) - history_len
    vov = np.zeros(n_windows)
    for i in range(n_windows):
        hist = surfaces[i:i + history_len]  # (H, 5, 5)
        mean_iv = hist.mean(axis=(1, 2))  # (H,)
        daily_chg = np.diff(mean_iv)
        vov[i] = daily_chg.std()

    labels = np.ones(n_windows, dtype=np.int64)  # default: neutral (1)
    p20 = np.percentile(vov, 20)
    p80 = np.percentile(vov, 80)
    labels[vov <= p20] = 0  # calm
    labels[vov >= p80] = 2  # turb
    return labels, vov


class ContrastiveDataset(Dataset):
    """Fixed 30-frame history → next frame + regime label."""

    def __init__(self, surfaces, regime_labels, start_idx=0, end_idx=None):
        end_idx = end_idx or len(surfaces)
        self.surfaces = surfaces[start_idx:end_idx]
        self.history_len = 30
        self.n_samples = len(self.surfaces) - self.history_len
        # Regime labels aligned to window starting positions within this slice
        self.labels = regime_labels[start_idx:start_idx + self.n_samples]
        print(f"ContrastiveDataset: {self.n_samples} samples, "
              f"calm={int((self.labels == 0).sum())}, "
              f"neutral={int((self.labels == 1).sum())}, "
              f"turb={int((self.labels == 2).sum())}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        history = self.surfaces[idx:idx + self.history_len]  # (30, 5, 5)
        target = self.surfaces[idx + self.history_len]  # (5, 5)
        label = self.labels[idx]
        return (
            torch.tensor(history, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss (SupCon, Khosla et al. 2020).

    For each anchor, positives = same label, negatives = different label.
    Only considers samples with non-neutral labels (0 or 2) as anchors.
    Neutral (1) samples are used as negatives but not as anchors.

    Args:
        features: (B, D) L2-normalized embeddings
        labels: (B,) regime labels (0=calm, 1=neutral, 2=turb)
        temperature: scaling factor

    Returns:
        scalar loss
    """
    device = features.device
    B = features.shape[0]

    # Mask: which samples are non-neutral (valid anchors)
    non_neutral = (labels == 0) | (labels == 2)
    if non_neutral.sum() < 2:
        return torch.tensor(0.0, device=device)

    # Similarity matrix
    sim = torch.mm(features, features.t()) / temperature  # (B, B)

    # For numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Mask: same label (positives)
    label_match = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    # Exclude self
    self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
    positive_mask = label_match & self_mask & non_neutral.unsqueeze(0) & non_neutral.unsqueeze(1)

    # Denominator: all non-self pairs
    exp_sim = torch.exp(sim) * self_mask.float()
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Log probability of positives
    log_prob = sim - log_denom  # (B, B)

    # Average over positives for each anchor
    n_positives = positive_mask.sum(dim=1).float()  # (B,)
    # Only compute for anchors that have at least 1 positive
    valid_anchors = (n_positives > 0) & non_neutral
    if valid_anchors.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss_per_anchor = -(log_prob * positive_mask.float()).sum(dim=1) / (n_positives + 1e-8)
    loss = loss_per_anchor[valid_anchors].mean()
    return loss


def main():
    parser = argparse.ArgumentParser(description="Pretrain GRU encoder: MSE + contrastive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lambda_contrast", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="models/backfill/gru_encoder_contrastive")
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

    # Precompute regime labels for all windows
    print("Computing regime labels...")
    all_labels, all_vov = compute_regime_labels(surfaces, history_len=30)
    print(f"Total windows: {len(all_labels)}, "
          f"calm={int((all_labels == 0).sum())}, "
          f"neutral={int((all_labels == 1).sum())}, "
          f"turb={int((all_labels == 2).sum())}")

    train_ds = ContrastiveDataset(surfaces, all_labels, start_idx=0, end_idx=4040)
    val_ds = ContrastiveDataset(surfaces, all_labels, start_idx=4040, end_idx=4540)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    history = []
    best_val = float("inf")

    print(f"\n{'='*70}")
    print(f"Pretraining GRU encoder: MSE + SupCon contrastive")
    print(f"  gru_hidden={args.gru_hidden}, bottleneck={args.bottleneck_dim}, dropout={args.dropout}")
    print(f"  lambda_contrast={args.lambda_contrast}, temperature={args.temperature}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        encoder.train()
        pred_head.train()
        total_mse = 0.0
        total_con = 0.0
        total_loss = 0.0
        n_batches = 0
        for hist_batch, target_batch, label_batch in train_loader:
            hist_batch = hist_batch.to(device)
            target_batch = target_batch.to(device)
            label_batch = label_batch.to(device)

            cond = encoder(hist_batch)  # (B, 128)
            pred = pred_head(cond)  # (B, 25)
            mse_loss = F.mse_loss(pred, target_batch.reshape(-1, 25))

            # Contrastive on L2-normalized features
            cond_norm = F.normalize(cond, dim=1)
            con_loss = supervised_contrastive_loss(
                cond_norm, label_batch, temperature=args.temperature
            )

            loss = mse_loss + args.lambda_contrast * con_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(pred_head.parameters()), 1.0
            )
            optimizer.step()

            total_mse += mse_loss.item()
            total_con += con_loss.item()
            total_loss += loss.item()
            n_batches += 1

        avg_mse = total_mse / n_batches
        avg_con = total_con / n_batches
        avg_loss = total_loss / n_batches

        # Validate (MSE only)
        encoder.eval()
        pred_head.eval()
        val_mse = 0.0
        val_con = 0.0
        n_val = 0
        with torch.no_grad():
            for hist_batch, target_batch, label_batch in val_loader:
                hist_batch = hist_batch.to(device)
                target_batch = target_batch.to(device)
                label_batch = label_batch.to(device)
                cond = encoder(hist_batch)
                pred = pred_head(cond)
                val_mse += F.mse_loss(pred, target_batch.reshape(-1, 25)).item()
                cond_norm = F.normalize(cond, dim=1)
                val_con += supervised_contrastive_loss(
                    cond_norm, label_batch, temperature=args.temperature
                ).item()
                n_val += 1
        val_mse /= n_val
        val_con /= n_val

        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"mse={avg_mse:.6f}  con={avg_con:.4f}  "
                  f"val_mse={val_mse:.6f}  val_con={val_con:.4f}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_mse": avg_mse, "train_con": avg_con,
            "train_loss": avg_loss, "val_mse": val_mse, "val_con": val_con,
        })

        # Save best on combined val loss
        val_combined = val_mse + args.lambda_contrast * val_con
        if val_combined < best_val:
            best_val = val_combined
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
                "val_con": val_con,
            }, f"{args.output_dir}/encoder.pt")

    # Save training history
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Quick regime separation check
    encoder.eval()
    all_conds, all_labels_list = [], []
    with torch.no_grad():
        for hist_batch, _, label_batch in val_loader:
            cond = encoder(hist_batch.to(device)).cpu().numpy()
            all_conds.append(cond)
            all_labels_list.append(label_batch.numpy())
    all_conds = np.concatenate(all_conds)
    all_labels_arr = np.concatenate(all_labels_list)

    calm_mask = all_labels_arr == 0
    turb_mask = all_labels_arr == 2
    if calm_mask.sum() > 0 and turb_mask.sum() > 0:
        calm_mean = all_conds[calm_mask].mean(axis=0)
        turb_mean = all_conds[turb_mask].mean(axis=0)
        l2_sep = np.linalg.norm(turb_mean - calm_mean)
        overall_std = all_conds.std()
        norm_sep = l2_sep / (overall_std * np.sqrt(128))
        print(f"\nRegime separation on val:")
        print(f"  L2 distance: {l2_sep:.4f}")
        print(f"  Normalized:  {norm_sep:.4f}")
        print(f"  (DDPM target: L2 > 0.25, normalized > 0.35)")

    # Per-cell MSE
    all_preds, all_targets = [], []
    with torch.no_grad():
        for hist_batch, target_batch, _ in val_loader:
            cond = encoder(hist_batch.to(device))
            pred = pred_head(cond).reshape(-1, 5, 5)
            all_preds.append(pred.cpu())
            all_targets.append(target_batch)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    per_cell_mse = ((all_preds - all_targets) ** 2).mean(dim=0)

    best_epoch = history[[h["val_mse"] + args.lambda_contrast * h["val_con"]
                          for h in history].index(best_val)]["epoch"]
    print(f"\nBest combined val loss: {best_val:.6f} at epoch {best_epoch}")
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
            print(f"  {per_cell_mse[i, j].item() * 1000:8.3f}", end="")
        print()

    print(f"\nEncoder saved to {args.output_dir}/encoder.pt")


if __name__ == "__main__":
    main()
