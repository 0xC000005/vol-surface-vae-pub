#!/usr/bin/env python
"""
162a: Direct Output E2E (RC19-H4-S1)

No MeanPredictor. CLN transformer outputs full surface directly.
Last-frame skip: output = last_frame + CLN(cond, noise).
20-epoch warm-up: freeze noise → train mean quality first (Stirn 2023).

Evidence: FGN (2506.10772) has no explicit mean/residual split.
FCN3 argues residual limits to Euler steps. But Wells Fargo cautions:
simple > complex at N~4000.

Architecture:
  history (30×5×5) → GRU encoder → condition (128-dim)
  (condition, noise) → CLNResidualTransformer → delta (30×25)
  output = last_frame_expanded + delta  (last-frame skip, zero-init)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_162a_direct.py \
        --epochs 80 --batch_size 8 --n_members 8 --noise_dim 32 \
        --warmup_epochs 20 \
        --output_dir models/backfill/flow_162a --device cuda
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
from experiments.backfill.block_ar.train_155d_cln_transformer import (
    CLNResidualTransformer, afcrps_loss, interval_score, per_frame_energy_score
)
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    normalize_iv, make_serializable
)


class MeanPredictor(nn.Module):
    """Predicts mean future surface from condition vector.
    Replaces the frozen 153a flow model."""

    def __init__(self, cond_dim=128, n_frames=30, n_cells=25, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_frames * n_cells),
        )
        # Initialize output near zero — combined with last_frame skip,
        # this means initial prediction ≈ repeat last frame
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_frames = n_frames
        self.n_cells = n_cells

    def forward(self, cond, last_frame):
        """
        Args:
            cond: (B, cond_dim) conditioning vector
            last_frame: (B, n_cells) last observed frame (for skip connection)
        Returns:
            mean_pred: (B, n_frames * n_cells) in [0, 1]
        """
        delta = self.net(cond).reshape(-1, self.n_frames, self.n_cells)
        # Skip connection: predict delta from last observed frame
        base = last_frame.unsqueeze(1).expand(-1, self.n_frames, -1)
        return (base + delta).clamp(0, 1).reshape(-1, self.n_frames * self.n_cells)


def load_pretrained_encoder(path, device):
    """Load pretrained GRU encoder for warm start.
    Uses the same load_encoder from train_cond_oneshot_flow which handles
    both DDPM and SinglePassBlockAR checkpoint formats."""
    from experiments.backfill.block_ar.train_cond_oneshot_flow import load_encoder
    encoder, cond_dim = load_encoder(path, device)
    return encoder, cond_dim


def evaluate_model(encoder, cln_model, surfaces, start_indices,
                   n_samples=50, device='cuda', ret=None):
    """Full evaluation on a set of windows (direct output, no MeanPredictor)."""
    encoder.eval(); cln_model.eval()
    H, T, C = 30, 30, 25
    N = len(start_indices)

    all_samples = []; all_gt = []
    with torch.no_grad():
        for idx in start_indices:
            hist = torch.from_numpy(
                surfaces[idx:idx+H][None].astype(np.float32)
            ).to(device)
            future = surfaces[idx+H:idx+H+T].reshape(T, C)
            all_gt.append(future)

            cond = encoder(normalize_iv(hist))  # (1, cond_dim)
            last_frame = hist[0, -1].reshape(1, C)  # (1, C)

            cond_K = cond.expand(n_samples, -1)
            noise = torch.randn(n_samples, cln_model.noise_dim, device=device)
            delta = cln_model(cond_K, noise)  # (K, T*C)
            base = last_frame.expand(n_samples, C).unsqueeze(1).expand(n_samples, T, C)
            combined = (base + delta.reshape(n_samples, T, C)).clamp(0, 1).cpu().numpy()
            all_samples.append(combined)

    samples = np.array(all_samples)  # (N, K, T, C)
    gt = np.array(all_gt)  # (N, T, C)

    # CI worst cell
    worst_ci = 1.0
    for c in range(C):
        lo = np.percentile(samples[:, :, :, c], 5, axis=1)
        hi = np.percentile(samples[:, :, :, c], 95, axis=1)
        cov = ((gt[:, :, c] >= lo) & (gt[:, :, c] <= hi)).mean()
        worst_ci = min(worst_ci, cov)

    # KS on daily changes
    gen_ch = np.diff(samples[:, 0], axis=1).reshape(-1, C)
    gt_ch = np.diff(gt, axis=1).reshape(-1, C)
    ks = sum(1 for c2 in range(C) if ks_2samp(gen_ch[:, c2], gt_ch[:, c2])[0] < 0.15)

    # Correlation
    gc = np.corrcoef(gen_ch.T); gtc = np.corrcoef(gt_ch.T)
    corr = np.abs(gc).mean() / (np.abs(gtc).mean() + 1e-6)

    # Kurtosis
    kr = kurtosis(gen_ch.flatten()) / (kurtosis(gt_ch.flatten()) + 1e-6)

    # Spread-skill
    ss = samples.std(axis=1).mean() / (np.abs(samples.mean(axis=1) - gt).mean() + 1e-8)

    # MAE (mean prediction quality)
    mae = np.abs(samples.mean(axis=1) - gt).mean()

    turb_calm = None
    if ret is not None and len(ret) >= N:
        spreads = samples.std(axis=1).mean(axis=(1, 2))
        vol_30d = np.array([np.std(ret[max(0,i-30):i]) if i >= 30 else np.std(ret[:i+1])
                           for i in range(N)])
        q75, q25 = np.percentile(vol_30d, 75), np.percentile(vol_30d, 25)
        turb_mask = vol_30d >= q75; calm_mask = vol_30d <= q25
        if turb_mask.sum() > 5 and calm_mask.sum() > 5:
            turb_calm = float(spreads[turb_mask].mean() / (spreads[calm_mask].mean() + 1e-8))

    return {
        "ci_worst": float(worst_ci), "ks": ks, "corr": float(corr),
        "kurt": float(kr), "ss": float(ss), "mae": float(mae),
        "turb_calm": turb_calm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_encoder", type=float, default=1e-4,
                        help="Encoder LR (lower — pretrained, fine-tune)")
    parser.add_argument("--lr_decoder", type=float, default=1e-3,
                        help="Mean predictor + CLN LR")
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_members", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--spread_weight", type=float, default=0.5)
    parser.add_argument("--lambda_is", type=float, default=0.5)
    parser.add_argument("--lambda_es", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=20,
                        help="Freeze noise for first N epochs (train mean quality)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)
    ret = data["ret"]  # (N,)
    N_total = surfaces.shape[0]
    H, T, C = 30, 30, 25

    # Data splits: train on full history, test starts at 4511
    TEST_START = 4511
    max_train_idx = TEST_START - H - T  # 4451
    VAL_SIZE = 441
    train_indices = np.arange(0, max_train_idx - VAL_SIZE)  # 0..4009
    val_indices = np.arange(max_train_idx - VAL_SIZE, max_train_idx)  # 4010..4450
    test_indices = np.arange(TEST_START, N_total - H - T + 1)  # 4511..

    print(f"Train: {len(train_indices)} windows, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Load pretrained encoder (warm start)
    encoder, cond_dim = load_pretrained_encoder(args.encoder_path, device)
    encoder = encoder.to(device)
    # Unfreeze encoder for joint training
    for p in encoder.parameters():
        p.requires_grad = True

    # CLN transformer — outputs full surface delta (no MeanPredictor)
    cln_model = CLNResidualTransformer(
        n_frames=T, n_cells=C, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, cond_dim=cond_dim, noise_dim=args.noise_dim,
    ).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_cln = sum(p.numel() for p in cln_model.parameters())
    print(f"\n{'='*60}")
    print(f"162a: Direct Output E2E (RC19-H4-S1)")
    print(f"{'='*60}")
    print(f"  Encoder: {n_enc:,} params (warm start, lr={args.lr_encoder})")
    print(f"  CLN Transformer: {n_cln:,} params (direct output, no MeanPredictor)")
    print(f"  Total: {n_enc + n_cln:,} params")
    print(f"  noise_dim={args.noise_dim}, d_model={args.d_model}")
    print(f"  Warm-up: {args.warmup_epochs} epochs (noise frozen)")

    # Preload surfaces to GPU
    surf_tensor = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_indices)),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_indices)),
        batch_size=args.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr_encoder, "weight_decay": 0.01},
        {"params": cln_model.parameters(), "lr": args.lr_decoder, "weight_decay": 0.01},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        encoder.train(); cln_model.train()
        ep_loss = 0; ep_mae = 0; ep_spread = 0; nb = 0

        # Warm-up: freeze noise for first N epochs (train mean only)
        warmup = epoch <= args.warmup_epochs

        for (idx_batch,) in train_loader:
            B = idx_batch.shape[0]
            K = args.n_members

            hist_list = []; future_list = []; last_frames = []
            for i in idx_batch:
                i = i.item()
                hist_list.append(surf_tensor[i:i+H].unsqueeze(0))
                future_list.append(surf_tensor[i+H:i+H+T].reshape(T, C))
                last_frames.append(surf_tensor[i+H-1].reshape(C))

            hist = torch.cat(hist_list, dim=0)  # (B, H, 5, 5)
            gt = torch.stack(future_list)  # (B, T, C)
            last_frame = torch.stack(last_frames)  # (B, C)

            # Forward: encoder
            cond = encoder(normalize_iv(hist))  # (B, cond_dim)

            # Forward: CLN direct output (K members)
            if warmup:
                # During warm-up: zero noise → all members identical → learn mean
                noise = torch.zeros(B * K, args.noise_dim, device=device)
            else:
                noise = torch.randn(B * K, args.noise_dim, device=device)
            cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            delta = cln_model(cond_K, noise).reshape(B, K, T, C)

            # Last-frame skip: output = last_frame + delta (CLN zero-init → start as repeat)
            base = last_frame.unsqueeze(1).unsqueeze(2).expand(B, K, T, C)
            combined = (base + delta).clamp(0, 1)  # (B, K, T, C)

            # Loss
            crps, mae, spread = afcrps_loss(combined, gt, alpha=args.alpha,
                                            spread_weight=args.spread_weight)
            is_loss = interval_score(combined, gt)
            es_loss = per_frame_energy_score(combined, gt) if args.lambda_es > 0 else torch.tensor(0.0, device=device)
            loss = crps + args.lambda_is * is_loss + args.lambda_es * es_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(cln_model.parameters()), 1.0
            )
            optimizer.step()

            ep_loss += loss.item(); ep_mae += mae.item(); ep_spread += spread.item()
            nb += 1

        scheduler.step()
        tl = ep_loss/nb; tm = ep_mae/nb; ts = ep_spread/nb
        elapsed = time.time() - t0

        # Validation
        encoder.eval(); cln_model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for (idx_batch,) in val_loader:
                B = idx_batch.shape[0]; K = args.n_members
                hist_list = []; future_list = []; last_frames = []
                for i in idx_batch:
                    i = i.item()
                    hist_list.append(surf_tensor[i:i+H].unsqueeze(0))
                    future_list.append(surf_tensor[i+H:i+H+T].reshape(T, C))
                    last_frames.append(surf_tensor[i+H-1].reshape(C))
                hist = torch.cat(hist_list, dim=0)
                gt = torch.stack(future_list)
                last_frame = torch.stack(last_frames)
                cond = encoder(normalize_iv(hist))
                noise = torch.randn(B * K, args.noise_dim, device=device)
                cond_K = cond.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
                delta = cln_model(cond_K, noise).reshape(B, K, T, C)
                base = last_frame.unsqueeze(1).unsqueeze(2).expand(B, K, T, C)
                combined = (base + delta).clamp(0, 1)
                crps, _, _ = afcrps_loss(combined, gt, alpha=args.alpha,
                                         spread_weight=args.spread_weight)
                vl += crps.item() * B; nv += B
        val_loss = vl / nv

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "encoder_state": encoder.state_dict(),
                "cln_state": cln_model.state_dict(),
                "epoch": epoch, "val_loss": val_loss,
                "config": {
                    "n_frames": T, "n_cells": C, "d_model": args.d_model,
                    "n_heads": args.n_heads, "n_layers": args.n_layers,
                    "cond_dim": cond_dim, "noise_dim": args.noise_dim,
                    "n_members": args.n_members, "alpha": args.alpha,
                    "spread_weight": args.spread_weight, "lambda_is": args.lambda_is,
                    "lambda_es": args.lambda_es,
                    "type": "direct_output_cln_transformer",
                    "warmup_epochs": args.warmup_epochs,
                    "train_windows": len(train_indices),
                },
            }, f"{args.output_dir}/best_model.pt")

        # Periodic eval
        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
            val_ret = ret[val_indices[0]:val_indices[0] + 100]
            val_metrics = evaluate_model(
                encoder, cln_model, surfaces,
                val_indices[:100], n_samples=50, device=device, ret=val_ret
            )
            test_ret = ret[test_indices[0]:test_indices[0] + 100]
            test_metrics = evaluate_model(
                encoder, cln_model, surfaces,
                test_indices[:100], n_samples=50, device=device, ret=test_ret
            )
            tc_v = f"{val_metrics['turb_calm']:.3f}" if val_metrics.get('turb_calm') else "N/A"
            tc_t = f"{test_metrics['turb_calm']:.3f}" if test_metrics.get('turb_calm') else "N/A"
            w = " [WARMUP]" if warmup else ""
            print(f"Ep {epoch:3d}  loss={tl:.4f}  val_loss={val_loss:.4f}  "
                  f"mae={tm:.4f}  spread={ts:.4f}  ({elapsed:.1f}s){w}")
            print(f"  VAL:  CI={val_metrics['ci_worst']:.3f}  KS={val_metrics['ks']}/25  "
                  f"corr={val_metrics['corr']:.3f}  mae={val_metrics['mae']:.4f}  "
                  f"turb/calm={tc_v}")
            print(f"  TEST: CI={test_metrics['ci_worst']:.3f}  KS={test_metrics['ks']}/25  "
                  f"corr={test_metrics['corr']:.3f}  mae={test_metrics['mae']:.4f}  "
                  f"turb/calm={tc_t}")
            history.append({
                "epoch": epoch, "train_loss": tl, "val_loss": val_loss,
                "mae": tm, "spread": ts, "warmup": warmup,
                "val_metrics": val_metrics, "test_metrics": test_metrics,
            })
        else:
            print(f"Ep {epoch:3d}  loss={tl:.4f}  val_loss={val_loss:.4f}  "
                  f"mae={tm:.4f}  spread={ts:.4f}  ({elapsed:.1f}s)")
            history.append({"epoch": epoch, "train_loss": tl, "val_loss": val_loss,
                           "mae": tm, "spread": ts})

    # Save final model
    torch.save({
        "encoder_state": encoder.state_dict(),
        "cln_state": cln_model.state_dict(),
        "epoch": args.epochs, "val_loss": val_loss,
        "config": {
            "n_frames": T, "n_cells": C, "d_model": args.d_model,
            "n_heads": args.n_heads, "n_layers": args.n_layers,
            "cond_dim": cond_dim, "noise_dim": args.noise_dim,
            "n_members": args.n_members, "alpha": args.alpha,
            "spread_weight": args.spread_weight, "lambda_is": args.lambda_is,
            "lambda_es": args.lambda_es,
            "type": "direct_output_cln_transformer",
            "warmup_epochs": args.warmup_epochs,
            "train_windows": len(train_indices),
        },
    }, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
