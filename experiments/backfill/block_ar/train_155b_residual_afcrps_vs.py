#!/usr/bin/env python
"""
155b: Single-Pass Residual MLP + afCRPS + Variogram Score (RC17-H2r)

Same as 155a but adds per-frame Variogram Score to the loss. VS targets
pairwise cell dependency structure (300 pairs for 25 cells).

Hypothesis: VS provides 300x more correlation gradient than ES. Adding VS
to afCRPS should (a) preserve cross-cell correlation and (b) change spread
contraction dynamics by providing additional gradient signal that rewards
spread in correlation-preserving directions.

Literature: Lakatos (2509.02784), STIPP (2601.02882)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_155b_residual_afcrps_vs.py \
        --base_model models/backfill/flow_153a/final_model.pt \
        --epochs 200 --batch_size 64 --n_members 8 --noise_dim 32 \
        --lambda_vs 0.1 \
        --output_dir models/backfill/flow_155b --device cuda
"""

import argparse
import json
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
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv,
    make_serializable
)
from experiments.backfill.block_ar.train_oneshot_flow import evaluate_samples


class ResidualMLP(nn.Module):
    """Simple MLP: (noise, condition) → residual.

    Zero-initialized output so model starts predicting base_pred exactly.
    """

    def __init__(self, noise_dim=32, cond_dim=128, hidden_dim=512, output_dim=750,
                 n_layers=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim

        layers = []
        in_dim = noise_dim + cond_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Zero-init output: at start, residual = 0 → combined = base_pred
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noise, condition):
        """
        Args:
            noise: (B, noise_dim)
            condition: (B, cond_dim)
        Returns:
            residual: (B, output_dim) in raw IV space
        """
        x = torch.cat([noise, condition], dim=-1)
        h = self.backbone(x)
        return self.output_proj(h)


def afcrps_loss(samples, gt, alpha=0.95):
    """Almost-Fair CRPS for flat (B, K, D) tensors.

    Args:
        samples: (B, K, D) ensemble members
        gt: (B, D) ground truth
    Returns:
        (loss, mae_term, spread_term)
    """
    K = samples.shape[1]
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)

    mae = (samples - gt.unsqueeze(1)).abs().mean()
    spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()

    fcrps = mae - 0.5 * spread
    loss = alpha * fcrps + (1 - alpha) * mae

    return loss, mae, spread


def variogram_score(samples, gt, p=0.5):
    """Variogram score for cross-cell dependency structure.

    Args:
        samples: (B, K, T, C) ensemble members
        gt: (B, T, C) ground truth
    Returns:
        Scalar VS loss
    """
    B, K, T, C = samples.shape
    eps = 1e-8
    s_diff = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    g_diff = (gt.unsqueeze(-1) - gt.unsqueeze(-2)).abs().clamp(min=eps).pow(p)
    s_mean = s_diff.mean(dim=1)
    loss = (g_diff - s_mean).pow(2)
    mask = torch.triu(torch.ones(C, C, device=samples.device), diagonal=1).bool()
    return loss[:, :, mask].sum(dim=(-2, -1)).mean()


def interval_score(samples, gt, alpha=0.9):
    """Interval score for CI calibration."""
    lo = samples.quantile(alpha / 2, dim=1)      # (B, D)
    hi = samples.quantile(1 - alpha / 2, dim=1)  # (B, D)
    width = hi - lo
    miss_lo = F.relu(lo - gt)
    miss_hi = F.relu(gt - hi)
    return (width + (2 / alpha) * (miss_lo + miss_hi)).mean()


def evaluate_155a(mlp, base_preds, gt_futures, conditions, res_std, res_mean,
                  n_samples=50, device='cuda'):
    """Full evaluation: CI, KS, kurtosis, correlation."""
    mlp.eval()
    N = len(base_preds)
    T, C = 30, 25
    DIM = T * C

    all_samples = []
    with torch.no_grad():
        for i in range(N):
            bp = base_preds[i]  # (750,)
            cond = torch.from_numpy(conditions[i:i+1]).float().to(device)  # (1, 128)
            cond_K = cond.expand(n_samples, -1)  # (K, 128)
            noise = torch.randn(n_samples, mlp.noise_dim, device=device)
            residual = mlp(noise, cond_K)  # (K, 750)
            combined = np.clip(bp + residual.cpu().numpy(), 0, 1)
            all_samples.append(combined.reshape(n_samples, T, C))

    samples = np.array(all_samples)  # (N, K, T, C)
    gt = gt_futures.reshape(N, T, C)

    # CI per-cell (worst cell)
    worst_ci = 1.0
    for c in range(C):
        lo = np.percentile(samples[:, :, :, c], 5, axis=1)   # (N, T)
        hi = np.percentile(samples[:, :, :, c], 95, axis=1)
        cov = ((gt[:, :, c] >= lo) & (gt[:, :, c] <= hi)).mean()
        worst_ci = min(worst_ci, cov)

    # CI per-horizon
    ci_h_pass = 0
    for h in range(T):
        lo = np.percentile(samples[:, :, h], 5, axis=1)
        hi = np.percentile(samples[:, :, h], 95, axis=1)
        if ((gt[:, h] >= lo) & (gt[:, h] <= hi)).mean() >= 0.85:
            ci_h_pass += 1

    # KS on daily changes
    gen_changes = np.diff(samples[:, 0], axis=1).reshape(-1, C)
    gt_changes = np.diff(gt, axis=1).reshape(-1, C)
    ks_pass = sum(1 for c in range(C) if ks_2samp(gen_changes[:, c], gt_changes[:, c])[0] < 0.15)

    # Kurtosis ratio
    kr = kurtosis(gen_changes.flatten()) / (kurtosis(gt_changes.flatten()) + 1e-6)

    # Cross-cell correlation
    gc = np.corrcoef(gen_changes.T)
    gtc = np.corrcoef(gt_changes.T)
    corr_ratio = np.abs(gc).mean() / (np.abs(gtc).mean() + 1e-6)

    # Effective rank
    def eff_rank(corr):
        ev = np.linalg.eigvalsh(corr)[::-1]
        ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10)
        p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))
    er_gen = eff_rank(gc)
    er_gt = eff_rank(gtc)

    # Spread per horizon
    spread_h1 = samples[:, :, 0].std(axis=1).mean()
    spread_h30 = samples[:, :, -1].std(axis=1).mean()

    # Spread-skill ratio
    ensemble_mean = samples.mean(axis=1)  # (N, T, C)
    skill = np.abs(ensemble_mean - gt).mean()
    spread_val = samples.std(axis=1).mean()
    ss_ratio = spread_val / (skill + 1e-8)

    return {
        "ci_worst_cell": float(worst_ci),
        "ci_h_pass": ci_h_pass,
        "ks_daily": ks_pass,
        "kurt_ratio": float(kr),
        "corr_ratio": float(corr_ratio),
        "eff_rank_gen": float(er_gen),
        "eff_rank_gt": float(er_gt),
        "eff_rank_ratio": float(er_gen / (er_gt + 1e-6)),
        "spread_h1": float(spread_h1),
        "spread_h30": float(spread_h30),
        "spread_skill_ratio": float(ss_ratio),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="models/backfill/flow_153a/final_model.pt")
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_members", type=int, default=8)
    parser.add_argument("--lambda_is", type=float, default=0.5,
                        help="Interval score weight")
    parser.add_argument("--alpha", type=float, default=0.95,
                        help="afCRPS alpha (0.95 = AIFS default)")
    parser.add_argument("--lambda_vs", type=float, default=0.1,
                        help="Variogram score weight")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load encoder (frozen, for condition computation)
    encoder, cond_dim = load_encoder(args.encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, T, DIM = 30, 30, 750

    # Load/generate base predictions and conditions
    cache_path = Path("models/backfill/flow_154b/base_predictions.npz")
    if not cache_path.exists():
        raise RuntimeError("Base predictions not cached. Run 154b first.")

    cached = np.load(cache_path)
    val_preds = cached["val_preds"]    # (441, 750) — train the residual MLP on these
    val_gts = cached["val_gts"]        # (441, 750)
    train_preds = cached["train_preds"]  # (441, 750) — use for validation
    train_gts = cached["train_gts"]

    # Compute encoder conditions
    print("Computing encoder conditions...")
    val_conds = []
    with torch.no_grad():
        for i in range(4040, 4040 + len(val_preds)):
            hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
            val_conds.append(encoder(normalize_iv(hist)).cpu().numpy())
    val_conds = np.concatenate(val_conds)

    train_conds = []
    with torch.no_grad():
        for i in range(3540, 3540 + len(train_preds)):
            hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
            train_conds.append(encoder(normalize_iv(hist)).cpu().numpy())
    train_conds = np.concatenate(train_conds)

    print(f"Data: {len(val_preds)} train windows, {len(train_preds)} val windows")
    print(f"Conditions: {val_conds.shape}, Preds: {val_preds.shape}")

    # Create model
    mlp = ResidualMLP(
        noise_dim=args.noise_dim, cond_dim=cond_dim,
        hidden_dim=args.hidden_dim, output_dim=DIM,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"\n{'='*60}")
    print(f"155b: Single-Pass Residual MLP + afCRPS + VS (RC17-H2r)")
    print(f"{'='*60}")
    print(f"  Parameters: {n_params:,}")
    print(f"  K members: {args.n_members}, noise_dim: {args.noise_dim}")
    print(f"  afCRPS alpha: {args.alpha}, lambda_IS: {args.lambda_is}, lambda_VS: {args.lambda_vs}")

    # Tensors for training
    bp_train = torch.from_numpy(val_preds).float().to(device)    # base preds (train set)
    gt_train = torch.from_numpy(val_gts).float().to(device)      # ground truth (train set)
    cd_train = torch.from_numpy(val_conds).float().to(device)    # conditions (train set)
    bp_val = torch.from_numpy(train_preds).float().to(device)    # base preds (val set)
    gt_val = torch.from_numpy(train_gts).float().to(device)
    cd_val = torch.from_numpy(train_conds).float().to(device)

    # Dataset: just indices — we index into the pre-loaded tensors
    train_indices = torch.arange(len(val_preds))
    train_loader = DataLoader(TensorDataset(train_indices), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_indices = torch.arange(len(train_preds))
    val_loader = DataLoader(TensorDataset(val_indices), batch_size=args.batch_size,
                            shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        mlp.train()
        epoch_loss = 0; epoch_mae = 0; epoch_spread = 0; epoch_is = 0; epoch_vs = 0; nb = 0

        for (idx,) in train_loader:
            B = idx.shape[0]
            K = args.n_members
            bp = bp_train[idx]   # (B, 750)
            gt = gt_train[idx]   # (B, 750)
            cd = cd_train[idx]   # (B, 128)

            # Generate K ensemble members
            noise = torch.randn(B, K, args.noise_dim, device=device)
            # Flatten for MLP: (B*K, noise_dim)
            noise_flat = noise.reshape(B * K, args.noise_dim)
            cond_flat = cd.unsqueeze(1).expand(B, K, cond_dim).reshape(B * K, cond_dim)

            residual_flat = mlp(noise_flat, cond_flat)  # (B*K, 750)
            residual = residual_flat.reshape(B, K, DIM)

            # Combined = base_pred + residual, clamped to [0, 1]
            combined = (bp.unsqueeze(1) + residual).clamp(0, 1)  # (B, K, 750)

            # Reshape for per-frame evaluation: (B, K, T, C)
            combined_4d = combined.reshape(B, K, T, 25)
            gt_4d = gt.reshape(B, T, 25)

            # afCRPS loss
            crps, mae, spread = afcrps_loss(combined_4d, gt_4d, alpha=args.alpha)

            # Interval score for CI calibration
            is_loss = interval_score(combined_4d, gt_4d)

            # Variogram score for cross-cell correlation
            vs_loss = variogram_score(combined_4d, gt_4d)

            loss = crps + args.lambda_is * is_loss + args.lambda_vs * vs_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += mae.item()
            epoch_spread += spread.item()
            epoch_is += is_loss.item()
            epoch_vs += vs_loss.item()
            nb += 1

        scheduler.step()
        train_loss = epoch_loss / nb
        train_mae = epoch_mae / nb
        train_spread = epoch_spread / nb
        train_is = epoch_is / nb
        train_vs = epoch_vs / nb
        elapsed = time.time() - t0

        # Validation
        mlp.eval()
        vl = 0; vm = 0; vs = 0; nv = 0
        with torch.no_grad():
            for (idx,) in val_loader:
                B = idx.shape[0]
                K = args.n_members
                bp = bp_val[idx]
                gt = gt_val[idx]
                cd = cd_val[idx]

                noise = torch.randn(B, K, args.noise_dim, device=device)
                noise_flat = noise.reshape(B * K, args.noise_dim)
                cond_flat = cd.unsqueeze(1).expand(B, K, cond_dim).reshape(B * K, cond_dim)
                residual = mlp(noise_flat, cond_flat).reshape(B, K, DIM)
                combined = (bp.unsqueeze(1) + residual).clamp(0, 1).reshape(B, K, T, 25)
                gt_4d = gt.reshape(B, T, 25)
                crps, mae, spread = afcrps_loss(combined, gt_4d, alpha=args.alpha)
                vl += crps.item() * B; vm += mae.item() * B; vs += spread.item() * B
                nv += B
        val_loss = vl / nv
        val_mae = vm / nv
        val_spread = vs / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": mlp.state_dict(), "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "noise_dim": args.noise_dim, "cond_dim": cond_dim,
                    "hidden_dim": args.hidden_dim, "output_dim": DIM,
                    "n_layers": args.n_layers, "n_members": args.n_members,
                    "alpha": args.alpha, "lambda_is": args.lambda_is,
                    "lambda_vs": args.lambda_vs,
                    "type": "residual_afcrps_vs_mlp",
                },
                "base_model_path": args.base_model,
            }, f"{args.output_dir}/best_model.pt")

        # Full eval every 40 epochs
        if epoch % 40 == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate_155a(
                mlp, val_preds, val_gts, val_conds, None, None,
                n_samples=50, device=device)
            print(f"Ep {epoch:3d}  loss={train_loss:.4f}  val={val_loss:.4f}  "
                  f"mae={train_mae:.4f}  spread={train_spread:.4f}  IS={train_is:.4f}  VS={train_vs:.4f}  "
                  f"({elapsed:.1f}s)")
            print(f"  -> CI worst={metrics['ci_worst_cell']:.3f}  "
                  f"CI_h={metrics['ci_h_pass']}/30  "
                  f"KS={metrics['ks_daily']}/25  kurt={metrics['kurt_ratio']:.3f}  "
                  f"corr={metrics['corr_ratio']:.3f}  "
                  f"SS={metrics['spread_skill_ratio']:.3f}  "
                  f"spread_h1={metrics['spread_h1']:.5f}  "
                  f"spread_h30={metrics['spread_h30']:.5f}")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, "train_mae": train_mae,
                           "train_spread": train_spread, "train_is": train_is, "train_vs": train_vs,
                           **metrics})
        else:
            print(f"Ep {epoch:3d}  loss={train_loss:.4f}  val={val_loss:.4f}  "
                  f"mae={train_mae:.4f}  spread={train_spread:.4f}  IS={train_is:.4f}  VS={train_vs:.4f}  "
                  f"({elapsed:.1f}s)")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, "train_mae": train_mae,
                           "train_spread": train_spread, "train_is": train_is})

    # Save final model
    torch.save({
        "model_state_dict": mlp.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss,
        "config": {
            "noise_dim": args.noise_dim, "cond_dim": cond_dim,
            "hidden_dim": args.hidden_dim, "output_dim": DIM,
            "n_layers": args.n_layers, "n_members": args.n_members,
            "alpha": args.alpha, "lambda_is": args.lambda_is,
            "type": "residual_afcrps_mlp",
        },
        "base_model_path": args.base_model,
    }, f"{args.output_dir}/final_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Training complete. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
