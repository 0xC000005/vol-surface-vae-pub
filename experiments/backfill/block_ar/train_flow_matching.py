#!/usr/bin/env python
"""
H2 Stage 1: Unconditional Flow Matching on 25-dim IV Surface Frames

Proxy experiment: Can flow matching generate realistic single IV surface
frames with GT-like cross-cell correlation (eff_rank)?

Architecture: Simple velocity MLP (x_t, t) → v_t
Training: OT-CFM loss (linear interpolation, predict velocity)
Sampling: 8-step Euler ODE from N(0,I) to data

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_flow_matching.py \
        --epochs 200 --batch_size 256 --lr 1e-3 \
        --output_dir models/backfill/flow_152a --device cuda
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


class VelocityMLP(nn.Module):
    """Velocity network: (x_t, t) → v_t for flow matching."""

    def __init__(self, dim=25, hidden=256, n_layers=4):
        super().__init__()
        # Time embedding (sinusoidal)
        self.time_dim = 64
        layers = []
        in_dim = dim + self.time_dim
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        # Zero-init last layer for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def time_embed(self, t):
        """Sinusoidal time embedding."""
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_t, t):
        """
        Args:
            x_t: (B, 25) noisy state at time t
            t: (B,) time in [0, 1]
        Returns:
            v: (B, 25) predicted velocity
        """
        t_emb = self.time_embed(t)
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)


def cfm_loss(model, x1, device):
    """Conditional Flow Matching loss with linear interpolation.

    x0 ~ N(0, I), x1 = data, x_t = (1-t)*x0 + t*x1
    Target velocity: u_t = x1 - x0
    Loss: ||v(x_t, t) - u_t||^2
    """
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=device)

    # Linear interpolation
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1

    # Target velocity (OT: x1 - x0)
    u_t = x1 - x0

    # Predict velocity
    v_t = model(x_t, t)

    # MSE loss
    return F.mse_loss(v_t, u_t)


@torch.no_grad()
def sample_ode(model, n_samples, dim=25, n_steps=8, device="cuda"):
    """Sample via Euler ODE integration from t=0 (noise) to t=1 (data)."""
    x = torch.randn(n_samples, dim, device=device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)
        v = model(x, t)
        x = x + v * dt

    return x


@torch.no_grad()
def evaluate_samples(model, gt_data, n_samples=1000, n_steps=8, device="cuda"):
    """Evaluate sample quality: eff_rank, KS, arbitrage, correlation."""
    from scipy.stats import ks_2samp

    samples = sample_ode(model, n_samples, dim=25, n_steps=n_steps, device=device)
    samples = samples.cpu().numpy()
    gt = gt_data[:n_samples] if len(gt_data) > n_samples else gt_data

    results = {}

    # 1. Cross-cell correlation structure
    gen_corr = np.corrcoef(samples.T)  # (25, 25)
    gt_corr = np.corrcoef(gt.T)  # (25, 25)

    # Eff rank
    def eff_rank(corr):
        eigvals = np.linalg.eigvalsh(corr)[::-1]
        eigvals = np.maximum(eigvals, 0)
        p = eigvals / (eigvals.sum() + 1e-10)
        p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    results["gen_eff_rank"] = eff_rank(gen_corr)
    results["gt_eff_rank"] = eff_rank(gt_corr)
    results["rank_ratio"] = results["gen_eff_rank"] / results["gt_eff_rank"]

    # Mean off-diagonal correlation
    mask = np.triu(np.ones((25, 25), dtype=bool), k=1)
    results["gen_mean_corr"] = float(gen_corr[mask].mean())
    results["gt_mean_corr"] = float(gt_corr[mask].mean())
    results["corr_ratio"] = results["gen_mean_corr"] / (results["gt_mean_corr"] + 1e-6)

    # Frobenius distance
    results["frob_dist"] = float(np.linalg.norm(gen_corr - gt_corr, 'fro'))

    # PC1 variance
    gen_eigvals = np.linalg.eigvalsh(gen_corr)[::-1]
    gen_eigvals = np.maximum(gen_eigvals, 0)
    results["gen_pc1_var"] = float(gen_eigvals[0] / (gen_eigvals.sum() + 1e-10))
    gt_eigvals = np.linalg.eigvalsh(gt_corr)[::-1]
    gt_eigvals = np.maximum(gt_eigvals, 0)
    results["gt_pc1_var"] = float(gt_eigvals[0] / (gt_eigvals.sum() + 1e-10))

    # 2. Per-cell KS test (marginal distributions)
    n_pass = 0
    ks_stats = []
    for c in range(25):
        stat, _ = ks_2samp(samples[:, c], gt[:, c])
        ks_stats.append(stat)
        if stat < 0.15:
            n_pass += 1
    results["ks_n_pass"] = n_pass
    results["ks_mean"] = float(np.mean(ks_stats))

    # 3. Arbitrage check (calendar: IV should decrease along tenor for same strike)
    # Reshape to 5x5 grid (5 strikes × 5 tenors, row=strike, col=tenor)
    gen_grids = samples.reshape(-1, 5, 5)
    cal_violations = 0
    total_pairs = 0
    for s in range(5):
        for t in range(4):
            total_pairs += len(gen_grids)
            cal_violations += (gen_grids[:, s, t+1] < gen_grids[:, s, t]).sum()
    results["calendar_arb_rate"] = float(cal_violations / max(total_pairs, 1))

    # 4. Basic statistics
    results["gen_mean"] = float(samples.mean())
    results["gen_std"] = float(samples.std())
    results["gt_mean"] = float(gt.mean())
    results["gt_std"] = float(gt.std())

    # PCA alignment: check if gen principal directions match GT
    gt_eigvecs = np.linalg.eigh(gt_corr)[1][:, ::-1]
    gen_eigvecs = np.linalg.eigh(gen_corr)[1][:, ::-1]
    pc1_align = abs(float(np.dot(gt_eigvecs[:, 0], gen_eigvecs[:, 0])))
    pc2_align = abs(float(np.dot(gt_eigvecs[:, 1], gen_eigvecs[:, 1])))
    results["pc1_alignment"] = pc1_align
    results["pc2_alignment"] = pc2_align

    return results


def main():
    parser = argparse.ArgumentParser(description="H2 Stage 1: Unconditional Flow Matching")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8, help="ODE integration steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H2 Stage 1: Unconditional Flow Matching (25-dim IV frames)")
    print("=" * 60)

    # Load data — use all IV surface frames as training data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    flat = surfaces.reshape(-1, 25).astype(np.float32)

    # Train/val split (same as afCRPS: first 4540 for training)
    train_data = flat[:4540]
    val_data = flat[4540:]

    print(f"  Training frames: {len(train_data)}")
    print(f"  Validation frames: {len(val_data)}")
    print(f"  Data range: [{train_data.min():.4f}, {train_data.max():.4f}]")
    print(f"  Hidden: {args.hidden}, Layers: {args.n_layers}")
    print(f"  ODE steps: {args.n_steps}")

    # Standardize for flow matching (mean=0, std=1 per cell)
    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True) + 1e-6
    train_norm = (train_data - train_mean) / train_std
    val_norm = (val_data - train_mean) / train_std

    train_ds = TensorDataset(torch.from_numpy(train_norm))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model
    model = VelocityMLP(dim=25, hidden=args.hidden, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for (x1,) in train_loader:
            x1 = x1.to(device)
            loss = cfm_loss(model, x1, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for i in range(0, len(val_norm), args.batch_size):
                x1 = torch.from_numpy(val_norm[i:i+args.batch_size]).to(device)
                loss = cfm_loss(model, x1, device)
                val_loss += loss.item() * x1.shape[0]
                n_val += x1.shape[0]
        val_loss = val_loss / n_val

        # Evaluate every 20 epochs
        metrics = {}
        if epoch % 20 == 0 or epoch == 1:
            # Generate standardized samples, then denormalize
            raw_samples = sample_ode(model, 2000, dim=25, n_steps=args.n_steps, device=device)
            denorm_samples = raw_samples.cpu().numpy() * train_std + train_mean
            denorm_samples = np.clip(denorm_samples, 0, 1)  # IV is in [0, 1]
            metrics = evaluate_samples(model, train_norm, n_samples=2000,
                                       n_steps=args.n_steps, device=device)
            # Also eval on denormalized
            gt_denorm = train_data
            denorm_corr = np.corrcoef(denorm_samples.T)
            gt_denorm_corr = np.corrcoef(gt_denorm.T)
            eigvals = np.linalg.eigvalsh(denorm_corr)[::-1]
            eigvals = np.maximum(eigvals, 0)
            p = eigvals / (eigvals.sum() + 1e-10)
            denorm_eff_rank = float(np.exp(-np.sum(p[p > 1e-10] * np.log(p[p > 1e-10]))))
            metrics["denorm_eff_rank"] = denorm_eff_rank
            metrics["denorm_frob"] = float(np.linalg.norm(denorm_corr - gt_denorm_corr, 'fro'))

        record = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **metrics,
        }
        history.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_mean": train_mean,
                "train_std": train_std,
                "config": {
                    "hidden": args.hidden, "n_layers": args.n_layers,
                    "n_steps": args.n_steps, "dim": 25,
                },
            }, f"{args.output_dir}/best_model.pt")

        # Print
        extra = ""
        if metrics:
            extra = (f"  eff_rank={metrics.get('gen_eff_rank', '?'):.2f}"
                     f"  corr_r={metrics.get('corr_ratio', '?'):.3f}"
                     f"  KS={metrics.get('ks_n_pass', '?')}/25"
                     f"  PC1_align={metrics.get('pc1_alignment', '?'):.3f}"
                     f"  denorm_eff={metrics.get('denorm_eff_rank', '?'):.2f}")
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}"
              f"{extra}")

    # Save final model and history
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_loss": val_loss,
        "train_mean": train_mean,
        "train_std": train_std,
        "config": {
            "hidden": args.hidden, "n_layers": args.n_layers,
            "n_steps": args.n_steps, "dim": 25,
        },
    }, f"{args.output_dir}/final_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    model.eval()
    final_metrics = evaluate_samples(model, train_norm, n_samples=5000,
                                     n_steps=args.n_steps, device=device)

    # Denormalized evaluation
    raw_samples = sample_ode(model, 5000, dim=25, n_steps=args.n_steps, device=device)
    denorm_samples = raw_samples.cpu().numpy() * train_std + train_mean
    denorm_samples = np.clip(denorm_samples, 0, 1)

    print(f"  Eff rank: {final_metrics['gen_eff_rank']:.2f} (GT: {final_metrics['gt_eff_rank']:.2f})")
    print(f"  Rank ratio: {final_metrics['rank_ratio']:.3f}")
    print(f"  Corr ratio: {final_metrics['corr_ratio']:.3f}")
    print(f"  Frobenius: {final_metrics['frob_dist']:.2f}")
    print(f"  KS pass: {final_metrics['ks_n_pass']}/25 (mean D={final_metrics['ks_mean']:.3f})")
    print(f"  PC1 alignment: {final_metrics['pc1_alignment']:.3f}")
    print(f"  PC2 alignment: {final_metrics['pc2_alignment']:.3f}")
    print(f"  Gen mean: {final_metrics['gen_mean']:.3f}, GT mean: {final_metrics['gt_mean']:.3f}")
    print(f"  Gen std: {final_metrics['gen_std']:.3f}, GT std: {final_metrics['gt_std']:.3f}")

    # Save evaluation
    final_metrics["best_val_loss"] = best_val_loss
    with open(f"{args.output_dir}/eval_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nModels saved to {args.output_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
