"""Diagnostic: Show model conditionality across time, space, and regime.

Generates per-horizon, per-cell, and per-regime uncertainty metrics to
demonstrate the model produces conditional (not unconditional) forecasts.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_conditionality_ar.py \
        --model_path models/backfill/afcrps_90d/best_model.pt \
        --no_ema --max_batches 20 --n_samples 50 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def classify_regime(history: np.ndarray) -> np.ndarray:
    """Classify windows as calm (0) or turb (1) based on vol-of-vol."""
    mean_iv = history.mean(axis=(-1, -2))  # (N, T)
    daily_chg = np.diff(mean_iv, axis=1)
    vov = daily_chg.std(axis=1)
    return vov > np.median(vov)


def load_model(args):
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if isinstance(cfg, dict):
        cfg = SinglePassConfig(**cfg)
    model = SinglePassBlockAR(cfg)
    key = "ema_state_dict" if not args.no_ema and "ema_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[key])
    model.eval().to(args.device)
    return model


def collect_samples(model, args):
    """Collect samples, GT, and regime labels."""
    data = np.load("data/vol_surface_with_ret.npz")
    dataset = VolSurfaceDataset(data["surface"], 30, 30, start_idx=4540)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_samples, all_gt, all_hist = [], [], []

    for i, batch in enumerate(tqdm(loader, desc="Sampling")):
        if i >= args.max_batches:
            break
        history = batch["history"].to(args.device)
        future = batch["future"].to(args.device)

        with torch.no_grad():
            samples = model.sample(history, n_samples=args.n_samples)
        # samples: (B, S, 30, 5, 5) in IV space

        all_samples.append(samples.cpu().numpy())
        all_gt.append(denormalize_iv(future).cpu().numpy())
        all_hist.append(denormalize_iv(history).cpu().numpy())

    samples = np.concatenate(all_samples, axis=0)  # (N, S, 30, 5, 5)
    gt = np.concatenate(all_gt, axis=0)            # (N, 30, 5, 5)
    hist = np.concatenate(all_hist, axis=0)        # (N, 30, 5, 5)
    return samples, gt, hist


def analyze_conditionality(samples, gt, hist, output_dir):
    """Compute and display all conditionality metrics."""
    N, S, T, H, W = samples.shape
    regime = classify_regime(hist)
    n_turb = regime.sum()
    n_calm = (~regime).sum()

    print(f"\nWindows: {N} total, {n_turb} turbulent, {n_calm} calm")

    # =========================================================================
    # 1. PER-HORIZON: ensemble spread and CI coverage
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. TEMPORAL CONDITIONALITY — Per-Horizon Spread & Coverage")
    print("=" * 70)

    ens_std = samples.std(axis=1)  # (N, T, 5, 5)
    mean_spread_per_h = ens_std.mean(axis=(0, 2, 3))  # (T,)

    # 90% CI coverage per horizon
    lo = np.percentile(samples, 5, axis=1)   # (N, T, 5, 5)
    hi = np.percentile(samples, 95, axis=1)  # (N, T, 5, 5)
    covered = (gt >= lo) & (gt <= hi)
    coverage_per_h = covered.mean(axis=(0, 2, 3))  # (T,)

    # CI width per horizon
    ci_width_per_h = (hi - lo).mean(axis=(0, 2, 3))  # (T,)

    # GT std per horizon (across windows)
    gt_std_per_h = gt.std(axis=0).mean(axis=(1, 2))  # (T,)

    horizons = [0, 4, 9, 14, 19, 29]
    print(f"\n{'Horizon':>8} {'Ens Spread':>12} {'CI Width':>10} {'Coverage':>10} {'GT Std':>10} {'Width/GT':>10}")
    print("-" * 65)
    for h in horizons:
        ratio = ci_width_per_h[h] / max(gt_std_per_h[h], 1e-8)
        print(f"  h={h+1:>2}   {mean_spread_per_h[h]:>10.4f}   {ci_width_per_h[h]:>8.4f}   "
              f"{coverage_per_h[h]:>8.1%}   {gt_std_per_h[h]:>8.4f}   {ratio:>8.2f}")

    # Growth ratio
    print(f"\n  Spread growth h30/h1: {mean_spread_per_h[29]/max(mean_spread_per_h[0], 1e-8):.2f}x")
    print(f"  CI width growth h30/h1: {ci_width_per_h[29]/max(ci_width_per_h[0], 1e-8):.2f}x")
    print(f"  GT std growth h30/h1: {gt_std_per_h[29]/max(gt_std_per_h[0], 1e-8):.2f}x")

    # =========================================================================
    # 2. PER-CELL: spread heterogeneity
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. SPATIAL CONDITIONALITY — Per-Cell Spread & Coverage")
    print("=" * 70)

    spread_per_cell = ens_std.mean(axis=(0, 1))  # (5, 5)
    coverage_per_cell = covered.mean(axis=(0, 1))  # (5, 5)

    print("\n  Ensemble spread (mean across horizons and windows):")
    print("  Moneyness →")
    print(f"  {'':>8}", end="")
    for c in range(5):
        print(f"  m={c:>1}  ", end="")
    print()
    for r in range(5):
        print(f"  τ={r:>1}   ", end="")
        for c in range(5):
            print(f" {spread_per_cell[r, c]:.4f}", end="")
        print()

    print(f"\n  Min: {spread_per_cell.min():.4f} at {np.unravel_index(spread_per_cell.argmin(), (5,5))}")
    print(f"  Max: {spread_per_cell.max():.4f} at {np.unravel_index(spread_per_cell.argmax(), (5,5))}")
    print(f"  Ratio max/min: {spread_per_cell.max()/max(spread_per_cell.min(), 1e-8):.2f}x")
    print(f"  CV (std/mean): {spread_per_cell.std()/spread_per_cell.mean():.3f}")

    print("\n  90% CI Coverage per cell:")
    for r in range(5):
        print(f"  τ={r:>1}   ", end="")
        for c in range(5):
            v = coverage_per_cell[r, c]
            mark = "✓" if v >= 0.90 else "✗"
            print(f" {v:.1%}{mark}", end="")
        print()

    # =========================================================================
    # 3. PER-REGIME: turb vs calm spread and coverage
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. REGIME CONDITIONALITY — Turbulent vs Calm")
    print("=" * 70)

    turb_mask = regime
    calm_mask = ~regime

    # Spread per regime per horizon
    turb_spread = ens_std[turb_mask].mean(axis=(0, 2, 3))  # (T,)
    calm_spread = ens_std[calm_mask].mean(axis=(0, 2, 3))

    turb_coverage = covered[turb_mask].mean(axis=(0, 2, 3))
    calm_coverage = covered[calm_mask].mean(axis=(0, 2, 3))

    print(f"\n{'Horizon':>8} {'Turb Spread':>12} {'Calm Spread':>12} {'Ratio':>8} {'Turb Cov':>10} {'Calm Cov':>10}")
    print("-" * 65)
    for h in horizons:
        ratio = turb_spread[h] / max(calm_spread[h], 1e-8)
        print(f"  h={h+1:>2}   {turb_spread[h]:>10.4f}   {calm_spread[h]:>10.4f}   "
              f"{ratio:>6.2f}x   {turb_coverage[h]:>8.1%}   {calm_coverage[h]:>8.1%}")

    # Overall regime summary
    turb_overall = ens_std[turb_mask].mean()
    calm_overall = ens_std[calm_mask].mean()
    turb_cov_all = covered[turb_mask].mean()
    calm_cov_all = covered[calm_mask].mean()
    print(f"\n  Overall turb spread: {turb_overall:.4f}, coverage: {turb_cov_all:.1%}")
    print(f"  Overall calm spread: {calm_overall:.4f}, coverage: {calm_cov_all:.1%}")
    print(f"  Turb/calm spread ratio: {turb_overall/max(calm_overall, 1e-8):.2f}x")

    # =========================================================================
    # 4. PER-REGIME × PER-CELL: interaction
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. REGIME × SPATIAL INTERACTION")
    print("=" * 70)

    turb_spread_cell = ens_std[turb_mask].mean(axis=(0, 1))  # (5, 5)
    calm_spread_cell = ens_std[calm_mask].mean(axis=(0, 1))
    ratio_cell = turb_spread_cell / np.maximum(calm_spread_cell, 1e-8)

    print("\n  Turb/Calm spread ratio per cell:")
    for r in range(5):
        print(f"  τ={r:>1}   ", end="")
        for c in range(5):
            print(f" {ratio_cell[r, c]:>5.2f}x", end="")
        print()
    print(f"\n  Mean ratio: {ratio_cell.mean():.2f}x, range: [{ratio_cell.min():.2f}x, {ratio_cell.max():.2f}x]")

    # =========================================================================
    # 5. PER-WINDOW SPREAD VARIABILITY (shows each window gets different width)
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. PER-WINDOW SPREAD VARIABILITY")
    print("=" * 70)

    # Mean spread per window (across cells and horizons)
    window_spread = ens_std.mean(axis=(1, 2, 3))  # (N,)
    print(f"\n  Per-window mean spread: min={window_spread.min():.4f}, max={window_spread.max():.4f}")
    print(f"  Ratio max/min: {window_spread.max()/max(window_spread.min(), 1e-8):.2f}x")
    print(f"  Std of window spreads: {window_spread.std():.4f}")
    print(f"  CV: {window_spread.std()/window_spread.mean():.3f}")

    # Percentiles
    pcts = [5, 25, 50, 75, 95]
    vals = np.percentile(window_spread, pcts)
    print(f"  Percentiles: " + ", ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, vals)))

    # =========================================================================
    # PLOTS
    # =========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Spread and coverage vs horizon
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(range(1, T+1), mean_spread_per_h, 'b-', label='Model spread')
    ax.plot(range(1, T+1), gt_std_per_h, 'r--', label='GT std')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Spread (IV pts)')
    ax.set_title('Temporal: Spread Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, T+1), coverage_per_h * 100, 'b-')
    ax.axhline(90, color='r', linestyle='--', alpha=0.5, label='90% target')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Temporal: CI Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 100)

    ax = axes[2]
    for h_idx, h_label in zip([0, 9, 29], ['h=1', 'h=10', 'h=30']):
        ax.plot(range(1, T+1)[:1], [0], alpha=0)  # dummy
    turb_h = [turb_spread[h] for h in range(T)]
    calm_h = [calm_spread[h] for h in range(T)]
    ax.plot(range(1, T+1), turb_h, 'r-', label='Turbulent', linewidth=2)
    ax.plot(range(1, T+1), calm_h, 'b-', label='Calm', linewidth=2)
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Spread (IV pts)')
    ax.set_title('Regime: Spread Separation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'conditionality_temporal.png', dpi=150)
    plt.close()

    # Plot 2: Spatial spread heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im = axes[0].imshow(spread_per_cell, cmap='YlOrRd', aspect='auto')
    axes[0].set_title('Overall Spread')
    plt.colorbar(im, ax=axes[0])

    im = axes[1].imshow(turb_spread_cell, cmap='YlOrRd', aspect='auto',
                         vmin=spread_per_cell.min(), vmax=turb_spread_cell.max())
    axes[1].set_title('Turbulent Spread')
    plt.colorbar(im, ax=axes[1])

    im = axes[2].imshow(calm_spread_cell, cmap='YlOrRd', aspect='auto',
                         vmin=spread_per_cell.min(), vmax=turb_spread_cell.max())
    axes[2].set_title('Calm Spread')
    plt.colorbar(im, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('Moneyness')
        ax.set_ylabel('Tenor')
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))

    plt.tight_layout()
    plt.savefig(output_dir / 'conditionality_spatial.png', dpi=150)
    plt.close()

    # Plot 3: Per-window spread distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    turb_ws = window_spread[turb_mask]
    calm_ws = window_spread[calm_mask]
    ax.hist(calm_ws, bins=30, alpha=0.6, label=f'Calm (n={len(calm_ws)})', color='blue')
    ax.hist(turb_ws, bins=30, alpha=0.6, label=f'Turb (n={len(turb_ws)})', color='red')
    ax.set_xlabel('Mean Ensemble Spread (IV pts)')
    ax.set_ylabel('Count')
    ax.set_title('Per-Window Spread Distribution by Regime')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'conditionality_window_spread.png', dpi=150)
    plt.close()

    # Plot 4: Example fan charts — one turb, one calm window
    turb_indices = np.where(turb_mask)[0]
    calm_indices = np.where(calm_mask)[0]

    # Pick windows with median spread for each regime
    turb_spreads = window_spread[turb_indices]
    calm_spreads = window_spread[calm_indices]
    turb_pick = turb_indices[np.argsort(turb_spreads)[len(turb_spreads)//2]]
    calm_pick = calm_indices[np.argsort(calm_spreads)[len(calm_spreads)//2]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cell = (2, 2)  # ATM mid-tenor

    for ax, idx, label in zip(axes, [turb_pick, calm_pick], ['Turbulent', 'Calm']):
        cell_samples = samples[idx, :, :, cell[0], cell[1]]  # (S, T)
        cell_gt = gt[idx, :, cell[0], cell[1]]  # (T,)

        # History
        cell_hist = hist[idx, :, cell[0], cell[1]]  # (30,)
        ax.plot(range(-29, 1), cell_hist, 'k-', linewidth=1.5, label='History')

        # Fan chart
        pcts_plot = [5, 10, 25, 50, 75, 90, 95]
        colors = ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7']
        for i in range(len(pcts_plot) - 1):
            lo_p = np.percentile(cell_samples, pcts_plot[i], axis=0)
            hi_p = np.percentile(cell_samples, pcts_plot[i+1], axis=0)
            ax.fill_between(range(1, T+1), lo_p, hi_p, color=colors[i], alpha=0.8)

        median = np.median(cell_samples, axis=0)
        ax.plot(range(1, T+1), median, 'b-', linewidth=1, label='Median')
        ax.plot(range(1, T+1), cell_gt, 'r-', linewidth=1.5, label='Ground Truth')
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{label} Window (idx={idx})')
        ax.set_xlabel('Day')
        ax.set_ylabel('IV')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Fan Charts — Cell ({cell[0]},{cell[1]}) ATM Mid-Tenor', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'conditionality_fan_charts.png', dpi=150)
    plt.close()

    print(f"\n  Plots saved to {output_dir}/")

    # Save summary JSON
    summary = {
        "n_windows": int(N),
        "n_samples": int(S),
        "n_turb": int(n_turb),
        "n_calm": int(n_calm),
        "temporal": {
            "spread_h1": float(mean_spread_per_h[0]),
            "spread_h30": float(mean_spread_per_h[29]),
            "spread_growth": float(mean_spread_per_h[29] / max(mean_spread_per_h[0], 1e-8)),
            "coverage_h1": float(coverage_per_h[0]),
            "coverage_h30": float(coverage_per_h[29]),
        },
        "spatial": {
            "spread_min": float(spread_per_cell.min()),
            "spread_max": float(spread_per_cell.max()),
            "spread_ratio": float(spread_per_cell.max() / max(spread_per_cell.min(), 1e-8)),
            "spread_cv": float(spread_per_cell.std() / spread_per_cell.mean()),
        },
        "regime": {
            "turb_spread": float(turb_overall),
            "calm_spread": float(calm_overall),
            "spread_ratio": float(turb_overall / max(calm_overall, 1e-8)),
            "turb_coverage": float(turb_cov_all),
            "calm_coverage": float(calm_cov_all),
        },
        "per_window": {
            "spread_min": float(window_spread.min()),
            "spread_max": float(window_spread.max()),
            "spread_ratio": float(window_spread.max() / max(window_spread.min(), 1e-8)),
            "spread_cv": float(window_spread.std() / window_spread.mean()),
        },
    }
    with open(output_dir / "conditionality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/block_ar/conditionality_90d")
    args = parser.parse_args()

    model = load_model(args)
    samples, gt, hist = collect_samples(model, args)
    analyze_conditionality(samples, gt, hist, Path(args.output_dir))


if __name__ == "__main__":
    main()
