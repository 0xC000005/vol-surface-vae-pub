"""Diagnose per-cell delta bias: calm vs turbulent, GT vs generated.

Checks whether the model learns unconditional downward drift (cause 2)
or the bias loss suppresses regime-specific deltas (cause 1).

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_delta_bias.py \
        --model_path models/backfill/afcrps_90d/best_model.pt \
        --no_ema --quantile_map models/backfill/afcrps_90d/quantile_map.npz \
        --qmap_alpha 0.3 --device cuda
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig, denormalize_iv, normalize_iv
from experiments.backfill.block_ar.quantile_mapper import QuantileMapper
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

MONEYNESS_LABELS = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
OUTPUT_DIR = "results/block_ar/exp90d_delta_bias"


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


def classify_regime(hist):
    mean_iv = hist.mean(axis=(-1, -2))
    daily_chg = np.diff(mean_iv, axis=1)
    vov = daily_chg.std(axis=1)
    return vov > np.median(vov)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--quantile_map", type=str, default=None)
    parser.add_argument("--qmap_alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model = load_model(args)

    # ── Load test data ──
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_samples, all_gt, all_hist = [], [], []
    max_batches = 20
    for i, batch in enumerate(tqdm(loader, desc="Sampling test", total=max_batches)):
        if i >= max_batches:
            break
        history = batch["history"].to(args.device)
        future = batch["future"].to(args.device)
        with torch.no_grad():
            samples = model.sample(history, n_samples=50)
        all_samples.append(samples.cpu().numpy())
        all_gt.append(denormalize_iv(future).cpu().numpy())
        all_hist.append(denormalize_iv(history).cpu().numpy())

    samples = np.concatenate(all_samples, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    hist = np.concatenate(all_hist, axis=0)

    if args.quantile_map:
        qmapper = QuantileMapper(args.quantile_map, alpha=args.qmap_alpha)
        samples = qmapper.apply(samples, hist)

    N, S, T, H, W = samples.shape
    regime = classify_regime(hist)

    # ── GT daily deltas from test future ──
    gt_deltas = np.diff(gt, axis=1)  # (N, 29, 5, 5)
    # Prepend delta from history[-1] to future[0]
    first_delta = gt[:, 0:1] - hist[:, -1:]  # (N, 1, 5, 5)
    gt_deltas = np.concatenate([first_delta, gt_deltas], axis=1)  # (N, 30, 5, 5)

    # ── Generated deltas (median trajectory) ──
    median_traj = np.median(samples, axis=1)  # (N, 30, 5, 5)
    first_gen_delta = median_traj[:, 0:1] - hist[:, -1:]
    gen_deltas = np.diff(median_traj, axis=1)
    gen_deltas = np.concatenate([first_gen_delta, gen_deltas], axis=1)  # (N, 30, 5, 5)

    # ── Also compute per-sample deltas for spread analysis ──
    anchor = hist[:, -1:]  # (N, 1, 5, 5)
    anchor_exp = np.broadcast_to(anchor[:, np.newaxis], (N, S, 1, H, W)).copy()
    full_traj = np.concatenate([anchor_exp, samples], axis=2)
    sample_deltas = np.diff(full_traj, axis=2)  # (N, S, 30, 5, 5)

    # ═══════════════════════════════════════════════════════════════
    # 1. Mean delta per cell: GT calm vs turb, Gen calm vs turb
    # ═══════════════════════════════════════════════════════════════
    print("=" * 80)
    print("1. MEAN DAILY DELTA PER CELL: GT vs GENERATED, CALM vs TURBULENT")
    print("=" * 80)
    print(f"   Test windows: {N} ({regime.sum()} turb, {(~regime).sum()} calm)")
    print()

    print("GT mean delta (×1000):")
    print(f"{'':>15}", end="")
    for c in range(5):
        print(f"  {MONEYNESS_LABELS[c]:>8}", end="")
    print()
    for label, mask in [("All", np.ones(N, bool)), ("Calm", ~regime), ("Turb", regime)]:
        print(f"  {label:<12}", end="")
        for c in range(5):
            # Row 0 only for focus, but print all rows below
            val = gt_deltas[mask, :, 0, c].mean() * 1000
            print(f"  {val:>+8.3f}", end="")
        print()

    print()
    print("Generated mean delta (×1000):")
    print(f"{'':>15}", end="")
    for c in range(5):
        print(f"  {MONEYNESS_LABELS[c]:>8}", end="")
    print()
    for label, mask in [("All", np.ones(N, bool)), ("Calm", ~regime), ("Turb", regime)]:
        print(f"  {label:<12}", end="")
        for c in range(5):
            val = gen_deltas[mask, :, 0, c].mean() * 1000
            print(f"  {val:>+8.3f}", end="")
        print()

    print()
    print("Delta bias = Gen - GT mean delta (×1000):")
    print(f"{'':>15}", end="")
    for c in range(5):
        print(f"  {MONEYNESS_LABELS[c]:>8}", end="")
    print()
    for label, mask in [("All", np.ones(N, bool)), ("Calm", ~regime), ("Turb", regime)]:
        print(f"  {label:<12}", end="")
        for c in range(5):
            gt_val = gt_deltas[mask, :, 0, c].mean() * 1000
            gen_val = gen_deltas[mask, :, 0, c].mean() * 1000
            print(f"  {gen_val - gt_val:>+8.3f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════════
    # 2. Full 5×5 grid: all rows, calm vs turb
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("2. FULL 5×5 GRID: GT MEAN DELTA, CALM vs TURB (×1000)")
    print("=" * 80)

    for label, mask in [("GT Calm", ~regime), ("GT Turb", regime),
                        ("Gen Calm", ~regime), ("Gen Turb", regime)]:
        print(f"\n  {label}:")
        print(f"  {'':>8}", end="")
        for c in range(5):
            print(f"  {MONEYNESS_LABELS[c]:>8}", end="")
        print()
        for r in range(5):
            print(f"  {MATURITY_LABELS[r]:>6}  ", end="")
            for c in range(5):
                if "GT" in label:
                    val = gt_deltas[mask, :, r, c].mean() * 1000
                else:
                    val = gen_deltas[mask, :, r, c].mean() * 1000
                print(f"  {val:>+8.3f}", end="")
            print()

    # ═══════════════════════════════════════════════════════════════
    # 3. Per-horizon delta for row 0: does sign flip?
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("3. PER-HORIZON MEAN DELTA FOR ROW 0 (×1000): EARLY vs LATE")
    print("=" * 80)

    for c in range(5):
        print(f"\n  Cell (0,{c}) {MONEYNESS_LABELS[c]}:")
        print(f"  {'':>12} {'h=1-5':>10} {'h=6-15':>10} {'h=16-30':>10} {'All':>10}")
        for label, mask in [("GT Calm", ~regime), ("GT Turb", regime),
                            ("Gen Calm", ~regime), ("Gen Turb", regime)]:
            vals = gt_deltas[mask, :, 0, c] if "GT" in label else gen_deltas[mask, :, 0, c]
            early = vals[:, :5].mean() * 1000
            mid = vals[:, 5:15].mean() * 1000
            late = vals[:, 15:].mean() * 1000
            all_h = vals.mean() * 1000
            print(f"  {label:<12} {early:>+10.3f} {mid:>+10.3f} {late:>+10.3f} {all_h:>+10.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 4. Per-sample delta spread: are turb samples asymmetric?
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("4. SAMPLE DELTA SPREAD: MEAN OF 50-SAMPLE DELTAS (×1000)")
    print("   (Do ALL samples drift the same direction?)")
    print("=" * 80)

    for c in range(5):
        # Per-window mean delta across all samples
        # sample_deltas: (N, S, 30, 5, 5)
        per_sample_mean = sample_deltas[:, :, :, 0, c].mean(axis=2)  # (N, S) mean over time
        per_window_spread = per_sample_mean.std(axis=1)  # (N,) spread across samples
        per_window_mean = per_sample_mean.mean(axis=1)  # (N,) mean across samples

        print(f"\n  Cell (0,{c}) {MONEYNESS_LABELS[c]}:")
        print(f"  {'':>12} {'Mean δ':>10} {'Spread':>10} {'% positive':>12}")
        for label, mask in [("Calm", ~regime), ("Turb", regime)]:
            mean_d = per_window_mean[mask].mean() * 1000
            spread_d = per_window_spread[mask].mean() * 1000
            # Fraction of individual samples with positive mean delta
            pct_pos = (per_sample_mean[mask] > 0).mean()
            print(f"  {label:<12} {mean_d:>+10.3f} {spread_d:>10.3f} {pct_pos:>11.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Plot: GT vs Gen mean delta by regime, per cell (row 0)
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle("Row 0 (1M): GT vs Generated Mean Daily Delta by Regime", fontsize=14)

    for c in range(5):
        ax = axes[c]
        # GT
        gt_calm = gt_deltas[~regime, :, 0, c].mean(axis=0) * 1000  # (30,)
        gt_turb = gt_deltas[regime, :, 0, c].mean(axis=0) * 1000
        gen_calm = gen_deltas[~regime, :, 0, c].mean(axis=0) * 1000
        gen_turb = gen_deltas[regime, :, 0, c].mean(axis=0) * 1000

        days = np.arange(1, 31)
        ax.plot(days, gt_calm, "b-", linewidth=1.5, label="GT Calm")
        ax.plot(days, gt_turb, "r-", linewidth=1.5, label="GT Turb")
        ax.plot(days, gen_calm, "b--", linewidth=1.5, label="Gen Calm")
        ax.plot(days, gen_turb, "r--", linewidth=1.5, label="Gen Turb")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_title(f"(0,{c}) {MONEYNESS_LABELS[c]}")
        ax.set_xlabel("Horizon (day)")
        if c == 0:
            ax.set_ylabel("Mean Δ (×1000)")
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(OUTPUT_DIR) / "delta_by_regime_row0.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    # ═══════════════════════════════════════════════════════════════
    # 6. Plot: full 5×5 heatmap of GT vs Gen mean delta
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    titles = ["GT Calm", "GT Turb", "Gen Calm", "Gen Turb"]
    masks = [~regime, regime, ~regime, regime]
    sources = ["gt", "gt", "gen", "gen"]

    vmax = 0
    grids = []
    for mask, src in zip(masks, sources):
        grid = np.zeros((5, 5))
        for r in range(5):
            for c in range(5):
                if src == "gt":
                    grid[r, c] = gt_deltas[mask, :, r, c].mean() * 1000
                else:
                    grid[r, c] = gen_deltas[mask, :, r, c].mean() * 1000
        grids.append(grid)
        vmax = max(vmax, np.abs(grid).max())

    for idx, (ax, title, grid) in enumerate(zip(axes.ravel(), titles, grids)):
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(MONEYNESS_LABELS, fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels(MATURITY_LABELS, fontsize=9)
        ax.set_title(title, fontsize=13)
        for r in range(5):
            for c in range(5):
                ax.text(c, r, f"{grid[r,c]:+.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Mean Daily Delta (×1000): GT vs Generated by Regime", fontsize=14, y=1.01)
    plt.tight_layout()
    path = Path(OUTPUT_DIR) / "delta_heatmap_5x5.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ═══════════════════════════════════════════════════════════════
    # 7. Diagnosis summary
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    for c in range(5):
        gt_calm_d = gt_deltas[~regime, :, 0, c].mean() * 1000
        gt_turb_d = gt_deltas[regime, :, 0, c].mean() * 1000
        gen_calm_d = gen_deltas[~regime, :, 0, c].mean() * 1000
        gen_turb_d = gen_deltas[regime, :, 0, c].mean() * 1000

        gt_sign_calm = "+" if gt_calm_d > 0 else "-"
        gt_sign_turb = "+" if gt_turb_d > 0 else "-"
        gen_sign_calm = "+" if gen_calm_d > 0 else "-"
        gen_sign_turb = "+" if gen_turb_d > 0 else "-"

        sign_match_calm = "✓" if (gt_calm_d > 0) == (gen_calm_d > 0) else "✗"
        sign_match_turb = "✓" if (gt_turb_d > 0) == (gen_turb_d > 0) else "✗"

        print(f"\n  Cell (0,{c}) {MONEYNESS_LABELS[c]}:")
        print(f"    GT  calm={gt_calm_d:>+.3f}  turb={gt_turb_d:>+.3f}")
        print(f"    Gen calm={gen_calm_d:>+.3f}  turb={gen_turb_d:>+.3f}")
        print(f"    Sign match: calm={sign_match_calm}  turb={sign_match_turb}")

        if (gt_turb_d > 0) and (gen_turb_d < 0):
            print(f"    >>> CAUSE 2: Model learned UNCONDITIONAL downward drift")
            print(f"        GT turb delta is POSITIVE but model produces NEGATIVE")
        elif (gt_turb_d > 0) and (gen_turb_d > 0) and (gen_turb_d < gt_turb_d * 0.5):
            print(f"    >>> CAUSE 1: Bias loss SUPPRESSING positive turb deltas")
            print(f"        Gen turb delta is positive but < 50% of GT")
        elif (gt_calm_d < 0) and (gen_calm_d < 0) and (gt_turb_d > 0) and (gen_turb_d < 0):
            print(f"    >>> CAUSE 2: Model learned calm-regime drift, applies to turb too")


if __name__ == "__main__":
    main()
