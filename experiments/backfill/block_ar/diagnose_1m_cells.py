"""Diagnose 1M (row 0) high-error cells for Exp 90d + quantile mapping.

Computes per-cell MAE by horizon, worst windows analysis, range comparison,
bias direction, and example path plots.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_1m_cells.py \
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
OUTPUT_DIR = "results/block_ar/exp90d_1m_diagnosis"


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


def collect_data(model, args):
    """Collect samples, GT, and history for test set."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_samples, all_gt, all_hist = [], [], []
    max_batches = 20

    for i, batch in enumerate(tqdm(loader, desc="Sampling", total=max_batches)):
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

    # Apply quantile mapping if requested
    if args.quantile_map:
        qmapper = QuantileMapper(args.quantile_map, alpha=args.qmap_alpha)
        samples = qmapper.apply(samples, hist)

    return samples, gt, hist


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

    print("Loading model and generating samples...")
    samples, gt, hist = collect_data(model := load_model(args), args)
    N, S, T, H, W = samples.shape
    regime = classify_regime(hist)
    median_samples = np.median(samples, axis=1)  # (N, T, 5, 5)

    print(f"Data: {N} windows, {S} samples, {T} horizons")
    print(f"Regime: {regime.sum()} turbulent, {N - regime.sum()} calm")

    row = 0  # 1M tenor
    horizons = {"h=1": 0, "h=7": 6, "h=14": 13, "h=30": 29}

    # ═══════════════════════════════════════════════════════════════
    # 1. Per-cell MAE at each horizon
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. PER-CELL MAE AT EACH HORIZON (row 0 = 1M tenor)")
    print("=" * 70)
    print(f"{'Cell':<15}", end="")
    for h_name in horizons:
        print(f"{h_name:>10}", end="")
    print(f"{'All':>10}")
    print("-" * 65)

    for c in range(5):
        print(f"  (0,{c}) {MONEYNESS_LABELS[c]:<8}", end="")
        for h_name, h_idx in horizons.items():
            mae = np.abs(median_samples[:, h_idx, row, c] - gt[:, h_idx, row, c]).mean()
            print(f"{mae:>9.4f}", end=" ")
        mae_all = np.abs(median_samples[:, :, row, c] - gt[:, :, row, c]).mean()
        print(f"{mae_all:>9.4f}")

    # Also show as percentage of GT mean
    print()
    print("As % of GT mean IV:")
    print(f"{'Cell':<15}", end="")
    for h_name in horizons:
        print(f"{h_name:>10}", end="")
    print(f"{'All':>10}")
    print("-" * 65)
    for c in range(5):
        print(f"  (0,{c}) {MONEYNESS_LABELS[c]:<8}", end="")
        for h_name, h_idx in horizons.items():
            mae = np.abs(median_samples[:, h_idx, row, c] - gt[:, h_idx, row, c]).mean()
            gt_mean = gt[:, h_idx, row, c].mean()
            print(f"{mae/gt_mean:>9.1%}", end=" ")
        mae_all = np.abs(median_samples[:, :, row, c] - gt[:, :, row, c]).mean()
        gt_mean_all = gt[:, :, row, c].mean()
        print(f"{mae_all/gt_mean_all:>9.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 2. Worst 10 windows by MAE — what do they have in common?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. WORST 10 WINDOWS BY MAE (per cell)")
    print("=" * 70)

    for c in range(5):
        per_window_mae = np.abs(median_samples[:, :, row, c] - gt[:, :, row, c]).mean(axis=1)
        worst_idx = np.argsort(per_window_mae)[-10:][::-1]

        print(f"\n  Cell (0,{c}) {MONEYNESS_LABELS[c]}:")
        print(f"  {'Rank':<6} {'Win':>5} {'MAE':>8} {'Regime':>8} {'GT h1':>7} {'GT h30':>7} "
              f"{'Δ(GT)':>7} {'Hist[-1]':>8} {'Med h30':>8} {'Bias':>8}")
        print("  " + "-" * 85)

        n_turb = 0
        n_gt_drop = 0
        n_bias_pos = 0
        for rank, wi in enumerate(worst_idx):
            mae_val = per_window_mae[wi]
            reg = "turb" if regime[wi] else "calm"
            if regime[wi]:
                n_turb += 1
            gt_h1 = gt[wi, 0, row, c]
            gt_h30 = gt[wi, 29, row, c]
            delta_gt = gt_h30 - gt_h1
            if delta_gt < 0:
                n_gt_drop += 1
            hist_last = hist[wi, -1, row, c]
            med_h30 = median_samples[wi, 29, row, c]
            bias = med_h30 - gt_h30
            if bias > 0:
                n_bias_pos += 1
            print(f"  {rank+1:<6} {wi:>5} {mae_val:>8.4f} {reg:>8} {gt_h1:>7.3f} {gt_h30:>7.3f} "
                  f"{delta_gt:>+7.3f} {hist_last:>8.3f} {med_h30:>8.3f} {bias:>+8.3f}")

        print(f"  Summary: {n_turb}/10 turbulent, {n_gt_drop}/10 GT drops, {n_bias_pos}/10 model above GT")

    # ═══════════════════════════════════════════════════════════════
    # 3. GT range vs generated range
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. GT RANGE vs GENERATED RANGE (across all windows)")
    print("=" * 70)
    print(f"{'Cell':<15} {'GT min':>8} {'GT max':>8} {'GT rng':>8} "
          f"{'Gen min':>8} {'Gen max':>8} {'Gen rng':>8} {'Ratio':>8}")
    print("-" * 80)

    for c in range(5):
        gt_vals = gt[:, :, row, c].ravel()
        # Use 1st and 99th percentile to avoid outlier influence
        gt_lo, gt_hi = np.percentile(gt_vals, [1, 99])
        gen_vals = samples[:, :, :, row, c].ravel()
        gen_lo, gen_hi = np.percentile(gen_vals, [1, 99])
        gt_rng = gt_hi - gt_lo
        gen_rng = gen_hi - gen_lo
        print(f"  (0,{c}) {MONEYNESS_LABELS[c]:<8} {gt_lo:>8.3f} {gt_hi:>8.3f} {gt_rng:>8.3f} "
              f"{gen_lo:>8.3f} {gen_hi:>8.3f} {gen_rng:>8.3f} {gen_rng/gt_rng:>7.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 4. Bias direction
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. BIAS DIRECTION (median prediction - GT)")
    print("=" * 70)

    print(f"\n{'Cell':<15}", end="")
    for h_name in horizons:
        print(f"{h_name:>10}", end="")
    print(f"{'All':>10}")
    print("-" * 65)

    for c in range(5):
        print(f"  (0,{c}) {MONEYNESS_LABELS[c]:<8}", end="")
        for h_name, h_idx in horizons.items():
            bias = (median_samples[:, h_idx, row, c] - gt[:, h_idx, row, c]).mean()
            print(f"{bias:>+9.4f}", end=" ")
        bias_all = (median_samples[:, :, row, c] - gt[:, :, row, c]).mean()
        print(f"{bias_all:>+9.4f}")

    # Regime split
    for regime_name, mask in [("Calm", ~regime), ("Turbulent", regime)]:
        print(f"\n  {regime_name} only:")
        print(f"  {'Cell':<15}", end="")
        for h_name in horizons:
            print(f"{h_name:>10}", end="")
        print()
        for c in range(5):
            print(f"    (0,{c}) {MONEYNESS_LABELS[c]:<8}", end="")
            for h_name, h_idx in horizons.items():
                bias = (median_samples[mask, h_idx, row, c] - gt[mask, h_idx, row, c]).mean()
                print(f"{bias:>+9.4f}", end=" ")
            print()

    # Fraction of windows where model is above GT
    print(f"\n  Fraction of windows where model ABOVE GT at h=30:")
    for c in range(5):
        above = (median_samples[:, 29, row, c] > gt[:, 29, row, c]).mean()
        print(f"    (0,{c}) {MONEYNESS_LABELS[c]}: {above:.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Example path plots — 3 worst windows per cell
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. EXAMPLE PATH PLOTS (saving to", OUTPUT_DIR, ")")
    print("=" * 70)

    for c in range(5):
        per_window_mae = np.abs(median_samples[:, :, row, c] - gt[:, :, row, c]).mean(axis=1)
        worst_idx = np.argsort(per_window_mae)[-3:][::-1]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Cell (0,{c}) — 1M / {MONEYNESS_LABELS[c]}: 3 Worst-Error Windows",
                     fontsize=14, fontweight="bold")

        for ax_idx, wi in enumerate(worst_idx):
            ax = axes[ax_idx]

            # History
            hist_days = np.arange(-29, 1)
            hist_vals = hist[wi, :, row, c]
            ax.plot(hist_days, hist_vals, "k-", linewidth=1.5, label="History")

            # GT future
            fwd_days = np.arange(1, 31)
            gt_vals = gt[wi, :, row, c]
            ax.plot(fwd_days, gt_vals, "r-", linewidth=2.0, label="GT", zorder=5)

            # Generated members (10 random)
            rng = np.random.RandomState(42)
            member_idx = rng.choice(S, 10, replace=False)
            for mi in member_idx:
                ax.plot(fwd_days, samples[wi, mi, :, row, c],
                        color="#4488CC", alpha=0.3, linewidth=0.8)
            ax.plot([], [], color="#4488CC", alpha=0.5, linewidth=1, label="Generated (10)")

            # Median
            med = median_samples[wi, :, row, c]
            ax.plot(fwd_days, med, "b--", linewidth=1.5, label="Median", zorder=4)

            # Vertical line at forecast boundary
            ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)

            mae_val = per_window_mae[wi]
            reg = "TURB" if regime[wi] else "CALM"
            ax.set_title(f"Window {wi} ({reg}) — MAE={mae_val:.4f}", fontsize=11)
            ax.set_xlabel("Day (0 = forecast start)")
            if ax_idx == 0:
                ax.set_ylabel("IV")
                ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = Path(OUTPUT_DIR) / f"worst_paths_cell_0_{c}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ═══════════════════════════════════════════════════════════════
    # Bonus: aggregated per-cell CI coverage for row 0
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BONUS: PER-CELL 90% CI COVERAGE (row 0)")
    print("=" * 70)
    print(f"{'Cell':<15}", end="")
    for h_name in horizons:
        print(f"{h_name:>10}", end="")
    print(f"{'All':>10}")
    print("-" * 65)

    for c in range(5):
        print(f"  (0,{c}) {MONEYNESS_LABELS[c]:<8}", end="")
        for h_name, h_idx in horizons.items():
            lo = np.percentile(samples[:, :, h_idx, row, c], 5, axis=1)
            hi = np.percentile(samples[:, :, h_idx, row, c], 95, axis=1)
            covered = ((gt[:, h_idx, row, c] >= lo) & (gt[:, h_idx, row, c] <= hi)).mean()
            print(f"{covered:>9.1%}", end=" ")
        # All horizons
        lo_all = np.percentile(samples[:, :, :, row, c], 5, axis=1)
        hi_all = np.percentile(samples[:, :, :, row, c], 95, axis=1)
        covered_all = ((gt[:, :, row, c] >= lo_all) & (gt[:, :, row, c] <= hi_all)).mean()
        print(f"{covered_all:>9.1%}")


if __name__ == "__main__":
    main()
