#!/usr/bin/env python
"""
Diagnose calm-regime overconfidence and per-cell CI violations.

Key issues identified:
1. Calm regime: median systematically above GT, narrow CI → GT outside band
2. Per-cell coverage hides behind aggregate 88% average
3. Term structure/smile: calm regime shows parallel offset

This script quantifies the problem and identifies root cause.
"""

import dataclasses
import numpy as np
import torch
from scipy import stats as sp_stats

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    normalize_iv,
)

MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
MONEYNESS_LABELS = ["0.70", "0.85", "1.00", "1.15", "1.30"]


def load_model(model_path, device):
    cp = torch.load(model_path, map_location=device, weights_only=False)
    c = cp["config"]
    if dataclasses.is_dataclass(c):
        c = dataclasses.asdict(c)
    config = BlockARConfig(**c)
    model = ConditionalBlockARDDPM(config)
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    return model, config


def main():
    device = "cuda"
    model_path = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
    n_samples = 50
    max_windows = 400

    print("Loading model...")
    model, config = load_model(model_path, device)

    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_start = 4540
    test_surfaces = surfaces[test_start:]

    history_len = config.history_len
    future_len = config.future_len
    total_len = history_len + future_len
    N = len(test_surfaces)
    n_windows = min(max_windows, N - total_len + 1)
    indices = np.linspace(0, N - total_len, n_windows, dtype=int)

    all_history, all_future = [], []
    for idx in indices:
        all_history.append(test_surfaces[idx:idx + history_len])
        all_future.append(test_surfaces[idx + history_len:idx + total_len])

    history_arr = np.stack(all_history)  # (W, 30, 5, 5)
    future_arr = np.stack(all_future)    # (W, 30, 5, 5)

    # vol_of_vol
    mean_iv = history_arr.mean(axis=(-1, -2))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)

    # Generate samples
    print(f"Generating {n_samples} samples for {n_windows} windows...")
    batch_size = 16
    all_samples = []
    for i in range(0, n_windows, batch_size):
        batch_end = min(i + batch_size, n_windows)
        hist = torch.tensor(history_arr[i:batch_end], dtype=torch.float32, device=device)
        hist_norm = normalize_iv(hist)
        with torch.no_grad():
            samples = model.sample(hist_norm, n_samples=n_samples)
        all_samples.append(samples.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {batch_end}/{n_windows}")
    all_samples = np.concatenate(all_samples, axis=0)  # (W, S, 30, 5, 5)

    # Regime masks
    quintiles = np.percentile(vol_of_vol, [0, 20, 40, 60, 80, 100])
    calm_mask = vol_of_vol <= quintiles[1]
    medium_mask = (vol_of_vol > quintiles[2]) & (vol_of_vol <= quintiles[3])
    turb_mask = vol_of_vol >= quintiles[4]

    # ═══════════════════════════════════════════════════════════════
    # 1. Per-cell, per-regime 90% CI coverage
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("1. PER-CELL, PER-REGIME 90% CI COVERAGE (h=1)")
    print(f"{'='*80}")

    for h_idx, h_name in [(0, "h=1"), (6, "h=7"), (29, "h=30")]:
        print(f"\n--- {h_name} ---")
        for mask, regime in [(calm_mask, "CALM"), (medium_mask, "MEDIUM"), (turb_mask, "TURBULENT"), (np.ones(n_windows, bool), "ALL")]:
            gt = future_arr[mask, h_idx]  # (M, 5, 5)
            s = all_samples[mask, :, h_idx]  # (M, S, 5, 5)
            q05 = np.percentile(s, 5, axis=1)
            q95 = np.percentile(s, 95, axis=1)
            covered = (gt >= q05) & (gt <= q95)
            percell_cov = covered.mean(axis=0)  # (5, 5)

            print(f"\n  {regime} ({mask.sum()} windows):")
            for r in range(5):
                row = "  ".join(f"{percell_cov[r,c]:.3f}" for c in range(5))
                print(f"    [{row}]")
            print(f"    Mean: {percell_cov.mean():.3f}  Min: {percell_cov.min():.3f}  Max: {percell_cov.max():.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 2. Per-cell median bias (median - GT), per regime
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("2. PER-CELL MEDIAN BIAS (median - GT) at h=1")
    print(f"{'='*80}")
    print("Positive = model overshoots, Negative = model undershoots")

    for mask, regime in [(calm_mask, "CALM"), (turb_mask, "TURBULENT")]:
        gt = future_arr[mask, 0]  # (M, 5, 5)
        s = all_samples[mask, :, 0]  # (M, S, 5, 5)
        median = np.median(s, axis=1)  # (M, 5, 5)
        bias = (median - gt).mean(axis=0)  # (5, 5)

        print(f"\n  {regime} — Mean bias (×1e3):")
        for r in range(5):
            row = "  ".join(f"{bias[r,c]*1e3:+7.2f}" for c in range(5))
            print(f"    [{row}]")
        print(f"    Mean abs bias: {np.abs(bias).mean()*1e3:.2f}×1e-3")

    # ═══════════════════════════════════════════════════════════════
    # 3. Bias direction analysis: is it always positive in calm?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("3. BIAS DIRECTION: Fraction of windows where median > GT (h=1)")
    print(f"{'='*80}")

    for mask, regime in [(calm_mask, "CALM"), (turb_mask, "TURBULENT")]:
        gt = future_arr[mask, 0]
        s = all_samples[mask, :, 0]
        median = np.median(s, axis=1)
        frac_above = (median > gt).mean(axis=0)

        print(f"\n  {regime} — P(median > GT) per cell:")
        for r in range(5):
            row = "  ".join(f"{frac_above[r,c]:.3f}" for c in range(5))
            print(f"    [{row}]")
        print(f"    Mean: {frac_above.mean():.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 4. CI width analysis: too narrow in calm?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("4. 90% CI WIDTH vs GT DEVIATION at h=1")
    print(f"{'='*80}")
    print("CI width = Q95 - Q5 of scenarios")
    print("GT deviation = |GT - baseline| where baseline = last history day")

    for mask, regime in [(calm_mask, "CALM"), (turb_mask, "TURBULENT")]:
        gt = future_arr[mask, 0]
        baseline = history_arr[mask, -1]  # last day of history
        s = all_samples[mask, :, 0]

        ci_width = (np.percentile(s, 95, axis=1) - np.percentile(s, 5, axis=1)).mean(axis=0)
        gt_dev = np.abs(gt - baseline).mean(axis=0)

        # Ratio: CI width should be > GT deviation for 90% coverage
        ratio = ci_width / gt_dev

        print(f"\n  {regime}:")
        print(f"    CI width (×1e3):")
        for r in range(5):
            row = "  ".join(f"{ci_width[r,c]*1e3:7.2f}" for c in range(5))
            print(f"      [{row}]")
        print(f"    GT deviation (×1e3):")
        for r in range(5):
            row = "  ".join(f"{gt_dev[r,c]*1e3:7.2f}" for c in range(5))
            print(f"      [{row}]")
        print(f"    CI width / GT deviation ratio:")
        for r in range(5):
            row = "  ".join(f"{ratio[r,c]:7.2f}" for c in range(5))
            print(f"      [{row}]")
        print(f"    Mean ratio: {ratio.mean():.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Root cause: baseline anchor bias
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("5. BASELINE ANCHOR ANALYSIS")
    print(f"{'='*80}")
    print("Model generates: exp(z * vol_scale) * baseline")
    print("If baseline is biased predictor of future, median inherits the bias")

    for mask, regime in [(calm_mask, "CALM"), (turb_mask, "TURBULENT")]:
        baseline = history_arr[mask, -1]  # (M, 5, 5)
        gt_h1 = future_arr[mask, 0]      # (M, 5, 5)

        # Baseline bias: does baseline systematically over/undershoot GT?
        baseline_bias = (baseline - gt_h1).mean(axis=0)
        # Also check model median bias
        median = np.median(all_samples[mask, :, 0], axis=1)
        model_bias = (median - gt_h1).mean(axis=0)

        print(f"\n  {regime}:")
        print(f"    Baseline bias (baseline - GT) ×1e3:")
        for r in range(5):
            row = "  ".join(f"{baseline_bias[r,c]*1e3:+7.2f}" for c in range(5))
            print(f"      [{row}]")
        print(f"    Model median bias (median - GT) ×1e3:")
        for r in range(5):
            row = "  ".join(f"{model_bias[r,c]*1e3:+7.2f}" for c in range(5))
            print(f"      [{row}]")
        # Correlation between baseline bias and model bias
        corr = np.corrcoef(baseline_bias.ravel(), model_bias.ravel())[0, 1]
        print(f"    Correlation(baseline_bias, model_bias): {corr:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 6. ACF diagnostic: per-cell, per-regime
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("6. ACF DIAGNOSTIC (lags 1-5)")
    print(f"{'='*80}")

    gt_daily = np.diff(future_arr, axis=1)  # (W, 29, 5, 5)
    gen_daily = np.diff(all_samples[:, 0, :, :, :], axis=1)  # (W, 29, 5, 5) single sample

    # The ACF computation in the management report pooled across windows
    # This creates artificial discontinuities at window boundaries
    # Check if intra-window ACF is different

    # Intra-window ACF at lag 1: corr(day_t, day_{t+1}) within each 29-day window
    print("\n  Intra-window ACF(|daily_change|, lag=1):")
    for label, daily in [("GT", gt_daily), ("Generated", gen_daily)]:
        acf1_vals = []
        for w in range(n_windows):
            for r in range(5):
                for c in range(5):
                    series = np.abs(daily[w, :, r, c])
                    if len(series) > 2:
                        acf1 = np.corrcoef(series[:-1], series[1:])[0, 1]
                        if not np.isnan(acf1):
                            acf1_vals.append(acf1)
        print(f"    {label}: mean={np.mean(acf1_vals):.3f}  std={np.std(acf1_vals):.3f}  median={np.median(acf1_vals):.3f}")

    # Pooled ACF (what the management report showed)
    print("\n  Pooled ACF (concatenated across windows, ATM 6M):")
    r, c = 2, 2
    for label, daily in [("GT", gt_daily), ("Generated", gen_daily)]:
        series = np.abs(daily[:, :, r, c].ravel())
        for lag in [1, 2, 3, 5, 10]:
            acf = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            print(f"    {label} lag={lag}: {acf:.3f}")
        print()


if __name__ == "__main__":
    main()
