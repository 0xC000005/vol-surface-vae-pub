#!/usr/bin/env python
"""
Compute ground truth Q5/Q1 ratio for cross-sample spread.

Groups real data windows by vol_of_vol quintile, computes the std of
future mean-IV values within each quintile at each horizon, and reports
Q5/Q1 ratio. This gives the "population-level" ground truth that models
should match.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/compute_gt_q5q1.py
"""

import numpy as np
from scipy.stats import spearmanr


def main():
    # --- 1. Load data ---
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5), raw IV in [0, 1]
    print(f"Total surfaces: {len(surfaces)}")

    # --- 2. Test set ---
    test_start = 4540
    test_surfaces = surfaces[test_start:]
    print(f"Test surfaces: {len(test_surfaces)} (from index {test_start})")

    # --- 3. Build sliding windows ---
    history_len = 30
    future_len = 30
    total_len = history_len + future_len

    N = len(test_surfaces)
    max_windows = 400
    n_windows = min(max_windows, N - total_len + 1)
    indices = np.linspace(0, N - total_len, n_windows, dtype=int)

    all_history = []
    all_future = []
    for idx in indices:
        all_history.append(test_surfaces[idx:idx + history_len])
        all_future.append(test_surfaces[idx + history_len:idx + total_len])

    history_arr = np.stack(all_history)  # (n_windows, 30, 5, 5)
    future_arr = np.stack(all_future)    # (n_windows, 30, 5, 5)
    print(f"Windows: {n_windows}")

    # --- 4. Compute vol_of_vol per window ---
    # Mean IV per day across spatial dims: (n_windows, 30)
    hist_mean_iv = history_arr.mean(axis=(-1, -2))
    daily_changes = np.diff(hist_mean_iv, axis=1)  # (n_windows, 29)
    vol_of_vol = daily_changes.std(axis=1)          # (n_windows,)

    # --- 4b. Compute future mean-IV trajectory: (n_windows, 30) ---
    future_mean_iv = future_arr.mean(axis=(-1, -2))  # (n_windows, 30)

    # Baseline: mean IV of last history day
    baseline_iv = hist_mean_iv[:, -1]  # (n_windows,)

    # --- 5. Group by vol_of_vol quintile ---
    quintiles = np.percentile(vol_of_vol, [0, 20, 40, 60, 80, 100])
    q1_mask = vol_of_vol <= quintiles[1]  # bottom 20%
    q5_mask = vol_of_vol >= quintiles[4]  # top 20%

    n_q1 = q1_mask.sum()
    n_q5 = q5_mask.sum()
    print(f"\nQuintile boundaries: {quintiles}")
    print(f"Q1 (bottom 20%): {n_q1} windows, vol_of_vol <= {quintiles[1]:.6f}")
    print(f"Q5 (top 20%):    {n_q5} windows, vol_of_vol >= {quintiles[4]:.6f}")

    # --- 6. Compute STD of future mean-IV within each quintile at each horizon ---
    horizons = [0, 6, 13, 29]  # h=1, h=7, h=14, h=30
    horizon_names = ["h=1", "h=7", "h=14", "h=30"]

    print(f"\n{'='*65}")
    print(f"Ground Truth Q5/Q1 (population-level std of future mean-IV)")
    print(f"{'='*65}")
    print(f"  Grouping variable: vol_of_vol")
    print(f"  Q1 = {n_q1} windows (calm), Q5 = {n_q5} windows (turbulent)\n")

    print(f"  {'Horizon':<10} {'Q1 std':<12} {'Q5 std':<12} {'Q5/Q1':<10}")
    print(f"  {'-'*44}")

    for h_idx, h_name in zip(horizons, horizon_names):
        q1_values = future_mean_iv[q1_mask, h_idx]
        q5_values = future_mean_iv[q5_mask, h_idx]

        q1_std = q1_values.std()
        q5_std = q5_values.std()
        ratio = q5_std / q1_std if q1_std > 0 else float("nan")

        print(f"  {h_name:<10} {q1_std:<12.6f} {q5_std:<12.6f} {ratio:<10.3f}x")

    # --- 7. Full horizon profile ---
    print(f"\n  Full horizon profile (Q5/Q1 at every horizon):")
    print(f"  {'h':<6} {'Q5/Q1':<10}")
    print(f"  {'-'*16}")
    for h in range(future_len):
        q1_std = future_mean_iv[q1_mask, h].std()
        q5_std = future_mean_iv[q5_mask, h].std()
        ratio = q5_std / q1_std if q1_std > 0 else float("nan")
        if h in [0, 1, 2, 4, 6, 9, 13, 19, 29]:
            print(f"  {h+1:<6} {ratio:<10.3f}x")

    # --- 8. Spearman correlation: vol_of_vol vs |future[h] - baseline| ---
    deviation_h1 = np.abs(future_mean_iv[:, 0] - baseline_iv)
    spearman_corr, spearman_p = spearmanr(vol_of_vol, deviation_h1)

    print(f"\n{'='*65}")
    print(f"Spearman correlation: vol_of_vol vs |future_mean_iv[h=1] - baseline|")
    print(f"  rho = {spearman_corr:.4f}, p = {spearman_p:.2e}")
    print(f"{'='*65}")

    # --- Bonus: also check other conditioning variables ---
    recent_change = np.abs(hist_mean_iv[:, -1] - hist_mean_iv[:, -6])

    print(f"\n{'='*65}")
    print(f"Comparison across conditioning variables (Q5/Q1 at h=1, h=30)")
    print(f"{'='*65}")

    for var_name, var_values in [
        ("vol_of_vol", vol_of_vol),
        ("recent_change", recent_change),
        ("baseline_iv", baseline_iv),
    ]:
        q_bounds = np.percentile(var_values, [20, 80])
        mask_q1 = var_values <= q_bounds[0]
        mask_q5 = var_values >= q_bounds[1]

        for h_idx, h_name in [(0, "h=1"), (29, "h=30")]:
            std_q1 = future_mean_iv[mask_q1, h_idx].std()
            std_q5 = future_mean_iv[mask_q5, h_idx].std()
            ratio = std_q5 / std_q1 if std_q1 > 0 else float("nan")

            sp, _ = spearmanr(var_values, np.abs(future_mean_iv[:, h_idx] - baseline_iv))
            print(f"  {var_name:<16} {h_name}: Q5/Q1 = {ratio:.3f}x  "
                  f"(Q5_std={std_q5:.6f}, Q1_std={std_q1:.6f})  "
                  f"Spearman={sp:.3f}")


if __name__ == "__main__":
    main()
