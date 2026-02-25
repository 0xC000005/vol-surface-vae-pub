#!/usr/bin/env python
"""
Ground Truth Uncertainty Investigation.

Investigates three questions about conditional uncertainty in GT data:
1. Per-cell (5x5 grid) conditional variance patterns
2. Uncertainty growth with horizon and mean-reversion plateau
3. Condition-dependent uncertainty by vol regime (high vs calm)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/investigate_gt_uncertainty.py \
        --output_dir results/block_ar/gt_uncertainty_investigation
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr


def compute_conditioning_variables(history):
    """Compute vol_of_vol, recent_change, baseline_iv from history windows."""
    # history: (N, 30, 5, 5)
    mean_iv = history.mean(axis=(-1, -2))  # (N, 30)
    daily_chg = np.diff(mean_iv, axis=1)  # (N, 29)

    vol_of_vol = daily_chg.std(axis=1)
    recent_change = np.abs(mean_iv[:, -1] - mean_iv[:, -6])
    baseline_iv = mean_iv[:, -1]

    return {
        "vol_of_vol": vol_of_vol,
        "recent_change": recent_change,
        "baseline_iv": baseline_iv,
    }


def investigate_per_cell_variance(surfaces, history_len=30, future_len=30):
    """Investigation 1: Per-cell conditional variance patterns."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 1: Per-Cell Conditional Variance Patterns")
    print("=" * 70)

    N = len(surfaces)
    n_windows = N - history_len - future_len + 1

    # Build all windows
    all_history = []
    all_future = []
    for i in range(n_windows):
        all_history.append(surfaces[i : i + history_len])
        all_future.append(surfaces[i + history_len : i + history_len + future_len])

    history = np.stack(all_history)  # (n_windows, 30, 5, 5)
    future = np.stack(all_future)  # (n_windows, 30, 5, 5)

    cond_vars = compute_conditioning_variables(history)

    # Per-cell, per-horizon GT cross-window std
    horizons = [0, 6, 13, 29]
    horizon_names = ["h=1", "h=7", "h=14", "h=30"]

    results = {}

    for h_idx, h_name in zip(horizons, horizon_names):
        # GT std per cell at this horizon: (5, 5)
        cell_stds = future[:, h_idx, :, :].std(axis=0)
        print(f"\n  {h_name} — GT cross-window std per cell:")
        for r in range(5):
            row = "    " + "  ".join(f"{cell_stds[r, c]:.5f}" for c in range(5))
            print(row)

        # Per-cell Q5/Q1 for vol_of_vol
        q20 = np.percentile(cond_vars["vol_of_vol"], 20)
        q80 = np.percentile(cond_vars["vol_of_vol"], 80)
        q1_mask = cond_vars["vol_of_vol"] <= q20
        q5_mask = cond_vars["vol_of_vol"] >= q80

        q5q1_grid = np.zeros((5, 5))
        for r in range(5):
            for c in range(5):
                cell_vals = future[:, h_idx, r, c]
                q1_std = cell_vals[q1_mask].std()
                q5_std = cell_vals[q5_mask].std()
                q5q1_grid[r, c] = q5_std / q1_std if q1_std > 0 else float("nan")

        print(f"\n  {h_name} — GT vol_of_vol Q5/Q1 per cell:")
        for r in range(5):
            row = "    " + "  ".join(f"{q5q1_grid[r, c]:.3f}" for c in range(5))
            print(row)

        results[h_name] = {
            "cell_stds": cell_stds.tolist(),
            "vol_of_vol_q5q1_per_cell": q5q1_grid.tolist(),
        }

    # Spearman per cell for vol_of_vol at h=1
    print(f"\n  h=1 — Spearman(vol_of_vol, cell_value_std) per cell:")
    spearman_grid = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            # Need per-window std for each cell
            # Use local variance: abs deviation from conditional mean
            cell_vals = future[:, 0, r, c]  # (n_windows,)
            # Can't compute cross-sample std for GT (only 1 realization per window)
            # Use absolute deviation from rolling mean as proxy
            rho, _ = spearmanr(cond_vars["vol_of_vol"], np.abs(cell_vals - cell_vals.mean()))
            spearman_grid[r, c] = rho

    for r in range(5):
        row = "    " + "  ".join(f"{spearman_grid[r, c]:+.3f}" for c in range(5))
        print(row)

    results["spearman_vol_of_vol_h1"] = spearman_grid.tolist()

    return results


def investigate_uncertainty_growth(surfaces, history_len=30, future_len=30):
    """Investigation 2: Uncertainty growth with horizon and mean-reversion plateau."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 2: Uncertainty Growth with Horizon")
    print("=" * 70)

    N = len(surfaces)
    n_windows = N - history_len - future_len + 1

    all_history = []
    all_future = []
    for i in range(n_windows):
        all_history.append(surfaces[i : i + history_len])
        all_future.append(surfaces[i + history_len : i + history_len + future_len])

    history = np.stack(all_history)
    future = np.stack(all_future)

    cond_vars = compute_conditioning_variables(history)

    # Cross-window variance at each horizon (mean IV)
    mean_future_iv = future.mean(axis=(-1, -2))  # (n_windows, 30)
    cross_window_std = mean_future_iv.std(axis=0)  # (30,)

    print("\n  GT cross-window std (mean IV) by horizon:")
    for h in range(30):
        marker = " <--" if h in [0, 6, 13, 29] else ""
        print(f"    h={h+1:2d}: std={cross_window_std[h]:.6f}{marker}")

    # Check monotonicity
    increasing = all(cross_window_std[i + 1] >= cross_window_std[i] for i in range(29))
    print(f"\n  Monotonically increasing: {increasing}")

    # Find plateau (where growth rate < 5% of initial growth rate)
    growth_rates = np.diff(cross_window_std)
    initial_growth = growth_rates[0]
    for h in range(len(growth_rates)):
        if growth_rates[h] < 0.05 * initial_growth:
            print(f"  Growth plateaus at h={h + 2} (growth rate < 5% of initial)")
            break
    else:
        print("  No plateau detected in 30-day horizon")

    # Condition-dependent growth: high-vol vs low-vol
    q20 = np.percentile(cond_vars["vol_of_vol"], 20)
    q80 = np.percentile(cond_vars["vol_of_vol"], 80)
    q1_mask = cond_vars["vol_of_vol"] <= q20
    q5_mask = cond_vars["vol_of_vol"] >= q80

    q1_std_curve = mean_future_iv[q1_mask].std(axis=0)
    q5_std_curve = mean_future_iv[q5_mask].std(axis=0)

    print("\n  Uncertainty growth by regime:")
    print(f"  {'Horizon':<10} {'All':<12} {'Q1 (calm)':<12} {'Q5 (turbul)':<12} {'Q5/Q1':<8}")
    results_growth = {}
    for h in [0, 6, 13, 29]:
        ratio = q5_std_curve[h] / q1_std_curve[h] if q1_std_curve[h] > 0 else float("nan")
        print(f"    h={h+1:<6d} {cross_window_std[h]:<12.6f} {q1_std_curve[h]:<12.6f} "
              f"{q5_std_curve[h]:<12.6f} {ratio:<8.3f}")
        results_growth[f"h={h+1}"] = {
            "all_std": float(cross_window_std[h]),
            "q1_std": float(q1_std_curve[h]),
            "q5_std": float(q5_std_curve[h]),
            "q5q1": float(ratio),
        }

    # Also check baseline_iv Q5/Q1 growth
    q20_biv = np.percentile(cond_vars["baseline_iv"], 20)
    q80_biv = np.percentile(cond_vars["baseline_iv"], 80)
    q1_biv = cond_vars["baseline_iv"] <= q20_biv
    q5_biv = cond_vars["baseline_iv"] >= q80_biv

    q1_biv_curve = mean_future_iv[q1_biv].std(axis=0)
    q5_biv_curve = mean_future_iv[q5_biv].std(axis=0)

    print("\n  Uncertainty growth by baseline_iv regime:")
    print(f"  {'Horizon':<10} {'Q1 (low IV)':<12} {'Q5 (high IV)':<12} {'Q5/Q1':<8}")
    for h in [0, 6, 13, 29]:
        ratio = q5_biv_curve[h] / q1_biv_curve[h] if q1_biv_curve[h] > 0 else float("nan")
        print(f"    h={h+1:<6d} {q1_biv_curve[h]:<12.6f} {q5_biv_curve[h]:<12.6f} {ratio:<8.3f}")

    return {
        "cross_window_std": cross_window_std.tolist(),
        "q1_std_curve": q1_std_curve.tolist(),
        "q5_std_curve": q5_std_curve.tolist(),
        "growth_by_horizon": results_growth,
        "monotonic": bool(increasing),
    }


def investigate_regime_uncertainty(surfaces, history_len=30, future_len=30):
    """Investigation 3: Condition-dependent uncertainty by vol regime."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: Condition-Dependent Uncertainty by Vol Regime")
    print("=" * 70)

    N = len(surfaces)
    n_windows = N - history_len - future_len + 1

    all_history = []
    all_future = []
    for i in range(n_windows):
        all_history.append(surfaces[i : i + history_len])
        all_future.append(surfaces[i + history_len : i + history_len + future_len])

    history = np.stack(all_history)
    future = np.stack(all_future)

    cond_vars = compute_conditioning_variables(history)

    # Quintile analysis for each conditioning variable
    results = {}
    for var_name in ["vol_of_vol", "recent_change", "baseline_iv"]:
        var = cond_vars[var_name]
        quintile_bounds = np.percentile(var, [0, 20, 40, 60, 80, 100])

        print(f"\n  --- {var_name} ---")
        print(f"  Quintile bounds: {quintile_bounds.round(5)}")

        var_results = {}
        for qi in range(5):
            mask = (var >= quintile_bounds[qi]) & (var <= quintile_bounds[qi + 1])
            n = mask.sum()

            # Per-horizon cross-window std within this quintile
            qf = future[mask]  # (n_q, 30, 5, 5)
            mean_iv_q = qf.mean(axis=(-1, -2))  # (n_q, 30)
            std_curve = mean_iv_q.std(axis=0)  # (30,)

            horizons = {1: 0, 7: 6, 14: 13, 30: 29}
            stds = {f"h={h}": float(std_curve[idx]) for h, idx in horizons.items()}

            label = f"Q{qi+1}"
            print(f"    {label} (n={n:4d}, range=[{quintile_bounds[qi]:.5f}, {quintile_bounds[qi+1]:.5f}]):")
            for h_name, s in stds.items():
                print(f"      {h_name}: std={s:.6f}")

            var_results[label] = {"n": int(n), "stds": stds}

        # Q5/Q1 at each horizon
        print(f"\n  {var_name} Q5/Q1 ratios:")
        for h_name in ["h=1", "h=7", "h=14", "h=30"]:
            q5_s = var_results["Q5"]["stds"][h_name]
            q1_s = var_results["Q1"]["stds"][h_name]
            ratio = q5_s / q1_s if q1_s > 0 else float("nan")
            print(f"    {h_name}: Q5/Q1 = {ratio:.3f}x (Q5={q5_s:.6f}, Q1={q1_s:.6f})")
            var_results[f"q5q1_{h_name}"] = float(ratio)

        # Spearman at each horizon
        print(f"\n  {var_name} Spearman correlations with per-window spread:")
        for h, h_idx in [(1, 0), (7, 6), (14, 13), (30, 29)]:
            # Per-window "spread" = abs deviation from cross-window mean
            cell_mean = future[:, h_idx, :, :].mean(axis=(-1, -2))  # (n_windows,)
            abs_dev = np.abs(cell_mean - cell_mean.mean())
            rho, p = spearmanr(var, abs_dev)
            print(f"    h={h}: rho={rho:+.3f} (p={p:.4f})")
            var_results[f"spearman_h{h}"] = float(rho)

        results[var_name] = var_results

    # Per-cell analysis for vol_of_vol at h=1
    print("\n  --- Per-Cell vol_of_vol Q5/Q1 at h=1 ---")
    q20 = np.percentile(cond_vars["vol_of_vol"], 20)
    q80 = np.percentile(cond_vars["vol_of_vol"], 80)
    q1_mask = cond_vars["vol_of_vol"] <= q20
    q5_mask = cond_vars["vol_of_vol"] >= q80

    q5q1_grid = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            q1_std = future[q1_mask, 0, r, c].std()
            q5_std = future[q5_mask, 0, r, c].std()
            q5q1_grid[r, c] = q5_std / q1_std if q1_std > 0 else float("nan")

    for r in range(5):
        row = "    " + "  ".join(f"{q5q1_grid[r, c]:.3f}" for c in range(5))
        print(row)

    results["per_cell_vol_of_vol_q5q1_h1"] = q5q1_grid.tolist()

    # Per-cell analysis for baseline_iv at h=1
    print("\n  --- Per-Cell baseline_iv Q5/Q1 at h=1 ---")
    q20_b = np.percentile(cond_vars["baseline_iv"], 20)
    q80_b = np.percentile(cond_vars["baseline_iv"], 80)
    q1_b = cond_vars["baseline_iv"] <= q20_b
    q5_b = cond_vars["baseline_iv"] >= q80_b

    q5q1_biv = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            q1_std = future[q1_b, 0, r, c].std()
            q5_std = future[q5_b, 0, r, c].std()
            q5q1_biv[r, c] = q5_std / q1_std if q1_std > 0 else float("nan")

    for r in range(5):
        row = "    " + "  ".join(f"{q5q1_biv[r, c]:.3f}" for c in range(5))
        print(row)

    results["per_cell_baseline_iv_q5q1_h1"] = q5q1_biv.tolist()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/block_ar/gt_uncertainty_investigation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)

    # Use test set
    test_start = 4540
    test_surfaces = surfaces[test_start:]
    print(f"Test surfaces: {len(test_surfaces)}")

    # Run all three investigations
    results = {}

    r1 = investigate_per_cell_variance(test_surfaces)
    results["per_cell_variance"] = r1

    r2 = investigate_uncertainty_growth(test_surfaces)
    results["uncertainty_growth"] = r2

    r3 = investigate_regime_uncertainty(test_surfaces)
    results["regime_uncertainty"] = r3

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    # Key Q5/Q1 numbers
    print("\n  GT vol_of_vol Q5/Q1 by horizon (mean IV):")
    for h_name in ["h=1", "h=7", "h=14", "h=30"]:
        r = r2["growth_by_horizon"][h_name]
        print(f"    {h_name}: Q5/Q1 = {r['q5q1']:.3f}x")

    print("\n  GT baseline_iv Q5/Q1 by horizon (from regime analysis):")
    for h_name in ["h=1", "h=7", "h=14", "h=30"]:
        q5 = r3["baseline_iv"]["Q5"]["stds"][h_name]
        q1 = r3["baseline_iv"]["Q1"]["stds"][h_name]
        ratio = q5 / q1 if q1 > 0 else float("nan")
        print(f"    {h_name}: Q5/Q1 = {ratio:.3f}x")

    # Save
    output_path = os.path.join(args.output_dir, "gt_investigation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
