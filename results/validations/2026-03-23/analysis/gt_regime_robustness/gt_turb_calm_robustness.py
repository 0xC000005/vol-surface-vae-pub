"""
GT Turb/Calm Spread Ratio Robustness Analysis
==============================================

Claim under verification: GT turb/calm spread ratio = 1.31x.
Model produces 2.1x (over-conditioning).

This script:
1. Replicates EXACTLY what test_block_ar_requirements_v2.py computes (Q20/Q80 vov split)
   applied to GT data (i.e., what ratio the test EXPECTS from the model, measured on GT).
2. Stress-tests: Q10/Q90, Q33/Q67
3. Vol-of-vol computed three ways: mean IV changes, ATM IV changes, returns std
4. Per-horizon analysis at h=7, h=14, h=30
5. Uses same test split (index 4540, history_len=30, future_len=30)

The key distinction:
- Test suite metric: std(mean_IV_daily_changes) over HISTORY window => vov for regime split
- "Spread" (GT): std(mean_IV_daily_changes) over FUTURE window => what we're comparing

Key insight from test_block_ar_requirements_v2.py:
  all_vov = std of daily changes of mean IV over history (30 days)
  per_window_cond_width = (Q95 - Q05) of conditional samples averaged over time and cells
  turb_calm_ratio = mean(per_window_width[turb_mask]) / mean(per_window_width[calm_mask])

For GT, we cannot compute Q95-Q05 CI width. Instead, we compute the analogous measure:
the std of actual daily changes in the FUTURE window. This is what we call "spread".
"""

import json
import sys
from pathlib import Path

import numpy as np

# ============================================================
# Configuration — mirrors test suite defaults exactly
# ============================================================
DATA_PATH = "data/vol_surface_with_ret.npz"
TEST_SPLIT_START = 4540
HISTORY_LEN = 30
FUTURE_LEN = 30
HORIZONS = [7, 14, 30]

# Test suite uses: vol_of_vol = std(diff(mean_iv_over_history))
# Regime split: Q20/Q80 on all_vov values


def load_data(data_path: str):
    data = np.load(data_path)
    surfaces = data["surface"]  # (N, 5, 5), raw IV in [0, 1]
    ret = data["ret"]           # (N,), daily SPX returns
    print(f"Loaded data: surfaces {surfaces.shape}, ret {ret.shape}")
    return surfaces, ret


def build_windows(surfaces, ret, split_start, history_len, future_len):
    """Build sliding-window test set starting at split_start."""
    N = surfaces.shape[0]
    windows_hist = []
    windows_fut = []
    windows_ret_hist = []

    for i in range(split_start, N - history_len - future_len + 1):
        hist = surfaces[i : i + history_len]          # (30, 5, 5)
        fut  = surfaces[i + history_len : i + history_len + future_len]  # (30, 5, 5)
        ret_h = ret[i : i + history_len]              # (30,)
        windows_hist.append(hist)
        windows_fut.append(fut)
        windows_ret_hist.append(ret_h)

    hist_arr = np.stack(windows_hist)  # (W, 30, 5, 5)
    fut_arr  = np.stack(windows_fut)   # (W, 30, 5, 5)
    ret_arr  = np.stack(windows_ret_hist)  # (W, 30)
    print(f"Windows: {hist_arr.shape[0]} test windows")
    return hist_arr, fut_arr, ret_arr


# ============================================================
# Vol-of-vol (regime classifier) — THREE DEFINITIONS
# ============================================================

def vov_mean_iv_changes(hist_arr):
    """
    EXACT match to test suite:
      mean_iv = history.mean(axis=(2,3))  -> (W, 30)
      daily_ch = diff(mean_iv, axis=1)    -> (W, 29)
      vov = std(daily_ch, axis=1)         -> (W,)
    """
    mean_iv = hist_arr.mean(axis=(2, 3))  # (W, 30)
    daily_ch = np.diff(mean_iv, axis=1)   # (W, 29)
    return daily_ch.std(axis=1)           # (W,)


def vov_atm_iv_changes(hist_arr):
    """
    ATM = moneyness index 2 (middle of 0-4), tenor index 2 (middle of 0-4).
    std of daily changes of ATM IV over history.
    """
    atm = hist_arr[:, :, 2, 2]           # (W, 30)
    daily_ch = np.diff(atm, axis=1)      # (W, 29)
    return daily_ch.std(axis=1)          # (W,)


def vov_returns_std(ret_arr):
    """
    Vol-of-vol proxy from return std over history window.
    """
    return ret_arr.std(axis=1)           # (W,)


# ============================================================
# Spread (GT) — THREE DEFINITIONS applied to FUTURE window
# ============================================================

def spread_mean_iv_changes(fut_arr):
    """
    Std of daily changes in mean IV over future window.
    Analogous to CI width — measures realized volatility of the forecast target.
    Shape: (W,) aggregate; also compute per-horizon.
    """
    mean_iv = fut_arr.mean(axis=(2, 3))   # (W, 30)
    daily_ch = np.diff(mean_iv, axis=1)   # (W, 29)
    return daily_ch.std(axis=1)           # (W,)


def spread_atm_iv_changes(fut_arr):
    """
    Std of daily changes in ATM IV over future window.
    """
    atm = fut_arr[:, :, 2, 2]            # (W, 30)
    daily_ch = np.diff(atm, axis=1)      # (W, 29)
    return daily_ch.std(axis=1)          # (W,)


def spread_per_horizon(fut_arr, horizon):
    """
    For a specific horizon h: compute spread as std across all cells at that time step.
    More analogous to per-step CI width.
    The std is across cells at step h (spatial spread), NOT temporal.

    A more meaningful per-horizon spread: for each window, what is the
    actual IV level std across cells at horizon h? This measures surface width.
    Returns (W,)
    """
    # IV at horizon h
    iv_h = fut_arr[:, horizon - 1, :, :]   # (W, 5, 5)
    # Deviation from per-window mean (comparable to CI width concept)
    # Use: std of the distribution of mean IVs across windows at that horizon
    return iv_h.mean(axis=(1, 2))           # (W,) mean IV at that horizon


def spread_temporal_std_up_to_horizon(fut_arr, horizon):
    """
    Temporal std of mean IV from day 1 to horizon h.
    Measures how much IV moves within the first h days.
    Comparable to CI width at horizon h.
    """
    fut_h = fut_arr[:, :horizon, :, :]     # (W, h, 5, 5)
    mean_iv = fut_h.mean(axis=(2, 3))      # (W, h)
    return mean_iv.std(axis=1)             # (W,) — std over time up to h


# ============================================================
# Regime split + ratio computation
# ============================================================

def compute_ratio(spread, vov, q_low, q_high, label=""):
    """
    Splits windows into calm (vov <= q_low-th percentile) and
    turb (vov >= q_high-th percentile).
    Returns turb_mean / calm_mean.
    """
    q_lo_val = np.quantile(vov, q_low)
    q_hi_val = np.quantile(vov, q_high)
    calm_mask = vov <= q_lo_val
    turb_mask = vov >= q_hi_val
    n_calm = calm_mask.sum()
    n_turb = turb_mask.sum()

    if n_calm == 0 or n_turb == 0:
        return {"ratio": None, "calm_mean": None, "turb_mean": None,
                "n_calm": int(n_calm), "n_turb": int(n_turb)}

    calm_mean = float(spread[calm_mask].mean())
    turb_mean = float(spread[turb_mask].mean())
    ratio = turb_mean / calm_mean if calm_mean > 0 else None

    if label:
        print(f"    {label}: calm={calm_mean:.5f}, turb={turb_mean:.5f}, "
              f"ratio={ratio:.3f}x (n_calm={n_calm}, n_turb={n_turb})")
    return {
        "ratio": float(ratio) if ratio is not None else None,
        "calm_mean": calm_mean,
        "turb_mean": turb_mean,
        "n_calm": int(n_calm),
        "n_turb": int(n_turb),
        "q_lo_val": float(q_lo_val),
        "q_hi_val": float(q_hi_val),
    }


def main():
    print("=" * 70)
    print("GT Turb/Calm Spread Ratio Robustness Analysis")
    print("=" * 70)

    # ---- Load data ----
    repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
    data_path = repo_root / DATA_PATH
    surfaces, ret = load_data(str(data_path))

    # ---- Build windows ----
    hist_arr, fut_arr, ret_arr = build_windows(
        surfaces, ret, TEST_SPLIT_START, HISTORY_LEN, FUTURE_LEN
    )
    W = hist_arr.shape[0]

    # ---- Compute three VoV definitions ----
    print("\n--- Vol-of-Vol (Regime Classifier) Definitions ---")
    vov_a = vov_mean_iv_changes(hist_arr)   # EXACT test suite match
    vov_b = vov_atm_iv_changes(hist_arr)
    vov_c = vov_returns_std(ret_arr)

    print(f"VoV (a) mean_IV_changes: mean={vov_a.mean():.5f}, "
          f"std={vov_a.std():.5f}, "
          f"Q20={np.quantile(vov_a, 0.20):.5f}, Q80={np.quantile(vov_a, 0.80):.5f}")
    print(f"VoV (b) ATM_IV_changes:  mean={vov_b.mean():.5f}, "
          f"std={vov_b.std():.5f}, "
          f"Q20={np.quantile(vov_b, 0.20):.5f}, Q80={np.quantile(vov_b, 0.80):.5f}")
    print(f"VoV (c) returns_std:     mean={vov_c.mean():.5f}, "
          f"std={vov_c.std():.5f}, "
          f"Q20={np.quantile(vov_c, 0.20):.5f}, Q80={np.quantile(vov_c, 0.80):.5f}")

    # ---- Compute three spread definitions ----
    spread_a = spread_mean_iv_changes(fut_arr)   # analogous to test suite width
    spread_b = spread_atm_iv_changes(fut_arr)

    print(f"\nSpread (a) mean_IV_future_std: mean={spread_a.mean():.5f}, std={spread_a.std():.5f}")
    print(f"Spread (b) ATM_IV_future_std:  mean={spread_b.mean():.5f}, std={spread_b.std():.5f}")

    # ---- Per-horizon temporal std spread ----
    spread_horizon = {}
    for h in HORIZONS:
        s = spread_temporal_std_up_to_horizon(fut_arr, h)
        spread_horizon[h] = s
        print(f"Spread temporal_std h={h:2d}: mean={s.mean():.5f}, std={s.std():.5f}")

    # ========================================================
    # SECTION 1: Test-suite-exact metric (Q20/Q80, vov_a, spread_a)
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Test-suite-exact metric (Q20/Q80 vov, mean_IV_changes spread)")
    print("=" * 70)
    print("Note: Test suite uses CI width (Q95-Q05 of samples). GT analog:")
    print("  std(mean_IV_daily_changes over FUTURE window)")
    print("  This is what the model SHOULD be able to match.\n")

    r_exact = compute_ratio(spread_a, vov_a, 0.20, 0.80,
                            "Q20/Q80, vov=mean_IV, spread=mean_IV_future")

    # ========================================================
    # SECTION 2: Q20/Q80 — vary VoV definition
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Q20/Q80 — vary VoV definition (spread=mean_IV_changes)")
    print("=" * 70)

    r_q2080_vov_a = compute_ratio(spread_a, vov_a, 0.20, 0.80,
                                   "vov=mean_IV")
    r_q2080_vov_b = compute_ratio(spread_a, vov_b, 0.20, 0.80,
                                   "vov=ATM_IV")
    r_q2080_vov_c = compute_ratio(spread_a, vov_c, 0.20, 0.80,
                                   "vov=returns_std")

    # ========================================================
    # SECTION 3: Q20/Q80 — vary spread definition
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Q20/Q80 — vary spread definition (vov=mean_IV_changes)")
    print("=" * 70)

    r_spread_a = compute_ratio(spread_a, vov_a, 0.20, 0.80,
                               "spread=mean_IV_future")
    r_spread_b = compute_ratio(spread_b, vov_a, 0.20, 0.80,
                               "spread=ATM_IV_future")

    # ========================================================
    # SECTION 4: Per-horizon temporal std spread (h=7, 14, 30)
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Per-horizon (Q20/Q80, vov=mean_IV_changes)")
    print("=" * 70)
    print("Spread = std of mean IV from day 1 to horizon h\n")

    per_horizon_results = {}
    for h in HORIZONS:
        print(f"  h={h}:")
        r = compute_ratio(spread_horizon[h], vov_a, 0.20, 0.80,
                          f"  Q20/Q80, h={h}")
        per_horizon_results[f"h{h}_q2080"] = r

    # ========================================================
    # SECTION 5: Q10/Q90 (more extreme regimes)
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Q10/Q90 — more extreme regime split")
    print("=" * 70)

    q1090_spread_a = compute_ratio(spread_a, vov_a, 0.10, 0.90,
                                    "Q10/Q90, vov=mean_IV, spread=mean_IV_future")
    q1090_per_horizon = {}
    for h in HORIZONS:
        print(f"  h={h}:")
        r = compute_ratio(spread_horizon[h], vov_a, 0.10, 0.90,
                          f"  Q10/Q90, h={h}")
        q1090_per_horizon[f"h{h}"] = r

    # ========================================================
    # SECTION 6: Q33/Q67 (tercile split)
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 6: Q33/Q67 — tercile split")
    print("=" * 70)

    q3367_spread_a = compute_ratio(spread_a, vov_a, 0.33, 0.67,
                                    "Q33/Q67, vov=mean_IV, spread=mean_IV_future")
    q3367_per_horizon = {}
    for h in HORIZONS:
        print(f"  h={h}:")
        r = compute_ratio(spread_horizon[h], vov_a, 0.33, 0.67,
                          f"  Q33/Q67, h={h}")
        q3367_per_horizon[f"h{h}"] = r

    # ========================================================
    # SECTION 7: All combinations — summary table
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 7: SUMMARY — Turb/Calm GT spread ratio")
    print("=" * 70)
    print(f"{'Config':<55} {'Ratio':>8}")
    print("-" * 65)

    summary = []
    configs = [
        ("Q20/Q80 | vov=mean_IV | spread=mean_IV_future [TEST-EXACT]",
         compute_ratio(spread_a, vov_a, 0.20, 0.80)),
        ("Q20/Q80 | vov=ATM_IV  | spread=mean_IV_future",
         compute_ratio(spread_a, vov_b, 0.20, 0.80)),
        ("Q20/Q80 | vov=ret_std | spread=mean_IV_future",
         compute_ratio(spread_a, vov_c, 0.20, 0.80)),
        ("Q20/Q80 | vov=mean_IV | spread=ATM_IV_future",
         compute_ratio(spread_b, vov_a, 0.20, 0.80)),
        ("Q10/Q90 | vov=mean_IV | spread=mean_IV_future",
         compute_ratio(spread_a, vov_a, 0.10, 0.90)),
        ("Q33/Q67 | vov=mean_IV | spread=mean_IV_future",
         compute_ratio(spread_a, vov_a, 0.33, 0.67)),
    ]
    for h in HORIZONS:
        configs.append(
            (f"Q20/Q80 | vov=mean_IV | spread=temporal_std_h={h}",
             compute_ratio(spread_horizon[h], vov_a, 0.20, 0.80))
        )
        configs.append(
            (f"Q10/Q90 | vov=mean_IV | spread=temporal_std_h={h}",
             compute_ratio(spread_horizon[h], vov_a, 0.10, 0.90))
        )
        configs.append(
            (f"Q33/Q67 | vov=mean_IV | spread=temporal_std_h={h}",
             compute_ratio(spread_horizon[h], vov_a, 0.33, 0.67))
        )

    for label, r in configs:
        ratio_str = f"{r['ratio']:.3f}x" if r["ratio"] is not None else "N/A"
        print(f"{label:<55} {ratio_str:>8}")
        summary.append({
            "config": label,
            "ratio": r["ratio"],
            "calm_mean": r.get("calm_mean"),
            "turb_mean": r.get("turb_mean"),
            "n_calm": r.get("n_calm"),
            "n_turb": r.get("n_turb"),
        })

    # ========================================================
    # Final interpretation
    # ========================================================
    ratios_all = [s["ratio"] for s in summary if s["ratio"] is not None]
    ratio_min = float(np.min(ratios_all))
    ratio_max = float(np.max(ratios_all))
    ratio_mean = float(np.mean(ratios_all))
    ratio_std = float(np.std(ratios_all))

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"GT turb/calm spread ratio across ALL {len(ratios_all)} configs:")
    print(f"  Min:  {ratio_min:.3f}x")
    print(f"  Max:  {ratio_max:.3f}x")
    print(f"  Mean: {ratio_mean:.3f}x")
    print(f"  Std:  {ratio_std:.3f}x")
    print(f"\nClaimed value: 1.31x")
    print(f"Model value:   2.1x (over-conditioning)")
    print(f"Test gate:     >1.15x")

    exact_ratio = r_exact["ratio"]
    print(f"\nTest-suite-exact GT ratio: {exact_ratio:.3f}x")
    if exact_ratio is not None:
        if ratio_min > 1.20 and ratio_max < 1.50:
            robustness = "ROBUST — ratio is stable across definitions"
        elif ratio_std > 0.15:
            robustness = "SENSITIVE — ratio varies significantly across definitions"
        else:
            robustness = "MODERATE — some sensitivity to definition"
        print(f"Robustness assessment: {robustness}")

    # ========================================================
    # Save results as JSON
    # ========================================================
    results = {
        "metadata": {
            "description": "GT turb/calm spread ratio robustness analysis",
            "claim": "GT turb/calm spread ratio = 1.31x, model = 2.1x",
            "test_suite_gate": ">1.15x",
            "test_split_start": TEST_SPLIT_START,
            "history_len": HISTORY_LEN,
            "future_len": FUTURE_LEN,
            "n_windows": int(W),
            "horizons": HORIZONS,
        },
        "vov_statistics": {
            "mean_iv_changes": {
                "mean": float(vov_a.mean()),
                "std": float(vov_a.std()),
                "q10": float(np.quantile(vov_a, 0.10)),
                "q20": float(np.quantile(vov_a, 0.20)),
                "q33": float(np.quantile(vov_a, 0.33)),
                "q67": float(np.quantile(vov_a, 0.67)),
                "q80": float(np.quantile(vov_a, 0.80)),
                "q90": float(np.quantile(vov_a, 0.90)),
            },
            "atm_iv_changes": {
                "mean": float(vov_b.mean()),
                "std": float(vov_b.std()),
                "q20": float(np.quantile(vov_b, 0.20)),
                "q80": float(np.quantile(vov_b, 0.80)),
            },
            "returns_std": {
                "mean": float(vov_c.mean()),
                "std": float(vov_c.std()),
                "q20": float(np.quantile(vov_c, 0.20)),
                "q80": float(np.quantile(vov_c, 0.80)),
            },
        },
        "spread_statistics": {
            "mean_iv_future_std": {
                "mean": float(spread_a.mean()),
                "std": float(spread_a.std()),
            },
            "atm_iv_future_std": {
                "mean": float(spread_b.mean()),
                "std": float(spread_b.std()),
            },
            "temporal_std_per_horizon": {
                str(h): {
                    "mean": float(spread_horizon[h].mean()),
                    "std": float(spread_horizon[h].std()),
                }
                for h in HORIZONS
            },
        },
        "test_suite_exact": {
            "description": "Q20/Q80 on vov=std(diff(mean_iv_history)), spread=std(diff(mean_iv_future))",
            "result": r_exact,
        },
        "q20_q80_results": {
            "vov_mean_iv_spread_mean_iv": r_q2080_vov_a,
            "vov_atm_iv_spread_mean_iv": r_q2080_vov_b,
            "vov_returns_std_spread_mean_iv": r_q2080_vov_c,
            "vov_mean_iv_spread_atm_iv": r_spread_b,
            "per_horizon": per_horizon_results,
        },
        "q10_q90_results": {
            "spread_mean_iv": q1090_spread_a,
            "per_horizon": q1090_per_horizon,
        },
        "q33_q67_results": {
            "spread_mean_iv": q3367_spread_a,
            "per_horizon": q3367_per_horizon,
        },
        "summary_all_configs": summary,
        "aggregate_stats": {
            "min_ratio": ratio_min,
            "max_ratio": ratio_max,
            "mean_ratio": ratio_mean,
            "std_ratio": ratio_std,
            "n_configs": len(ratios_all),
        },
        "interpretation": {
            "claimed_gt_ratio": 1.31,
            "model_ratio": 2.1,
            "test_gate": 1.15,
            "test_suite_exact_gt_ratio": float(exact_ratio) if exact_ratio is not None else None,
            "ratio_range": f"{ratio_min:.3f}x to {ratio_max:.3f}x",
            "is_robust": ratio_std < 0.15,
            "robustness_note": (
                "Ratio is stable (std < 0.15) across definitions — "
                "claim is robust" if ratio_std < 0.15
                else "Ratio varies significantly — claim is sensitive to definition"
            ),
        },
    }
    return results


if __name__ == "__main__":
    results = main()
    out_path = Path(__file__).parent / "gt_turb_calm_robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
