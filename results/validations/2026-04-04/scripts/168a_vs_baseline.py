#!/usr/bin/env python3
"""
168a (K=4) vs Baseline 164a_v3 (K=16) comprehensive comparison.

168a: K=4, 20 epochs, best_model (ep11)
Baseline: K=16, 80 epochs, best_model (ep11)

Both use the same architecture, only K differs.
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/max/Documents/vol-surface-vae-pub")

# Load summaries
with open(ROOT / "results/block_ar/168a_best_30d/summary.json") as f:
    k4 = json.load(f)
with open(ROOT / "results/block_ar/164a_v3_percell_bptt_softplus_best_30d/summary.json") as f:
    k16 = json.load(f)
# Load training history
with open(ROOT / "models/backfill/afcrps_168a/training_history.json") as f:
    history = json.load(f)

results = {}

# ============================================================
# 1. All 9 suite pass/fail comparison
# ============================================================
suite_names = {
    "surface": "S1 Surface Validity",
    "coverage": "S2 CI Coverage",
    "conditionality": "S3 Conditionality",
    "time_series": "S4 Time Series",
    "block_ar": "S5 Block-AR Boundary",
    "cointegration": "S6 Cointegration",
    "regime_coverage": "S7 Regime Coverage",
    "distributional": "S8 Distributional",
    "cross_cell_correlation": "S9 Cross-Cell Correlation",
}

suite_comparison = {}
for key, name in suite_names.items():
    k4_pass = k4[key]["overall_pass"]
    k16_pass = k16[key]["overall_pass"]
    suite_comparison[name] = {
        "168a_K4": "PASS" if k4_pass else "FAIL",
        "baseline_K16": "PASS" if k16_pass else "FAIL",
        "delta": "SAME" if k4_pass == k16_pass else ("REGRESSION" if k16_pass and not k4_pass else "IMPROVEMENT"),
    }

k4_pass_count = sum(1 for k in suite_names if k4[k]["overall_pass"])
k16_pass_count = sum(1 for k in suite_names if k16[k]["overall_pass"])

results["suite_comparison"] = {
    "per_suite": suite_comparison,
    "168a_total_pass": k4_pass_count,
    "baseline_total_pass": k16_pass_count,
    "delta": k4_pass_count - k16_pass_count,
}

print("=" * 70)
print("SUITE PASS/FAIL COMPARISON")
print("=" * 70)
print(f"{'Suite':<30} {'168a(K=4)':<12} {'Baseline(K=16)':<16} {'Delta'}")
print("-" * 70)
for name, d in suite_comparison.items():
    print(f"{name:<30} {d['168a_K4']:<12} {d['baseline_K16']:<16} {d['delta']}")
print(f"\n168a total: {k4_pass_count}/9, Baseline total: {k16_pass_count}/9 (delta: {k4_pass_count - k16_pass_count})")

# ============================================================
# 2. S1 - Surface validity details
# ============================================================
s1_168a = k4["surface"]
s1_base = k16["surface"]

results["s1_surface"] = {
    "explosion_rate": {
        "168a": s1_168a["explosion"]["explosion_total_rate"],
        "baseline": s1_base["explosion"]["explosion_total_rate"],
        "168a_gate_5pct": s1_168a["explosion"]["explosion_total_rate"] < 0.05,
    },
    "explosion_low_rate": {
        "168a": s1_168a["explosion"]["explosion_low_rate"],
        "baseline": s1_base["explosion"]["explosion_low_rate"],
    },
    "min_iv": {
        "168a": s1_168a["explosion"]["min_iv_observed"],
        "baseline": s1_base["explosion"]["min_iv_observed"],
    },
    "calendar_worst_strike": {
        "168a": s1_168a["calendar"]["worst_strike_rate"],
        "baseline": s1_base["calendar"]["worst_strike_rate"],
        "gt": s1_168a["calendar"]["gt_worst_strike_rate"],
    },
    "butterfly_worst_tenor": {
        "168a": s1_168a["butterfly"]["worst_tenor_rate"],
        "baseline": s1_base["butterfly"]["worst_tenor_rate"],
    },
}

print("\n" + "=" * 70)
print("S1 SURFACE VALIDITY")
print("=" * 70)
print(f"Explosion rate:     168a={s1_168a['explosion']['explosion_total_rate']:.4f} "
      f"(low={s1_168a['explosion']['explosion_low_rate']:.4f})  |  "
      f"Baseline={s1_base['explosion']['explosion_total_rate']:.4f} "
      f"(low={s1_base['explosion']['explosion_low_rate']:.4f})")
print(f"  168a at 4.55% -- borderline at 5% gate!")
print(f"  Baseline at 2.47% -- comfortable margin")
print(f"Min IV observed:    168a={s1_168a['explosion']['min_iv_observed']:.4f}  |  "
      f"Baseline={s1_base['explosion']['min_iv_observed']:.4f}")
print(f"Calendar worst:     168a={s1_168a['calendar']['worst_strike_rate']:.4f}  |  "
      f"Baseline={s1_base['calendar']['worst_strike_rate']:.4f}  (GT={s1_168a['calendar']['gt_worst_strike_rate']:.4f})")
print(f"Butterfly worst:    168a={s1_168a['butterfly']['worst_tenor_rate']:.4f}  |  "
      f"Baseline={s1_base['butterfly']['worst_tenor_rate']:.4f}")

# ============================================================
# 3. S2 - Per-horizon coverage breakdown
# ============================================================
print("\n" + "=" * 70)
print("S2 CI COVERAGE - PER-HORIZON 90% COVERAGE")
print("=" * 70)
s2_168a = k4["coverage"]
s2_base = k16["coverage"]

s2_horizon = {}
for h in ["1", "7", "14", "30"]:
    c4 = s2_168a["per_horizon"][h]["0.9"]
    c16 = s2_base["per_horizon"][h]["0.9"]
    s2_horizon[f"h{h}"] = {"168a": c4, "baseline": c16, "delta": c4 - c16}
    print(f"  h={h:>2}: 168a={c4:.4f}  |  Baseline={c16:.4f}  |  delta={c4-c16:+.4f}")

print(f"\nOverall 90% coverage: 168a={s2_168a['overall']['0.9']:.4f}  |  "
      f"Baseline={s2_base['overall']['0.9']:.4f}")
print(f"Calibration error:   168a={s2_168a['calibration_error']:.4f}  |  "
      f"Baseline={s2_base['calibration_error']:.4f}")

results["s2_coverage"] = {
    "per_horizon_90": s2_horizon,
    "overall_90": {"168a": s2_168a["overall"]["0.9"], "baseline": s2_base["overall"]["0.9"]},
    "calibration_error": {"168a": s2_168a["calibration_error"], "baseline": s2_base["calibration_error"]},
    "worst_cell_pass": {"168a": s2_168a["worst_cell_pass"], "baseline": s2_base["worst_cell_pass"]},
}

# ============================================================
# 4. S2 - Per-cell coverage comparison (all horizons)
# ============================================================
print("\n" + "=" * 70)
print("S2 PER-CELL COVERAGE (90%) - WHICH CELLS CHANGED?")
print("=" * 70)

cell_labels = [
    ["(0,0) 80%/1w", "(0,1) 90%/1w", "(0,2) 100%/1w", "(0,3) 110%/1w", "(0,4) 120%/1w"],
    ["(1,0) 80%/1m", "(1,1) 90%/1m", "(1,2) 100%/1m", "(1,3) 110%/1m", "(1,4) 120%/1m"],
    ["(2,0) 80%/3m", "(2,1) 90%/3m", "(2,2) 100%/3m", "(2,3) 110%/3m", "(2,4) 120%/3m"],
    ["(3,0) 80%/6m", "(3,1) 90%/6m", "(3,2) 100%/6m", "(3,3) 110%/6m", "(3,4) 120%/6m"],
    ["(4,0) 80%/1y", "(4,1) 90%/1y", "(4,2) 100%/1y", "(4,3) 110%/1y", "(4,4) 120%/1y"],
]

per_cell_deltas = {}
for h in ["1", "7", "14", "30"]:
    c4_grid = s2_168a["per_cell_coverage"][h]
    c16_grid = s2_base["per_cell_coverage"][h]
    print(f"\n  Horizon {h}d:")
    print(f"    {'Cell':<16} {'168a':>8} {'Baseline':>10} {'Delta':>8}  Status")
    print(f"    {'-'*52}")
    worst_168a = 1.0
    worst_base = 1.0
    for i in range(5):
        for j in range(5):
            v4 = c4_grid[i][j]
            v16 = c16_grid[i][j]
            delta = v4 - v16
            worst_168a = min(worst_168a, v4)
            worst_base = min(worst_base, v16)
            status = ""
            if abs(delta) > 0.05:
                status = "<<< LARGE" if delta < -0.05 else ">>> LARGE"
            elif abs(delta) > 0.02:
                status = "< notable" if delta < -0.02 else "> notable"
            per_cell_deltas[f"h{h}_r{i}_c{j}"] = {
                "168a": round(v4, 4),
                "baseline": round(v16, 4),
                "delta": round(delta, 4),
            }
            if abs(delta) > 0.02:
                print(f"    ({i},{j})         {v4:>8.4f} {v16:>10.4f} {delta:>+8.4f}  {status}")
    print(f"    Worst cell: 168a={worst_168a:.4f} vs Baseline={worst_base:.4f}")

results["s2_per_cell"] = per_cell_deltas

# Worst cells per horizon
print("\n  WORST CELL PER HORIZON:")
for h in ["1", "7", "14", "30"]:
    w4 = s2_168a["worst_cell_per_horizon"][h]
    w16 = s2_base["worst_cell_per_horizon"][h]
    print(f"    h={h:>2}: 168a={w4:.4f}  |  Baseline={w16:.4f}  |  delta={w4-w16:+.4f}")

results["s2_worst_cell_per_horizon"] = {
    h: {
        "168a": s2_168a["worst_cell_per_horizon"][h],
        "baseline": s2_base["worst_cell_per_horizon"][h],
    }
    for h in ["1", "7", "14", "30"]
}

# ============================================================
# 5. S3 - Conditionality details
# ============================================================
print("\n" + "=" * 70)
print("S3 CONDITIONALITY")
print("=" * 70)
s3_168a = k4["conditionality"]
s3_base = k16["conditionality"]

print(f"Width ratio:         168a={s3_168a['width_ratio']:.4f}  |  Baseline={s3_base['width_ratio']:.4f}")
print(f"Turb/calm ratio:     168a={s3_168a['turb_calm_ratio']:.4f}  |  Baseline={s3_base['turb_calm_ratio']:.4f}  (gate>1.15)")
print(f"MAE reduction:       168a={s3_168a['mae_reduction_pct']:.2f}%  |  Baseline={s3_base['mae_reduction_pct']:.2f}%")
print(f"Worst cell WR:       168a={s3_168a['worst_cell_width_ratio']:.4f}  |  Baseline={s3_base['worst_cell_width_ratio']:.4f}")
print(f"  WR pass:           168a={s3_168a['worst_cell_wr_pass']}  |  Baseline={s3_base['worst_cell_wr_pass']}")
print(f"Worst cell MAE red:  168a={s3_168a['worst_cell_mae_reduction']:.2f}%  |  "
      f"Baseline={s3_base['worst_cell_mae_reduction']:.2f}%")
print(f"Avg cond width:      168a={s3_168a['avg_cond_width']:.6f}  |  Baseline={s3_base['avg_cond_width']:.6f}")
print(f"Avg uncond width:    168a={s3_168a['avg_uncond_width']:.6f}  |  Baseline={s3_base['avg_uncond_width']:.6f}")

results["s3_conditionality"] = {
    "width_ratio": {"168a": s3_168a["width_ratio"], "baseline": s3_base["width_ratio"]},
    "turb_calm_ratio": {"168a": s3_168a["turb_calm_ratio"], "baseline": s3_base["turb_calm_ratio"]},
    "mae_reduction_pct": {"168a": s3_168a["mae_reduction_pct"], "baseline": s3_base["mae_reduction_pct"]},
    "worst_cell_wr": {"168a": s3_168a["worst_cell_width_ratio"], "baseline": s3_base["worst_cell_width_ratio"]},
    "worst_cell_wr_pass": {"168a": s3_168a["worst_cell_wr_pass"], "baseline": s3_base["worst_cell_wr_pass"]},
    "avg_cond_width": {"168a": s3_168a["avg_cond_width"], "baseline": s3_base["avg_cond_width"]},
}

# ============================================================
# 6. S4 - Time Series
# ============================================================
print("\n" + "=" * 70)
print("S4 TIME SERIES")
print("=" * 70)
s4_168a = k4["time_series"]
s4_base = k16["time_series"]

print(f"ACF correlation:     168a={s4_168a['acf']['acf_correlation']:.4f}  |  "
      f"Baseline={s4_base['acf']['acf_correlation']:.4f}")
print(f"Kurtosis ratio:      168a={s4_168a['kurtosis']['kurtosis_ratio']:.4f}  |  "
      f"Baseline={s4_base['kurtosis']['kurtosis_ratio']:.4f}  (gate: 0.5-2.0)")
print(f"  Gen kurtosis:      168a={s4_168a['kurtosis']['gen_kurtosis']:.2f}  |  "
      f"Baseline={s4_base['kurtosis']['gen_kurtosis']:.2f}  (GT={s4_168a['kurtosis']['gt_kurtosis']:.2f})")
print(f"Skewness ratio:      168a={s4_168a['kurtosis']['skewness_ratio']:.4f}  |  "
      f"Baseline={s4_base['kurtosis']['skewness_ratio']:.4f}")
print(f"  Skewness pass:     168a={s4_168a['kurtosis']['skewness_pass']}  |  "
      f"Baseline={s4_base['kurtosis']['skewness_pass']}")

results["s4_time_series"] = {
    "acf_correlation": {"168a": s4_168a["acf"]["acf_correlation"], "baseline": s4_base["acf"]["acf_correlation"]},
    "kurtosis_ratio": {"168a": s4_168a["kurtosis"]["kurtosis_ratio"], "baseline": s4_base["kurtosis"]["kurtosis_ratio"]},
    "gen_kurtosis": {"168a": s4_168a["kurtosis"]["gen_kurtosis"], "baseline": s4_base["kurtosis"]["gen_kurtosis"]},
    "skewness_pass": {"168a": s4_168a["kurtosis"]["skewness_pass"], "baseline": s4_base["kurtosis"]["skewness_pass"]},
}

# ============================================================
# 7. S5 - Block-AR
# ============================================================
print("\n" + "=" * 70)
print("S5 BLOCK-AR BOUNDARY")
print("=" * 70)
s5_168a = k4["block_ar"]
s5_base = k16["block_ar"]
print(f"Boundary ratio:      168a={s5_168a['boundary_smoothness']['boundary_ratio']:.4f}  |  "
      f"Baseline={s5_base['boundary_smoothness']['boundary_ratio']:.4f}")
print(f"Growing uncertainty:  168a={s5_168a['growing_uncertainty']['monotonic']}  |  "
      f"Baseline={s5_base['growing_uncertainty']['monotonic']}")
for h in ["1", "10", "20", "30"]:
    v4 = s5_168a["growing_uncertainty"]["key_horizon_vars"][h]
    v16 = s5_base["growing_uncertainty"]["key_horizon_vars"][h]
    print(f"  h={h:>2} var: 168a={v4:.6f}  |  Baseline={v16:.6f}")

results["s5_block_ar"] = {
    "boundary_ratio": {"168a": s5_168a["boundary_smoothness"]["boundary_ratio"],
                       "baseline": s5_base["boundary_smoothness"]["boundary_ratio"]},
    "key_horizon_vars": {
        h: {"168a": s5_168a["growing_uncertainty"]["key_horizon_vars"][h],
            "baseline": s5_base["growing_uncertainty"]["key_horizon_vars"][h]}
        for h in ["1", "10", "20", "30"]
    },
}

# ============================================================
# 8. S6 - Cointegration
# ============================================================
print("\n" + "=" * 70)
print("S6 COINTEGRATION")
print("=" * 70)
s6_168a = k4["cointegration"]
s6_base = k16["cointegration"]
print(f"Gen/GT ratio:        168a={s6_168a['gen_gt_ratio']:.4f}  |  Baseline={s6_base['gen_gt_ratio']:.4f}")
print(f"Legacy ratio:        168a={s6_168a['gen_gt_ratio_legacy']:.4f}  |  Baseline={s6_base['gen_gt_ratio_legacy']:.4f}")
print(f"Gen mean R^2:        168a={s6_168a['gen_mean_rsq']:.4f}  |  Baseline={s6_base['gen_mean_rsq']:.4f}")

results["s6_cointegration"] = {
    "gen_gt_ratio": {"168a": s6_168a["gen_gt_ratio"], "baseline": s6_base["gen_gt_ratio"]},
    "legacy_ratio": {"168a": s6_168a["gen_gt_ratio_legacy"], "baseline": s6_base["gen_gt_ratio_legacy"]},
    "gen_mean_rsq": {"168a": s6_168a["gen_mean_rsq"], "baseline": s6_base["gen_mean_rsq"]},
}

# ============================================================
# 9. S7 - Regime Coverage
# ============================================================
print("\n" + "=" * 70)
print("S7 REGIME COVERAGE")
print("=" * 70)
s7_168a = k4["regime_coverage"]
s7_base = k16["regime_coverage"]

print(f"Layer1 pass:         168a={s7_168a['layer1_pass']}  |  Baseline={s7_base['layer1_pass']}")
print(f"Layer2 pass:         168a={s7_168a['layer2_pass']}  |  Baseline={s7_base['layer2_pass']}")
print(f"  Layer2 n_passing:  168a={s7_168a['layer2_n_passing']}/8  |  Baseline={s7_base['layer2_n_passing']}/8")
print(f"Layer3 pass:         168a={s7_168a['layer3_pass']}  |  Baseline={s7_base['layer3_pass']}")
print(f"  Catastrophic rate: 168a={s7_168a['layer3_catastrophic_rate']:.4f}  |  "
      f"Baseline={s7_base['layer3_catastrophic_rate']:.4f}")

print("\nPer-regime per-horizon coverage:")
for regime in ["calm", "turb"]:
    for h in ["1", "7", "14", "30"]:
        c4 = s7_168a["layer1_regime_horizon"][regime][h]["coverage"]
        c16 = s7_base["layer1_regime_horizon"][regime][h]["coverage"]
        print(f"  {regime:>4} h={h:>2}: 168a={c4:.4f}  |  Baseline={c16:.4f}  |  delta={c4-c16:+.4f}")

results["s7_regime"] = {
    "layer1_pass": {"168a": s7_168a["layer1_pass"], "baseline": s7_base["layer1_pass"]},
    "layer2_pass": {"168a": s7_168a["layer2_pass"], "baseline": s7_base["layer2_pass"]},
    "layer2_n_passing": {"168a": s7_168a["layer2_n_passing"], "baseline": s7_base["layer2_n_passing"]},
    "layer3_catastrophic_rate": {"168a": s7_168a["layer3_catastrophic_rate"],
                                  "baseline": s7_base["layer3_catastrophic_rate"]},
}

# ============================================================
# 10. S8 - Distributional
# ============================================================
print("\n" + "=" * 70)
print("S8 DISTRIBUTIONAL")
print("=" * 70)
s8_168a = k4["distributional"]
s8_base = k16["distributional"]

print("KS test (daily changes):")
print(f"  n_pass:            168a={s8_168a['ks_test']['n_pass']}/25  |  Baseline={s8_base['ks_test']['n_pass']}/25")
print(f"  worst_stat:        168a={s8_168a['ks_test']['worst_stat']:.4f}  |  Baseline={s8_base['ks_test']['worst_stat']:.4f}")
print(f"  median_stat:       168a={s8_168a['ks_test']['median_stat']:.4f}  |  Baseline={s8_base['ks_test']['median_stat']:.4f}")

print("KS level test:")
print(f"  n_pass:            168a={s8_168a['ks_level_test']['n_pass']}/25  |  "
      f"Baseline={s8_base['ks_level_test']['n_pass']}/25")
print(f"  worst_stat:        168a={s8_168a['ks_level_test']['worst_stat']:.4f}  |  "
      f"Baseline={s8_base['ks_level_test']['worst_stat']:.4f}")

print("Median bias:")
print(f"  n_pass (frac):     168a={s8_168a['median_bias']['n_pass']}/25  |  "
      f"Baseline={s8_base['median_bias']['n_pass']}/25")
print(f"  n_mag_pass:        168a={s8_168a['median_bias']['n_mag_pass']}/25  |  "
      f"Baseline={s8_base['median_bias']['n_mag_pass']}/25")

print("Window floor:")
print(f"  n_bad_windows:     168a={s8_168a['window_floor']['n_bad_windows']}  |  "
      f"Baseline={s8_base['window_floor']['n_bad_windows']}")
print(f"  pct_bad:           168a={s8_168a['window_floor']['pct_bad']:.4f}  |  "
      f"Baseline={s8_base['window_floor']['pct_bad']:.4f}")
print(f"  p10_cov:           168a={s8_168a['window_floor']['p10_cov']:.4f}  |  "
      f"Baseline={s8_base['window_floor']['p10_cov']:.4f}")

# Per-cell KS comparison
ks_improved = 0
ks_worsened = 0
ks_same = 0
for i in range(5):
    for j in range(5):
        v4 = s8_168a["ks_test"]["ks_grid"][i][j]
        v16 = s8_base["ks_test"]["ks_grid"][i][j]
        if v4 < v16 - 0.01:
            ks_improved += 1
        elif v4 > v16 + 0.01:
            ks_worsened += 1
        else:
            ks_same += 1

print(f"\nPer-cell KS changes: {ks_improved} improved, {ks_worsened} worsened, {ks_same} similar")

results["s8_distributional"] = {
    "ks_daily_changes": {
        "n_pass": {"168a": s8_168a["ks_test"]["n_pass"], "baseline": s8_base["ks_test"]["n_pass"]},
        "worst_stat": {"168a": s8_168a["ks_test"]["worst_stat"], "baseline": s8_base["ks_test"]["worst_stat"]},
        "median_stat": {"168a": s8_168a["ks_test"]["median_stat"], "baseline": s8_base["ks_test"]["median_stat"]},
    },
    "ks_levels": {
        "n_pass": {"168a": s8_168a["ks_level_test"]["n_pass"], "baseline": s8_base["ks_level_test"]["n_pass"]},
        "worst_stat": {"168a": s8_168a["ks_level_test"]["worst_stat"], "baseline": s8_base["ks_level_test"]["worst_stat"]},
    },
    "median_bias": {
        "n_pass_frac": {"168a": s8_168a["median_bias"]["n_pass"], "baseline": s8_base["median_bias"]["n_pass"]},
        "n_mag_pass": {"168a": s8_168a["median_bias"]["n_mag_pass"], "baseline": s8_base["median_bias"]["n_mag_pass"]},
    },
    "window_floor": {
        "n_bad_windows": {"168a": s8_168a["window_floor"]["n_bad_windows"],
                          "baseline": s8_base["window_floor"]["n_bad_windows"]},
        "pct_bad": {"168a": s8_168a["window_floor"]["pct_bad"], "baseline": s8_base["window_floor"]["pct_bad"]},
        "p10_cov": {"168a": s8_168a["window_floor"]["p10_cov"], "baseline": s8_base["window_floor"]["p10_cov"]},
    },
}

# ============================================================
# 11. S9 - Cross-Cell Correlation
# ============================================================
print("\n" + "=" * 70)
print("S9 CROSS-CELL CORRELATION")
print("=" * 70)
s9_168a = k4["cross_cell_correlation"]
s9_base = k16["cross_cell_correlation"]

print(f"Correlation ratio:   168a={s9_168a['corr_ratio']:.4f}  |  Baseline={s9_base['corr_ratio']:.4f}")
print(f"  Gen mean corr:     168a={s9_168a['gen_mean_corr']:.4f}  |  Baseline={s9_base['gen_mean_corr']:.4f}  (GT={s9_168a['gt_mean_corr']:.4f})")
print(f"  corr_pass:         168a={s9_168a['corr_pass']}  |  Baseline={s9_base['corr_pass']}")
print(f"Rank ratio:          168a={s9_168a['rank_ratio']:.4f}  |  Baseline={s9_base['rank_ratio']:.4f}")
print(f"  Gen eff rank:      168a={s9_168a['gen_eff_rank']:.4f}  |  Baseline={s9_base['gen_eff_rank']:.4f}  (GT={s9_168a['gt_eff_rank']:.4f})")
print(f"  rank_pass:         168a={s9_168a['rank_pass']}  |  Baseline={s9_base['rank_pass']}")
print(f"Frob distance:       168a={s9_168a['frob_dist']:.4f}  |  Baseline={s9_base['frob_dist']:.4f}")
print(f"PC1 var explained:   168a={s9_168a['gen_pc1_var']:.4f}  |  Baseline={s9_base['gen_pc1_var']:.4f}  (GT={s9_168a['gt_pc1_var']:.4f})")

results["s9_cross_cell"] = {
    "corr_ratio": {"168a": s9_168a["corr_ratio"], "baseline": s9_base["corr_ratio"]},
    "gen_mean_corr": {"168a": s9_168a["gen_mean_corr"], "baseline": s9_base["gen_mean_corr"],
                      "gt": s9_168a["gt_mean_corr"]},
    "corr_pass": {"168a": s9_168a["corr_pass"], "baseline": s9_base["corr_pass"]},
    "rank_ratio": {"168a": s9_168a["rank_ratio"], "baseline": s9_base["rank_ratio"]},
    "gen_eff_rank": {"168a": s9_168a["gen_eff_rank"], "baseline": s9_base["gen_eff_rank"],
                     "gt": s9_168a["gt_eff_rank"]},
    "rank_pass": {"168a": s9_168a["rank_pass"], "baseline": s9_base["rank_pass"]},
    "frob_dist": {"168a": s9_168a["frob_dist"], "baseline": s9_base["frob_dist"]},
    "gen_pc1_var": {"168a": s9_168a["gen_pc1_var"], "baseline": s9_base["gen_pc1_var"],
                    "gt": s9_168a["gt_pc1_var"]},
}

# ============================================================
# 12. Training dynamics
# ============================================================
print("\n" + "=" * 70)
print("TRAINING DYNAMICS (168a, 20 epochs)")
print("=" * 70)

epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]
maes = [h["mae"] for h in history]
spreads = [h["spread"] for h in history]
vs_losses = [h["vs"] for h in history]

# Check if still improving at ep20
last_5_val = val_losses[-5:]
first_5_val = val_losses[:5]
last_5_train = train_losses[-5:]
last_3_mae = maes[-3:]
last_3_vs = vs_losses[-3:]

print(f"Epoch range: {epochs[0]}-{epochs[-1]}")
print(f"Best val_loss: {min(val_losses):.6f} at epoch {epochs[val_losses.index(min(val_losses))]}")
print(f"Final val_loss: {val_losses[-1]:.6f}")
print(f"\nTrain loss trend (last 5 epochs):")
for i in range(-5, 0):
    print(f"  ep{epochs[i]:>2}: train={train_losses[i]:.6f}  val={val_losses[i]:.6f}  mae={maes[i]:.6f}  vs={vs_losses[i]:.6f}")

# Check convergence: are losses still decreasing?
train_decreasing = all(train_losses[i] >= train_losses[i+1] for i in range(-5, -1))
mae_decreasing = all(maes[i] >= maes[i+1] for i in range(-4, -1))

print(f"\nTrain loss monotonically decreasing (last 5)? {train_decreasing}")
print(f"MAE monotonically decreasing (last 3)? {mae_decreasing}")
print(f"Train loss ep1 vs ep20: {train_losses[0]:.6f} -> {train_losses[-1]:.6f} ({(train_losses[-1]-train_losses[0])/train_losses[0]*100:+.1f}%)")
print(f"MAE ep1 vs ep20: {maes[0]:.6f} -> {maes[-1]:.6f} ({(maes[-1]-maes[0])/maes[0]*100:+.1f}%)")
print(f"VS ep1 vs ep20: {vs_losses[0]:.6f} -> {vs_losses[-1]:.6f} ({(vs_losses[-1]-vs_losses[0])/vs_losses[0]*100:+.1f}%)")
print(f"\nVS loss at ep10 (freeze point): {vs_losses[9]:.6f} -> ep20: {vs_losses[-1]:.6f} "
      f"({(vs_losses[-1]-vs_losses[9])/vs_losses[9]*100:+.1f}%)")

# Compute rate of improvement in last 5 epochs
train_rate = (train_losses[-5] - train_losses[-1]) / 5
mae_rate = (maes[-5] - maes[-1]) / 5
print(f"\nRate of improvement per epoch (last 5):")
print(f"  Train loss: {train_rate:.6f}/epoch")
print(f"  MAE: {mae_rate:.6f}/epoch")
still_improving = train_rate > 0.0001
print(f"\nSTILL IMPROVING AT EP20? {'YES' if still_improving else 'NO'} (train rate > 0.0001: {still_improving})")

results["training_dynamics"] = {
    "best_val_epoch": epochs[val_losses.index(min(val_losses))],
    "best_val_loss": min(val_losses),
    "final_val_loss": val_losses[-1],
    "train_loss_ep1": train_losses[0],
    "train_loss_ep20": train_losses[-1],
    "mae_ep1": maes[0],
    "mae_ep20": maes[-1],
    "vs_ep1": vs_losses[0],
    "vs_ep20": vs_losses[-1],
    "vs_ep10": vs_losses[9],
    "train_decreasing_last5": train_decreasing,
    "train_rate_per_epoch": train_rate,
    "mae_rate_per_epoch": mae_rate,
    "still_improving": still_improving,
}

# ============================================================
# 13. Checkpoint verification
# ============================================================
print("\n" + "=" * 70)
print("CHECKPOINT VERIFICATION")
print("=" * 70)
print(f"168a checkpoint epoch: 11 (best_model)")
print(f"Baseline checkpoint epoch: 11 (best_model)")
print(f"Both are best_model checkpoints from same relative epoch")
print(f"168a n_members (K): 4")
print(f"168a config hash: {k4['eval_config']['model_config_hash']}")
print(f"Baseline config hash: {k16['eval_config']['model_config_hash']}")
same_hash = k4["eval_config"]["model_config_hash"] == k16["eval_config"]["model_config_hash"]
print(f"Same model_config_hash? {same_hash}")
if same_hash:
    print("  NOTE: model_config_hash matches, but K is a training-time param (not in model config)")

results["checkpoint"] = {
    "168a_epoch": k4["eval_config"]["checkpoint_epoch"],
    "baseline_epoch": k16["eval_config"]["checkpoint_epoch"],
    "168a_config_hash": k4["eval_config"]["model_config_hash"],
    "baseline_config_hash": k16["eval_config"]["model_config_hash"],
    "same_config_hash": same_hash,
    "168a_K": 4,
    "baseline_K": 16,
}

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)
print(f"""
168a (K=4): {k4_pass_count}/9 PASS  [{', '.join(n.split(' ')[0] for n, d in suite_comparison.items() if d['168a_K4']=='PASS')}]
Baseline (K=16): {k16_pass_count}/9 PASS  [{', '.join(n.split(' ')[0] for n, d in suite_comparison.items() if d['baseline_K16']=='PASS')}]

KEY DIFFERENCES:
  S9 Cross-Cell: 168a rank_ratio=0.40 (FAIL) vs baseline=0.86 (PASS)
    -> K=4 collapses to rank-1 (eff_rank=2.02 vs GT=5.03)
    -> K=16 preserves factor structure (eff_rank=4.30)

  S2 Coverage: Both FAIL but 168a much worse
    -> 168a worst cell: 0.531 vs baseline 0.652 at h=30
    -> 168a calibration error: 0.109 vs baseline 0.046
    -> 168a 90% coverage: 0.734 vs baseline 0.813

  S1 Explosion: 168a at 4.55% (borderline at 5% gate)
    -> Baseline at 2.47% (comfortable)
    -> 168a has more negative IV floor breaches

  S4 Kurtosis: 168a=0.66 (barely passes) vs baseline=0.97 (near-perfect)
    -> K=4 destroys kurtosis (gen=50.9 vs GT=77.0)
    -> K=16 preserves it (gen=75.1)

  S3 Conditionality: Both PASS but 168a has TIGHTER intervals
    -> 168a width_ratio=0.56 vs baseline=0.70
    -> 168a turb/calm=1.25 vs baseline=1.18 (168a is MORE conditional)
    -> But 168a worst_cell_wr=0.98 (barely passes) vs baseline=1.58 (FAILS)
    -> 168a S3 PASSES because worst cell WR < 1.0 threshold

  S8 Distributional: Both FAIL
    -> 168a: 25/25 KS pass vs baseline 24/25
    -> But 168a: 174 bad windows (14.2%) vs baseline 91 (7.4%)
    -> 168a: p10_cov=0.42 vs baseline=0.57

  S7 Regime: Both FAIL, but 168a is worse
    -> 168a calm h=1 coverage: 0.835 vs baseline 0.882
    -> 168a layer2: 1/8 passing vs baseline 0/8 (168a technically better here)
""")

results["summary"] = {
    "168a_pass_count": k4_pass_count,
    "baseline_pass_count": k16_pass_count,
    "168a_passing_suites": [n.split(" ")[0] for n, d in suite_comparison.items() if d["168a_K4"] == "PASS"],
    "baseline_passing_suites": [n.split(" ")[0] for n, d in suite_comparison.items() if d["baseline_K16"] == "PASS"],
    "verdict": "K=4 LOSES S9, degrades S2/S4/S1/S8. K=16 is strictly better.",
    "k4_advantages": [
        "Tighter conditional intervals (width_ratio 0.56 vs 0.70)",
        "Better turb/calm differentiation (1.25 vs 1.18)",
        "Perfect KS daily changes (25/25 vs 24/25)",
        "S3 worst_cell_wr passes (0.98 vs 1.58)",
    ],
    "k4_disadvantages": [
        "S9 REGRESSION: rank collapse (eff_rank 2.02 vs 4.30, GT=5.03)",
        "S2 much worse: calibration error 0.109 vs 0.046",
        "S1 borderline: explosion rate 4.55% (5% gate)",
        "S4 kurtosis degraded: 0.66 vs 0.97",
        "S8 more bad windows: 174 vs 91",
        "Still improving at ep20 -- undertrained",
    ],
}

# Save results
out_dir = ROOT / "results/validations/2026-04-04"
with open(out_dir / "analysis/168a_followup/vs_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved analysis to: {out_dir / 'analysis/168a_followup/vs_baseline.json'}")

# Verification results
verification = {
    "comparison": "168a (K=4) vs 164a_v3 baseline (K=16)",
    "date": "2026-04-04",
    "168a_model": "models/backfill/afcrps_168a/best_model.pt",
    "baseline_model": "models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt",
    "168a_results": "results/block_ar/168a_best_30d/summary.json",
    "baseline_results": "results/block_ar/164a_v3_percell_bptt_softplus_best_30d/summary.json",
    "168a_K": 4,
    "baseline_K": 16,
    "168a_epochs_trained": 20,
    "baseline_epochs_trained": 80,
    "168a_best_epoch": 11,
    "baseline_best_epoch": 11,
    "168a_pass_count": k4_pass_count,
    "baseline_pass_count": k16_pass_count,
    "suite_results": suite_comparison,
    "key_metrics": {
        "s1_explosion_rate": {"168a": 0.0455, "baseline": 0.0247},
        "s2_90_coverage": {"168a": 0.734, "baseline": 0.813},
        "s2_calibration_error": {"168a": 0.109, "baseline": 0.046},
        "s3_turb_calm": {"168a": 1.250, "baseline": 1.185},
        "s4_kurtosis_ratio": {"168a": 0.661, "baseline": 0.974},
        "s8_pct_bad_windows": {"168a": 0.142, "baseline": 0.074},
        "s9_rank_ratio": {"168a": 0.401, "baseline": 0.855},
        "s9_eff_rank": {"168a": 2.016, "baseline": 4.301},
    },
    "verdict": "K=4 is strictly worse than K=16. Loses S9 (rank collapse), degrades S2/S4/S1/S8. The 4-member ensemble lacks sufficient diversity to represent the 5-factor GT correlation structure.",
    "training_still_improving": True,
}
with open(out_dir / "verification_results/168a_vs_baseline.json", "w") as f:
    json.dump(verification, f, indent=2)
print(f"Saved verification to: {out_dir / 'verification_results/168a_vs_baseline.json'}")
