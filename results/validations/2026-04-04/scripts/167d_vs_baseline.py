"""
167d vs Baseline (164a_v3 softplus best) — Suite-by-suite deep comparison.

Reads both summary.json files and produces a comprehensive metric-by-metric
comparison highlighting exactly where factorization helps vs hurts.

Usage:
    PYTHONPATH=. python results/validations/2026-04-04/scripts/167d_vs_baseline.py
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/max/Documents/vol-surface-vae-pub")

# Load both summary files
with open(ROOT / "results/block_ar/167d_best_30d/summary.json") as f:
    d167 = json.load(f)

with open(ROOT / "results/block_ar/164a_v3_percell_bptt_softplus_best_30d/summary.json") as f:
    base = json.load(f)

OUT_DIR = ROOT / "results/validations/2026-04-04/analysis/167d_followup"
VERIFY_DIR = ROOT / "results/validations/2026-04-04/verification_results"

# ============================================================
# Helper
# ============================================================

def delta(a, b):
    """Return a - b with sign indicating 167d vs baseline."""
    return a - b

def pf(val):
    return "PASS" if val else "FAIL"

def arrow(d_val, higher_is_better=True):
    """Return arrow indicating improvement/regression for 167d."""
    if abs(d_val) < 1e-9:
        return "="
    if higher_is_better:
        return "IMPROVED" if d_val > 0 else "REGRESSED"
    else:
        return "IMPROVED" if d_val < 0 else "REGRESSED"


# ============================================================
# Suite-by-suite comparison
# ============================================================

results = {}
print("=" * 80)
print("167d (E2E factorized, ep18) vs Baseline (164a_v3 softplus, ep11)")
print("=" * 80)

# --- S1: Surface Validity ---
print("\n--- S1: Surface Validity ---")
s1 = {}
s1["167d_pass"] = d167["surface"]["overall_pass"]
s1["base_pass"] = base["surface"]["overall_pass"]

s1["167d_explosion_rate"] = d167["surface"]["explosion"]["explosion_total_rate"]
s1["base_explosion_rate"] = base["surface"]["explosion"]["explosion_total_rate"]
s1["delta_explosion"] = delta(s1["167d_explosion_rate"], s1["base_explosion_rate"])

s1["167d_calendar_worst"] = d167["surface"]["calendar"]["worst_strike_rate"]
s1["base_calendar_worst"] = base["surface"]["calendar"]["worst_strike_rate"]
s1["delta_calendar"] = delta(s1["167d_calendar_worst"], s1["base_calendar_worst"])

s1["167d_butterfly_worst"] = d167["surface"]["butterfly"]["worst_tenor_rate"]
s1["base_butterfly_worst"] = base["surface"]["butterfly"]["worst_tenor_rate"]
s1["delta_butterfly"] = delta(s1["167d_butterfly_worst"], s1["base_butterfly_worst"])

print(f"  Pass:          167d={pf(s1['167d_pass'])}  base={pf(s1['base_pass'])}")
print(f"  Explosion:     167d={s1['167d_explosion_rate']:.4f}  base={s1['base_explosion_rate']:.4f}  delta={s1['delta_explosion']:+.4f} [{arrow(s1['delta_explosion'], False)}]")
print(f"  Calendar arb:  167d={s1['167d_calendar_worst']:.4f}  base={s1['base_calendar_worst']:.4f}  delta={s1['delta_calendar']:+.4f} [{arrow(s1['delta_calendar'], False)}]")
print(f"  Butterfly arb: 167d={s1['167d_butterfly_worst']:.4f}  base={s1['base_butterfly_worst']:.4f}  delta={s1['delta_butterfly']:+.4f} [{arrow(s1['delta_butterfly'], False)}]")
results["S1_surface"] = s1

# --- S2: CI Coverage ---
print("\n--- S2: CI Coverage ---")
s2 = {}
s2["167d_pass"] = d167["coverage"]["overall_pass"]
s2["base_pass"] = base["coverage"]["overall_pass"]

s2["167d_overall_90"] = d167["coverage"]["overall"]["0.9"]
s2["base_overall_90"] = base["coverage"]["overall"]["0.9"]
s2["delta_overall_90"] = delta(s2["167d_overall_90"], s2["base_overall_90"])

s2["167d_calib_error"] = d167["coverage"]["calibration_error"]
s2["base_calib_error"] = base["coverage"]["calibration_error"]
s2["delta_calib_error"] = delta(s2["167d_calib_error"], s2["base_calib_error"])

# Per-horizon coverage at 90%
s2["per_horizon_90"] = {}
for h in ["1", "7", "14", "30"]:
    d_val = d167["coverage"]["per_horizon"][h]["0.9"]
    b_val = base["coverage"]["per_horizon"][h]["0.9"]
    s2["per_horizon_90"][h] = {
        "167d": d_val, "base": b_val,
        "delta": delta(d_val, b_val)
    }

# Worst cell per horizon
s2["worst_cell_per_horizon"] = {}
for h in ["1", "7", "14", "30"]:
    d_val = d167["coverage"]["worst_cell_per_horizon"][h]
    b_val = base["coverage"]["worst_cell_per_horizon"][h]
    s2["worst_cell_per_horizon"][h] = {
        "167d": d_val, "base": b_val,
        "delta": delta(d_val, b_val)
    }

s2["167d_worst_cell_pass"] = d167["coverage"]["worst_cell_pass"]
s2["base_worst_cell_pass"] = base["coverage"]["worst_cell_pass"]

# Per-cell coverage deep analysis
print(f"  Pass:          167d={pf(s2['167d_pass'])}  base={pf(s2['base_pass'])}")
print(f"  Overall 90%:   167d={s2['167d_overall_90']:.4f}  base={s2['base_overall_90']:.4f}  delta={s2['delta_overall_90']:+.4f} [{arrow(s2['delta_overall_90'])}]")
print(f"  Calib error:   167d={s2['167d_calib_error']:.4f}  base={s2['base_calib_error']:.4f}  delta={s2['delta_calib_error']:+.4f} [{arrow(s2['delta_calib_error'], False)}]")
print(f"  Worst cell pass: 167d={pf(s2['167d_worst_cell_pass'])}  base={pf(s2['base_worst_cell_pass'])}")
print("  Per-horizon 90% coverage:")
for h in ["1", "7", "14", "30"]:
    info = s2["per_horizon_90"][h]
    print(f"    h={h:>2}: 167d={info['167d']:.4f}  base={info['base']:.4f}  delta={info['delta']:+.4f} [{arrow(info['delta'])}]")
print("  Worst cell per horizon:")
for h in ["1", "7", "14", "30"]:
    info = s2["worst_cell_per_horizon"][h]
    print(f"    h={h:>2}: 167d={info['167d']:.4f}  base={info['base']:.4f}  delta={info['delta']:+.4f} [{arrow(info['delta'])}]")

# Per-cell coverage comparison (horizon=30, most challenging)
d_cells_30 = np.array(d167["coverage"]["per_cell_coverage"]["30"])
b_cells_30 = np.array(base["coverage"]["per_cell_coverage"]["30"])
cell_delta_30 = d_cells_30 - b_cells_30
n_improved_cells = int(np.sum(cell_delta_30 > 0))
n_regressed_cells = int(np.sum(cell_delta_30 < 0))
s2["h30_cells_improved"] = n_improved_cells
s2["h30_cells_regressed"] = n_regressed_cells
s2["h30_mean_delta"] = float(np.mean(cell_delta_30))
s2["h30_worst_delta"] = float(np.min(cell_delta_30))
s2["h30_best_delta"] = float(np.max(cell_delta_30))

print(f"  Per-cell h=30: {n_improved_cells} improved, {n_regressed_cells} regressed, mean_delta={s2['h30_mean_delta']:+.4f}")
print(f"    Worst delta: {s2['h30_worst_delta']:+.4f}  Best delta: {s2['h30_best_delta']:+.4f}")

# Count cells below 70% at h=30
d_below_70 = int(np.sum(d_cells_30 < 0.70))
b_below_70 = int(np.sum(b_cells_30 < 0.70))
s2["h30_below_70pct_167d"] = d_below_70
s2["h30_below_70pct_base"] = b_below_70
print(f"    Cells below 70% at h=30: 167d={d_below_70}  base={b_below_70}")

results["S2_coverage"] = s2

# --- S3: Conditionality ---
print("\n--- S3: Conditionality ---")
s3 = {}
s3["167d_pass"] = d167["conditionality"]["overall_pass"]
s3["base_pass"] = base["conditionality"]["overall_pass"]

s3["167d_width_ratio"] = d167["conditionality"]["width_ratio"]
s3["base_width_ratio"] = base["conditionality"]["width_ratio"]
s3["167d_width_pass"] = d167["conditionality"]["width_pass"]
s3["base_width_pass"] = base["conditionality"]["width_pass"]

s3["167d_turb_calm"] = d167["conditionality"]["turb_calm_ratio"]
s3["base_turb_calm"] = base["conditionality"]["turb_calm_ratio"]

s3["167d_mae_reduction"] = d167["conditionality"]["mae_reduction_pct"]
s3["base_mae_reduction"] = base["conditionality"]["mae_reduction_pct"]

s3["167d_worst_cell_mae"] = d167["conditionality"]["worst_cell_mae_reduction"]
s3["base_worst_cell_mae"] = base["conditionality"]["worst_cell_mae_reduction"]

print(f"  Pass:           167d={pf(s3['167d_pass'])}  base={pf(s3['base_pass'])}")
print(f"  Width ratio:    167d={s3['167d_width_ratio']:.4f} ({pf(s3['167d_width_pass'])})  base={s3['base_width_ratio']:.4f} ({pf(s3['base_width_pass'])})")
print(f"  Turb/calm:      167d={s3['167d_turb_calm']:.4f}  base={s3['base_turb_calm']:.4f}")
print(f"  MAE reduction:  167d={s3['167d_mae_reduction']:.2f}%  base={s3['base_mae_reduction']:.2f}%")
print(f"  Worst cell MAE: 167d={s3['167d_worst_cell_mae']:.2f}%  base={s3['base_worst_cell_mae']:.2f}%")
results["S3_conditionality"] = s3

# --- S4: Time Series ---
print("\n--- S4: Time Series ---")
s4 = {}
s4["167d_pass"] = d167["time_series"]["overall_pass"]
s4["base_pass"] = base["time_series"]["overall_pass"]

s4["167d_acf_corr"] = d167["time_series"]["acf"]["acf_correlation"]
s4["base_acf_corr"] = base["time_series"]["acf"]["acf_correlation"]

s4["167d_kurtosis_ratio"] = d167["time_series"]["kurtosis"]["kurtosis_ratio"]
s4["base_kurtosis_ratio"] = base["time_series"]["kurtosis"]["kurtosis_ratio"]
# Kurtosis: ideal is 1.0, so closer to 1 is better
s4["167d_kurtosis_dist_from_1"] = abs(s4["167d_kurtosis_ratio"] - 1.0)
s4["base_kurtosis_dist_from_1"] = abs(s4["base_kurtosis_ratio"] - 1.0)

print(f"  Pass:          167d={pf(s4['167d_pass'])}  base={pf(s4['base_pass'])}")
print(f"  ACF corr:      167d={s4['167d_acf_corr']:.6f}  base={s4['base_acf_corr']:.6f}")
print(f"  Kurtosis ratio:167d={s4['167d_kurtosis_ratio']:.4f}  base={s4['base_kurtosis_ratio']:.4f}  (ideal=1.0)")
print(f"    |dist from 1|: 167d={s4['167d_kurtosis_dist_from_1']:.4f}  base={s4['base_kurtosis_dist_from_1']:.4f}")
results["S4_time_series"] = s4

# --- S5: Block-AR Boundary ---
print("\n--- S5: Block-AR Boundary ---")
s5 = {}
s5["167d_pass"] = d167["block_ar"]["overall_pass"]
s5["base_pass"] = base["block_ar"]["overall_pass"]

s5["167d_boundary_ratio"] = d167["block_ar"]["boundary_smoothness"]["boundary_ratio"]
s5["base_boundary_ratio"] = base["block_ar"]["boundary_smoothness"]["boundary_ratio"]

s5["167d_monotonic"] = d167["block_ar"]["growing_uncertainty"]["monotonic"]
s5["base_monotonic"] = base["block_ar"]["growing_uncertainty"]["monotonic"]

# Variance growth
s5["167d_var_h30"] = d167["block_ar"]["growing_uncertainty"]["key_horizon_vars"]["30"]
s5["base_var_h30"] = base["block_ar"]["growing_uncertainty"]["key_horizon_vars"]["30"]

print(f"  Pass:           167d={pf(s5['167d_pass'])}  base={pf(s5['base_pass'])}")
print(f"  Boundary ratio: 167d={s5['167d_boundary_ratio']:.4f}  base={s5['base_boundary_ratio']:.4f}")
print(f"  Monotonic:      167d={s5['167d_monotonic']}  base={s5['base_monotonic']}")
print(f"  Var h=30:       167d={s5['167d_var_h30']:.6f}  base={s5['base_var_h30']:.6f}")
results["S5_block_ar"] = s5

# --- S6: Cointegration ---
print("\n--- S6: Cointegration ---")
s6 = {}
s6["167d_pass"] = d167["cointegration"]["overall_pass"]
s6["base_pass"] = base["cointegration"]["overall_pass"]

s6["167d_gen_gt_ratio"] = d167["cointegration"]["gen_gt_ratio"]
s6["base_gen_gt_ratio"] = base["cointegration"]["gen_gt_ratio"]

s6["167d_gen_gt_ratio_legacy"] = d167["cointegration"]["gen_gt_ratio_legacy"]
s6["base_gen_gt_ratio_legacy"] = base["cointegration"]["gen_gt_ratio_legacy"]

s6["167d_worst_cell_ratio"] = d167["cointegration"]["worst_cell_ratio"]
s6["base_worst_cell_ratio"] = base["cointegration"]["worst_cell_ratio"]

s6["167d_mean_rsq"] = d167["cointegration"]["gen_mean_rsq"]
s6["base_mean_rsq"] = base["cointegration"]["gen_mean_rsq"]

print(f"  Pass:            167d={pf(s6['167d_pass'])}  base={pf(s6['base_pass'])}")
print(f"  Gen/GT ratio:    167d={s6['167d_gen_gt_ratio']:.4f}  base={s6['base_gen_gt_ratio']:.4f}")
print(f"  Legacy ratio:    167d={s6['167d_gen_gt_ratio_legacy']:.4f}  base={s6['base_gen_gt_ratio_legacy']:.4f}")
print(f"  Worst cell:      167d={s6['167d_worst_cell_ratio']:.4f}  base={s6['base_worst_cell_ratio']:.4f}")
print(f"  Mean R-sq:       167d={s6['167d_mean_rsq']:.4f}  base={s6['base_mean_rsq']:.4f}")
results["S6_cointegration"] = s6

# --- S7: Regime Coverage ---
print("\n--- S7: Regime Coverage ---")
s7 = {}
s7["167d_pass"] = d167["regime_coverage"]["overall_pass"]
s7["base_pass"] = base["regime_coverage"]["overall_pass"]
s7["167d_layer1_pass"] = d167["regime_coverage"]["layer1_pass"]
s7["base_layer1_pass"] = base["regime_coverage"]["layer1_pass"]
s7["167d_layer2_pass"] = d167["regime_coverage"]["layer2_pass"]
s7["base_layer2_pass"] = base["regime_coverage"]["layer2_pass"]
s7["167d_layer2_n_passing"] = d167["regime_coverage"]["layer2_n_passing"]
s7["base_layer2_n_passing"] = base["regime_coverage"]["layer2_n_passing"]
s7["167d_layer3_catastrophic"] = d167["regime_coverage"]["layer3_catastrophic_rate"]
s7["base_layer3_catastrophic"] = base["regime_coverage"]["layer3_catastrophic_rate"]

# Per-regime horizon coverage
s7["regime_horizon_coverage"] = {}
for regime in ["calm", "turb"]:
    for h in ["1", "7", "14", "30"]:
        d_cov = d167["regime_coverage"]["layer1_regime_horizon"][regime][h]["coverage"]
        b_cov = base["regime_coverage"]["layer1_regime_horizon"][regime][h]["coverage"]
        key = f"{regime}_h{h}"
        s7["regime_horizon_coverage"][key] = {
            "167d": d_cov, "base": b_cov, "delta": delta(d_cov, b_cov)
        }

print(f"  Pass:            167d={pf(s7['167d_pass'])}  base={pf(s7['base_pass'])}")
print(f"  Layer1 pass:     167d={pf(s7['167d_layer1_pass'])}  base={pf(s7['base_layer1_pass'])}")
print(f"  Layer2 passing:  167d={s7['167d_layer2_n_passing']}/8  base={s7['base_layer2_n_passing']}/8")
print(f"  Layer3 catast:   167d={s7['167d_layer3_catastrophic']:.4f}  base={s7['base_layer3_catastrophic']:.4f}")
print("  Regime coverage by horizon:")
for regime in ["calm", "turb"]:
    for h in ["1", "7", "14", "30"]:
        key = f"{regime}_h{h}"
        info = s7["regime_horizon_coverage"][key]
        print(f"    {regime:5s} h={h:>2}: 167d={info['167d']:.4f}  base={info['base']:.4f}  delta={info['delta']:+.4f} [{arrow(info['delta'])}]")
results["S7_regime_coverage"] = s7

# --- S8: Distributional ---
print("\n--- S8: Distributional ---")
s8 = {}
s8["167d_pass"] = d167["distributional"]["overall_pass"]
s8["base_pass"] = base["distributional"]["overall_pass"]

# KS daily changes
s8["167d_ks_daily_n_pass"] = d167["distributional"]["ks_test"]["n_pass"]
s8["base_ks_daily_n_pass"] = base["distributional"]["ks_test"]["n_pass"]
s8["167d_ks_daily_worst"] = d167["distributional"]["ks_test"]["worst_stat"]
s8["base_ks_daily_worst"] = base["distributional"]["ks_test"]["worst_stat"]
s8["167d_ks_daily_median"] = d167["distributional"]["ks_test"]["median_stat"]
s8["base_ks_daily_median"] = base["distributional"]["ks_test"]["median_stat"]

# KS IV levels
s8["167d_ks_level_n_pass"] = d167["distributional"]["ks_level_test"]["n_pass"]
s8["base_ks_level_n_pass"] = base["distributional"]["ks_level_test"]["n_pass"]
s8["167d_ks_level_worst"] = d167["distributional"]["ks_level_test"]["worst_stat"]
s8["base_ks_level_worst"] = base["distributional"]["ks_level_test"]["worst_stat"]
s8["167d_ks_level_pass"] = d167["distributional"]["ks_level_test"]["pass"]
s8["base_ks_level_pass"] = base["distributional"]["ks_level_test"]["pass"]

# Median bias
s8["167d_median_n_pass"] = d167["distributional"]["median_bias"]["n_pass"]
s8["base_median_n_pass"] = base["distributional"]["median_bias"]["n_pass"]
s8["167d_median_frac_pass"] = d167["distributional"]["median_bias"]["frac_pass"]
s8["base_median_frac_pass"] = base["distributional"]["median_bias"]["frac_pass"]

# Window floor
s8["167d_bad_windows"] = d167["distributional"]["window_floor"]["n_bad_windows"]
s8["base_bad_windows"] = base["distributional"]["window_floor"]["n_bad_windows"]
s8["167d_p10_cov"] = d167["distributional"]["window_floor"]["p10_cov"]
s8["base_p10_cov"] = base["distributional"]["window_floor"]["p10_cov"]

# Cell MAE
s8["167d_cell_mae_n_pass"] = d167["distributional"]["cell_mae"]["n_pass"]
s8["base_cell_mae_n_pass"] = base["distributional"]["cell_mae"]["n_pass"]

print(f"  Pass:            167d={pf(s8['167d_pass'])}  base={pf(s8['base_pass'])}")
print(f"  KS daily changes: 167d={s8['167d_ks_daily_n_pass']}/25  base={s8['base_ks_daily_n_pass']}/25  worst: 167d={s8['167d_ks_daily_worst']:.4f} base={s8['base_ks_daily_worst']:.4f}")
print(f"  KS IV levels:     167d={s8['167d_ks_level_n_pass']}/25 ({pf(s8['167d_ks_level_pass'])})  base={s8['base_ks_level_n_pass']}/25 ({pf(s8['base_ks_level_pass'])})")
print(f"    worst:          167d={s8['167d_ks_level_worst']:.4f}  base={s8['base_ks_level_worst']:.4f}")
print(f"  Median bias:      167d={s8['167d_median_n_pass']}/25  base={s8['base_median_n_pass']}/25")
print(f"  Bad windows:      167d={s8['167d_bad_windows']}  base={s8['base_bad_windows']}")
print(f"  P10 coverage:     167d={s8['167d_p10_cov']:.4f}  base={s8['base_p10_cov']:.4f}")
print(f"  Cell MAE:         167d={s8['167d_cell_mae_n_pass']}/25  base={s8['base_cell_mae_n_pass']}/25")
results["S8_distributional"] = s8

# --- S9: Cross-Cell Correlation ---
print("\n--- S9: Cross-Cell Correlation ---")
s9 = {}
s9["167d_pass"] = d167["cross_cell_correlation"]["overall_pass"]
s9["base_pass"] = base["cross_cell_correlation"]["overall_pass"]

s9["gt_mean_corr"] = d167["cross_cell_correlation"]["gt_mean_corr"]
s9["167d_gen_corr"] = d167["cross_cell_correlation"]["gen_mean_corr"]
s9["base_gen_corr"] = base["cross_cell_correlation"]["gen_mean_corr"]

s9["167d_corr_ratio"] = d167["cross_cell_correlation"]["corr_ratio"]
s9["base_corr_ratio"] = base["cross_cell_correlation"]["corr_ratio"]
# Ideal corr_ratio is 1.0
s9["167d_corr_dist_from_1"] = abs(s9["167d_corr_ratio"] - 1.0)
s9["base_corr_dist_from_1"] = abs(s9["base_corr_ratio"] - 1.0)

s9["gt_eff_rank"] = d167["cross_cell_correlation"]["gt_eff_rank"]
s9["167d_eff_rank"] = d167["cross_cell_correlation"]["gen_eff_rank"]
s9["base_eff_rank"] = base["cross_cell_correlation"]["gen_eff_rank"]

s9["167d_rank_ratio"] = d167["cross_cell_correlation"]["rank_ratio"]
s9["base_rank_ratio"] = base["cross_cell_correlation"]["rank_ratio"]
# Ideal rank_ratio is 1.0
s9["167d_rank_dist_from_1"] = abs(s9["167d_rank_ratio"] - 1.0)
s9["base_rank_dist_from_1"] = abs(s9["base_rank_ratio"] - 1.0)

s9["167d_frob_dist"] = d167["cross_cell_correlation"]["frob_dist"]
s9["base_frob_dist"] = base["cross_cell_correlation"]["frob_dist"]

s9["167d_pc1_var"] = d167["cross_cell_correlation"]["gen_pc1_var"]
s9["base_pc1_var"] = base["cross_cell_correlation"]["gen_pc1_var"]
s9["gt_pc1_var"] = d167["cross_cell_correlation"]["gt_pc1_var"]

print(f"  Pass:            167d={pf(s9['167d_pass'])}  base={pf(s9['base_pass'])}")
print(f"  GT mean corr:    {s9['gt_mean_corr']:.4f}")
print(f"  Gen mean corr:   167d={s9['167d_gen_corr']:.4f}  base={s9['base_gen_corr']:.4f}")
print(f"  Corr ratio:      167d={s9['167d_corr_ratio']:.4f}  base={s9['base_corr_ratio']:.4f}  (ideal=1.0)")
print(f"    |dist from 1|: 167d={s9['167d_corr_dist_from_1']:.4f}  base={s9['base_corr_dist_from_1']:.4f}  [{arrow(-delta(s9['167d_corr_dist_from_1'], s9['base_corr_dist_from_1']), True)}]")
print(f"  GT eff rank:     {s9['gt_eff_rank']:.4f}")
print(f"  Gen eff rank:    167d={s9['167d_eff_rank']:.4f}  base={s9['base_eff_rank']:.4f}")
print(f"  Rank ratio:      167d={s9['167d_rank_ratio']:.4f}  base={s9['base_rank_ratio']:.4f}  (ideal=1.0)")
print(f"    |dist from 1|: 167d={s9['167d_rank_dist_from_1']:.4f}  base={s9['base_rank_dist_from_1']:.4f}  [{arrow(-delta(s9['167d_rank_dist_from_1'], s9['base_rank_dist_from_1']), True)}]")
print(f"  Frob dist:       167d={s9['167d_frob_dist']:.4f}  base={s9['base_frob_dist']:.4f}  [{arrow(-delta(s9['167d_frob_dist'], s9['base_frob_dist']), True)}]")
print(f"  PC1 variance:    167d={s9['167d_pc1_var']:.4f}  base={s9['base_pc1_var']:.4f}  GT={s9['gt_pc1_var']:.4f}")
results["S9_cross_cell"] = s9

# ============================================================
# Summary Table
# ============================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY TABLE")
print("=" * 80)

suites = [
    ("S1", "Surface Validity", d167["surface"]["overall_pass"], base["surface"]["overall_pass"]),
    ("S2", "CI Coverage", d167["coverage"]["overall_pass"], base["coverage"]["overall_pass"]),
    ("S3", "Conditionality", d167["conditionality"]["overall_pass"], base["conditionality"]["overall_pass"]),
    ("S4", "Time Series", d167["time_series"]["overall_pass"], base["time_series"]["overall_pass"]),
    ("S5", "Block-AR Boundary", d167["block_ar"]["overall_pass"], base["block_ar"]["overall_pass"]),
    ("S6", "Cointegration", d167["cointegration"]["overall_pass"], base["cointegration"]["overall_pass"]),
    ("S7", "Regime Coverage", d167["regime_coverage"]["overall_pass"], base["regime_coverage"]["overall_pass"]),
    ("S8", "Distributional", d167["distributional"]["overall_pass"], base["distributional"]["overall_pass"]),
    ("S9", "Cross-Cell Corr", d167["cross_cell_correlation"]["overall_pass"], base["cross_cell_correlation"]["overall_pass"]),
]

d_total = sum(1 for _, _, d, _ in suites if d)
b_total = sum(1 for _, _, _, b in suites if b)

print(f"\n{'Suite':<5} {'Name':<20} {'167d':<8} {'Baseline':<8} {'Change':<12}")
print("-" * 55)
for sid, name, d_pass, b_pass in suites:
    change = ""
    if d_pass and not b_pass:
        change = "GAINED"
    elif not d_pass and b_pass:
        change = "LOST"
    elif d_pass and b_pass:
        change = "same (PASS)"
    else:
        change = "same (FAIL)"
    print(f"{sid:<5} {name:<20} {pf(d_pass):<8} {pf(b_pass):<8} {change:<12}")

print(f"\nTotal: 167d={d_total}/9  Baseline={b_total}/9")

# Identify key deltas
print("\n" + "=" * 80)
print("KEY METRIC DELTAS (167d vs baseline)")
print("=" * 80)

key_metrics = {
    "Overall 90% coverage": (s2["167d_overall_90"], s2["base_overall_90"], True, "higher=better"),
    "Calibration error": (s2["167d_calib_error"], s2["base_calib_error"], False, "lower=better"),
    "Worst cell h=30": (s2["worst_cell_per_horizon"]["30"]["167d"], s2["worst_cell_per_horizon"]["30"]["base"], True, "higher=better"),
    "Width ratio (cond/uncond)": (s3["167d_width_ratio"], s3["base_width_ratio"], False, "lower=better, <1 means cond tighter"),
    "Turb/calm ratio": (s3["167d_turb_calm"], s3["base_turb_calm"], True, "higher=better, >1.15 pass"),
    "MAE reduction %": (s3["167d_mae_reduction"], s3["base_mae_reduction"], True, "higher=better"),
    "Kurtosis ratio (ideal=1)": (s4["167d_kurtosis_ratio"], s4["base_kurtosis_ratio"], None, "closer to 1 better"),
    "ACF correlation": (s4["167d_acf_corr"], s4["base_acf_corr"], True, "higher=better"),
    "Coint gen/GT ratio": (s6["167d_gen_gt_ratio_legacy"], s6["base_gen_gt_ratio_legacy"], None, "closer to 1 better"),
    "KS daily n_pass": (s8["167d_ks_daily_n_pass"], s8["base_ks_daily_n_pass"], True, "higher=better"),
    "KS level n_pass": (s8["167d_ks_level_n_pass"], s8["base_ks_level_n_pass"], True, "higher=better"),
    "Median bias n_pass": (s8["167d_median_n_pass"], s8["base_median_n_pass"], True, "higher=better"),
    "Corr ratio (ideal=1)": (s9["167d_corr_ratio"], s9["base_corr_ratio"], None, "closer to 1 better"),
    "Rank ratio (ideal=1)": (s9["167d_rank_ratio"], s9["base_rank_ratio"], None, "closer to 1 better"),
    "Eff rank": (s9["167d_eff_rank"], s9["base_eff_rank"], None, f"GT={s9['gt_eff_rank']:.2f}"),
    "Frob distance": (s9["167d_frob_dist"], s9["base_frob_dist"], False, "lower=better"),
    "Layer3 catast rate": (s7["167d_layer3_catastrophic"], s7["base_layer3_catastrophic"], False, "lower=better"),
    "Bad windows": (s8["167d_bad_windows"], s8["base_bad_windows"], False, "lower=better"),
}

print(f"\n{'Metric':<30} {'167d':>10} {'Base':>10} {'Delta':>10} {'Note':<35}")
print("-" * 100)
for name, (d_val, b_val, higher_better, note) in key_metrics.items():
    d = d_val - b_val
    print(f"{name:<30} {d_val:>10.4f} {b_val:>10.4f} {d:>+10.4f} {note:<35}")


# ============================================================
# Verdict
# ============================================================

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Identify what changed
gains = [sid for sid, _, d, b in suites if d and not b]
losses = [sid for sid, _, d, b in suites if not d and b]

print(f"\n167d score: {d_total}/9  |  Baseline score: {b_total}/9")
if gains:
    print(f"  Suites GAINED by 167d: {gains}")
if losses:
    print(f"  Suites LOST by 167d:   {losses}")

# Store verdict
results["verdict"] = {
    "167d_total": d_total,
    "baseline_total": b_total,
    "suites_gained": gains,
    "suites_lost": losses,
    "167d_checkpoint_epoch": 18,
    "baseline_checkpoint_epoch": 11,
    "model_167d": "167d E2E factorized decoder (20 epochs)",
    "model_baseline": "164a_v3_percell_bptt_softplus_best (softplus barrier, ep11)",
}

# ============================================================
# Save results
# ============================================================

# Full comparison
with open(OUT_DIR / "vs_baseline_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {OUT_DIR / 'vs_baseline_comparison.json'}")

# Verification result
verify = {
    "comparison": "167d_vs_164a_v3_softplus_best",
    "date": "2026-04-04",
    "167d_total_pass": d_total,
    "baseline_total_pass": b_total,
    "suites_gained": gains,
    "suites_lost": losses,
    "pass_fail_grid": {sid: {"167d": pf(d), "baseline": pf(b)} for sid, _, d, b in suites},
    "key_findings": {
        "s2_coverage_delta": s2["delta_overall_90"],
        "s2_167d_under_coverage": s2["167d_overall_90"] < 0.80,
        "s2_worst_cell_h30_delta": s2["worst_cell_per_horizon"]["30"]["delta"],
        "s3_conditionality_lost": s3["167d_pass"] != s3["base_pass"],
        "s3_width_ratio_167d": s3["167d_width_ratio"],
        "s3_width_ratio_base": s3["base_width_ratio"],
        "s8_ks_level_167d": s8["167d_ks_level_n_pass"],
        "s8_ks_level_base": s8["base_ks_level_n_pass"],
        "s9_corr_ratio_167d": s9["167d_corr_ratio"],
        "s9_corr_ratio_base": s9["base_corr_ratio"],
        "s9_rank_ratio_167d": s9["167d_rank_ratio"],
        "s9_rank_ratio_base": s9["base_rank_ratio"],
        "s9_167d_closer_to_gt_corr": s9["167d_corr_dist_from_1"] < s9["base_corr_dist_from_1"],
        "s9_167d_closer_to_gt_rank": s9["167d_rank_dist_from_1"] < s9["base_rank_dist_from_1"],
    }
}
with open(VERIFY_DIR / "167d_vs_baseline.json", "w") as f:
    json.dump(verify, f, indent=2, default=str)
print(f"Saved: {VERIFY_DIR / '167d_vs_baseline.json'}")
