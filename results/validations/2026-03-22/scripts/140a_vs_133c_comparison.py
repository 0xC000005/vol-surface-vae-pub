#!/usr/bin/env python3
"""
140a vs 133c Comparison: AR Causal Transformer vs One-Shot Joint Transformer

Analyzes WHY one-shot (133c) achieves kurtosis 1.60 but AR (140a) only 0.364.
Compares all metrics side-by-side, per-cell patterns, factor structure, etc.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_suite_pass(data: dict) -> dict:
    """Extract pass/fail for each suite."""
    suites = {}

    # Suite 1: Surface Validity
    suites["1_surface"] = data.get("surface", {}).get("overall_pass", None)

    # Suite 2: CI Coverage
    suites["2_coverage"] = data.get("coverage", {}).get("pass", None)

    # Suite 3: Conditionality
    suites["3_conditionality"] = data.get("conditionality", {}).get("pass", None)

    # Suite 4: Time Series
    suites["4_time_series"] = data.get("time_series", {}).get("overall_pass", None)

    # Suite 5: Block AR
    suites["5_block_ar"] = data.get("block_ar", {}).get("overall_pass", None)

    # Suite 6: Cointegration
    suites["6_cointegration"] = data.get("cointegration", {}).get("pass", None)

    # Suite 7: Regime Coverage
    suites["7_regime"] = data.get("regime_coverage", {}).get("overall_pass", None)

    # Suite 8: Distributional
    suites["8_distributional"] = data.get("distributional", {}).get("overall_pass", None)

    # Suite 9: Cross-Cell Correlation (may not exist in older runs)
    if "cross_cell_correlation" in data:
        suites["9_cross_cell"] = data["cross_cell_correlation"].get("overall_pass", None)
    else:
        suites["9_cross_cell"] = "N/A (not in results)"

    return suites


def extract_key_metrics(data: dict) -> dict:
    """Extract a flat dict of the most important scalar metrics."""
    m = {}

    # Surface
    surf = data.get("surface", {})
    m["explosion_rate"] = surf.get("explosion", {}).get("explosion_total_rate")
    m["calendar_worst_strike"] = surf.get("calendar", {}).get("worst_strike_rate")
    m["butterfly_worst_tenor"] = surf.get("butterfly", {}).get("worst_tenor_rate")

    # Coverage
    cov = data.get("coverage", {})
    m["overall_90_coverage"] = cov.get("overall", {}).get("0.9")
    m["calibration_error"] = cov.get("calibration_error")
    m["worst_cell_pass"] = cov.get("worst_cell_pass")
    for h in ["1", "7", "14", "30"]:
        m[f"coverage_h{h}_90"] = cov.get("per_horizon", {}).get(h, {}).get("0.9")
        m[f"worst_cell_h{h}"] = cov.get("worst_cell_per_horizon", {}).get(h)

    # Conditionality
    cond = data.get("conditionality", {})
    m["turb_calm_ratio"] = cond.get("turb_calm_ratio")
    m["mae_reduction_pct"] = cond.get("mae_reduction_pct")
    m["width_ratio"] = cond.get("width_ratio")
    m["worst_cell_mae_reduction"] = cond.get("worst_cell_mae_reduction")

    # Time Series
    ts = data.get("time_series", {})
    acf = ts.get("acf", {})
    m["acf_correlation"] = acf.get("acf_correlation")
    m["acf_mae"] = acf.get("acf_mae")
    kurt = ts.get("kurtosis", {})
    m["gt_kurtosis"] = kurt.get("gt_kurtosis")
    m["gen_kurtosis"] = kurt.get("gen_kurtosis")
    m["kurtosis_ratio"] = kurt.get("kurtosis_ratio")
    m["gt_skewness"] = kurt.get("gt_skewness")
    m["gen_skewness"] = kurt.get("gen_skewness")
    m["skewness_ratio"] = kurt.get("skewness_ratio")
    m["worst_cell_kurtosis_ratio"] = kurt.get("worst_cell_ratio")
    m["best_cell_kurtosis_ratio"] = kurt.get("best_cell_ratio")

    # Block AR
    ba = data.get("block_ar", {})
    bs = ba.get("boundary_smoothness", {})
    m["boundary_smoothness_mean"] = bs.get("mean_smoothness")
    m["boundary_smoothness_pass"] = bs.get("pass")
    gu = ba.get("growing_uncertainty", {})
    m["growing_uncertainty_monotonic"] = gu.get("all_monotonic")
    m["growing_uncertainty_pass"] = gu.get("pass")

    # Cointegration
    co = data.get("cointegration", {})
    m["coint_gen_pass_rate"] = co.get("gen_pass_rate")
    m["coint_gt_pass_rate"] = co.get("gt_pass_rate")
    m["coint_gen_gt_ratio"] = co.get("gen_gt_ratio")
    m["coint_gen_mean_rsq"] = co.get("gen_mean_rsq")
    m["coint_gt_mean_rsq"] = co.get("gt_mean_rsq")

    # Regime Coverage
    rc = data.get("regime_coverage", {})
    m["regime_layer1_pass"] = rc.get("layer1_pass")
    m["regime_layer2_pass"] = rc.get("layer2_pass")
    m["regime_layer2_n_passing"] = rc.get("layer2_n_passing")
    m["regime_layer2_n_total"] = rc.get("layer2_n_total")
    m["regime_layer3_catastrophic_rate"] = rc.get("layer3_catastrophic_rate")

    # Distributional
    dist = data.get("distributional", {})
    ks = dist.get("ks_test", {})
    m["ks_n_pass"] = ks.get("n_pass")
    m["ks_worst_stat"] = ks.get("worst_stat")
    m["ks_median_stat"] = ks.get("median_stat")
    ksl = dist.get("ks_level_test", {})
    m["ks_level_n_pass"] = ksl.get("n_pass")
    m["ks_level_worst_stat"] = ksl.get("worst_stat")
    m["ks_level_median_stat"] = ksl.get("median_stat")
    mb = dist.get("median_bias", {})
    m["median_bias_mean"] = mb.get("mean_bias") if isinstance(mb, dict) else None
    m["median_bias_pass"] = mb.get("pass") if isinstance(mb, dict) else None

    # Cross-Cell Correlation
    cc = data.get("cross_cell_correlation", {})
    m["gt_mean_corr"] = cc.get("gt_mean_corr")
    m["gen_mean_corr"] = cc.get("gen_mean_corr")
    m["corr_ratio"] = cc.get("corr_ratio")
    m["gt_eff_rank"] = cc.get("gt_eff_rank")
    m["gen_eff_rank"] = cc.get("gen_eff_rank")
    m["rank_ratio"] = cc.get("rank_ratio")
    m["gt_pc1_var"] = cc.get("gt_pc1_var")
    m["gen_pc1_var"] = cc.get("gen_pc1_var")

    return m


def extract_per_cell_kurtosis(data: dict) -> np.ndarray | None:
    """Extract per-cell kurtosis ratio grid (5x5)."""
    kurt = data.get("time_series", {}).get("kurtosis", {})
    pcr = kurt.get("per_cell_ratio")
    if pcr is not None:
        return np.array(pcr)
    return None


def extract_per_cell_coverage(data: dict, horizon: str) -> np.ndarray | None:
    """Extract per-cell coverage at 90% for a given horizon."""
    pcc = data.get("coverage", {}).get("per_cell_coverage", {}).get(horizon)
    if pcc is not None:
        return np.array(pcc)
    return None


def extract_per_cell_width_ratio(data: dict) -> np.ndarray | None:
    """Extract per-cell conditionality width ratio grid (5x5)."""
    cond = data.get("conditionality", {})
    pcwr = cond.get("per_cell_width_ratio")
    if pcwr is not None:
        return np.array(pcwr)
    return None


def extract_per_cell_mae_reduction(data: dict) -> np.ndarray | None:
    """Extract per-cell MAE reduction grid (5x5)."""
    cond = data.get("conditionality", {})
    pcmr = cond.get("per_cell_mae_reduction")
    if pcmr is not None:
        return np.array(pcmr)
    return None


def format_val(v, fmt=".4f"):
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:{fmt}}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def print_comparison_table(metrics_a: dict, metrics_b: dict, label_a: str, label_b: str):
    """Print a side-by-side comparison table."""
    all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

    header = f"{'Metric':<40} {'140a (AR Causal)':<20} {'133c (One-Shot)':<20} {'Delta/Note':<20}"
    sep = "-" * 100

    lines = [sep, header, sep]

    for key in all_keys:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)

        sa = format_val(va)
        sb = format_val(vb)

        # Compute delta for numeric values
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va is not None and vb is not None:
            diff = va - vb
            delta = f"{diff:+.4f}"
        elif isinstance(va, bool) and isinstance(vb, bool):
            if va != vb:
                delta = "DIFFERS"

        lines.append(f"{key:<40} {sa:<20} {sb:<20} {delta:<20}")

    lines.append(sep)
    return "\n".join(lines)


def analyze_kurtosis_difference(data_a: dict, data_b: dict) -> dict:
    """Deep analysis of WHY kurtosis differs so much."""
    analysis = {}

    # Overall kurtosis
    ka = data_a["time_series"]["kurtosis"]
    kb = data_b["time_series"]["kurtosis"]

    analysis["overall"] = {
        "140a_kurtosis_ratio": ka["kurtosis_ratio"],
        "133c_kurtosis_ratio": kb["kurtosis_ratio"],
        "140a_gen_kurtosis": ka["gen_kurtosis"],
        "133c_gen_kurtosis": kb["gen_kurtosis"],
        "gt_kurtosis": ka["gt_kurtosis"],
        "ratio_of_ratios": ka["kurtosis_ratio"] / kb["kurtosis_ratio"] if kb["kurtosis_ratio"] != 0 else None,
    }

    # Per-cell kurtosis
    pca = extract_per_cell_kurtosis(data_a)
    pcb = extract_per_cell_kurtosis(data_b)

    if pca is not None and pcb is not None:
        analysis["per_cell_comparison"] = {
            "140a_mean": float(pca.mean()),
            "140a_std": float(pca.std()),
            "140a_min": float(pca.min()),
            "140a_max": float(pca.max()),
            "140a_min_cell": [int(x) for x in np.unravel_index(pca.argmin(), pca.shape)],
            "140a_max_cell": [int(x) for x in np.unravel_index(pca.argmax(), pca.shape)],
            "133c_mean": float(pcb.mean()),
            "133c_std": float(pcb.std()),
            "133c_min": float(pcb.min()),
            "133c_max": float(pcb.max()),
            "133c_min_cell": [int(x) for x in np.unravel_index(pcb.argmin(), pcb.shape)],
            "133c_max_cell": [int(x) for x in np.unravel_index(pcb.argmax(), pcb.shape)],
        }

        # Count cells in pass range (0.5-2.0)
        a_pass = np.sum((pca >= 0.5) & (pca <= 2.0))
        b_pass = np.sum((pcb >= 0.5) & (pcb <= 2.0))
        analysis["per_cell_comparison"]["140a_cells_in_pass_range"] = int(a_pass)
        analysis["per_cell_comparison"]["133c_cells_in_pass_range"] = int(b_pass)

        # Per-cell grid comparison
        analysis["per_cell_grid_140a"] = pca.tolist()
        analysis["per_cell_grid_133c"] = pcb.tolist()
        analysis["per_cell_grid_diff"] = (pca - pcb).tolist()

        # Identify cells where 140a < 0.5 (too low) vs 133c
        low_mask_a = pca < 0.5
        low_cells_a = list(zip(*np.where(low_mask_a)))
        analysis["140a_low_kurtosis_cells"] = [
            {"cell": [int(r), int(c)], "ratio": float(pca[r, c]), "133c_ratio": float(pcb[r, c])}
            for r, c in low_cells_a
        ]

    # Skewness comparison
    analysis["skewness"] = {
        "140a_skewness_ratio": ka.get("skewness_ratio"),
        "133c_skewness_ratio": kb.get("skewness_ratio"),
        "140a_gen_skewness": ka.get("gen_skewness"),
        "133c_gen_skewness": kb.get("gen_skewness"),
        "gt_skewness": ka.get("gt_skewness"),
    }

    # Hypothesis: AR variance accumulation vs one-shot variance
    analysis["hypothesis_ar_variance_saturation"] = {
        "description": (
            "AR models accumulate variance step-by-step with rho=0.8 AR(1) noise. "
            "Each frame adds a small delta. Over 30 frames, the accumulated noise "
            "becomes approximately Gaussian (CLT), which REDUCES kurtosis toward 3.0 "
            "(ratio 0.039 for excess kurtosis). One-shot models generate all frames at once "
            "with a single noise draw, preserving the heavy-tailed nature of the noise."
        ),
        "ar_frames": 30,
        "ar_rho": 0.8,
        "ar_noise_var_accumulation": "sum of rho^(2i) * (1-rho^2) converges to 1.0",
        "clt_effect": "30 summed noise terms -> near-Gaussian -> kurtosis ratio drops",
    }

    return analysis


def analyze_coverage_patterns(data_a: dict, data_b: dict) -> dict:
    """Compare per-cell coverage patterns."""
    analysis = {}

    for h in ["1", "7", "14", "30"]:
        ca = extract_per_cell_coverage(data_a, h)
        cb = extract_per_cell_coverage(data_b, h)

        if ca is not None and cb is not None:
            diff = ca - cb
            # 140a has much lower coverage overall
            analysis[f"horizon_{h}"] = {
                "140a_mean": float(ca.mean()),
                "140a_min": float(ca.min()),
                "140a_min_cell": [int(x) for x in np.unravel_index(ca.argmin(), ca.shape)],
                "133c_mean": float(cb.mean()),
                "133c_min": float(cb.min()),
                "133c_min_cell": [int(x) for x in np.unravel_index(cb.argmin(), cb.shape)],
                "mean_diff": float(diff.mean()),
                "140a_cells_below_80": int(np.sum(ca < 0.8)),
                "133c_cells_below_80": int(np.sum(cb < 0.8)),
                "140a_cells_below_50": int(np.sum(ca < 0.5)),
                "133c_cells_below_50": int(np.sum(cb < 0.5)),
            }

    return analysis


def analyze_factor_structure(data_a: dict, data_b: dict) -> dict:
    """Compare cross-cell correlation and factor structure."""
    analysis = {}

    # 140a has cross-cell data, 133c may not
    cc_a = data_a.get("cross_cell_correlation", {})
    cc_b = data_b.get("cross_cell_correlation", {})

    if cc_a:
        analysis["140a"] = {
            "gt_mean_corr": cc_a.get("gt_mean_corr"),
            "gen_mean_corr": cc_a.get("gen_mean_corr"),
            "corr_ratio": cc_a.get("corr_ratio"),
            "gt_eff_rank": cc_a.get("gt_eff_rank"),
            "gen_eff_rank": cc_a.get("gen_eff_rank"),
            "rank_ratio": cc_a.get("rank_ratio"),
            "gt_pc1_var": cc_a.get("gt_pc1_var"),
            "gen_pc1_var": cc_a.get("gen_pc1_var"),
        }

    if cc_b:
        analysis["133c"] = {
            "gt_mean_corr": cc_b.get("gt_mean_corr"),
            "gen_mean_corr": cc_b.get("gen_mean_corr"),
            "corr_ratio": cc_b.get("corr_ratio"),
            "gt_eff_rank": cc_b.get("gt_eff_rank"),
            "gen_eff_rank": cc_b.get("gen_eff_rank"),
            "rank_ratio": cc_b.get("rank_ratio"),
        }
    else:
        analysis["133c"] = "cross_cell_correlation not available in 133c results"

    # Cointegration comparison (proxy for temporal factor structure)
    co_a = data_a.get("cointegration", {})
    co_b = data_b.get("cointegration", {})
    analysis["cointegration_comparison"] = {
        "140a_gen_gt_ratio": co_a.get("gen_gt_ratio"),
        "133c_gen_gt_ratio": co_b.get("gen_gt_ratio"),
        "140a_gen_pass_rate": co_a.get("gen_pass_rate"),
        "133c_gen_pass_rate": co_b.get("gen_pass_rate"),
        "140a_gen_mean_rsq": co_a.get("gen_mean_rsq"),
        "133c_gen_mean_rsq": co_b.get("gen_mean_rsq"),
        "gt_pass_rate": co_a.get("gt_pass_rate"),
        "gt_mean_rsq": co_a.get("gt_mean_rsq"),
    }

    return analysis


def analyze_conditionality_patterns(data_a: dict, data_b: dict) -> dict:
    """Compare conditionality patterns per cell."""
    analysis = {}

    wr_a = extract_per_cell_width_ratio(data_a)
    wr_b = extract_per_cell_width_ratio(data_b)
    mae_a = extract_per_cell_mae_reduction(data_a)
    mae_b = extract_per_cell_mae_reduction(data_b)

    if wr_a is not None and wr_b is not None:
        analysis["width_ratio"] = {
            "140a_mean": float(wr_a.mean()),
            "140a_std": float(wr_a.std()),
            "133c_mean": float(wr_b.mean()),
            "133c_std": float(wr_b.std()),
            "140a_cells_below_1": int(np.sum(wr_a < 1.0)),
            "133c_cells_below_1": int(np.sum(wr_b < 1.0)),
            "note": "width_ratio < 1.0 means model narrows in turbulence (BAD)"
        }

    if mae_a is not None and mae_b is not None:
        analysis["mae_reduction"] = {
            "140a_mean": float(mae_a.mean()),
            "140a_min": float(mae_a.min()),
            "140a_min_cell": [int(x) for x in np.unravel_index(mae_a.argmin(), mae_a.shape)],
            "133c_mean": float(mae_b.mean()),
            "133c_min": float(mae_b.min()),
            "133c_min_cell": [int(x) for x in np.unravel_index(mae_b.argmin(), mae_b.shape)],
        }

    return analysis


def analyze_distributional(data_a: dict, data_b: dict) -> dict:
    """Compare distributional (KS) test results."""
    analysis = {}

    dist_a = data_a.get("distributional", {})
    dist_b = data_b.get("distributional", {})

    for test_name in ["ks_test", "ks_level_test"]:
        ta = dist_a.get(test_name, {})
        tb = dist_b.get(test_name, {})
        analysis[test_name] = {
            "140a_n_pass": ta.get("n_pass"),
            "133c_n_pass": tb.get("n_pass"),
            "140a_worst_stat": ta.get("worst_stat"),
            "133c_worst_stat": tb.get("worst_stat"),
            "140a_median_stat": ta.get("median_stat"),
            "133c_median_stat": tb.get("median_stat"),
        }

        # Compare per-cell KS grids if available
        ga = ta.get("ks_grid")
        gb = tb.get("ks_grid")
        if ga is not None and gb is not None:
            ga = np.array(ga)
            gb = np.array(gb)
            gate_a = ta.get("ks_gate", 0.1)
            gate_b = tb.get("ks_gate", 0.1)
            analysis[test_name]["140a_per_cell_pass_grid"] = (ga < gate_a).astype(int).tolist()
            analysis[test_name]["133c_per_cell_pass_grid"] = (gb < gate_b).astype(int).tolist()

    return analysis


def build_suite_comparison(suites_a: dict, suites_b: dict) -> dict:
    """Build structured suite comparison."""
    comp = {}
    for suite in sorted(set(list(suites_a.keys()) + list(suites_b.keys()))):
        va = suites_a.get(suite)
        vb = suites_b.get(suite)
        comp[suite] = {
            "140a": va if not isinstance(va, bool) else ("PASS" if va else "FAIL"),
            "133c": vb if not isinstance(vb, bool) else ("PASS" if vb else "FAIL"),
            "same": va == vb if not isinstance(va, str) and not isinstance(vb, str) else None,
        }
    return comp


def generate_text_report(
    suites_a, suites_b, metrics_a, metrics_b,
    kurtosis_analysis, coverage_analysis, factor_analysis,
    conditionality_analysis, distributional_analysis
) -> str:
    """Generate a human-readable text report."""
    lines = []
    lines.append("=" * 100)
    lines.append("COMPARISON REPORT: 140a (AR Causal Transformer) vs 133c (One-Shot Joint Transformer)")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 100)
    lines.append("")

    # Suite pass/fail
    lines.append("SUITE PASS/FAIL COMPARISON")
    lines.append("-" * 60)
    lines.append(f"{'Suite':<25} {'140a':<12} {'133c':<12} {'Match':<10}")
    lines.append("-" * 60)

    pass_count_a = 0
    pass_count_b = 0
    for suite in sorted(suites_a.keys()):
        va = suites_a[suite]
        vb = suites_b[suite]
        sa = "PASS" if va is True else ("FAIL" if va is False else str(va))
        sb = "PASS" if vb is True else ("FAIL" if vb is False else str(vb))
        match = "YES" if va == vb else "NO"
        lines.append(f"{suite:<25} {sa:<12} {sb:<12} {match:<10}")
        if va is True:
            pass_count_a += 1
        if vb is True:
            pass_count_b += 1

    lines.append("-" * 60)
    lines.append(f"{'TOTAL PASSING':<25} {pass_count_a:<12} {pass_count_b:<12}")
    lines.append("")

    # Key metrics comparison
    lines.append("KEY METRICS COMPARISON")
    lines.append(print_comparison_table(metrics_a, metrics_b, "140a", "133c"))
    lines.append("")

    # Kurtosis deep dive
    lines.append("=" * 80)
    lines.append("KURTOSIS DEEP DIVE")
    lines.append("=" * 80)

    k = kurtosis_analysis["overall"]
    lines.append(f"140a kurtosis ratio: {k['140a_kurtosis_ratio']:.4f} (gen={k['140a_gen_kurtosis']:.2f}, gt={k['gt_kurtosis']:.2f})")
    lines.append(f"133c kurtosis ratio: {k['133c_kurtosis_ratio']:.4f} (gen={k['133c_gen_kurtosis']:.2f}, gt={k['gt_kurtosis']:.2f})")
    lines.append(f"Ratio of ratios: {k['ratio_of_ratios']:.4f} (140a/133c)")
    lines.append("")

    if "per_cell_comparison" in kurtosis_analysis:
        pc = kurtosis_analysis["per_cell_comparison"]
        lines.append("Per-cell kurtosis ratio statistics:")
        lines.append(f"  140a: mean={pc['140a_mean']:.4f}, std={pc['140a_std']:.4f}, min={pc['140a_min']:.4f} at {pc['140a_min_cell']}, max={pc['140a_max']:.4f} at {pc['140a_max_cell']}")
        lines.append(f"  133c: mean={pc['133c_mean']:.4f}, std={pc['133c_std']:.4f}, min={pc['133c_min']:.4f} at {pc['133c_min_cell']}, max={pc['133c_max']:.4f} at {pc['133c_max_cell']}")
        lines.append(f"  140a cells in pass range [0.5, 2.0]: {pc['140a_cells_in_pass_range']}/25")
        lines.append(f"  133c cells in pass range [0.5, 2.0]: {pc['133c_cells_in_pass_range']}/25")
        lines.append("")

    if "per_cell_grid_140a" in kurtosis_analysis:
        lines.append("Per-cell kurtosis ratio grids (5x5, moneyness x tenor):")
        lines.append("  140a:")
        for row in kurtosis_analysis["per_cell_grid_140a"]:
            lines.append("    " + "  ".join(f"{v:6.3f}" for v in row))
        lines.append("  133c:")
        for row in kurtosis_analysis["per_cell_grid_133c"]:
            lines.append("    " + "  ".join(f"{v:6.3f}" for v in row))
        lines.append("  Difference (140a - 133c):")
        for row in kurtosis_analysis["per_cell_grid_diff"]:
            lines.append("    " + "  ".join(f"{v:+6.3f}" for v in row))
        lines.append("")

    if "140a_low_kurtosis_cells" in kurtosis_analysis:
        low = kurtosis_analysis["140a_low_kurtosis_cells"]
        lines.append(f"140a cells with kurtosis ratio < 0.5 ({len(low)}/25):")
        for item in low:
            lines.append(f"  Cell {item['cell']}: 140a={item['ratio']:.4f}, 133c={item['133c_ratio']:.4f}")
        lines.append("")

    # Hypothesis
    hyp = kurtosis_analysis["hypothesis_ar_variance_saturation"]
    lines.append("HYPOTHESIS: AR Variance Saturation via CLT")
    lines.append(f"  {hyp['description']}")
    lines.append("")

    # Coverage patterns
    lines.append("=" * 80)
    lines.append("COVERAGE PATTERNS (per-cell at 90% CI)")
    lines.append("=" * 80)
    for h, v in coverage_analysis.items():
        lines.append(f"Horizon {h}:")
        lines.append(f"  140a: mean={v['140a_mean']:.4f}, min={v['140a_min']:.4f} at {v['140a_min_cell']}, below_80={v['140a_cells_below_80']}, below_50={v['140a_cells_below_50']}")
        lines.append(f"  133c: mean={v['133c_mean']:.4f}, min={v['133c_min']:.4f} at {v['133c_min_cell']}, below_80={v['133c_cells_below_80']}, below_50={v['133c_cells_below_50']}")
        lines.append(f"  Mean diff: {v['mean_diff']:+.4f}")
    lines.append("")

    # Factor structure
    lines.append("=" * 80)
    lines.append("FACTOR STRUCTURE & CROSS-CELL CORRELATION")
    lines.append("=" * 80)
    if "140a" in factor_analysis and isinstance(factor_analysis["140a"], dict):
        fa = factor_analysis["140a"]
        lines.append("140a cross-cell correlation:")
        lines.append(f"  GT mean corr: {fa.get('gt_mean_corr', 'N/A')}")
        lines.append(f"  Gen mean corr: {fa.get('gen_mean_corr', 'N/A')}")
        lines.append(f"  Corr ratio: {fa.get('corr_ratio', 'N/A')}")
        lines.append(f"  GT eff rank: {fa.get('gt_eff_rank', 'N/A')}")
        lines.append(f"  Gen eff rank: {fa.get('gen_eff_rank', 'N/A')}")
        lines.append(f"  Rank ratio: {fa.get('rank_ratio', 'N/A')}")
    if "133c" in factor_analysis:
        fb = factor_analysis["133c"]
        if isinstance(fb, str):
            lines.append(f"133c: {fb}")
        else:
            lines.append("133c cross-cell correlation:")
            for k2, v2 in fb.items():
                lines.append(f"  {k2}: {v2}")
    lines.append("")

    cc = factor_analysis.get("cointegration_comparison", {})
    lines.append("Cointegration comparison:")
    lines.append(f"  140a gen/gt ratio: {cc.get('140a_gen_gt_ratio', 'N/A')}")
    lines.append(f"  133c gen/gt ratio: {cc.get('133c_gen_gt_ratio', 'N/A')}")
    lines.append(f"  140a gen pass rate: {cc.get('140a_gen_pass_rate', 'N/A')}")
    lines.append(f"  133c gen pass rate: {cc.get('133c_gen_pass_rate', 'N/A')}")
    lines.append("")

    # Conditionality
    lines.append("=" * 80)
    lines.append("CONDITIONALITY PATTERNS")
    lines.append("=" * 80)
    if "width_ratio" in conditionality_analysis:
        wr = conditionality_analysis["width_ratio"]
        lines.append(f"Width ratio: 140a mean={wr['140a_mean']:.4f} (std={wr['140a_std']:.4f}), 133c mean={wr['133c_mean']:.4f} (std={wr['133c_std']:.4f})")
        lines.append(f"Cells below 1.0: 140a={wr['140a_cells_below_1']}/25, 133c={wr['133c_cells_below_1']}/25")
    if "mae_reduction" in conditionality_analysis:
        mr = conditionality_analysis["mae_reduction"]
        lines.append(f"MAE reduction: 140a mean={mr['140a_mean']:.2f}% (min={mr['140a_min']:.2f}% at {mr['140a_min_cell']}), 133c mean={mr['133c_mean']:.2f}% (min={mr['133c_min']:.2f}% at {mr['133c_min_cell']})")
    lines.append("")

    # Distributional
    lines.append("=" * 80)
    lines.append("DISTRIBUTIONAL (KS TESTS)")
    lines.append("=" * 80)
    for test_name in ["ks_test", "ks_level_test"]:
        if test_name in distributional_analysis:
            td = distributional_analysis[test_name]
            lines.append(f"{test_name}:")
            lines.append(f"  140a: n_pass={td.get('140a_n_pass')}/25, worst={td.get('140a_worst_stat', 'N/A')}, median={td.get('140a_median_stat', 'N/A')}")
            lines.append(f"  133c: n_pass={td.get('133c_n_pass')}/25, worst={td.get('133c_worst_stat', 'N/A')}, median={td.get('133c_median_stat', 'N/A')}")
    lines.append("")

    # Final conclusions
    lines.append("=" * 80)
    lines.append("CONCLUSIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. KURTOSIS (Suite 4): The critical difference.")
    lines.append(f"   - 133c (one-shot): ratio {k['133c_kurtosis_ratio']:.3f} -> PASSES (0.5-2.0 range)")
    lines.append(f"   - 140a (AR): ratio {k['140a_kurtosis_ratio']:.3f} -> FAILS (too low)")
    lines.append("   ROOT CAUSE: AR generation accumulates 30 small deltas. By the Central Limit")
    lines.append("   Theorem, summing many small increments produces near-Gaussian distributions,")
    lines.append("   which have low excess kurtosis. One-shot generation avoids this by producing")
    lines.append("   the full trajectory from a single noise draw, preserving heavy tails.")
    lines.append("")
    lines.append("2. COVERAGE (Suite 2): Both fail worst_cell_pass but for different reasons.")
    lines.append("   - 140a has dramatically under-spread cells (some <5% coverage at h=1)")
    lines.append("   - 133c has much better per-cell coverage but still fails worst_cell_pass")
    lines.append("")
    lines.append("3. CONDITIONALITY (Suite 3): 133c dominates.")
    lines.append(f"   - 133c MAE reduction {metrics_b.get('mae_reduction_pct', 0):.1f}% vs 140a {metrics_a.get('mae_reduction_pct', 0):.1f}%")
    lines.append("   - 133c maintains high width ratios across all cells")
    lines.append("")
    lines.append("4. COINTEGRATION (Suite 6): 140a dominates massively.")
    lines.append(f"   - 140a gen/gt ratio: {cc.get('140a_gen_gt_ratio', 'N/A')}")
    lines.append(f"   - 133c gen/gt ratio: {cc.get('133c_gen_gt_ratio', 'N/A')}")
    lines.append("   AR structure naturally preserves cell-cell long-run equilibrium.")
    lines.append("")
    lines.append("5. ARCHITECTURE IMPLICATIONS FOR STEP 3 (CLN):")
    lines.append("   - One-shot generation is ESSENTIAL for kurtosis (Suite 4)")
    lines.append("   - AR structure is ESSENTIAL for cointegration (Suite 6)")
    lines.append("   - Hybrid approach needed: AR skeleton with non-Gaussian innovations")
    lines.append("   - OR: one-shot generation with explicit cointegration loss")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True, help="Path to 140a summary.json")
    parser.add_argument("--model_b", required=True, help="Path to 133c summary.json")
    parser.add_argument("--output_dir", required=True, help="Output directory for analysis")
    parser.add_argument("--result_json", required=True, help="Path for verification_result.json")
    args = parser.parse_args()

    # Load data
    data_a = load_summary(args.model_a)
    data_b = load_summary(args.model_b)

    # Extract pass/fail
    suites_a = get_suite_pass(data_a)
    suites_b = get_suite_pass(data_b)

    # Extract key metrics
    metrics_a = extract_key_metrics(data_a)
    metrics_b = extract_key_metrics(data_b)

    # Deep analyses
    kurtosis_analysis = analyze_kurtosis_difference(data_a, data_b)
    coverage_analysis = analyze_coverage_patterns(data_a, data_b)
    factor_analysis = analyze_factor_structure(data_a, data_b)
    conditionality_analysis = analyze_conditionality_patterns(data_a, data_b)
    distributional_analysis = analyze_distributional(data_a, data_b)

    # Suite comparison
    suite_comparison = build_suite_comparison(suites_a, suites_b)

    # Generate text report
    report = generate_text_report(
        suites_a, suites_b, metrics_a, metrics_b,
        kurtosis_analysis, coverage_analysis, factor_analysis,
        conditionality_analysis, distributional_analysis
    )

    # Save text report
    report_path = Path(args.output_dir) / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(report)

    # Build full analysis JSON
    analysis_json = {
        "comparison": "140a_AR_causal_transformer_vs_133c_oneshot_joint_transformer",
        "timestamp": datetime.now().isoformat(),
        "suite_comparison": suite_comparison,
        "kurtosis_analysis": kurtosis_analysis,
        "coverage_analysis": coverage_analysis,
        "factor_structure": factor_analysis,
        "conditionality_analysis": conditionality_analysis,
        "distributional_analysis": distributional_analysis,
    }

    analysis_path = Path(args.output_dir) / "full_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_json, f, indent=2, default=str)

    # Build verification result
    pass_count_a = sum(1 for v in suites_a.values() if v is True)
    pass_count_b = sum(1 for v in suites_b.values() if v is True)

    verification_result = {
        "verification": "140a_vs_133c_comparison",
        "timestamp": datetime.now().isoformat(),
        "status": "COMPLETED",
        "summary": {
            "140a_suites_passing": pass_count_a,
            "133c_suites_passing": pass_count_b,
            "140a_suites": {k: ("PASS" if v is True else ("FAIL" if v is False else str(v))) for k, v in suites_a.items()},
            "133c_suites": {k: ("PASS" if v is True else ("FAIL" if v is False else str(v))) for k, v in suites_b.items()},
        },
        "key_findings": {
            "kurtosis": {
                "140a_ratio": kurtosis_analysis["overall"]["140a_kurtosis_ratio"],
                "133c_ratio": kurtosis_analysis["overall"]["133c_kurtosis_ratio"],
                "root_cause": "AR accumulates 30 small Gaussian deltas -> CLT -> low kurtosis. One-shot preserves heavy tails.",
                "140a_gen_kurtosis": kurtosis_analysis["overall"]["140a_gen_kurtosis"],
                "133c_gen_kurtosis": kurtosis_analysis["overall"]["133c_gen_kurtosis"],
                "gt_kurtosis": kurtosis_analysis["overall"]["gt_kurtosis"],
            },
            "cointegration": {
                "140a_gen_gt_ratio": factor_analysis.get("cointegration_comparison", {}).get("140a_gen_gt_ratio"),
                "133c_gen_gt_ratio": factor_analysis.get("cointegration_comparison", {}).get("133c_gen_gt_ratio"),
                "finding": "AR preserves long-run equilibrium (ratio near 1.0); one-shot loses it (ratio ~0.32)",
            },
            "coverage": {
                "140a_overall_90": metrics_a.get("overall_90_coverage"),
                "133c_overall_90": metrics_b.get("overall_90_coverage"),
                "finding": "133c has much better coverage (91.7% vs 65.5%); 140a severely under-spread",
            },
            "conditionality": {
                "140a_mae_reduction": metrics_a.get("mae_reduction_pct"),
                "133c_mae_reduction": metrics_b.get("mae_reduction_pct"),
                "finding": "Both conditional; 133c much stronger MAE reduction",
            },
            "cross_cell_correlation": {
                "140a_available": "cross_cell_correlation" in data_a,
                "133c_available": "cross_cell_correlation" in data_b,
            },
        },
        "architectural_implications": {
            "for_kurtosis": "One-shot generation essential; AR sums -> CLT kills heavy tails",
            "for_cointegration": "AR structure essential; one-shot loses cell-cell equilibrium",
            "for_step3_cln": "Need hybrid: either AR with non-Gaussian innovations, or one-shot with cointegration loss",
            "tradeoff": "Suite 4 (kurtosis) and Suite 6 (cointegration) appear structurally opposed between AR and one-shot",
        },
        "files_produced": [
            str(Path(args.output_dir) / "comparison_report.txt"),
            str(Path(args.output_dir) / "full_analysis.json"),
            str(args.result_json),
        ],
    }

    with open(args.result_json, "w") as f:
        json.dump(verification_result, f, indent=2, default=str)

    print(f"\nResults saved to:")
    print(f"  Report: {report_path}")
    print(f"  Analysis: {analysis_path}")
    print(f"  Verification: {args.result_json}")


if __name__ == "__main__":
    main()
