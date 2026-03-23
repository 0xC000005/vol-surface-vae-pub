#!/bin/bash
# Cross-model comparison: Extract key metrics from 4 summary.json files
# (144b, 146a, 146b, 146c) and produce a structured JSON + markdown table.
#
# What this verifies: Side-by-side comparison of all metrics that matter
# for suite pass/fail across the 4 experiments in the RC10 series.
#
# Expected outcome: A comparison.json and comparison.md that make it
# trivially easy to see improvements and regressions.
#
# Usage: bash results/validations/2026-03-22/scripts/cross_model_comparison.sh

set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

mkdir -p results/validations/2026-03-22/analysis/cross_model

PYTHONPATH=. python3 - <<'PYEOF'
import json
import os

MODELS = {
    "144b": "results/block_ar/144b_best_30d/summary.json",
    "146a": "results/block_ar/146a_best_30d/summary.json",
    "146b": "results/block_ar/146b_best_30d/summary.json",
    "146c": "results/block_ar/146c_best_30d/summary.json",
}

OUT_DIR = "results/validations/2026-03-22/analysis/cross_model"


def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def extract_metrics(s):
    m = {}
    suite_names = [
        "surface", "coverage", "conditionality", "time_series",
        "block_ar", "cointegration", "regime_coverage",
        "distributional", "cross_cell_correlation"
    ]
    m["suites"] = {}
    pass_count = 0
    for sn in suite_names:
        p = safe_get(s, sn, "overall_pass", default=None)
        m["suites"][sn] = p
        if p:
            pass_count += 1
    m["suites"]["total_pass"] = pass_count
    m["suites"]["total"] = len(suite_names)

    m["coverage"] = {
        "overall_90": safe_get(s, "coverage", "overall", "0.9"),
        "per_horizon": {
            h: safe_get(s, "coverage", "per_horizon", h, "0.9")
            for h in ["1", "7", "14", "30"]
        },
        "calibration_error": safe_get(s, "coverage", "calibration_error"),
        "worst_cell_pass": safe_get(s, "coverage", "worst_cell_pass"),
    }

    m["distributional"] = {
        "ks_test": {
            "n_pass": safe_get(s, "distributional", "ks_test", "n_pass"),
            "median_stat": safe_get(s, "distributional", "ks_test", "median_stat"),
            "worst_stat": safe_get(s, "distributional", "ks_test", "worst_stat"),
            "pass": safe_get(s, "distributional", "ks_test", "pass"),
        },
        "ks_level_test": {
            "n_pass": safe_get(s, "distributional", "ks_level_test", "n_pass"),
            "median_stat": safe_get(s, "distributional", "ks_level_test", "median_stat"),
            "worst_stat": safe_get(s, "distributional", "ks_level_test", "worst_stat"),
            "pass": safe_get(s, "distributional", "ks_level_test", "pass"),
        },
        "window_floor": {
            "pct_bad": safe_get(s, "distributional", "window_floor", "pct_bad"),
            "pass": safe_get(s, "distributional", "window_floor", "pass"),
        },
        "median_bias": {
            "n_pass_frac": safe_get(s, "distributional", "median_bias", "n_pass"),
            "frac_pass": safe_get(s, "distributional", "median_bias", "frac_pass"),
            "n_mag_pass": safe_get(s, "distributional", "median_bias", "n_mag_pass"),
            "mag_pass": safe_get(s, "distributional", "median_bias", "mag_pass"),
            "pass": safe_get(s, "distributional", "median_bias", "pass"),
        },
    }

    m["time_series"] = {
        "kurtosis_ratio": safe_get(s, "time_series", "kurtosis", "kurtosis_ratio"),
        "gt_kurtosis": safe_get(s, "time_series", "kurtosis", "gt_kurtosis"),
        "gen_kurtosis": safe_get(s, "time_series", "kurtosis", "gen_kurtosis"),
        "kurtosis_pass": safe_get(s, "time_series", "kurtosis", "pass"),
        "acf_correlation": safe_get(s, "time_series", "acf", "acf_correlation"),
        "acf_mae": safe_get(s, "time_series", "acf", "acf_mae"),
        "acf_pass": safe_get(s, "time_series", "acf", "pass"),
    }

    m["cross_cell_correlation"] = {
        "gen_eff_rank": safe_get(s, "cross_cell_correlation", "gen_eff_rank"),
        "gt_eff_rank": safe_get(s, "cross_cell_correlation", "gt_eff_rank"),
        "rank_ratio": safe_get(s, "cross_cell_correlation", "rank_ratio"),
        "rank_pass": safe_get(s, "cross_cell_correlation", "rank_pass"),
        "corr_ratio": safe_get(s, "cross_cell_correlation", "corr_ratio"),
        "corr_pass": safe_get(s, "cross_cell_correlation", "corr_pass"),
    }

    m["conditionality"] = {
        "turb_calm_ratio": safe_get(s, "conditionality", "turb_calm_ratio"),
        "turb_calm_pass": safe_get(s, "conditionality", "turb_calm_pass"),
        "width_ratio": safe_get(s, "conditionality", "width_ratio"),
        "worst_cell_width_ratio": safe_get(s, "conditionality", "worst_cell_width_ratio"),
        "mae_reduction_pct": safe_get(s, "conditionality", "mae_reduction_pct"),
    }

    m["regime_coverage"] = {
        "layer1_pass": safe_get(s, "regime_coverage", "layer1_pass"),
        "layer2_pass": safe_get(s, "regime_coverage", "layer2_pass"),
        "layer2_n_passing": safe_get(s, "regime_coverage", "layer2_n_passing"),
        "layer2_n_total": safe_get(s, "regime_coverage", "layer2_n_total"),
        "layer3_pass": safe_get(s, "regime_coverage", "layer3_pass"),
        "layer3_catastrophic_rate": safe_get(s, "regime_coverage", "layer3_catastrophic_rate"),
    }

    m["cointegration"] = {
        "gen_pass_rate": safe_get(s, "cointegration", "gen_pass_rate"),
        "gt_pass_rate": safe_get(s, "cointegration", "gt_pass_rate"),
        "gen_gt_ratio": safe_get(s, "cointegration", "gen_gt_ratio"),
        "worst_cell_ratio": safe_get(s, "cointegration", "worst_cell_ratio"),
        "worst_cell_pass": safe_get(s, "cointegration", "worst_cell_pass"),
    }

    return m


comparison = {}
for name, path in MODELS.items():
    with open(path) as f:
        s = json.load(f)
    comparison[name] = extract_metrics(s)

json_path = os.path.join(OUT_DIR, "comparison.json")
with open(json_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"Saved: {json_path}")


def fmt(v, decimals=3):
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"


models = list(MODELS.keys())
lines = []
lines.append("# Cross-Model Comparison: 144b, 146a, 146b, 146c")
lines.append("")
lines.append("Generated from summary.json files on 2026-03-22.")
lines.append("")

suite_names = [
    "surface", "coverage", "conditionality", "time_series",
    "block_ar", "cointegration", "regime_coverage",
    "distributional", "cross_cell_correlation"
]

lines.append("## Suite Pass/Fail Summary")
lines.append("")
header = "| Suite | " + " | ".join(models) + " |"
sep = "|---|" + "|".join(["---"] * len(models)) + "|"
lines.append(header)
lines.append(sep)
for sn in suite_names:
    row = f"| {sn} | " + " | ".join(fmt(comparison[m]["suites"][sn]) for m in models) + " |"
    lines.append(row)
row = "| **TOTAL** | " + " | ".join(f"**{comparison[m]['suites']['total_pass']}/9**" for m in models) + " |"
lines.append(row)
lines.append("")

lines.append("## Coverage (CI 90%)")
lines.append("")
lines.append(header)
lines.append(sep)
row = "| Overall 90% CI | " + " | ".join(fmt_pct(comparison[m]["coverage"]["overall_90"]) for m in models) + " |"
lines.append(row)
for h in ["1", "7", "14", "30"]:
    row = f"| Horizon {h} CI | " + " | ".join(fmt_pct(comparison[m]["coverage"]["per_horizon"][h]) for m in models) + " |"
    lines.append(row)
row = "| Calibration Error | " + " | ".join(fmt(comparison[m]["coverage"]["calibration_error"]) for m in models) + " |"
lines.append(row)
row = "| Worst Cell Pass | " + " | ".join(fmt(comparison[m]["coverage"]["worst_cell_pass"]) for m in models) + " |"
lines.append(row)
lines.append("")

lines.append("## Distributional")
lines.append("")
lines.append(header)
lines.append(sep)
row = "| KS Changes n_pass/25 | " + " | ".join(str(comparison[m]["distributional"]["ks_test"]["n_pass"]) for m in models) + " |"
lines.append(row)
row = "| KS Changes median_stat | " + " | ".join(fmt(comparison[m]["distributional"]["ks_test"]["median_stat"]) for m in models) + " |"
lines.append(row)
row = "| KS Changes pass | " + " | ".join(fmt(comparison[m]["distributional"]["ks_test"]["pass"]) for m in models) + " |"
lines.append(row)
row = "| KS Levels n_pass/25 | " + " | ".join(str(comparison[m]["distributional"]["ks_level_test"]["n_pass"]) for m in models) + " |"
lines.append(row)
row = "| KS Levels median_stat | " + " | ".join(fmt(comparison[m]["distributional"]["ks_level_test"]["median_stat"]) for m in models) + " |"
lines.append(row)
row = "| KS Levels pass | " + " | ".join(fmt(comparison[m]["distributional"]["ks_level_test"]["pass"]) for m in models) + " |"
lines.append(row)
row = "| Window Floor pct_bad | " + " | ".join(fmt_pct(comparison[m]["distributional"]["window_floor"]["pct_bad"]) for m in models) + " |"
lines.append(row)
row = "| Window Floor pass | " + " | ".join(fmt(comparison[m]["distributional"]["window_floor"]["pass"]) for m in models) + " |"
lines.append(row)
row = "| Median Bias frac n_pass | " + " | ".join(str(comparison[m]["distributional"]["median_bias"]["n_pass_frac"]) for m in models) + " |"
lines.append(row)
row = "| Median Bias frac pass | " + " | ".join(fmt(comparison[m]["distributional"]["median_bias"]["frac_pass"]) for m in models) + " |"
lines.append(row)
row = "| Median Bias mag n_pass | " + " | ".join(str(comparison[m]["distributional"]["median_bias"]["n_mag_pass"]) for m in models) + " |"
lines.append(row)
row = "| Median Bias mag pass | " + " | ".join(fmt(comparison[m]["distributional"]["median_bias"]["mag_pass"]) for m in models) + " |"
lines.append(row)
lines.append("")

lines.append("## Time Series")
lines.append("")
lines.append(header)
lines.append(sep)
row = "| Kurtosis Ratio | " + " | ".join(fmt(comparison[m]["time_series"]["kurtosis_ratio"]) for m in models) + " |"
lines.append(row)
row = "| Kurtosis GT | " + " | ".join(fmt(comparison[m]["time_series"]["gt_kurtosis"], 1) for m in models) + " |"
lines.append(row)
row = "| Kurtosis Gen | " + " | ".join(fmt(comparison[m]["time_series"]["gen_kurtosis"], 1) for m in models) + " |"
lines.append(row)
row = "| Kurtosis Pass | " + " | ".join(fmt(comparison[m]["time_series"]["kurtosis_pass"]) for m in models) + " |"
lines.append(row)
row = "| ACF Correlation | " + " | ".join(fmt(comparison[m]["time_series"]["acf_correlation"]) for m in models) + " |"
lines.append(row)
row = "| ACF MAE | " + " | ".join(fmt(comparison[m]["time_series"]["acf_mae"]) for m in models) + " |"
lines.append(row)
row = "| ACF Pass | " + " | ".join(fmt(comparison[m]["time_series"]["acf_pass"]) for m in models) + " |"
lines.append(row)
lines.append("")

lines.append("## Cross-Cell Correlation")
lines.append("")
lines.append(header)
lines.append(sep)
row = "| GT Eff Rank | " + " | ".join(fmt(comparison[m]["cross_cell_correlation"]["gt_eff_rank"]) for m in models) + " |"
lines.append(row)
row = "| Gen Eff Rank | " + " | ".join(fmt(comparison[m]["cross_cell_correlation"]["gen_eff_rank"]) for m in models) + " |"
lines.append(row)
row = "| Rank Ratio | " + " | ".join(fmt(comparison[m]["cross_cell_correlation"]["rank_ratio"]) for m in models) + " |"
lines.append(row)
row = "| Rank Pass | " + " | ".join(fmt(comparison[m]["cross_cell_correlation"]["rank_pass"]) for m in models) + " |"
lines.append(row)
row = "| Corr Ratio | " + " | ".join(fmt(comparison[m]["cross_cell_correlation"]["corr_ratio"]) for m in models) + " |"
lines.append(row)
row = "| Corr Pass | " + " | ".join(fmt(comparison[m]["cross_cell_correlation"]["corr_pass"]) for m in models) + " |"
lines.append(row)
lines.append("")

lines.append("## Conditionality")
lines.append("")
lines.append(header)
lines.append(sep)
row = "| Turb/Calm Ratio | " + " | ".join(fmt(comparison[m]["conditionality"]["turb_calm_ratio"]) for m in models) + " |"
lines.append(row)
row = "| Turb/Calm Pass | " + " | ".join(fmt(comparison[m]["conditionality"]["turb_calm_pass"]) for m in models) + " |"
lines.append(row)
row = "| Width Ratio | " + " | ".join(fmt(comparison[m]["conditionality"]["width_ratio"]) for m in models) + " |"
lines.append(row)
row = "| Worst Cell Width Ratio | " + " | ".join(fmt(comparison[m]["conditionality"]["worst_cell_width_ratio"]) for m in models) + " |"
lines.append(row)
row = "| MAE Reduction % | " + " | ".join(fmt(comparison[m]["conditionality"]["mae_reduction_pct"], 1) for m in models) + " |"
lines.append(row)
lines.append("")

lines.append("## Regime Coverage")
lines.append("")
lines.append(header)
lines.append(sep)
row = "| Layer 1 Pass | " + " | ".join(fmt(comparison[m]["regime_coverage"]["layer1_pass"]) for m in models) + " |"
lines.append(row)
row = "| Layer 2 Pass | " + " | ".join(fmt(comparison[m]["regime_coverage"]["layer2_pass"]) for m in models) + " |"
lines.append(row)
l2_frac = []
for m in models:
    n = comparison[m]["regime_coverage"]["layer2_n_passing"]
    t = comparison[m]["regime_coverage"]["layer2_n_total"]
    l2_frac.append(f"{n}/{t}" if n is not None and t is not None else "N/A")
row = "| Layer 2 Passing | " + " | ".join(l2_frac) + " |"
lines.append(row)
row = "| Layer 3 Pass | " + " | ".join(fmt(comparison[m]["regime_coverage"]["layer3_pass"]) for m in models) + " |"
lines.append(row)
row = "| Layer 3 Catastrophic Rate | " + " | ".join(fmt(comparison[m]["regime_coverage"]["layer3_catastrophic_rate"]) for m in models) + " |"
lines.append(row)
lines.append("")

lines.append("## Cointegration")
lines.append("")
lines.append(header)
lines.append(sep)
row = "| Gen Pass Rate | " + " | ".join(fmt(comparison[m]["cointegration"]["gen_pass_rate"]) for m in models) + " |"
lines.append(row)
row = "| GT Pass Rate | " + " | ".join(fmt(comparison[m]["cointegration"]["gt_pass_rate"]) for m in models) + " |"
lines.append(row)
row = "| Gen/GT Ratio | " + " | ".join(fmt(comparison[m]["cointegration"]["gen_gt_ratio"]) for m in models) + " |"
lines.append(row)
row = "| Worst Cell Ratio | " + " | ".join(fmt(comparison[m]["cointegration"]["worst_cell_ratio"]) for m in models) + " |"
lines.append(row)
row = "| Worst Cell Pass | " + " | ".join(fmt(comparison[m]["cointegration"]["worst_cell_pass"]) for m in models) + " |"
lines.append(row)
lines.append("")

md_path = os.path.join(OUT_DIR, "comparison.md")
with open(md_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved: {md_path}")

print("\n=== SUITE PASS/FAIL ===")
for m in models:
    suites = comparison[m]["suites"]
    passed = [sn for sn in suite_names if suites[sn]]
    print(f"  {m}: {suites['total_pass']}/9 -- passed: {passed}")

print("\n=== KEY METRICS ===")
print(f"{'Metric':<30} " + "  ".join(f"{m:>8}" for m in models))
print("-" * 70)
print(f"{'CI 90%':<30} " + "  ".join(f"{comparison[m]['coverage']['overall_90']*100:>7.1f}%" for m in models))
print(f"{'Calib Error':<30} " + "  ".join(f"{comparison[m]['coverage']['calibration_error']:>8.3f}" for m in models))
print(f"{'KS Changes n_pass':<30} " + "  ".join(f"{comparison[m]['distributional']['ks_test']['n_pass']:>8}" for m in models))
print(f"{'KS Levels n_pass':<30} " + "  ".join(f"{comparison[m]['distributional']['ks_level_test']['n_pass']:>8}" for m in models))
print(f"{'Window Floor pct_bad':<30} " + "  ".join(f"{comparison[m]['distributional']['window_floor']['pct_bad']*100:>7.1f}%" for m in models))
print(f"{'Kurtosis Ratio':<30} " + "  ".join(f"{comparison[m]['time_series']['kurtosis_ratio']:>8.3f}" for m in models))
print(f"{'ACF Correlation':<30} " + "  ".join(f"{comparison[m]['time_series']['acf_correlation']:>8.3f}" for m in models))
print(f"{'Gen Eff Rank':<30} " + "  ".join(f"{comparison[m]['cross_cell_correlation']['gen_eff_rank']:>8.3f}" for m in models))
print(f"{'Rank Ratio':<30} " + "  ".join(f"{comparison[m]['cross_cell_correlation']['rank_ratio']:>8.3f}" for m in models))
print(f"{'Corr Ratio':<30} " + "  ".join(f"{comparison[m]['cross_cell_correlation']['corr_ratio']:>8.3f}" for m in models))
print(f"{'Turb/Calm Ratio':<30} " + "  ".join(f"{comparison[m]['conditionality']['turb_calm_ratio']:>8.3f}" for m in models))

PYEOF
