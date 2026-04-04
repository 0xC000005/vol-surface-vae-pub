#!/usr/bin/env python3
"""Cross-model comparison across ALL RC22 experiments.

Extracts key metrics from each model's summary.json and produces:
1. A master comparison JSON
2. A verification JSON with best-per-metric analysis
3. Prints a markdown table to stdout
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Script is at results/validations/2026-04-04/scripts/ -> go up 4 levels to repo root
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))

MODELS = {
    "164a_baseline": "results/block_ar/164a_v3_percell_bptt_softplus_best_30d/summary.json",
    "166a_corrected": "results/block_ar/166a_best_30d/summary.json",
    "167a_fact_Ldied": "results/block_ar/167a_best_30d/summary.json",
    "167b_fact_CLN": "results/block_ar/167b_best_30d/summary.json",
    "167d_fact_e2e": "results/block_ar/167d_best_30d/summary.json",
    "168a_K4_B64": "results/block_ar/168a_best_30d/summary.json",
    "168b_K8_B16": "results/block_ar/168b_best_30d/summary.json",
    "168c_K4_B16": "results/block_ar/168c_best_30d/summary.json",
}

MODEL_DESCRIPTIONS = {
    "164a_baseline": "Baseline K=16 (old loss)",
    "166a_corrected": "K=16, corrected loss (IS=0.005, VS=1.0)",
    "167a_fact_Ldied": "Factorized, L died",
    "167b_fact_CLN": "Factorized, CLN frozen",
    "167d_fact_e2e": "Factorized, e2e",
    "168a_K4_B64": "K=4, B=64 (confounded)",
    "168b_K8_B16": "K=8, B=16 (clean)",
    "168c_K4_B16": "K=4, B=16 (clean)",
}


def load_summary(path):
    full_path = os.path.join(BASE, path)
    if not os.path.exists(full_path):
        return None
    with open(full_path) as f:
        return json.load(f)


def count_passes(data):
    """Count suite passes (S1-S9)."""
    suite_map = {
        "surface": "S1",
        "coverage": "S2",
        "conditionality": "S3",
        "time_series": "S4",
        "block_ar": "S5",
        "cointegration": "S6",
        "regime_coverage": "S7",
        "distributional": "S8",
        "cross_cell_correlation": "S9",
    }
    passes = []
    for key, label in suite_map.items():
        if key in data and "overall_pass" in data[key]:
            if data[key]["overall_pass"]:
                passes.append(label)
    return passes


def extract_metrics(data):
    """Extract all requested metrics from a summary.json."""
    if data is None:
        return None

    passes = count_passes(data)
    m = {
        "suite_pass_count": len(passes),
        "suites_passing": passes,
    }

    # S1: explosion rate
    s1 = data.get("surface", {})
    expl = s1.get("explosion", {})
    m["s1_explosion_rate"] = expl.get("explosion_total_rate")
    m["s1_pass"] = s1.get("overall_pass")

    # S2: CI coverage
    s2 = data.get("coverage", {})
    overall = s2.get("overall", {})
    m["s2_90ci_coverage"] = float(overall.get("0.9", 0))
    m["s2_calibration_error"] = s2.get("calibration_error")
    m["s2_worst_cell_pass"] = s2.get("worst_cell_pass")
    m["s2_pass"] = s2.get("overall_pass")

    # S3: conditionality
    s3 = data.get("conditionality", {})
    m["s3_turb_calm_ratio"] = s3.get("turb_calm_ratio")
    m["s3_worst_cell_mae_reduction"] = s3.get("worst_cell_mae_reduction")
    m["s3_pass"] = s3.get("overall_pass")

    # S4: time series
    s4 = data.get("time_series", {})
    kurt = s4.get("kurtosis", {})
    m["s4_kurtosis_ratio"] = kurt.get("kurtosis_ratio")
    m["s4_pass"] = s4.get("overall_pass")

    # S5: block AR
    s5 = data.get("block_ar", {})
    m["s5_pass"] = s5.get("overall_pass")

    # S6: cointegration
    s6 = data.get("cointegration", {})
    m["s6_gen_gt_ratio"] = s6.get("gen_gt_ratio")
    m["s6_gen_pass_rate"] = s6.get("gen_pass_rate")
    m["s6_gt_pass_rate"] = s6.get("gt_pass_rate")
    m["s6_pass"] = s6.get("overall_pass")

    # S7: regime coverage
    s7 = data.get("regime_coverage", {})
    m["s7_pass"] = s7.get("overall_pass")

    # S8: distributional
    s8 = data.get("distributional", {})
    ks_daily = s8.get("ks_test", {})
    ks_level = s8.get("ks_level_test", {})
    med_bias = s8.get("median_bias", {})
    m["s8_ks_daily_npass"] = ks_daily.get("n_pass")
    m["s8_ks_daily_pass"] = ks_daily.get("pass")
    m["s8_ks_level_npass"] = ks_level.get("n_pass")
    m["s8_ks_level_pass"] = ks_level.get("pass")
    m["s8_median_bias_npass"] = med_bias.get("n_pass")
    m["s8_median_bias_frac_pass"] = med_bias.get("frac_pass")
    m["s8_median_bias_mag_pass"] = med_bias.get("mag_pass")
    m["s8_pass"] = s8.get("overall_pass")

    # S9: cross-cell correlation
    s9 = data.get("cross_cell_correlation", {})
    m["s9_corr_ratio"] = s9.get("corr_ratio")
    m["s9_rank_ratio"] = s9.get("rank_ratio")
    m["s9_gen_eff_rank"] = s9.get("gen_eff_rank")
    m["s9_gt_eff_rank"] = s9.get("gt_eff_rank")
    m["s9_gen_mean_corr"] = s9.get("gen_mean_corr")
    m["s9_gt_mean_corr"] = s9.get("gt_mean_corr")
    m["s9_pass"] = s9.get("overall_pass")

    return m


def find_best(all_metrics, metric_key, direction="high"):
    """Find best model for a given metric. direction='high' or 'low'."""
    best_name = None
    best_val = None
    for name, m in all_metrics.items():
        if m is None:
            continue
        val = m.get(metric_key)
        if val is None:
            continue
        if best_val is None:
            best_val = val
            best_name = name
        elif direction == "high" and val > best_val:
            best_val = val
            best_name = name
        elif direction == "low" and val < best_val:
            best_val = val
            best_name = name
    return best_name, best_val


def fmt(val, decimals=3):
    """Format a value for the table."""
    if val is None:
        return "N/A"
    if isinstance(val, bool):
        return "PASS" if val else "FAIL"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, list):
        return ",".join(val)
    return str(val)


def main():
    # Load all models
    all_metrics = {}
    for name, path in MODELS.items():
        data = load_summary(path)
        if data is None:
            print(f"WARNING: {path} not found, skipping {name}", file=sys.stderr)
            all_metrics[name] = None
        else:
            all_metrics[name] = extract_metrics(data)

    # --- Build comparison JSON ---
    comparison = {
        "models": {},
        "best_per_metric": {},
    }
    for name, m in all_metrics.items():
        comparison["models"][name] = {
            "description": MODEL_DESCRIPTIONS[name],
            "metrics": m,
        }

    # Best per metric
    metric_directions = {
        "suite_pass_count": "high",
        "s1_explosion_rate": "low",
        "s2_90ci_coverage": "high",
        "s2_calibration_error": "low",
        "s3_turb_calm_ratio": "high",
        "s3_worst_cell_mae_reduction": "high",
        "s4_kurtosis_ratio": "high",  # closer to 1.0 is best, but higher is better than <1
        "s6_gen_gt_ratio": "high",
        "s8_ks_daily_npass": "high",
        "s8_ks_level_npass": "high",
        "s8_median_bias_npass": "high",
        "s9_corr_ratio": "low",  # closer to 1.0 is best
        "s9_rank_ratio": "high",  # closer to 1.0 is best
        "s9_gen_eff_rank": "high",  # closer to GT is best
    }

    # For kurtosis_ratio, best = closest to 1.0
    # For corr_ratio, best = closest to 1.0
    # For rank_ratio, best = closest to 1.0
    # Handle these specially
    special_closest_to_1 = ["s4_kurtosis_ratio", "s9_corr_ratio", "s9_rank_ratio"]

    for metric_key, direction in metric_directions.items():
        if metric_key in special_closest_to_1:
            # Find closest to 1.0
            best_name = None
            best_dist = None
            for name, m in all_metrics.items():
                if m is None:
                    continue
                val = m.get(metric_key)
                if val is None:
                    continue
                dist = abs(val - 1.0)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_name = name
            best_val = all_metrics[best_name].get(metric_key) if best_name else None
            comparison["best_per_metric"][metric_key] = {
                "best_model": best_name,
                "value": best_val,
                "criterion": "closest_to_1.0",
            }
        else:
            best_name, best_val = find_best(all_metrics, metric_key, direction)
            comparison["best_per_metric"][metric_key] = {
                "best_model": best_name,
                "value": best_val,
                "direction": direction,
            }

    # --- Dominance analysis ---
    # Check if any model is best on ALL metrics
    best_counts = {}
    for metric_key, info in comparison["best_per_metric"].items():
        bm = info["best_model"]
        if bm:
            best_counts[bm] = best_counts.get(bm, 0) + 1

    comparison["dominance_analysis"] = {
        "best_count_per_model": best_counts,
        "total_metrics": len(metric_directions),
        "dominant_model": None,
    }

    # A model dominates if it's best on > 50% of metrics
    for name, count in best_counts.items():
        if count > len(metric_directions) / 2:
            comparison["dominance_analysis"]["dominant_model"] = name

    # Save comparison JSON
    comp_path = os.path.join(
        BASE,
        "results/validations/2026-04-04/analysis/168bc_followup/cross_model_comparison.json",
    )
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison to {comp_path}", file=sys.stderr)

    # Save verification JSON
    verif = {
        "script": "results/validations/2026-04-04/scripts/168bc_cross_model.py",
        "n_models": sum(1 for m in all_metrics.values() if m is not None),
        "n_models_missing": sum(1 for m in all_metrics.values() if m is None),
        "dominance_analysis": comparison["dominance_analysis"],
        "best_per_metric": comparison["best_per_metric"],
        "suite_pass_summary": {
            name: {
                "count": m["suite_pass_count"],
                "suites": m["suites_passing"],
            }
            for name, m in all_metrics.items()
            if m is not None
        },
    }
    verif_path = os.path.join(
        BASE,
        "results/validations/2026-04-04/verification_results/168bc_cross_model.json",
    )
    with open(verif_path, "w") as f:
        json.dump(verif, f, indent=2)
    print(f"Saved verification to {verif_path}", file=sys.stderr)

    # --- Print markdown table ---
    names = [n for n in MODELS.keys() if all_metrics.get(n) is not None]
    short_names = {
        "164a_baseline": "164a",
        "166a_corrected": "166a",
        "167a_fact_Ldied": "167a",
        "167b_fact_CLN": "167b",
        "167d_fact_e2e": "167d",
        "168a_K4_B64": "168a",
        "168b_K8_B16": "168b",
        "168c_K4_B16": "168c",
    }

    # Build rows
    rows = []
    rows.append(("**Model**", *[short_names[n] for n in names]))
    rows.append(("**Description**", *[MODEL_DESCRIPTIONS[n][:20] for n in names]))
    rows.append(("**Suites PASS**", *[fmt(all_metrics[n]["suite_pass_count"]) for n in names]))
    rows.append(("**Which suites**", *[fmt(all_metrics[n]["suites_passing"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S1 explosion", *[fmt(all_metrics[n]["s1_explosion_rate"], 4) for n in names]))
    rows.append(("S1 pass", *[fmt(all_metrics[n]["s1_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S2 90% CI", *[fmt(all_metrics[n]["s2_90ci_coverage"], 3) for n in names]))
    rows.append(("S2 cal error", *[fmt(all_metrics[n]["s2_calibration_error"], 4) for n in names]))
    rows.append(("S2 worst cell", *[fmt(all_metrics[n]["s2_worst_cell_pass"]) for n in names]))
    rows.append(("S2 pass", *[fmt(all_metrics[n]["s2_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S3 turb/calm", *[fmt(all_metrics[n]["s3_turb_calm_ratio"], 3) for n in names]))
    rows.append(("S3 worst MAE red", *[fmt(all_metrics[n]["s3_worst_cell_mae_reduction"], 2) for n in names]))
    rows.append(("S3 pass", *[fmt(all_metrics[n]["s3_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S4 kurtosis", *[fmt(all_metrics[n]["s4_kurtosis_ratio"], 3) for n in names]))
    rows.append(("S4 pass", *[fmt(all_metrics[n]["s4_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S5 pass", *[fmt(all_metrics[n]["s5_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S6 gen/gt ratio", *[fmt(all_metrics[n]["s6_gen_gt_ratio"], 3) for n in names]))
    rows.append(("S6 pass", *[fmt(all_metrics[n]["s6_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S7 pass", *[fmt(all_metrics[n]["s7_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S8 KS daily", *[fmt(all_metrics[n]["s8_ks_daily_npass"]) + "/25" for n in names]))
    rows.append(("S8 KS level", *[fmt(all_metrics[n]["s8_ks_level_npass"]) + "/25" for n in names]))
    rows.append(("S8 med bias", *[fmt(all_metrics[n]["s8_median_bias_npass"]) + "/25" for n in names]))
    rows.append(("S8 pass", *[fmt(all_metrics[n]["s8_pass"]) for n in names]))
    rows.append(("---", *["---" for _ in names]))
    rows.append(("S9 corr ratio", *[fmt(all_metrics[n]["s9_corr_ratio"], 3) for n in names]))
    rows.append(("S9 rank ratio", *[fmt(all_metrics[n]["s9_rank_ratio"], 3) for n in names]))
    rows.append(("S9 gen eff_rank", *[fmt(all_metrics[n]["s9_gen_eff_rank"], 2) for n in names]))
    rows.append(("S9 GT eff_rank", *[fmt(all_metrics[n]["s9_gt_eff_rank"], 2) for n in names]))
    rows.append(("S9 pass", *[fmt(all_metrics[n]["s9_pass"]) for n in names]))

    # Print table
    print()
    print("## RC22 Cross-Model Comparison (ALL experiments)")
    print()

    # Calculate column widths
    n_cols = len(names) + 1
    widths = [0] * n_cols
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Print header and rows
    for i, row in enumerate(rows):
        if row[0] == "---":
            # Separator
            continue
        cells = [cell.ljust(widths[j]) for j, cell in enumerate(row)]
        print("| " + " | ".join(cells) + " |")
        if i == 0:
            print("| " + " | ".join(["-" * widths[j] for j in range(n_cols)]) + " |")

    # Print best-per-metric summary
    print()
    print("## Best per metric")
    print()
    print("| Metric | Best Model | Value |")
    print("| ------ | ---------- | ----- |")
    for metric_key, info in comparison["best_per_metric"].items():
        bm = info["best_model"]
        val = info["value"]
        crit = info.get("criterion", info.get("direction", ""))
        print(f"| {metric_key} | {short_names.get(bm, bm)} | {fmt(val, 3)} ({crit}) |")

    # Print dominance
    print()
    print("## Dominance analysis")
    print()
    print("Best-on counts:")
    for name, count in sorted(best_counts.items(), key=lambda x: -x[1]):
        print(f"  {short_names.get(name, name)}: {count}/{len(metric_directions)} metrics")
    dom = comparison["dominance_analysis"]["dominant_model"]
    if dom:
        print(f"\nDominant model: {short_names.get(dom, dom)}")
    else:
        print("\nNo single model dominates (>50% of metrics). This is a Pareto frontier.")


if __name__ == "__main__":
    main()
