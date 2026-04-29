#!/usr/bin/env python
"""756a: attribute the 755a train-tail versus validation gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PATHS = {
    "734a_val_incumbent": "results/block_ar/734a_realvix_framework_lock_baseline/iv_val_full11_s64.json",
    "746a_train_tail_incumbent": "results/block_ar/746a_674a_train_tail_audit/iv_train_tail_full11_s64.json",
    "755a_val_shortprefix": "results/block_ar/755a_iv_shortprefix_fm_k5/iv_val_full11_s64.json",
    "755a_train_tail_shortprefix": "results/block_ar/755a_iv_shortprefix_fm_k5/iv_train_tail_full11_s64.json",
}

DEFAULT_HARD_CELLS = {
    "744a_incumbent": "results/block_ar/744a_iv_hard_cell_condition_response/summary.json",
    "756a_755a": "results/block_ar/756a_755_hard_cell_condition_response/summary.json",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fnum(value: Any, digits: int = 6) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def scorecard_row(name: str, path: str) -> dict[str, Any]:
    d = load_json(path)
    return {
        "name": name,
        "path": path,
        "n_pass": int(d["summary"]["n_pass"]),
        "failed_suites": list(d["summary"]["failed_suites"]),
        "cov90": fnum(d["coverage"]["overall"]["0.9"]),
        "calibration_error": fnum(d["coverage"]["calibration_error"]),
        "level_ks_pass_cells": int(d["distributional_fidelity"]["ks_level_test"]["n_pass"]),
        "median_bias_pass_cells": int(d["distributional_fidelity"]["median_bias"]["n_pass"]),
        "daily_ks_pass_cells": int(d["distributional_fidelity"]["ks_test"]["n_pass"]),
        "window_floor_pass": bool(d["distributional_fidelity"]["window_floor"]["pass"]),
        "cointegration_ratio": fnum(d["cointegration"]["gen_gt_ratio"]),
        "cointegration_worst_cell_ratio": fnum(d["cointegration"]["worst_cell_ratio"]),
        "regime_layer2_pass_count": int(d["regime_coverage"]["layer2_n_passing"]),
        "regime_layer3_catastrophic_rate": fnum(d["regime_coverage"]["layer3_catastrophic_rate"]),
        "risk_state_pass": bool(d["risk_state_allocation"]["overall_pass"]),
        "history_width_spearman": fnum(d["risk_state_allocation"].get("history_width_spearman")),
        "future_width_spearman": fnum(d["risk_state_allocation"].get("future_width_spearman")),
        "time_series_pass": bool(d["time_series"]["overall_pass"]),
        "kurtosis_ratio": fnum(d["time_series"]["kurtosis"]["kurtosis_ratio"]),
        "mean_reversion_pass": bool(d["mean_reversion"]["overall_pass"]),
        "mean_reversion_ratio": fnum(d["mean_reversion"]["mr_gt_ratio"]),
        "pathwise_pass": bool(d["pathwise_jump_realism"]["overall_pass"]),
        "pathwise_max_jump_ks": fnum(d["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
    }


def hard_cell_row(name: str, path: str) -> dict[str, Any]:
    d = load_json(path)
    summary = d["summary"]
    return {
        "name": name,
        "path": path,
        "median_low_tertile_coverage90": fnum(summary["median_low_tertile_coverage90"]),
        "median_hard_lower_miss_rate": fnum(summary["median_hard_lower_miss_rate"]),
        "median_abs_slope_gap": fnum(summary["median_abs_slope_gap"]),
    }


def build_report(scorecards: list[dict[str, Any]], hard_cells: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in scorecards}
    hard_by_name = {row["name"]: row for row in hard_cells}
    val_755 = by_name["755a_val_shortprefix"]
    train_755 = by_name["755a_train_tail_shortprefix"]
    hard_inc = hard_by_name["744a_incumbent"]
    hard_755 = hard_by_name["756a_755a"]
    return {
        "scorecards": scorecards,
        "hard_cells": hard_cells,
        "attribution": {
            "primary_read": "shortprefix_repairs_in_sample_path_law_but_not_validation_shifted_hard_cells",
            "evidence": [
                f"755a train-tail reaches {train_755['n_pass']}/11 and passes coverage, level distribution, median bias, cointegration, mean reversion, and pathwise realism.",
                f"755a validation stays {val_755['n_pass']}/11 and still fails coverage, conditionality, cointegration worst-cell, regime coverage, and median-bias distributional fidelity.",
                f"Validation hard-cell low-tertile coverage remains below incumbent: {hard_755['median_low_tertile_coverage90']} versus {hard_inc['median_low_tertile_coverage90']}.",
                f"Validation hard-cell lower-miss remains slightly worse than incumbent: {hard_755['median_hard_lower_miss_rate']} versus {hard_inc['median_hard_lower_miss_rate']}.",
            ],
            "failure_classification": {
                "core_path_realism": "mostly_repaired_in_sample",
                "validation_level_support": "still_binding",
                "validation_conditionality": "broad_risk_state_passes_but_per_cell_width_gate_fails",
                "regime_layer2": "still_binding_due_per_cell_regime_coverage_not_catastrophic_undercoverage",
                "pure_capacity_or_full_prefix_loss": "not_supported_by_755a",
            },
            "decision": (
                "Do not abandon short-prefix exposure. The next move should target validation level-support allocation, "
                "not generic path realism. A clean next experiment should either add a schedule around the short-prefix exposure "
                "or a split-robust, data-derived support/quantile calibration inside the same normalized-innovation law; avoid broad scalar temperature tuning."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 756a Short-Prefix Validation Gap",
        "",
        "## Scorecards",
        "",
        "| run | pass | cov90 | calerr | daily KS | level KS | bias | coint | worst coint | regime L2 | risk | MR | path KS | kurt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in report["scorecards"]:
        lines.append(
            "| {name} | {n_pass}/11 | {cov90:.3f} | {calibration_error:.3f} | "
            "{daily_ks_pass_cells}/25 | {level_ks_pass_cells}/25 | {median_bias_pass_cells}/25 | "
            "{cointegration_ratio:.3f} | {cointegration_worst_cell_ratio:.3f} | "
            "{regime_layer2_pass_count}/8 | {risk_state_pass} | {mean_reversion_pass} | "
            "{pathwise_max_jump_ks:.3f} | {kurtosis_ratio:.3f} |".format(**row)
        )
    lines.extend(["", "## Hard-Cell Audit", ""])
    lines.append("| run | low-tertile cov90 | lower-miss rate | abs slope gap |")
    lines.append("|---|---:|---:|---:|")
    for row in report["hard_cells"]:
        lines.append(
            "| {name} | {median_low_tertile_coverage90:.3f} | "
            "{median_hard_lower_miss_rate:.3f} | {median_abs_slope_gap:.3f} |".format(**row)
        )
    attr = report["attribution"]
    lines.extend(["", "## Attribution", "", f"- primary read: `{attr['primary_read']}`"])
    for item in attr["evidence"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Failure Classification", ""])
    for key, value in attr["failure_classification"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Decision", "", attr["decision"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="results/block_ar/756a_shortprefix_validation_gap")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecards = [scorecard_row(name, path) for name, path in DEFAULT_PATHS.items()]
    hard_cells = [hard_cell_row(name, path) for name, path in DEFAULT_HARD_CELLS.items()]
    report = build_report(scorecards, hard_cells)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "summary.md")
    print(json.dumps(report["attribution"], indent=2))


if __name__ == "__main__":
    main()
