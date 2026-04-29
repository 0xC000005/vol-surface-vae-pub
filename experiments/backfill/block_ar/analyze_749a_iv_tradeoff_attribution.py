#!/usr/bin/env python
"""749a: attribute recent IV strict-suite trade-offs before another model change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCORECARDS = {
    "734a_val_incumbent": "results/block_ar/734a_realvix_framework_lock_baseline/iv_val_full11_s64.json",
    "742a_val_interval_score": "results/block_ar/742a_iv_interval_score/iv_val_full11_s64.json",
    "745a_val_scale_local": "results/block_ar/745a_iv_scale_local/iv_val_full11_s64.json",
    "748a_val_state_tail_sampler": "results/block_ar/748a_iv_state_tail_sampler/iv_val_full11_s64.json",
    "746a_train_tail_incumbent": "results/block_ar/746a_674a_train_tail_audit/iv_train_tail_full11_s64.json",
    "748a_train_tail_state_tail_sampler": "results/block_ar/748a_iv_state_tail_sampler/iv_train_tail_full11_s64.json",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fnum(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
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
        "horizon_pass": bool(d["coverage"]["horizon_pass"]["1"])
        and bool(d["coverage"]["horizon_pass"]["7"])
        and bool(d["coverage"]["horizon_pass"]["14"])
        and bool(d["coverage"]["horizon_pass"]["30"]),
        "worst_cell_coverage_pass": bool(d["coverage"]["worst_cell_pass"]),
        "conditionality_mae_reduction_pct": fnum(d["conditionality"]["mae_reduction_pct"]),
        "worst_cell_width_ratio": fnum(d["conditionality"]["worst_cell_width_ratio"]),
        "risk_state_allocation_pass": bool(d["risk_state_allocation"]["overall_pass"]),
        "history_width_spearman": fnum(d["risk_state_allocation"]["history_width_spearman"]),
        "future_width_spearman": fnum(d["risk_state_allocation"]["future_width_spearman"]),
        "time_series_pass": bool(d["time_series"]["overall_pass"]),
        "kurtosis_ratio": fnum(d["time_series"]["kurtosis"]["kurtosis_ratio"]),
        "very_small_move_ratio": fnum(d["time_series"]["move_size_profile"]["very_small_moves"]["ratio"]),
        "cointegration_pass": bool(d["cointegration"]["overall_pass"]),
        "cointegration_ratio": fnum(d["cointegration"]["gen_gt_ratio"]),
        "cointegration_worst_cell_ratio": fnum(d["cointegration"]["worst_cell_ratio"]),
        "regime_pass": bool(d["regime_coverage"]["overall_pass"]),
        "regime_layer2_pass_count": int(d["regime_coverage"]["layer2_n_passing"]),
        "regime_layer3_catastrophic_rate": fnum(d["regime_coverage"]["layer3_catastrophic_rate"]),
        "distribution_pass": bool(d["distributional_fidelity"]["overall_pass"]),
        "daily_ks_pass_cells": int(d["distributional_fidelity"]["ks_test"]["n_pass"]),
        "level_ks_pass_cells": int(d["distributional_fidelity"]["ks_level_test"]["n_pass"]),
        "median_bias_pass_cells": int(d["distributional_fidelity"]["median_bias"]["n_pass"]),
        "mean_reversion_pass": bool(d["mean_reversion"]["overall_pass"]),
        "mean_reversion_ratio": fnum(d["mean_reversion"]["mr_gt_ratio"]),
        "pathwise_pass": bool(d["pathwise_jump_realism"]["overall_pass"]),
        "pathwise_max_jump_ks": fnum(d["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
    }


def hard_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["horizon"]), int(row["row"]), int(row["col"])


def hard_cell_comparison(incumbent_path: str, candidate_path: str) -> dict[str, Any]:
    incumbent = load_json(incumbent_path)
    candidate = load_json(candidate_path)
    inc_rows = {hard_key(row): row for row in incumbent["hard_cells"]}
    cand_rows = {hard_key(row): row for row in candidate["hard_cells"]}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(inc_rows) & set(cand_rows)):
        inc = inc_rows[key]
        cand = cand_rows[key]
        rows.append(
            {
                "horizon": key[0],
                "row": key[1],
                "col": key[2],
                "incumbent_coverage90": fnum(inc["coverage90"]),
                "candidate_coverage90": fnum(cand["coverage90"]),
                "delta_coverage90": fnum(cand["coverage90"] - inc["coverage90"]),
                "incumbent_low_coverage90": fnum(inc["low_coverage90"]),
                "candidate_low_coverage90": fnum(cand["low_coverage90"]),
                "delta_low_coverage90": fnum(cand["low_coverage90"] - inc["low_coverage90"]),
                "incumbent_lower_miss_rate": fnum(inc["lower_miss_rate"]),
                "candidate_lower_miss_rate": fnum(cand["lower_miss_rate"]),
                "delta_lower_miss_rate": fnum(cand["lower_miss_rate"] - inc["lower_miss_rate"]),
                "incumbent_width_current_corr": fnum(inc["width_vs_current_corr"]),
                "candidate_width_current_corr": fnum(cand["width_vs_current_corr"]),
                "incumbent_slope_gap": fnum(abs(inc["generated_median_vs_current_slope"] - inc["realized_vs_current_slope"])),
                "candidate_slope_gap": fnum(abs(cand["generated_median_vs_current_slope"] - cand["realized_vs_current_slope"])),
            }
        )
    return {
        "incumbent_path": incumbent_path,
        "candidate_path": candidate_path,
        "incumbent_summary": incumbent["summary"],
        "candidate_summary": candidate["summary"],
        "rows": rows,
    }


def attribution(scorecards: list[dict[str, Any]], hard_cells: dict[str, Any]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in scorecards}
    incumbent = by_name["734a_val_incumbent"]
    interval = by_name["742a_val_interval_score"]
    scale_local = by_name["745a_val_scale_local"]
    state_tail = by_name["748a_val_state_tail_sampler"]
    train_inc = by_name["746a_train_tail_incumbent"]
    train_tail = by_name["748a_train_tail_state_tail_sampler"]
    return {
        "objective_weighting": {
            "classification": "rejected_as_primary_bottleneck",
            "evidence": [
                f"742a interval-score objective improved aggregate cov90 to {interval['cov90']} but fell to {interval['n_pass']}/11 and damaged level/path realism.",
                f"748a state-tail sampler lowered validation cov90 from {incumbent['cov90']} to {state_tail['cov90']} and train-tail score from {train_inc['n_pass']}/11 to {train_tail['n_pass']}/11.",
            ],
        },
        "simple_state_geometry": {
            "classification": "insufficient_as_prefix_feature",
            "evidence": [
                f"745a scale-local prefix stayed {scale_local['n_pass']}/11 and reduced validation level-KS cells versus the incumbent.",
                "The validation risk-state allocation diagnostic already passes for 734a/748a, so the model responds to broad regimes; the miss is per-cell and late-horizon.",
            ],
        },
        "validation_shift": {
            "classification": "real_contributor_not_complete_explanation",
            "evidence": [
                "743a showed large train-tail/validation level shifts for the stable hard cells.",
                f"746a train-tail incumbent passes coverage/distribution but still fails conditionality, time-series, cointegration, regime coverage, and pathwise jumps at {train_inc['n_pass']}/11.",
            ],
        },
        "transition_readout_capacity": {
            "classification": "most_likely_next_target",
            "evidence": [
                "Daily increments are largely realistic, but integrated level distributions and late-horizon hard cells fail; this points to multi-step transition/readout allocation, not one-day marginal support.",
                f"748a hard-cell low-tertile median coverage is {hard_cells['candidate_summary']['median_low_tertile_coverage90']} with lower-miss median {hard_cells['candidate_summary']['median_hard_lower_miss_rate']}, worse than the incumbent hard-cell audit.",
                "The model can allocate broad risk-state width, but does not carry state-local directional tail geometry through the 30-day path.",
            ],
        },
        "decision": (
            "Do not continue tuning sampler weights or scalar losses. The next experiment should change one architectural axis: "
            "a minimal shared transition/readout capacity increase that preserves the normalized-innovation law, one stochastic source, "
            "and the same tri-scope framework recipe."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 749a IV Trade-Off Attribution",
        "",
        "## Scorecards",
        "",
        "| run | pass | cov90 | calerr | cond MAE% | risk-state | level KS | bias | coint | regime L2 | TS | MR | path KS |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in report["scorecards"]:
        lines.append(
            "| {name} | {n_pass}/11 | {cov90:.3f} | {calibration_error:.3f} | "
            "{conditionality_mae_reduction_pct:.2f} | {risk_state_allocation_pass} | "
            "{level_ks_pass_cells}/25 | {median_bias_pass_cells}/25 | {cointegration_ratio:.3f} | "
            "{regime_layer2_pass_count}/8 | {time_series_pass} | {mean_reversion_pass} | "
            "{pathwise_max_jump_ks:.3f} |".format(**row)
        )
    hc = report["hard_cell_comparison"]
    lines.extend(
        [
            "",
            "## Hard-Cell Candidate Versus Incumbent",
            "",
            f"- incumbent median low-tertile coverage90: `{hc['incumbent_summary']['median_low_tertile_coverage90']}`",
            f"- candidate median low-tertile coverage90: `{hc['candidate_summary']['median_low_tertile_coverage90']}`",
            f"- incumbent median hard lower-miss rate: `{hc['incumbent_summary']['median_hard_lower_miss_rate']}`",
            f"- candidate median hard lower-miss rate: `{hc['candidate_summary']['median_hard_lower_miss_rate']}`",
            "",
            "| horizon | cell | inc cov | cand cov | delta cov | inc low cov | cand low cov | delta low cov | cand lower miss |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in hc["rows"]:
        lines.append(
            "| {horizon} | ({row},{col}) | {incumbent_coverage90:.3f} | {candidate_coverage90:.3f} | "
            "{delta_coverage90:.3f} | {incumbent_low_coverage90:.3f} | {candidate_low_coverage90:.3f} | "
            "{delta_low_coverage90:.3f} | {candidate_lower_miss_rate:.3f} |".format(**row)
        )
    lines.extend(["", "## Attribution", ""])
    for key in ["objective_weighting", "simple_state_geometry", "validation_shift", "transition_readout_capacity"]:
        item = report["attribution"][key]
        lines.append(f"### {key}")
        lines.append(f"- classification: `{item['classification']}`")
        for evidence in item["evidence"]:
            lines.append(f"- {evidence}")
        lines.append("")
    lines.extend(["## Decision", "", report["attribution"]["decision"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="results/block_ar/749a_iv_tradeoff_attribution")
    parser.add_argument("--incumbent_hard_cell_json", default="results/block_ar/744a_iv_hard_cell_condition_response/summary.json")
    parser.add_argument("--candidate_hard_cell_json", default="results/block_ar/749a_748_hard_cell_condition_response/summary.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecards = [scorecard_row(name, path) for name, path in DEFAULT_SCORECARDS.items()]
    hard_cells = hard_cell_comparison(args.incumbent_hard_cell_json, args.candidate_hard_cell_json)
    report = {
        "scorecards": scorecards,
        "hard_cell_comparison": hard_cells,
        "attribution": attribution(scorecards, hard_cells),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "summary.md")
    print(json.dumps(report["attribution"], indent=2))


if __name__ == "__main__":
    main()
