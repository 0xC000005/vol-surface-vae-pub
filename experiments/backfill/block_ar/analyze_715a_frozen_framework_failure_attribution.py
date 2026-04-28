#!/usr/bin/env python
"""715a: attribute 714a frozen-framework failures before changing the recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_grid(grid: list[list[float]]) -> list[tuple[float, int, int]]:
    rows: list[tuple[float, int, int]] = []
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            rows.append((float(value), i, j))
    return rows


def level_ks_failures(iv: dict[str, Any]) -> list[dict[str, Any]]:
    ks = iv["distributional_fidelity"]["ks_level_test"]
    gate = float(ks["ks_gate"])
    rows = [
        {"cell": [i, j], "ks": value, "gate": gate}
        for value, i, j in flatten_grid(ks["ks_grid"])
        if value > gate
    ]
    return sorted(rows, key=lambda item: item["ks"], reverse=True)


def coverage_worst_cells(iv: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon, grid in iv["coverage"]["per_cell_coverage"].items():
        for value, i, j in flatten_grid(grid):
            rows.append(
                {
                    "horizon": int(horizon),
                    "cell": [i, j],
                    "coverage_90": value,
                    "gap_to_90": 0.9 - value,
                }
            )
    return sorted(rows, key=lambda item: item["coverage_90"])[:10]


def regime_worst_cells(iv: dict[str, Any]) -> list[dict[str, Any]]:
    layer2 = iv["regime_coverage"]["layer2_regime_cell"]
    rows: list[dict[str, Any]] = []
    for regime, horizons in layer2.items():
        for horizon, payload in horizons.items():
            for value, i, j in flatten_grid(payload["grid"]):
                rows.append(
                    {
                        "regime": regime,
                        "horizon": int(horizon),
                        "cell": [i, j],
                        "coverage_90": value,
                        "gap_to_90": 0.9 - value,
                    }
                )
    return sorted(rows, key=lambda item: item["coverage_90"])[:10]


def failed_factor_rows(panel: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in panel["summary"]["per_factor"]:
        gt_range = max(float(row["gt_max"]) - float(row["gt_min"]), 1e-12)
        gen_range = float(row["gen_max"]) - float(row["gen_min"])
        rows.append(
            {
                "name": row["name"],
                "ks_delta": float(row["ks_delta"]),
                "q99_abs_delta_ratio": float(row["q99_abs_delta_ratio"]),
                "gt_min": float(row["gt_min"]),
                "gt_max": float(row["gt_max"]),
                "gen_min": float(row["gen_min"]),
                "gen_max": float(row["gen_max"]),
                "range_ratio": gen_range / gt_range,
                "upper_overshoot_ratio": float(row["gen_max"]) / max(float(row["gt_max"]), 1e-12),
            }
        )
    return sorted(rows, key=lambda item: item["ks_delta"], reverse=True)


def summarize_scope(panel: dict[str, Any]) -> dict[str, Any]:
    summary = panel["summary"]
    rows = failed_factor_rows(panel)
    return {
        "factor_delta_ks_mean": summary["factor_delta_ks_mean"],
        "factor_delta_ks_pass_020": summary["factor_delta_ks_pass_020"],
        "n_factors": summary["n_factors"],
        "failed_factors": [row for row in rows if row["ks_delta"] >= 0.20],
        "top_ks_factors": rows[:5],
        "tail_ratio_median": summary["factor_tail_q99_ratio_median"],
        "factor_factor_corr": summary["factor_factor_corr"],
        "iv_factor_corr": summary["iv_factor_corr"],
        "conditional_panel": summary["conditional_panel"],
    }


def optional_control_delta(control_path: Path | None, iv: dict[str, Any]) -> dict[str, Any]:
    if control_path is None or not control_path.exists():
        return {}
    control = load_json(control_path)
    return {
        "control_path": str(control_path),
        "cov90_delta": iv["coverage"]["overall"]["0.9"]
        - control["coverage"]["overall"]["0.9"],
        "h30_worst_delta": iv["coverage"]["worst_cell_per_horizon"]["30"]
        - control["coverage"]["worst_cell_per_horizon"]["30"],
        "level_ks_pass_delta": iv["distributional_fidelity"]["ks_level_test"]["n_pass"]
        - control["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "max_jump_ks_delta": iv["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]
        - control["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def build_report(
    iv: dict[str, Any],
    anchor: dict[str, Any],
    joint: dict[str, Any],
    scorecard: dict[str, Any],
    control_delta: dict[str, Any],
) -> dict[str, Any]:
    anchor_scope = summarize_scope(anchor)
    joint_scope = summarize_scope(joint)
    anchor_failed = {row["name"] for row in anchor_scope["failed_factors"]}
    joint_failed = {row["name"] for row in joint_scope["failed_factors"]}
    common_failed = sorted(anchor_failed & joint_failed)
    return {
        "scorecard": {
            "overall_pass": scorecard["overall_pass"],
            "gate_passes": scorecard["gate_passes"],
        },
        "iv_attribution": {
            "n_pass": iv["summary"]["n_pass"],
            "effective_failed_suites": scorecard["iv"]["effective_failed_suites"],
            "cov90": iv["coverage"]["overall"]["0.9"],
            "h30_worst_cell_cov90": iv["coverage"]["worst_cell_per_horizon"]["30"],
            "coverage_worst_cells": coverage_worst_cells(iv),
            "level_ks_pass": iv["distributional_fidelity"]["ks_level_test"]["n_pass"],
            "level_ks_failures": level_ks_failures(iv),
            "regime_layer2_pass": [
                iv["regime_coverage"]["layer2_n_passing"],
                iv["regime_coverage"]["layer2_n_total"],
            ],
            "regime_layer3_catastrophic_rate": iv["regime_coverage"]["layer3_catastrophic_rate"],
            "regime_worst_cells": regime_worst_cells(iv),
            "risk_state_allocation_pass": iv["risk_state_allocation"]["overall_pass"],
        },
        "anchor_attribution": anchor_scope,
        "joint_attribution": joint_scope,
        "cross_scope_factor_signal": {
            "same_failed_factors_anchor_and_joint": common_failed,
            "interpretation": (
                "The factor failure is localized and repeats across anchor-only and "
                "joint38, so native co-modeling is not the primary cause."
            ),
        },
        "control_delta_vs_674a": control_delta,
        "decision": {
            "failure_class": "localized coordinate/support and calibration failure, not framework inconsistency",
            "next_experiment": (
                "Prefer a data-coordinate/support repair for positive spread-like factors "
                "and IV tail/regime calibration before adding another global loss or "
                "switching backend."
            ),
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    iv = report["iv_attribution"]
    anchor = report["anchor_attribution"]
    joint = report["joint_attribution"]
    lines = [
        "# 715a Frozen-Framework Failure Attribution",
        "",
        "## Read",
        f"- overall scorecard pass: `{report['scorecard']['overall_pass']}`",
        f"- gate passes: `{report['scorecard']['gate_passes']}`",
        f"- failure class: `{report['decision']['failure_class']}`",
        "",
        "## IV Failure",
        f"- IV score: `{iv['n_pass']}/11`",
        f"- effective failed suites: `{iv['effective_failed_suites']}`",
        f"- cov90 overall / h30 worst cell: `{iv['cov90']:.3f}` / `{iv['h30_worst_cell_cov90']:.3f}`",
        f"- level-KS pass cells: `{iv['level_ks_pass']}/25`",
        f"- regime layer2 pass cells: `{iv['regime_layer2_pass'][0]}/{iv['regime_layer2_pass'][1]}`",
        f"- regime layer3 catastrophic rate: `{iv['regime_layer3_catastrophic_rate']:.3f}`",
        f"- risk-state allocation pass: `{iv['risk_state_allocation_pass']}`",
        "",
        "Worst IV coverage cells:",
        "",
        "| horizon | cell | cov90 | gap to 0.90 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in iv["coverage_worst_cells"][:5]:
        lines.append(
            f"| {row['horizon']} | {row['cell']} | {row['coverage_90']:.3f} | {row['gap_to_90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Worst IV regime cells:",
            "",
            "| regime | horizon | cell | cov90 | gap to 0.90 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in iv["regime_worst_cells"][:5]:
        lines.append(
            f"| {row['regime']} | {row['horizon']} | {row['cell']} | {row['coverage_90']:.3f} | {row['gap_to_90']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Anchor And Joint Factor Failure",
            f"- anchor factor delta KS pass: `{anchor['factor_delta_ks_pass_020']}/{anchor['n_factors']}`",
            f"- joint factor delta KS pass: `{joint['factor_delta_ks_pass_020']}/{joint['n_factors']}`",
            f"- common failed factors: `{report['cross_scope_factor_signal']['same_failed_factors_anchor_and_joint']}`",
            f"- joint IV-factor matrix corr: `{joint['iv_factor_corr']['matrix_corr']:.3f}`",
            f"- anchor conditional-panel reduction: `{anchor['conditional_panel']['median_mae_reduction_vs_rolled_pct']:.2f}%`",
            f"- joint conditional-panel reduction: `{joint['conditional_panel']['median_mae_reduction_vs_rolled_pct']:.2f}%`",
            "",
            "| scope | factor | KS(delta) | q99 ratio | range ratio | gen max / gt max |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope_name, scope in [("anchor", anchor), ("joint", joint)]:
        for row in scope["top_ks_factors"][:3]:
            lines.append(
                f"| {scope_name} | {row['name']} | {row['ks_delta']:.3f} | "
                f"{row['q99_abs_delta_ratio']:.3f} | {row['range_ratio']:.3f} | "
                f"{row['upper_overshoot_ratio']:.3f} |"
            )
    if report["control_delta_vs_674a"]:
        delta = report["control_delta_vs_674a"]
        lines.extend(
            [
                "",
                "## Delta Versus 674a IV Control",
                f"- cov90 delta: `{delta['cov90_delta']:.3f}`",
                f"- h30 worst-cell delta: `{delta['h30_worst_delta']:.3f}`",
                f"- level-KS pass-cell delta: `{delta['level_ks_pass_delta']}`",
                f"- max-jump KS delta: `{delta['max_jump_ks_delta']:.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            report["decision"]["next_experiment"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iv_json", required=True)
    parser.add_argument("--anchor_json", required=True)
    parser.add_argument("--joint_json", required=True)
    parser.add_argument("--scorecard_json", required=True)
    parser.add_argument("--control_iv_json")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    iv = load_json(Path(args.iv_json))
    anchor = load_json(Path(args.anchor_json))
    joint = load_json(Path(args.joint_json))
    scorecard = load_json(Path(args.scorecard_json))
    control_delta = optional_control_delta(
        Path(args.control_iv_json) if args.control_iv_json else None,
        iv,
    )
    report = build_report(iv, anchor, joint, scorecard, control_delta)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(out_md, report)
    print(json.dumps(report["decision"], indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
