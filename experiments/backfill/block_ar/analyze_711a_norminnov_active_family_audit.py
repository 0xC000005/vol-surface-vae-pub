#!/usr/bin/env python
"""711a: summarize active normalized-innovation family versus 510a control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def summarize_full11(name: str, result: dict[str, Any], attribution: dict[str, Any]) -> dict[str, Any]:
    coverage = result.get("coverage", {})
    cond = result.get("conditionality", {})
    risk = result.get("risk_state_allocation", {})
    regime = result.get("regime_coverage", {})
    dist = result.get("distributional_fidelity", {})
    time_series = result.get("time_series", {})
    return {
        "name": name,
        "n_pass": int(_get(result, "summary", "n_pass", default=0)),
        "failed_suites": list(_get(result, "summary", "failed_suites", default=[])),
        "cov90": float(_get(coverage, "overall", "0.9", default=0.0)),
        "worst_cell_h30": float(_get(coverage, "worst_cell_per_horizon", "30", default=0.0)),
        "conditionality_mae_reduction_pct": float(cond.get("mae_reduction_pct", 0.0)),
        "risk_state_allocation_pass": _as_bool(risk.get("overall_pass", False)),
        "risk_state_width_history_rho": float(risk.get("width_history_activity_spearman", 0.0)),
        "risk_state_width_future_rho": float(risk.get("width_future_activity_spearman", 0.0)),
        "regime_layer2_pass": _as_bool(regime.get("layer2_pass", False)),
        "regime_layer3_pass": _as_bool(regime.get("layer3_pass", False)),
        "level_ks_n_pass": int(_get(dist, "ks_level_test", "n_pass", default=0)),
        "median_bias_n_pass": int(_get(dist, "median_bias", "n_pass", default=0)),
        "kurtosis_ratio": float(_get(time_series, "kurtosis", "ratio", default=0.0)),
        "pathwise_max_jump_ks": float(
            _get(result, "pathwise_jump_realism", "pathwise_max_jump", "ks_stat", default=0.0)
        ),
        "undercovered_slices": int(attribution.get("n_undercovered_slices", 0)),
        "unique_undercovered_cells": int(attribution.get("n_unique_undercovered_cells", 0)),
        "undercovered_level_ks_overlap": float(
            _get(attribution, "undercovered_cell_overlap", "level_ks_fail_rate", default=0.0)
        ),
        "undercovered_cointegration_overlap": float(
            _get(attribution, "undercovered_cell_overlap", "cointegration_fail_rate", default=0.0)
        ),
    }


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_report(
    readiness: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    summaries = [summarize_full11(name, result, attr) for name, result, attr in candidates]
    readiness_by_name = {row["name"]: row for row in readiness.get("ranked_candidates", [])}
    for row in summaries:
        ready = readiness_by_name.get(row["name"], {})
        row["stress_score"] = int(ready.get("stress_score", 0))
        row["stress_score_total"] = int(ready.get("stress_score_total", 4))
        row["stress_pass"] = _as_bool(ready.get("stress_pass", False))
        row["scenario_authenticity_pass"] = _as_bool(
            _get(ready, "scenario_authenticity", "pass", default=False)
        )
        row["lower_only_coverage_pass"] = _as_bool(
            _get(ready, "coverage_lower_only", "pass", default=False)
        )
        row["lower_only_regime_pass"] = _as_bool(
            _get(ready, "regime_lower_only", "pass", default=False)
        )

    active = [row for row in summaries if "norminnov" in row["name"]]
    risk_state_ok = bool(active and all(row["risk_state_allocation_pass"] for row in active))
    deployable = bool(active and any(row["stress_pass"] for row in active))
    common_failure = (
        "The normalized-innovation family now shows real state-dependent uncertainty "
        "allocation, but it is locally too narrow in regime/cell/horizon slices. "
        "That is a dispersion/allocation failure, not evidence that the encoder is unused."
    )
    next_step = (
        "Stay on the normalized-innovation framework and run a frozen-framework coverage "
        "repair experiment that targets lower-tail/regime under-inclusion inside the same "
        "state-normalized innovation law. Do not return to 510a except as a control, "
        "because 510a is less general even when its IV-only stress score is higher."
    )
    return {
        "iteration": "711a",
        "purpose": "current-code audit of later normalized-innovation active family versus 510a control",
        "summaries": summaries,
        "mechanism_read": {
            "risk_state_allocation_valid_in_active_family": risk_state_ok,
            "active_family_risk_manager_deployable": deployable,
            "common_failure": common_failure,
        },
        "decision": next_step,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 711a Normalized-Innovation Active-Family Audit",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Full 11 | Stress | Cov90 | h30 worst | Cond MAE | Risk-state | Regime under slices | Level-KS overlap | Key failed suites |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in report["summaries"]:
        failed = ", ".join(row["failed_suites"]) if row["failed_suites"] else "none"
        lines.append(
            f"| `{row['name']}` | `{row['n_pass']}/11` | "
            f"`{row['stress_score']}/{row['stress_score_total']}` | "
            f"`{row['cov90']:.3f}` | `{row['worst_cell_h30']:.3f}` | "
            f"`{row['conditionality_mae_reduction_pct']:.2f}%` | "
            f"`{row['risk_state_allocation_pass']}` | "
            f"`{row['undercovered_slices']}` | "
            f"`{row['undercovered_level_ks_overlap']:.3f}` | {failed} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Read",
            "",
            f"- risk-state allocation valid in active family: `{report['mechanism_read']['risk_state_allocation_valid_in_active_family']}`",
            f"- active family risk-manager deployable now: `{report['mechanism_read']['active_family_risk_manager_deployable']}`",
            f"- common failure: {report['mechanism_read']['common_failure']}",
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness_json", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("NAME", "FULL11_JSON", "ATTRIBUTION_JSON"),
        required=True,
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    readiness = load_json(args.readiness_json)
    candidates = [
        (name, load_json(full11_path), load_json(attribution_path))
        for name, full11_path, attribution_path in args.candidate
    ]
    report = build_report(readiness, candidates)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(json.dumps(make_serializable(report["mechanism_read"]), indent=2))


if __name__ == "__main__":
    main()
