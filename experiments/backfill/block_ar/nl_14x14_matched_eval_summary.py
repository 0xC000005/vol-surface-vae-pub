#!/usr/bin/env python
"""Summarize matched 14+14 frozen-SNI scenario evaluations by candidate method."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SCENARIO_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_matched_top3_90_scenario_eval_990g/"
    "scenario_level_eval_report.json"
)
DEFAULT_SUPPORT_AUDIT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_support_audit_990f/support_level_audit_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_matched_top3_90_scenario_eval_990g"
)
METRICS = (
    "ensemble_crps_z",
    "energy_score_z",
    "coverage_80",
    "mean_path_mae_z",
    "mean_path_rmse_z",
    "terminal_mae_z",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _mean(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        float(row[metric])
        for row in rows
        if row.get(metric) is not None and math.isfinite(float(row[metric]))
    ]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _round(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(round(float(value), 12))


def summarize_by_query_kind(
    scenario_report: dict[str, Any],
    *,
    generator_method: str = "narrative_generator_topk",
    baseline_method: str = "persistence",
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scenario_report.get("window_scores", []):
        if not isinstance(row, dict):
            continue
        kind = str(row.get("query_kind", ""))
        if not kind:
            continue
        grouped[kind].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for kind, rows in grouped.items():
        method_rows = [
            row.get("methods", {}).get(generator_method, {})
            for row in rows
            if isinstance(row.get("methods", {}).get(generator_method), dict)
        ]
        baseline_rows = [
            row.get("methods", {}).get(baseline_method, {})
            for row in rows
            if isinstance(row.get("methods", {}).get(baseline_method), dict)
        ]
        block: dict[str, Any] = {
            "window_count": int(len(rows)),
            "query_windows": sorted({int(row["window_index"]) for row in rows}),
        }
        for metric in METRICS:
            value = _mean(method_rows, metric)
            if value is not None:
                block[f"{metric}_mean"] = _round(value)
            if metric != "coverage_80" and value is not None:
                base = _mean(baseline_rows, metric)
                if base is not None and base > 0.0:
                    block[f"{metric}_improvement_vs_{baseline_method}"] = _round(
                        (base - value) / base
                    )
        summaries[kind] = block
    ranking = sorted(
        summaries.items(),
        key=lambda item: (
            -float(item[1].get("ensemble_crps_z_improvement_vs_persistence", -1e9)),
            -float(item[1].get("energy_score_z_improvement_vs_persistence", -1e9)),
            str(item[0]),
        ),
    )
    return {
        "generator_method": generator_method,
        "baseline_method": baseline_method,
        "method_summaries": summaries,
        "ranking_by_crps_then_energy": [
            {"rank": rank, "method": method, **summary}
            for rank, (method, summary) in enumerate(ranking, 1)
        ],
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Matched 14+14 Frozen-SNI Scenario Evaluation",
        "",
        f"- Status: `{report['status']}`",
        f"- Scenario rows: `{report['scenario_eval']['heldout_window_count']}`",
        f"- Matched query windows: `{len(report['matched_query_window_indices'])}`",
        f"- Methods compared: `{len(report['method_summaries'])}`",
        f"- Support sampling: `{report['scenario_eval']['support_sampling_mode']}`",
        f"- Samples per support: `{report['scenario_eval']['samples']}`",
        "",
        "## Ranked Methods",
        "",
        "| Rank | Method | Windows | CRPS mean | CRPS improvement | Energy mean | Energy improvement | Coverage80 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["ranking_by_crps_then_energy"]:
        lines.append(
            "| {rank} | {method} | {windows} | {crps:.4f} | {crps_imp:.4f} | {energy:.4f} | {energy_imp:.4f} | {coverage:.4f} |".format(
                rank=row["rank"],
                method=row["method"],
                windows=row["window_count"],
                crps=float(row.get("ensemble_crps_z_mean") or float("nan")),
                crps_imp=float(row.get("ensemble_crps_z_improvement_vs_persistence") or float("nan")),
                energy=float(row.get("energy_score_z_mean") or float("nan")),
                energy_imp=float(row.get("energy_score_z_improvement_vs_persistence") or float("nan")),
                coverage=float(row.get("coverage_80_mean") or float("nan")),
            )
        )
    support = report.get("support_method_summaries", {})
    if support:
        lines.extend(
            [
                "",
                "## Support Audit Context",
                "",
                "| Method | Unique supports | Mean top1 score |",
                "|---|---:|---:|",
            ]
        )
        for method, summary in sorted(support.items()):
            lines.append(
                "| {method} | {unique} | {score:.4f} |".format(
                    method=method,
                    unique=summary.get("unique_support_count", 0),
                    score=float(summary.get("mean_top1_score") or float("nan")),
                )
            )
    note = report.get("public_paper_demo_candidate_note")
    if note:
        lines.extend(["", "## Candidate Note", "", str(note), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_summary(
    *,
    scenario_report_path: str | Path,
    support_audit_path: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    scenario = _load_json(scenario_report_path)
    support = _load_json(support_audit_path) if support_audit_path else {}
    grouped = summarize_by_query_kind(scenario)
    query_windows = sorted(
        {
            int(row["window_index"])
            for row in scenario.get("window_scores", [])
            if isinstance(row, dict) and row.get("window_index") is not None
        }
    )
    output = _resolve(output_dir)
    report_path = output / "matched_scenario_eval_by_method.json"
    markdown_path = output / "matched_scenario_eval_by_method.md"
    report = {
        "schema_version": "nl_14x14_matched_scenario_eval_by_method_v1",
        "status": "pass" if scenario.get("status") == "ok" else "fail",
        "scenario_report": str(_resolve(scenario_report_path)),
        "support_audit_report": str(_resolve(support_audit_path))
        if support_audit_path
        else "",
        "matched_query_window_indices": query_windows,
        "scenario_eval": {
            "heldout_window_count": int(scenario.get("heldout_window_count", 0)),
            "top_k": int(scenario.get("top_k", 0)),
            "samples": int(scenario.get("samples", 0)),
            "support_sampling_mode": str(scenario.get("support_sampling_mode", "")),
            "common_random_numbers": scenario.get("common_random_numbers", {}),
        },
        "method_summaries": grouped["method_summaries"],
        "ranking_by_crps_then_energy": grouped["ranking_by_crps_then_energy"],
        "support_method_summaries": support.get("method_summaries", {}),
        "public_paper_demo_candidate_note": support.get(
            "public_paper_demo_candidate_note", ""
        ),
        "artifact_paths": {
            "report": str(report_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(report_path, report)
    _write_markdown(markdown_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-report", type=Path, default=DEFAULT_SCENARIO_REPORT)
    parser.add_argument("--support-audit", type=Path, default=DEFAULT_SUPPORT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = build_summary(
        scenario_report_path=args.scenario_report,
        support_audit_path=args.support_audit,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "methods": len(report["method_summaries"]),
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
                "top_method": report["ranking_by_crps_then_energy"][0]["method"]
                if report["ranking_by_crps_then_energy"]
                else "",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
