#!/usr/bin/env python
"""Summarize fixed-start component conditionality audits by start."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median
from typing import Any


RATIO_KEYS = [
    "repeat_to_observed_path_energy",
    "repeat_to_observed_path_variance",
    "repeat_to_observed_path_wasserstein",
    "bootstrap_to_observed_path_energy",
    "bootstrap_to_observed_path_variance",
    "bootstrap_to_observed_path_wasserstein",
    "bootstrap_to_sample_matched_observed_path_energy",
    "bootstrap_to_sample_matched_observed_path_variance",
    "bootstrap_to_sample_matched_observed_path_wasserstein",
    "start_only_to_observed_path_energy",
    "start_only_to_observed_path_variance",
    "start_only_to_observed_path_wasserstein",
]
START_RE = re.compile(r"_start(?P<start>\d+)(?:#|$)")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value == value]
    return float(median(finite)) if finite else None


def _start_indices(report: dict[str, Any]) -> list[int]:
    rows = report.get("start_reliability", {}).get("case_rows", [])
    values = sorted(
        {
            int(row["start_index"])
            for row in rows
            if isinstance(row, dict) and row.get("start_index") is not None
        }
    )
    if values:
        return values
    summaries = report.get("per_start_summaries", {})
    values = sorted(int(key) for key in summaries.keys())
    if values:
        return values
    parsed = set()
    for row in report.get("observed_pairwise", []):
        if not isinstance(row, dict):
            continue
        for key in ["left_case", "right_case"]:
            match = START_RE.search(str(row.get(key, "")))
            if match:
                parsed.add(int(match.group("start")))
    return sorted(parsed)


def _sample_count(report: dict[str, Any]) -> int | None:
    values = []
    for row in report.get("observed_pairwise", []):
        if not isinstance(row, dict):
            continue
        for key in ["sample_count_left", "sample_count_right"]:
            value = row.get(key)
            if value is not None:
                values.append(float(value))
    sample = _safe_median(values)
    return None if sample is None else int(round(sample))


def classify_audit(report: dict[str, Any]) -> tuple[str, str]:
    failures = [str(item) for item in report.get("failures", [])]
    warnings = [
        str(item)
        for item in (report.get("promotion_warnings") or report.get("warnings") or [])
    ]
    reliability = report.get("start_reliability", {})
    reliability_warnings = reliability.get("warning_counts", {}) or {}
    reliability_failures = reliability.get("failure_counts", {}) or {}
    if reliability_failures:
        return "fail", "start_incompatible"
    if reliability_warnings and failures:
        return "fail", "start_incompatible"
    if failures:
        if "observed_operational_start_failures" in failures:
            return "fail", "start_incompatible"
        if any("repeat" in item for item in failures):
            return "fail", "repeat_not_separated"
        return "fail", "metric_failure"
    if reliability_warnings:
        return "warning", "start_reliability_warning"
    if any("bootstrap" in item for item in warnings):
        return "warning", "bootstrap_readout_warning"
    if warnings:
        return "warning", "other_warning"
    return "pass", "clean_pass"


def audit_row(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    status, reason = classify_audit(report)
    ratios = report.get("ratios", {})
    row: dict[str, Any] = {
        "audit_path": str(path),
        "starts": _start_indices(report),
        "sample_count": _sample_count(report),
        "status": status,
        "reason": reason,
        "raw_status": report.get("status"),
        "promotion_status": report.get("promotion_status"),
        "warnings": list(report.get("promotion_warnings") or report.get("warnings") or []),
        "failures": list(report.get("failures") or []),
        "start_distance_z_median": report.get("start_reliability", {}).get(
            "start_distance_z_median"
        ),
    }
    for key in RATIO_KEYS:
        row[key] = _safe_float(ratios.get(key))
    return row


def build_stratification(paths: list[Path]) -> dict[str, Any]:
    rows = [audit_row(path) for path in paths]
    counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
    return {
        "status_counts": counts,
        "reason_counts": reason_counts,
        "audit_count": int(len(rows)),
        "rows": rows,
        "interpretation": [
            "pass means the audit clears repeat, start-only, bootstrap, and start-reliability gates.",
            "warning means narrative response is present but a reliability or bootstrap/readout gate is not clean enough for promotion.",
            "fail means the start should not support a broad component-path promotion claim under the current gate.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Component Fixed-Start Stratification",
        "",
        f"Audit count: {report['audit_count']}",
        f"Status counts: `{json.dumps(report['status_counts'], sort_keys=True)}`",
        f"Reason counts: `{json.dumps(report['reason_counts'], sort_keys=True)}`",
        "",
        "| Starts | Samples | Status | Reason | Repeat Energy | Bootstrap Energy | Sample-Matched Bootstrap Energy | Start-Only Energy |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| "
            + ", ".join(str(value) for value in row["starts"])
            + f" | {row.get('sample_count') or ''}"
            + f" | {row['status']}"
            + f" | {row['reason']}"
            + f" | {_format_float(row.get('repeat_to_observed_path_energy'))}"
            + f" | {_format_float(row.get('bootstrap_to_observed_path_energy'))}"
            + f" | {_format_float(row.get('bootstrap_to_sample_matched_observed_path_energy'))}"
            + f" | {_format_float(row.get('start_only_to_observed_path_energy'))}"
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: Any) -> str:
    numeric = _safe_float(value)
    return "" if numeric is None else f"{numeric:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_stratification([Path(path) for path in args.audit_json])
    json_path = output_dir / "component_start_stratification.json"
    markdown_path = output_dir / "component_start_stratification.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, markdown_path)
    print(
        json.dumps(
            {
                "audit_count": report["audit_count"],
                "status_counts": report["status_counts"],
                "reason_counts": report["reason_counts"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
