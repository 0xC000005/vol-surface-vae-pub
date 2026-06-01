#!/usr/bin/env python
"""Gate a readout calibration across multiple fixed-start conditionality audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_component_start_stratification import (
    audit_row,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _status_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _selected_quality_ok(selected: dict[str, Any]) -> bool:
    deltas = selected.get("quality_deltas", {})
    return (
        float(deltas.get("coverage_80_delta", 0.0)) >= 0.0
        and float(deltas.get("crps_delta", 0.0)) <= 0.0
        and float(deltas.get("energy_delta", 0.0)) <= 0.0
    )


def build_start_aware_readout_gate(
    readout_selection: dict[str, Any],
    audit_paths: list[Path],
    *,
    min_pass_starts: int = 2,
) -> dict[str, Any]:
    """Combine readout quality with multi-start conditionality evidence."""

    selected = dict(readout_selection.get("selected") or {})
    rows = [audit_row(path) for path in audit_paths]
    for row, path in zip(rows, audit_paths, strict=True):
        row["audit_path"] = str(path)
    status_counts = _status_counts(rows, "status")
    reason_counts = _status_counts(rows, "reason")
    pass_count = int(status_counts.get("pass", 0))
    warning_count = int(status_counts.get("warning", 0))
    fail_count = int(status_counts.get("fail", 0))
    selected_gate_ok = str(selected.get("gate_status")) == "pass"
    selected_quality_ok = _selected_quality_ok(selected)
    broad_promotion = (
        selected_gate_ok
        and selected_quality_ok
        and fail_count == 0
        and warning_count == 0
        and pass_count >= int(min_pass_starts)
    )
    narrow_promotion = selected_gate_ok and selected_quality_ok and pass_count >= 1

    findings: list[str] = []
    if not selected_gate_ok:
        findings.append("selected_readout_failed_base_gate")
    if not selected_quality_ok:
        findings.append("selected_readout_quality_not_improved")
    if pass_count < int(min_pass_starts):
        findings.append("too_few_clean_starts_for_broad_promotion")
    if warning_count:
        findings.append("warning_starts_remain_bootstrap_limited")
    if fail_count:
        findings.append("failed_starts_present")

    if broad_promotion:
        status = "pass"
        recommendation = "promote_selected_readout_broadly"
    elif narrow_promotion:
        status = "warning"
        recommendation = "keep_selected_readout_as_local_candidate_only"
    else:
        status = "fail" if fail_count else "warning"
        recommendation = "keep_uncalibrated_component_mixture_default"

    return {
        "status": status,
        "recommendation": recommendation,
        "selected_readout": selected,
        "min_pass_starts": int(min_pass_starts),
        "start_status_counts": status_counts,
        "start_reason_counts": reason_counts,
        "start_rows": rows,
        "findings": findings,
        "broad_promotion": bool(broad_promotion),
        "narrow_promotion": bool(narrow_promotion),
        "interpretation": [
            "The readout can be promoted broadly only when it improves held-out quality and clears the conditionality gate on multiple fixed starts.",
            "A clean start means narrative separation is larger than repeat noise, start-only controls, and bootstrap noise under that start.",
            "A warning start still shows narrative response, but finite-sample/bootstrap noise is too close to the observed narrative separation for a broad production claim.",
            "This gate prevents one clean start from being mistaken for a generally reliable readout calibration.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    selected = report.get("selected_readout", {})
    lines = [
        "# Start-Aware Readout Gate",
        "",
        f"Status: `{report['status']}`",
        f"Recommendation: `{report['recommendation']}`",
        f"Selected readout: `{selected.get('name', '')}`",
        f"Start counts: `{json.dumps(report['start_status_counts'], sort_keys=True)}`",
        f"Findings: `{json.dumps(report['findings'], sort_keys=True)}`",
        "",
        "| Starts | Status | Reason | Samples | Repeat Energy | Bootstrap Energy | Sample-Matched Bootstrap Energy | Start-Only Energy | Audit |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["start_rows"]:
        lines.append(
            "| "
            + ", ".join(str(value) for value in row.get("starts", []))
            + f" | {row.get('status', '')}"
            + f" | {row.get('reason', '')}"
            + f" | {row.get('sample_count') or ''}"
            + f" | {_format_float(row.get('repeat_to_observed_path_energy'))}"
            + f" | {_format_float(row.get('bootstrap_to_observed_path_energy'))}"
            + f" | {_format_float(row.get('bootstrap_to_sample_matched_observed_path_energy'))}"
            + f" | {_format_float(row.get('start_only_to_observed_path_energy'))}"
            + f" | `{row.get('audit_path', '')}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-selection-json", required=True)
    parser.add_argument("--audit-json", action="append", required=True)
    parser.add_argument("--min-pass-starts", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_start_aware_readout_gate(
        _read_json(args.readout_selection_json),
        [Path(path) for path in args.audit_json],
        min_pass_starts=int(args.min_pass_starts),
    )
    json_path = output_dir / "start_aware_readout_gate.json"
    markdown_path = output_dir / "start_aware_readout_gate.md"
    report["artifact_paths"] = {
        "report": str(json_path),
        "markdown": str(markdown_path),
    }
    _write_json(json_path, report)
    write_markdown(report, markdown_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "recommendation": report["recommendation"],
                "start_status_counts": report["start_status_counts"],
                "findings": report["findings"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
