#!/usr/bin/env python
"""Build and apply product-facing start reliability gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTROL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_865d_full_s192_symmetric/"
    "fixed_start_control_suite.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_reliability_gate_865d_full_s192_symmetric"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _product_status(control_status: str, warnings: list[str], failures: list[str]) -> str:
    if failures or control_status == "fail":
        return "warn_high_instability"
    if warnings or control_status == "warning":
        return "warn_needs_stronger_evidence"
    return "pass"


def _decision_text(status: str) -> str:
    if status == "pass":
        return "Start has enough fixed-start evidence for the current narrative support workflow."
    if status == "warn_high_instability":
        return (
            "Generate only with a prominent reliability warning: same-start "
            "reruns or other controls move too much relative to cross-narrative "
            "differences."
        )
    if status == "warn_needs_stronger_evidence":
        return (
            "Generate with caution: existing controls are not clean enough for a "
            "production default without stronger sample-count or repeat evidence."
        )
    return "No reliability evidence is available for this start."


def build_start_reliability_manifest(
    *,
    control_report: dict[str, Any],
    sample_count: int,
    min_sample_count: int = 192,
) -> dict[str, Any]:
    start_rows: list[dict[str, Any]] = []
    for row in _as_list(control_report.get("per_start_controls")):
        row = _as_dict(row)
        warnings = [str(item) for item in _as_list(row.get("warnings"))]
        failures = [str(item) for item in _as_list(row.get("failures"))]
        if int(sample_count) < int(min_sample_count):
            warnings = [*warnings, "scenario_sample_count_below_reliability_floor"]
        status = _product_status(str(row.get("status", "unknown")), warnings, failures)
        start_rows.append(
            {
                "start_name": str(row.get("start_name", "")),
                "product_status": status,
                "control_status": str(row.get("status", "unknown")),
                "decision": _decision_text(status),
                "observed_median_gap": _as_float(row.get("observed_median_gap")),
                "start_only_ratio": row.get("start_only_ratio"),
                "bootstrap_ratio": row.get("bootstrap_ratio"),
                "repeat_ratio": row.get("repeat_ratio"),
                "warnings": warnings,
                "failures": failures,
            }
        )
    counts: dict[str, int] = {}
    for row in start_rows:
        status = str(row["product_status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "status": "pass" if counts.get("warn_high_instability", 0) == 0 else "warning",
        "scope_note": (
            "Product-facing start reliability manifest derived from fixed-start "
            "control evidence. This gate does not replace narrative grounding; "
            "it decides how much trust to put in the selected starting level."
        ),
        "sample_count": int(sample_count),
        "min_sample_count": int(min_sample_count),
        "control_report_status": control_report.get("status", "unknown"),
        "status_counts": counts,
        "starts": start_rows,
    }


def evaluate_start_reliability(
    manifest: dict[str, Any], start_name: str
) -> dict[str, Any]:
    for row in _as_list(manifest.get("starts")):
        row = _as_dict(row)
        if row.get("start_name") == start_name:
            return {
                "start_name": start_name,
                "product_status": row.get("product_status", "unknown"),
                "decision": row.get("decision", ""),
                "warnings": _as_list(row.get("warnings")),
                "failures": _as_list(row.get("failures")),
                "source": "manifest",
            }
    return {
        "start_name": start_name,
        "product_status": "warn_needs_stronger_evidence",
        "decision": _decision_text("warn_needs_stronger_evidence"),
        "warnings": ["start_reliability_evidence_missing"],
        "failures": [],
        "source": "fallback",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Start Reliability Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Sample count: `{report['sample_count']}`",
        f"- Minimum sample count: `{report['min_sample_count']}`",
        "",
        "## Starts",
        "",
        "| Start | Product Status | Control | Observed Gap | Bootstrap Ratio | Repeat Ratio | Warnings | Failures |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in _as_list(report.get("starts")):
        row = _as_dict(row)
        warnings = ", ".join(_as_list(row.get("warnings"))) or "none"
        failures = ", ".join(_as_list(row.get("failures"))) or "none"
        lines.append(
            "| "
            f"`{row.get('start_name')}` | "
            f"`{row.get('product_status')}` | "
            f"`{row.get('control_status')}` | "
            f"`{_as_float(row.get('observed_median_gap')):.3f}` | "
            f"`{_as_float(row.get('bootstrap_ratio')):.3f}` | "
            f"`{_as_float(row.get('repeat_ratio')):.3f}` | "
            f"{warnings} | "
            f"{failures} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", type=Path, default=DEFAULT_CONTROL_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=192)
    parser.add_argument("--min-sample-count", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_start_reliability_manifest(
        control_report=_load_json(args.control_report),
        sample_count=args.sample_count,
        min_sample_count=args.min_sample_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["artifact_paths"] = {
        "report": str(args.output_dir / "start_reliability_gate.json"),
        "markdown": str(args.output_dir / "start_reliability_gate.md"),
    }
    _write_json(report["artifact_paths"]["report"], report)
    Path(report["artifact_paths"]["markdown"]).write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "status_counts": report["status_counts"],
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
