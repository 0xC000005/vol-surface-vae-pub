#!/usr/bin/env python
"""Select a readout calibration only if quality and conditionality gates agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _quality_deltas(report: dict[str, Any]) -> dict[str, float]:
    comparison = report.get("comparison", {})
    return {
        "coverage_80_delta": float(
            comparison.get("eval_calibrated_minus_component_coverage_80", 0.0)
        ),
        "crps_delta": float(comparison.get("eval_calibrated_minus_component_crps", 0.0)),
        "energy_delta": float(
            comparison.get("eval_calibrated_minus_component_energy", 0.0)
        ),
    }


def _gate_status(path_report: dict[str, Any]) -> str:
    failures = path_report.get("failures", [])
    warnings = path_report.get("warnings", [])
    if failures:
        return "fail"
    if warnings or str(path_report.get("status", "")) == "warning":
        return "warning"
    if str(path_report.get("status", "")) == "pass":
        return "pass"
    return str(path_report.get("status", "unknown"))


def _quality_score(deltas: dict[str, float]) -> float:
    # CRPS and energy are losses, so negative deltas are improvements.
    return (
        float(deltas["coverage_80_delta"])
        - float(deltas["crps_delta"])
        - float(deltas["energy_delta"])
    )


def _candidate_row(candidate: dict[str, Any]) -> dict[str, Any]:
    calibration = candidate["calibration_report"]
    path_report = candidate["path_report"]
    deltas = _quality_deltas(calibration)
    gate_status = _gate_status(path_report)
    alpha = float(calibration.get("selected_alpha", candidate.get("alpha", 1.0)))
    return {
        "name": str(candidate.get("name", f"alpha_{alpha:g}")),
        "alpha": alpha,
        "gate_status": gate_status,
        "quality_deltas": deltas,
        "quality_score": _quality_score(deltas),
        "path_status": str(path_report.get("status", "")),
        "path_warnings": list(path_report.get("warnings", [])),
        "path_failures": list(path_report.get("failures", [])),
        "path_ratios": dict(path_report.get("ratios", {})),
    }


def choose_readout_candidate(
    candidates: list[dict[str, Any]],
    *,
    fallback_name: str = "uncalibrated_component_mixture",
) -> dict[str, Any]:
    """Choose the best readout that passes the fixed-start conditionality gate."""

    rows = [_candidate_row(candidate) for candidate in candidates]
    eligible = [
        row
        for row in rows
        if row["gate_status"] == "pass"
        and row["quality_deltas"]["coverage_80_delta"] >= 0.0
        and row["quality_deltas"]["crps_delta"] <= 0.0
        and row["quality_deltas"]["energy_delta"] <= 0.0
    ]
    rejected = [row for row in rows if row not in eligible]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["quality_score"]),
                float(row["quality_deltas"]["coverage_80_delta"]),
                -float(row["alpha"]),
            ),
        )
        recommendation = "use_selected_readout_candidate"
    else:
        selected = {
            "name": str(fallback_name),
            "alpha": 1.0,
            "gate_status": "fallback",
            "quality_deltas": {
                "coverage_80_delta": 0.0,
                "crps_delta": 0.0,
                "energy_delta": 0.0,
            },
            "quality_score": 0.0,
        }
        recommendation = "keep_uncalibrated_default"
    return {
        "status": "pass" if eligible else "warning",
        "selected": selected,
        "eligible": eligible,
        "rejected": rejected,
        "recommendation": recommendation,
        "interpretation": [
            "A readout candidate must improve held-out quality and pass fixed-start narrative conditionality gates.",
            "Candidates with better CRPS/energy are rejected if bootstrap or repeat controls become too close to observed narrative differences.",
            "If no readout passes, keep the uncalibrated component-preserving mixture as the product default.",
        ],
    }


def _parse_candidate(text: str) -> dict[str, Any]:
    try:
        name, rest = text.split("=", 1)
        calibration_path, path_audit_path = rest.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "candidate must have form name=calibration_report:path_audit_report"
        ) from exc
    return {
        "name": name,
        "calibration_report": _load_json(calibration_path),
        "path_report": _load_json(path_audit_path),
        "calibration_report_path": calibration_path,
        "path_report_path": path_audit_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--fallback-name", default="uncalibrated_component_mixture")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = choose_readout_candidate(
        [_parse_candidate(item) for item in args.candidate],
        fallback_name=str(args.fallback_name),
    )
    output_path = Path(args.output_dir) / "readout_gate_selection.json"
    report["artifact_paths"] = {"report": str(output_path)}
    _write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected": report["selected"]["name"],
                "recommendation": report["recommendation"],
                "report": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
