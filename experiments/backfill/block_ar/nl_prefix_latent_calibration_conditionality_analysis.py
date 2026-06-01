#!/usr/bin/env python
"""Compare uncalibrated and calibrated fixed-start conditionality audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTROL_NAMES = [
    "observed_narrative",
    "same_narrative_repeat",
    "within_run_bootstrap",
    "start_only_null",
]
METRIC_KEYS = [
    (
        "path_energy_distance_median_across_pairs",
        "path_energy",
        "Multivariate path energy distance in pair-standardized path space.",
    ),
    (
        "path_wasserstein_z_mean_median_median_across_pairs",
        "path_wasserstein",
        "Median horizon-wise path Wasserstein in pooled-width units.",
    ),
    (
        "path_std_log_ratio_mean_median_median_across_pairs",
        "path_variance",
        "Median horizon-wise path width log-ratio.",
    ),
    (
        "path_drawdown_prob_gap_1sigma_mean_median_across_pairs",
        "drawdown_probability",
        "Median 1-sigma path drawdown probability gap.",
    ),
]


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def summary_value(
    report: dict[str, Any], control: str, metric_key: str
) -> float | None:
    value = report.get("summaries", {}).get(control, {}).get(metric_key)
    return float(value) if value is not None else None


def path_status(report: dict[str, Any]) -> str:
    return str(
        report.get("path_distribution_status") or report.get("status", "unknown")
    )


def build_calibration_conditionality_report(
    *,
    base_report: dict[str, Any],
    calibrated_report: dict[str, Any],
    base_report_path: str,
    calibrated_report_path: str,
) -> dict[str, Any]:
    rows = []
    for metric_key, metric_name, description in METRIC_KEYS:
        for control in CONTROL_NAMES:
            base_value = summary_value(base_report, control, metric_key)
            calibrated_value = summary_value(calibrated_report, control, metric_key)
            rows.append(
                {
                    "metric": metric_name,
                    "description": description,
                    "control": control,
                    "base_value": base_value,
                    "calibrated_value": calibrated_value,
                    "calibrated_to_base_ratio": safe_ratio(
                        calibrated_value,
                        base_value,
                    ),
                }
            )

    observed_retention = {
        metric_name: safe_ratio(
            summary_value(calibrated_report, "observed_narrative", metric_key),
            summary_value(base_report, "observed_narrative", metric_key),
        )
        for metric_key, metric_name, _description in METRIC_KEYS
    }
    bootstrap_retention = {
        metric_name: safe_ratio(
            summary_value(calibrated_report, "within_run_bootstrap", metric_key),
            summary_value(base_report, "within_run_bootstrap", metric_key),
        )
        for metric_key, metric_name, _description in METRIC_KEYS
    }

    base_status = path_status(base_report)
    calibrated_status = path_status(calibrated_report)
    if base_status == "pass" and calibrated_status in {"warning", "fail"}:
        decision = "base_conditioning_passes_calibrated_display_warns"
    elif base_status == calibrated_status == "pass":
        decision = "base_and_calibrated_pass"
    elif base_status == "fail":
        decision = "base_conditioning_fails"
    else:
        decision = "mixed"

    return {
        "status": (
            "warning" if "warns" in decision or decision == "mixed" else base_status
        ),
        "decision": decision,
        "scope_note": (
            "Compares fixed-start path-distribution audits before and after "
            "global fan-width calibration. Calibration preserves each ensemble "
            "mean and widens sample deviations, so standardized signal-to-width "
            "metrics can fall even when the underlying conditioning mechanism "
            "passes uncalibrated."
        ),
        "base_report": str(base_report_path),
        "calibrated_report": str(calibrated_report_path),
        "base_status": base_status,
        "calibrated_status": calibrated_status,
        "base_warnings": base_report.get("warnings", []),
        "calibrated_warnings": calibrated_report.get("warnings", []),
        "observed_signal_retention": observed_retention,
        "bootstrap_signal_retention": bootstrap_retention,
        "metric_rows": rows,
        "interpretation": [
            "Use the base uncalibrated audit to judge whether the support mixture responds to narrative under a fixed start.",
            "Use the calibrated audit to judge whether the displayed fan remains visibly separated after uncertainty calibration.",
            "A calibrated warning with a base pass is a display/readout calibration issue, not by itself evidence that the narrative-conditioning mechanism ignores the story.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--calibrated-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_calibration_conditionality_report(
        base_report=load_json(args.base_report),
        calibrated_report=load_json(args.calibrated_report),
        base_report_path=str(args.base_report),
        calibrated_report_path=str(args.calibrated_report),
    )
    report["artifact_paths"] = {"report": str(args.output)}
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "output": str(args.output),
                "base_status": report["base_status"],
                "calibrated_status": report["calibrated_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
