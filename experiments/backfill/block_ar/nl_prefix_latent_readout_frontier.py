#!/usr/bin/env python
"""Summarize the calibration versus conditionality readout frontier.

This is a post-experiment analysis tool. It does not generate new scenarios and
does not promote a new default. It compares fixed-start conditionality audits
before and after readout calibration to decide whether a calibration layer keeps
enough narrative signal visible to be a production candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_LABELS = {
    "path_energy": "full-path energy distance",
    "path_wasserstein": "path Wasserstein",
    "path_variance": "path width ratio",
    "drawdown_probability": "drawdown probability gap",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Prefix-Latent Readout Frontier",
        "",
        report["scope_note"],
        "",
        "## Decision",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Best candidate: `{report['best_candidate']['label']}`",
        f"- Best candidate status: `{report['best_candidate']['candidate_status']}`",
        "",
        "## Candidate Frontier",
        "",
        (
            "| Candidate | Status | Observed energy retention | Observed "
            "Wasserstein retention | Bootstrap/observed energy | Decision |"
        ),
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["candidate_rows"]:
        lines.append(
            "| {label} | {candidate_status} | {energy} | {wasserstein} | "
            "{bootstrap} | {decision} |".format(
                label=row["label"],
                candidate_status=row["candidate_status"],
                energy=_fmt(row["observed_signal_retention"].get("path_energy")),
                wasserstein=_fmt(
                    row["observed_signal_retention"].get("path_wasserstein")
                ),
                bootstrap=_fmt(row.get("bootstrap_to_observed_path_energy")),
                decision=row["candidate_decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *[f"- {item}" for item in report["interpretation"]],
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _ratio_from_audit(
    audit_report: dict[str, Any],
    key: str,
) -> float | None:
    ratios = audit_report.get("ratios", {})
    value = ratios.get(key)
    return None if value is None else float(value)


def _candidate_from_calibration_report(
    *,
    label: str,
    report: dict[str, Any],
    path: str,
    min_energy_retention: float,
    min_wasserstein_retention: float,
    max_bootstrap_to_observed: float,
) -> dict[str, Any]:
    observed = {
        key: (None if value is None else float(value))
        for key, value in report.get("observed_signal_retention", {}).items()
    }
    bootstrap = {
        key: (None if value is None else float(value))
        for key, value in report.get("bootstrap_signal_retention", {}).items()
    }
    calibrated_path = report.get("calibrated_report")
    calibrated_audit: dict[str, Any] = {}
    if calibrated_path:
        maybe_path = Path(str(calibrated_path))
        if maybe_path.exists():
            calibrated_audit = _load_json(maybe_path)
    bootstrap_to_observed_energy = _ratio_from_audit(
        calibrated_audit,
        "bootstrap_to_observed_path_energy",
    )
    bootstrap_to_observed_wasserstein = _ratio_from_audit(
        calibrated_audit,
        "bootstrap_to_observed_path_wasserstein",
    )
    warnings: list[str] = []
    failures: list[str] = []
    if (observed.get("path_energy") or 0.0) < float(min_energy_retention):
        failures.append("low_observed_path_energy_retention")
    if (observed.get("path_wasserstein") or 0.0) < float(min_wasserstein_retention):
        warnings.append("low_observed_path_wasserstein_retention")
    if (
        bootstrap_to_observed_energy is not None
        and bootstrap_to_observed_energy > float(max_bootstrap_to_observed)
    ):
        warnings.append("bootstrap_energy_too_close_to_observed")
    if (
        bootstrap_to_observed_wasserstein is not None
        and bootstrap_to_observed_wasserstein > float(max_bootstrap_to_observed)
    ):
        warnings.append("bootstrap_wasserstein_too_close_to_observed")
    candidate_decision = (
        "reject_readout_candidate"
        if failures
        else (
            "warn_readout_candidate"
            if warnings or str(report.get("status")) != "pass"
            else "candidate_viable"
        )
    )
    return {
        "label": str(label),
        "report": str(path),
        "base_report": str(report.get("base_report", "")),
        "calibrated_report": str(report.get("calibrated_report", "")),
        "candidate_status": str(report.get("status", "unknown")),
        "calibrated_status": str(report.get("calibrated_status", "unknown")),
        "observed_signal_retention": observed,
        "bootstrap_signal_retention": bootstrap,
        "bootstrap_to_observed_path_energy": bootstrap_to_observed_energy,
        "bootstrap_to_observed_path_wasserstein": bootstrap_to_observed_wasserstein,
        "warnings": warnings,
        "failures": failures,
        "candidate_decision": candidate_decision,
    }


def build_readout_frontier_report(
    *,
    calibration_reports: list[tuple[str, str]],
    min_energy_retention: float = 0.50,
    min_wasserstein_retention: float = 0.60,
    max_bootstrap_to_observed: float = 0.75,
) -> dict[str, Any]:
    if not calibration_reports:
        raise ValueError("at least one calibration report is required")
    candidate_rows = [
        _candidate_from_calibration_report(
            label=label,
            report=_load_json(path),
            path=path,
            min_energy_retention=float(min_energy_retention),
            min_wasserstein_retention=float(min_wasserstein_retention),
            max_bootstrap_to_observed=float(max_bootstrap_to_observed),
        )
        for label, path in calibration_reports
    ]
    viable = [
        row for row in candidate_rows if row["candidate_decision"] == "candidate_viable"
    ]
    best = max(
        candidate_rows,
        key=lambda row: (
            1 if row["candidate_decision"] == "candidate_viable" else 0,
            1 if row["candidate_decision"] == "warn_readout_candidate" else 0,
            float(row["observed_signal_retention"].get("path_energy") or 0.0),
            float(row["observed_signal_retention"].get("path_wasserstein") or 0.0),
        ),
    )
    status = "pass" if viable else "warning"
    decision = (
        "global_readout_candidate_viable"
        if viable
        else "global_readout_not_sufficient_continue_ecc_style_candidate"
    )
    return {
        "status": status,
        "decision": decision,
        "research_lane": "post_experiment_analysis",
        "result_status": "mechanism_found",
        "benchmark_floor_status": "not_applicable",
        "scope_note": (
            "Readout-frontier diagnostic over existing fixed-start calibration "
            "conditionality reports. The tool checks whether fan/readout "
            "calibration preserves enough narrative-conditioned path signal to "
            "be treated as a viable production readout candidate."
        ),
        "thresholds": {
            "min_energy_retention": float(min_energy_retention),
            "min_wasserstein_retention": float(min_wasserstein_retention),
            "max_bootstrap_to_observed": float(max_bootstrap_to_observed),
        },
        "candidate_count": int(len(candidate_rows)),
        "viable_candidate_count": int(len(viable)),
        "best_candidate": best,
        "candidate_rows": candidate_rows,
        "interpretation": [
            "Observed-signal retention measures how much fixed-start narrative separation survives calibration.",
            "Bootstrap-to-observed ratios measure whether display noise is close to the narrative effect.",
            "A rejected global readout candidate does not reject the narrative mixture; it says the readout/display layer needs a more conditionality-aware calibration.",
            "The next candidate should preserve path ranks/dependence while calibrating marginal spread, instead of only widening every fan around its own mean.",
        ],
    }


def _parse_report_arg(value: str) -> tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), path.strip()
    path = value.strip()
    return Path(path).parent.name, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-report",
        action="append",
        default=[],
        help="Calibration report path, optionally label=path. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-energy-retention", type=float, default=0.50)
    parser.add_argument("--min-wasserstein-retention", type=float, default=0.60)
    parser.add_argument("--max-bootstrap-to-observed", type=float, default=0.75)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = build_readout_frontier_report(
        calibration_reports=[
            _parse_report_arg(item) for item in args.calibration_report
        ],
        min_energy_retention=float(args.min_energy_retention),
        min_wasserstein_retention=float(args.min_wasserstein_retention),
        max_bootstrap_to_observed=float(args.max_bootstrap_to_observed),
    )
    report["artifact_paths"] = {
        "report": str(output_dir / "readout_frontier_report.json"),
        "markdown": str(output_dir / "readout_frontier_report.md"),
    }
    _write_json(report["artifact_paths"]["report"], report)
    _write_markdown(report["artifact_paths"]["markdown"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "best_candidate": report["best_candidate"]["label"],
                "viable_candidate_count": report["viable_candidate_count"],
                "report": report["artifact_paths"]["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
