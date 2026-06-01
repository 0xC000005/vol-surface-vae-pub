#!/usr/bin/env python
"""Build a gate-level readiness audit for the NL scenario generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CALIBRATION_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_narrative_ensemble_calibration_947b_full906b_66w_5start_"
    "broad_support_deck_matched_seed/narrative_ensemble_calibration_report.json"
)
DEFAULT_LIVE_STORY_DECK_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/"
    "fixed_start22_calibrated_story_deck_conditionality_summary.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _status(condition: bool) -> str:
    return "pass" if bool(condition) else "fail"


def _start_normalized_rows(
    report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attribution = _as_dict(report.get("fixed_start_attribution"))
    baseline = {}
    calibrated = {}
    for row in _as_list(attribution.get("baseline_results")):
        if isinstance(row, dict) and row.get("feature_space") == "start_normalized":
            baseline = row
            break
    for row in _as_list(attribution.get("calibrated_results")):
        if isinstance(row, dict) and row.get("feature_space") == "start_normalized":
            calibrated = row
            break
    return baseline, calibrated


def _heldout_quality_gate(report: dict[str, Any]) -> dict[str, Any]:
    heldout = _as_dict(report.get("heldout_quality"))
    comparison = _as_dict(heldout.get("comparison"))
    evaluation = _as_dict(
        _as_dict(heldout.get("evaluation_summary")).get("narrative_calibrated")
    )
    crps_delta = _float(comparison.get("calibrated_minus_identity_crps"))
    energy_delta = _float(comparison.get("calibrated_minus_identity_energy"))
    coverage_delta = _float(comparison.get("calibrated_minus_identity_coverage"))
    crps_improvement = _float(
        evaluation.get("ensemble_crps_z_improvement_vs_persistence")
    )
    energy_improvement = _float(
        evaluation.get("energy_score_z_improvement_vs_persistence")
    )
    coverage = _float(evaluation.get("coverage_80_mean"))
    status = _status(
        crps_delta <= 0.0
        and energy_delta <= 0.0
        and coverage_delta >= -0.03
        and crps_improvement > 0.0
        and energy_improvement > 0.0
    )
    return {
        "status": status,
        "calibrated_minus_identity_crps": crps_delta,
        "calibrated_minus_identity_energy": energy_delta,
        "calibrated_minus_identity_coverage": coverage_delta,
        "calibrated_crps_improvement_vs_persistence": crps_improvement,
        "calibrated_energy_improvement_vs_persistence": energy_improvement,
        "calibrated_coverage_80": coverage,
        "window_count": int(evaluation.get("window_count", 0) or 0),
    }


def _promotion_gate(report: dict[str, Any]) -> dict[str, Any]:
    gates = _as_dict(report.get("promotion_gates"))
    failed = [key for key, value in gates.items() if not bool(value)]
    return {
        "status": _status(bool(gates) and not failed),
        "failed": failed,
        "gate_count": len(gates),
    }


def _start_normalized_response_gate(report: dict[str, Any]) -> dict[str, Any]:
    baseline, calibrated = _start_normalized_rows(report)
    baseline_share = _float(baseline.get("narrative_share")) + _float(
        baseline.get("interaction_share")
    )
    calibrated_share = _float(calibrated.get("narrative_share")) + _float(
        calibrated.get("interaction_share")
    )
    return {
        "status": _status(
            calibrated_share > baseline_share and calibrated_share >= 0.5
        ),
        "baseline_narrative_plus_interaction": baseline_share,
        "calibrated_narrative_plus_interaction": calibrated_share,
        "improvement": calibrated_share - baseline_share,
    }


def _fixed_start_distribution_gate(report: dict[str, Any]) -> dict[str, Any]:
    _baseline, calibrated = _start_normalized_rows(report)
    factor_ks = _float(calibrated.get("mean_factor_ks_same_start_narrative"))
    portfolio_ks = _float(calibrated.get("mean_portfolio_ks_same_start_narrative"))
    support_jaccard = _float(
        calibrated.get("mean_support_jaccard_same_start_narrative")
    )
    return {
        "status": _status(
            factor_ks >= 0.30 and portfolio_ks >= 0.35 and support_jaccard <= 0.05
        ),
        "mean_factor_ks_same_start_narrative": factor_ks,
        "mean_portfolio_ks_same_start_narrative": portfolio_ks,
        "mean_support_jaccard_same_start_narrative": support_jaccard,
    }


def _existing_path(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and Path(text).exists()


def _qualitative_evidence_gate(
    report: dict[str, Any], live: dict[str, Any]
) -> dict[str, Any]:
    artifact_paths = _as_dict(report.get("artifact_paths"))
    expected = [
        artifact_paths.get("fixed_start_fans"),
        artifact_paths.get("narrative_relevant_raw_panels"),
        artifact_paths.get("start_only_null_contrasts"),
    ]
    live_cases = _as_list(live.get("cases"))
    snapshot_missing = [
        str(case.get("case_name", ""))
        for case in live_cases
        if isinstance(case, dict)
        and (
            not _existing_path(case.get("report_snapshot"))
            or not _existing_path(case.get("arrays_snapshot"))
        )
    ]
    missing_paths = [str(path) for path in expected if not _existing_path(path)]
    return {
        "status": _status(not missing_paths and not snapshot_missing),
        "missing_artifact_paths": missing_paths,
        "missing_live_snapshots": snapshot_missing,
    }


def _live_demo_support_gate(live: dict[str, Any]) -> dict[str, Any]:
    case_count = int(live.get("case_count", 0) or 0)
    pass_count = int(live.get("pass_count", 0) or 0)
    applied = int(live.get("calibration_applied_count", 0) or 0)
    min_gate = _float(live.get("min_calibration_support_gate"))
    max_jaccard = _float(live.get("max_pairwise_support_jaccard"), default=1.0)
    status = _status(
        str(live.get("status", "")) == "ok"
        and case_count >= 6
        and pass_count == case_count
        and applied == case_count
        and min_gate > 0.0
        and max_jaccard <= 0.25
    )
    return {
        "status": status,
        "case_count": case_count,
        "pass_count": pass_count,
        "calibration_applied_count": applied,
        "min_calibration_support_gate": min_gate,
        "max_pairwise_support_jaccard": max_jaccard,
        "fixed_start_index": live.get("fixed_start_index"),
        "total_openai_tokens": int(live.get("total_openai_tokens", 0) or 0),
    }


def _production_default_gate() -> dict[str, Any]:
    return {
        "status": "warning",
        "reason": (
            "Current evidence supports a paper/demo candidate. It remains a "
            "bounded post-rollout support-gated calibration layer, not a silent "
            "production default or direct prompt-to-scenario generator."
        ),
        "required_next_evidence": [
            "broader multi-start live UX sweep",
            "paper/demo figure synchronization with 947b and 948d artifacts",
            "risk-manager-facing visual acceptance of factor/path/portfolio readouts",
        ],
    }


def build_production_readiness_audit(
    *,
    calibration_path: str | Path = DEFAULT_CALIBRATION_REPORT,
    live_story_deck_path: str | Path = DEFAULT_LIVE_STORY_DECK_REPORT,
) -> dict[str, Any]:
    """Combine held-out calibration and live demo evidence into one audit."""

    calibration = _load_json(calibration_path)
    live = _load_json(live_story_deck_path)
    gates = {
        "heldout_quality": _heldout_quality_gate(calibration),
        "promotion_gates": _promotion_gate(calibration),
        "start_normalized_response": _start_normalized_response_gate(calibration),
        "fixed_start_distribution": _fixed_start_distribution_gate(calibration),
        "live_demo_support_conditionality": _live_demo_support_gate(live),
        "qualitative_evidence": _qualitative_evidence_gate(calibration, live),
        "production_default": _production_default_gate(),
    }
    hard_gates = [name for name in gates if name not in {"production_default"}]
    failed = [name for name in hard_gates if gates[name]["status"] != "pass"]
    overall = "paper_demo_candidate" if not failed else "needs_work"
    heldout = gates["heldout_quality"]
    live_gate = gates["live_demo_support_conditionality"]
    return {
        "status": "ok",
        "overall_status": overall,
        "goal_complete": False,
        "calibration_report": str(calibration_path),
        "live_story_deck_report": str(live_story_deck_path),
        "headline": {
            "calibrated_crps_improvement_vs_persistence": heldout[
                "calibrated_crps_improvement_vs_persistence"
            ],
            "calibrated_energy_improvement_vs_persistence": heldout[
                "calibrated_energy_improvement_vs_persistence"
            ],
            "calibrated_coverage_80": heldout["calibrated_coverage_80"],
            "start_normalized_narrative_plus_interaction": gates[
                "start_normalized_response"
            ]["calibrated_narrative_plus_interaction"],
            "fixed_start_factor_ks": gates["fixed_start_distribution"][
                "mean_factor_ks_same_start_narrative"
            ],
            "fixed_start_portfolio_ks": gates["fixed_start_distribution"][
                "mean_portfolio_ks_same_start_narrative"
            ],
            "live_fixed_start_case_count": live_gate["case_count"],
            "live_max_support_jaccard": live_gate["max_pairwise_support_jaccard"],
        },
        "failed_hard_gates": failed,
        "gates": gates,
        "interpretation": (
            "The combined evidence supports a paper/demo candidate for "
            "support-grounded narrative-conditioned scenario generation. The "
            "active goal should remain open because production readiness still "
            "requires synchronized paper/demo surfaces and broader UX/product "
            "validation beyond these quantitative and live-path gates."
        ),
    }


def _write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        "# NL Production Readiness Audit",
        "",
        f"- Overall status: `{audit['overall_status']}`",
        f"- Goal complete: `{audit['goal_complete']}`",
        f"- Calibration report: `{audit['calibration_report']}`",
        f"- Live story deck report: `{audit['live_story_deck_report']}`",
        "",
        "## Headline",
        "",
    ]
    for key, value in audit["headline"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Gates", ""])
    for name, gate in audit["gates"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Status: `{gate['status']}`")
        for key, value in gate.items():
            if key == "status":
                continue
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", default=DEFAULT_CALIBRATION_REPORT)
    parser.add_argument(
        "--live-story-deck-report", default=DEFAULT_LIVE_STORY_DECK_REPORT
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "nl_production_readiness_audit_949a/"
            "production_readiness_audit.json"
        ),
    )
    args = parser.parse_args()
    audit = build_production_readiness_audit(
        calibration_path=args.calibration_report,
        live_story_deck_path=args.live_story_deck_report,
    )
    output = Path(args.output)
    _write_json(output, audit)
    markdown = output.with_suffix(".md")
    _write_markdown(markdown, audit)
    print(
        json.dumps(
            {
                "audit": str(output),
                "markdown": str(markdown),
                "overall_status": audit["overall_status"],
                "goal_complete": audit["goal_complete"],
                "failed_hard_gates": audit["failed_hard_gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
