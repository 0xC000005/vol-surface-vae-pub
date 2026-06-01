import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_production_readiness_audit import (
    build_production_readiness_audit,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _calibration_report(tmp_path: Path) -> dict:
    fan = tmp_path / "fans.png"
    raw = tmp_path / "raw.png"
    null = tmp_path / "null.png"
    for path in [fan, raw, null]:
        path.write_bytes(b"png")
    return {
        "status": "ok",
        "calibration_row_count": 33,
        "evaluation_row_count": 33,
        "heldout_quality": {
            "comparison": {
                "calibrated_minus_identity_crps": -0.001,
                "calibrated_minus_identity_energy": -0.002,
                "calibrated_minus_identity_coverage": 0.001,
            },
            "evaluation_summary": {
                "narrative_calibrated": {
                    "ensemble_crps_z_improvement_vs_persistence": 0.21,
                    "energy_score_z_improvement_vs_persistence": 0.30,
                    "coverage_80_mean": 0.82,
                    "window_count": 33,
                }
            },
        },
        "promotion_gates": {
            "coverage_not_down_more_than_0p03": True,
            "crps_regression_within_2pct": True,
            "energy_regression_within_2pct": True,
            "per_start_factor_ks_min_ge_0p20": True,
            "per_start_portfolio_ks_min_ge_0p20": True,
            "per_start_support_jaccard_max_le_0p25": True,
            "selected_beta_nonzero": True,
            "start_normalized_narrative_share_improved": True,
        },
        "fixed_start_attribution": {
            "baseline_results": [
                {
                    "feature_space": "start_normalized",
                    "narrative_share": 0.18,
                    "interaction_share": 0.06,
                }
            ],
            "calibrated_results": [
                {
                    "feature_space": "start_normalized",
                    "narrative_share": 0.47,
                    "interaction_share": 0.05,
                    "mean_factor_ks_same_start_narrative": 0.33,
                    "mean_portfolio_ks_same_start_narrative": 0.41,
                    "mean_support_jaccard_same_start_narrative": 0.03,
                }
            ],
        },
        "artifact_paths": {
            "fixed_start_fans": str(fan),
            "narrative_relevant_raw_panels": str(raw),
            "start_only_null_contrasts": str(null),
        },
    }


def _live_summary(tmp_path: Path) -> dict:
    case_report = tmp_path / "case_a.json"
    case_arrays = tmp_path / "case_a.npz"
    case_report.write_text("{}", encoding="utf-8")
    case_arrays.write_bytes(b"npz")
    return {
        "status": "ok",
        "case_count": 6,
        "pass_count": 6,
        "fixed_start_index": 22,
        "calibration_applied_count": 6,
        "min_calibration_support_gate": 1.0,
        "mean_pairwise_support_jaccard": 0.0,
        "max_pairwise_support_jaccard": 0.0,
        "cases": [
            {
                "case_name": "case_a",
                "calibration_applied": True,
                "support_gate": 1.0,
                "support_count": 2,
                "support_windows": ["a", "b"],
                "terminal_mean_deltas": {"SPX": 1.0},
                "report_snapshot": str(case_report),
                "arrays_snapshot": str(case_arrays),
            }
            for _ in range(6)
        ],
    }


def test_production_readiness_audit_combines_947b_and_948d(tmp_path: Path) -> None:
    calibration_path = _write_json(
        tmp_path / "947b.json", _calibration_report(tmp_path)
    )
    live_path = _write_json(tmp_path / "948d.json", _live_summary(tmp_path))

    audit = build_production_readiness_audit(
        calibration_path=calibration_path,
        live_story_deck_path=live_path,
    )

    assert audit["overall_status"] == "paper_demo_candidate"
    assert audit["goal_complete"] is False
    assert audit["gates"]["heldout_quality"]["status"] == "pass"
    assert audit["gates"]["start_normalized_response"]["status"] == "pass"
    assert audit["gates"]["fixed_start_distribution"]["status"] == "pass"
    assert audit["gates"]["live_demo_support_conditionality"]["status"] == "pass"
    assert audit["gates"]["production_default"]["status"] == "warning"
    assert audit["headline"]["calibrated_crps_improvement_vs_persistence"] == 0.21
    assert audit["headline"]["live_fixed_start_case_count"] == 6


def test_production_readiness_audit_flags_missing_live_calibration(
    tmp_path: Path,
) -> None:
    live = _live_summary(tmp_path)
    live["calibration_applied_count"] = 5
    calibration_path = _write_json(
        tmp_path / "947b.json", _calibration_report(tmp_path)
    )
    live_path = _write_json(tmp_path / "948d.json", live)

    audit = build_production_readiness_audit(
        calibration_path=calibration_path,
        live_story_deck_path=live_path,
    )

    assert audit["overall_status"] == "needs_work"
    assert audit["gates"]["live_demo_support_conditionality"]["status"] == "fail"


def test_production_readiness_audit_flags_missing_qualitative_artifact(
    tmp_path: Path,
) -> None:
    calibration = _calibration_report(tmp_path)
    calibration["artifact_paths"]["fixed_start_fans"] = str(tmp_path / "missing.png")
    calibration_path = _write_json(tmp_path / "947b.json", calibration)
    live_path = _write_json(tmp_path / "948d.json", _live_summary(tmp_path))

    audit = build_production_readiness_audit(
        calibration_path=calibration_path,
        live_story_deck_path=live_path,
    )

    assert audit["overall_status"] == "needs_work"
    assert audit["gates"]["qualitative_evidence"]["status"] == "fail"
    assert (
        "missing.png"
        in audit["gates"]["qualitative_evidence"]["missing_artifact_paths"][0]
    )
