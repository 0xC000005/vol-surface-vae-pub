import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_grounding_reliability_audit import (
    build_grounding_reliability_audit,
    build_grounding_reliability_audit_from_prefix_reports,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _grounding() -> dict:
    return {
        "market_implications": [
            {
                "market": "SPX",
                "direction": "up",
                "confidence": "high",
                "horizon": "current_state",
                "evidence": ["equities are recovering"],
            },
            {
                "market": "VIX",
                "direction": "down",
                "confidence": "medium",
                "horizon": "current_state",
                "evidence": ["volatility is compressing"],
            },
        ],
        "non_conditioning_forward_language": [
            {
                "phrase": "the rebound could unwind if volatility returns",
                "handling": "warning_only",
            }
        ],
    }


def test_grounding_reliability_audit_scores_faithfulness_future_filter_and_history():
    audit = build_grounding_reliability_audit(
        [
            {
                "case_name": "fragile_risk_on",
                "story": (
                    "Equities are recovering and volatility is compressing. "
                    "The rebound could unwind if volatility returns."
                ),
                "grounding": _grounding(),
                "validation": {
                    "forward_warning_leakage_count": 0,
                    "condition_role_error_count": 0,
                },
                "expected_implications": [
                    {"market": "SPX", "direction": "up", "confidence": "high"},
                    {"market": "VIX", "direction": "down", "confidence": "high"},
                ],
                "expected_forward_language": True,
                "support_direction_check": {
                    "status": "pass",
                    "support_weighted_match_rate": 1.0,
                    "final_mixture_checked_count": 2,
                    "final_mixture_mismatch_count": 0,
                },
            }
        ]
    )

    summary = audit["summary"]
    assert summary["case_count"] == 1
    assert summary["claim_count"] == 2
    assert summary["claim_faithfulness_rate"] == pytest.approx(1.0)
    assert summary["future_language_detection_rate"] == pytest.approx(1.0)
    assert summary["future_language_leakage_rate"] == pytest.approx(0.0)
    assert summary["historical_direction_agreement_rate"] == pytest.approx(1.0)
    assert summary["historical_direction_coverage_rate"] == pytest.approx(1.0)
    assert summary["support_direction_pass_rate"] == pytest.approx(1.0)
    assert audit["cases"][0]["historical_direction"]["checked_count"] == 2


def test_grounding_reliability_audit_flags_unsupported_and_leaked_claims():
    grounding = _grounding()
    grounding["market_implications"][0]["evidence"] = ["unsupported equity claim"]

    audit = build_grounding_reliability_audit(
        [
            {
                "case_name": "bad_case",
                "story": "Volatility is compressing. Equities might rally next month.",
                "grounding": grounding,
                "validation": {"forward_warning_leakage_count": 1},
                "expected_forward_language": True,
            }
        ]
    )

    case = audit["cases"][0]
    assert case["claim_faithfulness"]["unsupported_claim_count"] == 1
    assert case["future_language"]["leakage_count"] == 1
    assert audit["summary"]["claim_faithfulness_rate"] == pytest.approx(0.5)
    assert audit["summary"]["future_language_leakage_rate"] == pytest.approx(1.0)


def test_grounding_reliability_audit_loads_prefix_report_support_checks(tmp_path):
    report_path = _write_json(
        tmp_path / "prefix_report.json",
        {
            "condition_only_case": {
                "story": "Equities are recovering and volatility is compressing.",
                "condition_only_validation": {
                    "forward_warning_leakage_count": 0,
                    "condition_role_error_count": 0,
                },
            },
            "cached_query": {
                "grounding": _grounding(),
                "memory_prior": {
                    "direction_check": {
                        "status": "pass",
                        "support_weighted_match_rate": 1.0,
                        "final_mixture_checked_count": 2,
                        "final_mixture_mismatch_count": 0,
                    }
                },
            },
        },
    )

    audit = build_grounding_reliability_audit_from_prefix_reports([report_path])

    assert audit["summary"]["case_count"] == 1
    assert audit["summary"]["support_direction_pass_rate"] == pytest.approx(1.0)
    assert audit["cases"][0]["source_path"] == str(report_path)

