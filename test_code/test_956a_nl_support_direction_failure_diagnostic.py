import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_support_direction_failure_diagnostic import (
    build_support_direction_failure_diagnostic,
    summarize_case_report,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _case_report(*, direction_status: str = "reject") -> dict:
    return {
        "cached_query": {
            "grounding": {
                "market_implications": [
                    {
                        "market": "SPX",
                        "direction": "down",
                        "confidence": "high",
                        "inferred": False,
                    },
                    {
                        "market": "VIX",
                        "direction": "up",
                        "confidence": "high",
                        "inferred": False,
                    },
                ]
            },
            "memory_prior": {
                "mode": "diverse_topk_narrative_start_checked",
                "direction_check": {
                    "status": direction_status,
                    "reason": "final_mixed_prefix_direction_mismatch",
                    "support_weighted_match_rate": 0.5,
                    "support_weighted_mismatch_rate": 0.5,
                    "min_support_match_rate": 0.6,
                    "final_mixture_checked_count": 2,
                    "final_mixture_mismatch_count": 1,
                    "final_mixture_alignment": {
                        "checked": [
                            {
                                "market": "SPX",
                                "direction": "down",
                                "aligned": False,
                                "mean_terminal_delta": 0.4,
                            },
                            {
                                "market": "VIX",
                                "direction": "up",
                                "aligned": True,
                                "mean_terminal_delta": 1.2,
                            },
                        ]
                    },
                },
                "candidate_details": [
                    {
                        "window_index": 10,
                        "window_id": "support_good",
                        "weight": 0.4,
                        "memory_support_cosine": 0.91,
                        "start_distance_z": 8.0,
                        "recent_prefix_match_count": 2,
                        "recent_prefix_mismatches": 0,
                        "recent_prefix_checked": 2,
                        "recent_prefix_alignment": {
                            "checked": [
                                {
                                    "market": "SPX",
                                    "direction": "down",
                                    "aligned": True,
                                    "mean_terminal_delta": -0.5,
                                },
                                {
                                    "market": "VIX",
                                    "direction": "up",
                                    "aligned": True,
                                    "mean_terminal_delta": 1.0,
                                },
                            ]
                        },
                    },
                    {
                        "window_index": 20,
                        "window_id": "support_start_close",
                        "weight": 0.6,
                        "memory_support_cosine": 0.80,
                        "start_distance_z": 0.0,
                        "recent_prefix_match_count": 1,
                        "recent_prefix_mismatches": 1,
                        "recent_prefix_checked": 2,
                        "recent_prefix_alignment": {
                            "checked": [
                                {
                                    "market": "SPX",
                                    "direction": "down",
                                    "aligned": False,
                                    "mean_terminal_delta": 0.8,
                                },
                                {
                                    "market": "VIX",
                                    "direction": "up",
                                    "aligned": True,
                                    "mean_terminal_delta": 0.6,
                                },
                            ]
                        },
                    },
                ],
            },
        },
        "variant_rows": [
            {
                "variant": "explicit_start_window",
                "is_operational": True,
                "memory_prior_mode": "diverse_topk_narrative_start_checked",
                "memory_prior_direction_status": direction_status,
                "memory_prior_direction_reason": "final_mixed_prefix_direction_mismatch",
                "start_window_index": 22,
            }
        ],
    }


def test_summarize_case_report_explains_direction_rejection() -> None:
    summary = summarize_case_report(
        "rates_selloff_22",
        _case_report(direction_status="reject"),
        smoke_case={"narrative_calibration_support_gate": 0.0},
    )

    assert summary["direction_status"] == "reject"
    assert summary["support_gate"] == 0.0
    assert summary["failure_reasons"] == [
        "final_mixed_prefix_direction_mismatch",
        "support_weighted_match_rate_below_min",
        "high_weight_on_direction_mismatched_support",
    ]
    assert summary["per_market_support"]["SPX"]["support_match_rate"] == pytest.approx(
        0.4
    )
    assert summary["per_market_support"]["SPX"]["final_aligned"] is False
    assert summary["per_market_support"]["VIX"]["support_match_rate"] == pytest.approx(
        1.0
    )
    assert summary["candidates"][0]["direction_match_rate"] == pytest.approx(1.0)


def test_summarize_case_report_does_not_call_pass_reason_a_failure() -> None:
    summary = summarize_case_report(
        "risk_on_22",
        _case_report(direction_status="pass"),
        smoke_case={"narrative_calibration_support_gate": 1.0},
    )

    assert summary["direction_status"] == "pass"
    assert summary["failure_reasons"] == []


def test_build_diagnostic_loads_casebook_summary_and_counts_rejections(
    tmp_path: Path,
) -> None:
    report_path = _write_json(tmp_path / "case" / "prefix_report.json", _case_report())
    summary_path = _write_json(
        tmp_path / "casebook.json",
        {
            "fixed_start_index": 22,
            "cases": [
                {
                    "case_name": "rates_selloff_22",
                    "status": "fail",
                    "narrative_calibration_support_gate": 0.0,
                    "prefix_report_snapshot_path": str(report_path),
                }
            ],
        },
    )

    result = build_support_direction_failure_diagnostic(summary_path)

    assert result["case_count"] == 1
    assert result["rejected_case_count"] == 1
    assert result["direction_status_counts"] == {"reject": 1}
    assert result["market_failure_counts"] == {"SPX": 1}
