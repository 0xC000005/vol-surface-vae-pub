import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_live_story_deck_analysis import (
    build_live_story_deck_analysis,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _case_report(
    *,
    support_windows: list[str],
    support_weights: list[float],
    spx_delta: float,
    vix_delta: float,
) -> dict:
    return {
        "generation": {
            "narrative_ensemble_calibration": {
                "applied": True,
                "effective_beta": 0.25,
                "support_gate": 1.0,
                "active_direction_count": 2,
            },
            "terminal_delta_summary": [
                {"market": "SPX", "mean_terminal_delta": spx_delta},
                {"market": "VIX", "mean_terminal_delta": vix_delta},
            ],
        },
        "cached_query": {
            "memory_prior": {
                "candidate_details": [
                    {"window_id": window, "weight": weight}
                    for window, weight in zip(support_windows, support_weights)
                ]
            }
        },
    }


def test_live_story_deck_analysis_summarizes_support_and_calibration(
    tmp_path: Path,
) -> None:
    report_a = _write_json(
        tmp_path / "case_a" / "prefix_report_snapshot.json",
        _case_report(
            support_windows=["a", "b"],
            support_weights=[0.6, 0.4],
            spx_delta=10.0,
            vix_delta=-1.0,
        ),
    )
    report_b = _write_json(
        tmp_path / "case_b" / "prefix_report_snapshot.json",
        _case_report(
            support_windows=["c"],
            support_weights=[1.0],
            spx_delta=-5.0,
            vix_delta=2.0,
        ),
    )
    summary = {
        "status": "ok",
        "case_count": 2,
        "pass_count": 2,
        "fixed_start_index": 22,
        "total_openai_tokens": 100,
        "cases": [
            {
                "case_name": "case_a",
                "status": "ok",
                "start_index": 22,
                "prefix_report_snapshot_path": str(report_a),
                "prefix_arrays_snapshot_path": str(tmp_path / "case_a" / "arrays.npz"),
                "market_implications": [{"market": "SPX", "direction": "up"}],
            },
            {
                "case_name": "case_b",
                "status": "ok",
                "start_index": 22,
                "prefix_report_snapshot_path": str(report_b),
                "prefix_arrays_snapshot_path": str(tmp_path / "case_b" / "arrays.npz"),
                "market_implications": [{"market": "VIX", "direction": "up"}],
            },
        ],
    }
    summary_path = _write_json(tmp_path / "summary.json", summary)

    analysis = build_live_story_deck_analysis(summary_path, markets=["SPX", "VIX"])

    assert analysis["case_count"] == 2
    assert analysis["calibration_applied_count"] == 2
    assert analysis["min_calibration_support_gate"] == 1.0
    assert analysis["mean_pairwise_support_jaccard"] == 0.0
    assert analysis["max_pairwise_support_jaccard"] == 0.0
    assert analysis["cases"][0]["support_windows"] == ["a", "b"]
    assert analysis["cases"][0]["terminal_mean_deltas"]["SPX"] == 10.0
    assert analysis["cases"][1]["terminal_mean_deltas"]["VIX"] == 2.0


def test_live_story_deck_analysis_rejects_missing_snapshot(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "summary.json",
        {
            "status": "ok",
            "case_count": 1,
            "pass_count": 1,
            "cases": [
                {
                    "case_name": "case_a",
                    "prefix_report_snapshot_path": str(
                        tmp_path / "missing_report.json"
                    ),
                }
            ],
        },
    )

    with pytest.raises(FileNotFoundError):
        build_live_story_deck_analysis(summary_path)
