import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_level_conditionality_audit import (
    build_prefix_level_conditionality_audit,
    direction_expected_sign,
    prefix_direction_alignment,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_report(
    path: Path,
    *,
    implications: list[dict],
    candidates: list[dict],
) -> Path:
    return _write_json(
        path,
        {
            "cached_query": {
                "grounding": {"market_implications": implications},
                "memory_prior": {"candidate_details": candidates},
            }
        },
    )


def test_direction_expected_sign_handles_common_market_words() -> None:
    assert direction_expected_sign("up") == 1
    assert direction_expected_sign("wider") == 1
    assert direction_expected_sign("down") == -1
    assert direction_expected_sign("tighter") == -1
    assert direction_expected_sign("stable") == 0
    assert direction_expected_sign("ambiguous") is None


def test_prefix_direction_alignment_scores_recent_prefix_only() -> None:
    prefix = np.asarray(
        [
            [100.0, 20.0, 2.5],
            [102.0, 18.0, 2.3],
            [105.0, 15.0, 2.0],
        ],
        dtype=np.float32,
    )
    implications = [
        {"market": "SPX", "direction": "up"},
        {"market": "VIX", "direction": "down"},
        {"market": "BBB_OAS", "direction": "tighter"},
    ]

    result = prefix_direction_alignment(
        prefix,
        implications=implications,
        market_to_index={"SPX": 0, "VIX": 1, "BBB_OAS": 2},
    )

    assert result["checked_count"] == 3
    assert result["match_count"] == 3
    assert result["mismatch_count"] == 0
    assert result["match_rate"] == 1.0


def test_prefix_level_audit_proves_support_prefixes_differ_before_rollout(
    tmp_path: Path,
) -> None:
    history_raw = np.asarray(
        [
            [
                [100.0, 20.0, 2.5],
                [102.0, 18.0, 2.3],
                [105.0, 15.0, 2.0],
            ],
            [
                [100.0, 15.0, 2.0],
                [98.0, 18.0, 2.4],
                [95.0, 23.0, 3.0],
            ],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"window_id": "risk_on_support", "calendar_end_date": "2020-01-03"},
        {"window_id": "risk_off_support", "calendar_end_date": "2020-02-03"},
    ]
    support_bank_path = tmp_path / "support_bank.npz"
    np.savez(
        support_bank_path,
        history_raw=history_raw,
        metadata=np.asarray(metadata, dtype=object),
    )

    risk_on_report = _write_report(
        tmp_path / "risk_on_report.json",
        implications=[
            {"market": "SPX", "direction": "up"},
            {"market": "VIX", "direction": "down"},
            {"market": "BBB_OAS", "direction": "tighter"},
        ],
        candidates=[
            {"bridge_local_index": 0, "weight": 1.0, "window_id": "risk_on_support"}
        ],
    )
    risk_off_report = _write_report(
        tmp_path / "risk_off_report.json",
        implications=[
            {"market": "SPX", "direction": "down"},
            {"market": "VIX", "direction": "up"},
            {"market": "BBB_OAS", "direction": "wider"},
        ],
        candidates=[
            {"bridge_local_index": 1, "weight": 1.0, "window_id": "risk_off_support"}
        ],
    )
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "fixed_start_index": 22,
            "cases": [
                {
                    "case_name": "risk_on_22",
                    "prefix_report_snapshot_path": str(risk_on_report),
                },
                {
                    "case_name": "risk_off_22",
                    "prefix_report_snapshot_path": str(risk_off_report),
                },
            ],
        },
    )

    report = build_prefix_level_conditionality_audit(
        summary,
        support_bank_npz=support_bank_path,
        markets=[("SPX", 0), ("VIX", 1), ("BBB_OAS", 2)],
    )

    assert report["case_count"] == 2
    assert report["support_jaccard"]["max"] == 0.0
    assert report["cases"][0]["weighted_alignment"]["match_rate"] == 1.0
    assert report["cases"][1]["weighted_alignment"]["match_rate"] == 1.0
    assert report["pairwise_prefix_signature_distance"]["median"] > 10.0
    assert report["cases"][0]["weighted_terminal_delta"]["SPX"] == pytest.approx(5.0)
    assert report["cases"][1]["weighted_terminal_delta"]["SPX"] == pytest.approx(-5.0)
