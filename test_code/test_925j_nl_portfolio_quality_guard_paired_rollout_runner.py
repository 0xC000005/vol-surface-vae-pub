import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_portfolio_quality_guard_paired_rollout_runner import (
    select_active_breadth_rows,
)


def test_select_active_breadth_rows_filters_fallback_low_breadth_and_unchanged() -> None:
    report = {
        "rows": [
            {
                "case_name": "fallback",
                "quality_guard_fallback": True,
                "quality_guard_low_breadth": False,
                "quality_guard_active": False,
                "support_jaccard": 0.0,
                "quality_guard_candidate_count": 64,
            },
            {
                "case_name": "low",
                "quality_guard_fallback": False,
                "quality_guard_low_breadth": True,
                "quality_guard_active": True,
                "support_jaccard": 0.0,
                "quality_guard_candidate_count": 1,
            },
            {
                "case_name": "unchanged",
                "quality_guard_fallback": False,
                "quality_guard_low_breadth": False,
                "quality_guard_active": True,
                "support_jaccard": 1.0,
                "quality_guard_candidate_count": 64,
            },
            {
                "case_name": "active",
                "quality_guard_fallback": False,
                "quality_guard_low_breadth": False,
                "quality_guard_active": True,
                "support_jaccard": 0.2,
                "quality_guard_candidate_count": 64,
                "start_window_index": 18,
            },
        ]
    }

    rows = select_active_breadth_rows(report, max_pairs=10)
    assert [row["case_name"] for row in rows] == ["active"]


def test_select_active_breadth_rows_can_include_unchanged_support() -> None:
    report = {
        "rows": [
            {
                "case_name": "unchanged",
                "quality_guard_fallback": False,
                "quality_guard_low_breadth": False,
                "quality_guard_active": True,
                "support_jaccard": 1.0,
                "quality_guard_candidate_count": 64,
                "start_window_index": 18,
            }
        ]
    }

    rows = select_active_breadth_rows(
        report,
        max_pairs=10,
        require_changed_support=False,
    )
    assert len(rows) == 1
