import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_portfolio_quality_guard_live_breadth_diagnostic import (
    _support_jaccard,
    summarize_rows,
)


def test_support_jaccard_handles_empty_and_partial_overlap() -> None:
    assert _support_jaccard([], []) == 1.0
    assert _support_jaccard([1, 2, 3], [2, 3, 4]) == 0.5


def test_summarize_rows_warns_when_candidate_breadth_is_low() -> None:
    report = summarize_rows(
        [
            {
                "quality_guard_candidate_count": 1,
                "quality_guard_low_breadth": True,
                "quality_guard_fallback": False,
                "support_jaccard": 0.5,
            },
            {
                "quality_guard_candidate_count": 8,
                "quality_guard_low_breadth": False,
                "quality_guard_fallback": False,
                "support_jaccard": 0.25,
            },
        ],
        min_candidate_mixtures=4,
    )

    assert report["status"] == "candidate_breadth_warning"
    assert report["summary"]["quality_guard_low_breadth_count"] == 1
    assert report["decision"]["promote_live_default"] is False


def test_summarize_rows_allows_paired_rollout_when_breadth_is_adequate() -> None:
    report = summarize_rows(
        [
            {
                "quality_guard_candidate_count": 6,
                "quality_guard_low_breadth": False,
                "quality_guard_fallback": False,
                "support_jaccard": 0.5,
            }
        ],
        min_candidate_mixtures=4,
    )

    assert report["status"] == "candidate_breadth_ok"
    assert "paired rollout comparison" in report["decision"]["interpretation"]
