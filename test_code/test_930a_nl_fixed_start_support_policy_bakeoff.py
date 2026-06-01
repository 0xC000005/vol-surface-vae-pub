import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_fixed_start_support_policy_bakeoff import (
    _pairwise_support_summary,
    _policy_summary,
)


def test_pairwise_support_summary_tracks_narrative_separation():
    cases = [
        {
            "case": "a",
            "support": [{"window_index": 1}, {"window_index": 2}],
        },
        {
            "case": "b",
            "support": [{"window_index": 3}, {"window_index": 4}],
        },
        {
            "case": "c",
            "support": [{"window_index": 2}, {"window_index": 4}],
        },
    ]

    summary = _pairwise_support_summary(cases)

    assert summary["min_support_jaccard"] == 0.0
    assert summary["max_support_jaccard"] == 1.0 / 3.0
    assert summary["mean_support_jaccard"] == (0.0 + 1.0 / 3.0 + 1.0 / 3.0) / 3.0


def test_policy_summary_separates_start_only_overlap_from_support_diversity():
    cases = [
        {
            "case": "risk_on",
            "support": [{"window_index": 1}, {"window_index": 2}],
            "metrics": {
                "support_count": 2,
                "unique_year_count": 1,
                "start_only_jaccard": 0.0,
                "direction_check_status": "pass",
            },
        },
        {
            "case": "risk_off",
            "support": [{"window_index": 3}, {"window_index": 4}],
            "metrics": {
                "support_count": 2,
                "unique_year_count": 2,
                "start_only_jaccard": 0.5,
                "direction_check_status": "warning",
            },
        },
    ]

    summary = _policy_summary(cases)

    assert summary["mean_support_count"] == 2.0
    assert summary["distinct_support_windows_across_cases"] == 4
    assert summary["mean_support_jaccard"] == 0.0
    assert summary["mean_start_only_jaccard"] == 0.25
    assert summary["direction_status_counts"] == {"pass": 1, "warning": 1}
