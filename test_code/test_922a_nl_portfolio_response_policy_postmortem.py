import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_policy_postmortem import (
    PORTFOLIO_LABEL_METRIC,
    candidate_headroom_report,
    decide_postmortem,
    pairwise_preference_accuracy,
    selection_regret,
)


def test_pairwise_preference_accuracy_groups_by_query():
    result = pairwise_preference_accuracy(
        predicted_scores=np.asarray([3.0, 2.0, 1.0, 1.0, 2.0]),
        true_scores=np.asarray([3.0, 2.0, 1.0, 2.0, 1.0]),
        query_ids=["a", "a", "a", "b", "b"],
    )

    assert result["pair_count"] == 4
    assert result["tie_count"] == 0
    assert result["accuracy"] == 0.75


def test_selection_regret_reports_oracle_gap():
    result = selection_regret(
        predicted_scores=np.asarray([3.0, 2.0, 1.0, 1.0, 2.0]),
        raw_losses=np.asarray([0.4, 0.1, 0.2, 1.0, 0.5]),
        query_ids=["a", "a", "a", "b", "b"],
    )

    assert result["query_count"] == 2
    assert np.isclose(result["mean_regret"], 0.15)
    assert result["selected_position_mean"] == 1.5
    assert result["oracle_position_mean"] == 2.0
    assert result["exact_oracle_selection_rate"] == 0.5


def test_candidate_headroom_report_uses_within_query_oracle():
    report = {
        "window_scores": [
            {
                "window_index": 1,
                "methods": {
                    "narrative_generator_topk": {PORTFOLIO_LABEL_METRIC: 1.0}
                },
            },
            {
                "window_index": 1,
                "methods": {
                    "narrative_generator_topk": {PORTFOLIO_LABEL_METRIC: 0.5}
                },
            },
            {
                "window_index": 1,
                "methods": {
                    "narrative_generator_topk": {PORTFOLIO_LABEL_METRIC: 0.75}
                },
            },
            {
                "window_index": 2,
                "methods": {
                    "narrative_generator_topk": {PORTFOLIO_LABEL_METRIC: 0.2}
                },
            },
            {
                "window_index": 2,
                "methods": {
                    "narrative_generator_topk": {PORTFOLIO_LABEL_METRIC: 0.3}
                },
            },
        ]
    }

    result = candidate_headroom_report(report)

    assert result["query_count"] == 2
    assert result["candidate_rows"] == 5
    assert np.isclose(result["median_top1_minus_oracle"], 0.25)
    assert result["oracle_position_counts"] == {"1": 1, "2": 1}


def test_decide_postmortem_flags_feature_predictability_when_labels_are_stable():
    report = {
        "predictability": {
            "test": {
                "pairwise": {"accuracy": 0.52},
                "rank_baseline_pairwise": {"accuracy": 0.51},
            }
        },
        "label_noise": {
            "half_a_vs_half_b_pearson": 0.90,
            "half_diff_to_query_spread_ratio_median": 0.10,
        },
        "candidate_headroom": {"median_top1_minus_oracle": 0.20},
    }

    decision = decide_postmortem(report)

    assert decision["status"] == "feature_predictability_is_primary_bottleneck"
