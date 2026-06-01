import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_support_reliability_policy import (
    _entropy,
    _group_candidate_rows,
    _softmax,
    _support_indices,
)


def test_support_indices_prefers_candidate_support_window_indices():
    row = {
        "candidate_support_window_indices": [3, "4"],
        "top_train_pool": [{"window_index": 9}],
    }

    assert _support_indices(row) == [3, 4]


def test_support_indices_falls_back_to_top_train_pool():
    row = {"top_train_pool": [{"window_index": 9}, {"window_index": "11"}]}

    assert _support_indices(row) == [9, 11]


def test_group_candidate_rows_sorts_by_candidate_rank():
    bridge = {
        "evaluation": {
            "heldout_examples": [
                {"window_index": 1, "candidate_mixture_rank": 2, "query_id": "b"},
                {"window_index": 1, "candidate_mixture_rank": 1, "query_id": "a"},
                {"window_index": 2, "candidate_mixture_rank": 1, "query_id": "c"},
            ]
        }
    }

    groups = _group_candidate_rows(bridge)

    assert [row["query_id"] for row in groups[1]] == ["a", "b"]
    assert [row["query_id"] for row in groups[2]] == ["c"]


def test_softmax_entropy_is_higher_for_flat_probabilities():
    sharp = _softmax([2.0, 0.0, -2.0], temperature=0.25)
    flat = _softmax([2.0, 0.0, -2.0], temperature=10.0)

    assert _entropy(flat) > _entropy(sharp)
