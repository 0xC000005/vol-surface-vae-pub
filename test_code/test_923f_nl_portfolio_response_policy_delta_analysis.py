import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_policy_delta_analysis import (
    _bridge_row_lookup,
    _entropy,
    _policy_features,
)


def test_bridge_row_lookup_reads_heldout_examples():
    bridge = {
        "evaluation": {
            "heldout_examples": [
                {"window_index": 5, "value": "a"},
                {"window_index": "6", "value": "b"},
            ]
        }
    }

    rows = _bridge_row_lookup(bridge)

    assert rows[5]["value"] == "a"
    assert rows[6]["value"] == "b"


def test_policy_features_summarize_weights_and_scores():
    row = {
        "support_policy": {
            "selected_support_weights": [0.5, 0.25, 0.25],
            "candidate_probabilities": [
                {"probability": 0.6, "score": 1.0},
                {"probability": 0.4, "score": -1.0},
            ],
        }
    }

    features = _policy_features(row)

    assert features["support_weight_max"] == 0.5
    assert features["support_effective_n"] > 2.0
    assert features["candidate_score_spread"] == 2.0


def test_entropy_ignores_zero_weights():
    assert _entropy([1.0, 0.0]) == 0.0
