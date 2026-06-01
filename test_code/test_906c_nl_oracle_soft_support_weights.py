import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_oracle_soft_support_weights import (
    build_oracle_weighted_bridge,
    candidate_label_scores,
    marginalize_candidate_support_weights,
    standardized_candidate_probabilities,
)


def _candidate_bridge() -> dict:
    return {
        "evaluation": {
            "heldout_examples": [
                {
                    "query_id": "q0__mix_a_b",
                    "window_index": 0,
                    "window_id": "q0",
                    "top_train_pool": [
                        {"window_index": 10, "window_id": "a", "cosine": 0.9},
                        {"window_index": 11, "window_id": "b", "cosine": 0.8},
                    ],
                },
                {
                    "query_id": "q0__mix_a_c",
                    "window_index": 0,
                    "window_id": "q0",
                    "top_train_pool": [
                        {"window_index": 10, "window_id": "a", "cosine": 0.7},
                        {"window_index": 12, "window_id": "c", "cosine": 0.6},
                    ],
                },
            ]
        }
    }


def test_candidate_label_scores_are_higher_for_lower_generator_metric() -> None:
    scenario = {
        "window_scores": [
            {
                "query_id": "q0__mix_a_b",
                "methods": {"narrative_generator_topk": {"energy_score_z": 1.0}},
            },
            {
                "query_id": "q0__mix_a_c",
                "methods": {"narrative_generator_topk": {"energy_score_z": 0.5}},
            },
        ]
    }

    scores = candidate_label_scores(
        _candidate_bridge()["evaluation"]["heldout_examples"], scenario
    )

    assert np.allclose(scores, [-1.0, -0.5])
    assert scores[1] > scores[0]


def test_standardized_candidate_probabilities_prefer_better_label() -> None:
    probs = standardized_candidate_probabilities(np.asarray([-1.0, -0.5]))

    assert probs.shape == (2,)
    assert np.isclose(float(probs.sum()), 1.0)
    assert probs[1] > probs[0]


def test_marginalize_candidate_support_weights_preserves_auditable_supports() -> None:
    candidates = _candidate_bridge()["evaluation"]["heldout_examples"]
    rows = marginalize_candidate_support_weights(
        candidates,
        np.asarray([0.25, 0.75]),
    )

    weights = {int(row["window_index"]): float(row["weight"]) for row in rows}
    assert np.isclose(sum(weights.values()), 1.0)
    assert np.isclose(weights[10], 0.5)
    assert weights[12] > weights[11]
    assert rows[0]["window_index"] == 10
    assert rows[0]["cosine"] == 0.9


def test_build_oracle_weighted_bridge_replaces_support_pool_with_weights() -> None:
    base = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 0,
                    "window_id": "q0",
                    "top_train_pool": [{"window_index": 99, "cosine": 0.1}],
                }
            ]
        }
    }
    scenario = {
        "window_scores": [
            {
                "query_id": "q0__mix_a_b",
                "methods": {"narrative_generator_topk": {"energy_score_z": 1.0}},
            },
            {
                "query_id": "q0__mix_a_c",
                "methods": {"narrative_generator_topk": {"energy_score_z": 0.5}},
            },
        ]
    }

    report = build_oracle_weighted_bridge(
        base_bridge=base,
        candidate_bridge=_candidate_bridge(),
        scenario_report=scenario,
    )

    row = report["evaluation"]["heldout_examples"][0]
    assert row["support_policy"]["policy_kind"] == "oracle_soft_listwise_upper_bound"
    assert row["support_policy"]["candidate_count"] == 2
    assert {item["window_index"] for item in row["top_train_pool"]} == {10, 11, 12}
    assert all("weight" in item for item in row["top_train_pool"])
    assert report["oracle_support_policy"]["changed_rows"] == 1
