import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_learned_mixture_policy_testflight import (
    build_mixture_policy_training_table,
    fit_linear_mixture_policy,
    rerank_bridge_report_with_mixture_policy,
)


def _candidate_bridge() -> dict:
    return {
        "evaluation": {
            "heldout_examples": [
                {
                    "query_id": "q0__mixture_001__a-b",
                    "window_index": 0,
                    "window_id": "q0",
                    "embedding_index": 0,
                    "role": "anchor",
                    "candidate_mixture_rank": 1,
                    "candidate_mixture_positions": [1, 2],
                    "candidate_support_window_indices": [1, 2],
                    "top_train_pool": [
                        {"window_index": 1, "window_id": "a", "cosine": 0.9},
                        {"window_index": 2, "window_id": "b", "cosine": 0.8},
                    ],
                },
                {
                    "query_id": "q0__mixture_002__a-c",
                    "window_index": 0,
                    "window_id": "q0",
                    "embedding_index": 0,
                    "role": "anchor",
                    "candidate_mixture_rank": 2,
                    "candidate_mixture_positions": [1, 3],
                    "candidate_support_window_indices": [1, 3],
                    "top_train_pool": [
                        {"window_index": 1, "window_id": "a", "cosine": 0.9},
                        {"window_index": 3, "window_id": "c", "cosine": 0.7},
                    ],
                },
            ]
        }
    }


def test_build_mixture_policy_training_table_uses_generator_metric() -> None:
    scenario_report = {
        "window_scores": [
            {
                "query_id": "q0__mixture_001__a-b",
                "methods": {
                    "narrative_generator_topk": {"energy_score_z": 1.2},
                },
            },
            {
                "query_id": "q0__mixture_002__a-c",
                "methods": {
                    "narrative_generator_topk": {"energy_score_z": 0.7},
                },
            },
        ]
    }
    condition_vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    memory_targets = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    history_level = np.zeros((4, 2, 3), dtype=np.float32)

    table = build_mixture_policy_training_table(
        candidate_bridge=_candidate_bridge(),
        scenario_report=scenario_report,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        history_level=history_level,
    )

    assert table.features.shape[0] == 2
    assert np.allclose(table.labels, [-1.2, -0.7])
    assert table.rows[1]["generator_energy_score_z"] == 0.7


def test_rerank_bridge_report_with_mixture_policy_selects_best_candidate() -> None:
    scenario_report = {
        "window_scores": [
            {
                "query_id": "q0__mixture_001__a-b",
                "methods": {
                    "narrative_generator_topk": {"energy_score_z": 1.2},
                },
            },
            {
                "query_id": "q0__mixture_002__a-c",
                "methods": {
                    "narrative_generator_topk": {"energy_score_z": 0.7},
                },
            },
        ]
    }
    condition_vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    memory_targets = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    history_level = np.zeros((4, 2, 3), dtype=np.float32)
    table = build_mixture_policy_training_table(
        candidate_bridge=_candidate_bridge(),
        scenario_report=scenario_report,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        history_level=history_level,
    )
    model = fit_linear_mixture_policy(table.features, table.labels, ridge_alpha=0.01)
    base_bridge = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 0,
                    "window_id": "q0",
                    "role": "anchor",
                    "top_train_pool": [],
                }
            ]
        }
    }

    reranked = rerank_bridge_report_with_mixture_policy(
        bridge_report=base_bridge,
        candidate_bridge=_candidate_bridge(),
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        history_level=history_level,
        model=model,
    )

    row = reranked["evaluation"]["heldout_examples"][0]
    assert row["support_policy"]["selected_query_id"] == "q0__mixture_002__a-c"
    assert [item["window_index"] for item in row["top_train_pool"]] == [1, 3]
