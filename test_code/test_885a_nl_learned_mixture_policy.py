import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_learned_mixture_policy_testflight import (
    build_support_set_training_table,
    build_mixture_policy_training_table,
    fit_listwise_mixture_policy,
    fit_linear_mixture_policy,
    fit_pairwise_mixture_ranker,
    fit_support_set_item_ranker,
    listwise_support_weights_for_candidates,
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


def test_pairwise_mixture_ranker_learns_within_query_preferences() -> None:
    features = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 0.0],
            [1.0, 0.5],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0.0, 1.0, 0.2, 1.2], dtype=np.float32)
    query_ids = ["q0", "q0", "q1", "q1"]

    model = fit_pairwise_mixture_ranker(
        features,
        labels,
        query_ids=query_ids,
        feature_names=["a", "b"],
        learning_rate=0.2,
        epochs=300,
        l2=1e-4,
    )

    scores = model.predict(features)
    assert scores[1] > scores[0]
    assert scores[3] > scores[2]


def test_pairwise_mixture_ranker_can_rerank_bridge_candidates() -> None:
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
    model = fit_pairwise_mixture_ranker(
        table.features,
        table.labels,
        query_ids=[row["window_index"] for row in table.rows],
        feature_names=table.feature_names,
        learning_rate=0.2,
        epochs=300,
    )
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
    assert row["support_policy"]["policy_kind"] == "pairwise_ranker"
    assert row["support_policy"]["selected_query_id"] == "q0__mixture_002__a-c"


def test_support_set_training_table_preserves_support_items() -> None:
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

    table = build_support_set_training_table(
        candidate_bridge=_candidate_bridge(),
        scenario_report=scenario_report,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        history_level=history_level,
    )

    assert table.item_features.shape[0] == 2
    assert table.item_features.shape[1] == 2
    assert np.allclose(table.labels, [-1.2, -0.7])
    assert table.rows[0]["support_window_indices"] == [1, 2]


def test_support_set_item_ranker_learns_set_preferences() -> None:
    item_features = np.asarray(
        [
            [[0.0, 0.0], [0.1, 0.0]],
            [[1.0, 0.0], [1.2, 0.1]],
            [[0.2, 0.0], [0.0, 0.1]],
            [[1.1, 0.0], [1.3, 0.1]],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0.0, 1.0, 0.1, 1.1], dtype=np.float32)
    query_ids = ["q0", "q0", "q1", "q1"]

    model = fit_support_set_item_ranker(
        item_features,
        labels,
        query_ids=query_ids,
        item_feature_names=["signal", "noise"],
        hidden_dim=4,
        learning_rate=0.05,
        epochs=250,
        seed=7,
    )

    scores = model.predict(item_features)
    assert scores[1] > scores[0]
    assert scores[3] > scores[2]


def test_support_set_item_ranker_can_rerank_bridge_candidates() -> None:
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
    table = build_support_set_training_table(
        candidate_bridge=_candidate_bridge(),
        scenario_report=scenario_report,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        history_level=history_level,
    )
    model = fit_support_set_item_ranker(
        table.item_features,
        table.labels,
        query_ids=[row["window_index"] for row in table.rows],
        item_feature_names=table.item_feature_names,
        hidden_dim=4,
        learning_rate=0.05,
        epochs=300,
        seed=11,
    )
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
    assert row["support_policy"]["policy_kind"] == "support_set_item_ranker"
    assert row["support_policy"]["selected_query_id"] == "q0__mixture_002__a-c"


def test_listwise_mixture_policy_learns_query_level_probabilities() -> None:
    features = np.asarray(
        [
            [0.0, 1.0],
            [2.0, 0.0],
            [1.0, 0.5],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0.0, 1.0, 0.2, 1.2], dtype=np.float32)
    query_ids = ["q0", "q0", "q1", "q1"]

    model = fit_listwise_mixture_policy(
        features,
        labels,
        query_ids=query_ids,
        feature_names=["a", "b"],
        learning_rate=0.2,
        epochs=300,
        l2=1e-4,
    )

    scores = model.predict(features)
    assert scores[1] > scores[0]
    assert scores[3] > scores[2]


def test_listwise_support_weights_for_candidates_marginalizes_candidate_probs() -> None:
    candidates = _candidate_bridge()["evaluation"]["heldout_examples"]
    scores = np.asarray([0.0, 2.0], dtype=np.float32)

    support_rows, candidate_probs = listwise_support_weights_for_candidates(
        candidates,
        scores,
        probability_temperature=1.0,
    )

    weight_by_index = {row["window_index"]: row["weight"] for row in support_rows}
    assert candidate_probs[1] > candidate_probs[0]
    assert weight_by_index[1] > weight_by_index[3]
    assert weight_by_index[3] > weight_by_index[2]


def test_listwise_mixture_policy_rerank_writes_support_weights() -> None:
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
    model = fit_listwise_mixture_policy(
        table.features,
        table.labels,
        query_ids=[row["window_index"] for row in table.rows],
        feature_names=table.feature_names,
        learning_rate=0.2,
        epochs=300,
    )
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
    assert row["support_policy"]["policy_kind"] == "listwise_mixture_policy"
    assert all("weight" in item for item in row["top_train_pool"])
    assert sum(item["weight"] for item in row["top_train_pool"]) == 1.0
