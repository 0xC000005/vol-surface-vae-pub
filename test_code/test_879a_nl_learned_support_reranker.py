import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_learned_support_reranker_testflight import (
    build_pairwise_training_table,
    fit_linear_support_reranker,
    rerank_bridge_report_with_model,
    score_candidate_pool,
)


def _toy_arrays():
    memory = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    history = np.zeros((4, 30, 3), dtype=np.float32)
    history[:, -1, 0] = np.asarray([0.0, 0.1, 2.0, 2.1], dtype=np.float32)
    future = np.zeros((4, 2, 3), dtype=np.float32)
    future[0, :, 0] = [1.0, 1.1]
    future[1, :, 0] = [1.0, 1.2]
    future[2, :, 0] = [-1.0, -1.1]
    future[3, :, 0] = [-1.0, -1.2]
    scale = np.ones((2, 3), dtype=np.float32)
    return memory, history, future, scale


def test_build_pairwise_training_table_excludes_query_from_candidates() -> None:
    memory, history, future, scale = _toy_arrays()

    table = build_pairwise_training_table(
        memory_targets=memory,
        history_level=history,
        future_delta=future,
        delta_scale=scale,
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        candidate_pool_size=2,
    )

    assert table.features.shape[0] > 0
    assert all(
        row["query_window_index"] != row["candidate_window_index"] for row in table.rows
    )
    assert set(row["query_window_index"] for row in table.rows) == {0, 1, 2}


def test_linear_support_reranker_prefers_lower_replay_loss_candidate() -> None:
    memory, history, future, scale = _toy_arrays()
    table = build_pairwise_training_table(
        memory_targets=memory,
        history_level=history,
        future_delta=future,
        delta_scale=scale,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        candidate_pool_size=3,
    )
    model = fit_linear_support_reranker(table.features, table.labels, ridge_alpha=1e-3)

    scored = score_candidate_pool(
        query_memory=memory[0],
        memory_targets=memory,
        history_level=history,
        future_delta=future,
        delta_scale=scale,
        candidate_indices=np.asarray([1, 2], dtype=np.int64),
        query_window_index=0,
        model=model,
    )

    assert scored[0]["window_index"] == 1
    assert scored[0]["learned_support_score"] > scored[1]["learned_support_score"]
    assert scored[0]["true_replay_loss_z"] < scored[1]["true_replay_loss_z"]


def test_rerank_bridge_report_with_model_preserves_query_and_reorders_pool() -> None:
    memory, history, future, scale = _toy_arrays()
    table = build_pairwise_training_table(
        memory_targets=memory,
        history_level=history,
        future_delta=future,
        delta_scale=scale,
        train_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        candidate_pool_size=3,
    )
    model = fit_linear_support_reranker(table.features, table.labels, ridge_alpha=1e-3)
    bridge = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 0,
                    "window_id": "q0",
                    "embedding_index": 0,
                    "role": "anchor",
                    "top_train_pool": [
                        {"window_index": 2, "window_id": "bad", "cosine": 0.3},
                        {"window_index": 1, "window_id": "good", "cosine": 0.2},
                    ],
                }
            ]
        }
    }

    reranked = rerank_bridge_report_with_model(
        bridge_report=bridge,
        condition_vectors=np.asarray([memory[0]], dtype=np.float32),
        memory_targets=memory,
        history_level=history,
        future_delta=future,
        delta_scale=scale,
        model=model,
    )

    row = reranked["evaluation"]["heldout_examples"][0]
    assert row["window_index"] == 0
    assert [item["window_index"] for item in row["top_train_pool"]] == [1, 2]
    assert row["top_train_pool"][0]["original_support_rank"] == 2
