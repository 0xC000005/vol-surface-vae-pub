from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_grounded_text_support_preference_reranker import (
    build_training_table_from_pools,
    evaluate_preference_rerank_replay,
    fit_text_support_preference_reranker,
    rerank_bridge_report_with_model,
)


def _bridge_report() -> dict:
    return {
        "schema_version": "unit",
        "split": {"train_indices": [0, 1, 2], "test_indices": [3]},
        "evaluation": {
            "heldout_examples": [
                {
                    "query_id": "q3",
                    "window_index": 3,
                    "window_id": "joint39_train_0003",
                    "role": "anchor",
                    "kind": "raw_openai_embedding_grounded_top3_90",
                    "required_grounding_claims": [
                        {"market": "SPX", "sign": 1, "direction": "up"}
                    ],
                    "pre_top3_90_candidate_pool": [
                        {
                            "rank": 1,
                            "window_index": 1,
                            "window_id": "joint39_train_0001",
                            "cosine": 0.90,
                            "weight": 0.55,
                            "retrieval_score": 0.90,
                            "scenario_title": "far future",
                            "score_components": {
                                "embedding_score": 0.90,
                                "view": "risk_manager_memo",
                                "direction_check": {"status": "pass", "checked_count": 1},
                            },
                        },
                        {
                            "rank": 2,
                            "window_index": 2,
                            "window_id": "joint39_train_0002",
                            "cosine": 0.80,
                            "weight": 0.30,
                            "retrieval_score": 0.80,
                            "scenario_title": "close future",
                            "score_components": {
                                "embedding_score": 0.80,
                                "view": "risk_manager_memo",
                                "direction_check": {"status": "pass", "checked_count": 1},
                            },
                        },
                        {
                            "rank": 3,
                            "window_index": 0,
                            "window_id": "joint39_train_0000",
                            "cosine": 0.70,
                            "weight": 0.15,
                            "retrieval_score": 0.70,
                            "scenario_title": "medium future",
                            "score_components": {
                                "embedding_score": 0.70,
                                "view": "risk_manager_memo",
                                "direction_check": {"status": "pass", "checked_count": 1},
                            },
                        },
                    ],
                    "top_train_pool": [],
                }
            ]
        },
    }


def _train_pools() -> list[dict]:
    return [
        {
            "query_window_index": 0,
            "candidates": [
                {
                    "rank": 1,
                    "window_index": 1,
                    "cosine": 0.95,
                    "weight": 0.6,
                    "score_components": {
                        "embedding_score": 0.95,
                        "view": "risk_manager_memo",
                        "direction_check": {"checked_count": 1},
                    },
                },
                {
                    "rank": 2,
                    "window_index": 2,
                    "cosine": 0.70,
                    "weight": 0.4,
                    "score_components": {
                        "embedding_score": 0.70,
                        "view": "risk_manager_memo",
                        "direction_check": {"checked_count": 1},
                    },
                },
            ],
        },
        {
            "query_window_index": 1,
            "candidates": [
                {
                    "rank": 1,
                    "window_index": 0,
                    "cosine": 0.90,
                    "weight": 0.6,
                    "score_components": {
                        "embedding_score": 0.90,
                        "view": "risk_manager_memo",
                        "direction_check": {"checked_count": 1},
                    },
                },
                {
                    "rank": 2,
                    "window_index": 2,
                    "cosine": 0.70,
                    "weight": 0.4,
                    "score_components": {
                        "embedding_score": 0.70,
                        "view": "risk_manager_memo",
                        "direction_check": {"checked_count": 1},
                    },
                },
            ],
        },
    ]


def test_training_table_labels_candidates_by_replay_closeness() -> None:
    future_delta = np.asarray(
        [
            [[0.0]],
            [[0.1]],
            [[3.0]],
            [[0.2]],
        ],
        dtype=np.float32,
    )
    history_raw = np.zeros((4, 2, 1), dtype=np.float32)
    delta_scale = np.ones((1, 1), dtype=np.float32)

    table = build_training_table_from_pools(
        query_pools=_train_pools(),
        history_raw=history_raw,
        future_delta=future_delta,
        delta_scale=delta_scale,
    )

    by_pair = {
        (row["query_window_index"], row["candidate_window_index"]): label
        for row, label in zip(table.rows, table.labels, strict=True)
    }
    assert by_pair[(0, 1)] > by_pair[(0, 2)]
    assert table.features.shape[0] == 4
    assert "embedding_score" in table.feature_names
    assert "start_distance_z" in table.feature_names


def test_preference_reranker_reorders_bridge_pool_and_preserves_top3_90() -> None:
    future_delta = np.asarray(
        [
            [[-1.0]],
            [[5.0]],
            [[0.2]],
            [[0.2]],
        ],
        dtype=np.float32,
    )
    history_raw = np.asarray(
        [
            [[0.0], [-1.0]],
            [[0.0], [5.0]],
            [[0.0], [0.2]],
            [[0.0], [0.2]],
        ],
        dtype=np.float32,
    )
    delta_scale = np.ones((1, 1), dtype=np.float32)
    table = build_training_table_from_pools(
        query_pools=_train_pools(),
        history_raw=history_raw,
        future_delta=future_delta,
        delta_scale=delta_scale,
    )
    model = fit_text_support_preference_reranker(table.features, table.labels)

    reranked = rerank_bridge_report_with_model(
        bridge_report=_bridge_report(),
        history_raw=history_raw,
        future_delta=future_delta,
        delta_scale=delta_scale,
        model=model,
    )

    pool = reranked["evaluation"]["heldout_examples"][0]["top_train_pool"]
    assert pool[0]["window_index"] == 2
    assert len(pool) <= 3
    assert abs(sum(float(item["weight"]) for item in pool) - 1.0) < 1e-8
    assert pool[0]["score_components"]["method"] == "grounded_text_preference_top3_90"


def test_replay_eval_reports_positive_delta_when_rerank_improves_pool() -> None:
    future_delta = np.asarray(
        [
            [[-1.0]],
            [[5.0]],
            [[0.2]],
            [[0.2]],
        ],
        dtype=np.float32,
    )
    history_raw = np.zeros((4, 2, 1), dtype=np.float32)
    delta_scale = np.ones((1, 1), dtype=np.float32)
    table = build_training_table_from_pools(
        query_pools=_train_pools(),
        history_raw=history_raw,
        future_delta=future_delta,
        delta_scale=delta_scale,
    )
    model = fit_text_support_preference_reranker(table.features, table.labels)
    original = _bridge_report()
    reranked = rerank_bridge_report_with_model(
        bridge_report=original,
        history_raw=history_raw,
        future_delta=future_delta,
        delta_scale=delta_scale,
        model=model,
    )

    replay = evaluate_preference_rerank_replay(
        bridge_report=original,
        reranked_bridge_report=reranked,
        future_delta=future_delta,
        delta_scale=delta_scale,
        top_k=1,
    )

    delta = replay["summary"]["learned_rerank_topk_replay"]["ensemble_crps_z"]
    assert delta["mean_delta_positive_is_better"] > 0.0
