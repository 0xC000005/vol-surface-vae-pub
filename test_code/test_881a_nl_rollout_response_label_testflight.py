import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_rollout_response_label_testflight import (
    build_candidate_label_bridge,
    build_cached_query_bridge_from_bridge_artifacts,
    build_mixture_label_bridge,
    build_query_bridge_from_examples,
    summarize_rollout_response_labels,
)


def _bridge_report() -> dict:
    return {
        "split": {"train_indices": [0, 1, 2, 3], "test_indices": [4]},
        "window_metadata": [{"window_index": idx} for idx in range(5)],
        "window_indices": [10, 11, 12, 13, 14],
        "evaluation": {
            "heldout_examples": [
                {
                    "embedding_index": 0,
                    "window_index": 4,
                    "window_id": "q4",
                    "role": "anchor",
                    "kind": "primary",
                    "target_cosine": 0.9,
                    "top_train_pool": [
                        {"window_index": 0, "window_id": "s0", "cosine": 0.95},
                        {"window_index": 1, "window_id": "s1", "cosine": 0.94},
                        {"window_index": 2, "window_id": "s2", "cosine": 0.93},
                        {"window_index": 3, "window_id": "s3", "cosine": 0.92},
                    ],
                }
            ]
        },
    }


def test_build_candidate_label_bridge_keeps_duplicate_query_rows() -> None:
    report = build_candidate_label_bridge(
        _bridge_report(),
        max_query_windows=1,
        candidate_pool_size=2,
    )

    rows = report["evaluation"]["heldout_examples"]
    assert report["purpose"] == "rollout_response_candidate_labels"
    assert len(rows) == 2
    assert rows[0]["window_index"] == rows[1]["window_index"] == 4
    assert rows[0]["query_id"] == "q4__candidate_001__s0"
    assert rows[1]["query_id"] == "q4__candidate_002__s1"
    assert [row["top_train_pool"][0]["window_index"] for row in rows] == [0, 1]


def test_summarize_rollout_response_labels_groups_by_query() -> None:
    candidate_bridge = build_candidate_label_bridge(
        _bridge_report(),
        max_query_windows=1,
        candidate_pool_size=2,
    )
    scenario_report = {
        "window_scores": [
            {
                "query_id": "q4__candidate_001__s0",
                "window_index": 4,
                "methods": {
                    "narrative_generator_topk": {
                        "energy_score_z": 1.0,
                        "ensemble_crps_z": 0.8,
                        "coverage_80": 0.4,
                    },
                    "historical_replay_topk": {
                        "energy_score_z": 0.9,
                        "ensemble_crps_z": 0.7,
                        "coverage_80": 0.5,
                    },
                },
            },
            {
                "query_id": "q4__candidate_002__s1",
                "window_index": 4,
                "methods": {
                    "narrative_generator_topk": {
                        "energy_score_z": 0.6,
                        "ensemble_crps_z": 0.5,
                        "coverage_80": 0.7,
                    },
                    "historical_replay_topk": {
                        "energy_score_z": 0.8,
                        "ensemble_crps_z": 0.6,
                        "coverage_80": 0.6,
                    },
                },
            },
        ]
    }

    summary = summarize_rollout_response_labels(
        candidate_bridge,
        scenario_report,
        baseline_scenario_report={
            "summary": {
                "narrative_generator_topk": {
                    "energy_score_z_mean": 0.55,
                    "ensemble_crps_z_mean": 0.45,
                }
            }
        },
    )

    assert summary["summary"]["query_count"] == 1
    assert summary["summary"]["candidate_row_count"] == 2
    assert summary["summary"]["best_generator_not_top1_count"] == 1
    assert summary["summary"]["mean_best_minus_top1_energy_score_z"] == -0.4
    assert summary["summary"]["best_generator_energy_score_z_mean"] == 0.6
    assert summary["summary"]["best_candidate_minus_baseline_topk_energy_score_z"] == (
        0.6 - 0.55
    )
    assert summary["summary"]["best_single_minus_baseline_topk_energy_score_z"] == (
        0.6 - 0.55
    )
    assert summary["groups"][0]["best_generator_support_window_indices"] == [1]


def test_build_mixture_label_bridge_builds_candidate_subsets() -> None:
    report = build_mixture_label_bridge(
        _bridge_report(),
        max_query_windows=1,
        candidate_pool_size=4,
        mixture_size=3,
        max_mixtures_per_query=3,
    )

    rows = report["evaluation"]["heldout_examples"]
    assert report["purpose"] == "rollout_response_mixture_labels"
    assert len(rows) == 3
    assert rows[0]["query_id"] == "q4__mixture_001__s0-s1-s2"
    assert [item["window_index"] for item in rows[0]["top_train_pool"]] == [0, 1, 2]
    assert len(rows[1]["top_train_pool"]) == 3


def test_build_query_bridge_from_examples_can_use_train_queries_without_self_support() -> None:
    examples = [
        {
            "window_index": 0,
            "window_id": "w0",
            "embedding_index": 0,
            "role": "anchor",
            "kind": "primary",
        },
        {
            "window_index": 1,
            "window_id": "w1",
            "embedding_index": 1,
            "role": "anchor",
            "kind": "primary",
        },
        {
            "window_index": 2,
            "window_id": "w2",
            "embedding_index": 2,
            "role": "anchor",
            "kind": "primary",
        },
    ]
    vectors = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        dtype=np.float32,
    )

    report = build_query_bridge_from_examples(
        examples=examples,
        condition_vectors=vectors,
        memory_targets=vectors,
        query_indices=[0, 1],
        support_indices=[0, 1, 2],
        candidate_pool_size=2,
        exclude_query_from_support=True,
    )

    rows = report["evaluation"]["heldout_examples"]
    assert [row["window_index"] for row in rows] == [0, 1]
    assert rows[0]["top_train_pool"][0]["window_index"] != 0
    assert rows[1]["top_train_pool"][0]["window_index"] != 1


def test_build_cached_query_bridge_from_bridge_artifacts_preserves_mapping() -> None:
    pipeline_report = {
        "narrative_bundles": [
            {
                "window_id": "w0",
                "narratives": [
                    {
                        "id": "anchor",
                        "text": "window zero",
                        "grounding_status": "ok",
                        "observed_fact_tokens": [],
                    }
                ],
            },
            {
                "window_id": "w1",
                "narratives": [
                    {
                        "id": "anchor",
                        "text": "window one",
                        "grounding_status": "ok",
                        "observed_fact_tokens": [],
                    }
                ],
            },
            {
                "window_id": "w2",
                "narratives": [
                    {
                        "id": "anchor",
                        "text": "window two",
                        "grounding_status": "ok",
                        "observed_fact_tokens": [],
                    }
                ],
            },
        ],
        "embedding_backend": "test",
        "embedding_model": "unit",
    }
    bridge_report = {
        "split": {
            "train_indices": [0, 1],
            "test_indices": [2],
            "excluded_indices": [],
        },
        "window_indices": [100, 101, 102],
        "window_metadata": [{"window_index": 100 + idx} for idx in range(3)],
    }
    arrays = {
        "condition_vectors": np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            dtype=np.float32,
        ),
        "memory_targets": np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            dtype=np.float32,
        ),
    }

    report = build_cached_query_bridge_from_bridge_artifacts(
        bridge_report=bridge_report,
        bridge_arrays=arrays,
        pipeline_report=pipeline_report,
        query_split="train",
        support_split="train",
        max_query_windows=2,
        candidate_pool_size=1,
    )

    assert report["purpose"] == "cached_query_bridge"
    assert report["window_indices"] == [100, 101, 102]
    assert report["embedding_backend"] == "test"
    assert [row["window_index"] for row in report["evaluation"]["heldout_examples"]] == [
        0,
        1,
    ]
    assert report["evaluation"]["heldout_examples"][0]["top_train_pool"][0][
        "window_index"
    ] == 1
