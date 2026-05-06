import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (
    build_bridge_examples,
    evaluate_condition_bridge,
    select_train_test_windows_from_report,
    select_train_test_windows,
    summarize_bridge_metrics,
)


def _report() -> dict:
    return {
        "narrative_bundles": [
            {
                "window_id": "w0",
                "narratives": [
                    {
                        "id": "primary",
                        "text": "Equities fall and volatility rises.",
                        "grounding_status": "market_fact_supported",
                        "observed_fact_tokens": "SPX: DOWN LARGE; VIX: UP LARGE",
                    },
                    {
                        "id": "market",
                        "text": "SPX down large, VIX up large.",
                        "grounding_status": "observed_market_facts_only",
                        "observed_fact_tokens": "SPX: DOWN LARGE; VIX: UP LARGE",
                    },
                ],
                "contrastive_narratives": [
                    {
                        "kind": "opposite",
                        "text": "Equities rally and volatility falls.",
                    }
                ],
            },
            {
                "window_id": "w1",
                "narratives": [
                    {
                        "id": "primary",
                        "text": "Equities rally and volatility compresses.",
                        "grounding_status": "market_fact_supported",
                        "observed_fact_tokens": "SPX: UP LARGE; VIX: DOWN LARGE",
                    }
                ],
                "contrastive_narratives": [
                    {
                        "kind": "opposite",
                        "text": "Equities fall and volatility jumps.",
                    }
                ],
            },
            {
                "window_id": "w2",
                "narratives": [
                    {
                        "id": "primary",
                        "text": "Rates rise while equity risk is mixed.",
                        "grounding_status": "market_fact_supported",
                        "observed_fact_tokens": "US10Y: UP LARGE; SPX: FLAT",
                    }
                ],
                "contrastive_narratives": [
                    {
                        "kind": "opposite",
                        "text": "Rates fall sharply in a safety bid.",
                    }
                ],
            },
        ]
    }


def test_select_train_test_windows_uses_tail_holdout() -> None:
    split = select_train_test_windows(10, train_windows=6, test_windows=3)

    assert split["train_indices"] == [0, 1, 2, 3, 4, 5]
    assert split["test_indices"] == [6, 7, 8]
    assert split["excluded_indices"] == [9]


def test_select_train_test_windows_from_report_uses_manifest_splits() -> None:
    report = {
        "window_metadata": [
            {"window_id": "w0", "manifest_split": "train"},
            {"window_id": "w1", "manifest_split": "validation"},
            {"window_id": "w2", "manifest_split": "train"},
            {"window_id": "w3", "manifest_split": "test"},
        ]
    }

    split = select_train_test_windows_from_report(
        report,
        n_windows=4,
        train_windows=2,
        test_windows=1,
    )

    assert split["train_indices"] == [0, 2]
    assert split["test_indices"] == [3]
    assert split["excluded_indices"] == [1]
    assert split["source"] == "manifest"


def test_build_bridge_examples_preserves_embedding_order_and_roles() -> None:
    examples = build_bridge_examples(_report())

    assert [example["window_id"] for example in examples] == [
        "w0",
        "w0",
        "w0",
        "w1",
        "w1",
        "w2",
        "w2",
    ]
    assert [example["role"] for example in examples] == [
        "anchor",
        "positive",
        "negative",
        "anchor",
        "negative",
        "anchor",
        "negative",
    ]
    assert [example["window_index"] for example in examples] == [0, 0, 0, 1, 1, 2, 2]
    assert [example["embedding_index"] for example in examples] == list(range(7))


def test_evaluate_condition_bridge_reports_rank_and_hard_negative_separation() -> None:
    examples = build_bridge_examples(_report())
    memory_targets = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    condition_vectors = np.asarray(
        [
            [0.95, 0.05],
            [0.9, 0.1],
            [-0.9, 0.0],
            [0.1, 0.9],
            [1.0, 0.0],
            [-0.95, 0.05],
            [0.9, 0.0],
        ],
        dtype=np.float32,
    )

    result = evaluate_condition_bridge(
        examples,
        condition_vectors,
        memory_targets,
        train_indices=[0, 1],
        test_indices=[2],
        top_k=2,
    )

    assert result["heldout_example_count"] == 1
    assert result["heldout_window_count"] == 1
    assert result["heldout_examples"][0]["true_rank_full_pool"] == 1
    assert result["heldout_examples"][0]["true_rank_test_pool"] == 1
    assert result["heldout_examples"][0]["top_full_pool"][0]["window_index"] == 2
    assert result["heldout_examples"][0]["top_test_pool"][0]["window_index"] == 2
    assert result["heldout_examples"][0]["top_train_pool"][0]["window_index"] in {0, 1}
    assert result["hard_negative_separation"]["window_count"] == 1
    assert result["hard_negative_separation"]["mean_hard_margin"] > 1.7


def test_summarize_bridge_metrics_handles_empty_optional_blocks() -> None:
    summary = summarize_bridge_metrics(
        {
            "heldout_examples": [
                {
                    "target_cosine": 0.9,
                    "true_rank_full_pool": 1,
                    "true_rank_test_pool": 1,
                    "top_train_pool": [{"cosine": 0.7}],
                },
                {
                    "target_cosine": 0.8,
                    "true_rank_full_pool": 3,
                    "true_rank_test_pool": 2,
                    "top_train_pool": [{"cosine": 0.6}],
                },
            ],
            "hard_negative_separation": {
                "window_count": 2,
                "mean_hard_margin": 0.4,
                "mean_negative_gap": 0.5,
            },
        }
    )

    assert summary["heldout_mean_target_cosine"] == 0.85
    assert summary["heldout_recall_at_1_full_pool"] == 0.5
    assert summary["heldout_recall_at_1_test_pool"] == 0.5
    assert summary["heldout_median_true_rank_full_pool"] == 2.0
    assert summary["heldout_median_true_rank_test_pool"] == 1.5
    assert summary["heldout_mean_top_train_cosine"] == 0.65
    assert summary["heldout_hard_negative_mean_margin"] == 0.4
