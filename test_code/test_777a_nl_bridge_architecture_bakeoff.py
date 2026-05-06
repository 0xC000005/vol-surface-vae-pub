import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_bridge_architecture_bakeoff import (
    choose_best_method,
    summarize_bakeoff,
    train_clip_condition_adapter,
)


def test_train_clip_condition_adapter_reduces_loss_and_outputs_condition_vectors() -> None:
    rng = np.random.default_rng(777)
    memory_targets = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    text_embeddings = np.asarray(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.9, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.1, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.1, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            rng.normal(size=4),
            rng.normal(size=4),
            rng.normal(size=4),
        ],
        dtype=np.float32,
    )
    target_indices = np.asarray([0, 0, 1, 1, 2, 2, -1, -1, -1], dtype=np.int64)
    roles = [
        "anchor",
        "positive",
        "anchor",
        "positive",
        "anchor",
        "positive",
        "negative",
        "negative",
        "negative",
    ]
    groups = ["w0", "w0", "w1", "w1", "w2", "w2", "w0", "w1", "w2"]

    result = train_clip_condition_adapter(
        text_embeddings,
        memory_targets,
        target_indices,
        roles,
        groups,
        condition_dim=3,
        hidden_dim=12,
        steps=180,
        lr=2e-2,
        temperature=0.1,
        seed=777,
    )

    assert result["condition_vectors"].shape == (9, 3)
    assert result["loss_last"] < result["loss_first"]
    assert np.isfinite(result["condition_vectors"]).all()


def test_summarize_bakeoff_compares_methods_and_keeps_best() -> None:
    results = {
        "mlp": {
            "summary": {
                "heldout_mean_target_cosine": 0.70,
                "heldout_recall_at_3_test_pool": 0.20,
                "heldout_hard_negative_mean_margin": 0.10,
            }
        },
        "clip": {
            "summary": {
                "heldout_mean_target_cosine": 0.80,
                "heldout_recall_at_3_test_pool": 0.30,
                "heldout_hard_negative_mean_margin": 0.40,
            }
        },
    }

    summary = summarize_bakeoff(results)

    assert summary["method_count"] == 2
    assert summary["best_by_heldout_mean_target_cosine"] == "clip"
    assert summary["best_by_heldout_recall_at_3_test_pool"] == "clip"
    assert summary["methods"]["clip"]["heldout_hard_negative_mean_margin"] == 0.40


def test_choose_best_method_ignores_missing_metric() -> None:
    assert (
        choose_best_method(
            {
                "a": {"summary": {"score": None}},
                "b": {"summary": {"score": 0.2}},
                "c": {"summary": {"other": 1.0}},
            },
            "score",
        )
        == "b"
    )
