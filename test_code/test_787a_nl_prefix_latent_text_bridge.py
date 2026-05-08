import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_text_bridge import (
    build_text_start_input_matrix,
    train_text_start_prefix_bridge,
)


def test_build_text_start_input_matrix_concatenates_normalized_text_and_start() -> None:
    text_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    start_state = np.asarray(
        [
            [10.0, 1.0],
            [12.0, 5.0],
        ],
        dtype=np.float32,
    )
    target_indices = np.asarray([0, 1, 0], dtype=np.int64)

    inputs, stats = build_text_start_input_matrix(
        text_embeddings,
        start_state,
        target_indices,
        fit_window_indices=np.asarray([0, 1], dtype=np.int64),
    )

    assert inputs.shape == (3, 4)
    np.testing.assert_allclose(
        inputs[:, :2],
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(inputs[0, 2:], inputs[2, 2:])
    assert np.isfinite(inputs).all()
    assert stats["start_mean"].shape == (1, 2)
    assert stats["start_std"].shape == (1, 2)


def test_build_text_start_input_matrix_supports_text_only_and_start_only() -> None:
    text_embeddings = np.asarray(
        [
            [3.0, 4.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    start_state = np.asarray(
        [
            [10.0, 1.0, 0.5],
            [12.0, 5.0, 1.5],
        ],
        dtype=np.float32,
    )
    target_indices = np.asarray([0, 1], dtype=np.int64)

    text_only, _text_stats = build_text_start_input_matrix(
        text_embeddings,
        start_state,
        target_indices,
        fit_window_indices=np.asarray([0, 1], dtype=np.int64),
        input_mode="text_only",
    )
    start_only, _start_stats = build_text_start_input_matrix(
        text_embeddings,
        start_state,
        target_indices,
        fit_window_indices=np.asarray([0, 1], dtype=np.int64),
        input_mode="start_only",
    )

    assert text_only.shape == (2, 2)
    assert start_only.shape == (2, 3)
    np.testing.assert_allclose(text_only[0], [0.6, 0.8])
    np.testing.assert_allclose(start_only.mean(axis=0), np.zeros(3), atol=1e-6)


def test_train_text_start_prefix_bridge_reduces_latent_loss() -> None:
    rng = np.random.default_rng(787)
    text = rng.normal(size=(24, 5)).astype(np.float32)
    starts = rng.normal(size=(6, 3)).astype(np.float32)
    target_indices = np.repeat(np.arange(6), 4)
    true_w = rng.normal(size=(8, 3)).astype(np.float32)
    inputs, _stats = build_text_start_input_matrix(
        text,
        starts,
        target_indices,
        fit_window_indices=np.arange(4),
    )
    targets = (inputs @ true_w).astype(np.float32)
    train_examples = np.where(target_indices < 4)[0]
    test_examples = np.where(target_indices >= 4)[0]

    result = train_text_start_prefix_bridge(
        inputs,
        targets,
        train_example_indices=train_examples,
        test_example_indices=test_examples,
        hidden_dim=32,
        steps=120,
        batch_size=16,
        lr=2e-3,
        seed=787,
    )

    assert result["predicted_latents"].shape == targets.shape
    assert result["loss_last"] < result["loss_first"]
    assert result["train_mse"] < 1.0
    assert np.isfinite(result["predicted_latents"]).all()
