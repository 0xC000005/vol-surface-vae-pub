import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_memory_decoder import (
    build_memory_start_input_matrix,
    train_memory_start_prefix_decoder,
)


def test_build_memory_start_input_matrix_standardizes_memory_and_start() -> None:
    memory = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    start = np.asarray(
        [
            [10.0, 1.0],
            [12.0, 5.0],
            [14.0, 9.0],
        ],
        dtype=np.float32,
    )

    inputs, stats = build_memory_start_input_matrix(
        memory,
        start,
        fit_indices=np.asarray([0, 1], dtype=np.int64),
    )

    assert inputs.shape == (3, 4)
    np.testing.assert_allclose(inputs[:2, :2].mean(axis=0), np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(inputs[:2, 2:].mean(axis=0), np.zeros(2), atol=1e-6)
    assert stats["memory_mean"].shape == (1, 2)
    assert stats["start_std"].shape == (1, 2)


def test_train_memory_start_prefix_decoder_reduces_feature_loss() -> None:
    rng = np.random.default_rng(788)
    memory = rng.normal(size=(18, 4)).astype(np.float32)
    start = rng.normal(size=(18, 3)).astype(np.float32)
    inputs, _stats = build_memory_start_input_matrix(
        memory,
        start,
        fit_indices=np.arange(12),
    )
    w = rng.normal(size=(7, 5)).astype(np.float32)
    features = (inputs @ w).astype(np.float32)

    result = train_memory_start_prefix_decoder(
        inputs,
        features,
        train_indices=np.arange(12),
        test_indices=np.arange(12, 18),
        hidden_dim=32,
        steps=120,
        batch_size=8,
        lr=2e-3,
        seed=788,
    )

    assert result["predicted_features"].shape == features.shape
    assert result["loss_last"] < result["loss_first"]
    assert result["train_mse"] < 1.0
