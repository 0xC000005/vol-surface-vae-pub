import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (
    PrefixFeatureLayout,
    build_prefix_feature_matrix,
    reconstruct_prefix_from_features,
    train_prefix_autoencoder,
)


def _toy_prefix_arrays(n: int = 8, t: int = 4, c: int = 3) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(786)
    start = rng.normal(size=(n, c)).astype(np.float32)
    backward = np.cumsum(rng.normal(scale=0.05, size=(n, t, c)), axis=1).astype(
        np.float32
    )
    backward[:, -1, :] = 0.0
    history_level = start[:, None, :] + backward
    history_norm = rng.normal(scale=0.2, size=(n, t, c)).astype(np.float32)
    center = rng.normal(scale=0.01, size=(n, c)).astype(np.float32)
    scale = np.exp(rng.normal(scale=0.05, size=(n, c))).astype(np.float32)
    drift = rng.normal(scale=0.01, size=(n, c)).astype(np.float32)
    return history_level, history_norm, center, scale, drift


def test_prefix_feature_roundtrip_preserves_start_state_and_shapes() -> None:
    history_level, history_norm, center, scale, drift = _toy_prefix_arrays()

    features, layout = build_prefix_feature_matrix(
        history_level,
        history_norm,
        center,
        scale,
        drift,
    )
    reconstructed = reconstruct_prefix_from_features(
        features,
        start_state=history_level[:, -1, :],
        layout=layout,
    )

    assert isinstance(layout, PrefixFeatureLayout)
    assert features.shape[0] == history_level.shape[0]
    np.testing.assert_allclose(reconstructed["history_level"], history_level)
    np.testing.assert_allclose(reconstructed["history_norm"], history_norm)
    np.testing.assert_allclose(reconstructed["center"], center)
    np.testing.assert_allclose(reconstructed["scale"], scale, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(reconstructed["drift_feature"], drift)
    np.testing.assert_allclose(
        reconstructed["history_level"][:, -1, :],
        history_level[:, -1, :],
    )


def test_train_prefix_autoencoder_reduces_reconstruction_loss() -> None:
    history_level, history_norm, center, scale, drift = _toy_prefix_arrays(n=16)
    features, _layout = build_prefix_feature_matrix(
        history_level,
        history_norm,
        center,
        scale,
        drift,
    )
    train_indices = np.arange(12)
    test_indices = np.arange(12, 16)

    result = train_prefix_autoencoder(
        features,
        train_indices=train_indices,
        test_indices=test_indices,
        latent_dim=4,
        hidden_dim=16,
        steps=80,
        batch_size=8,
        lr=2e-3,
        seed=786,
    )

    assert result["encoded_latents"].shape == (16, 4)
    assert result["reconstructed_features"].shape == features.shape
    assert result["loss_last"] < result["loss_first"]
    assert result["train_mse"] < 2.0
    assert np.isfinite(result["reconstructed_features"]).all()
