from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.context_target_jepa_data import (  # noqa: E402
    build_context_target_jepa_batch,
)
from experiments.world.part1_jepa_latent.context_target_jepa_clean_target_smoke import (  # noqa: E402
    make_clean_target_features,
    train_clean_context_target_smoke,
)


def test_clean_target_features_do_not_expose_synthetic_target_mask():
    batch = build_context_target_jepa_batch(
        split="val",
        history_len=8,
        future_len=4,
        max_windows=4,
        seed=2144,
    )
    values = torch.from_numpy(batch.clean_values)
    observed = torch.from_numpy(batch.observed_mask)
    target_mask = torch.from_numpy(batch.target_mask)

    features = make_clean_target_features(values, observed)
    n_tokens = batch.token_metadata.n_tokens

    assert features.shape == (4, 8, n_tokens * 3)
    assert torch.allclose(features[..., :n_tokens], values.float())
    assert torch.equal(features[..., n_tokens : 2 * n_tokens], observed.float())
    assert torch.equal(features[..., 2 * n_tokens :], observed.float())
    assert not torch.equal(features[..., 2 * n_tokens :], target_mask.float())


def test_clean_context_target_train_smoke_writes_artifacts(tmp_path):
    result_json = tmp_path / "context_target_clean_smoke.json"
    checkpoint = tmp_path / "context_target_clean_smoke.pt"
    args = argparse.Namespace(
        epochs=1,
        batch_size=4,
        history_len=8,
        future_len=4,
        hidden_dim=12,
        latent_dim=6,
        predictor_hidden_dim=8,
        max_train_windows=8,
        max_val_windows=4,
        lr=1e-3,
        weight_decay=1e-4,
        barlow_weight=0.05,
        barlow_offdiag_weight=0.005,
        ema_decay=0.99,
        grad_clip=1.0,
        seed=2144,
        device="cpu",
        output_json=result_json,
        checkpoint=checkpoint,
    )

    result = train_clean_context_target_smoke(args)

    assert result_json.exists()
    assert checkpoint.exists()
    assert result["objective_family"] == "context_to_target_jepa"
    assert result["target_input_mode"] == "clean_full_window"
    assert result["uses_future_targets"] is False
    assert result["uses_value_reconstruction"] is False
    assert result["train_shape"] == [8, 8, 58]
    assert result["val_shape"] == [4, 8, 58]
    assert np.isfinite(result["val_metrics"]["loss"])
