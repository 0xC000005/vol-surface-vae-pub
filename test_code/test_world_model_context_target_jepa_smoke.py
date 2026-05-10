from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.context_target_jepa_data import (  # noqa: E402
    build_context_target_jepa_batch,
)
from experiments.world.part1_jepa_latent.context_target_jepa_smoke import (  # noqa: E402
    ContextTargetJEPAConfig,
    ContextTargetJEPAModel,
    context_target_jepa_loss,
    make_context_target_features,
)


def test_context_target_jepa_forward_and_loss_are_target_masked():
    batch = build_context_target_jepa_batch(
        split="val",
        history_len=10,
        future_len=5,
        max_windows=4,
        seed=2139,
    )
    cfg = ContextTargetJEPAConfig(
        token_dim=batch.token_metadata.n_tokens,
        input_dim=batch.token_metadata.n_tokens * 3,
        hidden_dim=12,
        latent_dim=6,
        predictor_hidden_dim=8,
    )
    model = ContextTargetJEPAModel(cfg)

    context_features = make_context_target_features(
        torch.from_numpy(batch.context_values),
        torch.from_numpy(batch.observed_mask),
        torch.from_numpy(batch.context_mask),
    )
    target_features = make_context_target_features(
        torch.from_numpy(batch.target_values),
        torch.from_numpy(batch.observed_mask),
        torch.from_numpy(batch.target_mask),
    )
    out = model(context_features, target_features)
    loss, parts = context_target_jepa_loss(
        out,
        torch.from_numpy(batch.target_mask),
    )

    assert out["context"].shape == (4, 10, 6)
    assert out["predicted"].shape == (4, 10, 6)
    assert out["target"].shape == (4, 10, 6)
    assert torch.isfinite(loss)
    assert parts["target_time_rows"] > 0
    assert set(parts) == {
        "alignment",
        "barlow",
        "barlow_diag_loss",
        "barlow_offdiag_loss",
        "loss",
        "target_time_rows",
    }
