from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.surface_local_jepa_data import (  # noqa: E402
    build_surface_local_jepa_batch,
)
from experiments.world.part1_jepa_latent.masked_multiview_geometry_barlow_smoke import (  # noqa: E402
    build_token_descriptor_matrix,
)
from experiments.world.part1_jepa_latent.surface_local_jepa_model import (  # noqa: E402
    SurfaceLocalTokenJepaConfig,
    SurfaceLocalTokenJepaModel,
    select_target_token_rows,
    surface_local_context_target_loss,
)


def test_surface_local_token_jepa_selects_clean_target_token_rows():
    batch = build_surface_local_jepa_batch(
        split="val",
        history_len=10,
        future_len=5,
        max_windows=4,
        seed=2153,
        target_families=("surface_wing_moneyness", "surface_edge_maturity"),
    )
    descriptors = build_token_descriptor_matrix(batch.token_metadata)
    cfg = SurfaceLocalTokenJepaConfig(
        n_tokens=batch.token_metadata.n_tokens,
        token_descriptor_dim=descriptors.shape[1],
        token_hidden_dim=8,
        hidden_dim=12,
        latent_dim=6,
        predictor_hidden_dim=10,
    )
    model = SurfaceLocalTokenJepaModel(cfg, token_descriptors=descriptors)

    assert all(not param.requires_grad for param in model.target_encoder.parameters())

    target_positions = torch.from_numpy(batch.target_positions)
    out = model(
        context_values=torch.from_numpy(batch.context_values),
        clean_values=torch.from_numpy(batch.clean_values),
        observed_mask=torch.from_numpy(batch.observed_mask),
        context_mask=torch.from_numpy(batch.context_mask),
        target_positions=target_positions,
    )
    loss, parts = surface_local_context_target_loss(out)

    n_targets = int(batch.target_positions.shape[0])
    assert out["context_tokens"].shape == (4, 10, batch.token_metadata.n_tokens, 6)
    assert out["predicted_target_tokens"].shape == (n_targets, 6)
    assert out["target_tokens"].shape == (n_targets, 6)
    assert torch.isfinite(loss)
    assert parts["target_token_rows"] == n_targets
    assert set(parts) == {
        "alignment",
        "barlow",
        "barlow_diag_loss",
        "barlow_offdiag_loss",
        "loss",
        "target_token_rows",
    }

    target_mask = torch.from_numpy(batch.target_mask)
    for window, time, token in target_positions:
        assert bool(target_mask[int(window), int(time), int(token)])

    selected = select_target_token_rows(out["context_tokens"], target_positions)
    assert selected.shape == (n_targets, 6)


def test_select_target_token_rows_rejects_empty_or_wrong_shape():
    token_embeddings = torch.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match="at least two target token rows"):
        select_target_token_rows(token_embeddings, torch.empty(0, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="shape \\(K, 3\\)"):
        select_target_token_rows(token_embeddings, torch.zeros(2, 2, dtype=torch.long))
