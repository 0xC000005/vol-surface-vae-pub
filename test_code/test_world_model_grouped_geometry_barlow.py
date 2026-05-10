from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    build_geometry_token_metadata,
)
from experiments.world.part1_jepa_latent.masked_multiview_grouped_geometry_barlow_smoke import (  # noqa: E402
    GEOMETRY_GROUPS,
    GroupedGeometryDirectBarlowConfig,
    GroupedGeometryDirectBarlowModel,
    build_geometry_group_indices,
    build_token_descriptor_matrix,
    grouped_geometry_barlow_loss,
)


def test_grouped_geometry_encoder_keeps_geometry_groups_separate():
    import torch

    metadata = build_geometry_token_metadata(
        level_columns=["spx"],
        return_columns=["spx_logret"],
    )
    descriptors = build_token_descriptor_matrix(metadata)
    group_indices = build_geometry_group_indices(metadata)

    assert tuple(group_indices) == GEOMETRY_GROUPS
    assert all(group_indices[name].ndim == 1 for name in GEOMETRY_GROUPS)
    assert (
        sum(group_indices[name].size for name in GEOMETRY_GROUPS) == metadata.n_tokens
    )
    assert np.isfinite(descriptors).all()

    cfg = GroupedGeometryDirectBarlowConfig(
        n_tokens=metadata.n_tokens,
        token_descriptor_dim=descriptors.shape[1],
        token_hidden_dim=8,
        hidden_dim=12,
        latent_dim=6,
    )
    model = GroupedGeometryDirectBarlowModel(
        cfg,
        token_descriptors=descriptors,
        group_indices=group_indices,
    )
    values = torch.randn(2, 3, metadata.n_tokens)
    observed = torch.ones_like(values, dtype=torch.bool)
    synth_a = torch.ones_like(values, dtype=torch.bool)
    synth_b = torch.ones_like(values, dtype=torch.bool)
    synth_a[:, :, group_indices["iv_surface"][0]] = False
    synth_b[:, 1:, group_indices["factor_return"][0]] = False
    view_a = torch.where(observed & synth_a, values, torch.zeros_like(values))
    view_b = torch.where(observed & synth_b, values, torch.zeros_like(values))

    out = model(view_a, view_b, observed, synth_a, synth_b)

    assert out["view_a"].shape == (2, 3, 6)
    assert out["view_b"].shape == (2, 3, 6)
    assert model.encoder.daily_input_dim == len(GEOMETRY_GROUPS) * cfg.token_hidden_dim
    loss, parts = grouped_geometry_barlow_loss(out)
    assert torch.isfinite(loss)
    assert set(parts) == {"barlow", "barlow_diag_loss", "barlow_offdiag_loss", "loss"}
