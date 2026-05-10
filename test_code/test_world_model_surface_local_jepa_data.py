from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.surface_local_jepa_data import (  # noqa: E402
    build_surface_local_jepa_batch,
)


def test_surface_local_batch_preserves_target_token_geometry():
    batch = build_surface_local_jepa_batch(
        split="val",
        history_len=12,
        future_len=6,
        max_windows=8,
        seed=2150,
        target_families=("surface_wing_moneyness", "surface_edge_maturity"),
    )

    assert batch.clean_values.shape == (8, 12, batch.token_metadata.n_tokens)
    assert batch.context_values.shape == batch.clean_values.shape
    assert batch.target_mask.shape == batch.clean_values.shape
    assert batch.context_mask.shape == batch.clean_values.shape
    assert batch.relative_index.tolist() == list(range(12))
    assert batch.metadata["objective_family"] == "token_geometry_level_context_to_target_jepa"
    assert batch.metadata["uses_future_targets"] is False
    assert batch.metadata["target_representation_surface"] == "token_geometry"

    observed_targets = batch.target_mask & batch.observed_mask
    assert observed_targets.any(axis=(1, 2)).all()
    assert np.all(batch.context_values[observed_targets] == 0.0)
    assert np.allclose(
        batch.target_values[observed_targets],
        batch.clean_values[observed_targets],
    )
    assert batch.target_positions.shape[1] == 3
    assert batch.target_positions.shape[0] == int(observed_targets.sum())

    meta = batch.token_metadata
    for window, _time, token in batch.target_positions:
        assert observed_targets[int(window), int(_time), int(token)]
        assert meta.geometry_id[int(token)] == "iv_surface"

    wing_rows = batch.target_family.astype(str) == "surface_wing_moneyness"
    edge_rows = batch.target_family.astype(str) == "surface_edge_maturity"
    assert wing_rows.any()
    assert edge_rows.any()

    for token in np.flatnonzero(batch.target_mask[wing_rows].any(axis=(0, 1))):
        coord = meta.geometry_coord[int(token)]
        assert int(coord[0]) in {0, 4}
    for token in np.flatnonzero(batch.target_mask[edge_rows].any(axis=(0, 1))):
        coord = meta.geometry_coord[int(token)]
        assert int(coord[1]) in {0, 4}
