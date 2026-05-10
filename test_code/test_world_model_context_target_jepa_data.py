from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.context_target_jepa_data import (  # noqa: E402
    build_context_target_jepa_batch,
)


def test_context_target_batch_masks_same_window_targets_only():
    batch = build_context_target_jepa_batch(
        split="val",
        history_len=12,
        future_len=6,
        max_windows=8,
        seed=1937,
    )

    assert batch.clean_values.shape == (8, 12, batch.token_metadata.n_tokens)
    assert batch.context_values.shape == batch.clean_values.shape
    assert batch.target_values.shape == batch.clean_values.shape
    assert batch.context_mask.shape == batch.clean_values.shape
    assert batch.target_mask.shape == batch.clean_values.shape
    assert batch.relative_index.tolist() == list(range(12))
    assert batch.metadata["objective_family"] == "context_to_target_jepa"
    assert batch.metadata["uses_future_targets"] is False

    observed_targets = batch.target_mask & batch.observed_mask
    assert observed_targets.any(axis=(1, 2)).all()
    assert np.all(batch.context_values[observed_targets] == 0.0)
    assert np.allclose(
        batch.target_values[observed_targets],
        batch.clean_values[observed_targets],
    )
    assert np.all(batch.target_values[~observed_targets] == 0.0)
    assert set(batch.target_family.tolist()) <= {
        "surface_maturity",
        "surface_moneyness",
        "surface_rectangle",
        "vol_side_channel",
        "factor_family",
        "time_block",
    }
