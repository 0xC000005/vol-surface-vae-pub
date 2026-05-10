from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.masked_multiview_barlow_smoke import (
    build_batch_for_mask_preset,
)


def _mask_rates(batch) -> dict[str, float]:
    observed = np.asarray(batch.observed_mask, dtype=bool)
    synth_a = np.asarray(batch.synthetic_mask_a, dtype=bool)
    synth_b = np.asarray(batch.synthetic_mask_b, dtype=bool)
    return {
        "hidden_a": float(np.sum((~synth_a) & observed) / np.sum(observed)),
        "hidden_b": float(np.sum((~synth_b) & observed) / np.sum(observed)),
        "both_visible": float(np.sum(synth_a & synth_b & observed) / np.sum(observed)),
    }


def test_hard_head122_mask_preset_is_materially_harder_than_default():
    default = build_batch_for_mask_preset(
        split="val",
        history_len=30,
        future_len=30,
        max_windows=16,
        seed=1680,
        mask_preset="default",
    )
    hard = build_batch_for_mask_preset(
        split="val",
        history_len=30,
        future_len=30,
        max_windows=16,
        seed=1680,
        mask_preset="hard_head122",
    )

    default_rates = _mask_rates(default)
    hard_rates = _mask_rates(hard)

    assert hard_rates["hidden_a"] > default_rates["hidden_a"] + 0.10
    assert hard_rates["hidden_b"] > default_rates["hidden_b"] + 0.10
    assert hard_rates["both_visible"] < default_rates["both_visible"] - 0.20
    assert set(hard.mask_family_a.tolist()) <= {
        "surface_large_rectangle",
        "surface_whole_day_block",
        "factor_family_long_block",
        "cross_family_stress_block",
        "time_block_long",
    }
