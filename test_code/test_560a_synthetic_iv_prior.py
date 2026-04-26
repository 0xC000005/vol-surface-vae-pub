import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.synthetic_iv_prior import (
    SyntheticIVPriorConfig,
    generate_synthetic_iv_windows,
)


def test_synthetic_iv_prior_is_deterministic_for_fixed_seed() -> None:
    cfg = SyntheticIVPriorConfig(
        n_windows=8,
        history_len=6,
        future_len=4,
        seed=123,
    )

    first = generate_synthetic_iv_windows(cfg)
    second = generate_synthetic_iv_windows(cfg)

    assert torch.allclose(first.history, second.history)
    assert torch.allclose(first.future, second.future)
    assert torch.equal(first.regime_labels, second.regime_labels)


def test_synthetic_iv_prior_shapes_and_bounds_match_iv_windows() -> None:
    cfg = SyntheticIVPriorConfig(
        n_windows=11,
        history_len=7,
        future_len=5,
        seed=7,
        min_iv=0.04,
        max_iv=0.92,
    )

    batch = generate_synthetic_iv_windows(cfg)

    assert batch.history.shape == (11, 7, 5, 5)
    assert batch.future.shape == (11, 5, 5, 5)
    assert batch.full_sequence.shape == (11, 12, 5, 5)
    assert batch.history.dtype == torch.float32
    assert float(batch.history.min()) >= 0.04 - 1e-6
    assert float(batch.history.max()) <= 0.92 + 1e-6
    assert float(batch.future.min()) >= 0.04 - 1e-6
    assert float(batch.future.max()) <= 0.92 + 1e-6


def test_synthetic_iv_prior_exposes_calm_and_stress_regimes() -> None:
    cfg = SyntheticIVPriorConfig(
        n_windows=96,
        history_len=10,
        future_len=8,
        seed=42,
        stress_prob=0.35,
    )

    batch = generate_synthetic_iv_windows(cfg)
    labels = batch.regime_labels

    assert labels.shape == (96,)
    assert int((labels == 0).sum()) > 0
    assert int((labels == 1).sum()) > 0

    calm_future_mean = batch.future[labels == 0].mean()
    stress_future_mean = batch.future[labels == 1].mean()
    assert float(stress_future_mean - calm_future_mean) > 0.03
