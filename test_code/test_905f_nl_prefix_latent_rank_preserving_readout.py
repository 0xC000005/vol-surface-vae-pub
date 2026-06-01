import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_rank_preserving_readout import (
    fit_rank_preserving_alpha_map,
    scale_samples_by_alpha_map,
)


def test_scale_samples_by_alpha_map_preserves_variable_ranks() -> None:
    samples = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[2.0, 11.0], [4.0, 21.0]],
            [[3.0, 12.0], [8.0, 22.0]],
        ],
        dtype=np.float32,
    )
    alpha = np.asarray([[2.0, 1.5], [1.25, 3.0]], dtype=np.float32)

    scaled = scale_samples_by_alpha_map(samples, alpha)

    assert scaled.shape == samples.shape
    for step in range(samples.shape[1]):
        for col in range(samples.shape[2]):
            assert np.array_equal(
                np.argsort(samples[:, step, col]),
                np.argsort(scaled[:, step, col]),
            )


def test_scale_samples_by_alpha_map_rejects_non_positive_alpha() -> None:
    samples = np.ones((2, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="positive"):
        scale_samples_by_alpha_map(samples, np.zeros((1, 1), dtype=np.float32))


def test_fit_rank_preserving_alpha_map_selects_coverage_alpha(tmp_path: Path) -> None:
    arrays_path = tmp_path / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        samples=np.asarray([[[[-1.0]], [[0.0]], [[1.0]]]], dtype=np.float32),
        delta_scale=np.ones((1, 1), dtype=np.float32),
    )
    row = {
        "start_window_index": 0,
        "artifacts": {"component_prefix_mixture": {"arrays": str(arrays_path)}},
    }
    history_raw = np.zeros((1, 1, 1), dtype=np.float32)
    future_raw = np.asarray([[[1.2]]], dtype=np.float32)

    fit = fit_rank_preserving_alpha_map(
        [row],
        alpha_grid=[1.0, 2.0],
        history_raw=history_raw,
        future_raw=future_raw,
        target_coverage=1.0,
    )

    assert fit["alpha_map"].shape == (1, 1)
    assert fit["alpha_map"][0, 0] == pytest.approx(2.0)
    assert fit["chosen_coverage"][0, 0] == pytest.approx(1.0)
