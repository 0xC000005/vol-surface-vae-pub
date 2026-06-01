import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.plot_iv_level_conditional_case_study import (  # noqa: E402
    _bootstrap_iv_cell_paths,
    _bootstrap_iv_surface_paths,
    _coverage_1d,
)


def test_bootstrap_iv_cell_paths_adds_raw_level_deltas_and_clips_positive():
    paths = _bootstrap_iv_cell_paths(
        start_level=0.20,
        train_delta=np.array([0.05, -0.40], dtype=np.float64),
        n_samples=2,
        horizon=2,
        seed=7,
        sampled_transition_indices=np.array([[0, 0], [1, 0]], dtype=np.int64),
    )

    np.testing.assert_allclose(paths[0], np.array([0.25, 0.30]), rtol=1e-6)
    np.testing.assert_allclose(paths[1], np.array([1e-4, 0.0501]), rtol=1e-6)


def test_bootstrap_iv_surface_paths_uses_paired_raw_surface_deltas():
    start = np.full((5, 5), 0.20, dtype=np.float64)
    train_delta = np.zeros((2, 5, 5), dtype=np.float64)
    train_delta[0, 0, 0] = 0.05
    train_delta[0, 4, 4] = -0.03
    train_delta[1, 0, 0] = -0.40
    train_delta[1, 4, 4] = 0.10

    paths = _bootstrap_iv_surface_paths(
        start_surface=start,
        train_delta=train_delta,
        n_samples=1,
        horizon=2,
        seed=7,
        sampled_transition_indices=np.array([[0, 1]], dtype=np.int64),
    )

    np.testing.assert_allclose(paths[0, :, 0, 0], np.array([0.25, 1e-4]), rtol=1e-6)
    np.testing.assert_allclose(paths[0, :, 4, 4], np.array([0.17, 0.27]), rtol=1e-6)


def test_coverage_1d_uses_sample_axis_for_pointwise_90pct_band():
    future = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float64)
    samples = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        ],
        dtype=np.float64,
    )

    cov = _coverage_1d(future, samples)

    np.testing.assert_allclose(cov, np.array([1.0, 0.0]), rtol=1e-6)
