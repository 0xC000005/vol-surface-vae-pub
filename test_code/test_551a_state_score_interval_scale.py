import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.evaluate_551a_state_score_interval_scale_calibrated_system import (
    apply_state_interval_scaling,
    assign_state_bins,
    fit_state_horizon_scale_tables,
    state_score_from_history,
)


def test_state_score_uses_history_abs_move_quantile_only() -> None:
    quiet = np.full((1, 4, 2, 2), 0.20, dtype=np.float32)
    jumpy = quiet.copy()
    jumpy[:, 1:] += np.array([0.00, 0.05, -0.03], dtype=np.float32)[None, :, None, None]
    history = np.concatenate([quiet, jumpy], axis=0)

    score = state_score_from_history(history)

    assert score.shape == (2,)
    assert score[0] == 0.0
    assert score[1] > score[0]


def test_assign_state_bins_uses_calibration_quantiles() -> None:
    score = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    bins = assign_state_bins(score, low_q=0.15, high_q=0.35)

    assert bins.tolist() == [0, 0, 1, 1, 2]


def test_fit_state_horizon_scale_tables_widens_only_undercovered_state() -> None:
    # Two risk states, two horizons, one cell. Low-risk is already calibrated at
    # scale 1; high-risk needs wider residuals to cover targets.
    calib_history = np.zeros((4, 4, 1, 1), dtype=np.float32)
    calib_history[:2, 1:, 0, 0] = 0.01
    calib_history[2:, 1:, 0, 0] = np.array([0.00, 0.10, -0.05], dtype=np.float32)
    calib_samples = np.array(
        [
            [[[0.45]], [[0.45]]],
            [[[0.55]], [[0.55]]],
            [[[0.45]], [[0.45]]],
            [[[0.55]], [[0.55]]],
        ],
        dtype=np.float32,
    )
    calib_samples = np.repeat(calib_samples[:, None], 8, axis=1)
    offsets = np.linspace(-0.04, 0.04, 8, dtype=np.float32)[None, :, None, None, None]
    calib_samples = calib_samples + offsets
    calib_future = np.array(
        [
            [[[0.47]], [[0.53]]],
            [[[0.53]], [[0.47]]],
            [[[0.40]], [[0.60]]],
            [[[0.60]], [[0.40]]],
        ],
        dtype=np.float32,
    )

    tables = fit_state_horizon_scale_tables(
        calib_samples=calib_samples,
        calib_future=calib_future,
        calib_history=calib_history,
        target_coverage=0.90,
        coverage_lo=0.70,
        coverage_hi=0.95,
        scale_min=0.5,
        scale_max=2.0,
        scale_steps=16,
        min_bin_windows=1,
    )

    assert tables.scales.shape == (3, 2)
    assert tables.scales[2].mean() > tables.scales[0].mean()


def test_apply_state_interval_scaling_preserves_median_and_uses_bin_scale() -> None:
    samples = np.array(
        [
            [[[[0.40]], [[0.40]]], [[[0.60]], [[0.60]]]],
            [[[[0.40]], [[0.40]]], [[[0.60]], [[0.60]]]],
        ],
        dtype=np.float32,
    )
    history = np.zeros((2, 4, 1, 1), dtype=np.float32)
    history[1, 1:, 0, 0] = np.array([0.00, 0.10, -0.05], dtype=np.float32)
    tables = fit_state_horizon_scale_tables(
        calib_samples=np.repeat(samples, 2, axis=0),
        calib_future=np.full((4, 2, 1, 1), 0.50, dtype=np.float32),
        calib_history=np.repeat(history, 2, axis=0),
        target_coverage=0.90,
        coverage_lo=0.70,
        coverage_hi=0.95,
        scale_min=0.5,
        scale_max=2.0,
        scale_steps=4,
        min_bin_windows=1,
    )
    tables.scales[:] = 1.0
    tables.scales[2, :] = 1.5

    scaled = apply_state_interval_scaling(samples, history, tables, alpha=1.0)

    assert np.allclose(np.median(scaled, axis=1), np.median(samples, axis=1))
    assert scaled[1, 0, 0, 0, 0] < samples[1, 0, 0, 0, 0]
    assert scaled[1, 1, 0, 0, 0] > samples[1, 1, 0, 0, 0]
