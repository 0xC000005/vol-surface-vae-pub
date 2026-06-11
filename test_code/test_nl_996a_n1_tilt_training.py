"""Unit tests for the 996a N1 tilt teacher-label processing.

Pre-registered properties under test:
  (a) global calm-debiasing removes the replay-CRPS ~ candidate-future-activity
      correlation (the F1 calm shortcut);
  (b) noise-banding drops sub-noise pairwise contrasts and keeps supra-noise
      ones, with the per-query band max(0.048, 0.5 * pool replay std);
  (c) pair orientation: the first index of a banded pair is the BETTER
      (lower debiased residual) candidate;
  (d) the chassis-equivalent tilt selection simulation reproduces start-only
      top-3 at zero tilt and follows the tilt at extreme tilt weight.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_996a_n1_tilt_training import (
    NOISE_FLOOR,
    banded_pairs,
    chassis_terminal_z,
    contiguous_folds,
    global_calm_debias,
    noise_bands,
    per_query_calm_debias,
    replay_proxy_crps,
    simulate_tilt_selection,
    within_pool_z,
)


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)


# (a) calm-debiasing ----------------------------------------------------------


def test_global_debias_removes_activity_correlation() -> None:
    rng = np.random.default_rng(0)
    n_q, pool = 200, 50
    activity = rng.uniform(0.2, 2.0, size=(n_q, pool))
    signal = rng.normal(0.0, 0.05, size=(n_q, pool))
    replay = 0.3 + 0.45 * activity + signal

    raw_rho = _spearman(replay.reshape(-1), activity.reshape(-1))
    residual, stats = global_calm_debias(replay, activity)
    debiased_rho = _spearman(residual.reshape(-1), activity.reshape(-1))

    assert raw_rho > 0.5  # the planted calm shortcut is visible pre-debias
    assert abs(debiased_rho) < 0.05  # and gone post-debias
    assert stats["slope"] == pytest.approx(0.45, abs=0.01)
    assert stats["n_rows"] == n_q * pool


def test_global_debias_preserves_non_activity_signal() -> None:
    rng = np.random.default_rng(1)
    n_q, pool = 100, 50
    activity = rng.uniform(0.2, 2.0, size=(n_q, pool))
    quality = rng.normal(0.0, 0.1, size=(n_q, pool))  # the real teacher signal
    replay = 0.3 + 0.5 * activity + quality

    residual, _ = global_calm_debias(replay, activity)
    rho = _spearman(residual.reshape(-1), quality.reshape(-1))
    assert rho > 0.9  # debiasing must not destroy the matching signal


def test_per_query_debias_centers_each_pool() -> None:
    rng = np.random.default_rng(2)
    replay = rng.uniform(0.2, 1.0, size=(20, 50))
    activity = rng.uniform(0.2, 2.0, size=(20, 50))
    residual, stats = per_query_calm_debias(replay, activity)
    assert residual.shape == replay.shape
    np.testing.assert_allclose(residual.mean(axis=1), 0.0, atol=1e-10)
    assert "slope_mean" in stats


# (b) noise-banding ------------------------------------------------------------


def test_noise_band_floor_and_std_fraction() -> None:
    stds = np.asarray([0.0, 0.05, 0.096, 0.2, 1.0])
    bands = noise_bands(stds)
    expected = np.maximum(NOISE_FLOOR, 0.5 * stds)
    np.testing.assert_allclose(bands, expected)
    assert bands[0] == pytest.approx(NOISE_FLOOR)  # floor binds for tiny std
    assert bands[-1] == pytest.approx(0.5)  # 0.5 * std binds for large std


def test_banding_drops_sub_noise_pairs_and_keeps_supra_noise_pairs() -> None:
    # residuals: 0.0, 0.03, 0.30 with band 0.048:
    #   (0, 1): |delta|=0.03 < band  -> DROPPED (twin-noise)
    #   (0, 2): |delta|=0.30 > band  -> kept
    #   (1, 2): |delta|=0.27 > band  -> kept
    residuals = np.asarray([0.0, 0.03, 0.30])
    better, worse = banded_pairs(residuals, 0.048)
    kept = set(zip(better.tolist(), worse.tolist()))
    assert (0, 1) not in kept and (1, 0) not in kept
    assert (0, 2) in kept
    assert (1, 2) in kept
    assert len(kept) == 2


def test_banding_exact_band_boundary_is_dropped() -> None:
    residuals = np.asarray([0.0, 0.048])
    better, worse = banded_pairs(residuals, 0.048)
    assert better.size == 0 and worse.size == 0  # strict > band


def test_banding_with_huge_band_drops_everything() -> None:
    rng = np.random.default_rng(3)
    residuals = rng.normal(0.0, 0.01, size=50)
    better, worse = banded_pairs(residuals, 1.0)
    assert better.size == 0 and worse.size == 0


# (c) pair orientation ----------------------------------------------------------


def test_banded_pair_orientation_first_index_is_better() -> None:
    residuals = np.asarray([0.5, 0.1, 0.9])
    better, worse = banded_pairs(residuals, 0.05)
    for b, w in zip(better.tolist(), worse.tolist()):
        assert residuals[b] < residuals[w]


# (d) chassis-equivalent tilt selection -----------------------------------------


def _toy_pool():
    windows = np.asarray([100, 200, 300, 400])
    start_scores = np.asarray([-1.0, -2.0, -3.0, -4.0])  # locality order
    return windows, start_scores


def test_zero_tilt_reproduces_start_only_top3() -> None:
    windows, start_scores = _toy_pool()
    selected, weights = simulate_tilt_selection(
        pool_windows=windows,
        chassis_start_scores=start_scores,
        tilt_scores_z=np.zeros(4),
        tilt_weight=0.0,
        chassis_temperature=1.0,
    )
    assert selected.tolist() == [0, 1, 2]
    expected = np.exp(start_scores[:3] - start_scores[0])
    np.testing.assert_allclose(weights, expected / expected.sum(), atol=1e-12)


def test_extreme_tilt_follows_student_score() -> None:
    windows, start_scores = _toy_pool()
    tilt = np.asarray([0.0, 0.0, 0.0, 10.0])
    selected, weights = simulate_tilt_selection(
        pool_windows=windows,
        chassis_start_scores=start_scores,
        tilt_scores_z=tilt,
        tilt_weight=100.0,
        chassis_temperature=1.0,
    )
    assert selected[0] == 3  # tilted candidate promoted to rank 1
    assert weights[0] == pytest.approx(1.0, abs=1e-6)


def test_proxy_crps_is_weighted_mean() -> None:
    replay = np.asarray([0.4, 0.6, 0.8, 1.0])
    proxy = replay_proxy_crps(
        replay, np.asarray([0, 2]), np.asarray([0.25, 0.75])
    )
    assert proxy == pytest.approx(0.25 * 0.4 + 0.75 * 0.8)


# helpers ------------------------------------------------------------------------


def test_contiguous_folds_are_contiguous_and_cover_all() -> None:
    folds = contiguous_folds(1000, 5)
    assert len(folds) == 5
    flat = np.concatenate(folds)
    np.testing.assert_array_equal(flat, np.arange(1000))
    for fold in folds:
        np.testing.assert_array_equal(fold, np.arange(fold[0], fold[-1] + 1))


def test_within_pool_z_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(2.0, 3.0, size=(10, 50))
    z = within_pool_z(values)
    np.testing.assert_allclose(z.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=1), 1.0, atol=1e-10)


def test_within_pool_z_degenerate_pool_is_zero() -> None:
    z = within_pool_z(np.full((2, 5), 3.0))
    np.testing.assert_allclose(z, 0.0)


def test_chassis_terminal_z_matches_sqrt_d_scaling() -> None:
    # With no NaNs, the chassis L2 distance over z-scored terminals equals
    # sqrt(D) times the 994b RMS z-distance (same per-dim std, mean cancels).
    rng = np.random.default_rng(5)
    n, t, d = 40, 30, 7
    history = rng.normal(0.0, 1.0, size=(n, t, d)).astype(np.float32)
    fit = np.arange(n)
    z = chassis_terminal_z(history, fit)
    terminal = history[:, -1, :].astype(np.float64)
    scale = terminal[fit].std(axis=0)
    q, c = 0, 1
    l2 = float(np.linalg.norm(z[c] - z[q]))
    rms = float(np.sqrt(np.mean(((terminal[c] - terminal[q]) / scale) ** 2)))
    assert l2 == pytest.approx(np.sqrt(d) * rms, rel=1e-6)
