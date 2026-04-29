import numpy as np
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_iv_ewma_economic_link_tests,
)


def test_iv_ewma_economic_link_passes_when_generated_relation_matches_gt() -> None:
    n_windows, n_samples, horizon, rows, cols = 16, 4, 30, 5, 5
    returns = np.linspace(-0.03, 0.035, n_windows + 60)
    ewma = _ewma_for_windows(returns, n_windows=n_windows, history_len=30, future_len=horizon)
    grid_offsets = np.linspace(0.0, 0.024, rows * cols).reshape(rows, cols)
    gt = 0.20 + 1.3 * ewma[:, :, None, None] + grid_offsets[None, None, :, :]
    gt = np.clip(gt, 0.001, 0.99)
    samples = np.repeat(gt[:, None], n_samples, axis=1)
    samples += np.linspace(-0.002, 0.002, n_samples)[None, :, None, None, None]

    result = run_iv_ewma_economic_link_tests(
        samples,
        gt,
        returns=returns,
        test_start=0,
        history_len=30,
        future_len=horizon,
    )

    assert result["overall_pass"] is True
    assert result["slope_ratio_pass"] is True
    assert result["spearman_pass"] is True
    assert result["r2_ratio_pass"] is True
    assert result["gt_slope"] > 0
    assert 0.95 <= result["slope_ratio"] <= 1.05


def test_iv_ewma_economic_link_fails_when_generated_relation_is_flat() -> None:
    n_windows, n_samples, horizon, rows, cols = 16, 4, 30, 5, 5
    returns = np.linspace(-0.03, 0.035, n_windows + 60)
    ewma = _ewma_for_windows(returns, n_windows=n_windows, history_len=30, future_len=horizon)
    grid_offsets = np.linspace(0.0, 0.024, rows * cols).reshape(rows, cols)
    gt = 0.20 + 1.3 * ewma[:, :, None, None] + grid_offsets[None, None, :, :]
    gt = np.clip(gt, 0.001, 0.99)
    samples = np.full((n_windows, n_samples, horizon, rows, cols), float(np.mean(gt)))

    result = run_iv_ewma_economic_link_tests(
        samples,
        gt,
        returns=returns,
        test_start=0,
        history_len=30,
        future_len=horizon,
    )

    assert result["overall_pass"] is False
    assert result["slope_ratio_pass"] is False
    assert result["spearman_pass"] is False


def _ewma_for_windows(
    returns: np.ndarray,
    *,
    n_windows: int,
    history_len: int,
    future_len: int,
    ewma_lambda: float = 0.94,
) -> np.ndarray:
    rows = []
    for win_idx in range(n_windows):
        history = returns[win_idx : win_idx + history_len]
        future = returns[win_idx + history_len : win_idx + history_len + future_len]
        var = history[0] ** 2
        for ret in history[1:]:
            var = ewma_lambda * var + (1.0 - ewma_lambda) * ret**2
        path = []
        for ret in future:
            var = ewma_lambda * var + (1.0 - ewma_lambda) * ret**2
            path.append(np.sqrt(var * 252.0))
        rows.append(path)
    return np.asarray(rows, dtype=np.float64)
