import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_572a_joint_panel_quality import (
    ks_statistic_1d,
    quantile_scale_ratio,
    reconstruct_factor_levels_from_returns,
    summarize_factor_quality,
)


def test_ks_statistic_1d_detects_identical_and_shifted_samples() -> None:
    base = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    assert ks_statistic_1d(base, base.copy()) == 0.0
    shifted = base + 10.0
    assert ks_statistic_1d(base, shifted) > 0.9


def test_quantile_scale_ratio_handles_zero_ground_truth_scale() -> None:
    gt = np.zeros((2, 3), dtype=np.float64)
    gen = np.ones((4, 3), dtype=np.float64)
    ratio = quantile_scale_ratio(gt, gen, q=0.99)
    assert np.isfinite(ratio)
    assert ratio > 1.0


def test_summarize_factor_quality_separates_level_and_return_channels() -> None:
    rng = np.random.default_rng(7)
    gt = rng.normal(size=(5, 4, 5)).astype(np.float32)
    gen = np.repeat(gt[:, None], 3, axis=1)
    columns = [
        "iv:00",
        "factor:spx",
        "factor:gold",
        "factor:spx_logret",
        "factor:gold_diff",
    ]
    summary = summarize_factor_quality(gt, gen, columns=columns, iv_count=1)
    assert summary["factor_level_count"] == 2
    assert summary["factor_return_count"] == 2
    assert summary["finite_rate"] == 1.0
    assert summary["level_ks_median"] == 0.0
    assert summary["return_ks_median"] == 0.0


def test_reconstruct_factor_levels_from_returns_uses_last_history_level() -> None:
    columns = [
        "iv:00",
        "factor:spx",
        "factor:rate",
        "factor:spx_logret",
        "factor:rate_diff",
    ]
    history = np.zeros((1, 2, 5), dtype=np.float32)
    history[0, -1, 1] = 100.0
    history[0, -1, 2] = 5.0
    samples = np.zeros((1, 1, 2, 5), dtype=np.float32)
    samples[0, 0, :, 3] = np.log([1.10, 1.20])
    samples[0, 0, :, 4] = [0.5, -0.25]
    reconstructed = reconstruct_factor_levels_from_returns(
        history,
        samples,
        columns=columns,
        iv_count=1,
    )
    assert np.allclose(reconstructed[0, 0, :, 1], [110.0, 132.0], rtol=1e-5)
    assert np.allclose(reconstructed[0, 0, :, 2], [5.5, 5.25], rtol=1e-5)
