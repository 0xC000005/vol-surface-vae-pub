import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    corr_similarity,
    ks_statistic,
    safe_corrcoef,
    summarize_joint_quality,
)


def test_ks_statistic_is_zero_for_identical_samples() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert ks_statistic(x, x.copy()) == 0.0


def test_safe_corrcoef_handles_constant_columns() -> None:
    x = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    corr = safe_corrcoef(x)
    assert corr.shape == (2, 2)
    assert np.isfinite(corr).all()
    assert corr[0, 0] == 1.0


def test_corr_similarity_reports_perfect_match() -> None:
    matrix = np.array([[1.0, 0.2, -0.1], [0.2, 1.0, 0.3], [-0.1, 0.3, 1.0]])
    out = corr_similarity(matrix, matrix.copy())
    assert np.isclose(out["upper_corr"], 1.0)
    assert np.isclose(out["mae"], 0.0)


def test_summarize_joint_quality_shapes() -> None:
    rng = np.random.default_rng(627)
    history = rng.normal(size=(4, 3, 5)).astype(np.float32)
    future = rng.normal(size=(4, 2, 5)).astype(np.float32)
    samples = future[:, None, :, :] + 0.01 * rng.normal(size=(4, 3, 2, 5)).astype(np.float32)

    summary = summarize_joint_quality(
        history,
        future,
        samples,
        factor_names=["f0", "f1"],
        iv_count=3,
    )

    assert summary["n_factors"] == 2
    assert summary["finite_rate"] == 1.0
    assert len(summary["per_factor"]) == 2
    assert "iv_factor_corr" in summary

