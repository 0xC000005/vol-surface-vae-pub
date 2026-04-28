import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    conditional_panel_diagnostics,
    corr_similarity,
    ks_statistic,
    safe_corrcoef,
    state_block_alignment_diagnostics,
    summarize_joint_quality,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    UnifiedVariableSpec,
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
    assert "conditional_panel" in summary


def test_state_block_alignment_diagnostics_detects_exact_panel_alignment() -> None:
    class Block:
        pass

    panel = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    specs = [
        UnifiedVariableSpec("a", "a", 0, "diff_level"),
        UnifiedVariableSpec("c", "c", 2, "diff_level"),
    ]
    block = Block()
    block.indices = np.array([1, 2])
    state_panel = panel[:, [0, 2]]
    block.history_state = np.stack([state_panel[1:4], state_panel[2:5]]).astype(np.float32)
    block.future_state = np.stack([state_panel[4:6], state_panel[5:7]]).astype(np.float32)

    out = state_block_alignment_diagnostics(panel, block, specs)

    assert out["history_max_abs_error"] == 0.0
    assert out["future_max_abs_error"] == 0.0
    assert out["n_windows"] == 2


def test_conditional_panel_diagnostics_rewards_matched_scenario_centers() -> None:
    history = np.zeros((4, 2, 2), dtype=np.float32)
    future = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ],
        dtype=np.float32,
    )
    samples = future[:, None, :, :] + np.array([-0.1, 0.1], dtype=np.float32)[None, :, None, None]

    out = conditional_panel_diagnostics(history, future, samples)

    assert out["median_mae_reduction_vs_rolled_pct"] > 50.0
    assert out["conditional_median_mae"] < out["rolled_median_mae"]
