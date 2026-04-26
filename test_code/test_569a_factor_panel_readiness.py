import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_569a_factor_panel_readiness import (
    build_broad_factor_panel,
    build_local_factor_panel,
    fill_missing_by_column,
    window_boundary_summary,
)


def test_build_local_factor_panel_stacks_iv_and_local_features() -> None:
    surface = np.arange(3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
    ret = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    price = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    slopes = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    skews = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    levels = np.array([0.2, 0.3, 0.4], dtype=np.float32)

    panel, names = build_local_factor_panel(surface, ret, price, slopes, skews, levels)

    assert panel.shape == (3, 30)
    assert names[:2] == ["iv_0_0", "iv_0_1"]
    assert names[-5:] == ["ret", "price", "slopes", "skews", "levels"]
    assert np.allclose(panel[:, -5], ret)
    assert np.allclose(panel[:, -1], levels)


def test_build_broad_factor_panel_stacks_iv_levels_and_factor_returns() -> None:
    surface = np.zeros((4, 5, 5), dtype=np.float32)
    levels = np.ones((4, 2), dtype=np.float32)
    returns = np.full((4, 2), 0.5, dtype=np.float32)

    panel, names = build_broad_factor_panel(
        surface,
        levels,
        returns,
        level_names=["spx", "gold"],
        return_names=["spx_logret", "gold_logret"],
    )

    assert panel.shape == (4, 29)
    assert names[24] == "iv_4_4"
    assert names[25:] == [
        "factor_level:spx",
        "factor_level:gold",
        "factor_return:spx_logret",
        "factor_return:gold_logret",
    ]


def test_window_boundary_summary_prevents_test_leakage() -> None:
    summary = window_boundary_summary(
        n_obs=6000,
        history_len=30,
        future_lens=[30, 60, 252],
        test_start=4511,
        val_size=441,
    )

    assert summary["30"]["train_windows"] == 4010
    assert summary["30"]["val_windows"] == 441
    assert summary["30"]["last_val_future_end_exclusive"] == 4511
    assert summary["60"]["last_val_future_end_exclusive"] == 4511
    assert summary["252"]["last_val_future_end_exclusive"] == 4511
    assert all(item["no_test_leakage"] for item in summary.values())


def test_fill_missing_by_column_uses_forward_then_backward_fill() -> None:
    panel = np.array(
        [
            [np.nan, 1.0],
            [2.0, np.nan],
            [np.nan, 3.0],
        ],
        dtype=np.float32,
    )

    filled, stats = fill_missing_by_column(panel, names=["a", "b"])

    assert np.allclose(filled, np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    assert stats["finite_rate_before"] == 3 / 6
    assert stats["finite_rate_after"] == 1.0
    assert stats["columns_with_missing"] == ["a", "b"]
