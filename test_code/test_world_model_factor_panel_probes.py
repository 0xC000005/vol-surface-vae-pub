from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.evaluation.factor_panel_data import (  # noqa: E402
    build_factor_panel_world_windows,
    make_factor_panel_future_targets,
)


def test_factor_panel_world_windows_build_downstream_future_targets(tmp_path):
    path = tmp_path / "multi_factor_data.npz"
    levels = np.stack(
        [
            np.arange(12, dtype=np.float32),
            np.arange(12, dtype=np.float32) + 10.0,
        ],
        axis=1,
    )
    returns = (np.arange(12, dtype=np.float32) + 100.0)[:, None]
    np.savez(
        path,
        levels=levels,
        level_columns=np.asarray(["spx", "vix"]),
        returns=returns,
        return_columns=np.asarray(["spx_ret"]),
    )

    batch = build_factor_panel_world_windows(
        data_path=path,
        split="train",
        history_len=3,
        future_len=2,
        test_start=10,
        val_size=2,
        max_windows=2,
        normalize=False,
    )

    assert batch.past_panel.shape == (2, 3, 3)
    assert batch.future_panel.shape == (2, 2, 3)
    assert batch.columns == [
        "factor_level:spx",
        "factor_level:vix",
        "factor_return:spx_ret",
    ]
    assert batch.start_index.tolist() == [0, 1]

    targets = make_factor_panel_future_targets(
        batch.past_panel,
        batch.future_panel,
        columns=batch.columns,
    )

    assert (
        targets["metadata"]["target_scope"]
        == "factor_panel_future_downstream_probe_only"
    )
    assert targets["metadata"]["columns"] == batch.columns
    assert set(targets["regression"]) == {
        "factor_future_mean_delta",
        "factor_future_range",
        "factor_future_terminal_delta",
        "factor_future_max_abs_step",
    }

    expected_mean_delta = np.asarray([1.5, 1.5, 1.5], dtype=np.float32)
    assert np.allclose(
        targets["regression"]["factor_future_mean_delta"][0],
        expected_mean_delta,
    )
    assert np.allclose(
        targets["regression"]["factor_future_terminal_delta"][0],
        np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
    )
