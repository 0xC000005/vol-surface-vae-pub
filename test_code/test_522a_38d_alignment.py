import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.baselines.data_loader_38d import load_aligned_38d_data
from experiments.backfill.baselines.evaluate_baselines_38d import reconstruct_iv_surfaces
from experiments.backfill.block_ar._rollout_220_utils import build_rollout_windows, select_rollout_indices
from experiments.backfill.block_ar.evaluate_522a_38d_full11_bridge import (
    build_official_aligned_38d_windows,
)


def test_522a_38d_windows_align_to_official_full_suite_future():
    data = load_aligned_38d_data()
    windows = build_official_aligned_38d_windows(
        data=data,
        test_start=4511,
        val_size=441,
        history_len=30,
        future_len=30,
        max_windows=3,
    )
    batch = build_rollout_windows(
        data_path="data/vol_surface_with_ret.npz",
        history_len=30,
        future_len=30,
        test_start=4511,
        val_size=441,
        max_windows=3,
        device=torch.device("cpu"),
        split="val",
    )

    reconstructed_future = reconstruct_iv_surfaces(
        windows["future_changes"][:, None, :, :25],
        windows["anchor_surfaces"],
    ).squeeze(1)

    np.testing.assert_allclose(
        windows["history_surfaces"],
        batch.history_01.numpy(),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        reconstructed_future,
        batch.future_01.numpy(),
        atol=1e-7,
    )
    assert windows["history_changes"].shape == (3, 30, 38)


def test_select_rollout_indices_supports_train_tail_split():
    kwargs = dict(test_start=100, val_size=10, history_len=5, future_len=5)

    np.testing.assert_array_equal(
        select_rollout_indices(**kwargs, max_windows=3, split="train"),
        np.array([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        select_rollout_indices(**kwargs, max_windows=3, split="train_tail"),
        np.array([77, 78, 79]),
    )
    np.testing.assert_array_equal(
        select_rollout_indices(**kwargs, max_windows=3, split="val"),
        np.array([80, 81, 82]),
    )
