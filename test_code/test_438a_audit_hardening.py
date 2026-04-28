import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    rollout_start_for_split,
)


def test_rollout_start_for_split_matches_selected_window_indices() -> None:
    kwargs = dict(test_start=100, val_size=10, history_len=5, future_len=5)

    assert rollout_start_for_split(**kwargs, n_windows=3, eval_split="train") == 0
    assert rollout_start_for_split(**kwargs, n_windows=3, eval_split="train_tail") == 77
    assert rollout_start_for_split(**kwargs, n_windows=3, eval_split="val") == 80


def test_fixed_deployable_sampler_raises_on_missing_history_key() -> None:
    known_history = np.zeros((2, 5, 5), dtype=np.float32)
    known_samples = np.zeros((4, 3, 5, 5), dtype=np.float32)
    sampler = FixedDeployableSampler({history_key(known_history): known_samples})

    missing_history = torch.ones((1, 2, 5, 5), dtype=torch.float32)
    with pytest.raises(KeyError, match="missing history key"):
        sampler.sample_batched(missing_history, n_samples=2, n_steps=3)
