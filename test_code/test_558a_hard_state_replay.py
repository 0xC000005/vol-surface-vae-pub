import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_558a_hard_state_replay_patch_energy import (
    hard_state_scores_from_intervals,
    patch_energy_score_per_window,
    rank_replay_weights,
)


def test_hard_state_scores_increase_for_undercovered_and_upper_miss_windows() -> None:
    samples = np.array(
        [
            [[[[0.10]]], [[[0.10]]], [[[0.10]]]],
            [[[[0.00]]], [[[1.00]]], [[[2.00]]]],
        ],
        dtype=np.float64,
    )
    future = np.array([[[[0.50]]], [[[1.00]]]], dtype=np.float64)

    scores, metrics = hard_state_scores_from_intervals(samples, future)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    assert metrics["upper_miss_rate"][0] == 1.0
    assert metrics["coverage90"][1] == 1.0


def test_rank_replay_weights_are_mean_one_and_monotone() -> None:
    scores = np.array([0.0, 2.0, 1.0, 3.0], dtype=np.float64)

    weights = rank_replay_weights(scores, strength=2.0)

    assert np.isclose(weights.mean(), 1.0)
    assert weights[np.argmax(scores)] > weights[np.argmin(scores)]


def test_patch_energy_score_per_window_preserves_batch_dimension() -> None:
    samples = torch.zeros(2, 3, 2, 1)
    target = torch.tensor([[[0.0], [0.0]], [[1.0], [1.0]]])
    samples[1] = 1.0

    score, target_dist, pair_dist = patch_energy_score_per_window(
        samples,
        target,
        patch_len=1,
        eps=1e-6,
    )

    assert score.shape == (2,)
    assert target_dist.shape == (2,)
    assert pair_dist.shape == (2,)
    assert torch.all(score < 1e-2)
