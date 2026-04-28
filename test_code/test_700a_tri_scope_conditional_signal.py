import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_700a_tri_scope_conditional_signal import (  # noqa: E402
    conditional_signal_metrics,
    nearest_neighbor_future_prediction,
    ordinal_spearman,
)


def test_ordinal_spearman_detects_monotone_relationship() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 20.0, 30.0, 40.0])

    assert ordinal_spearman(x, y) == 1.0


def test_conditional_signal_metrics_reward_history_aware_prediction() -> None:
    history = np.array(
        [
            [[0.0], [1.0], [2.0]],
            [[0.0], [-1.0], [-2.0]],
            [[0.0], [0.5], [1.0]],
            [[0.0], [-0.5], [-1.0]],
        ],
        dtype=np.float32,
    )
    future = np.repeat(history[:, -1:, :], repeats=2, axis=1)

    out = conditional_signal_metrics(history, future)

    assert out["last_step_mae"] < out["rolled_future_mae"]
    assert out["last_step_improvement_vs_rolled_pct"] > 50.0
    assert out["history_future_activity_spearman"] > 0.5


def test_nearest_neighbor_future_prediction_excludes_same_index() -> None:
    reference_history = np.array(
        [
            [[0.0], [1.0]],
            [[0.0], [10.0]],
            [[0.0], [2.0]],
        ],
        dtype=np.float32,
    )
    reference_future = np.array(
        [
            [[1.0]],
            [[10.0]],
            [[2.0]],
        ],
        dtype=np.float32,
    )
    eval_history = reference_history[[0]]
    eval_indices = np.array([5])
    reference_indices = np.array([5, 6, 7])

    pred = nearest_neighbor_future_prediction(
        eval_history,
        reference_history,
        reference_future,
        eval_indices=eval_indices,
        reference_indices=reference_indices,
        k=1,
        chunk_size=2,
    )

    np.testing.assert_allclose(pred, reference_future[[2]])
