import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_605a_hard_window_support import (  # noqa: E402
    future_cell_delta,
    nearest_distance,
    nearest_self_distance,
    percentile_rank,
)


def test_percentile_rank_reports_fraction_at_or_below_value() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])

    assert percentile_rank(2.5, reference) == 0.5
    assert percentile_rank(4.0, reference) == 1.0


def test_nearest_distance_returns_zero_for_exact_match() -> None:
    reference = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    query = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)

    dist = nearest_distance(reference, query)

    assert dist.shape == (2,)
    assert np.isclose(dist[0], 0.0)
    assert dist[1] > dist[0]


def test_nearest_self_distance_excludes_the_same_row() -> None:
    reference = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)

    dist = nearest_self_distance(reference)

    assert np.allclose(dist, [5.0, 5.0, 5.0])


def test_future_cell_delta_uses_last_history_level() -> None:
    history = np.zeros((2, 3, 2, 2), dtype=np.float32)
    history[:, -1, 1, 0] = np.array([0.2, 0.5], dtype=np.float32)
    future = np.zeros((2, 4, 2, 2), dtype=np.float32)
    future[:, 2, 1, 0] = np.array([0.1, 0.8], dtype=np.float32)

    delta = future_cell_delta(history, future, horizon=3, row=1, col=0)

    assert np.allclose(delta, [-0.1, 0.3])
