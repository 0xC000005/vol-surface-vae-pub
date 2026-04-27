import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.evaluate_607a_support_aware_path_fallback import (  # noqa: E402
    inject_increment_fallback,
    select_increment_pool,
)


def test_select_increment_pool_keeps_high_severity_training_paths() -> None:
    train_history = np.zeros((3, 2, 1, 1), dtype=np.float32)
    train_future = np.array(
        [
            [[[0.01]], [[0.02]]],
            [[[0.20]], [[0.30]]],
            [[[0.03]], [[0.04]]],
        ],
        dtype=np.float32,
    )

    pool = select_increment_pool(train_history, train_future, stress_quantile=0.5)

    assert pool.shape[0] == 2
    assert np.isclose(pool.max(), 0.30)


def test_inject_increment_fallback_only_changes_low_support_rows() -> None:
    base = np.full((2, 4, 2, 1, 1), 0.50, dtype=np.float32)
    history = np.full((2, 3, 1, 1), 0.50, dtype=np.float32)
    support_scores = np.array([0.1, 2.0], dtype=np.float64)
    pool = np.array(
        [
            [[[-0.10]], [[-0.20]]],
            [[[0.10]], [[0.20]]],
        ],
        dtype=np.float32,
    )

    out = inject_increment_fallback(
        base_samples=base,
        history_01=history,
        support_scores=support_scores,
        support_threshold=1.0,
        increment_pool=pool,
        fallback_fraction=0.5,
        rng=np.random.default_rng(7),
    )

    assert np.allclose(out[0], base[0])
    assert not np.allclose(out[1], base[1])
    assert np.allclose(out[1, 2:], base[1, 2:])
    assert out.min() >= 0.0
    assert out.max() <= 1.0
