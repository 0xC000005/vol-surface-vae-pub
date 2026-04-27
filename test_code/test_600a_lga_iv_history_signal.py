import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_600a_lga_iv_history_signal import (  # noqa: E402
    build_lga_history_features,
    softmax_neg_squared_distance,
)


def test_softmax_neg_squared_distance_prefers_near_key() -> None:
    distances = np.array([[0.0, 2.0, 4.0]], dtype=np.float64)

    weights = softmax_neg_squared_distance(distances, temperature=1.0)

    assert weights.shape == distances.shape
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]


def test_build_lga_history_features_is_finite_and_named() -> None:
    history = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)

    features, names = build_lga_history_features(history, prefix="iv_lga", temperature=1.0)

    assert features.shape == (2, 21)
    assert len(names) == 21
    assert names[0] == "iv_lga_context_0"
    assert names[-1] == "iv_lga_last_weight"
    assert np.isfinite(features).all()


def test_build_lga_history_features_changes_with_local_match() -> None:
    base = np.zeros((2, 5, 2), dtype=np.float64)
    base[0, :, 0] = [0.0, 1.0, 2.0, 1.0, 0.0]
    base[1, :, 0] = [5.0, 4.0, 3.0, 2.0, 0.0]

    features, names = build_lga_history_features(base, prefix="iv_lga", temperature=0.5)
    min_dist_idx = names.index("iv_lga_min_dist")

    assert features[0, min_dist_idx] < features[1, min_dist_idx]
