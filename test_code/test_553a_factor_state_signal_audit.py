import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.audit_553a_factor_state_signal import (
    factor_history_features,
    future_stress_targets,
    rank_corr,
    top_feature_targets,
)


def test_factor_history_features_include_only_past_factor_values() -> None:
    factors = {
        "ret": np.array([0.00, 0.01, -0.02, 0.03, 0.04, -0.01], dtype=np.float64),
        "levels": np.array([0.20, 0.21, 0.23, 0.22, 0.25, 0.26], dtype=np.float64),
    }
    features, names = factor_history_features(factors, indices=np.array([0, 1]), history_len=3)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    assert features.shape == (2, len(names))
    assert "ret_hist_mean" in names
    assert "ret_hist_abs_q90" in names
    assert "levels_hist_trend" in names
    assert np.isclose(features[0, name_to_idx["levels_hist_trend"]], 0.03)
    assert np.isclose(features[1, name_to_idx["levels_hist_trend"]], 0.01)


def test_future_stress_targets_capture_future_only_surface_and_return_stress() -> None:
    surfaces = np.array(
        [
            [[[0.20]]],
            [[[0.22]]],
            [[[0.30]]],
            [[[0.25]]],
            [[[0.45]]],
        ],
        dtype=np.float32,
    )
    factors = {"ret": np.array([0.0, 0.01, -0.02, 0.03, -0.05], dtype=np.float64)}

    targets, names = future_stress_targets(
        surfaces=surfaces,
        factors=factors,
        indices=np.array([0]),
        history_len=2,
        future_len=3,
    )
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    assert targets.shape == (1, len(names))
    assert np.isclose(targets[0, name_to_idx["future_iv_abs_move_q90"]], 0.185)
    assert np.isclose(targets[0, name_to_idx["future_ret_abs_q90"]], 0.046)


def test_rank_corr_and_top_feature_targets_are_stable() -> None:
    assert rank_corr(np.ones(4), np.arange(4)) == 0.0
    assert np.isclose(rank_corr(np.array([1, 2, 3]), np.array([3, 2, 1])), -1.0)

    features = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    targets = np.array([[0.0], [1.0], [2.0], [3.0]])
    rows = top_feature_targets(features, ["up", "down"], targets, ["stress"])

    assert rows[0]["feature"] == "up"
    assert rows[0]["target"] == "stress"
    assert rows[0]["spearman"] > 0.99
