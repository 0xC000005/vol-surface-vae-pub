import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.audit_550a_data_framing_causal_state import (
    causal_state_features,
    effective_independent_windows,
    feature_drift_table,
    rank_corr,
    split_geometry,
)


def test_split_geometry_exposes_overlap_and_chronological_blocks() -> None:
    geom = split_geometry(
        n_total_days=5822,
        history_len=30,
        future_len=30,
        test_start=4511,
        val_size=441,
        max_windows=192,
    )

    assert geom["base_train_windows"] == 4010
    assert geom["calibration_start"] == 3569
    assert geom["calibration_end"] == 4009
    assert geom["official_val_start"] == 4010
    assert geom["eval_subset_end"] == 4201
    assert geom["future_overlap_days_between_adjacent_windows"] == 29
    assert geom["eval_subset_effective_independent_windows"] == effective_independent_windows(
        192,
        30,
    )


def test_causal_state_features_are_history_only_and_named() -> None:
    history = np.array(
        [
            [
                [[0.20, 0.30], [0.40, 0.50]],
                [[0.22, 0.32], [0.43, 0.53]],
                [[0.25, 0.35], [0.45, 0.55]],
            ],
            [
                [[0.60, 0.50], [0.40, 0.30]],
                [[0.58, 0.47], [0.39, 0.31]],
                [[0.57, 0.45], [0.41, 0.33]],
            ],
        ],
        dtype=np.float32,
    )

    features, names = causal_state_features(history)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    assert features.shape == (2, len(names))
    assert np.allclose(features[:, name_to_idx["last_mean"]], [0.40, 0.44])
    assert features[0, name_to_idx["history_trend_mean"]] > 0
    assert features[1, name_to_idx["history_trend_mean"]] < 0
    assert "history_vov" in names
    assert "term_slope" in names
    assert "smile_slope" in names


def test_rank_corr_and_feature_drift_table_are_stable_for_constant_inputs() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    assert np.isclose(rank_corr(x, y), -1.0)
    assert rank_corr(np.ones_like(x), y) == 0.0

    names = ["constant", "shifted"]
    train = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    eval_ = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    drift = feature_drift_table(train, eval_, names)
    by_feature = {row["feature"]: row for row in drift}

    assert by_feature["constant"]["ks_stat"] == 0.0
    assert by_feature["constant"]["standardized_mean_shift"] == 0.0
    assert by_feature["shifted"]["eval_mean"] > by_feature["shifted"]["train_mean"]
    assert drift[0]["feature"] == "shifted"
