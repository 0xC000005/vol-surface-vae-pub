import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.audit_554a_factor_failure_signal import (
    failure_signal_summary,
    fit_ridge_score,
    top_feature_failure_correlations,
    window_interval_miss_metrics,
)


def test_window_interval_miss_metrics_separate_stress_misses_from_coverage() -> None:
    samples = np.array(
        [
            [[[[0.10]]], [[[0.10]]], [[[0.10]]]],
            [[[[0.00]]], [[[1.00]]], [[[2.00]]]],
        ],
        dtype=np.float64,
    )
    future = np.array([[[[0.50]]], [[[1.00]]]], dtype=np.float64)

    metrics = window_interval_miss_metrics(samples, future)

    assert np.allclose(metrics["coverage90"], [0.0, 1.0])
    assert np.allclose(metrics["upper_miss_rate"], [1.0, 0.0])
    assert np.allclose(metrics["lower_miss_rate"], [0.0, 0.0])
    assert metrics["coverage_under_target"][0] > metrics["coverage_under_target"][1]


def test_fit_ridge_score_uses_train_statistics_and_preserves_order() -> None:
    train_x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    train_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    eval_x = np.array([[0.5], [2.5]], dtype=np.float64)

    score, fit = fit_ridge_score(train_x, train_y, eval_x, l2=1e-3)

    assert score.shape == (2,)
    assert score[1] > score[0]
    assert np.allclose(fit["feature_mean"], [1.5])


def test_top_feature_failure_correlations_and_summary_identify_material_signal() -> None:
    features = np.array(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    metrics = {
        "coverage_under_target": np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        "upper_miss_rate": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        "future_abs_move_mean": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
    }

    rows = top_feature_failure_correlations(features, ["stress_up", "stress_down"], metrics)
    summary = failure_signal_summary(rows, min_abs_spearman=0.75)

    assert rows[0]["feature"] == "stress_up"
    assert rows[0]["target"] == "coverage_under_target"
    assert rows[0]["spearman"] > 0.99
    assert summary["factor_failure_signal_material"] is True
