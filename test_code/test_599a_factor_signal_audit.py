import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_599a_factor_signal_for_iv_failures import (  # noqa: E402
    auc_score,
    build_history_summary_features,
    ridge_oos_score,
)


def test_auc_score_orders_binary_targets() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    assert auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_build_history_summary_features_is_finite_and_named() -> None:
    history = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    features, names = build_history_summary_features(history, prefix="x")

    assert features.shape == (2, 15)
    assert len(names) == 15
    assert names[0] == "x_last_0"
    assert np.isfinite(features).all()


def test_ridge_oos_score_detects_factor_lift() -> None:
    rng = np.random.default_rng(11)
    n = 80
    iv = rng.normal(size=(n, 2))
    factor = rng.normal(size=(n, 1))
    y = 2.0 * factor[:, 0] + 0.1 * rng.normal(size=n)

    iv_score = ridge_oos_score(iv, y, train_frac=0.7, alpha=1e-3)
    both_score = ridge_oos_score(np.column_stack([iv, factor]), y, train_frac=0.7, alpha=1e-3)

    assert both_score["r2"] > iv_score["r2"] + 0.5
    assert both_score["corr"] > 0.9
