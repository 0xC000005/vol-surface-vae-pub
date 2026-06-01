import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_feature_sufficiency import (
    _fit_linear_blend,
    _predict_linear_blend,
    _query_standardized,
    _score_support_prior,
    _support_reliability_prior,
)


def test_support_reliability_prior_uses_query_standardized_labels():
    prior = _support_reliability_prior(
        support_rows=[[1, 2], [1, 3], [4, 5], [4, 6]],
        labels=np.asarray([2.0, 0.0, 20.0, 10.0]),
        query_ids=["a", "a", "b", "b"],
    )

    assert prior[1] == 0.0
    assert prior[2] > 0.0
    assert prior[3] < 0.0
    assert prior[5] > 0.0
    assert prior[6] < 0.0


def test_score_support_prior_defaults_unknown_supports_to_zero():
    scores = _score_support_prior([[1, 2], [2, 3], [9]], {1: 1.0, 2: -0.5, 3: 0.5})

    assert np.allclose(scores, np.asarray([0.25, 0.0, 0.0]))


def test_linear_blend_recovers_positive_support_prior_weight():
    rank = np.asarray([0.0, -1.0, 0.0, -1.0])
    support = np.asarray([1.0, -1.0, 1.0, -1.0])
    labels = np.asarray([2.0, 0.0, 3.0, 1.0])

    coef = _fit_linear_blend(
        np.stack([rank, support], axis=1),
        _query_standardized(labels, ["a", "a", "b", "b"]),
    )
    pred = _predict_linear_blend(np.stack([rank, support], axis=1), coef)

    assert coef[2] > 0.0
    assert pred[0] > pred[1]
    assert pred[2] > pred[3]
