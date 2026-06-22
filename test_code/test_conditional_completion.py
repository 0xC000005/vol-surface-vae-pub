"""BJRS conditional-completion core (framework-v1 §B coherence + §I hull-gate target).

Given the historical 30-day-move covariance and a narrative-pinned factor set S with
target values, the Gaussian conditional mean of the OTHER factors,
mu_{-S|S} = Sigma_{-S,S} Sigma_{S,S}^{-1} x_S (zero-mean moves), is the closed-form
"what the other factors should do if S moves like this" used by the coherence sign-gate
and as the hull-feasibility target. Validation-side only (never enters conditioning).
"""

import numpy as np

from experiments.backfill.block_ar.nl_conditional_completion import conditional_completion


def test_correlated_completion_matches_closed_form():
    cov = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = conditional_completion(cov, pinned_indices=[0], pinned_values=[2.0])
    comp = out["completion"]
    assert comp.shape == (3,)
    assert np.isclose(comp[0], 2.0)          # pinned value preserved
    assert np.isclose(comp[1], 0.8 * 2.0)    # beta_{1|0} = 0.8 -> 1.6
    assert np.isclose(comp[2], 0.0)          # uncorrelated -> 0


def test_zero_correlation_completion_is_zero():
    cov = np.eye(3)
    out = conditional_completion(cov, pinned_indices=[0], pinned_values=[2.0])
    comp = out["completion"]
    assert np.isclose(comp[0], 2.0)
    assert np.allclose(comp[1:], 0.0)


def test_conditional_betas_reported():
    cov = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = conditional_completion(cov, pinned_indices=[0], pinned_values=[2.0])
    betas = out["conditional_betas"]  # (n_free, n_pinned)
    # factor 1 on factor 0 => 0.8 ; factor 2 on factor 0 => 0.0
    assert np.isclose(betas[0, 0], 0.8)
    assert np.isclose(betas[1, 0], 0.0)


def test_two_pinned_factors():
    # block-correlated 4-factor; pin {0,1}
    cov = np.array(
        [
            [1.0, 0.5, 0.3, 0.0],
            [0.5, 1.0, 0.0, 0.2],
            [0.3, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 1.0],
        ]
    )
    out = conditional_completion(cov, pinned_indices=[0, 1], pinned_values=[1.0, -1.0])
    comp = out["completion"]
    assert np.isclose(comp[0], 1.0)
    assert np.isclose(comp[1], -1.0)
    # closed form for the free factors
    sigma_ss = cov[np.ix_([0, 1], [0, 1])]
    sigma_fs = cov[np.ix_([2, 3], [0, 1])]
    expected = sigma_fs @ np.linalg.solve(sigma_ss, np.array([1.0, -1.0]))
    assert np.allclose(comp[[2, 3]], expected)
