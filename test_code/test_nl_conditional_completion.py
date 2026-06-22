"""Unit tests for the BJRS Gaussian conditional-completion core (framework-v1 §B/§I target)."""

import numpy as np
import pytest

from experiments.backfill.block_ar.nl_conditional_completion import conditional_completion


def test_analytic_two_factor():
    # Sigma with known correlation; pin factor 0 = 2.0. beta_{1|0} = cov_01/cov_00 = 0.5.
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    out = conditional_completion(cov, [0], [2.0])
    assert out["completion"][0] == pytest.approx(2.0)
    assert out["completion"][1] == pytest.approx(1.0)  # 0.5 * 2.0
    assert out["free_indices"] == [1]
    assert out["conditional_betas"].shape == (1, 1)
    assert out["conditional_betas"][0, 0] == pytest.approx(0.5)


def test_sign_follows_correlation():
    cov = np.array([[1.0, -0.7], [-0.7, 1.0]])
    # negative correlation: pinning factor 0 positive should push factor 1 negative
    out = conditional_completion(cov, [0], [1.5])
    assert out["completion"][1] < 0.0


def test_pin_all_factors_completion_is_x_s():
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    out = conditional_completion(cov, [0, 1], [1.0, -2.0])
    assert out["free_indices"] == []
    np.testing.assert_allclose(out["completion"], [1.0, -2.0])


def test_multi_pin_relative_magnitudes_matter():
    # With |S|>1, the free-factor sign depends on RELATIVE x_S magnitudes, not just signs.
    cov = np.array(
        [[1.0, 0.0, 0.8], [0.0, 1.0, -0.8], [0.8, -0.8, 1.0]]
    )
    # factor 2 is +0.8 with factor0, -0.8 with factor1. Pin 0=+1, 1=+1 (equal) -> betas cancel.
    out_equal = conditional_completion(cov, [0, 1], [1.0, 1.0])
    assert out_equal["completion"][2] == pytest.approx(0.0, abs=1e-9)
    # Now make factor 0 dominate -> factor 2 should go positive.
    out_dom = conditional_completion(cov, [0, 1], [3.0, 1.0])
    assert out_dom["completion"][2] > 0.0


def test_shape_validation():
    with pytest.raises(ValueError):
        conditional_completion(np.zeros((3, 2)), [0], [1.0])
    with pytest.raises(ValueError):
        conditional_completion(np.eye(3), [0, 1], [1.0])  # len mismatch
