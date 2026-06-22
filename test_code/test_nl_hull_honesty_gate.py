"""Unit tests for the convex-hull / support-honesty feasibility LP (framework-v1 §I)."""

import numpy as np
import pytest

from experiments.backfill.block_ar.nl_hull_honesty_gate import hull_feasibility


def _square_pool():
    # 2-D unit square corners -> convex hull = [0,1]^2.
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_interior_target_feasible():
    pool = _square_pool()
    out = hull_feasibility(np.array([0.5, 0.5]), pool)
    assert out["feasible"] is True
    assert out["l1_distance"] == pytest.approx(0.0, abs=1e-6)
    assert out["support_label"] == "historically_grounded"


def test_pool_vertex_feasible():
    pool = _square_pool()
    out = hull_feasibility(pool[1], pool)  # a vertex
    assert out["feasible"] is True
    assert out["l1_distance"] == pytest.approx(0.0, abs=1e-6)


def test_far_target_infeasible_and_graded():
    pool = _square_pool()
    near = hull_feasibility(np.array([1.3, 1.3]), pool)
    far = hull_feasibility(np.array([5.0, 5.0]), pool)
    assert near["feasible"] is False
    assert far["feasible"] is False
    assert far["support_label"] == "outside_historical_analogue_support"
    # graded: farther target -> strictly larger L1 miss
    assert far["l1_distance"] > near["l1_distance"] > 0.0


def test_shape_validation():
    with pytest.raises(ValueError):
        hull_feasibility(np.array([0.0, 0.0, 0.0]), _square_pool())  # D mismatch
