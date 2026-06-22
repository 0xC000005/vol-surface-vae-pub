"""Convex-hull / support-honesty gate (framework-v1 candidate I) unit tests.

The gate answers the one failure every divergence/cycle method is structurally blind to:
is the narrative's plausible completion target representable as a convex mixture of the
analogue pool? If not, the monitor must LABEL "outside historical analogue support"
rather than silently return the nearest representable scenario. This is a labeling
(honesty) gate, not a rejection gate.
"""

import numpy as np

from experiments.backfill.block_ar.nl_hull_honesty_gate import hull_feasibility


def test_target_inside_hull_is_feasible():
    # pool spans a 2-simplex in 3-D; centroid is strictly inside the hull
    pool = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = pool.mean(axis=0)  # convex combo with equal weights
    out = hull_feasibility(target, pool)
    assert out["feasible"] is True
    assert out["l1_distance"] < 1e-6
    assert out["support_label"] == "historically_grounded"


def test_target_outside_hull_is_infeasible():
    pool = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = np.array([5.0, 5.0, 5.0])  # far outside the unit simplex hull
    out = hull_feasibility(target, pool)
    assert out["feasible"] is False
    assert out["l1_distance"] > 1.0
    assert out["support_label"] == "outside_historical_analogue_support"


def test_vertex_is_feasible():
    pool = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    target = np.array([2.0, 0.0])  # exactly a pool vertex (w = e_2)
    out = hull_feasibility(target, pool)
    assert out["feasible"] is True
    assert out["l1_distance"] < 1e-6


def test_just_outside_is_graded():
    # a point slightly past a vertex along an axis -> infeasible but small distance
    pool = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    target = np.array([2.5, 0.0])
    out = hull_feasibility(target, pool)
    assert out["feasible"] is False
    assert 0.0 < out["l1_distance"] < 1.0  # graded honesty, not a cliff
