"""Tests for the hull-gate coordinate builder, incl. the discrimination go/no-go (advisor gate).

The binding correctness risk (per advisor) is coordinate consistency: pool, Σ, x_S must live in
one coordinate, and the hull label must actually DISCRIMINATE (calm -> feasible, extreme ->
infeasible, graded distance monotone in severity). If the label never varies, the gate is
decorative. These tests assert real discrimination on the real support bank.
"""

import os

import numpy as np
import pytest

from experiments.backfill.block_ar.nl_hull_gate_inputs import (
    DEFAULT_BANK,
    build_anchor_move_pool,
    emphasis_to_pinned,
    hull_label_for_emphasis,
    hull_label_kappa_ladder,
)

_BANK_PRESENT = os.path.exists(DEFAULT_BANK)
pytestmark = pytest.mark.skipif(not _BANK_PRESENT, reason="support bank arrays not present")


@pytest.fixture(scope="module")
def bundle():
    return build_anchor_move_pool(DEFAULT_BANK, train_only=True)


def test_pool_shape_and_anchor_map(bundle):
    assert bundle["pool"].shape[1] == 14
    assert bundle["pool"].shape[0] > 1000  # train region windows
    assert bundle["cov"].shape == (14, 14)
    assert np.all(bundle["factor_std"] > 0)
    assert bundle["anchor_names"][0] == "SPX"
    assert bundle["anchor_names"][-1] == "VIX"
    assert bundle["anchor_cols"] == list(range(25, 39))


def test_emphasis_to_pinned_signs_and_relative_magnitudes(bundle):
    names, std = bundle["anchor_names"], bundle["factor_std"]
    emphasis = {
        "SPX": {"direction": "down", "salience": 1.0},
        "VIX": {"direction": "up", "salience": 0.5},
    }
    pin = emphasis_to_pinned(emphasis, anchor_names=names, factor_std=std, kappa=1.0)
    by = dict(zip(pin["pinned_indices"], pin["x_raw"]))
    spx_i, vix_i = names.index("SPX"), names.index("VIX")
    assert by[spx_i] < 0  # down
    assert by[vix_i] > 0  # up
    # magnitude = kappa * (salience/max_salience) * std
    assert by[spx_i] == pytest.approx(-1.0 * 1.0 * std[spx_i])
    assert by[vix_i] == pytest.approx(1.0 * 0.5 * std[vix_i])


def test_emphasis_to_pinned_drops_invalid(bundle):
    names, std = bundle["anchor_names"], bundle["factor_std"]
    emphasis = {
        "NOT_A_FACTOR": {"direction": "up", "salience": 1.0},
        "GOLD": {"direction": "flat", "salience": 1.0},      # flat -> sign 0
        "COPPER": {"direction": "up", "salience": 0.0},       # zero salience
    }
    pin = emphasis_to_pinned(emphasis, anchor_names=names, factor_std=std, kappa=1.0)
    assert pin["dropped_all"] is True
    assert pin["pinned_indices"] == []


def test_discrimination_centroid_feasible_extreme_infeasible(bundle):
    """Go/no-go: a centroid target is feasible (~0 miss); a beyond-max target is infeasible."""
    from experiments.backfill.block_ar.nl_hull_honesty_gate import hull_feasibility

    pool = bundle["pool"]
    std = bundle["factor_std"]
    pool_sigma = pool / std[None, :]

    centroid = (pool.mean(axis=0)) / std
    extreme = (pool.max(axis=0) + 5.0 * std) / std  # exceeds per-dim max in EVERY coord

    feas = hull_feasibility(centroid, pool_sigma)
    infeas = hull_feasibility(extreme, pool_sigma)
    assert feas["feasible"] is True
    assert feas["l1_distance"] == pytest.approx(0.0, abs=1e-5)
    assert infeas["feasible"] is False
    assert infeas["l1_distance"] > 0.0
    # The label genuinely varies -> not decorative.
    assert feas["feasible"] != infeas["feasible"]


def test_kappa_ladder_monotone_and_trips_at_severity(bundle):
    """Graded honesty: sigma-distance non-decreasing in kappa; extreme move leaves the hull."""
    emphasis = {"SPX": {"direction": "down", "salience": 1.0}}  # single factor: clean monotone
    out = hull_label_kappa_ladder(emphasis, pool_bundle=bundle, kappas=(0.5, 2.0, 8.0))
    dists = [r["l1_distance_sigma"] for r in out["ladder"]]
    assert dists[0] <= dists[1] <= dists[2]      # monotone non-decreasing
    assert dists[2] > dists[0]                   # strictly grows with severity
    assert out["ladder"][0]["feasible"] is True  # mild 0.5-sigma move is in-support
    assert out["leaves_hull_at_kappa"] is not None  # 8-sigma move trips the honesty label


def test_joint_discrimination_plausible_vs_contradictory(bundle):
    """STRONG go/no-go (Codex review #5): a contradictory joint narrative is INFEASIBLE even
    though every completion coordinate stays inside its marginal min/max -- proving genuine JOINT
    (not marginal-bound) discrimination -- while the plausible risk-off counterpart is feasible
    and strictly closer to the historical support."""
    pool_sigma = bundle["pool_sigma"]
    lo, hi = pool_sigma.min(axis=0), pool_sigma.max(axis=0)

    risk_off = {  # plausible co-move: equities down, vol up, credit wider
        "SPX": {"direction": "down", "salience": 1.0},
        "VIX": {"direction": "up", "salience": 1.0},
        "BBB_OAS": {"direction": "wider", "salience": 1.0},
    }
    contradictory = {  # implausible: equities UP while vol up AND credit wider
        "SPX": {"direction": "up", "salience": 1.0},
        "VIX": {"direction": "up", "salience": 1.0},
        "BBB_OAS": {"direction": "wider", "salience": 1.0},
    }
    r_off = hull_label_for_emphasis(risk_off, pool_bundle=bundle, kappa=1.0)
    r_con = hull_label_for_emphasis(contradictory, pool_bundle=bundle, kappa=1.0)

    # The contradictory narrative is outside support; the plausible one is in-support.
    assert r_con["feasible"] is False
    assert r_off["feasible"] is True
    # ...and the contradictory miss is JOINT, not marginal: every completion coord is within range.
    comp_con_sigma = np.asarray(r_con["completion_target_raw"]) / bundle["factor_std"]
    assert np.all(comp_con_sigma >= lo - 1e-9) and np.all(comp_con_sigma <= hi + 1e-9)
    # Graded signal agrees with the binary label.
    assert r_con["l1_distance_sigma"] > r_off["l1_distance_sigma"]
    assert r_con["pool_mahalanobis"] > r_off["pool_mahalanobis"]


def test_single_kappa_label_well_formed(bundle):
    emphasis = {"SPX": {"direction": "down", "salience": 1.0}}
    r = hull_label_for_emphasis(emphasis, pool_bundle=bundle, kappa=1.0)
    assert r["status"] == "ok"
    assert len(r["completion_target_raw"]) == 14
    assert r["support_label"] in {"historically_grounded", "outside_historical_analogue_support"}
    assert [f["factor"] for f in r["pinned_factors"]] == ["SPX"]
