"""Convex-hull / support-honesty gate (framework-v1 acceptance candidate I).

The single most important NEW validator in the ratified framework
(`docs/research_protocols/nl_counterfactual_validation_framework_v1.md` §I): every
divergence / EL / cycle-consistency score is structurally blind to a narrative whose
plausible completion lies OUTSIDE the analogue pool's representable set. Retrieval-within-
history inherits exactly the historical-bias blind spot; nothing else in the stack detects
it — the system silently returns the nearest representable scenario instead.

This module answers, per query: is the conditional-completion target feasible as a convex
mixture of the analogue pool? It is a LABELING gate (honesty), not a rejection gate — an
infeasible target gets the label "outside historical analogue support" + the distance, never
a silent nearest-representable output. Bitter-Lesson-clean: validation-side only, no learned
constants, never enters the conditioning path.

Method (EL-feasibility, framework-v1 §I "scipy.linprog, per-query milliseconds"): solve the
L1-slack LP
    minimize   sum(s)
    over       w in R^N (>=0), s in R^D (>=0)
    s.t.       P^T w - target <=  s
              -(P^T w - target) <= s
               sum(w) = 1
where P is the (N, D) pool of analogue position vectors. The optimum `l1_distance = sum(s)`
is 0 iff `target` is in the convex hull of the pool; otherwise it is the minimal L1 mass by
which the hull misses the target (a graded honesty signal, not a cliff).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog

_FEASIBLE_TOL = 1e-6


def hull_feasibility(
    target: np.ndarray,
    pool: np.ndarray,
    *,
    feasible_tol: float = _FEASIBLE_TOL,
) -> dict[str, Any]:
    """Return {feasible, l1_distance, support_label} for `target` vs the analogue `pool`.

    target: (D,) conditional-completion target (e.g. BJRS mu_{-S|S} in the same coordinate
            as the pool position vectors).
    pool:   (N, D) analogue position vectors (the representable set whose convex hull is the
            pool's reachable completion space).
    """
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    pool = np.asarray(pool, dtype=np.float64)
    if pool.ndim != 2 or pool.shape[1] != target.shape[0]:
        raise ValueError(
            f"pool must be (N, D) matching target (D,); got pool {pool.shape}, target {target.shape}"
        )
    n, d = pool.shape

    # Variables: x = [w (n), s (d)]. Objective: minimize sum(s).
    c = np.concatenate([np.zeros(n), np.ones(d)])

    # Inequalities A_ub x <= b_ub:
    #   P^T w - s <= target          ->  [ P^T , -I ] x <= target
    #  -P^T w - s <= -target         ->  [-P^T , -I ] x <= -target
    pt = pool.T  # (d, n)
    neg_i = -np.eye(d)
    a_ub = np.block([[pt, neg_i], [-pt, neg_i]])
    b_ub = np.concatenate([target, -target])

    # Equality: sum(w) = 1 (s unconstrained here).
    a_eq = np.concatenate([np.ones(n), np.zeros(d)]).reshape(1, -1)
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n + [(0.0, None)] * d
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        # LP solver failure -> report as INDETERMINATE (distinct from a true infeasible/outside
        # verdict), never silently OK. A consumer must surface this as "could not assess", not as
        # "outside support".
        return {
            "feasible": False,
            "l1_distance": float("inf"),
            "support_label": "indeterminate_lp_failure",
            "lp_status": str(res.message),
        }

    l1_distance = float(res.fun)
    feasible = l1_distance <= feasible_tol
    return {
        "feasible": bool(feasible),
        "l1_distance": l1_distance,
        "support_label": (
            "historically_grounded" if feasible else "outside_historical_analogue_support"
        ),
        "lp_status": "optimal",
    }
