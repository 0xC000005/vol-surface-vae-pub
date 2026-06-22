"""Coordinate builder for the framework-v1 §I convex-hull support-honesty gate.

The gate (`nl_counterfactual_validation_framework_v1.md` §I) asks, per query: is the
narrative-implied BJRS conditional-completion target representable as a convex mixture of the
analogue pool? It is a graded *labeling* gate (honesty), never a silent nearest-representable
substitution. This module assembles the three objects that gate compares, all in ONE coordinate:

    - POOL : train-region windows' realized 30-day moves over the 14 named anchors,
             taken straight from the support bank in its OWN stored representation
             (`future_delta[:, -1, anchor_cols]` — verified cumulative move = future_raw[-1] -
             history_raw[-1]).
    - Σ    : the (14,14) covariance of those train-region moves, for the closed-form Gaussian
             completion mu_{-S|S} (`nl_conditional_completion.conditional_completion`).
    - x_S  : the narrative-pinned factor moves. `narrative_emphasis` gives {direction, salience}
             only, so x_S[i] = kappa * sign(direction_i) * (salience_i / max_salience) * std_i,
             where std_i is the train-region per-factor move std. kappa is a SINGLE global scalar
             in per-factor-sigma units (kappa=1 -> a 1-sigma move); salience carries the relative
             magnitudes across the pinned set (matters when |S|>1).

Coordinate consistency (the real correctness risk): pool, Σ, and x_S are all built in the bank's
raw 30-day-move units, so the BJRS completion is computed correctly in that space. The hull
*membership label* is affine-invariant to per-factor rescaling, so for numerical conditioning and
an interpretable graded distance we run the feasibility LP on the per-factor-sigma-normalized
coordinate (divide pool & completion by std). The pass/fail label is identical either way; only
the reported `l1_distance_sigma` changes (now in interpretable sigma units).

Bitter-Lesson clean: every quantity here is a property of HISTORICAL data used ONLY for
validation-side labeling. Nothing enters the conditioning/generation path. The per-factor std and
Σ are validation statistics, not learned model constants.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from experiments.backfill.block_ar.nl_conditional_completion import conditional_completion
from experiments.backfill.block_ar.nl_hull_honesty_gate import hull_feasibility
from experiments.backfill.block_ar.nl_joint39_anchor_map import joint39_anchor_columns
from experiments.backfill.block_ar.nl_narrative_reweighter import dir_sign, narrative_emphasis

DEFAULT_BANK = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_KAPPA_LADDER: tuple[float, ...] = (0.5, 1.0, 2.0)


def build_anchor_move_pool(
    bank_path: str = DEFAULT_BANK,
    *,
    train_only: bool = True,
) -> dict[str, Any]:
    """Build the analogue pool + Σ + per-factor std in the bank's 30-day anchor-move space.

    Returns dict with:
      pool         (N, 14) train-region terminal 30-day moves over the 14 anchors (raw units)
      cov          (14, 14) covariance of `pool`
      factor_std   (14,) per-factor std of `pool` (>0; used for kappa units + sigma distance)
      anchor_names (14,) UPPER factor names in column order
      anchor_cols  (14,) joint39 column indices (25..38)
      pool_window_indices  the bank-row indices used for the pool
    """
    anchor_map = joint39_anchor_columns()  # {NAME: joint39_col}
    names = [n for n, _ in sorted(anchor_map.items(), key=lambda kv: kv[1])]
    cols = [anchor_map[n] for n in names]

    data = np.load(bank_path, allow_pickle=True)
    future_delta = np.asarray(data["future_delta"], dtype=np.float64)  # (W, 30, 39) cumulative
    if train_only and "train_indices" in data.files:
        rows = np.asarray(data["train_indices"], dtype=np.int64)
    else:
        rows = np.arange(future_delta.shape[0], dtype=np.int64)

    # Terminal 30-day move = cumulative move at the last horizon step.
    pool = future_delta[rows][:, -1, :][:, cols].astype(np.float64)  # (N, 14)
    factor_std = pool.std(axis=0, ddof=1)
    factor_std = np.where(factor_std > 0.0, factor_std, 1.0)  # guard zero-variance
    cov = np.cov(pool.T, ddof=1)  # (14, 14) raw-unit covariance, kept for provenance only

    # All gate math runs in per-factor-sigma units. Two reasons (Codex review #6): (1) the raw
    # covariance has condition ~1e9 from the 0.02-vs-1000 unit-scale disparity across anchors, so
    # solving the BJRS system raw is ill-conditioned; cov_sigma is the correlation matrix and is
    # well-conditioned. (2) The hull MEMBERSHIP label is identical to raw space because a common
    # positive diagonal rescaling preserves convex hulls, and mu_sigma == mu_raw / std exactly, so
    # reporting raw is a pure rescale. Coordinate stays self-consistent: pool, Sigma, x_S all in
    # sigma units for the solve + LP.
    pool_sigma = pool / factor_std[None, :]
    cov_sigma = np.cov(pool_sigma.T, ddof=1)
    mean_sigma = pool_sigma.mean(axis=0)
    cov_sigma_inv = np.linalg.pinv(cov_sigma)  # for the Mahalanobis density diagnostic

    return {
        "pool": pool,
        "cov": cov,
        "factor_std": factor_std,
        "pool_sigma": pool_sigma,
        "cov_sigma": cov_sigma,
        "mean_sigma": mean_sigma,
        "cov_sigma_inv": cov_sigma_inv,
        "anchor_names": names,
        "anchor_cols": cols,
        "pool_window_indices": rows,
    }


@lru_cache(maxsize=2)
def _cached_pool(bank_path: str, train_only: bool) -> dict[str, Any]:
    """Cached pool bundle (read-only; do not mutate the returned arrays)."""
    return build_anchor_move_pool(bank_path, train_only=train_only)


def emphasis_to_pinned(
    emphasis: dict[str, dict[str, Any]],
    *,
    anchor_names: Sequence[str],
    factor_std: np.ndarray,
    kappa: float = 1.0,
) -> dict[str, Any]:
    """Map narrative emphasis -> (pinned_indices, pinned_values) in raw move units.

    x_S[i] = kappa * sign(direction) * (salience / max_salience) * std_i .
    Factors not in the anchor panel, with zero/unknown direction, or non-positive salience are
    dropped. Relative magnitudes across the pinned set come from `salience` (so |S|>1 is handled).
    """
    name_to_local = {str(n).upper(): i for i, n in enumerate(anchor_names)}
    factor_std = np.asarray(factor_std, dtype=np.float64).reshape(-1)

    raw: list[tuple[int, int, float]] = []  # (local_idx, sign, salience)
    for factor, spec in (emphasis or {}).items():
        key = str(factor).upper()
        if key not in name_to_local:
            continue
        s = dir_sign(str(spec.get("direction", "")))
        sal = float(spec.get("salience", 0.0))
        if s == 0 or sal <= 0.0:
            continue
        raw.append((name_to_local[key], s, sal))

    if not raw:
        return {"pinned_indices": [], "x_sigma": [], "x_raw": [], "dropped_all": True}

    max_sal = max(sal for _, _, sal in raw)
    pinned_indices: list[int] = []
    x_sigma: list[float] = []
    x_raw: list[float] = []
    for local_idx, s, sal in raw:
        rel = sal / max_sal if max_sal > 0 else 1.0
        mag_sigma = float(kappa) * s * rel  # sigma units: kappa=1 -> a 1-sigma move
        pinned_indices.append(int(local_idx))
        x_sigma.append(mag_sigma)
        x_raw.append(mag_sigma * float(factor_std[local_idx]))  # raw move units, for reporting
    return {"pinned_indices": pinned_indices, "x_sigma": x_sigma, "x_raw": x_raw, "dropped_all": False}


def hull_label_for_emphasis(
    emphasis: dict[str, dict[str, Any]],
    *,
    pool_bundle: dict[str, Any],
    kappa: float = 1.0,
) -> dict[str, Any]:
    """Per-query hull honesty label for a narrative emphasis at a single kappa.

    Returns the BJRS completion target + hull feasibility (label is coordinate-invariant; the
    graded l1 distance is reported in per-factor-sigma units).
    """
    pool_sigma = np.asarray(pool_bundle["pool_sigma"], dtype=np.float64)
    cov_sigma = np.asarray(pool_bundle["cov_sigma"], dtype=np.float64)
    mean_sigma = np.asarray(pool_bundle["mean_sigma"], dtype=np.float64).reshape(-1)
    cov_sigma_inv = np.asarray(pool_bundle["cov_sigma_inv"], dtype=np.float64)
    factor_std = np.asarray(pool_bundle["factor_std"], dtype=np.float64).reshape(-1)
    names = list(pool_bundle["anchor_names"])

    pin = emphasis_to_pinned(emphasis, anchor_names=names, factor_std=factor_std, kappa=kappa)
    if pin["dropped_all"]:
        return {
            "status": "no_anchor_emphasis",
            "kappa": float(kappa),
            "pinned_factors": [],
            "support_label": "no_emphasis_to_test",
            "feasible": None,
            "l1_distance_sigma": None,
            "pool_mahalanobis": None,
        }

    # Entirely in sigma units: well-conditioned Sigma (correlation matrix), and the hull label is
    # identical to raw space. mu_sigma == mu_raw / std, so completion_target_raw is a pure rescale.
    completion_sigma = conditional_completion(cov_sigma, pin["pinned_indices"], pin["x_sigma"])[
        "completion"
    ]  # (14,) sigma units
    hull = hull_feasibility(completion_sigma, pool_sigma)
    delta = completion_sigma - mean_sigma
    maha = float(np.sqrt(max(float(delta @ cov_sigma_inv @ delta), 0.0)))  # density diagnostic
    completion_raw = completion_sigma * factor_std

    pinned_factors = [
        {"factor": names[i], "x_S_raw": float(r), "x_S_sigma": float(sg)}
        for i, r, sg in zip(pin["pinned_indices"], pin["x_raw"], pin["x_sigma"])
    ]
    return {
        "status": "ok",
        "kappa": float(kappa),
        "pinned_factors": pinned_factors,
        "feasible": bool(hull["feasible"]),
        "l1_distance_sigma": float(hull["l1_distance"]),
        "support_label": str(hull["support_label"]),
        "pool_mahalanobis": maha,
        "completion_target_raw": [float(x) for x in completion_raw],
        "completion_factors": names,
        "lp_status": hull.get("lp_status", ""),
    }


def hull_label_kappa_ladder(
    emphasis: dict[str, dict[str, Any]],
    *,
    pool_bundle: dict[str, Any],
    kappas: Sequence[float] = DEFAULT_KAPPA_LADDER,
) -> dict[str, Any]:
    """Graded hull honesty label across a kappa ladder (severity of the requested move).

    Reports, per kappa, feasibility + sigma distance, plus the smallest kappa at which the
    narrative-implied completion leaves the historical analogue hull (None if feasible throughout).
    """
    rungs = [hull_label_for_emphasis(emphasis, pool_bundle=pool_bundle, kappa=float(k)) for k in kappas]
    leaves_at = None
    indeterminate = False
    for r in rungs:
        if r.get("status") != "ok":
            continue
        label = r.get("support_label")
        if label == "indeterminate_lp_failure":
            # LP solver could not assess: this is "could not check", NOT "outside support".
            indeterminate = True
        elif label == "outside_historical_analogue_support" and leaves_at is None:
            leaves_at = r["kappa"]
    return {
        "ladder": rungs,
        "kappas": [float(k) for k in kappas],
        "leaves_hull_at_kappa": leaves_at,
        "any_infeasible": leaves_at is not None,
        "any_indeterminate": bool(indeterminate),
    }


def hull_label_from_grounding(
    grounding_output: dict[str, Any],
    *,
    pool_bundle: dict[str, Any] | None = None,
    bank_path: str = DEFAULT_BANK,
    kappas: Sequence[float] = DEFAULT_KAPPA_LADDER,
) -> dict[str, Any]:
    """End-to-end: grounding_output -> narrative_emphasis -> graded hull honesty label."""
    if pool_bundle is None:
        pool_bundle = _cached_pool(bank_path, True)
    emphasis = narrative_emphasis(grounding_output or {})
    out = hull_label_kappa_ladder(emphasis, pool_bundle=pool_bundle, kappas=kappas)
    out["emphasis"] = emphasis
    return out
