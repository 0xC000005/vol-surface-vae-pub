#!/usr/bin/env python
"""
Aggregate results/block_ar/233a/*/suite.json files into a comparison table
and apply the decision tree from spec §6.5 / §7.5.

Usage:
  PYTHONPATH=. python experiments/backfill/block_ar/compare_233a_variants.py

Incumbent 229a native+anchor baseline (v3 harness reference):
  n_pass=3, turb_calm=1.142, worst_cell_cov=0.27, max_jump_ks=0.94, change_ks_npass=19

Decision tree (from spec §6.5 / §7.5):
  Compute delta_full_minus_B for {n_pass, worst_cell_cov, max_jump_ks}
  "Generous gain" (any one):
    delta_n_pass >= 2  OR  delta_worst_cell_cov >= 0.15  OR  delta_max_jump_ks <= -0.20
  "Generous loss" (any one):
    delta_n_pass <= -2  OR  delta_worst_cell_cov <= -0.15  OR  delta_max_jump_ks >= 0.20
  Rule:
    (generous gain) AND (full also beats C similarly) -> CONTINUE to v2
    (generous gain) BUT (C matches full)              -> STOP + deploy v1-C (HAR beats learned state)
    (generous loss)                                   -> STOP, decomposition harmful, deploy v1-B
    (neither)                                         -> STOP, decomposition not load-bearing, deploy v1-B

JSON schema (from evaluate_220b_multihorizon_path_suite.py output):
  Top-level keys: config, surface, coverage, conditionality, distributional_fidelity,
                  cross_cell_correlation, mean_reversion, pathwise_jump_realism, summary
  n_pass:          summary.n_pass
  turb_calm:       conditionality.turb_calm_ratio
  worst_cell_cov:  coverage.worst_cell_per_horizon.30  (string key "30" in JSON)
  max_jump_ks:     pathwise_jump_realism.pathwise_max_jump.ks_stat
  change_ks_npass: distributional_fidelity.ks_test.n_pass  (count of cells 0-25,
                   no per-horizon breakdown available — plan's "change_ks_h30" label
                   is retained for display, but maps to the aggregate pass count)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

RESULT_DIR = Path("results/block_ar/233a")
SEEDS = [42, 1337, 2024]

INCUMBENT_229a = {
    "n_pass": 3,
    "turb_calm": 1.142,
    "worst_cell_cov": 0.27,
    "max_jump_ks": 0.94,
    "change_ks_h30": 19,
}

# Dotted JSON paths into suite.json.
# Keys in the JSON are strings (int horizon keys become "30" etc. after JSON serialisation).
# Verified against evaluate_220b_multihorizon_path_suite.py + test_block_ar_requirements_v2.py.
METRIC_PATHS = {
    "n_pass":         "summary.n_pass",
    "turb_calm":      "conditionality.turb_calm_ratio",
    # worst_cell_per_horizon is a dict keyed by horizon (as string in JSON).
    # We take h=30 as the gate horizon.
    "worst_cell_cov": "coverage.worst_cell_per_horizon.30",
    # pathwise_jump_realism nests under pathwise_max_jump.
    "max_jump_ks":    "pathwise_jump_realism.pathwise_max_jump.ks_stat",
    # No per-horizon KS breakdown; plan's change_ks_h30 maps to aggregate cell-pass count.
    "change_ks_h30":  "distributional_fidelity.ks_test.n_pass",
}


def _get(tree: dict, dotted: str):
    """Navigate a nested dict via a dotted key path. Raises KeyError on missing keys."""
    cur = tree
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"missing key '{key}' in path '{dotted}'")
        cur = cur[key]
    return cur


def load_seed(variant: str, seed: int) -> Optional[dict]:
    path = RESULT_DIR / f"{variant}_s{seed}" / "suite.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summarise_variant(variant: str) -> dict:
    """Return {metric_name: (mean, std, n)} across available seeds; None entries for missing."""
    per_metric: dict[str, list[float]] = {name: [] for name in METRIC_PATHS}
    missing: list[int] = []
    for s in SEEDS:
        data = load_seed(variant, s)
        if data is None:
            missing.append(s)
            continue
        for metric, path in METRIC_PATHS.items():
            try:
                v = float(_get(data, path))
                per_metric[metric].append(v)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"  WARNING: {variant}_s{s}: metric '{metric}' not found ({exc})")

    summary: dict = {}
    for metric, vals in per_metric.items():
        if vals:
            summary[metric] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            summary[metric] = (None, None, 0)
    summary["_missing_seeds"] = missing
    return summary


def fmt(summary: dict, metric: str) -> str:
    mean, std, n = summary[metric]
    if mean is None:
        return "(no data)"
    return f"{mean:.3f} ± {std:.3f} (n={n})"


def _mean(summary: dict, metric: str) -> Optional[float]:
    """Return mean or None."""
    return summary[metric][0]


def decide(full: dict, B: dict, C: dict) -> str:
    """Apply decision tree from spec §6.5 / §7.5."""
    fN, BN = _mean(full, "n_pass"), _mean(B, "n_pass")
    fW, BW = _mean(full, "worst_cell_cov"), _mean(B, "worst_cell_cov")
    fJ, BJ = _mean(full, "max_jump_ks"), _mean(B, "max_jump_ks")

    if None in (fN, BN, fW, BW, fJ, BJ):
        return "INSUFFICIENT DATA: one or more key metrics missing for v1-full or v1-B"

    dN = fN - BN
    dW = fW - BW
    dJ = fJ - BJ  # lower KS stat is better, so negative delta means full is better

    gain = (dN >= 2) or (dW >= 0.15) or (dJ <= -0.20)
    loss = (dN <= -2) or (dW <= -0.15) or (dJ >= 0.20)

    if gain:
        CN, CW, CJ = _mean(C, "n_pass"), _mean(C, "worst_cell_cov"), _mean(C, "max_jump_ks")
        if None in (CN, CW, CJ):
            return "PARTIAL DATA: v1-full gains vs v1-B, but v1-C data missing — cannot complete decision"
        dNc = fN - CN
        dWc = fW - CW
        dJc = fJ - CJ
        gain_vs_C = (dNc >= 2) or (dWc >= 0.15) or (dJc <= -0.20)
        if gain_vs_C:
            return "CONTINUE to v2: decomposition load-bearing AND learned slow state beats HAR"
        else:
            return "STOP latent expansion: deploy v1-C (HAR-conditioned 226a matches full)"
    elif loss:
        return "STOP line: v1-B outperforms v1-full; decomposition harmful"
    else:
        return "STOP line: decomposition not load-bearing; deploy v1-B"


def main() -> None:
    full = summarise_variant("full")
    B = summarise_variant("B")
    C = summarise_variant("C")

    print("=" * 100)
    print("233a Two-Path Factor AR v1 — Three-Variant Ladder (3 seeds each)")
    print("=" * 100)
    col_w = 24
    print(
        f"\n{'metric':<18} {'229a ref':>12}   "
        f"{'v1-full':<{col_w}}   {'v1-B':<{col_w}}   {'v1-C':<{col_w}}"
    )
    print("-" * 100)
    for metric in ["n_pass", "turb_calm", "worst_cell_cov", "max_jump_ks", "change_ks_h30"]:
        ref_val = INCUMBENT_229a.get(metric, "—")
        ref_str = f"{ref_val}" if isinstance(ref_val, int) else f"{ref_val:.3f}"
        print(
            f"{metric:<18} {ref_str:>12}   "
            f"{fmt(full, metric):<{col_w}}   {fmt(B, metric):<{col_w}}   {fmt(C, metric):<{col_w}}"
        )

    for variant_name, s in [("full", full), ("B", B), ("C", C)]:
        missing = s.get("_missing_seeds", [])
        if missing:
            print(f"\n  v1-{variant_name}: missing seeds {missing} "
                  f"(results/block_ar/233a/{variant_name}_s<seed>/suite.json not found)")

    print()
    print("=" * 100)
    decision = decide(full, B, C)
    print(f"DECISION: {decision}")
    print("=" * 100)


if __name__ == "__main__":
    main()
