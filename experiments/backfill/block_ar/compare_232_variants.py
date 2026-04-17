#!/usr/bin/env python
"""Compare 232a/b/c/d attacks on the 3 architectural gates.

Gates:
  1. turb/calm ≥ 1.15 (conditionality)            — 232a target
  2. max-jump KS < 0.20 (pathwise)                — 232b/c target
  3. worst_cell_per_horizon[30] ≥ 0.70 (coverage) — 232d target

Baselines:
  229a@ep30 (wrapper path, with inference anchor implicit)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS = Path("results/block_ar/232_eval")
BASELINE_PATH = Path("results/block_ar/229a_ep30_suite.json")
VARIANTS = ["232a", "232b", "232c", "232d"]


def get_num(d: Any, *keys: Any, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        if k in d:
            d = d[k]
        elif isinstance(k, int) and str(k) in d:
            d = d[str(k)]
        elif isinstance(k, str) and k.isdigit() and int(k) in d:
            d = d[int(k)]
        else:
            return default
    return d


def extract_metrics(d: dict) -> dict[str, Any]:
    return {
        "n_pass": get_num(d, "summary", "n_pass"),
        "failed": get_num(d, "summary", "failed_suites") or [],
        "chg_ks": get_num(d, "distributional_fidelity", "ks_test", "n_pass"),
        "lvl_ks": get_num(d, "distributional_fidelity", "ks_level_test", "n_pass"),
        "turb_calm": get_num(d, "conditionality", "turb_calm_ratio"),
        "max_jump_ks": get_num(d, "pathwise_jump_realism", "pathwise_max_jump", "ks_stat"),
        "h30_worst_cov": get_num(d, "coverage", "worst_cell_per_horizon", 30),
        "h1_worst_cov": get_num(d, "coverage", "worst_cell_per_horizon", 1),
        "mr_ratio": get_num(d, "mean_reversion", "mr_gt_ratio"),
        "corr_ratio": get_num(d, "cross_cell_correlation", "corr_ratio"),
    }


def fmt(v: Any, width: int, precision: int = 3) -> str:
    if isinstance(v, (int, float)):
        return f"{v:{width}.{precision}f}"
    return f"{'--':>{width}}"


def main() -> None:
    rows: list[tuple[str, dict]] = []
    if BASELINE_PATH.exists():
        rows.append(("229a@ep30 (ref)", extract_metrics(json.loads(BASELINE_PATH.read_text()))))

    for v in VARIANTS:
        p = RESULTS / f"{v}_suite.json"
        if p.exists():
            rows.append((v, extract_metrics(json.loads(p.read_text()))))
        else:
            rows.append((v, {"n_pass": "MISS"}))

    header = (
        f"{'variant':<20} {'n/7':>5} {'ChgKS':>6} {'LvlKS':>6} "
        f"{'turb/calm':>10} {'maxJumpKS':>10} "
        f"{'h1 wcov':>8} {'h30 wcov':>9} {'mr':>7} {'corr':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(
            f"{name:<20} "
            f"{str(m.get('n_pass')):>5} "
            f"{str(m.get('chg_ks')):>6} "
            f"{str(m.get('lvl_ks')):>6} "
            f"{fmt(m.get('turb_calm'), 10)} "
            f"{fmt(m.get('max_jump_ks'), 10)} "
            f"{fmt(m.get('h1_worst_cov'), 8)} "
            f"{fmt(m.get('h30_worst_cov'), 9)} "
            f"{fmt(m.get('mr_ratio'), 7)} "
            f"{fmt(m.get('corr_ratio'), 6)}"
        )

    # Per-gate winner
    print()
    print("=" * 70)
    print("Per-gate architectural-attack analysis")
    print("=" * 70)

    def best_by(attr: str, higher_better: bool, label: str, gate: str) -> None:
        candidates = [(n, m.get(attr)) for n, m in rows if isinstance(m.get(attr), (int, float))]
        if not candidates:
            return
        sign = 1 if higher_better else -1
        best = max(candidates, key=lambda x: sign * x[1])
        print(f"\n{label}  (gate: {gate})")
        for n, v in candidates:
            marker = " <-- best" if n == best[0] else ""
            print(f"  {n:<20}  {v:.3f}{marker}")

    best_by("turb_calm", True,
            "Conditionality turb/calm", "≥ 1.15 (gate)")
    best_by("max_jump_ks", False,
            "Pathwise max-jump KS", "< 0.20 (gate)")
    best_by("h30_worst_cov", True,
            "Worst-cell coverage h30", "≥ 0.70 (gate)")

    # Save
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparison.json").write_text(
        json.dumps({"rows": [{"variant": n, **m} for n, m in rows]}, indent=2, default=str)
    )
    md_lines = [
        "# 232 Architectural Attacks — Comparison",
        "",
        "| variant | n/7 | ChgKS | LvlKS | turb/calm | maxJumpKS | h30 wcov | mr | corr |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        if not isinstance(m.get("turb_calm"), (int, float)):
            md_lines.append(f"| {name} | MISSING | | | | | | | |")
            continue
        md_lines.append(
            f"| {name} | {m.get('n_pass')} | {m.get('chg_ks')} | {m.get('lvl_ks')} | "
            f"{m['turb_calm']:.3f} | "
            f"{m.get('max_jump_ks'):.3f} | "
            f"{m.get('h30_worst_cov'):.3f} | "
            f"{m.get('mr_ratio'):.3f} | "
            f"{m.get('corr_ratio'):.3f} |"
        )
    (RESULTS / "comparison.md").write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
