#!/usr/bin/env python
"""Compare 231a three-way ablation suite results and output promotion decision.

Reads:
  results/block_ar/231a_eval/231a_{none,fixed,learn}_suite.json
  results/block_ar/229a_ep30_suite.json  (incumbent)

Writes:
  results/block_ar/231a_eval/comparison.md
  results/block_ar/231a_eval/comparison.json
Also prints a summary table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS = Path("results/block_ar/231a_eval")
# 229a suite.json uses WRAPPER path. For native-path comparison, we also print
# the native-path ChgKS from the 230a diagnostic decision_memo as a reference.
INCUMBENT_PATHS = [
    ("229a@ep30 (wrapper)", "results/block_ar/229a_ep30_suite.json"),
]
DIAGNOSTIC_PATH = Path("results/block_ar/230a_scale_state/230a_consolidated.json")

VARIANT_ORDER = ["none", "fixed", "learn"]


def load_native_chgks_at_h30() -> dict[str, int] | None:
    """Extract the 229a native-path ChgKS @ h30 for each mode from 230a diagnostic.

    Returns dict with keys:
      native_self_feed  -> m2 (baseline, no anchor)
      native_anchor050  -> m4 (inference-only anchor, the 231a target)
      wrapper_self_feed -> m7 (reference for what wrapper gives)
    """
    if not DIAGNOSTIC_PATH.exists():
        return None
    d = json.loads(DIAGNOSTIC_PATH.read_text())
    dm = d.get("229a_ep30", {}).get("decision_memo", {})
    q1 = dm.get("Q1_oracle_scale_recovery", [])
    q2 = dm.get("Q2_wrapper_vs_anchor", [])
    h30_q1 = next((x for x in q1 if x.get("horizon") == 30), None)
    h30_q2 = next((x for x in q2 if x.get("horizon") == 30), None)
    if h30_q1 is None or h30_q2 is None:
        return None
    return {
        "native_self_feed": h30_q1.get("native"),
        "native_oracle_scale": h30_q1.get("oracle"),
        "native_anchor050": h30_q2.get("anchor050"),
        "wrapper_self_feed": h30_q2.get("wrapper"),
        "tf": h30_q1.get("tf"),
    }


def get_num(d: Any, *keys: Any, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        # try both int and str keys (json preserves type)
        if k in d:
            d = d[k]
        elif isinstance(k, int) and str(k) in d:
            d = d[str(k)]
        elif isinstance(k, str) and k.isdigit() and int(k) in d:
            d = d[int(k)]
        else:
            return default
    return d


def fmt(v: Any, width: int, precision: int = 3) -> str:
    if isinstance(v, (int, float)):
        return f"{v:>{width}.{precision}f}"
    return f"{'--':>{width}}"


def load_suite(path: str | Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def extract_metrics(d: dict) -> dict[str, Any]:
    return {
        "n_pass": get_num(d, "summary", "n_pass"),
        "failed": get_num(d, "summary", "failed_suites") or [],
        "chg_ks": get_num(d, "distributional_fidelity", "ks_test", "n_pass"),
        "lvl_ks": get_num(d, "distributional_fidelity", "ks_level_test", "n_pass"),
        "turb_calm": get_num(d, "conditionality", "turb_calm_ratio"),
        "max_jump_ks": get_num(d, "pathwise_jump_realism", "pathwise_max_jump", "ks_stat"),
        "h30_worst_cov": get_num(d, "coverage", "worst_cell_per_horizon", 30),
        "mr_ratio": get_num(d, "mean_reversion", "mr_gt_ratio"),
        "corr_ratio": get_num(d, "cross_cell_correlation", "corr_ratio"),
        "h30_cov90": get_num(d, "coverage", "per_horizon", 30, 0.9),
    }


def main() -> None:
    rows: list[tuple[str, dict]] = []

    # 229a native-path ChgKS reference (from 230a diagnostic)
    native_ref = load_native_chgks_at_h30()
    if native_ref:
        print("=== 229a@ep30 NATIVE-path ChgKS reference (from 230a diagnostic) ===")
        print(f"  h=30  TF={native_ref['tf']}  native_self={native_ref['native_self_feed']}  "
              f"native+anchor(0.5)={native_ref['native_anchor050']}  "
              f"oracle_scale={native_ref['native_oracle_scale']}  "
              f"wrapper={native_ref['wrapper_self_feed']}")
        print("  231a-none expected ~=", native_ref["native_self_feed"],
              "(no anchor)")
        print("  231a-fixed expected ~=", native_ref["native_anchor050"],
              "(inference anchor = training anchor; 230b regime)")
        print("  231a-learn gate: >= max(none,fixed) + 2")
        print()

    # Incumbent(s)
    for name, path in INCUMBENT_PATHS:
        d = load_suite(path)
        if d is not None:
            rows.append((name, extract_metrics(d)))

    # Variants
    for v in VARIANT_ORDER:
        d = load_suite(RESULTS / f"231a_{v}_suite.json")
        if d is None:
            rows.append((f"231a-{v}", {"n_pass": "MISSING"}))
        else:
            rows.append((f"231a-{v}", extract_metrics(d)))

    # Console table
    header = (
        f"{'variant':<24} {'n/7':>5} {'ChgKS':>6} {'LvlKS':>6} "
        f"{'turb/calm':>10} {'maxJumpKS':>10} {'h30 wcov':>9} "
        f"{'mr':>7} {'corr':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(
            f"{name:<24} "
            f"{str(m.get('n_pass')):>5} "
            f"{str(m.get('chg_ks')):>6} "
            f"{str(m.get('lvl_ks')):>6} "
            f"{fmt(m.get('turb_calm'), 10)} "
            f"{fmt(m.get('max_jump_ks'), 10)} "
            f"{fmt(m.get('h30_worst_cov'), 9)} "
            f"{fmt(m.get('mr_ratio'), 7)} "
            f"{fmt(m.get('corr_ratio'), 6)}"
        )

    # Promotion decision — plan gate:
    #   231a-learn must beat 231a-fixed AND 231a-none by >= 2 cells ChgKS at h=30
    chg: dict[str, int | None] = {
        v: None
        for v in VARIANT_ORDER
    }
    for v in VARIANT_ORDER:
        d = load_suite(RESULTS / f"231a_{v}_suite.json")
        if d is not None:
            chg[v] = extract_metrics(d).get("chg_ks")
    print()
    print(f"ChgKS @ h30:  none={chg['none']}  fixed={chg['fixed']}  learn={chg['learn']}")
    print("Promotion gate: learn >= fixed + 2 AND learn >= none + 2")

    verdict = None
    if all(v is not None for v in chg.values()):
        if chg["learn"] >= chg["fixed"] + 2 and chg["learn"] >= chg["none"] + 2:
            verdict = "231a-learn PROMOTES"
            next_step = "Proceed to 231b (regime mixture)."
        elif chg["fixed"] >= chg["none"] + 2:
            verdict = "231a-fixed PROMOTES (learn did not beat it)"
            next_step = "Proceed to 231b using 231a-fixed as base (note in log: regime-coupling trap not escaped)."
        else:
            verdict = "ALL VARIANTS UNDERPERFORM"
            next_step = "Publish negative result; stop 231 series."
        print(f"VERDICT: {verdict}")
        print(f"NEXT: {next_step}")

    # Persist comparison
    out = {
        "rows": [{"variant": n, **m} for n, m in rows],
        "chg_ks_at_h30": chg,
        "verdict": verdict,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparison.json").write_text(json.dumps(out, indent=2, default=str))
    md_lines = [
        "# 231a Three-Way Ablation — Comparison",
        "",
        f"- Change KS @ h30: none={chg['none']}  fixed={chg['fixed']}  learn={chg['learn']}",
        f"- Verdict: **{verdict or 'INCOMPLETE'}**",
        "",
        "## Suite metrics",
        "",
        "| variant | n/7 | ChgKS | LvlKS | turb/calm | maxJumpKS | h30 wcov | mr | corr |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in rows:
        md_lines.append(
            f"| {name} | {m.get('n_pass')} | {m.get('chg_ks')} | {m.get('lvl_ks')} | "
            f"{m.get('turb_calm'):.3f} | "
            f"{m.get('max_jump_ks') if isinstance(m.get('max_jump_ks'), (int, float)) else '--'} | "
            f"{m.get('h30_worst_cov'):.3f} | "
            f"{m.get('mr_ratio'):.3f} | "
            f"{m.get('corr_ratio'):.3f} |"
            if isinstance(m.get("turb_calm"), (int, float))
            else f"| {name} | {m.get('n_pass')} | -- | -- | -- | -- | -- | -- | -- |"
        )
    (RESULTS / "comparison.md").write_text("\n".join(md_lines))
    print(f"\nWrote: {RESULTS / 'comparison.md'}")


if __name__ == "__main__":
    main()
