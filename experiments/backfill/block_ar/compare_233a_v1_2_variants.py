#!/usr/bin/env python
"""
Compare 233a-v1.2 variants vs 229a and v1 baselines.

Produces:
  1. Per-variant metrics table (n_pass, turb_calm, worstC_h30, max_jump_ks, ks_test, MR)
  2. Attribution deltas (from design spec §7)
  3. Roll-up table: 5 diagnostics × 7 variants
  4. Decision tree outcome (Branches 1, 2a, 2b, 3, 4, 5, 6, 7 + H=252 demotion)
"""

import json
from pathlib import Path
import numpy as np

RESULT_DIR = Path("results/block_ar/233a_v1_2")
V1_RESULT_DIR = Path("results/block_ar/233a")
SEEDS = [42]
VARIANTS = ["control", "minreg", "minimal", "aux", "link", "both", "noreg"]

INCUMBENT_229a = {
    "n_pass": 3, "turb_calm": 1.025, "worst_cell_cov": 0.270,
    "max_jump_ks": 0.940, "change_ks_h30": 19, "MR_ratio": 1.318,
}

METRIC_PATHS = {
    "n_pass":         "summary.n_pass",
    "turb_calm":      "conditionality.turb_calm_ratio",
    "worst_cell_cov": "coverage.worst_cell_per_horizon.30",
    "max_jump_ks":    "pathwise_jump_realism.pathwise_max_jump.ks_stat",
    "change_ks_h30":  "distributional_fidelity.ks_test.n_pass",
    "MR_ratio":       "mean_reversion.gt_ratio",
}

DIAGNOSTIC_FIELDS = [
    ("film_logit_std", "_diagnostic_film_collapse.json", "logit_std_final_seed_42"),
    ("alpha_std", "_diagnostic_emission_link.json", "alpha_overall_stats.std"),
    ("lag1_autocorr", "_diagnostic_ar_compounding.json", "model_v1_2.lag1_autocorr_h29"),
    ("h_slow_auc", "_diagnostic_slow_state.json", "h_slow_pc1_auc_seed_42"),
    ("regime_inversion_flag", "_diagnostic_regime_breakdown.json", "regime_inversion_detected"),
]


def _get(tree, dotted):
    cur = tree
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_suite(variant, seed=42):
    path = RESULT_DIR / f"{variant}_s{seed}" / "suite.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summarise_variant(variant):
    d = load_suite(variant)
    if d is None:
        return None
    return {name: _get(d, path) for name, path in METRIC_PATHS.items()}


def diagnostic_rollup(variant):
    row = {"variant": variant}
    for field_name, file_basename, dotted in DIAGNOSTIC_FIELDS:
        path = RESULT_DIR / f"{variant}_s42" / file_basename
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            row[field_name] = _get(d, dotted)
        else:
            row[field_name] = None
    return row


def main():
    # Per-variant metrics
    results = {v: summarise_variant(v) for v in VARIANTS}
    # Baselines
    v1_path = V1_RESULT_DIR / "full_s42" / "suite.json"
    b229_path = V1_RESULT_DIR / "_baseline_229a_newproxy" / "suite.json"
    v1_metrics = {name: _get(json.load(open(v1_path)), path) for name, path in METRIC_PATHS.items()} if v1_path.exists() else None
    b229_metrics = {name: _get(json.load(open(b229_path)), path) for name, path in METRIC_PATHS.items()} if b229_path.exists() else None

    # Print metrics table
    print("=" * 110)
    print("233a-v1.2 — 7-variant comparison (seed 42)")
    print("=" * 110)
    header = f"{'config':<24}"
    for m in ["n_pass", "turb_calm", "worstC_h30", "max_jump_ks", "change_ks_h30", "MR_ratio"]:
        header += f" {m:>12}"
    print(header)
    print("-" * 110)

    def _fmt(row, m):
        v = row.get(m) if row else None
        if v is None: return "     —"
        if m == "n_pass": return f"{int(v):>12}"
        if m == "change_ks_h30": return f"{int(v):>12}"
        return f"{float(v):>12.3f}"

    if b229_metrics:
        print(f"{'229a @ep30 (incumbent)':<24}" + "".join(_fmt(b229_metrics, m) for m in METRIC_PATHS))
    if v1_metrics:
        print(f"{'v1-full_s42 (prev)':<24}" + "".join(_fmt(v1_metrics, m) for m in METRIC_PATHS))
    for v in VARIANTS:
        row = results[v]
        label = f"v1.2-{v}_s42"
        print(f"{label:<24}" + "".join(_fmt(row, m) for m in METRIC_PATHS))

    # Attribution deltas
    print("\n" + "=" * 110)
    print("Attribution deltas")
    print("=" * 110)
    def _d(a, b, key):
        va = a.get(key) if a else None
        vb = b.get(key) if b else None
        if va is None or vb is None: return None
        return va - vb

    if results["control"] and results["minreg"]:
        print(f"C1+C2 alone (minreg - control): Δn_pass={_d(results['minreg'], results['control'], 'n_pass')}, "
              f"Δturb_calm={_d(results['minreg'], results['control'], 'turb_calm'):.3f}")
    if results["minimal"] and results["minreg"]:
        print(f"C3 state-reg alone (minimal - minreg): Δn_pass={_d(results['minimal'], results['minreg'], 'n_pass')}")
    if results["aux"] and results["minimal"]:
        print(f"C4a twCRPS alone (aux - minimal): Δmax_jump_ks={_d(results['aux'], results['minimal'], 'max_jump_ks'):.3f}")
    if results["link"] and results["minimal"]:
        print(f"C4b learned-link alone (link - minimal): Δmax_jump_ks={_d(results['link'], results['minimal'], 'max_jump_ks'):.3f}")
    if results["both"] and results["minimal"]:
        print(f"C4a+C4b combined (both - minimal): Δmax_jump_ks={_d(results['both'], results['minimal'], 'max_jump_ks'):.3f}")

    # Diagnostic roll-up table
    print("\n" + "=" * 110)
    print("Diagnostic roll-up (5 diagnostics × 7 variants)")
    print("=" * 110)
    diag_header = f"{'variant':<12}"
    for f, _, _ in DIAGNOSTIC_FIELDS:
        diag_header += f" {f:>22}"
    print(diag_header)
    print("-" * 110)
    for v in VARIANTS:
        row = diagnostic_rollup(v)
        cells = [f"{row['variant']:<12}"]
        for f, _, _ in DIAGNOSTIC_FIELDS:
            val = row.get(f)
            cells.append(f"{'—' if val is None else (str(val) if isinstance(val, bool) else f'{val:.3f}'):>22}")
        print("".join(cells))

    # Decision tree
    print("\n" + "=" * 110)
    print("Decision tree (per Section 8 of design spec)")
    print("=" * 110)
    best = None
    best_n = -1
    for v in VARIANTS:
        r = results[v]
        if r and r["n_pass"] is not None and r["n_pass"] > best_n:
            best = v; best_n = r["n_pass"]
    if best is None:
        print("INSUFFICIENT DATA — missing suite.json for all variants")
        return
    bm = results[best]
    n = bm["n_pass"]; jk = bm.get("max_jump_ks", 1.0) or 1.0
    if n >= 5 and jk < 0.50:
        print(f"BRANCH 1 (clean success): best={best}, n/7={n}, max_jump_ks={jk:.3f}")
        print("  → architectural signal only; multi-seed replication required in v1.3")
        print("  → H=252 smoke test next; if fails, DEMOTE to Branch 2a")
    elif n >= 4 and jk < 0.50:
        print(f"BRANCH 2b (partial success + emission cracked): best={best}, n/7={n}, max_jump_ks={jk:.3f}")
        print("  → most informative partial; design targeted v1.2.x fix for remaining failing suite")
    elif n >= 4:
        print(f"BRANCH 2a (partial — FiLM fixed, emission cap): best={best}, n/7={n}, max_jump_ks={jk:.3f}")
        print("  → publish 4/7; Bug 6 is next bottleneck; parallel H3 scaffold design")
    else:
        print(f"BRANCH 3 (failure): best={best}, n/7={n}")
        print("  → paradigm pivot justified; launch H3 + joint-path flow matching")

    # === PA-02: Branches 4, 5, 6, 7 + H=252 demotion ===

    # Branch 4: control anomaly — v1.2-control should match v1-full_s42's pass-set
    if results["control"] and v1_metrics:
        ctrl_suite = load_suite("control")
        v1_suite = json.load(open(V1_RESULT_DIR / "full_s42" / "suite.json"))
        ctrl_pass = set(ctrl_suite["summary"].get("passed_suites", []))
        v1_pass = set(v1_suite["summary"].get("passed_suites", []))
        if ctrl_pass != v1_pass:
            print(f"\nBRANCH 4 WARNING (control anomaly): v1.2-control passed {ctrl_pass}, "
                  f"v1-full passed {v1_pass}. Drift detected — investigate before trusting "
                  f"attribution deltas.")

    # Branch 5: YAGNI — minimal ≈ both (within 1 n_pass)
    if results["minimal"] and results["both"]:
        dN = results["both"]["n_pass"] - results["minimal"]["n_pass"]
        if abs(dN) <= 1:
            print(f"\nBRANCH 5 (YAGNI): v1.2-minimal ≈ v1.2-both (ΔN={dN}). "
                  f"Emission fixes add no value; deploy minimal.")

    # Branch 6: one-fix dominates
    if results["aux"] and results["link"] and results["both"]:
        aux_n = results["aux"]["n_pass"]
        link_n = results["link"]["n_pass"]
        both_n = results["both"]["n_pass"]
        if aux_n >= both_n - 0.5 and aux_n > link_n:
            print(f"\nBRANCH 6a: v1.2-aux dominates; prefer aux-only (λ_twcrps) over both.")
        if link_n >= both_n - 0.5 and link_n > aux_n:
            print(f"\nBRANCH 6b: v1.2-link dominates; prefer link-only (learned g_θ) over both.")

    # Branch 7: C3 not load-bearing — minreg ≈ minimal
    if results["minreg"] and results["minimal"]:
        dN7 = abs(results["minreg"]["n_pass"] - results["minimal"]["n_pass"])
        if dN7 <= 1:
            print(f"\nBRANCH 7 (C3 NOT load-bearing): v1.2-minreg ≈ v1.2-minimal (|ΔN|={dN7}). "
                  f"Drop state consistency reg from production recipe.")

    # H=252 smoke demotion: if Branch 1 triggered AND h252_smoke.json fails any gate,
    # demote to Branch 2a
    if n >= 5 and jk < 0.50:
        h252_path = RESULT_DIR / "h252_smoke.json"
        if h252_path.exists():
            h252 = json.load(open(h252_path))
            if not h252.get("all_gates_pass", False):
                print(f"\nBRANCH 1 → 2a DEMOTION: H=252 smoke failed gates: "
                      f"{[k for k, v in h252.items() if k.endswith('_ok') and not v]}. "
                      f"Long-horizon rollout fails despite H=30 success. "
                      f"Bug 6 or state collapse at long horizons; v1.3 scope.")


if __name__ == "__main__":
    main()
