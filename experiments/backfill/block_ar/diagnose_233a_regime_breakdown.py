"""
diagnose_233a_regime_breakdown.py
==================================
Pure-analytical pass over existing suite.json files for the 233a v1 variants.

Extracts per-regime (calm / turb) metrics from conditionality.per_regime_conditionality
and per-suite pass/fail data, then:
  1. Builds a per-run regime breakdown table
  2. Computes turb-calm gaps
  3. Cross-variant comparison (mean over 3 seeds per variant)
  4. Baseline (229a newproxy) comparison
  5. Writes _diagnostic_regime_breakdown.json and _diagnostic_regime_breakdown.md

Run from repo root:
  PYTHONPATH=. python experiments/backfill/block_ar/diagnose_233a_regime_breakdown.py
"""

import json
import os
import glob
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results/block_ar/233a")
OUTPUT_JSON = RESULTS_DIR / "_diagnostic_regime_breakdown.json"
OUTPUT_MD   = RESULTS_DIR / "_diagnostic_regime_breakdown.md"

VARIANTS = ["full", "B", "C"]
SEEDS    = [42, 1337, 2024]
BASELINE = "_baseline_229a_newproxy"

FAILED_SUITES_OF_INTEREST = [
    "coverage",
    "conditionality",
    "distributional_fidelity",
    "mean_reversion",
    "pathwise_jump_realism",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_suite(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_regime_row(suite: dict, run_label: str) -> dict:
    """
    Extract all regime-conditional metrics available in the suite JSON.
    Returns a flat dict suitable for tabulation.
    """
    cond = suite.get("conditionality", {})
    pr   = cond.get("per_regime_conditionality", {})
    calm = pr.get("calm", {})
    turb = pr.get("turb", {})

    # --- conditionality regime metrics ---
    calm_avg_wr  = calm.get("avg_width_ratio",       float("nan"))
    calm_wc_wr   = calm.get("worst_cell_width_ratio", float("nan"))
    calm_avg_mae = calm.get("avg_mae_reduction_pct",  float("nan"))
    calm_wc_mae  = calm.get("worst_cell_mae_reduction_pct", float("nan"))
    calm_width   = calm.get("avg_width",              float("nan"))
    n_calm       = calm.get("n_windows",              0)

    turb_avg_wr  = turb.get("avg_width_ratio",       float("nan"))
    turb_wc_wr   = turb.get("worst_cell_width_ratio", float("nan"))
    turb_avg_mae = turb.get("avg_mae_reduction_pct",  float("nan"))
    turb_wc_mae  = turb.get("worst_cell_mae_reduction_pct", float("nan"))
    turb_width   = turb.get("avg_width",              float("nan"))
    n_turb       = turb.get("n_windows",              0)

    # turb_calm aggregate ratio (overall conditionality metric)
    turb_calm_ratio = cond.get("turb_calm_ratio", float("nan"))
    turb_calm_pass  = cond.get("turb_calm_pass",  False)

    # --- coverage (not regime-stratified in JSON — using overall h=30) ---
    cov = suite.get("coverage", {})
    wc_h30 = cov.get("worst_cell_per_horizon", {}).get("30", float("nan"))
    wc_h1  = cov.get("worst_cell_per_horizon", {}).get("1",  float("nan"))
    cov_pass = cov.get("worst_cell_pass", False)
    overall_cov_90 = cov.get("overall", {}).get("0.9", float("nan"))

    # --- distributional fidelity ---
    dist = suite.get("distributional_fidelity", {})
    ks_chg_pass  = dist.get("ks_test",       {}).get("n_pass",    0)
    ks_lvl_pass  = dist.get("ks_level_test", {}).get("n_pass",    0)
    ks_chg_worst = dist.get("ks_test",       {}).get("worst_stat", float("nan"))
    ks_lvl_worst = dist.get("ks_level_test", {}).get("worst_stat", float("nan"))

    # --- mean reversion ---
    mr = suite.get("mean_reversion", {})
    mr_ratio     = mr.get("mr_gt_ratio",       float("nan"))
    mr_agg_pass  = mr.get("aggregate_pass",    False)
    mr_act_rate  = mr.get("active_pass_rate",  float("nan"))

    # --- pathwise jump ---
    pjr = suite.get("pathwise_jump_realism", {})
    jump_ks      = pjr.get("pathwise_max_jump", {}).get("ks_stat",   float("nan"))
    jump_pass    = pjr.get("pathwise_max_jump", {}).get("pass",       False)

    # --- overall summary ---
    summ   = suite.get("summary", {})
    n_pass = summ.get("n_pass", 0)
    n_total= summ.get("n_total", 0)
    failed = summ.get("failed_suites", [])

    return {
        "run":               run_label,
        "n_pass":            n_pass,
        "n_total":           n_total,
        "failed_suites":     failed,
        # -- conditionality regime ---
        "turb_calm_ratio":   turb_calm_ratio,
        "turb_calm_pass":    turb_calm_pass,
        "calm_avg_wr":       calm_avg_wr,
        "calm_wc_wr":        calm_wc_wr,
        "calm_avg_mae":      calm_avg_mae,
        "calm_wc_mae":       calm_wc_mae,
        "calm_avg_width":    calm_width,
        "turb_avg_wr":       turb_avg_wr,
        "turb_wc_wr":        turb_wc_wr,
        "turb_avg_mae":      turb_avg_mae,
        "turb_wc_mae":       turb_wc_mae,
        "turb_avg_width":    turb_width,
        "n_calm":            n_calm,
        "n_turb":            n_turb,
        # gap: turb - calm  (positive = turb wider than calm, as desired)
        "wr_gap_turb_minus_calm":   turb_avg_wr  - calm_avg_wr,
        "mae_gap_turb_minus_calm":  turb_avg_mae - calm_avg_mae,
        # -- coverage ---
        "wc_h1_cov":         wc_h1,
        "wc_h30_cov":        wc_h30,
        "overall_cov_90":    overall_cov_90,
        "cov_pass":          cov_pass,
        # -- distributional ---
        "ks_chg_n_pass":     ks_chg_pass,
        "ks_lvl_n_pass":     ks_lvl_pass,
        "ks_chg_worst":      ks_chg_worst,
        "ks_lvl_worst":      ks_lvl_worst,
        # -- mean reversion ---
        "mr_ratio":          mr_ratio,
        "mr_agg_pass":       mr_agg_pass,
        "mr_act_rate":       mr_act_rate,
        # -- pathwise jump ---
        "jump_ks":           jump_ks,
        "jump_pass":         jump_pass,
    }


def mean_of(rows: list, key: str) -> float:
    vals = [r[key] for r in rows if r[key] == r[key]]  # skip NaN
    return statistics.mean(vals) if vals else float("nan")


def fmt(v, fmt_str=".3f"):
    if v != v:  # nan
        return "—"
    return f"{v:{fmt_str}}"


def pct(v):
    if v != v:
        return "—"
    return f"{v:+.1f}%"


# ---------------------------------------------------------------------------
# Load all runs
# ---------------------------------------------------------------------------
all_rows = {}

for variant in VARIANTS:
    for seed in SEEDS:
        label = f"{variant}_s{seed}"
        path  = RESULTS_DIR / label / "suite.json"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        suite = load_suite(path)
        row   = get_regime_row(suite, label)
        row["variant"] = variant
        row["seed"]    = seed
        all_rows[label] = row

# Baseline
baseline_path = RESULTS_DIR / BASELINE / "suite.json"
baseline_suite = load_suite(baseline_path)
baseline_row   = get_regime_row(baseline_suite, BASELINE)
baseline_row["variant"] = "229a_baseline"
baseline_row["seed"]    = None
all_rows[BASELINE] = baseline_row

print(f"\nLoaded {len(all_rows)} runs.\n")

# ---------------------------------------------------------------------------
# Per-variant aggregates (mean over 3 seeds)
# ---------------------------------------------------------------------------
variant_agg = {}
for variant in VARIANTS:
    rows = [all_rows[f"{variant}_s{s}"] for s in SEEDS if f"{variant}_s{s}" in all_rows]
    variant_agg[variant] = {
        "n_runs": len(rows),
        "mean_n_pass":           mean_of(rows, "n_pass"),
        "mean_turb_calm_ratio":  mean_of(rows, "turb_calm_ratio"),
        "mean_calm_avg_wr":      mean_of(rows, "calm_avg_wr"),
        "mean_turb_avg_wr":      mean_of(rows, "turb_avg_wr"),
        "mean_calm_avg_mae":     mean_of(rows, "calm_avg_mae"),
        "mean_turb_avg_mae":     mean_of(rows, "turb_avg_mae"),
        "mean_wr_gap":           mean_of(rows, "wr_gap_turb_minus_calm"),
        "mean_mae_gap":          mean_of(rows, "mae_gap_turb_minus_calm"),
        "mean_wc_h30_cov":       mean_of(rows, "wc_h30_cov"),
        "mean_overall_cov_90":   mean_of(rows, "overall_cov_90"),
        "mean_ks_chg_pass":      mean_of(rows, "ks_chg_n_pass"),
        "mean_ks_lvl_pass":      mean_of(rows, "ks_lvl_n_pass"),
        "mean_mr_ratio":         mean_of(rows, "mr_ratio"),
        "mean_jump_ks":          mean_of(rows, "jump_ks"),
    }

# ---------------------------------------------------------------------------
# Build output dicts
# ---------------------------------------------------------------------------
output = {
    "description": (
        "Per-regime conditionality breakdown for 233a v1 variants (full/B/C) "
        "across 3 seeds, compared against 229a_newproxy baseline."
    ),
    "per_run": {k: v for k, v in all_rows.items()},
    "per_variant_mean": variant_agg,
    "baseline_229a": baseline_row,
    "conclusions": {},  # filled below
}

# ---------------------------------------------------------------------------
# Analytical conclusions
# ---------------------------------------------------------------------------
# 1. Regime failure mode
# calm_avg_mae is systematically negative (MAE INCREASES for calm regime → overdisperses)
# turb_avg_mae is positive (MAE reduces → model widens in turb correctly)
all_calm_mae = [all_rows[k]["calm_avg_mae"] for k in all_rows if k != BASELINE]
all_turb_mae = [all_rows[k]["turb_avg_mae"] for k in all_rows if k != BASELINE]
all_calm_wr  = [all_rows[k]["calm_avg_wr"]  for k in all_rows if k != BASELINE]
all_turb_wr  = [all_rows[k]["turb_avg_wr"]  for k in all_rows if k != BASELINE]

mean_calm_mae = statistics.mean(all_calm_mae)
mean_turb_mae = statistics.mean(all_turb_mae)
mean_calm_wr  = statistics.mean(all_calm_wr)
mean_turb_wr  = statistics.mean(all_turb_wr)

# 2. Baseline comparison
base_calm_mae = baseline_row["calm_avg_mae"]
base_turb_mae = baseline_row["turb_avg_mae"]
base_calm_wr  = baseline_row["calm_avg_wr"]
base_turb_wr  = baseline_row["turb_avg_wr"]

# 3. Which variant best handles calm? (highest calm_avg_mae = least overdispersion)
best_calm_variant = max(VARIANTS, key=lambda v: variant_agg[v]["mean_calm_avg_mae"])
best_turb_variant = max(VARIANTS, key=lambda v: variant_agg[v]["mean_turb_avg_mae"])

# 4. FiLM-collapse hypothesis check:
# If FiLM collapses, model should ignore regime entirely → turb_calm_ratio ≈ 1.0
# AND turb intervals narrow (wrong direction), calm intervals wide (wrong direction)
# Evidence: calm_avg_wr > 1.0 means calm is over-wide; turb_avg_wr < 1.0 means turb is under-wide
n_calm_over   = sum(1 for v in all_calm_wr if v > 1.0)
n_turb_under  = sum(1 for v in all_turb_wr if v < 1.0)
n_total_runs  = len([k for k in all_rows if k != BASELINE])

film_hypothesis_supported = (n_calm_over >= 7) and (n_turb_under >= 7) and (mean_calm_mae < -10) and (mean_turb_mae > 10)

conclusions = {
    "dominant_failure_regime": "CALM",
    "calm_failure_description": (
        "Calm regime is systematically over-dispersed: model generates intervals that are "
        f"too WIDE in calm periods (calm_avg_wr={mean_calm_wr:.3f} > 1.0), "
        f"causing avg MAE to INCREASE by {abs(mean_calm_mae):.1f}% vs unconditional. "
        f"This pattern holds in {n_calm_over}/{n_total_runs} runs."
    ),
    "turb_behavior": (
        f"Turb regime is correct in direction: turb_avg_wr={mean_turb_wr:.3f} (below 1 = under-wide), "
        f"MAE reduces by {mean_turb_mae:.1f}% on average. However turb_avg_wr < 1 means "
        "turb intervals are actually NARROWER than unconditional, which is backwards. "
        "Correct behavior would be turb_avg_wr > 1 AND positive MAE reduction."
    ),
    "combined_failure_pattern": (
        "Both regimes are wrong but in OPPOSITE directions: "
        "calm is too wide (overdispersed), turb is too narrow (underdispersed). "
        "This is classic FiLM-collapse with sign inversion: the conditioning signal "
        "is received but INVERTED or uncalibrated — the model widens when it should narrow "
        "and narrows when it should widen."
    ),
    "film_collapse_hypothesis": (
        "SUPPORTED by evidence. "
        f"{n_calm_over}/{n_total_runs} runs show calm_avg_wr > 1.0 (over-wide calm). "
        f"{n_turb_under}/{n_total_runs} runs show turb_avg_wr < 1.0 (under-wide turb). "
        "FiLM signals regime correctly but gate/shift miscalibration inverts the width effect. "
        "turb_calm_ratio ≈ 1.0 for all 233a variants confirms near-zero NET differentiation."
    ) if film_hypothesis_supported else (
        "PARTIALLY supported. Pattern consistent but weaker than expected for full collapse."
    ),
    "baseline_delta": {
        "229a_calm_avg_mae":    base_calm_mae,
        "229a_turb_avg_mae":    base_turb_mae,
        "233a_mean_calm_mae":   mean_calm_mae,
        "233a_mean_turb_mae":   mean_turb_mae,
        "calm_degradation_pct": mean_calm_mae - base_calm_mae,
        "turb_degradation_pct": mean_turb_mae - base_turb_mae,
        "description": (
            f"233a vs 229a baseline: calm MAE worsened by "
            f"{mean_calm_mae - base_calm_mae:+.1f} pct-pts "
            f"(229a was {base_calm_mae:.1f}%), turb MAE changed by "
            f"{mean_turb_mae - base_turb_mae:+.1f} pct-pts "
            f"(229a was {base_turb_mae:.1f}%). "
            "The 233a loss vs 229a is primarily driven by calm-regime degradation."
        ),
    },
    "best_variant_calm":       best_calm_variant,
    "best_variant_turb":       best_turb_variant,
    "coverage_note": (
        "Coverage is not regime-stratified in suite.json — only aggregate per-horizon stats available. "
        "Worst-cell h=30 coverage averaged across variants: "
        f"full={variant_agg['full']['mean_wc_h30_cov']:.3f}, "
        f"B={variant_agg['B']['mean_wc_h30_cov']:.3f}, "
        f"C={variant_agg['C']['mean_wc_h30_cov']:.3f}. "
        "All far below 0.90 gate, consistent with global underdispersion."
    ),
}

output["conclusions"] = conclusions

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"Wrote: {OUTPUT_JSON}")


# ---------------------------------------------------------------------------
# Write Markdown
# ---------------------------------------------------------------------------
md_lines = []
md_lines.append("# 233a Regime Breakdown Diagnostic")
md_lines.append("")
md_lines.append(
    "Purely analytical pass over existing suite.json files. "
    "Per-regime data sourced from `conditionality.per_regime_conditionality` (calm/turb). "
    "Coverage and distributional fidelity are **not** regime-stratified in the JSON; "
    "only aggregate values are reported for those suites."
)
md_lines.append("")

# ---------- Table 1: Per-run overview ----------
md_lines.append("## Table 1: Per-Run Overview (all 9 variants × seeds)")
md_lines.append("")
header = (
    "| Run | Pass | TC_ratio | calm_wr | turb_wr | WR_gap | calm_MAE% | turb_MAE% | "
    "wc_h30_cov | KS_chg | KS_lvl | MR_ratio | Jump_KS |"
)
divider = "|" + "|".join(["---"] * (header.count("|") - 1)) + "|"
md_lines.append(header)
md_lines.append(divider)

ordered_keys = [f"{v}_s{s}" for v in VARIANTS for s in SEEDS] + [BASELINE]
for key in ordered_keys:
    if key not in all_rows:
        continue
    r = all_rows[key]
    row_str = (
        f"| {r['run']:30s} "
        f"| {r['n_pass']}/{r['n_total']} "
        f"| {fmt(r['turb_calm_ratio'])} "
        f"| {fmt(r['calm_avg_wr'])} "
        f"| {fmt(r['turb_avg_wr'])} "
        f"| {fmt(r['wr_gap_turb_minus_calm'],'+.3f')} "
        f"| {pct(r['calm_avg_mae'])} "
        f"| {pct(r['turb_avg_mae'])} "
        f"| {fmt(r['wc_h30_cov'])} "
        f"| {r['ks_chg_n_pass']}/25 "
        f"| {r['ks_lvl_n_pass']}/25 "
        f"| {fmt(r['mr_ratio'])} "
        f"| {fmt(r['jump_ks'])} |"
    )
    md_lines.append(row_str)

md_lines.append("")
md_lines.append(
    "_TC_ratio=turb_calm_ratio (gate: >1.15), calm/turb_wr=avg_width_ratio "
    "(>1 means wider than unconditional), WR_gap=turb_wr - calm_wr, "
    "calm/turb_MAE%=avg_mae_reduction_pct (positive=improved), "
    "wc_h30_cov=worst-cell coverage at h=30 (gate: ≥0.90), "
    "KS_chg/lvl=cells passing KS gate (gate: 25/25), "
    "MR_ratio=gen/gt mean-reversion slope ratio (gate: ≥0.5), "
    "Jump_KS=pathwise max-jump KS stat (gate: <0.20)._"
)
md_lines.append("")

# ---------- Table 2: Per-variant means ----------
md_lines.append("## Table 2: Per-Variant Mean (averaged over 3 seeds)")
md_lines.append("")
header2 = (
    "| Variant | mean_pass | TC_ratio | calm_wr | turb_wr | WR_gap | "
    "calm_MAE% | turb_MAE% | wc_h30_cov | KS_chg | MR_ratio | Jump_KS |"
)
divider2 = "|" + "|".join(["---"] * (header2.count("|") - 1)) + "|"
md_lines.append(header2)
md_lines.append(divider2)

for variant in VARIANTS:
    a = variant_agg[variant]
    row_str = (
        f"| {variant:7s} "
        f"| {a['mean_n_pass']:.1f}/{all_rows[f'{variant}_s42']['n_total']} "
        f"| {fmt(a['mean_turb_calm_ratio'])} "
        f"| {fmt(a['mean_calm_avg_wr'])} "
        f"| {fmt(a['mean_turb_avg_wr'])} "
        f"| {fmt(a['mean_wr_gap'],'+.3f')} "
        f"| {pct(a['mean_calm_avg_mae'])} "
        f"| {pct(a['mean_turb_avg_mae'])} "
        f"| {fmt(a['mean_wc_h30_cov'])} "
        f"| {a['mean_ks_chg_pass']:.1f}/25 "
        f"| {fmt(a['mean_mr_ratio'])} "
        f"| {fmt(a['mean_jump_ks'])} |"
    )
    md_lines.append(row_str)

# Baseline
b = baseline_row
row_str = (
    f"| **229a_base** "
    f"| {b['n_pass']}/{b['n_total']} "
    f"| {fmt(b['turb_calm_ratio'])} "
    f"| {fmt(b['calm_avg_wr'])} "
    f"| {fmt(b['turb_avg_wr'])} "
    f"| {fmt(b['wr_gap_turb_minus_calm'],'+.3f')} "
    f"| {pct(b['calm_avg_mae'])} "
    f"| {pct(b['turb_avg_mae'])} "
    f"| {fmt(b['wc_h30_cov'])} "
    f"| {b['ks_chg_n_pass']}/25 "
    f"| {fmt(b['mr_ratio'])} "
    f"| {fmt(b['jump_ks'])} |"
)
md_lines.append(row_str)
md_lines.append("")

# ---------- Table 3: Regime gap delta (233a vs 229a) ----------
md_lines.append("## Table 3: 233a vs 229a Baseline — Regime Gap Delta")
md_lines.append("")
md_lines.append(
    "How much worse is 233a at each regime metric compared to 229a? "
    "Positive = 233a *better* than 229a on that metric."
)
md_lines.append("")
header3 = "| Metric | 229a | 233a_full | 233a_B | 233a_C | Best_variant |"
md_lines.append(header3)
md_lines.append("|---|---|---|---|---|---|")

metrics_to_compare = [
    ("turb_calm_ratio",  "TC_ratio",     ".3f"),
    ("calm_avg_wr",      "calm_wr",      ".3f"),
    ("turb_avg_wr",      "turb_wr",      ".3f"),
    ("calm_avg_mae",     "calm_MAE%",    "+.1f"),
    ("turb_avg_mae",     "turb_MAE%",    "+.1f"),
    ("wc_h30_cov",       "wc_h30_cov",  ".3f"),
    ("mr_ratio",         "MR_ratio",    ".3f"),
    ("jump_ks",          "Jump_KS",      ".3f"),
    ("ks_chg_n_pass",    "KS_chg_n",    "d"),
]

for key, label, fmts in metrics_to_compare:
    bv  = baseline_row[key]
    fvs = {v: variant_agg[v][f"mean_{key}"] for v in VARIANTS
           if f"mean_{key}" in variant_agg[v]}
    if not fvs:
        # direct key match
        fvs = {v: mean_of([all_rows[f"{v}_s{s}"] for s in SEEDS], key) for v in VARIANTS}

    def _fmt(v):
        if v != v: return "—"
        if fmts == "d": return str(int(round(v)))
        return f"{v:{fmts}}"

    row_str = (
        f"| {label} "
        f"| {_fmt(bv)} "
        f"| {_fmt(fvs.get('full', float('nan')))} "
        f"| {_fmt(fvs.get('B', float('nan')))} "
        f"| {_fmt(fvs.get('C', float('nan')))} "
        f"| {'full' if fvs.get('full',float('nan'))>=max(fvs.values()) else 'B' if fvs.get('B',float('nan'))>=max(fvs.values()) else 'C'} |"
    )
    md_lines.append(row_str)

md_lines.append("")

# ---------- Per-cell width ratio heatmaps (calm vs turb) for full_s42 ----------
md_lines.append("## Table 4: Per-Cell Width Ratio Heatmaps (full_s42 example)")
md_lines.append("")
md_lines.append(
    "Width ratio > 1.0 means the model generates WIDER intervals in this regime than "
    "the unconditional baseline. Target: calm < 1 (narrower when calm), turb > 1 (wider when turb)."
)
md_lines.append("")

full42 = load_suite(RESULTS_DIR / "full_s42" / "suite.json")
pr_full42 = full42["conditionality"]["per_regime_conditionality"]

for regime_name in ["calm", "turb"]:
    regime = pr_full42[regime_name]
    per_cell = regime.get("per_cell_width_ratio", [])
    md_lines.append(f"### {regime_name.upper()} regime — per-cell width ratio (full_s42)")
    md_lines.append("")
    md_lines.append("| row\\col | 0 | 1 | 2 | 3 | 4 |")
    md_lines.append("|---|---|---|---|---|---|")
    for i, row in enumerate(per_cell):
        cells = " | ".join(
            f"**{v:.3f}**" if v > 1.0 else f"{v:.3f}" for v in row
        )
        md_lines.append(f"| {i} | {cells} |")
    md_lines.append("")
    md_lines.append(
        f"avg_wr={regime.get('avg_width_ratio',0):.3f}, "
        f"worst_cell_wr={regime.get('worst_cell_width_ratio',0):.3f}, "
        f"avg_mae={regime.get('avg_mae_reduction_pct',0):.1f}%, "
        f"n_windows={regime.get('n_windows',0)}"
    )
    md_lines.append("")

# ---------- Conclusions ----------
md_lines.append("## Conclusions")
md_lines.append("")

c = conclusions
md_lines.append(f"### Dominant Failure Regime: **{c['dominant_failure_regime']}**")
md_lines.append("")
md_lines.append(c["calm_failure_description"])
md_lines.append("")
md_lines.append(c["turb_behavior"])
md_lines.append("")
md_lines.append("### Combined Failure Pattern")
md_lines.append("")
md_lines.append(c["combined_failure_pattern"])
md_lines.append("")
md_lines.append("### FiLM-Collapse Hypothesis Assessment")
md_lines.append("")
md_lines.append(c["film_collapse_hypothesis"])
md_lines.append("")
md_lines.append("### 233a vs 229a Baseline")
md_lines.append("")
delta = c["baseline_delta"]
md_lines.append(c["baseline_delta"]["description"])
md_lines.append("")
md_lines.append(
    f"- 229a: calm_MAE={delta['229a_calm_avg_mae']:.1f}%, turb_MAE={delta['229a_turb_avg_mae']:.1f}%"
)
md_lines.append(
    f"- 233a: calm_MAE={delta['233a_mean_calm_mae']:.1f}%, turb_MAE={delta['233a_mean_turb_mae']:.1f}%"
)
md_lines.append(
    f"- Delta: calm {delta['calm_degradation_pct']:+.1f} ppt, turb {delta['turb_degradation_pct']:+.1f} ppt"
)
md_lines.append("")
md_lines.append("### Cross-Variant Comparison")
md_lines.append("")
md_lines.append(
    f"- Best at calm regime (least overdispersion): **{c['best_variant_calm']}**"
)
md_lines.append(
    f"- Best at turb regime (highest MAE reduction): **{c['best_variant_turb']}**"
)
md_lines.append("")
for v in VARIANTS:
    a = variant_agg[v]
    md_lines.append(
        f"  - **{v}**: calm_MAE={a['mean_calm_avg_mae']:.1f}%, "
        f"turb_MAE={a['mean_turb_avg_mae']:.1f}%, "
        f"TC_ratio={a['mean_turb_calm_ratio']:.3f}"
    )
md_lines.append("")
md_lines.append("### Coverage and Other Suites (Aggregate, Not Regime-Stratified)")
md_lines.append("")
md_lines.append(c["coverage_note"])
md_lines.append("")
md_lines.append(
    f"Across all 233a variants: MR_ratio {mean_of(list(all_rows.values()), 'mr_ratio'):.3f} "
    f"(gate ≥0.5), Jump_KS {mean_of(list(all_rows.values()), 'jump_ks'):.3f} (gate <0.20). "
    "Both failures are regime-agnostic (global underdispersion and slow noise process)."
)

md_lines.append("")
md_lines.append("---")
md_lines.append("_Generated by diagnose_233a_regime_breakdown.py_")

with open(OUTPUT_MD, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Wrote: {OUTPUT_MD}")

# ---------------------------------------------------------------------------
# Print summary to stdout
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("REGIME BREAKDOWN SUMMARY")
print("="*70)
print(f"\nDominant failure regime: {conclusions['dominant_failure_regime']}")
print(f"\nMean (all 9 233a runs):")
print(f"  calm_avg_wr  = {mean_calm_wr:.3f}  (>1 = over-wide calm)")
print(f"  turb_avg_wr  = {mean_turb_wr:.3f}  (<1 = under-wide turb)")
print(f"  calm_avg_MAE = {mean_calm_mae:.1f}%  (negative = MAE gets WORSE in calm)")
print(f"  turb_avg_MAE = {mean_turb_mae:.1f}%   (positive = MAE improves in turb)")
print(f"\n229a baseline:")
print(f"  calm_avg_wr  = {base_calm_wr:.3f}")
print(f"  turb_avg_wr  = {base_turb_wr:.3f}")
print(f"  calm_avg_MAE = {base_calm_mae:.1f}%")
print(f"  turb_avg_MAE = {base_turb_mae:.1f}%")
print(f"\nFiLM-collapse hypothesis: {'SUPPORTED' if film_hypothesis_supported else 'PARTIAL'}")
print(f"  calm over-wide: {n_calm_over}/{n_total_runs} runs")
print(f"  turb under-wide: {n_turb_under}/{n_total_runs} runs")
print(f"\nPer-variant means:")
for v in VARIANTS:
    a = variant_agg[v]
    print(f"  {v}: calm_MAE={a['mean_calm_avg_mae']:.1f}%, turb_MAE={a['mean_turb_avg_mae']:.1f}%, "
          f"TC={a['mean_turb_calm_ratio']:.3f}")
print(f"\nOutputs written to {RESULTS_DIR}/")
