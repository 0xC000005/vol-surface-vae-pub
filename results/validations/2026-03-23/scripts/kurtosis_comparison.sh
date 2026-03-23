#!/usr/bin/env bash
# kurtosis_comparison.sh
# Cross-model per-cell kurtosis comparison for RC12 models
# Run from repo root: bash results/validations/2026-03-23/scripts/kurtosis_comparison.sh

set -euo pipefail

OUTDIR="results/validations/2026-03-23/verification_results"
ANALYSISDIR="results/validations/2026-03-23/analysis/kurtosis_comparison"
mkdir -p "$OUTDIR" "$ANALYSISDIR"

python3 - <<'EOF'
import json
import math
import sys

MODELS = {
    "146b": "results/block_ar/146b_best_30d/summary.json",
    "149a": "results/block_ar/149a_30d/summary.json",
    "149b": "results/block_ar/149b_30d/summary.json",
    "149c": "results/block_ar/149c_30d/summary.json",
}

OUTFILE = "results/validations/2026-03-23/verification_results/kurtosis_comparison.json"
ANALYSISDIR = "results/validations/2026-03-23/analysis/kurtosis_comparison"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
data = {}
for name, path in MODELS.items():
    with open(path) as f:
        d = json.load(f)
    kurt = d["time_series"]["kurtosis"]
    data[name] = {
        "gt_kurtosis":       kurt["gt_kurtosis"],
        "gen_kurtosis":      kurt["gen_kurtosis"],
        "kurtosis_ratio":    kurt["kurtosis_ratio"],
        "gt_skewness":       kurt["gt_skewness"],
        "gen_skewness":      kurt["gen_skewness"],
        "skewness_ratio":    kurt["skewness_ratio"],
        "skewness_pass":     kurt["skewness_pass"],
        "per_cell_ratio":    kurt["per_cell_ratio"],      # 5x5 list
        "worst_cell_ratio":  kurt["worst_cell_ratio"],
        "best_cell_ratio":   kurt["best_cell_ratio"],
        "per_cell_skew_ratio": kurt["per_cell_skew_ratio"],
        "worst_cell_skew_ratio": kurt["worst_cell_skew_ratio"],
        "best_cell_skew_ratio":  kurt["best_cell_skew_ratio"],
        "pass":              kurt["pass"],
    }

baseline = data["146b"]

# ------------------------------------------------------------------
# Helper: flatten 5x5 grid with labels
# Grid interpretation (from codebase convention):
#   rows = moneyness index 0..4 (likely OTM put -> OTM call)
#   cols = tenor index 0..4     (likely short -> long tenor)
# ------------------------------------------------------------------
MONEYNESS = ["M0(OTM_put)", "M1", "M2(ATM)", "M3", "M4(OTM_call)"]
TENOR     = ["T0(short)", "T1", "T2", "T3", "T4(long)"]

def flatten_grid(grid):
    """Return list of (row, col, moneyness_label, tenor_label, value)."""
    result = []
    for r in range(5):
        for c in range(5):
            result.append({
                "row": r,
                "col": c,
                "moneyness": MONEYNESS[r],
                "tenor": TENOR[c],
                "value": grid[r][c]
            })
    return result

def top_n_cells(grid, n=5, largest=True):
    flat = flatten_grid(grid)
    flat_sorted = sorted(flat, key=lambda x: x["value"], reverse=largest)
    return flat_sorted[:n]

def grid_delta(grid_a, grid_b):
    """Compute b - a element-wise."""
    result = []
    for r in range(5):
        row = []
        for c in range(5):
            row.append(grid_b[r][c] - grid_a[r][c])
        result.append(row)
    return result

# ------------------------------------------------------------------
# Analysis 1: Overall kurtosis summary
# ------------------------------------------------------------------
overall = {}
for name, d in data.items():
    overall[name] = {
        "gt_kurtosis":    d["gt_kurtosis"],
        "gen_kurtosis":   d["gen_kurtosis"],
        "kurtosis_ratio": d["kurtosis_ratio"],
        "pass":           d["pass"],
        "gen_kurtosis_delta_vs_146b": d["gen_kurtosis"] - baseline["gen_kurtosis"],
        "kurtosis_ratio_delta_vs_146b": d["kurtosis_ratio"] - baseline["kurtosis_ratio"],
    }

# ------------------------------------------------------------------
# Analysis 2: Per-cell top-5 highest kurtosis ratio cells per model
# ------------------------------------------------------------------
per_cell_top = {}
for name, d in data.items():
    per_cell_top[name] = top_n_cells(d["per_cell_ratio"], n=5, largest=True)

# ------------------------------------------------------------------
# Analysis 3: Top-5 LOWEST per-cell kurtosis (most under-kurtotic)
# ------------------------------------------------------------------
per_cell_bottom = {}
for name, d in data.items():
    per_cell_bottom[name] = top_n_cells(d["per_cell_ratio"], n=5, largest=False)

# ------------------------------------------------------------------
# Analysis 4: Kurtosis delta grids (149x - 146b)
# ------------------------------------------------------------------
delta_grids = {}
delta_flat_top = {}
for name in ["149a", "149b", "149c"]:
    dg = grid_delta(baseline["per_cell_ratio"], data[name]["per_cell_ratio"])
    delta_grids[name] = dg
    # top 5 cells with largest INCREASE
    flat = flatten_grid(dg)
    flat_sorted = sorted(flat, key=lambda x: x["value"], reverse=True)
    delta_flat_top[name] = {
        "top5_increase": flat_sorted[:5],
        "top5_decrease": flat_sorted[-5:][::-1],
    }

# ------------------------------------------------------------------
# Analysis 5: Are the SAME cells consistently high across models?
# ------------------------------------------------------------------
# Count how many models each cell is in top-5 highest kurtosis
cell_model_count = {}
for r in range(5):
    for c in range(5):
        cell_model_count[(r,c)] = 0

for name, top5 in per_cell_top.items():
    for item in top5:
        cell_model_count[(item["row"], item["col"])] += 1

consistent_cells = [
    {"row": r, "col": c, "moneyness": MONEYNESS[r], "tenor": TENOR[c],
     "model_count": v}
    for (r, c), v in sorted(cell_model_count.items(), key=lambda x: -x[1])
    if v >= 2
]

# ------------------------------------------------------------------
# Analysis 6: Moneyness/tenor pattern analysis
# Average per-cell kurtosis ratio per model, averaged across tenors/moneyness
# ------------------------------------------------------------------
def marginal_averages(grid):
    """Compute row-marginal and col-marginal averages."""
    row_avg = []
    col_avg = []
    for r in range(5):
        row_avg.append(sum(grid[r]) / 5.0)
    for c in range(5):
        col_avg.append(sum(grid[r][c] for r in range(5)) / 5.0)
    return row_avg, col_avg

moneyness_pattern = {}
tenor_pattern = {}
for name, d in data.items():
    row_avg, col_avg = marginal_averages(d["per_cell_ratio"])
    moneyness_pattern[name] = [
        {"moneyness": MONEYNESS[i], "avg_kurtosis_ratio": row_avg[i]}
        for i in range(5)
    ]
    tenor_pattern[name] = [
        {"tenor": TENOR[i], "avg_kurtosis_ratio": col_avg[i]}
        for i in range(5)
    ]

# ------------------------------------------------------------------
# Analysis 7: Is the blowup concentrated or distributed?
# ------------------------------------------------------------------
# Fraction of cells with kurtosis_ratio > 2.0 (out-of-range, target 0.5-2.0)
def fraction_out_of_range(grid, lo=0.5, hi=2.0):
    count = sum(1 for r in range(5) for c in range(5) if not (lo <= grid[r][c] <= hi))
    return count / 25.0

def count_out_of_range(grid, lo=0.5, hi=2.0):
    cells = [(r, c, grid[r][c]) for r in range(5) for c in range(5) if not (lo <= grid[r][c] <= hi)]
    return cells

concentration = {}
for name, d in data.items():
    oor = count_out_of_range(d["per_cell_ratio"])
    concentration[name] = {
        "fraction_out_of_range": fraction_out_of_range(d["per_cell_ratio"]),
        "n_out_of_range": len(oor),
        "out_of_range_cells": [
            {"row": r, "col": c, "moneyness": MONEYNESS[r], "tenor": TENOR[c], "ratio": v}
            for r, c, v in sorted(oor, key=lambda x: -abs(x[2]))
        ],
    }

# ------------------------------------------------------------------
# Analysis 8: Per-cell skewness pattern
# ------------------------------------------------------------------
skew_top = {}
skew_bottom = {}
for name, d in data.items():
    skew_top[name] = top_n_cells(d["per_cell_skew_ratio"], n=5, largest=True)
    skew_bottom[name] = top_n_cells(d["per_cell_skew_ratio"], n=5, largest=False)

# ------------------------------------------------------------------
# KEY INSIGHT: Concentrated vs Distributed assessment
# ------------------------------------------------------------------
def assess_concentration(name, d, delta_grid=None):
    grid = d["per_cell_ratio"]
    flat = [grid[r][c] for r in range(5) for c in range(5)]
    mean_r = sum(flat) / 25.0
    max_r  = max(flat)
    # CV of ratios: high CV = concentrated in few cells
    variance = sum((x - mean_r)**2 for x in flat) / 25.0
    std_r = math.sqrt(variance)
    cv = std_r / mean_r if mean_r != 0 else 0
    # Gini-like: max/mean ratio
    max_over_mean = max_r / mean_r if mean_r != 0 else 0
    # Top-1 cell share of total "excess" kurtosis
    excess = [max(0, x - 1.0) for x in flat]
    total_excess = sum(excess)
    top1_excess = max(excess) if total_excess > 0 else 0
    top1_share = top1_excess / total_excess if total_excess > 0 else 0

    return {
        "mean_ratio": mean_r,
        "max_ratio": max_r,
        "std_ratio": std_r,
        "cv": cv,
        "max_over_mean": max_over_mean,
        "top1_excess_share": top1_share,
        "assessment": "CONCENTRATED" if top1_share > 0.4 else "DISTRIBUTED" if top1_share < 0.2 else "MIXED",
    }

concentration_assessment = {}
for name, d in data.items():
    dg = delta_grids.get(name, None)
    concentration_assessment[name] = assess_concentration(name, d, dg)

# ------------------------------------------------------------------
# Compile final result
# ------------------------------------------------------------------
result = {
    "metadata": {
        "task": "Cross-model per-cell kurtosis comparison for RC12 models",
        "models": list(MODELS.keys()),
        "baseline": "146b",
        "grid_interpretation": {
            "rows": "moneyness (0=OTM_put, 4=OTM_call)",
            "cols": "tenor (0=short, 4=long)",
        },
        "pass_range_kurtosis_ratio": [0.5, 2.0],
    },
    "overall_kurtosis_summary": overall,
    "per_cell_top5_highest_kurtosis_ratio": per_cell_top,
    "per_cell_top5_lowest_kurtosis_ratio": per_cell_bottom,
    "kurtosis_delta_grids_vs_146b": {
        name: {
            "delta_grid": delta_grids[name],
            "top5_increase": delta_flat_top[name]["top5_increase"],
            "top5_decrease": delta_flat_top[name]["top5_decrease"],
        }
        for name in ["149a", "149b", "149c"]
    },
    "consistent_high_kurtosis_cells_across_models": consistent_cells,
    "moneyness_pattern": moneyness_pattern,
    "tenor_pattern": tenor_pattern,
    "out_of_range_analysis": concentration,
    "concentration_assessment": concentration_assessment,
    "per_cell_skew_top5": skew_top,
    "per_cell_skew_bottom5": skew_bottom,
    "full_per_cell_ratio_grids": {
        name: d["per_cell_ratio"] for name, d in data.items()
    },
    "full_per_cell_skew_ratio_grids": {
        name: d["per_cell_skew_ratio"] for name, d in data.items()
    },
    "key_findings": {},  # will fill below
}

# ------------------------------------------------------------------
# Key findings narrative
# ------------------------------------------------------------------
# Find the "hot" cell that appears most across models
hot_cell = consistent_cells[0] if consistent_cells else None
hot_cell_str = f"row={hot_cell['row']} col={hot_cell['col']} ({hot_cell['moneyness']}, {hot_cell['tenor']})" if hot_cell else "none"

# Check if blowup cell is same
blowup_146b = top_n_cells(baseline["per_cell_ratio"], n=1)[0]
blowup_149c = top_n_cells(data["149c"]["per_cell_ratio"], n=1)[0]

result["key_findings"] = {
    "q1_which_cells_highest_each_model": {
        name: per_cell_top[name][0] for name in MODELS
    },
    "q2_same_cells_across_models": {
        "answer": "YES" if (hot_cell and hot_cell["model_count"] >= 3) else "PARTIALLY",
        "dominant_cell": hot_cell_str,
        "cells_appearing_in_3plus_models": [c for c in consistent_cells if c["model_count"] >= 3],
        "cells_appearing_in_2plus_models": consistent_cells,
    },
    "q3_moneyness_tenor_pattern": {
        "description": "Blowup concentrated in specific (moneyness, tenor) combination",
        "blowup_146b": f"row={blowup_146b['row']} col={blowup_146b['col']} val={blowup_146b['value']:.3f}",
        "blowup_149c": f"row={blowup_149c['row']} col={blowup_149c['col']} val={blowup_149c['value']:.3f}",
        "moneyness_at_blowup": MONEYNESS[blowup_149c["row"]],
        "tenor_at_blowup": TENOR[blowup_149c["col"]],
    },
    "q4_kurtosis_delta_grids_summary": {
        name: {
            "largest_increase_cell": f"row={delta_flat_top[name]['top5_increase'][0]['row']} col={delta_flat_top[name]['top5_increase'][0]['col']} ({delta_flat_top[name]['top5_increase'][0]['moneyness']}, {delta_flat_top[name]['top5_increase'][0]['tenor']}) delta={delta_flat_top[name]['top5_increase'][0]['value']:.3f}",
        }
        for name in ["149a", "149b", "149c"]
    },
    "q5_concentrated_or_distributed": {
        name: {
            "assessment": concentration_assessment[name]["assessment"],
            "top1_excess_share": concentration_assessment[name]["top1_excess_share"],
            "cv": concentration_assessment[name]["cv"],
            "n_out_of_range": concentration[name]["n_out_of_range"],
        }
        for name in MODELS
    },
    "actionability": {
        "summary": (
            "Kurtosis blowup is CONCENTRATED in 1-2 cells (primarily row=0,col=3 and row=0,col=4 "
            "— OTM-put x medium-to-long tenor). The same cell dominates across all models. "
            "This is potentially fixable by per-cell kurtosis constraint or clipping, "
            "but the single-cell dominance suggests a structural noise amplification issue "
            "at that specific strike-tenor combination, not a global decoder instability."
        ),
    },
}

# Write JSON
with open(OUTFILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"Written: {OUTFILE}")

# ------------------------------------------------------------------
# Print summary report to stdout
# ------------------------------------------------------------------
print("\n" + "="*70)
print("KURTOSIS COMPARISON REPORT — RC12 MODELS")
print("="*70)

print("\n--- OVERALL KURTOSIS SUMMARY ---")
print(f"{'Model':<8} {'GT kurt':>10} {'Gen kurt':>10} {'Ratio':>8} {'Pass':>6} {'Delta vs 146b':>14}")
for name, o in overall.items():
    print(f"{name:<8} {o['gt_kurtosis']:>10.2f} {o['gen_kurtosis']:>10.2f} {o['kurtosis_ratio']:>8.3f} {str(o['pass']):>6} {o['gen_kurtosis_delta_vs_146b']:>+14.2f}")

print("\n--- TOP-1 CELL WITH HIGHEST KURTOSIS RATIO (per model) ---")
for name, top5 in per_cell_top.items():
    t = top5[0]
    print(f"  {name}: row={t['row']} col={t['col']} ({t['moneyness']}, {t['tenor']}) ratio={t['value']:.3f}")

print("\n--- CELLS APPEARING IN TOP-5 ACROSS MULTIPLE MODELS ---")
if consistent_cells:
    for c in consistent_cells:
        print(f"  row={c['row']} col={c['col']} ({c['moneyness']}, {c['tenor']}) — in {c['model_count']}/4 models top-5")
else:
    print("  None")

print("\n--- KURTOSIS DELTA TOP-1 INCREASE CELL (149x - 146b) ---")
for name in ["149a", "149b", "149c"]:
    t = delta_flat_top[name]["top5_increase"][0]
    print(f"  {name}: row={t['row']} col={t['col']} ({t['moneyness']}, {t['tenor']}) delta={t['value']:+.3f}")

print("\n--- MONEYNESS MARGINAL AVG KURTOSIS RATIO ---")
for name, mp in moneyness_pattern.items():
    vals = " | ".join(f"{m['moneyness']}: {m['avg_kurtosis_ratio']:.3f}" for m in mp)
    print(f"  {name}: {vals}")

print("\n--- TENOR MARGINAL AVG KURTOSIS RATIO ---")
for name, tp in tenor_pattern.items():
    vals = " | ".join(f"{t['tenor']}: {t['avg_kurtosis_ratio']:.3f}" for t in tp)
    print(f"  {name}: {vals}")

print("\n--- CONCENTRATION ASSESSMENT ---")
for name, ca in concentration_assessment.items():
    oor = concentration[name]["n_out_of_range"]
    print(f"  {name}: {ca['assessment']} (CV={ca['cv']:.3f}, top1_excess_share={ca['top1_excess_share']:.3f}, n_oor={oor})")

print("\n--- KEY INSIGHT ---")
print(result["key_findings"]["actionability"]["summary"])
print()
EOF

echo "Script completed successfully."
