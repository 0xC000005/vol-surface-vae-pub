#!/usr/bin/env bash
# Verification script: 148_probe per-cell CI analysis
# Verifies claim: "Scaling noise by 3x has ZERO effect on CI"
# Run from repo root: bash results/validations/2026-03-23/scripts/148_probe_percell_ci.sh

set -euo pipefail

REPO_ROOT="/home/max/Documents/vol-surface-vae-pub"
ANALYSIS_DIR="$REPO_ROOT/results/validations/2026-03-23/analysis/148_probe_percell"
RESULTS_DIR="$REPO_ROOT/results/validations/2026-03-23/verification_results"

mkdir -p "$ANALYSIS_DIR"
mkdir -p "$RESULTS_DIR"

cd "$REPO_ROOT"

python3 - <<'PYEOF'
import json
import os
import numpy as np

REPO_ROOT = "/home/max/Documents/vol-surface-vae-pub"
ANALYSIS_DIR = os.path.join(REPO_ROOT, "results/validations/2026-03-23/analysis/148_probe_percell")
RESULTS_DIR = os.path.join(REPO_ROOT, "results/validations/2026-03-23/verification_results")

BETAS = [1.0, 1.5, 2.0, 3.0]
HORIZONS = [1, 7, 14, 30]
HORIZON_KEYS = ["1", "7", "14", "30"]

# ---------- Load all summary.json files ----------
data = {}
for beta in BETAS:
    path = os.path.join(REPO_ROOT, f"results/block_ar/148_probe_beta{beta}/summary.json")
    with open(path) as f:
        data[beta] = json.load(f)
    print(f"Loaded beta={beta}: {path}")

# ---------- Extract per_cell_coverage grids ----------
# per_cell_coverage[horizon_str] -> 5x5 list of floats
grids = {}  # grids[beta][horizon_int] = np.array 5x5
for beta in BETAS:
    grids[beta] = {}
    pcc = data[beta]["coverage"]["per_cell_coverage"]
    for hk, h in zip(HORIZON_KEYS, HORIZONS):
        arr = np.array(pcc[hk])  # shape (5,5)
        assert arr.shape == (5, 5), f"Expected (5,5), got {arr.shape} for beta={beta}, h={h}"
        grids[beta][h] = arr

print("\n--- Per-cell coverage grids loaded (5x5, coverage fractions) ---")

# ---------- Compute delta grids (beta_X - beta_1.0) ----------
baseline = grids[1.0]

results = {
    "claim": "Scaling noise by 3x has ZERO effect on CI",
    "specific_claims": {
        "h7_mean_delta_beta3_minus_beta1_pp": -1.14,
        "h30_mean_delta_beta3_minus_beta1_pp": 0.63,
        "no_cell_improves_more_than_3pp": True
    },
    "horizons_analyzed": HORIZONS,
    "betas_analyzed": BETAS,
    "per_horizon_analysis": {},
    "exceptional_cells": [],
    "claim_verification": {}
}

print("\n--- Delta analysis (beta_X - beta_1.0), in percentage points ---")
print(f"{'Horizon':>8} | {'Beta':>5} | {'Mean delta (pp)':>16} | {'Max improve (pp)':>17} | {'Max degrade (pp)':>17}")
print("-" * 75)

for h in HORIZONS:
    results["per_horizon_analysis"][str(h)] = {}
    for beta in BETAS:
        delta = (grids[beta][h] - baseline[h]) * 100.0  # convert to pp
        mean_d = float(delta.mean())
        max_improve = float(delta.max())
        max_degrade = float(delta.min())
        results["per_horizon_analysis"][str(h)][str(beta)] = {
            "mean_delta_pp": round(mean_d, 4),
            "max_improvement_pp": round(max_improve, 4),
            "max_degradation_pp": round(max_degrade, 4),
            "delta_grid_pp": delta.tolist()
        }
        print(f"{h:>8} | {beta:>5} | {mean_d:>+16.4f} | {max_improve:>+17.4f} | {max_degrade:>+17.4f}")

# ---------- Verify specific claims ----------
h7_delta_beta3 = results["per_horizon_analysis"]["7"]["3.0"]["mean_delta_pp"]
h30_delta_beta3 = results["per_horizon_analysis"]["30"]["3.0"]["mean_delta_pp"]

claim_h7 = abs(h7_delta_beta3 - (-1.14)) < 0.5  # within 0.5pp tolerance
claim_h30 = abs(h30_delta_beta3 - 0.63) < 0.5   # within 0.5pp tolerance

# Check if any cell improves more than 3pp at beta=3.0 across all horizons
any_cell_gt3pp = False
any_cell_gt5pp = False
for h in HORIZONS:
    delta_grid = np.array(results["per_horizon_analysis"][str(h)]["3.0"]["delta_grid_pp"])
    if delta_grid.max() > 3.0:
        any_cell_gt3pp = True
        # Find which cells
        rows, cols = np.where(delta_grid > 3.0)
        for r, c in zip(rows, cols):
            results["exceptional_cells"].append({
                "horizon": h, "beta": 3.0,
                "row": int(r), "col": int(c),
                "delta_pp": round(float(delta_grid[r, c]), 4)
            })
    if delta_grid.max() > 5.0:
        any_cell_gt5pp = True

results["claim_verification"] = {
    "h7_mean_delta_beta3_actual_pp": round(h7_delta_beta3, 4),
    "h7_mean_delta_beta3_claimed_pp": -1.14,
    "h7_claim_confirmed": claim_h7,
    "h30_mean_delta_beta3_actual_pp": round(h30_delta_beta3, 4),
    "h30_mean_delta_beta3_claimed_pp": 0.63,
    "h30_claim_confirmed": claim_h30,
    "any_cell_improves_gt3pp_beta3": any_cell_gt3pp,
    "any_cell_improves_gt5pp_beta3": any_cell_gt5pp,
    "no_cell_gt3pp_claim_confirmed": not any_cell_gt3pp,
    "overall_zero_effect_claim_confirmed": (claim_h7 and claim_h30 and not any_cell_gt3pp)
}

print("\n--- Claim Verification ---")
print(f"H7  mean delta beta=3.0: actual={h7_delta_beta3:+.4f}pp, claimed={-1.14}pp => {'CONFIRMED' if claim_h7 else 'REJECTED'}")
print(f"H30 mean delta beta=3.0: actual={h30_delta_beta3:+.4f}pp, claimed={+0.63}pp => {'CONFIRMED' if claim_h30 else 'REJECTED'}")
print(f"No cell improves >3pp at beta=3.0: {'CONFIRMED' if not any_cell_gt3pp else 'REJECTED'}")
print(f"Any cell improves >5pp at beta=3.0: {any_cell_gt5pp}")
print(f"Overall 'ZERO effect' claim: {'CONFIRMED' if results['claim_verification']['overall_zero_effect_claim_confirmed'] else 'REJECTED'}")

if results["exceptional_cells"]:
    print(f"\nExceptional cells (>3pp improvement at beta=3.0): {len(results['exceptional_cells'])}")
    for ec in results["exceptional_cells"]:
        print(f"  h={ec['horizon']} row={ec['row']} col={ec['col']} delta={ec['delta_pp']:+.4f}pp")

# ---------- Save per-cell grids for visual inspection ----------
grids_output = {}
for beta in BETAS:
    grids_output[str(beta)] = {}
    for h in HORIZONS:
        grids_output[str(beta)][str(h)] = (grids[beta][h] * 100.0).tolist()  # as %

with open(os.path.join(ANALYSIS_DIR, "percell_coverage_grids_pct.json"), "w") as f:
    json.dump(grids_output, f, indent=2)
print(f"\nSaved per-cell coverage grids to: {ANALYSIS_DIR}/percell_coverage_grids_pct.json")

# ---------- Save full analysis ----------
with open(os.path.join(ANALYSIS_DIR, "percell_ci_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved full analysis to: {ANALYSIS_DIR}/percell_ci_analysis.json")

# ---------- Write verification_result.json ----------
verification_result = {
    "task": "148_probe_percell_ci",
    "date": "2026-03-23",
    "claim": "Scaling noise by 3x has ZERO effect on CI",
    "data_sources": [
        f"results/block_ar/148_probe_beta{b}/summary.json" for b in BETAS
    ],
    "verification_outcome": results["claim_verification"],
    "summary": {
        "h7_mean_delta_pp": round(h7_delta_beta3, 4),
        "h30_mean_delta_pp": round(h30_delta_beta3, 4),
        "any_cell_gt3pp": any_cell_gt3pp,
        "any_cell_gt5pp": any_cell_gt5pp,
        "n_exceptional_cells": len(results["exceptional_cells"]),
        "exceptional_cells": results["exceptional_cells"]
    },
    "per_horizon_summary": {
        str(h): {
            str(beta): {
                "mean_delta_pp": results["per_horizon_analysis"][str(h)][str(beta)]["mean_delta_pp"],
                "max_improvement_pp": results["per_horizon_analysis"][str(h)][str(beta)]["max_improvement_pp"],
                "max_degradation_pp": results["per_horizon_analysis"][str(h)][str(beta)]["max_degradation_pp"]
            }
            for beta in BETAS
        }
        for h in HORIZONS
    },
    "output_files": {
        "full_analysis": f"{ANALYSIS_DIR}/percell_ci_analysis.json",
        "coverage_grids": f"{ANALYSIS_DIR}/percell_coverage_grids_pct.json",
        "verification_result": f"{RESULTS_DIR}/148_probe_percell.json"
    }
}

out_path = os.path.join(RESULTS_DIR, "148_probe_percell.json")
with open(out_path, "w") as f:
    json.dump(verification_result, f, indent=2)
print(f"Saved verification_result.json to: {out_path}")

print("\n=== VERIFICATION COMPLETE ===")
PYEOF

echo "Script completed."
