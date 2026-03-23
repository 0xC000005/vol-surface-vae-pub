#!/usr/bin/env bash
# GT Turb/Calm Spread Ratio Robustness — Verification Script
# Runs the GT robustness analysis and saves the JSON result.
set -euo pipefail

REPO_ROOT="/home/max/Documents/vol-surface-vae-pub"
SCRIPT="${REPO_ROOT}/results/validations/2026-03-23/analysis/gt_regime_robustness/gt_turb_calm_robustness.py"
OUT_JSON="${REPO_ROOT}/results/validations/2026-03-23/verification_results/gt_regime_robustness.json"
ANALYSIS_OUT="${REPO_ROOT}/results/validations/2026-03-23/analysis/gt_regime_robustness/gt_turb_calm_robustness_results.json"

cd "${REPO_ROOT}"

echo "======================================================"
echo "GT Turb/Calm Robustness — $(date -Iseconds)"
echo "======================================================"

PYTHONPATH="${REPO_ROOT}" python "${SCRIPT}" 2>&1

# Copy the analysis output JSON to the required verification_results location
cp "${ANALYSIS_OUT}" "${OUT_JSON}"

echo ""
echo "Verification result saved to: ${OUT_JSON}"
