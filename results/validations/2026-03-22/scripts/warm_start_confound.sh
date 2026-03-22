#!/usr/bin/env bash
# Warm Start Confound Analysis — Reproducible Script
# Investigates whether 144b's results are an artifact of the warm start chain
# (DDPM -> 143a -> 144a -> 144b) rather than a genuine effect of per-cell scale.
#
# Requirements:
#   - Python 3.13+ with torch
#   - Model checkpoints: 143a, 144a, 144b, DDPM base
#   - Run from repo root
#
# Usage: bash results/validations/2026-03-22/scripts/warm_start_confound.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "Running warm start confound analysis from: $REPO_ROOT"

# Verify model files exist
for model in \
    models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
    models/backfill/afcrps_143a/best_model.pt \
    models/backfill/afcrps_144a/best_model.pt \
    models/backfill/afcrps_144b/best_model.pt; do
    if [ ! -f "$model" ]; then
        echo "ERROR: Missing model: $model"
        exit 1
    fi
done

# Create output dirs
mkdir -p results/validations/2026-03-22/analysis/warm_start_confound
mkdir -p results/validations/2026-03-22/verification_results

# Run analysis (CPU only, no GPU needed)
PYTHONPATH=. python results/validations/2026-03-22/scripts/warm_start_confound.py

echo ""
echo "Outputs:"
echo "  Analysis: results/validations/2026-03-22/analysis/warm_start_confound/warm_start_confound_analysis.json"
echo "  Verification: results/validations/2026-03-22/verification_results/warm_start_confound.json"
