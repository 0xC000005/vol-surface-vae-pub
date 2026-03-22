#!/bin/bash
set -euo pipefail

# Exp 144b comprehensive diagnostics verification script
# Run from repo root: bash results/validations/2026-03-22/scripts/144b_diagnostics.sh

cd /home/max/Documents/vol-surface-vae-pub

# Create output directories
mkdir -p results/validations/2026-03-22/analysis/144b
mkdir -p results/validations/2026-03-22/verification_results

# Run the diagnostics
PYTHONPATH=. python results/validations/2026-03-22/scripts/144b_diagnostics.py \
    2>&1 | tee results/validations/2026-03-22/analysis/144b/diagnostics_stdout.log

echo ""
echo "Diagnostics complete. Results saved to:"
echo "  results/validations/2026-03-22/analysis/144b/diagnostics_results.json"
echo "  results/validations/2026-03-22/analysis/144b/diagnostics_stdout.log"
