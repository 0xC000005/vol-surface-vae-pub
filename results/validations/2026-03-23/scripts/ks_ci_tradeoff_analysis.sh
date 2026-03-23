#!/bin/bash
# KS-CI Trade-off Deep Investigation
# Runs the analysis script and saves outputs to the designated directories.
#
# Usage: bash results/validations/2026-03-23/scripts/ks_ci_tradeoff_analysis.sh

set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

echo "=== KS-CI Trade-off Deep Investigation ==="
echo "Date: $(date -Iseconds)"
echo ""

# Ensure output directories exist
mkdir -p results/validations/2026-03-23/analysis/ks_ci_tradeoff
mkdir -p results/validations/2026-03-23/verification_results

# Run analysis
PYTHONPATH=. python results/validations/2026-03-23/scripts/ks_ci_tradeoff_analysis.py \
    2>&1 | tee results/validations/2026-03-23/analysis/ks_ci_tradeoff/run_log.txt

echo ""
echo "=== Analysis complete ==="
echo "Outputs:"
echo "  - results/validations/2026-03-23/analysis/ks_ci_tradeoff/analysis_results.json"
echo "  - results/validations/2026-03-23/analysis/ks_ci_tradeoff/report.txt"
echo "  - results/validations/2026-03-23/verification_results/ks_ci_tradeoff.json"
