#!/bin/bash
# Cross-model comparison and factor analysis
# Models: 144a, 144b, 144c, 145a, 145c
# All CausalARTransformerDecoder with CLN
#
# Usage: bash results/validations/2026-03-22/scripts/cross_model_comparison.sh

set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

echo "=== Cross-Model Comparison (2026-03-22) ==="
echo "Starting at $(date)"
echo ""

# Create output directories
mkdir -p results/validations/2026-03-22/analysis/cross_model
mkdir -p results/validations/2026-03-22/verification_results

# Run the Python analysis script
PYTHONPATH=. python results/validations/2026-03-22/scripts/cross_model_comparison.py \
    2>&1 | tee results/validations/2026-03-22/analysis/cross_model/run_log.txt

echo ""
echo "=== Completed at $(date) ==="
echo "Results saved to:"
echo "  - results/validations/2026-03-22/analysis/cross_model/cross_model_results.json"
echo "  - results/validations/2026-03-22/analysis/cross_model/cross_model_summary.md"
echo "  - results/validations/2026-03-22/verification_results/cross_model_comparison.json"
