#!/usr/bin/env bash
# 140a vs 133c comparison — AR causal transformer vs one-shot joint transformer
# This script runs the Python comparison analysis
set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

SCRIPT_DIR="results/validations/2026-03-22/scripts"
OUTPUT_DIR="results/validations/2026-03-22/analysis/140a_vs_133c"
RESULT_DIR="results/validations/2026-03-22/verification_results"

mkdir -p "$OUTPUT_DIR" "$RESULT_DIR"

PYTHONPATH=. python3 "$SCRIPT_DIR/140a_vs_133c_comparison.py" \
    --model_a results/block_ar/140a_30d/summary.json \
    --model_b results/block_ar/133c_30d/summary.json \
    --output_dir "$OUTPUT_DIR" \
    --result_json "$RESULT_DIR/140a_vs_133c_comparison.json"

echo "Comparison complete. Results in $OUTPUT_DIR"
