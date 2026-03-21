#!/usr/bin/env bash
# Long-horizon (252-day) verification for 120b_v6 best_coverage_model
# Model: models/backfill/afcrps_120b_v6/best_coverage_model.pt
# Purpose: Verify whether IS fix maintains distributional quality at 252 days
# Date: 2026-03-21

set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

MODEL_PATH="models/backfill/afcrps_120b_v6/best_coverage_model.pt"
OUTPUT_DIR="results/block_ar/long_horizon_120b_v6"

echo "=== Long-Horizon Test: 120b_v6 best_coverage_model ==="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Start time: $(date -Iseconds)"

PYTHONPATH=. python experiments/backfill/block_ar/test_long_horizon.py \
    --model_path "${MODEL_PATH}" \
    --no_ema \
    --n_windows 10 \
    --n_samples 50 \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda

echo "End time: $(date -Iseconds)"
echo "=== Done ==="
