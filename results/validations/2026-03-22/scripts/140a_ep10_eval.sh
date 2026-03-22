#!/bin/bash
set -euo pipefail

# Verification script: Evaluate 140a epoch 10 checkpoint
# Purpose: Test whether peak factor structure (eff_rank ~2.68 at ep5)
#          translates to better test suite scores at ep10 vs best_model (ep9)
# Date: 2026-03-22

cd /home/max/Documents/vol-surface-vae-pub

MODEL_PATH="models/backfill/afcrps_140a/checkpoint_epoch_10.pt"
OUTPUT_DIR="results/block_ar/140a_ep10_30d"

echo "=== Evaluating 140a epoch 10 checkpoint ==="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Run full validation (9 test suites)
PYTHONPATH=. python experiments/backfill/block_ar/test_block_ar_requirements_v2.py \
    --model_path "${MODEL_PATH}" \
    --no_ema --max_batches 20 --n_samples 50 \
    --output_dir "${OUTPUT_DIR}" --device cuda

echo ""
echo "=== Computing composite score ==="
python autoresearch-session/compute_score.py "${OUTPUT_DIR}/summary.json"

echo ""
echo "=== Done ==="
