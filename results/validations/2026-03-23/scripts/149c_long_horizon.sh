#!/usr/bin/env bash
# Verification script: 149c long-horizon (252-day) test
# Date: 2026-03-23
# Question: Does 149c's h=1 CI improvement (+8.1pp) persist at longer horizons?
#           Does kurtosis blowup (3.33 at 30d) worsen at 252d?

set -euo pipefail

REPO_ROOT="/home/max/Documents/vol-surface-vae-pub"
MODEL_PATH="${REPO_ROOT}/models/backfill/afcrps_149c/best_model.pt"
OUTPUT_DIR="${REPO_ROOT}/results/validations/2026-03-23/analysis/149c_long_horizon"

cd "${REPO_ROOT}"

PYTHONPATH=. python experiments/backfill/block_ar/test_long_horizon.py \
    --model_path "${MODEL_PATH}" \
    --no_ema \
    --n_windows 10 \
    --n_samples 50 \
    --device cuda \
    2>&1 | tee "${OUTPUT_DIR}/stdout.txt"

echo "Done. Output saved to ${OUTPUT_DIR}/stdout.txt"
