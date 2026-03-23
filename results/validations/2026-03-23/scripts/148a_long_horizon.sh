#!/bin/bash
# Verification: Exp 148a long-horizon (252-day) test
# Why: Boss requirement — verify ensemble spread grows with horizon, no explosion/ceiling
#      at 252-day horizon for Exp 148a model.
# Expected outcome:
#   - 252-day CI coverage > 0 (model survives to 252 days)
#   - Ensemble spread grows monotonically (or at least does not collapse) with horizon
#   - No explosion (IV stays in reasonable range)
# Run from repo root: bash results/validations/2026-03-23/scripts/148a_long_horizon.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="models/backfill/afcrps_148a/best_model.pt"
OUTPUT_DIR="results/validations/2026-03-23/analysis/148a_long_horizon"

mkdir -p "$OUTPUT_DIR"

PYTHONPATH=. python experiments/backfill/block_ar/test_long_horizon.py \
    --model_path "$MODEL_PATH" \
    --no_ema \
    --n_windows 10 \
    --n_samples 50 \
    --device cuda \
    2>&1 | tee "$OUTPUT_DIR/stdout.txt"

echo "Done. Output saved to $OUTPUT_DIR/stdout.txt"
