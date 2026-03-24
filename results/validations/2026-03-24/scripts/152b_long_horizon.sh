#!/bin/bash
set -euo pipefail

# Long-horizon verification for 152b AR Flow Matching model
# Generates 252-day paths and checks CI coverage, kurtosis, spread growth, cross-cell correlation

cd /home/max/Documents/vol-surface-vae-pub

PYTHONPATH=. python results/validations/2026-03-24/scripts/152b_long_horizon_test.py \
    --model_path models/backfill/flow_152b/best_model.pt \
    --max_batches 5 \
    --n_samples 50 \
    --output_dir results/validations/2026-03-24/analysis/152b_long_horizon \
    --device cuda \
    --seed 42
