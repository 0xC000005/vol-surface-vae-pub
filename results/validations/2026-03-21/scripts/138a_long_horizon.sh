#!/usr/bin/env bash
# Long-horizon (252-day) verification for model 138a
# The first 6/8 model — checking if per-horizon IS + noise-free MLP holds at extended horizons
set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

PYTHONPATH=. python experiments/backfill/block_ar/test_long_horizon.py \
    --model_path models/backfill/afcrps_138a/best_model.pt \
    --no_ema \
    --n_samples 50 \
    --n_windows 10 \
    --batch_size 8 \
    --output_dir results/block_ar/long_horizon_138a \
    --device cuda
