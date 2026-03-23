#!/bin/bash
# Verification: Exp 148c long-horizon (252-day) test
# Why: Boss requirement — long-horizon generation must remain stable and well-calibrated
# Expected outcome: CI coverage reported at 252-day horizon, monotonically growing spread,
#                   no explosion or ceiling effects
set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub
PYTHONPATH=. python experiments/backfill/block_ar/test_long_horizon.py \
    --model_path models/backfill/afcrps_148c/best_model.pt \
    --no_ema --n_windows 10 --n_samples 50 --device cuda
