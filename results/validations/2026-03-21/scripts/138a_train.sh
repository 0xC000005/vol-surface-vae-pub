#!/bin/bash
# Training: Exp 138a — Per-Horizon IS Aggregation — FIRST EVER 6/8
# Purpose: Mean(H,W) per timestep then sum(T) instead of sum(T,H,W)
# Result: 6/8 suites (S1,S3,S4,S5,S6,S8), score 76.62, CI 78.7%
# Note: IS magnitude accidentally reduced 25x (mean vs sum), creating weak IS
set -euo pipefail
PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
    --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
    --no_ema --epochs 30 --batch_size 8 --noise_dim 32 --n_members 8 \
    --lr_decoder 1e-3 --lambda_vs 0.1 --lambda_es 1.0 --lambda_is 0.05 \
    --ar_frame --ar_cell_spread --ar_noise_skip --ar_skip_bypass_spread \
    --ar_reflect --ar_floor_clamp 0.01 --ar_bias_lambda 0.01 \
    --lambda_cell_var 1.0 --freeze_after_epoch 10 \
    --ar_noisefree_mlp --noise_dist student_t --student_t_df 6.0 \
    --disable_early_stop \
    --output_dir models/backfill/afcrps_138a --device cuda
# NOTE: 138a uses per-horizon IS aggregation (code change in interval_score()),
# not a CLI flag. The code change is in commit d5d3a4e.
