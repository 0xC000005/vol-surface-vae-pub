#!/bin/bash
# Training: Exp 136a — RC5-H1: Asymmetric CRPS (spread_weight=0.3)
# Result: FALSIFIED — IS has superior gradient structure (score 36.74)
set -euo pipefail
PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
    --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
    --no_ema --epochs 30 --batch_size 8 --noise_dim 32 --n_members 8 \
    --lr_decoder 1e-3 --lambda_vs 0.1 --lambda_es 1.0 \
    --spread_weight 0.3 \
    --ar_frame --ar_cell_spread --ar_noise_skip --ar_skip_bypass_spread \
    --ar_reflect --ar_floor_clamp 0.01 --ar_bias_lambda 0.01 \
    --lambda_cell_var 1.0 --freeze_after_epoch 10 \
    --ar_noisefree_mlp --noise_dist student_t --student_t_df 6.0 \
    --disable_early_stop \
    --output_dir models/backfill/afcrps_136a --device cuda
