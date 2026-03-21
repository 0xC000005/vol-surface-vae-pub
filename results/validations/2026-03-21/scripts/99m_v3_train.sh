#!/bin/bash
# Training: Exp 99m_v3 — IS Fix on Standard AR MLP
# Result: CATASTROPHIC — IS creates tug-of-war with MLP noise pathway
set -euo pipefail
PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
    --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
    --no_ema --epochs 30 --batch_size 8 --noise_dim 32 --n_members 8 \
    --lr_decoder 1e-3 --lambda_vs 0.1 --lambda_es 1.0 --lambda_is 0.05 \
    --ar_frame --ar_cell_spread --ar_noise_skip --ar_skip_bypass_spread \
    --ar_reflect --ar_floor_clamp 0.01 --ar_bias_lambda 0.01 \
    --lambda_cell_var 1.0 --freeze_after_epoch 10 \
    --disable_early_stop \
    --output_dir models/backfill/afcrps_99m_v3 --device cuda
