#!/bin/bash
# Training: Exp 120b_v8 — IS Warmup (CRPS-only 10ep + IS 20ep)
# Result: Warmup doesn't help (confounded with freeze timing)
set -euo pipefail
PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
    --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
    --no_ema --epochs 30 --batch_size 8 --noise_dim 32 --n_members 8 \
    --lr_decoder 1e-3 --lambda_vs 0.1 --lambda_es 1.0 --lambda_is 0.05 \
    --is_warmup_epoch 10 \
    --ar_frame --ar_cell_spread --ar_noise_skip --ar_skip_bypass_spread \
    --ar_reflect --ar_floor_clamp 0.01 --ar_bias_lambda 0.01 \
    --lambda_cell_var 1.0 --freeze_after_epoch 10 \
    --ar_noisefree_mlp --noise_dist student_t --student_t_df 6.0 \
    --disable_early_stop \
    --output_dir models/backfill/afcrps_120b_v8 --device cuda
