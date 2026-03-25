#!/bin/bash
set -euo pipefail
# Evaluate 155e (CLN transformer, sw=0.5, lambda_vs=0.01) on val split
PYTHONPATH=. python experiments/backfill/block_ar/eval_cln_transformer.py \
    --model_path models/backfill/flow_155e/final_model.pt \
    --n_samples 50 --eval_split val \
    --output_dir results/block_ar/155e_val_eval --device cuda
