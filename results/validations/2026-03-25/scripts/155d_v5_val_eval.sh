#!/bin/bash
set -euo pipefail
# Evaluate 155d_v5 (CLN transformer, sw=0.55) on val split
PYTHONPATH=. python experiments/backfill/block_ar/eval_cln_transformer.py \
    --model_path models/backfill/flow_155d_v5/final_model.pt \
    --n_samples 50 --eval_split val \
    --output_dir results/block_ar/155d_v5_val_eval --device cuda
