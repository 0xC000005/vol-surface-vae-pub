#!/bin/bash
set -euo pipefail
# Evaluate 155d (CLN transformer, sw=0.5) on val split
# Expects: GPU available, repo root as working directory
PYTHONPATH=. python experiments/backfill/block_ar/eval_cln_transformer.py \
    --model_path models/backfill/flow_155d/final_model.pt \
    --n_samples 50 --eval_split val \
    --output_dir results/block_ar/155d_val_eval --device cuda
