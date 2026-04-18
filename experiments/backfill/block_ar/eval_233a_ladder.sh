#!/usr/bin/env bash
set -euo pipefail

# 233a ladder: 3 variants × 3 seeds = 9 training runs, then eval each.
# Sequential execution, all on CUDA. Expected wall time ~6-8h on RTX 3070 Ti.
#
# Training output dirs pre-existed (created in Task 0.1 with .gitkeep).
# Results dir created as needed.

COMMON_FLAGS="--pca_artifact models/backfill/coarse_pca_233a.npz \
    --epochs 60 --batch_size 32 --n_members 8 \
    --lambda_vs 0.05 --lambda_rv 0.1 --lambda_jump 0.05 --lambda_state 0.1 \
    --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
    --device cuda"

for SEED in 42 1337 2024; do
  for VARIANT in full B C; do
    OUT_DIR=models/backfill/233a_v1_${VARIANT}_25d_s${SEED}
    mkdir -p ${OUT_DIR}
    echo "=== Training variant=${VARIANT} seed=${SEED} (output=${OUT_DIR}) ==="
    PYTHONPATH=. python experiments/backfill/block_ar/train_233a_twopath_factor_ar.py \
        --variant ${VARIANT} --seed ${SEED} --output_dir ${OUT_DIR} \
        ${COMMON_FLAGS} 2>&1 | tee ${OUT_DIR}/training.log
  done
done

# Evaluate each
for SEED in 42 1337 2024; do
  for VARIANT in full B C; do
    OUT_DIR=models/backfill/233a_v1_${VARIANT}_25d_s${SEED}
    RESULT_DIR=results/block_ar/233a/${VARIANT}_s${SEED}
    mkdir -p ${RESULT_DIR}
    echo "=== Eval variant=${VARIANT} seed=${SEED} ==="
    PYTHONPATH=. python experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py \
        --model_type 233a_${VARIANT} \
        --checkpoint ${OUT_DIR}/best_model.pt \
        --force_native_anchor \
        --max_windows 192 --samples 48 \
        --output_json ${RESULT_DIR}/suite.json \
        --output_md   ${RESULT_DIR}/suite.md \
        --device cuda 2>&1 | tee ${RESULT_DIR}/eval.log
  done
done

echo "Ladder complete: 9 training runs + 9 evaluations."
