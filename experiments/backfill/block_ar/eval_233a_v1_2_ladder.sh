#!/usr/bin/env bash
set -euo pipefail

# 233a-v1.2 ladder: 7 variants × 1 seed = 7 training runs + 7 evals.
# Sequential training (GPU OOM-safe per PA-03: max 2 concurrent; we run 1-at-a-time for simplicity).
# Expected wall time ~3-4h total (7 trainings × ~20-30 min each).
# For parallel execution, see Task 5.2 instructions in plan.md.

SEED=42
COMMON="--seed ${SEED} --epochs 60 --batch_size 32 --n_members 8 \
        --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
        --lambda_vs 0.05 --lambda_rv 0.10 --lambda_jump 0.05 \
        --state_reg_window 5 --device cuda"

for VARIANT in control minreg minimal aux link both noreg; do
  OUT_DIR=models/backfill/233a_v1_2_${VARIANT}_25d_s${SEED}
  mkdir -p ${OUT_DIR}
  echo "=== Training v1.2-${VARIANT} seed=${SEED} ==="
  PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
      --variant_name ${VARIANT} --output_dir ${OUT_DIR} \
      ${COMMON} 2>&1 | tee ${OUT_DIR}/training.log
done

echo ""
echo "=== Evaluating 7 v1.2 variants ==="
for VARIANT in control minreg minimal aux link both noreg; do
  OUT_DIR=models/backfill/233a_v1_2_${VARIANT}_25d_s${SEED}
  RESULT_DIR=results/block_ar/233a_v1_2/${VARIANT}_s${SEED}
  mkdir -p ${RESULT_DIR}
  echo "=== Eval v1.2-${VARIANT} ==="
  PYTHONPATH=. python experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py \
      --model_type 233a_v1_2 \
      --checkpoint ${OUT_DIR}/best_model.pt \
      --force_native_anchor \
      --max_windows 192 --samples 48 \
      --output_json ${RESULT_DIR}/suite.json \
      --output_md   ${RESULT_DIR}/suite.md \
      --device cuda 2>&1 | tee ${RESULT_DIR}/eval.log
done

echo "=== Ladder complete. Run compare_233a_v1_2_variants.py for decision. ==="
