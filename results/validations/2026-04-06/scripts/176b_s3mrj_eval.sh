#!/bin/bash
# 176b evaluation under s3mrj harness (11 suites) for fair comparison with 183c
# Run date: 2026-04-06

PYTHONPATH=. python -u experiments/backfill/block_ar/test_block_ar_requirements_v2.py \
    --model_path models/backfill/shared_local_template_mixture_residual_flow_structured_joint_student_t_176b/best_model.pt \
    --no_ema --max_batches 20 --n_samples 50 \
    --output_dir results/block_ar/176b_best_v2_s3mrj_full_30d --device cuda
