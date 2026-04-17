#!/bin/bash
# Evaluate the 231a three-way ablation + comparison table.
# Runs evaluate_220b (7-suite) + diagnose_230a_scale_state on each variant
# and writes consolidated comparison JSON.

set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p results/block_ar/231a_eval

for VARIANT in none fixed learn; do
    CKPT="models/backfill/hybrid_231a_${VARIANT}/best_model.pt"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP 231a-${VARIANT}: $CKPT does not exist yet"
        continue
    fi
    echo ""
    echo "=============================================================="
    echo "  Evaluating 231a-${VARIANT}"
    echo "=============================================================="

    # 7-suite evaluation (native rollout, uses model's sample_batched)
    PYTHONPATH=. python -u experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py \
        --model_type 231a \
        --checkpoint "$CKPT" \
        --max_windows 192 --samples 48 \
        --batch_size 16 --chunk_size 8 \
        --conditionality_samples 32 \
        --output_json "results/block_ar/231a_eval/231a_${VARIANT}_suite.json" \
        --output_md   "results/block_ar/231a_eval/231a_${VARIANT}_suite.md" \
        --device cuda

    # 230a-style scale/state diagnostic (9-mode intervention matrix)
    PYTHONPATH=. python -u experiments/backfill/block_ar/diagnose_230a_scale_state.py \
        --checkpoints "$CKPT" \
        --checkpoint_tags "231a_${VARIANT}_best" \
        --model_type 231a \
        --max_windows 200 --n_members 48 --batch_size 16 \
        --horizons 1,5,10,20,30 \
        --output_dir "results/block_ar/231a_eval/231a_${VARIANT}_diagnostic" \
        --device cuda
done

# Comparison table + promotion decision
echo ""
echo "=============================================================="
echo "  Comparison Table (231a three-way ablation)"
echo "=============================================================="
PYTHONPATH=. python -u experiments/backfill/block_ar/compare_231a_variants.py
