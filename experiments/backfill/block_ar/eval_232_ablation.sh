#!/bin/bash
# Evaluate the 232a/b/c/d architectural-attack series.
# For each variant: evaluate_220b suite + diagnose_230a mechanistic + compare.

set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p results/block_ar/232_eval

declare -A CKPT_PATH=(
    [232a]="models/backfill/hybrid_232a_mixture/best_model.pt"
    [232b]="models/backfill/hybrid_232b_heavytail/best_model.pt"
    [232c]="models/backfill/hybrid_232c_jumpaux/best_model.pt"
    [232d]="models/backfill/hybrid_232d_cellweights/best_model.pt"
)

for VARIANT in 232a 232b 232c 232d; do
    CKPT="${CKPT_PATH[$VARIANT]}"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP $VARIANT: $CKPT does not exist yet"
        continue
    fi
    echo ""
    echo "=============================================================="
    echo "  Evaluating $VARIANT"
    echo "=============================================================="

    # Main 7-suite eval (with inference anchor flipped on for 232 variants)
    PYTHONPATH=. python -u experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py \
        --model_type $VARIANT \
        --checkpoint "$CKPT" \
        --max_windows 192 --samples 48 \
        --batch_size 16 --chunk_size 8 \
        --conditionality_samples 32 \
        --output_json "results/block_ar/232_eval/${VARIANT}_suite.json" \
        --output_md   "results/block_ar/232_eval/${VARIANT}_suite.md" \
        --device cuda

    # 230a-style mechanistic diagnostic (only for 232a/b — they have new forwards)
    # 232c/d use 227a's forward, so m4_native_anc050 ≈ same as 229a
    if [ "$VARIANT" = "232a" ] || [ "$VARIANT" = "232b" ]; then
        PYTHONPATH=. python -u experiments/backfill/block_ar/diagnose_230a_scale_state.py \
            --checkpoints "$CKPT" \
            --checkpoint_tags "${VARIANT}_best" \
            --model_type $VARIANT \
            --max_windows 200 --n_members 48 --batch_size 16 \
            --horizons 1,5,10,20,30 \
            --output_dir "results/block_ar/232_eval/${VARIANT}_diagnostic" \
            --device cuda
    fi
done

echo ""
echo "=============================================================="
echo "  Comparison Table (232 architectural attacks)"
echo "=============================================================="
PYTHONPATH=. python -u experiments/backfill/block_ar/compare_232_variants.py
