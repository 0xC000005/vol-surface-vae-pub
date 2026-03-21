#!/bin/bash
# Evaluate 138a_v2 best_coverage_model.pt (epoch 3)
# 138a_v2 uses per-horizon IS at lambda=1.0 (25x stronger than 138a)
# best_model (ep30) scored 4/8, CI 71.0%
# best_coverage_model saved at epoch 3 — likely higher CI, unknown suite count
#
# Date: 2026-03-21

set -euo pipefail
cd /home/max/Documents/vol-surface-vae-pub

MODEL_PATH="models/backfill/afcrps_138a_v2/best_coverage_model.pt"
OUTPUT_DIR="results/block_ar/138a_v2_bestcov_30d"

echo "=== 138a_v2 best_coverage_model evaluation ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Timestamp: $(date -Iseconds)"
echo ""

# Step 1: Run v2 test suite
echo "--- Step 1: Running v2 test suite ---"
PYTHONPATH=. python experiments/backfill/block_ar/test_block_ar_requirements_v2.py \
    --model_path "$MODEL_PATH" \
    --no_ema \
    --max_batches 20 \
    --n_samples 50 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "--- Step 2: Computing composite score ---"
PYTHONPATH=. python autoresearch-session/compute_score.py "$OUTPUT_DIR/summary.json"

echo ""
echo "--- Step 3: Summary ---"
python3 -c "
import json
with open('$OUTPUT_DIR/summary.json') as f:
    s = json.load(f)
suites = s.get('suites', s.get('test_suites', {}))
passed = sum(1 for v in suites.values() if v.get('passed', v.get('pass', False)))
total = len(suites)
print(f'Result: {passed}/{total} suites PASS')
for name, v in suites.items():
    status = 'PASS' if v.get('passed', v.get('pass', False)) else 'FAIL'
    print(f'  {name}: {status}')
"

echo ""
echo "=== Evaluation complete ==="
