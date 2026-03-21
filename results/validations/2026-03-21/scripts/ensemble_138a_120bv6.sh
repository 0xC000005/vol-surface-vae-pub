#!/bin/bash
# Ensemble evaluation: 138a best_model + 120b_v6 best_coverage_model
# 50/50 ensemble (25 samples each) on v2 test suites
set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

echo "=== Ensemble Verification: 138a + 120b_v6 ==="
echo "Start: $(date)"

# Verify model files exist
for f in models/backfill/afcrps_138a/best_model.pt \
         models/backfill/afcrps_120b_v6/best_coverage_model.pt; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Model not found: $f"
        exit 1
    fi
    echo "Found: $f ($(stat -c%s "$f") bytes)"
done

# Run ensemble evaluation
PYTHONPATH=. python results/validations/2026-03-21/scripts/ensemble_138a_120bv6.py \
    2>&1 | tee results/block_ar/ensemble_138a_120bv6_30d/eval.log

echo ""
echo "End: $(date)"
echo "Results: results/block_ar/ensemble_138a_120bv6_30d/summary.json"
