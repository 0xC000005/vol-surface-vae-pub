#!/bin/bash
set -euo pipefail

# ============================================================================
# Multi-seed verification for Exp 138a
# ============================================================================
# WHAT: Runs v2 test suite on 138a best_model with seeds 123 and 456
# WHY:  138a claims 6/8 suites PASS. Test suite has sampling noise (133f
#       varied by 20 points between runs). We need to confirm 6/8 is robust.
# EXPECTED: If robust, both seeds should produce 6/8 PASS (S1,S3,S4,S5,S6,S8)
#           Key borderline: S4 kurtosis (0.794, threshold 0.5-2.0),
#                           S8 KS levels (15/25, threshold 15)
# ============================================================================

cd /home/max/Documents/vol-surface-vae-pub

MODEL_PATH="models/backfill/afcrps_138a/best_model.pt"
TEST_SCRIPT="experiments/backfill/block_ar/test_block_ar_requirements_v2.py"
COMMON_ARGS="--no_ema --max_batches 20 --n_samples 50 --device cuda"

echo "=== Seed 123 ==="
PYTHONPATH=. python "$TEST_SCRIPT" \
    --model_path "$MODEL_PATH" \
    $COMMON_ARGS \
    --seed 123 \
    --output_dir results/block_ar/138a_seed123_30d

echo ""
echo "=== Seed 456 ==="
PYTHONPATH=. python "$TEST_SCRIPT" \
    --model_path "$MODEL_PATH" \
    $COMMON_ARGS \
    --seed 456 \
    --output_dir results/block_ar/138a_seed456_30d

echo ""
echo "=== Comparison ==="
python3 -c "
import json, sys

seeds = {'seed42': 'results/block_ar/138a_best_30d/summary.json',
         'seed123': 'results/block_ar/138a_seed123_30d/summary.json',
         'seed456': 'results/block_ar/138a_seed456_30d/summary.json'}

suite_names = ['surface', 'coverage', 'conditionality', 'time_series',
               'block_ar', 'cointegration', 'regime_coverage', 'distributional']
suite_labels = ['S1:Surface', 'S2:Coverage', 'S3:Condition', 'S4:TimeSeries',
                'S5:BlockAR', 'S6:Coint', 'S7:Regime', 'S8:Distrib']

results = {}
for label, path in seeds.items():
    try:
        with open(path) as f:
            d = json.load(f)
        passes = []
        for s in suite_names:
            if s in d:
                p = d[s].get('overall_pass', d[s].get('pass', False))
                passes.append(p)
            else:
                passes.append(False)
        results[label] = {'passes': passes, 'count': sum(passes)}
    except FileNotFoundError:
        print(f'WARNING: {path} not found (seed42 is existing result)')
        results[label] = {'passes': [None]*8, 'count': -1}

print(f\"{'Suite':<15} {'seed42':<10} {'seed123':<10} {'seed456':<10}\")
print('-' * 45)
for i, sl in enumerate(suite_labels):
    row = f'{sl:<15}'
    for label in ['seed42', 'seed123', 'seed456']:
        v = results[label]['passes'][i]
        row += f' {\"PASS\" if v else \"FAIL\":<10}' if v is not None else f' {\"N/A\":<10}'
    print(row)
print('-' * 45)
for label in ['seed42', 'seed123', 'seed456']:
    c = results[label]['count']
    print(f'{label}: {c}/8 PASS' if c >= 0 else f'{label}: N/A')

# Save comparison JSON
comparison = {}
for label in ['seed42', 'seed123', 'seed456']:
    comparison[label] = {
        'suite_pass': dict(zip(suite_names, results[label]['passes'])),
        'total_pass': results[label]['count']
    }
comparison['robust'] = all(r['count'] >= 6 for r in results.values() if r['count'] >= 0)

import os
os.makedirs('results/validations/2026-03-21/analysis/138a_multiseed', exist_ok=True)
with open('results/validations/2026-03-21/analysis/138a_multiseed/seed_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print(f'\nComparison saved to results/validations/2026-03-21/analysis/138a_multiseed/seed_comparison.json')
print(f'Robust across seeds: {comparison[\"robust\"]}')
"

echo "=== Done ==="
