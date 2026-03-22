#!/usr/bin/env bash
# Ensemble Verification: 140a (AR Causal Transformer) + 99m_v2 (AR MLP Baseline)
#
# Hypothesis: 140a has GT-matching cross-cell correlation (0.389) and high output rank (4-5.6)
# but terrible coverage (65.5%) and low kurtosis (0.364). 99m_v2 has good coverage (91.3%)
# and kurtosis (0.845) but poor correlation (0.433) and low rank (1.57). Ensemble should
# combine complementary strengths.
#
# Configuration:
#   - 140a: 25 samples (AR causal transformer)
#   - 99m_v2: 25 samples (AR MLP baseline)
#   - Total: 50 samples per window
#   - Max batches: 20, device: cuda
#
# Output: results/block_ar/E10_140a_99m_30d/

set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

OUTDIR="results/block_ar/E10_140a_99m_30d"
ANALYSIS_DIR="results/validations/2026-03-22/analysis/ensemble_140a_99m"

echo "============================================================"
echo "ENSEMBLE VERIFICATION: E10 (140a + 99m_v2)"
echo "Started: $(date -Iseconds)"
echo "============================================================"

# Check GPU availability
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo ""

# Step 1: Run the existing 8-suite ensemble test
echo "Step 1: Running 8-suite ensemble evaluation..."
PYTHONPATH=. python experiments/backfill/block_ar/test_ensemble_full.py \
    --models 140a 99m_v2 \
    --samples_per_model 25 25 \
    --max_batches 20 \
    --device cuda \
    --output_dir "$OUTDIR" \
    2>&1 | tee "${ANALYSIS_DIR}/ensemble_8suite.log"

echo ""
echo "Step 2: Running v2 full test suite (includes Suite 9: cross-cell correlation)..."

# Step 2: Run a dedicated cross-cell correlation analysis using v2 test suite functions
# This extracts the ensemble samples and computes cross-cell correlation metrics
PYTHONPATH=. python -c "
import json
import numpy as np
import sys
sys.path.insert(0, '.')

# Load the 8-suite results
with open('${OUTDIR}/summary.json') as f:
    results_8 = json.load(f)

# Print 8-suite summary
print()
print('=' * 60)
print('8-SUITE RESULTS SUMMARY')
print('=' * 60)

suite_names = {
    'surface': 'Surface Validity',
    'coverage': 'CI Coverage',
    'conditionality': 'Conditionality',
    'time_series': 'Time Series',
    'block_ar': 'Block-AR Boundary',
    'cointegration': 'Cointegration',
    'regime_coverage': 'Regime Coverage',
    'distributional': 'Distributional',
}

passes = 0
for key, name in suite_names.items():
    if key in results_8:
        p = results_8[key].get('pass', False)
        passes += int(p)
        print(f'  {name:25s} {\"PASS\" if p else \"FAIL\"}'
              + ('' if p else '  <--- FAIL'))

print(f'  -------')
print(f'  Total: {passes}/8 suites PASS')

# Key metrics comparison
print()
print('=' * 60)
print('KEY METRICS')
print('=' * 60)

if 'coverage' in results_8:
    cov = results_8['coverage']
    print(f'  CI Coverage (overall): {cov.get(\"overall_coverage\", \"?\"):.1%}' if isinstance(cov.get('overall_coverage'), (int,float)) else f'  CI Coverage: {cov.get(\"overall_coverage\", \"?\")}')
    print(f'  Worst cell pass: {cov.get(\"worst_cell_pass\", \"?\")}')

if 'time_series' in results_8:
    ts = results_8['time_series']
    print(f'  Kurtosis ratio: {ts.get(\"kurtosis_ratio\", \"?\")}')
    print(f'  ACF correlation: {ts.get(\"acf_correlation\", \"?\")}')

if 'distributional' in results_8:
    dist = results_8['distributional']
    ks_changes = dist.get('ks_changes_pass_count', '?')
    ks_levels = dist.get('ks_levels_pass_count', '?')
    print(f'  KS changes pass: {ks_changes}/25')
    print(f'  KS levels pass: {ks_levels}/25')

if 'cointegration' in results_8:
    coint = results_8['cointegration']
    print(f'  Cointegration rate: {coint.get(\"coint_pass_rate\", \"?\")}')

print()
print('Analysis complete.')
" 2>&1 | tee "${ANALYSIS_DIR}/results_summary.log"

# Step 3: Now generate ensemble samples for cross-cell correlation (Suite 9 from v2)
echo ""
echo "Step 3: Computing cross-cell correlation (Suite 9 from v2)..."
PYTHONPATH=. python -c "
import numpy as np
import torch
import json
import sys
sys.path.insert(0, '.')

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig
from diffusion.block_ar.block_ar_ddpm import denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.block_ar.config_block_ar import get_default_config
from torch.utils.data import DataLoader

device = 'cuda'
max_batches = 20

# Load models
def load_model(path):
    cp = torch.load(path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in cp['config'].items() if k in SinglePassConfig.__dataclass_fields__}
    model = SinglePassBlockAR(SinglePassConfig(**cfg))
    model.load_state_dict(cp['model_state_dict'], strict=False)
    model.to(device).eval()
    return model

print('Loading models...')
m1 = load_model('models/backfill/afcrps_140a/best_model.pt')
m2 = load_model('models/backfill/afcrps_99m_v2/best_model.pt')

# Load test data
config = get_default_config()
data = np.load(config.data_path)
surfaces = data['surface']
test_dataset = VolSurfaceDataset(surfaces, config.history_len, config.future_len, start_idx=config.test_start)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

# Generate samples
all_cond = []
all_gt = []
for batch_idx, batch in enumerate(test_loader):
    if batch_idx >= max_batches:
        break
    history = batch['history'].to(device)
    extra = batch.get('history_returns')
    if extra is not None:
        extra = extra.to(device)

    with torch.no_grad():
        s1 = m1.sample_batched(history, n_samples=25, extra_hist=extra).cpu().numpy()
        s2 = m2.sample_batched(history, n_samples=25, extra_hist=extra).cpu().numpy()

    combined = np.concatenate([s1, s2], axis=1)  # (B, 50, T, 5, 5)
    all_cond.append(combined)
    gt = denormalize_iv(batch['future'].to(device)).cpu().numpy()
    all_gt.append(gt)
    torch.cuda.empty_cache()

cond_samples = np.concatenate(all_cond, axis=0)
ground_truth = np.concatenate(all_gt, axis=0)
print(f'Ensemble samples: {cond_samples.shape}')
print(f'Ground truth: {ground_truth.shape}')

# Run cross-cell correlation test from v2
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import run_cross_cell_correlation_tests
xcorr = run_cross_cell_correlation_tests(cond_samples, ground_truth)

print()
print('=' * 60)
print('SUITE 9: CROSS-CELL CORRELATION')
print('=' * 60)
print(f'  Correlation ratio: {xcorr.get(\"corr_ratio\", \"?\")}')
print(f'  Effective rank ratio: {xcorr.get(\"eff_rank_ratio\", \"?\")}')
print(f'  Pass: {xcorr.get(\"pass\", \"?\")}')
print()

# Also compute individual model cross-cell stats for comparison
print('Per-model cross-cell correlation (for comparison):')
s1_all = np.concatenate([a[:, :25] for a in all_cond], axis=0)
s2_all = np.concatenate([a[:, 25:] for a in all_cond], axis=0)

for label, samples in [('140a', s1_all), ('99m_v2', s2_all), ('E10 ensemble', cond_samples)]:
    N, K, T, _, _ = samples.shape
    flat = samples.reshape(N, K, T, 25)
    changes = np.diff(flat, axis=2)
    median_changes = np.median(changes, axis=1)  # (N, T-1, 25)
    flat_changes = median_changes.reshape(-1, 25)
    if flat_changes.shape[0] > 10:
        corr_mat = np.corrcoef(flat_changes.T)
        gt_flat = ground_truth.reshape(N, T, 25)
        gt_changes = np.diff(gt_flat, axis=1).reshape(-1, 25)
        gt_corr = np.corrcoef(gt_changes.T)

        # Effective rank
        def eff_rank(mat):
            eigs = np.linalg.eigvalsh(mat)
            eigs = np.maximum(eigs, 0)
            eigs = eigs / eigs.sum()
            eigs = eigs[eigs > 1e-10]
            return float(np.exp(-np.sum(eigs * np.log(eigs))))

        model_er = eff_rank(corr_mat)
        gt_er = eff_rank(gt_corr)

        # Mean off-diagonal correlation
        mask = ~np.eye(25, dtype=bool)
        model_corr = float(np.mean(np.abs(corr_mat[mask])))
        gt_corr_val = float(np.mean(np.abs(gt_corr[mask])))

        print(f'  {label:20s}: corr={model_corr:.3f} (GT {gt_corr_val:.3f}), '
              f'eff_rank={model_er:.2f} (GT {gt_er:.2f}), '
              f'corr_ratio={model_corr/max(gt_corr_val,1e-8):.3f}, '
              f'rank_ratio={model_er/max(gt_er,1e-8):.3f}')

# Save cross-cell results
xcorr_path = '${ANALYSIS_DIR}/cross_cell_correlation.json'
with open(xcorr_path, 'w') as f:
    json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in xcorr.items()}, f, indent=2)
print(f'  Saved to {xcorr_path}')
" 2>&1 | tee "${ANALYSIS_DIR}/cross_cell_correlation.log"

# Copy summary.json for analysis
cp "${OUTDIR}/summary.json" "${ANALYSIS_DIR}/summary.json" 2>/dev/null || true
cp "${OUTDIR}/composite_score.json" "${ANALYSIS_DIR}/composite_score.json" 2>/dev/null || true

echo ""
echo "============================================================"
echo "ENSEMBLE VERIFICATION COMPLETE: $(date -Iseconds)"
echo "Output dir: ${OUTDIR}"
echo "Analysis dir: ${ANALYSIS_DIR}"
echo "============================================================"
