"""
Co-integration Preservation Testing

Tests whether VAE and econometric model predictions preserve the co-integration
relationship with EWMA realized volatility that exists in ground truth data.

If models preserve co-integration → learned fundamental economic relationships
If models break co-integration → may be overfitting noise

Tests on three periods:
1. In-sample (training): 2004-2019 (indices 1000-5000)
2. Out-of-sample (test): 2019-2023 (indices 5001-5821)
3. Crisis: 2008-2010 (indices 2000-2765)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CO-INTEGRATION PRESERVATION TESTING")
print("=" * 80)
print()

# ============================================================================
# Configuration
# ============================================================================

# Test periods
PERIODS = {
    'in_sample': (1000, 5000),   # Training data
    'out_of_sample': (5001, 5821),  # Test data
    'crisis': (2000, 2765),      # 2008-2010 crisis
}

HORIZONS = [1, 7, 14, 30]
EWMA_WARMUP = 20
ADF_LAGS = 5

# Output directory
output_dir = Path("tables/cointegration_preservation")
output_dir.mkdir(parents=True, exist_ok=True)

print("Test periods:")
for period_name, (start, end) in PERIODS.items():
    print(f"  {period_name}: indices {start}-{end} ({end-start+1} days)")
print()

# ============================================================================
# 1. Load Data
# ============================================================================

print("1. Loading data...")

# Load ground truth and compute EWMA
full_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_full = full_data['surface']
log_returns_full = full_data['ret']

# Compute EWMA
def compute_ewma_volatility(returns, lambda_=0.94, warmup=20):
    n = len(returns)
    variance = np.zeros(n)
    variance[0] = returns[0] ** 2
    for t in range(1, n):
        variance[t] = lambda_ * variance[t-1] + (1 - lambda_) * returns[t]**2
    ewma_vol = np.sqrt(variance * 252)
    valid_indices = np.arange(warmup, n)
    return ewma_vol, valid_indices

ewma_vol_full, valid_indices = compute_ewma_volatility(log_returns_full)

# Load model predictions
# VAE: in-sample
vae_insample = np.load("models_backfill/insample_reconstruction_16yr.npz")

# VAE: out-of-sample
vae_oos = np.load("models_backfill/oos_reconstruction_16yr.npz")

# Econometric: crisis period
econ_crisis = np.load("tables/econometric_backfill/econometric_backfill_2008_2010.npz")

# Econometric: OOS
econ_oos = np.load("tables/econometric_backfill/econometric_backfill_oos.npz")

# VAE Prior: in-sample (z ~ N(0,1) for future)
vae_prior_insample = np.load("models_backfill/vae_prior_insample_16yr.npz")

# VAE Prior: out-of-sample
vae_prior_oos = np.load("models_backfill/vae_prior_oos_16yr.npz")

print("  ✓ Ground truth and EWMA loaded")
print("  ✓ VAE Oracle predictions loaded (in-sample + OOS)")
print("  ✓ VAE Prior predictions loaded (in-sample + OOS)")
print("  ✓ Econometric predictions loaded (crisis + OOS)")
print()

# ============================================================================
# 2. Define Co-integration Test Function
# ============================================================================

def test_cointegration(iv_series, ewma_series, lags=ADF_LAGS):
    """
    Test co-integration between IV and EWMA.

    Returns:
        dict with regression and ADF test results
    """
    # Regression: IV ~ EWMA
    X = add_constant(ewma_series)
    y = iv_series
    model = OLS(y, X).fit()

    residuals = model.resid
    beta = model.params[0]
    alpha1 = model.params[1]
    rsquared = model.rsquared

    # ADF test on residuals
    adf_result = adfuller(residuals, maxlag=lags, regression='c')
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]

    # Breusch-Pagan test for heteroskedasticity
    bp_test = het_breuschpagan(residuals, X)
    bp_pvalue = bp_test[1]

    # Ljung-Box test for autocorrelation
    if len(residuals) > 10:
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[-1]
    else:
        lb_pvalue = np.nan

    return {
        'beta': beta,
        'alpha1': alpha1,
        'rsquared': rsquared,
        'adf_statistic': adf_statistic,
        'adf_pvalue': adf_pvalue,
        'cointegrated': adf_pvalue < 0.05,
        'bp_pvalue': bp_pvalue,
        'lb_pvalue': lb_pvalue,
        'n_obs': len(iv_series),
    }

# ============================================================================
# 3. Test Ground Truth (Baseline)
# ============================================================================

print("2. Testing ground truth co-integration (baseline)...")

ground_truth_results = {}

for period_name, (start_idx, end_idx) in PERIODS.items():
    print(f"  Period: {period_name}")

    # Extract data for this period
    period_mask = (valid_indices >= start_idx) & (valid_indices <= end_idx)
    period_indices = valid_indices[period_mask]

    ewma_period = ewma_vol_full[period_indices]
    surf_period = vol_surf_full[period_indices]

    # Test each grid point
    period_results = np.zeros((5, 5), dtype=object)

    for i in range(5):
        for j in range(5):
            iv_ij = surf_period[:, i, j]
            result = test_cointegration(iv_ij, ewma_period)
            period_results[i, j] = result

    ground_truth_results[period_name] = period_results

    # Summary statistics
    adf_pvalues = np.array([[period_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
    n_cointegrated = np.sum(adf_pvalues < 0.05)

    print(f"    Co-integrated grid points: {n_cointegrated}/25 ({100*n_cointegrated/25:.1f}%)")
    print(f"    Median ADF p-value: {np.median(adf_pvalues):.4f}")

print()

# ============================================================================
# 4. Test VAE Predictions
# ============================================================================

print("3. Testing VAE co-integration preservation...")

vae_results = {}

# Helper function to align data
def align_data(model_indices, model_preds, start_idx, end_idx, ewma_full, surf_full):
    """Align model predictions with EWMA and ground truth for a period."""
    # Find overlapping indices
    period_mask = (model_indices >= start_idx) & (model_indices <= end_idx)
    aligned_indices = model_indices[period_mask]

    if len(aligned_indices) == 0:
        return None, None, None

    # Extract predictions
    preds_aligned = model_preds[period_mask]  # (N, 3, 5, 5) or similar

    # Extract EWMA and ground truth
    ewma_aligned = ewma_full[aligned_indices]
    surf_aligned = surf_full[aligned_indices]

    return preds_aligned, ewma_aligned, surf_aligned

# Loop over all horizons
for horizon in HORIZONS:
    print(f"  Testing horizon {horizon} days...")

    # Test in-sample period
    insample_start, insample_end = PERIODS['in_sample']
    vae_h_insample, ewma_insample, surf_insample = align_data(
        vae_insample[f'indices_h{horizon}'],
        vae_insample[f'recon_h{horizon}'],
        insample_start, insample_end,
        ewma_vol_full, vol_surf_full
    )

    if vae_h_insample is not None:
        vae_h_p50 = vae_h_insample[:, 1, :, :]  # Median predictions

        insample_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                vae_pred_ij = vae_h_p50[:, i, j]
                result = test_cointegration(vae_pred_ij, ewma_insample)
                insample_results[i, j] = result

        vae_results[f'in_sample_h{horizon}'] = insample_results

        adf_pvalues = np.array([[insample_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    VAE H{horizon} In-sample: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

    # Test out-of-sample period
    oos_start, oos_end = PERIODS['out_of_sample']
    vae_h_oos, ewma_oos_period, surf_oos_period = align_data(
        vae_oos[f'indices_h{horizon}'],
        vae_oos[f'recon_h{horizon}'],
        oos_start, oos_end,
        ewma_vol_full, vol_surf_full
    )

    if vae_h_oos is not None:
        vae_h_p50_oos = vae_h_oos[:, 1, :, :]

        oos_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                vae_pred_ij = vae_h_p50_oos[:, i, j]
                result = test_cointegration(vae_pred_ij, ewma_oos_period)
                oos_results[i, j] = result

        vae_results[f'out_of_sample_h{horizon}'] = oos_results

        adf_pvalues = np.array([[oos_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    VAE H{horizon} OOS: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

    # Test crisis period
    crisis_start, crisis_end = PERIODS['crisis']
    vae_h_crisis, ewma_crisis_period, surf_crisis_period = align_data(
        vae_insample[f'indices_h{horizon}'],
        vae_insample[f'recon_h{horizon}'],
        crisis_start, crisis_end,
        ewma_vol_full, vol_surf_full
    )

    if vae_h_crisis is not None:
        vae_h_p50_crisis = vae_h_crisis[:, 1, :, :]

        crisis_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                vae_pred_ij = vae_h_p50_crisis[:, i, j]
                result = test_cointegration(vae_pred_ij, ewma_crisis_period)
                crisis_results[i, j] = result

        vae_results[f'crisis_h{horizon}'] = crisis_results

        adf_pvalues = np.array([[crisis_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    VAE H{horizon} Crisis: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

print()

# ============================================================================
# 4.5. Test VAE Prior Predictions (z ~ N(0,1) for future)
# ============================================================================

print("3.5. Testing VAE PRIOR co-integration preservation...")
print("     (Realistic generation: z ~ N(0,1) for future, no target encoding)")
print()

vae_prior_results = {}

# Loop over all horizons
for horizon in HORIZONS:
    print(f"  Testing horizon {horizon} days...")

    # Test in-sample period
    insample_start, insample_end = PERIODS['in_sample']
    vae_prior_h_insample, ewma_insample, surf_insample = align_data(
        vae_prior_insample[f'indices_h{horizon}'],
        vae_prior_insample[f'recon_h{horizon}'],
        insample_start, insample_end,
        ewma_vol_full, vol_surf_full
    )

    if vae_prior_h_insample is not None:
        vae_prior_h_p50 = vae_prior_h_insample[:, 1, :, :]  # Median predictions

        insample_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                vae_pred_ij = vae_prior_h_p50[:, i, j]
                result = test_cointegration(vae_pred_ij, ewma_insample)
                insample_results[i, j] = result

        vae_prior_results[f'in_sample_h{horizon}'] = insample_results

        adf_pvalues = np.array([[insample_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    VAE Prior H{horizon} In-sample: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

    # Test out-of-sample period
    oos_start, oos_end = PERIODS['out_of_sample']
    vae_prior_h_oos, ewma_oos_period, surf_oos_period = align_data(
        vae_prior_oos[f'indices_h{horizon}'],
        vae_prior_oos[f'recon_h{horizon}'],
        oos_start, oos_end,
        ewma_vol_full, vol_surf_full
    )

    if vae_prior_h_oos is not None:
        vae_prior_h_p50_oos = vae_prior_h_oos[:, 1, :, :]

        oos_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                vae_pred_ij = vae_prior_h_p50_oos[:, i, j]
                result = test_cointegration(vae_pred_ij, ewma_oos_period)
                oos_results[i, j] = result

        vae_prior_results[f'out_of_sample_h{horizon}'] = oos_results

        adf_pvalues = np.array([[oos_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    VAE Prior H{horizon} OOS: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

    # Test crisis period (from in-sample data)
    crisis_start, crisis_end = PERIODS['crisis']
    vae_prior_h_crisis, ewma_crisis_period, surf_crisis_period = align_data(
        vae_prior_insample[f'indices_h{horizon}'],
        vae_prior_insample[f'recon_h{horizon}'],
        crisis_start, crisis_end,
        ewma_vol_full, vol_surf_full
    )

    if vae_prior_h_crisis is not None:
        vae_prior_h_p50_crisis = vae_prior_h_crisis[:, 1, :, :]

        crisis_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                vae_pred_ij = vae_prior_h_p50_crisis[:, i, j]
                result = test_cointegration(vae_pred_ij, ewma_crisis_period)
                crisis_results[i, j] = result

        vae_prior_results[f'crisis_h{horizon}'] = crisis_results

        adf_pvalues = np.array([[crisis_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    VAE Prior H{horizon} Crisis: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

print()

# ============================================================================
# 5. Test Econometric Predictions
# ============================================================================

print("4. Testing econometric co-integration preservation...")

econ_results = {}

# Loop over all horizons
for horizon in HORIZONS:
    print(f"  Testing horizon {horizon} days...")

    # Test crisis period
    crisis_start, crisis_end = PERIODS['crisis']
    econ_h_crisis, ewma_econ_crisis, _ = align_data(
        econ_crisis[f'indices_h{horizon}'],
        econ_crisis[f'recon_h{horizon}'],
        crisis_start, crisis_end,
        ewma_vol_full, vol_surf_full
    )

    if econ_h_crisis is not None:
        econ_h_p50_crisis = econ_h_crisis[:, 1, :, :]

        crisis_econ_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                econ_pred_ij = econ_h_p50_crisis[:, i, j]
                result = test_cointegration(econ_pred_ij, ewma_econ_crisis)
                crisis_econ_results[i, j] = result

        econ_results[f'crisis_h{horizon}'] = crisis_econ_results

        adf_pvalues = np.array([[crisis_econ_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    Econ H{horizon} Crisis: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

    # Test OOS period
    oos_start, oos_end = PERIODS['out_of_sample']
    econ_h_oos, ewma_econ_oos, _ = align_data(
        econ_oos[f'indices_h{horizon}'],
        econ_oos[f'recon_h{horizon}'],
        oos_start, oos_end,
        ewma_vol_full, vol_surf_full
    )

    if econ_h_oos is not None:
        econ_h_p50_oos = econ_h_oos[:, 1, :, :]

        oos_econ_results = np.zeros((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                econ_pred_ij = econ_h_p50_oos[:, i, j]
                result = test_cointegration(econ_pred_ij, ewma_econ_oos)
                oos_econ_results[i, j] = result

        econ_results[f'out_of_sample_h{horizon}'] = oos_econ_results

        adf_pvalues = np.array([[oos_econ_results[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        n_cointegrated = np.sum(adf_pvalues < 0.05)
        print(f"    Econ H{horizon} OOS: {n_cointegrated}/25 co-integrated ({100*n_cointegrated/25:.1f}%)")

print()

# ============================================================================
# 6. Save Results
# ============================================================================

print("5. Saving results...")

# Save as npz (pickled objects)
np.savez(
    output_dir / "ground_truth_results.npz",
    **{period: ground_truth_results[period] for period in PERIODS.keys()}
)

np.savez(
    output_dir / "vae_oracle_results.npz",
    **vae_results
)

np.savez(
    output_dir / "vae_prior_results.npz",
    **vae_prior_results
)

np.savez(
    output_dir / "econometric_results.npz",
    **econ_results
)

# Create summary CSV
summary_rows = []

# Ground truth summary
for period_name in PERIODS.keys():
    if period_name in ground_truth_results:
        results_grid = ground_truth_results[period_name]
        adf_pvalues = np.array([[results_grid[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
        rsquared = np.array([[results_grid[i, j]['rsquared'] for j in range(5)] for i in range(5)])

        summary_rows.append({
            'model': 'Ground Truth',
            'period': period_name,
            'n_cointegrated': np.sum(adf_pvalues < 0.05),
            'pct_cointegrated': 100 * np.sum(adf_pvalues < 0.05) / 25,
            'median_adf_pvalue': np.median(adf_pvalues),
            'mean_rsquared': np.mean(rsquared),
        })

# VAE Oracle summary
for test_name in vae_results.keys():
    results_grid = vae_results[test_name]
    adf_pvalues = np.array([[results_grid[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
    rsquared = np.array([[results_grid[i, j]['rsquared'] for j in range(5)] for i in range(5)])

    summary_rows.append({
        'model': 'VAE_Oracle',
        'period': test_name,
        'n_cointegrated': np.sum(adf_pvalues < 0.05),
        'pct_cointegrated': 100 * np.sum(adf_pvalues < 0.05) / 25,
        'median_adf_pvalue': np.median(adf_pvalues),
        'mean_rsquared': np.mean(rsquared),
    })

# VAE Prior summary
for test_name in vae_prior_results.keys():
    results_grid = vae_prior_results[test_name]
    adf_pvalues = np.array([[results_grid[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
    rsquared = np.array([[results_grid[i, j]['rsquared'] for j in range(5)] for i in range(5)])

    summary_rows.append({
        'model': 'VAE_Prior',
        'period': test_name,
        'n_cointegrated': np.sum(adf_pvalues < 0.05),
        'pct_cointegrated': 100 * np.sum(adf_pvalues < 0.05) / 25,
        'median_adf_pvalue': np.median(adf_pvalues),
        'mean_rsquared': np.mean(rsquared),
    })

# Econometric summary
for test_name in econ_results.keys():
    results_grid = econ_results[test_name]
    adf_pvalues = np.array([[results_grid[i, j]['adf_pvalue'] for j in range(5)] for i in range(5)])
    rsquared = np.array([[results_grid[i, j]['rsquared'] for j in range(5)] for i in range(5)])

    summary_rows.append({
        'model': 'Econometric',
        'period': test_name,
        'n_cointegrated': np.sum(adf_pvalues < 0.05),
        'pct_cointegrated': 100 * np.sum(adf_pvalues < 0.05) / 25,
        'median_adf_pvalue': np.median(adf_pvalues),
        'mean_rsquared': np.mean(rsquared),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_dir / "summary_comparison.csv", index=False)

print(f"  ✓ Saved: {output_dir / 'ground_truth_results.npz'}")
print(f"  ✓ Saved: {output_dir / 'vae_oracle_results.npz'}")
print(f"  ✓ Saved: {output_dir / 'vae_prior_results.npz'}")
print(f"  ✓ Saved: {output_dir / 'econometric_results.npz'}")
print(f"  ✓ Saved: {output_dir / 'summary_comparison.csv'}")
print()

# ============================================================================
# 7. Summary
# ============================================================================

print("=" * 80)
print("CO-INTEGRATION PRESERVATION TEST COMPLETE")
print("=" * 80)
print()

print("Summary:")
print(summary_df.to_string(index=False))
print()

print("Key Findings:")
print("  - Ground truth: Strongly co-integrated with EWMA (as expected)")
print("  - VAE Oracle: Preservation quality when encoding target (upper bound)")
print("  - VAE Prior: Preservation quality with realistic generation (z ~ N(0,1))")
print("  - Econometric: Should preserve by design (linear model)")
print()

print("Files saved:")
print(f"  - {output_dir / 'ground_truth_results.npz'}")
print(f"  - {output_dir / 'vae_oracle_results.npz'}")
print(f"  - {output_dir / 'vae_prior_results.npz'}")
print(f"  - {output_dir / 'econometric_results.npz'}")
print(f"  - {output_dir / 'summary_comparison.csv'}")
print()

print("Next step: Generate visualizations and comprehensive OOS comparison")
print()

print("✓ Done!")
print()
