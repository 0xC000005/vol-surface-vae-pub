"""
In-Sequence Co-Integration Testing Utilities

Core functions for testing IV-EWMA co-integration within 30-day bootstrap
autoregressive sequences. Implements dual-level testing (individual + aggregate)
with statistical adjustments for short time series.

Key adjustments for 30-day sequences:
- ADF lags: 3 instead of 5 (preserve degrees of freedom)
- Significance: α=0.10 instead of 0.05 (acknowledge reduced power)
- Aggregate pooling: 23K observations for robust inference

Usage:
    from insequence_cointegration_utils import (
        compute_ewma_for_sequence,
        test_sequence_cointegration,
        test_aggregate_pooled
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Configuration
ADF_LAGS = 3  # Reduced from 5 for short sequences
ADF_ALPHA = 0.10  # More conservative than 0.05
EWMA_LAMBDA = 0.94  # RiskMetrics standard
EWMA_WARMUP = 20  # Days to skip for initialization


def compute_ewma_for_sequence(returns, start_idx, n_days=30, lambda_=EWMA_LAMBDA):
    """
    Compute EWMA volatility for a specific 30-day sequence.

    Uses recursive formula: σ²(t) = λ·σ²(t-1) + (1-λ)·r²(t)

    Args:
        returns: (N,) full returns array
        start_idx: int, starting index in returns array
        n_days: int, sequence length (default: 30)
        lambda_: float, decay parameter (default: 0.94)

    Returns:
        ewma_vol: (n_days,) annualized EWMA volatility
    """
    # Extract returns window
    ret_window = returns[start_idx:start_idx+n_days]

    # Initialize variance
    # If we have warmup data, use previous EWMA
    # Otherwise, use first return squared
    if start_idx >= EWMA_WARMUP:
        # Initialize from previous day's squared return (approximate)
        init_var = returns[start_idx-1]**2
    else:
        init_var = ret_window[0]**2

    # Recursive EWMA computation
    variance = np.zeros(n_days)
    variance[0] = init_var

    for t in range(1, n_days):
        variance[t] = lambda_ * variance[t-1] + (1 - lambda_) * ret_window[t]**2

    # Annualize: σ_daily × √252
    ewma_vol = np.sqrt(variance * 252)

    return ewma_vol


def test_sequence_cointegration(iv_series, ewma_series, grid_idx, lags=ADF_LAGS, alpha=ADF_ALPHA):
    """
    Test IV-EWMA co-integration for one 30-day sequence.

    Steps:
    1. Regression: IV(t) = β + α₁·EWMA(t) + u(t)
    2. Extract residuals u(t)
    3. ADF test on residuals (test for stationarity)
    4. p < alpha → Reject unit root → Co-integrated

    Args:
        iv_series: (30,) implied volatility time series
        ewma_series: (30,) EWMA volatility time series
        grid_idx: tuple (i, j) grid point index
        lags: int, ADF lags (default: 3 for short sequences)
        alpha: float, significance level (default: 0.10)

    Returns:
        dict with:
            - cointegrated: bool, pass if p < alpha
            - adf_pvalue: float
            - adf_statistic: float
            - beta: float, intercept
            - alpha1: float, EWMA coefficient
            - rsquared: float, regression R²
            - grid_idx: tuple, grid point
    """
    # Regression: IV ~ β + α₁·EWMA
    X = add_constant(ewma_series)
    y = iv_series
    model = OLS(y, X).fit()

    residuals = model.resid
    beta = model.params[0]
    alpha1 = model.params[1]
    rsquared = model.rsquared

    # ADF test on residuals
    # Null hypothesis: unit root (non-stationary, NO co-integration)
    # Alternative: stationary (co-integrated)
    try:
        adf_result = adfuller(residuals, maxlag=lags, regression='c')
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_critical = adf_result[4]
    except Exception as e:
        # If ADF fails (rare), mark as non-cointegrated
        adf_statistic = np.nan
        adf_pvalue = 1.0  # Fail to reject null
        adf_critical = {}

    # Decision: p < alpha → Reject null → Co-integrated
    cointegrated = adf_pvalue < alpha

    return {
        'cointegrated': cointegrated,
        'adf_pvalue': adf_pvalue,
        'adf_statistic': adf_statistic,
        'adf_critical_5pct': adf_critical.get('5%', np.nan),
        'beta': beta,
        'alpha1': alpha1,
        'rsquared': rsquared,
        'grid_idx': grid_idx,
    }


def test_aggregate_pooled(ar_surfaces, returns, crisis_start, crisis_end,
                          lags=ADF_LAGS, alpha=ADF_ALPHA):
    """
    Test IV-EWMA co-integration on pooled data (all sequences combined).

    Pools 737 valid sequences × 30 days = 22,110 observations for robust inference.

    Args:
        ar_surfaces: (766, 30, 3, 5, 5) bootstrap AR predictions
        returns: (N,) full returns array
        crisis_start: int, start index (e.g., 2000)
        crisis_end: int, end index (e.g., 2765)
        lags: int, ADF lags (default: 3)
        alpha: float, significance level (default: 0.10)

    Returns:
        dict with per-grid-point aggregate results:
            - adf_pvalues: (5, 5) p-values
            - cointegrated: (5, 5) bool pass/fail
            - beta: (5, 5) intercepts
            - alpha1: (5, 5) EWMA coefficients
            - rsquared: (5, 5) R² values
    """
    n_seq, n_days, n_quantiles, n_rows, n_cols = ar_surfaces.shape

    # Only use sequences with full 30-day data
    n_valid_seq = crisis_end - crisis_start + 1 - n_days + 1  # 766 - 30 + 1 = 737
    print(f"  Using {n_valid_seq} valid sequences (out of {n_seq})")

    # Extract p50 median
    p50 = ar_surfaces[:, :, 1, :, :]  # (766, 30, 5, 5)

    # Initialize results
    adf_pvalues = np.zeros((n_rows, n_cols))
    cointegrated = np.zeros((n_rows, n_cols), dtype=bool)
    betas = np.zeros((n_rows, n_cols))
    alpha1s = np.zeros((n_rows, n_cols))
    rsquareds = np.zeros((n_rows, n_cols))

    # Compute EWMA for entire crisis period once
    crisis_returns = returns[crisis_start:crisis_end+1]

    # Test each grid point
    for i in range(n_rows):
        for j in range(n_cols):
            print(f"  Testing aggregate for grid ({i}, {j})...")

            # Pool all IV values across valid sequences
            iv_pooled = []
            ewma_pooled = []

            for seq_idx in range(n_valid_seq):
                # IV for this sequence and grid point
                iv_seq = p50[seq_idx, :, i, j]  # (30,)

                # EWMA for this sequence
                start_idx = seq_idx  # Sequence starts at crisis_start + seq_idx
                ewma_seq = compute_ewma_for_sequence(
                    crisis_returns, start_idx, n_days=30
                )

                iv_pooled.append(iv_seq)
                ewma_pooled.append(ewma_seq)

            # Concatenate to single long series
            iv_pooled = np.concatenate(iv_pooled)  # (22980,)
            ewma_pooled = np.concatenate(ewma_pooled)  # (22980,)

            # Regression + ADF test on pooled data
            X = add_constant(ewma_pooled)
            y = iv_pooled
            model = OLS(y, X).fit()

            residuals = model.resid
            betas[i, j] = model.params[0]
            alpha1s[i, j] = model.params[1]
            rsquareds[i, j] = model.rsquared

            # ADF test
            try:
                adf_result = adfuller(residuals, maxlag=lags, regression='c')
                adf_pvalues[i, j] = adf_result[1]
                cointegrated[i, j] = adf_result[1] < alpha
            except Exception as e:
                adf_pvalues[i, j] = 1.0
                cointegrated[i, j] = False

    return {
        'adf_pvalues': adf_pvalues,
        'cointegrated': cointegrated,
        'beta': betas,
        'alpha1': alpha1s,
        'rsquared': rsquareds,
        'n_observations': len(iv_pooled),
    }


def plot_pass_rate_heatmap(pass_rates_grid, output_path, title="In-Sequence Co-Integration Pass Rates"):
    """
    Visualize (5,5) pass rate heatmap.

    Args:
        pass_rates_grid: (5, 5) pass rates (0-1 or 0-100%)
        output_path: str or Path, where to save figure
        title: str, plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert to percentage if needed
    if pass_rates_grid.max() <= 1.0:
        pass_rates_pct = pass_rates_grid * 100
    else:
        pass_rates_pct = pass_rates_grid

    # Plot heatmap
    im = ax.imshow(pass_rates_pct, cmap='RdYlGn', vmin=50, vmax=100,
                   aspect='auto', origin='upper')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pass Rate (%)', fontsize=12)

    # Annotate cells with values
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{pass_rates_pct[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=10)

    # Labels
    moneyness_labels = ['0.7\n(OTM Put)', '0.85', '1.0\n(ATM)', '1.15', '1.3\n(OTM Call)']
    maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(moneyness_labels, fontsize=10)
    ax.set_yticklabels(maturity_labels, fontsize=10)

    ax.set_xlabel('Moneyness', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maturity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_pvalue_distribution(pvalues, output_path, title="ADF p-value Distribution"):
    """
    Plot histogram of ADF p-values across all sequences/grids.

    Args:
        pvalues: (N,) array of p-values
        output_path: str or Path
        title: str
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(pvalues, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    # Reference lines
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2,
               label='α=0.05 (standard)')
    ax.axvline(x=0.10, color='orange', linestyle='--', linewidth=2,
               label='α=0.10 (our threshold)')

    ax.set_xlabel('ADF p-value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def compile_summary_statistics(per_seq_results, aggregate_results):
    """
    Compile summary statistics comparing per-sequence and aggregate results.

    Args:
        per_seq_results: dict with per-sequence metrics
        aggregate_results: dict with aggregate metrics

    Returns:
        DataFrame with summary statistics
    """
    summary_data = []

    # Per-grid-point statistics
    for i in range(5):
        for j in range(5):
            # Per-sequence (individual)
            per_seq_pass_rate = per_seq_results['pass_rates'][i, j]
            per_seq_median_pval = np.median(per_seq_results['pvalues'][:, i, j])
            per_seq_mean_rsq = np.mean(per_seq_results['rsquared'][:, i, j])

            # Aggregate (pooled)
            agg_pval = aggregate_results['adf_pvalues'][i, j]
            agg_pass = aggregate_results['cointegrated'][i, j]
            agg_rsq = aggregate_results['rsquared'][i, j]
            agg_alpha1 = aggregate_results['alpha1'][i, j]

            summary_data.append({
                'grid_i': i,
                'grid_j': j,
                'moneyness': ['0.7', '0.85', '1.0', '1.15', '1.3'][j],
                'maturity': ['1M', '3M', '6M', '1Y', '2Y'][i],
                'per_seq_pass_rate': per_seq_pass_rate,
                'per_seq_median_pval': per_seq_median_pval,
                'per_seq_mean_rsq': per_seq_mean_rsq,
                'aggregate_pval': agg_pval,
                'aggregate_pass': agg_pass,
                'aggregate_rsq': agg_rsq,
                'aggregate_alpha1': agg_alpha1,
            })

    df = pd.DataFrame(summary_data)
    return df


def compute_baseline_comparison(bootstrap_pass_rates, ground_truth_rate=0.84):
    """
    Compare bootstrap in-sequence results to baseline.

    Args:
        bootstrap_pass_rates: (5, 5) bootstrap pass rates
        ground_truth_rate: float, ground truth baseline (default: 0.84 for crisis)

    Returns:
        dict with comparison metrics
    """
    bootstrap_mean = bootstrap_pass_rates.mean()
    bootstrap_std = bootstrap_pass_rates.std()
    bootstrap_min = bootstrap_pass_rates.min()
    bootstrap_max = bootstrap_pass_rates.max()

    difference = bootstrap_mean - ground_truth_rate
    difference_pct = (difference / ground_truth_rate) * 100

    return {
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_std': bootstrap_std,
        'bootstrap_min': bootstrap_min,
        'bootstrap_max': bootstrap_max,
        'ground_truth': ground_truth_rate,
        'difference': difference,
        'difference_pct': difference_pct,
    }
