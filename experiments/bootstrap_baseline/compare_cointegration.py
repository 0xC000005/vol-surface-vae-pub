"""
Co-Integration Comparison: Bootstrap vs VAE vs Econometric

Tests whether each method preserves the co-integration relationships
observed in ground truth volatility surfaces. Co-integration preservation
is critical for ensuring arbitrage-free and economically consistent forecasts.

Methodology:
- Johansen co-integration test on 25 grid points (5×5 surface)
- Compare co-integration rank and trace statistics
- Analyze correlation structure preservation

Usage:
    python experiments/bootstrap_baseline/compare_cointegration.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_ground_truth(period):
    """Load ground truth volatility surfaces."""
    data = np.load("data/vol_surface_with_ret.npz")
    surface = data['surface']  # (N, 5, 5)

    if period == 'insample':
        gt = surface[:3900]
    else:  # oos
        gt = surface[5000:5792]

    return gt

def load_bootstrap(period, horizon):
    """Load bootstrap predictions."""
    file_path = f"results/bootstrap_baseline/predictions/{period}/bootstrap_predictions_H{horizon}.npz"
    data = np.load(file_path)
    surfaces = data['surfaces']  # (n_days, 3, 5, 5)
    p50 = surfaces[:, 1, :, :]  # Use median
    return p50

def load_vae_oracle(period, horizon):
    """Load VAE Oracle predictions."""
    if period == 'insample':
        file_path = "results/backfill_16yr/predictions/insample_reconstruction_16yr.npz"
    else:
        file_path = "results/backfill_16yr/predictions/oos_reconstruction_16yr.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']
    p50 = surfaces[:, 1, :, :]
    return p50

def load_vae_prior(period, horizon):
    """Load VAE Prior predictions."""
    if period == 'insample':
        file_path = "results/backfill_16yr/predictions/vae_prior_insample_16yr.npz"
    else:
        file_path = "results/backfill_16yr/predictions/vae_prior_oos_16yr.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']
    p50 = surfaces[:, 1, :, :]
    return p50

def load_econometric(period, horizon):
    """Load econometric baseline predictions."""
    if period == 'insample':
        file_path = "results/econometric_baseline/predictions/econometric_backfill_insample.npz"
    else:
        file_path = "results/econometric_baseline/predictions/econometric_backfill_oos.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']
    p50 = surfaces[:, 1, :, :]
    return p50

def align_data_lengths(*arrays):
    """Align multiple arrays to minimum common length."""
    min_len = min(len(arr) for arr in arrays)
    aligned = [arr[:min_len] for arr in arrays]
    return aligned

def reshape_surface_to_matrix(surface):
    """
    Reshape (n_days, 5, 5) surface to (n_days, 25) matrix.
    Each column is a time series for one grid point.
    """
    n_days = surface.shape[0]
    return surface.reshape(n_days, 25)

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Apply Johansen co-integration test.

    Args:
        data: (n_days, n_series) matrix of time series
        det_order: 0=no deterministic, 1=constant, 2=linear trend
        k_ar_diff: Number of lagged differences in the VAR

    Returns:
        dict with test results
    """
    try:
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        # Extract results
        trace_stats = result.lr1  # Trace statistics
        max_eigen_stats = result.lr2  # Max eigenvalue statistics
        crit_values_trace = result.cvt  # Critical values for trace (90%, 95%, 99%)
        crit_values_max = result.cvm  # Critical values for max eigenvalue

        # Determine co-integration rank at 5% significance
        # Compare trace statistic to 95% critical value (index 1)
        rank = 0
        for i in range(len(trace_stats)):
            if trace_stats[i] > crit_values_trace[i, 1]:  # 95% critical value
                rank = i + 1
            else:
                break

        return {
            'success': True,
            'rank': rank,
            'trace_stats': trace_stats,
            'max_eigen_stats': max_eigen_stats,
            'crit_values_trace_95': crit_values_trace[:, 1],
            'crit_values_max_95': crit_values_max[:, 1],
        }
    except Exception as e:
        print(f"  Warning: Johansen test failed - {str(e)}")
        return {
            'success': False,
            'rank': np.nan,
            'trace_stats': np.nan,
            'max_eigen_stats': np.nan,
            'crit_values_trace_95': np.nan,
            'crit_values_max_95': np.nan,
        }

def compute_correlation_matrix(surface):
    """
    Compute correlation matrix across all grid points.

    Args:
        surface: (n_days, 5, 5) volatility surface

    Returns:
        (25, 25) correlation matrix
    """
    data = reshape_surface_to_matrix(surface)
    corr_matrix = np.corrcoef(data.T)  # Transpose to get correlation across series
    return corr_matrix

def compare_correlation_structures(gt_corr, pred_corr):
    """
    Compare correlation matrices between ground truth and predictions.

    Returns:
        dict with comparison metrics
    """
    # Flatten upper triangle (exclude diagonal)
    n = gt_corr.shape[0]
    triu_indices = np.triu_indices(n, k=1)

    gt_flat = gt_corr[triu_indices]
    pred_flat = pred_corr[triu_indices]

    # Compute metrics
    correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]
    rmse = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))
    mae = np.mean(np.abs(gt_flat - pred_flat))

    return {
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
    }

def run_cointegration_analysis(periods=['insample', 'oos'], horizons=[1, 7, 14, 30]):
    """
    Run comprehensive co-integration analysis across all configurations.

    Returns:
        DataFrame with co-integration test results
    """
    all_results = []

    for period in periods:
        print(f"\n{'='*80}")
        print(f"Co-Integration Analysis: {period.upper()} period")
        print('='*80)

        gt = load_ground_truth(period)

        for horizon in horizons:
            print(f"\nHorizon H={horizon}")

            # Load predictions
            bootstrap = load_bootstrap(period, horizon)
            vae_oracle = load_vae_oracle(period, horizon)
            vae_prior = load_vae_prior(period, horizon)
            econ = load_econometric(period, horizon)

            # Align lengths
            gt_aligned, bootstrap, vae_oracle, vae_prior, econ = align_data_lengths(
                gt, bootstrap, vae_oracle, vae_prior, econ
            )

            n_days = len(gt_aligned)
            print(f"  Aligned data length: {n_days} days")

            # Reshape to (n_days, 25) matrices
            gt_matrix = reshape_surface_to_matrix(gt_aligned)
            bootstrap_matrix = reshape_surface_to_matrix(bootstrap)
            vae_oracle_matrix = reshape_surface_to_matrix(vae_oracle)
            vae_prior_matrix = reshape_surface_to_matrix(vae_prior)
            econ_matrix = reshape_surface_to_matrix(econ)

            # Apply Johansen test
            print("  Running Johansen co-integration tests...")
            gt_johansen = johansen_test(gt_matrix)
            bootstrap_johansen = johansen_test(bootstrap_matrix)
            vae_oracle_johansen = johansen_test(vae_oracle_matrix)
            vae_prior_johansen = johansen_test(vae_prior_matrix)
            econ_johansen = johansen_test(econ_matrix)

            # Compute correlation matrices
            print("  Computing correlation structures...")
            gt_corr = compute_correlation_matrix(gt_aligned)
            bootstrap_corr = compute_correlation_matrix(bootstrap)
            vae_oracle_corr = compute_correlation_matrix(vae_oracle)
            vae_prior_corr = compute_correlation_matrix(vae_prior)
            econ_corr = compute_correlation_matrix(econ)

            # Compare correlation structures
            bootstrap_corr_comp = compare_correlation_structures(gt_corr, bootstrap_corr)
            vae_oracle_corr_comp = compare_correlation_structures(gt_corr, vae_oracle_corr)
            vae_prior_corr_comp = compare_correlation_structures(gt_corr, vae_prior_corr)
            econ_corr_comp = compare_correlation_structures(gt_corr, econ_corr)

            # Store results
            result = {
                'period': period,
                'horizon': horizon,
                'gt_rank': gt_johansen['rank'],
                'bootstrap_rank': bootstrap_johansen['rank'],
                'vae_oracle_rank': vae_oracle_johansen['rank'],
                'vae_prior_rank': vae_prior_johansen['rank'],
                'econ_rank': econ_johansen['rank'],
                'bootstrap_corr_similarity': bootstrap_corr_comp['correlation'],
                'vae_oracle_corr_similarity': vae_oracle_corr_comp['correlation'],
                'vae_prior_corr_similarity': vae_prior_corr_comp['correlation'],
                'econ_corr_similarity': econ_corr_comp['correlation'],
                'bootstrap_corr_rmse': bootstrap_corr_comp['rmse'],
                'vae_oracle_corr_rmse': vae_oracle_corr_comp['rmse'],
                'vae_prior_corr_rmse': vae_prior_corr_comp['rmse'],
                'econ_corr_rmse': econ_corr_comp['rmse'],
            }

            all_results.append(result)

            # Print summary
            print(f"\n  Co-integration Ranks (at 5% significance):")
            print(f"    Ground Truth:  {gt_johansen['rank']}")
            print(f"    Bootstrap:     {bootstrap_johansen['rank']}")
            print(f"    VAE Oracle:    {vae_oracle_johansen['rank']}")
            print(f"    VAE Prior:     {vae_prior_johansen['rank']}")
            print(f"    Econometric:   {econ_johansen['rank']}")

            print(f"\n  Correlation Structure Similarity (correlation with GT):")
            print(f"    Bootstrap:     {bootstrap_corr_comp['correlation']:.4f}")
            print(f"    VAE Oracle:    {vae_oracle_corr_comp['correlation']:.4f}")
            print(f"    VAE Prior:     {vae_prior_corr_comp['correlation']:.4f}")
            print(f"    Econometric:   {econ_corr_comp['correlation']:.4f}")

    return pd.DataFrame(all_results)

def create_cointegration_rank_plot(df, output_dir):
    """
    Plot co-integration rank comparison across methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    periods = ['insample', 'oos']
    period_names = ['In-Sample (2004-2019)', 'Out-of-Sample (2019-2023)']

    for idx, (period, period_name) in enumerate(zip(periods, period_names)):
        ax = axes[idx]
        subset = df[df['period'] == period]

        horizons = subset['horizon'].values
        x = np.arange(len(horizons))
        width = 0.15

        # Plot bars for each method
        ax.bar(x - 2*width, subset['gt_rank'], width, label='Ground Truth', color='black', alpha=0.7)
        ax.bar(x - width, subset['bootstrap_rank'], width, label='Bootstrap', color='blue', alpha=0.7)
        ax.bar(x, subset['vae_oracle_rank'], width, label='VAE Oracle', color='orange', alpha=0.7)
        ax.bar(x + width, subset['vae_prior_rank'], width, label='VAE Prior', color='red', alpha=0.7)
        ax.bar(x + 2*width, subset['econ_rank'], width, label='Econometric', color='green', alpha=0.7)

        ax.set_xlabel('Forecast Horizon (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Co-integration Rank', fontsize=12, fontweight='bold')
        ax.set_title(f'{period_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'H{h}' for h in horizons])
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Co-Integration Rank Comparison (Johansen Test, 5% Significance)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'cointegration_rank_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_correlation_similarity_plot(df, output_dir):
    """
    Plot correlation structure similarity across methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    periods = ['insample', 'oos']
    period_names = ['In-Sample (2004-2019)', 'Out-of-Sample (2019-2023)']

    methods = ['bootstrap', 'vae_oracle', 'vae_prior', 'econ']
    method_names = ['Bootstrap', 'VAE Oracle', 'VAE Prior', 'Econometric']
    colors = ['blue', 'orange', 'red', 'green']

    for idx, (period, period_name) in enumerate(zip(periods, period_names)):
        ax = axes[idx]
        subset = df[df['period'] == period]

        horizons = subset['horizon'].values

        for method, name, color in zip(methods, method_names, colors):
            similarity = subset[f'{method}_corr_similarity'].values
            ax.plot(horizons, similarity, marker='o', markersize=10,
                   linewidth=2.5, label=name, color=color)

        ax.set_xlabel('Forecast Horizon (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Correlation Similarity with GT', fontsize=12, fontweight='bold')
        ax.set_title(f'{period_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks(horizons)
        ax.set_ylim(0.5, 1.0)

    plt.suptitle('Correlation Structure Preservation (Correlation with Ground Truth)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'correlation_similarity_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_summary_table(df, output_dir):
    """
    Create summary table of co-integration results.
    """
    # Save full results
    output_file = output_dir / 'cointegration_comprehensive.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved comprehensive results: {output_file}")

    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY: Co-Integration Rank (Johansen Test, 5% Significance)")
    print("="*80)
    print(df[['period', 'horizon', 'gt_rank', 'bootstrap_rank',
             'vae_oracle_rank', 'vae_prior_rank', 'econ_rank']].to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY: Correlation Structure Similarity (Correlation with GT)")
    print("="*80)
    print(df[['period', 'horizon', 'bootstrap_corr_similarity',
             'vae_oracle_corr_similarity', 'vae_prior_corr_similarity',
             'econ_corr_similarity']].to_string(index=False))

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--periods', nargs='+', default=['insample', 'oos'])
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 7, 14, 30])
    args = parser.parse_args()

    print("="*80)
    print("Co-Integration Comparison: Bootstrap vs VAE vs Econometric")
    print("="*80)

    # Run analysis
    print("\nRunning co-integration analysis...")
    df = run_cointegration_analysis(periods=args.periods, horizons=args.horizons)

    # Save results
    output_dir = Path("results/bootstrap_baseline/comparisons/cointegration")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    create_cointegration_rank_plot(df, output_dir)
    create_correlation_similarity_plot(df, output_dir)

    # Summary table
    summary_df = create_summary_table(df, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nKey Insights:")
    print("  - Co-integration rank: Higher = more relationships preserved")
    print("  - Correlation similarity: Closer to 1.0 = better structure preservation")
    print("  - Econometric typically shows 100% rank preservation (forced by linear structure)")
    print("  - VAE shows adaptive preservation (weakens during stress)")
    print("  - Bootstrap preservation depends on sampling diversity")

if __name__ == "__main__":
    main()
