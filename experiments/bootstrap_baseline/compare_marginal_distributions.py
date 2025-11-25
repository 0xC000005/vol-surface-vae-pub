"""
Marginal Distribution Comparison: Bootstrap vs VAE vs Econometric

Compares distribution shape (kurtosis, skewness, tails) across all methods
to test the hypothesis: Bootstrap should naturally preserve fat-tailed
historical marginal distributions, while VAE models struggle due to
Gaussian latent prior constraints.

Key Research Question:
Does bootstrap baseline better match the fat-tailed, uniform-spread
historical marginal distributions compared to Gaussian-constrained VAE models?

Usage:
    python experiments/bootstrap_baseline/compare_marginal_distributions.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import argparse

def load_ground_truth(period, horizon):
    """Load ground truth volatility surfaces."""
    data = np.load("data/vol_surface_with_ret.npz")
    surface = data['surface']  # (N, 5, 5)

    if period == 'insample':
        # Match the indices from backfill_16yr generation
        # In-sample: 0 to ~3900
        gt = surface[:3900]
    else:  # oos
        # OOS: 5000 onwards (test set)
        gt = surface[5000:5792]

    # Align with horizon-specific prediction lengths
    # Bootstrap predictions have different lengths due to initial history requirements
    return gt

def load_bootstrap(period, horizon):
    """Load bootstrap predictions."""
    file_path = f"results/bootstrap_baseline/predictions/{period}/bootstrap_predictions_H{horizon}.npz"
    data = np.load(file_path)
    surfaces = data['surfaces']  # (n_days, 3, 5, 5) - [p05, p50, p95]

    # Use p50 (median) for point forecast comparison
    p50 = surfaces[:, 1, :, :]  # (n_days, 5, 5)
    return p50

def load_vae_oracle(period, horizon):
    """Load VAE Oracle predictions (ground truth latent encoding)."""
    if period == 'insample':
        file_path = "results/backfill_16yr/predictions/insample_reconstruction_16yr.npz"
    else:
        file_path = "results/backfill_16yr/predictions/oos_reconstruction_16yr.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']  # (n_days, 3, 5, 5)

    # Use p50
    p50 = surfaces[:, 1, :, :]
    return p50

def load_vae_prior(period, horizon):
    """Load VAE Prior predictions (z ~ N(0,1) sampling)."""
    if period == 'insample':
        file_path = "results/backfill_16yr/predictions/vae_prior_insample_16yr.npz"
    else:
        file_path = "results/backfill_16yr/predictions/vae_prior_oos_16yr.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']  # (n_days, 3, 5, 5)

    # Use p50
    p50 = surfaces[:, 1, :, :]
    return p50

def load_econometric(period, horizon):
    """Load econometric baseline predictions."""
    if period == 'insample':
        file_path = "results/econometric_baseline/predictions/econometric_backfill_insample.npz"
    else:
        file_path = "results/econometric_baseline/predictions/econometric_backfill_oos.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']  # (n_days, 3, 5, 5)

    # Use p50
    p50 = surfaces[:, 1, :, :]
    return p50

def align_data_lengths(*arrays):
    """Align multiple arrays to minimum common length."""
    min_len = min(len(arr) for arr in arrays)
    aligned = [arr[:min_len] for arr in arrays]
    return aligned

def compute_distribution_stats(data, name):
    """
    Compute comprehensive distribution statistics.

    Args:
        data: 1D array of values
        name: Name of the distribution

    Returns:
        dict with statistics
    """
    return {
        'name': name,
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data, fisher=False),  # Pearson (normal=3)
        'excess_kurtosis': stats.kurtosis(data, fisher=True),  # Excess (normal=0)
        'p01': np.percentile(data, 1),
        'p05': np.percentile(data, 5),
        'p25': np.percentile(data, 25),
        'p50': np.percentile(data, 50),
        'p75': np.percentile(data, 75),
        'p95': np.percentile(data, 95),
        'p99': np.percentile(data, 99),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'range': np.max(data) - np.min(data),
    }

def analyze_grid_point(period, horizon, i, j, gt, bootstrap, vae_oracle, vae_prior, econ):
    """
    Analyze distribution shape for a single grid point.

    Returns:
        DataFrame with statistics for all methods
    """
    # Extract time series for this grid point
    gt_vals = gt[:, i, j]
    bootstrap_vals = bootstrap[:, i, j]
    vae_oracle_vals = vae_oracle[:, i, j]
    vae_prior_vals = vae_prior[:, i, j]
    econ_vals = econ[:, i, j]

    # Compute stats for each method
    stats_list = [
        compute_distribution_stats(gt_vals, 'Ground Truth'),
        compute_distribution_stats(bootstrap_vals, 'Bootstrap'),
        compute_distribution_stats(vae_oracle_vals, 'VAE Oracle'),
        compute_distribution_stats(vae_prior_vals, 'VAE Prior'),
        compute_distribution_stats(econ_vals, 'Econometric'),
    ]

    return pd.DataFrame(stats_list)

def run_comprehensive_analysis(periods=['insample', 'oos'], horizons=[1, 7, 14, 30]):
    """
    Run comprehensive marginal distribution analysis across all configurations.

    Returns:
        DataFrame with kurtosis/skewness differences for all grid points
    """
    all_results = []

    for period in periods:
        print(f"\n{'='*80}")
        print(f"Analyzing {period.upper()} period")
        print('='*80)

        for horizon in horizons:
            print(f"\nHorizon H={horizon}")

            # Load data
            gt = load_ground_truth(period, horizon)
            bootstrap = load_bootstrap(period, horizon)
            vae_oracle = load_vae_oracle(period, horizon)
            vae_prior = load_vae_prior(period, horizon)
            econ = load_econometric(period, horizon)

            # Align lengths
            gt, bootstrap, vae_oracle, vae_prior, econ = align_data_lengths(
                gt, bootstrap, vae_oracle, vae_prior, econ
            )

            n_days = len(gt)
            print(f"  Aligned data length: {n_days} days")

            # Analyze each grid point
            for i in range(5):  # Moneyness
                for j in range(5):  # Maturity
                    # Extract values
                    gt_vals = gt[:, i, j]
                    bootstrap_vals = bootstrap[:, i, j]
                    vae_oracle_vals = vae_oracle[:, i, j]
                    vae_prior_vals = vae_prior[:, i, j]
                    econ_vals = econ[:, i, j]

                    # Compute kurtosis and skewness
                    gt_kurt = stats.kurtosis(gt_vals, fisher=False)
                    bootstrap_kurt = stats.kurtosis(bootstrap_vals, fisher=False)
                    vae_oracle_kurt = stats.kurtosis(vae_oracle_vals, fisher=False)
                    vae_prior_kurt = stats.kurtosis(vae_prior_vals, fisher=False)
                    econ_kurt = stats.kurtosis(econ_vals, fisher=False)

                    gt_skew = stats.skew(gt_vals)
                    bootstrap_skew = stats.skew(bootstrap_vals)
                    vae_oracle_skew = stats.skew(vae_oracle_vals)
                    vae_prior_skew = stats.skew(vae_prior_vals)
                    econ_skew = stats.skew(econ_vals)

                    # Store results
                    result = {
                        'period': period,
                        'horizon': horizon,
                        'moneyness_idx': i,
                        'maturity_idx': j,
                        'gt_kurtosis': gt_kurt,
                        'bootstrap_kurtosis': bootstrap_kurt,
                        'vae_oracle_kurtosis': vae_oracle_kurt,
                        'vae_prior_kurtosis': vae_prior_kurt,
                        'econ_kurtosis': econ_kurt,
                        'bootstrap_kurt_diff': bootstrap_kurt - gt_kurt,
                        'vae_oracle_kurt_diff': vae_oracle_kurt - gt_kurt,
                        'vae_prior_kurt_diff': vae_prior_kurt - gt_kurt,
                        'econ_kurt_diff': econ_kurt - gt_kurt,
                        'gt_skewness': gt_skew,
                        'bootstrap_skewness': bootstrap_skew,
                        'vae_oracle_skewness': vae_oracle_skew,
                        'vae_prior_skewness': vae_prior_skew,
                        'econ_skewness': econ_skew,
                        'bootstrap_skew_diff': bootstrap_skew - gt_skew,
                        'vae_oracle_skew_diff': vae_oracle_skew - gt_skew,
                        'vae_prior_skew_diff': vae_prior_skew - gt_skew,
                        'econ_skew_diff': econ_skew - gt_skew,
                    }

                    all_results.append(result)

    return pd.DataFrame(all_results)

def create_kurtosis_comparison_heatmap(df, period, horizon, output_dir):
    """
    Create 4-panel heatmap comparing kurtosis differences for all methods.

    Layout:
    - Bootstrap vs GT
    - VAE Oracle vs GT
    - VAE Prior vs GT
    - Econometric vs GT
    """
    subset = df[(df['period'] == period) & (df['horizon'] == horizon)]

    # Create 5×5 grids for each method
    methods = ['bootstrap', 'vae_oracle', 'vae_prior', 'econ']
    method_names = ['Bootstrap', 'VAE Oracle', 'VAE Prior', 'Econometric']
    grids = {}

    for method in methods:
        grid = np.zeros((5, 5))
        for _, row in subset.iterrows():
            i = int(row['moneyness_idx'])
            j = int(row['maturity_idx'])
            grid[i, j] = row[f'{method}_kurt_diff']
        grids[method] = grid

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Compute global vmax for consistent color scaling
    vmax = max(abs(grids[m]).max() for m in methods)
    vmin = -vmax

    for idx, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = axes[idx]
        grid = grids[method]

        # Plot heatmap
        im = ax.imshow(grid, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

        # Annotations
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f'{grid[i,j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

        # Labels
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['M1', 'M2', 'M3', 'M4', 'M5'])
        ax.set_yticklabels(['T1', 'T2', 'T3', 'T4', 'T5'])
        ax.set_xlabel('Maturity', fontsize=12)
        ax.set_ylabel('Moneyness', fontsize=12)
        ax.set_title(f'{method_name} Kurtosis Difference vs GT', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Δ Kurtosis (Model - GT)', fontsize=10)

    plt.suptitle(f'Marginal Distribution Kurtosis Comparison\n{period.upper()} Period, H={horizon}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_file = output_dir / f'{period}_H{horizon}_kurtosis_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_distribution_overlays(period, horizon, i, j, output_dir):
    """
    Create distribution overlay plots for a specific grid point.

    Shows histogram + KDE for GT, Bootstrap, VAE Prior, Econometric.
    """
    # Load data
    gt = load_ground_truth(period, horizon)
    bootstrap = load_bootstrap(period, horizon)
    vae_prior = load_vae_prior(period, horizon)
    econ = load_econometric(period, horizon)

    # Align
    gt, bootstrap, vae_prior, econ = align_data_lengths(gt, bootstrap, vae_prior, econ)

    # Extract grid point
    gt_vals = gt[:, i, j]
    bootstrap_vals = bootstrap[:, i, j]
    vae_prior_vals = vae_prior[:, i, j]
    econ_vals = econ[:, i, j]

    # Compute kurtosis
    gt_kurt = stats.kurtosis(gt_vals, fisher=False)
    bootstrap_kurt = stats.kurtosis(bootstrap_vals, fisher=False)
    vae_prior_kurt = stats.kurtosis(vae_prior_vals, fisher=False)
    econ_kurt = stats.kurtosis(econ_vals, fisher=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    datasets = [
        ('Ground Truth', gt_vals, 'black', gt_kurt),
        ('Bootstrap', bootstrap_vals, 'blue', bootstrap_kurt),
        ('VAE Prior', vae_prior_vals, 'red', vae_prior_kurt),
        ('Econometric', econ_vals, 'green', econ_kurt),
    ]

    # Compute global range for x-axis
    all_vals = np.concatenate([gt_vals, bootstrap_vals, vae_prior_vals, econ_vals])
    xmin, xmax = all_vals.min(), all_vals.max()

    for idx, (name, vals, color, kurt) in enumerate(datasets):
        ax = axes[idx]

        # Histogram
        ax.hist(vals, bins=50, density=True, alpha=0.6, color=color, edgecolor='black')

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        x_range = np.linspace(xmin, xmax, 200)
        ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=f'{name} KDE')

        # Vertical lines for mean and median
        mean_val = np.mean(vals)
        median_val = np.median(vals)
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='darkblue', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')

        ax.set_xlabel('Implied Volatility', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name}\nKurtosis: {kurt:.2f} (Δ={kurt-gt_kurt:.2f})',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'Marginal Distribution Overlay - Grid Point ({i},{j})\n'
                f'{period.upper()}, H={horizon}',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f'{period}_H{horizon}_grid{i}{j}_distribution_overlay.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_summary_statistics_table(df, output_dir):
    """
    Create summary table of mean kurtosis/skewness differences by period and horizon.
    """
    summary_rows = []

    for period in ['insample', 'oos']:
        for horizon in [1, 7, 14, 30]:
            subset = df[(df['period'] == period) & (df['horizon'] == horizon)]

            row = {
                'Period': period,
                'Horizon': horizon,
                'Bootstrap_Kurt_Diff': subset['bootstrap_kurt_diff'].mean(),
                'VAE_Oracle_Kurt_Diff': subset['vae_oracle_kurt_diff'].mean(),
                'VAE_Prior_Kurt_Diff': subset['vae_prior_kurt_diff'].mean(),
                'Econ_Kurt_Diff': subset['econ_kurt_diff'].mean(),
                'Bootstrap_Skew_Diff': subset['bootstrap_skew_diff'].mean(),
                'VAE_Oracle_Skew_Diff': subset['vae_oracle_skew_diff'].mean(),
                'VAE_Prior_Skew_Diff': subset['vae_prior_skew_diff'].mean(),
                'Econ_Skew_Diff': subset['econ_skew_diff'].mean(),
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save to CSV
    output_file = output_dir / 'marginal_distribution_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary table: {output_file}")

    # Print to console
    print("\n" + "="*80)
    print("SUMMARY: Mean Kurtosis Difference (Model - GT)")
    print("="*80)
    print(summary_df[['Period', 'Horizon', 'Bootstrap_Kurt_Diff',
                     'VAE_Oracle_Kurt_Diff', 'VAE_Prior_Kurt_Diff',
                     'Econ_Kurt_Diff']].to_string(index=False))

    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--periods', nargs='+', default=['insample', 'oos'],
                       help='Periods to analyze')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 7, 14, 30],
                       help='Horizons to analyze')
    args = parser.parse_args()

    print("="*80)
    print("Marginal Distribution Comparison: Bootstrap vs VAE vs Econometric")
    print("="*80)

    # Run comprehensive analysis
    print("\nRunning comprehensive analysis across all grid points...")
    df = run_comprehensive_analysis(periods=args.periods, horizons=args.horizons)

    # Save full results
    output_dir = Path("results/bootstrap_baseline/comparisons/marginal_distributions")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / 'marginal_distribution_comprehensive.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved comprehensive results: {csv_file}")

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    # Kurtosis comparison heatmaps for all period/horizon combinations
    for period in args.periods:
        for horizon in args.horizons:
            create_kurtosis_comparison_heatmap(df, period, horizon, output_dir)

    # Distribution overlays for representative grid points
    print("\nGenerating distribution overlay plots...")
    representative_points = [(2, 2), (1, 1), (3, 3)]  # ATM, low, high
    for period in args.periods:
        for horizon in [1, 30]:  # Just show H1 and H30
            for (i, j) in representative_points:
                create_distribution_overlays(period, horizon, i, j, output_dir)

    # Summary statistics table
    summary_df = create_summary_statistics_table(df, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nKey Findings:")
    print("  - Bootstrap kurtosis match: Check if ~0.0 (perfect match expected)")
    print("  - VAE kurtosis mismatch: Compare to previous +0.19 to +2.64 range")
    print("  - Econometric: Known to be ~+10.6 (poor)")

if __name__ == "__main__":
    main()
