"""
RMSE Comparison: Bootstrap vs VAE vs Econometric

Compares root mean squared error across all methods to assess
point forecast accuracy. While bootstrap excels at distribution matching,
this analysis reveals whether it can also compete on RMSE.

Usage:
    python experiments/bootstrap_baseline/compare_rmse_all_methods.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

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

def compute_rmse(pred, gt):
    """Compute RMSE between prediction and ground truth."""
    return np.sqrt(np.mean((pred - gt) ** 2))

def compute_grid_rmse(pred, gt):
    """
    Compute RMSE for each grid point separately.

    Args:
        pred: (n_days, 5, 5) predictions
        gt: (n_days, 5, 5) ground truth

    Returns:
        rmse_grid: (5, 5) RMSE for each grid point
    """
    rmse_grid = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            rmse_grid[i, j] = compute_rmse(pred[:, i, j], gt[:, i, j])
    return rmse_grid

def run_comprehensive_rmse_analysis(periods=['insample', 'oos'], horizons=[1, 7, 14, 30]):
    """
    Compute RMSE for all methods across all configurations.

    Returns:
        DataFrame with grid-level and aggregate RMSE
    """
    all_results = []

    for period in periods:
        print(f"\n{'='*80}")
        print(f"Computing RMSE for {period.upper()} period")
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

            # Compute grid-level RMSE
            bootstrap_rmse_grid = compute_grid_rmse(bootstrap, gt_aligned)
            vae_oracle_rmse_grid = compute_grid_rmse(vae_oracle, gt_aligned)
            vae_prior_rmse_grid = compute_grid_rmse(vae_prior, gt_aligned)
            econ_rmse_grid = compute_grid_rmse(econ, gt_aligned)

            # Store results for each grid point
            for i in range(5):
                for j in range(5):
                    result = {
                        'period': period,
                        'horizon': horizon,
                        'moneyness_idx': i,
                        'maturity_idx': j,
                        'bootstrap_rmse': bootstrap_rmse_grid[i, j],
                        'vae_oracle_rmse': vae_oracle_rmse_grid[i, j],
                        'vae_prior_rmse': vae_prior_rmse_grid[i, j],
                        'econ_rmse': econ_rmse_grid[i, j],
                    }
                    all_results.append(result)

    return pd.DataFrame(all_results)

def create_rmse_comparison_heatmap(df, period, horizon, output_dir):
    """
    Create 4-panel heatmap comparing RMSE for all methods.
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
            grid[i, j] = row[f'{method}_rmse']
        grids[method] = grid

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Compute global vmax for consistent color scaling
    vmax = max(grids[m].max() for m in methods)
    vmin = 0

    for idx, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = axes[idx]
        grid = grids[method]

        # Plot heatmap
        im = ax.imshow(grid, cmap='YlOrRd', vmin=vmin, vmax=vmax, aspect='auto')

        # Annotations
        for i in range(5):
            for j in range(5):
                text = ax.text(j, i, f'{grid[i,j]:.4f}',
                             ha="center", va="center", color="black", fontsize=9)

        # Labels
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['M1', 'M2', 'M3', 'M4', 'M5'])
        ax.set_yticklabels(['T1', 'T2', 'T3', 'T4', 'T5'])
        ax.set_xlabel('Maturity', fontsize=12)
        ax.set_ylabel('Moneyness', fontsize=12)

        # Compute mean RMSE for this method
        mean_rmse = grid.mean()
        ax.set_title(f'{method_name}\nMean RMSE: {mean_rmse:.4f}',
                    fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RMSE', fontsize=10)

    plt.suptitle(f'RMSE Comparison Across Methods\n{period.upper()} Period, H={horizon}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_file = output_dir / f'{period}_H{horizon}_rmse_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_horizon_degradation_plot(df, period, output_dir):
    """
    Plot RMSE vs horizon for all methods.
    """
    subset = df[df['period'] == period]

    # Compute mean RMSE for each method and horizon
    horizon_stats = subset.groupby('horizon').agg({
        'bootstrap_rmse': ['mean', 'std'],
        'vae_oracle_rmse': ['mean', 'std'],
        'vae_prior_rmse': ['mean', 'std'],
        'econ_rmse': ['mean', 'std'],
    }).reset_index()

    horizons = horizon_stats['horizon'].values

    # Extract means and stds
    methods = ['bootstrap', 'vae_oracle', 'vae_prior', 'econ']
    method_names = ['Bootstrap', 'VAE Oracle', 'VAE Prior', 'Econometric']
    colors = ['blue', 'orange', 'red', 'green']

    fig, ax = plt.subplots(figsize=(12, 8))

    for method, name, color in zip(methods, method_names, colors):
        means = horizon_stats[(f'{method}_rmse', 'mean')].values
        stds = horizon_stats[(f'{method}_rmse', 'std')].values

        ax.errorbar(horizons, means, yerr=stds, marker='o', markersize=10,
                   linewidth=2.5, capsize=5, label=name, color=color)

    ax.set_xlabel('Forecast Horizon (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean RMSE', fontsize=14, fontweight='bold')
    ax.set_title(f'RMSE vs Forecast Horizon - {period.upper()} Period',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticks(horizons)

    plt.tight_layout()

    output_file = output_dir / f'{period}_rmse_vs_horizon.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_summary_table(df, output_dir):
    """
    Create summary table of mean RMSE by period and horizon.
    """
    summary_rows = []

    for period in ['insample', 'oos']:
        for horizon in [1, 7, 14, 30]:
            subset = df[(df['period'] == period) & (df['horizon'] == horizon)]

            row = {
                'Period': period,
                'Horizon': horizon,
                'Bootstrap_RMSE': subset['bootstrap_rmse'].mean(),
                'VAE_Oracle_RMSE': subset['vae_oracle_rmse'].mean(),
                'VAE_Prior_RMSE': subset['vae_prior_rmse'].mean(),
                'Econ_RMSE': subset['econ_rmse'].mean(),
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Add relative performance columns (vs VAE Oracle as baseline)
    summary_df['Bootstrap_vs_Oracle_%'] = 100 * (
        summary_df['Bootstrap_RMSE'] / summary_df['VAE_Oracle_RMSE'] - 1
    )
    summary_df['VAE_Prior_vs_Oracle_%'] = 100 * (
        summary_df['VAE_Prior_RMSE'] / summary_df['VAE_Oracle_RMSE'] - 1
    )
    summary_df['Econ_vs_Oracle_%'] = 100 * (
        summary_df['Econ_RMSE'] / summary_df['VAE_Oracle_RMSE'] - 1
    )

    # Save to CSV
    output_file = output_dir / 'rmse_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary table: {output_file}")

    # Print to console
    print("\n" + "="*80)
    print("SUMMARY: Mean RMSE Across Methods")
    print("="*80)
    print(summary_df[['Period', 'Horizon', 'Bootstrap_RMSE',
                     'VAE_Oracle_RMSE', 'VAE_Prior_RMSE',
                     'Econ_RMSE']].to_string(index=False))

    print("\n" + "="*80)
    print("Relative Performance (% difference vs VAE Oracle)")
    print("="*80)
    print(summary_df[['Period', 'Horizon', 'Bootstrap_vs_Oracle_%',
                     'VAE_Prior_vs_Oracle_%', 'Econ_vs_Oracle_%']].to_string(index=False))

    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--periods', nargs='+', default=['insample', 'oos'])
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 7, 14, 30])
    args = parser.parse_args()

    print("="*80)
    print("RMSE Comparison: Bootstrap vs VAE vs Econometric")
    print("="*80)

    # Run analysis
    print("\nComputing RMSE for all methods...")
    df = run_comprehensive_rmse_analysis(periods=args.periods, horizons=args.horizons)

    # Save results
    output_dir = Path("results/bootstrap_baseline/comparisons/rmse")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / 'rmse_comprehensive.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved comprehensive results: {csv_file}")

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    # RMSE heatmaps for all period/horizon combinations
    for period in args.periods:
        for horizon in args.horizons:
            create_rmse_comparison_heatmap(df, period, horizon, output_dir)

    # Horizon degradation plots
    for period in args.periods:
        create_horizon_degradation_plot(df, period, output_dir)

    # Summary table
    summary_df = create_summary_table(df, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
