"""
Unified 4-Way Comparison Visualizations

Creates comprehensive multi-panel comparison figures showing
Bootstrap, VAE Oracle, VAE Prior, and Econometric predictions
side-by-side for direct visual comparison.

Usage:
    python experiments/bootstrap_baseline/visualize_4way_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_ground_truth(period):
    """Load ground truth volatility surfaces."""
    data = np.load("data/vol_surface_with_ret.npz")
    surface = data['surface']

    if period == 'insample':
        gt = surface[:3900]
    else:
        gt = surface[5000:5792]

    return gt

def load_all_methods(period, horizon):
    """Load predictions from all methods."""
    # Bootstrap
    bootstrap_file = f"results/bootstrap_baseline/predictions/{period}/bootstrap_predictions_H{horizon}.npz"
    bootstrap_data = np.load(bootstrap_file)
    bootstrap = bootstrap_data['surfaces']  # (n_days, 3, 5, 5)

    # VAE Oracle
    if period == 'insample':
        vae_oracle_file = "results/backfill_16yr/predictions/insample_reconstruction_16yr.npz"
    else:
        vae_oracle_file = "results/backfill_16yr/predictions/oos_reconstruction_16yr.npz"
    vae_oracle_data = np.load(vae_oracle_file)
    vae_oracle = vae_oracle_data[f'recon_h{horizon}']

    # VAE Prior
    if period == 'insample':
        vae_prior_file = "results/backfill_16yr/predictions/vae_prior_insample_16yr.npz"
    else:
        vae_prior_file = "results/backfill_16yr/predictions/vae_prior_oos_16yr.npz"
    vae_prior_data = np.load(vae_prior_file)
    vae_prior = vae_prior_data[f'recon_h{horizon}']

    # Econometric
    if period == 'insample':
        econ_file = "results/econometric_baseline/predictions/econometric_backfill_insample.npz"
    else:
        econ_file = "results/econometric_baseline/predictions/econometric_backfill_oos.npz"
    econ_data = np.load(econ_file)
    econ = econ_data[f'recon_h{horizon}']

    return bootstrap, vae_oracle, vae_prior, econ

def align_data_lengths(*arrays):
    """Align multiple arrays to minimum common length."""
    min_len = min(len(arr) for arr in arrays)
    aligned = [arr[:min_len] for arr in arrays]
    return aligned

def create_teacher_forcing_comparison(period, horizon, grid_points, output_dir):
    """
    Create 12-panel teacher forcing comparison.
    3 grid points × 4 methods = 12 panels.
    """
    # Load data
    gt = load_ground_truth(period)
    bootstrap, vae_oracle, vae_prior, econ = load_all_methods(period, horizon)

    # Align
    gt, bootstrap, vae_oracle, vae_prior, econ = align_data_lengths(
        gt, bootstrap, vae_oracle, vae_prior, econ
    )

    n_days = len(gt)
    time_axis = np.arange(n_days)

    # Create figure
    n_grid_points = len(grid_points)
    fig, axes = plt.subplots(n_grid_points, 4, figsize=(20, 4*n_grid_points))

    method_names = ['Bootstrap', 'VAE Oracle', 'VAE Prior', 'Econometric']
    method_data = [bootstrap, vae_oracle, vae_prior, econ]
    colors = ['blue', 'orange', 'red', 'green']

    for row_idx, (i, j) in enumerate(grid_points):
        gt_vals = gt[:, i, j]

        for col_idx, (method_name, method_pred, color) in enumerate(zip(method_names, method_data, colors)):
            ax = axes[row_idx, col_idx]

            # Extract quantiles
            p05 = method_pred[:, 0, i, j]
            p50 = method_pred[:, 1, i, j]
            p95 = method_pred[:, 2, i, j]

            # Plot ground truth
            ax.plot(time_axis, gt_vals, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.7)

            # Plot p50 prediction
            ax.plot(time_axis, p50, color=color, linewidth=1.5, label=f'{method_name} p50', alpha=0.8)

            # Plot confidence band
            ax.fill_between(time_axis, p05, p95, color=color, alpha=0.2, label='90% CI')

            # Labels
            if row_idx == 0:
                ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'Grid ({i},{j})\nIV', fontsize=11)
            if row_idx == n_grid_points - 1:
                ax.set_xlabel('Days', fontsize=11)

            # Legend only on first row
            if row_idx == 0:
                ax.legend(fontsize=8, loc='upper right')

            ax.grid(alpha=0.3)

    plt.suptitle(f'Teacher Forcing Comparison - {period.upper()}, H={horizon}\n'
                 f'Ground Truth vs 4 Methods ({n_days} days)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f'{period}_H{horizon}_teacher_forcing_4way.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_rmse_4panel_comparison(period, horizon, output_dir):
    """
    Create 4-panel RMSE heatmap comparison.
    """
    # Load RMSE data
    rmse_file = "results/bootstrap_baseline/comparisons/rmse/rmse_comprehensive.csv"
    df = pd.read_csv(rmse_file)

    subset = df[(df['period'] == period) & (df['horizon'] == horizon)]

    # Create grids
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

    vmax = max(grids[m].max() for m in methods)

    for idx, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = axes[idx]
        grid = grids[method]

        im = ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')

        # Annotations
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{grid[i,j]:.4f}',
                       ha="center", va="center", color="black", fontsize=9)

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['M1', 'M2', 'M3', 'M4', 'M5'])
        ax.set_yticklabels(['T1', 'T2', 'T3', 'T4', 'T5'])
        ax.set_xlabel('Maturity', fontsize=12)
        ax.set_ylabel('Moneyness', fontsize=12)

        mean_rmse = grid.mean()
        ax.set_title(f'{method_name}\nMean RMSE: {mean_rmse:.4f}',
                    fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RMSE', fontsize=10)

    plt.suptitle(f'RMSE Comparison: 4 Methods\n{period.upper()}, H={horizon}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_file = output_dir / f'{period}_H{horizon}_rmse_4panel.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_distribution_4panel_comparison(period, horizon, i, j, output_dir):
    """
    Create 4-panel distribution overlay comparison.
    """
    # Load data
    gt = load_ground_truth(period)
    bootstrap, vae_oracle, vae_prior, econ = load_all_methods(period, horizon)

    # Align
    gt, bootstrap, vae_oracle, vae_prior, econ = align_data_lengths(
        gt, bootstrap, vae_oracle, vae_prior, econ
    )

    # Extract grid point
    gt_vals = gt[:, i, j]
    bootstrap_vals = bootstrap[:, 1, i, j]  # p50
    vae_oracle_vals = vae_oracle[:, 1, i, j]
    vae_prior_vals = vae_prior[:, 1, i, j]
    econ_vals = econ[:, 1, i, j]

    # Compute kurtosis
    from scipy import stats
    gt_kurt = stats.kurtosis(gt_vals, fisher=False)
    bootstrap_kurt = stats.kurtosis(bootstrap_vals, fisher=False)
    vae_oracle_kurt = stats.kurtosis(vae_oracle_vals, fisher=False)
    vae_prior_kurt = stats.kurtosis(vae_prior_vals, fisher=False)
    econ_kurt = stats.kurtosis(econ_vals, fisher=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    datasets = [
        ('Bootstrap', bootstrap_vals, 'blue', bootstrap_kurt),
        ('VAE Oracle', vae_oracle_vals, 'orange', vae_oracle_kurt),
        ('VAE Prior', vae_prior_vals, 'red', vae_prior_kurt),
        ('Econometric', econ_vals, 'green', econ_kurt),
    ]

    # Compute global range
    all_vals = np.concatenate([gt_vals, bootstrap_vals, vae_oracle_vals, vae_prior_vals, econ_vals])
    xmin, xmax = all_vals.min(), all_vals.max()

    for idx, (name, vals, color, kurt) in enumerate(datasets):
        ax = axes[idx]

        # Ground truth histogram (background)
        ax.hist(gt_vals, bins=40, density=True, alpha=0.3, color='gray',
               edgecolor='black', label='GT')

        # Method histogram
        ax.hist(vals, bins=40, density=True, alpha=0.6, color=color,
               edgecolor='black', label=name)

        # KDE
        from scipy.stats import gaussian_kde
        gt_kde = gaussian_kde(gt_vals)
        method_kde = gaussian_kde(vals)
        x_range = np.linspace(xmin, xmax, 200)
        ax.plot(x_range, gt_kde(x_range), color='black', linewidth=2,
               linestyle='--', label='GT KDE')
        ax.plot(x_range, method_kde(x_range), color=color, linewidth=2,
               label=f'{name} KDE')

        ax.set_xlabel('Implied Volatility', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)

        kurt_diff = kurt - gt_kurt
        ax.set_title(f'{name}\nGT Kurt: {gt_kurt:.2f}, {name} Kurt: {kurt:.2f}\n'
                    f'Δ={kurt_diff:+.2f}',
                    fontsize=13, fontweight='bold')

        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f'Marginal Distribution Comparison - Grid ({i},{j})\n'
                 f'{period.upper()}, H={horizon}',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / f'{period}_H{horizon}_grid{i}{j}_dist_4panel.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_calibration_summary_2panel(output_dir):
    """
    Create 2-panel CI calibration summary (in-sample vs OOS).
    """
    # Load data
    ci_file = "results/bootstrap_baseline/comparisons/ci_calibration/ci_calibration_comprehensive.csv"
    df = pd.read_csv(ci_file)

    # Aggregate by period and horizon
    summary = df.groupby(['period', 'horizon']).agg({
        'bootstrap_violation_rate': 'mean',
        'vae_oracle_violation_rate': 'mean',
        'vae_prior_violation_rate': 'mean',
        'econ_violation_rate': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    periods = ['insample', 'oos']
    period_names = ['In-Sample (2004-2019)', 'Out-of-Sample (2019-2023)']

    methods = ['bootstrap', 'vae_oracle', 'vae_prior', 'econ']
    method_names = ['Bootstrap', 'VAE Oracle', 'VAE Prior', 'Econometric']
    colors = ['blue', 'orange', 'red', 'green']

    for idx, (period, period_name) in enumerate(zip(periods, period_names)):
        ax = axes[idx]
        subset = summary[summary['period'] == period]

        horizons = subset['horizon'].values

        for method, name, color in zip(methods, method_names, colors):
            rates = subset[f'{method}_violation_rate'].values * 100
            ax.plot(horizons, rates, marker='o', markersize=10,
                   linewidth=2.5, label=name, color=color)

        # Target line
        ax.axhline(10, color='black', linestyle='--', linewidth=2, label='Target (10%)')

        ax.set_xlabel('Forecast Horizon (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CI Violation Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(period_name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks(horizons)
        ax.set_ylim(0, max(85, ax.get_ylim()[1]))

    plt.suptitle('CI Calibration Summary (90% CIs, Target: 10% Violations)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'ci_calibration_summary_2panel.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--periods', nargs='+', default=['insample', 'oos'])
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 30])
    args = parser.parse_args()

    print("="*80)
    print("Unified 4-Way Comparison Visualizations")
    print("="*80)

    output_dir = Path("results/bootstrap_baseline/comparisons/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Representative grid points
    grid_points = [(1, 1), (2, 2), (3, 3)]  # Low, ATM, High

    # 1. Teacher forcing comparisons
    print("\nGenerating teacher forcing comparisons...")
    for period in args.periods:
        for horizon in args.horizons:
            create_teacher_forcing_comparison(period, horizon, grid_points, output_dir)

    # 2. RMSE 4-panel comparisons
    print("\nGenerating RMSE 4-panel comparisons...")
    for period in args.periods:
        for horizon in args.horizons:
            create_rmse_4panel_comparison(period, horizon, output_dir)

    # 3. Distribution 4-panel comparisons (ATM only)
    print("\nGenerating distribution 4-panel comparisons...")
    for period in args.periods:
        for horizon in args.horizons:
            create_distribution_4panel_comparison(period, horizon, 2, 2, output_dir)

    # 4. Calibration summary
    print("\nGenerating calibration summary...")
    create_calibration_summary_2panel(output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated visualizations:")
    print("  - Teacher forcing comparisons (4 files)")
    print("  - RMSE 4-panel heatmaps (4 files)")
    print("  - Distribution 4-panel overlays (4 files)")
    print("  - CI calibration summary (1 file)")
    print("\nTotal: 13 unified comparison figures")

if __name__ == "__main__":
    main()
