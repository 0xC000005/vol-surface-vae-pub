"""
CI Calibration Comparison: Bootstrap vs VAE vs Econometric

Compares confidence interval calibration across methods.
Well-calibrated 90% CIs should have ~10% violation rate (5% below p05, 5% above p95).

Usage:
    python experiments/bootstrap_baseline/compare_ci_calibration.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
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
    """Load bootstrap predictions with quantiles."""
    file_path = f"results/bootstrap_baseline/predictions/{period}/bootstrap_predictions_H{horizon}.npz"
    data = np.load(file_path)
    surfaces = data['surfaces']  # (n_days, 3, 5, 5) - [p05, p50, p95]
    return surfaces[:, 0, :, :], surfaces[:, 1, :, :], surfaces[:, 2, :, :]

def load_vae_oracle(period, horizon):
    """Load VAE Oracle predictions with quantiles."""
    if period == 'insample':
        file_path = "results/backfill_16yr/predictions/insample_reconstruction_16yr.npz"
    else:
        file_path = "results/backfill_16yr/predictions/oos_reconstruction_16yr.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']
    return surfaces[:, 0, :, :], surfaces[:, 1, :, :], surfaces[:, 2, :, :]

def load_vae_prior(period, horizon):
    """Load VAE Prior predictions with quantiles."""
    if period == 'insample':
        file_path = "results/backfill_16yr/predictions/vae_prior_insample_16yr.npz"
    else:
        file_path = "results/backfill_16yr/predictions/vae_prior_oos_16yr.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']
    return surfaces[:, 0, :, :], surfaces[:, 1, :, :], surfaces[:, 2, :, :]

def load_econometric(period, horizon):
    """Load econometric baseline predictions with quantiles."""
    if period == 'insample':
        file_path = "results/econometric_baseline/predictions/econometric_backfill_insample.npz"
    else:
        file_path = "results/econometric_baseline/predictions/econometric_backfill_oos.npz"

    data = np.load(file_path)
    surfaces = data[f'recon_h{horizon}']
    return surfaces[:, 0, :, :], surfaces[:, 1, :, :], surfaces[:, 2, :, :]

def align_data_lengths(*arrays):
    """Align multiple arrays to minimum common length."""
    min_len = min(len(arr) for arr in arrays)
    aligned = [arr[:min_len] for arr in arrays]
    return aligned

def compute_ci_violations(gt, p05, p95):
    """
    Compute CI violation statistics.

    Returns:
        dict with violation metrics
    """
    # Check violations
    below_p05 = gt < p05
    above_p95 = gt > p95
    violations = below_p05 | above_p95

    # Compute rates
    violation_rate = violations.mean()
    below_rate = below_p05.mean()
    above_rate = above_p95.mean()

    # CI width
    ci_width = p95 - p05

    return {
        'violation_rate': violation_rate,
        'below_p05_rate': below_rate,
        'above_p95_rate': above_rate,
        'mean_ci_width': ci_width.mean(),
        'std_ci_width': ci_width.std(),
    }

def run_comprehensive_ci_analysis(periods=['insample', 'oos'], horizons=[1, 7, 14, 30]):
    """
    Compute CI calibration for all methods across all configurations.

    Returns:
        DataFrame with grid-level CI statistics
    """
    all_results = []

    for period in periods:
        print(f"\n{'='*80}")
        print(f"Computing CI Calibration for {period.upper()} period")
        print('='*80)

        gt = load_ground_truth(period)

        for horizon in horizons:
            print(f"\nHorizon H={horizon}")

            # Load predictions with quantiles
            bootstrap_p05, bootstrap_p50, bootstrap_p95 = load_bootstrap(period, horizon)
            vae_oracle_p05, vae_oracle_p50, vae_oracle_p95 = load_vae_oracle(period, horizon)
            vae_prior_p05, vae_prior_p50, vae_prior_p95 = load_vae_prior(period, horizon)
            econ_p05, econ_p50, econ_p95 = load_econometric(period, horizon)

            # Align lengths
            aligned = align_data_lengths(
                gt, bootstrap_p05, bootstrap_p50, bootstrap_p95,
                vae_oracle_p05, vae_oracle_p50, vae_oracle_p95,
                vae_prior_p05, vae_prior_p50, vae_prior_p95,
                econ_p05, econ_p50, econ_p95
            )

            (gt_aligned, bootstrap_p05, bootstrap_p50, bootstrap_p95,
             vae_oracle_p05, vae_oracle_p50, vae_oracle_p95,
             vae_prior_p05, vae_prior_p50, vae_prior_p95,
             econ_p05, econ_p50, econ_p95) = aligned

            n_days = len(gt_aligned)
            print(f"  Aligned data length: {n_days} days")

            # Compute CI violations for each grid point
            for i in range(5):
                for j in range(5):
                    # Extract grid point time series
                    gt_vals = gt_aligned[:, i, j]

                    bootstrap_stats = compute_ci_violations(
                        gt_vals, bootstrap_p05[:, i, j], bootstrap_p95[:, i, j]
                    )
                    vae_oracle_stats = compute_ci_violations(
                        gt_vals, vae_oracle_p05[:, i, j], vae_oracle_p95[:, i, j]
                    )
                    vae_prior_stats = compute_ci_violations(
                        gt_vals, vae_prior_p05[:, i, j], vae_prior_p95[:, i, j]
                    )
                    econ_stats = compute_ci_violations(
                        gt_vals, econ_p05[:, i, j], econ_p95[:, i, j]
                    )

                    # Store results
                    result = {
                        'period': period,
                        'horizon': horizon,
                        'moneyness_idx': i,
                        'maturity_idx': j,
                        'bootstrap_violation_rate': bootstrap_stats['violation_rate'],
                        'bootstrap_below_rate': bootstrap_stats['below_p05_rate'],
                        'bootstrap_above_rate': bootstrap_stats['above_p95_rate'],
                        'bootstrap_ci_width': bootstrap_stats['mean_ci_width'],
                        'vae_oracle_violation_rate': vae_oracle_stats['violation_rate'],
                        'vae_oracle_below_rate': vae_oracle_stats['below_p05_rate'],
                        'vae_oracle_above_rate': vae_oracle_stats['above_p95_rate'],
                        'vae_oracle_ci_width': vae_oracle_stats['mean_ci_width'],
                        'vae_prior_violation_rate': vae_prior_stats['violation_rate'],
                        'vae_prior_below_rate': vae_prior_stats['below_p05_rate'],
                        'vae_prior_above_rate': vae_prior_stats['above_p95_rate'],
                        'vae_prior_ci_width': vae_prior_stats['mean_ci_width'],
                        'econ_violation_rate': econ_stats['violation_rate'],
                        'econ_below_rate': econ_stats['below_p05_rate'],
                        'econ_above_rate': econ_stats['above_p95_rate'],
                        'econ_ci_width': econ_stats['mean_ci_width'],
                    }

                    all_results.append(result)

    return pd.DataFrame(all_results)

def create_violation_rate_heatmap(df, period, horizon, output_dir):
    """
    Create 4-panel heatmap comparing CI violation rates for all methods.
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
            grid[i, j] = row[f'{method}_violation_rate'] * 100  # Convert to percentage
        grids[method] = grid

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    # Color scale: 10% = perfect (green), <10% = too wide (blue), >10% = too narrow (red)
    vmin = 0
    vmax = max(30, max(grids[m].max() for m in methods))  # Cap at 30% or max value

    for idx, (method, method_name) in enumerate(zip(methods, method_names)):
        ax = axes[idx]
        grid = grids[method]

        # Plot heatmap
        im = ax.imshow(grid, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, aspect='auto')

        # Annotations
        for i in range(5):
            for j in range(5):
                # Color text based on proximity to 10%
                deviation = abs(grid[i,j] - 10)
                text_color = 'black' if deviation < 5 else 'white'
                text = ax.text(j, i, f'{grid[i,j]:.1f}%',
                             ha="center", va="center", color=text_color,
                             fontsize=9, fontweight='bold')

        # Labels
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['M1', 'M2', 'M3', 'M4', 'M5'])
        ax.set_yticklabels(['T1', 'T2', 'T3', 'T4', 'T5'])
        ax.set_xlabel('Maturity', fontsize=12)
        ax.set_ylabel('Moneyness', fontsize=12)

        # Compute mean violation rate for this method
        mean_viol = grid.mean()
        ax.set_title(f'{method_name}\nMean Violation: {mean_viol:.1f}% (Target: 10%)',
                    fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('CI Violation Rate (%)', fontsize=10)

    plt.suptitle(f'CI Calibration Comparison (90% CIs)\n{period.upper()} Period, H={horizon}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_file = output_dir / f'{period}_H{horizon}_ci_violation_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_calibration_plot(df, period, output_dir):
    """
    Plot violation rate vs horizon for all methods.
    """
    subset = df[df['period'] == period]

    # Compute mean violation rate for each method and horizon
    horizon_stats = subset.groupby('horizon').agg({
        'bootstrap_violation_rate': 'mean',
        'vae_oracle_violation_rate': 'mean',
        'vae_prior_violation_rate': 'mean',
        'econ_violation_rate': 'mean',
    }).reset_index()

    horizons = horizon_stats['horizon'].values

    methods = ['bootstrap', 'vae_oracle', 'vae_prior', 'econ']
    method_names = ['Bootstrap', 'VAE Oracle', 'VAE Prior', 'Econometric']
    colors = ['blue', 'orange', 'red', 'green']

    fig, ax = plt.subplots(figsize=(12, 8))

    for method, name, color in zip(methods, method_names, colors):
        rates = horizon_stats[f'{method}_violation_rate'].values * 100  # Convert to %

        ax.plot(horizons, rates, marker='o', markersize=10,
               linewidth=2.5, label=name, color=color)

    # Add target line at 10%
    ax.axhline(10, color='black', linestyle='--', linewidth=2, label='Target (10%)')

    ax.set_xlabel('Forecast Horizon (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CI Violation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'CI Calibration vs Forecast Horizon - {period.upper()} Period',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xticks(horizons)
    ax.set_ylim(0, max(40, ax.get_ylim()[1]))

    plt.tight_layout()

    output_file = output_dir / f'{period}_ci_calibration_vs_horizon.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

def create_summary_table(df, output_dir):
    """
    Create summary table of mean CI violation rates by period and horizon.
    """
    summary_rows = []

    for period in ['insample', 'oos']:
        for horizon in [1, 7, 14, 30]:
            subset = df[(df['period'] == period) & (df['horizon'] == horizon)]

            row = {
                'Period': period,
                'Horizon': horizon,
                'Bootstrap_Violation_%': subset['bootstrap_violation_rate'].mean() * 100,
                'VAE_Oracle_Violation_%': subset['vae_oracle_violation_rate'].mean() * 100,
                'VAE_Prior_Violation_%': subset['vae_prior_violation_rate'].mean() * 100,
                'Econ_Violation_%': subset['econ_violation_rate'].mean() * 100,
                'Bootstrap_CI_Width': subset['bootstrap_ci_width'].mean(),
                'VAE_Oracle_CI_Width': subset['vae_oracle_ci_width'].mean(),
                'VAE_Prior_CI_Width': subset['vae_prior_ci_width'].mean(),
                'Econ_CI_Width': subset['econ_ci_width'].mean(),
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save to CSV
    output_file = output_dir / 'ci_calibration_summary.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary table: {output_file}")

    # Print to console
    print("\n" + "="*80)
    print("SUMMARY: CI Violation Rates (Target: 10%)")
    print("="*80)
    print(summary_df[['Period', 'Horizon', 'Bootstrap_Violation_%',
                     'VAE_Oracle_Violation_%', 'VAE_Prior_Violation_%',
                     'Econ_Violation_%']].to_string(index=False))

    print("\n" + "="*80)
    print("Mean CI Width")
    print("="*80)
    print(summary_df[['Period', 'Horizon', 'Bootstrap_CI_Width',
                     'VAE_Oracle_CI_Width', 'VAE_Prior_CI_Width',
                     'Econ_CI_Width']].to_string(index=False))

    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--periods', nargs='+', default=['insample', 'oos'])
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 7, 14, 30])
    args = parser.parse_args()

    print("="*80)
    print("CI Calibration Comparison: Bootstrap vs VAE vs Econometric")
    print("="*80)

    # Run analysis
    print("\nComputing CI calibration for all methods...")
    df = run_comprehensive_ci_analysis(periods=args.periods, horizons=args.horizons)

    # Save results
    output_dir = Path("results/bootstrap_baseline/comparisons/ci_calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / 'ci_calibration_comprehensive.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved comprehensive results: {csv_file}")

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    # Violation rate heatmaps for all period/horizon combinations
    for period in args.periods:
        for horizon in args.horizons:
            create_violation_rate_heatmap(df, period, horizon, output_dir)

    # Calibration plots
    for period in args.periods:
        create_calibration_plot(df, period, output_dir)

    # Summary table
    summary_df = create_summary_table(df, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nKey Insight:")
    print("  - Well-calibrated CIs should have ~10% violation rate")
    print("  - Lower = too conservative (wide CIs)")
    print("  - Higher = too aggressive (narrow CIs)")

if __name__ == "__main__":
    main()
