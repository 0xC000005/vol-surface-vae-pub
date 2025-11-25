"""
Diagnose distribution SHAPE differences (not just variance).

Computes kurtosis and skewness to understand why GT appears flat/spread
while models appear peaked, even when variance is similar.

Updated to:
- Use new file path structure (results/backfill_16yr/predictions/)
- Run comprehensive grid-level analysis
- Generate CSV outputs for further analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

# Run modes
RUN_ALL = True  # Set to True for comprehensive analysis, False for single grid point
SAVE_CSV = True  # Save grid-level statistics to CSV

# Single grid point test (when RUN_ALL=False)
TEST_PERIOD = 'oos'  # or 'insample'
TEST_HORIZON = 1
TEST_GRID_I = 2  # ATM
TEST_GRID_J = 2  # 6M

moneyness_labels = ['0.70', '0.85', '1.00', '1.15', '1.30']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

output_dir = Path("results/distribution_analysis/shape_diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_data(period, horizon):
    """Load data for given period and horizon."""
    if period == 'insample':
        oracle_file = "results/backfill_16yr/predictions/insample_reconstruction_16yr.npz"
        vae_file = "results/backfill_16yr/predictions/vae_prior_insample_16yr.npz"
        econ_file = "results/econometric_baseline/predictions/econometric_backfill_insample.npz"
    else:  # oos
        oracle_file = "results/backfill_16yr/predictions/oos_reconstruction_16yr.npz"
        vae_file = "results/backfill_16yr/predictions/vae_prior_oos_16yr.npz"
        econ_file = "results/econometric_baseline/predictions/econometric_backfill_oos.npz"

    oracle_data = np.load(oracle_file)
    vae_data = np.load(vae_file)
    econ_data = np.load(econ_file)
    gt_data = np.load("data/vol_surface_with_ret.npz")

    recon_key = f'recon_h{horizon}'
    indices_key = f'indices_h{horizon}'

    oracle_p50 = oracle_data[recon_key][:, 1, :, :]
    vae_p50 = vae_data[recon_key][:, 1, :, :]
    econ_p50 = econ_data[recon_key][:, 1, :, :]

    indices = oracle_data[indices_key]
    gt = gt_data["surface"][indices]

    # Align lengths
    min_len = min(gt.shape[0], oracle_p50.shape[0], vae_p50.shape[0], econ_p50.shape[0])
    gt = gt[:min_len]
    oracle_p50 = oracle_p50[:min_len]
    vae_p50 = vae_p50[:min_len]
    econ_p50 = econ_p50[:min_len]

    return gt, oracle_p50, vae_p50, econ_p50, min_len


def compute_stats(data, name):
    """Compute comprehensive distributional statistics."""
    return {
        'name': name,
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data, fisher=False),  # fisher=False gives Pearson (normal=3)
        'excess_kurtosis': stats.kurtosis(data, fisher=True),  # fisher=True gives excess (normal=0)
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


def analyze_single_grid_point(period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50,
                               verbose=True, create_plot=True):
    """Analyze distribution shape for a single grid point."""

    gt_vals = gt[:, i, j]
    oracle_vals = oracle_p50[:, i, j]
    vae_vals = vae_p50[:, i, j]
    econ_vals = econ_p50[:, i, j]

    # Compute distributional statistics
    gt_stats = compute_stats(gt_vals, 'Ground Truth')
    oracle_stats = compute_stats(oracle_vals, 'Oracle')
    vae_stats = compute_stats(vae_vals, 'VAE Prior')
    econ_stats = compute_stats(econ_vals, 'Econometric')

    if verbose:
        print(f"\nGrid Point ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity")
        print("="*80)
        print()

        # Print statistics table
        print("DISTRIBUTIONAL STATISTICS")
        print("-"*80)
        print(f"{'Metric':<20} {'GT':<12} {'Oracle':<12} {'VAE':<12} {'Econometric':<12}")
        print("-"*80)
        print(f"{'Mean':<20} {gt_stats['mean']:<12.6f} {oracle_stats['mean']:<12.6f} {vae_stats['mean']:<12.6f} {econ_stats['mean']:<12.6f}")
        print(f"{'Std':<20} {gt_stats['std']:<12.6f} {oracle_stats['std']:<12.6f} {vae_stats['std']:<12.6f} {econ_stats['std']:<12.6f}")
        print()
        print(f"{'Skewness':<20} {gt_stats['skewness']:<12.4f} {oracle_stats['skewness']:<12.4f} {vae_stats['skewness']:<12.4f} {econ_stats['skewness']:<12.4f}")
        print(f"{'Kurtosis':<20} {gt_stats['kurtosis']:<12.4f} {oracle_stats['kurtosis']:<12.4f} {vae_stats['kurtosis']:<12.4f} {econ_stats['kurtosis']:<12.4f}")
        print(f"{'Excess Kurtosis':<20} {gt_stats['excess_kurtosis']:<12.4f} {oracle_stats['excess_kurtosis']:<12.4f} {vae_stats['excess_kurtosis']:<12.4f} {econ_stats['excess_kurtosis']:<12.4f}")
        print()
        print(f"{'p01':<20} {gt_stats['p01']:<12.6f} {oracle_stats['p01']:<12.6f} {vae_stats['p01']:<12.6f} {econ_stats['p01']:<12.6f}")
        print(f"{'p25':<20} {gt_stats['p25']:<12.6f} {oracle_stats['p25']:<12.6f} {vae_stats['p25']:<12.6f} {econ_stats['p25']:<12.6f}")
        print(f"{'p50 (median)':<20} {gt_stats['p50']:<12.6f} {oracle_stats['p50']:<12.6f} {vae_stats['p50']:<12.6f} {econ_stats['p50']:<12.6f}")
        print(f"{'p75':<20} {gt_stats['p75']:<12.6f} {oracle_stats['p75']:<12.6f} {vae_stats['p75']:<12.6f} {econ_stats['p75']:<12.6f}")
        print(f"{'p99':<20} {gt_stats['p99']:<12.6f} {oracle_stats['p99']:<12.6f} {vae_stats['p99']:<12.6f} {econ_stats['p99']:<12.6f}")
        print()
        print(f"{'IQR (p75-p25)':<20} {gt_stats['iqr']:<12.6f} {oracle_stats['iqr']:<12.6f} {vae_stats['iqr']:<12.6f} {econ_stats['iqr']:<12.6f}")
        print(f"{'Range':<20} {gt_stats['range']:<12.6f} {oracle_stats['range']:<12.6f} {vae_stats['range']:<12.6f} {econ_stats['range']:<12.6f}")
        print()

        # Interpretation
        print("="*80)
        print("INTERPRETATION")
        print("="*80)
        print()

        print("Kurtosis interpretation (Pearson, normal=3.0):")
        for stat_dict in [gt_stats, oracle_stats, vae_stats, econ_stats]:
            k = stat_dict['kurtosis']
            if k > 4.0:
                shape = "VERY PEAKED (leptokurtic)"
            elif k > 3.5:
                shape = "Peaked (leptokurtic)"
            elif k >= 2.5:
                shape = "Normal-ish"
            elif k > 2.0:
                shape = "Flat (platykurtic)"
            else:
                shape = "VERY FLAT (platykurtic)"
            print(f"  {stat_dict['name']:<15}: {k:.2f} → {shape}")
        print()

        print("Skewness interpretation:")
        for stat_dict in [gt_stats, oracle_stats, vae_stats, econ_stats]:
            s = stat_dict['skewness']
            if abs(s) < 0.5:
                shape = "Approximately symmetric"
            elif s > 0:
                shape = f"Right-skewed (long right tail)"
            else:
                shape = f"Left-skewed (long left tail)"
            print(f"  {stat_dict['name']:<15}: {s:+.2f} → {shape}")
        print()

    # Create visualization if requested
    if create_plot:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Overlaid histograms
        ax1 = fig.add_subplot(gs[0, :])
        bins = 40
        alpha = 0.4
        ax1.hist(gt_vals, bins=bins, alpha=alpha, color='black', density=True, label='Ground Truth', edgecolor='black', linewidth=1.5)
        ax1.hist(oracle_vals, bins=bins, alpha=alpha, color='blue', density=True, label='Oracle', edgecolor='blue', linewidth=1.5)
        ax1.hist(vae_vals, bins=bins, alpha=alpha, color='red', density=True, label='VAE Prior', edgecolor='red', linewidth=1.5)
        ax1.hist(econ_vals, bins=bins, alpha=alpha, color='green', density=True, label='Econometric', edgecolor='green', linewidth=1.5)
        ax1.set_title(f'Overlaid Histograms\n{period.upper()} H{horizon}, Grid ({i},{j})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Implied Volatility')
        ax1.set_ylabel('Density')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Plot 2-5: Individual distributions
        for idx, (vals, stat_dict, color) in enumerate([
            (gt_vals, gt_stats, 'black'),
            (oracle_vals, oracle_stats, 'blue'),
            (vae_vals, vae_stats, 'red'),
            (econ_vals, econ_stats, 'green')
        ]):
            row = 1 + idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            ax.hist(vals, bins=bins, alpha=0.7, color=color, density=True, edgecolor=color, linewidth=1.5)
            ax.axvline(stat_dict['mean'], color='darkred', linestyle='--', linewidth=2, label=f"Mean={stat_dict['mean']:.4f}")
            ax.axvline(stat_dict['p50'], color='darkblue', linestyle=':', linewidth=2, label=f"Median={stat_dict['p50']:.4f}")

            # Add statistics text
            stats_text = (
                f"Std: {stat_dict['std']:.4f}\n"
                f"Skew: {stat_dict['skewness']:+.2f}\n"
                f"Kurt: {stat_dict['kurtosis']:.2f}\n"
                f"IQR: {stat_dict['iqr']:.4f}"
            )
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))

            ax.set_title(stat_dict['name'], fontsize=12, fontweight='bold', color=color)
            ax.set_xlabel('Implied Volatility')
            ax.set_ylabel('Density')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Distribution Shape Analysis: {period.upper()} H{horizon}\n'
                     f'Grid ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity',
                     fontsize=16, fontweight='bold')

        output_file = output_dir / f'{period}_h{horizon}_shape_analysis_grid_{i}_{j}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"✓ Saved: {output_file}")
        plt.close()

    # Summary
    if verbose:
        print("="*80)
        print("SUMMARY: SHAPE MISMATCH ANALYSIS")
        print("="*80)
        print()

        kurt_diff_oracle = oracle_stats['kurtosis'] - gt_stats['kurtosis']
        kurt_diff_vae = vae_stats['kurtosis'] - gt_stats['kurtosis']

        print(f"GT Kurtosis: {gt_stats['kurtosis']:.2f}")
        print(f"Oracle Kurtosis: {oracle_stats['kurtosis']:.2f} (diff: {kurt_diff_oracle:+.2f})")
        print(f"VAE Kurtosis: {vae_stats['kurtosis']:.2f} (diff: {kurt_diff_vae:+.2f})")
        print()

        if kurt_diff_oracle > 0.5 or kurt_diff_vae > 0.5:
            print("✓ MODELS ARE MORE PEAKED (higher kurtosis):")
            print(f"  - Oracle is {kurt_diff_oracle:+.2f} more peaked than GT")
            print(f"  - VAE is {kurt_diff_vae:+.2f} more peaked than GT")
            print()
            print("  Interpretation:")
            print("  - Models produce predictions clustered around mean (tall peak)")
            print("  - GT has more spread-out values (flatter distribution)")
            print("  - This is due to Gaussian latent prior constraining marginal shape")
            print("  - GT mixes crisis + normal periods → more dispersed")
        else:
            print("⚠ Kurtosis difference is small (<0.5)")

        print()

    return gt_stats, oracle_stats, vae_stats, econ_stats


def run_comprehensive_analysis(periods=['insample', 'oos'], horizons=[1, 7, 14, 30]):
    """Run comprehensive analysis across all periods, horizons, and grid points."""

    print("="*80)
    print("COMPREHENSIVE DISTRIBUTION SHAPE ANALYSIS")
    print("="*80)
    print()
    print(f"Periods: {periods}")
    print(f"Horizons: {horizons}")
    print(f"Grid: 5×5 (25 points)")
    print()

    # Storage for grid-level statistics
    all_results = []

    for period in periods:
        for horizon in horizons:
            print(f"\n{'='*80}")
            print(f"Processing: {period.upper()} H{horizon}")
            print(f"{'='*80}")

            # Load data
            print(f"Loading data...")
            gt, oracle_p50, vae_p50, econ_p50, n_days = load_data(period, horizon)
            print(f"  Loaded {n_days} days")

            # Analyze all 25 grid points
            print(f"Analyzing 25 grid points...")
            for i in range(5):
                for j in range(5):
                    # Run analysis (non-verbose for comprehensive mode)
                    gt_stats, oracle_stats, vae_stats, econ_stats = analyze_single_grid_point(
                        period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50,
                        verbose=False, create_plot=False
                    )

                    # Store results
                    result = {
                        'period': period,
                        'horizon': horizon,
                        'moneyness_idx': i,
                        'maturity_idx': j,
                        'moneyness': moneyness_labels[i],
                        'maturity': maturity_labels[j],
                        'n_days': n_days,
                        # GT stats
                        'gt_kurt': gt_stats['kurtosis'],
                        'gt_skew': gt_stats['skewness'],
                        'gt_std': gt_stats['std'],
                        'gt_iqr': gt_stats['iqr'],
                        # Oracle stats
                        'oracle_kurt': oracle_stats['kurtosis'],
                        'oracle_skew': oracle_stats['skewness'],
                        'oracle_std': oracle_stats['std'],
                        'oracle_iqr': oracle_stats['iqr'],
                        # VAE stats
                        'vae_kurt': vae_stats['kurtosis'],
                        'vae_skew': vae_stats['skewness'],
                        'vae_std': vae_stats['std'],
                        'vae_iqr': vae_stats['iqr'],
                        # Econometric stats
                        'econ_kurt': econ_stats['kurtosis'],
                        'econ_skew': econ_stats['skewness'],
                        'econ_std': econ_stats['std'],
                        'econ_iqr': econ_stats['iqr'],
                        # Differences
                        'oracle_kurt_diff': oracle_stats['kurtosis'] - gt_stats['kurtosis'],
                        'vae_kurt_diff': vae_stats['kurtosis'] - gt_stats['kurtosis'],
                        'econ_kurt_diff': econ_stats['kurtosis'] - gt_stats['kurtosis'],
                        'oracle_skew_diff': oracle_stats['skewness'] - gt_stats['skewness'],
                        'vae_skew_diff': vae_stats['skewness'] - gt_stats['skewness'],
                        'econ_skew_diff': econ_stats['skewness'] - gt_stats['skewness'],
                    }
                    all_results.append(result)

            print(f"  ✓ Completed {period.upper()} H{horizon}")

    # Convert to DataFrame and save
    if SAVE_CSV:
        df = pd.DataFrame(all_results)
        csv_file = output_dir / "shape_statistics_comprehensive.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Saved comprehensive statistics to: {csv_file}")

        # Print summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}\n")

        for period in periods:
            for horizon in horizons:
                subset = df[(df['period'] == period) & (df['horizon'] == horizon)]
                print(f"{period.upper()} H{horizon}:")
                print(f"  Oracle kurtosis diff: {subset['oracle_kurt_diff'].mean():+.3f} ± {subset['oracle_kurt_diff'].std():.3f}")
                print(f"  VAE kurtosis diff:    {subset['vae_kurt_diff'].mean():+.3f} ± {subset['vae_kurt_diff'].std():.3f}")
                print(f"  Econ kurtosis diff:   {subset['econ_kurt_diff'].mean():+.3f} ± {subset['econ_kurt_diff'].std():.3f}")
                print()

    return all_results


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    if RUN_ALL:
        # Run comprehensive analysis
        results = run_comprehensive_analysis(
            periods=['insample', 'oos'],
            horizons=[1, 7, 14, 30]
        )
        print("\n✓ Comprehensive analysis complete!")
        print(f"  Results saved to: {output_dir}")

    else:
        # Run single grid point analysis
        print("="*80)
        print(f"SINGLE GRID POINT ANALYSIS: {TEST_PERIOD.upper()} H{TEST_HORIZON}")
        print("="*80)
        print()

        # Load data
        gt, oracle_p50, vae_p50, econ_p50, n_days = load_data(TEST_PERIOD, TEST_HORIZON)
        print(f"Data loaded: {n_days} days\n")

        # Analyze
        analyze_single_grid_point(
            TEST_PERIOD, TEST_HORIZON, TEST_GRID_I, TEST_GRID_J,
            gt, oracle_p50, vae_p50, econ_p50,
            verbose=True, create_plot=True
        )
