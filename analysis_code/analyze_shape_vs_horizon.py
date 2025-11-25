"""
Analyze how distribution shape mismatch degrades with forecast horizon.

Creates plots showing kurtosis/skewness difference as a function of horizon (H1 → H30).

Outputs to: results/distribution_analysis/shape_vs_horizon/
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

csv_file = Path("results/distribution_analysis/shape_diagnostics/shape_statistics_comprehensive.csv")
output_dir = Path("results/distribution_analysis/shape_vs_horizon")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Load Data
# ============================================================================

print("="*80)
print("SHAPE VS HORIZON DEGRADATION ANALYSIS")
print("="*80)
print()

print(f"Loading data from: {csv_file}")
df = pd.read_csv(csv_file)
print(f"  Loaded {len(df)} rows\n")

# ============================================================================
# Aggregate by Horizon
# ============================================================================

def plot_horizon_degradation(period):
    """Plot kurtosis and skewness difference vs horizon."""

    subset = df[df['period'] == period]

    # Group by horizon and compute mean/std
    horizon_stats = subset.groupby('horizon').agg({
        'oracle_kurt_diff': ['mean', 'std'],
        'vae_kurt_diff': ['mean', 'std'],
        'econ_kurt_diff': ['mean', 'std'],
        'oracle_skew_diff': ['mean', 'std'],
        'vae_skew_diff': ['mean', 'std'],
        'econ_skew_diff': ['mean', 'std'],
    }).reset_index()

    horizons = horizon_stats['horizon'].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Distribution Shape Degradation vs Horizon: {period.upper()}\n'
                 f'Grid-Level Averages (±1 std)',
                 fontsize=16, fontweight='bold')

    # Panel 1: Kurtosis difference (all models)
    ax = axes[0, 0]

    oracle_kurt_mean = horizon_stats[('oracle_kurt_diff', 'mean')].values
    oracle_kurt_std = horizon_stats[('oracle_kurt_diff', 'std')].values
    vae_kurt_mean = horizon_stats[('vae_kurt_diff', 'mean')].values
    vae_kurt_std = horizon_stats[('vae_kurt_diff', 'std')].values
    econ_kurt_mean = horizon_stats[('econ_kurt_diff', 'mean')].values
    econ_kurt_std = horizon_stats[('econ_kurt_diff', 'std')].values

    ax.errorbar(horizons, oracle_kurt_mean, yerr=oracle_kurt_std, marker='o', markersize=8,
               linewidth=2, capsize=5, label='Oracle', color='blue')
    ax.errorbar(horizons, vae_kurt_mean, yerr=vae_kurt_std, marker='s', markersize=8,
               linewidth=2, capsize=5, label='VAE Prior', color='red')
    ax.errorbar(horizons, econ_kurt_mean, yerr=econ_kurt_std, marker='^', markersize=8,
               linewidth=2, capsize=5, label='Econometric', color='green')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Perfect match')
    ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
    ax.set_ylabel('Kurtosis Difference (Model - GT)', fontsize=12)
    ax.set_title('Kurtosis Degradation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)

    # Panel 2: Kurtosis difference (Oracle vs VAE only, zoomed)
    ax = axes[0, 1]

    ax.errorbar(horizons, oracle_kurt_mean, yerr=oracle_kurt_std, marker='o', markersize=10,
               linewidth=2.5, capsize=5, label='Oracle', color='blue')
    ax.errorbar(horizons, vae_kurt_mean, yerr=vae_kurt_std, marker='s', markersize=10,
               linewidth=2.5, capsize=5, label='VAE Prior', color='red')

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect match')
    ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
    ax.set_ylabel('Kurtosis Difference (Model - GT)', fontsize=12)
    ax.set_title('Kurtosis Degradation (Oracle & VAE only)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)

    # Panel 3: Skewness difference (all models)
    ax = axes[1, 0]

    oracle_skew_mean = horizon_stats[('oracle_skew_diff', 'mean')].values
    oracle_skew_std = horizon_stats[('oracle_skew_diff', 'std')].values
    vae_skew_mean = horizon_stats[('vae_skew_diff', 'mean')].values
    vae_skew_std = horizon_stats[('vae_skew_diff', 'std')].values
    econ_skew_mean = horizon_stats[('econ_skew_diff', 'mean')].values
    econ_skew_std = horizon_stats[('econ_skew_diff', 'std')].values

    ax.errorbar(horizons, oracle_skew_mean, yerr=oracle_skew_std, marker='o', markersize=8,
               linewidth=2, capsize=5, label='Oracle', color='blue')
    ax.errorbar(horizons, vae_skew_mean, yerr=vae_skew_std, marker='s', markersize=8,
               linewidth=2, capsize=5, label='VAE Prior', color='red')
    ax.errorbar(horizons, econ_skew_mean, yerr=econ_skew_std, marker='^', markersize=8,
               linewidth=2, capsize=5, label='Econometric', color='green')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Perfect match')
    ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
    ax.set_ylabel('Skewness Difference (Model - GT)', fontsize=12)
    ax.set_title('Skewness Degradation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_text = f"DEGRADATION SUMMARY: {period.upper()}\n" + "="*50 + "\n\n"
    table_text += "Kurtosis Difference (Model - GT):\n"
    table_text += "-"*50 + "\n"
    table_text += f"{'Horizon':<10} {'Oracle':>12} {'VAE':>12} {'Econ':>12}\n"
    table_text += "-"*50 + "\n"

    for idx, h in enumerate(horizons):
        table_text += f"{'H'+str(int(h)):<10} "
        table_text += f"{oracle_kurt_mean[idx]:>12.2f} "
        table_text += f"{vae_kurt_mean[idx]:>12.2f} "
        table_text += f"{econ_kurt_mean[idx]:>12.2f}\n"

    table_text += "\n\nSkewness Difference (Model - GT):\n"
    table_text += "-"*50 + "\n"
    table_text += f"{'Horizon':<10} {'Oracle':>12} {'VAE':>12} {'Econ':>12}\n"
    table_text += "-"*50 + "\n"

    for idx, h in enumerate(horizons):
        table_text += f"{'H'+str(int(h)):<10} "
        table_text += f"{oracle_skew_mean[idx]:>12.3f} "
        table_text += f"{vae_skew_mean[idx]:>12.3f} "
        table_text += f"{econ_skew_mean[idx]:>12.3f}\n"

    # Add degradation rate
    if period == 'insample':
        kurt_slope = (vae_kurt_mean[-1] - vae_kurt_mean[0]) / (horizons[-1] - horizons[0])
        table_text += f"\n\nVAE Kurtosis Degradation Rate:\n"
        table_text += f"  {kurt_slope:+.3f} per day\n"
        table_text += f"  {kurt_slope * (horizons[-1] - horizons[0]):+.2f} total (H1 → H30)"

    ax.text(0.1, 0.9, table_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = output_dir / f'{period}_shape_vs_horizon.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    return horizon_stats


def plot_intra_grid_variability(period):
    """Plot how grid-level variability changes with horizon."""

    subset = df[df['period'] == period]

    # Group by horizon
    horizons = [1, 7, 14, 30]

    oracle_kurt_std_by_horizon = []
    vae_kurt_std_by_horizon = []

    for h in horizons:
        h_subset = subset[subset['horizon'] == h]
        oracle_kurt_std_by_horizon.append(h_subset['oracle_kurt_diff'].std())
        vae_kurt_std_by_horizon.append(h_subset['vae_kurt_diff'].std())

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Grid-Level Variability vs Horizon: {period.upper()}\n'
                 f'Standard Deviation of Kurtosis Difference Across 25 Grid Points',
                 fontsize=14, fontweight='bold')

    ax.plot(horizons, oracle_kurt_std_by_horizon, marker='o', markersize=10, linewidth=2.5,
           color='blue', label='Oracle')
    ax.plot(horizons, vae_kurt_std_by_horizon, marker='s', markersize=10, linewidth=2.5,
           color='red', label='VAE Prior')

    ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
    ax.set_ylabel('Std Dev of Kurtosis Difference', fontsize=12)
    ax.set_title('Spatial Heterogeneity in Shape Mismatch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)

    # Add interpretation
    interp_text = (
        "Interpretation: Higher std dev = more spatial heterogeneity\n"
        "(some grid points have good shape matching, others poor)"
    )
    ax.text(0.5, 0.02, interp_text, transform=ax.transAxes, fontsize=9,
           ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    output_file = output_dir / f'{period}_grid_variability_vs_horizon.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_comparison_plot():
    """Compare in-sample vs OOS horizon degradation."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Shape Mismatch: In-Sample vs Out-of-Sample Comparison\n'
                 'VAE Prior Kurtosis Difference',
                 fontsize=16, fontweight='bold')

    for period_idx, period in enumerate(['insample', 'oos']):
        ax = axes[period_idx]
        subset = df[df['period'] == period]

        horizon_stats = subset.groupby('horizon').agg({
            'vae_kurt_diff': ['mean', 'std'],
        }).reset_index()

        horizons = horizon_stats['horizon'].values
        vae_kurt_mean = horizon_stats[('vae_kurt_diff', 'mean')].values
        vae_kurt_std = horizon_stats[('vae_kurt_diff', 'std')].values

        ax.errorbar(horizons, vae_kurt_mean, yerr=vae_kurt_std, marker='s', markersize=10,
                   linewidth=3, capsize=5, label='VAE Prior', color='red')
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Perfect match')

        ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
        ax.set_ylabel('Kurtosis Difference (VAE - GT)', fontsize=12)
        ax.set_title(f'{period.upper()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons)

        # Add pattern annotation
        if period == 'insample':
            pattern = "Pattern: Positive & Increasing\n(Models more peaked, worsens with horizon)"
        else:
            pattern = "Pattern: Negative & Decreasing\n(Models flatter, improves with horizon)"

        ax.text(0.95, 0.05, pattern, transform=ax.transAxes, fontsize=10,
               ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_file = output_dir / 'insample_vs_oos_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    periods = ['insample', 'oos']

    print("Generating horizon degradation plots...\n")

    for period in periods:
        print(f"Processing: {period.upper()}")
        plot_horizon_degradation(period)
        plot_intra_grid_variability(period)
        print()

    print("Creating in-sample vs OOS comparison...")
    create_comparison_plot()
    print()

    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated plots in: {output_dir}")
    print()
    print("Files generated:")
    print("  ✓ insample_shape_vs_horizon.png")
    print("  ✓ insample_grid_variability_vs_horizon.png")
    print("  ✓ oos_shape_vs_horizon.png")
    print("  ✓ oos_grid_variability_vs_horizon.png")
    print("  ✓ insample_vs_oos_comparison.png")
