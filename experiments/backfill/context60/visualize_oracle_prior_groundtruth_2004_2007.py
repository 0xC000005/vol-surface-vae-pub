"""
Oracle vs Prior with Ground Truth Overlay Visualization (2004-2007)

Creates three comprehensive visualizations showing how oracle and prior sampling modes
compare against actual ground truth ATM 6M implied volatility during 2004-2007:
1. Primary 2-panel overlay: Ground truth + Oracle + Prior predictions
2. 4-panel analysis: Overlays + prediction error timeseries
3. CI coverage analysis: Binary indicators showing when ground truth falls within CIs

Usage:
    python experiments/backfill/context60/visualize_oracle_prior_groundtruth_2004_2007.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# Configuration
START_DATE = "2004-01-01"
END_DATE = "2007-12-31"


def plot_overlay_panel(ax, df_h, horizon):
    """
    Plot ground truth with oracle and prior predictions.

    Args:
        ax: matplotlib axis
        df_h: DataFrame with columns for date, ground_truth, oracle/prior quantiles
        horizon: int (60 or 90)
    """
    dates = df_h['date'].values
    gt = df_h['ground_truth'].values

    # Oracle
    oracle_p50 = df_h['oracle_p50'].values
    oracle_p05 = df_h['oracle_p05'].values
    oracle_p95 = df_h['oracle_p95'].values

    # Prior
    prior_p50 = df_h['prior_p50'].values
    prior_p05 = df_h['prior_p05'].values
    prior_p95 = df_h['prior_p95'].values

    # Plot CI bands first (background)
    ax.fill_between(dates, oracle_p05, oracle_p95,
                     alpha=0.2, color='blue', label='Oracle 90% CI', zorder=3)
    ax.fill_between(dates, prior_p05, prior_p95,
                     alpha=0.2, color='orange', label='Prior 90% CI', zorder=2)

    # Plot medians
    ax.plot(dates, oracle_p50, color='blue', linestyle='--',
            linewidth=1.5, label='Oracle p50', zorder=5)
    ax.plot(dates, prior_p50, color='orange', linestyle='--',
            linewidth=1.5, label='Prior p50', zorder=4)

    # Plot ground truth on top
    ax.plot(dates, gt, color='black', linewidth=2,
            label='Ground Truth', zorder=10)

    # Styling
    ax.set_ylabel('ATM 6M IV', fontsize=11)
    ax.set_title(f'H={horizon} Days: Predictions vs Ground Truth (2004-2007)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # Statistics box
    oracle_rmse = np.sqrt(np.mean((oracle_p50 - gt)**2))
    prior_rmse = np.sqrt(np.mean((prior_p50 - gt)**2))
    oracle_coverage = ((gt >= oracle_p05) & (gt <= oracle_p95)).mean() * 100
    prior_coverage = ((gt >= prior_p05) & (gt <= prior_p95)).mean() * 100

    stats_text = (f"Oracle RMSE: {oracle_rmse:.5f}\n"
                 f"Prior RMSE: {prior_rmse:.5f}\n"
                 f"Oracle CI Coverage: {oracle_coverage:.1f}%\n"
                 f"Prior CI Coverage: {prior_coverage:.1f}%")

    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_error_panel(ax, df_h, horizon):
    """
    Plot prediction errors over time.

    Args:
        ax: matplotlib axis
        df_h: DataFrame with oracle_error and prior_error columns
        horizon: int (60 or 90)
    """
    dates = df_h['date'].values
    oracle_err = df_h['oracle_error'].values
    prior_err = df_h['prior_error'].values

    # Plot errors
    ax.plot(dates, oracle_err, color='blue', linewidth=1,
            label='Oracle error', alpha=0.7)
    ax.plot(dates, prior_err, color='orange', linewidth=1,
            label='Prior error', alpha=0.7)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Styling
    ax.set_ylabel('Prediction Error (IV points)', fontsize=10)
    ax.set_title(f'H={horizon} Days: Prediction Errors',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # Statistics
    oracle_bias = np.mean(oracle_err)
    prior_bias = np.mean(prior_err)
    oracle_std = np.std(oracle_err)
    prior_std = np.std(prior_err)

    stats_text = (f"Oracle bias: {oracle_bias:.5f} (±{oracle_std:.5f})\n"
                 f"Prior bias: {prior_bias:.5f} (±{prior_std:.5f})")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_coverage_panel(ax, df_h, horizon):
    """
    Plot binary CI coverage indicators.

    Args:
        ax: matplotlib axis
        df_h: DataFrame with ground_truth and quantile columns
        horizon: int (60 or 90)
    """
    dates = df_h['date'].values
    gt = df_h['ground_truth'].values

    # Oracle coverage
    oracle_covered = ((gt >= df_h['oracle_p05']) &
                     (gt <= df_h['oracle_p95'])).values.astype(float)

    # Prior coverage
    prior_covered = ((gt >= df_h['prior_p05']) &
                    (gt <= df_h['prior_p95'])).values.astype(float)

    # Offset prior slightly for visibility
    prior_covered_offset = prior_covered - 0.05

    # Plot
    ax.scatter(dates, oracle_covered, color='blue', s=10, alpha=0.6,
              label=f'Oracle ({oracle_covered.mean()*100:.1f}% coverage)')
    ax.scatter(dates, prior_covered_offset, color='orange', s=10, alpha=0.6,
              label=f'Prior ({prior_covered.mean()*100:.1f}% coverage)')

    # Reference line at target
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5,
              label='Target: 90%')

    # Styling
    ax.set_ylabel('Coverage (1=covered, 0=violation)', fontsize=10)
    ax.set_title(f'H={horizon} Days: CI Coverage of Ground Truth',
                 fontsize=11, fontweight='bold')
    ax.set_ylim([-0.2, 1.2])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=9)


def main():
    """Main execution pipeline"""
    print("="*80)
    print("ORACLE VS PRIOR WITH GROUND TRUTH OVERLAY: 2004-2007")
    print("Context60 Model - ATM 6M Point")
    print("="*80)

    # Load data
    print("\nLoading data...")
    csv_file = "results/context60_baseline/analysis/comparison/predicted_values_divergence_2004_2008.csv"
    df = pd.read_csv(csv_file, parse_dates=['date'])
    print(f"  Loaded {len(df)} rows from CSV")

    # Filter to 2004-2007
    df_2004_2007 = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)].copy()
    print(f"  Filtered to {len(df_2004_2007)} rows (2004-2007)")

    # Split by horizon
    df_h60 = df_2004_2007[df_2004_2007['horizon'] == 60].sort_values('date')
    df_h90 = df_2004_2007[df_2004_2007['horizon'] == 90].sort_values('date')
    print(f"  H=60: {len(df_h60)} days")
    print(f"  H=90: {len(df_h90)} days")

    # Create output directory
    output_dir = Path("results/context60_baseline/visualizations/2004_2007_confirmation")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Plot 1: Primary 2-panel overlay
    print("\nCreating primary overlay plot...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    plot_overlay_panel(axes[0], df_h60, 60)
    plot_overlay_panel(axes[1], df_h90, 90)
    axes[-1].set_xlabel('Date', fontsize=11)
    plt.tight_layout()
    output_file = output_dir / "oracle_prior_groundtruth_overlay_2004_2007.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")

    # Plot 2: 4-panel analysis
    print("\nCreating 4-panel analysis plot...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plot_overlay_panel(axes[0, 0], df_h60, 60)
    plot_error_panel(axes[0, 1], df_h60, 60)
    plot_overlay_panel(axes[1, 0], df_h90, 90)
    plot_error_panel(axes[1, 1], df_h90, 90)
    axes[1, 0].set_xlabel('Date', fontsize=11)
    axes[1, 1].set_xlabel('Date', fontsize=11)
    plt.tight_layout()
    output_file = output_dir / "oracle_prior_groundtruth_4panel_2004_2007.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")

    # Plot 3: CI coverage
    print("\nCreating CI coverage plot...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_coverage_panel(axes[0], df_h60, 60)
    plot_coverage_panel(axes[1], df_h90, 90)
    axes[-1].set_xlabel('Date', fontsize=11)
    plt.tight_layout()
    output_file = output_dir / "oracle_prior_ci_coverage_2004_2007.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for horizon, df_h in [(60, df_h60), (90, df_h90)]:
        print(f"\nHorizon {horizon} days:")
        print("-" * 40)

        gt = df_h['ground_truth'].values
        oracle_p50 = df_h['oracle_p50'].values
        prior_p50 = df_h['prior_p50'].values

        # Prediction accuracy
        oracle_rmse = np.sqrt(np.mean((oracle_p50 - gt)**2))
        prior_rmse = np.sqrt(np.mean((prior_p50 - gt)**2))
        oracle_bias = np.mean(oracle_p50 - gt)
        prior_bias = np.mean(prior_p50 - gt)

        # CI coverage
        oracle_coverage = ((gt >= df_h['oracle_p05']) &
                          (gt <= df_h['oracle_p95'])).mean() * 100
        prior_coverage = ((gt >= df_h['prior_p05']) &
                         (gt <= df_h['prior_p95'])).mean() * 100

        # Ground truth statistics
        gt_mean = gt.mean()
        gt_std = gt.std()
        gt_min = gt.min()
        gt_max = gt.max()

        print(f"  Ground Truth:")
        print(f"    Mean: {gt_mean:.5f} ± {gt_std:.5f}")
        print(f"    Range: [{gt_min:.5f}, {gt_max:.5f}]")
        print(f"\n  Prediction Accuracy:")
        print(f"    Oracle RMSE: {oracle_rmse:.5f}")
        print(f"    Prior RMSE: {prior_rmse:.5f}")
        print(f"    Oracle Bias: {oracle_bias:.5f}")
        print(f"    Prior Bias: {prior_bias:.5f}")
        print(f"\n  CI Coverage (Target: 90%):")
        print(f"    Oracle: {oracle_coverage:.1f}%")
        print(f"    Prior: {prior_coverage:.1f}%")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nVisualizations saved to: {output_dir}")
    print("\nFiles created:")
    print("  1. oracle_prior_groundtruth_overlay_2004_2007.png")
    print("  2. oracle_prior_groundtruth_4panel_2004_2007.png")
    print("  3. oracle_prior_ci_coverage_2004_2007.png")
    print("="*80)


if __name__ == "__main__":
    main()
