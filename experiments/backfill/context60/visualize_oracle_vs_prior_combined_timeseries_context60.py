"""
Visualize Oracle vs Prior CI Width - Context60 Model

Generates 9-panel timeseries plot showing oracle vs prior CI widths for ATM 6M point
across all horizons (6 TF + 2 AR + 1 vol reference).

Output: oracle_vs_prior_combined_timeseries_with_vol_KS1.00_6M.png

Usage:
    python experiments/backfill/context60/visualize_oracle_vs_prior_combined_timeseries_context60.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path


def plot_horizon_panel(ax, oracle_ci, prior_ci, dates, horizon, is_ar=False):
    """
    Plot single horizon panel with oracle vs prior comparison.

    Args:
        ax: matplotlib axis
        oracle_ci: dict with keys 'min', 'max', 'avg'
        prior_ci: dict with keys 'min', 'max', 'avg'
        dates: (n_dates,) datetime array
        horizon: int (1,7,14,30,60,90,180,270)
        is_ar: bool - True if autoregressive horizon
    """
    # Plot oracle (blue)
    ax.fill_between(dates, oracle_ci['min'], oracle_ci['max'],
                     alpha=0.2, color='blue', label='Oracle range')
    ax.plot(dates, oracle_ci['avg'], color='blue', linewidth=1.5,
            label='Oracle avg', zorder=3)

    # Plot prior (orange)
    ax.fill_between(dates, prior_ci['min'], prior_ci['max'],
                     alpha=0.2, color='orange', label='Prior range')
    ax.plot(dates, prior_ci['avg'], color='orange', linewidth=1.5,
            label='Prior avg', zorder=3)

    # Formatting
    horizon_type = 'AR' if is_ar else 'TF'
    ax.set_title(f'H={horizon} days ({horizon_type})', fontsize=11, fontweight='bold')
    ax.set_ylabel('CI Width (IV points)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Statistics box
    oracle_mean = np.mean(oracle_ci['avg'])
    prior_mean = np.mean(prior_ci['avg'])
    ratio = prior_mean / oracle_mean
    pct_increase = (ratio - 1) * 100

    stats_text = f"Oracle: {oracle_mean:.4f}\nPrior: {prior_mean:.4f}\nRatio: {ratio:.2f}× (+{pct_increase:.1f}%)"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def extract_atm_6m_timeseries(data, periods, horizons, atm_6m=(2, 2)):
    """
    Extract ATM 6M timeseries from statistics data.

    Args:
        data: loaded NPZ file
        periods: list of period names
        horizons: list of horizons
        atm_6m: tuple (m_idx, t_idx) - grid indices for ATM 6M

    Returns:
        combined_timeseries: dict[horizon] -> dict['dates', 'min', 'max', 'avg']
    """
    combined_timeseries = {}

    for h in horizons:
        dates_list = []
        min_ci_list = []
        max_ci_list = []
        avg_ci_list = []

        for period in periods:
            # Load statistics
            indices = data[f'{period}_h{h}_indices']
            min_ci_grid = data[f'{period}_h{h}_min_ci_width']  # (n_dates, 5, 5)
            max_ci_grid = data[f'{period}_h{h}_max_ci_width']
            avg_ci_grid = data[f'{period}_h{h}_avg_ci_width']

            # Extract ATM 6M point
            min_ci = min_ci_grid[:, atm_6m[0], atm_6m[1]]  # (n_dates,)
            max_ci = max_ci_grid[:, atm_6m[0], atm_6m[1]]
            avg_ci = avg_ci_grid[:, atm_6m[0], atm_6m[1]]

            # Store (will convert to dates later)
            dates_list.append(indices)
            min_ci_list.append(min_ci)
            max_ci_list.append(max_ci)
            avg_ci_list.append(avg_ci)

        # Concatenate periods
        indices_combined = np.concatenate(dates_list)
        min_ci_combined = np.concatenate(min_ci_list)
        max_ci_combined = np.concatenate(max_ci_list)
        avg_ci_combined = np.concatenate(avg_ci_list)

        combined_timeseries[h] = {
            'indices': indices_combined,
            'min': min_ci_combined,
            'max': max_ci_combined,
            'avg': avg_ci_combined
        }

    return combined_timeseries


def convert_indices_to_dates(timeseries, gt_dates):
    """
    Convert indices to dates and sort by date.

    Args:
        timeseries: dict[horizon] -> dict with 'indices' key
        gt_dates: pandas datetime series

    Returns:
        timeseries_with_dates: dict[horizon] -> dict['dates', 'min', 'max', 'avg']
    """

    timeseries_with_dates = {}

    for h, ts in timeseries.items():
        # Convert indices to dates
        dates = gt_dates[ts['indices']]

        # Sort by date
        sort_idx = np.argsort(dates)

        timeseries_with_dates[h] = {
            'dates': dates.values[sort_idx],
            'min': ts['min'][sort_idx],
            'max': ts['max'][sort_idx],
            'avg': ts['avg'][sort_idx]
        }

    return timeseries_with_dates


def main():
    """Main visualization pipeline."""
    print("="*80)
    print("CONTEXT60 ORACLE VS PRIOR CI WIDTH VISUALIZATION")
    print("="*80)

    # Configuration
    periods = ['insample', 'gap', 'oos']  # Crisis is subset of insample
    tf_horizons = [1, 7, 14, 30, 60, 90]
    ar_horizons = [180, 270]
    all_horizons = tf_horizons + ar_horizons
    atm_6m = (2, 2)  # Middle of 5×5 grid

    print(f"\nConfiguration:")
    print(f"  Periods: {periods}")
    print(f"  TF Horizons: {tf_horizons}")
    print(f"  AR Horizons: {ar_horizons}")
    print(f"  Grid point: ATM 6M (index {atm_6m})")

    # Load data
    print(f"\nLoading data...")
    oracle_data = np.load("results/context60_baseline/analysis/oracle/sequence_ci_width_stats.npz")
    prior_data = np.load("results/context60_baseline/analysis/prior/sequence_ci_width_stats.npz")
    ground_truth = np.load("data/vol_surface_with_ret.npz")

    # Load dates from parquet file
    dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    gt_dates = pd.to_datetime(dates_df["date"].values)

    print(f"  ✓ Oracle statistics: {len(oracle_data.files)} keys")
    print(f"  ✓ Prior statistics: {len(prior_data.files)} keys")
    print(f"  ✓ Ground truth data: {ground_truth['surface'].shape}")
    print(f"  ✓ Ground truth dates: {len(gt_dates)} days")

    # Extract ATM 6M timeseries
    print(f"\nExtracting ATM 6M timeseries...")
    oracle_ts = extract_atm_6m_timeseries(oracle_data, periods, all_horizons, atm_6m)
    prior_ts = extract_atm_6m_timeseries(prior_data, periods, all_horizons, atm_6m)
    print(f"  ✓ Extracted {len(oracle_ts)} horizon timeseries per mode")

    # Convert indices to dates
    print(f"\nConverting indices to dates...")
    oracle_ts = convert_indices_to_dates(oracle_ts, gt_dates)
    prior_ts = convert_indices_to_dates(prior_ts, gt_dates)
    print(f"  ✓ Date conversion complete")

    # Create 9-panel plot
    print(f"\nCreating 9-panel plot...")
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    fig.suptitle('Oracle vs Prior CI Width - Context60 Model (ATM 6M)',
                 fontsize=16, fontweight='bold')

    # Row 1: TF Short Horizons (H=1, 7, 14)
    for i, h in enumerate([1, 7, 14]):
        ax = axes[0, i]
        plot_horizon_panel(ax, oracle_ts[h], prior_ts[h],
                          oracle_ts[h]['dates'], h, is_ar=False)

    # Row 2: TF Medium Horizons (H=30, 60, 90)
    for i, h in enumerate([30, 60, 90]):
        ax = axes[1, i]
        plot_horizon_panel(ax, oracle_ts[h], prior_ts[h],
                          oracle_ts[h]['dates'], h, is_ar=False)

    # Row 3: AR Long Horizons (H=180, 270) + Vol Reference
    for i, h in enumerate([180, 270]):
        ax = axes[2, i]
        plot_horizon_panel(ax, oracle_ts[h], prior_ts[h],
                          oracle_ts[h]['dates'], h, is_ar=True)

    # Row 3, Col 3: ATM Volatility Reference
    print(f"  Adding volatility reference panel...")
    ax = axes[2, 2]
    atm_vol = ground_truth['surface'][:, 2, 2]  # ATM 6M from full dataset
    ax.plot(gt_dates, atm_vol, color='black', linewidth=1, alpha=0.7)
    ax.set_title('ATM 6M Volatility (Ground Truth)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Implied Volatility', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.3 threshold')
    ax.legend(loc='upper right', fontsize=8)

    # Format x-axis for all panels
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_dir = "results/context60_baseline/visualizations/comparison/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "oracle_vs_prior_combined_timeseries_with_vol_KS1.00_6M.png")

    print(f"\nSaving plot...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024**2:.1f} MB")

    plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}\n")

    print("Output:")
    print(f"  {output_file}")
    print(f"\nPlot details:")
    print(f"  - Layout: 3×3 grid (9 panels)")
    print(f"  - Horizons: 6 TF + 2 AR + 1 vol reference")
    print(f"  - Size: 20×12 inches, 300 DPI")
    print(f"  - Grid point: ATM 6M (K/S=1.00, 6-month)")
    print(f"\nNext step:")
    print(f"  Run comparison: python experiments/backfill/context60/compare_oracle_vs_prior_ci_context60.py")
    print()


if __name__ == "__main__":
    main()
