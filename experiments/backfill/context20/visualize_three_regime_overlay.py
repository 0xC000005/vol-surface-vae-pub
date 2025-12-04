"""
Visualize Three-Regime CI Width Overlay - CVAE Conditional Generation Demo

Creates a single overlaid plot comparing three market regimes from prior sampling
to demonstrate how the CVAE produces different CI widths under different market conditions.

**Purpose**: Demonstrate the anomalous "high CI despite low vol" regime discovered in investigation

**Three regimes compared:**
- 2007-03-28 (normal/calm period, narrow CI)
- 2007-10-09 (low-vol high-CI, anomalous - pre-crisis)
- 2008-10-30 (high-vol high-CI, crisis peak)

Usage:
    python experiments/backfill/context20/visualize_three_regime_overlay.py

Output:
    results/vae_baseline/visualizations/comparison/three_regime_overlay_prior.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def load_ci_width_stats(sampling_mode):
    """
    Load CI width statistics from sequence_ci_width_stats.npz

    Args:
        sampling_mode: 'oracle' or 'prior'

    Returns:
        dict with CI width data, dates, and indices for H=30 insample period
    """
    stats_file = f"results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz"
    print(f"Loading CI width stats: {stats_file}")

    data = np.load(stats_file, allow_pickle=True)

    # Extract H=30 insample data
    # avg_ci_width has shape (n_seq, 5, 5) - extract ATM 6M grid point (2, 2)
    grid_row, grid_col = 2, 2  # ATM 6M
    avg_ci_width_full = data['insample_h30_avg_ci_width']
    avg_ci_width_atm = avg_ci_width_full[:, grid_row, grid_col]

    result = {
        'avg_ci_width': avg_ci_width_atm,
        'dates': data['insample_h30_dates'],
        'indices': data['insample_h30_indices'],
        'atm_vol': data['insample_h30_atm_vol']
    }

    print(f"  Loaded {len(result['avg_ci_width'])} sequences for H=30")
    print(f"  Extracted ATM 6M CI width (grid point {grid_row}, {grid_col})")

    return result


def load_tf_predictions(sampling_mode):
    """
    Load teacher forcing predictions for H=30

    Args:
        sampling_mode: 'oracle' or 'prior'

    Returns:
        npz data with surfaces (n_seq, 30, 3, 5, 5) and indices
    """
    pred_file = f"results/vae_baseline/predictions/autoregressive/{sampling_mode}/vae_tf_insample_h30.npz"
    print(f"Loading TF predictions: {pred_file}")

    data = np.load(pred_file)

    print(f"  Surfaces shape: {data['surfaces'].shape}")
    print(f"  Indices shape: {data['indices'].shape}")

    return data


def load_ground_truth():
    """
    Load ground truth volatility surfaces and dates

    Returns:
        tuple of (surfaces, dates, ex_data)
    """
    print("Loading ground truth data...")

    # Load surfaces
    data_file = "data/vol_surface_with_ret.npz"
    data = np.load(data_file)
    surfaces = data["surface"]

    # Load dates
    dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    dates = pd.to_datetime(dates_df["date"].values)

    # Load extra features
    if "ret" in data and "skews" in data and "slopes" in data:
        ret = data["ret"]
        skews = data["skews"]
        slopes = data["slopes"]
        ex_data = np.stack([ret, skews, slopes], axis=1)
    else:
        ex_data = None

    print(f"  Surfaces shape: {surfaces.shape}")
    print(f"  Dates shape: {dates.shape}")

    return surfaces, dates, ex_data


def find_date_sequence(target_date_str, dates_array):
    """
    Find the sequence index for a specific date

    Args:
        target_date_str: 'YYYY-MM-DD' format
        dates_array: Array of dates from CI stats

    Returns:
        int: Sequence index matching the target date
    """
    target_date = pd.Timestamp(target_date_str)

    for idx, date_item in enumerate(dates_array):
        # Handle various date formats
        date_str = str(date_item) if not isinstance(date_item, str) else date_item
        if pd.Timestamp(date_str) == target_date:
            return idx

    raise ValueError(f"Date {target_date_str} not found in dates array")


def extract_context_forecast_truth(sequence_idx,
                                     tf_predictions,
                                     indices_array,
                                     ground_truth_surfaces,
                                     dates):
    """
    Extract aligned data for one specific date

    Args:
        sequence_idx: Index in tf_predictions arrays
        tf_predictions: Teacher forcing predictions npz data
        indices_array: Mapping from sequence to data indices
        ground_truth_surfaces: Full ground truth (N, 5, 5)
        dates: Full date array

    Returns:
        dict with context_dates, forecast_dates, truth values, and predictions
    """
    context_len = 20
    horizon = 30
    grid_row, grid_col = 2, 2  # ATM 6M

    # Map sequence index to data index
    data_start_idx = indices_array[sequence_idx]

    # Extract context (20 days before forecast start)
    context_start = data_start_idx - context_len
    context_truth = ground_truth_surfaces[context_start:data_start_idx, grid_row, grid_col]
    context_dates = dates[context_start:data_start_idx]

    # Extract forecast ground truth (30 days starting at data_start_idx)
    forecast_truth = ground_truth_surfaces[data_start_idx:data_start_idx+horizon, grid_row, grid_col]
    forecast_dates = dates[data_start_idx:data_start_idx+horizon]

    # Extract predictions (quantiles: 0=p05, 1=p50, 2=p95)
    forecast_p05 = tf_predictions['surfaces'][sequence_idx, :, 0, grid_row, grid_col]
    forecast_p50 = tf_predictions['surfaces'][sequence_idx, :, 1, grid_row, grid_col]
    forecast_p95 = tf_predictions['surfaces'][sequence_idx, :, 2, grid_row, grid_col]

    return {
        'context_dates': context_dates,
        'forecast_dates': forecast_dates,
        'context_truth': context_truth,
        'forecast_truth': forecast_truth,
        'forecast_p05': forecast_p05,
        'forecast_p50': forecast_p50,
        'forecast_p95': forecast_p95,
        'data_start_idx': data_start_idx
    }


def create_three_regime_overlay(normal_data, low_vol_data, high_vol_data,
                                 normal_date, low_vol_date, high_vol_date,
                                 output_path):
    """
    Create overlaid plot comparing three regimes: normal, low-vol high-CI, high-vol high-CI

    Args:
        normal_data: dict with context, forecast, truth for normal period
        low_vol_data: dict with context, forecast, truth for low-vol high-CI period
        high_vol_data: dict with context, forecast, truth for high-vol high-CI period
        normal_date: Date string for normal period
        low_vol_date: Date string for low-vol high-CI period
        high_vol_date: Date string for high-vol high-CI period
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    # Create relative day numbers for x-axis
    context_days = np.arange(0, 20)
    forecast_days = np.arange(20, 50)

    # =========================================================================
    # Plot normal/calm period (green)
    # =========================================================================

    # Context (solid green line)
    ax.plot(context_days, normal_data['context_truth'],
            color='green', linewidth=2.5, label=f'Normal Context ({normal_date})', zorder=6)

    # Ground truth continuation in forecast region (green dotted)
    ax.plot(forecast_days, normal_data['forecast_truth'],
            color='green', linewidth=2, linestyle=':', alpha=0.6, zorder=5,
            label='Normal Ground Truth')

    # Forecast CI band (light green fill)
    ax.fill_between(forecast_days, normal_data['forecast_p05'], normal_data['forecast_p95'],
                    alpha=0.25, color='green', label='Normal 90% CI', zorder=1)

    # Forecast median (green dashed)
    ax.plot(forecast_days, normal_data['forecast_p50'],
            color='green', linewidth=2, linestyle='--', label='Normal Forecast (p50)', zorder=4)

    # =========================================================================
    # Plot low-vol high-CI period (orange) - THE ANOMALOUS REGIME
    # =========================================================================

    # Context (solid orange line)
    ax.plot(context_days, low_vol_data['context_truth'],
            color='orange', linewidth=2.5, label=f'Low-Vol High-CI Context ({low_vol_date})', zorder=6)

    # Ground truth continuation in forecast region (orange dotted)
    ax.plot(forecast_days, low_vol_data['forecast_truth'],
            color='orange', linewidth=2, linestyle=':', alpha=0.6, zorder=5,
            label='Low-Vol High-CI Ground Truth')

    # Forecast CI band (light orange fill)
    ax.fill_between(forecast_days, low_vol_data['forecast_p05'], low_vol_data['forecast_p95'],
                    alpha=0.25, color='orange', label='Low-Vol High-CI 90% CI', zorder=1)

    # Forecast median (orange dashed)
    ax.plot(forecast_days, low_vol_data['forecast_p50'],
            color='orange', linewidth=2, linestyle='--', label='Low-Vol High-CI Forecast (p50)', zorder=4)

    # =========================================================================
    # Plot high-vol high-CI / crisis period (red)
    # =========================================================================

    # Context (solid red line)
    ax.plot(context_days, high_vol_data['context_truth'],
            color='red', linewidth=2.5, label=f'High-Vol High-CI Context ({high_vol_date})', zorder=6)

    # Ground truth continuation in forecast region (red dotted)
    ax.plot(forecast_days, high_vol_data['forecast_truth'],
            color='red', linewidth=2, linestyle=':', alpha=0.6, zorder=5,
            label='High-Vol High-CI Ground Truth')

    # Forecast CI band (light red fill)
    ax.fill_between(forecast_days, high_vol_data['forecast_p05'], high_vol_data['forecast_p95'],
                    alpha=0.25, color='red', label='High-Vol High-CI 90% CI', zorder=1)

    # Forecast median (red dashed)
    ax.plot(forecast_days, high_vol_data['forecast_p50'],
            color='red', linewidth=2, linestyle='--', label='High-Vol High-CI Forecast (p50)', zorder=4)

    # =========================================================================
    # Visual elements
    # =========================================================================

    # Vertical line at context/forecast boundary
    ax.axvline(x=20, color='gray', linewidth=2, linestyle='-',
               alpha=0.6, label='Context/Forecast Boundary', zorder=3)

    # Background shading
    ax.axvspan(0, 20, alpha=0.05, color='gray', zorder=0)  # Context region

    # Subtitle annotation highlighting key finding
    ax.text(0.5, 0.985,
            'Key Finding: High CI despite low ATM vol (orange) - Pre-Crisis Detection',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontstyle='italic', color='orange',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='orange', linewidth=2))

    # Styling
    ax.set_xlabel('Day (Relative to Forecast Start)', fontsize=14, fontweight='bold')
    ax.set_ylabel('ATM 6M Implied Volatility', fontsize=14, fontweight='bold')
    ax.set_title('CVAE Conditional Generation: Three Regime Comparison\n'
                 'Normal vs Low-Vol High-CI vs High-Vol High-CI',
                 fontsize=16, fontweight='bold', pad=30)

    # Legend positioned below plot to avoid covering subtitle
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
             fontsize=9, framealpha=0.95, ncol=3)

    # Grid with minimal styling
    ax.grid(True, alpha=0.2, linestyle='--')

    # X-axis ticks every 5 days
    ax.set_xticks(np.arange(0, 51, 5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved three-regime overlay plot: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    """Main execution function"""
    print("=" * 80)
    print("CVAE CONDITIONAL GENERATION: THREE REGIME COMPARISON")
    print("=" * 80)
    print()

    # Target dates (User-selected Option 2)
    normal_date = '2007-03-28'
    low_vol_high_ci_date = '2007-10-09'
    high_vol_high_ci_date = '2008-10-30'

    # Always use prior sampling
    sampling_mode = 'prior'

    print(f"Target dates:")
    print(f"  Normal:          {normal_date} (narrow CI)")
    print(f"  Low-Vol High-CI: {low_vol_high_ci_date} (anomalous - wide CI despite low vol)")
    print(f"  High-Vol High-CI: {high_vol_high_ci_date} (crisis - widest CI)")
    print(f"  Sampling mode: {sampling_mode}")
    print()

    # =========================================================================
    # Load data
    # =========================================================================

    ci_stats = load_ci_width_stats(sampling_mode)
    print()

    tf_preds = load_tf_predictions(sampling_mode)
    print()

    gt_surfaces, gt_dates, gt_ex_data = load_ground_truth()
    print()

    # =========================================================================
    # Find sequence indices for target dates
    # =========================================================================

    print("Finding sequence indices for target dates...")

    normal_seq_idx = find_date_sequence(normal_date, ci_stats['dates'])
    print(f"  Normal date {normal_date} → sequence index {normal_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][normal_seq_idx]:.4f}")
    print(f"    ATM vol:  {ci_stats['atm_vol'][normal_seq_idx]:.3f}")

    low_vol_seq_idx = find_date_sequence(low_vol_high_ci_date, ci_stats['dates'])
    print(f"  Low-Vol High-CI date {low_vol_high_ci_date} → sequence index {low_vol_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][low_vol_seq_idx]:.4f}")
    print(f"    ATM vol:  {ci_stats['atm_vol'][low_vol_seq_idx]:.3f}")

    high_vol_seq_idx = find_date_sequence(high_vol_high_ci_date, ci_stats['dates'])
    print(f"  High-Vol High-CI date {high_vol_high_ci_date} → sequence index {high_vol_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][high_vol_seq_idx]:.4f}")
    print(f"    ATM vol:  {ci_stats['atm_vol'][high_vol_seq_idx]:.3f}")

    # Calculate ratios
    ci_ratio_low_vs_normal = ci_stats['avg_ci_width'][low_vol_seq_idx] / ci_stats['avg_ci_width'][normal_seq_idx]
    ci_ratio_high_vs_normal = ci_stats['avg_ci_width'][high_vol_seq_idx] / ci_stats['avg_ci_width'][normal_seq_idx]
    ci_ratio_high_vs_low = ci_stats['avg_ci_width'][high_vol_seq_idx] / ci_stats['avg_ci_width'][low_vol_seq_idx]

    print(f"\n  CI width ratios:")
    print(f"    Low-Vol/Normal:  {ci_ratio_low_vs_normal:.2f}×")
    print(f"    High-Vol/Normal: {ci_ratio_high_vs_normal:.2f}×")
    print(f"    High-Vol/Low-Vol: {ci_ratio_high_vs_low:.2f}×")
    print()

    # =========================================================================
    # Extract data for all three dates
    # =========================================================================

    print("Extracting context and forecast data...")

    normal_data = extract_context_forecast_truth(
        normal_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted normal period data")

    low_vol_data = extract_context_forecast_truth(
        low_vol_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted low-vol high-CI period data")

    high_vol_data = extract_context_forecast_truth(
        high_vol_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted high-vol high-CI period data")
    print()

    # =========================================================================
    # Create visualization
    # =========================================================================

    print("Creating three-regime overlay visualization...")

    output_dir = Path("results/vae_baseline/visualizations/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "three_regime_overlay_prior.png"

    create_three_regime_overlay(
        normal_data, low_vol_data, high_vol_data,
        normal_date, low_vol_high_ci_date, high_vol_high_ci_date,
        output_path
    )

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print("Key insights:")
    print(f"  - Low-Vol High-CI (orange) has {ci_ratio_low_vs_normal:.2f}× wider CI than Normal")
    print(f"  - This anomalous regime demonstrates pre-crisis detection")
    print(f"  - Model widens CI based on surface shape, not just ATM vol level")
    print(f"  - High-Vol High-CI (red) is {ci_ratio_high_vs_normal:.2f}× wider than Normal (expected)")
    print()


if __name__ == "__main__":
    main()
