"""
Visualize Calm vs Crisis CI Width Overlay - CVAE Conditional Generation Demo

Creates a single overlaid plot comparing two specific dates from prior sampling
to demonstrate how the CVAE produces different CI widths under different market conditions.

**Purpose**: CVAE presentation showing conditional generation - same model, different contexts → different uncertainty

**Two dates compared:**
- 2007-03-28 (calm period, narrowest CI)
- 2008-10-30 (crisis period, widest CI)

Usage:
    python experiments/backfill/context20/visualize_calm_vs_crisis_overlay.py

Output:
    results/vae_baseline/visualizations/comparison/calm_vs_crisis_overlay_prior.png
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


def create_overlay_plot(calm_data, crisis_data, output_path):
    """
    Create overlaid plot comparing calm vs crisis CI widths

    Args:
        calm_data: dict with context, forecast, truth for 2007-03-28
        crisis_data: dict with context, forecast, truth for 2008-10-30
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Create relative day numbers for x-axis
    context_days = np.arange(0, 20)
    forecast_days = np.arange(20, 50)

    # =========================================================================
    # Plot calm period (blue)
    # =========================================================================

    # Context (solid blue line)
    ax.plot(context_days, calm_data['context_truth'],
            color='blue', linewidth=2.5, label='Calm Context (2007-03-28)', zorder=4)

    # Ground truth continuation in forecast region (blue dotted)
    ax.plot(forecast_days, calm_data['forecast_truth'],
            color='blue', linewidth=2, linestyle=':', alpha=0.7, zorder=3,
            label='Calm Ground Truth')

    # Forecast CI band (light blue fill)
    ax.fill_between(forecast_days, calm_data['forecast_p05'], calm_data['forecast_p95'],
                    alpha=0.3, color='blue', label='Calm 90% CI', zorder=1)

    # Forecast median (blue dashed)
    ax.plot(forecast_days, calm_data['forecast_p50'],
            color='blue', linewidth=2, linestyle='--', label='Calm Forecast (p50)', zorder=2)

    # =========================================================================
    # Plot crisis period (red)
    # =========================================================================

    # Context (solid red line)
    ax.plot(context_days, crisis_data['context_truth'],
            color='red', linewidth=2.5, label='Crisis Context (2008-10-30)', zorder=4)

    # Ground truth continuation in forecast region (red dotted)
    ax.plot(forecast_days, crisis_data['forecast_truth'],
            color='red', linewidth=2, linestyle=':', alpha=0.7, zorder=3,
            label='Crisis Ground Truth')

    # Forecast CI band (light red fill)
    ax.fill_between(forecast_days, crisis_data['forecast_p05'], crisis_data['forecast_p95'],
                    alpha=0.3, color='red', label='Crisis 90% CI', zorder=1)

    # Forecast median (red dashed)
    ax.plot(forecast_days, crisis_data['forecast_p50'],
            color='red', linewidth=2, linestyle='--', label='Crisis Forecast (p50)', zorder=2)

    # =========================================================================
    # Visual elements
    # =========================================================================

    # Vertical line at context/forecast boundary
    ax.axvline(x=20, color='gray', linewidth=2, linestyle='-',
               alpha=0.6, label='Context/Forecast Boundary', zorder=5)

    # Background shading
    ax.axvspan(0, 20, alpha=0.05, color='gray', zorder=0)  # Context region

    # Styling
    ax.set_xlabel('Day (Relative to Forecast Start)', fontsize=14, fontweight='bold')
    ax.set_ylabel('ATM 6M Implied Volatility', fontsize=14, fontweight='bold')
    ax.set_title('CVAE Conditional Generation: Calm vs Crisis Context\n'
                 'Same Model, Different Contexts → Different Uncertainty',
                 fontsize=16, fontweight='bold', pad=20)

    # Legend with better positioning
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)

    # Grid with minimal styling
    ax.grid(True, alpha=0.2, linestyle='--')

    # X-axis ticks every 5 days
    ax.set_xticks(np.arange(0, 51, 5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved overlay plot: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    """Main execution function"""
    print("=" * 80)
    print("CVAE CONDITIONAL GENERATION: CALM VS CRISIS OVERLAY")
    print("=" * 80)
    print()

    # Target dates
    calm_date = '2007-03-28'
    crisis_date = '2008-10-30'

    # Always use prior sampling
    sampling_mode = 'prior'

    print(f"Target dates:")
    print(f"  Calm:   {calm_date} (narrowest CI)")
    print(f"  Crisis: {crisis_date} (widest CI)")
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

    calm_seq_idx = find_date_sequence(calm_date, ci_stats['dates'])
    print(f"  Calm date {calm_date} → sequence index {calm_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][calm_seq_idx]:.4f}")

    crisis_seq_idx = find_date_sequence(crisis_date, ci_stats['dates'])
    print(f"  Crisis date {crisis_date} → sequence index {crisis_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][crisis_seq_idx]:.4f}")

    ci_width_ratio = ci_stats['avg_ci_width'][crisis_seq_idx] / ci_stats['avg_ci_width'][calm_seq_idx]
    print(f"  CI width ratio (crisis/calm): {ci_width_ratio:.2f}×")
    print()

    # =========================================================================
    # Extract data for both dates
    # =========================================================================

    print("Extracting context and forecast data...")

    calm_data = extract_context_forecast_truth(
        calm_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted calm period data")

    crisis_data = extract_context_forecast_truth(
        crisis_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted crisis period data")
    print()

    # =========================================================================
    # Create visualization
    # =========================================================================

    print("Creating overlay visualization...")

    output_dir = Path("results/vae_baseline/visualizations/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "calm_vs_crisis_overlay_prior.png"

    create_overlay_plot(calm_data, crisis_data, output_path)

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print("Key insights:")
    print(f"  - Crisis CI is {ci_width_ratio:.2f}× wider than calm CI")
    print("  - Same CVAE model adapts uncertainty based on context")
    print("  - Demonstrates conditional generation capability")
    print()


if __name__ == "__main__":
    main()
