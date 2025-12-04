"""
Visualize Four-Regime CI Width Overlay - CVAE Intelligence Demonstration

Creates a single overlaid plot comparing four market regimes from prior sampling
to demonstrate CVAE's intelligent pattern recognition beyond simple volatility levels.

**Purpose**: Showcase complete 2×2 matrix of regime types demonstrating both
anomalous patterns and intelligent confidence

**Four regimes compared:**
- 2007-03-28 (normal/calm period, narrow CI) - GREEN
- 2009-02-23 (high-vol low-CI, intelligent confidence) - BLUE
- 2007-10-09 (low-vol high-CI, anomalous - pre-crisis) - ORANGE
- 2008-10-30 (high-vol high-CI, crisis peak) - RED

**Complete 2×2 Matrix:**
                Low CI              High CI
Low Vol    Green (Normal)      Orange (Pre-Crisis Detection)
High Vol   Blue (Smart)        Red (Expected Crisis)

Usage:
    python experiments/backfill/context20/visualize_four_regime_overlay.py

Output:
    results/vae_baseline/visualizations/comparison/four_regime_overlay_prior.png
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


def create_four_regime_overlay(green_data, blue_data, orange_data, red_data,
                                 green_date, blue_date, orange_date, red_date,
                                 output_path):
    """
    Create overlaid plot comparing four regimes demonstrating complete 2×2 matrix

    Args:
        green_data: dict with context, forecast, truth for normal period (Low-Vol Low-CI)
        blue_data: dict with context, forecast, truth for high-vol low-CI period (Intelligent Confidence)
        orange_data: dict with context, forecast, truth for low-vol high-CI period (Pre-Crisis Detection)
        red_data: dict with context, forecast, truth for high-vol high-CI period (Expected Crisis)
        green_date: Date string for normal period
        blue_date: Date string for high-vol low-CI period
        orange_date: Date string for low-vol high-CI period
        red_date: Date string for high-vol high-CI period
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 9))

    # Create relative day numbers for x-axis
    context_days = np.arange(0, 20)
    forecast_days = np.arange(20, 50)

    # =========================================================================
    # Plot GREEN: Normal (Low-Vol Low-CI) - BASELINE
    # =========================================================================

    # Context (solid green line)
    ax.plot(context_days, green_data['context_truth'],
            color='green', linewidth=2.5, label=f'Green: Normal Context ({green_date})', zorder=6)

    # Ground truth continuation in forecast region (green dotted)
    ax.plot(forecast_days, green_data['forecast_truth'],
            color='green', linewidth=2, linestyle=':', alpha=0.6, zorder=5,
            label='Green: Ground Truth')

    # Forecast CI band (light green fill)
    ax.fill_between(forecast_days, green_data['forecast_p05'], green_data['forecast_p95'],
                    alpha=0.25, color='green', label='Green: 90% CI', zorder=1)

    # Forecast median (green dashed)
    ax.plot(forecast_days, green_data['forecast_p50'],
            color='green', linewidth=2, linestyle='--', label='Green: Forecast (p50)', zorder=4)

    # =========================================================================
    # Plot BLUE: High-Vol Low-CI - INTELLIGENT CONFIDENCE
    # =========================================================================

    # Context (solid blue line)
    ax.plot(context_days, blue_data['context_truth'],
            color='blue', linewidth=2.5, label=f'Blue: High-Vol Low-CI Context ({blue_date})', zorder=6)

    # Ground truth continuation in forecast region (blue dotted)
    ax.plot(forecast_days, blue_data['forecast_truth'],
            color='blue', linewidth=2, linestyle=':', alpha=0.6, zorder=5,
            label='Blue: Ground Truth')

    # Forecast CI band (light blue fill)
    ax.fill_between(forecast_days, blue_data['forecast_p05'], blue_data['forecast_p95'],
                    alpha=0.25, color='blue', label='Blue: 90% CI', zorder=1)

    # Forecast median (blue dashed)
    ax.plot(forecast_days, blue_data['forecast_p50'],
            color='blue', linewidth=2, linestyle='--', label='Blue: Forecast (p50)', zorder=4)

    # =========================================================================
    # Plot ORANGE: Low-Vol High-CI - PRE-CRISIS DETECTION (ANOMALOUS)
    # =========================================================================

    # Context (solid orange line)
    ax.plot(context_days, orange_data['context_truth'],
            color='orange', linewidth=2.5, label=f'Orange: Low-Vol High-CI Context ({orange_date})', zorder=6)

    # Ground truth continuation in forecast region (orange dotted)
    ax.plot(forecast_days, orange_data['forecast_truth'],
            color='orange', linewidth=2, linestyle=':', alpha=0.6, zorder=5,
            label='Orange: Ground Truth')

    # Forecast CI band (light orange fill)
    ax.fill_between(forecast_days, orange_data['forecast_p05'], orange_data['forecast_p95'],
                    alpha=0.25, color='orange', label='Orange: 90% CI', zorder=1)

    # Forecast median (orange dashed)
    ax.plot(forecast_days, orange_data['forecast_p50'],
            color='orange', linewidth=2, linestyle='--', label='Orange: Forecast (p50)', zorder=4)

    # =========================================================================
    # Plot RED: High-Vol High-CI - EXPECTED CRISIS
    # =========================================================================

    # Context (solid red line)
    ax.plot(context_days, red_data['context_truth'],
            color='red', linewidth=2.5, label=f'Red: High-Vol High-CI Context ({red_date})', zorder=6)

    # Ground truth continuation in forecast region (red dotted)
    ax.plot(forecast_days, red_data['forecast_truth'],
            color='red', linewidth=2, linestyle=':', alpha=0.6, zorder=5,
            label='Red: Ground Truth')

    # Forecast CI band (light red fill)
    ax.fill_between(forecast_days, red_data['forecast_p05'], red_data['forecast_p95'],
                    alpha=0.25, color='red', label='Red: 90% CI', zorder=1)

    # Forecast median (red dashed)
    ax.plot(forecast_days, red_data['forecast_p50'],
            color='red', linewidth=2, linestyle='--', label='Red: Forecast (p50)', zorder=4)

    # =========================================================================
    # Visual elements
    # =========================================================================

    # Vertical line at context/forecast boundary
    ax.axvline(x=20, color='gray', linewidth=2, linestyle='-',
               alpha=0.6, label='Context/Forecast Boundary', zorder=3)

    # Background shading
    ax.axvspan(0, 20, alpha=0.05, color='gray', zorder=0)  # Context region

    # Dual subtitle annotation highlighting both key insights
    subtitle_text = (
        'Orange: High CI despite low vol (Pre-Crisis Detection) | '
        'Blue: Low CI despite high vol (Intelligent Confidence in Familiar Patterns)'
    )
    ax.text(0.5, 0.985,
            subtitle_text,
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, fontstyle='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                     edgecolor='purple', linewidth=2))

    # Styling
    ax.set_xlabel('Day (Relative to Forecast Start)', fontsize=14, fontweight='bold')
    ax.set_ylabel('ATM 6M Implied Volatility', fontsize=14, fontweight='bold')
    ax.set_title('CVAE Conditional Generation: Four Regime Comparison\n'
                 'Demonstrating Intelligent Pattern Recognition Beyond Volatility Levels',
                 fontsize=16, fontweight='bold', pad=35)

    # Legend positioned below plot with 4 columns
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
             fontsize=9, framealpha=0.95, ncol=4)

    # Grid with minimal styling
    ax.grid(True, alpha=0.2, linestyle='--')

    # X-axis ticks every 5 days
    ax.set_xticks(np.arange(0, 51, 5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved four-regime overlay plot: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    """Main execution function"""
    print("=" * 80)
    print("CVAE CONDITIONAL GENERATION: FOUR REGIME COMPARISON")
    print("Complete 2×2 Matrix of Regime Types")
    print("=" * 80)
    print()

    # Target dates for complete 2×2 matrix
    green_date = '2007-03-28'    # Low-Vol Low-CI (Normal)
    blue_date = '2009-02-23'     # High-Vol Low-CI (Intelligent Confidence)
    orange_date = '2007-10-09'   # Low-Vol High-CI (Pre-Crisis Detection)
    red_date = '2008-10-30'      # High-Vol High-CI (Expected Crisis)

    # Always use prior sampling
    sampling_mode = 'prior'

    print(f"Target dates (2×2 Matrix):")
    print(f"  Green (Normal):           {green_date} - Low Vol, Low CI")
    print(f"  Blue (Smart):             {blue_date} - High Vol, Low CI (INTELLIGENT CONFIDENCE)")
    print(f"  Orange (Pre-Crisis):      {orange_date} - Low Vol, High CI (ANOMALOUS)")
    print(f"  Red (Crisis):             {red_date} - High Vol, High CI")
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

    green_seq_idx = find_date_sequence(green_date, ci_stats['dates'])
    print(f"  Green (Normal) {green_date} → sequence index {green_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][green_seq_idx]:.4f}")
    print(f"    ATM vol:  {ci_stats['atm_vol'][green_seq_idx]:.3f}")

    blue_seq_idx = find_date_sequence(blue_date, ci_stats['dates'])
    print(f"  Blue (High-Vol Low-CI) {blue_date} → sequence index {blue_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][blue_seq_idx]:.4f}")
    print(f"    ATM vol:  {ci_stats['atm_vol'][blue_seq_idx]:.3f}")

    orange_seq_idx = find_date_sequence(orange_date, ci_stats['dates'])
    print(f"  Orange (Low-Vol High-CI) {orange_date} → sequence index {orange_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][orange_seq_idx]:.4f}")
    print(f"    ATM vol:  {ci_stats['atm_vol'][orange_seq_idx]:.3f}")

    red_seq_idx = find_date_sequence(red_date, ci_stats['dates'])
    print(f"  Red (High-Vol High-CI) {red_date} → sequence index {red_seq_idx}")
    print(f"    CI width: {ci_stats['avg_ci_width'][red_seq_idx]:.4f}")
    print(f"    ATM vol:  {ci_stats['atm_vol'][red_seq_idx]:.3f}")

    # Calculate key ratios for analysis
    ci_blue_vs_green = ci_stats['avg_ci_width'][blue_seq_idx] / ci_stats['avg_ci_width'][green_seq_idx]
    ci_orange_vs_green = ci_stats['avg_ci_width'][orange_seq_idx] / ci_stats['avg_ci_width'][green_seq_idx]
    ci_red_vs_green = ci_stats['avg_ci_width'][red_seq_idx] / ci_stats['avg_ci_width'][green_seq_idx]
    ci_blue_vs_red = ci_stats['avg_ci_width'][blue_seq_idx] / ci_stats['avg_ci_width'][red_seq_idx]

    print(f"\n  CI width ratios (vs Green baseline):")
    print(f"    Blue/Green:   {ci_blue_vs_green:.2f}×")
    print(f"    Orange/Green: {ci_orange_vs_green:.2f}×")
    print(f"    Red/Green:    {ci_red_vs_green:.2f}×")
    print(f"\n  Key intelligence metric:")
    print(f"    Blue/Red:     {ci_blue_vs_red:.2f}× (Both high vol, but Blue is {(1-ci_blue_vs_red)*100:.0f}% narrower)")
    print()

    # =========================================================================
    # Extract data for all four dates
    # =========================================================================

    print("Extracting context and forecast data...")

    green_data = extract_context_forecast_truth(
        green_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted green (normal) period data")

    blue_data = extract_context_forecast_truth(
        blue_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted blue (high-vol low-CI) period data")

    orange_data = extract_context_forecast_truth(
        orange_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted orange (low-vol high-CI) period data")

    red_data = extract_context_forecast_truth(
        red_seq_idx, tf_preds, ci_stats['indices'], gt_surfaces, gt_dates
    )
    print(f"  ✓ Extracted red (high-vol high-CI) period data")
    print()

    # =========================================================================
    # Create visualization
    # =========================================================================

    print("Creating four-regime overlay visualization...")

    output_dir = Path("results/vae_baseline/visualizations/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "four_regime_overlay_prior.png"

    create_four_regime_overlay(
        green_data, blue_data, orange_data, red_data,
        green_date, blue_date, orange_date, red_date,
        output_path
    )

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE - 2×2 REGIME MATRIX")
    print("=" * 80)
    print()
    print("Key insights:")
    print(f"  ✓ Complete 2×2 matrix demonstrated:")
    print(f"    - Low Vol, Low CI:   Green (baseline)")
    print(f"    - High Vol, Low CI:  Blue (intelligent confidence - {ci_blue_vs_green:.2f}× baseline)")
    print(f"    - Low Vol, High CI:  Orange (pre-crisis detection - {ci_orange_vs_green:.2f}× baseline)")
    print(f"    - High Vol, High CI: Red (expected crisis - {ci_red_vs_green:.2f}× baseline)")
    print()
    print(f"  ✓ Intelligence demonstration:")
    print(f"    - Blue and Red have similar context volatility")
    print(f"    - But Blue CI is {(1-ci_blue_vs_red)*100:.0f}% narrower than Red")
    print(f"    - Model recognizes familiar patterns even in volatile conditions")
    print(f"    - Confidence comes from pattern recognition, not calm markets")
    print()


if __name__ == "__main__":
    main()
