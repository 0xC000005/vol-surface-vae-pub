"""
Visualize Top 5 Widest CI Moments - Teacher Forcing Plots

Creates teacher forcing visualizations for the top 5 times when confidence interval (CI)
width is widest, showing:
- 20-day context (real historical data)
- 30-day horizon forecast (autoregressive generation)
- Ground truth comparison
- Separate figures for oracle and prior sampling modes

Usage:
    # Generate both oracle and prior visualizations
    python experiments/backfill/context20/visualize_top_ci_width_moments.py

    # Or specify sampling mode
    python experiments/backfill/context20/visualize_top_ci_width_moments.py --sampling_mode oracle
    python experiments/backfill/context20/visualize_top_ci_width_moments.py --sampling_mode prior

Output:
    results/vae_baseline/visualizations/top_ci_width_moments/oracle_top5_widest_ci_h30.png
    results/vae_baseline/visualizations/top_ci_width_moments/prior_top5_widest_ci_h30.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime


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


def identify_top_n_ci(ci_width_array, dates_array, n=5, ci_type='widest'):
    """
    Find top N dates with widest or narrowest CI

    Args:
        ci_width_array: (n_days,) CI width values at ATM 6M
        dates_array: (n_days,) datetime objects
        n: Number of top moments to return (default 5)
        ci_type: 'widest' or 'narrowest' (default 'widest')

    Returns:
        List of tuples: [(date, ci_width, sequence_idx), ...]
        Sorted by CI width (descending for widest, ascending for narrowest)
    """
    if ci_type == 'widest':
        # Get top N largest values
        top_n_indices = np.argsort(ci_width_array)[-n:][::-1]
    elif ci_type == 'narrowest':
        # Get top N smallest values
        top_n_indices = np.argsort(ci_width_array)[:n]
    else:
        raise ValueError(f"ci_type must be 'widest' or 'narrowest', got {ci_type}")

    # Extract dates and CI widths
    results = [
        (dates_array[idx], ci_width_array[idx], idx)
        for idx in top_n_indices
    ]

    return results


def extract_context_forecast_truth(sequence_idx,
                                     tf_predictions,
                                     indices_array,
                                     ground_truth_surfaces,
                                     dates):
    """
    Extract aligned data for one visualization row

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


def compute_metrics(forecast_truth, forecast_p05, forecast_p50, forecast_p95):
    """
    Compute RMSE and CI violations for forecast window

    Returns:
        dict with rmse, ci_violations, ci_violation_pct, mean_ci_width
    """
    # RMSE
    rmse = np.sqrt(np.mean((forecast_truth - forecast_p50)**2))

    # CI violations (ground truth outside [p05, p95])
    violations = (forecast_truth < forecast_p05) | (forecast_truth > forecast_p95)
    ci_violations = np.sum(violations)
    ci_violation_pct = 100 * ci_violations / len(forecast_truth)

    # Mean CI width
    mean_ci_width = np.mean(forecast_p95 - forecast_p05)

    return {
        'rmse': rmse,
        'ci_violations': ci_violations,
        'ci_violation_pct': ci_violation_pct,
        'mean_ci_width': mean_ci_width
    }


def create_top5_visualization(top5_data_list, sampling_mode, ci_type, output_path):
    """
    Create 5-row stacked figure showing top 5 widest/narrowest CI moments

    Args:
        top5_data_list: List of 5 dicts from extract_context_forecast_truth()
        sampling_mode: 'oracle' or 'prior'
        ci_type: 'widest' or 'narrowest'
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))

    # Update title to indicate widest/narrowest
    title_type = ci_type.capitalize()
    fig.suptitle(f'Top 5 {title_type} CI Moments - {sampling_mode.upper()} Sampling (H=30, ATM 6M)',
                 fontsize=16, fontweight='bold')

    for i, (ax, data) in enumerate(zip(axes, top5_data_list)):
        # Combine dates for continuous timeline
        all_dates = np.concatenate([data['context_dates'], data['forecast_dates']])

        # Combine ground truth for continuous line
        all_truth = np.concatenate([data['context_truth'], data['forecast_truth']])

        # Plot ground truth (continuous black line)
        ax.plot(all_dates, all_truth, 'k-', linewidth=2, label='Ground Truth', zorder=3)

        # Plot forecast region only
        forecast_dates = data['forecast_dates']
        ax.plot(forecast_dates, data['forecast_p50'],
                color='#1f77b4', linewidth=1.5, label='Forecast (p50)', zorder=2)

        # CI band (forecast region only)
        ax.fill_between(forecast_dates,
                        data['forecast_p05'],
                        data['forecast_p95'],
                        color='#1f77b4', alpha=0.25, label='90% CI', zorder=1)

        # Context/forecast boundary (vertical red line)
        boundary_date = data['context_dates'][-1]
        ax.axvline(boundary_date, color='red', linestyle='--',
                   linewidth=2, label='Context/Forecast Split', zorder=4)

        # Background shading
        ax.axvspan(all_dates[0], boundary_date, color='gray', alpha=0.1)  # Context region

        # CI violations (red circles on ground truth)
        violations_mask = (data['forecast_truth'] < data['forecast_p05']) | \
                          (data['forecast_truth'] > data['forecast_p95'])
        violation_dates = forecast_dates[violations_mask]
        violation_values = data['forecast_truth'][violations_mask]
        ax.scatter(violation_dates, violation_values,
                   color='red', s=50, zorder=5, label='CI Violations')

        # Metrics
        metrics = compute_metrics(data['forecast_truth'],
                                   data['forecast_p05'],
                                   data['forecast_p50'],
                                   data['forecast_p95'])

        # Text box with metrics
        metrics_text = (f"RMSE: {metrics['rmse']:.4f}\n"
                       f"CI Violations: {metrics['ci_violations']}/30 "
                       f"({metrics['ci_violation_pct']:.1f}%)\n"
                       f"Mean CI Width: {metrics['mean_ci_width']:.4f}")

        ax.text(0.02, 0.95, metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Styling
        ax.set_ylabel('Implied Volatility', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Row title (date + CI width)
        # Convert numpy.datetime64 to pandas Timestamp for strftime
        forecast_start_date = pd.Timestamp(all_dates[20]).strftime('%Y-%m-%d')
        row_title = f"#{i+1}: {forecast_start_date} " \
                   f"(CI Width: {metrics['mean_ci_width']:.4f})"
        ax.set_title(row_title, fontsize=12, fontweight='bold', loc='left')

        # X-axis formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main(sampling_mode='oracle', ci_type='widest'):
    """Main execution pipeline"""

    print(f"\n{'='*80}")
    print(f"Processing {sampling_mode.upper()} - {ci_type.upper()} CI moments")
    print(f"{'='*80}\n")

    # 1. Load data
    ci_stats = load_ci_width_stats(sampling_mode)
    tf_preds = load_tf_predictions(sampling_mode)
    ground_truth_surfaces, gt_dates, _ = load_ground_truth()

    # 2. Identify top 5 (widest or narrowest)
    top5 = identify_top_n_ci(
        ci_stats['avg_ci_width'],
        ci_stats['dates'],
        n=5,
        ci_type=ci_type
    )

    print(f"\nTop 5 {ci_type} CI moments ({sampling_mode}):")
    for i, (date, ci_width, seq_idx) in enumerate(top5, 1):
        # dates are already strings in YYYY-MM-DD format
        date_str = str(date) if not isinstance(date, str) else date
        print(f"  {i}. {date_str} - CI width: {ci_width:.6f} (seq_idx: {seq_idx})")

    # 3. Extract data for each top moment
    print(f"\nExtracting context + forecast data for top 5 {ci_type} moments...")
    top5_data = []
    for date, ci_width, seq_idx in top5:
        data = extract_context_forecast_truth(
            seq_idx,
            tf_preds,
            ci_stats['indices'],
            ground_truth_surfaces,
            gt_dates
        )
        top5_data.append(data)

    # 4. Create visualization
    output_dir = Path("results/vae_baseline/visualizations/top_ci_width_moments")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sampling_mode}_top5_{ci_type}_ci_h30.png"

    print(f"\nCreating visualization...")
    create_top5_visualization(top5_data, sampling_mode, ci_type, output_path)

    print(f"\n{'='*80}")
    print(f"Completed {sampling_mode} - {ci_type} visualization!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize top 5 widest/narrowest CI moments with teacher forcing plots"
    )
    parser.add_argument("--sampling_mode", type=str, default="both",
                       choices=["oracle", "prior", "both"],
                       help="Sampling mode to visualize")
    parser.add_argument("--ci_type", type=str, default="narrowest",
                       choices=["widest", "narrowest", "both"],
                       help="CI type to visualize (widest, narrowest, or both)")
    args = parser.parse_args()

    # Determine which combinations to run
    sampling_modes = ["oracle", "prior"] if args.sampling_mode == "both" else [args.sampling_mode]
    ci_types = ["widest", "narrowest"] if args.ci_type == "both" else [args.ci_type]

    # Run all combinations
    for sampling_mode in sampling_modes:
        for ci_type in ci_types:
            main(sampling_mode, ci_type)
