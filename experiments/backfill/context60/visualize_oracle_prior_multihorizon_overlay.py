#!/usr/bin/env python3
"""
Oracle vs Prior Multi-Horizon Comparison with 2 Regimes Overlaid

Compares oracle and prior sampling modes across H=30, H=60, H=90 horizons,
overlaying two market regimes (2007-03-28 calm vs 2007-10-09 pre-crisis) to
demonstrate that longer prediction horizons show more pronounced oracle-prior
CI width gaps.

Author: Generated with Claude Code
Date: 2025-12-04
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================================
# Constants
# ============================================================================

CONTEXT_LEN = 60
ATM_6M = (2, 2)  # Grid point: K/S=1.00, 6-month maturity


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_predictions(horizon, sampling_mode='oracle'):
    """Load teacher forcing predictions for given horizon and mode.

    Args:
        horizon: 30, 60, or 90
        sampling_mode: 'oracle' or 'prior'

    Returns:
        dict with 'surfaces', 'indices', 'horizon'
    """
    filepath = (f"results/context60_baseline/predictions/teacher_forcing/"
                f"{sampling_mode}/vae_tf_insample_h{horizon}.npz")

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Prediction file not found: {filepath}")

    data = np.load(filepath)

    print(f"  Loaded {sampling_mode} predictions: shape {data['surfaces'].shape}")

    return {
        'surfaces': data['surfaces'],  # (n_seq, H, 3, 5, 5)
        'indices': data['indices'],
        'horizon': int(data['horizon'])
    }


def load_ground_truth():
    """Load ground truth surfaces and dates.

    Returns:
        (surfaces, dates) tuple
    """
    gt_data = np.load("data/vol_surface_with_ret.npz")
    dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    dates = pd.to_datetime(dates_df["date"].values)

    print(f"  Loaded ground truth: {gt_data['surface'].shape}, dates: {len(dates)}")

    return gt_data['surface'], dates


def find_sequence_index(target_date, dates, indices, context_len=60):
    """Find sequence index for specific target date with error handling.

    Args:
        target_date: 'YYYY-MM-DD' format
        dates: Full date array from ground truth
        indices: Array of data indices where forecasts start
        context_len: Number of context days (60)

    Returns:
        Sequence index in the predictions arrays
    """
    try:
        target_ts = pd.Timestamp(target_date)
    except Exception as e:
        raise ValueError(f"Invalid date format '{target_date}': {e}")

    # Find date in ground truth
    try:
        date_idx = dates.get_loc(target_ts)
    except KeyError:
        # Try finding nearest date within 1 day
        date_diffs = np.abs((dates - target_ts).total_seconds())
        nearest_idx = np.argmin(date_diffs)
        nearest_date = dates[nearest_idx]

        if abs((nearest_date - target_ts).total_seconds()) < 86400:  # Within 1 day
            print(f"  Warning: Date {target_date} not found, using nearest: {nearest_date}")
            date_idx = nearest_idx
        else:
            raise ValueError(f"Date {target_date} not found. Range: {dates[0]} to {dates[-1]}")

    # Validate sufficient context
    if date_idx < context_len:
        raise ValueError(f"Date {target_date} needs {context_len} days context (insufficient)")

    # Find in predictions
    seq_idx = np.where(indices == date_idx)[0]
    if len(seq_idx) == 0:
        raise ValueError(f"Date {target_date} not in predictions. "
                        f"Available: {dates[indices.min()]} to {dates[indices.max()]}")

    return seq_idx[0]


# ============================================================================
# Data Extraction Functions
# ============================================================================

def extract_regime_data(oracle_data, prior_data, gt_surface, seq_idx, horizon):
    """Extract all data for one regime at one horizon.

    Args:
        oracle_data: Oracle predictions dict
        prior_data: Prior predictions dict
        gt_surface: Ground truth surfaces (N, 5, 5)
        seq_idx: Sequence index
        horizon: Forecast horizon

    Returns:
        dict with all plotting data
    """
    grid_row, grid_col = ATM_6M

    # Get data start index
    data_start_idx = oracle_data['indices'][seq_idx]

    # Validate alignment
    if oracle_data['indices'][seq_idx] != prior_data['indices'][seq_idx]:
        raise ValueError("Oracle and prior indices misaligned!")

    # Extract context ground truth
    context_start = data_start_idx - CONTEXT_LEN
    context_truth = gt_surface[context_start:data_start_idx, grid_row, grid_col]

    # Extract forecast ground truth
    forecast_truth = gt_surface[data_start_idx:data_start_idx + horizon, grid_row, grid_col]

    # Extract oracle predictions (quantile axis: 0=p05, 1=p50, 2=p95)
    oracle_p05 = oracle_data['surfaces'][seq_idx, :, 0, grid_row, grid_col]
    oracle_p50 = oracle_data['surfaces'][seq_idx, :, 1, grid_row, grid_col]
    oracle_p95 = oracle_data['surfaces'][seq_idx, :, 2, grid_row, grid_col]

    # Extract prior predictions
    prior_p05 = prior_data['surfaces'][seq_idx, :, 0, grid_row, grid_col]
    prior_p50 = prior_data['surfaces'][seq_idx, :, 1, grid_row, grid_col]
    prior_p95 = prior_data['surfaces'][seq_idx, :, 2, grid_row, grid_col]

    return {
        'context_truth': context_truth,
        'forecast_truth': forecast_truth,
        'oracle_p05': oracle_p05,
        'oracle_p50': oracle_p50,
        'oracle_p95': oracle_p95,
        'prior_p05': prior_p05,
        'prior_p50': prior_p50,
        'prior_p95': prior_p95,
    }


# ============================================================================
# Statistics Functions
# ============================================================================

def compute_statistics(forecast_truth, oracle_p05, oracle_p50, oracle_p95,
                      prior_p05, prior_p50, prior_p95):
    """Compute RMSE and CI coverage statistics.

    Args:
        forecast_truth: Ground truth values (H,)
        oracle_p05, oracle_p50, oracle_p95: Oracle predicted quantiles (H,)
        prior_p05, prior_p50, prior_p95: Prior predicted quantiles (H,)

    Returns:
        dict with statistics
    """
    # RMSE (using p50 as point forecast)
    oracle_rmse = np.sqrt(np.mean((oracle_p50 - forecast_truth)**2))
    prior_rmse = np.sqrt(np.mean((prior_p50 - forecast_truth)**2))

    # CI Coverage (percentage of ground truth within [p05, p95])
    oracle_coverage = np.mean((forecast_truth >= oracle_p05) &
                             (forecast_truth <= oracle_p95)) * 100
    prior_coverage = np.mean((forecast_truth >= prior_p05) &
                            (forecast_truth <= prior_p95)) * 100

    # CI Width (average)
    oracle_ci_width = np.mean(oracle_p95 - oracle_p05)
    prior_ci_width = np.mean(prior_p95 - prior_p05)
    ci_ratio = prior_ci_width / oracle_ci_width

    return {
        'oracle_rmse': oracle_rmse,
        'prior_rmse': prior_rmse,
        'oracle_coverage': oracle_coverage,
        'prior_coverage': prior_coverage,
        'oracle_ci_width': oracle_ci_width,
        'prior_ci_width': prior_ci_width,
        'ci_ratio': ci_ratio
    }


def add_statistics_box(ax, r1_stats, r2_stats):
    """Add side-by-side statistics boxes for both regimes.

    Args:
        ax: Matplotlib axis
        r1_stats: Statistics dict for regime 1
        r2_stats: Statistics dict for regime 2
    """
    # Regime 1 stats (Calm)
    text1 = (
        f"Calm (2007-03-28)\n"
        f"{'─'*20}\n"
        f"Oracle RMSE:  {r1_stats['oracle_rmse']:.4f}\n"
        f"Prior RMSE:   {r1_stats['prior_rmse']:.4f}\n"
        f"Oracle Cov:   {r1_stats['oracle_coverage']:.1f}%\n"
        f"Prior Cov:    {r1_stats['prior_coverage']:.1f}%\n"
        f"CI Ratio: {r1_stats['ci_ratio']:.2f}×"
    )

    # Regime 2 stats (Pre-crisis)
    text2 = (
        f"Pre-Crisis (2007-10-09)\n"
        f"{'─'*20}\n"
        f"Oracle RMSE:  {r2_stats['oracle_rmse']:.4f}\n"
        f"Prior RMSE:   {r2_stats['prior_rmse']:.4f}\n"
        f"Oracle Cov:   {r2_stats['oracle_coverage']:.1f}%\n"
        f"Prior Cov:    {r2_stats['prior_coverage']:.1f}%\n"
        f"CI Ratio: {r2_stats['ci_ratio']:.2f}×"
    )

    # Box styles matching regime colors
    box1 = dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9',  # Light green
                alpha=0.8, edgecolor='#2E7D32', linewidth=1.5)  # Dark green
    box2 = dict(boxstyle='round,pad=0.5', facecolor='#FFE0B2',  # Light orange
                alpha=0.8, edgecolor='#E65100', linewidth=1.5)  # Dark orange

    # Place boxes side by side
    ax.text(0.02, 0.03, text1, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            family='monospace', bbox=box1)

    ax.text(0.30, 0.03, text2, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            family='monospace', bbox=box2)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_horizon_comparison(ax, horizon, r1_data, r2_data):
    """Plot oracle vs prior for both regimes in single panel.

    Args:
        ax: Matplotlib axis
        horizon: Forecast horizon
        r1_data: Regime 1 data dict
        r2_data: Regime 2 data dict
    """
    # X-axis: relative day numbers
    context_days = np.arange(0, CONTEXT_LEN)  # 0-59
    forecast_days = np.arange(CONTEXT_LEN, CONTEXT_LEN + horizon)  # 60-89/119/149

    # === REGIME 1 (Calm) - SOLID lines, filled bands ===

    # Context ground truth (dark green solid)
    ax.plot(context_days, r1_data['context_truth'],
            color='#2E7D32', linewidth=2, linestyle='-', alpha=0.8,
            zorder=6)

    # Ground truth forecast (green solid)
    ax.plot(forecast_days, r1_data['forecast_truth'],
            color='#4CAF50', linewidth=2, linestyle='-', alpha=0.8,
            zorder=5)

    # Oracle CI band (blue fill)
    ax.fill_between(forecast_days,
                    r1_data['oracle_p05'], r1_data['oracle_p95'],
                    color='#1E88E5', alpha=0.25, zorder=1)

    # Oracle p50 (dark blue solid)
    ax.plot(forecast_days, r1_data['oracle_p50'],
            color='#0D47A1', linewidth=1.5, linestyle='-',
            alpha=0.9, zorder=3)

    # Prior CI band (red fill)
    ax.fill_between(forecast_days,
                    r1_data['prior_p05'], r1_data['prior_p95'],
                    color='#E53935', alpha=0.25, zorder=1)

    # Prior p50 (dark red solid)
    ax.plot(forecast_days, r1_data['prior_p50'],
            color='#B71C1C', linewidth=1.5, linestyle='-',
            alpha=0.9, zorder=3)

    # === REGIME 2 (Pre-crisis) - DASHED lines, hatched bands ===

    # Context ground truth (dark orange dashed)
    ax.plot(context_days, r2_data['context_truth'],
            color='#E65100', linewidth=2.5, linestyle='--', alpha=0.8,
            zorder=6)

    # Ground truth forecast (orange dashed)
    ax.plot(forecast_days, r2_data['forecast_truth'],
            color='#FF9800', linewidth=2.5, linestyle='--', alpha=0.8,
            zorder=5)

    # Oracle CI band (blue hatch)
    ax.fill_between(forecast_days,
                    r2_data['oracle_p05'], r2_data['oracle_p95'],
                    color='#1E88E5', alpha=0.18, hatch='///',
                    edgecolor='none', zorder=1)

    # Oracle p50 (dark blue dashed)
    ax.plot(forecast_days, r2_data['oracle_p50'],
            color='#0D47A1', linewidth=1.8, linestyle='--',
            alpha=0.9, zorder=3)

    # Prior CI band (red hatch)
    ax.fill_between(forecast_days,
                    r2_data['prior_p05'], r2_data['prior_p95'],
                    color='#E53935', alpha=0.18, hatch='///',
                    edgecolor='none', zorder=1)

    # Prior p50 (dark red dashed)
    ax.plot(forecast_days, r2_data['prior_p50'],
            color='#B71C1C', linewidth=1.8, linestyle='--',
            alpha=0.9, zorder=3)

    # Vertical boundary line at context/forecast split
    ax.axvline(x=CONTEXT_LEN, color='gray', linewidth=2,
              linestyle='-', alpha=0.5, zorder=8)

    # Background shading for context region
    ax.axvspan(0, CONTEXT_LEN, alpha=0.05, color='gray', zorder=0)

    # Labels and title
    ax.set_xlabel('Day (Relative to Forecast Start)', fontsize=11)
    ax.set_ylabel('ATM 6M Implied Volatility', fontsize=11)
    ax.set_title(f'Horizon = {horizon} days', fontsize=13,
                 fontweight='bold', pad=10)

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("MULTI-HORIZON ORACLE VS PRIOR COMPARISON")
    print("="*80)

    # Load ground truth
    print("\nLoading ground truth...")
    gt_surface, dates = load_ground_truth()

    # Target regimes
    regime1_date = '2007-03-28'
    regime2_date = '2007-10-09'

    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.35, left=0.07, right=0.98,
                       top=0.95, bottom=0.08)

    horizons = [30, 60, 90]

    # Storage for computing global y-axis limits
    all_data = []

    for idx, (ax, horizon) in enumerate(zip(axes, horizons)):
        print(f"\nProcessing H={horizon}...")

        # Load predictions
        print(f"  Loading predictions for H={horizon}...")
        oracle_data = load_predictions(horizon, 'oracle')
        prior_data = load_predictions(horizon, 'prior')

        # Find sequence indices
        print(f"  Finding sequence indices...")
        r1_idx = find_sequence_index(regime1_date, dates, oracle_data['indices'])
        r2_idx = find_sequence_index(regime2_date, dates, oracle_data['indices'])
        print(f"    Regime 1 ({regime1_date}): seq_idx={r1_idx}")
        print(f"    Regime 2 ({regime2_date}): seq_idx={r2_idx}")

        # Extract all data for both regimes
        print(f"  Extracting regime data...")
        r1_data = extract_regime_data(oracle_data, prior_data, gt_surface,
                                     r1_idx, horizon)
        r2_data = extract_regime_data(oracle_data, prior_data, gt_surface,
                                     r2_idx, horizon)

        # Collect all data for global y-axis limits
        all_data.extend([
            r1_data['context_truth'], r1_data['forecast_truth'],
            r1_data['oracle_p05'], r1_data['oracle_p95'],
            r1_data['prior_p05'], r1_data['prior_p95'],
            r2_data['context_truth'], r2_data['forecast_truth'],
            r2_data['oracle_p05'], r2_data['oracle_p95'],
            r2_data['prior_p05'], r2_data['prior_p95'],
        ])

        # Plot
        print(f"  Plotting...")
        plot_horizon_comparison(ax, horizon, r1_data, r2_data)

        # Compute statistics
        print(f"  Computing statistics...")
        r1_stats = compute_statistics(r1_data['forecast_truth'],
                                      r1_data['oracle_p05'], r1_data['oracle_p50'], r1_data['oracle_p95'],
                                      r1_data['prior_p05'], r1_data['prior_p50'], r1_data['prior_p95'])

        r2_stats = compute_statistics(r2_data['forecast_truth'],
                                      r2_data['oracle_p05'], r2_data['oracle_p50'], r2_data['oracle_p95'],
                                      r2_data['prior_p05'], r2_data['prior_p50'], r2_data['prior_p95'])

        print(f"    Regime 1 CI ratio: {r1_stats['ci_ratio']:.2f}×")
        print(f"    Regime 2 CI ratio: {r2_stats['ci_ratio']:.2f}×")

        # Add statistics boxes
        add_statistics_box(ax, r1_stats, r2_stats)

    # Set consistent axis limits across all panels
    print(f"\nSetting consistent axis limits...")

    # Y-axis: same range for all panels
    all_values = np.concatenate([np.asarray(d).flatten() for d in all_data])
    y_min = np.min(all_values)
    y_max = np.max(all_values)
    y_margin = (y_max - y_min) * 0.05  # 5% margin

    # X-axis: same range for all panels (0 to max horizon + context)
    max_horizon = max(horizons)
    x_max = CONTEXT_LEN + max_horizon  # 60 + 90 = 150

    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    print(f"  X-axis range: [0, {x_max}]")
    print(f"  Y-axis range: [{y_min - y_margin:.4f}, {y_max + y_margin:.4f}]")

    # Overall title
    fig.suptitle('Oracle vs Prior Sampling: CI Width Evolution Across Horizons\n'
                 'Comparing Two Regimes (Calm vs Pre-Crisis Anomaly)',
                fontsize=15, fontweight='bold', y=0.98)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color='#2E7D32', lw=2, linestyle='-',
               label='■ Regime 1 (Calm): Context'),
        Line2D([0], [0], color='#4CAF50', lw=2, linestyle='-',
               label='■ Regime 1 (Calm): Forecast'),
        Line2D([0], [0], color='#E65100', lw=2.5, linestyle='--',
               label='▬ Regime 2 (Pre-crisis): Context'),
        Line2D([0], [0], color='#FF9800', lw=2.5, linestyle='--',
               label='▬ Regime 2 (Pre-crisis): Forecast'),
        Patch(facecolor='#1E88E5', alpha=0.25, label='Oracle 90% CI'),
        Line2D([0], [0], color='#0D47A1', lw=1.5, label='Oracle p50'),
        Patch(facecolor='#E53935', alpha=0.25, label='Prior 90% CI'),
        Line2D([0], [0], color='#B71C1C', lw=1.5, label='Prior p50'),
    ]

    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=10,
               frameon=True, fancybox=True, shadow=True)

    # Save
    output_dir = Path('results/context60_baseline/visualizations/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'oracle_prior_multihorizon_overlay_2regimes.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"✓ Saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    print(f"{'='*80}")

    plt.close()


if __name__ == '__main__':
    main()
