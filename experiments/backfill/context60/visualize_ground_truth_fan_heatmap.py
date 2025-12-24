#!/usr/bin/env python3
"""
Ground Truth Historical Path Fan Heat Plot

Creates a density heatmap showing how 5,762 historical volatility paths progress
over 90 days after being normalized to a common anchor point (end of 60-day context).

The color intensity reveals "where most paths actually look like" - preserving
full distribution information that simple percentile bands would lose.

This provides the empirical baseline for comparing against VAE model behavior.

Author: Generated with Claude Code
Date: 2025-12-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Constants
# ============================================================================

CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)  # Grid point: K/S=1.00, 6-month maturity
OUTPUT_DIR = Path("results/context60_baseline/analysis/ground_truth_paths")


# ============================================================================
# Data Loading and Extraction
# ============================================================================

def load_ground_truth_data():
    """Load ground truth volatility surfaces and dates.

    Returns:
        tuple: (atm_6m array (5822,), dates)
    """
    print("Loading ground truth data...")

    # Load surfaces
    gt_data = np.load("data/vol_surface_with_ret.npz")

    # Load dates
    dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    dates = pd.to_datetime(dates_df["date"].values)

    # Extract ATM 6M values
    grid_row, grid_col = ATM_6M
    atm_6m = gt_data['surface'][:, grid_row, grid_col]

    print(f"  Loaded {len(atm_6m)} daily observations")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  ATM 6M IV range: [{atm_6m.min():.4f}, {atm_6m.max():.4f}]")
    print()

    return atm_6m, dates


def extract_normalized_sequences(atm_6m, context_len=60, horizon=90):
    """Extract all valid sequences and normalize to context endpoint.

    Args:
        atm_6m: (N,) array of ATM 6M volatility values
        context_len: Number of context days
        horizon: Number of forecast days

    Returns:
        tuple: (normalized_sequences (n_seq, horizon), context_endpoints (n_seq,))
    """
    print(f"Extracting sequences (context={context_len}, horizon={horizon})...")

    total_len = context_len + horizon
    n_sequences = len(atm_6m) - total_len + 1

    if n_sequences <= 0:
        raise ValueError(f"Not enough data: need {total_len} days, have {len(atm_6m)}")

    # Pre-allocate
    normalized_sequences = np.zeros((n_sequences, horizon))
    context_endpoints = np.zeros(n_sequences)

    for i in range(n_sequences):
        # Extract context and forecast
        context = atm_6m[i:i + context_len]
        forecast = atm_6m[i + context_len:i + total_len]

        # Normalize: anchor to context endpoint (set it to 0)
        anchor_value = context[-1]
        normalized_forecast = forecast - anchor_value

        normalized_sequences[i] = normalized_forecast
        context_endpoints[i] = anchor_value

    print(f"  Extracted {n_sequences} sequences")
    print(f"  Context endpoints range: [{context_endpoints.min():.4f}, {context_endpoints.max():.4f}]")
    print(f"  Normalized forecast range: [{normalized_sequences.min():.4f}, {normalized_sequences.max():.4f}]")
    print()

    return normalized_sequences, context_endpoints


# ============================================================================
# Statistics Computation
# ============================================================================

def compute_path_statistics(normalized_sequences):
    """Compute percentiles and mean reversion metrics.

    Args:
        normalized_sequences: (n_seq, horizon) array

    Returns:
        dict with statistics
    """
    print("Computing path statistics...")

    horizon = normalized_sequences.shape[1]

    # Percentiles at each day
    p05 = np.percentile(normalized_sequences, 5, axis=0)
    p25 = np.percentile(normalized_sequences, 25, axis=0)
    p50 = np.percentile(normalized_sequences, 50, axis=0)
    p75 = np.percentile(normalized_sequences, 75, axis=0)
    p95 = np.percentile(normalized_sequences, 95, axis=0)
    mean = np.mean(normalized_sequences, axis=0)

    # Spread metrics
    spread_day1 = p95[0] - p05[0]
    spread_day90 = p95[-1] - p05[-1]
    spread_change_pct = (spread_day90 / spread_day1 - 1) * 100

    # Distribution at day 90
    endpoints = normalized_sequences[:, -1]
    pct_above_zero = (endpoints > 0).mean() * 100
    pct_below_zero = (endpoints < 0).mean() * 100

    # Mean reversion metrics
    # Half-life: day when |mean| falls to 50% of day 1 value
    abs_mean = np.abs(mean)
    threshold = abs_mean[0] / 2
    half_life_days = np.where(abs_mean < threshold)[0]
    half_life = half_life_days[0] if len(half_life_days) > 0 else horizon

    stats = {
        'days': np.arange(1, horizon + 1),
        'p05': p05,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p95': p95,
        'mean': mean,
        'spread_day1': spread_day1,
        'spread_day90': spread_day90,
        'spread_change_pct': spread_change_pct,
        'pct_above_zero': pct_above_zero,
        'pct_below_zero': pct_below_zero,
        'half_life': half_life,
        'mean_at_day90': mean[-1],
    }

    print(f"  p05-p95 spread: Day 1 = {spread_day1:.4f}, Day 90 = {spread_day90:.4f}")
    print(f"  Spread change: {spread_change_pct:+.1f}%")
    print(f"  Day 90 distribution: {pct_above_zero:.1f}% above anchor, {pct_below_zero:.1f}% below")
    print(f"  Mean reversion half-life: {half_life} days")
    print(f"  Mean deviation at day 90: {mean[-1]:.4f}")
    print()

    return stats


# ============================================================================
# [PRIMARY] Density Heatmap Visualization
# ============================================================================

def plot_fan_heatmap(normalized_sequences, statistics, output_dir):
    """Create density heatmap showing where most historical paths go.

    Args:
        normalized_sequences: (n_seq, horizon) array
        statistics: dict from compute_path_statistics
        output_dir: Path object for output directory
    """
    print("[PRIMARY] Generating fan heat plot...")

    n_sequences, horizon = normalized_sequences.shape

    # Flatten to (day, value) pairs for density plot
    days_all = []
    values_all = []

    for seq in normalized_sequences:
        days_all.extend(np.arange(1, horizon + 1))
        values_all.extend(seq)

    print(f"  Total data points: {len(days_all):,}")

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))

    # Create density heatmap using hexbin
    hexbin = ax.hexbin(days_all, values_all, gridsize=60,
                       cmap='YlOrRd', mincnt=1, linewidths=0.2,
                       edgecolors='face', alpha=0.9)

    # Colorbar
    cbar = plt.colorbar(hexbin, ax=ax, pad=0.02)
    cbar.set_label('Path Density (count)', fontsize=13, weight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Overlay percentile lines (thin, for reference)
    days = statistics['days']
    ax.plot(days, statistics['p50'], 'k-', linewidth=2.5, alpha=0.8,
            label='p50 (median)', zorder=10)
    ax.plot(days, statistics['p05'], 'b--', linewidth=2, alpha=0.7,
            label='p05', zorder=10)
    ax.plot(days, statistics['p95'], 'r--', linewidth=2, alpha=0.7,
            label='p95', zorder=10)

    # Reference line: y=0 (anchor point)
    ax.axhline(y=0, color='green', linestyle='-', linewidth=2.5,
               alpha=0.8, label='Context endpoint (anchor)', zorder=5)

    # Annotations
    ax.set_xlabel('Days After Context End', fontsize=14, weight='bold')
    ax.set_ylabel('Normalized IV (relative to context endpoint)', fontsize=14, weight='bold')
    ax.set_title(f'Ground Truth Historical Path Distribution\n'
                 f'{n_sequences:,} Sequences (60-day context + 90-day forecast, 2000-2023)',
                 fontsize=16, weight='bold', pad=15)

    ax.legend(loc='upper left', fontsize=12, framealpha=0.95,
              edgecolor='black', fancybox=True)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8, zorder=1)

    # Statistics box
    stats_text = (
        f'N sequences: {n_sequences:,}\n'
        f'N data points: {len(days_all):,}\n'
        f'\n'
        f'Spread (p95-p05):\n'
        f'  Day 1: {statistics["spread_day1"]:.4f}\n'
        f'  Day 90: {statistics["spread_day90"]:.4f}\n'
        f'  Change: {statistics["spread_change_pct"]:+.1f}%\n'
        f'\n'
        f'Day 90 Distribution:\n'
        f'  Above anchor: {statistics["pct_above_zero"]:.1f}%\n'
        f'  Below anchor: {statistics["pct_below_zero"]:.1f}%\n'
        f'\n'
        f'Mean Reversion:\n'
        f'  Half-life: {statistics["half_life"]} days\n'
        f'  Mean @ day 90: {statistics["mean_at_day90"]:+.4f}'
    )

    props = dict(boxstyle='round,pad=0.8', facecolor='wheat',
                 alpha=0.95, edgecolor='black', linewidth=2)
    ax.text(0.98, 0.03, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom',
            horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()

    # Save
    filename = 'ground_truth_fan_heatmap.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    # Report file size
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()


# ============================================================================
# [OPTIONAL] Percentile Bands Only
# ============================================================================

def plot_percentile_bands_only(statistics, output_dir):
    """Create simpler view with just percentile bands.

    Args:
        statistics: dict from compute_path_statistics
        output_dir: Path object for output directory
    """
    print("[OPTIONAL] Generating percentile bands plot...")

    fig, ax = plt.subplots(figsize=(16, 9))

    days = statistics['days']

    # Plot bands
    ax.fill_between(days, statistics['p05'], statistics['p95'],
                    alpha=0.25, color='red', label='5-95%')
    ax.fill_between(days, statistics['p25'], statistics['p75'],
                    alpha=0.35, color='orange', label='25-75%')
    ax.plot(days, statistics['p50'], color='darkred',
            linewidth=2.5, label='Median')
    ax.plot(days, statistics['mean'], color='blue',
            linewidth=2, linestyle='--', label='Mean')

    # Reference line
    ax.axhline(y=0, color='green', linestyle='-', linewidth=2,
               alpha=0.7, label='Context endpoint (anchor)')

    # Annotations
    ax.set_xlabel('Days After Context End', fontsize=14, weight='bold')
    ax.set_ylabel('Normalized IV (relative to context endpoint)', fontsize=14, weight='bold')
    ax.set_title('Ground Truth Path Percentile Bands\n'
                 '60-day context, 90-day forecast (5,762 sequences)',
                 fontsize=16, weight='bold', pad=15)
    ax.legend(loc='best', fontsize=12, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)

    plt.tight_layout()

    # Save
    filename = 'ground_truth_percentile_bands.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(statistics, n_sequences, output_dir):
    """Generate CSV summary and text report.

    Args:
        statistics: dict from compute_path_statistics
        n_sequences: int, number of sequences analyzed
        output_dir: Path object for output directory
    """
    print("Generating summary report...")

    # CSV: Percentiles at each day
    df = pd.DataFrame({
        'Day': statistics['days'],
        'p05': statistics['p05'],
        'p25': statistics['p25'],
        'p50': statistics['p50'],
        'p75': statistics['p75'],
        'p95': statistics['p95'],
        'mean': statistics['mean'],
        'spread_p95_p05': statistics['p95'] - statistics['p05'],
    })

    csv_path = output_dir / 'path_statistics_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path}")

    # Text report
    report = []
    report.append("=" * 80)
    report.append("GROUND TRUTH HISTORICAL PATH ANALYSIS REPORT")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    report.append("OBJECTIVE:")
    report.append("  Visualize how historical volatility paths empirically progress over 90 days")
    report.append("  after being normalized to a common anchor point (end of 60-day context).")
    report.append("  This provides the empirical baseline for VAE model comparison.")
    report.append("")

    report.append("DATA:")
    report.append(f"  Sequences analyzed: {n_sequences:,}")
    report.append(f"  Context length: {CONTEXT_LEN} days")
    report.append(f"  Forecast horizon: {HORIZON} days")
    report.append(f"  Grid point: ATM 6M (K/S=1.00, 6-month maturity)")
    report.append(f"  Date range: 2000-2023")
    report.append("")

    report.append("KEY FINDINGS:")
    report.append("")

    report.append("1. SPREAD EVOLUTION:")
    report.append(f"   - Day 1 spread (p95-p05): {statistics['spread_day1']:.4f}")
    report.append(f"   - Day 90 spread (p95-p05): {statistics['spread_day90']:.4f}")
    report.append(f"   - Change: {statistics['spread_change_pct']:+.1f}%")
    if statistics['spread_change_pct'] > 0:
        report.append("   → Uncertainty INCREASES with horizon (paths diverge)")
    else:
        report.append("   → Uncertainty DECREASES with horizon (paths converge)")
    report.append("")

    report.append("2. DISTRIBUTION AT DAY 90:")
    report.append(f"   - Paths above anchor (0): {statistics['pct_above_zero']:.1f}%")
    report.append(f"   - Paths below anchor (0): {statistics['pct_below_zero']:.1f}%")
    if statistics['pct_above_zero'] > statistics['pct_below_zero']:
        report.append("   → Upward bias: volatility tends to increase from context level")
    else:
        report.append("   → Downward bias: volatility tends to decrease from context level")
    report.append("")

    report.append("3. MEAN REVERSION:")
    report.append(f"   - Mean deviation at day 90: {statistics['mean_at_day90']:+.4f}")
    report.append(f"   - Half-life: {statistics['half_life']} days")
    if abs(statistics['mean_at_day90']) < 0.01:
        report.append("   → Strong mean reversion: paths return to anchor level")
    else:
        report.append("   → Weak mean reversion: paths drift from anchor level")
    report.append("")

    report.append("INTERPRETATION:")
    report.append("  The density heatmap reveals WHERE MOST PATHS go by showing concentration")
    report.append("  through color intensity. High-density regions (bright colors) indicate where")
    report.append("  historical volatility paths empirically tend to progress.")
    report.append("")
    report.append("  This empirical distribution can be compared against VAE model predictions")
    report.append("  to evaluate whether the model learns realistic mean reversion patterns.")
    report.append("")

    report.append("=" * 80)

    report_text = "\n".join(report)

    report_path = output_dir / 'ground_truth_path_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  ✓ Saved: {report_path}")

    # Also print to console
    print("\n" + report_text)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run complete ground truth path analysis."""

    print("=" * 80)
    print("GROUND TRUTH HISTORICAL PATH FAN HEAT PLOT")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    atm_6m, dates = load_ground_truth_data()

    # Extract and normalize sequences
    normalized_sequences, context_endpoints = extract_normalized_sequences(
        atm_6m, CONTEXT_LEN, HORIZON
    )

    # Compute statistics
    statistics = compute_path_statistics(normalized_sequences)

    # [PRIMARY] Generate density heatmap
    plot_fan_heatmap(normalized_sequences, statistics, OUTPUT_DIR)
    print()

    # [OPTIONAL] Generate percentile bands plot
    plot_percentile_bands_only(statistics, OUTPUT_DIR)
    print()

    # Generate report
    generate_report(statistics, len(normalized_sequences), OUTPUT_DIR)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
