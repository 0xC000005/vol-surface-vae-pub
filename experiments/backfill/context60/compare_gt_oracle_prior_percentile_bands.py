#!/usr/bin/env python3
"""
Ground Truth vs Oracle vs Prior Percentile Bands Comparison

Overlays percentile bands from three sources:
1. Ground truth - Empirical historical path distribution
2. Oracle model - Posterior sampling predictions
3. Prior model - Realistic sampling predictions

By pooling all normalized sequences from each source, we obtain the marginal
distribution of 90-day path progressions for comparison.

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
# Ground Truth Loading
# ============================================================================

def load_ground_truth_normalized():
    """Load ground truth and compute normalized percentile bands.

    Returns:
        dict with 'days', 'p05', 'p25', 'p50', 'p75', 'p95', 'n_sequences'
    """
    print("Loading ground truth data...")

    # Load surfaces
    gt_data = np.load("data/vol_surface_with_ret.npz")
    grid_row, grid_col = ATM_6M
    atm_6m = gt_data['surface'][:, grid_row, grid_col]

    print(f"  Loaded {len(atm_6m)} daily observations")

    # Extract sequences
    print(f"  Extracting sequences (context={CONTEXT_LEN}, horizon={HORIZON})...")
    total_len = CONTEXT_LEN + HORIZON
    n_sequences = len(atm_6m) - total_len + 1

    # Pre-allocate
    normalized_sequences = np.zeros((n_sequences, HORIZON))

    for i in range(n_sequences):
        context = atm_6m[i:i + CONTEXT_LEN]
        forecast = atm_6m[i + CONTEXT_LEN:i + total_len]

        # Normalize to context endpoint
        anchor_value = context[-1]
        normalized_sequences[i] = forecast - anchor_value

    print(f"  Extracted {n_sequences} sequences")

    # Compute percentiles
    print("  Computing percentiles...")
    percentiles = {
        'days': np.arange(1, HORIZON + 1),
        'p05': np.percentile(normalized_sequences, 5, axis=0),
        'p25': np.percentile(normalized_sequences, 25, axis=0),
        'p50': np.percentile(normalized_sequences, 50, axis=0),
        'p75': np.percentile(normalized_sequences, 75, axis=0),
        'p95': np.percentile(normalized_sequences, 95, axis=0),
        'n_sequences': n_sequences,
        'source': 'Ground Truth'
    }

    print(f"  Day 1 spread (p95-p05): {percentiles['p95'][0] - percentiles['p05'][0]:.4f}")
    print(f"  Day 90 spread (p95-p05): {percentiles['p95'][-1] - percentiles['p05'][-1]:.4f}")
    print()

    return percentiles


# ============================================================================
# Model Predictions Loading
# ============================================================================

def load_model_predictions_normalized(sampling_mode):
    """Load model predictions and compute normalized percentile bands.

    Uses only the p50 (median) trajectory from each prediction to compute
    the marginal distribution across contexts.

    Args:
        sampling_mode: 'oracle' or 'prior'

    Returns:
        dict with 'days', 'p05', 'p25', 'p50', 'p75', 'p95', 'n_sequences'
    """
    print(f"Loading {sampling_mode.upper()} predictions...")

    # Load predictions
    pred_file = (f"results/context60_baseline/predictions/teacher_forcing/"
                 f"{sampling_mode}/vae_tf_insample_h{HORIZON}.npz")
    pred_data = np.load(pred_file)

    surfaces = pred_data['surfaces']  # (N, 90, 3, 5, 5)
    indices = pred_data['indices']

    n_sequences = len(indices)
    grid_row, grid_col = ATM_6M

    print(f"  Loaded {n_sequences} prediction sequences")

    # Load ground truth for context endpoints
    gt_data = np.load("data/vol_surface_with_ret.npz")
    gt_surface = gt_data['surface'][:, grid_row, grid_col]

    print("  Extracting and normalizing p50 (median) trajectories...")

    # Extract only p50 from each sequence
    normalized_p50_trajectories = np.zeros((n_sequences, HORIZON))

    for i, idx in enumerate(indices):
        # Get context endpoint from ground truth
        context_start = max(0, idx - CONTEXT_LEN)
        context = gt_surface[context_start:idx]

        if len(context) < CONTEXT_LEN:
            # Skip sequences without full context
            continue

        anchor_value = context[-1]

        # Extract model's p50 prediction at ATM 6M
        model_p50 = surfaces[i, :, 1, grid_row, grid_col]  # (90,)

        # Normalize p50 trajectory
        normalized_p50_trajectories[i] = model_p50 - anchor_value

    print(f"  Normalized {n_sequences} p50 trajectories")

    # Compute marginal percentiles across p50 trajectories
    print("  Computing marginal percentiles across p50 paths...")

    percentiles = {
        'days': np.arange(1, HORIZON + 1),
        'p05': np.percentile(normalized_p50_trajectories, 5, axis=0),
        'p25': np.percentile(normalized_p50_trajectories, 25, axis=0),
        'p50': np.percentile(normalized_p50_trajectories, 50, axis=0),
        'p75': np.percentile(normalized_p50_trajectories, 75, axis=0),
        'p95': np.percentile(normalized_p50_trajectories, 95, axis=0),
        'n_sequences': n_sequences,
        'source': sampling_mode.capitalize()
    }

    print(f"  Day 1 spread (p95-p05): {percentiles['p95'][0] - percentiles['p05'][0]:.4f}")
    print(f"  Day 90 spread (p95-p05): {percentiles['p95'][-1] - percentiles['p05'][-1]:.4f}")
    print()

    return percentiles


# ============================================================================
# Comparison Statistics
# ============================================================================

def compute_comparison_statistics(gt_stats, oracle_stats, prior_stats):
    """Compute comparison metrics across the three sources.

    Args:
        gt_stats: dict from load_ground_truth_normalized
        oracle_stats: dict from load_model_predictions_normalized
        prior_stats: dict from load_model_predictions_normalized

    Returns:
        dict with comparison metrics
    """
    print("Computing comparison statistics...")

    # Band widths at key days
    days_to_check = [0, 44, 89]  # Day 1, 45, 90 (0-indexed)
    day_labels = [1, 45, 90]

    stats = {
        'days_checked': day_labels,
        'gt_widths': [],
        'oracle_widths': [],
        'prior_widths': [],
        'oracle_gt_ratios': [],
        'prior_gt_ratios': [],
    }

    for i, day_idx in enumerate(days_to_check):
        gt_width = gt_stats['p95'][day_idx] - gt_stats['p05'][day_idx]
        oracle_width = oracle_stats['p95'][day_idx] - oracle_stats['p05'][day_idx]
        prior_width = prior_stats['p95'][day_idx] - prior_stats['p05'][day_idx]

        stats['gt_widths'].append(gt_width)
        stats['oracle_widths'].append(oracle_width)
        stats['prior_widths'].append(prior_width)
        stats['oracle_gt_ratios'].append(oracle_width / gt_width)
        stats['prior_gt_ratios'].append(prior_width / gt_width)

        print(f"  Day {day_labels[i]}:")
        print(f"    GT width: {gt_width:.4f}")
        print(f"    Oracle width: {oracle_width:.4f} (ratio: {oracle_width/gt_width:.3f})")
        print(f"    Prior width: {prior_width:.4f} (ratio: {prior_width/gt_width:.3f})")

    # Spread growth rates
    gt_growth = (stats['gt_widths'][-1] / stats['gt_widths'][0] - 1) * 100
    oracle_growth = (stats['oracle_widths'][-1] / stats['oracle_widths'][0] - 1) * 100
    prior_growth = (stats['prior_widths'][-1] / stats['prior_widths'][0] - 1) * 100

    stats['gt_growth_pct'] = gt_growth
    stats['oracle_growth_pct'] = oracle_growth
    stats['prior_growth_pct'] = prior_growth

    print(f"\n  Spread growth (day 1 → 90):")
    print(f"    GT: {gt_growth:+.1f}%")
    print(f"    Oracle: {oracle_growth:+.1f}%")
    print(f"    Prior: {prior_growth:+.1f}%")
    print()

    return stats


# ============================================================================
# [PRIMARY] Pairwise Comparison Visualization
# ============================================================================

def compute_pairwise_statistics(stats1, stats2):
    """Compute comparison statistics for two sources.

    Args:
        stats1: dict with percentiles for source 1
        stats2: dict with percentiles for source 2

    Returns:
        dict with pairwise comparison metrics
    """
    days_to_check = [0, 44, 89]  # Day 1, 45, 90 (0-indexed)
    day_labels = [1, 45, 90]

    pairwise_stats = {
        'days_checked': day_labels,
        'widths1': [],
        'widths2': [],
        'ratios': [],
    }

    for day_idx in days_to_check:
        width1 = stats1['p95'][day_idx] - stats1['p05'][day_idx]
        width2 = stats2['p95'][day_idx] - stats2['p05'][day_idx]

        pairwise_stats['widths1'].append(width1)
        pairwise_stats['widths2'].append(width2)
        pairwise_stats['ratios'].append(width2 / width1)

    # Spread growth rates
    growth1 = (pairwise_stats['widths1'][-1] / pairwise_stats['widths1'][0] - 1) * 100
    growth2 = (pairwise_stats['widths2'][-1] / pairwise_stats['widths2'][0] - 1) * 100

    pairwise_stats['growth1_pct'] = growth1
    pairwise_stats['growth2_pct'] = growth2

    return pairwise_stats


def plot_pairwise_comparison(stats1, stats2, colors, labels, filename, output_dir):
    """Plot pairwise comparison of two sources.

    Args:
        stats1: dict from load_ground_truth_normalized or load_model_predictions_normalized
        stats2: dict from load_ground_truth_normalized or load_model_predictions_normalized
        colors: tuple of (color1, color2) for the two sources
        labels: tuple of (label1, label2) for the two sources
        filename: output filename (e.g., 'comparison_gt_vs_oracle.png')
        output_dir: Path object for output directory
    """
    print(f"  Generating {filename}...")

    fig, ax = plt.subplots(figsize=(16, 9))

    days = stats1['days']
    color1, color2 = colors
    label1, label2 = labels

    # Compute pairwise statistics
    pairwise_stats = compute_pairwise_statistics(stats1, stats2)

    # ========================================================================
    # Source 1
    # ========================================================================

    # Outer band: p05-p95
    ax.fill_between(days, stats1['p05'], stats1['p95'],
                    color=color1, alpha=0.2, label=f'{label1} p05-p95',
                    zorder=1, edgecolor='none')

    # Inner band: p25-p75
    ax.fill_between(days, stats1['p25'], stats1['p75'],
                    color=color1, alpha=0.35, label=f'{label1} p25-p75 (IQR)',
                    zorder=2, edgecolor='none')

    # Median line
    ax.plot(days, stats1['p50'], color=color1, linewidth=3,
            label=f'{label1} p50 (median)', zorder=7, linestyle='-', alpha=0.9)

    # ========================================================================
    # Source 2
    # ========================================================================

    # Outer band: p05-p95
    ax.fill_between(days, stats2['p05'], stats2['p95'],
                    color=color2, alpha=0.2, label=f'{label2} p05-p95',
                    zorder=3, edgecolor='none')

    # Inner band: p25-p75
    ax.fill_between(days, stats2['p25'], stats2['p75'],
                    color=color2, alpha=0.35, label=f'{label2} p25-p75 (IQR)',
                    zorder=4, edgecolor='none')

    # Median line
    ax.plot(days, stats2['p50'], color=color2, linewidth=3,
            label=f'{label2} p50 (median)', zorder=8, linestyle='-', alpha=0.9)

    # ========================================================================
    # Reference Elements
    # ========================================================================

    # Anchor line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2,
               alpha=0.6, label='Context endpoint (anchor)', zorder=10)

    # ========================================================================
    # Annotations
    # ========================================================================

    ax.set_xlabel('Days After Context End', fontsize=14, weight='bold')
    ax.set_ylabel('Normalized IV (relative to context endpoint)', fontsize=14, weight='bold')
    ax.set_title(f'{label1} vs {label2}: Marginal Path Distribution Comparison\n'
                 f'60-day context, 90-day forecast',
                 fontsize=16, weight='bold', pad=15)

    ax.legend(loc='upper left', fontsize=11,
              framealpha=0.95, edgecolor='black', fancybox=True)

    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8, zorder=0)

    # ========================================================================
    # Statistics Box
    # ========================================================================

    stats_text = (
        f'Band Width (p95-p05):\n'
        f'\n'
        f'Day 1:\n'
        f'  {label1}: {pairwise_stats["widths1"][0]:.4f}\n'
        f'  {label2}: {pairwise_stats["widths2"][0]:.4f}\n'
        f'  Ratio: {pairwise_stats["ratios"][0]:.3f}×\n'
        f'\n'
        f'Day 45:\n'
        f'  {label1}: {pairwise_stats["widths1"][1]:.4f}\n'
        f'  {label2}: {pairwise_stats["widths2"][1]:.4f}\n'
        f'  Ratio: {pairwise_stats["ratios"][1]:.3f}×\n'
        f'\n'
        f'Day 90:\n'
        f'  {label1}: {pairwise_stats["widths1"][2]:.4f}\n'
        f'  {label2}: {pairwise_stats["widths2"][2]:.4f}\n'
        f'  Ratio: {pairwise_stats["ratios"][2]:.3f}×\n'
        f'\n'
        f'Spread Growth:\n'
        f'  {label1}: {pairwise_stats["growth1_pct"]:+.1f}%\n'
        f'  {label2}: {pairwise_stats["growth2_pct"]:+.1f}%'
    )

    props = dict(boxstyle='round,pad=0.8', facecolor='wheat',
                 alpha=0.95, edgecolor='black', linewidth=2)
    ax.text(0.98, 0.03, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()

    # Save
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"    File size: {size_mb:.1f} MB")

    plt.close()


def plot_overlaid_percentile_bands(gt_stats, oracle_stats, prior_stats, comp_stats, output_dir):
    """Plot three percentile band sources overlaid.

    Args:
        gt_stats: dict from load_ground_truth_normalized
        oracle_stats: dict from load_model_predictions_normalized
        prior_stats: dict from load_model_predictions_normalized
        comp_stats: dict from compute_comparison_statistics
        output_dir: Path object for output directory
    """
    print("[PRIMARY] Generating overlaid percentile bands plot...")

    fig, ax = plt.subplots(figsize=(18, 10))

    days = gt_stats['days']

    # ========================================================================
    # Ground Truth (Blue, Solid Fills)
    # ========================================================================

    # Outer band: p05-p95
    ax.fill_between(days, gt_stats['p05'], gt_stats['p95'],
                    color='blue', alpha=0.15, label='GT p05-p95',
                    zorder=1, edgecolor='none')

    # Inner band: p25-p75
    ax.fill_between(days, gt_stats['p25'], gt_stats['p75'],
                    color='blue', alpha=0.3, label='GT p25-p75 (IQR)',
                    zorder=2, edgecolor='none')

    # Median line
    ax.plot(days, gt_stats['p50'], color='darkblue', linewidth=2.5,
            label='GT p50 (median)', zorder=7, linestyle='-')

    # ========================================================================
    # Oracle (Green, Hatched Fills)
    # ========================================================================

    # Outer band: p05-p95
    ax.fill_between(days, oracle_stats['p05'], oracle_stats['p95'],
                    color='green', alpha=0.12, hatch='///', label='Oracle p05-p95',
                    zorder=3, edgecolor='green', linewidth=0.3)

    # Inner band: p25-p75
    ax.fill_between(days, oracle_stats['p25'], oracle_stats['p75'],
                    color='green', alpha=0.25, hatch='///', label='Oracle p25-p75 (IQR)',
                    zorder=4, edgecolor='green', linewidth=0.3)

    # Median line
    ax.plot(days, oracle_stats['p50'], color='darkgreen', linewidth=2.5,
            label='Oracle p50 (median)', zorder=8, linestyle='-')

    # ========================================================================
    # Prior (Red, Hatched Fills)
    # ========================================================================

    # Outer band: p05-p95
    ax.fill_between(days, prior_stats['p05'], prior_stats['p95'],
                    color='red', alpha=0.12, hatch='\\\\\\', label='Prior p05-p95',
                    zorder=5, edgecolor='red', linewidth=0.3)

    # Inner band: p25-p75
    ax.fill_between(days, prior_stats['p25'], prior_stats['p75'],
                    color='red', alpha=0.25, hatch='\\\\\\', label='Prior p25-p75 (IQR)',
                    zorder=6, edgecolor='red', linewidth=0.3)

    # Median line
    ax.plot(days, prior_stats['p50'], color='darkred', linewidth=2.5,
            label='Prior p50 (median)', zorder=9, linestyle='-')

    # ========================================================================
    # Reference Elements
    # ========================================================================

    # Anchor line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2,
               alpha=0.6, label='Context endpoint (anchor)', zorder=10)

    # ========================================================================
    # Annotations
    # ========================================================================

    ax.set_xlabel('Days After Context End', fontsize=14, weight='bold')
    ax.set_ylabel('Normalized IV (relative to context endpoint)', fontsize=14, weight='bold')
    ax.set_title('Ground Truth vs Oracle vs Prior: Marginal Path Distribution Comparison\n'
                 f'60-day context, 90-day forecast (GT: {gt_stats["n_sequences"]:,} sequences)',
                 fontsize=16, weight='bold', pad=15)

    # Legend - organized by source
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=10,
              framealpha=0.95, edgecolor='black', fancybox=True, ncol=3)

    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8, zorder=0)

    # ========================================================================
    # Statistics Box
    # ========================================================================

    stats_text = (
        f'Band Width (p95-p05):\n'
        f'\n'
        f'Day 1:\n'
        f'  GT: {comp_stats["gt_widths"][0]:.4f}\n'
        f'  Oracle: {comp_stats["oracle_widths"][0]:.4f} (×{comp_stats["oracle_gt_ratios"][0]:.2f})\n'
        f'  Prior: {comp_stats["prior_widths"][0]:.4f} (×{comp_stats["prior_gt_ratios"][0]:.2f})\n'
        f'\n'
        f'Day 45:\n'
        f'  GT: {comp_stats["gt_widths"][1]:.4f}\n'
        f'  Oracle: {comp_stats["oracle_widths"][1]:.4f} (×{comp_stats["oracle_gt_ratios"][1]:.2f})\n'
        f'  Prior: {comp_stats["prior_widths"][1]:.4f} (×{comp_stats["prior_gt_ratios"][1]:.2f})\n'
        f'\n'
        f'Day 90:\n'
        f'  GT: {comp_stats["gt_widths"][2]:.4f}\n'
        f'  Oracle: {comp_stats["oracle_widths"][2]:.4f} (×{comp_stats["oracle_gt_ratios"][2]:.2f})\n'
        f'  Prior: {comp_stats["prior_widths"][2]:.4f} (×{comp_stats["prior_gt_ratios"][2]:.2f})\n'
        f'\n'
        f'Spread Growth (Day 1→90):\n'
        f'  GT: {comp_stats["gt_growth_pct"]:+.1f}%\n'
        f'  Oracle: {comp_stats["oracle_growth_pct"]:+.1f}%\n'
        f'  Prior: {comp_stats["prior_growth_pct"]:+.1f}%'
    )

    props = dict(boxstyle='round,pad=0.8', facecolor='wheat',
                 alpha=0.95, edgecolor='black', linewidth=2)
    ax.text(0.98, 0.03, stats_text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='bottom',
            horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()

    # Save
    filename = 'gt_oracle_prior_percentile_bands_comparison.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()


# ============================================================================
# Report Generation
# ============================================================================

def generate_comparison_report(gt_stats, oracle_stats, prior_stats, comp_stats, output_dir):
    """Generate CSV and text report.

    Args:
        gt_stats: dict from load_ground_truth_normalized
        oracle_stats: dict from load_model_predictions_normalized
        prior_stats: dict from load_model_predictions_normalized
        comp_stats: dict from compute_comparison_statistics
        output_dir: Path object for output directory
    """
    print("Generating comparison report...")

    # CSV: Day-by-day percentiles for all three sources
    df = pd.DataFrame({
        'Day': gt_stats['days'],
        'GT_p05': gt_stats['p05'],
        'GT_p25': gt_stats['p25'],
        'GT_p50': gt_stats['p50'],
        'GT_p75': gt_stats['p75'],
        'GT_p95': gt_stats['p95'],
        'Oracle_p05': oracle_stats['p05'],
        'Oracle_p25': oracle_stats['p25'],
        'Oracle_p50': oracle_stats['p50'],
        'Oracle_p75': oracle_stats['p75'],
        'Oracle_p95': oracle_stats['p95'],
        'Prior_p05': prior_stats['p05'],
        'Prior_p25': prior_stats['p25'],
        'Prior_p50': prior_stats['p50'],
        'Prior_p75': prior_stats['p75'],
        'Prior_p95': prior_stats['p95'],
    })

    csv_path = output_dir / 'percentile_comparison_statistics.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path}")

    # Text report
    report = []
    report.append("=" * 80)
    report.append("MODEL vs EMPIRICAL COMPARISON REPORT")
    report.append("Ground Truth vs Oracle vs Prior Percentile Bands")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    report.append("OBJECTIVE:")
    report.append("  Compare marginal distributions of 90-day path progressions from:")
    report.append("    1. Ground truth (empirical historical paths)")
    report.append("    2. Oracle model (posterior sampling)")
    report.append("    3. Prior model (realistic sampling)")
    report.append("")

    report.append("DATA:")
    report.append(f"  Ground truth sequences: {gt_stats['n_sequences']:,}")
    report.append(f"  Oracle sequences: {oracle_stats['n_sequences']:,}")
    report.append(f"  Prior sequences: {prior_stats['n_sequences']:,}")
    report.append(f"  Context length: {CONTEXT_LEN} days")
    report.append(f"  Forecast horizon: {HORIZON} days")
    report.append("")

    report.append("KEY FINDINGS:")
    report.append("")

    report.append("1. UNCERTAINTY PREDICTION:")
    report.append(f"   Day 1 band widths:")
    report.append(f"     Ground truth: {comp_stats['gt_widths'][0]:.4f}")
    report.append(f"     Oracle: {comp_stats['oracle_widths'][0]:.4f} (×{comp_stats['oracle_gt_ratios'][0]:.2f})")
    report.append(f"     Prior: {comp_stats['prior_widths'][0]:.4f} (×{comp_stats['prior_gt_ratios'][0]:.2f})")
    report.append("")
    report.append(f"   Day 90 band widths:")
    report.append(f"     Ground truth: {comp_stats['gt_widths'][2]:.4f}")
    report.append(f"     Oracle: {comp_stats['oracle_widths'][2]:.4f} (×{comp_stats['oracle_gt_ratios'][2]:.2f})")
    report.append(f"     Prior: {comp_stats['prior_widths'][2]:.4f} (×{comp_stats['prior_gt_ratios'][2]:.2f})")
    report.append("")

    # Diagnosis
    if comp_stats['prior_gt_ratios'][2] < 0.8:
        report.append("   → UNDER-PREDICTION: Models significantly underestimate long-horizon uncertainty")
    elif comp_stats['prior_gt_ratios'][2] > 1.2:
        report.append("   → OVER-PREDICTION: Models overestimate long-horizon uncertainty")
    else:
        report.append("   → WELL-CALIBRATED: Models reasonably match empirical uncertainty")
    report.append("")

    report.append("2. SPREAD GROWTH PATTERNS:")
    report.append(f"   Ground truth: {comp_stats['gt_growth_pct']:+.1f}% (day 1 → 90)")
    report.append(f"   Oracle: {comp_stats['oracle_growth_pct']:+.1f}% (day 1 → 90)")
    report.append(f"   Prior: {comp_stats['prior_growth_pct']:+.1f}% (day 1 → 90)")
    report.append("")

    if comp_stats['prior_growth_pct'] < comp_stats['gt_growth_pct'] * 0.5:
        report.append("   → OVER-CONVERGENCE: Models show excessive mean reversion")
    elif comp_stats['prior_growth_pct'] > comp_stats['gt_growth_pct'] * 1.5:
        report.append("   → OVER-DIVERGENCE: Models show excessive uncertainty growth")
    else:
        report.append("   → REALISTIC: Models match empirical spread growth patterns")
    report.append("")

    report.append("3. ORACLE vs PRIOR:")
    oracle_wider_pct = (comp_stats['oracle_gt_ratios'][2] / comp_stats['prior_gt_ratios'][2] - 1) * 100
    report.append(f"   Oracle bands are {abs(oracle_wider_pct):.1f}% {'wider' if oracle_wider_pct > 0 else 'narrower'} than prior at day 90")
    report.append("   → Oracle sees target data, should have more realistic uncertainty")
    report.append("")

    report.append("INTERPRETATION:")
    report.append("  The marginal distribution comparison reveals whether models learn realistic")
    report.append("  patterns of uncertainty growth and mean reversion from empirical data.")
    report.append("")
    report.append("  - Blue bands: Empirical baseline from 23 years of SPX volatility")
    report.append("  - Green bands: Oracle predictions (theoretical upper bound)")
    report.append("  - Red bands: Prior predictions (realistic deployment scenario)")
    report.append("")
    report.append("  If model bands are narrower than ground truth, they are over-confident.")
    report.append("  If model bands grow slower than ground truth, they over-predict convergence.")
    report.append("")

    report.append("=" * 80)

    report_text = "\n".join(report)

    report_path = output_dir / 'model_vs_empirical_comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  ✓ Saved: {report_path}")

    # Also print to console
    print("\n" + report_text)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run complete GT vs Oracle vs Prior comparison."""

    print("=" * 80)
    print("GROUND TRUTH vs ORACLE vs PRIOR COMPARISON")
    print("Marginal Path Distribution Percentile Bands")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all three sources
    gt_stats = load_ground_truth_normalized()
    oracle_stats = load_model_predictions_normalized('oracle')
    prior_stats = load_model_predictions_normalized('prior')

    # Compute comparison statistics
    comp_stats = compute_comparison_statistics(gt_stats, oracle_stats, prior_stats)

    # Generate pairwise comparison visualizations
    print("[PRIMARY] Generating pairwise comparison plots...")

    plot_pairwise_comparison(
        gt_stats, oracle_stats,
        colors=('blue', 'green'),
        labels=('Ground Truth', 'Oracle'),
        filename='comparison_gt_vs_oracle.png',
        output_dir=OUTPUT_DIR
    )

    plot_pairwise_comparison(
        gt_stats, prior_stats,
        colors=('blue', 'red'),
        labels=('Ground Truth', 'Prior'),
        filename='comparison_gt_vs_prior.png',
        output_dir=OUTPUT_DIR
    )

    plot_pairwise_comparison(
        oracle_stats, prior_stats,
        colors=('green', 'red'),
        labels=('Oracle', 'Prior'),
        filename='comparison_oracle_vs_prior.png',
        output_dir=OUTPUT_DIR
    )
    print()

    # Generate report
    generate_comparison_report(gt_stats, oracle_stats, prior_stats, comp_stats, OUTPUT_DIR)
    print()

    print("=" * 80)
    print("COMPARISON COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
