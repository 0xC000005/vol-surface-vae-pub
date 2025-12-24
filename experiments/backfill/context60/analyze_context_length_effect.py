#!/usr/bin/env python3
"""
Experiment 1: Context Length Stratification Analysis

Tests hypothesis: Context randomization causes day-1 over-dispersion via epistemic uncertainty.

Strategy:
- Stratify test sequences by their actual context length [10, 20, 30, 40, 50, 60]
- For each stratum, compute p50 marginal spread (p95-p05)
- Compare across context lengths and horizons

Decision Criteria:
- If p50 spread for context=60 is <0.04 (vs current 0.0858):
  ✅ FIX: Use fixed context=60 in next training
- If spread is still high even for context=60:
  ❌ Context randomization is NOT the main cause

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
OUTPUT_DIR = Path("results/context60_baseline/analysis/preliminary_experiments/context_length_stratification")
CONTEXT_LENGTHS = [10, 20, 30, 40, 50, 60]  # Training used randomized context lengths


# ============================================================================
# Data Loading
# ============================================================================

def load_predictions_and_indices():
    """Load oracle predictions and sequence indices.

    Returns:
        dict with 'surfaces', 'indices'
    """
    print("Loading oracle predictions...")

    pred_file = (f"results/context60_baseline/predictions/teacher_forcing/"
                 f"oracle/vae_tf_insample_h{HORIZON}.npz")
    pred_data = np.load(pred_file)

    surfaces = pred_data['surfaces']  # (N, 90, 3, 5, 5)
    indices = pred_data['indices']     # (N,)

    print(f"  Loaded {len(indices)} prediction sequences")
    print(f"  Surfaces shape: {surfaces.shape}")

    return {'surfaces': surfaces, 'indices': indices}


def load_ground_truth():
    """Load ground truth volatility surfaces.

    Returns:
        numpy array (T, 5, 5)
    """
    print("Loading ground truth...")

    gt_data = np.load("data/vol_surface_with_ret.npz")
    gt_surface = gt_data['surface']  # (T, 5, 5)

    print(f"  Loaded {len(gt_surface)} daily observations")

    return gt_surface


# ============================================================================
# Context Length Extraction
# ============================================================================

def extract_context_length_for_each_sequence(indices, gt_surface, context_len=60):
    """Determine actual context length for each test sequence.

    Context length is limited by data availability at start of time series.

    Args:
        indices: (N,) - sequence indices (end of context)
        gt_surface: (T, 5, 5) - ground truth surface
        context_len: Maximum context length (60 for context60)

    Returns:
        numpy array (N,) - actual context length for each sequence
    """
    print(f"Extracting actual context lengths...")

    actual_context_lengths = np.zeros(len(indices), dtype=int)

    for i, idx in enumerate(indices):
        # Context would be data[idx-context_len:idx]
        # But if idx < context_len, context is shorter
        actual_len = min(idx, context_len)
        actual_context_lengths[i] = actual_len

    # Count sequences per context length
    unique_lens, counts = np.unique(actual_context_lengths, return_counts=True)
    print(f"  Context length distribution:")
    for length, count in zip(unique_lens, counts):
        print(f"    {length} days: {count} sequences ({count/len(indices)*100:.1f}%)")

    return actual_context_lengths


# ============================================================================
# Stratification
# ============================================================================

def stratify_by_context_length(actual_context_lengths, surfaces, gt_surface, indices):
    """Group sequences by context length.

    Args:
        actual_context_lengths: (N,) - actual context length for each sequence
        surfaces: (N, 90, 3, 5, 5) - oracle predictions
        gt_surface: (T, 5, 5) - ground truth
        indices: (N,) - sequence indices

    Returns:
        dict: {context_length: {'p50_predictions': (M, 90), 'context_endpoints': (M,)}}
    """
    print("Stratifying by context length...")

    grid_row, grid_col = ATM_6M
    stratified = {}

    for context_len in CONTEXT_LENGTHS:
        # Find sequences with this context length
        mask = actual_context_lengths == context_len
        n_sequences = mask.sum()

        if n_sequences == 0:
            print(f"  Context length {context_len}: 0 sequences (skipping)")
            continue

        print(f"  Context length {context_len}: {n_sequences} sequences")

        # Extract p50 predictions for these sequences
        p50_predictions = surfaces[mask, :, 1, grid_row, grid_col]  # (M, 90)

        # Extract context endpoints for normalization
        context_endpoints = np.zeros(n_sequences)
        for i, idx in enumerate(indices[mask]):
            context_start = max(0, idx - CONTEXT_LEN)
            context = gt_surface[context_start:idx, grid_row, grid_col]
            context_endpoints[i] = context[-1]

        # Normalize p50 predictions to context endpoint
        normalized_p50 = p50_predictions - context_endpoints[:, None]

        stratified[context_len] = {
            'p50_predictions': normalized_p50,
            'context_endpoints': context_endpoints,
            'n_sequences': n_sequences
        }

    return stratified


# ============================================================================
# Analysis
# ============================================================================

def compute_p50_spread_per_stratum(stratified_data, horizons=[1, 30, 60, 90]):
    """For each context length, compute p50 marginal spread.

    Args:
        stratified_data: dict from stratify_by_context_length
        horizons: list of day indices to analyze (1-indexed)

    Returns:
        pandas DataFrame with columns [context_length, day, p05, p50, p95, spread]
    """
    print("Computing p50 spreads...")

    results = []

    for context_len, data in sorted(stratified_data.items()):
        p50_predictions = data['p50_predictions']  # (N, 90)
        n_sequences = data['n_sequences']

        for day in horizons:
            day_idx = day - 1  # Convert to 0-indexed

            # Get all p50 predictions for this day
            p50_values = p50_predictions[:, day_idx]

            # Compute marginal percentiles
            p05 = np.percentile(p50_values, 5)
            p50_median = np.percentile(p50_values, 50)
            p95 = np.percentile(p50_values, 95)
            spread = p95 - p05

            results.append({
                'context_length': context_len,
                'day': day,
                'p05': p05,
                'p50': p50_median,
                'p95': p95,
                'spread': spread,
                'n_sequences': n_sequences
            })

            if day == 1:
                print(f"  Context {context_len:2d} days - Day {day:2d}: "
                      f"spread = {spread:.4f} (n={n_sequences})")

    df = pd.DataFrame(results)
    return df


# ============================================================================
# Visualization
# ============================================================================

def plot_spread_vs_context_length(results_df, output_dir):
    """Plot p50 spread vs context length for different horizons.

    Args:
        results_df: DataFrame from compute_p50_spread_per_stratum
        output_dir: Path for output files
    """
    print("[PRIMARY] Generating spread vs context length plots...")

    horizons = results_df['day'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = ['red', 'orange', 'blue', 'green']

    for i, (day, ax) in enumerate(zip(horizons, axes)):
        data = results_df[results_df['day'] == day]

        # Plot spread vs context length
        ax.plot(data['context_length'], data['spread'],
                marker='o', markersize=10, linewidth=2.5,
                color=colors[i], label=f'Day {day}')

        # Scatter points sized by number of sequences
        sizes = data['n_sequences'] / data['n_sequences'].max() * 300 + 50
        ax.scatter(data['context_length'], data['spread'],
                  s=sizes, color=colors[i], alpha=0.6,
                  edgecolor='black', linewidth=1)

        # Reference lines
        if day == 1:
            # Ground truth day 1 spread
            ax.axhline(y=0.0248, color='black', linestyle='--',
                      linewidth=2, alpha=0.7, label='GT Day 1 (0.0248)')
            # Current overall spread
            ax.axhline(y=0.0858, color='gray', linestyle=':',
                      linewidth=2, alpha=0.7, label='Overall Day 1 (0.0858)')
            # Decision threshold
            ax.axhline(y=0.04, color='purple', linestyle='-.',
                      linewidth=2, alpha=0.7, label='Decision Threshold (0.04)')

        ax.set_xlabel('Context Length (days)', fontsize=12, weight='bold')
        ax.set_ylabel('P50 Marginal Spread (p95-p05)', fontsize=12, weight='bold')
        ax.set_title(f'Day {day}: P50 Spread vs Context Length',
                    fontsize=14, weight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_xticks(CONTEXT_LENGTHS)

        # Annotate points
        for _, row in data.iterrows():
            ax.annotate(f"n={row['n_sequences']}",
                       xy=(row['context_length'], row['spread']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.7)

    plt.tight_layout()

    # Save
    filepath = output_dir / 'p50_spread_vs_context_length_all_days.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()


def plot_day1_focus(results_df, output_dir):
    """Focused plot for day 1 analysis (the key diagnostic).

    Args:
        results_df: DataFrame from compute_p50_spread_per_stratum
        output_dir: Path for output files
    """
    print("[PRIMARY] Generating day-1 focused plot...")

    data = results_df[results_df['day'] == 1].copy()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Main line plot
    ax.plot(data['context_length'], data['spread'],
            marker='o', markersize=14, linewidth=3,
            color='red', label='Oracle Day 1', zorder=5)

    # Scatter with size proportional to n_sequences
    sizes = data['n_sequences'] / data['n_sequences'].max() * 500 + 100
    ax.scatter(data['context_length'], data['spread'],
              s=sizes, color='red', alpha=0.4,
              edgecolor='darkred', linewidth=2, zorder=4)

    # Reference lines
    ax.axhline(y=0.0248, color='blue', linestyle='--',
              linewidth=2.5, alpha=0.8, label='Ground Truth Day 1 (0.0248)', zorder=3)
    ax.axhline(y=0.0858, color='gray', linestyle=':',
              linewidth=2.5, alpha=0.8, label='Overall Oracle Day 1 (0.0858)', zorder=3)
    ax.axhline(y=0.04, color='purple', linestyle='-.',
              linewidth=2.5, alpha=0.8, label='Decision Threshold (0.04)', zorder=3)

    # Shaded regions
    ax.axhspan(0, 0.04, alpha=0.1, color='green', label='Target Zone (<0.04)')
    ax.axhspan(0.04, 0.10, alpha=0.1, color='yellow')
    ax.axhspan(0.10, ax.get_ylim()[1], alpha=0.1, color='red')

    ax.set_xlabel('Context Length (days)', fontsize=14, weight='bold')
    ax.set_ylabel('P50 Marginal Spread (p95-p05)', fontsize=14, weight='bold')
    ax.set_title('Day 1: Does Context Length Reduce P50 Spread?\n'
                 'Hypothesis: Longer context → lower epistemic uncertainty → narrower spread',
                fontsize=16, weight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8, zorder=0)
    ax.set_xticks(CONTEXT_LENGTHS)

    # Annotate each point
    for _, row in data.iterrows():
        # Spread value
        ax.annotate(f"{row['spread']:.4f}",
                   xy=(row['context_length'], row['spread']),
                   xytext=(0, 10), textcoords='offset points',
                   fontsize=10, weight='bold', ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Sample size
        ax.annotate(f"n={row['n_sequences']}",
                   xy=(row['context_length'], row['spread']),
                   xytext=(0, -20), textcoords='offset points',
                   fontsize=9, alpha=0.7, ha='center')

    # Add interpretation box
    # Check if context=60 spread < 0.04
    context60_spread = data[data['context_length'] == 60]['spread'].values[0]

    if context60_spread < 0.04:
        verdict = "✅ CONFIRMED: Fixed context reduces spread to <0.04"
        recommendation = "RECOMMENDATION: Train with fixed context=60"
        box_color = 'lightgreen'
    else:
        verdict = "❌ NOT CONFIRMED: Context=60 spread still high"
        recommendation = "RECOMMENDATION: Investigate other causes (latent space, etc.)"
        box_color = 'lightcoral'

    interpretation_text = (
        f"HYPOTHESIS TEST RESULT:\n"
        f"{verdict}\n\n"
        f"Context=60 spread: {context60_spread:.4f}\n"
        f"Reduction from overall: {(1 - context60_spread/0.0858)*100:.1f}%\n\n"
        f"{recommendation}"
    )

    props = dict(boxstyle='round,pad=1', facecolor=box_color, alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, interpretation_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    # Save
    filepath = output_dir / 'day1_spread_vs_context_length_focused.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()


# ============================================================================
# Report Generation
# ============================================================================

def generate_interpretation_report(results_df, output_dir):
    """Generate text report with interpretation and recommendations.

    Args:
        results_df: DataFrame from compute_p50_spread_per_stratum
        output_dir: Path for output files
    """
    print("Generating interpretation report...")

    report = []
    report.append("=" * 80)
    report.append("EXPERIMENT 1: CONTEXT LENGTH STRATIFICATION ANALYSIS")
    report.append("=" * 80)
    report.append("")

    report.append("HYPOTHESIS:")
    report.append("  Context randomization causes day-1 over-dispersion via epistemic uncertainty.")
    report.append("  Prediction: Shorter contexts → wider p50 spread due to ambiguous signals.")
    report.append("")

    report.append("METHOD:")
    report.append(f"  - Stratified {len(results_df[results_df['day']==1])} context length bins")
    report.append(f"  - Analyzed horizons: {sorted(results_df['day'].unique())}")
    report.append(f"  - Computed p50 marginal spread (p95-p05) for each stratum")
    report.append("")

    report.append("RESULTS:")
    report.append("")

    # Day 1 results (key diagnostic)
    day1_data = results_df[results_df['day'] == 1].sort_values('context_length')
    report.append("Day 1 P50 Spread by Context Length:")
    report.append(f"{'Context':>8} | {'Spread':>8} | {'N Seqs':>8} | {'vs Overall':>12} | {'vs GT':>12}")
    report.append("-" * 65)

    overall_spread = 0.0858  # From previous analysis
    gt_spread = 0.0248

    for _, row in day1_data.iterrows():
        vs_overall = (row['spread'] / overall_spread - 1) * 100
        vs_gt = (row['spread'] / gt_spread - 1) * 100
        report.append(f"{int(row['context_length']):>8d} | {row['spread']:>8.4f} | {int(row['n_sequences']):>8d} | "
                     f"{vs_overall:>+11.1f}% | {vs_gt:>+11.1f}%")

    report.append("")

    # Trend analysis
    context60_spread = day1_data[day1_data['context_length'] == 60]['spread'].values[0]
    context10_spread = day1_data[day1_data['context_length'] == 10]['spread'].values

    if len(context10_spread) > 0:
        context10_spread = context10_spread[0]
        reduction = (1 - context60_spread / context10_spread) * 100
        report.append(f"TREND: Context=10 → Context=60 reduces spread by {reduction:.1f}%")
    else:
        report.append(f"TREND: Context=60 spread = {context60_spread:.4f}")

    report.append("")

    # Check if we actually have multiple context lengths
    n_unique_lengths = len(day1_data)

    if n_unique_lengths == 1:
        # Special case: All sequences have same context length
        only_length = int(day1_data.iloc[0]['context_length'])
        report.append("CRITICAL FINDING:")
        report.append(f"  ALL test sequences have the same context length: {only_length} days")
        report.append("")
        report.append("INTERPRETATION:")
        report.append("  The test set uses FIXED context length, not randomized context lengths.")
        report.append("  This means:")
        report.append(f"    - The current day-1 spread ({context60_spread:.4f}) occurs WITH fixed context")
        report.append(f"    - Context randomization is NOT testable with this data")
        report.append("")
        report.append("VERDICT:")
        if context60_spread >= 0.04:
            report.append(f"  ❌ HYPOTHESIS REJECTED")
            report.append(f"  Day-1 over-dispersion ({context60_spread:.4f}) persists EVEN with fixed context=60")
            report.append(f"  This is {(context60_spread/gt_spread):.1f}× higher than ground truth ({gt_spread:.4f})")
            report.append("")
            report.append("CONCLUSION:")
            report.append("  Context randomization is NOT the primary cause of day-1 over-dispersion.")
            report.append("  The problem exists even with consistent context length.")
            report.append("")
            report.append("NEXT STEPS:")
            report.append("  **Investigate other root causes:**")
            report.append("  1. Under-regularized latent space (KL weight = 1e-5) → Run Experiment 2")
            report.append("  2. Training horizon mismatch (H=1 → H=90) → Run Experiment 5")
            report.append("  3. Mean reversion bias (23 years of training data)")
        else:
            report.append(f"  ⚠️  INCONCLUSIVE")
            report.append(f"  Day-1 spread ({context60_spread:.4f}) is reasonable (<0.04 threshold)")
            report.append(f"  But we cannot test effect of context randomization with this data")
    else:
        # Multiple context lengths available
        report.append("DECISION CRITERIA:")
        report.append(f"  - If context=60 spread < 0.04: ✅ FIX: Use fixed context=60")
        report.append(f"  - If context=60 spread >= 0.04: ❌ Context randomization NOT main cause")
        report.append("")

        report.append("VERDICT:")
        if context60_spread < 0.04:
            report.append(f"  ✅ HYPOTHESIS CONFIRMED")
            report.append(f"  Context=60 spread ({context60_spread:.4f}) is below threshold (0.04)")
            report.append(f"  Reduction from overall: {(1 - context60_spread/overall_spread)*100:.1f}%")
            report.append("")
            report.append("RECOMMENDATION:")
            report.append("  **Train next model with FIXED context length = 60 days**")
            report.append("")
            report.append("EXPECTED IMPACT:")
            report.append(f"  - Day 1 p50 spread: {overall_spread:.4f} → {context60_spread:.4f}")
            report.append(f"  - Reduction: {(1 - context60_spread/overall_spread)*100:.1f}%")
            report.append(f"  - Brings closer to GT ({gt_spread:.4f})")
        else:
            report.append(f"  ❌ HYPOTHESIS NOT CONFIRMED")
            report.append(f"  Context=60 spread ({context60_spread:.4f}) still above threshold (0.04)")
            report.append(f"  Reduction from overall: {(1 - context60_spread/overall_spread)*100:.1f}%")
            report.append("")
            report.append("INTERPRETATION:")
            report.append("  Context randomization is NOT the primary cause of day-1 over-dispersion.")
            report.append("  Other factors dominate:")
            report.append("    - Under-regularized latent space (KL weight = 1e-5)")
            report.append("    - Training horizon mismatch (H=1 → H=90 extrapolation)")
            report.append("")
            report.append("RECOMMENDATION:")
            report.append("  **Investigate latent space structure (Experiment 2)**")
            report.append("  **Consider increasing KL weight from 1e-5 to 1e-3**")

    report.append("")
    report.append("=" * 80)

    # Write report
    report_text = "\n".join(report)
    report_path = output_dir / 'interpretation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  ✓ Saved: {report_path}")

    # Also print to console
    print("\n" + report_text)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run complete context length stratification analysis."""

    print("=" * 80)
    print("EXPERIMENT 1: CONTEXT LENGTH STRATIFICATION ANALYSIS")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    predictions = load_predictions_and_indices()
    gt_surface = load_ground_truth()

    # Extract context lengths
    actual_context_lengths = extract_context_length_for_each_sequence(
        predictions['indices'], gt_surface, CONTEXT_LEN
    )

    # Stratify by context length
    stratified_data = stratify_by_context_length(
        actual_context_lengths, predictions['surfaces'], gt_surface, predictions['indices']
    )

    # Compute spreads
    results_df = compute_p50_spread_per_stratum(stratified_data, horizons=[1, 30, 60, 90])

    # Save results
    csv_path = OUTPUT_DIR / 'context_length_effect_statistics.csv'
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Saved statistics: {csv_path}")

    # Generate visualizations
    plot_spread_vs_context_length(results_df, OUTPUT_DIR)
    plot_day1_focus(results_df, OUTPUT_DIR)

    # Generate report
    generate_interpretation_report(results_df, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("EXPERIMENT 1 COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
