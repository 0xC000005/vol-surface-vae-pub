#!/usr/bin/env python3
"""
Investigation: "Opposite Dynamics" Pattern - Bug or Feature?

Systematically investigates whether the observed pattern is miscalibration or
correct uncertainty representation:
- GT: Starts narrow (0.0248), explodes wide (0.1572), +535% growth
- Models: Start wide (0.1255), stay flat (0.1624), +29% growth

Key analyses:
1. Coverage statistics: Do model bands contain ~90% of GT paths?
2. Individual vs marginal: Is wide marginal due to pooling?
3. Diagnosis: Miscalibration (bug) or epistemic uncertainty (feature)?

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
ATM_6M = (2, 2)
OUTPUT_DIR = Path("results/context60_latent12_v2/analysis/opposite_dynamics_investigation")

# Days to check for coverage
COVERAGE_DAYS = [0, 4, 9, 19, 29, 44, 59, 74, 89]  # Days 1, 5, 10, 20, 30, 45, 60, 75, 90


# ============================================================================
# Data Loading
# ============================================================================

def load_all_data():
    """Load ground truth and model predictions.

    Returns:
        dict with gt_surface, gt_dates, oracle_pred, prior_pred
    """
    print("Loading data...")

    # Ground truth
    gt_data = np.load("data/vol_surface_with_ret.npz")
    dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")

    grid_row, grid_col = ATM_6M
    gt_surface = gt_data['surface'][:, grid_row, grid_col]
    gt_dates = pd.to_datetime(dates_df["date"].values)

    print(f"  Ground truth: {len(gt_surface)} days")

    # Oracle predictions
    oracle_file = f"results/context60_latent12_v2/predictions/teacher_forcing/oracle/vae_tf_insample_h{HORIZON}.npz"
    oracle_data = np.load(oracle_file)
    oracle_surfaces = oracle_data['surfaces']  # (N, 90, 3, 5, 5)
    oracle_indices = oracle_data['indices']

    print(f"  Oracle predictions: {len(oracle_indices)} sequences")

    # Prior predictions
    prior_file = f"results/context60_latent12_v2/predictions/teacher_forcing/prior/vae_tf_insample_h{HORIZON}.npz"
    prior_data = np.load(prior_file)
    prior_surfaces = prior_data['surfaces']
    prior_indices = prior_data['indices']

    print(f"  Prior predictions: {len(prior_indices)} sequences")
    print()

    return {
        'gt_surface': gt_surface,
        'gt_dates': gt_dates,
        'oracle_surfaces': oracle_surfaces,
        'oracle_indices': oracle_indices,
        'prior_surfaces': prior_surfaces,
        'prior_indices': prior_indices,
    }


# ============================================================================
# Analysis 1: Coverage Statistics
# ============================================================================

def compute_coverage_statistics(data):
    """Compute what % of GT paths fall within model prediction bands.

    Args:
        data: dict from load_all_data()

    Returns:
        dict with coverage statistics for oracle and prior
    """
    print("[ANALYSIS 1] Computing coverage statistics...")

    gt_surface = data['gt_surface']
    grid_row, grid_col = ATM_6M

    results = {}

    for mode_name, surfaces, indices in [
        ('oracle', data['oracle_surfaces'], data['oracle_indices']),
        ('prior', data['prior_surfaces'], data['prior_indices'])
    ]:
        print(f"  Computing {mode_name} coverage...")

        n_sequences = len(indices)
        coverage_by_day = {day: 0 for day in COVERAGE_DAYS}
        valid_sequences = {day: 0 for day in COVERAGE_DAYS}

        for i, idx in enumerate(indices):
            # Get context and forecast from ground truth
            context_start = max(0, idx - CONTEXT_LEN)
            context = gt_surface[context_start:idx]

            if len(context) < CONTEXT_LEN:
                continue

            forecast_start = idx
            forecast_end = idx + HORIZON
            gt_forecast = gt_surface[forecast_start:forecast_end]

            if len(gt_forecast) < HORIZON:
                continue

            # Get model predictions (p05, p50, p95) at ATM 6M
            model_p05 = surfaces[i, :, 0, grid_row, grid_col]  # (90,)
            model_p95 = surfaces[i, :, 2, grid_row, grid_col]  # (90,)

            # Check coverage at each day
            for day_idx in COVERAGE_DAYS:
                gt_value = gt_forecast[day_idx]
                pred_p05 = model_p05[day_idx]
                pred_p95 = model_p95[day_idx]

                valid_sequences[day_idx] += 1

                if pred_p05 <= gt_value <= pred_p95:
                    coverage_by_day[day_idx] += 1

        # Convert to percentages
        coverage_pct = {}
        for day_idx in COVERAGE_DAYS:
            if valid_sequences[day_idx] > 0:
                coverage_pct[day_idx + 1] = (coverage_by_day[day_idx] / valid_sequences[day_idx]) * 100
            else:
                coverage_pct[day_idx + 1] = np.nan

        results[mode_name] = {
            'days': list(coverage_pct.keys()),
            'coverage_pct': list(coverage_pct.values()),
            'n_sequences': n_sequences,
        }

        print(f"    Day 1 coverage: {coverage_pct[1]:.1f}%")
        print(f"    Day 45 coverage: {coverage_pct[45]:.1f}%")
        print(f"    Day 90 coverage: {coverage_pct[90]:.1f}%")

    print()
    return results


# ============================================================================
# Analysis 2: Individual vs Marginal Uncertainty
# ============================================================================

def analyze_individual_vs_marginal(data):
    """Compare individual prediction widths to marginal widths.

    Args:
        data: dict from load_all_data()

    Returns:
        dict with individual and marginal width statistics
    """
    print("[ANALYSIS 2] Analyzing individual vs marginal uncertainty...")

    gt_surface = data['gt_surface']
    grid_row, grid_col = ATM_6M

    results = {}

    for mode_name, surfaces, indices in [
        ('oracle', data['oracle_surfaces'], data['oracle_indices']),
        ('prior', data['prior_surfaces'], data['prior_indices'])
    ]:
        print(f"  Analyzing {mode_name}...")

        n_sequences = len(indices)

        # Pre-allocate arrays for normalized predictions
        normalized_p05 = []
        normalized_p50 = []
        normalized_p95 = []

        for i, idx in enumerate(indices):
            # Get context endpoint
            context_start = max(0, idx - CONTEXT_LEN)
            context = gt_surface[context_start:idx]

            if len(context) < CONTEXT_LEN:
                continue

            anchor = context[-1]

            # Extract and normalize model quantiles
            p05 = surfaces[i, :, 0, grid_row, grid_col] - anchor
            p50 = surfaces[i, :, 1, grid_row, grid_col] - anchor
            p95 = surfaces[i, :, 2, grid_row, grid_col] - anchor

            normalized_p05.append(p05)
            normalized_p50.append(p50)
            normalized_p95.append(p95)

        normalized_p05 = np.array(normalized_p05)  # (N, 90)
        normalized_p50 = np.array(normalized_p50)
        normalized_p95 = np.array(normalized_p95)

        # Individual widths (per sequence)
        individual_widths = normalized_p95 - normalized_p05  # (N, 90)
        avg_individual_width = np.mean(individual_widths, axis=0)  # (90,)

        # Marginal widths (pooled across sequences)
        marginal_p05 = np.percentile(normalized_p05, 5, axis=0)  # (90,)
        marginal_p50 = np.percentile(normalized_p50, 50, axis=0)
        marginal_p95 = np.percentile(normalized_p95, 95, axis=0)
        marginal_width = marginal_p95 - marginal_p05

        # Spread of centers (epistemic uncertainty proxy)
        spread_of_p50 = np.std(normalized_p50, axis=0)  # (90,)

        results[mode_name] = {
            'avg_individual_width': avg_individual_width,
            'marginal_width': marginal_width,
            'spread_of_centers': spread_of_p50,
            'individual_widths': individual_widths,  # Keep for detailed analysis
        }

        print(f"    Day 1:")
        print(f"      Avg individual width: {avg_individual_width[0]:.4f}")
        print(f"      Marginal width: {marginal_width[0]:.4f}")
        print(f"      Ratio: {marginal_width[0] / avg_individual_width[0]:.2f}×")
        print(f"    Day 90:")
        print(f"      Avg individual width: {avg_individual_width[-1]:.4f}")
        print(f"      Marginal width: {marginal_width[-1]:.4f}")
        print(f"      Ratio: {marginal_width[-1] / avg_individual_width[-1]:.2f}×")

    print()
    return results


# ============================================================================
# Visualization: Coverage Evolution
# ============================================================================

def plot_coverage_evolution(coverage_stats, output_dir):
    """Plot coverage vs horizon for oracle and prior.

    Args:
        coverage_stats: dict from compute_coverage_statistics
        output_dir: Path for output directory
    """
    print("[VISUALIZATION] Generating coverage evolution plots...")

    for mode_name in ['oracle', 'prior']:
        fig, ax = plt.subplots(figsize=(12, 7))

        stats = coverage_stats[mode_name]
        days = stats['days']
        coverage = stats['coverage_pct']

        # Plot coverage
        ax.plot(days, coverage, 'o-', linewidth=2.5, markersize=8,
                color='blue', label=f'{mode_name.capitalize()} Coverage')

        # Reference lines
        ax.axhline(y=90, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label='Target (90%)')
        ax.axhline(y=85, color='orange', linestyle=':', linewidth=1.5,
                   alpha=0.6, label='Warning threshold (85%)')
        ax.axhline(y=95, color='orange', linestyle=':', linewidth=1.5,
                   alpha=0.6, label='Warning threshold (95%)')

        # Diagnostic zones
        ax.axhspan(85, 95, color='green', alpha=0.1, label='Well-calibrated zone')
        ax.axhspan(95, 100, color='yellow', alpha=0.1, label='Over-conservative')
        ax.axhspan(0, 85, color='red', alpha=0.1, label='Under-confident')

        # Annotations
        ax.set_xlabel('Forecast Horizon (days)', fontsize=13, weight='bold')
        ax.set_ylabel('Coverage (%)', fontsize=13, weight='bold')
        ax.set_title(f'GT Path Coverage by {mode_name.capitalize()} Prediction Bands\n'
                     f'p05-p95 bands should contain ~90% of GT paths',
                     fontsize=14, weight='bold', pad=15)

        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_ylim(0, 100)

        # Statistics box
        day1_cov = coverage[0] if len(coverage) > 0 else np.nan
        day90_cov = coverage[-1] if len(coverage) > 0 else np.nan

        # Diagnosis
        if day1_cov > 95 and day90_cov < 85:
            diagnosis = "BUG: Miscalibrated\n(over-conservative → under-confident)"
        elif 85 <= day1_cov <= 95 and 85 <= day90_cov <= 95:
            diagnosis = "FEATURE: Well-Calibrated\n(appropriate uncertainty)"
        elif day1_cov > 95:
            diagnosis = "Partial: Over-conservative\n(predicts too much uncertainty)"
        elif day90_cov < 85:
            diagnosis = "Partial: Under-confident\n(under-predicts tail risk)"
        else:
            diagnosis = "Mixed"

        stats_text = (
            f'Coverage Statistics:\n'
            f'  Day 1: {day1_cov:.1f}%\n'
            f'  Day 45: {coverage[len(coverage)//2] if len(coverage) > 2 else np.nan:.1f}%\n'
            f'  Day 90: {day90_cov:.1f}%\n'
            f'\n'
            f'Diagnosis:\n'
            f'{diagnosis}'
        )

        props = dict(boxstyle='round,pad=0.8', facecolor='wheat',
                     alpha=0.95, edgecolor='black', linewidth=2)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                horizontalalignment='left', bbox=props, family='monospace')

        plt.tight_layout()

        filename = f'coverage_evolution_{mode_name}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"    File size: {size_mb:.1f} MB")

        plt.close()

    print()


# ============================================================================
# Visualization: Individual vs Marginal
# ============================================================================

def plot_uncertainty_decomposition(uncertainty_stats, output_dir):
    """Plot individual vs marginal width evolution.

    Args:
        uncertainty_stats: dict from analyze_individual_vs_marginal
        output_dir: Path for output directory
    """
    print("[VISUALIZATION] Generating uncertainty decomposition plot...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    days = np.arange(1, HORIZON + 1)

    for idx, mode_name in enumerate(['oracle', 'prior']):
        ax = axes[idx]
        stats = uncertainty_stats[mode_name]

        avg_individual = stats['avg_individual_width']
        marginal = stats['marginal_width']
        spread = stats['spread_of_centers']

        # Plot widths
        ax.plot(days, marginal, linewidth=2.5, color='red',
                label='Marginal Width (pooled)', zorder=3)
        ax.plot(days, avg_individual, linewidth=2.5, color='blue',
                label='Avg Individual Width', zorder=2)
        ax.plot(days, spread, linewidth=2, color='purple', linestyle='--',
                label='Spread of Centers (epistemic)', zorder=1, alpha=0.7)

        # Reference line
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

        # Annotations
        ax.set_ylabel('Uncertainty (IV)', fontsize=12, weight='bold')
        ax.set_title(f'{mode_name.capitalize()} Mode: Uncertainty Decomposition',
                     fontsize=13, weight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)

        # Statistics box
        ratio_day1 = marginal[0] / avg_individual[0]
        ratio_day90 = marginal[-1] / avg_individual[-1]

        if ratio_day1 > 2.0:
            interpretation = "Wide marginal due to\nspread of centers\n(epistemic uncertainty)"
        else:
            interpretation = "Wide marginal due to\nwide individuals\n(aleatory uncertainty)"

        stats_text = (
            f'Marginal / Individual:\n'
            f'  Day 1: {ratio_day1:.2f}×\n'
            f'  Day 90: {ratio_day90:.2f}×\n'
            f'\n'
            f'{interpretation}'
        )

        props = dict(boxstyle='round,pad=0.6', facecolor='lightblue',
                     alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                horizontalalignment='right', bbox=props, family='monospace')

    axes[1].set_xlabel('Forecast Horizon (days)', fontsize=12, weight='bold')

    plt.tight_layout()

    filename = 'individual_vs_marginal_widths.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()
    print()


# ============================================================================
# Report Generation
# ============================================================================

def generate_diagnostic_report(coverage_stats, uncertainty_stats, output_dir):
    """Generate diagnostic report: Bug or Feature?

    Args:
        coverage_stats: dict from compute_coverage_statistics
        uncertainty_stats: dict from analyze_individual_vs_marginal
        output_dir: Path for output directory
    """
    print("Generating diagnostic report...")

    # CSV: Coverage statistics
    coverage_df = pd.DataFrame({
        'Day': coverage_stats['oracle']['days'],
        'Oracle_Coverage_%': coverage_stats['oracle']['coverage_pct'],
        'Prior_Coverage_%': coverage_stats['prior']['coverage_pct'],
    })

    csv_path = output_dir / 'coverage_statistics.csv'
    coverage_df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"  ✓ Saved: {csv_path}")

    # Text report
    report = []
    report.append("=" * 80)
    report.append("OPPOSITE DYNAMICS INVESTIGATION: BUG OR FEATURE?")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    report.append("OBSERVATION:")
    report.append("  Ground truth: Starts narrow (spread 0.0248), explodes wide (0.1572), +535% growth")
    report.append("  Models: Start wide (spread 0.1255), stay flat (0.1624), +29% growth")
    report.append("  Question: Is this miscalibration (BUG) or correct uncertainty (FEATURE)?")
    report.append("")

    report.append("=" * 80)
    report.append("FINDING 1: COVERAGE ANALYSIS")
    report.append("=" * 80)
    report.append("")

    for mode_name in ['oracle', 'prior']:
        stats = coverage_stats[mode_name]
        day1_cov = stats['coverage_pct'][0]
        day90_cov = stats['coverage_pct'][-1]

        report.append(f"{mode_name.upper()}:")
        report.append(f"  Day 1 coverage: {day1_cov:.1f}% (target: 90%)")
        report.append(f"  Day 90 coverage: {day90_cov:.1f}% (target: 90%)")

        if 85 <= day1_cov <= 95 and 85 <= day90_cov <= 95:
            report.append("  → WELL-CALIBRATED: Both within acceptable range")
        elif day1_cov > 95:
            report.append("  → OVER-CONSERVATIVE at day 1: Predicting too much initial uncertainty")
        elif day90_cov < 85:
            report.append("  → UNDER-CONFIDENT at day 90: Under-predicting long-term uncertainty")
        report.append("")

    report.append("=" * 80)
    report.append("FINDING 2: INDIVIDUAL VS MARGINAL UNCERTAINTY")
    report.append("=" * 80)
    report.append("")

    for mode_name in ['oracle', 'prior']:
        stats = uncertainty_stats[mode_name]
        ratio_day1 = stats['marginal_width'][0] / stats['avg_individual_width'][0]
        ratio_day90 = stats['marginal_width'][-1] / stats['avg_individual_width'][-1]

        report.append(f"{mode_name.upper()}:")
        report.append(f"  Marginal / Individual ratio:")
        report.append(f"    Day 1: {ratio_day1:.2f}× (marginal {ratio_day1:.2f}× wider)")
        report.append(f"    Day 90: {ratio_day90:.2f}× (marginal {ratio_day90:.2f}× wider)")

        if ratio_day1 > 2.0:
            report.append("  → Wide marginal comes from SPREAD OF CENTERS (epistemic uncertainty)")
            report.append("     Individual predictions are narrower but centered at different locations")
        else:
            report.append("  → Wide marginal comes from WIDE INDIVIDUALS (aleatory uncertainty)")
            report.append("     Individual predictions are genuinely wide")
        report.append("")

    report.append("=" * 80)
    report.append("CONCLUSION")
    report.append("=" * 80)
    report.append("")

    # Determine overall diagnosis
    oracle_day1_cov = coverage_stats['oracle']['coverage_pct'][0]
    oracle_day90_cov = coverage_stats['oracle']['coverage_pct'][-1]
    oracle_ratio_day1 = uncertainty_stats['oracle']['marginal_width'][0] / uncertainty_stats['oracle']['avg_individual_width'][0]

    if 85 <= oracle_day1_cov <= 95 and 85 <= oracle_day90_cov <= 95 and oracle_ratio_day1 > 2.0:
        report.append("VERDICT: FEATURE (Not a Bug)")
        report.append("")
        report.append("The 'opposite dynamics' pattern is a CORRECT representation of uncertainty:")
        report.append("  1. Coverage is well-calibrated at both day 1 and day 90 (~90%)")
        report.append("  2. Wide marginal bands at day 1 come from epistemic uncertainty:")
        report.append("     - Model doesn't know which of many plausible paths will occur")
        report.append("     - Individual predictions are narrow but centered at different locations")
        report.append("  3. The model correctly represents: 'I'm uncertain which scenario, but")
        report.append("     within each scenario, the path is relatively constrained'")
        report.append("")
        report.append("This is EPISTEMIC uncertainty (model uncertainty about context interpretation)")
        report.append("vs ALEATORY uncertainty (natural path variability).")
    elif oracle_day1_cov > 95:
        report.append("VERDICT: BUG (Miscalibration)")
        report.append("")
        report.append("The model is OVER-CONSERVATIVE at short horizons:")
        report.append(f"  - Day 1 coverage is {oracle_day1_cov:.1f}% (should be ~90%)")
        report.append("  - Predicting too much initial uncertainty")
        report.append("")
        report.append("RECOMMENDATION: Retrain with adjusted loss weighting or confidence interval")
        report.append("calibration to reduce initial band width.")
    elif oracle_day90_cov < 85:
        report.append("VERDICT: BUG (Miscalibration)")
        report.append("")
        report.append("The model UNDER-PREDICTS long-term uncertainty:")
        report.append(f"  - Day 90 coverage is {oracle_day90_cov:.1f}% (should be ~90%)")
        report.append("  - Not capturing tail risk at longer horizons")
        report.append("")
        report.append("RECOMMENDATION: Retrain with increased multi-horizon diversity or")
        report.append("adjust quantile loss to penalize under-coverage at long horizons.")
    else:
        report.append("VERDICT: MIXED")
        report.append("")
        report.append("The pattern has both correct and incorrect aspects.")
        report.append("Review coverage and individual/marginal analysis for details.")

    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    report_path = output_dir / 'diagnostic_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  ✓ Saved: {report_path}")

    # Also print to console
    print("\n" + report_text)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run complete investigation."""

    print("=" * 80)
    print("INVESTIGATING 'OPPOSITE DYNAMICS' PATTERN")
    print("Bug or Feature?")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_all_data()

    # Analysis 1: Coverage statistics
    coverage_stats = compute_coverage_statistics(data)

    # Analysis 2: Individual vs marginal
    uncertainty_stats = analyze_individual_vs_marginal(data)

    # Generate visualizations
    plot_coverage_evolution(coverage_stats, OUTPUT_DIR)
    plot_uncertainty_decomposition(uncertainty_stats, OUTPUT_DIR)

    # Generate diagnostic report
    generate_diagnostic_report(coverage_stats, uncertainty_stats, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("INVESTIGATION COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
