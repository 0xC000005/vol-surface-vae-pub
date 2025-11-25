"""
Test In-Sequence Co-Integration for Bootstrap Autoregressive Sequences

Tests whether IV-EWMA co-integration holds WITHIN each 30-day bootstrap
autoregressive sequence, following milestone presentation methodology.

Implements dual-level testing:
1. Per-sequence (766 individual tests per grid point) → pass rate
2. Aggregate pooled (23K observations) → robust inference

Statistical adjustments for 30-day limitation:
- ADF lags: 3 instead of 5
- Significance: α=0.10 instead of 0.05

Usage:
    python experiments/bootstrap_baseline/test_insequence_cointegration.py --period crisis

Outputs:
    results/bootstrap_baseline/analysis/insequence/
    ├── summary_statistics.csv
    ├── per_sequence_pvalues.npz
    ├── aggregate_pooled_results.npz
    └── visualizations/
        ├── pass_rate_heatmap_crisis.png
        └── pvalue_distribution_crisis.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from insequence_cointegration_utils import (
    compute_ewma_for_sequence,
    test_sequence_cointegration,
    test_aggregate_pooled,
    plot_pass_rate_heatmap,
    plot_pvalue_distribution,
    compile_summary_statistics,
    compute_baseline_comparison,
    ADF_LAGS,
    ADF_ALPHA,
)


def load_data():
    """Load bootstrap AR predictions and ground truth data."""
    print("Loading data...")

    # Bootstrap AR predictions
    ar_file = "results/bootstrap_baseline/predictions/autoregressive/bootstrap_ar_crisis.npz"
    ar_data = np.load(ar_file)
    ar_surfaces = ar_data['surfaces']  # (766, 30, 3, 5, 5)
    print(f"  Bootstrap AR: {ar_surfaces.shape}")

    # Ground truth data
    gt_file = "data/vol_surface_with_ret.npz"
    gt_data = np.load(gt_file)
    returns = gt_data['ret']  # (5822,)
    gt_surfaces = gt_data['surface']  # (5822, 5, 5)
    print(f"  Returns: {returns.shape}")
    print(f"  Ground truth surfaces: {gt_surfaces.shape}")

    return ar_surfaces, returns, gt_surfaces


def test_per_sequence_cointegration(ar_surfaces, returns, crisis_start, crisis_end):
    """
    Test co-integration for each of 766 sequences individually.

    For each sequence and grid point:
    - Extract 30-day IV series
    - Compute 30-day EWMA series
    - Run ADF test on residuals
    - Record pass/fail, p-value, regression stats

    Args:
        ar_surfaces: (766, 30, 3, 5, 5)
        returns: (5822,)
        crisis_start: int (e.g., 2000)
        crisis_end: int (e.g., 2765)

    Returns:
        dict with:
            - pvalues: (737, 5, 5) p-values
            - cointegrated: (737, 5, 5) bool pass/fail
            - pass_rates: (5, 5) % of sequences passing per grid
            - rsquared: (737, 5, 5) R² values
            - alpha1: (737, 5, 5) EWMA coefficients
    """
    n_seq, n_days, n_quantiles, n_rows, n_cols = ar_surfaces.shape

    # Only process sequences that have full 30-day EWMA data
    # Crisis period: 766 days, 30-day sequences can start from day 0 to 736
    n_valid_seq = crisis_end - crisis_start + 1 - n_days + 1  # 766 - 30 + 1 = 737
    print(f"  Valid sequences with full 30-day data: {n_valid_seq} (out of {n_seq})")

    # Initialize storage
    pvalues = np.zeros((n_valid_seq, n_rows, n_cols))
    cointegrated = np.zeros((n_valid_seq, n_rows, n_cols), dtype=bool)
    rsquared = np.zeros((n_valid_seq, n_rows, n_cols))
    alpha1 = np.zeros((n_valid_seq, n_rows, n_cols))
    beta = np.zeros((n_valid_seq, n_rows, n_cols))

    # Extract p50 median
    p50 = ar_surfaces[:, :, 1, :, :]  # (766, 30, 5, 5)

    # Extract crisis returns
    crisis_returns = returns[crisis_start:crisis_end+1]

    print(f"\nTesting per-sequence co-integration ({n_valid_seq} sequences × 25 grid points)...")

    # Loop over valid sequences only
    for seq_idx in tqdm(range(n_valid_seq), desc="Testing sequences"):
        # IV for this sequence
        iv_sequence = p50[seq_idx, :, :, :]  # (30, 5, 5)

        # Compute EWMA for this sequence
        # Sequence starts at crisis_start + seq_idx
        start_idx = seq_idx
        ewma_sequence = compute_ewma_for_sequence(
            crisis_returns, start_idx, n_days=30
        )

        # Test each grid point
        for i in range(n_rows):
            for j in range(n_cols):
                iv_series = iv_sequence[:, i, j]  # (30,)

                # Test co-integration
                result = test_sequence_cointegration(
                    iv_series=iv_series,
                    ewma_series=ewma_sequence,
                    grid_idx=(i, j),
                    lags=ADF_LAGS,
                    alpha=ADF_ALPHA
                )

                # Store results
                pvalues[seq_idx, i, j] = result['adf_pvalue']
                cointegrated[seq_idx, i, j] = result['cointegrated']
                rsquared[seq_idx, i, j] = result['rsquared']
                alpha1[seq_idx, i, j] = result['alpha1']
                beta[seq_idx, i, j] = result['beta']

    # Compute pass rates per grid point
    pass_rates = cointegrated.mean(axis=0)  # (5, 5)

    print(f"\nPer-Sequence Results:")
    print(f"  Overall pass rate: {cointegrated.mean():.1%}")
    print(f"  Pass rate range: {pass_rates.min():.1%} to {pass_rates.max():.1%}")
    print(f"  Best grid: {np.unravel_index(pass_rates.argmax(), pass_rates.shape)}")
    print(f"  Worst grid: {np.unravel_index(pass_rates.argmin(), pass_rates.shape)}")

    return {
        'pvalues': pvalues,
        'cointegrated': cointegrated,
        'pass_rates': pass_rates,
        'rsquared': rsquared,
        'alpha1': alpha1,
        'beta': beta,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test in-sequence co-integration for bootstrap AR'
    )
    parser.add_argument(
        '--period',
        type=str,
        choices=['crisis'],
        default='crisis',
        help='Period to analyze (currently only crisis supported)'
    )
    args = parser.parse_args()

    print("="*80)
    print("In-Sequence Co-Integration Testing")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Period: {args.period}")
    print(f"  ADF lags: {ADF_LAGS} (reduced for 30-day sequences)")
    print(f"  Significance: {ADF_ALPHA} (conservative for low power)")

    # Crisis period configuration
    crisis_start = 2000
    crisis_end = 2765
    n_crisis_days = crisis_end - crisis_start + 1

    print(f"\nCrisis Period:")
    print(f"  Indices: {crisis_start}-{crisis_end}")
    print(f"  Days: {n_crisis_days}")

    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    ar_surfaces, returns, gt_surfaces = load_data()

    # Layer 1: Per-Sequence Testing
    print("\n" + "="*80)
    print("Layer 1: Per-Sequence Individual Tests")
    print("="*80)
    per_seq_results = test_per_sequence_cointegration(
        ar_surfaces, returns, crisis_start, crisis_end
    )

    # Layer 2: Aggregate Pooled Testing
    print("\n" + "="*80)
    print("Layer 2: Aggregate Pooled Test")
    print("="*80)
    print("Pooling all sequences for robust inference...")
    aggregate_results = test_aggregate_pooled(
        ar_surfaces, returns, crisis_start, crisis_end,
        lags=ADF_LAGS, alpha=ADF_ALPHA
    )

    print(f"\nAggregate Results:")
    print(f"  Pass rate: {aggregate_results['cointegrated'].mean():.1%} ({aggregate_results['cointegrated'].sum()}/25 grids)")
    print(f"  Median p-value: {np.median(aggregate_results['adf_pvalues']):.4f}")
    print(f"  Mean R²: {aggregate_results['rsquared'].mean():.3f}")
    print(f"  Total observations: {aggregate_results['n_observations']}")

    # Layer 3: Comparative Context
    print("\n" + "="*80)
    print("Layer 3: Comparative Context")
    print("="*80)

    # Compare to ground truth baseline (84% for crisis from milestone)
    comparison = compute_baseline_comparison(
        per_seq_results['pass_rates'],
        ground_truth_rate=0.84
    )

    print(f"\nComparison to Ground Truth (Milestone Baseline):")
    print(f"  Bootstrap mean: {comparison['bootstrap_mean']:.1%}")
    print(f"  Ground truth: {comparison['ground_truth']:.1%}")
    print(f"  Difference: {comparison['difference']:+.1%} ({comparison['difference_pct']:+.1f}%)")
    print(f"  Bootstrap range: {comparison['bootstrap_min']:.1%} to {comparison['bootstrap_max']:.1%}")

    # Save Results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)

    output_dir = Path("results/bootstrap_baseline/analysis/insequence")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save NPZ files
    np.savez(
        output_dir / f'per_sequence_pvalues_{args.period}.npz',
        pvalues=per_seq_results['pvalues'],
        cointegrated=per_seq_results['cointegrated'],
        pass_rates=per_seq_results['pass_rates'],
        rsquared=per_seq_results['rsquared'],
        alpha1=per_seq_results['alpha1'],
        beta=per_seq_results['beta'],
    )
    print(f"\nSaved: {output_dir / f'per_sequence_pvalues_{args.period}.npz'}")

    np.savez(
        output_dir / f'aggregate_pooled_results_{args.period}.npz',
        adf_pvalues=aggregate_results['adf_pvalues'],
        cointegrated=aggregate_results['cointegrated'],
        rsquared=aggregate_results['rsquared'],
        alpha1=aggregate_results['alpha1'],
        beta=aggregate_results['beta'],
        n_observations=aggregate_results['n_observations'],
    )
    print(f"Saved: {output_dir / f'aggregate_pooled_results_{args.period}.npz'}")

    # Save summary statistics CSV
    summary_df = compile_summary_statistics(per_seq_results, aggregate_results)
    summary_file = output_dir / f'summary_statistics_{args.period}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")

    # Generate Visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Pass rate heatmap
    plot_pass_rate_heatmap(
        per_seq_results['pass_rates'],
        viz_dir / f'pass_rate_heatmap_{args.period}.png',
        title=f"In-Sequence Co-Integration Pass Rates ({args.period.upper()})"
    )

    # p-value distribution
    pvalues_flat = per_seq_results['pvalues'].flatten()
    plot_pvalue_distribution(
        pvalues_flat,
        viz_dir / f'pvalue_distribution_{args.period}.png',
        title=f"ADF p-value Distribution - Per-Sequence Tests ({args.period.upper()})"
    )

    # Summary Report
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    print(f"\n{'Metric':<40} {'Per-Sequence':<15} {'Aggregate':<15}")
    print("-"*70)
    print(f"{'Pass Rate':<40} {per_seq_results['pass_rates'].mean():<15.1%} {aggregate_results['cointegrated'].mean():<15.1%}")
    print(f"{'Median p-value':<40} {np.median(per_seq_results['pvalues']):<15.4f} {np.median(aggregate_results['adf_pvalues']):<15.4f}")
    print(f"{'Mean R²':<40} {per_seq_results['rsquared'].mean():<15.3f} {aggregate_results['rsquared'].mean():<15.3f}")
    print(f"{'Mean α₁ (EWMA coef)':<40} {per_seq_results['alpha1'].mean():<15.3f} {aggregate_results['alpha1'].mean():<15.3f}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nKey Findings:")
    print(f"  1. Per-sequence pass rate: {per_seq_results['pass_rates'].mean():.1%}")
    print(f"  2. Aggregate pass rate: {aggregate_results['cointegrated'].mean():.1%}")
    print(f"  3. Difference from ground truth: {comparison['difference']:+.1%}")

    print(f"\nInterpretation:")
    if per_seq_results['pass_rates'].mean() >= 0.70:
        print("  ✓ GOOD: Most sequences maintain IV-EWMA co-integration")
    elif per_seq_results['pass_rates'].mean() >= 0.60:
        print("  ⚠ MODERATE: Co-integration preserved in majority but not all")
    else:
        print("  ✗ POOR: Many sequences fail co-integration test")

    if abs(comparison['difference']) <= 0.10:
        print("  ✓ GOOD: Within 10pp of ground truth baseline")
    else:
        print(f"  ⚠ NOTE: {abs(comparison['difference']):.1%} difference from ground truth")

    print(f"\nStatistical Caveats:")
    print(f"  - 30-day sequences below ADF minimum (50+)")
    print(f"  - Reduced lags (3 vs 5) and conservative alpha (0.10 vs 0.05)")
    print(f"  - Individual tests have low power, aggregate has full power")
    print(f"  - Results not directly comparable to cross-sectional analysis")

    print(f"\nOutput location:")
    print(f"  {output_dir}/")


if __name__ == "__main__":
    main()
