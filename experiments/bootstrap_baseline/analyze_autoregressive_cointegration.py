"""
Autoregressive Co-Integration Drift Analysis

Measures whether co-integration relationships are preserved over 30-day autoregressive
sequences. Tracks correlation similarity and Johansen rank AT EACH STEP to detect drift.

Novel analysis:
- Current co-integration tests are STATIC (one matrix for entire period)
- This implements DYNAMIC measurement (correlation at each autoregressive step)

Key metrics:
1. Correlation similarity at steps [1, 7, 14, 30]
2. Johansen co-integration rank evolution
3. Drift percentage: (similarity_step1 - similarity_step30) / similarity_step1 * 100

Usage:
    python experiments/bootstrap_baseline/analyze_autoregressive_cointegration.py --period crisis

Outputs:
    - correlation_drift_metrics.csv
    - johansen_rank_evolution.csv
    - correlation_drift_curve.png
    - drift_percentage_summary.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_ground_truth(period):
    """Load ground truth volatility surfaces."""
    data = np.load("data/vol_surface_with_ret.npz")
    surface = data['surface']  # (N, 5, 5)

    if period == 'crisis':
        gt = surface[2000:2765+1]
    elif period == 'insample':
        gt = surface[:3900]
    else:  # oos
        gt = surface[5000:5792+1]

    return gt


def load_bootstrap_ar(period):
    """Load bootstrap autoregressive predictions."""
    file_path = f"results/bootstrap_baseline/predictions/autoregressive/bootstrap_ar_{period}.npz"
    data = np.load(file_path)
    surfaces = data['surfaces']  # (n_days, horizon, 3, 5, 5)
    # Extract p50 median for analysis
    p50 = surfaces[:, :, 1, :, :]  # (n_days, horizon, 5, 5)
    return p50


def reshape_surface_to_matrix(surface):
    """
    Reshape (n_days, 5, 5) surface to (n_days, 25) matrix.
    Each column is a time series for one grid point.
    """
    n_days = surface.shape[0]
    return surface.reshape(n_days, 25)


def compute_correlation_matrix(surface):
    """
    Compute 25×25 correlation matrix across grid points.

    Args:
        surface: (n_days, 5, 5) volatility surfaces

    Returns:
        corr_matrix: (25, 25) correlation matrix
    """
    matrix = reshape_surface_to_matrix(surface)  # (n_days, 25)
    corr_matrix = np.corrcoef(matrix.T)  # Transpose to get correlation across series
    return corr_matrix


def compare_correlation_structures(gt_corr, pred_corr):
    """
    Compare two correlation matrices.

    Flattens upper triangle and computes correlation, RMSE, MAE.

    Args:
        gt_corr: (25, 25) ground truth correlation matrix
        pred_corr: (25, 25) predicted correlation matrix

    Returns:
        dict with correlation, rmse, mae
    """
    n = gt_corr.shape[0]
    triu_indices = np.triu_indices(n, k=1)  # Upper triangle (exclude diagonal)

    gt_flat = gt_corr[triu_indices]
    pred_flat = pred_corr[triu_indices]

    # Correlation of correlations (meta-correlation)
    correlation = np.corrcoef(gt_flat, pred_flat)[0, 1]
    rmse = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))
    mae = np.mean(np.abs(gt_flat - pred_flat))

    return {
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae
    }


def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Apply Johansen co-integration test.

    Args:
        data: (n_days, 25) matrix of 25 time series

    Returns:
        dict with rank, trace_stats, and test results
    """
    try:
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        trace_stats = result.lr1
        max_eigen_stats = result.lr2
        crit_values_trace = result.cvt
        crit_values_max = result.cvm

        # Determine co-integration rank at 5% significance
        rank = 0
        for i in range(len(trace_stats)):
            if trace_stats[i] > crit_values_trace[i, 1]:  # 95% critical value
                rank = i + 1
            else:
                break

        return {
            'success': True,
            'rank': rank,
            'trace_stats': trace_stats,
            'max_eigen_stats': max_eigen_stats,
            'crit_values_trace_95': crit_values_trace[:, 1],
            'crit_values_max_95': crit_values_max[:, 1],
        }
    except Exception as e:
        print(f"Warning: Johansen test failed: {e}")
        return {
            'success': False,
            'rank': np.nan,
            'trace_stats': None,
            'max_eigen_stats': None,
        }


def measure_correlation_drift_over_steps(ar_surfaces, gt_surfaces, steps=[1, 7, 14, 30]):
    """
    Measure correlation preservation at each autoregressive step.

    Novel approach: Extract predictions AT EACH STEP across all days,
    then compute correlation matrix for that step.

    Args:
        ar_surfaces: (n_days, horizon, 5, 5) - autoregressive predictions
        gt_surfaces: (n_days, 5, 5) - ground truth for reference
        steps: list of steps to analyze

    Returns:
        DataFrame with columns [step, correlation_similarity, rmse, mae]
    """
    results = []

    # Compute ground truth correlation (reference)
    print("Computing ground truth correlation matrix...")
    gt_corr = compute_correlation_matrix(gt_surfaces)

    for step in steps:
        print(f"\nAnalyzing step {step}...")

        # Extract predictions at this step across all days
        # Step 1: Use day 1 from each sequence
        # Step 7: Use day 7 from each sequence
        # Step 30: Use day 30 from each sequence
        pred_at_step = ar_surfaces[:, step-1, :, :]  # (n_days, 5, 5)

        # Compute correlation for predictions at this step
        pred_corr = compute_correlation_matrix(pred_at_step)

        # Compare to ground truth
        comparison = compare_correlation_structures(gt_corr, pred_corr)

        print(f"  Correlation similarity: {comparison['correlation']:.4f}")
        print(f"  RMSE: {comparison['rmse']:.4f}")
        print(f"  MAE: {comparison['mae']:.4f}")

        results.append({
            'step': step,
            'correlation_similarity': comparison['correlation'],
            'rmse': comparison['rmse'],
            'mae': comparison['mae']
        })

    return pd.DataFrame(results)


def measure_johansen_rank_over_steps(ar_surfaces, steps=[1, 7, 14, 30]):
    """
    Test co-integration rank degradation over autoregressive steps.

    Args:
        ar_surfaces: (n_days, horizon, 5, 5) - autoregressive predictions
        steps: list of steps to analyze

    Returns:
        DataFrame with columns [step, rank, success]
    """
    results = []

    for step in steps:
        print(f"\nJohansen test for step {step}...")

        # Extract predictions at this step
        pred_at_step = ar_surfaces[:, step-1, :, :]  # (n_days, 5, 5)
        matrix = reshape_surface_to_matrix(pred_at_step)  # (n_days, 25)

        # Run Johansen test
        result = johansen_test(matrix)

        if result['success']:
            print(f"  Co-integration rank: {result['rank']} out of 24")
        else:
            print(f"  Test failed")

        results.append({
            'step': step,
            'rank': result['rank'],
            'success': result['success']
        })

    return pd.DataFrame(results)


def compute_drift_percentage(similarity_step1, similarity_step30):
    """
    Quantify correlation drift as percentage degradation.

    drift = (similarity_1 - similarity_30) / similarity_1 * 100

    Examples:
        0.997 → 0.95 = 4.7% drift (good)
        0.997 → 0.80 = 19.8% drift (concerning)
    """
    drift = (similarity_step1 - similarity_step30) / similarity_step1 * 100
    return drift


def visualize_correlation_drift(drift_df, output_dir, period):
    """
    Plot correlation similarity vs autoregressive step.

    Args:
        drift_df: DataFrame with columns [step, correlation_similarity, ...]
        output_dir: Path to save directory
        period: Period name (e.g., 'crisis')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot correlation similarity curve
    ax.plot(drift_df['step'], drift_df['correlation_similarity'],
            marker='o', linewidth=2, markersize=8,
            color='#2E86AB', label='Bootstrap Autoregressive')

    # Add horizontal reference line at 0.95
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5,
               label='0.95 threshold (good preservation)')

    ax.set_xlabel('Autoregressive Step', fontsize=12)
    ax.set_ylabel('Correlation Similarity', fontsize=12)
    ax.set_title(f'Co-Integration Preservation Over 30-Day Sequences ({period.upper()})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.0])

    # Annotate drift percentage
    step1_sim = drift_df.loc[drift_df['step']==1, 'correlation_similarity'].values[0]
    step30_sim = drift_df.loc[drift_df['step']==30, 'correlation_similarity'].values[0]
    drift_pct = compute_drift_percentage(step1_sim, step30_sim)

    ax.text(0.5, 0.05,
            f'Drift: {drift_pct:.1f}% (step 1 → step 30)',
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            ha='center')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'correlation_drift_curve_{period}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()


def visualize_johansen_rank(rank_df, output_dir, period):
    """
    Plot Johansen co-integration rank evolution.

    Args:
        rank_df: DataFrame with columns [step, rank, ...]
        output_dir: Path to save directory
        period: Period name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot rank evolution
    ax.plot(rank_df['step'], rank_df['rank'],
            marker='s', linewidth=2, markersize=8,
            color='#A23B72', label='Co-integration Rank')

    ax.set_xlabel('Autoregressive Step', fontsize=12)
    ax.set_ylabel('Co-Integration Rank (out of 24)', fontsize=12)
    ax.set_title(f'Johansen Rank Evolution Over 30-Day Sequences ({period.upper()})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 25])

    # Annotate rank change
    step1_rank = rank_df.loc[rank_df['step']==1, 'rank'].values[0]
    step30_rank = rank_df.loc[rank_df['step']==30, 'rank'].values[0]
    rank_change = step30_rank - step1_rank

    ax.text(0.5, 0.95,
            f'Rank change: {rank_change:+.0f} (step 1 → step 30)',
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            ha='center', va='top')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_file = output_path / f'johansen_rank_evolution_{period}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze co-integration preservation in autoregressive sequences'
    )
    parser.add_argument(
        '--period',
        type=str,
        choices=['crisis', 'insample', 'oos'],
        default='crisis',
        help='Which period to analyze'
    )
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1, 7, 14, 30],
        help='Steps to analyze (default: 1 7 14 30)'
    )
    args = parser.parse_args()

    print("="*80)
    print("Autoregressive Co-Integration Drift Analysis")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Period: {args.period}")
    print(f"  Steps: {args.steps}")

    # Load data
    print("\nLoading data...")
    gt_surfaces = load_ground_truth(args.period)
    ar_surfaces = load_bootstrap_ar(args.period)

    print(f"Ground truth shape: {gt_surfaces.shape}")
    print(f"Autoregressive predictions shape: {ar_surfaces.shape}")

    # Align lengths
    min_len = min(len(gt_surfaces), len(ar_surfaces))
    gt_surfaces = gt_surfaces[:min_len]
    ar_surfaces = ar_surfaces[:min_len]

    print(f"Aligned to {min_len} days")

    # Measure correlation drift
    print("\n" + "="*80)
    print("Measuring Correlation Drift Over Steps")
    print("="*80)

    drift_df = measure_correlation_drift_over_steps(
        ar_surfaces=ar_surfaces,
        gt_surfaces=gt_surfaces,
        steps=args.steps
    )

    # Measure Johansen rank evolution
    print("\n" + "="*80)
    print("Measuring Johansen Rank Evolution")
    print("="*80)

    rank_df = measure_johansen_rank_over_steps(
        ar_surfaces=ar_surfaces,
        steps=args.steps
    )

    # Compute drift percentage
    print("\n" + "="*80)
    print("Drift Summary")
    print("="*80)

    step1_sim = drift_df.loc[drift_df['step']==1, 'correlation_similarity'].values[0]
    step30_sim = drift_df.loc[drift_df['step']==30, 'correlation_similarity'].values[0]
    drift_pct = compute_drift_percentage(step1_sim, step30_sim)

    print(f"\nCorrelation Similarity:")
    print(f"  Step 1:  {step1_sim:.4f}")
    print(f"  Step 30: {step30_sim:.4f}")
    print(f"  Drift:   {drift_pct:.2f}%")

    step1_rank = rank_df.loc[rank_df['step']==1, 'rank'].values[0]
    step30_rank = rank_df.loc[rank_df['step']==30, 'rank'].values[0]
    rank_change = step30_rank - step1_rank

    print(f"\nJohansen Co-Integration Rank:")
    print(f"  Step 1:  {step1_rank:.0f}")
    print(f"  Step 30: {step30_rank:.0f}")
    print(f"  Change:  {rank_change:+.0f}")

    # Interpret results
    print("\n" + "="*80)
    print("Interpretation")
    print("="*80)

    if drift_pct < 5:
        print("\n✓ EXCELLENT: Drift < 5% - Bootstrap preserves co-integration remarkably well!")
    elif drift_pct < 10:
        print("\n✓ GOOD: Drift < 10% - Bootstrap maintains co-integration adequately")
    elif drift_pct < 15:
        print("\n⚠ MODERATE: Drift 10-15% - Some correlation degradation over 30 days")
    else:
        print("\n✗ CONCERNING: Drift > 15% - Significant correlation drift, unsuitable for long-term backfilling")

    if abs(rank_change) <= 2:
        print("✓ Johansen rank stable (±2 or less)")
    else:
        print(f"⚠ Johansen rank changed by {abs(rank_change):.0f} (concerning)")

    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)

    output_dir = Path("results/bootstrap_baseline/analysis/autoregressive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    drift_csv = output_dir / f'correlation_drift_metrics_{args.period}.csv'
    drift_df.to_csv(drift_csv, index=False)
    print(f"\nSaved: {drift_csv}")

    rank_csv = output_dir / f'johansen_rank_evolution_{args.period}.csv'
    rank_df.to_csv(rank_csv, index=False)
    print(f"Saved: {rank_csv}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_correlation_drift(drift_df, output_dir, args.period)
    visualize_johansen_rank(rank_df, output_dir, args.period)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nAnalyzed {args.period} period autoregressive sequences")
    print(f"Measured correlation drift at {len(args.steps)} steps")
    print("\nOutput location:")
    print(f"  {output_dir}/")
    print("\nNext step:")
    print("  Run compare_autoregressive_methods.py to compare Bootstrap vs Econometric vs VAE")


if __name__ == "__main__":
    main()
