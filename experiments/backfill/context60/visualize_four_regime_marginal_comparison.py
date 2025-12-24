"""
Visualize Four-Regime Marginal Distribution Comparison - Context60 Latent12 V2

Creates a 2×2 plot comparing ground truth vs latent12_v2 model percentile bands
stratified by volatility regime (quartiles).

**Purpose**: Demonstrate whether latent12_v2 model improvements are consistent
across different volatility regimes.

**Four regimes (volatility quartiles):**
- Q1 (0-25%): Low volatility regime
- Q2 (25-50%): Medium-low volatility regime
- Q3 (50-75%): Medium-high volatility regime
- Q4 (75-100%): High volatility regime

**Layout:** 2×2 grid showing GT vs Oracle vs Prior for each regime

Usage:
    PYTHONPATH=. python experiments/backfill/context60/visualize_four_regime_marginal_comparison.py

Output:
    results/context60_latent12_v2/analysis/regime_percentile_bands/
    ├── four_regime_gt_vs_latent12v2_comparison.png
    └── regime_statistics.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Constants
CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)  # Grid point: K/S=1.00, 6-month maturity
N_REGIMES = 4    # Quartiles
OUTPUT_DIR = Path("results/context60_latent12_v2/analysis/regime_percentile_bands")


def load_ground_truth_by_regime():
    """
    Load GT and stratify into 4 regimes by context volatility.

    Returns:
        dict with:
        - 'regimes': dict of regime data (p05-p95, sequences, context vol stats)
        - 'thresholds': [Q1, Q2, Q3] percentile thresholds
        - 'total_sequences': total number of sequences
    """
    print("Loading ground truth data...")

    # Load data
    gt_data = np.load("data/vol_surface_with_ret.npz")
    atm_6m = gt_data['surface'][:, ATM_6M[0], ATM_6M[1]]

    # Extract sequences
    total_len = CONTEXT_LEN + HORIZON
    n_sequences = len(atm_6m) - total_len + 1

    sequences = []
    context_vols = []

    for i in range(n_sequences):
        context = atm_6m[i:i + CONTEXT_LEN]
        forecast = atm_6m[i + CONTEXT_LEN:i + total_len]

        # Calculate context volatility (60-day mean)
        context_vol = np.mean(context)
        context_vols.append(context_vol)

        # Normalize to context endpoint
        anchor = context[-1]
        sequences.append(forecast - anchor)

    sequences = np.array(sequences)  # (N, 90)
    context_vols = np.array(context_vols)

    print(f"  Total sequences: {n_sequences}")
    print(f"  Context vol range: [{context_vols.min():.3f}, {context_vols.max():.3f}]")

    # Compute quartile thresholds
    thresholds = np.percentile(context_vols, [25, 50, 75])
    print(f"  Quartile thresholds: {thresholds}")

    # Stratify into 4 regimes
    regimes = {}
    regime_labels = [
        'Low Vol (Q1: 0-25%)',
        'Medium-Low (Q2: 25-50%)',
        'Medium-High (Q3: 50-75%)',
        'High Vol (Q4: 75-100%)'
    ]

    for regime_idx in range(N_REGIMES):
        if regime_idx == 0:
            mask = context_vols <= thresholds[0]
        elif regime_idx == N_REGIMES - 1:
            mask = context_vols > thresholds[-1]
        else:
            mask = (context_vols > thresholds[regime_idx-1]) & \
                   (context_vols <= thresholds[regime_idx])

        regime_sequences = sequences[mask]

        regimes[f'regime_{regime_idx+1}'] = {
            'sequences': regime_sequences,
            'n_sequences': len(regime_sequences),
            'context_vol_mean': np.mean(context_vols[mask]),
            'context_vol_std': np.std(context_vols[mask]),
            'label': regime_labels[regime_idx],
            'p05': np.percentile(regime_sequences, 5, axis=0),
            'p25': np.percentile(regime_sequences, 25, axis=0),
            'p50': np.percentile(regime_sequences, 50, axis=0),
            'p75': np.percentile(regime_sequences, 75, axis=0),
            'p95': np.percentile(regime_sequences, 95, axis=0),
        }

        print(f"  {regime_labels[regime_idx]}: {len(regime_sequences)} sequences")

    return {
        'regimes': regimes,
        'thresholds': thresholds,
        'total_sequences': n_sequences
    }


def load_model_predictions_by_regime(sampling_mode, regime_thresholds):
    """
    Load latent12_v2 predictions and stratify by context volatility.

    Args:
        sampling_mode: 'oracle' or 'prior'
        regime_thresholds: [Q1, Q2, Q3] from GT

    Returns:
        dict with 'regime_1', 'regime_2', 'regime_3', 'regime_4'
    """
    print(f"Loading {sampling_mode} predictions...")

    # Load predictions
    pred_file = (f"results/context60_latent12_v2/predictions/teacher_forcing/"
                 f"{sampling_mode}/vae_tf_insample_h90.npz")
    pred_data = np.load(pred_file)

    surfaces = pred_data['surfaces']  # (N, 90, 3, 5, 5)
    indices = pred_data['indices']

    print(f"  Loaded {len(indices)} sequences")

    # Load GT for context calculation
    gt_data = np.load("data/vol_surface_with_ret.npz")
    gt_surface = gt_data['surface'][:, ATM_6M[0], ATM_6M[1]]

    # Extract sequences and calculate context vols
    sequences = []
    context_vols = []

    for i, idx in enumerate(indices):
        # Calculate context volatility
        context_start = max(0, idx - CONTEXT_LEN)
        context = gt_surface[context_start:idx]

        if len(context) < CONTEXT_LEN:
            continue

        context_vol = np.mean(context)
        context_vols.append(context_vol)

        # Extract all three quantiles and normalize
        anchor = context[-1]

        # Extract p05, p50, p95 for this sequence
        seq_p05 = surfaces[i, :, 0, ATM_6M[0], ATM_6M[1]] - anchor  # (90,)
        seq_p50 = surfaces[i, :, 1, ATM_6M[0], ATM_6M[1]] - anchor  # (90,)
        seq_p95 = surfaces[i, :, 2, ATM_6M[0], ATM_6M[1]] - anchor  # (90,)

        sequences.append(np.stack([seq_p05, seq_p50, seq_p95], axis=0))  # (3, 90)

    sequences = np.array(sequences)  # (N, 3, 90)
    context_vols = np.array(context_vols)

    print(f"  Valid sequences after filtering: {len(sequences)}")

    # Stratify using GT thresholds
    regimes = {}

    for regime_idx in range(N_REGIMES):
        if regime_idx == 0:
            mask = context_vols <= regime_thresholds[0]
        elif regime_idx == N_REGIMES - 1:
            mask = context_vols > regime_thresholds[-1]
        else:
            mask = (context_vols > regime_thresholds[regime_idx-1]) & \
                   (context_vols <= regime_thresholds[regime_idx])

        regime_sequences = sequences[mask]  # (n, 3, 90)

        # Extract percentile bands across sequences for this regime
        # We want p05, p50, p95 bands, computed across the p50 trajectories
        regimes[f'regime_{regime_idx+1}'] = {
            'sequences': regime_sequences,
            'n_sequences': len(regime_sequences),
            'p05': np.percentile(regime_sequences[:, 1, :], 5, axis=0),    # p05 of medians
            'p25': np.percentile(regime_sequences[:, 1, :], 25, axis=0),   # p25 of medians
            'p50': np.percentile(regime_sequences[:, 1, :], 50, axis=0),   # p50 of medians
            'p75': np.percentile(regime_sequences[:, 1, :], 75, axis=0),   # p75 of medians
            'p95': np.percentile(regime_sequences[:, 1, :], 95, axis=0),   # p95 of medians
        }

        print(f"  Regime {regime_idx+1}: {len(regime_sequences)} sequences")

    return regimes


def plot_four_regime_comparison(gt_regimes, oracle_regimes, prior_regimes):
    """
    Create 2×2 plot showing GT vs latent12_v2 for each regime.

    Layout:
        Regime 1 (Low)     | Regime 2 (Med-Low)
        Regime 3 (Med-High)| Regime 4 (High)
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    days = np.arange(1, HORIZON + 1)

    for regime_idx in range(N_REGIMES):
        regime_key = f'regime_{regime_idx+1}'
        ax = axes[regime_idx // 2, regime_idx % 2]

        gt = gt_regimes['regimes'][regime_key]
        oracle = oracle_regimes[regime_key]
        prior = prior_regimes[regime_key]

        # Plot GT bands (BLUE)
        ax.fill_between(days, gt['p05'], gt['p95'],
                        color='blue', alpha=0.15, label='GT p05-p95')
        ax.fill_between(days, gt['p25'], gt['p75'],
                        color='blue', alpha=0.25, label='GT IQR')
        ax.plot(days, gt['p50'], color='darkblue',
                linewidth=2.5, label='GT Median')

        # Plot Oracle bands (GREEN)
        ax.fill_between(days, oracle['p05'], oracle['p95'],
                        color='green', alpha=0.15, hatch='///',
                        edgecolor='green', linewidth=0,
                        label='Oracle p05-p95')
        ax.plot(days, oracle['p50'], color='darkgreen',
                linewidth=2, linestyle='--', label='Oracle Median')

        # Plot Prior bands (RED)
        ax.fill_between(days, prior['p05'], prior['p95'],
                        color='red', alpha=0.15, hatch='\\\\\\',
                        edgecolor='red', linewidth=0,
                        label='Prior p05-p95')
        ax.plot(days, prior['p50'], color='darkred',
                linewidth=2, linestyle=':', label='Prior Median')

        # Statistics box
        gt_day1_width = gt['p95'][0] - gt['p05'][0]
        gt_day90_width = gt['p95'][-1] - gt['p05'][-1]
        oracle_day1_width = oracle['p95'][0] - oracle['p05'][0]
        oracle_day90_width = oracle['p95'][-1] - oracle['p05'][-1]

        stats_text = (
            f"GT: Day-1 = {gt_day1_width:.4f}, Day-90 = {gt_day90_width:.4f}\n"
            f"Oracle: Day-1 = {oracle_day1_width:.4f} ({oracle_day1_width/gt_day1_width:.2f}×),\n"
            f"        Day-90 = {oracle_day90_width:.4f} ({oracle_day90_width/gt_day90_width:.2f}×)"
        )

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Title
        ax.set_title(
            f"{gt['label']}\n"
            f"Context Vol: {gt['context_vol_mean']:.3f} ± {gt['context_vol_std']:.3f}\n"
            f"N = {gt['n_sequences']} sequences",
            fontsize=13, fontweight='bold'
        )

        ax.set_xlabel('Days Ahead', fontsize=11)
        ax.set_ylabel('Normalized IV (relative to context endpoint)', fontsize=11)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=2)

    fig.suptitle(
        'Ground Truth vs Latent12 V2: 4-Regime Marginal Distribution Comparison\n'
        'Context60, Horizon=90, ATM 6M (K/S=1.00, τ=6mo)',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = OUTPUT_DIR / 'four_regime_gt_vs_latent12v2_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    print(f"\n✓ Saved: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

    return output_file


def generate_regime_statistics(gt_regimes, oracle_regimes, prior_regimes):
    """
    Generate CSV with regime-specific metrics.

    Columns:
    - regime, label, n_sequences, context_vol_mean, context_vol_std
    - gt_day1_width, gt_day90_width, gt_growth_pct
    - oracle_day1_width, oracle_day1_ratio, oracle_day90_width, oracle_day90_ratio
    - prior_day1_width, prior_day1_ratio, prior_day90_width, prior_day90_ratio
    """
    rows = []

    for regime_idx in range(N_REGIMES):
        regime_key = f'regime_{regime_idx+1}'
        gt = gt_regimes['regimes'][regime_key]
        oracle = oracle_regimes[regime_key]
        prior = prior_regimes[regime_key]

        gt_day1 = gt['p95'][0] - gt['p05'][0]
        gt_day90 = gt['p95'][-1] - gt['p05'][-1]
        gt_growth = ((gt_day90 - gt_day1) / gt_day1) * 100

        oracle_day1 = oracle['p95'][0] - oracle['p05'][0]
        oracle_day90 = oracle['p95'][-1] - oracle['p05'][-1]

        prior_day1 = prior['p95'][0] - prior['p05'][0]
        prior_day90 = prior['p95'][-1] - prior['p05'][-1]

        rows.append({
            'regime': regime_idx + 1,
            'label': gt['label'],
            'n_sequences': gt['n_sequences'],
            'context_vol_mean': gt['context_vol_mean'],
            'context_vol_std': gt['context_vol_std'],
            'gt_day1_width': gt_day1,
            'gt_day90_width': gt_day90,
            'gt_growth_pct': gt_growth,
            'oracle_day1_width': oracle_day1,
            'oracle_day1_ratio': oracle_day1 / gt_day1,
            'oracle_day90_width': oracle_day90,
            'oracle_day90_ratio': oracle_day90 / gt_day90,
            'prior_day1_width': prior_day1,
            'prior_day1_ratio': prior_day1 / gt_day1,
            'prior_day90_width': prior_day90,
            'prior_day90_ratio': prior_day90 / gt_day90,
        })

    df = pd.DataFrame(rows)
    csv_file = OUTPUT_DIR / 'regime_statistics.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')

    print(f"✓ Saved: {csv_file}")
    return df


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("4-REGIME MARGINAL DISTRIBUTION COMPARISON")
    print("Ground Truth vs Latent12 V2")
    print("=" * 80)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load GT stratified by regime
    gt_regimes = load_ground_truth_by_regime()
    print()

    # Load model predictions
    oracle_regimes = load_model_predictions_by_regime(
        'oracle', gt_regimes['thresholds']
    )
    print()

    prior_regimes = load_model_predictions_by_regime(
        'prior', gt_regimes['thresholds']
    )
    print()

    # Generate visualization
    print("Generating 4-regime comparison plot...")
    plot_file = plot_four_regime_comparison(
        gt_regimes, oracle_regimes, prior_regimes
    )
    print()

    # Generate statistics
    print("Generating regime statistics CSV...")
    stats_df = generate_regime_statistics(
        gt_regimes, oracle_regimes, prior_regimes
    )
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(stats_df.to_string(index=False))
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
