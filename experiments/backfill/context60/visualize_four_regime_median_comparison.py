"""
Visualize Four-Regime Median Marginal Distribution Comparison - Context60 Latent12 V2

Creates a 2×2 plot comparing ONLY the p50 (median) marginal distributions,
ignoring the p05 and p95 quantile heads. Shows ground truth median vs model
median for each volatility regime.

**Purpose**: Provide a cleaner, simpler view focusing only on median trajectories
to understand central tendency behavior across regimes, without the visual clutter
of percentile bands.

**Four regimes (volatility quartiles):**
- Q1 (0-25%): Low volatility regime
- Q2 (25-50%): Medium-low volatility regime
- Q3 (50-75%): Medium-high volatility regime
- Q4 (75-100%): High volatility regime

**Layout:** 2×2 grid showing GT median vs Oracle median vs Prior median for each regime

Usage:
    PYTHONPATH=. python experiments/backfill/context60/visualize_four_regime_median_comparison.py

Output:
    results/context60_latent12_v2/analysis/regime_median_comparison/
    ├── four_regime_median_comparison.png
    └── median_statistics.csv
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
OUTPUT_DIR = Path("results/context60_latent12_v2/analysis/regime_median_comparison")


def load_ground_truth_by_regime():
    """
    Load GT and stratify into 4 regimes by context volatility.
    Compute ONLY p50 (median) for each regime.

    Returns:
        dict with:
        - 'regimes': dict of regime data (p50, sequences, context vol stats)
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
            'p50': np.median(regime_sequences, axis=0),  # ONLY median
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
    Extract ONLY p50 (median) channel.

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

    # Extract ONLY p50 channel (idx=1 in quantile dimension)
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

        # Extract ONLY p50 and normalize
        anchor = context[-1]
        seq_p50 = surfaces[i, :, 1, ATM_6M[0], ATM_6M[1]] - anchor  # (90,)
        sequences.append(seq_p50)

    sequences = np.array(sequences)  # (N, 90)
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

        regime_sequences = sequences[mask]  # (n, 90)

        regimes[f'regime_{regime_idx+1}'] = {
            'sequences': regime_sequences,
            'n_sequences': len(regime_sequences),
            'p50': np.median(regime_sequences, axis=0),  # Median of medians
        }

        print(f"  Regime {regime_idx+1}: {len(regime_sequences)} sequences")

    return regimes


def plot_four_regime_median_comparison(gt_regimes, oracle_regimes, prior_regimes):
    """
    Create 2×2 plot showing GT vs Oracle vs Prior MEDIANS ONLY.

    NO percentile bands - just clean line plots.

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

        # Plot ONLY medians (no bands)
        ax.plot(days, gt['p50'], color='darkblue',
                linewidth=3, label='GT Median', alpha=0.9, zorder=3)

        ax.plot(days, oracle['p50'], color='darkgreen',
                linewidth=2.5, linestyle='--', label='Oracle Median', alpha=0.8, zorder=2)

        ax.plot(days, prior['p50'], color='darkred',
                linewidth=2.5, linestyle=':', label='Prior Median', alpha=0.8, zorder=1)

        # Statistics box with RMSE
        oracle_rmse = np.sqrt(np.mean((oracle['p50'] - gt['p50'])**2))
        prior_rmse = np.sqrt(np.mean((prior['p50'] - gt['p50'])**2))

        stats_text = (
            f"Oracle RMSE: {oracle_rmse:.6f}\n"
            f"Prior RMSE: {prior_rmse:.6f}\n"
            f"Day-1: GT={gt['p50'][0]:.4f}, Oracle={oracle['p50'][0]:.4f}, Prior={prior['p50'][0]:.4f}\n"
            f"Day-90: GT={gt['p50'][-1]:.4f}, Oracle={oracle['p50'][-1]:.4f}, Prior={prior['p50'][-1]:.4f}"
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
        ax.set_ylabel('Normalized IV (median, relative to context endpoint)', fontsize=11)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

    fig.suptitle(
        'Ground Truth vs Latent12 V2: Median Marginal Distribution Comparison (4 Regimes)\n'
        'Context60, Horizon=90, ATM 6M - Median Trajectories Only',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = OUTPUT_DIR / 'four_regime_median_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    print(f"\n✓ Saved: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

    return output_file


def generate_median_statistics(gt_regimes, oracle_regimes, prior_regimes):
    """
    Generate CSV with median-specific metrics.

    Columns:
    - regime, label, n_sequences, context_vol_mean, context_vol_std
    - gt_median_day1, gt_median_day90
    - oracle_median_day1, oracle_median_day90, oracle_rmse
    - prior_median_day1, prior_median_day90, prior_rmse
    """
    rows = []

    for regime_idx in range(N_REGIMES):
        regime_key = f'regime_{regime_idx+1}'
        gt = gt_regimes['regimes'][regime_key]
        oracle = oracle_regimes[regime_key]
        prior = prior_regimes[regime_key]

        oracle_rmse = np.sqrt(np.mean((oracle['p50'] - gt['p50'])**2))
        prior_rmse = np.sqrt(np.mean((prior['p50'] - gt['p50'])**2))

        rows.append({
            'regime': regime_idx + 1,
            'label': gt['label'],
            'n_sequences': gt['n_sequences'],
            'context_vol_mean': gt['context_vol_mean'],
            'context_vol_std': gt['context_vol_std'],
            'gt_median_day1': gt['p50'][0],
            'gt_median_day90': gt['p50'][-1],
            'oracle_median_day1': oracle['p50'][0],
            'oracle_median_day90': oracle['p50'][-1],
            'oracle_rmse': oracle_rmse,
            'prior_median_day1': prior['p50'][0],
            'prior_median_day90': prior['p50'][-1],
            'prior_rmse': prior_rmse,
        })

    df = pd.DataFrame(rows)
    csv_file = OUTPUT_DIR / 'median_statistics.csv'
    df.to_csv(csv_file, index=False, float_format='%.6f')

    print(f"✓ Saved: {csv_file}")
    return df


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("4-REGIME MEDIAN DISTRIBUTION COMPARISON")
    print("Ground Truth vs Latent12 V2 (Medians Only)")
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
    print("Generating median comparison plot...")
    plot_file = plot_four_regime_median_comparison(
        gt_regimes, oracle_regimes, prior_regimes
    )
    print()

    # Generate statistics
    print("Generating median statistics CSV...")
    stats_df = generate_median_statistics(
        gt_regimes, oracle_regimes, prior_regimes
    )
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY (RMSE across 90-day horizon)")
    print("=" * 80)
    print()
    print(stats_df[['regime', 'label', 'oracle_rmse', 'prior_rmse']].to_string(index=False))
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
