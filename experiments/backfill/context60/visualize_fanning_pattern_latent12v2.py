"""
Visualize Fanning Pattern Comparison - Context60 Latent12 V2

Creates visualizations showing the "fanning pattern" - when all trajectories are
anchored to a fixed starting point (0), uncertainty naturally grows over time,
creating a visual "fan" shape.

**Purpose**: Compare whether the model replicates the natural trajectory dispersion
dynamics seen in ground truth data. Unlike percentile bands (aggregated statistics)
or median plots (central tendency), fanning patterns show RAW trajectory spread.

**Two visualizations:**
1. All Regimes Combined (2×2): GT / Oracle / Prior / Overlay
2. By Regime (2×2): One panel per volatility quartile

Usage:
    PYTHONPATH=. python experiments/backfill/context60/visualize_fanning_pattern_latent12v2.py

Output:
    results/context60_latent12_v2/analysis/fanning_pattern/
    ├── fanning_pattern_all_regimes.png
    ├── fanning_pattern_by_regime.png
    └── fanning_metrics.csv
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
SAMPLE_SIZE = 5000  # Subsample for visualization (too many lines = clutter)
ALPHA = 0.02  # Transparency for individual trajectories
OUTPUT_DIR = Path("results/context60_latent12_v2/analysis/fanning_pattern")


def load_ground_truth_trajectories():
    """
    Load ALL ground truth sequences, anchor to starting point (0).

    Returns:
        sequences: (N, 90) array, each row starts at 0
        context_vols: (N,) array for regime stratification
    """
    print("Loading ground truth trajectories...")

    gt_data = np.load("data/vol_surface_with_ret.npz")
    atm_6m = gt_data['surface'][:, ATM_6M[0], ATM_6M[1]]

    total_len = CONTEXT_LEN + HORIZON
    n_sequences = len(atm_6m) - total_len + 1

    sequences = []
    context_vols = []

    for i in range(n_sequences):
        context = atm_6m[i:i + CONTEXT_LEN]
        forecast = atm_6m[i + CONTEXT_LEN:i + total_len]

        # Calculate context volatility for regime stratification
        context_vol = np.mean(context)
        context_vols.append(context_vol)

        # Anchor to 0: subtract first value (starting point)
        anchor = forecast[0]
        sequences.append(forecast - anchor)

    sequences = np.array(sequences)  # (N, 90)
    context_vols = np.array(context_vols)

    print(f"  Loaded {n_sequences} sequences")
    print(f"  Context vol range: [{context_vols.min():.3f}, {context_vols.max():.3f}]")
    print(f"  All sequences anchored to starting point (0)")

    return sequences, context_vols


def load_model_trajectories(sampling_mode):
    """
    Load model predictions, extract p50 (median), anchor to starting point (0).

    Args:
        sampling_mode: 'oracle' or 'prior'

    Returns:
        sequences: (N, 90) array, each row starts at 0
        context_vols: (N,) for regime stratification
    """
    print(f"\nLoading {sampling_mode} predictions...")

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

        # Extract p50 (median prediction)
        seq_p50 = surfaces[i, :, 1, ATM_6M[0], ATM_6M[1]]  # (90,)

        # Anchor to 0: subtract first value (starting point)
        anchor = seq_p50[0]
        sequences.append(seq_p50 - anchor)

    sequences = np.array(sequences)  # (N, 90)
    context_vols = np.array(context_vols)

    print(f"  Valid sequences after filtering: {len(sequences)}")
    print(f"  All sequences anchored to starting point (0)")

    return sequences, context_vols


def plot_fanning_pattern_all_regimes(gt_seqs, oracle_seqs, prior_seqs):
    """
    Create 2×2 layout showing fanning patterns.

    Top-Left: GT Fan (all sequences)
    Top-Right: Oracle Fan (all sequences)
    Bottom-Left: Prior Fan (all sequences)
    Bottom-Right: Overlay comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    days = np.arange(0, HORIZON)

    # Subsample for visualization (too many lines = clutter)
    np.random.seed(42)  # For reproducibility
    n_gt = min(SAMPLE_SIZE, len(gt_seqs))
    n_oracle = min(SAMPLE_SIZE, len(oracle_seqs))
    n_prior = min(SAMPLE_SIZE, len(prior_seqs))

    gt_indices = np.random.choice(len(gt_seqs), n_gt, replace=False)
    oracle_indices = np.random.choice(len(oracle_seqs), n_oracle, replace=False)
    prior_indices = np.random.choice(len(prior_seqs), n_prior, replace=False)

    # =========================================================================
    # Top-Left: Ground Truth Fan
    # =========================================================================
    ax = axes[0, 0]
    for idx in gt_indices:
        ax.plot(days, gt_seqs[idx], color='darkblue', alpha=ALPHA, linewidth=0.5)

    # Add envelope (p05-p95)
    p05 = np.percentile(gt_seqs, 5, axis=0)
    p95 = np.percentile(gt_seqs, 95, axis=0)
    ax.plot(days, p05, color='blue', linewidth=2, label='p05-p95', linestyle='--')
    ax.plot(days, p95, color='blue', linewidth=2, linestyle='--')

    ax.set_title(f'Ground Truth Fan\n{len(gt_seqs)} sequences ({n_gt} shown)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================================
    # Top-Right: Oracle Fan
    # =========================================================================
    ax = axes[0, 1]
    for idx in oracle_indices:
        ax.plot(days, oracle_seqs[idx], color='darkgreen', alpha=ALPHA, linewidth=0.5)

    p05 = np.percentile(oracle_seqs, 5, axis=0)
    p95 = np.percentile(oracle_seqs, 95, axis=0)
    ax.plot(days, p05, color='green', linewidth=2, label='p05-p95', linestyle='--')
    ax.plot(days, p95, color='green', linewidth=2, linestyle='--')

    ax.set_title(f'Oracle Fan\n{len(oracle_seqs)} sequences ({n_oracle} shown)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================================
    # Bottom-Left: Prior Fan
    # =========================================================================
    ax = axes[1, 0]
    for idx in prior_indices:
        ax.plot(days, prior_seqs[idx], color='darkred', alpha=ALPHA, linewidth=0.5)

    p05 = np.percentile(prior_seqs, 5, axis=0)
    p95 = np.percentile(prior_seqs, 95, axis=0)
    ax.plot(days, p05, color='red', linewidth=2, label='p05-p95', linestyle='--')
    ax.plot(days, p95, color='red', linewidth=2, linestyle='--')

    ax.set_title(f'Prior Fan\n{len(prior_seqs)} sequences ({n_prior} shown)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================================
    # Bottom-Right: Overlay Comparison
    # =========================================================================
    ax = axes[1, 1]

    # Plot envelopes only (too busy with all lines)
    gt_p05 = np.percentile(gt_seqs, 5, axis=0)
    gt_p95 = np.percentile(gt_seqs, 95, axis=0)
    oracle_p05 = np.percentile(oracle_seqs, 5, axis=0)
    oracle_p95 = np.percentile(oracle_seqs, 95, axis=0)
    prior_p05 = np.percentile(prior_seqs, 5, axis=0)
    prior_p95 = np.percentile(prior_seqs, 95, axis=0)

    ax.fill_between(days, gt_p05, gt_p95, color='blue', alpha=0.2, label='GT p05-p95')
    ax.fill_between(days, oracle_p05, oracle_p95, color='green', alpha=0.2, label='Oracle p05-p95')
    ax.fill_between(days, prior_p05, prior_p95, color='red', alpha=0.2, label='Prior p05-p95')

    ax.plot(days, gt_p05, color='blue', linewidth=2, linestyle='--')
    ax.plot(days, gt_p95, color='blue', linewidth=2, linestyle='--')
    ax.plot(days, oracle_p05, color='green', linewidth=2, linestyle='--')
    ax.plot(days, oracle_p95, color='green', linewidth=2, linestyle='--')
    ax.plot(days, prior_p05, color='red', linewidth=2, linestyle='--')
    ax.plot(days, prior_p95, color='red', linewidth=2, linestyle='--')

    ax.set_title('Overlay: Envelope Comparison', fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()

    # =========================================================================
    # Main title and save
    # =========================================================================
    fig.suptitle(
        'Fanning Pattern Comparison: Ground Truth vs Latent12 V2\n'
        'All Trajectories Anchored to Starting Point (0)',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = OUTPUT_DIR / 'fanning_pattern_all_regimes.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    print(f"\n✓ Saved: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

    return output_file


def plot_fanning_pattern_by_regime(gt_seqs, gt_vols, oracle_seqs, oracle_vols,
                                    prior_seqs, prior_vols):
    """
    Create 2×2 layout showing fanning patterns PER REGIME.

    Shows whether fanning dynamics vary across volatility regimes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    days = np.arange(0, HORIZON)

    # Compute quartile thresholds (use GT as reference)
    thresholds = np.percentile(gt_vols, [25, 50, 75])

    regime_labels = [
        'Low Vol (Q1: 0-25%)',
        'Medium-Low (Q2: 25-50%)',
        'Medium-High (Q3: 50-75%)',
        'High Vol (Q4: 75-100%)'
    ]

    np.random.seed(42)  # For reproducibility

    for regime_idx in range(N_REGIMES):
        ax = axes[regime_idx // 2, regime_idx % 2]

        # Create regime mask
        if regime_idx == 0:
            gt_mask = gt_vols <= thresholds[0]
            oracle_mask = oracle_vols <= thresholds[0]
            prior_mask = prior_vols <= thresholds[0]
        elif regime_idx == N_REGIMES - 1:
            gt_mask = gt_vols > thresholds[-1]
            oracle_mask = oracle_vols > thresholds[-1]
            prior_mask = prior_vols > thresholds[-1]
        else:
            gt_mask = (gt_vols > thresholds[regime_idx-1]) & (gt_vols <= thresholds[regime_idx])
            oracle_mask = (oracle_vols > thresholds[regime_idx-1]) & (oracle_vols <= thresholds[regime_idx])
            prior_mask = (prior_vols > thresholds[regime_idx-1]) & (prior_vols <= thresholds[regime_idx])

        # Extract regime sequences
        gt_regime = gt_seqs[gt_mask]
        oracle_regime = oracle_seqs[oracle_mask]
        prior_regime = prior_seqs[prior_mask]

        # Subsample GT
        n_sample = min(500, len(gt_regime))
        if len(gt_regime) > 0:
            sample_indices = np.random.choice(len(gt_regime), n_sample, replace=False)
            for idx in sample_indices:
                ax.plot(days, gt_regime[idx], color='blue', alpha=0.03, linewidth=0.5)

        # Subsample Oracle
        n_oracle_sample = min(200, len(oracle_regime))
        if len(oracle_regime) > 0:
            oracle_sample = np.random.choice(len(oracle_regime), n_oracle_sample, replace=False)
            for idx in oracle_sample:
                ax.plot(days, oracle_regime[idx], color='green', alpha=0.05, linewidth=0.5)

        # Plot envelopes
        if len(gt_regime) > 0:
            gt_p05 = np.percentile(gt_regime, 5, axis=0)
            gt_p95 = np.percentile(gt_regime, 95, axis=0)
            ax.plot(days, gt_p05, color='blue', linewidth=2.5, label='GT p05-p95', linestyle='--')
            ax.plot(days, gt_p95, color='blue', linewidth=2.5, linestyle='--')

        if len(oracle_regime) > 0:
            oracle_p05 = np.percentile(oracle_regime, 5, axis=0)
            oracle_p95 = np.percentile(oracle_regime, 95, axis=0)
            ax.plot(days, oracle_p05, color='green', linewidth=2, label='Oracle p05-p95', linestyle=':')
            ax.plot(days, oracle_p95, color='green', linewidth=2, linestyle=':')

        ax.set_title(f'{regime_labels[regime_idx]}\n'
                     f'GT: {len(gt_regime)} seqs, Oracle: {len(oracle_regime)} seqs',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Days Ahead', fontsize=10)
        ax.set_ylabel('Normalized Change from Start', fontsize=10)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        'Fanning Pattern by Volatility Regime\n'
        'Blue: Ground Truth | Green: Oracle Predictions',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    output_file = OUTPUT_DIR / 'fanning_pattern_by_regime.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    print(f"✓ Saved: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

    return output_file


def compute_fanning_metrics(gt_seqs, oracle_seqs, prior_seqs):
    """
    Compute quantitative metrics for fanning pattern comparison.

    Metrics:
    - Spread growth rate (p95-p05 at day 1 vs day 90)
    - Envelope width at various horizons
    """
    print("\nComputing fanning metrics...")

    metrics = {}

    for name, seqs in [('GT', gt_seqs), ('Oracle', oracle_seqs), ('Prior', prior_seqs)]:
        # Envelope widths
        p05 = np.percentile(seqs, 5, axis=0)
        p95 = np.percentile(seqs, 95, axis=0)
        width = p95 - p05

        growth_rate = (width[89] - width[0]) / width[0] * 100 if width[0] > 0 else 0

        metrics[name] = {
            'day1_width': width[0],
            'day30_width': width[29],
            'day60_width': width[59],
            'day90_width': width[89],
            'growth_rate': growth_rate,
            'mean_width': np.mean(width),
        }

        print(f"  {name} day-1 width: {width[0]:.6f}")
        print(f"  {name} day-90 width: {width[89]:.6f}")
        print(f"  {name} growth rate: {growth_rate:.1f}%")

    return metrics


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("FANNING PATTERN ANALYSIS")
    print("Ground Truth vs Latent12 V2")
    print("=" * 80)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    gt_seqs, gt_vols = load_ground_truth_trajectories()
    oracle_seqs, oracle_vols = load_model_trajectories('oracle')
    prior_seqs, prior_vols = load_model_trajectories('prior')

    # Compute metrics
    metrics = compute_fanning_metrics(gt_seqs, oracle_seqs, prior_seqs)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    print("\n1. Generating all-regimes fanning plot...")
    plot_fanning_pattern_all_regimes(gt_seqs, oracle_seqs, prior_seqs)

    print("\n2. Generating by-regime fanning plot...")
    plot_fanning_pattern_by_regime(gt_seqs, gt_vols, oracle_seqs, oracle_vols,
                                    prior_seqs, prior_vols)

    # Save metrics CSV
    print("\n3. Generating fanning metrics CSV...")
    df = pd.DataFrame(metrics).T
    csv_file = OUTPUT_DIR / 'fanning_metrics.csv'
    df.to_csv(csv_file, float_format='%.6f')
    print(f"✓ Saved: {csv_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("FANNING PATTERN SUMMARY")
    print("=" * 80)
    print()
    print(df[['day1_width', 'day90_width', 'growth_rate']].to_string())
    print()

    print("Key findings:")
    gt_growth = metrics['GT']['growth_rate']
    oracle_growth = metrics['Oracle']['growth_rate']
    prior_growth = metrics['Prior']['growth_rate']

    print(f"  - GT natural divergence: +{gt_growth:.1f}%")
    print(f"  - Oracle model divergence: +{oracle_growth:.1f}%")
    print(f"  - Prior model divergence: +{prior_growth:.1f}%")
    print()

    if abs(oracle_growth - gt_growth) < 50:
        print("  ✅ Model fans show similar divergence dynamics to GT")
    else:
        print(f"  ⚠️  Model fans differ from GT by {abs(oracle_growth - gt_growth):.1f}pp")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
