"""
Visualize Fanning Pattern Comparison - Context60 V3 FIXED

Creates visualizations showing the "fanning pattern" - when all trajectories are
anchored to a fixed starting point (0), uncertainty naturally grows over time,
creating a visual "fan" shape.

Usage:
    PYTHONPATH=. python experiments/backfill/context60/visualize_fanning_pattern_v3_fixed.py

Output:
    results/context60_latent12_v3_FIXED/analysis/fanning_pattern/
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
OUTPUT_DIR = Path("results/context60_latent12_v3_FIXED/analysis/fanning_pattern")


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
    pred_file = (f"results/context60_latent12_v3_FIXED/predictions/teacher_forcing/"
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

    return sequences, context_vols


def compute_fanning_metrics(gt_seqs, oracle_seqs, prior_seqs):
    """
    Compute quantitative metrics for fanning patterns.

    Returns:
        DataFrame with metrics by horizon
    """
    # Align lengths
    n_total = min(len(gt_seqs), len(oracle_seqs), len(prior_seqs))
    gt_seqs = gt_seqs[:n_total]
    oracle_seqs = oracle_seqs[:n_total]
    prior_seqs = prior_seqs[:n_total]

    horizons = [1, 7, 14, 30, 60, 90]
    metrics = []

    for h in horizons:
        h_idx = h - 1  # 0-indexed

        gt_std = gt_seqs[:, h_idx].std()
        oracle_std = oracle_seqs[:, h_idx].std()
        prior_std = prior_seqs[:, h_idx].std()

        metrics.append({
            'horizon': h,
            'gt_std': gt_std,
            'oracle_std': oracle_std,
            'prior_std': prior_std,
            'oracle_ratio': oracle_std / gt_std if gt_std > 0 else 0,
            'prior_ratio': prior_std / gt_std if gt_std > 0 else 0,
            'oracle_prior_gap': (prior_std - oracle_std) / gt_std if gt_std > 0 else 0,
        })

    return pd.DataFrame(metrics)


def plot_all_regimes(gt_seqs, oracle_seqs, prior_seqs, output_path):
    """
    Create 2x2 plot: GT / Oracle / Prior / Overlay.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    days = np.arange(1, HORIZON + 1)

    # Subsample - use minimum length across all datasets
    n_total = min(len(gt_seqs), len(oracle_seqs), len(prior_seqs))
    gt_seqs = gt_seqs[:n_total]  # Truncate to common length
    oracle_seqs = oracle_seqs[:n_total]
    prior_seqs = prior_seqs[:n_total]

    subsample_idx = np.random.choice(n_total, min(SAMPLE_SIZE, n_total), replace=False)

    # Top-left: Ground Truth
    ax = axes[0, 0]
    for seq in gt_seqs[subsample_idx]:
        ax.plot(days, seq, color='black', alpha=ALPHA, linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title("Ground Truth Trajectories", fontsize=14, fontweight='bold')
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Change from Starting Point")
    ax.grid(True, alpha=0.3)

    # Top-right: Oracle
    ax = axes[0, 1]
    for seq in oracle_seqs[subsample_idx]:
        ax.plot(days, seq, color='blue', alpha=ALPHA, linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title("Oracle (Posterior) Predictions", fontsize=14, fontweight='bold')
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Change from Starting Point")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Prior
    ax = axes[1, 0]
    for seq in prior_seqs[subsample_idx]:
        ax.plot(days, seq, color='green', alpha=ALPHA, linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title("Prior (Conditional) Predictions", fontsize=14, fontweight='bold')
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Change from Starting Point")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Overlay
    ax = axes[1, 1]
    # Plot means + std bands
    gt_mean = gt_seqs.mean(axis=0)
    gt_std = gt_seqs.std(axis=0)
    oracle_mean = oracle_seqs.mean(axis=0)
    oracle_std = oracle_seqs.std(axis=0)
    prior_mean = prior_seqs.mean(axis=0)
    prior_std = prior_seqs.std(axis=0)

    ax.fill_between(days, gt_mean - gt_std, gt_mean + gt_std,
                     color='black', alpha=0.2, label='GT ±1σ')
    ax.fill_between(days, oracle_mean - oracle_std, oracle_mean + oracle_std,
                     color='blue', alpha=0.2, label='Oracle ±1σ')
    ax.fill_between(days, prior_mean - prior_std, prior_mean + prior_std,
                     color='green', alpha=0.2, label='Prior ±1σ')

    ax.plot(days, gt_mean, color='black', linewidth=2, label='GT Mean')
    ax.plot(days, oracle_mean, color='blue', linewidth=2, label='Oracle Mean')
    ax.plot(days, prior_mean, color='green', linewidth=2, label='Prior Mean')

    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title("Overlay: Mean ± 1σ", fontsize=14, fontweight='bold')
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Change from Starting Point")
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Fanning Pattern Analysis - V3 FIXED (N={n_total} sequences)",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FANNING PATTERN VISUALIZATION - V3 FIXED")
    print("="*80)

    # Load data
    gt_seqs, gt_vols = load_ground_truth_trajectories()

    try:
        oracle_seqs, oracle_vols = load_model_trajectories('oracle')
    except FileNotFoundError:
        print("\n⚠️  Oracle predictions not found, skipping oracle analysis")
        oracle_seqs = gt_seqs  # Fallback to GT

    prior_seqs, prior_vols = load_model_trajectories('prior')

    # Compute metrics
    print("\n" + "="*80)
    print("FANNING METRICS")
    print("="*80)

    metrics_df = compute_fanning_metrics(gt_seqs, oracle_seqs, prior_seqs)
    print(metrics_df.to_string(index=False))

    # Save metrics
    metrics_path = OUTPUT_DIR / "fanning_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Saved metrics: {metrics_path}")

    # Plot
    plot_path = OUTPUT_DIR / "fanning_pattern_all_regimes.png"
    plot_all_regimes(gt_seqs, oracle_seqs, prior_seqs, plot_path)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
