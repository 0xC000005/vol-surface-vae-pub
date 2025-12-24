"""
Visualize Fanning Pattern: Standard vs Fitted Prior vs Ground Truth

Comprehensive visualization comparing standard N(0,1) prior vs fitted GMM prior
vs ground truth to show bias correction in the fanning pattern.

Creates 4-panel comparison:
- Panel 1: Standard Prior N(0,1) (shows systematic negative bias)
- Panel 2: Fitted GMM Prior (shows reduced bias)
- Panel 3: Ground Truth (empirical marginal distribution)
- Panel 4: Overlay comparison of all three sources

Usage:
    PYTHONPATH=. python experiments/backfill/context60/visualize_fitted_prior_fanning.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from vae.cvae_with_mem_randomized import CVAEMemRand

# Config
MODEL_PATH = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_best.pt"
FITTED_PRIOR_PATH = "models/backfill/context60_experiment/fitted_prior_gmm.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_DIR = Path("results/context60_latent12_v2/analysis/fitted_prior_comparison")
CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)
N_SAMPLES = 500  # Number of sequences to visualize
ALPHA = 0.02  # Transparency


def load_model():
    """Load trained model."""
    model_data = torch.load(MODEL_PATH, weights_only=False)
    model = CVAEMemRand(model_data["model_config"])
    model.load_weights(dict_to_load=model_data)
    model.eval()
    return model


def generate_predictions(model, data, prior_mode, n_samples):
    """Generate predictions for visualization."""
    print(f"Generating {n_samples} predictions with prior_mode='{prior_mode}'...")

    vol_surf = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    predictions = []
    indices = []  # Track indices for GT alignment

    with torch.no_grad():
        for i in range(CONTEXT_LEN, min(CONTEXT_LEN + n_samples, len(vol_surf) - HORIZON)):
            context_surface = vol_surf[i-CONTEXT_LEN:i]
            context_ex = ex_data[i-CONTEXT_LEN:i]

            context = {
                "surface": torch.from_numpy(context_surface).unsqueeze(0).double(),
                "ex_feats": torch.from_numpy(context_ex).unsqueeze(0).double()
            }

            surf_pred, _ = model.get_surface_given_conditions(
                context, z=None, horizon=HORIZON, prior_mode=prior_mode
            )

            # Extract ATM 6M p50 (median)
            pred_p50 = surf_pred[0, :, 1, ATM_6M[0], ATM_6M[1]].cpu().numpy()

            # Anchor to starting point
            pred_p50 = pred_p50 - pred_p50[0]
            predictions.append(pred_p50)
            indices.append(i)  # Store index for GT alignment

    print(f"  Generated {len(predictions)} predictions")
    return np.array(predictions), indices


def extract_ground_truth_trajectories(data, indices, context_len, horizon):
    """
    Extract ground truth trajectories aligned to model prediction indices.

    Args:
        data: npz data with 'surface' key
        indices: list of forecast start positions (same as model sampling)
        context_len: context window length (60)
        horizon: forecast horizon (90)

    Returns:
        gt_trajectories: (n_sequences, horizon) array, normalized to context endpoint
    """
    vol_surf = data['surface']
    atm_6m_series = vol_surf[:, ATM_6M[0], ATM_6M[1]]  # Extract ATM 6M time series

    gt_trajectories = []

    for start_idx in indices:
        # Context: 60 days before forecast start
        context_start = start_idx - context_len
        context = atm_6m_series[context_start:start_idx]
        anchor = context[-1]  # Last day of context (normalization point)

        # Forecast: 90 days after context
        forecast = atm_6m_series[start_idx:start_idx + horizon]

        # Normalize to anchor (SAME as model predictions!)
        normalized = forecast - anchor
        gt_trajectories.append(normalized)

    return np.array(gt_trajectories)


def plot_comparison(standard_preds, fitted_preds, gt_preds):
    """Create 4-panel fanning pattern comparison with ground truth."""
    fig, axes = plt.subplots(1, 4, figsize=(32, 6))
    days = np.arange(0, HORIZON)

    # Subsample for visualization
    n_plot = min(500, len(standard_preds))
    std_indices = np.random.choice(len(standard_preds), n_plot, replace=False)
    fit_indices = np.random.choice(len(fitted_preds), n_plot, replace=False)
    gt_indices = np.random.choice(len(gt_preds), n_plot, replace=False)

    # =========================================================================
    # Left: Standard Prior
    # =========================================================================
    ax = axes[0]
    for idx in std_indices:
        ax.plot(days, standard_preds[idx], color='darkred', alpha=ALPHA, linewidth=0.5)

    # Envelope
    p05 = np.percentile(standard_preds, 5, axis=0)
    p50 = np.percentile(standard_preds, 50, axis=0)
    p95 = np.percentile(standard_preds, 95, axis=0)

    ax.plot(days, p05, color='red', linewidth=2.5, linestyle='--', label='p05-p95')
    ax.plot(days, p95, color='red', linewidth=2.5, linestyle='--')
    ax.plot(days, p50, color='darkred', linewidth=3, label='median')

    ax.set_title(f'Standard Prior N(0,1)\n{len(standard_preds)} sequences ({n_plot} shown)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.15, 0.15)

    # Add statistics
    mean_traj = np.mean(standard_preds, axis=0)
    stats_text = (
        f"Day-30 mean: {mean_traj[29]:.4f}\n"
        f"Day-60 mean: {mean_traj[59]:.4f}\n"
        f"Day-90 mean: {mean_traj[89]:.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # Middle: Fitted Prior
    # =========================================================================
    ax = axes[1]
    for idx in fit_indices:
        ax.plot(days, fitted_preds[idx], color='darkgreen', alpha=ALPHA, linewidth=0.5)

    # Envelope
    p05 = np.percentile(fitted_preds, 5, axis=0)
    p50 = np.percentile(fitted_preds, 50, axis=0)
    p95 = np.percentile(fitted_preds, 95, axis=0)

    ax.plot(days, p05, color='green', linewidth=2.5, linestyle='--', label='p05-p95')
    ax.plot(days, p95, color='green', linewidth=2.5, linestyle='--')
    ax.plot(days, p50, color='darkgreen', linewidth=3, label='median')

    ax.set_title(f'Fitted Prior (GMM)\n{len(fitted_preds)} sequences ({n_plot} shown)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.15, 0.15)

    # Add statistics
    mean_traj = np.mean(fitted_preds, axis=0)
    stats_text = (
        f"Day-30 mean: {mean_traj[29]:.4f}\n"
        f"Day-60 mean: {mean_traj[59]:.4f}\n"
        f"Day-90 mean: {mean_traj[89]:.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # Panel 3: Ground Truth
    # =========================================================================
    ax = axes[2]
    for idx in gt_indices:
        ax.plot(days, gt_preds[idx], color='darkblue', alpha=ALPHA, linewidth=0.5)

    # Envelope
    gt_p05 = np.percentile(gt_preds, 5, axis=0)
    gt_p50 = np.percentile(gt_preds, 50, axis=0)
    gt_p95 = np.percentile(gt_preds, 95, axis=0)

    ax.plot(days, gt_p05, color='blue', linewidth=2.5, linestyle='--', label='p05-p95')
    ax.plot(days, gt_p95, color='blue', linewidth=2.5, linestyle='--')
    ax.plot(days, gt_p50, color='darkblue', linewidth=3, label='median')

    ax.set_title(f'Ground Truth\n{len(gt_preds)} sequences ({n_plot} shown)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.15, 0.15)

    # Add statistics
    gt_mean_traj = np.mean(gt_preds, axis=0)
    stats_text = (
        f"Day-30 mean: {gt_mean_traj[29]:.4f}\n"
        f"Day-60 mean: {gt_mean_traj[59]:.4f}\n"
        f"Day-90 mean: {gt_mean_traj[89]:.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # Panel 4: Overlay Comparison
    # =========================================================================
    ax = axes[3]

    # Compute percentiles
    std_p05 = np.percentile(standard_preds, 5, axis=0)
    std_p50 = np.percentile(standard_preds, 50, axis=0)
    std_p95 = np.percentile(standard_preds, 95, axis=0)

    fit_p05 = np.percentile(fitted_preds, 5, axis=0)
    fit_p50 = np.percentile(fitted_preds, 50, axis=0)
    fit_p95 = np.percentile(fitted_preds, 95, axis=0)

    # Plot envelopes (filled bands)
    ax.fill_between(days, gt_p05, gt_p95, color='blue', alpha=0.15, label='GT p05-p95')
    ax.fill_between(days, fit_p05, fit_p95, color='green', alpha=0.15, label='Fitted p05-p95')
    ax.fill_between(days, std_p05, std_p95, color='red', alpha=0.15, label='Standard p05-p95')

    # Plot medians
    ax.plot(days, gt_p50, color='darkblue', linewidth=2.5, linestyle='-', label='GT median')
    ax.plot(days, fit_p50, color='darkgreen', linewidth=2.5, linestyle='--', label='Fitted median')
    ax.plot(days, std_p50, color='darkred', linewidth=2.5, linestyle='-.', label='Standard median')

    ax.set_title('Overlay: Standard vs Fitted vs Ground Truth',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Normalized Change from Start', fontsize=11)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.15, 0.15)

    # Add comparison statistics
    gt_mean_abs = np.abs([gt_p50[29], gt_p50[59], gt_p50[89]]).mean()
    fit_mean_abs = np.abs([fit_p50[29], fit_p50[59], fit_p50[89]]).mean()
    std_mean_abs = np.abs([std_p50[29], std_p50[59], std_p50[89]]).mean()

    stats_text = (
        f"Avg |median|:\n"
        f"GT:       {gt_mean_abs:.4f}\n"
        f"Fitted:   {fit_mean_abs:.4f}\n"
        f"Standard: {std_mean_abs:.4f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle(
        'Fanning Pattern Comparison: Standard N(0,1) vs Fitted GMM vs Ground Truth\n'
        'Context60 Latent12 V2 - All Trajectories Anchored to Starting Point (0)',
        fontsize=16, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    output_file = OUTPUT_DIR / 'fanning_pattern_standard_vs_fitted_vs_gt.png'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    print(f"\n✓ Saved: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

    return output_file


def main():
    """Main visualization pipeline."""
    print("=" * 80)
    print("FANNING PATTERN: STANDARD VS FITTED PRIOR COMPARISON")
    print("=" * 80)
    print()

    torch.set_default_dtype(torch.float64)

    # Load model
    print("Loading model...")
    model = load_model()
    print("✓ Model loaded")

    # Load data
    print("\nLoading data...")
    data = np.load(DATA_PATH)
    print(f"✓ Data loaded ({len(data['surface'])} days)")

    # Generate with standard prior
    print()
    standard_preds, indices = generate_predictions(model, data, "standard", N_SAMPLES)

    # Load fitted prior and generate
    print(f"\nLoading fitted prior from {FITTED_PRIOR_PATH}...")
    model.load_fitted_prior(FITTED_PRIOR_PATH)
    print("✓ Fitted prior loaded")

    print()
    fitted_preds, _ = generate_predictions(model, data, "fitted", N_SAMPLES)

    # Extract ground truth for same indices
    print("\nExtracting ground truth trajectories...")
    gt_preds = extract_ground_truth_trajectories(data, indices, CONTEXT_LEN, HORIZON)
    print(f"  Extracted {len(gt_preds)} ground truth sequences")

    # Create visualization
    print("\nCreating visualization...")
    plot_file = plot_comparison(standard_preds, fitted_preds, gt_preds)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print()
    print("Key findings:")

    std_mean = np.mean(standard_preds, axis=0)
    fit_mean = np.mean(fitted_preds, axis=0)
    gt_mean = np.mean(gt_preds, axis=0)

    print(f"  Day-90 mean:")
    print(f"    Ground Truth: {gt_mean[89]:.6f} (empirical)")
    print(f"    Standard:     {std_mean[89]:.6f} (negative drift)")
    print(f"    Fitted:       {fit_mean[89]:.6f} (reduced bias)")
    print(f"    Standard bias vs GT: {abs(std_mean[89] - gt_mean[89]):.6f}")
    print(f"    Fitted bias vs GT:   {abs(fit_mean[89] - gt_mean[89]):.6f}")

    if abs(fit_mean[89] - gt_mean[89]) < abs(std_mean[89] - gt_mean[89]):
        print("\n  ✓ Fitted prior is closer to ground truth than standard prior!")

    print(f"\nVisualization saved to: {plot_file}")


if __name__ == "__main__":
    main()
