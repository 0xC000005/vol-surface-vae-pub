"""
Visualize marginal distributions of predictions at each grid point.

Compares Oracle, VAE Prior, and Econometric predictions across:
- Periods: In-Sample (2004-2019), OOS (2019-2023)
- Horizons: H1, H7, H14, H30

For each grid point (5x5 surface), plots overlaid histograms showing
the distribution of predicted values across all test days.

Output: 8 figures (4 horizons × 2 periods) saved to tables/marginal_distribution_plots/
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Grid labels
moneyness_labels = ['0.70', '0.85', '1.00', '1.15', '1.30']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

# Colors
colors = {
    'ground_truth': 'black',
    'oracle': 'blue',
    'vae_prior': 'red',
    'econometric': 'green'
}

# Output directory
output_dir = Path("tables/marginal_distribution_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def load_predictions(period, horizon):
    """Load predictions from all three methods for given period and horizon."""

    if period == 'insample':
        oracle_file = "models_backfill/insample_reconstruction_16yr.npz"
        vae_file = "models_backfill/vae_prior_insample_16yr.npz"
        econ_file = "tables/econometric_backfill/econometric_backfill_insample.npz"
    else:  # oos
        oracle_file = "models_backfill/oos_reconstruction_16yr.npz"
        vae_file = "models_backfill/vae_prior_oos_16yr.npz"
        econ_file = "tables/econometric_backfill/econometric_backfill_oos.npz"

    # Load data
    oracle_data = np.load(oracle_file)
    vae_data = np.load(vae_file)
    econ_data = np.load(econ_file)
    gt_data = np.load("data/vol_surface_with_ret.npz")

    # Extract reconstructions and indices
    recon_key = f'recon_h{horizon}'
    indices_key = f'indices_h{horizon}'

    # Extract p50 (median) quantile - index 1
    oracle_p50 = oracle_data[recon_key][:, 1, :, :]  # (N, 5, 5)
    vae_p50 = vae_data[recon_key][:, 1, :, :]  # (N, 5, 5)
    econ_p50 = econ_data[recon_key][:, 1, :, :]  # (N, 5, 5)

    # Get indices and ground truth
    indices = oracle_data[indices_key]
    gt = gt_data["surface"][indices]  # (N, 5, 5)

    return oracle_p50, vae_p50, econ_p50, gt


def plot_marginal_distributions(period, horizon):
    """Create 5x5 grid of marginal distribution plots."""

    print(f"\n{'='*80}")
    print(f"Period: {period.upper()}, Horizon: H{horizon}")
    print(f"{'='*80}")

    # Load data
    print("Loading predictions...")
    oracle_p50, vae_p50, econ_p50, gt = load_predictions(period, horizon)
    n_days = oracle_p50.shape[0]
    print(f"  Days: {n_days}")
    print(f"  Oracle shape: {oracle_p50.shape}")
    print(f"  VAE Prior shape: {vae_p50.shape}")
    print(f"  Econometric shape: {econ_p50.shape}")
    print(f"  Ground truth shape: {gt.shape}")

    # Create figure with 5x5 subplots
    fig, axes = plt.subplots(5, 5, figsize=(24, 24))
    fig.suptitle(f'Marginal Distributions - {period.upper()} H{horizon}\n'
                 f'p50 Predictions across {n_days} days',
                 fontsize=20, fontweight='bold', y=0.995)

    # Global min/max for consistent x-axis (optional)
    all_data = np.concatenate([
        oracle_p50.flatten(),
        vae_p50.flatten(),
        econ_p50.flatten(),
        gt.flatten()
    ])
    global_min = np.percentile(all_data, 1)
    global_max = np.percentile(all_data, 99)

    print("\nGenerating plots...")

    # Plot each grid point
    for i in range(5):  # Moneyness (rows)
        for j in range(5):  # Maturity (columns)
            ax = axes[i, j]

            # Extract data for this grid point across all days
            gt_vals = gt[:, i, j]
            oracle_vals = oracle_p50[:, i, j]
            vae_vals = vae_p50[:, i, j]
            econ_vals = econ_p50[:, i, j]

            # Plot overlaid histograms with density normalization
            bins = 40
            alpha = 0.4

            ax.hist(gt_vals, bins=bins, alpha=alpha, color=colors['ground_truth'],
                   density=True, label='Ground Truth', histtype='stepfilled', edgecolor='black', linewidth=0.5)
            ax.hist(oracle_vals, bins=bins, alpha=alpha, color=colors['oracle'],
                   density=True, label='Oracle', histtype='stepfilled', edgecolor=colors['oracle'], linewidth=0.5)
            ax.hist(vae_vals, bins=bins, alpha=alpha, color=colors['vae_prior'],
                   density=True, label='VAE Prior', histtype='stepfilled', edgecolor=colors['vae_prior'], linewidth=0.5)
            ax.hist(econ_vals, bins=bins, alpha=alpha, color=colors['econometric'],
                   density=True, label='Econometric', histtype='stepfilled', edgecolor=colors['econometric'], linewidth=0.5)

            # Set x-axis limits
            ax.set_xlim(global_min, global_max)

            # Labels
            if i == 0:
                ax.set_title(f'{maturity_labels[j]}', fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'K={moneyness_labels[i]}\nDensity', fontsize=12, fontweight='bold')

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # Statistics
            stats_text = (
                f'GT: μ={gt_vals.mean():.3f}, σ={gt_vals.std():.3f}\n'
                f'Or: μ={oracle_vals.mean():.3f}, σ={oracle_vals.std():.3f}\n'
                f'VAE: μ={vae_vals.mean():.3f}, σ={vae_vals.std():.3f}\n'
                f'Econ: μ={econ_vals.mean():.3f}, σ={econ_vals.std():.3f}'
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5))

    # Add legend (only once, in top-right corner)
    handles, labels = axes[0, 4].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=14, frameon=True,
              bbox_to_anchor=(0.99, 0.99), ncol=1)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.995])

    # Save figure
    output_file = output_dir / f'{period}_h{horizon}_marginal_distributions.png'
    print(f"\nSaving figure to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_file}")
    print()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MARGINAL DISTRIBUTION COMPARISON")
    print("Oracle vs VAE Prior vs Econometric")
    print("=" * 80)

    periods = ['insample', 'oos']
    horizons = [1, 7, 14, 30]

    total_plots = len(periods) * len(horizons)
    count = 0

    for period in periods:
        for horizon in horizons:
            count += 1
            print(f"\n[{count}/{total_plots}] Generating {period.upper()} H{horizon}...")

            try:
                plot_marginal_distributions(period, horizon)
            except Exception as e:
                print(f"ERROR: Failed to generate {period} H{horizon}")
                print(f"  {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated {count} figures in: {output_dir}")
    print("\nOutput files:")
    for period in periods:
        for horizon in horizons:
            filename = f'{period}_h{horizon}_marginal_distributions.png'
            filepath = output_dir / filename
            if filepath.exists():
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (missing)")
    print()
