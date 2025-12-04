"""
Visualize variance ratios as heatmaps to show spatial patterns.

Creates 5×5 heatmaps showing model_std / GT_std for each grid point.
Helps identify where models track GT variance well (ratio ~1.0) vs
where they underestimate (ratio <0.9) or overestimate (ratio >1.1).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

moneyness_labels = ['0.70', '0.85', '1.00', '1.15', '1.30']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

output_dir = Path("results/variance_ratio_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================

def compute_variance_ratios(period, horizon):
    """Compute variance ratios for all grid points."""

    if period == 'insample':
        oracle_file = "models/backfill/insample_reconstruction_16yr.npz"
        vae_file = "models/backfill/vae_prior_insample_16yr.npz"
        econ_file = "results/econometric_backfill/econometric_backfill_insample.npz"
    else:  # oos
        oracle_file = "models/backfill/oos_reconstruction_16yr.npz"
        vae_file = "models/backfill/vae_prior_oos_16yr.npz"
        econ_file = "results/econometric_backfill/econometric_backfill_oos.npz"

    # Load data
    oracle_data = np.load(oracle_file)
    vae_data = np.load(vae_file)
    econ_data = np.load(econ_file)
    gt_data = np.load("data/vol_surface_with_ret.npz")

    # Extract reconstructions
    recon_key = f'recon_h{horizon}'
    indices_key = f'indices_h{horizon}'

    oracle_p50 = oracle_data[recon_key][:, 1, :, :]  # (N, 5, 5)
    vae_p50 = vae_data[recon_key][:, 1, :, :]
    econ_p50 = econ_data[recon_key][:, 1, :, :]

    indices = oracle_data[indices_key]
    gt = gt_data["surface"][indices]

    # Align lengths
    min_len = min(gt.shape[0], oracle_p50.shape[0], vae_p50.shape[0], econ_p50.shape[0])
    gt = gt[:min_len]
    oracle_p50 = oracle_p50[:min_len]
    vae_p50 = vae_p50[:min_len]
    econ_p50 = econ_p50[:min_len]

    # Compute variance ratios
    oracle_ratios = np.zeros((5, 5))
    vae_ratios = np.zeros((5, 5))
    econ_ratios = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_std = np.std(gt[:, i, j])
            oracle_std = np.std(oracle_p50[:, i, j])
            vae_std = np.std(vae_p50[:, i, j])
            econ_std = np.std(econ_p50[:, i, j])

            oracle_ratios[i, j] = oracle_std / gt_std if gt_std > 0 else np.nan
            vae_ratios[i, j] = vae_std / gt_std if gt_std > 0 else np.nan
            econ_ratios[i, j] = econ_std / gt_std if gt_std > 0 else np.nan

    return oracle_ratios, vae_ratios, econ_ratios


def plot_variance_ratios(period, horizon):
    """Create heatmap visualization of variance ratios."""

    print(f"\n{'='*80}")
    print(f"Period: {period.upper()}, Horizon: H{horizon}")
    print(f"{'='*80}")

    # Compute ratios
    print("Computing variance ratios...")
    oracle_ratios, vae_ratios, econ_ratios = compute_variance_ratios(period, horizon)

    # Create figure with 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'Variance Ratio Heatmaps: {period.upper()} H{horizon}\n'
                 f'model_std / GT_std (target = 1.0)',
                 fontsize=16, fontweight='bold')

    # Common colormap and range
    vmin, vmax = 0.0, 1.2
    cmap = 'RdYlGn'  # Red = underestimate, Yellow = ok, Green = good

    # Plot Oracle
    ax = axes[0]
    im = ax.imshow(oracle_ratios, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'Oracle\nMean: {np.nanmean(oracle_ratios):.3f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    # Annotate values
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{oracle_ratios[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    # Plot VAE Prior
    ax = axes[1]
    im = ax.imshow(vae_ratios, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'VAE Prior\nMean: {np.nanmean(vae_ratios):.3f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{vae_ratios[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    # Plot Econometric
    ax = axes[2]
    im = ax.imshow(econ_ratios, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(f'Econometric\nMean: {np.nanmean(econ_ratios):.3f}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(maturity_labels)
    ax.set_yticklabels(moneyness_labels)
    ax.set_xlabel('Maturity', fontsize=12)
    ax.set_ylabel('Moneyness', fontsize=12)

    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{econ_ratios[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, aspect=40)
    cbar.set_label('Variance Ratio (model_std / GT_std)', fontsize=12)
    cbar.ax.axvline(1.0, color='black', linewidth=2, linestyle='--', label='Target = 1.0')

    # Add interpretation guide
    guide_text = (
        "Interpretation:\n"
        "  Green (0.9-1.1): Model tracks GT variance well\n"
        "  Yellow (0.7-0.9 or 1.1-1.2): Moderate under/over-estimation\n"
        "  Red (<0.7 or >1.2): Significant mismatch"
    )
    fig.text(0.5, 0.01, guide_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    # Save
    output_file = output_dir / f'{period}_h{horizon}_variance_ratio_heatmap.png'
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_file}")

    # Print summary
    print(f"\nSummary statistics:")
    print(f"  Oracle:      Mean={np.nanmean(oracle_ratios):.3f}, Median={np.nanmedian(oracle_ratios):.3f}")
    print(f"  VAE Prior:   Mean={np.nanmean(vae_ratios):.3f}, Median={np.nanmedian(vae_ratios):.3f}")
    print(f"  Econometric: Mean={np.nanmean(econ_ratios):.3f}, Median={np.nanmedian(econ_ratios):.3f}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("VARIANCE RATIO HEATMAP VISUALIZATION")
    print("=" * 80)

    periods = ['insample', 'oos']
    horizons = [1, 7, 14, 30]

    for period in periods:
        for horizon in horizons:
            try:
                plot_variance_ratios(period, horizon)
            except Exception as e:
                print(f"\nERROR: Failed for {period} H{horizon}")
                print(f"  {str(e)}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated heatmaps in: {output_dir}")
