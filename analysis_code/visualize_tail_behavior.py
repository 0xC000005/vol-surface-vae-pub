"""
Visualize tail behavior using Q-Q plots and tail probability analysis.

Q-Q (Quantile-Quantile) plots compare empirical quantiles between distributions.
- Diagonal line = perfect match
- Deviation in tails = shape mismatch

Tail probability plots show P(IV > threshold) for extreme values.

Outputs to: results/distribution_analysis/tail_behavior/
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

output_dir = Path("results/distribution_analysis/tail_behavior")
output_dir.mkdir(parents=True, exist_ok=True)

moneyness_labels = ['0.70', '0.85', '1.00', '1.15', '1.30']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

# Test specific grid points (representative sample)
TEST_GRID_POINTS = [
    (2, 2),  # ATM, 6M (most liquid)
    (0, 0),  # Deep OTM put, 1M (tail risk)
    (4, 4),  # Deep OTM call, 2Y (long-dated)
]

# ============================================================================
# Helper Functions
# ============================================================================

def load_data(period, horizon):
    """Load data for given period and horizon."""
    if period == 'insample':
        oracle_file = "results/backfill_16yr/predictions/insample_reconstruction_16yr.npz"
        vae_file = "results/backfill_16yr/predictions/vae_prior_insample_16yr.npz"
        econ_file = "results/econometric_baseline/predictions/econometric_backfill_insample.npz"
    else:  # oos
        oracle_file = "results/backfill_16yr/predictions/oos_reconstruction_16yr.npz"
        vae_file = "results/backfill_16yr/predictions/vae_prior_oos_16yr.npz"
        econ_file = "results/econometric_baseline/predictions/econometric_backfill_oos.npz"

    oracle_data = np.load(oracle_file)
    vae_data = np.load(vae_file)
    econ_data = np.load(econ_file)
    gt_data = np.load("data/vol_surface_with_ret.npz")

    recon_key = f'recon_h{horizon}'
    indices_key = f'indices_h{horizon}'

    oracle_p50 = oracle_data[recon_key][:, 1, :, :]
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

    return gt, oracle_p50, vae_p50, econ_p50


def create_qq_plot_4panel(period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50):
    """Create 4-panel Q-Q plot comparing all models to GT."""

    gt_vals = gt[:, i, j]
    oracle_vals = oracle_p50[:, i, j]
    vae_vals = vae_p50[:, i, j]
    econ_vals = econ_p50[:, i, j]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'Q-Q Plots vs Ground Truth: {period.upper()} H{horizon}\n'
                 f'Grid ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity',
                 fontsize=16, fontweight='bold')

    # Compute quantiles for all distributions
    quantiles = np.linspace(0, 1, 100)
    gt_quantiles = np.percentile(gt_vals, quantiles * 100)

    # Panel 1: Oracle vs GT
    ax = axes[0, 0]
    oracle_quantiles = np.percentile(oracle_vals, quantiles * 100)

    # Plot Q-Q
    ax.scatter(gt_quantiles, oracle_quantiles, alpha=0.6, s=20, color='blue', label='Oracle')

    # Add diagonal (perfect match)
    min_val = min(gt_quantiles.min(), oracle_quantiles.min())
    max_val = max(gt_quantiles.max(), oracle_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')

    ax.set_xlabel('Ground Truth Quantiles', fontsize=12)
    ax.set_ylabel('Oracle Quantiles', fontsize=12)
    ax.set_title('Oracle vs Ground Truth', fontsize=14, fontweight='bold', color='blue')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate tail deviations
    lower_dev = oracle_quantiles[5] - gt_quantiles[5]  # p5 deviation
    upper_dev = oracle_quantiles[95] - gt_quantiles[95]  # p95 deviation
    ax.text(0.05, 0.95, f'Lower tail dev: {lower_dev:+.4f}\nUpper tail dev: {upper_dev:+.4f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: VAE vs GT
    ax = axes[0, 1]
    vae_quantiles = np.percentile(vae_vals, quantiles * 100)

    ax.scatter(gt_quantiles, vae_quantiles, alpha=0.6, s=20, color='red', label='VAE Prior')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')

    ax.set_xlabel('Ground Truth Quantiles', fontsize=12)
    ax.set_ylabel('VAE Prior Quantiles', fontsize=12)
    ax.set_title('VAE Prior vs Ground Truth', fontsize=14, fontweight='bold', color='red')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    lower_dev = vae_quantiles[5] - gt_quantiles[5]
    upper_dev = vae_quantiles[95] - gt_quantiles[95]
    ax.text(0.05, 0.95, f'Lower tail dev: {lower_dev:+.4f}\nUpper tail dev: {upper_dev:+.4f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 3: Econometric vs GT
    ax = axes[1, 0]
    econ_quantiles = np.percentile(econ_vals, quantiles * 100)

    ax.scatter(gt_quantiles, econ_quantiles, alpha=0.6, s=20, color='green', label='Econometric')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')

    ax.set_xlabel('Ground Truth Quantiles', fontsize=12)
    ax.set_ylabel('Econometric Quantiles', fontsize=12)
    ax.set_title('Econometric vs Ground Truth', fontsize=14, fontweight='bold', color='green')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    lower_dev = econ_quantiles[5] - gt_quantiles[5]
    upper_dev = econ_quantiles[95] - gt_quantiles[95]
    ax.text(0.05, 0.95, f'Lower tail dev: {lower_dev:+.4f}\nUpper tail dev: {upper_dev:+.4f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 4: All models overlay
    ax = axes[1, 1]
    ax.scatter(gt_quantiles, oracle_quantiles, alpha=0.5, s=15, color='blue', label='Oracle')
    ax.scatter(gt_quantiles, vae_quantiles, alpha=0.5, s=15, color='red', label='VAE Prior')
    ax.scatter(gt_quantiles, econ_quantiles, alpha=0.5, s=15, color='green', label='Econometric')
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect match')

    ax.set_xlabel('Ground Truth Quantiles', fontsize=12)
    ax.set_ylabel('Model Quantiles', fontsize=12)
    ax.set_title('All Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f'{period}_h{horizon}_qq_plot_grid_{i}_{j}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_tail_probability_plot(period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50):
    """Create tail probability plot: P(IV > threshold)."""

    gt_vals = gt[:, i, j]
    oracle_vals = oracle_p50[:, i, j]
    vae_vals = vae_p50[:, i, j]
    econ_vals = econ_p50[:, i, j]

    # Define threshold range (from min to max across all data)
    all_data = np.concatenate([gt_vals, oracle_vals, vae_vals, econ_vals])
    thresholds = np.linspace(all_data.min(), all_data.max(), 200)

    # Compute tail probabilities: P(X > threshold)
    gt_tail_prob = np.array([np.mean(gt_vals > t) for t in thresholds])
    oracle_tail_prob = np.array([np.mean(oracle_vals > t) for t in thresholds])
    vae_tail_prob = np.array([np.mean(vae_vals > t) for t in thresholds])
    econ_tail_prob = np.array([np.mean(econ_vals > t) for t in thresholds])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Tail Probability Analysis: {period.upper()} H{horizon}\n'
                 f'Grid ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity',
                 fontsize=16, fontweight='bold')

    # Panel 1: Linear scale
    ax = axes[0]
    ax.plot(thresholds, gt_tail_prob, 'k-', linewidth=2, label='Ground Truth')
    ax.plot(thresholds, oracle_tail_prob, 'b--', linewidth=2, label='Oracle')
    ax.plot(thresholds, vae_tail_prob, 'r--', linewidth=2, label='VAE Prior')
    ax.plot(thresholds, econ_tail_prob, 'g--', linewidth=2, label='Econometric')

    ax.set_xlabel('Threshold (IV)', fontsize=12)
    ax.set_ylabel('P(IV > threshold)', fontsize=12)
    ax.set_title('Tail Probability (Linear Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Log-log scale (emphasize tails)
    ax = axes[1]
    # Filter out zeros for log scale
    mask = (gt_tail_prob > 0) & (oracle_tail_prob > 0) & (vae_tail_prob > 0) & (econ_tail_prob > 0)

    ax.loglog(thresholds[mask], gt_tail_prob[mask], 'k-', linewidth=2, label='Ground Truth')
    ax.loglog(thresholds[mask], oracle_tail_prob[mask], 'b--', linewidth=2, label='Oracle')
    ax.loglog(thresholds[mask], vae_tail_prob[mask], 'r--', linewidth=2, label='VAE Prior')
    ax.loglog(thresholds[mask], econ_tail_prob[mask], 'g--', linewidth=2, label='Econometric')

    ax.set_xlabel('Threshold (IV)', fontsize=12)
    ax.set_ylabel('P(IV > threshold)', fontsize=12)
    ax.set_title('Tail Probability (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Add interpretation guide
    guide_text = (
        "Interpretation:\n"
        "  Models above GT → overestimate tail probability (conservative)\n"
        "  Models below GT → underestimate tail probability (risky)\n"
        "  Log-log scale: Straight line = power law (fat tail)"
    )
    fig.text(0.5, 0.01, guide_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    output_file = output_dir / f'{period}_h{horizon}_tail_prob_grid_{i}_{j}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_combined_analysis(period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50):
    """Create comprehensive 6-panel analysis combining Q-Q, histograms, and tail probs."""

    gt_vals = gt[:, i, j]
    oracle_vals = oracle_p50[:, i, j]
    vae_vals = vae_p50[:, i, j]
    econ_vals = econ_p50[:, i, j]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Comprehensive Tail Behavior Analysis: {period.upper()} H{horizon}\n'
                 f'Grid ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity',
                 fontsize=18, fontweight='bold')

    # Row 1: Q-Q plots
    quantiles = np.linspace(0, 1, 100)
    gt_quantiles = np.percentile(gt_vals, quantiles * 100)

    for idx, (vals, name, color) in enumerate([
        (oracle_vals, 'Oracle', 'blue'),
        (vae_vals, 'VAE Prior', 'red'),
        (econ_vals, 'Econometric', 'green')
    ]):
        ax = fig.add_subplot(gs[0, idx])
        model_quantiles = np.percentile(vals, quantiles * 100)

        ax.scatter(gt_quantiles, model_quantiles, alpha=0.6, s=15, color=color)
        min_val = min(gt_quantiles.min(), model_quantiles.min())
        max_val = max(gt_quantiles.max(), model_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel('GT Quantiles', fontsize=10)
        ax.set_ylabel(f'{name} Quantiles', fontsize=10)
        ax.set_title(f'Q-Q: {name}', fontsize=12, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)

    # Row 2: Histograms
    bins = 40
    for idx, (vals, name, color) in enumerate([
        (oracle_vals, 'Oracle', 'blue'),
        (vae_vals, 'VAE Prior', 'red'),
        (econ_vals, 'Econometric', 'green')
    ]):
        ax = fig.add_subplot(gs[1, idx])
        ax.hist(gt_vals, bins=bins, alpha=0.5, color='black', density=True, label='GT', edgecolor='black')
        ax.hist(vals, bins=bins, alpha=0.6, color=color, density=True, label=name, edgecolor=color)

        # Add statistics
        gt_kurt = stats.kurtosis(gt_vals, fisher=False)
        model_kurt = stats.kurtosis(vals, fisher=False)
        stats_text = f'GT κ: {gt_kurt:.2f}\n{name} κ: {model_kurt:.2f}\nΔκ: {model_kurt-gt_kurt:+.2f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('IV', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Distribution: {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Row 3: Tail probabilities (2 panels + summary)

    # Panel 1: Linear tail prob
    ax = fig.add_subplot(gs[2, 0:2])
    all_data = np.concatenate([gt_vals, oracle_vals, vae_vals, econ_vals])
    thresholds = np.linspace(all_data.min(), all_data.max(), 200)

    gt_tail_prob = np.array([np.mean(gt_vals > t) for t in thresholds])
    oracle_tail_prob = np.array([np.mean(oracle_vals > t) for t in thresholds])
    vae_tail_prob = np.array([np.mean(vae_vals > t) for t in thresholds])
    econ_tail_prob = np.array([np.mean(econ_vals > t) for t in thresholds])

    ax.plot(thresholds, gt_tail_prob, 'k-', linewidth=2.5, label='Ground Truth')
    ax.plot(thresholds, oracle_tail_prob, 'b--', linewidth=2, label='Oracle')
    ax.plot(thresholds, vae_tail_prob, 'r--', linewidth=2, label='VAE Prior')
    ax.plot(thresholds, econ_tail_prob, 'g--', linewidth=2, label='Econometric')

    ax.set_xlabel('Threshold (IV)', fontsize=11)
    ax.set_ylabel('P(IV > threshold)', fontsize=11)
    ax.set_title('Tail Probability', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Summary statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary_text = "TAIL BEHAVIOR SUMMARY\n" + "="*30 + "\n\n"
    summary_text += f"Ground Truth:\n"
    summary_text += f"  Kurtosis: {stats.kurtosis(gt_vals, fisher=False):.2f}\n"
    summary_text += f"  Skewness: {stats.skew(gt_vals):+.2f}\n"
    summary_text += f"  p01-p99: {np.percentile(gt_vals, 99) - np.percentile(gt_vals, 1):.4f}\n\n"

    for vals, name in [(oracle_vals, 'Oracle'), (vae_vals, 'VAE'), (econ_vals, 'Econ')]:
        kurt_diff = stats.kurtosis(vals, fisher=False) - stats.kurtosis(gt_vals, fisher=False)
        summary_text += f"{name}:\n"
        summary_text += f"  Δκ: {kurt_diff:+.2f}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    output_file = output_dir / f'{period}_h{horizon}_combined_analysis_grid_{i}_{j}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TAIL BEHAVIOR VISUALIZATION")
    print("="*80)
    print()

    periods = ['insample', 'oos']
    horizons = [1, 30]  # Just H1 and H30 for key comparisons

    print(f"Test grid points: {len(TEST_GRID_POINTS)}")
    for i, j in TEST_GRID_POINTS:
        print(f"  ({i},{j}): {moneyness_labels[i]} moneyness, {maturity_labels[j]} maturity")
    print()

    for period in periods:
        for horizon in horizons:
            print(f"\n{'='*80}")
            print(f"Processing: {period.upper()} H{horizon}")
            print(f"{'='*80}\n")

            # Load data
            print("Loading data...")
            gt, oracle_p50, vae_p50, econ_p50 = load_data(period, horizon)
            print(f"  Loaded {gt.shape[0]} days\n")

            # Generate plots for each test grid point
            for i, j in TEST_GRID_POINTS:
                print(f"Analyzing grid point ({i},{j})...")

                try:
                    create_qq_plot_4panel(period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50)
                    create_tail_probability_plot(period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50)
                    create_combined_analysis(period, horizon, i, j, gt, oracle_p50, vae_p50, econ_p50)
                    print(f"  ✓ Completed grid ({i},{j})\n")
                except Exception as e:
                    print(f"  ERROR: Failed for grid ({i},{j})")
                    print(f"    {str(e)}\n")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated tail behavior plots in: {output_dir}")
    print()
    print("Files generated:")
    for period in periods:
        for horizon in horizons:
            for i, j in TEST_GRID_POINTS:
                print(f"  ✓ {period}_h{horizon}_qq_plot_grid_{i}_{j}.png")
                print(f"  ✓ {period}_h{horizon}_tail_prob_grid_{i}_{j}.png")
                print(f"  ✓ {period}_h{horizon}_combined_analysis_grid_{i}_{j}.png")
