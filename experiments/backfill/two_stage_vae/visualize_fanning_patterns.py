"""
Visualize Fanning Patterns for GT and VAE Paths.

Creates visualizations showing:
1. Ground truth horizons forming a fanning pattern
2. VAE-generated paths under posterior mode (z conditioned on target)
3. Both unconditional and conditional (per-cluster) cases

Usage:
    python experiments/backfill/prior_encoder_ablation/visualize_fanning_patterns.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, ".")


def load_data():
    """Load pre-computed analysis data."""
    uncond_path = Path("models/backfill/two_stage/unconditional_analysis.npz")
    cond_path = Path("models/backfill/two_stage/conditional_analysis.npz")

    if not uncond_path.exists():
        raise FileNotFoundError(f"Run analyze_unconditional_marginal.py first: {uncond_path}")

    uncond_data = np.load(uncond_path)

    # Load cluster labels if available, otherwise recompute
    if cond_path.exists():
        cond_data = np.load(cond_path, allow_pickle=True)
        cluster_labels = cond_data["cluster_labels"]
    else:
        # Recompute clusters
        ctx_embeddings = uncond_data["ctx_embeddings"]
        scaler = StandardScaler()
        ctx_scaled = scaler.fit_transform(ctx_embeddings)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(ctx_scaled)

    return uncond_data, cluster_labels


def compute_cumulative_paths(log_returns):
    """
    Convert log-returns to cumulative paths starting from 0.

    Args:
        log_returns: (..., horizon, 5, 5) array

    Returns:
        cumulative: (..., horizon+1, 5, 5) array with 0 prepended
    """
    # Prepend zeros for starting point
    shape = list(log_returns.shape)
    shape[-3] = 1  # One timestep for the origin
    zeros = np.zeros(shape)

    # Cumulative sum
    cumsum = np.cumsum(log_returns, axis=-3)

    # Concatenate origin + cumsum
    return np.concatenate([zeros, cumsum], axis=-3)


def plot_unconditional_fanning(samples_log, targets_log, output_dir, max_paths=100):
    """
    Plot unconditional fanning pattern comparing GT vs VAE.

    Shows paths for a single grid point (ATM, mid-tenor).
    For fair comparison: show individual VAE samples, not means.
    """
    print("Generating unconditional fanning plot...")

    # Use center grid point (ATM, mid-tenor)
    grid_h, grid_w = 2, 2

    # Extract paths for this grid point
    # targets_log: (n_seq, horizon, 5, 5)
    gt_paths = targets_log[:, :, grid_h, grid_w]  # (n_seq, horizon)

    # samples_log: (n_seq, n_samples, horizon, 5, 5)
    # Take ONE sample per sequence for fair comparison (not mean which collapses variance)
    vae_single_sample_paths = samples_log[:, 0, :, grid_h, grid_w]  # (n_seq, horizon)

    # Compute cumulative paths
    n_seq = min(max_paths, len(gt_paths))

    # Prepend zero for origin
    gt_cumsum = np.concatenate([np.zeros((n_seq, 1)), np.cumsum(gt_paths[:n_seq], axis=1)], axis=1)
    vae_cumsum = np.concatenate([np.zeros((n_seq, 1)), np.cumsum(vae_single_sample_paths[:n_seq], axis=1)], axis=1)

    horizons = np.arange(gt_cumsum.shape[1])  # 0 to 30

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: GT paths
    ax1 = axes[0]
    for i in range(n_seq):
        ax1.plot(horizons, gt_cumsum[i], color='gray', alpha=0.3, linewidth=0.5)

    # Add mean and percentiles
    gt_mean = gt_cumsum.mean(axis=0)
    gt_p05 = np.percentile(gt_cumsum, 5, axis=0)
    gt_p95 = np.percentile(gt_cumsum, 95, axis=0)

    ax1.fill_between(horizons, gt_p05, gt_p95, color='gray', alpha=0.2, label='90% CI')
    ax1.plot(horizons, gt_mean, color='black', linewidth=2, label='Mean')

    ax1.set_xlabel('Horizon (days)', fontsize=12)
    ax1.set_ylabel('Cumulative Log-Return', fontsize=12)
    ax1.set_title(f'Ground Truth Paths (n={n_seq})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Right: VAE paths
    ax2 = axes[1]
    for i in range(n_seq):
        ax2.plot(horizons, vae_cumsum[i], color='blue', alpha=0.3, linewidth=0.5)

    vae_mean = vae_cumsum.mean(axis=0)
    vae_p05 = np.percentile(vae_cumsum, 5, axis=0)
    vae_p95 = np.percentile(vae_cumsum, 95, axis=0)

    ax2.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2, label='90% CI')
    ax2.plot(horizons, vae_mean, color='darkblue', linewidth=2, label='Mean')

    ax2.set_xlabel('Horizon (days)', fontsize=12)
    ax2.set_ylabel('Cumulative Log-Return', fontsize=12)
    ax2.set_title(f'VAE Posterior Paths (n={n_seq})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Match y-axis limits
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    plt.suptitle('Unconditional Fanning Pattern: GT vs VAE (Posterior Mode)\n'
                 'Grid Point: ATM, Mid-Tenor', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "unconditional_fanning.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_conditional_fanning(samples_log, targets_log, cluster_labels, output_dir, max_paths=50):
    """
    Plot conditional fanning patterns per cluster.
    For fair comparison: show individual VAE samples, not means.
    """
    print("Generating conditional fanning plots...")

    # Use center grid point
    grid_h, grid_w = 2, 2

    gt_paths = targets_log[:, :, grid_h, grid_w]
    # Take ONE sample per sequence for fair comparison
    vae_single_sample_paths = samples_log[:, 0, :, grid_h, grid_w]

    n_clusters = len(np.unique(cluster_labels))

    fig, axes = plt.subplots(n_clusters, 2, figsize=(14, 4*n_clusters))

    for c in range(n_clusters):
        mask = cluster_labels == c
        n_in_cluster = mask.sum()
        n_show = min(max_paths, n_in_cluster)

        gt_cluster = gt_paths[mask][:n_show]
        vae_cluster = vae_single_sample_paths[mask][:n_show]

        # Cumulative paths
        gt_cumsum = np.concatenate([np.zeros((n_show, 1)), np.cumsum(gt_cluster, axis=1)], axis=1)
        vae_cumsum = np.concatenate([np.zeros((n_show, 1)), np.cumsum(vae_cluster, axis=1)], axis=1)

        horizons = np.arange(gt_cumsum.shape[1])

        # Left: GT
        ax1 = axes[c, 0]
        for i in range(n_show):
            ax1.plot(horizons, gt_cumsum[i], color='gray', alpha=0.4, linewidth=0.5)

        gt_mean = gt_cumsum.mean(axis=0)
        gt_p05 = np.percentile(gt_cumsum, 5, axis=0)
        gt_p95 = np.percentile(gt_cumsum, 95, axis=0)

        ax1.fill_between(horizons, gt_p05, gt_p95, color='gray', alpha=0.2)
        ax1.plot(horizons, gt_mean, color='black', linewidth=2)

        ax1.set_ylabel('Cumulative Log-Return', fontsize=10)
        ax1.set_title(f'Cluster {c} - GT (n={n_in_cluster})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Right: VAE
        ax2 = axes[c, 1]
        for i in range(n_show):
            ax2.plot(horizons, vae_cumsum[i], color='blue', alpha=0.4, linewidth=0.5)

        vae_mean = vae_cumsum.mean(axis=0)
        vae_p05 = np.percentile(vae_cumsum, 5, axis=0)
        vae_p95 = np.percentile(vae_cumsum, 95, axis=0)

        ax2.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2)
        ax2.plot(horizons, vae_mean, color='darkblue', linewidth=2)

        ax2.set_title(f'Cluster {c} - VAE Posterior (n={n_in_cluster})', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Match y-axis limits within row
        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        if c == n_clusters - 1:
            ax1.set_xlabel('Horizon (days)', fontsize=10)
            ax2.set_xlabel('Horizon (days)', fontsize=10)

    plt.suptitle('Conditional Fanning Pattern by Context Cluster\n'
                 'GT vs VAE (Posterior Mode) - Grid Point: ATM, Mid-Tenor',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "conditional_fanning_clusters.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_single_sequence_ci(samples_log, targets_log, cluster_labels, output_dir):
    """
    Plot single sequence examples with CI bands.
    """
    print("Generating single sequence CI examples...")

    grid_h, grid_w = 2, 2
    n_clusters = len(np.unique(cluster_labels))

    # Pick one representative sequence from each cluster
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5))

    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]

        # Pick a sequence near the cluster center (median index)
        seq_idx = cluster_indices[len(cluster_indices) // 2]

        # GT path
        gt_path = targets_log[seq_idx, :, grid_h, grid_w]  # (horizon,)
        gt_cumsum = np.concatenate([[0], np.cumsum(gt_path)])

        # VAE samples for this sequence
        vae_samples = samples_log[seq_idx, :, :, grid_h, grid_w]  # (n_samples, horizon)
        vae_cumsum = np.concatenate([np.zeros((vae_samples.shape[0], 1)),
                                      np.cumsum(vae_samples, axis=1)], axis=1)

        horizons = np.arange(len(gt_cumsum))

        ax = axes[c] if n_clusters > 1 else axes

        # Plot VAE sample paths (subset)
        n_show = min(50, vae_cumsum.shape[0])
        for i in range(n_show):
            ax.plot(horizons, vae_cumsum[i], color='lightblue', alpha=0.3, linewidth=0.5)

        # VAE CI band
        vae_p05 = np.percentile(vae_cumsum, 5, axis=0)
        vae_p50 = np.percentile(vae_cumsum, 50, axis=0)
        vae_p95 = np.percentile(vae_cumsum, 95, axis=0)

        ax.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2, label='VAE 90% CI')
        ax.plot(horizons, vae_p50, color='blue', linewidth=1.5, linestyle='--', label='VAE Median')

        # GT path (thick line)
        ax.plot(horizons, gt_cumsum, color='red', linewidth=2.5, label='Ground Truth')

        ax.set_xlabel('Horizon (days)', fontsize=11)
        ax.set_ylabel('Cumulative Log-Return', fontsize=11)
        ax.set_title(f'Cluster {c} - Sequence {seq_idx}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.suptitle('Single Sequence Examples: GT Path with VAE 90% CI\n'
                 '(Posterior Mode - z conditioned on target)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "single_sequence_ci_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_fanning_summary(samples_log, targets_log, cluster_labels, output_dir):
    """
    Create a combined summary plot.
    For fair comparison: show individual VAE samples, not means.
    """
    print("Generating summary plot...")

    grid_h, grid_w = 2, 2

    gt_paths = targets_log[:, :, grid_h, grid_w]
    # Take ONE sample per sequence for fair comparison
    vae_single_sample_paths = samples_log[:, 0, :, grid_h, grid_w]

    # Compute statistics
    gt_cumsum = np.concatenate([np.zeros((len(gt_paths), 1)),
                                 np.cumsum(gt_paths, axis=1)], axis=1)
    vae_cumsum = np.concatenate([np.zeros((len(vae_single_sample_paths), 1)),
                                  np.cumsum(vae_single_sample_paths, axis=1)], axis=1)

    horizons = np.arange(gt_cumsum.shape[1])

    fig = plt.figure(figsize=(16, 10))

    # Main comparison plot
    ax1 = fig.add_subplot(2, 2, 1)

    # GT band
    gt_p05 = np.percentile(gt_cumsum, 5, axis=0)
    gt_p50 = np.percentile(gt_cumsum, 50, axis=0)
    gt_p95 = np.percentile(gt_cumsum, 95, axis=0)

    ax1.fill_between(horizons, gt_p05, gt_p95, color='gray', alpha=0.3, label='GT 90% CI')
    ax1.plot(horizons, gt_p50, color='black', linewidth=2, label='GT Median')

    # VAE band
    vae_p05 = np.percentile(vae_cumsum, 5, axis=0)
    vae_p50 = np.percentile(vae_cumsum, 50, axis=0)
    vae_p95 = np.percentile(vae_cumsum, 95, axis=0)

    ax1.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.3, label='VAE 90% CI')
    ax1.plot(horizons, vae_p50, color='blue', linewidth=2, linestyle='--', label='VAE Median')

    ax1.set_xlabel('Horizon (days)', fontsize=11)
    ax1.set_ylabel('Cumulative Log-Return', fontsize=11)
    ax1.set_title('GT vs VAE 90% Confidence Intervals', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # CI Width comparison
    ax2 = fig.add_subplot(2, 2, 2)

    gt_ci_width = gt_p95 - gt_p05
    vae_ci_width = vae_p95 - vae_p05

    ax2.plot(horizons, gt_ci_width, color='black', linewidth=2, label='GT CI Width')
    ax2.plot(horizons, vae_ci_width, color='blue', linewidth=2, linestyle='--', label='VAE CI Width')

    # Theoretical sqrt growth
    theoretical = gt_ci_width[1] * np.sqrt(horizons[1:])
    ax2.plot(horizons[1:], theoretical, color='green', linewidth=1,
             linestyle=':', label='Theoretical √H')

    ax2.set_xlabel('Horizon (days)', fontsize=11)
    ax2.set_ylabel('CI Width', fontsize=11)
    ax2.set_title('CI Width Growth with Horizon', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Std comparison
    ax3 = fig.add_subplot(2, 2, 3)

    gt_std = gt_cumsum.std(axis=0)
    vae_std = vae_cumsum.std(axis=0)

    ax3.plot(horizons, gt_std, color='black', linewidth=2, label='GT Std')
    ax3.plot(horizons, vae_std, color='blue', linewidth=2, linestyle='--', label='VAE Std')

    ax3.set_xlabel('Horizon (days)', fontsize=11)
    ax3.set_ylabel('Standard Deviation', fontsize=11)
    ax3.set_title('Path Standard Deviation by Horizon', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Per-cluster violation rates
    ax4 = fig.add_subplot(2, 2, 4)

    n_clusters = len(np.unique(cluster_labels))
    bar_width = 0.25
    x = np.arange(n_clusters)

    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_labels == c

        # Compute violations
        gt_c = targets_log[mask, :, grid_h, grid_w]
        vae_c = samples_log[mask, :, :, grid_h, grid_w]

        ci_lower = np.percentile(vae_c, 5, axis=1)
        ci_upper = np.percentile(vae_c, 95, axis=1)

        violations = (gt_c < ci_lower) | (gt_c > ci_upper)

        short_viol = violations[:, :5].mean() * 100
        medium_viol = violations[:, 5:15].mean() * 100
        long_viol = violations[:, 15:].mean() * 100

        cluster_stats.append((short_viol, medium_viol, long_viol))

    cluster_stats = np.array(cluster_stats)

    ax4.bar(x - bar_width, cluster_stats[:, 0], bar_width, label='Short (H=1-5)', color='green')
    ax4.bar(x, cluster_stats[:, 1], bar_width, label='Medium (H=6-15)', color='orange')
    ax4.bar(x + bar_width, cluster_stats[:, 2], bar_width, label='Long (H=16-30)', color='red')

    ax4.axhline(y=10, color='black', linestyle='--', alpha=0.7, label='Target (10%)')

    ax4.set_xlabel('Cluster', fontsize=11)
    ax4.set_ylabel('CI Violations (%)', fontsize=11)
    ax4.set_title('CI Violations by Cluster and Horizon', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'C{i}' for i in range(n_clusters)])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Fanning Pattern Analysis Summary\n'
                 'VAE Posterior Mode (z conditioned on target) - ATM Grid Point',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "fanning_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("Fanning Pattern Visualization")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    uncond_data, cluster_labels = load_data()

    samples_log = uncond_data["samples_log"]
    targets_log = uncond_data["targets_log"]

    print(f"  Samples shape: {samples_log.shape}")
    print(f"  Targets shape: {targets_log.shape}")
    print(f"  Clusters: {len(np.unique(cluster_labels))}")

    # Create output directory
    output_dir = Path("models/backfill/two_stage/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate all plots
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    plot_unconditional_fanning(samples_log, targets_log, output_dir)
    plot_conditional_fanning(samples_log, targets_log, cluster_labels, output_dir)
    plot_single_sequence_ci(samples_log, targets_log, cluster_labels, output_dir)
    plot_fanning_summary(samples_log, targets_log, cluster_labels, output_dir)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
