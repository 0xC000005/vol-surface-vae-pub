"""
Visualize Fanning Patterns in IV Level Space (after transform-back).

Same visualizations as visualize_fanning_patterns.py but in IV levels
instead of log-return space.

Usage:
    python experiments/backfill/prior_encoder_ablation/visualize_fanning_iv_space.py
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

    if cond_path.exists():
        cond_data = np.load(cond_path, allow_pickle=True)
        cluster_labels = cond_data["cluster_labels"]
    else:
        ctx_embeddings = uncond_data["ctx_embeddings"]
        scaler = StandardScaler()
        ctx_scaled = scaler.fit_transform(ctx_embeddings)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(ctx_scaled)

    return uncond_data, cluster_labels


def plot_unconditional_fanning_iv(samples_iv, targets_iv, output_dir, max_paths=100):
    """
    Plot unconditional fanning pattern in IV space.
    For fair comparison: show individual VAE samples, not means.
    """
    print("Generating unconditional fanning plot (IV space)...")

    grid_h, grid_w = 2, 2

    # Extract paths - already in IV space
    # Need to prepend the initial IV level (we'll normalize to start at 1.0)
    gt_paths = targets_iv[:, :, grid_h, grid_w]  # (n_seq, horizon)
    # Take ONE sample per sequence for fair comparison (not mean which collapses variance)
    vae_single_sample_paths = samples_iv[:, 0, :, grid_h, grid_w]  # (n_seq, horizon)

    n_seq = min(max_paths, len(gt_paths))

    # Normalize: divide by first value to start at 1.0 (relative IV change)
    # This allows comparison across sequences with different starting IV levels
    gt_initial = gt_paths[:n_seq, 0:1]  # (n_seq, 1)
    vae_initial = vae_single_sample_paths[:n_seq, 0:1]

    gt_normalized = gt_paths[:n_seq] / gt_initial
    vae_normalized = vae_single_sample_paths[:n_seq] / vae_initial

    # Prepend 1.0 for the starting point
    gt_with_start = np.concatenate([np.ones((n_seq, 1)), gt_normalized], axis=1)
    vae_with_start = np.concatenate([np.ones((n_seq, 1)), vae_normalized], axis=1)

    horizons = np.arange(gt_with_start.shape[1])  # 0 to 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: GT paths
    ax1 = axes[0]
    for i in range(n_seq):
        ax1.plot(horizons, gt_with_start[i], color='gray', alpha=0.3, linewidth=0.5)

    gt_mean = gt_with_start.mean(axis=0)
    gt_p05 = np.percentile(gt_with_start, 5, axis=0)
    gt_p95 = np.percentile(gt_with_start, 95, axis=0)

    ax1.fill_between(horizons, gt_p05, gt_p95, color='gray', alpha=0.2, label='90% CI')
    ax1.plot(horizons, gt_mean, color='black', linewidth=2, label='Mean')

    ax1.set_xlabel('Horizon (days)', fontsize=12)
    ax1.set_ylabel('Relative IV (normalized to 1.0 at H=0)', fontsize=12)
    ax1.set_title(f'Ground Truth Paths (n={n_seq})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    # Right: VAE paths
    ax2 = axes[1]
    for i in range(n_seq):
        ax2.plot(horizons, vae_with_start[i], color='blue', alpha=0.3, linewidth=0.5)

    vae_mean = vae_with_start.mean(axis=0)
    vae_p05 = np.percentile(vae_with_start, 5, axis=0)
    vae_p95 = np.percentile(vae_with_start, 95, axis=0)

    ax2.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2, label='90% CI')
    ax2.plot(horizons, vae_mean, color='darkblue', linewidth=2, label='Mean')

    ax2.set_xlabel('Horizon (days)', fontsize=12)
    ax2.set_ylabel('Relative IV (normalized to 1.0 at H=0)', fontsize=12)
    ax2.set_title(f'VAE Posterior Paths (n={n_seq})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    # Match y-axis limits
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    plt.suptitle('Unconditional Fanning Pattern (IV Space): GT vs VAE (Posterior Mode)\n'
                 'Grid Point: ATM, Mid-Tenor - Normalized to start at 1.0', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "unconditional_fanning_iv_space.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_conditional_fanning_iv(samples_iv, targets_iv, cluster_labels, output_dir, max_paths=50):
    """
    Plot conditional fanning patterns per cluster in IV space.
    For fair comparison: show individual VAE samples, not means.
    """
    print("Generating conditional fanning plots (IV space)...")

    grid_h, grid_w = 2, 2

    gt_paths = targets_iv[:, :, grid_h, grid_w]
    # Take ONE sample per sequence for fair comparison
    vae_single_sample_paths = samples_iv[:, 0, :, grid_h, grid_w]

    n_clusters = len(np.unique(cluster_labels))

    fig, axes = plt.subplots(n_clusters, 2, figsize=(14, 4*n_clusters))

    for c in range(n_clusters):
        mask = cluster_labels == c
        n_in_cluster = mask.sum()
        n_show = min(max_paths, n_in_cluster)

        gt_cluster = gt_paths[mask][:n_show]
        vae_cluster = vae_single_sample_paths[mask][:n_show]

        # Normalize to start at 1.0
        gt_initial = gt_cluster[:, 0:1]
        vae_initial = vae_cluster[:, 0:1]

        gt_normalized = gt_cluster / gt_initial
        vae_normalized = vae_cluster / vae_initial

        gt_with_start = np.concatenate([np.ones((n_show, 1)), gt_normalized], axis=1)
        vae_with_start = np.concatenate([np.ones((n_show, 1)), vae_normalized], axis=1)

        horizons = np.arange(gt_with_start.shape[1])

        # Left: GT
        ax1 = axes[c, 0]
        for i in range(n_show):
            ax1.plot(horizons, gt_with_start[i], color='gray', alpha=0.4, linewidth=0.5)

        gt_mean = gt_with_start.mean(axis=0)
        gt_p05 = np.percentile(gt_with_start, 5, axis=0)
        gt_p95 = np.percentile(gt_with_start, 95, axis=0)

        ax1.fill_between(horizons, gt_p05, gt_p95, color='gray', alpha=0.2)
        ax1.plot(horizons, gt_mean, color='black', linewidth=2)

        ax1.set_ylabel('Relative IV', fontsize=10)
        ax1.set_title(f'Cluster {c} - GT (n={n_in_cluster})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

        # Right: VAE
        ax2 = axes[c, 1]
        for i in range(n_show):
            ax2.plot(horizons, vae_with_start[i], color='blue', alpha=0.4, linewidth=0.5)

        vae_mean = vae_with_start.mean(axis=0)
        vae_p05 = np.percentile(vae_with_start, 5, axis=0)
        vae_p95 = np.percentile(vae_with_start, 95, axis=0)

        ax2.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2)
        ax2.plot(horizons, vae_mean, color='darkblue', linewidth=2)

        ax2.set_title(f'Cluster {c} - VAE Posterior (n={n_in_cluster})', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

        # Match y-axis limits within row
        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        if c == n_clusters - 1:
            ax1.set_xlabel('Horizon (days)', fontsize=10)
            ax2.set_xlabel('Horizon (days)', fontsize=10)

    plt.suptitle('Conditional Fanning Pattern (IV Space) by Context Cluster\n'
                 'GT vs VAE (Posterior Mode) - Normalized to start at 1.0',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "conditional_fanning_iv_space.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_single_sequence_iv(samples_iv, targets_iv, cluster_labels, output_dir):
    """
    Plot single sequence examples with CI bands in IV space.
    """
    print("Generating single sequence CI examples (IV space)...")

    grid_h, grid_w = 2, 2
    n_clusters = len(np.unique(cluster_labels))

    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5))

    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]
        seq_idx = cluster_indices[len(cluster_indices) // 2]

        # GT path in IV space
        gt_path = targets_iv[seq_idx, :, grid_h, grid_w]  # (horizon,)
        gt_initial = gt_path[0]
        gt_normalized = gt_path / gt_initial
        gt_with_start = np.concatenate([[1.0], gt_normalized])

        # VAE samples for this sequence
        vae_samples = samples_iv[seq_idx, :, :, grid_h, grid_w]  # (n_samples, horizon)
        vae_initial = vae_samples[:, 0:1]
        vae_normalized = vae_samples / vae_initial
        vae_with_start = np.concatenate([np.ones((vae_normalized.shape[0], 1)),
                                          vae_normalized], axis=1)

        horizons = np.arange(len(gt_with_start))

        ax = axes[c] if n_clusters > 1 else axes

        # Plot VAE sample paths (subset)
        n_show = min(50, vae_with_start.shape[0])
        for i in range(n_show):
            ax.plot(horizons, vae_with_start[i], color='lightblue', alpha=0.3, linewidth=0.5)

        # VAE CI band
        vae_p05 = np.percentile(vae_with_start, 5, axis=0)
        vae_p50 = np.percentile(vae_with_start, 50, axis=0)
        vae_p95 = np.percentile(vae_with_start, 95, axis=0)

        ax.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2, label='VAE 90% CI')
        ax.plot(horizons, vae_p50, color='blue', linewidth=1.5, linestyle='--', label='VAE Median')

        # GT path (thick line)
        ax.plot(horizons, gt_with_start, color='red', linewidth=2.5, label='Ground Truth')

        ax.set_xlabel('Horizon (days)', fontsize=11)
        ax.set_ylabel('Relative IV', fontsize=11)
        ax.set_title(f'Cluster {c} - Sequence {seq_idx}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    plt.suptitle('Single Sequence Examples (IV Space): GT Path with VAE 90% CI\n'
                 '(Posterior Mode - Normalized to start at 1.0)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "single_sequence_ci_iv_space.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_absolute_iv_levels(samples_iv, targets_iv, cluster_labels, output_dir, max_paths=100):
    """
    Plot absolute IV levels (not normalized) to show actual volatility values.
    For fair comparison: show individual VAE samples, not means.
    """
    print("Generating absolute IV level plot...")

    grid_h, grid_w = 2, 2

    gt_paths = targets_iv[:, :, grid_h, grid_w]  # (n_seq, horizon)
    # Take ONE sample per sequence for fair comparison
    vae_single_sample_paths = samples_iv[:, 0, :, grid_h, grid_w]  # (n_seq, horizon)

    n_seq = min(max_paths, len(gt_paths))

    horizons = np.arange(1, gt_paths.shape[1] + 1)  # 1 to 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: GT paths (absolute IV)
    ax1 = axes[0]
    for i in range(n_seq):
        ax1.plot(horizons, gt_paths[i], color='gray', alpha=0.3, linewidth=0.5)

    gt_mean = gt_paths[:n_seq].mean(axis=0)
    gt_p05 = np.percentile(gt_paths[:n_seq], 5, axis=0)
    gt_p95 = np.percentile(gt_paths[:n_seq], 95, axis=0)

    ax1.fill_between(horizons, gt_p05, gt_p95, color='gray', alpha=0.2, label='90% CI')
    ax1.plot(horizons, gt_mean, color='black', linewidth=2, label='Mean')

    ax1.set_xlabel('Horizon (days)', fontsize=12)
    ax1.set_ylabel('Implied Volatility (absolute)', fontsize=12)
    ax1.set_title(f'Ground Truth IV Levels (n={n_seq})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: VAE paths (absolute IV)
    ax2 = axes[1]
    for i in range(n_seq):
        ax2.plot(horizons, vae_single_sample_paths[i], color='blue', alpha=0.3, linewidth=0.5)

    vae_mean = vae_single_sample_paths[:n_seq].mean(axis=0)
    vae_p05 = np.percentile(vae_single_sample_paths[:n_seq], 5, axis=0)
    vae_p95 = np.percentile(vae_single_sample_paths[:n_seq], 95, axis=0)

    ax2.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2, label='90% CI')
    ax2.plot(horizons, vae_mean, color='darkblue', linewidth=2, label='Mean')

    ax2.set_xlabel('Horizon (days)', fontsize=12)
    ax2.set_ylabel('Implied Volatility (absolute)', fontsize=12)
    ax2.set_title(f'VAE Posterior IV Levels (n={n_seq})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Match y-axis limits
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    plt.suptitle('Absolute IV Levels: GT vs VAE (Posterior Mode)\n'
                 'Grid Point: ATM, Mid-Tenor', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "absolute_iv_levels.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("Fanning Pattern Visualization (IV Space)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    uncond_data, cluster_labels = load_data()

    samples_iv = uncond_data["samples_iv"]
    targets_iv = uncond_data["targets_iv"]

    print(f"  Samples shape (IV): {samples_iv.shape}")
    print(f"  Targets shape (IV): {targets_iv.shape}")
    print(f"  Clusters: {len(np.unique(cluster_labels))}")

    # Create output directory
    output_dir = Path("models/backfill/two_stage/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate all plots
    print("\n" + "=" * 70)
    print("Generating IV Space Visualizations")
    print("=" * 70)

    plot_unconditional_fanning_iv(samples_iv, targets_iv, output_dir)
    plot_conditional_fanning_iv(samples_iv, targets_iv, cluster_labels, output_dir)
    plot_single_sequence_iv(samples_iv, targets_iv, cluster_labels, output_dir)
    plot_absolute_iv_levels(samples_iv, targets_iv, cluster_labels, output_dir)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nAll IV-space plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*iv*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
