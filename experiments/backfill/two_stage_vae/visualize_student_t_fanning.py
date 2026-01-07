"""
Visualize Fanning Patterns for Student-t VAE.

Creates visualizations comparing GT vs Student-t VAE paths in:
1. Log-return space (unconditional + conditional by cluster)
2. IV space (unconditional + conditional by cluster)

Usage:
    python experiments/backfill/two_stage_vae/visualize_student_t_fanning.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, ".")


def load_data():
    """Load Student-t analysis data."""
    data_path = Path("models/backfill/two_stage/student_t_unconditional_analysis.npz")

    if not data_path.exists():
        raise FileNotFoundError(f"Run analyze_unconditional_marginal_student_t.py first: {data_path}")

    data = np.load(data_path)

    # Compute clusters
    ctx_embeddings = data["ctx_embeddings"]
    scaler = StandardScaler()
    ctx_scaled = scaler.fit_transform(ctx_embeddings)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ctx_scaled)

    return data, cluster_labels


def compute_cumulative_paths(log_returns):
    """Convert log-returns to cumulative paths starting from 0."""
    shape = list(log_returns.shape)
    shape[-3] = 1
    zeros = np.zeros(shape)
    cumsum = np.cumsum(log_returns, axis=-3)
    return np.concatenate([zeros, cumsum], axis=-3)


def plot_unconditional_fanning_log_return(samples_log, targets_log, output_dir, max_paths=100):
    """Plot unconditional fanning in log-return space."""
    print("Generating unconditional fanning plot (log-return space)...")

    grid_h, grid_w = 2, 2  # ATM

    gt_paths = targets_log[:, :, grid_h, grid_w]
    vae_single_sample_paths = samples_log[:, 0, :, grid_h, grid_w]

    n_seq = min(max_paths, len(gt_paths))

    gt_cumsum = np.concatenate([np.zeros((n_seq, 1)), np.cumsum(gt_paths[:n_seq], axis=1)], axis=1)
    vae_cumsum = np.concatenate([np.zeros((n_seq, 1)), np.cumsum(vae_single_sample_paths[:n_seq], axis=1)], axis=1)

    horizons = np.arange(gt_cumsum.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # GT paths
    ax1 = axes[0]
    for i in range(n_seq):
        ax1.plot(horizons, gt_cumsum[i], color='gray', alpha=0.3, linewidth=0.5)

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

    # VAE paths
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
    ax2.set_title(f'Student-t VAE Paths (n={n_seq})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Match y-axis limits
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    plt.suptitle('Student-t VAE: Unconditional Fanning (Log-Return Space, ATM)', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = output_dir / "student_t_unconditional_fanning_log_return.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_unconditional_fanning_iv(samples_iv, targets_iv, output_dir, max_paths=100):
    """Plot unconditional fanning in IV space."""
    print("Generating unconditional fanning plot (IV space)...")

    grid_h, grid_w = 2, 2

    gt_paths = targets_iv[:, :, grid_h, grid_w]
    vae_single_sample_paths = samples_iv[:, 0, :, grid_h, grid_w]

    n_seq = min(max_paths, len(gt_paths))

    # Normalize to start at 1.0
    gt_initial = gt_paths[:n_seq, 0:1]
    vae_initial = vae_single_sample_paths[:n_seq, 0:1]

    gt_normalized = gt_paths[:n_seq] / gt_initial
    vae_normalized = vae_single_sample_paths[:n_seq] / vae_initial

    gt_with_start = np.concatenate([np.ones((n_seq, 1)), gt_normalized], axis=1)
    vae_with_start = np.concatenate([np.ones((n_seq, 1)), vae_normalized], axis=1)

    horizons = np.arange(gt_with_start.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # GT
    ax1 = axes[0]
    for i in range(n_seq):
        ax1.plot(horizons, gt_with_start[i], color='gray', alpha=0.3, linewidth=0.5)

    gt_mean = gt_with_start.mean(axis=0)
    gt_p05 = np.percentile(gt_with_start, 5, axis=0)
    gt_p95 = np.percentile(gt_with_start, 95, axis=0)

    ax1.fill_between(horizons, gt_p05, gt_p95, color='gray', alpha=0.2, label='90% CI')
    ax1.plot(horizons, gt_mean, color='black', linewidth=2, label='Mean')

    ax1.set_xlabel('Horizon (days)', fontsize=12)
    ax1.set_ylabel('Relative IV', fontsize=12)
    ax1.set_title(f'Ground Truth (n={n_seq})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    # VAE
    ax2 = axes[1]
    for i in range(n_seq):
        ax2.plot(horizons, vae_with_start[i], color='blue', alpha=0.3, linewidth=0.5)

    vae_mean = vae_with_start.mean(axis=0)
    vae_p05 = np.percentile(vae_with_start, 5, axis=0)
    vae_p95 = np.percentile(vae_with_start, 95, axis=0)

    ax2.fill_between(horizons, vae_p05, vae_p95, color='blue', alpha=0.2, label='90% CI')
    ax2.plot(horizons, vae_mean, color='darkblue', linewidth=2, label='Mean')

    ax2.set_xlabel('Horizon (days)', fontsize=12)
    ax2.set_ylabel('Relative IV', fontsize=12)
    ax2.set_title(f'Student-t VAE (n={n_seq})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)

    # Match y-axis
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    plt.suptitle('Student-t VAE: Unconditional Fanning (IV Space, ATM)', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = output_dir / "student_t_unconditional_fanning_iv.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_conditional_fanning(samples_log, targets_log, cluster_labels, output_dir, max_paths=50):
    """Plot conditional fanning by cluster."""
    print("Generating conditional fanning plot (log-return space, by cluster)...")

    grid_h, grid_w = 2, 2
    n_clusters = 3

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = ['red', 'green', 'purple']
    cluster_names = ['Low Vol', 'Medium Vol', 'High Vol']

    for c in range(n_clusters):
        mask = cluster_labels == c
        n_in_cluster = mask.sum()

        gt_cluster = targets_log[mask][:, :, grid_h, grid_w]
        vae_cluster = samples_log[mask][:, 0, :, grid_h, grid_w]

        n_paths = min(max_paths, len(gt_cluster))

        gt_cumsum = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(gt_cluster[:n_paths], axis=1)], axis=1)
        vae_cumsum = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(vae_cluster[:n_paths], axis=1)], axis=1)

        horizons = np.arange(gt_cumsum.shape[1])

        # GT
        ax_gt = axes[0, c]
        for i in range(n_paths):
            ax_gt.plot(horizons, gt_cumsum[i], color=colors[c], alpha=0.3, linewidth=0.5)

        gt_mean = gt_cumsum.mean(axis=0)
        ax_gt.plot(horizons, gt_mean, color='black', linewidth=2)
        ax_gt.set_title(f'GT - {cluster_names[c]} (n={n_in_cluster})', fontsize=12)
        ax_gt.grid(True, alpha=0.3)
        ax_gt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # VAE
        ax_vae = axes[1, c]
        for i in range(n_paths):
            ax_vae.plot(horizons, vae_cumsum[i], color=colors[c], alpha=0.3, linewidth=0.5)

        vae_mean = vae_cumsum.mean(axis=0)
        ax_vae.plot(horizons, vae_mean, color='black', linewidth=2)
        ax_vae.set_title(f'Student-t VAE - {cluster_names[c]}', fontsize=12)
        ax_vae.grid(True, alpha=0.3)
        ax_vae.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_vae.set_xlabel('Horizon (days)')

    axes[0, 0].set_ylabel('Cumulative Log-Return')
    axes[1, 0].set_ylabel('Cumulative Log-Return')

    plt.suptitle('Student-t VAE: Conditional Fanning by Cluster (Log-Return Space, ATM)', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = output_dir / "student_t_conditional_fanning_clusters.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary(samples_log, targets_log, samples_iv, targets_iv, output_dir, learned_nu):
    """Create summary plot with key statistics."""
    print("Generating summary plot...")

    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    grid_h, grid_w = 2, 2

    # 1. Cumulative path comparison
    ax1 = axes[0, 0]
    n_paths = 50
    gt_paths = targets_log[:n_paths, :, grid_h, grid_w]
    vae_paths = samples_log[:n_paths, 0, :, grid_h, grid_w]

    gt_cumsum = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(gt_paths, axis=1)], axis=1)
    vae_cumsum = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(vae_paths, axis=1)], axis=1)

    horizons = np.arange(gt_cumsum.shape[1])

    for i in range(n_paths):
        ax1.plot(horizons, gt_cumsum[i], color='gray', alpha=0.2, linewidth=0.5)
        ax1.plot(horizons, vae_cumsum[i], color='blue', alpha=0.2, linewidth=0.5)

    ax1.plot(horizons, gt_cumsum.mean(axis=0), color='black', linewidth=2, label='GT mean')
    ax1.plot(horizons, vae_cumsum.mean(axis=0), color='darkblue', linewidth=2, label='VAE mean')
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Cumulative Log-Return')
    ax1.set_title('Path Comparison (ATM)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribution at H=15
    ax2 = axes[0, 1]
    h = 14  # H=15
    gt_h15 = targets_log[:, h, grid_h, grid_w]
    vae_h15 = samples_log[:, :, h, grid_h, grid_w].flatten()

    bins = np.linspace(-1.5, 1.5, 50)
    ax2.hist(gt_h15, bins=bins, alpha=0.5, density=True, label=f'GT (kurt={stats.kurtosis(gt_h15):.1f})')
    ax2.hist(vae_h15, bins=bins, alpha=0.5, density=True, label=f'VAE (kurt={stats.kurtosis(vae_h15):.1f})')
    ax2.set_xlabel('Log-Return')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Distribution at H=15 (nu={learned_nu:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Kurtosis across horizons
    ax3 = axes[1, 0]
    gt_kurtosis = [stats.kurtosis(targets_log[:, h, grid_h, grid_w]) for h in range(30)]
    vae_kurtosis = [stats.kurtosis(samples_log[:, :, h, grid_h, grid_w].flatten()) for h in range(30)]

    ax3.plot(range(1, 31), gt_kurtosis, 'o-', color='black', label='GT')
    ax3.plot(range(1, 31), vae_kurtosis, 's-', color='blue', label='VAE')
    ax3.axhline(y=6/(learned_nu-4)+3-3, color='red', linestyle='--', label=f'Theoretical (nu={learned_nu:.2f})')
    ax3.set_xlabel('Horizon')
    ax3.set_ylabel('Excess Kurtosis')
    ax3.set_title('Kurtosis Across Horizons (ATM)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. CI width comparison
    ax4 = axes[1, 1]
    gt_iv_p05 = np.percentile(targets_iv[:, :, grid_h, grid_w], 5, axis=0)
    gt_iv_p95 = np.percentile(targets_iv[:, :, grid_h, grid_w], 95, axis=0)
    gt_width = gt_iv_p95 - gt_iv_p05

    vae_iv_p05 = np.percentile(samples_iv[:, :, :, grid_h, grid_w], 5, axis=(0, 1))
    vae_iv_p95 = np.percentile(samples_iv[:, :, :, grid_h, grid_w], 95, axis=(0, 1))
    vae_width = vae_iv_p95 - vae_iv_p05

    ax4.plot(range(1, 31), gt_width, 'o-', color='black', label='GT 90% CI width')
    ax4.plot(range(1, 31), vae_width, 's-', color='blue', label='VAE 90% CI width')
    ax4.set_xlabel('Horizon')
    ax4.set_ylabel('CI Width (IV)')
    ax4.set_title('Confidence Interval Width (ATM)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Student-t VAE Summary (learned nu={learned_nu:.2f})', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = output_dir / "student_t_fanning_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("Student-t VAE Fanning Visualization")
    print("=" * 70)

    # Load data
    data, cluster_labels = load_data()

    samples_log = data["samples_log"]
    targets_log = data["targets_log"]
    samples_iv = data["samples_iv"]
    targets_iv = data["targets_iv"]
    learned_nu = float(data["learned_nu"])

    print(f"\nLoaded data:")
    print(f"  Samples: {samples_log.shape}")
    print(f"  Targets: {targets_log.shape}")
    print(f"  Learned nu: {learned_nu:.2f}")

    output_dir = Path("models/backfill/two_stage/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_unconditional_fanning_log_return(samples_log, targets_log, output_dir)
    plot_unconditional_fanning_iv(samples_iv, targets_iv, output_dir)
    plot_conditional_fanning(samples_log, targets_log, cluster_labels, output_dir)
    plot_summary(samples_log, targets_log, samples_iv, targets_iv, output_dir, learned_nu)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - student_t_unconditional_fanning_log_return.png")
    print(f"  - student_t_unconditional_fanning_iv.png")
    print(f"  - student_t_conditional_fanning_clusters.png")
    print(f"  - student_t_fanning_summary.png")


if __name__ == "__main__":
    main()
