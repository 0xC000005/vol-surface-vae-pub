"""
Experiment 14: Context Coarsening - Conditional Variance Analysis

Goal: Verify that conditional variance exists in the data when we group
similar 60-day contexts together.

Hypothesis: The context encoder is "too good" - it encodes each context so
precisely that no conditional variance emerges. If we group similar contexts,
their targets should show variance ("fanning pattern").

Method:
1. Extract all 60-day contexts and their next-day targets
2. Cluster contexts using K-Means (K = 10, 50, 100, 500)
3. For each cluster, compute within-cluster target variance
4. Calculate P1 = E[Var(X|C)] / Var(X) at each K level
5. Visualize fanning: Stack GT targets within clusters to see spread
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm

# Configuration
CONTEXT_LEN = 60  # Fixed context length
HORIZON = 30  # Look 30 days ahead for fanning visualization
N_CLUSTERS_LIST = [10, 25, 50, 100, 200, 500]


def load_data():
    """Load volatility surface data."""
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = data["surface"]  # Shape: (N, 5, 5)
    print(f"Loaded surface data: {surface.shape}")
    print(f"Total days: {len(surface)}")
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Available samples: {len(surface) - CONTEXT_LEN}")
    return surface


def extract_contexts_and_targets(surface, context_len=CONTEXT_LEN):
    """
    Extract all context-target pairs.

    Returns:
        contexts: (N, context_len * 25) - flattened contexts
        targets: (N, 5, 5) - next-day surfaces
        indices: (N,) - original indices
    """
    n_samples = len(surface) - context_len
    contexts = []
    targets = []
    indices = []

    for i in range(n_samples):
        ctx = surface[i:i + context_len]  # (60, 5, 5)
        tgt = surface[i + context_len]     # (5, 5)
        contexts.append(ctx.flatten())
        targets.append(tgt)
        indices.append(i + context_len)

    return np.array(contexts), np.array(targets), np.array(indices)


def compute_p1_by_clustering(contexts, targets, n_clusters_list=N_CLUSTERS_LIST):
    """
    Compute P1 = E[Var(X|C)] / Var(X) for different clustering granularities.

    P1 measures conditional variance: how much variance remains when we
    condition on context. If P1 > 2%, meaningful conditional variance exists.
    """
    # Normalize contexts for better clustering
    scaler = StandardScaler()
    contexts_scaled = scaler.fit_transform(contexts)

    # Total variance (unconditional)
    total_var = targets.var()
    print(f"\nTotal target variance: {total_var:.6f}")

    results = {}

    for n_clusters in n_clusters_list:
        print(f"\nClustering with K={n_clusters}...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(contexts_scaled)

        within_vars = []
        group_sizes = []
        cluster_stats = []

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_size = mask.sum()

            if cluster_size < 2:
                continue

            cluster_targets = targets[mask]
            within_var = cluster_targets.var()
            within_vars.append(within_var)
            group_sizes.append(cluster_size)

            # ATM IV stats for this cluster
            atm_ivs = cluster_targets[:, 2, 2]
            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'size': int(cluster_size),
                'within_var': float(within_var),
                'atm_mean': float(atm_ivs.mean()),
                'atm_std': float(atm_ivs.std()),
            })

        # Weighted average within-group variance
        E_var_given_C = np.average(within_vars, weights=group_sizes)
        P1 = E_var_given_C / total_var * 100  # As percentage

        results[n_clusters] = {
            'n_clusters': n_clusters,
            'n_valid_clusters': len(within_vars),
            'avg_cluster_size': float(np.mean(group_sizes)),
            'min_cluster_size': int(np.min(group_sizes)),
            'max_cluster_size': int(np.max(group_sizes)),
            'E_var_given_C': float(E_var_given_C),
            'total_var': float(total_var),
            'P1_percent': float(P1),
            'labels': labels,
            'cluster_stats': sorted(cluster_stats, key=lambda x: -x['size'])[:10],
        }

        print(f"  Valid clusters: {len(within_vars)}")
        print(f"  Avg cluster size: {np.mean(group_sizes):.1f}")
        print(f"  E[Var(X|C)]: {E_var_given_C:.6f}")
        print(f"  P1 = {P1:.2f}%")

    return results


def visualize_fanning(targets, labels, n_clusters, output_dir, top_k=6):
    """
    Visualize "fanning pattern" for largest clusters.

    For each cluster, show all GT targets stacked. If they fan out,
    conditional variance exists in the data.
    """
    cluster_sizes = [(labels == i).sum() for i in range(n_clusters)]
    largest = np.argsort(cluster_sizes)[-top_k:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, cluster_id in zip(axes, largest):
        mask = labels == cluster_id
        cluster_targets = targets[mask]  # (M, 5, 5)

        # Plot ATM IV for each target in cluster
        atm_ivs = cluster_targets[:, 2, 2]  # ATM point

        # Sort by value for better visualization
        sorted_ivs = np.sort(atm_ivs)

        # Histogram-style visualization
        ax.hist(atm_ivs, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(atm_ivs.mean(), color='red', linestyle='--',
                   label=f'mean={atm_ivs.mean():.3f}')

        ax.set_title(f"Cluster {cluster_id}\nn={mask.sum()}, std={atm_ivs.std():.4f}")
        ax.set_xlabel("ATM IV (next day)")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle(f"Fanning Pattern: K={n_clusters} clusters\n"
                 "Wide spread = conditional variance EXISTS", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"fanning_k{n_clusters}.png", dpi=150)
    plt.close()
    print(f"  Saved: fanning_k{n_clusters}.png")


def visualize_fanning_lines(targets, labels, n_clusters, output_dir, top_k=6):
    """
    Alternative fanning visualization: show each target as a horizontal line.
    """
    cluster_sizes = [(labels == i).sum() for i in range(n_clusters)]
    largest = np.argsort(cluster_sizes)[-top_k:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, cluster_id in zip(axes, largest):
        mask = labels == cluster_id
        cluster_targets = targets[mask]

        atm_ivs = cluster_targets[:, 2, 2]
        sorted_ivs = np.sort(atm_ivs)

        # Plot each IV as a horizontal line, stacked vertically
        for i, iv in enumerate(sorted_ivs):
            ax.plot([0, 1], [iv, iv], 'b-', alpha=0.3, linewidth=1)

        ax.set_xlim(-0.1, 1.1)
        ax.set_title(f"Cluster {cluster_id}\nn={mask.sum()}, std={atm_ivs.std():.4f}")
        ax.set_ylabel("ATM IV")
        ax.set_xticks([])

    plt.suptitle(f"Fanning Lines: K={n_clusters} clusters\n"
                 "Vertical spread = conditional variance", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"fanning_lines_k{n_clusters}.png", dpi=150)
    plt.close()
    print(f"  Saved: fanning_lines_k{n_clusters}.png")


def extract_30day_trajectories(surface, context_len=CONTEXT_LEN, horizon=HORIZON):
    """
    Extract 60-day contexts and their NEXT 30 DAYS of ATM IV.

    Returns:
        contexts: (N, context_len * 25) - flattened contexts
        trajectories: (N, horizon) - ATM IV for next 30 days
    """
    n_samples = len(surface) - context_len - horizon
    contexts = []
    trajectories = []

    for i in range(n_samples):
        ctx = surface[i:i + context_len]  # 60-day context
        # Next 30 days of ATM IV
        future_atm = surface[i + context_len:i + context_len + horizon, 2, 2]
        contexts.append(ctx.flatten())
        trajectories.append(future_atm)

    return np.array(contexts), np.array(trajectories)


def visualize_30day_fanning(trajectories, labels, n_clusters, output_dir, top_k=6):
    """
    For each cluster, plot all 30-day ATM IV trajectories stacked.
    Shows fanning pattern over time.

    This is the key visualization showing:
    - X-axis: Days 1-30 after context ends
    - Y-axis: ATM IV
    - Each blue line: One actual GT trajectory from that cluster
    - Fanning: Lines diverge over time = conditional variance exists
    """
    cluster_sizes = [(labels == i).sum() for i in range(n_clusters)]
    largest = np.argsort(cluster_sizes)[-top_k:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    days = np.arange(1, HORIZON + 1)  # Days 1-30

    for ax, cluster_id in zip(axes, largest):
        mask = labels == cluster_id
        cluster_trajs = trajectories[mask]  # (M, 30)

        # Limit number of trajectories for visibility
        n_trajs = min(200, len(cluster_trajs))
        sample_idx = np.random.choice(len(cluster_trajs), n_trajs, replace=False)
        sampled_trajs = cluster_trajs[sample_idx]

        # Plot each trajectory as a line
        for traj in sampled_trajs:
            ax.plot(days, traj, 'b-', alpha=0.15, linewidth=0.5)

        # Plot mean and std bands
        mean_traj = cluster_trajs.mean(axis=0)
        std_traj = cluster_trajs.std(axis=0)
        ax.plot(days, mean_traj, 'r-', linewidth=2, label='Mean')
        ax.fill_between(days, mean_traj - std_traj, mean_traj + std_traj,
                       alpha=0.3, color='red', label='±1 std')

        ax.set_xlabel("Days after context ends")
        ax.set_ylabel("ATM IV")
        ax.set_title(f"Cluster {cluster_id}\nn={mask.sum()}, std@day1={std_traj[0]:.4f}, std@day30={std_traj[-1]:.4f}")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"30-Day Fanning Pattern: K={n_clusters} clusters\n"
                 "Blue lines = actual GT trajectories, Red = mean ± std", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"fanning_30day_k{n_clusters}.png", dpi=150)
    plt.close()
    print(f"  Saved: fanning_30day_k{n_clusters}.png")


def visualize_std_vs_horizon(trajectories, labels, n_clusters, output_dir, top_k=6):
    """
    Plot how standard deviation increases with horizon for each cluster.
    Shows if uncertainty grows over time (fanning out).
    """
    cluster_sizes = [(labels == i).sum() for i in range(n_clusters)]
    largest = np.argsort(cluster_sizes)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(12, 6))
    days = np.arange(1, HORIZON + 1)

    colors = plt.cm.tab10(np.linspace(0, 1, top_k))

    for color, cluster_id in zip(colors, largest):
        mask = labels == cluster_id
        cluster_trajs = trajectories[mask]

        std_by_day = cluster_trajs.std(axis=0)
        ax.plot(days, std_by_day, '-', color=color, linewidth=2,
               label=f'Cluster {cluster_id} (n={mask.sum()})')

    ax.set_xlabel("Days after context ends")
    ax.set_ylabel("Standard Deviation of ATM IV")
    ax.set_title(f"Conditional Variance Growth Over 30-Day Horizon\n"
                 f"K={n_clusters} clusters")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"std_vs_horizon_k{n_clusters}.png", dpi=150)
    plt.close()
    print(f"  Saved: std_vs_horizon_k{n_clusters}.png")


def visualize_p1_vs_k(results, output_dir):
    """Plot P1 vs number of clusters."""
    ks = sorted(results.keys())
    p1s = [results[k]['P1_percent'] for k in ks]
    avg_sizes = [results[k]['avg_cluster_size'] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # P1 vs K
    ax1.plot(ks, p1s, 'bo-', markersize=10, linewidth=2)
    ax1.axhline(2, color='red', linestyle='--', label='Target: P1 > 2%')
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("P1 = E[Var(X|C)] / Var(X) (%)")
    ax1.set_title("Conditional Variance vs Clustering Granularity")
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Avg cluster size vs K
    ax2.plot(ks, avg_sizes, 'go-', markersize=10, linewidth=2)
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Average Cluster Size")
    ax2.set_title("Cluster Size vs K")
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "p1_vs_k.png", dpi=150)
    plt.close()
    print(f"Saved: p1_vs_k.png")


def analyze_by_regime(surface, targets, context_len=CONTEXT_LEN):
    """
    Alternative analysis: group by volatility regime instead of clustering.
    """
    # Use ATM IV of the last day of context to determine regime
    atm_ivs = surface[context_len-1:-1, 2, 2]  # Last day of each context

    # Define regimes by terciles
    terciles = np.percentile(atm_ivs, [33, 67])

    regimes = np.zeros(len(atm_ivs), dtype=int)
    regimes[atm_ivs <= terciles[0]] = 0  # Low vol
    regimes[(atm_ivs > terciles[0]) & (atm_ivs <= terciles[1])] = 1  # Mid vol
    regimes[atm_ivs > terciles[1]] = 2  # High vol

    regime_names = ['Low Vol', 'Mid Vol', 'High Vol']
    total_var = targets.var()

    print("\n" + "="*60)
    print("REGIME-BASED ANALYSIS")
    print("="*60)

    within_vars = []
    group_sizes = []

    for regime_id, name in enumerate(regime_names):
        mask = regimes == regime_id
        regime_targets = targets[mask]
        within_var = regime_targets.var()

        within_vars.append(within_var)
        group_sizes.append(mask.sum())

        atm_ivs_regime = regime_targets[:, 2, 2]
        print(f"\n{name}:")
        print(f"  N samples: {mask.sum()}")
        print(f"  Within-group variance: {within_var:.6f}")
        print(f"  ATM IV range: [{atm_ivs_regime.min():.3f}, {atm_ivs_regime.max():.3f}]")
        print(f"  ATM IV std: {atm_ivs_regime.std():.4f}")

    E_var_given_C = np.average(within_vars, weights=group_sizes)
    P1 = E_var_given_C / total_var * 100

    print(f"\nRegime-based P1 = {P1:.2f}%")

    return {
        'method': 'regime',
        'n_groups': 3,
        'P1_percent': float(P1),
        'E_var_given_C': float(E_var_given_C),
        'total_var': float(total_var),
    }


def main():
    print("="*70)
    print("EXPERIMENT 14: Context Coarsening - Conditional Variance Analysis")
    print("="*70)

    # Output directory
    output_dir = Path("results/prior_encoder_ablation/exp14_context_granularity")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    surface = load_data()

    # Extract contexts and targets
    print("\nExtracting context-target pairs...")
    contexts, targets, indices = extract_contexts_and_targets(surface)
    print(f"Extracted {len(contexts)} pairs")
    print(f"Context shape: {contexts.shape}")
    print(f"Target shape: {targets.shape}")

    # K-Means clustering analysis
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("="*60)
    results = compute_p1_by_clustering(contexts, targets)

    # Visualizations (next-day only)
    print("\nGenerating next-day visualizations...")
    for n_clusters in [10, 50, 100]:
        if n_clusters in results:
            visualize_fanning(targets, results[n_clusters]['labels'],
                            n_clusters, output_dir)
            visualize_fanning_lines(targets, results[n_clusters]['labels'],
                                   n_clusters, output_dir)

    visualize_p1_vs_k(results, output_dir)

    # 30-Day Fanning Analysis
    print("\n" + "="*60)
    print("30-DAY FANNING ANALYSIS")
    print("="*60)

    print("\nExtracting 30-day trajectories...")
    contexts_30d, trajectories_30d = extract_30day_trajectories(surface)
    print(f"Extracted {len(contexts_30d)} trajectory pairs")
    print(f"Trajectory shape: {trajectories_30d.shape}")

    # Cluster the 30-day contexts (re-cluster because we have fewer samples)
    scaler = StandardScaler()
    contexts_30d_scaled = scaler.fit_transform(contexts_30d)

    print("\nGenerating 30-day fanning visualizations...")
    for n_clusters in [10, 25, 50]:
        print(f"\nClustering with K={n_clusters} for 30-day analysis...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_30d = kmeans.fit_predict(contexts_30d_scaled)

        visualize_30day_fanning(trajectories_30d, labels_30d, n_clusters, output_dir)
        visualize_std_vs_horizon(trajectories_30d, labels_30d, n_clusters, output_dir)

    # Regime-based analysis
    regime_results = analyze_by_regime(surface, targets)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nP1 by clustering granularity:")
    print("-" * 50)
    print(f"{'K':<10} {'Avg Size':<12} {'P1 (%)':<10} {'Status'}")
    print("-" * 50)
    for k in sorted(results.keys()):
        r = results[k]
        status = "✓ PASS" if r['P1_percent'] > 2 else "✗ FAIL"
        print(f"{k:<10} {r['avg_cluster_size']:<12.1f} {r['P1_percent']:<10.2f} {status}")

    print(f"\nRegime-based: P1 = {regime_results['P1_percent']:.2f}%")

    # Save results
    save_results = {
        'context_len': CONTEXT_LEN,
        'n_samples': len(contexts),
        'total_var': float(targets.var()),
        'clustering': {k: {key: val for key, val in v.items() if key != 'labels'}
                       for k, v in results.items()},
        'regime': regime_results,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_dir}/results.json")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    max_p1 = max(r['P1_percent'] for r in results.values())
    if max_p1 > 2:
        print(f"✓ Conditional variance EXISTS in data!")
        print(f"  Maximum P1 = {max_p1:.2f}% achieved at coarse clustering")
        print(f"  → Context encoder is too precise, not a data limitation")
        print(f"  → Proceed to Phase 2: train with context bottleneck")
    else:
        print(f"✗ Conditional variance is LOW even at coarse clustering")
        print(f"  Maximum P1 = {max_p1:.2f}%")
        print(f"  → May need alternative approach (synthetic augmentation)")


if __name__ == "__main__":
    main()
