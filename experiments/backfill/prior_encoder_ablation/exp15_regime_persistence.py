"""
Experiment 15: Regime Persistence Analysis

Measures how often a context stays in the same cluster (regime) over time.
Tests whether we can say "risky context -> risky horizon".

Key Question: If context at time t is in cluster X ("high vol regime"),
what's the probability that context at time t+h is still in cluster X?

High persistence (>70% at h=30): Regimes are sticky, we CAN assume stability
Low persistence (<50% at h=30): Regimes change frequently, CANNOT assume stability
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import json

CONTEXT_LEN = 60
HORIZONS = [1, 7, 14, 30, 60, 90]
N_CLUSTERS_LIST = [5, 10, 25]


def compute_transition_matrix(labels, horizon, n_clusters):
    """
    Compute P(cluster at t+h | cluster at t)

    Returns:
        transition_matrix: (K, K) where [i,j] = P(cluster_j at t+h | cluster_i at t)
    """
    transitions = np.zeros((n_clusters, n_clusters))
    for t in range(len(labels) - horizon):
        transitions[labels[t], labels[t + horizon]] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return transitions / row_sums


def plot_transition_matrix(T, K, h, output_dir):
    """Plot transition matrix heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(T, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='P(to | from)')
    plt.xlabel('To Cluster')
    plt.ylabel('From Cluster')
    plt.title(f'Transition Matrix (K={K}, h={h} days)\nDiagonal = Persistence')

    for i in range(K):
        for j in range(K):
            plt.text(j, i, f'{T[i,j]:.2f}', ha='center', va='center',
                    color='white' if T[i,j] > 0.5 else 'black', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f'transition_k{K}_h{h}.png', dpi=150)
    plt.close()


def plot_persistence_vs_horizon(labels, K, horizons, output_dir):
    """Plot persistence decay over horizon."""
    persistence_by_horizon = []
    for h in horizons:
        T = compute_transition_matrix(labels, h, K)
        persistence_by_horizon.append(np.diag(T))

    persistence_by_horizon = np.array(persistence_by_horizon)  # (len(horizons), K)

    plt.figure(figsize=(12, 6))
    for cluster_id in range(K):
        plt.plot(horizons, persistence_by_horizon[:, cluster_id] * 100,
                'o-', alpha=0.5, label=f'Cluster {cluster_id}')
    plt.plot(horizons, persistence_by_horizon.mean(axis=1) * 100,
            'k-', linewidth=3, label='Average')

    plt.xlabel('Horizon (days)')
    plt.ylabel('Persistence (%)')
    plt.title(f'Cluster Persistence vs Horizon (K={K})\n'
              f'P(stay in same cluster after h days)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    plt.tight_layout()
    plt.savefig(output_dir / f'persistence_vs_horizon_k{K}.png', dpi=150)
    plt.close()

    return persistence_by_horizon


def analyze_cluster_characteristics(surface, contexts, labels, kmeans, K, output_dir):
    """Analyze what each cluster represents (vol level, etc.)."""
    cluster_stats = {}

    for cluster_id in range(K):
        mask = labels == cluster_id
        cluster_contexts = contexts[mask]

        # Get last day ATM IV for each context in this cluster
        # Context is flattened (60*25), so reshape to get ATM
        cluster_contexts_reshaped = cluster_contexts.reshape(-1, 60, 5, 5)
        last_day_atm = cluster_contexts_reshaped[:, -1, 2, 2]  # ATM IV on last day

        cluster_stats[cluster_id] = {
            'count': int(mask.sum()),
            'mean_atm': float(last_day_atm.mean()),
            'std_atm': float(last_day_atm.std()),
            'min_atm': float(last_day_atm.min()),
            'max_atm': float(last_day_atm.max()),
        }

    # Sort clusters by mean ATM IV
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['mean_atm'])

    print("\nCluster Characteristics (sorted by mean ATM IV):")
    print("-" * 70)
    print(f"{'Cluster':<10} {'Count':<10} {'Mean ATM':<12} {'Std ATM':<12} {'Range':<20}")
    print("-" * 70)
    for cluster_id, stats in sorted_clusters:
        print(f"{cluster_id:<10} {stats['count']:<10} {stats['mean_atm']:.4f}      "
              f"{stats['std_atm']:.4f}      [{stats['min_atm']:.3f}, {stats['max_atm']:.3f}]")

    return cluster_stats


def main():
    print("=" * 70)
    print("EXPERIMENT 15: Regime Persistence Analysis")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surface = data["surface"]  # (N, 5, 5)
    print(f"\nData shape: {surface.shape}")

    # Extract all contexts
    n_samples = len(surface) - CONTEXT_LEN
    contexts = np.array([surface[i:i+CONTEXT_LEN].flatten()
                        for i in range(n_samples)])
    print(f"Number of contexts: {n_samples}")

    output_dir = Path("results/prior_encoder_ablation/exp15_regime_persistence")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for K in N_CLUSTERS_LIST:
        print(f"\n{'='*70}")
        print(f"K = {K} clusters")
        print('='*70)

        # Cluster
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(contexts)

        # Analyze cluster characteristics
        cluster_stats = analyze_cluster_characteristics(
            surface, contexts, labels, kmeans, K, output_dir
        )

        # Compute transitions for each horizon
        results_k = {'cluster_stats': cluster_stats, 'persistence': {}}

        print("\nPersistence by Horizon:")
        print("-" * 50)

        for h in HORIZONS:
            T = compute_transition_matrix(labels, h, K)
            persistence = np.diag(T)

            results_k['persistence'][h] = {
                'mean': float(persistence.mean()),
                'min': float(persistence.min()),
                'max': float(persistence.max()),
                'per_cluster': {i: float(p) for i, p in enumerate(persistence)}
            }

            print(f"h={h:2d}: Avg={persistence.mean()*100:5.1f}%  "
                  f"Min={persistence.min()*100:5.1f}%  "
                  f"Max={persistence.max()*100:5.1f}%")

            # Save heatmap for key horizons
            if h in [1, 7, 30]:
                plot_transition_matrix(T, K, h, output_dir)

        # Plot persistence vs horizon
        persistence_matrix = plot_persistence_vs_horizon(labels, K, HORIZONS, output_dir)
        all_results[K] = results_k

    # Summary comparison across K values
    print("\n" + "=" * 70)
    print("SUMMARY: Average Persistence by K and Horizon")
    print("=" * 70)
    print(f"{'K':<8}", end='')
    for h in HORIZONS:
        print(f"h={h:<5}", end='')
    print()
    print("-" * 70)

    for K in N_CLUSTERS_LIST:
        print(f"K={K:<5}", end='')
        for h in HORIZONS:
            p = all_results[K]['persistence'][h]['mean'] * 100
            print(f"{p:6.1f}%", end='')
        print()

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Use K=10 as reference
    K_ref = 10
    p30 = all_results[K_ref]['persistence'][30]['mean']
    p90 = all_results[K_ref]['persistence'][90]['mean']

    print(f"\nUsing K={K_ref} as reference:")
    print(f"  30-day persistence: {p30*100:.1f}%")
    print(f"  90-day persistence: {p90*100:.1f}%")

    if p30 > 0.7:
        print("\n  [CONCLUSION] Regimes are STICKY (>70% at h=30)")
        print("  -> We CAN say 'risky context -> expect risky horizon'")
        print("  -> Model could use regime as conditioning variable")
    elif p30 > 0.5:
        print("\n  [CONCLUSION] Regimes are MODERATELY persistent (50-70% at h=30)")
        print("  -> Some regime stability, but transitions are common")
        print("  -> Need to model regime transitions explicitly")
    else:
        print("\n  [CONCLUSION] Regimes are VOLATILE (<50% at h=30)")
        print("  -> CANNOT assume horizon stays in same regime")
        print("  -> Regime-based conditioning may not be reliable")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json")
    print(f"  - transition_k*_h*.png")
    print(f"  - persistence_vs_horizon_k*.png")


if __name__ == "__main__":
    main()
