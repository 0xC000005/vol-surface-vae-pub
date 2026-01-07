"""
Analyze Conditional Marginal Distribution Matching.

This script:
1. Clusters context embeddings using K-means
2. Analyzes CI violations within each cluster
3. Checks if conditional distributions match GT within similar contexts

The goal is to verify the context encoder groups similar contexts together
without over-discriminating.

Usage:
    python experiments/backfill/prior_encoder_ablation/analyze_conditional_marginal.py
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sys
sys.path.insert(0, ".")


def main():
    print("=" * 70)
    print("Conditional Marginal Distribution Analysis")
    print("(Grouped by Similar Context Embeddings)")
    print("=" * 70)

    # Load results from unconditional analysis
    results_path = Path("models/backfill/two_stage/unconditional_analysis.npz")
    if not results_path.exists():
        print(f"ERROR: Run analyze_unconditional_marginal.py first")
        return

    print(f"\nLoading results from {results_path}...")
    data = np.load(results_path)

    samples_log = data["samples_log"]  # (n_eval, n_samples, horizon, 5, 5)
    targets_log = data["targets_log"]  # (n_eval, horizon, 5, 5)
    samples_iv = data["samples_iv"]
    targets_iv = data["targets_iv"]
    ctx_embeddings = data["ctx_embeddings"]  # (n_eval, ctx_dim)

    n_eval, n_samples, horizon = samples_log.shape[:3]
    ctx_dim = ctx_embeddings.shape[1]

    print(f"  Sequences: {n_eval}")
    print(f"  Samples per sequence: {n_samples}")
    print(f"  Horizon: {horizon}")
    print(f"  Context embedding dim: {ctx_dim}")

    # ========================================
    # PHASE 1: Context Embedding Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("1. CONTEXT EMBEDDING ANALYSIS")
    print("=" * 70)

    # Standardize embeddings
    scaler = StandardScaler()
    ctx_scaled = scaler.fit_transform(ctx_embeddings)

    # Analyze embedding statistics
    print("\n1a. Embedding Statistics:")
    for i in range(ctx_dim):
        print(f"  Dim {i}: mean={ctx_embeddings[:, i].mean():.4f}, "
              f"std={ctx_embeddings[:, i].std():.4f}, "
              f"range=[{ctx_embeddings[:, i].min():.4f}, {ctx_embeddings[:, i].max():.4f}]")

    # ========================================
    # PHASE 2: Determine Optimal Clusters
    # ========================================
    print("\n" + "=" * 70)
    print("2. CLUSTER SELECTION")
    print("=" * 70)

    # Try different K values
    k_values = [2, 3, 4, 5, 6, 8, 10]
    silhouette_scores = []
    inertias = []

    print("\nEvaluating cluster quality:")
    print(f"{'K':<5} {'Inertia':<12} {'Silhouette':<12}")
    print("-" * 30)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(ctx_scaled)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(ctx_scaled, labels)

        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        print(f"{k:<5} {inertia:<12.2f} {silhouette:<12.4f}")

    # Choose K with best silhouette score (but at least 3 clusters)
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_k_idx]
    if best_k < 3:
        best_k = 3  # Ensure at least 3 clusters for meaningful analysis

    print(f"\nSelected K={best_k} (silhouette={silhouette_scores[best_k_idx]:.4f})")

    # ========================================
    # PHASE 3: Cluster Assignment
    # ========================================
    print("\n" + "=" * 70)
    print("3. CLUSTER ASSIGNMENT")
    print("=" * 70)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ctx_scaled)

    # Cluster statistics
    print("\nCluster Sizes:")
    for c in range(best_k):
        count = (cluster_labels == c).sum()
        pct = count / n_eval * 100
        print(f"  Cluster {c}: {count} sequences ({pct:.1f}%)")

    # Cluster centers in original space
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    print("\nCluster Centers (original space):")
    for c in range(best_k):
        print(f"  Cluster {c}: {centers_original[c]}")

    # ========================================
    # PHASE 4: Conditional CI Violations
    # ========================================
    print("\n" + "=" * 70)
    print("4. CONDITIONAL CI VIOLATIONS (IV Space)")
    print("=" * 70)

    # Compute CI violations per cluster
    ci_lower_iv = np.percentile(samples_iv, 5, axis=1)  # (n_eval, horizon, 5, 5)
    ci_upper_iv = np.percentile(samples_iv, 95, axis=1)
    violations_iv = (targets_iv < ci_lower_iv) | (targets_iv > ci_upper_iv)

    # Per-cluster, per-horizon violations
    cluster_violations = {}

    print("\n4a. Per-Cluster Violations by Horizon Group:")
    print(f"{'Cluster':<10} {'Size':<8} {'Short':<10} {'Medium':<10} {'Long':<10} {'Overall':<10}")
    print("-" * 60)

    for c in range(best_k):
        mask = cluster_labels == c
        cluster_viols = violations_iv[mask]  # (n_cluster, horizon, 5, 5)

        # Per-horizon group
        short_viol = cluster_viols[:, :5].mean() * 100
        medium_viol = cluster_viols[:, 5:15].mean() * 100
        long_viol = cluster_viols[:, 15:].mean() * 100
        overall_viol = cluster_viols.mean() * 100

        cluster_violations[c] = {
            "short": short_viol,
            "medium": medium_viol,
            "long": long_viol,
            "overall": overall_viol,
            "size": mask.sum()
        }

        print(f"C{c:<9} {mask.sum():<8} {short_viol:<10.1f} {medium_viol:<10.1f} "
              f"{long_viol:<10.1f} {overall_viol:<10.1f}")

    # ========================================
    # PHASE 5: Cross-Cluster Consistency
    # ========================================
    print("\n" + "=" * 70)
    print("5. CROSS-CLUSTER CONSISTENCY")
    print("=" * 70)

    overall_viols = [cluster_violations[c]["overall"] for c in range(best_k)]
    short_viols = [cluster_violations[c]["short"] for c in range(best_k)]
    medium_viols = [cluster_violations[c]["medium"] for c in range(best_k)]
    long_viols = [cluster_violations[c]["long"] for c in range(best_k)]

    print("\n5a. Violation Statistics Across Clusters:")
    print(f"  Overall: mean={np.mean(overall_viols):.2f}%, std={np.std(overall_viols):.2f}%, "
          f"range=[{np.min(overall_viols):.2f}%, {np.max(overall_viols):.2f}%]")
    print(f"  Short:   mean={np.mean(short_viols):.2f}%, std={np.std(short_viols):.2f}%")
    print(f"  Medium:  mean={np.mean(medium_viols):.2f}%, std={np.std(medium_viols):.2f}%")
    print(f"  Long:    mean={np.mean(long_viols):.2f}%, std={np.std(long_viols):.2f}%")

    # Check for over-discrimination
    max_diff = np.max(overall_viols) - np.min(overall_viols)
    print(f"\n5b. Over-Discrimination Check:")
    print(f"  Max cluster difference: {max_diff:.2f}%")
    if max_diff < 3:
        print(f"  ✓ Good: Clusters have consistent calibration (diff < 3%)")
    elif max_diff < 5:
        print(f"  ⚠ Moderate: Some variation across clusters (3% < diff < 5%)")
    else:
        print(f"  ✗ Warning: Large variation across clusters (diff >= 5%)")
        print(f"    This may indicate over-discrimination by context encoder")

    # ========================================
    # PHASE 6: Distribution Matching per Cluster
    # ========================================
    print("\n" + "=" * 70)
    print("6. DISTRIBUTION MATCHING PER CLUSTER")
    print("=" * 70)

    print("\n6a. KS Test (VAE vs GT) per Cluster at H=15:")
    for c in range(best_k):
        mask = cluster_labels == c
        vae_samples = samples_log[mask, :, 14].flatten()  # H=15
        gt_targets = targets_log[mask, 14].flatten()

        ks_stat, p_value = stats.ks_2samp(vae_samples, gt_targets)
        status = "Similar" if p_value > 0.05 else "Different"
        print(f"  Cluster {c}: KS={ks_stat:.4f}, p={p_value:.4f} → {status}")

    print("\n6b. Sample Std Ratio (VAE/GT) per Cluster at H=15:")
    for c in range(best_k):
        mask = cluster_labels == c
        vae_std = samples_log[mask, :, 14].std()
        gt_std = targets_log[mask, 14].std()
        ratio = vae_std / gt_std * 100
        print(f"  Cluster {c}: VAE std={vae_std:.4f}, GT std={gt_std:.4f}, ratio={ratio:.1f}%")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY: CONDITIONAL MARGINAL ANALYSIS")
    print("=" * 70)

    print(f"""
Context Embedding Clustering Results:

1. Optimal Clusters: K={best_k}
   - Silhouette Score: {silhouette_scores[k_values.index(best_k)]:.4f}

2. Conditional CI Violations (IV Space, Posterior Mode):
   - All clusters have violations ~{np.mean(overall_viols):.1f}% (target: 10%)
   - Cross-cluster std: {np.std(overall_viols):.2f}%
   - Max difference: {max_diff:.2f}%

3. Interpretation:
   - {"✓ Context encoder groups similar contexts appropriately" if max_diff < 5 else "⚠ Potential over-discrimination"}
   - {"✓ Consistent calibration across clusters" if np.std(overall_viols) < 2 else "⚠ Calibration varies by context"}
   - Overall CIs are {"over-conservative (too wide)" if np.mean(overall_viols) < 8 else "well-calibrated" if np.mean(overall_viols) < 12 else "under-conservative (too narrow)"}
""")

    # Save results
    output_path = Path("models/backfill/two_stage/conditional_analysis.npz")
    np.savez(output_path,
             cluster_labels=cluster_labels,
             cluster_centers=centers_original,
             cluster_violations={str(k): v for k, v in cluster_violations.items()},
             ctx_embeddings=ctx_embeddings,
             k=best_k,
             silhouette=silhouette_scores[k_values.index(best_k)])
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
