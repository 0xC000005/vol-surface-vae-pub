#!/usr/bin/env python3
"""
Experiment 2: Latent Space Analysis

Tests hypothesis: Under-regularized latent space (KL weight=1e-5) causes fragmented representations
leading to high epistemic uncertainty.

Strategy:
- Extract latent embeddings z for all test sequences
- Visualize latent space with PCA (colored by context endpoint)
- Compute correlation between latent distance and prediction distance
- Analyze effective dimensionality

Decision Criteria:
- If latent space shows NO clear clustering by endpoint: ✅ FIX: Increase KL weight
- If correlation(latent_dist, pred_dist) is LOW (<0.3): ✅ Increase KL weight
- If effective dimensionality is HIGH (>50% of latent_dim): ✅ Under-constrained

Author: Generated with Claude Code
Date: 2025-12-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Constants
# ============================================================================

CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)
OUTPUT_DIR = Path("results/context60_baseline/analysis/preliminary_experiments/latent_space_analysis")
MODEL_PATH = "models/backfill/context60_experiment/checkpoints/backfill_context60_best.pt"


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path):
    """Load trained context60 model.

    Returns:
        model, model_config
    """
    print(f"Loading model from {model_path}...")

    # Load model data (weights_only=False since this is a trusted checkpoint)
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = model_data['model_config']

    print(f"  Model config:")
    print(f"    Latent dim: {model_config.get('latent_dim', 'unknown')}")
    print(f"    KL weight: {model_config.get('kl_weight', 'unknown')}")
    print(f"    Context lengths: {model_config.get('context_len', 'unknown')}")

    # Import model class
    from vae.cvae_with_mem_randomized import CVAEMemRand

    # Initialize model
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()

    print(f"  ✓ Model loaded successfully")

    return model, model_config


# ============================================================================
# Data Loading
# ============================================================================

def load_predictions_and_data():
    """Load oracle predictions, ground truth, and indices.

    Returns:
        dict with 'surfaces', 'indices', 'gt_surface'
    """
    print("Loading predictions and ground truth...")

    # Load oracle predictions
    pred_file = (f"results/context60_baseline/predictions/teacher_forcing/"
                 f"oracle/vae_tf_insample_h{HORIZON}.npz")
    pred_data = np.load(pred_file)
    surfaces = pred_data['surfaces']  # (N, 90, 3, 5, 5)
    indices = pred_data['indices']     # (N,)

    # Load ground truth
    gt_data = np.load("data/vol_surface_with_ret.npz")
    gt_surface = gt_data['surface']  # (T, 5, 5)

    print(f"  Loaded {len(indices)} prediction sequences")
    print(f"  Ground truth: {len(gt_surface)} days")

    return {
        'surfaces': surfaces,
        'indices': indices,
        'gt_surface': gt_surface
    }


# ============================================================================
# Latent Embedding Extraction
# ============================================================================

def extract_latent_embeddings(model, gt_surface, indices, context_len=60):
    """Extract latent embeddings for all test sequences.

    Args:
        model: Trained VAE model
        gt_surface: (T, 5, 5) ground truth
        indices: (N,) sequence indices
        context_len: Context length

    Returns:
        dict with 'latent_z', 'context_endpoints', 'p50_predictions'
    """
    print("Extracting latent embeddings...")

    n_sequences = len(indices)
    latent_dim = model.config.get('latent_dim', 32)
    grid_row, grid_col = ATM_6M

    latent_z = np.zeros((n_sequences, latent_dim))
    context_endpoints = np.zeros(n_sequences)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i % 500 == 0:
                print(f"  Processing {i}/{n_sequences}...")

            # Extract context
            context_start = max(0, idx - context_len)
            context_surfaces = gt_surface[context_start:idx]  # (C, 5, 5)

            if len(context_surfaces) < context_len:
                # Pad if necessary
                padding = np.zeros((context_len - len(context_surfaces), 5, 5))
                context_surfaces = np.concatenate([padding, context_surfaces], axis=0)

            # Convert to tensor and add batch dimension
            context_tensor = torch.FloatTensor(context_surfaces).unsqueeze(0)  # (1, C, 5, 5)
            context_tensor = context_tensor.to(model.device)  # Move to same device as model

            # Create dummy ex_feats (zeros) since model was trained with them
            # Shape: (1, C, 3) where 3 = [returns, skew, slope]
            ex_feats_tensor = torch.zeros(1, context_len, 3).to(model.device)

            # Create input dict
            input_dict = {
                'surface': context_tensor,
                'ex_feats': ex_feats_tensor
            }

            # Extract latent embedding using encoder
            # encoder returns (z_mean, z_logvar, z_sample) with shape (B, C, latent_dim)
            z_mean, z_logvar, z_sample = model.encoder(input_dict)

            # Pool temporal dimension: take mean across context timesteps
            # This gives us a single latent vector per sequence
            z_pooled = z_mean.mean(dim=1).squeeze(0).cpu().numpy()  # (latent_dim,)
            latent_z[i] = z_pooled

            # Store context endpoint
            context_endpoints[i] = context_surfaces[-1, grid_row, grid_col]

    print(f"  ✓ Extracted {n_sequences} latent embeddings (dim={latent_dim})")

    return {
        'latent_z': latent_z,
        'context_endpoints': context_endpoints
    }


# ============================================================================
# Analysis 1: PCA Visualization
# ============================================================================

def visualize_latent_space_pca(latent_z, context_endpoints, output_dir):
    """Visualize latent space using PCA, colored by context endpoint.

    Args:
        latent_z: (N, latent_dim) latent embeddings
        context_endpoints: (N,) context endpoint values
        output_dir: Output directory
    """
    print("[PRIMARY] Generating PCA visualization...")

    # Perform PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_z)

    explained_var = pca.explained_variance_ratio_
    print(f"  PCA explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter plot colored by context endpoint
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                        c=context_endpoints, cmap='viridis',
                        alpha=0.6, s=50, edgecolor='black', linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Context Endpoint (ATM 6M IV)', fontsize=12, weight='bold')

    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12, weight='bold')
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12, weight='bold')
    ax.set_title('Latent Space PCA Visualization (Colored by Context Endpoint)\n'
                 'Well-structured latent space should show clear clustering by endpoint',
                fontsize=14, weight='bold', pad=15)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)

    # Add interpretation box
    # Check if there's clear structure
    # Heuristic: If PC1+PC2 explain <30% variance, likely fragmented
    total_explained = explained_var[0] + explained_var[1]

    if total_explained < 0.3:
        verdict = "❌ FRAGMENTED: First 2 PCs explain <30% variance"
        recommendation = "INCREASE KL weight from 1e-5 to 1e-3"
        box_color = 'lightcoral'
    elif total_explained < 0.5:
        verdict = "⚠️  MODERATE: First 2 PCs explain 30-50% variance"
        recommendation = "Consider increasing KL weight"
        box_color = 'lightyellow'
    else:
        verdict = "✅ STRUCTURED: First 2 PCs explain >50% variance"
        recommendation = "Latent space appears well-structured"
        box_color = 'lightgreen'

    interpretation_text = (
        f"LATENT SPACE STRUCTURE:\n"
        f"{verdict}\n\n"
        f"Total explained (PC1+PC2): {total_explained:.1%}\n"
        f"KL weight: 1e-5 (very low)\n\n"
        f"{recommendation}"
    )

    props = dict(boxstyle='round,pad=1', facecolor=box_color, alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, interpretation_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    # Save
    filepath = output_dir / 'latent_pca_by_endpoint.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()

    return {'pca_explained_var': explained_var, 'pca': pca}


# ============================================================================
# Analysis 2: Latent Distance vs Prediction Distance
# ============================================================================

def compute_latent_prediction_correlation(latent_z, p50_predictions, context_endpoints, output_dir):
    """Compute correlation between latent distance and prediction distance.

    For sequences with similar context endpoints, check if similar latent embeddings
    lead to similar predictions.

    Args:
        latent_z: (N, latent_dim)
        p50_predictions: (N, 90) - p50 predictions for all horizons
        context_endpoints: (N,)
        output_dir: Output directory
    """
    print("[PRIMARY] Computing latent vs prediction distance correlation...")

    # Focus on day 1 predictions (most relevant for epistemic uncertainty)
    day1_predictions = p50_predictions[:, 0]  # (N,)

    # Find pairs of sequences with similar context endpoints (±1%)
    n_sequences = len(context_endpoints)
    similarity_threshold = 0.01  # ±1%

    latent_distances = []
    prediction_distances = []
    endpoint_diffs = []

    print(f"  Finding similar context pairs (±{similarity_threshold*100:.0f}%)...")

    for i in range(n_sequences):
        for j in range(i+1, min(i+100, n_sequences)):  # Sample 100 neighbors to keep it tractable
            endpoint_diff = abs(context_endpoints[i] - context_endpoints[j])

            if endpoint_diff <= similarity_threshold:
                # Compute latent distance
                latent_dist = np.linalg.norm(latent_z[i] - latent_z[j])

                # Compute prediction distance
                pred_dist = abs(day1_predictions[i] - day1_predictions[j])

                latent_distances.append(latent_dist)
                prediction_distances.append(pred_dist)
                endpoint_diffs.append(endpoint_diff)

    print(f"  Found {len(latent_distances)} similar pairs")

    if len(latent_distances) == 0:
        print("  ⚠️  No similar pairs found, skipping correlation analysis")
        return None

    latent_distances = np.array(latent_distances)
    prediction_distances = np.array(prediction_distances)

    # Compute correlation
    correlation, p_value = pearsonr(latent_distances, prediction_distances)
    print(f"  Correlation: {correlation:.3f} (p={p_value:.3e})")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(latent_distances, prediction_distances,
              alpha=0.4, s=30, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Latent Distance (L2 norm)', fontsize=12, weight='bold')
    ax.set_ylabel('Prediction Distance (|p50_i - p50_j|)', fontsize=12, weight='bold')
    ax.set_title('Latent Distance vs Prediction Distance (Day 1)\n'
                 'For sequences with similar context endpoints (±1%)',
                fontsize=14, weight='bold', pad=15)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)

    # Add correlation line
    z = np.polyfit(latent_distances, prediction_distances, 1)
    p = np.poly1d(z)
    ax.plot(latent_distances, p(latent_distances), "r--", linewidth=2, alpha=0.8,
           label=f'Linear fit (r={correlation:.3f})')

    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

    # Interpretation box
    if abs(correlation) < 0.3:
        verdict = "❌ LOW CORRELATION: Latent space not meaningful"
        interpretation = "Similar latents → different predictions\nFragmented representation"
        box_color = 'lightcoral'
    elif abs(correlation) < 0.5:
        verdict = "⚠️  MODERATE CORRELATION: Weak structure"
        interpretation = "Some relationship, but noisy"
        box_color = 'lightyellow'
    else:
        verdict = "✅ HIGH CORRELATION: Meaningful latent space"
        interpretation = "Similar latents → similar predictions\nWell-structured representation"
        box_color = 'lightgreen'

    interpretation_text = (
        f"CORRELATION ANALYSIS:\n"
        f"{verdict}\n\n"
        f"Pearson r = {correlation:.3f}\n"
        f"p-value = {p_value:.2e}\n"
        f"N pairs = {len(latent_distances)}\n\n"
        f"{interpretation}"
    )

    props = dict(boxstyle='round,pad=1', facecolor=box_color, alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.98, 0.02, interpretation_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace')

    plt.tight_layout()

    # Save
    filepath = output_dir / 'latent_distance_vs_prediction_distance.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()

    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_pairs': len(latent_distances)
    }


# ============================================================================
# Analysis 3: Effective Dimensionality
# ============================================================================

def compute_effective_dimensionality(latent_z, output_dir):
    """Analyze effective dimensionality of latent space.

    Args:
        latent_z: (N, latent_dim)
        output_dir: Output directory
    """
    print("[PRIMARY] Computing effective dimensionality...")

    # Full PCA to get all eigenvalues
    pca = PCA()
    pca.fit(latent_z)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    latent_dim = latent_z.shape[1]

    # Compute effective dimensionality
    # Method 1: Number of components to explain 90% variance
    n_90 = np.argmax(cumulative_var >= 0.9) + 1

    # Method 2: Number of components above 1% individual variance
    n_above_1pct = np.sum(explained_var > 0.01)

    # Method 3: Participation ratio (inverse Simpson index)
    participation_ratio = 1.0 / np.sum(explained_var ** 2)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Components for 90% var: {n_90} ({n_90/latent_dim*100:.1f}%)")
    print(f"  Components >1% var: {n_above_1pct} ({n_above_1pct/latent_dim*100:.1f}%)")
    print(f"  Participation ratio: {participation_ratio:.1f} ({participation_ratio/latent_dim*100:.1f}%)")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Scree plot (eigenvalues)
    ax1.plot(range(1, len(explained_var)+1), explained_var,
            marker='o', linewidth=2, markersize=4)
    ax1.axhline(y=0.01, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='1% threshold')
    ax1.set_xlabel('Principal Component', fontsize=12, weight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12, weight='bold')
    ax1.set_title('Scree Plot: Individual Component Variance', fontsize=14, weight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_xlim(0, min(50, latent_dim))  # Show first 50 components

    # Plot 2: Cumulative variance
    ax2.plot(range(1, len(cumulative_var)+1), cumulative_var,
            marker='o', linewidth=2, markersize=4, color='green')
    ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='90% threshold')
    ax2.axvline(x=n_90, color='blue', linestyle=':', linewidth=2,
               alpha=0.7, label=f'N={n_90} for 90%')
    ax2.set_xlabel('Number of Components', fontsize=12, weight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, weight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=14, weight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.set_xlim(0, min(50, latent_dim))
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save
    filepath = output_dir / 'effective_dimensionality_analysis.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    plt.close()

    return {
        'latent_dim': latent_dim,
        'n_90pct': n_90,
        'n_above_1pct': n_above_1pct,
        'participation_ratio': participation_ratio,
        'explained_var': explained_var,
        'cumulative_var': cumulative_var
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_diagnostic_report(pca_results, correlation_results, dim_results, output_dir):
    """Generate comprehensive diagnostic report.

    Args:
        pca_results: dict from visualize_latent_space_pca
        correlation_results: dict from compute_latent_prediction_correlation
        dim_results: dict from compute_effective_dimensionality
        output_dir: Output directory
    """
    print("Generating diagnostic report...")

    report = []
    report.append("=" * 80)
    report.append("EXPERIMENT 2: LATENT SPACE ANALYSIS")
    report.append("=" * 80)
    report.append("")

    report.append("HYPOTHESIS:")
    report.append("  Under-regularized latent space (KL weight=1e-5) causes fragmented")
    report.append("  representations, leading to high epistemic uncertainty.")
    report.append("")

    report.append("METHOD:")
    report.append("  1. Extract latent embeddings z for all test sequences")
    report.append("  2. Visualize with PCA (colored by context endpoint)")
    report.append("  3. Compute latent distance vs prediction distance correlation")
    report.append("  4. Analyze effective dimensionality")
    report.append("")

    report.append("RESULTS:")
    report.append("")

    # PCA results
    report.append("1. PCA VISUALIZATION:")
    pca_var = pca_results['pca_explained_var']
    total_explained = pca_var[0] + pca_var[1]
    report.append(f"   PC1 variance: {pca_var[0]:.1%}")
    report.append(f"   PC2 variance: {pca_var[1]:.1%}")
    report.append(f"   Total (PC1+PC2): {total_explained:.1%}")
    report.append("")

    if total_explained < 0.3:
        report.append("   ❌ FRAGMENTED: First 2 PCs explain <30% variance")
        report.append("   → Latent space is highly fragmented")
    elif total_explained < 0.5:
        report.append("   ⚠️  MODERATE: First 2 PCs explain 30-50% variance")
    else:
        report.append("   ✅ STRUCTURED: First 2 PCs explain >50% variance")
    report.append("")

    # Correlation results
    if correlation_results:
        report.append("2. LATENT-PREDICTION CORRELATION:")
        corr = correlation_results['correlation']
        p_val = correlation_results['p_value']
        n_pairs = correlation_results['n_pairs']
        report.append(f"   Pearson r: {corr:.3f}")
        report.append(f"   p-value: {p_val:.2e}")
        report.append(f"   N pairs: {n_pairs}")
        report.append("")

        if abs(corr) < 0.3:
            report.append("   ❌ LOW CORRELATION: Latent space not meaningful")
            report.append("   → Similar contexts have different latent representations")
        elif abs(corr) < 0.5:
            report.append("   ⚠️  MODERATE CORRELATION: Weak structure")
        else:
            report.append("   ✅ HIGH CORRELATION: Meaningful latent space")
        report.append("")
    else:
        report.append("2. LATENT-PREDICTION CORRELATION:")
        report.append("   ⚠️  Could not compute (no similar pairs found)")
        report.append("")

    # Dimensionality results
    report.append("3. EFFECTIVE DIMENSIONALITY:")
    latent_dim = dim_results['latent_dim']
    n_90 = dim_results['n_90pct']
    n_above_1pct = dim_results['n_above_1pct']
    participation = dim_results['participation_ratio']

    report.append(f"   Latent dim: {latent_dim}")
    report.append(f"   Components for 90% var: {n_90} ({n_90/latent_dim*100:.1f}%)")
    report.append(f"   Components >1% var: {n_above_1pct} ({n_above_1pct/latent_dim*100:.1f}%)")
    report.append(f"   Participation ratio: {participation:.1f} ({participation/latent_dim*100:.1f}%)")
    report.append("")

    if n_90 > latent_dim * 0.5:
        report.append("   ❌ HIGH DIMENSIONALITY: Need >50% of dims for 90% variance")
        report.append("   → Latent space is under-constrained")
    elif n_90 > latent_dim * 0.3:
        report.append("   ⚠️  MODERATE: Need 30-50% of dims for 90% variance")
    else:
        report.append("   ✅ LOW DIMENSIONALITY: Need <30% of dims for 90% variance")
        report.append("   → Latent space is well-constrained")
    report.append("")

    # Overall verdict
    report.append("=" * 80)
    report.append("VERDICT:")
    report.append("")

    # Count evidence
    fragmented_count = 0
    if total_explained < 0.3:
        fragmented_count += 1
    if correlation_results and abs(correlation_results['correlation']) < 0.3:
        fragmented_count += 1
    if n_90 > latent_dim * 0.5:
        fragmented_count += 1

    if fragmented_count >= 2:
        report.append("  ❌ HYPOTHESIS CONFIRMED")
        report.append("  Latent space is FRAGMENTED and under-regularized")
        report.append(f"  {fragmented_count}/3 metrics indicate poor structure")
        report.append("")
        report.append("CONCLUSION:")
        report.append("  KL weight = 1e-5 is TOO LOW, allowing the latent space to become")
        report.append("  fragmented with many independent modes. This causes high epistemic")
        report.append("  uncertainty as similar contexts map to different latent representations.")
        report.append("")
        report.append("RECOMMENDATION:")
        report.append("  **INCREASE KL weight from 1e-5 to 1e-3 (100× increase)**")
        report.append("")
        report.append("EXPECTED IMPACT:")
        report.append("  - Tighter latent space with clear structure")
        report.append("  - Reduced epistemic uncertainty")
        report.append("  - Day-1 p50 spread should decrease significantly")
        report.append("  - Estimated reduction: 30-50% (0.0858 → ~0.04-0.06)")
    else:
        report.append("  ✅ HYPOTHESIS REJECTED")
        report.append("  Latent space appears reasonably structured")
        report.append(f"  Only {fragmented_count}/3 metrics indicate poor structure")
        report.append("")
        report.append("CONCLUSION:")
        report.append("  Under-regularized latent space is NOT the primary cause of")
        report.append("  day-1 over-dispersion.")
        report.append("")
        report.append("NEXT STEPS:")
        report.append("  **Investigate remaining causes:**")
        report.append("  - Training horizon mismatch (H=1 → H=90) → Experiment 5")
        report.append("  - Mean reversion bias")
        report.append("  - Consider ensemble approach or conformal calibration")

    report.append("")
    report.append("=" * 80)

    # Write report
    report_text = "\n".join(report)
    report_path = output_dir / 'latent_space_diagnostics.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  ✓ Saved: {report_path}")

    # Also print to console
    print("\n" + report_text)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run complete latent space analysis."""

    print("=" * 80)
    print("EXPERIMENT 2: LATENT SPACE ANALYSIS")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, model_config = load_model(MODEL_PATH)

    # Load data
    data = load_predictions_and_data()

    # Extract latent embeddings
    latent_data = extract_latent_embeddings(
        model, data['gt_surface'], data['indices'], CONTEXT_LEN
    )

    # Extract p50 predictions for correlation analysis
    grid_row, grid_col = ATM_6M
    p50_predictions = data['surfaces'][:, :, 1, grid_row, grid_col]  # (N, 90)

    # Run analyses
    print("\n" + "=" * 80)
    print("RUNNING ANALYSES")
    print("=" * 80 + "\n")

    pca_results = visualize_latent_space_pca(
        latent_data['latent_z'],
        latent_data['context_endpoints'],
        OUTPUT_DIR
    )

    correlation_results = compute_latent_prediction_correlation(
        latent_data['latent_z'],
        p50_predictions,
        latent_data['context_endpoints'],
        OUTPUT_DIR
    )

    dim_results = compute_effective_dimensionality(
        latent_data['latent_z'],
        OUTPUT_DIR
    )

    # Generate report
    generate_diagnostic_report(pca_results, correlation_results, dim_results, OUTPUT_DIR)

    print()
    print("=" * 80)
    print("EXPERIMENT 2 COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
