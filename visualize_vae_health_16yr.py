"""
Comprehensive VAE Health Visualizations for backfill_16yr Model

Creates visualizations for:
1. Posterior collapse (KL divergence heatmaps, bar charts)
2. Latent space structure (PCA, variance analysis)
3. Temporal dynamics (crisis vs normal, time series)
4. Reconstruction quality correlation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("VAE HEALTH VISUALIZATIONS - backfill_16yr")
print("=" * 80)
print()

# Create output directory
output_dir = Path("models_backfill/vae_health_figs")
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading VAE health metrics...")
data = np.load("models_backfill/vae_health_16yr.npz", allow_pickle=True)

latent_dim = int(data['latent_dim'])
context_len = int(data['context_len'])
horizons = data['horizons']
crisis_start = int(data['crisis_start'])
crisis_end = int(data['crisis_end'])

print(f"  Latent dim: {latent_dim}")
print(f"  Context length: {context_len}")
print(f"  Horizons: {horizons}")
print()

# ============================================================================
# 1. POSTERIOR COLLAPSE VISUALIZATIONS
# ============================================================================

print("1. Creating posterior collapse visualizations...")

# Extract KL divergence data
kl_data = np.zeros((len(horizons), latent_dim))
for i, h in enumerate(horizons):
    kl_data[i] = data[f'h{h}_mean_kl_per_dim']

# 1.1 Heatmap: KL divergence per dimension per horizon
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(kl_data.T, annot=True, fmt='.4f', cmap='RdYlGn',
            xticklabels=[f'H{h}' for h in horizons],
            yticklabels=[f'Dim {d}' for d in range(latent_dim)],
            ax=ax, cbar_kws={'label': 'KL Divergence'})
ax.axhline(y=0, color='k', linewidth=2)
ax.axhline(y=latent_dim, color='k', linewidth=2)
ax.set_xlabel('Horizon')
ax.set_ylabel('Latent Dimension')
ax.set_title('Per-Dimension KL Divergence Across Horizons\n(Green = Higher KL = Active, Red = Lower KL = Collapsed)')

# Add collapse threshold line
collapse_threshold = 0.01
ax.text(len(horizons) + 0.1, latent_dim/2, f'Threshold\n{collapse_threshold}',
        rotation=-90, va='center', fontsize=9, color='red')

plt.tight_layout()
plt.savefig(output_dir / 'kl_divergence_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved kl_divergence_heatmap.png")

# 1.2 Bar chart: Collapsed vs Active dimensions
collapsed_counts = []
for h in horizons:
    collapsed_dims = data[f'h{h}_collapsed_dims']
    collapsed_counts.append(np.sum(collapsed_dims))

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(horizons))
ax.bar(x, collapsed_counts, color='red', alpha=0.7, label='Collapsed')
ax.bar(x, [latent_dim - c for c in collapsed_counts], bottom=collapsed_counts,
       color='green', alpha=0.7, label='Active')
ax.set_xticks(x)
ax.set_xticklabels([f'H{h}' for h in horizons])
ax.set_xlabel('Horizon')
ax.set_ylabel('Number of Dimensions')
ax.set_title(f'Active vs Collapsed Dimensions (Threshold: KL < {collapse_threshold})')
ax.legend()
ax.set_ylim([0, latent_dim + 0.5])

# Add count labels
for i, (active, collapsed) in enumerate(zip([latent_dim - c for c in collapsed_counts], collapsed_counts)):
    if collapsed > 0:
        ax.text(i, collapsed/2, str(collapsed), ha='center', va='center', fontweight='bold')
    if active > 0:
        ax.text(i, collapsed + active/2, str(active), ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'collapsed_dimensions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved collapsed_dimensions.png")

# 1.3 Context vs Future KL comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, h in enumerate(horizons):
    ax = axes[i]
    context_kl = data[f'h{h}_context_kl']
    future_kl = data[f'h{h}_future_kl']

    x = np.arange(latent_dim)
    width = 0.35

    ax.bar(x - width/2, context_kl, width, label='Context', color='blue', alpha=0.7)
    ax.bar(x + width/2, future_kl, width, label='Future', color='orange', alpha=0.7)
    ax.axhline(y=collapse_threshold, color='red', linestyle='--', label='Collapse threshold')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('KL Divergence')
    ax.set_title(f'Horizon {h}: Context vs Future KL')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'context_vs_future_kl.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved context_vs_future_kl.png")

# ============================================================================
# 2. LATENT SPACE STRUCTURE VISUALIZATIONS
# ============================================================================

print("2. Creating latent space structure visualizations...")

# 2.1 Per-dimension variance
variance_data = np.zeros((len(horizons), latent_dim))
for i, h in enumerate(horizons):
    variance_data[i] = data[f'h{h}_latent_variance']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(latent_dim)
width = 0.2

for i, h in enumerate(horizons):
    ax.bar(x + i*width, variance_data[i], width, label=f'H{h}', alpha=0.7)

utilization_threshold = 0.1
ax.axhline(y=utilization_threshold, color='red', linestyle='--',
           label=f'Active threshold ({utilization_threshold})')
ax.set_xlabel('Latent Dimension')
ax.set_ylabel('Variance')
ax.set_title('Per-Dimension Latent Variance Across Horizons')
ax.set_xticks(x + width * (len(horizons) - 1) / 2)
ax.set_xticklabels([f'Dim {d}' for d in range(latent_dim)])
ax.legend()
ax.set_yscale('log')  # Log scale to see small values
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'latent_variance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved latent_variance.png")

# 2.2 Effective dimensionality
effective_dims = [float(data[f'h{h}_effective_dim']) for h in horizons]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(horizons, effective_dims, marker='o', linewidth=2, markersize=8, color='blue')
ax.axhline(y=latent_dim, color='red', linestyle='--', label=f'Max possible ({latent_dim})')
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('Effective Dimensionality')
ax.set_title('Effective Dimensionality vs Horizon\n(Participation Ratio = (Σ var)² / Σ(var²))')
ax.grid(alpha=0.3)
ax.legend()
ax.set_ylim([0, latent_dim + 0.5])

# Add value labels
for h, eff_dim in zip(horizons, effective_dims):
    ax.text(h, eff_dim + 0.1, f'{eff_dim:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'effective_dimensionality.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved effective_dimensionality.png")

# 2.3 PCA visualization (for H30 - most latent variation)
h30_z_means = data['h30_z_means']  # (N, T, latent_dim)
h30_regime_labels = data['h30_regime_labels']  # (N,)

# Flatten across time dimension for PCA
N, T, D = h30_z_means.shape
z_flat = h30_z_means.reshape(N, T * D)  # (N, T*D)

# Perform PCA to 2D
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_flat)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
crisis_mask = h30_regime_labels == 1
normal_mask = h30_regime_labels == 0

ax.scatter(z_pca[normal_mask, 0], z_pca[normal_mask, 1],
           c='blue', alpha=0.3, s=20, label='Normal')
ax.scatter(z_pca[crisis_mask, 0], z_pca[crisis_mask, 1],
           c='red', alpha=0.5, s=20, label='Crisis (2008-2010)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('PCA of Latent Representations (H30)\nColored by Market Regime')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pca_regime_separation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved pca_regime_separation.png")

# ============================================================================
# 3. TEMPORAL DYNAMICS VISUALIZATIONS
# ============================================================================

print("3. Creating temporal dynamics visualizations...")

# 3.1 Crisis vs Normal mean latent
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, h in enumerate(horizons):
    ax = axes[i]
    crisis_mean = data[f'h{h}_crisis_mean']
    normal_mean = data[f'h{h}_normal_mean']

    x = np.arange(latent_dim)
    width = 0.35

    ax.bar(x - width/2, crisis_mean, width, label='Crisis', color='red', alpha=0.7)
    ax.bar(x + width/2, normal_mean, width, label='Normal', color='blue', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean Value')
    ax.set_title(f'Horizon {h}: Mean Latent per Regime')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'regime_mean_latent.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved regime_mean_latent.png")

# 3.2 Centroid distance vs horizon
centroid_distances = [float(data[f'h{h}_centroid_distance']) for h in horizons]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(horizons, centroid_distances, marker='o', linewidth=2, markersize=8, color='purple')
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('L2 Distance')
ax.set_title('Crisis vs Normal Regime Separation\n(L2 Distance Between Centroid Means)')
ax.grid(alpha=0.3)

# Add value labels
for h, dist in zip(horizons, centroid_distances):
    ax.text(h, dist + 0.002, f'{dist:.4f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'regime_separation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved regime_separation.png")

# 3.3 Time series of latent means (H30 only)
h30_z_means = data['h30_z_means']  # (N, T, latent_dim)
h30_date_indices = data['h30_date_indices']  # (N,)

# Compute mean across time dimension for each sample
latent_means_over_time = np.mean(h30_z_means, axis=1)  # (N, latent_dim)

fig, axes = plt.subplots(latent_dim, 1, figsize=(14, 10), sharex=True)

for d in range(latent_dim):
    ax = axes[d]
    ax.plot(h30_date_indices, latent_means_over_time[:, d], linewidth=0.5, alpha=0.7)
    ax.axvspan(crisis_start, crisis_end, alpha=0.2, color='red', label='Crisis' if d == 0 else '')
    ax.set_ylabel(f'Dim {d}')
    ax.grid(alpha=0.3)
    if d == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Date Index')
axes[0].set_title('Temporal Evolution of Mean Latent Dimensions (H30)')

plt.tight_layout()
plt.savefig(output_dir / 'temporal_latent_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved temporal_latent_evolution.png")

# ============================================================================
# 4. RECONSTRUCTION QUALITY CORRELATION VISUALIZATIONS
# ============================================================================

print("4. Creating reconstruction quality visualizations...")

# 4.1 Correlation heatmap
corr_data = np.zeros((len(horizons), 3))
for i, h in enumerate(horizons):
    corr_data[i, 0] = data[f'h{h}_corr_l2']
    corr_data[i, 1] = data[f'h{h}_corr_var']
    corr_data[i, 2] = data[f'h{h}_corr_kl']

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_data.T, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=[f'H{h}' for h in horizons],
            yticklabels=['L2 Norm', 'Variance', 'KL Divergence'],
            ax=ax, cbar_kws={'label': 'Correlation'}, vmin=-0.3, vmax=0.3)
ax.set_xlabel('Horizon')
ax.set_ylabel('Latent Statistic')
ax.set_title('Correlation Between Latent Statistics and RMSE\n(Positive = Higher latent stat → Higher RMSE)')

plt.tight_layout()
plt.savefig(output_dir / 'rmse_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved rmse_correlation_heatmap.png")

# 4.2 Scatter plot: Latent L2 norm vs RMSE (for H30)
h30_z_means = data['h30_z_means']
h30_rmse = data['h30_rmse_per_sample']
h30_regime_labels = data['h30_regime_labels']

# Compute future timestep L2 norm
future_z_mean = h30_z_means[:, context_len:, :]  # (N, horizon, latent_dim)
latent_l2_norm = np.linalg.norm(future_z_mean, axis=(1, 2))  # (N,)

fig, ax = plt.subplots(figsize=(10, 6))
crisis_mask = h30_regime_labels == 1
normal_mask = h30_regime_labels == 0

ax.scatter(latent_l2_norm[normal_mask], h30_rmse[normal_mask],
           c='blue', alpha=0.3, s=10, label='Normal')
ax.scatter(latent_l2_norm[crisis_mask], h30_rmse[crisis_mask],
           c='red', alpha=0.5, s=10, label='Crisis')
ax.set_xlabel('Latent L2 Norm (Future Timesteps)')
ax.set_ylabel('RMSE')
ax.set_title(f'Latent L2 Norm vs RMSE (H30)\nCorrelation: {data["h30_corr_l2"]:.3f}')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rmse_vs_latent_l2.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved rmse_vs_latent_l2.png")

# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"All figures saved to: {output_dir}")
print()
print("Generated visualizations:")
print("  1. Posterior Collapse:")
print("     - kl_divergence_heatmap.png")
print("     - collapsed_dimensions.png")
print("     - context_vs_future_kl.png")
print()
print("  2. Latent Space Structure:")
print("     - latent_variance.png")
print("     - effective_dimensionality.png")
print("     - pca_regime_separation.png")
print()
print("  3. Temporal Dynamics:")
print("     - regime_mean_latent.png")
print("     - regime_separation.png")
print("     - temporal_latent_evolution.png")
print()
print("  4. Reconstruction Quality:")
print("     - rmse_correlation_heatmap.png")
print("     - rmse_vs_latent_l2.png")
print()
