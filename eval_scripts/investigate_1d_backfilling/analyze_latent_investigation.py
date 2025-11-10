"""
Analyze Latent Distributions to Diagnose Posterior Collapse.

Checks:
1. Latent statistics (mean, variance) vs N(0,1) prior
2. Correlation between latent dimensions and AMZN/MSFT/SP500 returns
3. Mutual information I(z; returns)
4. Difference between z[T-1] and z[T] distributions
5. Effective latent dimensionality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, entropy
from sklearn.feature_selection import mutual_info_regression
import os

# Configuration
INVESTIGATION_FILE = "models_1d_backfilling/latent_selection_investigation.npz"
DATA_FILE = "data/stock_returns_multifeature.npz"
OUTPUT_DIR = "models_1d_backfilling/latent_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("ANALYZING LATENT DISTRIBUTIONS")
print("=" * 80)
print(f"Investigation file: {INVESTIGATION_FILE}")
print(f"Output dir: {OUTPUT_DIR}")
print()

# Load data
print("Loading data...")
investigation = np.load(INVESTIGATION_FILE)
data = np.load(DATA_FILE)

actuals = investigation["actuals"]  # (N,) - AMZN returns
dates = investigation["dates"]

# Latent variables (N, latent_dim)
s1_z_mean = investigation["s1_z_mean"]
s1_z_logvar = investigation["s1_z_logvar"]
s3_z_mean = investigation["s3_z_mean"]
s3_z_logvar = investigation["s3_z_logvar"]
s4_z_mean = investigation["s4_z_mean"]
s4_z_logvar = investigation["s4_z_logvar"]

# Load MSFT and SP500 returns
all_features = data["all_features"]
test_start = 5000
ctx_len = 5
first_day = test_start + ctx_len

# Extract returns for prediction days
amzn_returns = all_features[first_day:first_day+len(actuals), 0]  # Channel 0
msft_returns = all_features[first_day:first_day+len(actuals), 4]  # Channel 4 (MSFT return)
sp500_returns = all_features[first_day:first_day+len(actuals), 8]  # Channel 8 (SP500 return)

print(f"  Loaded {len(actuals)} predictions")
print(f"  Latent dimension: {s1_z_mean.shape[1]}")
print()

# ============================================================================
# 1. Posterior Collapse Check: Compare to N(0, 1) Prior
# ============================================================================

print("=" * 80)
print("1. POSTERIOR COLLAPSE CHECK")
print("=" * 80)
print()

scenarios = [
    ("S1: Oracle (z[T] from real)", s1_z_mean, s1_z_logvar),
    ("S3: Realistic Original (z[T-1])", s3_z_mean, s3_z_logvar),
    ("S4: Realistic Fixed (z[T])", s4_z_mean, s4_z_logvar),
]

for name, z_mean, z_logvar in scenarios:
    # Compute statistics
    mean_of_means = np.mean(z_mean)  # Should be ~0 if posterior ≈ N(0,1)
    std_of_means = np.std(z_mean)    # Should be ~1 if posterior ≈ N(0,1)

    z_var = np.exp(z_logvar)
    mean_variance = np.mean(z_var)   # Should be ~1 if posterior ≈ N(0,1)

    # KL divergence from N(0,1): KL(q||p) = 0.5 * (μ² + σ² - log(σ²) - 1)
    kl_per_sample = 0.5 * np.sum(z_mean**2 + z_var - z_logvar - 1, axis=1)
    mean_kl = np.mean(kl_per_sample)

    print(f"{name}:")
    print(f"  Mean of z_mean: {mean_of_means:.4f} (target: ~0)")
    print(f"  Std of z_mean: {std_of_means:.4f} (target: ~1)")
    print(f"  Mean of z_var: {mean_variance:.4f} (target: ~1)")
    print(f"  Mean KL divergence: {mean_kl:.4f} (lower = closer to prior)")

    if mean_kl < 0.1:
        print(f"  ⚠ SEVERE POSTERIOR COLLAPSE (KL < 0.1)")
    elif mean_kl < 0.5:
        print(f"  ⚠ MODERATE POSTERIOR COLLAPSE (KL < 0.5)")
    else:
        print(f"  ✓ Posterior diverged from prior (KL > 0.5)")
    print()

# ============================================================================
# 2. Correlation Analysis: z vs Returns
# ============================================================================

print("=" * 80)
print("2. CORRELATION ANALYSIS: Latent vs Returns")
print("=" * 80)
print()

print("Analyzing S1 (Oracle) - Best case:")
for dim in range(min(3, s1_z_mean.shape[1])):  # Check first 3 dimensions
    corr_amzn, p_amzn = pearsonr(s1_z_mean[:, dim], amzn_returns)
    corr_msft, p_msft = pearsonr(s1_z_mean[:, dim], msft_returns)
    corr_sp500, p_sp500 = pearsonr(s1_z_mean[:, dim], sp500_returns)

    print(f"  z_dim_{dim}:")
    print(f"    AMZN:  r={corr_amzn:+.3f} (p={p_amzn:.4f})")
    print(f"    MSFT:  r={corr_msft:+.3f} (p={p_msft:.4f})")
    print(f"    SP500: r={corr_sp500:+.3f} (p={p_sp500:.4f})")

print()
print("Analyzing S4 (Realistic Fixed) - Should capture MSFT/SP500:")
for dim in range(min(3, s4_z_mean.shape[1])):
    corr_amzn, p_amzn = pearsonr(s4_z_mean[:, dim], amzn_returns)
    corr_msft, p_msft = pearsonr(s4_z_mean[:, dim], msft_returns)
    corr_sp500, p_sp500 = pearsonr(s4_z_mean[:, dim], sp500_returns)

    print(f"  z_dim_{dim}:")
    print(f"    AMZN:  r={corr_amzn:+.3f} (p={p_amzn:.4f})")
    print(f"    MSFT:  r={corr_msft:+.3f} (p={p_msft:.4f})")
    print(f"    SP500: r={corr_sp500:+.3f} (p={p_sp500:.4f})")

print()

# Max correlation across all dimensions
max_corr_s1_amzn = np.max([abs(pearsonr(s1_z_mean[:, i], amzn_returns)[0]) for i in range(s1_z_mean.shape[1])])
max_corr_s4_amzn = np.max([abs(pearsonr(s4_z_mean[:, i], amzn_returns)[0]) for i in range(s4_z_mean.shape[1])])
max_corr_s4_msft = np.max([abs(pearsonr(s4_z_mean[:, i], msft_returns)[0]) for i in range(s4_z_mean.shape[1])])
max_corr_s4_sp500 = np.max([abs(pearsonr(s4_z_mean[:, i], sp500_returns)[0]) for i in range(s4_z_mean.shape[1])])

print(f"Max |correlation| across all latent dimensions:")
print(f"  S1 (Oracle) → AMZN: {max_corr_s1_amzn:.3f}")
print(f"  S4 (Fixed) → AMZN: {max_corr_s4_amzn:.3f}")
print(f"  S4 (Fixed) → MSFT: {max_corr_s4_msft:.3f}")
print(f"  S4 (Fixed) → SP500: {max_corr_s4_sp500:.3f}")
print()

if max_corr_s1_amzn < 0.1:
    print("⚠ Even Oracle latent has minimal correlation with AMZN returns!")
    print("  → Posterior collapse confirmed")
print()

# ============================================================================
# 3. Mutual Information
# ============================================================================

print("=" * 80)
print("3. MUTUAL INFORMATION I(z; returns)")
print("=" * 80)
print()

# Compute MI for Oracle
mi_s1_amzn = mutual_info_regression(s1_z_mean, amzn_returns, random_state=42)
mi_s1_total = np.sum(mi_s1_amzn)

# Compute MI for Realistic Fixed
mi_s4_amzn = mutual_info_regression(s4_z_mean, amzn_returns, random_state=42)
mi_s4_msft = mutual_info_regression(s4_z_mean, msft_returns, random_state=42)
mi_s4_sp500 = mutual_info_regression(s4_z_mean, sp500_returns, random_state=42)
mi_s4_total = np.sum(mi_s4_amzn)

print(f"S1 (Oracle):")
print(f"  I(z; AMZN) = {mi_s1_total:.4f}")
print(f"  Top 3 dimensions: {sorted(mi_s1_amzn, reverse=True)[:3]}")
print()

print(f"S4 (Realistic Fixed):")
print(f"  I(z; AMZN) = {mi_s4_total:.4f}")
print(f"  I(z; MSFT) = {np.sum(mi_s4_msft):.4f}")
print(f"  I(z; SP500) = {np.sum(mi_s4_sp500):.4f}")
print()

if mi_s1_total < 0.01:
    print("⚠ Mutual information near zero → Latent is uninformative!")
    print("  → Severe posterior collapse")
print()

# ============================================================================
# 4. Difference Between z[T-1] and z[T]
# ============================================================================

print("=" * 80)
print("4. z[T-1] vs z[T] COMPARISON")
print("=" * 80)
print()

# Euclidean distance between S3 (z[T-1]) and S4 (z[T])
distances = np.linalg.norm(s3_z_mean - s4_z_mean, axis=1)
mean_distance = np.mean(distances)
std_distance = np.std(distances)

print(f"Euclidean distance between z[T-1] and z[T]:")
print(f"  Mean: {mean_distance:.4f}")
print(f"  Std: {std_distance:.4f}")
print()

if mean_distance < 0.1:
    print("⚠ z[T-1] ≈ z[T] → LSTM not encoding new information at timestep T")
    print("  → Even though MSFT[T+1] and SP500[T+1] are available, they're ignored!")
print()

# Correlation between z[T-1] and z[T]
correlations = [pearsonr(s3_z_mean[:, i], s4_z_mean[:, i])[0] for i in range(s3_z_mean.shape[1])]
mean_corr = np.mean(correlations)

print(f"Correlation between z[T-1] and z[T] (per dimension):")
print(f"  Mean: {mean_corr:.3f}")
print(f"  Min: {np.min(correlations):.3f}, Max: {np.max(correlations):.3f}")
print()

if mean_corr > 0.95:
    print("⚠ z[T-1] and z[T] are nearly identical (r > 0.95)")
    print("  → Latent is constant, not responding to new data")
print()

# ============================================================================
# 5. Effective Latent Dimensionality (PCA)
# ============================================================================

print("=" * 80)
print("5. EFFECTIVE LATENT DIMENSIONALITY")
print("=" * 80)
print()

from sklearn.decomposition import PCA

for name, z_mean in [("S1 (Oracle)", s1_z_mean), ("S4 (Fixed)", s4_z_mean)]:
    pca = PCA()
    pca.fit(z_mean)

    # Cumulative variance explained
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_dims_90 = np.argmax(cum_var >= 0.90) + 1
    n_dims_99 = np.argmax(cum_var >= 0.99) + 1

    print(f"{name}:")
    print(f"  Dimensions for 90% variance: {n_dims_90}/{z_mean.shape[1]}")
    print(f"  Dimensions for 99% variance: {n_dims_99}/{z_mean.shape[1]}")
    print(f"  Top 3 eigenvalues: {pca.explained_variance_[:3]}")
    print()

    if n_dims_90 <= 2:
        print(f"  ⚠ Only {n_dims_90} dimensions explain 90% variance")
        print(f"    → Latent has collapsed to low-dimensional manifold")
        print()

# ============================================================================
# 6. Visualizations
# ============================================================================

print("=" * 80)
print("6. GENERATING VISUALIZATIONS")
print("=" * 80)
print()

# Plot 1: Latent distributions vs N(0,1)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Latent Distributions vs N(0,1) Prior", fontsize=16)

for idx, (name, z_mean, z_logvar) in enumerate(scenarios):
    # Plot mean distribution
    ax = axes[0, idx]
    ax.hist(z_mean.flatten(), bins=50, density=True, alpha=0.7, label='Posterior')
    x = np.linspace(-3, 3, 100)
    ax.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), 'r-', linewidth=2, label='N(0,1) Prior')
    ax.set_title(f"{name}\nμ Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot variance distribution
    ax = axes[1, idx]
    z_var = np.exp(z_logvar)
    ax.hist(z_var.flatten(), bins=50, density=True, alpha=0.7, label='Posterior')
    ax.axvline(1.0, color='r', linewidth=2, label='Prior (σ²=1)')
    ax.set_title(f"σ² Distribution")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/latent_vs_prior.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/latent_vs_prior.png")

# Plot 2: Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Latent-Return Correlations", fontsize=16)

# S1 (Oracle)
corr_matrix_s1 = np.zeros((s1_z_mean.shape[1], 3))
for i in range(s1_z_mean.shape[1]):
    corr_matrix_s1[i, 0] = pearsonr(s1_z_mean[:, i], amzn_returns)[0]
    corr_matrix_s1[i, 1] = pearsonr(s1_z_mean[:, i], msft_returns)[0]
    corr_matrix_s1[i, 2] = pearsonr(s1_z_mean[:, i], sp500_returns)[0]

im1 = axes[0].imshow(corr_matrix_s1.T, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
axes[0].set_yticks([0, 1, 2])
axes[0].set_yticklabels(['AMZN', 'MSFT', 'SP500'])
axes[0].set_xlabel('Latent Dimension')
axes[0].set_title('S1: Oracle (z[T] from real)')
plt.colorbar(im1, ax=axes[0], label='Pearson r')

# S4 (Realistic Fixed)
corr_matrix_s4 = np.zeros((s4_z_mean.shape[1], 3))
for i in range(s4_z_mean.shape[1]):
    corr_matrix_s4[i, 0] = pearsonr(s4_z_mean[:, i], amzn_returns)[0]
    corr_matrix_s4[i, 1] = pearsonr(s4_z_mean[:, i], msft_returns)[0]
    corr_matrix_s4[i, 2] = pearsonr(s4_z_mean[:, i], sp500_returns)[0]

im2 = axes[1].imshow(corr_matrix_s4.T, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(['AMZN', 'MSFT', 'SP500'])
axes[1].set_xlabel('Latent Dimension')
axes[1].set_title('S4: Realistic Fixed (z[T] from masked)')
plt.colorbar(im2, ax=axes[1], label='Pearson r')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/correlation_heatmap.png")

# Plot 3: z[T-1] vs z[T] scatter
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("z[T-1] vs z[T]: First 6 Dimensions", fontsize=16)

for dim in range(min(6, s3_z_mean.shape[1])):
    ax = axes[dim // 3, dim % 3]
    ax.scatter(s3_z_mean[:, dim], s4_z_mean[:, dim], alpha=0.3, s=10)
    ax.plot([-3, 3], [-3, 3], 'r--', linewidth=1, label='y=x')

    corr = pearsonr(s3_z_mean[:, dim], s4_z_mean[:, dim])[0]
    ax.set_title(f"Dimension {dim} (r={corr:.3f})")
    ax.set_xlabel("z[T-1]")
    ax.set_ylabel("z[T]")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/z_comparison_scatter.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/z_comparison_scatter.png")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()

print("Key Findings Summary:")
print("-" * 80)
print(f"1. Posterior collapse severity: Check KL divergence above")
print(f"2. Max correlation (S1 → AMZN): {max_corr_s1_amzn:.3f}")
print(f"3. Max correlation (S4 → MSFT): {max_corr_s4_msft:.3f}")
print(f"4. Mean distance z[T-1] ↔ z[T]: {mean_distance:.4f}")
print(f"5. Correlation z[T-1] ↔ z[T]: {mean_corr:.3f}")
print()
print("Visualizations saved in:", OUTPUT_DIR)
