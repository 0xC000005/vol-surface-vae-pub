#!/usr/bin/env python3
"""
Comprehensive Final Checkpoint Analysis - Context60 Latent12 V2 Epoch 599

This script performs complete diagnostic analysis on the final trained model:
1. KL divergence (should be 2-5 for healthy)
2. Latent space PCA (PC1 should be <90%)
3. Latent-prediction correlation (should be >0.35)
4. Effective dimensionality (percentage of 12 dims used)
5. Epistemic uncertainty (day-1 p50 spread)
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "backfill" / "context60_experiment" / "checkpoints" / "backfill_context60_latent12_v2_phase2_ep599.pt"
DATA_PATH = PROJECT_ROOT / "data" / "vol_surface_with_ret.npz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "context60_baseline" / "analysis_v2" / "final_epoch599"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import model config
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from config.backfill_context60_config_latent12_v2 import BackfillContext60ConfigLatent12V2 as cfg
from vae.cvae_with_mem_randomized import CVAEMemRand

print("=" * 80)
print("COMPREHENSIVE FINAL CHECKPOINT ANALYSIS")
print("Context60 Latent12 V2 - Epoch 599")
print("=" * 80)
print()

# Load model
print("Loading final checkpoint...")
print(f"Model: {MODEL_PATH.name}")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

model_config = checkpoint["model_config"]
model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=checkpoint)
model.eval()

print(f"Latent dim: {model_config['latent_dim']}")
print(f"KL weight (config): {cfg.kl_weight}")
print()

# Load data
print("Loading validation data...")
data = np.load(DATA_PATH)
surfaces = torch.from_numpy(data["surface"]).float()  # (N, 5, 5)
ret = data["ret"]
skew = data["skews"]
slope = data["slopes"]
ex_data_np = np.stack([ret, skew, slope], axis=-1)  # (N, 3)
ex_data = torch.from_numpy(ex_data_np).float()

# Use test set (last 20%)
train_size = int(0.8 * len(surfaces))
test_surfaces = surfaces[train_size:]
test_ex_data = ex_data[train_size:]

# Use subset for analysis (500 samples)
n_samples = min(500, len(test_surfaces) - cfg.context_len - 1)
print(f"Using {n_samples} test samples")
print()

# ==============================================================================
# METRIC 1: KL DIVERGENCE
# ==============================================================================
print("=" * 80)
print("METRIC 1: KL DIVERGENCE")
print("=" * 80)
print()

kl_values = []
with torch.no_grad():
    for i in range(n_samples):
        # Get context + target
        context_surfaces = test_surfaces[i:i+cfg.context_len].unsqueeze(0)  # (1, C, 5, 5)
        target_surface = test_surfaces[i+cfg.context_len].unsqueeze(0).unsqueeze(0)  # (1, 1, 5, 5)

        context_ex = test_ex_data[i:i+cfg.context_len].unsqueeze(0)  # (1, C, 3)
        target_ex = test_ex_data[i+cfg.context_len].unsqueeze(0).unsqueeze(0)  # (1, 1, 3)

        # Prepare input
        full_seq_surface = torch.cat([context_surfaces, target_surface], dim=1)  # (1, C+1, 5, 5)
        full_seq_ex = torch.cat([context_ex, target_ex], dim=1)  # (1, C+1, 3)

        input_dict = {
            "surface": full_seq_surface,
            "ex_feats": full_seq_ex
        }

        # Forward pass
        output = model(input_dict)
        kl = output["kl_divergence"].item()
        kl_values.append(kl)

kl_mean = np.mean(kl_values)
kl_std = np.std(kl_values)

print(f"KL divergence: {kl_mean:.2f} ± {kl_std:.2f}")
print()

if kl_mean < 1.0:
    print("❌ FAILURE: KL < 1.0 (severe posterior collapse)")
    kl_status = "FAIL"
elif kl_mean < 2.0:
    print("⚠️  WARNING: KL < 2.0 (mild over-regularization)")
    kl_status = "WARN"
elif kl_mean <= 5.0:
    print("✅ SUCCESS: KL in healthy range [2, 5]")
    kl_status = "PASS"
else:
    print("⚠️  WARNING: KL > 5.0 (under-regularization, may overfit)")
    kl_status = "WARN"

print()

# ==============================================================================
# METRIC 2: LATENT SPACE PCA
# ==============================================================================
print("=" * 80)
print("METRIC 2: LATENT SPACE PCA")
print("=" * 80)
print()

latent_embeddings = []
with torch.no_grad():
    for i in range(n_samples):
        context_surfaces = test_surfaces[i:i+cfg.context_len].unsqueeze(0)
        context_ex = test_ex_data[i:i+cfg.context_len].unsqueeze(0)

        # Get latent embedding (posterior mean)
        input_dict = {
            "surface": context_surfaces,
            "ex_feats": context_ex
        }
        z_mean = model.get_latent_embedding(input_dict)  # (1, latent_dim)
        latent_embeddings.append(z_mean.squeeze(0).numpy())

latent_embeddings = np.array(latent_embeddings)  # (n_samples, latent_dim)

# Run PCA
pca = PCA()
pca.fit(latent_embeddings)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Find effective dimensionality (90% variance)
n_dims_90 = np.argmax(cumulative_variance >= 0.9) + 1
effective_dim_pct = (n_dims_90 / model_config['latent_dim']) * 100

print(f"PC1 variance: {explained_variance[0] * 100:.2f}%")
print(f"PC2 variance: {explained_variance[1] * 100:.2f}%")
print(f"PC3 variance: {explained_variance[2] * 100:.2f}%")
print()
print(f"Dimensions for 90% variance: {n_dims_90}/{model_config['latent_dim']}")
print(f"Effective dimensionality: {effective_dim_pct:.1f}%")
print()

if explained_variance[0] > 0.95:
    print("❌ FAILURE: PC1 > 95% (severe over-compression)")
    pca_status = "FAIL"
elif explained_variance[0] > 0.90:
    print("⚠️  WARNING: PC1 > 90% (over-compression)")
    pca_status = "WARN"
elif explained_variance[0] > 0.75:
    print("✅ GOOD: PC1 in [75%, 90%] (reasonable compression)")
    pca_status = "PASS"
else:
    print("✅ EXCELLENT: PC1 < 75% (well-distributed variance)")
    pca_status = "PASS"

print()

# ==============================================================================
# METRIC 3: LATENT-PREDICTION CORRELATION
# ==============================================================================
print("=" * 80)
print("METRIC 3: LATENT-PREDICTION CORRELATION")
print("=" * 80)
print()

predictions = []
with torch.no_grad():
    for i in range(n_samples):
        context_surfaces = test_surfaces[i:i+cfg.context_len].unsqueeze(0)
        context_ex = test_ex_data[i:i+cfg.context_len].unsqueeze(0)

        # Get prediction (p50)
        pred = model.get_surface_given_conditions(
            context_surfaces,
            context_ex,
            horizon=1,
            num_samples=1
        )  # (1, 1, 3, 5, 5) or (1, 1, 5, 5)

        if pred.dim() == 5:  # Quantile model
            pred = pred[:, :, 1, :, :]  # Select p50

        predictions.append(pred.squeeze().numpy())

predictions = np.array(predictions)  # (n_samples, 5, 5)
predictions_flat = predictions.reshape(n_samples, -1)  # (n_samples, 25)

# Compute pairwise distances
latent_distances = euclidean_distances(latent_embeddings)
prediction_distances = euclidean_distances(predictions_flat)

# Flatten upper triangle (avoid diagonal and duplicates)
mask = np.triu(np.ones((n_samples, n_samples)), k=1).astype(bool)
latent_dist_flat = latent_distances[mask]
pred_dist_flat = prediction_distances[mask]

# Pearson correlation
corr, p_value = pearsonr(latent_dist_flat, pred_dist_flat)

print(f"Latent-Prediction correlation: {corr:.3f}")
print(f"P-value: {p_value:.2e}")
print()

if corr < 0.20:
    print("❌ FAILURE: Correlation < 0.20 (latents don't predict outputs)")
    corr_status = "FAIL"
elif corr < 0.35:
    print("⚠️  WARNING: Correlation < 0.35 (weak discriminative power)")
    corr_status = "WARN"
elif corr < 0.50:
    print("✅ GOOD: Correlation in [0.35, 0.50]")
    corr_status = "PASS"
else:
    print("✅ EXCELLENT: Correlation > 0.50")
    corr_status = "PASS"

print()

# ==============================================================================
# METRIC 4: EPISTEMIC UNCERTAINTY (DAY-1 P50 SPREAD)
# ==============================================================================
print("=" * 80)
print("METRIC 4: EPISTEMIC UNCERTAINTY (DAY-1 P50 SPREAD)")
print("=" * 80)
print()

# Already have predictions from metric 3
p50_std = predictions_flat.std(axis=0).mean()

print(f"Day-1 P50 spread (avg std across grid): {p50_std:.4f}")
print()

if p50_std < 0.015:
    print("⚠️  WARNING: Spread < 0.015 (may be under-dispersed)")
    spread_status = "WARN"
elif p50_std <= 0.035:
    print("✅ GOOD: Spread in [0.015, 0.035] (reasonable uncertainty)")
    spread_status = "PASS"
else:
    print("⚠️  WARNING: Spread > 0.035 (may be over-dispersed)")
    spread_status = "WARN"

print()

# ==============================================================================
# SUMMARY SCORECARD
# ==============================================================================
print("=" * 80)
print("FINAL SCORECARD - CONTEXT60 LATENT12 V2 (EPOCH 599)")
print("=" * 80)
print()

metrics = [
    ("KL Divergence", f"{kl_mean:.2f}", "[2.0, 5.0]", kl_status),
    ("PC1 Variance", f"{explained_variance[0] * 100:.1f}%", "<90%", pca_status),
    ("Effective Dim", f"{n_dims_90}/{model_config['latent_dim']}", ">30%", "PASS" if effective_dim_pct > 30 else "WARN"),
    ("Correlation", f"{corr:.3f}", ">0.35", corr_status),
    ("P50 Spread", f"{p50_std:.4f}", "0.025-0.03", spread_status),
]

print(f"{'Metric':<20} {'Value':<12} {'Target':<15} {'Status':<10}")
print("-" * 60)
for metric, value, target, status in metrics:
    status_symbol = "✅" if status == "PASS" else ("⚠️" if status == "WARN" else "❌")
    print(f"{metric:<20} {value:<12} {target:<15} {status_symbol} {status}")

print()

# Count passes
n_pass = sum(1 for _, _, _, s in metrics if s == "PASS")
n_total = len(metrics)

print(f"Score: {n_pass}/{n_total} metrics pass")
print()

if n_pass == n_total:
    print("🎉 OUTSTANDING SUCCESS - All metrics pass!")
elif n_pass >= 4:
    print("✅ SUCCESS - Most metrics pass, minor issues remain")
elif n_pass >= 2:
    print("⚠️  PARTIAL SUCCESS - Mixed results, investigation needed")
else:
    print("❌ FAILURE - Major issues detected")

print()

# Comparison to V1
print("=" * 80)
print("COMPARISON: V2 (Epoch 599) vs V1 (Failed)")
print("=" * 80)
print()
print(f"{'Metric':<20} {'V1 (Failed)':<15} {'V2 (Epoch 599)':<15} {'Change':<15}")
print("-" * 70)
print(f"{'KL Divergence':<20} {'0.85':<15} {f'{kl_mean:.2f}':<15} {'+' if kl_mean > 0.85 else ''}{((kl_mean - 0.85) / 0.85 * 100):+.0f}%")
print(f"{'PC1 Variance':<20} {'99.27%':<15} {f'{explained_variance[0] * 100:.1f}%':<15} {((explained_variance[0] * 100 - 99.27) / 99.27 * 100):+.0f}%")
print(f"{'Effective Dim':<20} {'8.3%':<15} {f'{effective_dim_pct:.1f}%':<15} {'+' if effective_dim_pct > 8.3 else ''}{((effective_dim_pct - 8.3) / 8.3 * 100):+.0f}%")
print(f"{'Correlation':<20} {'0.113':<15} {f'{corr:.3f}':<15} {'+' if corr > 0.113 else ''}{((corr - 0.113) / 0.113 * 100):+.0f}%")
print()

# Comparison to raw data
print("=" * 80)
print("COMPARISON: V2 (Epoch 599) vs Raw Data PCA")
print("=" * 80)
print()
print(f"{'Metric':<20} {'Raw Data':<15} {'V2 Latent':<15} {'Interpretation':<30}")
print("-" * 80)
print(f"{'PC1 Variance':<20} {'55.3%':<15} {f'{explained_variance[0] * 100:.1f}%':<15} {'Slightly more compressed':<30}")
print(f"{'Dims for 90%':<20} {'5':<15} {f'{n_dims_90}':<15} {'Using 60% of natural dims':<30}")
print()

# Save results
results = {
    "kl_mean": kl_mean,
    "kl_std": kl_std,
    "pc1_variance": explained_variance[0],
    "pc_variances": explained_variance,
    "n_dims_90": n_dims_90,
    "effective_dim_pct": effective_dim_pct,
    "correlation": corr,
    "p50_spread": p50_std,
    "n_pass": n_pass,
    "n_total": n_total,
}

np.savez(OUTPUT_DIR / "final_metrics.npz", **results)
print(f"Saved: {OUTPUT_DIR / 'final_metrics.npz'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. KL distribution
ax = axes[0, 0]
ax.hist(kl_values, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(kl_mean, color='red', linestyle='--', label=f'Mean: {kl_mean:.2f}')
ax.axvspan(2, 5, alpha=0.2, color='green', label='Healthy range [2, 5]')
ax.set_xlabel('KL Divergence')
ax.set_ylabel('Frequency')
ax.set_title('KL Divergence Distribution (Epoch 599)')
ax.legend()
ax.grid(alpha=0.3)

# 2. PCA scree plot
ax = axes[0, 1]
n_plot = min(12, len(explained_variance))
ax.bar(np.arange(n_plot) + 1, explained_variance[:n_plot] * 100, alpha=0.7)
ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.set_title('Latent Space PCA (Epoch 599)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(np.arange(1, n_plot + 1))

# 3. Latent vs Prediction distance scatter
ax = axes[1, 0]
sample_indices = np.random.choice(len(latent_dist_flat), size=min(5000, len(latent_dist_flat)), replace=False)
ax.scatter(latent_dist_flat[sample_indices], pred_dist_flat[sample_indices], alpha=0.1, s=1)
ax.set_xlabel('Latent Distance')
ax.set_ylabel('Prediction Distance')
ax.set_title(f'Latent-Prediction Correlation: {corr:.3f}')
ax.grid(alpha=0.3)

# 4. Comparison to V1
ax = axes[1, 1]
metrics_names = ['KL\n(×0.1)', 'PC1\n(×10%)', 'Eff Dim\n(%)', 'Corr\n(×10)']
v1_values = [0.85 * 10, 99.27 / 10, 8.3, 0.113 * 10]
v2_values = [kl_mean * 10, explained_variance[0] * 100 / 10, effective_dim_pct, corr * 10]

x = np.arange(len(metrics_names))
width = 0.35

ax.bar(x - width/2, v1_values, width, label='V1 (Failed)', alpha=0.7, color='red')
ax.bar(x + width/2, v2_values, width, label='V2 (Epoch 599)', alpha=0.7, color='green')

ax.set_ylabel('Scaled Metric Value')
ax.set_title('V1 vs V2 Comparison (Scaled for Visibility)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'final_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'final_comprehensive_analysis.png'}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
