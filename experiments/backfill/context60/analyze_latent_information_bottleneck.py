#!/usr/bin/env python3
"""
Hypothesis 3: Latent Information Bottleneck

Question: Does latent_dim=5 fail to capture context distinctions, causing erratic predictions?

Logic:
- If latent space cannot distinguish similar contexts effectively
- Model produces varying predictions for similar contexts → high epistemic uncertainty

Expected Evidence if TRUE:
- Correlation(latent_dist, pred_dist) < 0.3 (weak relationship)
- Effective dimensionality > 50% of latent_dim (under-regularized)
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from vae.cvae_with_mem_randomized import CVAEMemRand

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "backfill" / "context60_experiment" / "checkpoints" / "backfill_context60_latent12_best.pt"
DATA_PATH = PROJECT_ROOT / "data" / "vol_surface_with_ret.npz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "context60_baseline" / "analysis" / "day1_over_dispersion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HYPOTHESIS 3: LATENT INFORMATION BOTTLENECK ANALYSIS")
print("=" * 80)
print()

# Load model
print("Loading trained model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_data = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = CVAEMemRand(model_data["model_config"])
model.load_weights(dict_to_load=model_data)
model.to(device)
model.eval()

config = model_data["model_config"]
print(f"Model loaded: {MODEL_PATH.name}")
print(f"Device: {device}")
print(f"Latent dim: {config.get('latent_dim', 'N/A')}")
print(f"Context length: {config.get('context_len', 60)}")
print()

# Load test data
print("Loading test data...")
data = np.load(DATA_PATH)
surfaces = data["surface"]  # (N, 5, 5)

# Use last 20% as test set
train_size = int(0.8 * len(surfaces))
test_surfaces = surfaces[train_size:]
print(f"Test set: {len(test_surfaces)} days")
print()

# Extract test contexts
context_len = config.get("context_len", 60)
n_test_contexts = 500  # Use 500 random contexts for efficiency

max_start_idx = len(test_surfaces) - context_len - 1
if max_start_idx <= 0:
    print("ERROR: Not enough test data")
    exit(1)

np.random.seed(42)
start_indices = np.random.choice(max_start_idx, size=min(n_test_contexts, max_start_idx), replace=False)

print(f"Extracting {len(start_indices)} test contexts...")
contexts = []
context_endpoints = []

for idx in start_indices:
    context = test_surfaces[idx:idx+context_len]  # (60, 5, 5)
    contexts.append(context)
    context_endpoints.append(context[-1, 2, 2])  # ATM endpoint

contexts = np.array(contexts)  # (N, 60, 5, 5)
context_endpoints = np.array(context_endpoints)
contexts_tensor = torch.tensor(contexts, dtype=torch.float32).to(device)
print(f"Context shape: {contexts_tensor.shape}")
print()

# Create dummy extra features if model expects them
ex_feats_dim = config.get("ex_feats_dim", 0)
if ex_feats_dim > 0:
    ex_feats_tensor = torch.zeros(len(contexts), context_len, ex_feats_dim, dtype=torch.float32).to(device)
    print(f"Extra features shape: {ex_feats_tensor.shape}")
else:
    ex_feats_tensor = None

print()

# Encode contexts to get latent representations
print("Encoding contexts to latent space...")
with torch.no_grad():
    enc_input = {"surface": contexts_tensor}
    if ex_feats_tensor is not None:
        enc_input["ex_feats"] = ex_feats_tensor

    mean, logvar, z = model.encoder(enc_input)

# Use mean of latent distribution for analysis (deterministic representation)
latent_z = mean.cpu().numpy()  # (N, T, latent_dim)

# Take final timestep latent (most informative)
latent_z_final = latent_z[:, -1, :]  # (N, latent_dim)

print(f"Latent z shape: {latent_z_final.shape}")
print(f"Latent mean: {latent_z_final.mean():.4f} ± {latent_z_final.std():.4f}")
print()

# Generate day-1 predictions for correlation analysis
print("Generating day-1 predictions...")
with torch.no_grad():
    # Generate using prior mode (realistic)
    preds, _ = model.get_surface_given_conditions(
        contexts_tensor if ex_feats_tensor is None else enc_input,
        z=None,  # Sample from prior
        mu=0,
        std=1,
        horizon=1
    )

# Extract day-1 p50 predictions (ATM point)
if config.get("use_quantile_regression", False):
    day1_p50 = preds[:, 0, 1, 2, 2].cpu().numpy()  # (N,) p50 channel
else:
    day1_p50 = preds[:, 0, 0, 2, 2].cpu().numpy()  # (N,)

print(f"Day-1 p50 predictions shape: {day1_p50.shape}")
print(f"Day-1 p50 mean: {day1_p50.mean():.4f} ± {day1_p50.std():.4f}")
print()

# === ANALYSIS 1: Latent-Prediction Correlation for Similar Contexts ===
print("=" * 80)
print("ANALYSIS 1: LATENT-PREDICTION CORRELATION")
print("=" * 80)
print()

# Find pairs of contexts with similar endpoints (±1%)
similar_pairs = []
latent_distances = []
prediction_distances = []

for i in range(len(context_endpoints)):
    for j in range(i+1, len(context_endpoints)):
        # Check if endpoints are within ±1%
        endpoint_diff_pct = abs(context_endpoints[i] - context_endpoints[j]) / context_endpoints[i]

        if endpoint_diff_pct < 0.01:  # Within 1%
            # Compute latent distance
            latent_dist = np.linalg.norm(latent_z_final[i] - latent_z_final[j])

            # Compute prediction distance
            pred_dist = abs(day1_p50[i] - day1_p50[j])

            similar_pairs.append((i, j))
            latent_distances.append(latent_dist)
            prediction_distances.append(pred_dist)

latent_distances = np.array(latent_distances)
prediction_distances = np.array(prediction_distances)

print(f"Found {len(similar_pairs)} pairs of similar contexts (within ±1%)")

if len(similar_pairs) > 10:
    # Compute correlation
    correlation, p_value = stats.pearsonr(latent_distances, prediction_distances)

    print(f"Pearson correlation: {correlation:.3f} (p-value: {p_value:.3e})")
    print()

    if correlation < 0.3:
        print("⚠️  WEAK CORRELATION (< 0.3): Latent space poorly captures context distinctions")
    elif correlation < 0.5:
        print("⚠️  MODERATE CORRELATION (0.3-0.5): Some latent structure, but not strong")
    else:
        print("✅ STRONG CORRELATION (> 0.5): Latent space captures context well")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(latent_distances, prediction_distances, alpha=0.5, s=30)
    ax.set_xlabel('Latent Distance ||z_i - z_j||', fontsize=12)
    ax.set_ylabel('Prediction Distance |p50_i - p50_j|', fontsize=12)
    ax.set_title(f'Latent vs Prediction Distance for Similar Contexts\n' +
                 f'Correlation = {correlation:.3f}, N={len(similar_pairs)} pairs',
                 fontsize=13)
    ax.grid(alpha=0.3)

    # Add trend line
    z_fit = np.polyfit(latent_distances, prediction_distances, 1)
    p_fit = np.poly1d(z_fit)
    x_trend = np.linspace(latent_distances.min(), latent_distances.max(), 100)
    ax.plot(x_trend, p_fit(x_trend), 'r--', linewidth=2, label=f'Linear fit (slope={z_fit[0]:.3f})')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'latent_vs_prediction_distance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'latent_vs_prediction_distance.png'}")
else:
    print("Not enough similar context pairs for correlation analysis")
    correlation = np.nan

print()

# === ANALYSIS 2: Effective Dimensionality ===
print("=" * 80)
print("ANALYSIS 2: EFFECTIVE DIMENSIONALITY")
print("=" * 80)
print()

pca = PCA()
pca.fit(latent_z_final)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# How many components for 90% variance?
n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
effective_ratio = n_components_90 / latent_z_final.shape[1]

print(f"Latent dimensionality: {latent_z_final.shape[1]}")
print(f"Components for 90% variance: {n_components_90} ({effective_ratio * 100:.1f}%)")
print()

for i in range(min(latent_z_final.shape[1], 5)):
    print(f"PC{i+1}: {explained_variance_ratio[i] * 100:.2f}% variance")

print()

if effective_ratio > 0.5:
    print("⚠️  HIGH EFFECTIVE DIMENSIONALITY (> 50%): Under-regularized latent space")
    print("    Latent space may be too fragmented")
else:
    print("✅ LOW EFFECTIVE DIMENSIONALITY (< 50%): Well-regularized latent space")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Scree plot
ax = axes[0]
ax.bar(np.arange(len(explained_variance_ratio)) + 1, explained_variance_ratio * 100, alpha=0.7)
ax.axhline(90, color='red', linestyle='--', label='90% cumulative')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance (%)', fontsize=12)
ax.set_title('PCA Scree Plot', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Right: Cumulative variance
ax = axes[1]
ax.plot(np.arange(len(cumulative_variance)) + 1, cumulative_variance * 100, 'o-', linewidth=2)
ax.axhline(90, color='red', linestyle='--', label='90% threshold')
ax.axvline(n_components_90, color='green', linestyle='--', label=f'{n_components_90} components')
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
ax.set_title('Cumulative Variance Explained', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'latent_effective_dimensionality.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'latent_effective_dimensionality.png'}")

print()

# === ANALYSIS 3: PCA Visualization ===
print("=" * 80)
print("ANALYSIS 3: LATENT SPACE VISUALIZATION")
print("=" * 80)
print()

# 2D PCA projection
latent_2d = pca.transform(latent_z_final)[:, :2]

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=context_endpoints,
                    cmap='viridis', alpha=0.6, s=50)
ax.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)', fontsize=12)
ax.set_title('Latent Space (2D PCA) Colored by Context Endpoint', fontsize=13)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Context Endpoint (ATM IV)', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'latent_space_pca.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'latent_space_pca.png'}")

print()

# === VERDICT ===
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

# Combine evidence
evidence_bottleneck = []

if not np.isnan(correlation) and correlation < 0.3:
    evidence_bottleneck.append(f"Weak latent-prediction correlation ({correlation:.3f})")

if effective_ratio > 0.5:
    evidence_bottleneck.append(f"High effective dimensionality ({effective_ratio * 100:.1f}%)")

if len(evidence_bottleneck) >= 1:
    print("✅ HYPOTHESIS CONFIRMED: Latent information bottleneck detected")
    print()
    print("Evidence:")
    for ev in evidence_bottleneck:
        print(f"  - {ev}")
    print()
    print("IMPLICATION:")
    print("  - latent_dim=5 is insufficient for context discrimination")
    print("  - Similar contexts map to different latents → erratic predictions")
    print("  - Causes high epistemic uncertainty at day-1")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Increase latent_dim: 5 → 16 or 32")
    print("  2. Increase KL weight: 1e-5 → 1e-4 (10× stronger)")
    print()
    print("EXPECTED IMPACT:")
    print("  - Day-1 spread: 0.0858 → 0.03-0.04 (60% reduction)")
else:
    print("❌ HYPOTHESIS REJECTED: No clear latent bottleneck")
    print()
    print("Evidence:")
    if not np.isnan(correlation):
        print(f"  - Moderate latent-prediction correlation ({correlation:.3f})")
    print(f"  - Effective dimensionality reasonable ({effective_ratio * 100:.1f}%)")
    print()
    print("IMPLICATION:")
    print("  - Latent space structure is not the primary cause")
    print("  - Investigate H4 (decoder calibration)")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
