#!/usr/bin/env python3
"""
Hypothesis 4: Decoder Architecture (Time Embedding Issue)

Question: Does the decoder fail to differentiate short vs long horizons?

Method:
1. Load trained model
2. Extract decoder activations for same latent z at different horizons
3. Measure: ||decoder(z, h=1) - decoder(z, h=90)||
4. Check if decoder treats all horizons similarly

If small distance: Decoder collapses horizons → Add stronger time embeddings
If large distance: Decoder is horizon-aware, problem is upstream
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from vae.cvae_with_mem_randomized import CVAEMemRand

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "backfill" / "context60_experiment" / "checkpoints" / "backfill_context60_best.pt"
DATA_PATH = PROJECT_ROOT / "data" / "vol_surface_with_ret.npz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "context60_baseline" / "analysis" / "mean_reversion_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HYPOTHESIS 4: DECODER HORIZON SENSITIVITY ANALYSIS")
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

print(f"Model loaded: {MODEL_PATH.name}")
print(f"Device: {device}")
print()

# Model config
config = model_data["model_config"]
print("Model configuration:")
print(f"  Latent dim: {config.get('latent_dim', 'N/A')}")
print(f"  Context length: {config.get('context_len', 'N/A')}")
print(f"  Horizon: {config.get('horizon', 'N/A')}")
print(f"  Quantile regression: {config.get('quantile_regression', False)}")
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

# Extract a few test contexts
context_len = config.get("context_len", 60)
n_test_contexts = 50  # Use 50 random contexts

max_start_idx = len(test_surfaces) - context_len - 90
if max_start_idx <= 0:
    print("ERROR: Not enough test data")
    exit(1)

# Sample random starting indices
np.random.seed(42)
start_indices = np.random.choice(max_start_idx, size=min(n_test_contexts, max_start_idx), replace=False)

print(f"Extracting {len(start_indices)} test contexts...")
contexts = []
for idx in start_indices:
    context = test_surfaces[idx:idx+context_len]  # (60, 5, 5)
    contexts.append(context)

contexts = np.array(contexts)  # (N, 60, 5, 5)
contexts_tensor = torch.tensor(contexts, dtype=torch.float32).to(device)
print(f"Context shape: {contexts_tensor.shape}")

# Create dummy extra features if model expects them
ex_feats_dim = config.get("ex_feats_dim", 0)
if ex_feats_dim > 0:
    # Create zeros for extra features
    ex_feats_tensor = torch.zeros(len(contexts), context_len, ex_feats_dim, dtype=torch.float32).to(device)
    print(f"Extra features shape: {ex_feats_tensor.shape}")
else:
    ex_feats_tensor = None

print()

# Encode contexts to get latent representations
print("Encoding contexts to latent space...")
with torch.no_grad():
    # Use encoder to get latent z
    enc_input = {"surface": contexts_tensor}
    if ex_feats_tensor is not None:
        enc_input["ex_feats"] = ex_feats_tensor
    mean, logvar, z = model.encoder(enc_input)

print(f"Latent z shape: {z.shape}")
print(f"Latent mean: {mean.mean().item():.4f} ± {mean.std().item():.4f}")
print(f"Latent std: {torch.exp(0.5 * logvar).mean().item():.4f}")
print()

# Test decoder sensitivity across horizons
horizons_to_test = [1, 7, 14, 30, 60, 90]
print(f"Testing decoder at horizons: {horizons_to_test}")
print()

# Store decoder outputs for each horizon
decoder_outputs = {}

for h in horizons_to_test:
    print(f"Generating predictions at horizon={h}...")

    with torch.no_grad():
        # Generate surface predictions using decoder
        # Use get_surface_given_conditions method
        pred_input = contexts_tensor
        if ex_feats_tensor is not None:
            # For models with ex_feats, pass the full input dict
            pred_input = {"surface": contexts_tensor, "ex_feats": ex_feats_tensor}
            preds, _ = model.get_surface_given_conditions(
                pred_input,
                z=z,  # Use same latent for all horizons
                mu=0,
                std=1,
                horizon=h
            )
        else:
            preds, _ = model.get_surface_given_conditions(
                contexts_tensor,
                z=z,  # Use same latent for all horizons
                mu=0,
                std=1,
                horizon=h
            )

    # preds shape: (N, h, C, 5, 5) where C=3 for quantile, C=1 for standard
    # We want the final prediction at horizon h
    final_pred = preds[:, -1, :, :, :]  # (N, C, 5, 5)

    decoder_outputs[h] = final_pred.cpu().numpy()
    print(f"  Output shape: {final_pred.shape}")

print()

# Compute pairwise distances between decoder outputs
print("=" * 80)
print("DECODER OUTPUT DISTANCES")
print("=" * 80)
print()

# For each horizon pair, compute L2 distance
distance_matrix = np.zeros((len(horizons_to_test), len(horizons_to_test)))

for i, h1 in enumerate(horizons_to_test):
    for j, h2 in enumerate(horizons_to_test):
        out1 = decoder_outputs[h1]  # (N, C, 5, 5)
        out2 = decoder_outputs[h2]  # (N, C, 5, 5)

        # L2 distance per sample, averaged across samples
        distances = np.linalg.norm(out1 - out2, axis=(1, 2, 3))  # (N,)
        mean_distance = distances.mean()

        distance_matrix[i, j] = mean_distance

# Print distance matrix
print("Distance matrix (L2 norm):")
print()
print(f"{'':>8}", end="")
for h in horizons_to_test:
    print(f"{f'H={h}':>10}", end="")
print()

for i, h1 in enumerate(horizons_to_test):
    print(f"{f'H={h1}':>8}", end="")
    for j, h2 in enumerate(horizons_to_test):
        print(f"{distance_matrix[i, j]:>10.4f}", end="")
    print()

print()

# Analyze key distances
dist_1_90 = distance_matrix[horizons_to_test.index(1), horizons_to_test.index(90)]
dist_1_7 = distance_matrix[horizons_to_test.index(1), horizons_to_test.index(7)]
dist_30_90 = distance_matrix[horizons_to_test.index(30), horizons_to_test.index(90)]

print("Key distances:")
print(f"  H=1 vs H=90: {dist_1_90:.4f}")
print(f"  H=1 vs H=7:  {dist_1_7:.4f}")
print(f"  H=30 vs H=90: {dist_30_90:.4f}")
print()

# Compute relative sensitivity
# Sensitivity = distance(H=1, H=90) / distance(H=1, H=7)
# High sensitivity: decoder strongly differentiates long vs short horizons
# Low sensitivity: decoder treats horizons similarly
sensitivity_ratio = dist_1_90 / dist_1_7 if dist_1_7 > 0 else 0

print(f"Sensitivity ratio (H=1 vs H=90) / (H=1 vs H=7): {sensitivity_ratio:.3f}")
print()

# Visualization 1: Distance heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(distance_matrix, cmap='RdYlGn_r', aspect='auto')

ax.set_xticks(np.arange(len(horizons_to_test)))
ax.set_yticks(np.arange(len(horizons_to_test)))
ax.set_xticklabels([f'H={h}' for h in horizons_to_test])
ax.set_yticklabels([f'H={h}' for h in horizons_to_test])

# Add text annotations
for i in range(len(horizons_to_test)):
    for j in range(len(horizons_to_test)):
        text = ax.text(j, i, f'{distance_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Decoder Output Distance Matrix\n(L2 norm between predictions at different horizons)',
             fontsize=13)
fig.colorbar(im, ax=ax, label='L2 Distance')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'decoder_horizon_distance_matrix.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'decoder_horizon_distance_matrix.png'}")

# Visualization 2: Distance growth with horizon gap
fig, ax = plt.subplots(figsize=(10, 6))

# For H=1 as reference, plot distance vs horizon gap
reference_h = 1
ref_idx = horizons_to_test.index(reference_h)
distances_from_ref = distance_matrix[ref_idx, :]

ax.plot(horizons_to_test, distances_from_ref, 'o-', linewidth=2, markersize=8, color='blue')
ax.set_xlabel('Horizon (days)', fontsize=12)
ax.set_ylabel(f'L2 Distance from H={reference_h}', fontsize=12)
ax.set_title(f'Decoder Sensitivity to Horizon\n(Distance from H={reference_h} predictions)',
             fontsize=13)
ax.grid(alpha=0.3)

# Add value labels
for h, dist in zip(horizons_to_test, distances_from_ref):
    ax.text(h, dist + 0.002, f'{dist:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'decoder_horizon_sensitivity_growth.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'decoder_horizon_sensitivity_growth.png'}")

# Visualization 3: Actual predictions at different horizons (ATM point)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, h in enumerate(horizons_to_test):
    ax = axes[i]

    # Extract ATM predictions (center point 2, 2)
    if config.get('quantile_regression', False):
        atm_preds = decoder_outputs[h][:, :, 2, 2]  # (N, 3) for p05, p50, p95
        p50_preds = atm_preds[:, 1]  # Median
    else:
        atm_preds = decoder_outputs[h][:, 0, 2, 2]  # (N,)
        p50_preds = atm_preds

    # Histogram of predictions
    ax.hist(p50_preds, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(p50_preds.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {p50_preds.mean():.4f}')
    ax.axvline(p50_preds.std(), color='orange', linestyle=':', linewidth=1.5,
               label=f'Std: {p50_preds.std():.4f}')

    ax.set_xlabel('Predicted ATM IV', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'H={h} (same latent z)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'decoder_predictions_by_horizon.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'decoder_predictions_by_horizon.png'}")

print()

# Verdict
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

# Threshold: If distance(H=1, H=90) < 0.02, decoder is horizon-blind
horizon_blind_threshold = 0.02

if dist_1_90 < horizon_blind_threshold:
    print("✅ HYPOTHESIS CONFIRMED: Decoder is horizon-blind")
    print(f"   Distance(H=1, H=90) = {dist_1_90:.4f} < {horizon_blind_threshold}")
    print()
    print("IMPLICATION:")
    print("  - Decoder produces similar outputs regardless of horizon")
    print("  - Time embeddings are too weak or not utilized")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Add stronger time embeddings (sinusoidal or learnable)")
    print("  2. Use horizon-conditional decoder (separate paths per horizon range)")
    print("  3. Add explicit horizon information to decoder input")
    print("  4. Consider progressive decoding (sequential, not parallel)")
elif sensitivity_ratio < 2.0:
    print("⚠️  HYPOTHESIS PARTIALLY CONFIRMED: Decoder has weak horizon sensitivity")
    print(f"   Sensitivity ratio = {sensitivity_ratio:.3f} < 2.0")
    print(f"   Distance(H=1, H=90) = {dist_1_90:.4f}")
    print()
    print("IMPLICATION:")
    print("  - Decoder differentiates horizons, but weakly")
    print("  - Time information may not be fully utilized")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Strengthen time embeddings")
    print("  2. Add horizon-specific decoder layers")
    print("  3. Consider increasing model capacity for temporal modeling")
else:
    print("❌ HYPOTHESIS REJECTED: Decoder is horizon-aware")
    print(f"   Sensitivity ratio = {sensitivity_ratio:.3f} >= 2.0")
    print(f"   Distance(H=1, H=90) = {dist_1_90:.4f}")
    print()
    print("IMPLICATION:")
    print("  - Decoder properly differentiates short vs long horizons")
    print("  - Problem is NOT with decoder architecture")
    print()
    print("RECOMMENDED NEXT STEPS:")
    print("  1. Investigate latent information bottleneck (H3)")
    print("  2. Check if encoder loses context information over time")
    print("  3. Investigate prediction target representation (H5)")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
