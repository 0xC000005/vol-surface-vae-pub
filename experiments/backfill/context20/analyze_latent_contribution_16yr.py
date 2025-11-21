"""
R² Decomposition Analysis - Latent Contribution to Predictions

TESTS: How much does latent variable contribute to prediction quality?

Method:
1. Extract context embeddings and latent variables for test set
2. Compute RMSE for each prediction
3. Use linear regression to predict RMSE from:
   - Model A: Context embeddings only
   - Model B: Context embeddings + latent variables
4. Compute: Latent contribution = R²_B - R²_A

Expected if latent IS used:
  - R²_B > R²_A (adding latent improves RMSE prediction)
  - Latent contribution > 0.05 (at least 5% additional variance explained)

Expected if latent COLLAPSED:
  - R²_B ≈ R²_A (latent adds nothing)
  - Latent contribution ≈ 0
"""
import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set seeds
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("R² DECOMPOSITION ANALYSIS - backfill_16yr")
print("=" * 80)
print()
print("Objective: Quantify latent variable contribution to prediction quality")
print()
print("Method:")
print("  1. Extract context embeddings + latent variables")
print("  2. Compute RMSE for each prediction")
print("  3. Linear regression: RMSE ~ context_embeddings")
print("  4. Linear regression: RMSE ~ context_embeddings + latent")
print("  5. Compare R² values")
print()
print("Expected if latent IS used:")
print("  ✓ R² increases when adding latent variables")
print("  ✓ Latent contribution (ΔR²) > 5%")
print()

# ============================================================================
# Load Model
# ============================================================================

print("Loading model...")
model_file = "models/backfill/backfill_16yr.pt"
model_data = torch.load(model_file, weights_only=False)
model_config = model_data["model_config"]

latent_dim = model_config['latent_dim']
context_len = model_config['context_len']

print(f"Model config:")
print(f"  Context length: {context_len}")
print(f"  Latent dim: {latent_dim}")
print(f"  Memory hidden: {model_config['mem_hidden']}")

model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
device = model.device
print("✓ Model loaded")

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]

ex_data = np.concatenate([
    ret_data[..., np.newaxis],
    skew_data[..., np.newaxis],
    slope_data[..., np.newaxis]
], axis=-1)

print(f"  Data shape: {vol_surf_data.shape}")

# ============================================================================
# Load Existing Predictions
# ============================================================================

print("\nLoading VAE prior predictions...")
vae_prior_file = "models/backfill/vae_prior_insample_16yr.npz"
vae_prior_data = np.load(vae_prior_file)
print("✓ VAE Prior predictions loaded")

# ============================================================================
# Configuration
# ============================================================================

train_start = 1000
train_end = 5000
horizons = [1, 7, 14, 30]
num_samples_per_horizon = 1000  # Subsample for computational efficiency

print(f"\nTest Configuration:")
print(f"  Test period: indices [{train_start}, {train_end}]")
print(f"  Samples per horizon: {num_samples_per_horizon}")
print(f"  Horizons: {horizons}")
print()

# ============================================================================
# Extract Features and Compute RMSE
# ============================================================================

print("=" * 80)
print("EXTRACTING FEATURES AND COMPUTING RMSE")
print("=" * 80)
print()

results_by_horizon = {}

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    # Get predictions and indices
    predictions = vae_prior_data[f'recon_h{horizon}']  # (N, 3, 5, 5)
    indices = vae_prior_data[f'indices_h{horizon}']

    # Subsample
    np.random.seed(42)
    if len(indices) > num_samples_per_horizon:
        sample_idx = np.random.choice(len(indices), num_samples_per_horizon, replace=False)
        sample_idx = np.sort(sample_idx)
        predictions = predictions[sample_idx]
        indices = indices[sample_idx]

    num_samples = len(indices)
    print(f"  Using {num_samples} samples")

    # Ground truth
    ground_truth = vol_surf_data[indices]  # (N, 5, 5)

    # Compute RMSE for each prediction (using median quantile)
    median_pred = predictions[:, 1, :, :]  # (N, 5, 5)
    rmse_per_sample = np.sqrt(np.mean((median_pred - ground_truth) ** 2, axis=(1, 2)))  # (N,)

    print(f"  RMSE range: [{rmse_per_sample.min():.6f}, {rmse_per_sample.max():.6f}]")
    print(f"  Mean RMSE: {rmse_per_sample.mean():.6f}")

    # Extract features for these samples
    print(f"  Extracting context embeddings and latents...")

    context_embeddings = []
    latent_means = []

    with torch.no_grad():
        for i, day_idx in enumerate(tqdm(indices, desc=f"    H{horizon}")):
            # Extract context
            context_surface = vol_surf_data[day_idx - context_len : day_idx]
            context_ex = ex_data[day_idx - context_len : day_idx]

            ctx_input = {
                "surface": torch.from_numpy(context_surface).unsqueeze(0).to(device),
                "ex_feats": torch.from_numpy(context_ex).unsqueeze(0).to(device)
            }

            # Get context embedding (LSTM encoding)
            ctx_embedding = model.ctx_encoder(ctx_input)  # (1, C, mem_hidden)

            # Get latent encoding
            latent_mean, _, _ = model.encoder(ctx_input)  # (1, C, latent_dim)

            # Take LAST timestep embedding (represents full context)
            final_ctx_embedding = ctx_embedding[0, -1, :].cpu().numpy()  # (mem_hidden,)
            final_latent = latent_mean[0, -1, :].cpu().numpy()  # (latent_dim,)

            context_embeddings.append(final_ctx_embedding)
            latent_means.append(final_latent)

    context_embeddings = np.array(context_embeddings)  # (N, mem_hidden)
    latent_means = np.array(latent_means)  # (N, latent_dim)

    print(f"  ✓ Features extracted:")
    print(f"    Context embeddings: {context_embeddings.shape}")
    print(f"    Latent means: {latent_means.shape}")
    print(f"    RMSE values: {rmse_per_sample.shape}")

    # Store
    results_by_horizon[horizon] = {
        'context_embeddings': context_embeddings,
        'latent_means': latent_means,
        'rmse': rmse_per_sample,
        'num_samples': num_samples
    }

print("\n✓ Feature extraction complete")
print()

# ============================================================================
# R² Analysis
# ============================================================================

print("=" * 80)
print("R² DECOMPOSITION ANALYSIS")
print("=" * 80)
print()

regression_results = []

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    data = results_by_horizon[horizon]
    X_context = data['context_embeddings']
    X_latent = data['latent_means']
    y_rmse = data['rmse']

    # Standardize features
    scaler_context = StandardScaler()
    scaler_latent = StandardScaler()

    X_context_scaled = scaler_context.fit_transform(X_context)
    X_latent_scaled = scaler_latent.fit_transform(X_latent)

    # Model A: Context embeddings only
    model_context = LinearRegression()
    model_context.fit(X_context_scaled, y_rmse)
    r2_context = model_context.score(X_context_scaled, y_rmse)

    # Model B: Context embeddings + latent variables
    X_full = np.concatenate([X_context_scaled, X_latent_scaled], axis=1)
    model_full = LinearRegression()
    model_full.fit(X_full, y_rmse)
    r2_full = model_full.score(X_full, y_rmse)

    # Model C: Latent only (for comparison)
    model_latent = LinearRegression()
    model_latent.fit(X_latent_scaled, y_rmse)
    r2_latent = model_latent.score(X_latent_scaled, y_rmse)

    # Compute latent contribution
    latent_contribution = r2_full - r2_context
    latent_contribution_pct = latent_contribution * 100

    print(f"\n  Regression R² Scores:")
    print(f"    Context only:          R² = {r2_context:.4f}")
    print(f"    Context + Latent:      R² = {r2_full:.4f}")
    print(f"    Latent only:           R² = {r2_latent:.4f}")
    print(f"    Latent contribution:   ΔR² = {latent_contribution:+.4f} ({latent_contribution_pct:+.2f}%)")
    print()

    if latent_contribution > 0.05:
        print(f"  ✓ STRONG latent contribution (ΔR² > 5%)")
    elif latent_contribution > 0.01:
        print(f"  ~ MODERATE latent contribution (1% < ΔR² < 5%)")
    elif latent_contribution > 0:
        print(f"  ⚠ WEAK latent contribution (ΔR² < 1%)")
    else:
        print(f"  ✗ NEGATIVE contribution (latent hurts prediction!)")

    # Per-dimension analysis (which latent dims correlate with RMSE)
    print(f"\n  Per-Dimension Correlation with RMSE:")
    print(f"    {'Dimension':<15} {'Correlation':<15} {'Abs Correlation':<18}")
    print(f"    {'-'*50}")

    dim_correlations = []
    for dim in range(latent_dim):
        corr = np.corrcoef(X_latent[:, dim], y_rmse)[0, 1]
        dim_correlations.append(corr)
        print(f"    Dim {dim:<12} {corr:<15.4f} {abs(corr):<18.4f}")

    # Most important dimension
    most_important_dim = np.argmax(np.abs(dim_correlations))
    print(f"\n  Most important dimension: Dim {most_important_dim} (|r| = {abs(dim_correlations[most_important_dim]):.4f})")

    # Store results
    regression_results.append({
        'horizon': horizon,
        'r2_context': r2_context,
        'r2_full': r2_full,
        'r2_latent': r2_latent,
        'latent_contribution': latent_contribution,
        'dim_correlations': dim_correlations,
        'num_samples': data['num_samples']
    })

# ============================================================================
# Summary Table
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()

print(f"{'Horizon':<10} {'R² Context':<15} {'R² Full':<15} {'ΔR² (Latent)':<18} {'Interpretation'}")
print("-" * 80)

for r in regression_results:
    delta_r2_pct = r['latent_contribution'] * 100

    if r['latent_contribution'] > 0.05:
        interpretation = "✓ Strong contribution"
    elif r['latent_contribution'] > 0.01:
        interpretation = "~ Moderate contribution"
    elif r['latent_contribution'] > 0:
        interpretation = "⚠ Weak contribution"
    else:
        interpretation = "✗ Negative contribution"

    print(f"H{r['horizon']:<9} {r['r2_context']:<15.4f} {r['r2_full']:<15.4f} "
          f"{delta_r2_pct:+<18.2f}% {interpretation}")

print()

# ============================================================================
# Visualization
# ============================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

output_dir = "models/backfill/latent_contribution_figs"
import os
os.makedirs(output_dir, exist_ok=True)

# Plot 1: R² comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))

horizons_plot = [r['horizon'] for r in regression_results]
r2_context = [r['r2_context'] for r in regression_results]
r2_full = [r['r2_full'] for r in regression_results]
r2_latent = [r['r2_latent'] for r in regression_results]

x = np.arange(len(horizons_plot))
width = 0.25

ax.bar(x - width, r2_context, width, label='Context only', color='blue', alpha=0.7)
ax.bar(x, r2_full, width, label='Context + Latent', color='green', alpha=0.7)
ax.bar(x + width, r2_latent, width, label='Latent only', color='red', alpha=0.7)

ax.set_xlabel('Horizon (days)')
ax.set_ylabel('R² Score')
ax.set_title('R² Decomposition: Predicting RMSE from Features')
ax.set_xticks(x)
ax.set_xticklabels([f'H{h}' for h in horizons_plot])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plot1_file = f"{output_dir}/r2_comparison.png"
plt.savefig(plot1_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot1_file}")
plt.close()

# Plot 2: Latent contribution by horizon
fig, ax = plt.subplots(figsize=(10, 6))

latent_contrib = [r['latent_contribution'] * 100 for r in regression_results]

ax.bar(range(len(horizons_plot)), latent_contrib, color='purple', alpha=0.7)
ax.axhline(5, color='green', linestyle='--', linewidth=2, label='Strong threshold (5%)')
ax.axhline(1, color='orange', linestyle='--', linewidth=2, label='Moderate threshold (1%)')
ax.axhline(0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Horizon (days)')
ax.set_ylabel('Latent Contribution (ΔR² %)')
ax.set_title('Latent Variable Contribution to RMSE Prediction')
ax.set_xticks(range(len(horizons_plot)))
ax.set_xticklabels([f'H{h}' for h in horizons_plot])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plot2_file = f"{output_dir}/latent_contribution.png"
plt.savefig(plot2_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot2_file}")
plt.close()

# Plot 3: Per-dimension correlation heatmap
fig, ax = plt.subplots(figsize=(10, 6))

corr_matrix = np.array([r['dim_correlations'] for r in regression_results])  # (4 horizons, latent_dim)

im = ax.imshow(corr_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
ax.set_xlabel('Horizon')
ax.set_ylabel('Latent Dimension')
ax.set_title('Per-Dimension Correlation with RMSE')
ax.set_xticks(range(len(horizons_plot)))
ax.set_xticklabels([f'H{h}' for h in horizons_plot])
ax.set_yticks(range(latent_dim))
ax.set_yticklabels([f'Dim {d}' for d in range(latent_dim)])

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation with RMSE')

# Annotate cells
for i in range(len(horizons_plot)):
    for j in range(latent_dim):
        text = ax.text(i, j, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

plot3_file = f"{output_dir}/dimension_correlation_heatmap.png"
plt.savefig(plot3_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot3_file}")
plt.close()

print()

# ============================================================================
# Final Verdict
# ============================================================================

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

avg_latent_contrib = np.mean([r['latent_contribution'] for r in regression_results])
avg_latent_contrib_pct = avg_latent_contrib * 100

positive_contrib_count = sum(1 for r in regression_results if r['latent_contribution'] > 0)
strong_contrib_count = sum(1 for r in regression_results if r['latent_contribution'] > 0.05)

print(f"Average latent contribution across horizons: {avg_latent_contrib_pct:+.2f}%")
print(f"Horizons with positive contribution: {positive_contrib_count}/{len(horizons)}")
print(f"Horizons with strong contribution (>5%): {strong_contrib_count}/{len(horizons)}")
print()

if avg_latent_contrib > 0.05:
    print("CONCLUSION: ✅ Latent variables provide STRONG contribution")
    print("Adding latent to context improves RMSE prediction significantly.")
    print("This is evidence that latent IS being used effectively.")
elif avg_latent_contrib > 0.01:
    print("CONCLUSION: ⚠️ Latent variables provide MODERATE contribution")
    print("Latent adds some value beyond context, but effect is modest.")
    print("Model may rely primarily on context.")
elif avg_latent_contrib > 0:
    print("CONCLUSION: ⚠️ Latent variables provide WEAK contribution")
    print("Latent improves RMSE prediction minimally.")
    print("Context dominates prediction quality.")
else:
    print("CONCLUSION: ❌ Latent variables provide NO contribution")
    print("Adding latent does not improve RMSE prediction.")
    print("This suggests latent collapse or redundancy with context.")

print()

# ============================================================================
# Save Results
# ============================================================================

output_file = "models/backfill/latent_contribution_16yr.npz"
print(f"Saving results to {output_file}...")

np.savez(
    output_file,
    regression_summary=np.array([(
        r['horizon'],
        r['r2_context'],
        r['r2_full'],
        r['r2_latent'],
        r['latent_contribution']
    ) for r in regression_results], dtype=[
        ('horizon', 'i4'),
        ('r2_context', 'f8'),
        ('r2_full', 'f8'),
        ('r2_latent', 'f8'),
        ('latent_contribution', 'f8')
    ]),
    dimension_correlations=np.array([r['dim_correlations'] for r in regression_results])
)

print("✓ Results saved")
print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Figures saved to: {output_dir}/")
print("  - r2_comparison.png")
print("  - latent_contribution.png")
print("  - dimension_correlation_heatmap.png")
