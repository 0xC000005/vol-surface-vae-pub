"""
Per-Dimension Latent Ablation for backfill_16yr model.

TESTS WHICH LATENT DIMENSIONS ARE IMPORTANT:
For each dimension d in [0, 1, 2, 3, 4]:
  - Disable dimension: z[:, :, d] = 0
  - Keep other dimensions: z[:, :, ~d] ~ N(0,1)
  - Measure RMSE degradation

EXPECTED RESULTS (if latent IS used):
  - Disabling high-variance dimensions (0, 3) should increase RMSE
  - Disabling collapsed dimensions (1, 4 at H1) should have minimal impact
  - RMSE degradation correlates with dimension variance

This script:
1. Generates predictions with each dimension ablated
2. Compares to full VAE prior (all dims active)
3. Ranks dimensions by importance (RMSE degradation)
4. Saves ablation results
"""
import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
from tqdm import tqdm

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("PER-DIMENSION LATENT ABLATION - backfill_16yr")
print("=" * 80)
print()
print("Test: Which latent dimensions contribute to predictions?")
print()
print("Method:")
print("  For each dimension d:")
print("    - Set z[:, :, d] = 0 (disable)")
print("    - Sample z[:, :, ~d] ~ N(0,1) (keep others active)")
print("    - Measure RMSE degradation vs full VAE prior")
print()
print("Expected:")
print("  ✓ High-variance dims (0, 3) → large RMSE increase when disabled")
print("  ✓ Collapsed dims (1, 4) → minimal RMSE increase when disabled")
print()

# ============================================================================
# Load Model
# ============================================================================

print("Loading model...")
model_file = "models_backfill/backfill_16yr.pt"
model_data = torch.load(model_file, weights_only=False)
model_config = model_data["model_config"]

latent_dim = model_config['latent_dim']
print(f"Model config:")
print(f"  Context length: {model_config['context_len']}")
print(f"  Latent dim: {latent_dim}")
print(f"  Quantiles: {model_config['quantiles']}")

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

print(f"  Total data shape: {vol_surf_data.shape}")

# ============================================================================
# Test Configuration
# ============================================================================

# Test on subset (faster, 500 days per horizon)
train_start = 1000
train_end = 5000
context_len = model_config['context_len']
horizons = [1, 7, 14, 30]
num_samples_per_horizon = 500  # Subsample for speed

print(f"\nTest Configuration:")
print(f"  Test period: indices [{train_start}, {train_end}]")
print(f"  Samples per horizon: {num_samples_per_horizon}")
print(f"  Horizons to test: {horizons}")
print()

# ============================================================================
# Load VAE Prior Predictions (Full Baseline)
# ============================================================================

print("=" * 80)
print("LOADING VAE PRIOR BASELINE (all dims active)")
print("=" * 80)
print()

vae_prior_file = "models_backfill/vae_prior_insample_16yr.npz"
vae_prior_data = np.load(vae_prior_file)
print("✓ VAE Prior baseline loaded")
print()

# Load ground truth
print("Extracting ground truth...")
ground_truth = {}
for horizon in horizons:
    indices = vae_prior_data[f'indices_h{horizon}']
    gt = vol_surf_data[indices]
    ground_truth[f'gt_h{horizon}'] = gt

print("✓ Ground truth extracted")
print()

# ============================================================================
# Per-Dimension Ablation
# ============================================================================

print("=" * 80)
print("PER-DIMENSION ABLATION")
print("=" * 80)
print()

ablation_results = {}

for ablate_dim in range(latent_dim):
    print(f"\n{'='*80}")
    print(f"ABLATING DIMENSION {ablate_dim}")
    print(f"{'='*80}")
    print()

    ablation_predictions = {}

    for horizon in horizons:
        print(f"  Horizon {horizon}...")

        # Available test days
        min_idx = train_start + context_len
        max_idx = train_end - horizon
        available_days = list(range(min_idx, max_idx))

        # Subsample for speed
        np.random.seed(42)
        test_days = sorted(np.random.choice(
            available_days,
            size=min(num_samples_per_horizon, len(available_days)),
            replace=False
        ))
        num_test = len(test_days)

        # Storage
        preds = np.zeros((num_test, 3, 5, 5))

        # Temporarily change horizon
        original_horizon = model.horizon
        model.horizon = horizon

        with torch.no_grad():
            for i, day_idx in enumerate(tqdm(test_days, desc=f"    Dim {ablate_dim}, H{horizon}")):
                # 1. Extract context
                context_surface = vol_surf_data[day_idx - context_len : day_idx]
                context_ex = ex_data[day_idx - context_len : day_idx]

                ctx_input = {
                    "surface": torch.from_numpy(context_surface).unsqueeze(0).to(device),
                    "ex_feats": torch.from_numpy(context_ex).unsqueeze(0).to(device)
                }

                # 2. Encode context
                ctx_latent_mean, _, _ = model.encoder(ctx_input)
                ctx_embedding = model.ctx_encoder(ctx_input)

                # 3. Sample future latents from N(0,1) BUT set dimension `ablate_dim` to ZERO
                C = context_len
                H = horizon
                z_future = torch.randn(1, H, latent_dim, device=device)
                z_future[:, :, ablate_dim] = 0.0  # ABLATE this dimension

                # 4. Concatenate context + ablated future
                z_full = torch.cat([ctx_latent_mean, z_future], dim=1)

                # 5. Zero-pad context embeddings
                T = C + H
                ctx_embedding_dim = ctx_embedding.shape[2]
                ctx_embedding_padded = torch.zeros(1, T, ctx_embedding_dim, device=device)
                ctx_embedding_padded[:, :C, :] = ctx_embedding

                # 6. Decode
                decoder_input = torch.cat([ctx_embedding_padded, z_full], dim=-1)
                decoded_surface, _ = model.decoder(decoder_input)

                # 7. Extract future prediction
                future_preds = decoded_surface[0, C:, :, :, :]
                last_day_pred = future_preds[-1, :, :, :]

                preds[i] = last_day_pred.cpu().numpy()

        # Restore horizon
        model.horizon = original_horizon

        # Store
        ablation_predictions[f'ablate_dim{ablate_dim}_h{horizon}'] = preds
        ablation_predictions[f'ablate_dim{ablate_dim}_indices_h{horizon}'] = np.array(test_days)

    # Save ablation results for this dimension
    ablation_results[ablate_dim] = ablation_predictions

print("\n✓ All ablations complete")
print()

# ============================================================================
# Comparison Analysis
# ============================================================================

print("=" * 80)
print("ABLATION IMPACT ANALYSIS")
print("=" * 80)
print()

comparison_results = []

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    # Get baseline (full VAE prior)
    baseline_pred = vae_prior_data[f'recon_h{horizon}']
    baseline_indices = vae_prior_data[f'indices_h{horizon}']
    gt = ground_truth[f'gt_h{horizon}']

    # Baseline RMSE (full latent)
    baseline_median = baseline_pred[:, 1, :, :]
    baseline_rmse = np.sqrt(np.mean((baseline_median - gt) ** 2))

    print(f"\nBaseline RMSE (all dims active): {baseline_rmse:.6f}")
    print()
    print(f"{'Ablated Dim':<15} {'RMSE':<15} {'RMSE Increase':<18} {'% Degradation':<18}")
    print("-" * 70)

    dim_results = []

    for ablate_dim in range(latent_dim):
        # Get ablated predictions (on subsampled days)
        ablated_pred = ablation_results[ablate_dim][f'ablate_dim{ablate_dim}_h{horizon}']
        ablated_indices = ablation_results[ablate_dim][f'ablate_dim{ablate_dim}_indices_h{horizon}']

        # Find matching indices (intersection of ablated and baseline)
        valid_indices = np.intersect1d(ablated_indices, baseline_indices)

        # Get masks for both sets
        ablated_mask = np.isin(ablated_indices, valid_indices)
        baseline_mask = np.isin(baseline_indices, valid_indices)

        # Filter to matching indices only
        ablated_pred_matched = ablated_pred[ablated_mask]
        baseline_pred_matched = baseline_pred[baseline_mask]
        gt_ablated = vol_surf_data[valid_indices]

        # Compute RMSEs on matched data
        baseline_rmse_matched = np.sqrt(np.mean((baseline_pred_matched[:, 1, :, :] - gt_ablated) ** 2))
        ablated_rmse = np.sqrt(np.mean((ablated_pred_matched[:, 1, :, :] - gt_ablated) ** 2))

        rmse_increase = ablated_rmse - baseline_rmse_matched
        pct_degradation = (rmse_increase / baseline_rmse_matched) * 100

        print(f"Dim {ablate_dim:<12} {ablated_rmse:<15.6f} {rmse_increase:+18.6f} {pct_degradation:+18.2f}%")

        dim_results.append({
            'dimension': ablate_dim,
            'rmse': ablated_rmse,
            'rmse_increase': rmse_increase,
            'pct_degradation': pct_degradation
        })

    # Sort by RMSE increase (descending)
    dim_results_sorted = sorted(dim_results, key=lambda x: x['rmse_increase'], reverse=True)

    print()
    print("Dimension Importance Ranking (by RMSE increase when ablated):")
    for rank, result in enumerate(dim_results_sorted, 1):
        print(f"  {rank}. Dim {result['dimension']}: +{result['rmse_increase']:.6f} ({result['pct_degradation']:+.2f}%)")

    comparison_results.append({
        'horizon': horizon,
        'baseline_rmse': baseline_rmse,
        'dim_results': dim_results_sorted
    })

# ============================================================================
# Cross-Horizon Dimension Importance
# ============================================================================

print("\n" + "=" * 80)
print("CROSS-HORIZON DIMENSION IMPORTANCE")
print("=" * 80)
print()

# Aggregate importance across horizons
dim_importance = {d: [] for d in range(latent_dim)}

for comp in comparison_results:
    for result in comp['dim_results']:
        dim_importance[result['dimension']].append(result['rmse_increase'])

# Average importance
dim_avg_importance = {d: np.mean(increases) for d, increases in dim_importance.items()}
dim_ranking = sorted(dim_avg_importance.items(), key=lambda x: x[1], reverse=True)

print("Average RMSE Increase When Ablated (across all horizons):")
print()
print(f"{'Rank':<8} {'Dimension':<12} {'Avg RMSE Increase':<20} {'Interpretation':<30}")
print("-" * 80)

for rank, (dim, avg_increase) in enumerate(dim_ranking, 1):
    if avg_increase > 0.0001:
        interpretation = "IMPORTANT (contributes to accuracy)"
        symbol = "✓"
    elif avg_increase > 0.00001:
        interpretation = "MODERATE (some contribution)"
        symbol = "~"
    else:
        interpretation = "MINIMAL (possibly collapsed)"
        symbol = "✗"

    print(f"{rank:<8} Dim {dim:<9} {avg_increase:<20.6f} {symbol} {interpretation}")

print()

# ============================================================================
# Load VAE Health Metrics for Comparison
# ============================================================================

print("=" * 80)
print("COMPARING TO VAE HEALTH METRICS")
print("=" * 80)
print()

try:
    vae_health_file = "models_backfill/vae_health_16yr.npz"
    vae_health = np.load(vae_health_file)

    # Check if we have per-dimension KL or variance
    if 'kl_per_dim_h30' in vae_health:
        print("Dimension KL Divergence (H30) vs Ablation Importance:")
        print()
        kl_per_dim = vae_health['kl_per_dim_h30']

        print(f"{'Dimension':<12} {'KL Divergence':<18} {'RMSE Increase':<18} {'Correlation'}")
        print("-" * 70)

        for dim in range(latent_dim):
            kl = kl_per_dim[dim]
            rmse_inc = dim_avg_importance[dim]
            print(f"Dim {dim:<9} {kl:<18.6f} {rmse_inc:<18.6f}")

        # Compute correlation
        kl_values = [kl_per_dim[d] for d in range(latent_dim)]
        rmse_values = [dim_avg_importance[d] for d in range(latent_dim)]
        correlation = np.corrcoef(kl_values, rmse_values)[0, 1]

        print()
        print(f"Correlation (KL vs RMSE increase): {correlation:.3f}")

        if correlation > 0.5:
            print("✓ STRONG correlation: Higher KL → more important dimension")
        elif correlation > 0.2:
            print("~ MODERATE correlation: Some relationship exists")
        else:
            print("✗ WEAK correlation: KL may not reflect importance")
    else:
        print("(Per-dimension KL not available in VAE health metrics)")

except FileNotFoundError:
    print("(VAE health metrics file not found)")

print()

# ============================================================================
# Final Verdict
# ============================================================================

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

# Count important dimensions
important_dims = sum(1 for avg_inc in dim_avg_importance.values() if avg_inc > 0.0001)
moderate_dims = sum(1 for avg_inc in dim_avg_importance.values() if 0.00001 < avg_inc <= 0.0001)
minimal_dims = sum(1 for avg_inc in dim_avg_importance.values() if avg_inc <= 0.00001)

print(f"Dimension Utilization (across all horizons):")
print(f"  ✓ Important dimensions (RMSE increase > 0.0001): {important_dims}/{latent_dim}")
print(f"  ~ Moderate dimensions (0.00001 < increase ≤ 0.0001): {moderate_dims}/{latent_dim}")
print(f"  ✗ Minimal dimensions (increase ≤ 0.00001): {minimal_dims}/{latent_dim}")
print()

if important_dims >= 3:
    print("CONCLUSION: ✅ Multiple latent dimensions ARE being used")
    print(f"At least {important_dims} dimensions contribute meaningfully to predictions.")
    print("This is strong evidence AGAINST complete latent collapse.")
elif important_dims >= 1:
    print("CONCLUSION: ⚠️ Partial latent utilization")
    print(f"Only {important_dims} dimension(s) contribute significantly.")
    print("Model may be operating with reduced effective dimensionality.")
else:
    print("CONCLUSION: ❌ Minimal latent utilization")
    print("Ablating dimensions has negligible impact on RMSE.")
    print("This suggests potential latent collapse.")

print()

# ============================================================================
# Save Results
# ============================================================================

output_file = "models_backfill/dimension_ablation_16yr.npz"
print(f"Saving results to {output_file}...")

# Flatten ablation predictions for saving
save_dict = {}
for ablate_dim in range(latent_dim):
    for key, value in ablation_results[ablate_dim].items():
        save_dict[key] = value

# Add summary statistics
save_dict['dimension_importance'] = np.array([
    (dim, avg_inc) for dim, avg_inc in dim_ranking
], dtype=[('dimension', 'i4'), ('avg_rmse_increase', 'f8')])

save_dict['horizon_summaries'] = np.array([
    (comp['horizon'], comp['baseline_rmse']) for comp in comparison_results
], dtype=[('horizon', 'i4'), ('baseline_rmse', 'f8')])

np.savez(output_file, **save_dict)

print("✓ Results saved")
print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
