"""
Zero-Latent vs VAE Prior Comparison for backfill_16yr model.

TESTS LATENT COLLAPSE HYPOTHESIS:
If context is too strong and model ignores latent variable, then:
  - z=0 (no latent information) should perform similarly to z~N(0,1)
  - Prediction variance should be identical
  - RMSE should be identical

EXPECTED RESULTS (if latent IS used):
  - z=0 should have WORSE RMSE (latent adds value)
  - z=0 should have NARROWER prediction variance (no stochastic component)
  - z~N(0,1) should have better CI calibration

This script:
1. Generates predictions with z=0 for all timesteps
2. Loads existing VAE prior predictions (z~N(0,1))
3. Compares RMSE, prediction variance, CI calibration
4. Saves comparison results
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
print("ZERO-LATENT vs VAE PRIOR COMPARISON - backfill_16yr")
print("=" * 80)
print()
print("Test Hypothesis: Does model ignore latent variable (collapse)?")
print()
print("Generation Methods:")
print("  1. ZERO-LATENT: z=0 for all timesteps (context + future)")
print("  2. VAE PRIOR: z~N(0,1) for future, encoded for context")
print()
print("Expected if latent IS used:")
print("  ✓ VAE Prior has lower RMSE than Zero-Latent")
print("  ✓ VAE Prior has wider prediction variance")
print("  ✓ VAE Prior has better CI calibration")
print()
print("Expected if latent COLLAPSED (ignored):")
print("  ✗ Both methods have identical RMSE")
print("  ✗ Both methods have identical variance")
print()

# ============================================================================
# Load Model
# ============================================================================

print("Loading model...")
model_file = "models_backfill/backfill_16yr.pt"
model_data = torch.load(model_file, weights_only=False)
model_config = model_data["model_config"]

print(f"Model config:")
print(f"  Context length: {model_config['context_len']}")
print(f"  Latent dim: {model_config['latent_dim']}")
print(f"  Horizon (training): {model_config['horizon']}")
print(f"  Quantiles: {model_config['quantiles']}")

model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
device = model.device
latent_dim = model_config['latent_dim']
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

# Concatenate extra features [return, skew, slope]
ex_data = np.concatenate([
    ret_data[..., np.newaxis],
    skew_data[..., np.newaxis],
    slope_data[..., np.newaxis]
], axis=-1)

print(f"  Total data shape: {vol_surf_data.shape}")
print(f"  Extra features shape: {ex_data.shape}")

# ============================================================================
# Test Configuration
# ============================================================================

# Test on FULL training data
train_start = 1000
train_end = 5000
context_len = model_config['context_len']
horizons = [1, 7, 14, 30]

print(f"\nTest Configuration:")
print(f"  Test period: indices [{train_start}, {train_end}]")
print(f"  Test days: {train_end - train_start}")
print(f"  Context length: {context_len}")
print(f"  Horizons to test: {horizons}")
print()

# ============================================================================
# ZERO-LATENT Generation (z=0 for ALL timesteps)
# ============================================================================

print("=" * 80)
print("GENERATING WITH ZERO-LATENT (z=0)")
print("=" * 80)
print()

# Storage for zero-latent predictions
zero_predictions = {}
zero_indices = {}

for horizon in horizons:
    print(f"{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    # Available test days for this horizon
    min_idx = train_start + context_len
    max_idx = train_end - horizon
    num_test_days = max_idx - min_idx

    print(f"  Available test days: {num_test_days}")
    print(f"  Date range: [{min_idx}, {max_idx}]")

    # Storage for this horizon
    preds = np.zeros((num_test_days, 3, 5, 5))  # (N, 3, 5, 5) - 3 quantiles
    indices = []

    # Temporarily change model horizon
    original_horizon = model.horizon
    model.horizon = horizon

    with torch.no_grad():
        for i, day_idx in enumerate(tqdm(range(min_idx, max_idx), desc=f"  Horizon {horizon}")):
            # 1. Extract CONTEXT
            context_surface = vol_surf_data[day_idx - context_len : day_idx]  # (C, 5, 5)
            context_ex = ex_data[day_idx - context_len : day_idx]  # (C, 3)

            # Create context batch
            ctx_input = {
                "surface": torch.from_numpy(context_surface).unsqueeze(0).to(device),
                "ex_feats": torch.from_numpy(context_ex).unsqueeze(0).to(device)
            }

            # 2. Encode CONTEXT (to get context embeddings)
            ctx_embedding = model.ctx_encoder(ctx_input)  # (1, C, mem_hidden)

            # 3. CREATE ZERO LATENTS for ALL timesteps (context + future)
            C = context_len
            H = horizon
            T = C + H
            z_full = torch.zeros(1, T, latent_dim, device=device)  # ALL ZEROS

            # 4. Zero-pad context embeddings for future timesteps
            ctx_embedding_dim = ctx_embedding.shape[2]
            ctx_embedding_padded = torch.zeros(1, T, ctx_embedding_dim, device=device)
            ctx_embedding_padded[:, :C, :] = ctx_embedding  # First C: real context
            # Last H timesteps remain ZEROS

            # 5. Create decoder input: [context_embedding_padded || z_zeros]
            decoder_input = torch.cat([ctx_embedding_padded, z_full], dim=-1)

            # 6. Decode to get quantile predictions
            decoded_surface, decoded_ex_feat = model.decoder(decoder_input)

            # 7. Extract FUTURE predictions only
            future_preds = decoded_surface[0, C:, :, :, :]  # (H, 3, 5, 5)

            # Evaluate the H-th day
            last_day_pred = future_preds[-1, :, :, :]  # (3, 5, 5)

            preds[i] = last_day_pred.cpu().numpy()
            indices.append(day_idx + horizon - 1)

    # Restore original horizon
    model.horizon = original_horizon

    # Store results
    zero_predictions[f'zero_h{horizon}'] = preds
    zero_indices[f'indices_h{horizon}'] = np.array(indices)

    print(f"  ✓ Generated {num_test_days} predictions with zero-latent")
    print()

# ============================================================================
# Load VAE Prior Predictions for Comparison
# ============================================================================

print("=" * 80)
print("LOADING VAE PRIOR PREDICTIONS (z~N(0,1))")
print("=" * 80)
print()

vae_prior_file = "models_backfill/vae_prior_insample_16yr.npz"
print(f"Loading {vae_prior_file}...")
vae_prior_data = np.load(vae_prior_file)

print("Available keys:", list(vae_prior_data.keys()))
print()

# ============================================================================
# Load Ground Truth
# ============================================================================

print("=" * 80)
print("LOADING GROUND TRUTH")
print("=" * 80)
print()

# Ground truth is just the actual surfaces at the predicted indices
print("Extracting ground truth from data...")
ground_truth = {}

for horizon in horizons:
    indices = zero_indices[f'indices_h{horizon}']
    gt = vol_surf_data[indices]  # (N, 5, 5)
    ground_truth[f'gt_h{horizon}'] = gt
    print(f"  H{horizon}: {len(indices)} days")

print("✓ Ground truth extracted")
print()

# ============================================================================
# Comparison Analysis
# ============================================================================

print("=" * 80)
print("COMPARISON ANALYSIS")
print("=" * 80)
print()

results = []

for horizon in horizons:
    print(f"\n{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    # Get predictions
    zero_pred = zero_predictions[f'zero_h{horizon}']  # (N, 3, 5, 5)
    vae_pred = vae_prior_data[f'recon_h{horizon}']  # (N, 3, 5, 5)
    gt = ground_truth[f'gt_h{horizon}']  # (N, 5, 5)

    # Use median (quantile 1) for point forecast
    zero_median = zero_pred[:, 1, :, :]  # (N, 5, 5)
    vae_median = vae_pred[:, 1, :, :]  # (N, 5, 5)

    # 1. RMSE Comparison
    zero_rmse = np.sqrt(np.mean((zero_median - gt) ** 2))
    vae_rmse = np.sqrt(np.mean((vae_median - gt) ** 2))
    rmse_diff = vae_rmse - zero_rmse
    rmse_pct_change = (rmse_diff / zero_rmse) * 100

    print(f"\n1. RMSE Comparison (Median Forecast):")
    print(f"   Zero-Latent (z=0):    {zero_rmse:.6f}")
    print(f"   VAE Prior (z~N(0,1)): {vae_rmse:.6f}")
    print(f"   Difference:           {rmse_diff:+.6f} ({rmse_pct_change:+.2f}%)")

    if vae_rmse < zero_rmse:
        print(f"   ✓ VAE Prior is BETTER (latent IS used)")
    else:
        print(f"   ✗ Zero-Latent is BETTER (suggests collapse?)")

    # 2. Prediction Variance (across quantiles)
    # Measure: std of predictions across quantiles at each grid point
    zero_var = np.mean(np.std(zero_pred, axis=1))  # Average std across quantiles
    vae_var = np.mean(np.std(vae_pred, axis=1))
    var_ratio = vae_var / zero_var

    print(f"\n2. Prediction Variance (across quantiles):")
    print(f"   Zero-Latent variance: {zero_var:.6f}")
    print(f"   VAE Prior variance:   {vae_var:.6f}")
    print(f"   Ratio (VAE/Zero):     {var_ratio:.3f}")

    if var_ratio > 1.1:
        print(f"   ✓ VAE has wider variance (latent adds uncertainty)")
    elif var_ratio < 0.9:
        print(f"   ✗ Zero has wider variance (unexpected!)")
    else:
        print(f"   ⚠ Similar variance (latent may be collapsed)")

    # 3. CI Width Comparison
    zero_ci_width = np.mean(zero_pred[:, 2, :, :] - zero_pred[:, 0, :, :])  # p95 - p05
    vae_ci_width = np.mean(vae_pred[:, 2, :, :] - vae_pred[:, 0, :, :])
    ci_width_ratio = vae_ci_width / zero_ci_width

    print(f"\n3. Confidence Interval Width (p95 - p05):")
    print(f"   Zero-Latent CI width: {zero_ci_width:.6f}")
    print(f"   VAE Prior CI width:   {vae_ci_width:.6f}")
    print(f"   Ratio (VAE/Zero):     {ci_width_ratio:.3f}")

    if ci_width_ratio > 1.05:
        print(f"   ✓ VAE has wider CIs (latent provides uncertainty)")
    elif ci_width_ratio < 0.95:
        print(f"   ✗ Zero has wider CIs (unexpected!)")
    else:
        print(f"   ⚠ Similar CI width (latent may be collapsed)")

    # 4. CI Violation Rate
    p05_zero = zero_pred[:, 0, :, :]  # (N, 5, 5)
    p95_zero = zero_pred[:, 2, :, :]
    p05_vae = vae_pred[:, 0, :, :]
    p95_vae = vae_pred[:, 2, :, :]

    violations_zero = np.mean((gt < p05_zero) | (gt > p95_zero))
    violations_vae = np.mean((gt < p05_vae) | (gt > p95_vae))

    print(f"\n4. CI Violation Rate (90% interval):")
    print(f"   Zero-Latent violations: {violations_zero*100:.2f}%")
    print(f"   VAE Prior violations:   {violations_vae*100:.2f}%")
    print(f"   Target (well-calibrated): ~10%")

    # 5. Grid Point-wise RMSE difference
    grid_rmse_zero = np.sqrt(np.mean((zero_median - gt) ** 2, axis=0))  # (5, 5)
    grid_rmse_vae = np.sqrt(np.mean((vae_median - gt) ** 2, axis=0))
    grid_improvement = grid_rmse_zero - grid_rmse_vae  # Positive = VAE better

    vae_better_count = np.sum(grid_improvement > 0)
    total_grid_points = 25
    vae_win_rate = vae_better_count / total_grid_points * 100

    print(f"\n5. Grid Point-wise Comparison:")
    print(f"   VAE better at {vae_better_count}/{total_grid_points} grid points ({vae_win_rate:.1f}%)")

    if vae_win_rate > 60:
        print(f"   ✓ VAE wins majority (latent IS used)")
    elif vae_win_rate < 40:
        print(f"   ✗ Zero wins majority (suggests collapse)")
    else:
        print(f"   ⚠ Mixed results")

    # Store results
    results.append({
        'horizon': horizon,
        'zero_rmse': zero_rmse,
        'vae_rmse': vae_rmse,
        'rmse_diff': rmse_diff,
        'rmse_pct_change': rmse_pct_change,
        'zero_variance': zero_var,
        'vae_variance': vae_var,
        'variance_ratio': var_ratio,
        'zero_ci_width': zero_ci_width,
        'vae_ci_width': vae_ci_width,
        'ci_width_ratio': ci_width_ratio,
        'zero_violations': violations_zero,
        'vae_violations': violations_vae,
        'vae_win_rate': vae_win_rate,
    })

# ============================================================================
# Summary Table
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()
print(f"{'Horizon':<10} {'Zero RMSE':<12} {'VAE RMSE':<12} {'Improvement':<15} {'Var Ratio':<12} {'CI Ratio':<12}")
print("-" * 80)

for r in results:
    improvement_str = f"{r['rmse_pct_change']:+.2f}%"
    print(f"H{r['horizon']:<9} {r['zero_rmse']:<12.6f} {r['vae_rmse']:<12.6f} {improvement_str:<15} "
          f"{r['variance_ratio']:<12.3f} {r['ci_width_ratio']:<12.3f}")

print()

# ============================================================================
# Final Verdict
# ============================================================================

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

# Count how many horizons show VAE is better
vae_better_horizons = sum(1 for r in results if r['vae_rmse'] < r['zero_rmse'])
wider_variance_horizons = sum(1 for r in results if r['variance_ratio'] > 1.05)
wider_ci_horizons = sum(1 for r in results if r['ci_width_ratio'] > 1.05)

print(f"Evidence AGAINST latent collapse:")
print(f"  ✓ VAE Prior has lower RMSE: {vae_better_horizons}/{len(horizons)} horizons")
print(f"  ✓ VAE Prior has wider variance: {wider_variance_horizons}/{len(horizons)} horizons")
print(f"  ✓ VAE Prior has wider CIs: {wider_ci_horizons}/{len(horizons)} horizons")
print()

if vae_better_horizons >= 3 and wider_variance_horizons >= 3:
    print("CONCLUSION: ✅ Latent variable IS being used effectively")
    print("The model does NOT suffer from complete latent collapse.")
    print("VAE Prior consistently outperforms zero-latent baseline.")
elif vae_better_horizons <= 1 and wider_variance_horizons <= 1:
    print("CONCLUSION: ❌ Latent variable may be COLLAPSED")
    print("Zero-latent performs similarly to VAE Prior.")
    print("Context may be overpowering latent information.")
else:
    print("CONCLUSION: ⚠️ MIXED EVIDENCE")
    print("Latent provides some value but may be underutilized.")
    print("Further investigation needed.")

print()

# ============================================================================
# Save Results
# ============================================================================

output_file = "models_backfill/zero_vs_prior_comparison_16yr.npz"
print(f"Saving results to {output_file}...")

np.savez(
    output_file,
    # Zero-latent predictions
    **zero_predictions,
    **zero_indices,
    # Ground truth
    **ground_truth,
    # Comparison metrics
    comparison_table=np.array([(
        r['horizon'],
        r['zero_rmse'],
        r['vae_rmse'],
        r['rmse_pct_change'],
        r['variance_ratio'],
        r['ci_width_ratio'],
        r['vae_win_rate']
    ) for r in results], dtype=[
        ('horizon', 'i4'),
        ('zero_rmse', 'f8'),
        ('vae_rmse', 'f8'),
        ('rmse_pct_change', 'f8'),
        ('variance_ratio', 'f8'),
        ('ci_width_ratio', 'f8'),
        ('vae_win_rate', 'f8')
    ])
)

print("✓ Results saved")
print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
