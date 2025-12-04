"""
VAE Health Analysis for backfill_16yr Model

Analyzes VAE health from multiple perspectives:
1. Posterior collapse detection (per-dimension KL divergence)
2. Latent utilization (variance, active units, effective dimensionality)
3. Temporal dynamics (crisis vs normal regime differences)
4. Reconstruction quality correlation with latent statistics

Runs on full training set (4000 days) for all horizons [1, 7, 14, 30].
"""
import numpy as np
import pandas as pd
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
from tqdm import tqdm

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("VAE HEALTH ANALYSIS - backfill_16yr")
print("=" * 80)
print()

# ============================================================================
# Load Model
# ============================================================================

print("Loading model...")
model_file = "models/backfill/backfill_16yr.pt"
model_data = torch.load(model_file, weights_only=False)
model_config = model_data["model_config"]

print(f"Model config:")
print(f"  Context length: {model_config['context_len']}")
print(f"  Latent dim: {model_config['latent_dim']}")
print(f"  Quantiles: {model_config['quantiles']}")
print()

model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
print("✓ Model loaded")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]

# Concatenate extra features
ex_data = np.concatenate([
    ret_data[..., np.newaxis],
    skew_data[..., np.newaxis],
    slope_data[..., np.newaxis]
], axis=-1)

print(f"  Total data shape: {vol_surf_data.shape}")
print(f"  Extra features shape: {ex_data.shape}")
print()

# ============================================================================
# Define Regimes
# ============================================================================

# Crisis period: 2008-2010 (roughly indices 2000-2765)
crisis_start = 2000
crisis_end = 2765

print("Regime definitions:")
print(f"  Crisis period: indices [{crisis_start}, {crisis_end}]")
print(f"  Normal periods: < {crisis_start} or > {crisis_end}")
print()

# ============================================================================
# Load RMSE Data for Correlation Analysis
# ============================================================================

print("Loading RMSE data...")
rmse_recon = np.load("models/backfill/insample_reconstruction_16yr.npz")
print("✓ RMSE data loaded")
print()

# ============================================================================
# Extract Latent Representations for Each Horizon
# ============================================================================

train_start = 1000
train_end = 5000
context_len = model_config['context_len']
latent_dim = model_config['latent_dim']
horizons = [1, 7, 14, 30]

# Storage for all metrics
all_metrics = {}

for horizon in horizons:
    print("=" * 80)
    print(f"HORIZON = {horizon} days")
    print("=" * 80)
    print()

    # Available test days for this horizon
    min_idx = train_start + context_len
    max_idx = train_end - horizon
    num_samples = max_idx - min_idx

    print(f"Extracting latent representations...")
    print(f"  Number of samples: {num_samples}")
    print(f"  Date range: [{min_idx}, {max_idx}]")
    print()

    # Storage for latent representations
    full_seq_len = context_len + horizon
    z_means = np.zeros((num_samples, full_seq_len, latent_dim))
    z_log_vars = np.zeros((num_samples, full_seq_len, latent_dim))
    z_samples = np.zeros((num_samples, full_seq_len, latent_dim))
    date_indices = []
    regime_labels = []  # 0=normal, 1=crisis

    # Load reconstruction and ground truth for RMSE
    recon_key = f'recon_h{horizon}'
    indices_key = f'indices_h{horizon}'
    recons = rmse_recon[recon_key]  # (N, 3, 5, 5)
    recon_indices = rmse_recon[indices_key]  # (N,)

    # Compute RMSE per sample
    gt = vol_surf_data[recon_indices]
    p50 = recons[:, 1, :, :]
    rmse_per_sample = np.sqrt(np.mean((p50 - gt) ** 2, axis=(1, 2)))  # (N,)

    # Temporarily change model horizon
    original_horizon = model.horizon
    model.horizon = horizon

    with torch.no_grad():
        for i, day_idx in enumerate(tqdm(range(min_idx, max_idx), desc=f"  Extracting latents H{horizon}")):
            # Create full sequence: context + target
            surface_seq = vol_surf_data[day_idx - context_len : day_idx + horizon]
            ex_seq = ex_data[day_idx - context_len : day_idx + horizon]

            # Create input batch
            x = {
                "surface": torch.from_numpy(surface_seq).unsqueeze(0),
                "ex_feats": torch.from_numpy(ex_seq).unsqueeze(0)
            }

            # Forward pass - get latent representations
            _, _, z_mean, z_log_var, z = model.forward(x)

            # Store (squeeze batch dimension)
            z_means[i] = z_mean.cpu().numpy()[0]
            z_log_vars[i] = z_log_var.cpu().numpy()[0]
            z_samples[i] = z.cpu().numpy()[0]

            # Store metadata
            date_indices.append(day_idx)
            is_crisis = (day_idx >= crisis_start) and (day_idx <= crisis_end)
            regime_labels.append(1 if is_crisis else 0)

    # Restore original horizon
    model.horizon = original_horizon

    date_indices = np.array(date_indices)
    regime_labels = np.array(regime_labels)

    print(f"✓ Extracted {num_samples} latent representations")
    print()

    # ========================================================================
    # 1. POSTERIOR COLLAPSE DETECTION
    # ========================================================================

    print("1. POSTERIOR COLLAPSE DETECTION")
    print("-" * 80)

    # Compute per-dimension KL divergence
    # KL(q(z)||p(z)) = -0.5 * (1 + log_var - exp(log_var) - mean^2)
    kl_per_dim = -0.5 * (1 + z_log_vars - np.exp(z_log_vars) - z_means ** 2)
    # Shape: (num_samples, full_seq_len, latent_dim)

    # Aggregate across samples and timesteps
    mean_kl_per_dim = np.mean(kl_per_dim, axis=(0, 1))  # (latent_dim,)

    # Threshold for posterior collapse
    collapse_threshold = 0.01
    collapsed_dims = mean_kl_per_dim < collapse_threshold
    num_collapsed = np.sum(collapsed_dims)

    print(f"Per-dimension KL divergence (averaged over all samples/timesteps):")
    for d in range(latent_dim):
        status = "COLLAPSED" if collapsed_dims[d] else "active"
        print(f"  Dimension {d}: KL = {mean_kl_per_dim[d]:.6f} [{status}]")
    print()
    print(f"Collapsed dimensions: {num_collapsed}/{latent_dim} (threshold: {collapse_threshold})")
    print()

    # Separate context vs future timesteps
    context_kl = np.mean(kl_per_dim[:, :context_len, :], axis=(0, 1))  # (latent_dim,)
    future_kl = np.mean(kl_per_dim[:, context_len:, :], axis=(0, 1))  # (latent_dim,)

    print("Context vs Future KL:")
    print("  Context KL per dim:", ", ".join([f"{x:.4f}" for x in context_kl]))
    print("  Future KL per dim: ", ", ".join([f"{x:.4f}" for x in future_kl]))
    print()

    # ========================================================================
    # 2. LATENT UTILIZATION ANALYSIS
    # ========================================================================

    print("2. LATENT UTILIZATION ANALYSIS")
    print("-" * 80)

    # Variance of each dimension across dataset
    # Use z_mean for stability (z_samples would include sampling noise)
    latent_variance = np.var(z_means, axis=(0, 1))  # (latent_dim,)

    # Active units: variance > threshold
    utilization_threshold = 0.1
    active_dims = latent_variance > utilization_threshold
    num_active = np.sum(active_dims)

    print(f"Per-dimension variance (z_mean across all samples/timesteps):")
    for d in range(latent_dim):
        status = "ACTIVE" if active_dims[d] else "inactive"
        print(f"  Dimension {d}: Var = {latent_variance[d]:.6f} [{status}]")
    print()
    print(f"Active dimensions: {num_active}/{latent_dim} (threshold: {utilization_threshold})")
    print()

    # Effective dimensionality (participation ratio)
    sum_var = np.sum(latent_variance)
    sum_var_sq = np.sum(latent_variance ** 2)
    effective_dim = (sum_var ** 2) / sum_var_sq if sum_var_sq > 0 else 0

    print(f"Effective dimensionality: {effective_dim:.2f}/{latent_dim}")
    print(f"  (Participation ratio = (Σ var)² / Σ(var²))")
    print()

    # ========================================================================
    # 3. TEMPORAL DYNAMICS
    # ========================================================================

    print("3. TEMPORAL DYNAMICS")
    print("-" * 80)

    # Mean latent per regime
    crisis_mask = regime_labels == 1
    normal_mask = regime_labels == 0

    crisis_mean = np.mean(z_means[crisis_mask], axis=(0, 1))  # (latent_dim,)
    normal_mean = np.mean(z_means[normal_mask], axis=(0, 1))  # (latent_dim,)

    crisis_var = np.var(z_means[crisis_mask], axis=(0, 1))  # (latent_dim,)
    normal_var = np.var(z_means[normal_mask], axis=(0, 1))  # (latent_dim,)

    print("Crisis vs Normal Period Latent Statistics:")
    print()
    print("Mean latent per dimension:")
    print(f"  Crisis: {', '.join([f'{x:.3f}' for x in crisis_mean])}")
    print(f"  Normal: {', '.join([f'{x:.3f}' for x in normal_mean])}")
    print()

    print("Variance per dimension:")
    print(f"  Crisis: {', '.join([f'{x:.3f}' for x in crisis_var])}")
    print(f"  Normal: {', '.join([f'{x:.3f}' for x in normal_var])}")
    print()

    # Distance between regime centroids
    centroid_distance = np.linalg.norm(crisis_mean - normal_mean)
    print(f"L2 distance between crisis and normal centroids: {centroid_distance:.4f}")
    print()

    # ========================================================================
    # 4. RECONSTRUCTION QUALITY CORRELATION
    # ========================================================================

    print("4. RECONSTRUCTION QUALITY CORRELATION")
    print("-" * 80)

    # For each sample, compute latent statistics and correlate with RMSE
    # Use only the future timestep latents (these are what decoder uses for prediction)
    future_z_mean = z_means[:, context_len:, :]  # (num_samples, horizon, latent_dim)

    # Compute statistics per sample
    latent_l2_norm = np.linalg.norm(future_z_mean, axis=(1, 2))  # (num_samples,)
    latent_variance_sample = np.var(future_z_mean, axis=(1, 2))  # (num_samples,)

    # KL divergence per sample (future timesteps only)
    future_kl_sample = np.mean(kl_per_dim[:, context_len:, :], axis=(1, 2))  # (num_samples,)

    # Correlate with RMSE
    corr_l2 = np.corrcoef(latent_l2_norm, rmse_per_sample)[0, 1]
    corr_var = np.corrcoef(latent_variance_sample, rmse_per_sample)[0, 1]
    corr_kl = np.corrcoef(future_kl_sample, rmse_per_sample)[0, 1]

    print(f"Correlation between latent statistics and RMSE:")
    print(f"  Latent L2 norm vs RMSE:     {corr_l2:+.4f}")
    print(f"  Latent variance vs RMSE:    {corr_var:+.4f}")
    print(f"  Latent KL vs RMSE:          {corr_kl:+.4f}")
    print()

    # Per-dimension contribution to RMSE prediction
    print("Per-dimension correlation with RMSE:")
    for d in range(latent_dim):
        dim_mean = np.mean(future_z_mean[:, :, d], axis=1)
        dim_corr = np.corrcoef(dim_mean, rmse_per_sample)[0, 1]
        print(f"  Dimension {d}: {dim_corr:+.4f}")
    print()

    # ========================================================================
    # Store Results
    # ========================================================================

    all_metrics[f'h{horizon}'] = {
        # Latent representations
        'z_means': z_means,
        'z_log_vars': z_log_vars,
        'z_samples': z_samples,
        'date_indices': date_indices,
        'regime_labels': regime_labels,
        'rmse_per_sample': rmse_per_sample,

        # Posterior collapse
        'mean_kl_per_dim': mean_kl_per_dim,
        'context_kl': context_kl,
        'future_kl': future_kl,
        'collapsed_dims': collapsed_dims,
        'num_collapsed': num_collapsed,

        # Latent utilization
        'latent_variance': latent_variance,
        'active_dims': active_dims,
        'num_active': num_active,
        'effective_dim': effective_dim,

        # Temporal dynamics
        'crisis_mean': crisis_mean,
        'normal_mean': normal_mean,
        'crisis_var': crisis_var,
        'normal_var': normal_var,
        'centroid_distance': centroid_distance,

        # Reconstruction correlation
        'corr_l2': corr_l2,
        'corr_var': corr_var,
        'corr_kl': corr_kl,
    }

# ============================================================================
# Summary Across Horizons
# ============================================================================

print("=" * 80)
print("SUMMARY ACROSS HORIZONS")
print("=" * 80)
print()

summary_data = []
for horizon in horizons:
    metrics = all_metrics[f'h{horizon}']
    summary_data.append({
        'horizon': horizon,
        'num_collapsed': metrics['num_collapsed'],
        'num_active': metrics['num_active'],
        'effective_dim': metrics['effective_dim'],
        'centroid_distance': metrics['centroid_distance'],
        'corr_l2_rmse': metrics['corr_l2'],
        'corr_var_rmse': metrics['corr_var'],
        'corr_kl_rmse': metrics['corr_kl'],
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print()

# ============================================================================
# Save Results
# ============================================================================

output_file = "models/backfill/vae_health_16yr.npz"
print(f"Saving results to {output_file}...")

# Prepare save dict
save_dict = {
    'latent_dim': latent_dim,
    'context_len': context_len,
    'horizons': np.array(horizons),
    'crisis_start': crisis_start,
    'crisis_end': crisis_end,
}

# Add per-horizon data
for horizon in horizons:
    metrics = all_metrics[f'h{horizon}']
    prefix = f'h{horizon}_'

    # Save all arrays
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            save_dict[prefix + key] = val
        else:
            # Save scalars as 0-d arrays
            save_dict[prefix + key] = np.array(val)

np.savez(output_file, **save_dict)
print("✓ Saved!")
print()

# Save summary CSV
summary_csv = "models/backfill/vae_health_summary_16yr.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"✓ Summary saved to {summary_csv}")
print()

print("=" * 80)
print("VAE HEALTH ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Key Findings:")
print(f"  - Latent dimensionality: {latent_dim}")
print(f"  - Effective dimensionality range: {df_summary['effective_dim'].min():.2f}-{df_summary['effective_dim'].max():.2f}")
print(f"  - Posterior collapse: {df_summary['num_collapsed'].max()}/{latent_dim} dimensions at worst")
print(f"  - Active dimensions: {df_summary['num_active'].min()}-{df_summary['num_active'].max()}/{latent_dim}")
print(f"  - Regime separation: centroid L2 distance = {df_summary['centroid_distance'].mean():.4f} (avg)")
print()
print("Next step: Run visualize_vae_health_16yr.py")
