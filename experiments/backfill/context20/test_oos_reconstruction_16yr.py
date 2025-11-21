"""
Out-of-sample reconstruction test for backfill_16yr model.

Tests CI calibration on TEST SET using Variant 2 (posterior sampling):
- Use TRUE context (real past 20 days)
- Use TRUE latent (z sampled from q(z|x) via reparameterization)
- This matches training distribution

Tests on TEST data (indices 5001-5821, ~800 days) to get:
- Out-of-sample calibration performance
- Breakdown by normal vs crisis periods (if any in test set)
- Comparison baseline for in-sample performance
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
print("OUT-OF-SAMPLE RECONSTRUCTION TEST - backfill_16yr")
print("=" * 80)
print()
print("Test Type: Variant 2 (Posterior Sampling)")
print("  - Use TRUE context (real past)")
print("  - Use TRUE latent z ~ q(z|x) via reparameterization")
print("  - Matches training distribution")
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
print(f"  Horizon (training): {model_config['horizon']}")
print(f"  Quantiles: {model_config['quantiles']}")
print(f"  Quantile weights: {model_config.get('quantile_loss_weights', [1,1,1])}")

model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
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

# Test on OUT-OF-SAMPLE data (test set)
# Training was on indices 1000-5000
# Test set starts at 5001
test_start = 5001
test_end = 5821  # End of available data
context_len = model_config['context_len']
horizons = [1, 7, 14, 30]

print(f"\nTest Configuration:")
print(f"  Test period: indices [{test_start}, {test_end}]")
print(f"  Test days: {test_end - test_start}")
print(f"  Context length: {context_len}")
print(f"  Horizons to test: {horizons}")
print()

# ============================================================================
# Generate Reconstructions
# ============================================================================

# Storage for reconstructions
reconstructions = {}
date_indices = {}

for horizon in horizons:
    print(f"{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    # Available test days for this horizon
    # Need: context_len past + horizon future
    # First prediction needs context from training data boundary
    min_idx = test_start + context_len  # First valid prediction index
    max_idx = test_end - horizon
    num_test_days = max_idx - min_idx

    print(f"  Available test days: {num_test_days}")
    print(f"  Date range: [{min_idx}, {max_idx}]")

    # Storage for this horizon
    recons = np.zeros((num_test_days, 3, 5, 5))  # (N, 3, 5, 5) - 3 quantiles
    indices = []

    # Temporarily change model horizon
    original_horizon = model.horizon
    model.horizon = horizon

    with torch.no_grad():
        for i, day_idx in enumerate(tqdm(range(min_idx, max_idx), desc=f"  Horizon {horizon}")):
            # Create full sequence: context + target
            # Context: [day_idx - context_len, day_idx)
            # Target: [day_idx, day_idx + horizon)
            full_seq_len = context_len + horizon

            # Extract data (context may come from training set boundary)
            surface_seq = vol_surf_data[day_idx - context_len : day_idx + horizon]  # (C+H, 5, 5)
            ex_seq = ex_data[day_idx - context_len : day_idx + horizon]  # (C+H, 3)

            # Create input batch
            x = {
                "surface": torch.from_numpy(surface_seq).unsqueeze(0),  # (1, C+H, 5, 5)
                "ex_feats": torch.from_numpy(ex_seq).unsqueeze(0)  # (1, C+H, 3)
            }

            # Forward pass - get posterior sample
            # This internally does:
            #   z_mean, z_logvar, z = encoder(x)
            #   where z = z_mean + exp(0.5 * z_logvar) * eps
            surf_recon, ex_recon, z_mean, z_logvar, z = model.forward(x)

            # surf_recon shape: (1, H, 3, 5, 5)
            # Extract reconstruction: (H, 3, 5, 5)
            surf_recon = surf_recon.cpu().numpy()[0]

            # For this test, we only care about the LAST day's reconstruction
            # (multi-horizon training predicts all H days, we evaluate the H-th day)
            last_day_recon = surf_recon[-1, :, :, :]  # (3, 5, 5)

            recons[i] = last_day_recon
            indices.append(day_idx + horizon - 1)  # Index of predicted day

    # Restore original horizon
    model.horizon = original_horizon

    # Store results
    reconstructions[f'recon_h{horizon}'] = recons
    date_indices[f'indices_h{horizon}'] = np.array(indices)

    print(f"  ✓ Generated {num_test_days} reconstructions")
    print()

# ============================================================================
# Save Results
# ============================================================================

output_file = "models/backfill/oos_reconstruction_16yr.npz"
print(f"Saving results to {output_file}...")

# Combine all results
save_dict = {}
save_dict.update(reconstructions)
save_dict.update(date_indices)

np.savez(output_file, **save_dict)

print("✓ Saved!")
print()
print("Contents:")
for key, val in save_dict.items():
    if isinstance(val, np.ndarray):
        print(f"  {key}: {val.shape}")

print()
print("=" * 80)
print("OUT-OF-SAMPLE RECONSTRUCTION COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print("  1. Run analyze_vae_health_oos_16yr.py for VAE health analysis")
print("  2. Run visualize_backfill_oos_16yr_plotly.py for interactive visualization")
