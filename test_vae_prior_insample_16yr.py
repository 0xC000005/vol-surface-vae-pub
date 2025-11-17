"""
In-sample VAE prior generation test for backfill_16yr model.

Tests CI calibration using VAE Prior Sampling (Strategy 2):
- Use TRUE context (real past 20 days)
- Use SAMPLED latent (z ~ N(0,1) for future timesteps, NOT encoded from target)
- This is REALISTIC generation (no cheating with future information)

Difference from oracle reconstruction:
- Oracle: Encodes full sequence (context + target) → z from q(z|x)
- VAE Prior: Encodes only context → z ~ N(0,1) for future

Expected: Higher CI violations due to prior mismatch (~20-22% vs ~18% oracle)

Tests on FULL training data (indices 1000-5000, 4000 days).
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
print("VAE PRIOR GENERATION TEST - backfill_16yr (IN-SAMPLE)")
print("=" * 80)
print()
print("Generation Strategy: VAE Prior Sampling (Strategy 2)")
print("  - Use TRUE context (real past 20 days)")
print("  - Sample future latents: z ~ N(0,1) [NO TARGET ENCODING]")
print("  - Zero-pad context embeddings for future")
print("  - This is REALISTIC generation (theoretically correct)")
print()
print("Expected: ~20-22% CI violations (vs ~18% oracle due to prior mismatch)")
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
print(f"  Device: {model_config['device']}")

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
# VAE Prior Generation
# ============================================================================

# Storage for predictions
predictions = {}
date_indices = {}

for horizon in horizons:
    print(f"{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    # Available test days for this horizon
    # Need: context_len past + horizon future
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
            # ===================================================================
            # VAE PRIOR GENERATION (Context-Only Encoding)
            # ===================================================================

            # 1. Extract CONTEXT ONLY (no target!)
            context_surface = vol_surf_data[day_idx - context_len : day_idx]  # (C, 5, 5)
            context_ex = ex_data[day_idx - context_len : day_idx]  # (C, 3)

            # Create context batch
            ctx_input = {
                "surface": torch.from_numpy(context_surface).unsqueeze(0).to(device),  # (1, C, 5, 5)
                "ex_feats": torch.from_numpy(context_ex).unsqueeze(0).to(device)  # (1, C, 3)
            }

            # 2. Encode CONTEXT ONLY (no target information)
            ctx_latent_mean, ctx_latent_log_var, _ = model.encoder(ctx_input)  # (1, C, latent_dim)
            ctx_embedding = model.ctx_encoder(ctx_input)  # (1, C, mem_hidden)

            # 3. Sample FUTURE latents from N(0,1) prior (realistic generation!)
            # This is the key difference from oracle: we DON'T encode the target
            z_future = torch.randn(1, horizon, latent_dim, device=device)  # (1, H, latent_dim)

            # 4. Concatenate: [encoded context | sampled future]
            z_full = torch.cat([ctx_latent_mean, z_future], dim=1)  # (1, C+H, latent_dim)

            # 5. Zero-pad context embeddings for future timesteps
            C = context_len
            H = horizon
            T = C + H
            ctx_embedding_dim = ctx_embedding.shape[2]

            ctx_embedding_padded = torch.zeros(1, T, ctx_embedding_dim, device=device)
            ctx_embedding_padded[:, :C, :] = ctx_embedding  # First C: real context
            # Last H timesteps remain ZEROS

            # 6. Create decoder input: [context_embedding_padded || z_full]
            decoder_input = torch.cat([ctx_embedding_padded, z_full], dim=-1)  # (1, C+H, embed_dim+latent_dim)

            # 7. Decode to get quantile predictions
            decoded_surface, decoded_ex_feat = model.decoder(decoder_input)  # (1, C+H, 3, 5, 5)

            # 8. Extract FUTURE predictions only (discard context reconstruction)
            future_preds = decoded_surface[0, C:, :, :, :]  # (H, 3, 5, 5)

            # For this test, we only care about the LAST day's prediction
            # (multi-horizon training predicts all H days, we evaluate the H-th day)
            last_day_pred = future_preds[-1, :, :, :]  # (3, 5, 5)

            preds[i] = last_day_pred.cpu().numpy()
            indices.append(day_idx + horizon - 1)  # Index of predicted day

    # Restore original horizon
    model.horizon = original_horizon

    # Store results
    predictions[f'recon_h{horizon}'] = preds
    date_indices[f'indices_h{horizon}'] = np.array(indices)

    print(f"  ✓ Generated {num_test_days} predictions via VAE prior")
    print()

# ============================================================================
# Save Results
# ============================================================================

output_file = "models_backfill/vae_prior_insample_16yr.npz"
print(f"Saving results to {output_file}...")

# Combine all results
save_dict = {}
save_dict.update(predictions)
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
print("VAE PRIOR GENERATION COMPLETE")
print("=" * 80)
print()
print("Generation method: VAE Prior Sampling (z ~ N(0,1) for future)")
print("  - Context latents: Encoded from real observations")
print("  - Future latents: Sampled from standard normal N(0,1)")
print("  - Context embeddings: Zero-padded for future timesteps")
print()
print("This is REALISTIC generation without future information!")
print()
print("Next step: Run evaluate_vae_prior_ci_insample_16yr.py")
