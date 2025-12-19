"""
Diagnostic script to compute KL divergence and loss components for latent12 V2 model.

This script loads the trained latent12 V2 model (with corrected KL weight) and computes
the detailed loss breakdown on validation data to verify that over-regularization was fixed.

Expected Results:
- KL divergence: 2-5 (healthy range, vs V1: 0.854)
- Reconstruction loss: similar to V1
- Total loss: balanced between reconstruction and KL terms
"""

import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from vae.cvae_with_mem_randomized import CVAEMemRand
from config.backfill_context60_config_latent12_v2 import BackfillContext60ConfigLatent12V2 as cfg

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = PROJECT_ROOT / "models" / "backfill" / "context60_experiment" / "checkpoints" / "backfill_context60_latent12_v2_best.pt"
DATA_PATH = PROJECT_ROOT / "data" / "vol_surface_with_ret.npz"

# Use same validation indices as training
TRAIN_START = cfg.train_start_idx
TRAIN_END = cfg.train_end_idx
SPLIT_RATIO = 0.8
SPLIT_IDX = int((TRAIN_END - TRAIN_START) * SPLIT_RATIO) + TRAIN_START

print("=" * 80)
print("KL DIVERGENCE DIAGNOSTIC FOR LATENT12 V2 MODEL (CORRECTED)")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Config: latent_dim={cfg.latent_dim}, kl_weight={cfg.kl_weight} (CORRECTED from 5e-5)")
print(f"V1 Result: KL=0.854 (collapsed)")
print(f"V2 Target: KL=2-5 (healthy)")
print(f"Validation indices: [{SPLIT_IDX}, {TRAIN_END}]")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading data...")
data = np.load(DATA_PATH)
vol_data = data["surface"]
ret_data = data["ret"].reshape(-1, 1)
skew_data = data["skews"].reshape(-1, 1)
slope_data = data["slopes"].reshape(-1, 1)
ex_data = np.concatenate([ret_data, skew_data, slope_data], axis=1)

# Validation data
vol_valid = vol_data[SPLIT_IDX:TRAIN_END]
ex_valid = ex_data[SPLIT_IDX:TRAIN_END]

print(f"Validation data shape: {vol_valid.shape}")

# ============================================================================
# Load Model
# ============================================================================

print("\nLoading model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model = CVAEMemRand(checkpoint["model_config"])
model.load_weights(dict_to_load=checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded successfully")
print(f"Device: {device}")
print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"Checkpoint val_loss: {checkpoint.get('val_metrics', {}).get('loss', 'unknown')}")

# ============================================================================
# Compute Loss Components on Validation Data
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING LOSS COMPONENTS ON VALIDATION DATA")
print("=" * 80)

# Create batches manually
batch_size = 256
seq_len = 80  # context=60 + horizon=1 + buffer
num_samples = 500  # Sample 500 sequences for diagnostics

loss_components = defaultdict(list)

with torch.no_grad():
    for i in range(num_samples):
        # Random start index for sequence
        start_idx = np.random.randint(0, len(vol_valid) - seq_len)

        # Extract sequence
        vol_seq = vol_valid[start_idx:start_idx + seq_len]
        ex_seq = ex_valid[start_idx:start_idx + seq_len]

        # Convert to torch tensors
        batch = {
            "surface": torch.FloatTensor(vol_seq).unsqueeze(0).to(device),  # (1, T, 5, 5)
            "ex_feats": torch.FloatTensor(ex_seq).unsqueeze(0).to(device),   # (1, T, 3)
        }

        surface = batch["surface"]
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - model.horizon  # Context length = total length - horizon
        surface_real = surface[:, C:, :, :].to(device)

        ex_feats = batch["ex_feats"]
        ex_feats_real = ex_feats[:, C:, :].to(device)

        # Forward pass with full sequence (context + target)
        # This computes q(z|context, target) - the posterior
        surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = model(batch)

        # Compute reconstruction loss using quantile loss
        re_surface = model.quantile_loss_fn(surface_reconstruction, surface_real)

        if model.config["ex_loss_on_ret_only"]:
            ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
            ex_feats_real = ex_feats_real[:, :, :1]
        re_ex_feats = model.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
        reconstruction_error = re_surface + model.config["re_feat_weight"] * re_ex_feats

        # Compute KL divergence
        # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        # Total loss
        total_loss = reconstruction_error + model.kl_weight * kl_loss

        # Store components
        loss_components['loss'].append(total_loss.item())
        loss_components['reconstruction_loss'].append(reconstruction_error.item())
        loss_components['re_surface'].append(re_surface.item())
        loss_components['re_ex_feats'].append(re_ex_feats.item())
        loss_components['kl'].append(kl_loss.item())
        loss_components['weighted_kl'].append((model.kl_weight * kl_loss).item())

# Average loss components
print("\nLoss Components (averaged over 500 validation sequences):")
print("-" * 80)

for key in sorted(loss_components.keys()):
    values = loss_components[key]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{key:20s}: {mean_val:10.6f} ± {std_val:.6f}")

# ============================================================================
# KL Divergence Analysis
# ============================================================================

if 'kl' in loss_components:
    kl_values = loss_components['kl']
    kl_mean = np.mean(kl_values)
    kl_std = np.std(kl_values)

    print("\n" + "=" * 80)
    print("KL DIVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"KL divergence: {kl_mean:.4f} ± {kl_std:.4f}")
    print()

    # Diagnostic interpretation
    if kl_mean < 1.0:
        print("⚠️  SEVERE OVER-REGULARIZATION DETECTED")
        print("   KL < 1.0 indicates posterior has collapsed to prior")
        print("   Latent space is not being used effectively")
        print()
        print("   RECOMMENDATION: Reduce kl_weight significantly")
        print(f"   Current: kl_weight = {cfg.kl_weight}")
        print(f"   Suggested: kl_weight = 1e-5 (5× reduction)")

    elif kl_mean < 2.0:
        print("⚠️  MODERATE OVER-REGULARIZATION")
        print("   KL < 2.0 suggests posterior is too close to prior")
        print("   Latent space utilization is suboptimal")
        print()
        print("   RECOMMENDATION: Reduce kl_weight moderately")
        print(f"   Current: kl_weight = {cfg.kl_weight}")
        print(f"   Suggested: kl_weight = 2e-5 (2.5× reduction)")

    elif kl_mean < 5.0:
        print("✅ HEALTHY KL DIVERGENCE RANGE")
        print("   KL in [2, 5] indicates good balance")
        print("   Posterior is learning meaningful representations")
        print()
        print("   RECOMMENDATION: Issue is NOT over-regularization")
        print("   Consider alternative diagnoses:")
        print("   - Insufficient latent capacity (try latent_dim=16-24)")
        print("   - Decoder bottleneck (increase decoder capacity)")
        print("   - Fundamental architecture limitation")

    elif kl_mean < 10.0:
        print("⚠️  MODERATE UNDER-REGULARIZATION")
        print("   KL in [5, 10] suggests posterior is diverging from prior")
        print("   May lead to overfitting")
        print()
        print("   RECOMMENDATION: Consider increasing kl_weight slightly")
        print(f"   Current: kl_weight = {cfg.kl_weight}")
        print(f"   Suggested: kl_weight = 7e-5 (1.4× increase)")

    else:
        print("❌ SEVERE UNDER-REGULARIZATION")
        print("   KL > 10 indicates posterior has collapsed to delta function")
        print("   Model is effectively ignoring the prior")
        print()
        print("   RECOMMENDATION: Increase kl_weight significantly")
        print(f"   Current: kl_weight = {cfg.kl_weight}")
        print(f"   Suggested: kl_weight = 1e-4 (2× increase)")

    print("=" * 80)

# ============================================================================
# Reconstruction Loss Analysis
# ============================================================================

if 're_loss' in loss_components or 'recon' in loss_components or 'reconstruction' in loss_components:
    recon_key = 're_loss' if 're_loss' in loss_components else ('recon' if 'recon' in loss_components else 'reconstruction')
    recon_values = loss_components[recon_key]
    recon_mean = np.mean(recon_values)
    recon_std = np.std(recon_values)

    print("\nRECONSTRUCTION LOSS ANALYSIS")
    print("-" * 80)
    print(f"Reconstruction loss: {recon_mean:.6f} ± {recon_std:.6f}")

    if 'kl' in loss_components:
        kl_mean = np.mean(loss_components['kl'])
        weighted_kl = kl_mean * cfg.kl_weight

        print(f"Weighted KL term:    {weighted_kl:.6f} (kl_weight={cfg.kl_weight})")
        print(f"Total loss:          {recon_mean + weighted_kl:.6f}")
        print()
        print(f"Loss ratio (recon/kl): {recon_mean / weighted_kl:.2f}x")

        if recon_mean / weighted_kl < 1.0:
            print("⚠️  KL term dominates loss - strong over-regularization signal")
        elif recon_mean / weighted_kl < 10.0:
            print("✅ Balanced loss components")
        else:
            print("⚠️  Reconstruction dominates - KL weight may be too weak")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
