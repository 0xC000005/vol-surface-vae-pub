"""
Train CVAEMemRand for backfilling task using limited data.

This script implements Phase 3 of BACKFILL_MVP_PLAN.md:
- Trains on limited recent data (1-3 years before test set)
- Uses scheduled sampling (Phase 2.2): teacher forcing → multi-horizon training
- Prepares model for generating 30-day backfill sequences

Training approach:
    Phase 1 (0 to teacher_forcing_epochs): Standard single-step teacher forcing
    Phase 2 (teacher_forcing_epochs+1 to epochs): Multi-horizon training [1, 7, 14, 30]
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import train_with_scheduled_sampling, set_seeds
from config.backfill_config import BackfillConfig
import os

# ============================================================================
# Setup
# ============================================================================

# Set seeds and dtype for reproducibility
set_seeds(0)
torch.set_default_dtype(torch.float64)

# Print configuration
BackfillConfig.summary()

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
# Extract Limited Training Data
# ============================================================================

train_start, train_end = BackfillConfig.get_train_indices()
vol_train = vol_surf_data[train_start:train_end]
ex_train = ex_data[train_start:train_end]

print(f"\nTraining data:")
print(f"  Shape: {vol_train.shape}")
print(f"  Period: {BackfillConfig.train_period_years} years ({train_end - train_start} days)")
print(f"  Indices: [{train_start}, {train_end}]")

# Split into train/validation (80/20)
split_idx = int(0.8 * len(vol_train))
vol_train_split = vol_train[:split_idx]
ex_train_split = ex_train[:split_idx]
vol_valid_split = vol_train[split_idx:]
ex_valid_split = ex_train[split_idx:]

print(f"  Train split: {vol_train_split.shape[0]} days")
print(f"  Valid split: {vol_valid_split.shape[0]} days")

# ============================================================================
# Create Datasets
# ============================================================================

# Need longer sequences for multi-horizon training
# Sequence length must be: context_len + max_horizon
max_horizon = max(BackfillConfig.training_horizons)
min_seq_len = BackfillConfig.context_len + max_horizon
max_seq_len = min_seq_len + 10  # Some variability for randomization

print(f"\nCreating datasets:")
print(f"  Sequence length range: {min_seq_len}-{max_seq_len}")
print(f"  Context length: {BackfillConfig.context_len}")
print(f"  Max horizon: {max_horizon}")

train_dataset = VolSurfaceDataSetRand(
    (vol_train_split, ex_train_split),
    min_seq_len=min_seq_len,
    max_seq_len=max_seq_len
)

valid_dataset = VolSurfaceDataSetRand(
    (vol_valid_split, ex_valid_split),
    min_seq_len=min_seq_len,
    max_seq_len=max_seq_len
)

# ============================================================================
# Create DataLoaders
# ============================================================================

train_loader = DataLoader(
    train_dataset,
    batch_sampler=CustomBatchSampler(
        train_dataset,
        BackfillConfig.batch_size,
        min_seq_len
    )
)

valid_loader = DataLoader(
    valid_dataset,
    batch_sampler=CustomBatchSampler(
        valid_dataset,
        16,  # Smaller batch size for validation
        min_seq_len
    )
)

print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")

# ============================================================================
# Model Configuration
# ============================================================================

model_config = {
    # Input dimensions
    "feat_dim": (5, 5),
    "ex_feats_dim": 3,  # [return, skew, slope]

    # Latent space
    "latent_dim": BackfillConfig.latent_dim,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Loss weights
    "kl_weight": BackfillConfig.kl_weight,
    "re_feat_weight": 1.0,  # ex_loss variant: optimize feature reconstruction

    # Surface encoder/decoder
    "surface_hidden": BackfillConfig.surface_hidden,
    "ex_feats_hidden": None,

    # Memory module
    "mem_type": "lstm",
    "mem_hidden": BackfillConfig.mem_hidden,
    "mem_layers": BackfillConfig.mem_layers,
    "mem_dropout": BackfillConfig.mem_dropout,

    # Context encoder
    "ctx_surface_hidden": BackfillConfig.surface_hidden,
    "ctx_ex_feats_hidden": None,
    "interaction_layers": None,
    "compress_context": True,

    # Architecture options
    "use_dense_surface": False,

    # Quantile regression
    "use_quantile_regression": BackfillConfig.use_quantile_regression,
    "num_quantiles": BackfillConfig.num_quantiles,
    "quantiles": BackfillConfig.quantiles,
    "quantile_loss_weights": BackfillConfig.quantile_loss_weights,

    # Extra features configuration
    "ex_loss_on_ret_only": True,  # Only optimize return, not skew/slope
    "ex_feats_loss_type": "l2",   # L2 loss for return prediction

    # Horizon (defaults to 1 for scheduled sampling)
    "horizon": 1,

    # Context length (for reference, not used directly by model)
    "context_len": BackfillConfig.context_len,
}

print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)
for key, value in model_config.items():
    print(f"  {key}: {value}")
print("=" * 80)

# ============================================================================
# Create Model
# ============================================================================

print("\nInitializing model...")
model = CVAEMemRand(model_config)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {num_params:,}")
print(f"  Device: {model.device}")

# ============================================================================
# Train
# ============================================================================

output_dir = "models_backfill"
os.makedirs(output_dir, exist_ok=True)
model_name = f"backfill_{BackfillConfig.train_period_years}yr.pt"

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"  Output directory: {output_dir}/")
print(f"  Model name: {model_name}")
print(f"  Total epochs: {BackfillConfig.epochs}")
print(f"  Phase 1 (teacher forcing): {BackfillConfig.teacher_forcing_epochs} epochs")
print(f"  Phase 2 (multi-horizon): {BackfillConfig.epochs - BackfillConfig.teacher_forcing_epochs} epochs")
print(f"  Training horizons: {BackfillConfig.training_horizons}")
print("=" * 80)
print()

train_with_scheduled_sampling(
    model=model,
    train_dataloader=train_loader,
    valid_dataloader=valid_loader,
    lr=BackfillConfig.learning_rate,
    epochs=BackfillConfig.epochs,
    model_dir=output_dir,
    file_name=model_name,
    teacher_forcing_epochs=BackfillConfig.teacher_forcing_epochs,
    horizons=BackfillConfig.training_horizons
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Model saved to: {output_dir}/{model_name}")
print(f"Log file: {output_dir}/backfill_{BackfillConfig.train_period_years}yr-{BackfillConfig.epochs}-log.txt")
print()
print("Next steps:")
print("  - Phase 4: Implement backfill generation script")
print("  - Generate 30-day sequences for 2008-2010 crisis period")
print("  - Evaluate backfill quality vs ground truth")
print("=" * 80)
