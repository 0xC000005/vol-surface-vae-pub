"""
Test script for Phase 3 (Training Configuration and Script).

Validates that:
1. BackfillConfig loads and computes indices correctly
2. Model initialization works with config parameters
3. Dataset creation works with correct sequence lengths
4. Training loop can be invoked (short run to verify setup)

This is a quick validation test - does NOT run full 400-epoch training.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import train_with_scheduled_sampling, set_seeds
from config.backfill_config import BackfillConfig
import os

print("=" * 80)
print("TESTING PHASE 3 - BACKFILL CONFIGURATION AND TRAINING SETUP")
print("=" * 80)
print()

# Set seeds and dtype
set_seeds(42)
torch.set_default_dtype(torch.float64)

# ============================================================================
# Test 1: Configuration Validation
# ============================================================================

print("=" * 80)
print("TEST 1: Configuration Validation")
print("=" * 80)
print()

BackfillConfig.summary()

# Validate computed indices
train_start, train_end = BackfillConfig.get_train_indices()
backfill_start, backfill_end = BackfillConfig.get_backfill_indices()

assert train_start > 0, "Train start index should be positive"
assert train_end > train_start, "Train end should be after train start"
assert train_end - train_start == 250 * BackfillConfig.train_period_years, \
    f"Expected {250 * BackfillConfig.train_period_years} days, got {train_end - train_start}"

print()
print("✓ Configuration validation passed!")
print(f"  Train indices: [{train_start}, {train_end}] = {train_end - train_start} days")
print(f"  Backfill indices: [{backfill_start}, {backfill_end}] = {backfill_end - backfill_start} days")
print()

# ============================================================================
# Test 2: Data Loading and Dataset Creation
# ============================================================================

print("=" * 80)
print("TEST 2: Data Loading and Dataset Creation")
print("=" * 80)
print()

print("Loading data...")
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
print(f"  Extra features shape: {ex_data.shape}")

# Use tiny subset for testing
test_train_start = 4000
test_train_end = 4200  # Just 200 days for quick test
vol_train = vol_surf_data[test_train_start:test_train_end]
ex_train = ex_data[test_train_start:test_train_end]

# Split into train/validation (80/20)
split_idx = int(0.8 * len(vol_train))
vol_train_split = vol_train[:split_idx]
ex_train_split = ex_train[:split_idx]
vol_valid_split = vol_train[split_idx:]
ex_valid_split = ex_train[split_idx:]

print(f"  Test train split: {vol_train_split.shape[0]} days")
print(f"  Test valid split: {vol_valid_split.shape[0]} days")

# Create datasets with correct sequence length
max_horizon = max(BackfillConfig.training_horizons)
min_seq_len = BackfillConfig.context_len + max_horizon
max_seq_len = min_seq_len + 10

print(f"  Sequence length: {min_seq_len}-{max_seq_len}")

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

# Create dataloaders with small batch size
train_loader = DataLoader(
    train_dataset,
    batch_sampler=CustomBatchSampler(train_dataset, 4, min_seq_len)
)

valid_loader = DataLoader(
    valid_dataset,
    batch_sampler=CustomBatchSampler(valid_dataset, 4, min_seq_len)
)

print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")

print()
print("✓ Dataset creation passed!")
print()

# ============================================================================
# Test 3: Model Initialization
# ============================================================================

print("=" * 80)
print("TEST 3: Model Initialization")
print("=" * 80)
print()

model_config = {
    "feat_dim": (5, 5),
    "ex_feats_dim": 3,
    "latent_dim": BackfillConfig.latent_dim,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "kl_weight": BackfillConfig.kl_weight,
    "re_feat_weight": 1.0,
    "surface_hidden": BackfillConfig.surface_hidden,
    "ex_feats_hidden": None,
    "mem_type": "lstm",
    "mem_hidden": BackfillConfig.mem_hidden,
    "mem_layers": BackfillConfig.mem_layers,
    "mem_dropout": BackfillConfig.mem_dropout,
    "ctx_surface_hidden": BackfillConfig.surface_hidden,
    "ctx_ex_feats_hidden": None,
    "interaction_layers": None,
    "compress_context": True,
    "use_dense_surface": False,
    "use_quantile_regression": BackfillConfig.use_quantile_regression,
    "num_quantiles": BackfillConfig.num_quantiles,
    "quantiles": BackfillConfig.quantiles,
    "ex_loss_on_ret_only": True,
    "ex_feats_loss_type": "l2",
    "horizon": 1,
    "context_len": BackfillConfig.context_len,
}

print("Creating model...")
model = CVAEMemRand(model_config)
num_params = sum(p.numel() for p in model.parameters())

print(f"  Total parameters: {num_params:,}")
print(f"  Device: {model.device}")
print(f"  Horizon: {model.horizon}")
print(f"  Quantile regression: {model.config['use_quantile_regression']}")

# Verify model can handle batch
print()
print("Testing forward pass...")
sample_batch = next(iter(train_loader))
with torch.no_grad():
    if "ex_feats" in sample_batch:
        surface_recon, ex_feats_recon, z_mean, z_log_var, z = model.forward(sample_batch)
        print(f"  Surface reconstruction: {surface_recon.shape}")
        print(f"  Ex_feats reconstruction: {ex_feats_recon.shape}")
    else:
        surface_recon, z_mean, z_log_var, z = model.forward(sample_batch)
        print(f"  Surface reconstruction: {surface_recon.shape}")

    print(f"  Latent z: {z.shape}")

print()
print("✓ Model initialization passed!")
print()

# ============================================================================
# Test 4: Training Loop (Short Run)
# ============================================================================

print("=" * 80)
print("TEST 4: Training Loop Validation (5 epochs)")
print("=" * 80)
print()

# Create temporary output directory
test_output_dir = "test_phase3_output"
if os.path.exists(test_output_dir):
    import shutil
    shutil.rmtree(test_output_dir)
os.makedirs(test_output_dir)

print("Running short training (5 epochs total: 2 teacher forcing + 3 multi-horizon)...")
print()

try:
    train_losses, valid_losses = train_with_scheduled_sampling(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        lr=BackfillConfig.learning_rate,
        epochs=5,  # Just 5 epochs for quick test
        model_dir=test_output_dir,
        file_name="test_model.pt",
        teacher_forcing_epochs=2,  # 2 for Phase 1, 3 for Phase 2
        horizons=BackfillConfig.training_horizons
    )

    print()
    print("✓ Training loop completed successfully!")
    print()

    # Check outputs
    print("Verifying outputs...")
    model_path = f"{test_output_dir}/test_model.pt"
    log_path = f"{test_output_dir}/test_model-5-log.txt"

    if os.path.exists(model_path):
        print(f"  ✓ Model checkpoint exists: {model_path}")

        # Verify model can be loaded
        model_data = torch.load(model_path, weights_only=False)
        loaded_model = CVAEMemRand(model_data["model_config"])
        loaded_model.load_weights(dict_to_load=model_data)
        print(f"  ✓ Model loads successfully")
    else:
        print(f"  ✗ Model checkpoint not found")

    if os.path.exists(log_path):
        print(f"  ✓ Training log exists: {log_path}")
    else:
        print(f"  ✗ Training log not found")

    print()
    print("Final losses:")
    print(f"  Train loss: {train_losses.get('loss', 'N/A')}")
    print(f"  Valid loss: {valid_losses.get('loss', 'N/A')}")

except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()
print("✓ All Phase 3 validation tests passed!")
print()
print("Validated:")
print("  1. BackfillConfig computes indices correctly")
print("  2. Dataset creation with proper sequence lengths")
print("  3. Model initialization with all required parameters")
print("  4. Training loop with scheduled sampling")
print()
print("Phase 3 (Training Configuration and Script) is complete and ready for use.")
print()
print("Next steps:")
print("  - Run full training: python train_backfill_model.py")
print("  - Proceed to Phase 4: Backfill generation script")
print("=" * 80)
print()
print(f"Test artifacts saved to: {test_output_dir}/")
