"""
Test script to verify monotonic quantile reparameterization.

Trains a small model for 10 epochs and verifies:
1. No quantile crossings (p05 ≤ p50 ≤ p95)
2. Loss convergence
3. Softplus transformation works correctly
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.utils import set_seeds

# Configuration
set_seeds(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
BATCH_SIZE = 32

print("="*60)
print("MONOTONIC QUANTILE IMPLEMENTATION TEST")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print()

# Model configuration - small for fast training
model_config = {
    "feat_dim": (5, 5),
    "latent_dim": 3,
    "surface_hidden": [3, 3],
    "mem_hidden": 20,
    "mem_layers": 1,
    "mem_type": "lstm",
    "mem_dropout": 0.0,
    "ctx_surface_hidden": [3, 3],
    "ex_feats_dim": 0,  # No extra features
    "ex_feats_hidden": None,
    "ctx_ex_feats_hidden": None,
    "kl_weight": 1e-5,
    "re_feat_weight": 1.0,
    "ex_feats_loss_type": "l1",
    "ex_loss_on_ret_only": False,
    "use_dense_surface": False,
    "compress_context": False,
    "interaction_layers": None,
    "padding": 1,
    "deconv_output_padding": 0,
    "device": DEVICE,
    # Quantile parameters (REQUIRED)
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
}

print("Model Configuration:")
print(f"  Latent dim: {model_config['latent_dim']}")
print(f"  Surface hidden: {model_config['surface_hidden']}")
print(f"  Memory hidden: {model_config['mem_hidden']}")
print(f"  Quantiles: {model_config['quantiles']}")
print()

# Load data (use small subset)
print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
surface = data["surface"][:1000].astype(np.float32)  # First 1000 days only, convert to float32
print(f"  Surface shape: {surface.shape}")
print(f"  Data type: {surface.dtype}")

# Create dataset
min_seq_len = 5
max_seq_len = 10
train_size = 800

train_data = surface[:train_size]
val_data = surface[train_size:]

train_dataset = VolSurfaceDataSetRand(
    train_data, min_seq_len=min_seq_len, max_seq_len=max_seq_len
)
val_dataset = VolSurfaceDataSetRand(
    val_data, min_seq_len=min_seq_len, max_seq_len=max_seq_len
)

train_sampler = CustomBatchSampler(train_dataset, BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

val_sampler = CustomBatchSampler(val_dataset, BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

print(f"  Train size: {train_size}")
print(f"  Val size: {len(surface) - train_size}")
print()

# Create model
print("Creating model...")
model = CVAEMemRand(model_config)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
print("="*60)
print("TRAINING")
print("="*60)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    batch_count = 0

    for batch in train_loader:
        loss_dict = model.train_step(batch, optimizer)
        epoch_loss += loss_dict["loss"].item()
        batch_count += 1

    avg_train_loss = epoch_loss / batch_count
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    val_batch_count = 0

    with torch.no_grad():
        for batch in val_loader:
            loss_dict = model.test_step(batch)
            val_loss += loss_dict["loss"].item()
            val_batch_count += 1

    avg_val_loss = val_loss / val_batch_count
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

print()
print("✓ Training completed")
print()

# Verification
print("="*60)
print("VERIFICATION")
print("="*60)

def verify_monotonicity(predictions):
    """
    Verify p05 ≤ p50 ≤ p95 for all predictions.

    Args:
        predictions: (N, 3, H, W) - [p05, p50, p95]

    Returns:
        crossing_rate: float
        violations_count: int
        total_points: int
    """
    p05 = predictions[:, 0, :, :]
    p50 = predictions[:, 1, :, :]
    p95 = predictions[:, 2, :, :]

    # Check violations
    violations_p05_p50 = (p05 > p50)
    violations_p50_p95 = (p50 > p95)
    violations = violations_p05_p50 | violations_p50_p95

    total = p05.size
    violation_count = violations.sum()
    crossing_rate = violation_count / total

    return crossing_rate, violation_count, total


# Generate predictions on validation set
print("Generating predictions on validation set...")
all_predictions = []
all_targets = []

model.eval()
with torch.no_grad():
    for batch in val_loader:
        # Forward pass
        surface = batch["surface"]
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B = surface.shape[0]
        T = surface.shape[1]
        C = T - 1

        # Get predictions
        surface_pred, z_mean, z_log_var, z = model.forward(batch)
        # surface_pred: (B, 1, 3, H, W)

        # Extract quantiles
        preds = surface_pred[:, 0, :, :, :].cpu().numpy()  # (B, 3, H, W)
        targets = surface[:, C:, :, :].cpu().numpy()  # (B, 1, H, W)

        all_predictions.append(preds)
        all_targets.append(targets[:, 0, :, :])  # (B, H, W)

all_predictions = np.concatenate(all_predictions, axis=0)  # (N, 3, H, W)
all_targets = np.concatenate(all_targets, axis=0)  # (N, H, W)

print(f"  Predictions shape: {all_predictions.shape}")
print(f"  Targets shape: {all_targets.shape}")
print()

# Check monotonicity
print("Checking quantile ordering...")
crossing_rate, violations, total = verify_monotonicity(all_predictions)

print(f"  Total predictions: {total:,}")
print(f"  Quantile violations: {violations}")
print(f"  Crossing rate: {crossing_rate*100:.2f}%")

if violations == 0:
    print("  ✓ PERFECT MONOTONICITY - No crossings detected!")
else:
    print(f"  ✗ WARNING: {violations} crossings detected")

print()

# Show sample predictions
print("Sample predictions:")
sample_idx = 0
sample_grid = (2, 2)  # Center grid point

p05_sample = all_predictions[sample_idx, 0, sample_grid[0], sample_grid[1]]
p50_sample = all_predictions[sample_idx, 1, sample_grid[0], sample_grid[1]]
p95_sample = all_predictions[sample_idx, 2, sample_grid[0], sample_grid[1]]
target_sample = all_targets[sample_idx, sample_grid[0], sample_grid[1]]

print(f"  Day {sample_idx}, Grid {sample_grid}:")
print(f"    p05: {p05_sample:.6f}")
print(f"    p50: {p50_sample:.6f}  (> p05: {p50_sample > p05_sample})")
print(f"    p95: {p95_sample:.6f}  (> p50: {p95_sample > p50_sample})")
print(f"    Target: {target_sample:.6f}")
print()

# Check if target is within CI
within_ci = (target_sample >= p05_sample) and (target_sample <= p95_sample)
print(f"    Target within CI: {within_ci}")
print()

# Summary
print("="*60)
print("TEST SUMMARY")
print("="*60)
print(f"Training: {NUM_EPOCHS} epochs")
print(f"  Initial loss: {train_losses[0]:.6f}")
print(f"  Final loss: {train_losses[-1]:.6f}")
print(f"  Loss improvement: {(train_losses[0] - train_losses[-1])/train_losses[0]*100:.1f}%")
print()
print(f"Verification:")
print(f"  Total predictions: {total:,}")
print(f"  Quantile violations: {violations}")
print(f"  Crossing rate: {crossing_rate*100:.4f}%")
print()

# Final verdict
if violations == 0 and train_losses[-1] < train_losses[0]:
    print("✓✓✓ TEST PASSED ✓✓✓")
    print("  - Model trains successfully")
    print("  - Loss converges")
    print("  - Zero quantile crossings")
    print("  - Monotonic transform works correctly")
else:
    print("✗ TEST FAILED")
    if violations > 0:
        print(f"  - {violations} quantile crossings detected")
    if train_losses[-1] >= train_losses[0]:
        print("  - Loss did not improve")

print("="*60)
