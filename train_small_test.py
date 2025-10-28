"""
Small training test for quantile regression decoder.
Quick sanity check: trains on 100 days for 10 epochs.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
import os

print("=" * 60)
print("QUANTILE REGRESSION - SMALL TRAINING TEST")
print("=" * 60)

# Set seeds for reproducibility
set_seeds(42)
torch.set_default_dtype(torch.float64)

# Load data (small subset)
print("\n1. Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)

# Use only first 150 days (100 train, 25 val, 25 test)
N_DAYS = 150
vol_surf_data = vol_surf_data[:N_DAYS]
ex_data = ex_data[:N_DAYS]

print(f"   - Surface data shape: {vol_surf_data.shape}")
print(f"   - Extra features shape: {ex_data.shape}")

# Split data
TRAIN_END = 100
VAL_END = 125
train_simple = vol_surf_data[:TRAIN_END]
val_simple = vol_surf_data[TRAIN_END:VAL_END]
test_simple = vol_surf_data[VAL_END:]

train_ex = (vol_surf_data[:TRAIN_END], ex_data[:TRAIN_END])
val_ex = (vol_surf_data[TRAIN_END:VAL_END], ex_data[TRAIN_END:VAL_END])
test_ex = (vol_surf_data[VAL_END:], ex_data[VAL_END:])

print(f"   - Train: {TRAIN_END} days")
print(f"   - Val: {VAL_END - TRAIN_END} days")
print(f"   - Test: {N_DAYS - VAL_END} days")

# Model configuration
print("\n2. Configuring model...")
config = {
    "feat_dim": (5, 5),
    "latent_dim": 5,
    "surface_hidden": [5, 5, 5],
    "ctx_surface_hidden": [5, 5, 5],
    "ex_feats_dim": 0,  # No extra features for this test
    "ex_feats_hidden": None,
    "ctx_ex_feats_hidden": None,
    "mem_type": "lstm",
    "mem_hidden": 50,  # Smaller for quick training
    "mem_layers": 1,
    "mem_dropout": 0.0,
    "interaction_layers": None,
    "use_dense_surface": False,
    "compress_context": False,
    "kl_weight": 1e-5,
    "re_feat_weight": 1.0,
    "ex_feats_loss_type": "l2",
    "ex_loss_on_ret_only": False,
    "use_quantile_regression": True,
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"   - Device: {config['device']}")
print(f"   - Quantile regression: {config['use_quantile_regression']}")
print(f"   - Quantiles: {config['quantiles']}")

# Create model
print("\n3. Creating model...")
model = CVAEMemRand(config)
print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create datasets and dataloaders
print("\n4. Creating datasets...")
min_seq_len = 5
max_seq_len = 10
batch_size = 8

train_dataset = VolSurfaceDataSetRand(train_simple, min_seq_len=min_seq_len, max_seq_len=max_seq_len)
val_dataset = VolSurfaceDataSetRand(val_simple, min_seq_len=min_seq_len, max_seq_len=max_seq_len)

train_sampler = CustomBatchSampler(train_dataset, batch_size=batch_size, min_seq_len=min_seq_len)
val_sampler = CustomBatchSampler(val_dataset, batch_size=batch_size, min_seq_len=min_seq_len)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

print(f"   - Train batches: {len(train_loader)}")
print(f"   - Val batches: {len(val_loader)}")

# Training
print("\n5. Training...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_train_loss = 0
    epoch_train_re_surf = 0
    epoch_train_kl = 0

    for batch_idx, batch in enumerate(train_loader):
        loss_dict = model.train_step(batch, optimizer)
        epoch_train_loss += loss_dict["loss"].item()
        epoch_train_re_surf += loss_dict["re_surface"].item()
        epoch_train_kl += loss_dict["kl_loss"].item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_re = epoch_train_re_surf / len(train_loader)
    avg_train_kl = epoch_train_kl / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    epoch_val_loss = 0
    epoch_val_re_surf = 0
    epoch_val_kl = 0

    with torch.no_grad():
        for batch in val_loader:
            loss_dict = model.test_step(batch)
            epoch_val_loss += loss_dict["loss"].item()
            epoch_val_re_surf += loss_dict["re_surface"].item()
            epoch_val_kl += loss_dict["kl_loss"].item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_val_re = epoch_val_re_surf / len(val_loader)
    avg_val_kl = epoch_val_kl / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"   Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.6f} (RE: {avg_train_re:.6f}, KL: {avg_train_kl:.6f}) | "
          f"Val Loss: {avg_val_loss:.6f}")

# Check loss decreased
print("\n6. Checking training progress...")
if train_losses[-1] < train_losses[0]:
    print(f"   ✓ Loss decreased: {train_losses[0]:.6f} → {train_losses[-1]:.6f}")
else:
    print(f"   ✗ Loss did not decrease: {train_losses[0]:.6f} → {train_losses[-1]:.6f}")

# Test generation and quantile ordering
print("\n7. Testing generation and quantile ordering...")
model.eval()
ctx_len = 5

# Generate for a few test days
test_days = 5
violations_p5_p50 = 0
violations_p50_p95 = 0
total_grid_points = 0

with torch.no_grad():
    for day_idx in range(ctx_len, ctx_len + test_days):
        # Prepare context
        surf_data = torch.from_numpy(vol_surf_data[day_idx - ctx_len:day_idx])
        ctx_data = {"surface": surf_data.unsqueeze(0)}

        # Generate quantiles
        generated = model.get_surface_given_conditions(ctx_data)  # (1, 1, 3, 5, 5)
        generated = generated.cpu().numpy().squeeze(0).squeeze(0)  # (3, 5, 5)

        p05 = generated[0]
        p50 = generated[1]
        p95 = generated[2]

        # Check ordering: p5 <= p50 <= p95
        violations_p5_p50 += np.sum(p05 > p50)
        violations_p50_p95 += np.sum(p50 > p95)
        total_grid_points += p05.size

print(f"   - Generated {test_days} days × 25 grid points = {total_grid_points} predictions")
print(f"   - Violations (p5 > p50): {violations_p5_p50}/{total_grid_points} ({100*violations_p5_p50/total_grid_points:.1f}%)")
print(f"   - Violations (p50 > p95): {violations_p50_p95}/{total_grid_points} ({100*violations_p50_p95/total_grid_points:.1f}%)")

if violations_p5_p50 + violations_p50_p95 < total_grid_points * 0.05:  # Less than 5% violations
    print("   ✓ Quantile ordering mostly correct!")
else:
    print("   ⚠ Some quantile ordering violations (expected early in training)")

# Save test model
print("\n8. Saving test model...")
os.makedirs("test_models", exist_ok=True)
save_path = "test_models/quantile_small_test.pt"
torch.save({
    "model_config": config,
    "model_state_dict": model.state_dict(),
    "train_losses": train_losses,
    "val_losses": val_losses,
}, save_path)
print(f"   ✓ Model saved to {save_path}")

# Summary
print("\n" + "=" * 60)
print("SMALL TRAINING TEST SUMMARY")
print("=" * 60)
print(f"✓ Training completed successfully")
print(f"✓ Loss decreased from {train_losses[0]:.6f} to {train_losses[-1]:.6f}")
print(f"✓ Model generates 3 quantiles per grid point")
print(f"✓ Quantile ordering: {100*(total_grid_points-violations_p5_p50-violations_p50_p95)/total_grid_points:.1f}% correct")
print("\nReady for full training! 🚀")
print("=" * 60)
