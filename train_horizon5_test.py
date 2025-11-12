"""
Train a test model with horizon=5 to validate multi-horizon training.

This is a smaller, faster model to test that multi-horizon training works.
Compare results to horizon=1 baseline to see if it improves long-term forecasts.

Configuration:
- Variant: no_ex (simplest, no extra features)
- Horizon: 5 (predict 5 days ahead)
- Model size: Smaller (mem_hidden=50 vs 100)
- Epochs: 200
- Training time: ~2-3 hours on GPU
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.utils import set_seeds
import os
from datetime import datetime

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("HORIZON=5 TEST MODEL TRAINING")
print("=" * 80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
HORIZON = 5
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
MODEL_DIR = "test_horizon5"
MODEL_NAME = "no_ex_horizon5"  # Note: save_weights() adds .pt automatically

print("Configuration:")
print(f"  Horizon: {HORIZON}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Output: {MODEL_DIR}/{MODEL_NAME}.pt")
print()

# Create output directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]

print(f"  Total data shape: {vol_surf_data.shape}")

# Split into train/validation/test
# Use standard splits from original training
train_end_idx = 4000
valid_end_idx = 4500

vol_train = vol_surf_data[:train_end_idx]
vol_valid = vol_surf_data[train_end_idx:valid_end_idx]
vol_test = vol_surf_data[valid_end_idx:]

print(f"  Train: {vol_train.shape[0]} days")
print(f"  Valid: {vol_valid.shape[0]} days")
print(f"  Test:  {vol_test.shape[0]} days")

# Create datasets
# Need sequence length >= context + horizon
# min_context = 5, so min_seq_len = 5 + 5 = 10
min_seq_len = 5 + HORIZON  # context + horizon
max_seq_len = min_seq_len + 5

print(f"\nCreating datasets (seq_len: {min_seq_len}-{max_seq_len})...")

train_dataset = VolSurfaceDataSetRand(
    vol_train,
    min_seq_len=min_seq_len,
    max_seq_len=max_seq_len
)

valid_dataset = VolSurfaceDataSetRand(
    vol_valid,
    min_seq_len=min_seq_len,
    max_seq_len=max_seq_len
)

test_dataset = VolSurfaceDataSetRand(
    vol_test,
    min_seq_len=min_seq_len,
    max_seq_len=max_seq_len
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_sampler=CustomBatchSampler(train_dataset, BATCH_SIZE, min_seq_len)
)

valid_loader = DataLoader(
    valid_dataset,
    batch_sampler=CustomBatchSampler(valid_dataset, 16, min_seq_len)
)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=CustomBatchSampler(test_dataset, 16, min_seq_len)
)

print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")
print(f"  Test batches:  {len(test_loader)}")

# Model configuration
# Using smaller model for faster training
model_config = {
    "feat_dim": (5, 5),
    "latent_dim": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "kl_weight": 1e-5,
    "re_feat_weight": 0.0,
    "surface_hidden": [5, 5, 5],
    "ex_feats_dim": 0,  # no_ex variant
    "ex_feats_hidden": None,
    "mem_type": "lstm",
    "mem_hidden": 50,  # Smaller than default (100)
    "mem_layers": 2,
    "mem_dropout": 0.3,
    "ctx_surface_hidden": [5, 5, 5],
    "ctx_ex_feats_hidden": None,
    "interaction_layers": None,
    "compress_context": True,
    "use_dense_surface": False,
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
    "horizon": HORIZON,  # Key parameter!
}

print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)
for key, value in model_config.items():
    print(f"  {key}: {value}")
print("=" * 80)

# Create model
print("\nInitializing model...")
model = CVAEMemRand(model_config)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Device: {model.device}")
print(f"  Horizon: {model.horizon}")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print()

# Training loop
best_valid_loss = float('inf')
train_losses = []
valid_losses = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_train_loss = 0
    epoch_re_loss = 0
    epoch_kl_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        losses = model.train_step(batch, optimizer)
        epoch_train_loss += losses["loss"].item()
        epoch_re_loss += losses["re_surface"].item()
        epoch_kl_loss += losses["kl_loss"].item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_re_loss = epoch_re_loss / len(train_loader)
    avg_kl_loss = epoch_kl_loss / len(train_loader)

    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    epoch_valid_loss = 0

    with torch.no_grad():
        for batch in valid_loader:
            val_losses = model.test_step(batch)
            epoch_valid_loss += val_losses["loss"].item()

    avg_valid_loss = epoch_valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)

    # Save best model
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        model.save_weights(optimizer, MODEL_DIR, MODEL_NAME)
        best_epoch = epoch + 1
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Valid: {avg_valid_loss:.6f} | "
              f"RE: {avg_re_loss:.6f} | "
              f"KL: {avg_kl_loss:.6f} | "
              f"✓ BEST")
    else:
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Valid: {avg_valid_loss:.6f} | "
                  f"RE: {avg_re_loss:.6f} | "
                  f"KL: {avg_kl_loss:.6f}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nBest model saved at epoch {best_epoch}")
print(f"Best validation loss: {best_valid_loss:.6f}")

# Load best model and evaluate on test set
print("\nEvaluating on test set...")
model_data = torch.load(f"{MODEL_DIR}/{MODEL_NAME}.pt", weights_only=False)
model = CVAEMemRand(model_data["model_config"])
model.load_weights(dict_to_load=model_data)
model.eval()

test_loss = 0
test_re_loss = 0
test_kl_loss = 0

with torch.no_grad():
    for batch in test_loader:
        losses = model.test_step(batch)
        test_loss += losses["loss"].item()
        test_re_loss += losses["re_surface"].item()
        test_kl_loss += losses["kl_loss"].item()

avg_test_loss = test_loss / len(test_loader)
avg_test_re = test_re_loss / len(test_loader)
avg_test_kl = test_kl_loss / len(test_loader)

print(f"  Test Loss: {avg_test_loss:.6f}")
print(f"  Test RE:   {avg_test_re:.6f}")
print(f"  Test KL:   {avg_test_kl:.6f}")

# Save training history
np.savez(
    f"{MODEL_DIR}/training_history.npz",
    train_losses=np.array(train_losses),
    valid_losses=np.array(valid_losses),
    test_loss=avg_test_loss,
    test_re=avg_test_re,
    test_kl=avg_test_kl,
    config=model_config
)

print(f"\n✓ Training history saved to: {MODEL_DIR}/training_history.npz")
print(f"✓ Model saved to: {MODEL_DIR}/{MODEL_NAME}.pt")

print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\nTo compare with horizon=1 baseline:")
print(f"  1. Train horizon=1 model for comparison")
print(f"  2. Generate 5-day sequences from both models")
print(f"  3. Compare test losses and forecast accuracy")
print()
print("Expected outcome:")
print("  - horizon=5 model should have lower loss on 5-day forecasts")
print("  - Validates that multi-horizon training works")
print("  - Provides confidence for horizon=30 training")
print("=" * 80)
