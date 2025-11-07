"""
Train 1D Backfilling VAE Model.

Trains a single backfilling model variant using multi-stock, multi-feature input:
- Target: AMZN return (1D)
- Extra features: 11 features (AMZN vol/volume/range + MSFT×4 + SP500×4)
- Total: 12D joint market state encoding

Implements 80/20 training strategy with forward-fill masking:
- 70% batches: Standard training with full information
- 30% batches: Masked training (forward-fill AMZN at T+1, use previous latent)

This simulates realistic backfilling scenarios where:
- MSFT and SP500 data available at T+1
- AMZN data missing at T+1, only available up to T
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from vae.datasets_1d_randomized import TimeSeriesDataSetRand, CustomBatchSampler1D
from vae.cvae_1d_with_mem_randomized import CVAE1DMemRand
from vae.utils import train, test, set_seeds
import os

# Configuration
DATA_FILE = "data/stock_returns_multifeature.npz"
OUTPUT_DIR = "models_1d_backfilling"
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-5

# Backfilling-specific hyperparameters (from BACKFILLING_PROPOSAL.md)
LATENT_DIM = 12  # Increased from 5 to handle 12D state space
TARGET_HIDDEN = [32, 32]  # AMZN return embedding
MEM_HIDDEN = 32  # Match proposal (was 100 in old models)
MEM_LAYERS = 2
MEM_DROPOUT = 0.2
KL_WEIGHT = 1e-5
MASK_PROB = 0.3  # 30% batches use forward-fill masking (80/20 split from proposal)

# Sequence length
MIN_SEQ_LEN = 4
MAX_SEQ_LEN = 10

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("TRAINING 1D BACKFILLING VAE MODEL")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Mask probability: {MASK_PROB} (forward-fill masking)")
print()

# Set random seed for reproducibility
set_seeds(0)
torch.set_default_dtype(torch.float64)

# Load data
print("Loading multifeature data...")
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        f"Please run: python prepare_stock_data_multifeature.py"
    )

data = np.load(DATA_FILE)
amzn_return = data["amzn_return"]  # (N, 1) - Target
extra_features = data["extra_features"]  # (N, 11) - Conditioning

print(f"  AMZN return shape: {amzn_return.shape}")
print(f"  Extra features shape: {extra_features.shape}")
print(f"  Total samples: {len(amzn_return)}")
print()

# Split data: train (4000), valid (1000), test (805)
TRAIN_END = 4000
VALID_END = 5000

amzn_train = amzn_return[:TRAIN_END]
amzn_valid = amzn_return[TRAIN_END:VALID_END]
amzn_test = amzn_return[VALID_END:]

extra_train = extra_features[:TRAIN_END]
extra_valid = extra_features[TRAIN_END:VALID_END]
extra_test = extra_features[VALID_END:]

print("Data splits:")
print(f"  Train: {TRAIN_END} samples")
print(f"  Valid: {VALID_END - TRAIN_END} samples")
print(f"  Test:  {len(amzn_return) - VALID_END} samples")
print()

# Create datasets
print("Creating datasets...")
train_dataset = TimeSeriesDataSetRand(
    (amzn_train, extra_train),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
valid_dataset = TimeSeriesDataSetRand(
    (amzn_valid, extra_valid),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
test_dataset = TimeSeriesDataSetRand(
    (amzn_test, extra_test),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_sampler=CustomBatchSampler1D(train_dataset, BATCH_SIZE, MIN_SEQ_LEN)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_sampler=CustomBatchSampler1D(valid_dataset, 16, MIN_SEQ_LEN)
)
test_loader = DataLoader(
    test_dataset,
    batch_sampler=CustomBatchSampler1D(test_dataset, 16, MIN_SEQ_LEN)
)

print("Datasets created successfully!")
print()

# Model configuration
config = {
    "feat_dim": 1,  # Scalar time series for target (AMZN return)
    "latent_dim": LATENT_DIM,  # Increased to handle 12D state space
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "kl_weight": KL_WEIGHT,
    "ex_feat_weight": 0.0,  # Passive conditioning (no loss on extra features)
    "target_hidden": TARGET_HIDDEN,  # AMZN return embedding
    "ex_feats_dim": 11,  # AMZN vol/volume/range + MSFT×4 + SP500×4
    "ex_feats_hidden": None,  # Identity mapping for extra features
    "mem_type": "lstm",
    "mem_hidden": MEM_HIDDEN,
    "mem_layers": MEM_LAYERS,
    "mem_dropout": MEM_DROPOUT,
    "ctx_target_hidden": TARGET_HIDDEN,
    "ctx_ex_feats_hidden": None,
    "interaction_layers": None,  # Match 2D model (disabled)
    "compress_context": True,
    "ex_loss_type": "l2",
    # Quantile regression
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
}

print("=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)
print(f"Architecture:")
print(f"  latent_dim: {config['latent_dim']} (increased from 5)")
print(f"  target_hidden: {config['target_hidden']}")
print(f"  ex_feats_dim: {config['ex_feats_dim']} (11 conditioning features)")
print(f"  mem_hidden: {config['mem_hidden']}")
print(f"  mem_type: {config['mem_type']}")
print()
print(f"Training:")
print(f"  kl_weight: {config['kl_weight']}")
print(f"  ex_feat_weight: {config['ex_feat_weight']} (passive conditioning)")
print(f"  mask_prob: {MASK_PROB} (forward-fill masking)")
print()
print(f"Quantile regression:")
print(f"  num_quantiles: {config['num_quantiles']}")
print(f"  quantiles: {config['quantiles']}")
print()
print(f"Device: {config['device']}")
print("=" * 80)
print()

# Create model
print("Initializing model...")
model = CVAE1DMemRand(config)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
print()

# Train model
model_name = "backfill_model.pt"
model_path = os.path.join(OUTPUT_DIR, model_name)

if not os.path.exists(model_path):
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Model will be saved to: {model_path}")
    print()

    train(
        model,
        train_loader,
        valid_loader,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        mask_prob=MASK_PROB,  # Enable backfilling training
        model_dir=OUTPUT_DIR,
        file_name=model_name
    )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
else:
    print(f"Model already exists: {model_path}")
    print("Skipping training...")
    print()

# Test model
print("=" * 80)
print("EVALUATING MODEL")
print("=" * 80)

valid_losses, test_losses = test(
    model,
    valid_loader,
    test_loader,
    model_path
)

print()
print("Final Results:")
print(f"  Valid loss: {valid_losses.get('loss', 0):.6f}")
print(f"  Test loss:  {test_losses.get('loss', 0):.6f}")
print()

# Save results
results = {
    "model_name": "backfill_model",
    "description": "Backfilling: AMZN return + 11 multifeatures (3 stocks × 4 features - AMZN return)",
    "latent_dim": config["latent_dim"],
    "ex_feats_dim": config["ex_feats_dim"],
    "target_hidden": str(config["target_hidden"]),
    "mem_hidden": config["mem_hidden"],
    "kl_weight": config["kl_weight"],
    "mask_prob": MASK_PROB,
    "valid_loss": valid_losses.get("loss", 0),
    "valid_target_loss": valid_losses.get("target_loss", 0),
    "valid_kl_loss": valid_losses.get("kl_loss", 0),
    "test_loss": test_losses.get("loss", 0),
    "test_target_loss": test_losses.get("target_loss", 0),
    "test_kl_loss": test_losses.get("kl_loss", 0),
}

results_df = pd.DataFrame([results])
results_path = os.path.join(OUTPUT_DIR, "results.csv")
results_df.to_csv(results_path, index=False)

print(f"Results saved to: {results_path}")
print()
print("=" * 80)
print("ALL DONE")
print("=" * 80)
print()
print("Next steps:")
print("  1. Create generate_1d_backfilling_predictions.py")
print("  2. Create evaluate_1d_backfilling_model.py")
print("  3. Analyze CI calibration and compare with baseline")
