"""
Train 1D Time Series VAE Models for Stock Returns.

Trains 4 model variants:
1. Amazon only (baseline)
2. Amazon + SP500 (no loss on SP500)
3. Amazon + MSFT (no loss on MSFT)
4. Amazon + SP500 + MSFT (no loss on both)

All models use passive conditioning (cond_feat_weight=0) to test whether
external features improve predictions without multi-task learning overhead.
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
DATA_FILE = "data/stock_returns.npz"
OUTPUT_DIR = "models_1d"
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-5

# Training hyperparameters
LATENT_DIM = 5
TARGET_HIDDEN = [32, 32, 32]
MEM_HIDDEN = 100
MEM_LAYERS = 2
MEM_DROPOUT = 0.2
KL_WEIGHT = 1e-5
MIN_SEQ_LEN = 4
MAX_SEQ_LEN = 10

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("TRAINING 1D TIME SERIES VAE MODELS")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print()

# Set random seed for reproducibility
set_seeds(0)
torch.set_default_dtype(torch.float64)

# Load data
print("Loading data...")
data = np.load(DATA_FILE)
amzn_returns = data["amzn_returns"]
sp500_returns = data["sp500_returns"]
msft_returns = data["msft_returns"]
cond_sp500 = data["cond_sp500"]
cond_msft = data["cond_msft"]
cond_both = data["cond_both"]

print(f"  Amazon returns shape: {amzn_returns.shape}")
print(f"  SP500 returns shape: {sp500_returns.shape}")
print(f"  MSFT returns shape: {msft_returns.shape}")
print(f"  Total samples: {len(amzn_returns)}")
print()

# Split data: train (4000), valid (1000), test (824)
TRAIN_END = 4000
VALID_END = 5000

amzn_train = amzn_returns[:TRAIN_END]
amzn_valid = amzn_returns[TRAIN_END:VALID_END]
amzn_test = amzn_returns[VALID_END:]

sp500_train = cond_sp500[:TRAIN_END]
sp500_valid = cond_sp500[TRAIN_END:VALID_END]
sp500_test = cond_sp500[VALID_END:]

msft_train = cond_msft[:TRAIN_END]
msft_valid = cond_msft[TRAIN_END:VALID_END]
msft_test = cond_msft[VALID_END:]

both_train = cond_both[:TRAIN_END]
both_valid = cond_both[TRAIN_END:VALID_END]
both_test = cond_both[VALID_END:]

print("Data splits:")
print(f"  Train: {TRAIN_END} samples")
print(f"  Valid: {VALID_END - TRAIN_END} samples")
print(f"  Test:  {len(amzn_returns) - VALID_END} samples")
print()

# Create datasets and dataloaders

# 1. Amazon only (baseline)
print("Creating datasets: Amazon only...")
train_dataset_amzn = TimeSeriesDataSetRand(
    amzn_train,
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
valid_dataset_amzn = TimeSeriesDataSetRand(
    amzn_valid,
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
test_dataset_amzn = TimeSeriesDataSetRand(
    amzn_test,
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)

train_loader_amzn = DataLoader(
    train_dataset_amzn,
    batch_sampler=CustomBatchSampler1D(train_dataset_amzn, BATCH_SIZE, MIN_SEQ_LEN)
)
valid_loader_amzn = DataLoader(
    valid_dataset_amzn,
    batch_sampler=CustomBatchSampler1D(valid_dataset_amzn, 16, MIN_SEQ_LEN)
)
test_loader_amzn = DataLoader(
    test_dataset_amzn,
    batch_sampler=CustomBatchSampler1D(test_dataset_amzn, 16, MIN_SEQ_LEN)
)

# 2. Amazon + SP500
print("Creating datasets: Amazon + SP500...")
train_dataset_sp500 = TimeSeriesDataSetRand(
    (amzn_train, sp500_train),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
valid_dataset_sp500 = TimeSeriesDataSetRand(
    (amzn_valid, sp500_valid),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
test_dataset_sp500 = TimeSeriesDataSetRand(
    (amzn_test, sp500_test),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)

train_loader_sp500 = DataLoader(
    train_dataset_sp500,
    batch_sampler=CustomBatchSampler1D(train_dataset_sp500, BATCH_SIZE, MIN_SEQ_LEN)
)
valid_loader_sp500 = DataLoader(
    valid_dataset_sp500,
    batch_sampler=CustomBatchSampler1D(valid_dataset_sp500, 16, MIN_SEQ_LEN)
)
test_loader_sp500 = DataLoader(
    test_dataset_sp500,
    batch_sampler=CustomBatchSampler1D(test_dataset_sp500, 16, MIN_SEQ_LEN)
)

# 3. Amazon + MSFT
print("Creating datasets: Amazon + MSFT...")
train_dataset_msft = TimeSeriesDataSetRand(
    (amzn_train, msft_train),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
valid_dataset_msft = TimeSeriesDataSetRand(
    (amzn_valid, msft_valid),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
test_dataset_msft = TimeSeriesDataSetRand(
    (amzn_test, msft_test),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)

train_loader_msft = DataLoader(
    train_dataset_msft,
    batch_sampler=CustomBatchSampler1D(train_dataset_msft, BATCH_SIZE, MIN_SEQ_LEN)
)
valid_loader_msft = DataLoader(
    valid_dataset_msft,
    batch_sampler=CustomBatchSampler1D(valid_dataset_msft, 16, MIN_SEQ_LEN)
)
test_loader_msft = DataLoader(
    test_dataset_msft,
    batch_sampler=CustomBatchSampler1D(test_dataset_msft, 16, MIN_SEQ_LEN)
)

# 4. Amazon + SP500 + MSFT
print("Creating datasets: Amazon + SP500 + MSFT...")
train_dataset_both = TimeSeriesDataSetRand(
    (amzn_train, both_train),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
valid_dataset_both = TimeSeriesDataSetRand(
    (amzn_valid, both_valid),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)
test_dataset_both = TimeSeriesDataSetRand(
    (amzn_test, both_test),
    min_seq_len=MIN_SEQ_LEN,
    max_seq_len=MAX_SEQ_LEN
)

train_loader_both = DataLoader(
    train_dataset_both,
    batch_sampler=CustomBatchSampler1D(train_dataset_both, BATCH_SIZE, MIN_SEQ_LEN)
)
valid_loader_both = DataLoader(
    valid_dataset_both,
    batch_sampler=CustomBatchSampler1D(valid_dataset_both, 16, MIN_SEQ_LEN)
)
test_loader_both = DataLoader(
    test_dataset_both,
    batch_sampler=CustomBatchSampler1D(test_dataset_both, 16, MIN_SEQ_LEN)
)

print("Datasets created successfully!")
print()

# Define model configurations
model_configs = [
    {
        "name": "amzn_only",
        "description": "Amazon only (baseline)",
        "train_loader": train_loader_amzn,
        "valid_loader": valid_loader_amzn,
        "test_loader": test_loader_amzn,
        "cond_feats_dim": 0,
        "cond_feat_weight": 0.0,
    },
    {
        "name": "amzn_sp500_no_loss",
        "description": "Amazon + SP500 (no loss)",
        "train_loader": train_loader_sp500,
        "valid_loader": valid_loader_sp500,
        "test_loader": test_loader_sp500,
        "cond_feats_dim": 1,
        "cond_feat_weight": 0.0,
    },
    {
        "name": "amzn_msft_no_loss",
        "description": "Amazon + MSFT (no loss)",
        "train_loader": train_loader_msft,
        "valid_loader": valid_loader_msft,
        "test_loader": test_loader_msft,
        "cond_feats_dim": 1,
        "cond_feat_weight": 0.0,
    },
    {
        "name": "amzn_both_no_loss",
        "description": "Amazon + SP500 + MSFT (no loss)",
        "train_loader": train_loader_both,
        "valid_loader": valid_loader_both,
        "test_loader": test_loader_both,
        "cond_feats_dim": 2,
        "cond_feat_weight": 0.0,
    },
]

# Results tracking
results = {
    "model_name": [],
    "description": [],
    "cond_feats_dim": [],
    "latent_dim": [],
    "target_hidden": [],
    "mem_hidden": [],
    "kl_weight": [],
    "valid_loss": [],
    "valid_recon_loss": [],
    "valid_target_loss": [],
    "valid_cond_loss": [],
    "valid_kl_loss": [],
    "test_loss": [],
    "test_recon_loss": [],
    "test_target_loss": [],
    "test_cond_loss": [],
    "test_kl_loss": [],
}

# Train each model
for model_config_dict in model_configs:
    print("=" * 80)
    print(f"Training: {model_config_dict['description']}")
    print("=" * 80)

    set_seeds(0)

    model_name = model_config_dict["name"] + ".pt"
    model_path = os.path.join(OUTPUT_DIR, model_name)

    # Model configuration
    config = {
        "feat_dim": 1,  # Scalar time series
        "latent_dim": LATENT_DIM,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "kl_weight": KL_WEIGHT,
        "cond_feat_weight": model_config_dict["cond_feat_weight"],
        "target_hidden": TARGET_HIDDEN,
        "cond_feats_dim": model_config_dict["cond_feats_dim"],
        "cond_feats_hidden": None,  # Identity mapping for conditioning
        "mem_type": "lstm",
        "mem_hidden": MEM_HIDDEN,
        "mem_layers": MEM_LAYERS,
        "mem_dropout": MEM_DROPOUT,
        "ctx_target_hidden": TARGET_HIDDEN,
        "ctx_cond_feats_hidden": None,
        "interaction_layers": 2,  # Nonlinear mixing layers
        "compress_context": True,
        "cond_loss_type": "l2",
        # Quantile regression
        "num_quantiles": 3,
        "quantiles": [0.05, 0.5, 0.95],
    }

    print(f"Model configuration:")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  target_hidden: {config['target_hidden']}")
    print(f"  cond_feats_dim: {config['cond_feats_dim']}")
    print(f"  mem_hidden: {config['mem_hidden']}")
    print(f"  kl_weight: {config['kl_weight']}")
    print(f"  cond_feat_weight: {config['cond_feat_weight']}")
    print(f"  num_quantiles: {config['num_quantiles']}")
    print(f"  device: {config['device']}")
    print()

    # Create model
    model = CVAE1DMemRand(config)

    # Train model (skip if already trained)
    if not os.path.exists(model_path):
        print(f"Training model: {model_name}")
        train(
            model,
            model_config_dict["train_loader"],
            model_config_dict["valid_loader"],
            epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            model_dir=OUTPUT_DIR,
            file_name=model_name
        )
    else:
        print(f"Model already exists: {model_name}")
        print("Skipping training...")
        print()

    # Test model
    print(f"Evaluating model: {model_name}")
    valid_losses, test_losses = test(
        model,
        model_config_dict["valid_loader"],
        model_config_dict["test_loader"],
        model_path
    )

    # Record results
    results["model_name"].append(model_config_dict["name"])
    results["description"].append(model_config_dict["description"])
    results["cond_feats_dim"].append(config["cond_feats_dim"])
    results["latent_dim"].append(config["latent_dim"])
    results["target_hidden"].append(str(config["target_hidden"]))
    results["mem_hidden"].append(config["mem_hidden"])
    results["kl_weight"].append(config["kl_weight"])

    results["valid_loss"].append(valid_losses.get("loss", 0))
    results["valid_recon_loss"].append(valid_losses.get("reconstruction_loss", 0))
    results["valid_target_loss"].append(valid_losses.get("re_surface", 0))  # Note: utils uses "re_surface" key
    results["valid_cond_loss"].append(valid_losses.get("re_ex_feats", 0))
    results["valid_kl_loss"].append(valid_losses.get("kl_loss", 0))

    results["test_loss"].append(test_losses.get("loss", 0))
    results["test_recon_loss"].append(test_losses.get("reconstruction_loss", 0))
    results["test_target_loss"].append(test_losses.get("re_surface", 0))
    results["test_cond_loss"].append(test_losses.get("re_ex_feats", 0))
    results["test_kl_loss"].append(test_losses.get("kl_loss", 0))

    print()
    print("Results:")
    print(f"  Valid loss: {valid_losses.get('loss', 0):.6f}")
    print(f"  Test loss:  {test_losses.get('loss', 0):.6f}")
    print()

# Save results to CSV
results_df = pd.DataFrame(results)
results_path = os.path.join(OUTPUT_DIR, "results.csv")
results_df.to_csv(results_path, index=False)

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Results saved to: {results_path}")
print()
print("Summary:")
print(results_df[["model_name", "description", "test_loss", "test_target_loss", "test_kl_loss"]].to_string(index=False))
print()
print("Next steps:")
print("  1. Run: python generate_1d_predictions.py")
print("  2. Visualize results and compare model performance")
