"""
Train full quantile regression models (no_ex, ex_no_loss, ex_loss).
Based on param_search.py but with quantile regression decoder enabled.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import *
import os

print("=" * 60)
print("QUANTILE REGRESSION - FULL MODEL TRAINING")
print("=" * 60)

set_seeds(0)
torch.set_default_dtype(torch.float64)
num_epochs = 500

# Load data
print("\n1. Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
level_data = data["levels"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)
print(f"   - Total data shape: {vol_surf_data.shape}")
print(f"   - Extra features shape: {ex_data.shape}")

# Create datasets (no_ex: surface only)
print("\n2. Creating datasets...")
train_dataset = VolSurfaceDataSetRand(vol_surf_data[:4000])
valid_dataset = VolSurfaceDataSetRand(vol_surf_data[4000:5000])
test_dataset = VolSurfaceDataSetRand(vol_surf_data[5000:])
train_batch_sampler = CustomBatchSampler(train_dataset, 64)
valid_batch_sampler = CustomBatchSampler(valid_dataset, 16)
test_batch_sampler = CustomBatchSampler(test_dataset, 16)
train_simple = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
valid_simple = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler)
test_simple = DataLoader(test_dataset, batch_sampler=test_batch_sampler)

# Create datasets (ex models: surface + extra features)
train_dataset2 = VolSurfaceDataSetRand((vol_surf_data[:4000], ex_data[:4000]))
valid_dataset2 = VolSurfaceDataSetRand((vol_surf_data[4000:5000], ex_data[4000:5000]))
test_dataset2 = VolSurfaceDataSetRand((vol_surf_data[5000:], ex_data[5000:]))
train_batch_sampler2 = CustomBatchSampler(train_dataset2, 64)
valid_batch_sampler2 = CustomBatchSampler(valid_dataset2, 16)
test_batch_sampler2 = CustomBatchSampler(test_dataset2, 16)
train_ex = DataLoader(train_dataset2, batch_sampler=train_batch_sampler2)
valid_ex = DataLoader(valid_dataset2, batch_sampler=valid_batch_sampler2)
test_ex = DataLoader(test_dataset2, batch_sampler=test_batch_sampler2)

print(f"   - Train: 4000 days")
print(f"   - Valid: 1000 days")
print(f"   - Test: {vol_surf_data.shape[0] - 5000} days")

# Hyperparameters
conv_param_grid = {
    "latent_dim": 5,
    "surface_hidden": [5, 5, 5],
    "mem_hidden": 100,
    "kl_weight": 1e-5,
}

# Save to new folder for quantile models
base_folder_name = "test_spx/quantile_regression"
os.makedirs(base_folder_name, exist_ok=True)
print(f"\n3. Saving models to: {base_folder_name}/")

# Results tracking
df = {
    "fn": [],
    "latent_dim": [],
    "surface_hidden": [],
    "mem_hidden": [],
    "kl_weight": [],
    "dev_loss": [],
    "dev_re_surface": [],
    "dev_re_ex_feats": [],
    "dev_re_loss": [],
    "dev_kl_loss": [],
    "test_loss": [],
    "test_re_surface": [],
    "test_re_ex_feats": [],
    "test_re_loss": [],
    "test_kl_loss": [],
}

# Model variants
params = [
    {"model_name": "no_ex", "train_data": train_simple, "valid_data": valid_simple, "test_data": test_simple},
    {"model_name": "ex_no_loss", "train_data": train_ex, "valid_data": valid_ex, "test_data": test_ex},
    {"model_name": "ex_loss", "train_data": train_ex, "valid_data": valid_ex, "test_data": test_ex},
]

print("\n4. Training models...")
print("=" * 60)

for i, param in enumerate(params):
    set_seeds(0)

    model_name = param["model_name"] + ".pt"
    train_data = param["train_data"]
    valid_data = param["valid_data"]
    test_data = param["test_data"]

    print(f"\n>>> Training model {i+1}/3: {param['model_name']}")
    print("-" * 60)

    config = {
        "feat_dim": (5, 5),
        "latent_dim": conv_param_grid["latent_dim"],
        "device": "cuda",
        "kl_weight": conv_param_grid["kl_weight"],
        "re_feat_weight": 1.0 if param["model_name"] == "ex_loss" else 0.0,
        "surface_hidden": conv_param_grid["surface_hidden"],
        "ex_feats_dim": 0 if param["model_name"] == "no_ex" else ex_data.shape[-1],
        "ex_feats_hidden": None,
        "mem_type": "lstm",
        "mem_hidden": conv_param_grid["mem_hidden"],
        "mem_layers": 2,
        "mem_dropout": 0.2,
        "ctx_surface_hidden": conv_param_grid["surface_hidden"],
        "ctx_ex_feats_hidden": None,
        "interaction_layers": None,
        "use_dense_surface": False,
        "compress_context": True,
        "ex_loss_on_ret_only": True,
        "ex_feats_loss_type": "l2",
        # Quantile regression configuration
        "use_quantile_regression": True,
        "num_quantiles": 3,
        "quantiles": [0.05, 0.5, 0.95],
    }

    print(f"   Config:")
    print(f"   - Latent dim: {config['latent_dim']}")
    print(f"   - Mem hidden: {config['mem_hidden']}")
    print(f"   - Extra features: {config['ex_feats_dim']}")
    print(f"   - Feature loss weight: {config['re_feat_weight']}")
    print(f"   - Quantile regression: {config['use_quantile_regression']}")
    print(f"   - Quantiles: {config['quantiles']}")
    print()

    model = CVAEMemRand(config)
    if not os.path.exists(f"{base_folder_name}/{model_name}"):
        print(f"   Starting training ({num_epochs} epochs)...")
        train(model, train_data, valid_data, epochs=num_epochs, lr=1e-05, model_dir=base_folder_name, file_name=model_name)
        print(f"   ✓ Training complete!")
    else:
        print(f"   Model already exists, skipping training")

    print(f"   Evaluating on validation and test sets...")
    dev_losses, test_losses = test(model, valid_data, test_data, f"{base_folder_name}/{model_name}")

    df["fn"].append(model_name)
    df["latent_dim"].append(conv_param_grid["latent_dim"])
    df["surface_hidden"].append(conv_param_grid["surface_hidden"])
    df["mem_hidden"].append(conv_param_grid["mem_hidden"])
    df["kl_weight"].append(conv_param_grid["kl_weight"])
    df["dev_loss"].append(dev_losses["loss"])
    df["dev_re_surface"].append(dev_losses["re_surface"])
    df["dev_re_ex_feats"].append(dev_losses["re_ex_feats"])
    df["dev_re_loss"].append(dev_losses["reconstruction_loss"])
    df["dev_kl_loss"].append(dev_losses["kl_loss"])
    df["test_loss"].append(test_losses["loss"])
    df["test_re_surface"].append(test_losses["re_surface"])
    df["test_re_ex_feats"].append(test_losses["re_ex_feats"])
    df["test_re_loss"].append(test_losses["reconstruction_loss"])
    df["test_kl_loss"].append(test_losses["kl_loss"])

    print(f"   Validation loss: {dev_losses['loss']:.6f}")
    print(f"   Test loss: {test_losses['loss']:.6f}")

# Save results
df = pd.DataFrame(df)
df.to_csv(f"{base_folder_name}/results.csv", index=False)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nResults saved to: {base_folder_name}/results.csv")
print("\nModel performance:")
print(df[["fn", "test_loss", "test_re_surface", "test_kl_loss"]].to_string(index=False))
print("\n✓ All models trained successfully!")
print("=" * 60)
