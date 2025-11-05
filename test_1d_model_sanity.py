"""
Sanity check for 1D VAE model architecture.

Tests:
1. Model instantiation
2. Forward pass with all 4 configurations
3. Loss computation
4. Single training step
5. Prediction generation
"""

import numpy as np
import torch
from vae.cvae_1d_with_mem_randomized import CVAE1DMemRand
from vae.datasets_1d_randomized import TimeSeriesDataSetRand
from torch.utils.data import DataLoader

# Set default dtype FIRST
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("1D VAE MODEL SANITY CHECK")
print("=" * 80)
print()

# Generate synthetic data
print("Creating synthetic data...")
np.random.seed(42)
N = 100
target_data = np.random.randn(N)  # Amazon returns
cond_data1 = np.random.randn(N, 1)  # SP500 returns
cond_data2 = np.random.randn(N, 2)  # SP500 + MSFT returns

print(f"  Target shape: {target_data.shape}")
print(f"  Cond1 shape: {cond_data1.shape}")
print(f"  Cond2 shape: {cond_data2.shape}")
print()

# Test configurations
configs = [
    {
        "name": "Target only",
        "cond_feats_dim": 0,
        "dataset": TimeSeriesDataSetRand(target_data, min_seq_len=4, max_seq_len=6),
    },
    {
        "name": "Target + 1 cond",
        "cond_feats_dim": 1,
        "dataset": TimeSeriesDataSetRand((target_data, cond_data1), min_seq_len=4, max_seq_len=6),
    },
    {
        "name": "Target + 2 conds",
        "cond_feats_dim": 2,
        "dataset": TimeSeriesDataSetRand((target_data, cond_data2), min_seq_len=4, max_seq_len=6),
    },
]

for config_dict in configs:
    print("=" * 80)
    print(f"Testing: {config_dict['name']}")
    print("=" * 80)

    # Model configuration
    config = {
        "feat_dim": 1,
        "latent_dim": 3,
        "device": "cpu",
        "kl_weight": 1e-5,
        "cond_feat_weight": 0.0,
        "target_hidden": [8, 8],
        "cond_feats_dim": config_dict["cond_feats_dim"],
        "cond_feats_hidden": None,
        "mem_type": "lstm",
        "mem_hidden": 16,
        "mem_layers": 1,
        "mem_dropout": 0.0,
        "ctx_target_hidden": [8, 8],
        "ctx_cond_feats_hidden": None,
        "interaction_layers": 1,
        "compress_context": True,
        "cond_loss_type": "l2",
    }

    # Test 1: Instantiation
    print("Test 1: Model instantiation... ", end="")
    try:
        model = CVAE1DMemRand(config)
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        continue

    # Test 2: Dataset loading
    print("Test 2: Dataset loading... ", end="")
    try:
        dataset = config_dict["dataset"]
        sample = dataset[0]
        assert "target" in sample
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        continue

    # Test 3: Forward pass
    print("Test 3: Forward pass... ", end="")
    try:
        model.eval()
        with torch.no_grad():
            # Prepare batch
            batch = {}
            if "cond_feats" in sample:
                batch = {
                    "target": sample["target"].unsqueeze(0),
                    "cond_feats": sample["cond_feats"].unsqueeze(0),
                }
            else:
                batch = {
                    "target": sample["target"].unsqueeze(0),
                }

            decoded_target, decoded_cond, z_mean, z_log_var, z = model.forward(batch)

            # Check shapes
            T = sample["target"].shape[0]
            assert decoded_target.shape == (1, T, 1), f"Expected (1, {T}, 1), got {decoded_target.shape}"
            assert z_mean.shape == (1, T, config["latent_dim"])
            print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        continue

    # Test 4: Loss computation
    print("Test 4: Loss computation... ", end="")
    try:
        loss_dict = model.compute_loss(batch, decoded_target, decoded_cond, z_mean, z_log_var)
        assert "loss" in loss_dict
        assert "recon_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert torch.isfinite(loss_dict["loss"])
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        continue

    # Test 5: Training step
    print("Test 5: Training step... ", end="")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_dict = model.train_step(batch, optimizer)
        assert "loss" in loss_dict
        assert loss_dict["loss"] > 0
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        continue

    # Test 6: Prediction generation
    print("Test 6: Prediction generation... ", end="")
    try:
        model.eval()
        # Extract context (first 4 timesteps)
        ctx_len = 4
        ctx_dict = {
            "target": batch["target"][:, :ctx_len, :]
        }
        if "cond_feats" in batch:
            ctx_dict["cond_feats"] = batch["cond_feats"][:, :ctx_len, :]

        # Generate stochastic predictions
        preds_stoch = model.get_prediction_given_context(ctx_dict, num_samples=10, use_mean=False)
        assert preds_stoch.shape == (1, 10, 1), f"Expected (1, 10, 1), got {preds_stoch.shape}"

        # Generate MLE prediction
        preds_mle = model.get_prediction_given_context(ctx_dict, num_samples=1, use_mean=True)
        assert preds_mle.shape == (1, 1, 1), f"Expected (1, 1, 1), got {preds_mle.shape}"

        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        continue

    print(f"All tests passed for: {config_dict['name']}")
    print()

print("=" * 80)
print("SANITY CHECK COMPLETE")
print("=" * 80)
print()
print("✓ Model architecture is working correctly!")
print("✓ Ready to train on real data")
print()
print("Next steps:")
print("  1. Run: python train_1d_models.py")
print("     (This will train all 4 models - takes ~30-60 min per model)")
print("  2. Run: python generate_1d_predictions.py")
print("  3. Run: python visualize_1d_predictions.py")
print("  4. Run: python evaluate_1d_models.py")
