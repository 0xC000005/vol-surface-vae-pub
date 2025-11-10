"""
Sanity check for 1D backfilling model (multi-channel architecture).

Tests:
1. Model instantiation with 12D multi-channel target (target_dim=12, ex_feats_dim=0)
2. Forward pass with unified 12D target
3. Masked training step (forward-fill channel 0 only)
4. Multi-channel output shapes and data flow
"""

import numpy as np
import torch
from vae.cvae_1d_with_mem_randomized import CVAE1DMemRand
from vae.utils import set_seeds

print("=" * 80)
print("SANITY TEST: 1D BACKFILLING MODEL")
print("=" * 80)
print()

# Set random seed
set_seeds(0)
torch.set_default_dtype(torch.float64)

# Test configuration (multi-channel architecture)
config = {
    "target_dim": 12,  # Multi-channel target: 3 stocks × 4 features
    "latent_dim": 12,  # Increased for 12D state space
    "device": "cpu",
    "kl_weight": 1e-5,
    "ex_feat_weight": 0.0,
    "target_hidden": [32, 32],
    "ex_feats_dim": 0,  # No additional conditioning (all features in target)
    "ex_feats_hidden": None,
    "mem_type": "lstm",
    "mem_hidden": 32,
    "mem_layers": 2,
    "mem_dropout": 0.2,
    "ctx_target_hidden": [32, 32],
    "ctx_ex_feats_hidden": None,
    "interaction_layers": None,
    "compress_context": True,
    "ex_loss_type": "l2",
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
    # Multi-channel loss and masking
    "target_loss_on_channel_0_only": True,
    "mask_channel_0_only": True,
}

print("Test 1: Model Instantiation")
print("-" * 80)
print(f"Config: target_dim={config['target_dim']}, latent_dim={config['latent_dim']}, ex_feats_dim={config['ex_feats_dim']}")

try:
    model = CVAE1DMemRand(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {num_params}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    exit(1)

print()

# Test 2: Forward pass
print("Test 2: Forward Pass with 12D Input")
print("-" * 80)

B = 8  # Batch size
T = 5  # Sequence length (4 context + 1 future)

# Create dummy data - unified 12D target
target = torch.randn(B, T, 12)  # All 12 stock features

batch = {"target": target}  # No ex_feats

print(f"Batch shapes:")
print(f"  target: {target.shape} (12D: 3 stocks × 4 features)")
print()

try:
    decoded_target, decoded_ex_feats, z_mean, z_log_var, z = model.forward(batch)

    print(f"✓ Forward pass successful")
    print(f"Output shapes:")
    print(f"  decoded_target: {decoded_target.shape} (expected: (B, 1, num_quantiles, target_dim) = (8, 1, 3, 12))")
    print(f"  decoded_ex_feats: {decoded_ex_feats} (expected: None)")
    print(f"  z_mean: {z_mean.shape} (expected: (B, T, latent_dim) = (8, 5, 12))")
    print(f"  z: {z.shape} (expected: (B, T, latent_dim) = (8, 5, 12))")

    # Verify shapes
    assert decoded_target.shape == (B, 1, 3, 12), f"Wrong decoded_target shape: {decoded_target.shape}"
    assert decoded_ex_feats is None, f"Expected None for decoded_ex_feats"
    assert z_mean.shape == (B, T, 12), f"Wrong z_mean shape: {z_mean.shape}"
    assert z.shape == (B, T, 12), f"Wrong z shape: {z.shape}"

    print(f"✓ All output shapes correct")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 3: Masked training step
print("Test 3: Masked Training Step (Forward-Fill)")
print("-" * 80)

if not hasattr(model, 'train_step_masked'):
    print("✗ Model does not have train_step_masked() method")
    exit(1)

try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Standard training step
    losses_standard = model.train_step(batch, optimizer)
    print(f"✓ Standard training step successful")
    print(f"  Standard losses: {', '.join([f'{k}: {v.item():.3f}' for k, v in losses_standard.items()])}")

    # Masked training step
    losses_masked = model.train_step_masked(batch, optimizer)
    print(f"✓ Masked training step successful")
    print(f"  Masked losses: {', '.join([f'{k}: {v.item():.3f}' for k, v in losses_masked.items()])}")

    # Verify both have the same keys
    assert set(losses_standard.keys()) == set(losses_masked.keys()), "Loss dict keys don't match"
    print(f"✓ Both training modes produce consistent loss dictionaries")

except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 4: Generation (backfilling scenario)
print("Test 4: Generation with Masked Context")
print("-" * 80)

try:
    # Create context (historical data)
    C = T - 1  # Context length
    ctx_dict = {
        "target": target[:, :C, :],  # All features [0:C]
    }

    # Generate prediction
    model.eval()
    with torch.no_grad():
        prediction = model.get_prediction_given_context(ctx_dict)

    print(f"✓ Generation successful")
    print(f"  Prediction shape: {prediction.shape} (expected: (B, num_quantiles, target_dim) = (8, 3, 12))")
    print(f"  Channel 0 (AMZN) quantiles (batch 0): p05={prediction[0, 0, 0].item():.4f}, "
          f"p50={prediction[0, 1, 0].item():.4f}, p95={prediction[0, 2, 0].item():.4f}")

    # Check monotonicity for channel 0 (AMZN) only: p05 <= p50 <= p95
    non_monotonic_count = 0
    for i in range(B):
        p05 = prediction[i, 0, 0]  # Channel 0, quantile 0
        p50 = prediction[i, 1, 0]  # Channel 0, quantile 1
        p95 = prediction[i, 2, 0]  # Channel 0, quantile 2
        if not (p05 <= p50 <= p95):
            non_monotonic_count += 1

    if non_monotonic_count > 0:
        print(f"⚠ Warning: {non_monotonic_count}/{B} samples have non-monotonic quantiles on channel 0 (expected for untrained model)")
        print(f"  This will be resolved during training via quantile loss function")
    else:
        print(f"✓ All channel 0 quantiles are monotonic (p05 ≤ p50 ≤ p95)")

except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Final summary
print("=" * 80)
print("ALL SANITY TESTS PASSED ✓")
print("=" * 80)
print()
print("Model is ready for training with:")
print(f"  - target_dim: {config['target_dim']} (12D multi-channel)")
print(f"  - latent_dim: {config['latent_dim']}")
print(f"  - ex_feats_dim: {config['ex_feats_dim']} (no additional conditioning)")
print(f"  - target_loss_on_channel_0_only: {config['target_loss_on_channel_0_only']}")
print(f"  - mask_channel_0_only: {config['mask_channel_0_only']}")
print(f"  - Total parameters: {num_params}")
print()
print("Next step:")
print("  Run: python train_1d_backfilling_model.py")
