"""
Sanity check for 1D backfilling model.

Tests:
1. Model instantiation with 12D input (latent_dim=12, ex_feats_dim=11)
2. Forward pass with multifeature data
3. Masked training step (forward-fill masking)
4. Output shapes and data flow
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

# Test configuration (from BACKFILLING_PROPOSAL.md)
config = {
    "feat_dim": 1,
    "latent_dim": 12,  # Increased for 12D state space
    "device": "cpu",
    "kl_weight": 1e-5,
    "ex_feat_weight": 0.0,
    "target_hidden": [32, 32],
    "ex_feats_dim": 11,  # 11 conditioning features
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
}

print("Test 1: Model Instantiation")
print("-" * 80)
print(f"Config: latent_dim={config['latent_dim']}, ex_feats_dim={config['ex_feats_dim']}")

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

# Create dummy data
target = torch.randn(B, T, 1)  # AMZN return (1D)
ex_feats = torch.randn(B, T, 11)  # 11 conditioning features

batch = {
    "target": target,
    "ex_feats": ex_feats
}

print(f"Batch shapes:")
print(f"  target: {target.shape}")
print(f"  ex_feats: {ex_feats.shape}")
print()

try:
    decoded_target, decoded_ex_feats, z_mean, z_log_var, z = model.forward(batch)

    print(f"✓ Forward pass successful")
    print(f"Output shapes:")
    print(f"  decoded_target: {decoded_target.shape} (expected: (B, 1, num_quantiles) = (8, 1, 3))")
    print(f"  z_mean: {z_mean.shape} (expected: (B, T, latent_dim) = (8, 5, 12))")
    print(f"  z: {z.shape} (expected: (B, T, latent_dim) = (8, 5, 12))")

    # Verify shapes
    assert decoded_target.shape == (B, 1, 3), f"Wrong decoded_target shape: {decoded_target.shape}"
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
        "target": target[:, :C, :],  # AMZN[0:C]
        "ex_feats": ex_feats[:, :C, :],  # Market data[0:C]
    }

    # Generate prediction
    model.eval()
    with torch.no_grad():
        prediction = model.get_prediction_given_context(ctx_dict)

    print(f"✓ Generation successful")
    print(f"  Prediction shape: {prediction.shape} (expected: (B, num_quantiles) = (8, 3))")
    print(f"  Quantile values (batch 0): p05={prediction[0, 0].item():.4f}, "
          f"p50={prediction[0, 1].item():.4f}, p95={prediction[0, 2].item():.4f}")

    # Check monotonicity (p05 <= p50 <= p95) - expected to fail for untrained model
    non_monotonic_count = 0
    for i in range(B):
        if not (prediction[i, 0] <= prediction[i, 1] <= prediction[i, 2]):
            non_monotonic_count += 1

    if non_monotonic_count > 0:
        print(f"⚠ Warning: {non_monotonic_count}/{B} samples have non-monotonic quantiles (expected for untrained model)")
        print(f"  This will be resolved during training via quantile loss function")
    else:
        print(f"✓ All quantiles are monotonic (p05 ≤ p50 ≤ p95)")

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
print(f"  - latent_dim: {config['latent_dim']}")
print(f"  - ex_feats_dim: {config['ex_feats_dim']}")
print(f"  - Total parameters: {num_params}")
print()
print("Next step:")
print("  Run: python train_1d_backfilling_model.py")
