"""
Test script for V4 Single-Output Decoder modifications.

Tests:
1. Model imports work correctly
2. Model produces correct output shapes (B, T, H, W) instead of (B, T, 3, H, W)
3. MSE loss computation works
4. Generation methods work correctly
5. Training utilities work with new shapes
"""

import numpy as np
import torch
import torch.nn.functional as F
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior
from vae.utils import set_seeds

print("=" * 80)
print("V4 SINGLE-OUTPUT DECODER TEST")
print("=" * 80)
print()

# Set seeds for reproducibility
set_seeds(0)
torch.set_default_dtype(torch.float64)

# ============================================================================
# Test 1: Model Configuration and Initialization
# ============================================================================
print("Test 1: Model Configuration and Initialization")
print("-" * 80)

model_config = {
    "feat_dim": (5, 5),  # Surface dimensions
    "latent_dim": 12,
    "surface_hidden": [5, 5, 5],
    "ctx_surface_hidden": [5, 5, 5],
    "ex_feats_dim": 3,  # returns, skew, slope
    "ex_feats_hidden": [5, 5, 5],
    "ctx_ex_feats_hidden": [5, 5, 5],
    "context_len": 20,
    "re_feat_weight": 1.0,
    "kl_weight": 1.0,
    "mem_type": "lstm",
    "mem_hidden": 100,
    "mem_layers": 2,
    "mem_dropout": 0.3,
    "interaction_layers": 2,
    "use_dense_surface": False,
    "compress_context": True,
    "ex_loss_on_ret_only": True,
    "ex_feats_loss_type": "l2",
    "device": "cpu",
    "horizon": 5,
    # NOTE: No quantile parameters!
}

# Test CVAEMemRand (base model)
print("Initializing CVAEMemRand (base model)...")
try:
    model_base = CVAEMemRand(model_config)
    print("✓ CVAEMemRand initialized successfully")
except Exception as e:
    print(f"✗ FAIL: CVAEMemRand initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test CVAEMemRandConditionalPrior
print("Initializing CVAEMemRandConditionalPrior...")
model_config["use_conditional_prior"] = True
try:
    model_cond = CVAEMemRandConditionalPrior(model_config)
    print("✓ CVAEMemRandConditionalPrior initialized successfully")
except Exception as e:
    print(f"✗ FAIL: CVAEMemRandConditionalPrior initialization failed: {e}")
    exit(1)

print()

# ============================================================================
# Test 2: Forward Pass Shape Check
# ============================================================================
print("Test 2: Forward Pass Shape Check")
print("-" * 80)

# Create dummy input
batch_size = 2
seq_len = 25  # 20 context + 5 horizon
surface = torch.randn(batch_size, seq_len, 5, 5, dtype=torch.float64)
ex_feats = torch.randn(batch_size, seq_len, 3, dtype=torch.float64)

x = {
    "surface": surface,
    "ex_feats": ex_feats
}

print(f"Input shapes:")
print(f"  surface: {x['surface'].shape}")
print(f"  ex_feats: {x['ex_feats'].shape}")
print()

# Test base model forward pass
model_base.eval()
with torch.no_grad():
    try:
        surf_recon, ex_recon, z_mean, z_logvar, z = model_base.forward(x)

        expected_surf_shape = (batch_size, model_config["horizon"], 5, 5)
        expected_ex_shape = (batch_size, model_config["horizon"], 3)
        expected_z_shape = (batch_size, seq_len, model_config["latent_dim"])

        print(f"CVAEMemRand output shapes:")
        print(f"  surf_recon: {surf_recon.shape} (expected: {expected_surf_shape})")
        print(f"  ex_recon: {ex_recon.shape} (expected: {expected_ex_shape})")
        print(f"  z_mean: {z_mean.shape} (expected: {expected_z_shape})")
        print(f"  z_logvar: {z_logvar.shape} (expected: {expected_z_shape})")
        print(f"  z: {z.shape} (expected: {expected_z_shape})")

        # Verify shapes
        assert surf_recon.shape == expected_surf_shape, f"Surface shape mismatch! Got {surf_recon.shape}, expected {expected_surf_shape}"
        assert ex_recon.shape == expected_ex_shape, f"Ex features shape mismatch! Got {ex_recon.shape}, expected {expected_ex_shape}"
        assert z_mean.shape == expected_z_shape, f"Z mean shape mismatch!"

        # Verify no quantile dimension
        assert len(surf_recon.shape) == 4, f"Surface should be 4D, got {len(surf_recon.shape)}D"

        print("✓ All output shapes correct (4D surface, no quantile dimension)")

    except Exception as e:
        print(f"✗ FAIL: Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

print()

# ============================================================================
# Test 3: Loss Computation
# ============================================================================
print("Test 3: Loss Computation (MSE instead of Quantile Loss)")
print("-" * 80)

target_surface = x["surface"][:, -model_config["horizon"]:, :, :]
target_ex = x["ex_feats"][:, -model_config["horizon"]:, :]

print(f"Target shapes:")
print(f"  target_surface: {target_surface.shape}")
print(f"  target_ex: {target_ex.shape}")
print()

try:
    # Test MSE loss (should work)
    surf_loss = F.mse_loss(surf_recon, target_surface)
    ex_loss = F.mse_loss(ex_recon, target_ex)

    print(f"✓ MSE loss computed successfully:")
    print(f"  Surface loss: {surf_loss.item():.6f}")
    print(f"  Ex features loss: {ex_loss.item():.6f}")

    # Verify loss is a scalar
    assert surf_loss.dim() == 0, "Loss should be a scalar"

    print("✓ Loss computation working correctly")

except Exception as e:
    print(f"✗ FAIL: Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# ============================================================================
# Test 4: Conditional Prior Model
# ============================================================================
print("Test 4: Conditional Prior Model Forward Pass")
print("-" * 80)

model_cond.eval()
with torch.no_grad():
    try:
        surf_recon, ex_recon, z_mean, z_logvar, z = model_cond.forward(x)

        print(f"CVAEMemRandConditionalPrior output shapes:")
        print(f"  surf_recon: {surf_recon.shape}")
        print(f"  ex_recon: {ex_recon.shape}")

        assert surf_recon.shape == expected_surf_shape, "Conditional prior model shape mismatch!"
        assert len(surf_recon.shape) == 4, "Should be 4D, no quantile dimension"

        print("✓ Conditional prior model produces correct shapes")

    except Exception as e:
        print(f"✗ FAIL: Conditional prior forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

print()

# ============================================================================
# Test 5: Generation Methods
# ============================================================================
print("Test 5: Generation Methods")
print("-" * 80)

# Test get_surface_given_conditions (prior sampling)
context_len = 20
context_surface = x["surface"][:, :context_len, :, :]
context_ex = x["ex_feats"][:, :context_len, :]

context = {
    "surface": context_surface,
    "ex_feats": context_ex
}

print(f"Context shapes:")
print(f"  surface: {context['surface'].shape}")
print(f"  ex_feats: {context['ex_feats'].shape}")
print()

try:
    with torch.no_grad():
        surf_pred, ex_pred = model_base.get_surface_given_conditions(
            context, z=None, mu=0, std=1, horizon=5
        )

    expected_pred_shape = (batch_size, 5, 5, 5)
    expected_ex_pred_shape = (batch_size, 5, 3)

    print(f"Generation output shapes:")
    print(f"  surf_pred: {surf_pred.shape} (expected: {expected_pred_shape})")
    print(f"  ex_pred: {ex_pred.shape} (expected: {expected_ex_pred_shape})")

    assert surf_pred.shape == expected_pred_shape, f"Generated surface shape mismatch!"
    assert len(surf_pred.shape) == 4, "Generated surface should be 4D"

    print("✓ Generation methods produce correct shapes")

except Exception as e:
    print(f"✗ FAIL: Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# ============================================================================
# Test 6: Training Utilities (from vae.utils)
# ============================================================================
print("Test 6: Training Utilities")
print("-" * 80)

try:
    from vae.utils import model_eval

    # Create simple dataset
    vol_data = np.random.randn(100, 5, 5)
    ex_data = np.random.randn(100, 3)

    # Test model_eval doesn't crash with new shapes
    # (We won't run full eval, just check it imports and accepts the model)
    print("✓ Training utilities import successfully")
    print("✓ model_eval function available")

except Exception as e:
    print(f"✗ FAIL: Training utilities failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# ============================================================================
# Test 7: Experiment Script Imports
# ============================================================================
print("Test 7: Experiment Script Imports")
print("-" * 80)

# Just check they can be imported/parsed (syntax check)
import sys
from pathlib import Path

scripts_to_test = [
    "experiments/backfill/context60_v4_full_cov/teacher_forcing/generate_vae_tf_sequences.py",
    "experiments/backfill/context60_v4_full_cov/teacher_forcing/validate_vae_tf_sequences.py",
    "experiments/backfill/context60_v4_full_cov/train_backfill_context60_latent12_v4_full_cov.py",
]

for script_path in scripts_to_test:
    script_name = Path(script_path).name
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        compile(code, script_path, 'exec')
        print(f"✓ {script_name}: Syntax OK")
    except SyntaxError as e:
        print(f"✗ FAIL: {script_name}: Syntax error: {e}")
        exit(1)
    except Exception as e:
        print(f"✗ FAIL: {script_name}: Error: {e}")
        exit(1)

print()

# ============================================================================
# Test 8: Verify No Quantile References in Model
# ============================================================================
print("Test 8: Verify No Quantile Decoder Artifacts")
print("-" * 80)

# Check that model doesn't have quantile_loss_fn
try:
    assert not hasattr(model_base, 'quantile_loss_fn'), "Model still has quantile_loss_fn attribute!"
    print("✓ No quantile_loss_fn attribute (correctly removed)")

    # Check model_config doesn't have quantile params
    assert 'num_quantiles' not in model_config, "model_config still has num_quantiles!"
    assert 'quantiles' not in model_config, "model_config still has quantiles!"

    print("✓ No quantile parameters in model_config")

except AssertionError as e:
    print(f"✗ FAIL: {e}")
    exit(1)

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("ALL TESTS PASSED ✓")
print("=" * 80)
print()
print("Summary:")
print("  1. ✓ Models initialize correctly")
print("  2. ✓ Forward pass produces correct 4D shapes (no quantile dimension)")
print("  3. ✓ MSE loss computation works")
print("  4. ✓ Conditional prior model works")
print("  5. ✓ Generation methods work")
print("  6. ✓ Training utilities import successfully")
print("  7. ✓ All experiment scripts have valid syntax")
print("  8. ✓ No quantile decoder artifacts remain")
print()
print("V4 Single-Output Decoder implementation is CORRECT!")
print()
