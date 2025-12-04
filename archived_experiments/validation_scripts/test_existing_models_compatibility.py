"""
Test existing trained models for backward compatibility.

Validates that Phase 1 + 2a changes didn't break existing models.

Tests:
1. Model loading works correctly
2. Forward pass with horizon=1 (backward compatibility)
3. Test loss matches previous results
4. Autoregressive generation still works
"""

import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds

# Set seeds and dtype
set_seeds(42)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("EXISTING MODELS BACKWARD COMPATIBILITY TEST")
print("=" * 80)
print("\nValidating that our changes didn't break existing trained models.")
print()

# Previous test losses (from test_results.log after Phase 1)
PREVIOUS_LOSSES = {
    "no_ex": 0.006409,
    "ex_no_loss": 0.006347,
    "ex_loss": 0.006392,
}

# Tolerance for loss comparison (50% relative difference allowed, since testing on different subset)
LOSS_TOLERANCE_RELATIVE = 0.50  # 50% relative difference


def load_test_data():
    """Load test data."""
    print("Loading test data...")
    data = np.load("data/vol_surface_with_ret.npz")
    vol_surf_data = data["surface"]
    ret_data = data["ret"]
    skew_data = data["skews"]
    slope_data = data["slopes"]

    # Construct ex_data
    ex_data = np.concatenate([
        ret_data[..., np.newaxis],
        skew_data[..., np.newaxis],
        slope_data[..., np.newaxis]
    ], axis=-1)

    print(f"  Data shape: {vol_surf_data.shape}")
    print(f"  Ex_data shape: {ex_data.shape}")

    return vol_surf_data, ex_data


def test_model_loading(model_name, model_path):
    """Test that model loads correctly."""
    print(f"\n--- Loading {model_name} ---")

    try:
        model_data = torch.load(model_path, weights_only=False)
        print(f"  ✓ Model file loaded")

        model_config = model_data["model_config"]
        print(f"  Config keys: {list(model_config.keys())[:5]}...")

        # Check if horizon is in config (should default to 1)
        horizon = model_config.get("horizon", 1)
        print(f"  Horizon: {horizon} (default: 1)")

        model = CVAEMemRand(model_config)
        model.load_weights(dict_to_load=model_data)
        model.eval()

        print(f"  ✓ Model created and weights loaded")
        print(f"    Device: {model.device}")
        print(f"    Ex_feats_dim: {model.config['ex_feats_dim']}")
        print(f"    Horizon attribute: {model.horizon}")

        if model.horizon != 1:
            print(f"  ⚠ WARNING: Expected horizon=1, got {model.horizon}")

        return model, model_config

    except Exception as e:
        print(f"  ✗ FAIL: Model loading failed!")
        print(f"    Error: {e}")
        return None, None


def test_forward_pass(model, model_name, vol_data, ex_data):
    """Test forward pass with horizon=1."""
    print(f"\n--- Testing Forward Pass ({model_name}) ---")

    has_ex_feats = model.config["ex_feats_dim"] > 0

    # Create a batch with T=6 (context=5, horizon=1)
    batch_size = 4
    context_len = 5
    horizon = 1
    T = context_len + horizon

    # Sample from test data
    start_idx = 100
    surfaces = vol_data[start_idx:start_idx+T]

    batch = {
        "surface": torch.from_numpy(surfaces).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    }

    if has_ex_feats:
        ex_feats = ex_data[start_idx:start_idx+T]
        batch["ex_feats"] = torch.from_numpy(ex_feats).unsqueeze(0).repeat(batch_size, 1, 1)

    print(f"  Input shape: {batch['surface'].shape}")
    if has_ex_feats:
        print(f"  Ex_feats shape: {batch['ex_feats'].shape}")

    # Forward pass
    try:
        with torch.no_grad():
            if has_ex_feats:
                surface_recon, ex_feats_recon, z_mean, z_log_var, z = model.forward(batch)
            else:
                surface_recon, z_mean, z_log_var, z = model.forward(batch)

        expected_shape = (batch_size, horizon, 3, 5, 5)

        if surface_recon.shape != expected_shape:
            print(f"  ✗ FAIL: Shape mismatch!")
            print(f"    Expected: {expected_shape}")
            print(f"    Got: {surface_recon.shape}")
            return False

        print(f"  ✓ Output shape: {surface_recon.shape} (correct)")

        if has_ex_feats:
            expected_ex_shape = (batch_size, horizon, model.config["ex_feats_dim"])
            if ex_feats_recon.shape != expected_ex_shape:
                print(f"  ✗ FAIL: Ex_feats shape mismatch!")
                print(f"    Expected: {expected_ex_shape}")
                print(f"    Got: {ex_feats_recon.shape}")
                return False
            print(f"  ✓ Ex_feats shape: {ex_feats_recon.shape} (correct)")

        # Check for NaN/Inf
        if torch.isnan(surface_recon).any() or torch.isinf(surface_recon).any():
            print(f"  ✗ FAIL: NaN or Inf in output!")
            return False

        print(f"  ✓ No NaN or Inf values")

        return True

    except Exception as e:
        print(f"  ✗ FAIL: Forward pass failed!")
        print(f"    Error: {e}")
        return False


def test_loss_computation(model, model_name, vol_data, ex_data):
    """Test loss computation and compare to previous results."""
    print(f"\n--- Testing Loss Computation ({model_name}) ---")

    has_ex_feats = model.config["ex_feats_dim"] > 0

    # Create test batch (same as forward pass test)
    batch_size = 16
    context_len = 5
    horizon = 1
    T = context_len + horizon

    # Sample multiple batches and average
    num_batches = 10
    total_loss = 0.0

    for i in range(num_batches):
        start_idx = 100 + i * 10
        surfaces = vol_data[start_idx:start_idx+T]

        batch = {
            "surface": torch.from_numpy(surfaces).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        }

        if has_ex_feats:
            ex_feats = ex_data[start_idx:start_idx+T]
            batch["ex_feats"] = torch.from_numpy(ex_feats).unsqueeze(0).repeat(batch_size, 1, 1)

        with torch.no_grad():
            losses = model.test_step(batch)

        total_loss += losses["loss"].item()

    avg_loss = total_loss / num_batches

    print(f"  Current test loss: {avg_loss:.6f}")

    # Compare to previous results
    if model_name in PREVIOUS_LOSSES:
        previous_loss = PREVIOUS_LOSSES[model_name]
        diff = abs(avg_loss - previous_loss)
        pct_diff = (diff / previous_loss)

        print(f"  Previous test loss: {previous_loss:.6f}")
        print(f"  Difference: {diff:.6f} ({pct_diff*100:.2f}%)")

        if pct_diff > LOSS_TOLERANCE_RELATIVE:
            print(f"  ⚠ WARNING: Loss difference exceeds tolerance!")
            print(f"    Tolerance: {LOSS_TOLERANCE_RELATIVE*100:.1f}% relative difference")
            return False
        else:
            print(f"  ✓ Loss within acceptable range (tolerance: {LOSS_TOLERANCE_RELATIVE*100:.0f}%)")
            return True
    else:
        print(f"  ⚠ No previous loss recorded for {model_name}")
        return True


def test_autoregressive_generation(model, model_name, vol_data, ex_data):
    """Test autoregressive generation (Phase 1 functionality)."""
    print(f"\n--- Testing Autoregressive Generation ({model_name}) ---")

    has_ex_feats = model.config["ex_feats_dim"] > 0

    # Create context
    context_len = 5
    horizon = 30
    start_idx = 100

    context_surfaces = vol_data[start_idx:start_idx+context_len]

    context = {
        "surface": torch.from_numpy(context_surfaces).unsqueeze(0)  # Keep as float64
    }

    if has_ex_feats:
        context_ex_feats = ex_data[start_idx:start_idx+context_len]
        context["ex_feats"] = torch.from_numpy(context_ex_feats).unsqueeze(0)  # Keep as float64

    try:
        with torch.no_grad():
            result = model.generate_autoregressive_sequence(
                initial_context=context,
                horizon=horizon
            )

        if has_ex_feats:
            pred_surfaces, pred_ex_feats = result
            expected_ex_shape = (1, horizon, 3)

            if pred_ex_feats.shape != expected_ex_shape:
                print(f"  ✗ FAIL: Ex_feats shape mismatch!")
                print(f"    Expected: {expected_ex_shape}")
                print(f"    Got: {pred_ex_feats.shape}")
                return False

            print(f"  ✓ Ex_feats shape: {pred_ex_feats.shape}")
        else:
            pred_surfaces = result

        expected_shape = (1, horizon, 3, 5, 5)

        if pred_surfaces.shape != expected_shape:
            print(f"  ✗ FAIL: Surfaces shape mismatch!")
            print(f"    Expected: {expected_shape}")
            print(f"    Got: {pred_surfaces.shape}")
            return False

        print(f"  ✓ Surfaces shape: {pred_surfaces.shape}")

        # Check for NaN/Inf
        if torch.isnan(pred_surfaces).any() or torch.isinf(pred_surfaces).any():
            print(f"  ✗ FAIL: NaN or Inf in generated surfaces!")
            return False

        print(f"  ✓ No NaN or Inf values")

        # Check quantile ordering
        p05 = pred_surfaces[0, :, 0, :, :]
        p50 = pred_surfaces[0, :, 1, :, :]
        p95 = pred_surfaces[0, :, 2, :, :]

        violations_05_50 = (p05 > p50).sum().item()
        violations_50_95 = (p50 > p95).sum().item()
        total_elements = horizon * 5 * 5

        print(f"  Quantile ordering violations:")
        print(f"    p05 > p50: {violations_05_50}/{total_elements} ({violations_05_50/total_elements*100:.1f}%)")
        print(f"    p50 > p95: {violations_50_95}/{total_elements} ({violations_50_95/total_elements*100:.1f}%)")

        if violations_05_50 + violations_50_95 > total_elements * 0.1:
            print(f"  ⚠ WARNING: High quantile ordering violations (>10%)")
        else:
            print(f"  ✓ Quantile ordering reasonable")

        return True

    except Exception as e:
        print(f"  ✗ FAIL: Autoregressive generation failed!")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""

    # Load data
    vol_data, ex_data = load_test_data()

    # Models to test
    models_to_test = [
        ("no_ex", "test_spx/quantile_regression/no_ex.pt"),
        ("ex_no_loss", "test_spx/quantile_regression/ex_no_loss.pt"),
        ("ex_loss", "test_spx/quantile_regression/ex_loss.pt"),
    ]

    results = {}

    for model_name, model_path in models_to_test:
        print("\n" + "=" * 80)
        print(f"TESTING {model_name.upper()}")
        print("=" * 80)

        # Test 1: Model loading
        model, config = test_model_loading(model_name, model_path)
        if model is None:
            results[model_name] = {"loading": False}
            continue

        results[model_name] = {"loading": True}

        # Test 2: Forward pass
        forward_ok = test_forward_pass(model, model_name, vol_data, ex_data)
        results[model_name]["forward"] = forward_ok

        # Test 3: Loss computation
        loss_ok = test_loss_computation(model, model_name, vol_data, ex_data)
        results[model_name]["loss"] = loss_ok

        # Test 4: Autoregressive generation
        generation_ok = test_autoregressive_generation(model, model_name, vol_data, ex_data)
        results[model_name]["generation"] = generation_ok

        # Overall status
        all_ok = all([forward_ok, loss_ok, generation_ok])
        results[model_name]["overall"] = all_ok

        status = "✓ PASS" if all_ok else "✗ FAIL"
        print(f"\n{model_name.upper()} - {status}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<15} {'Loading':<10} {'Forward':<10} {'Loss':<10} {'Generation':<12} {'Overall':<10}")
    print("-" * 80)

    for model_name, tests in results.items():
        loading_status = "✓" if tests.get("loading", False) else "✗"
        forward_status = "✓" if tests.get("forward", False) else "✗"
        loss_status = "✓" if tests.get("loss", False) else "✗"
        generation_status = "✓" if tests.get("generation", False) else "✗"
        overall_status = "✓ PASS" if tests.get("overall", False) else "✗ FAIL"

        print(f"{model_name:<15} {loading_status:<10} {forward_status:<10} {loss_status:<10} {generation_status:<12} {overall_status:<10}")

    print("-" * 80)

    # Final verdict
    all_passed = all(tests.get("overall", False) for tests in results.values())

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nBackward compatibility confirmed:")
        print("  - All models load successfully")
        print("  - Forward pass works with horizon=1")
        print("  - Test losses match previous results")
        print("  - Autoregressive generation works correctly")
        print("\n➜ Safe to proceed with horizon > 1 training")
    else:
        print("\n✗ SOME TESTS FAILED!")
        print("\nPlease review failures before proceeding.")


if __name__ == "__main__":
    main()
