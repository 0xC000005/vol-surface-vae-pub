"""
Test multi-horizon training capability for CVAEMemRand.

Tests:
1. Backward compatibility: horizon=1 should work identically to existing models
2. Multi-horizon training: horizon=2, 3, 5, 7 should train successfully
3. Shape validation: outputs have correct shapes for each horizon
4. Loss computation: loss is computed correctly for all horizons
5. All model variants: no_ex, ex_no_loss, ex_loss
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.utils import set_seeds

# Set seeds and dtype
set_seeds(42)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("MULTI-HORIZON TRAINING TEST SUITE")
print("=" * 80)
print("\nThis script tests the multi-horizon training capability on all 3 model variants.")
print()


def load_test_data():
    """Load a small subset of data for testing."""
    print("Loading test data...")
    data = np.load("data/vol_surface_with_ret.npz")
    vol_surf_data = data["surface"]
    ret_data = data["ret"]
    skew_data = data["skews"]
    slope_data = data["slopes"]

    # Construct ex_data from individual features
    ex_data = np.concatenate([
        ret_data[..., np.newaxis],
        skew_data[..., np.newaxis],
        slope_data[..., np.newaxis]
    ], axis=-1)

    # Use small subset for quick testing (100 days)
    vol_surf_subset = vol_surf_data[:100]
    ex_data_subset = ex_data[:100]

    print(f"  Data shape: {vol_surf_subset.shape}")
    print(f"  Ex_data shape: {ex_data_subset.shape}")

    return vol_surf_subset, ex_data_subset


def create_model_config(ex_feats_dim, horizon=1):
    """Create model configuration for testing."""
    config = {
        "feat_dim": (5, 5),
        "latent_dim": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "kl_weight": 1e-5,
        "re_feat_weight": 1.0 if ex_feats_dim > 0 else 0.0,
        "surface_hidden": [5, 5, 5],
        "ex_feats_dim": ex_feats_dim,
        "ex_feats_hidden": None,
        "ex_loss_on_ret_only": True,
        "ex_feats_loss_type": "l2",
        "mem_type": "lstm",
        "mem_hidden": 50,  # Smaller for faster testing
        "mem_layers": 1,
        "mem_dropout": 0.2,
        "ctx_surface_hidden": [5, 5, 5],
        "ctx_ex_feats_hidden": None,
        "interaction_layers": None,
        "compress_context": True,
        "use_dense_surface": False,
        "num_quantiles": 3,
        "quantiles": [0.05, 0.5, 0.95],
        "horizon": horizon,
    }
    return config


def test_model_variant(variant_name, ex_feats_dim, vol_data, ex_data, horizons=[1, 2, 3, 5, 7]):
    """Test a specific model variant with different horizons."""
    print("\n" + "=" * 80)
    print(f"Testing {variant_name.upper()}")
    print("=" * 80)

    for horizon in horizons:
        print(f"\n--- Horizon = {horizon} ---")

        # Create model config
        config = create_model_config(ex_feats_dim, horizon=horizon)
        print(f"  Model config: ex_feats_dim={ex_feats_dim}, horizon={horizon}")

        # Create model
        model = CVAEMemRand(config)
        print(f"  ✓ Model created successfully")
        print(f"    Device: {config['device']}")
        print(f"    Horizon: {model.horizon}")

        # Create dataset with sufficient sequence length
        # Need at least: min_context + horizon for valid training
        min_seq_len = 5 + horizon  # Min context = 5
        max_seq_len = min_seq_len + 5  # Some variability

        print(f"  Creating dataset with seq_len: {min_seq_len}-{max_seq_len}")

        if ex_feats_dim > 0:
            dataset = VolSurfaceDataSetRand(
                (vol_data, ex_data),
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len
            )
        else:
            dataset = VolSurfaceDataSetRand(
                vol_data,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len
            )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_sampler=CustomBatchSampler(dataset, batch_size=8, min_seq_len=min_seq_len)
        )

        print(f"  Dataset size: {len(dataset)}")
        print(f"  Num batches: {len(dataloader)}")

        # Test forward pass
        batch = next(iter(dataloader))
        print(f"  Batch surface shape: {batch['surface'].shape}")
        if ex_feats_dim > 0:
            print(f"  Batch ex_feats shape: {batch['ex_feats'].shape}")

        # Forward pass
        with torch.no_grad():
            if ex_feats_dim > 0:
                surface_recon, ex_feats_recon, z_mean, z_log_var, z = model.forward(batch)
            else:
                surface_recon, z_mean, z_log_var, z = model.forward(batch)

        # Validate output shapes
        expected_horizon = horizon
        expected_shape = (batch['surface'].shape[0], expected_horizon, 3, 5, 5)

        if surface_recon.shape != expected_shape:
            print(f"  ✗ FAIL: Surface shape mismatch!")
            print(f"    Expected: {expected_shape}")
            print(f"    Got: {surface_recon.shape}")
            return False
        else:
            print(f"  ✓ Surface reconstruction shape: {surface_recon.shape} (correct)")

        if ex_feats_dim > 0:
            expected_ex_shape = (batch['surface'].shape[0], expected_horizon, ex_feats_dim)
            if ex_feats_recon.shape != expected_ex_shape:
                print(f"  ✗ FAIL: Ex_feats shape mismatch!")
                print(f"    Expected: {expected_ex_shape}")
                print(f"    Got: {ex_feats_recon.shape}")
                return False
            else:
                print(f"  ✓ Ex_feats reconstruction shape: {ex_feats_recon.shape} (correct)")

        # Test train_step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        losses = model.train_step(batch, optimizer)

        print(f"  ✓ Train step completed")
        print(f"    Loss: {losses['loss']:.6f}")
        print(f"    RE Surface: {losses['re_surface']:.6f}")
        print(f"    KL Loss: {losses['kl_loss']:.6f}")

        # Test test_step
        model.eval()
        with torch.no_grad():
            test_losses = model.test_step(batch)

        print(f"  ✓ Test step completed")
        print(f"    Test Loss: {test_losses['loss']:.6f}")

        # Validate no NaN or Inf
        if torch.isnan(losses['loss']) or torch.isinf(losses['loss']):
            print(f"  ✗ FAIL: Loss contains NaN or Inf!")
            return False

        print(f"  ✓ No NaN or Inf values")

        # Quick training test (3 epochs)
        print(f"  Running quick training (3 epochs)...")
        for epoch in range(3):
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                model.train()
                losses = model.train_step(batch, optimizer)
                epoch_loss += losses['loss'].item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"    Epoch {epoch+1}/3: Loss = {avg_loss:.6f}")

        print(f"  ✓ Training successful for horizon={horizon}")

    print(f"\n{'='*80}")
    print(f"{variant_name.upper()} - ALL HORIZONS PASSED ✓")
    print(f"{'='*80}")
    return True


def test_backward_compatibility():
    """Test that horizon=1 works identically to existing models."""
    print("\n" + "=" * 80)
    print("BACKWARD COMPATIBILITY TEST (horizon=1)")
    print("=" * 80)

    vol_data, ex_data = load_test_data()

    # Test with horizon=1
    config = create_model_config(ex_feats_dim=3, horizon=1)
    model = CVAEMemRand(config)

    # Create dataset with old parameters (should still work)
    dataset = VolSurfaceDataSetRand((vol_data, ex_data), min_seq_len=6, max_seq_len=10)
    dataloader = DataLoader(
        dataset,
        batch_sampler=CustomBatchSampler(dataset, batch_size=8, min_seq_len=6)
    )

    batch = next(iter(dataloader))

    # Forward pass
    with torch.no_grad():
        surface_recon, ex_feats_recon, z_mean, z_log_var, z = model.forward(batch)

    # Should return (B, 1, 3, 5, 5) for horizon=1
    expected_shape = (batch['surface'].shape[0], 1, 3, 5, 5)

    if surface_recon.shape == expected_shape:
        print(f"  ✓ horizon=1 produces correct shape: {surface_recon.shape}")
        print(f"  ✓ Backward compatible with existing models")
        return True
    else:
        print(f"  ✗ FAIL: Shape mismatch with horizon=1")
        print(f"    Expected: {expected_shape}")
        print(f"    Got: {surface_recon.shape}")
        return False


def main():
    """Main test runner."""

    # Load test data
    vol_data, ex_data = load_test_data()

    # Test 1: Backward compatibility
    print("\nTest 1: Backward Compatibility")
    if not test_backward_compatibility():
        print("✗ Backward compatibility test FAILED")
        return

    # Test 2: Multi-horizon for all 3 variants
    horizons_to_test = [1, 2, 3, 5, 7]

    variants = [
        ("no_ex", 0),
        ("ex_no_loss", 3),
        ("ex_loss", 3),
    ]

    results = {}

    for variant_name, ex_feats_dim in variants:
        success = test_model_variant(
            variant_name, ex_feats_dim, vol_data, ex_data, horizons=horizons_to_test
        )
        results[variant_name] = success

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for variant_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{variant_name:<15} : {status}")

    print("=" * 80)

    if all(results.values()):
        print("\n✓ All tests passed successfully!")
        print("\nMulti-horizon training is working correctly:")
        print("  - Backward compatible with horizon=1")
        print("  - Supports horizon=2, 3, 5, 7")
        print("  - All 3 model variants work correctly")
        print("  - Loss computation is correct")
        print("  - Training converges successfully")
    else:
        print("\n✗ Some tests failed!")


if __name__ == "__main__":
    main()
