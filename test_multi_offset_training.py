"""
Unit Tests for Multi-Offset Autoregressive Training

Tests the new multi-offset training functionality added to vae/utils.py
for the Context=60 4-phase curriculum.

Usage:
    python test_multi_offset_training.py
"""

import torch
import numpy as np
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import train_autoregressive_step, train_autoregressive_multi_offset
from config.backfill_context60_config import BackfillContext60Config as cfg


def create_test_model(use_quantile=True):
    """Create a small test model for unit testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = {
        "feat_dim": (5, 5),
        "latent_dim": 5,
        "context_len": 60,
        "horizon": 1,  # Will be overridden during AR training
        "surface_hidden": [5, 5, 5],
        "ctx_surface_hidden": [5, 5, 5],
        "ex_feats_dim": 3,
        "ex_feats_hidden": None,
        "ctx_ex_feats_hidden": None,
        "interaction_layers": None,
        "mem_type": "lstm",
        "mem_hidden": 50,
        "mem_layers": 2,
        "mem_dropout": 0.3,
        "compress_context": True,
        "use_dense_surface": False,
        "kl_weight": 1e-5,
        "use_quantile_regression": use_quantile,
        "num_quantiles": 3 if use_quantile else None,
        "quantiles": [0.05, 0.5, 0.95] if use_quantile else None,
        "quantile_loss_weights": [5.0, 1.0, 5.0] if use_quantile else None,
        "re_feat_weight": 1.0,
        "ex_loss_on_ret_only": True,
        "ex_feats_loss_type": "l2",
        "device": device,
    }

    model = CVAEMemRand(model_config)
    return model


def create_test_batch(seq_len, batch_size=4):
    """
    Create synthetic test batch.

    Args:
        seq_len: Sequence length (e.g., 240 for Phase 3)
        batch_size: Batch size

    Returns:
        Dict with 'surface' and 'ex_feats' tensors
    """
    surface = torch.randn(batch_size, seq_len, 5, 5) * 0.1 + 0.2  # Around 20% vol
    ex_feats = torch.randn(batch_size, seq_len, 3) * 0.01  # Small features

    return {
        "surface": surface,
        "ex_feats": ex_feats
    }


def test_sequence_length_validation():
    """Test that sequence length validation works correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Sequence Length Validation")
    print("=" * 80)

    model = create_test_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Phase 3 parameters
    context_len = 60
    ar_steps = 3
    horizon = 60
    offset = 60

    # Required length: 60 + (3 * 60) = 240
    required_len = context_len + (ar_steps * offset)
    print(f"\nRequired sequence length: {required_len}")

    # Test with sufficient length
    print("\n✓ Testing with seq_len = 240 (sufficient)...")
    batch = create_test_batch(seq_len=240, batch_size=2)

    try:
        losses = train_autoregressive_step(
            model, batch, optimizer,
            ar_steps=ar_steps,
            horizon=horizon,
            offset=offset
        )
        print("  ✅ PASS: Training succeeded with sufficient sequence length")
        print(f"  Returned loss keys: {list(losses.keys())}")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")

    # Test with insufficient length
    print("\n✓ Testing with seq_len = 200 (insufficient)...")
    batch_short = create_test_batch(seq_len=200, batch_size=2)

    try:
        losses = train_autoregressive_step(
            model, batch_short, optimizer,
            ar_steps=ar_steps,
            horizon=horizon,
            offset=offset
        )
        print("  ❌ FAIL: Should have raised ValueError for insufficient length")
    except ValueError as e:
        print(f"  ✅ PASS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"  ❌ FAIL: Raised unexpected error: {e}")


def test_offset_sampling():
    """Test that multi-offset training samples offsets correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: Offset Sampling")
    print("=" * 80)

    model = create_test_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Phase 3 parameters
    horizon = 60
    offsets = [30, 60]
    ar_steps = 3
    batch_size = 1

    # Run 100 iterations and count offset distribution
    offset_counts = {30: 0, 60: 0}
    num_iterations = 100

    print(f"\nRunning {num_iterations} iterations with offsets={offsets}...")
    print("Expected: ~50% each offset")

    for i in range(num_iterations):
        # Need seq_len = 240 to support both offsets
        # offset=30: 60 + (3*30) = 150
        # offset=60: 60 + (3*60) = 240
        batch = create_test_batch(seq_len=240, batch_size=batch_size)

        losses = train_autoregressive_multi_offset(
            model, batch, optimizer,
            horizon=horizon,
            offsets=offsets,
            ar_steps=ar_steps
        )

        sampled_offset = losses['offset']
        offset_counts[sampled_offset] += 1

    # Print results
    print("\nResults:")
    for offset, count in offset_counts.items():
        pct = (count / num_iterations) * 100
        print(f"  Offset {offset}: {count}/{num_iterations} ({pct:.1f}%)")

    # Check if distribution is reasonable (within 20-80% for each)
    pct_30 = (offset_counts[30] / num_iterations) * 100
    pct_60 = (offset_counts[60] / num_iterations) * 100

    if 20 <= pct_30 <= 80 and 20 <= pct_60 <= 80:
        print("\n  ✅ PASS: Offset distribution is reasonable")
    else:
        print("\n  ⚠️  WARNING: Offset distribution seems skewed")


def test_overlapping_vs_nonoverlapping():
    """Test both overlapping and non-overlapping offset scenarios."""
    print("\n" + "=" * 80)
    print("TEST 3: Overlapping vs Non-Overlapping Offsets")
    print("=" * 80)

    model = create_test_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    horizon = 60
    ar_steps = 3

    # Test Case 1: Overlapping (offset < horizon)
    print("\n✓ Testing overlapping: offset=30, horizon=60...")
    batch = create_test_batch(seq_len=240, batch_size=2)

    try:
        losses = train_autoregressive_step(
            model, batch, optimizer,
            ar_steps=ar_steps,
            horizon=horizon,
            offset=30
        )
        print(f"  ✅ PASS: Overlapping training succeeded")
        print(f"  Loss: {losses['loss']:.6f}")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")

    # Test Case 2: Non-overlapping (offset = horizon)
    print("\n✓ Testing non-overlapping: offset=60, horizon=60...")
    batch = create_test_batch(seq_len=240, batch_size=2)

    try:
        losses = train_autoregressive_step(
            model, batch, optimizer,
            ar_steps=ar_steps,
            horizon=horizon,
            offset=60
        )
        print(f"  ✅ PASS: Non-overlapping training succeeded")
        print(f"  Loss: {losses['loss']:.6f}")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")

    # Test Case 3: Large overlap (offset = 45, horizon = 90)
    print("\n✓ Testing large overlap: offset=45, horizon=90...")
    batch = create_test_batch(seq_len=330, batch_size=2)  # 60 + 3*90 = 330

    try:
        losses = train_autoregressive_step(
            model, batch, optimizer,
            ar_steps=ar_steps,
            horizon=90,
            offset=45
        )
        print(f"  ✅ PASS: Large overlap training succeeded")
        print(f"  Loss: {losses['loss']:.6f}")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")


def test_loss_components():
    """Test that loss components are returned correctly."""
    print("\n" + "=" * 80)
    print("TEST 4: Loss Components")
    print("=" * 80)

    model = create_test_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    batch = create_test_batch(seq_len=240, batch_size=2)

    losses = train_autoregressive_multi_offset(
        model, batch, optimizer,
        horizon=60,
        offsets=[30, 60],
        ar_steps=3
    )

    print("\nReturned loss components:")
    expected_keys = ['loss', 'reconstruction_loss', 'kl_loss', 'offset', 'ar_steps']

    for key in expected_keys:
        if key in losses:
            if key in ['offset', 'ar_steps']:
                print(f"  ✅ {key}: {losses[key]}")
            else:
                print(f"  ✅ {key}: {losses[key]:.6f}")
        else:
            print(f"  ❌ Missing key: {key}")

    # Check all expected keys present
    if all(key in losses for key in expected_keys):
        print("\n  ✅ PASS: All expected loss components present")
    else:
        print("\n  ❌ FAIL: Missing some loss components")

    # Check loss values are reasonable
    checks_passed = True

    if losses['loss'] <= 0:
        print("  ❌ FAIL: Total loss should be positive")
        checks_passed = False

    if losses['reconstruction_loss'] < 0:
        print("  ❌ FAIL: Reconstruction loss should be non-negative")
        checks_passed = False

    if losses['kl_loss'] < 0:
        print("  ❌ FAIL: KL loss should be non-negative")
        checks_passed = False

    if losses['offset'] not in [30, 60]:
        print("  ❌ FAIL: Offset should be 30 or 60")
        checks_passed = False

    if losses['ar_steps'] != 3:
        print("  ❌ FAIL: AR steps should be 3")
        checks_passed = False

    if checks_passed:
        print("  ✅ PASS: All loss values are reasonable")


def test_phase3_configuration():
    """Test Phase 3 configuration matches expected settings."""
    print("\n" + "=" * 80)
    print("TEST 5: Phase 3 Configuration")
    print("=" * 80)

    print("\nPhase 3 parameters from config:")
    print(f"  Horizon: {cfg.phase3_horizon}")
    print(f"  Offsets: {cfg.phase3_offsets}")
    print(f"  AR steps: {cfg.phase3_ar_steps}")
    print(f"  Sequence length range: {cfg.phase3_seq_len}")
    print(f"  Epoch range: {cfg.phase2_end}-{cfg.phase3_end-1}")

    # Validate configuration
    checks = []

    # Check horizon
    if cfg.phase3_horizon == 60:
        print("\n  ✅ Horizon = 60 (correct)")
        checks.append(True)
    else:
        print(f"\n  ❌ Horizon = {cfg.phase3_horizon} (expected 60)")
        checks.append(False)

    # Check offsets
    if cfg.phase3_offsets == [30, 60]:
        print("  ✅ Offsets = [30, 60] (correct)")
        checks.append(True)
    else:
        print(f"  ❌ Offsets = {cfg.phase3_offsets} (expected [30, 60])")
        checks.append(False)

    # Check AR steps
    if cfg.phase3_ar_steps == 3:
        print("  ✅ AR steps = 3 (correct)")
        checks.append(True)
    else:
        print(f"  ❌ AR steps = {cfg.phase3_ar_steps} (expected 3)")
        checks.append(False)

    # Check sequence length
    min_len, max_len = cfg.phase3_seq_len
    required_len = cfg.context_len + (cfg.phase3_ar_steps * max(cfg.phase3_offsets))

    if min_len <= required_len <= max_len:
        print(f"  ✅ Sequence length {cfg.phase3_seq_len} supports required length {required_len}")
        checks.append(True)
    else:
        print(f"  ❌ Sequence length {cfg.phase3_seq_len} doesn't support required length {required_len}")
        checks.append(False)

    if all(checks):
        print("\n  ✅ PASS: Phase 3 configuration is correct")
    else:
        print("\n  ❌ FAIL: Some configuration checks failed")


def test_phase4_configuration():
    """Test Phase 4 configuration matches expected settings."""
    print("\n" + "=" * 80)
    print("TEST 6: Phase 4 Configuration")
    print("=" * 80)

    print("\nPhase 4 parameters from config:")
    print(f"  Horizon: {cfg.phase4_horizon}")
    print(f"  Offsets: {cfg.phase4_offsets}")
    print(f"  AR steps: {cfg.phase4_ar_steps}")
    print(f"  Sequence length range: {cfg.phase4_seq_len}")
    print(f"  Epoch range: {cfg.phase3_end}-{cfg.phase4_end-1}")

    # Validate configuration
    checks = []

    # Check horizon
    if cfg.phase4_horizon == 90:
        print("\n  ✅ Horizon = 90 (correct)")
        checks.append(True)
    else:
        print(f"\n  ❌ Horizon = {cfg.phase4_horizon} (expected 90)")
        checks.append(False)

    # Check offsets
    if cfg.phase4_offsets == [45, 90]:
        print("  ✅ Offsets = [45, 90] (correct)")
        checks.append(True)
    else:
        print(f"  ❌ Offsets = {cfg.phase4_offsets} (expected [45, 90])")
        checks.append(False)

    # Check AR steps
    if cfg.phase4_ar_steps == 3:
        print("  ✅ AR steps = 3 (correct)")
        checks.append(True)
    else:
        print(f"  ❌ AR steps = {cfg.phase4_ar_steps} (expected 3)")
        checks.append(False)

    # Check sequence length
    min_len, max_len = cfg.phase4_seq_len
    required_len = cfg.context_len + (cfg.phase4_ar_steps * max(cfg.phase4_offsets))

    if min_len <= required_len <= max_len:
        print(f"  ✅ Sequence length {cfg.phase4_seq_len} supports required length {required_len}")
        checks.append(True)
    else:
        print(f"  ❌ Sequence length {cfg.phase4_seq_len} doesn't support required length {required_len}")
        checks.append(False)

    if all(checks):
        print("\n  ✅ PASS: Phase 4 configuration is correct")
    else:
        print("\n  ❌ FAIL: Some configuration checks failed")


def test_gradient_flow():
    """Test that gradients flow correctly through the model."""
    print("\n" + "=" * 80)
    print("TEST 7: Gradient Flow")
    print("=" * 80)

    model = create_test_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Store initial parameters
    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    # Run one training step
    batch = create_test_batch(seq_len=240, batch_size=2)

    print("\n✓ Running one training step...")
    losses = train_autoregressive_multi_offset(
        model, batch, optimizer,
        horizon=60,
        offsets=[30, 60],
        ar_steps=3
    )

    # Check if parameters changed
    num_changed = 0
    num_total = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_total += 1
            initial_param = initial_params[name]
            if not torch.allclose(param, initial_param, atol=1e-8):
                num_changed += 1

    print(f"\nParameters updated: {num_changed} / {num_total}")

    if num_changed > 0:
        print("  ✅ PASS: Gradients are flowing (parameters updated)")
    else:
        print("  ❌ FAIL: No parameters updated (gradient flow issue?)")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 100)
    print(" " * 30 + "MULTI-OFFSET TRAINING UNIT TESTS")
    print("=" * 100)

    try:
        test_sequence_length_validation()
        test_offset_sampling()
        test_overlapping_vs_nonoverlapping()
        test_loss_components()
        test_phase3_configuration()
        test_phase4_configuration()
        test_gradient_flow()

        print("\n" + "=" * 100)
        print(" " * 40 + "ALL TESTS COMPLETED")
        print("=" * 100 + "\n")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_all_tests()
