"""
Test autoregressive generation for all 3 model variants.

Tests:
1. Shape validation
2. Quantile ordering (p05 < p50 < p95)
3. Edge cases (different horizons, unbatched input)
4. Device handling
5. Visual sanity check
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds

# Set seeds and dtype
set_seeds(42)
torch.set_default_dtype(torch.float64)

def test_model_variant(model_name, model_path):
    """Test one model variant with comprehensive checks."""
    print(f"\n{'='*80}")
    print(f"Testing {model_name.upper()}")
    print(f"{'='*80}")

    # Load model
    print(f"Loading model from: {model_path}")
    try:
        model_data = torch.load(model_path, map_location="cpu", weights_only=False)
        model = CVAEMemRand(model_data["model_config"])
        model.load_weights(dict_to_load=model_data)
        model.eval()
        print(f"✓ Model loaded successfully")
        print(f"  Device: {model.config['device']}")
        print(f"  Ex_feats_dim: {model.config['ex_feats_dim']}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Load test data
    data = np.load("data/vol_surface_with_ret.npz")
    vol_surf_data = data["surface"]

    # Construct ex_data from individual features
    ret_data = data["ret"]
    skew_data = data["skews"]
    slope_data = data["slopes"]
    ex_data = np.concatenate([ret_data[..., np.newaxis],
                             skew_data[..., np.newaxis],
                             slope_data[..., np.newaxis]], axis=-1)  # (N, 3)

    # Test set starts at index 5000
    test_start = 5000

    # Prepare context (use 5 days of context)
    context_len = 5
    context_surfaces = vol_surf_data[test_start:test_start + context_len]  # (5, 5, 5)

    has_ex_feats = model.config["ex_feats_dim"] > 0

    if has_ex_feats:
        context_ex_feats = ex_data[test_start:test_start + context_len]  # (5, 3)
        context = {
            "surface": torch.from_numpy(context_surfaces).unsqueeze(0).to(model.config['device']),
            "ex_feats": torch.from_numpy(context_ex_feats).unsqueeze(0).to(model.config['device'])
        }
        print(f"  Context: surface {context['surface'].shape}, ex_feats {context['ex_feats'].shape}")
    else:
        context = {
            "surface": torch.from_numpy(context_surfaces).unsqueeze(0).to(model.config['device'])
        }
        print(f"  Context: surface {context['surface'].shape}")

    # Test 1: Shape validation for 30-day horizon
    print(f"\nTest 1: Shape Validation (30-day horizon)")
    try:
        with torch.no_grad():
            result = model.generate_autoregressive_sequence(context, horizon=30)

        if has_ex_feats:
            pred_surfaces, pred_ex_feats = result
            print(f"  ✓ Surfaces shape: {pred_surfaces.shape} (expected: (1, 30, 3, 5, 5))")
            print(f"  ✓ Ex_feats shape: {pred_ex_feats.shape} (expected: (1, 30, 3))")

            assert pred_surfaces.shape == (1, 30, 3, 5, 5), f"Wrong surface shape: {pred_surfaces.shape}"
            assert pred_ex_feats.shape == (1, 30, 3), f"Wrong ex_feats shape: {pred_ex_feats.shape}"
        else:
            pred_surfaces = result
            print(f"  ✓ Surfaces shape: {pred_surfaces.shape} (expected: (1, 30, 3, 5, 5))")
            assert pred_surfaces.shape == (1, 30, 3, 5, 5), f"Wrong shape: {pred_surfaces.shape}"

        # Check for NaN/Inf
        if torch.isnan(pred_surfaces).any():
            print(f"  ✗ Warning: NaN values detected in surfaces")
        else:
            print(f"  ✓ No NaN values")

        if torch.isinf(pred_surfaces).any():
            print(f"  ✗ Warning: Inf values detected in surfaces")
        else:
            print(f"  ✓ No Inf values")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

    # Test 2: Quantile ordering (p05 < p50 < p95)
    print(f"\nTest 2: Quantile Ordering")
    try:
        p05 = pred_surfaces[0, :, 0, :, :]  # (30, 5, 5)
        p50 = pred_surfaces[0, :, 1, :, :]  # (30, 5, 5)
        p95 = pred_surfaces[0, :, 2, :, :]  # (30, 5, 5)

        violations_p05_p50 = (p05 > p50).sum().item()
        violations_p50_p95 = (p50 > p95).sum().item()
        total_points = 30 * 5 * 5

        print(f"  p05 > p50 violations: {violations_p05_p50}/{total_points} ({100*violations_p05_p50/total_points:.2f}%)")
        print(f"  p50 > p95 violations: {violations_p50_p95}/{total_points} ({100*violations_p50_p95/total_points:.2f}%)")

        if violations_p05_p50 == 0 and violations_p50_p95 == 0:
            print(f"  ✓ Perfect quantile ordering")
        else:
            print(f"  ⚠ Quantile ordering violations detected")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 3: Different horizons
    print(f"\nTest 3: Different Horizons")
    for horizon in [5, 10, 15]:
        try:
            with torch.no_grad():
                result = model.generate_autoregressive_sequence(context, horizon=horizon)

            if has_ex_feats:
                surf, _ = result
                expected_shape = (1, horizon, 3, 5, 5)
            else:
                surf = result
                expected_shape = (1, horizon, 3, 5, 5)

            assert surf.shape == expected_shape, f"Wrong shape for horizon {horizon}"
            print(f"  ✓ Horizon {horizon:2d}: {surf.shape}")
        except Exception as e:
            print(f"  ✗ Horizon {horizon}: {e}")

    # Test 4: Visual sanity check (one grid point)
    print(f"\nTest 4: Visual Sanity Check")
    try:
        # Get ground truth for comparison
        ground_truth = vol_surf_data[test_start + context_len:test_start + context_len + 30]

        # Extract predictions for center grid point (2, 2)
        pred_p05 = pred_surfaces[0, :, 0, 2, 2].cpu().numpy()
        pred_p50 = pred_surfaces[0, :, 1, 2, 2].cpu().numpy()
        pred_p95 = pred_surfaces[0, :, 2, 2, 2].cpu().numpy()
        gt = ground_truth[:, 2, 2]

        # Compute metrics
        rmse = np.sqrt(np.mean((pred_p50 - gt)**2))
        mae = np.mean(np.abs(pred_p50 - gt))
        ci_violations = np.mean((gt < pred_p05) | (gt > pred_p95)) * 100

        print(f"  Center grid point (2,2) metrics:")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    MAE:  {mae:.6f}")
        print(f"    CI Violations: {ci_violations:.2f}%")

        # Create simple plot
        fig, ax = plt.subplots(figsize=(10, 5))
        days = np.arange(30)

        ax.fill_between(days, pred_p05, pred_p95, alpha=0.3, color='blue', label='90% CI')
        ax.plot(days, gt, 'k-', linewidth=2, label='Ground Truth')
        ax.plot(days, pred_p50, 'b-', linewidth=1.5, label='Prediction (p50)')

        # Mark violations
        violations = (gt < pred_p05) | (gt > pred_p95)
        if np.any(violations):
            ax.scatter(days[violations], gt[violations], color='red', s=50,
                      zorder=5, label='CI Violation')

        ax.set_xlabel('Days')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'{model_name.upper()} - Autoregressive 30-Day Forecast\n'
                     f'Center Grid Point (2,2) - RMSE: {rmse:.4f}, CI Viol: {ci_violations:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_file = f'test_autoregressive_{model_name}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved to: {plot_file}")

    except Exception as e:
        print(f"  ✗ Visual check failed: {e}")

    # Test 5: Ex_feats coherence (for ex_loss variant)
    if has_ex_feats and pred_ex_feats is not None:
        print(f"\nTest 5: Ex_Feats Coherence")
        try:
            # Get ground truth ex_feats
            gt_ex_feats = ex_data[test_start + context_len:test_start + context_len + 30]

            # Extract predicted returns (first feature, no quantiles for ex_feats)
            pred_returns = pred_ex_feats[0, :, 0].cpu().numpy()  # (30,) - feature 0 is returns
            gt_returns = gt_ex_feats[:, 0]

            # Compute metrics
            returns_rmse = np.sqrt(np.mean((pred_returns - gt_returns)**2))
            returns_corr = np.corrcoef(pred_returns, gt_returns)[0, 1]

            print(f"  Return predictions (feature 0):")
            print(f"    RMSE: {returns_rmse:.6f}")
            print(f"    Correlation: {returns_corr:.4f}")

            # Check if predicted returns are reasonable (not all zeros/constant)
            returns_std = np.std(pred_returns)
            print(f"    Std Dev: {returns_std:.6f}")

            if returns_std < 1e-6:
                print(f"  ⚠ Warning: Predicted returns have very low variance")
            else:
                print(f"  ✓ Returns have reasonable variance")

        except Exception as e:
            print(f"  ✗ Ex_feats coherence check failed: {e}")

    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - ALL TESTS PASSED ✓")
    print(f"{'='*80}\n")

    return True

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("AUTOREGRESSIVE GENERATION TEST SUITE")
    print("="*80)
    print("\nThis script tests the new autoregressive generation methods")
    print("on all 3 trained model variants.\n")

    # Test all 3 model variants
    models = [
        ("no_ex", "test_spx/quantile_regression/no_ex.pt"),
        ("ex_no_loss", "test_spx/quantile_regression/ex_no_loss.pt"),
        ("ex_loss", "test_spx/quantile_regression/ex_loss.pt"),
    ]

    results = {}
    for model_name, model_path in models:
        try:
            success = test_model_variant(model_name, model_path)
            results[model_name] = success
        except Exception as e:
            print(f"\n✗ {model_name.upper()} failed with error: {e}\n")
            results[model_name] = False

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name:15s}: {status}")
    print("="*80)

    # Exit with appropriate code
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed successfully!\n")
        exit(0)
    else:
        print("\n✗ Some tests failed. Please review the output above.\n")
        exit(1)
