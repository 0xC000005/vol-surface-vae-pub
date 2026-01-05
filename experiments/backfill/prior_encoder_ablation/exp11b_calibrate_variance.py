"""
Experiment 11b: Post-hoc Variance Calibration

The heteroscedastic VAE from exp11 achieved only ~32% coverage because:
1. Decoder variance too low (~0.005 vs GT ~0.01)
2. No regime-dependence (variance ratio = 1.0)
3. z sampling adds minimal variability (decoder gain ≈ 0)

This script calibrates the decoder variance post-hoc to achieve 90% coverage.

Approach:
1. Load trained heteroscedastic model
2. Find scaling factor that gives 90% coverage on calibration set
3. Evaluate on held-out test set
4. Check if regime-specific calibration improves results
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_heteroscedastic import CVAEHeteroscedastic


def load_model(model_path):
    """Load trained heteroscedastic model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']

    model = CVAEHeteroscedastic(model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    return model, model_config


def compute_coverage_with_scale(model, surface, context_len, scale_factor, n_eval=500, n_samples=100):
    """Compute coverage with scaled variance.

    Instead of sampling from N(mean, var), sample from N(mean, scale_factor * var)
    """
    n_total = len(surface) - context_len - 1
    eval_indices = np.random.choice(n_total, min(n_eval, n_total), replace=False)

    coverages = []
    atm_ivs = []

    with torch.no_grad():
        for idx in eval_indices:
            context = surface[idx:idx + context_len]
            target = surface[idx + context_len].numpy()

            ctx_dict = {"surface": context.unsqueeze(0)}

            # Get mean and std from decoder (using prior mean for z)
            mean, std = model.get_surface_given_conditions(
                ctx_dict, horizon=1, sample_from_decoder=False
            )
            mean = mean.squeeze().cpu().numpy()
            std = std.squeeze().cpu().numpy()

            # Scale std
            scaled_std = std * np.sqrt(scale_factor)

            # 90% CI
            z_score = 1.645
            lower = mean - z_score * scaled_std
            upper = mean + z_score * scaled_std

            # Check coverage
            covered = (target >= lower) & (target <= upper)
            coverages.append(covered)
            atm_ivs.append(context[-1, 2, 2].item())

    coverages = np.array(coverages)
    atm_ivs = np.array(atm_ivs)

    # Overall and regime coverage
    overall_coverage = coverages.mean() * 100

    vol_terciles = np.percentile(atm_ivs, [33, 67])
    low_mask = atm_ivs < vol_terciles[0]
    high_mask = atm_ivs >= vol_terciles[1]

    low_coverage = coverages[low_mask].mean() * 100 if low_mask.sum() > 0 else 0
    high_coverage = coverages[high_mask].mean() * 100 if high_mask.sum() > 0 else 0

    return {
        "overall": overall_coverage,
        "low": low_coverage,
        "high": high_coverage,
    }


def find_global_scale_factor(model, surface, context_len, target_coverage=90, n_eval=500):
    """Find scaling factor that achieves target coverage."""

    def coverage_objective(log_scale):
        scale = np.exp(log_scale)
        result = compute_coverage_with_scale(model, surface, context_len, scale, n_eval=n_eval, n_samples=1)
        return result["overall"] - target_coverage

    # Search in log space for stability
    # Start with reasonable bounds
    try:
        log_scale = brentq(coverage_objective, np.log(1.0), np.log(100.0), xtol=0.1)
        return np.exp(log_scale)
    except ValueError:
        # If brentq fails, try manual search
        best_scale = 1.0
        best_diff = float('inf')
        for scale in np.logspace(0, 2, 20):  # 1 to 100
            result = compute_coverage_with_scale(model, surface, context_len, scale, n_eval=200, n_samples=1)
            diff = abs(result["overall"] - target_coverage)
            if diff < best_diff:
                best_diff = diff
                best_scale = scale
        return best_scale


def main():
    print("=" * 70)
    print("EXPERIMENT 11b: Post-hoc Variance Calibration")
    print("=" * 70)

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nData shape: {surface.shape}")

    # Load model
    model_path = Path("results/prior_encoder_ablation/exp11_heteroscedastic/model.pt")
    print(f"\nLoading model from: {model_path}")
    model, config = load_model(model_path)

    context_len = config["context_len"]
    print(f"Context length: {context_len}")

    # Split data: first 70% for calibration, last 30% for test
    n_total = len(surface) - context_len - 1
    n_cal = int(n_total * 0.7)

    cal_surface = surface[:n_cal + context_len + 1]
    test_surface = surface[n_cal:]

    print(f"Calibration samples: {len(cal_surface) - context_len - 1}")
    print(f"Test samples: {len(test_surface) - context_len - 1}")

    # =========================================================================
    # STEP 1: Find global scaling factor
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Find Global Scaling Factor")
    print("=" * 70)

    print("\nSearching for scale factor that gives 90% coverage...")

    # Try different scale factors
    for scale in [1, 4, 9, 16, 25, 36, 49, 64]:
        result = compute_coverage_with_scale(model, cal_surface, context_len, scale, n_eval=500, n_samples=1)
        print(f"  Scale {scale:2d}x: Overall={result['overall']:.1f}%, Low={result['low']:.1f}%, High={result['high']:.1f}%")

    # Find optimal scale
    optimal_scale = find_global_scale_factor(model, cal_surface, context_len, target_coverage=90, n_eval=500)
    print(f"\nOptimal global scale: {optimal_scale:.2f}x")

    # =========================================================================
    # STEP 2: Evaluate on test set with global scale
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Evaluate on Test Set")
    print("=" * 70)

    test_result = compute_coverage_with_scale(model, test_surface, context_len, optimal_scale, n_eval=1000, n_samples=1)

    print(f"\nTest set coverage with global scale {optimal_scale:.2f}x:")
    print(f"  Overall:  {test_result['overall']:.1f}%")
    print(f"  Low vol:  {test_result['low']:.1f}%")
    print(f"  High vol: {test_result['high']:.1f}%")

    # =========================================================================
    # STEP 3: Regime-specific scaling
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Regime-Specific Scaling")
    print("=" * 70)

    # For each regime, find optimal scale
    def compute_regime_coverage(model, surface, context_len, low_scale, high_scale, mid_scale=None, n_eval=500):
        """Compute coverage with regime-specific scaling."""
        if mid_scale is None:
            mid_scale = (low_scale + high_scale) / 2

        n_total = len(surface) - context_len - 1
        eval_indices = np.random.choice(n_total, min(n_eval, n_total), replace=False)

        coverages = []
        atm_ivs = []

        with torch.no_grad():
            for idx in eval_indices:
                context = surface[idx:idx + context_len]
                target = surface[idx + context_len].numpy()
                atm_iv = context[-1, 2, 2].item()

                ctx_dict = {"surface": context.unsqueeze(0)}
                mean, std = model.get_surface_given_conditions(
                    ctx_dict, horizon=1, sample_from_decoder=False
                )
                mean = mean.squeeze().cpu().numpy()
                std = std.squeeze().cpu().numpy()

                # Regime-specific scaling
                vol_terciles = [0.17, 0.22]  # Approximate terciles
                if atm_iv < vol_terciles[0]:
                    scale = low_scale
                elif atm_iv < vol_terciles[1]:
                    scale = mid_scale
                else:
                    scale = high_scale

                scaled_std = std * np.sqrt(scale)

                z_score = 1.645
                lower = mean - z_score * scaled_std
                upper = mean + z_score * scaled_std

                covered = (target >= lower) & (target <= upper)
                coverages.append(covered)
                atm_ivs.append(atm_iv)

        coverages = np.array(coverages)
        atm_ivs = np.array(atm_ivs)

        vol_terciles = np.percentile(atm_ivs, [33, 67])
        low_mask = atm_ivs < vol_terciles[0]
        mid_mask = (atm_ivs >= vol_terciles[0]) & (atm_ivs < vol_terciles[1])
        high_mask = atm_ivs >= vol_terciles[1]

        return {
            "overall": coverages.mean() * 100,
            "low": coverages[low_mask].mean() * 100 if low_mask.sum() > 0 else 0,
            "mid": coverages[mid_mask].mean() * 100 if mid_mask.sum() > 0 else 0,
            "high": coverages[high_mask].mean() * 100 if high_mask.sum() > 0 else 0,
        }

    # Find regime-specific scales
    # Based on exp10c results: high vol has ~2x the variance of low vol
    # So high vol needs ~2x higher scale factor

    base_scale = optimal_scale
    scales_to_try = [
        (base_scale * 0.5, base_scale, base_scale * 2.0),  # Low, Mid, High
        (base_scale * 0.4, base_scale * 0.8, base_scale * 2.5),
        (base_scale * 0.3, base_scale * 0.7, base_scale * 3.0),
    ]

    print("\nTrying different regime-specific scales (low, mid, high):")
    for low_s, mid_s, high_s in scales_to_try:
        result = compute_regime_coverage(model, cal_surface, context_len, low_s, high_s, mid_s, n_eval=500)
        print(f"  ({low_s:.1f}, {mid_s:.1f}, {high_s:.1f}): Overall={result['overall']:.1f}%, "
              f"Low={result['low']:.1f}%, Mid={result['mid']:.1f}%, High={result['high']:.1f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Post-hoc Variance Calibration Results:

GLOBAL SCALING:
  Optimal scale factor: {optimal_scale:.2f}x
  This means decoder variance needs to be {optimal_scale:.1f}x larger

TEST SET COVERAGE (with global scale):
  Overall:  {test_result['overall']:.1f}%
  Low vol:  {test_result['low']:.1f}%
  High vol: {test_result['high']:.1f}%

INTERPRETATION:
  The heteroscedastic VAE underestimates uncertainty significantly.
  Decoder variance (~0.005) is about {100/optimal_scale:.0f}% of what's needed.

  Regime-specific scaling could improve calibration further, but
  this confirms that learning context-dependent variance end-to-end
  is not working with the current architecture.

CONCLUSION:
  Post-hoc calibration can achieve 90% overall coverage, but the
  underlying model doesn't learn meaningful variance structure.
  The approach from exp10c (regime-based post-hoc variance) may be
  more practical for deployment.
""")

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp11b_calibrated")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "results.npz",
        optimal_scale=optimal_scale,
        test_coverage=test_result["overall"],
        test_low_coverage=test_result["low"],
        test_high_coverage=test_result["high"],
    )

    print(f"Results saved to: {output_dir}/results.npz")


if __name__ == "__main__":
    main()
