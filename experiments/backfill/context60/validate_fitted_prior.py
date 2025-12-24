"""
Validate Fitted Prior Implementation

Validates that fitted GMM prior fixes the systematic negative bias by:
1. Generating test predictions with both standard and fitted prior
2. Computing marginal statistics (mean, median, spread)
3. Comparing fanning patterns
4. Checking for systematic bias

Usage:
    PYTHONPATH=. python experiments/backfill/context60/validate_fitted_prior.py

Output:
    Console report comparing standard vs fitted prior performance
"""

import numpy as np
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from pathlib import Path


# Configuration
MODEL_PATH = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_best.pt"
FITTED_PRIOR_PATH = "models/backfill/context60_experiment/fitted_prior_gmm.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)
N_TEST_SAMPLES = 100  # Number of sequences to test


def load_model():
    """Load trained VAE model."""
    print(f"Loading model from {MODEL_PATH}...")
    model_data = torch.load(MODEL_PATH, weights_only=False)

    model = CVAEMemRand(model_data["model_config"])
    model.load_weights(dict_to_load=model_data)
    model.eval()

    print(f"✓ Model loaded (latent_dim={model.config['latent_dim']})")
    return model


def generate_test_predictions(model, data, prior_mode="standard"):
    """Generate test predictions using specified prior mode."""
    print(f"\nGenerating predictions with prior_mode='{prior_mode}'...")

    vol_surf = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    predictions = []
    indices = []

    # Generate predictions for test samples
    with torch.no_grad():
        for i in range(CONTEXT_LEN, min(CONTEXT_LEN + N_TEST_SAMPLES, len(vol_surf) - HORIZON)):
            # Get context
            context_surface = vol_surf[i-CONTEXT_LEN:i]
            context_ex = ex_data[i-CONTEXT_LEN:i]

            context = {
                "surface": torch.from_numpy(context_surface).unsqueeze(0).double(),
                "ex_feats": torch.from_numpy(context_ex).unsqueeze(0).double()
            }

            # Generate prediction
            surf_pred, ex_pred = model.get_surface_given_conditions(
                context, z=None, horizon=HORIZON, prior_mode=prior_mode
            )

            # Extract ATM 6M p50 (median)
            pred_p50 = surf_pred[0, :, 1, ATM_6M[0], ATM_6M[1]].cpu().numpy()  # (90,)

            # Anchor to starting point (0)
            pred_p50 = pred_p50 - pred_p50[0]

            predictions.append(pred_p50)
            indices.append(i)

    predictions = np.array(predictions)  # (N, 90)
    print(f"  Generated {len(predictions)} predictions")

    return predictions, indices


def compute_fanning_statistics(predictions, label=""):
    """Compute fanning pattern statistics."""
    # Compute percentile envelope
    p05 = np.percentile(predictions, 5, axis=0)
    p50 = np.percentile(predictions, 50, axis=0)
    p95 = np.percentile(predictions, 95, axis=0)

    # Envelope widths
    width = p95 - p05

    # Mean trajectory (should be near 0 if unbiased)
    mean_traj = np.mean(predictions, axis=0)

    stats = {
        'day1_width': width[0],
        'day30_width': width[29],
        'day60_width': width[59],
        'day90_width': width[89],
        'mean_day1': mean_traj[0],
        'mean_day30': mean_traj[29],
        'mean_day60': mean_traj[59],
        'mean_day90': mean_traj[89],
        'p50_day1': p50[0],
        'p50_day30': p50[29],
        'p50_day60': p50[59],
        'p50_day90': p50[89],
    }

    return stats


def validate():
    """Main validation pipeline."""
    print("=" * 80)
    print("FITTED PRIOR VALIDATION")
    print("=" * 80)
    print()

    # Load model
    model = load_model()

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    print(f"✓ Data loaded ({len(data['surface'])} days)")

    # Generate predictions with standard prior
    standard_preds, indices = generate_test_predictions(model, data, prior_mode="standard")
    standard_stats = compute_fanning_statistics(standard_preds, "Standard")

    # Load fitted prior and generate predictions
    print(f"\nLoading fitted prior from {FITTED_PRIOR_PATH}...")
    model.load_fitted_prior(FITTED_PRIOR_PATH)
    print("✓ Fitted prior loaded")

    fitted_preds, _ = generate_test_predictions(model, data, prior_mode="fitted")
    fitted_stats = compute_fanning_statistics(fitted_preds, "Fitted")

    # Print comparison
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    print("Envelope Width (p05-p95 spread):")
    print(f"{'Horizon':<12} {'Standard':<15} {'Fitted':<15} {'Difference':<15}")
    print("-" * 60)
    for day in [1, 30, 60, 90]:
        std_width = standard_stats[f'day{day}_width']
        fit_width = fitted_stats[f'day{day}_width']
        diff = fit_width - std_width
        print(f"{'Day-' + str(day):<12} {std_width:>14.6f}  {fit_width:>14.6f}  {diff:>+14.6f}")

    print("\nMean Trajectory (should be near 0 if unbiased):")
    print(f"{'Horizon':<12} {'Standard':<15} {'Fitted':<15} {'Improvement':<15}")
    print("-" * 60)
    for day in [1, 30, 60, 90]:
        std_mean = standard_stats[f'mean_day{day}']
        fit_mean = fitted_stats[f'mean_day{day}']
        improvement = abs(std_mean) - abs(fit_mean)
        marker = "✓" if improvement > 0 else ("=" if abs(improvement) < 1e-6 else "↓")
        print(f"{'Day-' + str(day):<12} {std_mean:>+14.6f}  {fit_mean:>+14.6f}  {improvement:>+14.6f} {marker}")

    print("\nMedian Trajectory (p50 across contexts):")
    print(f"{'Horizon':<12} {'Standard':<15} {'Fitted':<15} {'Difference':<15}")
    print("-" * 60)
    for day in [1, 30, 60, 90]:
        std_p50 = standard_stats[f'p50_day{day}']
        fit_p50 = fitted_stats[f'p50_day{day}']
        diff = fit_p50 - std_p50
        print(f"{'Day-' + str(day):<12} {std_p50:>+14.6f}  {fit_p50:>+14.6f}  {diff:>+14.6f}")

    # Overall assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    # Check for bias reduction
    std_mean_abs = np.abs([standard_stats[f'mean_day{d}'] for d in [1, 30, 60, 90]]).mean()
    fit_mean_abs = np.abs([fitted_stats[f'mean_day{d}'] for d in [1, 30, 60, 90]]).mean()

    print(f"\nAverage absolute mean trajectory:")
    print(f"  Standard: {std_mean_abs:.6f}")
    print(f"  Fitted:   {fit_mean_abs:.6f}")
    print(f"  Improvement: {std_mean_abs - fit_mean_abs:+.6f}")

    if fit_mean_abs < std_mean_abs:
        print("  ✓ Fitted prior reduces systematic bias!")
    elif abs(fit_mean_abs - std_mean_abs) < 1e-5:
        print("  = Similar bias levels (fitted prior may not be necessary)")
    else:
        print("  ⚠ Fitted prior does not reduce bias (investigate further)")

    # Check envelope width
    std_width_90 = standard_stats['day90_width']
    fit_width_90 = fitted_stats['day90_width']
    width_ratio = fit_width_90 / std_width_90

    print(f"\nDay-90 envelope width ratio (fitted/standard): {width_ratio:.3f}")
    if 0.9 < width_ratio < 1.1:
        print("  ✓ Envelope widths are similar (good)")
    elif width_ratio < 0.9:
        print("  ⚠ Fitted prior produces narrower envelopes (under-dispersed?)")
    else:
        print("  ⚠ Fitted prior produces wider envelopes (over-dispersed?)")

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. If validation looks good, regenerate full predictions with fitted prior")
    print("  2. Re-run fanning pattern visualization to verify bias correction")
    print("  3. Compare CI calibration and co-integration preservation")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    validate()
