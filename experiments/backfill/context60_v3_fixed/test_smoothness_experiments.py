"""
Test Smoothness Experiments - Quick Variants (2 + 4)

Tests four variants:
1. Baseline: sample_context=False, ar_phi=0.0
2. Exp 2 only: sample_context=True, ar_phi=0.0
3. Exp 4 only: sample_context=False, ar_phi=0.9
4. Both: sample_context=True, ar_phi=0.9

Measures roughness ratio compared to ground truth.
"""

import numpy as np
import torch
from pathlib import Path
from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior

# Constants
CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)
N_TEST_SAMPLES = 500  # Subsample for speed


def measure_roughness(trajectories):
    """
    Compute roughness as std of day-to-day changes.

    Args:
        trajectories: (N, 90) array

    Returns:
        roughness: float, average std of first differences
    """
    diffs = np.diff(trajectories, axis=1)  # (N, 89)
    roughness = diffs.std(axis=1).mean()
    return roughness


def generate_predictions(model, data, indices, sample_context, ar_phi):
    """Generate predictions with specified experiment settings."""
    model.eval()

    gt_surface = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    predictions = []

    with torch.no_grad():
        for idx in indices:
            # Prepare context
            ctx_start = idx - CONTEXT_LEN
            ctx_end = idx

            if ctx_start < 0 or ctx_end + HORIZON > len(gt_surface):
                continue

            ctx_surf = torch.tensor(
                gt_surface[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            ctx_ex = torch.tensor(
                ex_data[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            context = {"surface": ctx_surf, "ex_feats": ctx_ex}

            # Generate with experiment parameters
            surf_pred, _ = model.get_surface_given_conditions(
                context,
                horizon=HORIZON,
                sample_context=sample_context,
                ar_phi=ar_phi
            )

            # Extract p50 at ATM 6M
            p50 = surf_pred[0, :, 1, ATM_6M[0], ATM_6M[1]].cpu().numpy()
            predictions.append(p50)

    return np.array(predictions)


def main():
    print("="*80)
    print("SMOOTHNESS EXPERIMENTS - Quick Tests (2 + 4)")
    print("="*80)

    # Load model
    model_path = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_best.pt"
    print(f"\nLoading model: {model_path}")

    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    model_config = checkpoint.get('model_config', checkpoint.get('config', {}))

    model = CVAEMemRandConditionalPrior(model_config)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_weights(checkpoint)

    model = model.to('cuda')
    model.eval()
    print("✓ Model loaded")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")

    # Get test indices (insample period)
    all_indices = np.arange(CONTEXT_LEN, 3971 - HORIZON)
    np.random.seed(42)
    test_indices = np.random.choice(all_indices, N_TEST_SAMPLES, replace=False)
    print(f"✓ Testing on {N_TEST_SAMPLES} sequences")

    # Load ground truth trajectories
    gt_trajectories = []
    for idx in test_indices:
        traj = data['surface'][idx:idx+HORIZON, ATM_6M[0], ATM_6M[1]]
        gt_trajectories.append(traj)
    gt_trajectories = np.array(gt_trajectories)
    gt_roughness = measure_roughness(gt_trajectories)

    print(f"\nGround Truth roughness: {gt_roughness:.6f}")
    print()
    print("="*80)
    print("TESTING VARIANTS")
    print("="*80)

    # Test configurations
    configs = [
        ("Baseline", False, 0.0),
        ("Exp 2: Sample Context", True, 0.0),
        ("Exp 4: AR(1) phi=0.9", False, 0.9),
        ("Both: Sample + AR(1)", True, 0.9),
    ]

    results = []

    for name, sample_ctx, ar_phi in configs:
        print(f"\n{name}:")
        print(f"  sample_context={sample_ctx}, ar_phi={ar_phi}")

        # Generate predictions
        preds = generate_predictions(model, data, test_indices, sample_ctx, ar_phi)

        # Measure roughness
        roughness = measure_roughness(preds)
        ratio = roughness / gt_roughness

        print(f"  Roughness: {roughness:.6f}")
        print(f"  Ratio: {ratio:.2%} of GT")

        # Assess
        if ratio > 0.5:
            status = "✅ GOOD"
        elif ratio > 0.3:
            status = "⚠️  MODERATE"
        else:
            status = "❌ TOO SMOOTH"

        print(f"  Status: {status}")

        results.append({
            'name': name,
            'sample_context': sample_ctx,
            'ar_phi': ar_phi,
            'roughness': roughness,
            'ratio': ratio
        })

    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nGT Roughness: {gt_roughness:.6f}")
    print()
    print(f"{'Variant':<25} {'Roughness':>12} {'Ratio':>10} {'Status':>12}")
    print("-"*80)

    for r in results:
        status = "✅ GOOD" if r['ratio'] > 0.5 else "⚠️  MOD" if r['ratio'] > 0.3 else "❌ SMOOTH"
        print(f"{r['name']:<25} {r['roughness']:>12.6f} {r['ratio']:>9.1%} {status:>12}")

    print()

    # Find best
    best = max(results, key=lambda x: x['ratio'])
    print(f"Best variant: {best['name']} ({best['ratio']:.1%} of GT)")
    print()

    # Check if target met
    if best['ratio'] > 0.4:
        print("✅ SUCCESS: Roughness increased to >40% of GT")
    else:
        print(f"⚠️  Target not met: Best is {best['ratio']:.1%}, need >40%")
        print("   Recommendation: Try Experiment 1 (Autoregressive prior)")

    print("="*80)


if __name__ == "__main__":
    main()
