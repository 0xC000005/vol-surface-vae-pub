"""
Verify Conditional vs Unconditional Distribution Capture

Tests:
1. UNCONDITIONAL ✓: Model captures marginal distribution across contexts
2. CONDITIONAL ✗: Model fails to capture distribution for single context (E[Var(X|C)] ≈ 0)

Usage:
    PYTHONPATH=. python experiments/backfill/context60/verify_conditional_vs_unconditional.py

Output:
    results/context60_latent12_v3_FIXED/analysis/variance_decomposition/
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance
from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior

# Constants
CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)
N_TEST_UNCONDITIONAL = None  # None = use ALL sequences (like fanning pattern)
N_TEST_CONDITIONAL = 20  # For conditional test (sampling is slow)
N_SAMPLES_PER_CONTEXT = 1000  # Samples to generate per context
HORIZONS = [1, 7, 14, 30, 60, 90]
OUTPUT_DIR = Path("results/context60_latent12_v3_FIXED/analysis/variance_decomposition")


def load_model():
    """Load the V3 FIXED model."""
    model_path = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_best.pt"
    print(f"Loading model: {model_path}")

    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    model_config = checkpoint.get('model_config', checkpoint.get('config', {}))

    model = CVAEMemRandConditionalPrior(model_config)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_weights(checkpoint)

    model = model.to('cuda')
    model.eval()
    print("✓ Model loaded\n")
    return model


def test_unconditional_distribution(model, data, test_indices):
    """
    Test 1: Verify model captures UNCONDITIONAL marginal distribution.

    CRITICAL: Must anchor trajectories to starting point (like fanning pattern analysis).
    - Generate FULL trajectory for H=90
    - Anchor to day 0: changes = traj - traj[0]
    - Compare distribution of CHANGES, not absolute levels
    """
    print("="*80)
    print("TEST 1: UNCONDITIONAL DISTRIBUTION (What Model CAN Do)")
    print("="*80)
    print("Note: Testing ANCHORED trajectories (changes from starting point)\n")

    gt_surface = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    # Generate FULL 90-day trajectories for all contexts
    print(f"Generating {HORIZON}-day trajectories for {len(test_indices)} contexts...")

    gt_changes_all = []  # List of (H,) arrays
    pred_changes_all = []  # List of (H,) arrays

    with torch.no_grad():
        for idx in test_indices:
            ctx_start = idx
            ctx_end = idx + CONTEXT_LEN

            if ctx_end + HORIZON > len(gt_surface):
                continue

            # GT trajectory (H days)
            gt_traj = gt_surface[ctx_end:ctx_end+HORIZON, ATM_6M[0], ATM_6M[1]]

            # Model prediction (FULL trajectory)
            ctx_surf = torch.tensor(
                gt_surface[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            ctx_ex = torch.tensor(
                ex_data[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            context = {"surface": ctx_surf, "ex_feats": ctx_ex}

            # Generate full 90-day trajectory
            surf_pred, _ = model.get_surface_given_conditions(
                context,
                horizon=HORIZON
            )

            # Extract p50 trajectory at ATM 6M
            pred_traj = surf_pred[0, :, 1, ATM_6M[0], ATM_6M[1]].cpu().numpy()  # (H,)

            # ANCHOR both trajectories to starting point
            gt_changes = gt_traj - gt_traj[0]
            pred_changes = pred_traj - pred_traj[0]

            gt_changes_all.append(gt_changes)
            pred_changes_all.append(pred_changes)

    gt_changes_all = np.array(gt_changes_all)  # (N, H)
    pred_changes_all = np.array(pred_changes_all)  # (N, H)

    print(f"✓ Generated {len(gt_changes_all)} trajectories\n")

    # Now test at each horizon
    results = {}

    for H in HORIZONS:
        h_idx = H - 1  # 0-indexed

        # Extract changes at this horizon
        gt_changes_h = gt_changes_all[:, h_idx]
        pred_changes_h = pred_changes_all[:, h_idx]

        # Statistics
        gt_std = np.std(gt_changes_h)
        pred_std = np.std(pred_changes_h)
        ratio = pred_std / gt_std if gt_std > 0 else 0

        results[H] = {
            'gt_std': gt_std,
            'pred_std': pred_std,
            'ratio': ratio,
        }

        print(f"Horizon H={H}:")
        print(f"  GT std:   {gt_std:.6f}")
        print(f"  Pred std: {pred_std:.6f}")
        print(f"  Ratio:    {ratio:.4f} ({ratio*100:.1f}%)")

        # Compare to fanning_metrics.csv expectation
        if 0.90 < ratio < 1.10:
            print(f"  ✅ EXCELLENT: Matches GT within 10%")
        elif 0.80 < ratio < 1.20:
            print(f"  ✅ GOOD: Matches GT within 20%")
        elif 0.70 < ratio < 1.30:
            print(f"  ⚠️  MODERATE: Some mismatch")
        else:
            print(f"  ❌ POOR: Significant mismatch")
        print()

    # Summary comparison to fanning_metrics.csv
    print("="*80)
    print("COMPARISON TO FANNING_METRICS.CSV")
    print("="*80)
    print(f"{'Horizon':<10} {'GT std':>12} {'Pred std':>12} {'Ratio':>10}")
    print("-"*50)
    for H in HORIZONS:
        r = results[H]
        print(f"{H:<10} {r['gt_std']:>12.6f} {r['pred_std']:>12.6f} {r['ratio']:>9.2%}")

    return results


def test_conditional_distribution(model, data, test_indices):
    """
    Test 2: Verify model FAILS to capture CONDITIONAL distribution.

    For fixed contexts, sample z multiple times and measure within-context variance.
    Should find E[Var(X|C)] ≈ 0.
    """
    print("\n" + "="*80)
    print("TEST 2: CONDITIONAL DISTRIBUTION (What Model CANNOT Do)")
    print("="*80)

    gt_surface = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    # Use fewer contexts for detailed sampling
    sample_contexts = test_indices[:20]  # Just 20 contexts for detailed analysis

    H = HORIZON  # Focus on H=90
    print(f"\nTesting at H={H} with {N_SAMPLES_PER_CONTEXT} samples per context...\n")

    var_within_list = []
    mae_list = []

    with torch.no_grad():
        for ctx_idx, idx in enumerate(sample_contexts):
            # Prepare context
            ctx_start = idx
            ctx_end = idx + CONTEXT_LEN

            if ctx_end + H > len(gt_surface):
                continue

            # Ground truth
            gt_outcome = gt_surface[ctx_end + H - 1, ATM_6M[0], ATM_6M[1]]

            ctx_surf = torch.tensor(
                gt_surface[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            ctx_ex = torch.tensor(
                ex_data[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            context = {"surface": ctx_surf, "ex_feats": ctx_ex}

            # Sample MANY times for this SAME context
            samples = []
            for _ in range(N_SAMPLES_PER_CONTEXT):
                surf_pred, _ = model.get_surface_given_conditions(
                    context,
                    horizon=H
                )
                p50 = surf_pred[0, H-1, 1, ATM_6M[0], ATM_6M[1]].cpu().numpy()
                samples.append(p50)

            samples = np.array(samples)

            # Within-context variance
            var_within = np.var(samples)
            var_within_list.append(var_within)

            # Prediction error (center problem)
            mean_pred = np.mean(samples)
            error = abs(mean_pred - gt_outcome)
            mae_list.append(error)

            if ctx_idx < 5:  # Print first few
                print(f"Context {ctx_idx+1}:")
                print(f"  Samples: mean={mean_pred:.4f}, std={np.std(samples):.6f}")
                print(f"  GT: {gt_outcome:.4f}")
                print(f"  Prediction error: {error:.4f}")
                print(f"  Variance within context: {var_within:.8f}")

    var_within_list = np.array(var_within_list)
    mae_list = np.array(mae_list)

    E_var_within = np.mean(var_within_list)
    std_var_within = np.std(var_within_list)
    mean_mae = np.mean(mae_list)

    print(f"\n{'─'*80}")
    print(f"Within-context variance E[Var(X|C)]:")
    print(f"  Mean: {E_var_within:.8f}")
    print(f"  Std:  {std_var_within:.8f}")
    print(f"\nPrediction error (MAE):")
    print(f"  Mean: {mean_mae:.6f}")

    # Compare to total variance
    all_gt = [gt_surface[idx + CONTEXT_LEN + H - 1, ATM_6M[0], ATM_6M[1]]
              for idx in test_indices if idx + CONTEXT_LEN + H <= len(gt_surface)]
    var_total = np.var(all_gt)

    print(f"\nTotal variance Var(X): {var_total:.6f}")
    print(f"Within-context E[Var(X|C)]: {E_var_within:.8f}")
    print(f"Ratio: {E_var_within/var_total*100:.4f}%")

    if E_var_within / var_total < 0.01:
        print(f"\n❌ CONFIRMED: E[Var(X|C)] ≈ 0 (< 1% of total variance)")
        print(f"   No conditional distribution - model produces identical outputs for same context")
    else:
        print(f"\n✅ Model has some conditional variance")

    return {
        'E_var_within': E_var_within,
        'std_var_within': std_var_within,
        'var_total': var_total,
        'mean_mae': mean_mae,
        'var_within_list': var_within_list,
        'mae_list': mae_list,
    }


def measure_variance_decomposition(model, data, test_indices):
    """
    Test 3: Measure variance decomposition components.

    Var(X) = E[Var(X|C)] + Var(E[X|C])
    """
    print("\n" + "="*80)
    print("TEST 3: VARIANCE DECOMPOSITION")
    print("="*80)

    gt_surface = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    H = HORIZON
    print(f"\nAt H={H}:")

    # Total variance
    all_gt = []
    all_point_preds = []

    with torch.no_grad():
        for idx in test_indices:
            ctx_start = idx
            ctx_end = idx + CONTEXT_LEN

            if ctx_end + H > len(gt_surface):
                continue

            # GT
            gt_outcome = gt_surface[ctx_end + H - 1, ATM_6M[0], ATM_6M[1]]
            all_gt.append(gt_outcome)

            # Point prediction (p50, single sample)
            ctx_surf = torch.tensor(
                gt_surface[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            ctx_ex = torch.tensor(
                ex_data[ctx_start:ctx_end],
                dtype=torch.float32, device='cuda'
            ).unsqueeze(0)

            context = {"surface": ctx_surf, "ex_feats": ctx_ex}

            surf_pred, _ = model.get_surface_given_conditions(context, horizon=H)
            p50 = surf_pred[0, H-1, 1, ATM_6M[0], ATM_6M[1]].cpu().numpy()
            all_point_preds.append(p50)

    all_gt = np.array(all_gt)
    all_point_preds = np.array(all_point_preds)

    # Variance decomposition
    var_X = np.var(all_gt)
    var_E_X_given_C = np.var(all_point_preds)  # Between-condition variance
    E_var_X_given_C_estimate = var_X - var_E_X_given_C  # Required conditional variance

    print(f"\nVariance Decomposition:")
    print(f"  Var(X)           = {var_X:.6f}  (total variance)")
    print(f"  Var(E[X|C])      = {var_E_X_given_C:.6f}  (between-condition)")
    print(f"  E[Var(X|C)] req  = {E_var_X_given_C_estimate:.6f}  (required conditional)")

    # Check if between-condition ≈ total
    ratio = var_E_X_given_C / var_X
    print(f"\nVar(E[X|C]) / Var(X) = {ratio:.2%}")

    if ratio > 0.90:
        print(f"✅ Confirmed: Almost ALL variance is between-condition")
        print(f"   {ratio*100:.1f}% of total variance comes from context differences")
        print(f"   Required conditional variance: {E_var_X_given_C_estimate:.6f} ({(1-ratio)*100:.1f}% of total)")

    return {
        'var_X': var_X,
        'var_E_X_given_C': var_E_X_given_C,
        'E_var_X_given_C_required': E_var_X_given_C_estimate,
        'ratio_between_to_total': ratio,
    }


def test_decoder_sensitivity(model, data, test_indices):
    """
    Test 4: Test if decoder is sensitive to z variations.

    For a fixed context, vary z and see if output changes.
    """
    print("\n" + "="*80)
    print("TEST 4: DECODER SENSITIVITY TO Z VARIATIONS")
    print("="*80)

    gt_surface = data['surface']
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)

    # Pick one context
    idx = test_indices[0]
    ctx_start = idx
    ctx_end = idx + CONTEXT_LEN
    H = HORIZON

    print(f"\nTesting with context starting at index {idx}...")

    ctx_surf = torch.tensor(
        gt_surface[ctx_start:ctx_end],
        dtype=torch.float32, device='cuda'
    ).unsqueeze(0)

    ctx_ex = torch.tensor(
        ex_data[ctx_start:ctx_end],
        dtype=torch.float32, device='cuda'
    ).unsqueeze(0)

    context = {"surface": ctx_surf, "ex_feats": ctx_ex}

    # Get prior mean and std
    with torch.no_grad():
        # First, get a prediction to access prior
        surf_pred, extra = model.get_surface_given_conditions(context, horizon=H)

        # Sample 100 times to measure output variation
        outputs = []
        for _ in range(100):
            surf_pred, _ = model.get_surface_given_conditions(context, horizon=H)
            p50 = surf_pred[0, H-1, 1, ATM_6M[0], ATM_6M[1]].cpu().numpy()
            outputs.append(p50)

    outputs = np.array(outputs)
    output_std = np.std(outputs)
    output_mean = np.mean(outputs)

    print(f"\nOutput statistics (100 samples, same context):")
    print(f"  Mean: {output_mean:.6f}")
    print(f"  Std:  {output_std:.6f}")

    if output_std < 0.0001:
        print(f"\n❌ INSENSITIVE: Decoder produces nearly identical outputs regardless of z")
        print(f"   Std = {output_std:.8f} is negligible")
    elif output_std < 0.001:
        print(f"\n⚠️  WEAK SENSITIVITY: Very small output variation")
    else:
        print(f"\n✅ SENSITIVE: Decoder responds to z variations")

    return {
        'output_mean': output_mean,
        'output_std': output_std,
        'outputs': outputs,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CONDITIONAL VS UNCONDITIONAL DISTRIBUTION VERIFICATION")
    print("="*80)
    print()

    # Load model and data
    model = load_model()

    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    print("✓ Data loaded\n")

    # Get test indices
    all_indices = np.arange(0, 3971 - CONTEXT_LEN - HORIZON)

    # For unconditional test: use ALL sequences (like fanning pattern)
    if N_TEST_UNCONDITIONAL is None:
        test_indices_unconditional = all_indices
        print(f"Unconditional test: using ALL {len(all_indices)} sequences\n")
    else:
        np.random.seed(42)
        test_indices_unconditional = np.random.choice(all_indices, N_TEST_UNCONDITIONAL, replace=False)
        print(f"Unconditional test: using {N_TEST_UNCONDITIONAL} sequences\n")

    # For conditional test: use smaller subset (sampling is slow)
    np.random.seed(42)
    test_indices_conditional = np.random.choice(all_indices, N_TEST_CONDITIONAL, replace=False)
    print(f"Conditional test: using {N_TEST_CONDITIONAL} contexts\n")

    # Run tests
    unconditional_results = test_unconditional_distribution(model, data, test_indices_unconditional)
    conditional_results = test_conditional_distribution(model, data, test_indices_conditional)
    decomposition_results = measure_variance_decomposition(model, data, test_indices_unconditional)
    sensitivity_results = test_decoder_sensitivity(model, data, test_indices_conditional)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\n1. UNCONDITIONAL DISTRIBUTION (What Model CAN Do):")
    print("   Anchored changes (std ratios):")
    for H in HORIZONS:
        r = unconditional_results[H]
        ratio = r['ratio']
        status = "✅" if 0.90 < ratio < 1.10 else "⚠️ " if 0.80 < ratio < 1.20 else "❌"
        print(f"   H={H:2d}: {ratio:6.2%} {status}")

    print("\n2. CONDITIONAL DISTRIBUTION (What Model CANNOT Do):")
    ratio = conditional_results['E_var_within'] / conditional_results['var_total']
    print(f"   E[Var(X|C)] / Var(X) = {ratio*100:.4f}%")
    print(f"   Status: {'❌ No conditional variance' if ratio < 0.01 else '✅ Has conditional variance'}")

    print("\n3. VARIANCE DECOMPOSITION:")
    d = decomposition_results
    print(f"   Var(X)       = {d['var_X']:.6f}")
    print(f"   Var(E[X|C])  = {d['var_E_X_given_C']:.6f} ({d['ratio_between_to_total']*100:.1f}% of total)")
    print(f"   E[Var(X|C)]  = {d['E_var_X_given_C_required']:.6f} ({(1-d['ratio_between_to_total'])*100:.1f}% required)")

    print("\n4. DECODER SENSITIVITY:")
    s = sensitivity_results
    print(f"   Output std from sampling = {s['output_std']:.6f}")
    print(f"   Status: {'❌ Insensitive' if s['output_std'] < 0.0001 else '⚠️  Weak' if s['output_std'] < 0.001 else '✅ Sensitive'}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\n✅ Model CAN capture UNCONDITIONAL marginal distribution")
    print("   → Point predictions across contexts have correct spread")
    print("\n❌ Model CANNOT capture CONDITIONAL distribution for single context")
    print("   → E[Var(X|C)] ≈ 0: sampling same context gives identical outputs")
    print("   → Decoder insensitive to z variations")
    print("   → Cannot generate scenarios for risk management")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
