"""
Experiment 9a Debug: Why did conformal prediction achieve only 47.5% coverage?

HYPOTHESIS:
1. Distribution shift between consecutive time periods (not exchangeable)
2. Systematic prediction bias varies by grid point
3. Absolute errors hide signed bias direction

DIAGNOSTICS:
1. Compute residuals (signed, not absolute)
2. Check for systematic bias per grid point
3. Compare calibration vs test distributions
4. Test with signed quantile-based intervals
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior


def load_model(model_path, model_class):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']

    model = model_class(model_config)

    state_dict = checkpoint['state_dict']
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            cleaned_state_dict[k[10:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()

    return model, model_config


def get_deterministic_prediction(model, context):
    """Get deterministic prediction (z=0, no sampling)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    C = model.config["context_len"]
    latent_dim = model.config["latent_dim"]

    context = context.unsqueeze(0).to(device).to(dtype)
    ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
    ctx_input = {"surface": context, "ex_feats": ctx_feats}

    with torch.no_grad():
        # Get context encoding
        ctx_out = model.ctx_encoder(ctx_input)
        context_summary = ctx_out[:, -1, :]

        # Get prior mean (deterministic, z = mu_p)
        mu_p, _ = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

        # Decode with z = mu_p (deterministic)
        # mu_p shape is already (B, 1, latent_dim)
        if model.config.get("compress_context", True):
            ctx_embedding_dim = latent_dim
        else:
            ctx_embedding_dim = model.config["mem_hidden"]

        ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
        decoder_input = torch.cat([mu_p, ctx_zeros], dim=-1)
        decoded = model.decoder(decoder_input)

        if isinstance(decoded, tuple):
            decoded = decoded[0]

    return decoded.squeeze().cpu().numpy()


def main():
    print("="*70)
    print("EXPERIMENT 9a DEBUG: Why conformal got 47.5% coverage")
    print("="*70)

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nTotal data: {surface.shape}")

    # Load model
    model_path = Path("results/prior_encoder_ablation/extended_training_v5/baseline_ep200.pt")
    print(f"\nLoading model from: {model_path}")
    model, config = load_model(model_path, CVAEFullCovPrior)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    C = config["context_len"]
    print(f"Context length: {C}")

    # Same split as exp9a
    n_total = len(surface) - C - 1
    n_cal = n_total // 5
    n_test = n_total // 5
    cal_start = n_total - n_cal - n_test
    test_start = n_total - n_test

    print(f"\nData split (CONSECUTIVE - potential distribution shift!):")
    print(f"  Calibration: indices {cal_start} to {test_start-1}")
    print(f"  Test: indices {test_start} to {n_total-1}")

    # Subsample
    n_sample = 200
    cal_indices = np.random.choice(range(cal_start, test_start), n_sample, replace=False)
    test_indices = np.random.choice(range(test_start, n_total), n_sample, replace=False)

    # =========================================================================
    # DIAGNOSTIC 1: Compute signed residuals
    # =========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: Signed Residuals (prediction - target)")
    print("="*70)

    cal_residuals = []
    for i in tqdm(cal_indices, desc="Calibration residuals"):
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        residual = pred - target  # SIGNED
        cal_residuals.append(residual)
    cal_residuals = np.array(cal_residuals)

    test_residuals = []
    for i in tqdm(test_indices, desc="Test residuals"):
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        residual = pred - target  # SIGNED
        test_residuals.append(residual)
    test_residuals = np.array(test_residuals)

    # Per-grid-point bias
    cal_bias = cal_residuals.mean(axis=0)
    test_bias = test_residuals.mean(axis=0)
    bias_shift = test_bias - cal_bias

    print("\nCalibration set bias (mean residual) per grid point:")
    np.set_printoptions(precision=4)
    print(cal_bias)

    print("\nTest set bias (mean residual) per grid point:")
    print(test_bias)

    print("\nBias SHIFT (test - cal):")
    print(bias_shift)
    print(f"  Max shift: {np.abs(bias_shift).max():.4f}")

    # =========================================================================
    # DIAGNOSTIC 2: Distribution shift in targets
    # =========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: Distribution Shift in Targets")
    print("="*70)

    cal_targets = surface[cal_indices + C].numpy()
    test_targets = surface[test_indices + C].numpy()

    print(f"\nCalibration targets: mean={cal_targets.mean():.4f}, std={cal_targets.std():.4f}")
    print(f"Test targets: mean={test_targets.mean():.4f}, std={test_targets.std():.4f}")

    target_mean_shift = test_targets.mean() - cal_targets.mean()
    target_std_shift = test_targets.std() - cal_targets.std()
    print(f"\nTarget distribution shift:")
    print(f"  Mean shift: {target_mean_shift:.4f}")
    print(f"  Std shift: {target_std_shift:.4f}")

    # =========================================================================
    # DIAGNOSTIC 3: Conformal with signed quantiles
    # =========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Signed Quantile Intervals")
    print("="*70)

    # Compute empirical quantiles from signed residuals
    lower_q = np.percentile(cal_residuals, 5, axis=0)  # 5th percentile
    upper_q = np.percentile(cal_residuals, 95, axis=0)  # 95th percentile

    print("\nCalibration residual quantiles:")
    print(f"  5th percentile (lower_q):")
    print(lower_q)
    print(f"  95th percentile (upper_q):")
    print(upper_q)

    # Check coverage on test set with signed intervals
    # Interval: [pred + lower_q, pred + upper_q]
    # Note: lower_q is typically negative, upper_q is typically positive
    covered_signed = []
    for i in range(len(test_residuals)):
        residual = test_residuals[i]
        in_interval = (residual >= lower_q) & (residual <= upper_q)
        covered_signed.append(in_interval)
    covered_signed = np.array(covered_signed)

    signed_coverage = covered_signed.mean() * 100
    signed_per_grid = covered_signed.mean(axis=0) * 100

    print(f"\nSigned interval coverage (90% target):")
    print(f"  Overall: {signed_coverage:.1f}%")
    print(f"  Per-grid:")
    print(signed_per_grid)

    # =========================================================================
    # DIAGNOSTIC 4: Compare with exp9a's absolute error approach
    # =========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: Absolute Error Approach (exp9a method)")
    print("="*70)

    cal_abs_errors = np.abs(cal_residuals)
    n = len(cal_abs_errors)
    adjusted_quantile = min(np.ceil(0.90 * (n + 1)) / n, 1.0)
    q_abs = np.quantile(cal_abs_errors, adjusted_quantile, axis=0)

    print(f"\nAbsolute error thresholds (90th percentile):")
    print(q_abs)

    # Coverage with symmetric intervals
    covered_abs = []
    for i in range(len(test_residuals)):
        abs_error = np.abs(test_residuals[i])
        in_interval = abs_error <= q_abs
        covered_abs.append(in_interval)
    covered_abs = np.array(covered_abs)

    abs_coverage = covered_abs.mean() * 100
    abs_per_grid = covered_abs.mean(axis=0) * 100

    print(f"\nAbsolute error coverage (90% target):")
    print(f"  Overall: {abs_coverage:.1f}%")
    print(f"  Per-grid:")
    print(abs_per_grid)

    # =========================================================================
    # DIAGNOSTIC 5: Test with random split (check exchangeability)
    # =========================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 5: Random Split (test exchangeability)")
    print("="*70)

    # Combine and randomly shuffle
    all_indices = np.concatenate([cal_indices, test_indices])
    np.random.shuffle(all_indices)
    rand_cal_idx = all_indices[:n_sample]
    rand_test_idx = all_indices[n_sample:]

    # Compute residuals for random split
    rand_cal_residuals = []
    for i in rand_cal_idx:
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        rand_cal_residuals.append(pred - target)
    rand_cal_residuals = np.array(rand_cal_residuals)

    rand_test_residuals = []
    for i in rand_test_idx:
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        rand_test_residuals.append(pred - target)
    rand_test_residuals = np.array(rand_test_residuals)

    # Signed quantiles on random cal
    rand_lower_q = np.percentile(rand_cal_residuals, 5, axis=0)
    rand_upper_q = np.percentile(rand_cal_residuals, 95, axis=0)

    # Coverage on random test
    rand_covered = []
    for i in range(len(rand_test_residuals)):
        residual = rand_test_residuals[i]
        in_interval = (residual >= rand_lower_q) & (residual <= rand_upper_q)
        rand_covered.append(in_interval)
    rand_covered = np.array(rand_covered)

    rand_coverage = rand_covered.mean() * 100
    rand_per_grid = rand_covered.mean(axis=0) * 100

    print(f"\nRandom split + signed quantiles (90% target):")
    print(f"  Overall: {rand_coverage:.1f}%")
    print(f"  Per-grid:")
    print(rand_per_grid)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Coverage comparison:
  1. Original exp9a (consecutive split, |error|):  47.5%
  2. Consecutive split + signed quantiles:         {signed_coverage:.1f}%
  3. Consecutive split + |error| (reproduced):     {abs_coverage:.1f}%
  4. Random split + signed quantiles:              {rand_coverage:.1f}%

Diagnosis:
  - Bias shift between cal/test: {np.abs(bias_shift).max():.4f}
  - Target mean shift: {target_mean_shift:.4f}

Root cause analysis:
""")

    if np.abs(bias_shift).max() > 0.01:
        print("  [!] SIGNIFICANT BIAS SHIFT detected between calibration and test periods")
        print("      → Time-ordered split violates exchangeability assumption")

    if rand_coverage > 85:
        print("  [✓] Random split achieves ~90% coverage")
        print("      → Confirms exchangeability is the key issue")
    else:
        print("  [?] Random split coverage still low - other issues may exist")

    if signed_coverage > abs_coverage:
        print("  [✓] Signed quantiles outperform absolute error approach")
        print("      → Asymmetric intervals handle bias better")

    # Save diagnostic results
    output_dir = Path("results/prior_encoder_ablation/exp9_conformal")
    np.savez(
        output_dir / "debug_results.npz",
        cal_residuals=cal_residuals,
        test_residuals=test_residuals,
        cal_bias=cal_bias,
        test_bias=test_bias,
        bias_shift=bias_shift,
        signed_coverage=signed_coverage,
        abs_coverage=abs_coverage,
        rand_coverage=rand_coverage,
        signed_per_grid=signed_per_grid,
        abs_per_grid=abs_per_grid,
        rand_per_grid=rand_per_grid
    )
    print(f"\nResults saved to: {output_dir}/debug_results.npz")


if __name__ == "__main__":
    main()
