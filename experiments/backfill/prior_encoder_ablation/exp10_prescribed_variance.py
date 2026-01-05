"""
Experiment 10: Prescribed Conditional Variance (Analytical Approach)

INSIGHT:
If conditional variance cannot be learned from data (one outcome per context),
it must be PRESCRIBED analytically using the Law of Total Variance:

    Var(X) = E[Var(X|C)] + Var(E[X|C])

Rearranging:
    σ²_conditional = Var(X) - Var(E[X|C])
                   = GT_variance - prediction_variance

This guarantees:
1. Marginal variance is preserved by construction
2. CI calibration through empirical quantiles (not Gaussian assumption)

KEY LESSON FROM EXP9A DEBUG:
- Consecutive time splits violate exchangeability → 47.5% coverage
- Random splits preserve exchangeability → 85.6% coverage
- Use RANDOM splits for calibration/test

ALGORITHM:
1. Compute deterministic predictions for all data points
2. Measure Var(predictions) = Var(E[X|C])
3. Compute prescribed σ² = Var(X) - Var(E[X|C])
4. Use empirical residual quantiles for CI (handles non-Gaussian)
5. Verify coverage with random split
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
    model = model.to(device)
    model.eval()

    return model, model_config


def get_deterministic_prediction(model, context):
    """Get deterministic prediction (z = prior mean, no sampling)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    C = model.config["context_len"]
    latent_dim = model.config["latent_dim"]

    context = context.unsqueeze(0).to(device).to(dtype)
    ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
    ctx_input = {"surface": context, "ex_feats": ctx_feats}

    with torch.no_grad():
        ctx_out = model.ctx_encoder(ctx_input)
        context_summary = ctx_out[:, -1, :]

        # Get prior mean (deterministic)
        mu_p, _ = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

        # Decode with z = mu_p
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


def generate_with_prescribed_noise(prediction, sigma_conditional):
    """Generate sample by adding prescribed noise to prediction."""
    noise = np.random.normal(0, sigma_conditional, size=prediction.shape)
    return prediction + noise


def main():
    print("=" * 70)
    print("EXPERIMENT 10: Prescribed Conditional Variance")
    print("=" * 70)

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nTotal data: {surface.shape}")

    # Load model
    model_path = Path("results/prior_encoder_ablation/extended_training_v5/baseline_ep200.pt")
    print(f"\nLoading model from: {model_path}")
    model, config = load_model(model_path, CVAEFullCovPrior)

    C = config["context_len"]
    print(f"Context length: {C}")

    # =========================================================================
    # STEP 1: Compute GT statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Ground Truth Statistics")
    print("=" * 70)

    gt_surfaces = surface[C:].numpy()  # All surfaces after context window
    gt_mean = gt_surfaces.mean(axis=0)
    gt_var = gt_surfaces.var(axis=0)
    gt_std = np.sqrt(gt_var)

    print(f"\nGT statistics per grid point:")
    print(f"  Overall mean: {gt_mean.mean():.4f}")
    print(f"  Overall var:  {gt_var.mean():.6f}")
    print(f"  Overall std:  {gt_std.mean():.4f}")

    # =========================================================================
    # STEP 2: Compute predictions on full dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Compute Deterministic Predictions")
    print("=" * 70)

    n_total = len(surface) - C - 1
    n_sample = min(1000, n_total)  # Use up to 1000 points
    sample_indices = np.random.choice(n_total, n_sample, replace=False)

    predictions = []
    targets = []

    for i in tqdm(sample_indices, desc="Computing predictions"):
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        predictions.append(pred)
        targets.append(target)

    predictions = np.array(predictions)  # (N, 5, 5)
    targets = np.array(targets)  # (N, 5, 5)

    print(f"\nPredictions shape: {predictions.shape}")

    # =========================================================================
    # STEP 3: Analytical Variance Prescription
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Analytical Variance Prescription")
    print("=" * 70)

    # Var(E[X|C]) - variance of predictions
    var_predictions = predictions.var(axis=0)

    # Var(X) - variance of targets (should match GT)
    var_targets = targets.var(axis=0)

    # σ²_conditional = Var(X) - Var(E[X|C])
    # Use GT variance for more stable estimate
    sigma_sq_analytical = np.maximum(0, gt_var - var_predictions)
    sigma_conditional = np.sqrt(sigma_sq_analytical)

    print("\nLaw of Total Variance decomposition:")
    print(f"  Var(X) [GT]:           {gt_var.mean():.6f}")
    print(f"  Var(E[X|C]) [pred]:    {var_predictions.mean():.6f}")
    print(f"  σ²_conditional [diff]: {sigma_sq_analytical.mean():.6f}")
    print(f"  σ_conditional [std]:   {sigma_conditional.mean():.4f}")

    # Sanity check: var_targets should be close to gt_var
    print(f"\n  Sample Var(X):         {var_targets.mean():.6f}")
    print(f"  Ratio to GT:           {var_targets.mean() / gt_var.mean():.2%}")

    # Check what fraction of variance is explained by predictions
    r_squared = var_predictions.mean() / gt_var.mean()
    print(f"\n  R² (prediction variance / total variance): {r_squared:.2%}")

    # =========================================================================
    # STEP 4: Empirical Residual Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Empirical Residual Analysis")
    print("=" * 70)

    residuals = targets - predictions  # (N, 5, 5) - Note: target - pred (not pred - target)
    residual_mean = residuals.mean(axis=0)
    residual_std = residuals.std(axis=0)

    print("\nResidual statistics:")
    print(f"  Mean (bias): {residual_mean.mean():.4f}")
    print(f"  Std:         {residual_std.mean():.4f}")

    # Compare with analytical σ_conditional
    print(f"\n  Analytical σ_conditional: {sigma_conditional.mean():.4f}")
    print(f"  Empirical residual std:   {residual_std.mean():.4f}")
    print(f"  Ratio:                    {residual_std.mean() / sigma_conditional.mean():.2%}")

    # =========================================================================
    # STEP 5: CI Calibration with RANDOM Split
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CI Calibration (Random Split)")
    print("=" * 70)

    # Random split for calibration and test
    np.random.shuffle(sample_indices)
    n_cal = n_sample // 2
    cal_indices = sample_indices[:n_cal]
    test_indices = sample_indices[n_cal:]

    # Recompute for clean split
    cal_residuals = []
    test_predictions = []
    test_targets = []

    for i in tqdm(cal_indices, desc="Calibration set"):
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        cal_residuals.append(target - pred)

    for i in tqdm(test_indices, desc="Test set"):
        context = surface[i:i+C]
        target = surface[i+C].numpy()
        pred = get_deterministic_prediction(model, context)
        test_predictions.append(pred)
        test_targets.append(target)

    cal_residuals = np.array(cal_residuals)
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    print(f"\nCalibration set: {len(cal_residuals)}")
    print(f"Test set: {len(test_predictions)}")

    # =========================================================================
    # METHOD A: Gaussian CI with prescribed σ
    # =========================================================================
    print("\n" + "-" * 50)
    print("METHOD A: Gaussian CI (mean ± 1.645σ)")
    print("-" * 50)

    # Compute bias correction from calibration set
    bias = cal_residuals.mean(axis=0)
    print(f"  Bias (cal): {bias.mean():.4f}")

    # CI = (pred + bias) ± 1.645 * σ_conditional
    covered_gaussian = []
    for i in range(len(test_predictions)):
        pred_corrected = test_predictions[i] + bias
        lower = pred_corrected - 1.645 * sigma_conditional
        upper = pred_corrected + 1.645 * sigma_conditional
        in_interval = (test_targets[i] >= lower) & (test_targets[i] <= upper)
        covered_gaussian.append(in_interval)

    covered_gaussian = np.array(covered_gaussian)
    gaussian_coverage = covered_gaussian.mean() * 100
    gaussian_per_grid = covered_gaussian.mean(axis=0) * 100

    print(f"\n  Gaussian CI coverage (90% target):")
    print(f"    Overall: {gaussian_coverage:.1f}%")
    print(f"    Per-grid range: [{gaussian_per_grid.min():.1f}%, {gaussian_per_grid.max():.1f}%]")

    # =========================================================================
    # METHOD B: Empirical Quantile CI (handles non-Gaussian)
    # =========================================================================
    print("\n" + "-" * 50)
    print("METHOD B: Empirical Quantile CI")
    print("-" * 50)

    # Compute empirical quantiles from calibration residuals
    lower_q = np.percentile(cal_residuals, 5, axis=0)
    upper_q = np.percentile(cal_residuals, 95, axis=0)

    print(f"  Lower quantile (5th): {lower_q.mean():.4f}")
    print(f"  Upper quantile (95th): {upper_q.mean():.4f}")
    print(f"  Interval width: {(upper_q - lower_q).mean():.4f}")

    # CI = [pred + lower_q, pred + upper_q]
    covered_quantile = []
    for i in range(len(test_predictions)):
        lower = test_predictions[i] + lower_q
        upper = test_predictions[i] + upper_q
        in_interval = (test_targets[i] >= lower) & (test_targets[i] <= upper)
        covered_quantile.append(in_interval)

    covered_quantile = np.array(covered_quantile)
    quantile_coverage = covered_quantile.mean() * 100
    quantile_per_grid = covered_quantile.mean(axis=0) * 100

    print(f"\n  Quantile CI coverage (90% target):")
    print(f"    Overall: {quantile_coverage:.1f}%")
    print(f"    Per-grid range: [{quantile_per_grid.min():.1f}%, {quantile_per_grid.max():.1f}%]")

    # =========================================================================
    # STEP 6: Marginal Verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Marginal Verification")
    print("=" * 70)

    # Generate samples with prescribed noise
    n_gen = 1000
    generated_samples = []
    gen_indices = np.random.choice(n_total, min(n_gen, n_total), replace=False)

    for i in tqdm(gen_indices, desc="Generating with prescribed noise"):
        context = surface[i:i+C]
        pred = get_deterministic_prediction(model, context)
        # Add bias correction and prescribed noise
        sample = pred + bias + np.random.normal(0, sigma_conditional)
        generated_samples.append(sample)

    generated_samples = np.array(generated_samples)

    gen_mean = generated_samples.mean(axis=0)
    gen_var = generated_samples.var(axis=0)

    mean_ratio = gen_mean.mean() / gt_mean.mean() * 100
    var_ratio = gen_var.mean() / gt_var.mean() * 100

    print(f"\nMarginal comparison:")
    print(f"  GT Mean:  {gt_mean.mean():.4f}")
    print(f"  Gen Mean: {gen_mean.mean():.4f}")
    print(f"  Ratio:    {mean_ratio:.1f}%")

    print(f"\n  GT Var:   {gt_var.mean():.6f}")
    print(f"  Gen Var:  {gen_var.mean():.6f}")
    print(f"  Ratio:    {var_ratio:.1f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Experiment 10: Prescribed Conditional Variance Results

ANALYTICAL PRESCRIPTION:
  Law of Total Variance: Var(X) = Var(E[X|C]) + σ²_conditional

  GT Variance:           {gt_var.mean():.6f}
  Prediction Variance:   {var_predictions.mean():.6f}
  Prescribed σ²:         {sigma_sq_analytical.mean():.6f}
  R² (pred var / total): {r_squared:.2%}

CI CALIBRATION (Random Split, 90% target):
  Method A (Gaussian):   {gaussian_coverage:.1f}%
  Method B (Quantile):   {quantile_coverage:.1f}%

MARGINAL MATCHING:
  Gen Mean / GT Mean:    {mean_ratio:.1f}%
  Gen Var / GT Var:      {var_ratio:.1f}%

SUCCESS CRITERIA:
  ✓ Gen Mean in [95%, 105%]: {"[OK]" if 95 <= mean_ratio <= 105 else "[FAIL]"}
  ✓ Gen Var in [95%, 105%]:  {"[OK]" if 95 <= var_ratio <= 105 else "[FAIL]"}
  ✓ CI Coverage ≥ 88%:       {"[OK]" if quantile_coverage >= 88 else "[FAIL]"}
""")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = Path("results/prior_encoder_ablation/exp10_prescribed_variance")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "results.npz",
        # Analytical
        gt_var=gt_var,
        var_predictions=var_predictions,
        sigma_sq_analytical=sigma_sq_analytical,
        sigma_conditional=sigma_conditional,
        bias=bias,
        # Empirical quantiles
        lower_q=lower_q,
        upper_q=upper_q,
        # Coverage
        gaussian_coverage=gaussian_coverage,
        quantile_coverage=quantile_coverage,
        gaussian_per_grid=gaussian_per_grid,
        quantile_per_grid=quantile_per_grid,
        # Marginal
        gen_mean=gen_mean,
        gen_var=gen_var,
        gt_mean=gt_mean,
        mean_ratio=mean_ratio,
        var_ratio=var_ratio
    )

    print(f"Results saved to: {output_dir}/results.npz")


if __name__ == "__main__":
    main()
