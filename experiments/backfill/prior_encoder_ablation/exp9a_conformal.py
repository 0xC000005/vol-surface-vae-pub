"""
Experiment 9a: Conformal Prediction for Calibrated Confidence Intervals

BACKGROUND:
exp8 showed that P1 loss can increase decoder gain (1.2e-7 → 2.2e-4) but
breaks the unconditional marginal distribution.

APPROACH:
Conformal prediction provides guaranteed coverage without retraining.
It's a post-hoc calibration method that works with any model.

ALGORITHM:
1. Split data into calibration set and test set
2. On calibration set, compute nonconformity scores: |prediction - target|
3. Find q = 90th percentile of scores (for 90% coverage)
4. On test set, CI = [prediction - q, prediction + q]
5. Verify coverage

EXPECTED RESULTS:
- Guaranteed ~90% coverage on test set (by theory)
- Intervals may be wide if model predictions are poor
- Per-grid-point calibration for heterogeneous uncertainty
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

    # Handle compiled model state dict
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


def generate_samples(model, context, num_samples=100):
    """Generate samples from prior for a given context."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    C = model.config["context_len"]
    latent_dim = model.config["latent_dim"]

    context = context.unsqueeze(0).to(device).to(dtype)
    ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
    ctx_input = {"surface": context, "ex_feats": ctx_feats}

    samples = []
    with torch.no_grad():
        # Get context encoding
        ctx_out = model.ctx_encoder(ctx_input)
        context_summary = ctx_out[:, -1, :]

        # Get prior parameters
        mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)
        sigma_p = torch.sqrt(torch.diag(Sigma_p)).unsqueeze(0).unsqueeze(0)

        for _ in range(num_samples):
            # Sample z from prior
            epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
            z = mu_p + sigma_p * epsilon

            # Decode
            if model.config.get("compress_context", True):
                ctx_embedding_dim = latent_dim
            else:
                ctx_embedding_dim = model.config["mem_hidden"]

            ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
            decoder_input = torch.cat([z, ctx_zeros], dim=-1)
            decoded = model.decoder(decoder_input)

            if isinstance(decoded, tuple):
                decoded = decoded[0]

            samples.append(decoded.squeeze().cpu().numpy())

    return np.array(samples)  # (num_samples, 5, 5)


def conformal_calibration(model, calibration_contexts, calibration_targets,
                         num_samples=100, alpha=0.10):
    """
    Compute conformal prediction thresholds.

    Args:
        model: Trained model
        calibration_contexts: (N, C, 5, 5) context sequences
        calibration_targets: (N, 5, 5) ground truth targets
        num_samples: Number of samples to generate per context
        alpha: 1 - coverage level (e.g., 0.10 for 90% coverage)

    Returns:
        q: (5, 5) per-grid-point thresholds
    """
    print(f"\nComputing conformal thresholds on {len(calibration_contexts)} calibration points...")

    scores = []  # Nonconformity scores

    for i in tqdm(range(len(calibration_contexts))):
        context = calibration_contexts[i]
        target = calibration_targets[i]

        # Generate samples and compute mean prediction
        samples = generate_samples(model, context, num_samples)
        prediction = samples.mean(axis=0)  # (5, 5)

        # Nonconformity score: absolute error
        score = np.abs(prediction - target.numpy())  # (5, 5)
        scores.append(score)

    scores = np.array(scores)  # (N, 5, 5)

    # Compute quantile for each grid point
    # Use (1 - alpha) * (n + 1) / n adjustment for finite samples
    n = len(scores)
    adjusted_quantile = np.ceil((1 - alpha) * (n + 1)) / n
    adjusted_quantile = min(adjusted_quantile, 1.0)

    q = np.quantile(scores, adjusted_quantile, axis=0)  # (5, 5)

    print(f"\nConformal thresholds (90% CI half-width):")
    print(f"  Mean: {q.mean():.6f}")
    print(f"  Min:  {q.min():.6f}")
    print(f"  Max:  {q.max():.6f}")

    return q


def predict_interval(model, context, q, num_samples=100):
    """
    Generate prediction interval using conformal thresholds.

    Args:
        model: Trained model
        context: (C, 5, 5) context sequence
        q: (5, 5) conformal thresholds
        num_samples: Number of samples for mean prediction

    Returns:
        lower: (5, 5) lower bound of CI
        upper: (5, 5) upper bound of CI
        mean: (5, 5) point prediction
    """
    samples = generate_samples(model, context, num_samples)
    mean = samples.mean(axis=0)

    lower = mean - q
    upper = mean + q

    return lower, upper, mean


def evaluate_coverage(model, test_contexts, test_targets, q, num_samples=100):
    """
    Evaluate CI coverage on test set.

    Args:
        model: Trained model
        test_contexts: (N, C, 5, 5) test context sequences
        test_targets: (N, 5, 5) ground truth targets
        q: (5, 5) conformal thresholds
        num_samples: Number of samples per context

    Returns:
        coverage: Overall coverage percentage
        per_grid_coverage: (5, 5) coverage per grid point
    """
    print(f"\nEvaluating coverage on {len(test_contexts)} test points...")

    covered = []

    for i in tqdm(range(len(test_contexts))):
        context = test_contexts[i]
        target = test_targets[i].numpy()

        lower, upper, mean = predict_interval(model, context, q, num_samples)

        # Check if target is within interval
        in_interval = (target >= lower) & (target <= upper)  # (5, 5)
        covered.append(in_interval)

    covered = np.array(covered)  # (N, 5, 5)

    # Per-grid-point coverage
    per_grid_coverage = covered.mean(axis=0) * 100  # (5, 5)

    # Overall coverage
    overall_coverage = covered.mean() * 100

    return overall_coverage, per_grid_coverage


def main():
    print("="*70)
    print("EXPERIMENT 9a: Conformal Prediction for Calibrated CIs")
    print("="*70)

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nTotal data: {surface.shape}")

    # Load model
    model_path = Path("results/prior_encoder_ablation/extended_training_v5/baseline_ep200.pt")

    if not model_path.exists():
        # Try alternative path
        model_path = Path("results/prior_encoder_ablation/exp8_p1_loss/model_ep200.pt")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please specify a valid model path")
        return

    print(f"\nLoading model from: {model_path}")
    model, config = load_model(model_path, CVAEFullCovPrior)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    C = config["context_len"]
    print(f"Context length: {C}")

    # Split data: 60% train (not used here), 20% calibration, 20% test
    n_total = len(surface) - C - 1
    n_cal = n_total // 5  # 20%
    n_test = n_total // 5  # 20%

    # Use last 40% for calibration and test
    cal_start = n_total - n_cal - n_test
    test_start = n_total - n_test

    print(f"\nData split:")
    print(f"  Calibration: {n_cal} points (indices {cal_start} to {test_start-1})")
    print(f"  Test: {n_test} points (indices {test_start} to {n_total-1})")

    # Prepare calibration data
    cal_contexts = []
    cal_targets = []
    for i in range(cal_start, test_start):
        cal_contexts.append(surface[i:i+C])
        cal_targets.append(surface[i+C])

    cal_contexts = torch.stack(cal_contexts)
    cal_targets = torch.stack(cal_targets)

    # Prepare test data
    test_contexts = []
    test_targets = []
    for i in range(test_start, n_total):
        test_contexts.append(surface[i:i+C])
        test_targets.append(surface[i+C])

    test_contexts = torch.stack(test_contexts)
    test_targets = torch.stack(test_targets)

    print(f"\nCalibration data: {cal_contexts.shape}")
    print(f"Test data: {test_contexts.shape}")

    # Subsample for faster computation
    n_cal_sample = min(200, len(cal_contexts))
    n_test_sample = min(200, len(test_contexts))

    cal_idx = np.random.choice(len(cal_contexts), n_cal_sample, replace=False)
    test_idx = np.random.choice(len(test_contexts), n_test_sample, replace=False)

    cal_contexts_sample = cal_contexts[cal_idx]
    cal_targets_sample = cal_targets[cal_idx]
    test_contexts_sample = test_contexts[test_idx]
    test_targets_sample = test_targets[test_idx]

    print(f"\nUsing {n_cal_sample} calibration and {n_test_sample} test points")

    # Compute conformal thresholds
    q = conformal_calibration(
        model, cal_contexts_sample, cal_targets_sample,
        num_samples=50, alpha=0.10
    )

    # Evaluate on test set
    overall_cov, per_grid_cov = evaluate_coverage(
        model, test_contexts_sample, test_targets_sample, q,
        num_samples=50
    )

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n90% Conformal Prediction CI:")
    print(f"  Target coverage: 90%")
    print(f"  Achieved coverage: {overall_cov:.1f}%")

    if overall_cov >= 85:
        print(f"  Status: [OK] Coverage within acceptable range")
    else:
        print(f"  Status: [WARNING] Coverage below 85%")

    print(f"\nConformal thresholds (CI half-width):")
    print(f"  Mean:  {q.mean():.6f}")
    print(f"  Range: [{q.min():.6f}, {q.max():.6f}]")

    print(f"\nPer-grid-point coverage (%):")
    np.set_printoptions(precision=1)
    print(per_grid_cov)

    # Compare with GT statistics
    gt_mean = surface[C:].mean().item()
    gt_std = surface[C:].std().item()

    print(f"\nContext:")
    print(f"  GT mean: {gt_mean:.4f}")
    print(f"  GT std:  {gt_std:.4f}")
    print(f"  CI half-width / GT std: {q.mean() / gt_std:.2f}x")

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp9_conformal")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "conformal_results.npz",
        thresholds=q,
        overall_coverage=overall_cov,
        per_grid_coverage=per_grid_cov,
        n_calibration=n_cal_sample,
        n_test=n_test_sample
    )

    print(f"\nResults saved to: {output_dir}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Conformal Prediction Results:
  - Achieved {overall_cov:.1f}% coverage (target: 90%)
  - CI half-width: {q.mean():.6f} (~{q.mean()/gt_std:.1f}x GT std)
  - Per-grid coverage range: [{per_grid_cov.min():.1f}%, {per_grid_cov.max():.1f}%]

Key Insight:
  Conformal prediction provides guaranteed coverage by construction,
  but the intervals may be wide if the model predictions are poor.

  If intervals are too wide, Track B (retraining with marginal constraints)
  should produce better predictions with tighter calibrated CIs.
""")


if __name__ == "__main__":
    main()
