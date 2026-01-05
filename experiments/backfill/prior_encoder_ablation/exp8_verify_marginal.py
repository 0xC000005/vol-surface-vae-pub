"""
Exp8 Verification: Does P1 Loss Preserve the Unconditional Marginal?

The Law of Total Variance states:
    Var(X) = E[Var(X|C)] + Var(E[X|C])

We increased E[Var(X|C)] (conditional variance) via P1 loss.
This test verifies whether the unconditional marginal P(X) is preserved.

TESTS:
1. Compare generated total variance vs ground truth variance
2. Compare marginal distribution statistics (mean, std, skewness, kurtosis)
3. Verify Law of Total Variance decomposition
4. Compare per-grid-point statistics
"""

import torch
import numpy as np
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior


def load_exp8_model(model_path):
    """Load the trained exp8 model."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']

    model = CVAEFullCovPrior(model_config)

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


def generate_samples_for_contexts(model, val_surface, num_contexts=200, num_samples=100):
    """
    Generate samples for many contexts and collect statistics.

    Returns:
        all_samples: (num_contexts, num_samples, 5, 5) - all generated samples
        context_means: (num_contexts, 5, 5) - mean prediction per context
        context_vars: (num_contexts, 5, 5) - variance per context
        gt_targets: (num_contexts, 5, 5) - ground truth targets
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    C = model.config["context_len"]
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    all_samples = []
    context_means = []
    context_vars = []
    gt_targets = []

    print(f"Generating {num_samples} samples for {num_contexts} contexts...")

    with torch.no_grad():
        for i in range(num_contexts):
            if i + C + 1 > len(val_surface):
                break

            if (i + 1) % 50 == 0:
                print(f"  Context {i+1}/{num_contexts}")

            # Get context and ground truth
            context = val_surface[i:i+C].unsqueeze(0).to(device).to(dtype)
            gt_target = val_surface[i+C].numpy()  # Ground truth next day
            gt_targets.append(gt_target)

            ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
            ctx_input = {"surface": context, "ex_feats": ctx_feats}

            # Get prior parameters
            ctx_out = model.ctx_encoder(ctx_input)
            context_summary = ctx_out[:, -1, :]
            mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

            # Sample from prior
            eps = 1e-4
            eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
            L = torch.linalg.cholesky(Sigma_p + eps * eye)

            samples = []
            for _ in range(num_samples):
                epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                z = mu_p + z_centered

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

            samples = np.array(samples)  # (num_samples, 5, 5)
            all_samples.append(samples)
            context_means.append(samples.mean(axis=0))
            context_vars.append(samples.var(axis=0))

    all_samples = np.array(all_samples)  # (num_contexts, num_samples, 5, 5)
    context_means = np.array(context_means)  # (num_contexts, 5, 5)
    context_vars = np.array(context_vars)  # (num_contexts, 5, 5)
    gt_targets = np.array(gt_targets)  # (num_contexts, 5, 5)

    return all_samples, context_means, context_vars, gt_targets


def verify_law_of_total_variance(all_samples, context_means, context_vars, gt_targets):
    """
    Verify the Law of Total Variance:
    Var(X) = E[Var(X|C)] + Var(E[X|C])
    """
    print("\n" + "="*70)
    print("LAW OF TOTAL VARIANCE VERIFICATION")
    print("="*70)

    # Ground truth variance
    gt_var = gt_targets.var(axis=0)  # (5, 5)
    gt_var_mean = gt_var.mean()

    # E[Var(X|C)] - expected conditional variance (within-context)
    within_var = context_vars.mean(axis=0)  # (5, 5)
    within_var_mean = within_var.mean()

    # Var(E[X|C]) - variance of conditional means (between-context)
    between_var = context_means.var(axis=0)  # (5, 5)
    between_var_mean = between_var.mean()

    # Generated total variance (should equal within + between)
    all_flat = all_samples.reshape(-1, 5, 5)  # (num_contexts * num_samples, 5, 5)
    gen_total_var = all_flat.var(axis=0)  # (5, 5)
    gen_total_var_mean = gen_total_var.mean()

    # Law of Total Variance check
    lotv_sum = within_var_mean + between_var_mean

    print("\n| Component | Value | Formula |")
    print("|-----------|-------|---------|")
    print(f"| GT Var(X) | {gt_var_mean:.6f} | Ground truth variance |")
    print(f"| Gen Var(X) | {gen_total_var_mean:.6f} | Variance of all generated samples |")
    print(f"| E[Var(X|C)] | {within_var_mean:.6f} | Within-context variance |")
    print(f"| Var(E[X|C]) | {between_var_mean:.6f} | Between-context variance |")
    print(f"| Sum | {lotv_sum:.6f} | E[Var(X|C)] + Var(E[X|C]) |")

    print("\n" + "-"*70)
    print("VERIFICATION:")
    print(f"  Law of Total Variance: {within_var_mean:.6f} + {between_var_mean:.6f} = {lotv_sum:.6f}")
    print(f"  Generated total:       {gen_total_var_mean:.6f}")
    print(f"  Difference:            {abs(lotv_sum - gen_total_var_mean):.6f} ({abs(lotv_sum - gen_total_var_mean)/gen_total_var_mean*100:.1f}%)")

    print(f"\n  Ground Truth Var(X):   {gt_var_mean:.6f}")
    print(f"  Generated Var(X):      {gen_total_var_mean:.6f}")
    print(f"  Ratio (Gen/GT):        {gen_total_var_mean/gt_var_mean:.2%}")

    # P1 metric
    p1 = within_var_mean / gt_var_mean
    print(f"\n  P1 = E[Var(X|C)] / Var(X) = {p1:.4%}")

    return {
        'gt_var': gt_var_mean,
        'gen_var': gen_total_var_mean,
        'within_var': within_var_mean,
        'between_var': between_var_mean,
        'p1': p1,
        'var_ratio': gen_total_var_mean / gt_var_mean
    }


def compare_marginal_distributions(all_samples, gt_targets):
    """
    Compare marginal distribution statistics between generated and GT.
    """
    print("\n" + "="*70)
    print("MARGINAL DISTRIBUTION COMPARISON")
    print("="*70)

    # Flatten for overall statistics
    gen_flat = all_samples.reshape(-1)
    gt_flat = gt_targets.reshape(-1)

    print("\n| Statistic | Ground Truth | Generated | Ratio |")
    print("|-----------|--------------|-----------|-------|")

    # Mean
    gt_mean = gt_flat.mean()
    gen_mean = gen_flat.mean()
    print(f"| Mean | {gt_mean:.6f} | {gen_mean:.6f} | {gen_mean/gt_mean:.2%} |")

    # Std
    gt_std = gt_flat.std()
    gen_std = gen_flat.std()
    print(f"| Std | {gt_std:.6f} | {gen_std:.6f} | {gen_std/gt_std:.2%} |")

    # Skewness
    gt_skew = stats.skew(gt_flat)
    gen_skew = stats.skew(gen_flat)
    print(f"| Skewness | {gt_skew:.4f} | {gen_skew:.4f} | - |")

    # Kurtosis
    gt_kurt = stats.kurtosis(gt_flat)
    gen_kurt = stats.kurtosis(gen_flat)
    print(f"| Kurtosis | {gt_kurt:.4f} | {gen_kurt:.4f} | - |")

    # Percentiles
    print("\n| Percentile | Ground Truth | Generated | Diff |")
    print("|------------|--------------|-----------|------|")
    for p in [5, 25, 50, 75, 95]:
        gt_p = np.percentile(gt_flat, p)
        gen_p = np.percentile(gen_flat, p)
        print(f"| p{p} | {gt_p:.6f} | {gen_p:.6f} | {gen_p - gt_p:+.6f} |")

    return {
        'gt_mean': gt_mean,
        'gen_mean': gen_mean,
        'gt_std': gt_std,
        'gen_std': gen_std,
        'gt_skew': gt_skew,
        'gen_skew': gen_skew,
        'gt_kurt': gt_kurt,
        'gen_kurt': gen_kurt
    }


def compare_per_grid_point(all_samples, gt_targets):
    """
    Compare variance per grid point.
    """
    print("\n" + "="*70)
    print("PER-GRID-POINT VARIANCE COMPARISON")
    print("="*70)

    # GT variance per grid point
    gt_var_grid = gt_targets.var(axis=0)  # (5, 5)

    # Generated variance per grid point
    gen_var_grid = all_samples.reshape(-1, 5, 5).var(axis=0)  # (5, 5)

    # Ratio
    ratio_grid = gen_var_grid / gt_var_grid

    print("\nGround Truth Variance (5x5 grid):")
    print(np.array2string(gt_var_grid, precision=6, suppress_small=True))

    print("\nGenerated Variance (5x5 grid):")
    print(np.array2string(gen_var_grid, precision=6, suppress_small=True))

    print("\nRatio (Generated / GT):")
    print(np.array2string(ratio_grid, precision=2, suppress_small=True))

    print(f"\nMean ratio across grid: {ratio_grid.mean():.2%}")
    print(f"Min ratio: {ratio_grid.min():.2%}, Max ratio: {ratio_grid.max():.2%}")

    return ratio_grid


def check_ci_coverage(all_samples, gt_targets):
    """
    Check if 90% CI contains ground truth 90% of the time.
    """
    print("\n" + "="*70)
    print("CONFIDENCE INTERVAL COVERAGE")
    print("="*70)

    num_contexts = all_samples.shape[0]

    # Compute 5th and 95th percentiles for each context
    p5 = np.percentile(all_samples, 5, axis=1)  # (num_contexts, 5, 5)
    p95 = np.percentile(all_samples, 95, axis=1)  # (num_contexts, 5, 5)

    # Check coverage
    in_ci = (gt_targets >= p5) & (gt_targets <= p95)  # (num_contexts, 5, 5)

    coverage_per_context = in_ci.mean(axis=(1, 2))  # (num_contexts,)
    overall_coverage = in_ci.mean()

    print(f"\n90% CI Coverage:")
    print(f"  Target:  90%")
    print(f"  Actual:  {overall_coverage:.1%}")
    print(f"  Gap:     {overall_coverage - 0.9:+.1%}")

    # Per-grid-point coverage
    coverage_per_grid = in_ci.mean(axis=0)  # (5, 5)
    print(f"\nPer-grid-point coverage:")
    print(np.array2string(coverage_per_grid * 100, precision=1, suppress_small=True))

    # Violations
    if overall_coverage < 0.9:
        print(f"\n  [WARNING] Under-coverage by {0.9 - overall_coverage:.1%}")
        print(f"            CIs are too narrow")
    else:
        print(f"\n  [OK] Coverage meets or exceeds target")

    return overall_coverage, coverage_per_grid


def main():
    print("="*70)
    print("EXP8 VERIFICATION: Marginal Distribution Preservation")
    print("="*70)

    # Load model
    model_path = Path("results/prior_encoder_ablation/exp8_p1_loss/model_ep200.pt")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"\nLoading model from: {model_path}")
    model, config = load_exp8_model(model_path)

    # Load validation data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    # Use validation set
    train_end = 4000
    val_surface = surface[train_end:train_end+500]
    print(f"Validation data: {val_surface.shape}")

    # Generate samples
    all_samples, context_means, context_vars, gt_targets = generate_samples_for_contexts(
        model, val_surface,
        num_contexts=200,
        num_samples=100
    )

    print(f"\nGenerated samples shape: {all_samples.shape}")
    print(f"  {all_samples.shape[0]} contexts x {all_samples.shape[1]} samples x 5x5 grid")

    # Run verification tests
    lotv_results = verify_law_of_total_variance(all_samples, context_means, context_vars, gt_targets)
    marginal_results = compare_marginal_distributions(all_samples, gt_targets)
    ratio_grid = compare_per_grid_point(all_samples, gt_targets)
    coverage, coverage_grid = check_ci_coverage(all_samples, gt_targets)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Is the Unconditional Marginal Preserved?")
    print("="*70)

    var_ratio = lotv_results['var_ratio']
    std_ratio = marginal_results['gen_std'] / marginal_results['gt_std']

    print(f"\n| Check | Result | Status |")
    print(f"|-------|--------|--------|")

    # Variance check
    var_ok = 0.8 <= var_ratio <= 1.2
    var_status = "[OK]" if var_ok else "[FAIL]"
    print(f"| Variance ratio | {var_ratio:.2%} | {var_status} (target: 80-120%) |")

    # Std check
    std_ok = 0.8 <= std_ratio <= 1.2
    std_status = "[OK]" if std_ok else "[FAIL]"
    print(f"| Std ratio | {std_ratio:.2%} | {std_status} (target: 80-120%) |")

    # Mean check
    mean_diff = abs(marginal_results['gen_mean'] - marginal_results['gt_mean'])
    mean_ok = mean_diff < 0.01
    mean_status = "[OK]" if mean_ok else "[FAIL]"
    print(f"| Mean diff | {mean_diff:.6f} | {mean_status} (target: <0.01) |")

    # P1 check
    p1_ok = lotv_results['p1'] > 0.02
    p1_status = "[OK]" if p1_ok else "[FAIL]"
    print(f"| P1 (cond var) | {lotv_results['p1']:.2%} | {p1_status} (target: >2%) |")

    # CI coverage
    cov_ok = coverage > 0.85
    cov_status = "[OK]" if cov_ok else "[FAIL]"
    print(f"| 90% CI coverage | {coverage:.1%} | {cov_status} (target: >85%) |")

    # Overall assessment
    print("\n" + "-"*70)
    all_ok = var_ok and std_ok and mean_ok and p1_ok

    if all_ok:
        print("VERDICT: Unconditional marginal IS PRESERVED")
        print("  - Total variance matches GT within 20%")
        print("  - Conditional variance (P1) > 2% target")
        print("  - Mean is unbiased")
    elif var_ratio > 1.3:
        print("VERDICT: Generated samples have TOO MUCH variance")
        print(f"  - Gen/GT variance ratio = {var_ratio:.2%}")
        print("  - P1 loss may be too strong")
    elif var_ratio < 0.7:
        print("VERDICT: Generated samples have TOO LITTLE variance")
        print(f"  - Gen/GT variance ratio = {var_ratio:.2%}")
        print("  - P1 loss may need to be stronger")
    else:
        print("VERDICT: Partial success - some metrics off")

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp8_p1_loss")
    np.savez(
        output_dir / "marginal_verification.npz",
        all_samples=all_samples,
        context_means=context_means,
        context_vars=context_vars,
        gt_targets=gt_targets,
        ratio_grid=ratio_grid,
        coverage_grid=coverage_grid,
        **lotv_results,
        **marginal_results,
        coverage=coverage
    )

    print(f"\nResults saved to: {output_dir / 'marginal_verification.npz'}")


if __name__ == "__main__":
    main()
