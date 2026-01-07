"""
Validate conditional variance capability of trained autoencoder.

Tests (using GROUND TRUTH ctx_emb and z):
1. Conditional variance: Same ctx_emb + different z samples -> variance for CI
2. Unconditional marginal: Aggregate samples match data distribution
3. Regime-specific variance: High vol contexts -> higher conditional variance

This tests model CAPABILITY under training conditions, not inference.

Usage:
    python experiments/backfill/prior_encoder_ablation/validate_conditional_variance.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage
from config.two_stage_config import TwoStageConfig


def load_model(checkpoint_path):
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["model_config"]
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    model = CVAETwoStage(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

    return model, config


def test_conditional_variance(model, surfaces, n_samples=100):
    """
    Test 1: Conditional Variance

    For each position, use GROUND TRUTH ctx_emb but SAMPLE different z.
    Measure variance of decoded surfaces.

    Goal: Variance should be large enough for meaningful CI (>1% of data variance)
    """
    print("\n" + "=" * 70)
    print("TEST 1: Conditional Variance (Ground Truth ctx_emb)")
    print("=" * 70)

    config = model.config
    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon
    device = model.device

    # Take test sequences spread across data
    n_test = min(50, len(surfaces) - seq_len + 1)
    test_indices = np.linspace(0, len(surfaces) - seq_len - 1, n_test, dtype=int)

    all_variances = []
    all_z_logvar = []

    print(f"\nSampling {n_samples} z per sequence for {n_test} sequences...")

    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Computing variance"):
            sequence = surfaces[idx:idx+seq_len].unsqueeze(0).to(device)
            batch = {"surface": sequence}

            # Get GROUND TRUTH ctx_emb
            ctx_emb = model.ctx_encoder(batch)  # (1, T, ctx_dim)

            # Get GROUND TRUTH z distribution parameters
            z_mean, z_logvar, _ = model.encoder(batch)  # (1, T, latent_dim)
            all_z_logvar.append(z_logvar[:, context_len:].cpu())

            # Sample multiple z from the posterior
            samples = []
            for _ in range(n_samples):
                eps = torch.randn_like(z_mean)
                z_sample = z_mean + torch.exp(0.5 * z_logvar) * eps

                # Decode with ground truth ctx_emb + sampled z
                recon = model.decoder(ctx_emb, z_sample)
                samples.append(recon[:, context_len:].cpu())  # horizon only

            # Stack samples: (n_samples, 1, horizon, 5, 5)
            samples = torch.stack(samples, dim=0)

            # Compute variance across samples for this sequence
            variance = samples.var(dim=0)  # (1, horizon, 5, 5)
            all_variances.append(variance)

    # Aggregate variances
    all_variances = torch.cat(all_variances, dim=0)  # (n_test, horizon, 5, 5)
    all_z_logvar = torch.cat(all_z_logvar, dim=0)  # (n_test, horizon, latent_dim)

    mean_variance = all_variances.mean().item()
    data_variance = surfaces.var().item()
    variance_ratio = mean_variance / data_variance

    # z_logvar statistics
    mean_z_logvar = all_z_logvar.mean().item()
    mean_z_std = torch.exp(0.5 * all_z_logvar).mean().item()

    print(f"\nResults:")
    print(f"  Mean conditional variance: {mean_variance:.6f}")
    print(f"  Data variance: {data_variance:.6f}")
    print(f"  Ratio (conditional/data): {variance_ratio:.4f} ({variance_ratio*100:.2f}%)")

    print(f"\n  z_logvar statistics:")
    print(f"    Mean z_logvar: {mean_z_logvar:.4f}")
    print(f"    Mean z_std (exp(logvar/2)): {mean_z_std:.4f}")
    print(f"    z_logvar_floor: {config.get('z_logvar_floor', None)}")

    print(f"\n  Per-horizon variance:")
    for h in [0, 4, 9, 14, 19, 29]:
        if h < all_variances.shape[1]:
            h_var = all_variances[:, h].mean().item()
            print(f"    H={h+1:2d}: {h_var:.6f} ({h_var/data_variance*100:.2f}% of data var)")

    # CI coverage estimate (assuming Gaussian)
    mean_std = np.sqrt(mean_variance)
    ci_width_90 = 2 * 1.645 * mean_std
    ci_width_95 = 2 * 1.96 * mean_std

    print(f"\n  Estimated CI widths:")
    print(f"    90% CI half-width: {ci_width_90:.4f}")
    print(f"    95% CI half-width: {ci_width_95:.4f}")
    print(f"    As fraction of data std: {ci_width_90 / np.sqrt(data_variance):.4f}")

    # Pass/Fail
    passed = variance_ratio > 0.01
    if passed:
        print(f"\n  [PASS] Conditional variance is {variance_ratio*100:.1f}% of data variance (>1%)")
    else:
        print(f"\n  [FAIL] Conditional variance too small ({variance_ratio*100:.2f}% < 1%)")

    return {
        "mean_variance": mean_variance,
        "data_variance": data_variance,
        "variance_ratio": variance_ratio,
        "per_horizon_variance": all_variances.mean(dim=(0, 2, 3)).numpy(),
        "mean_z_logvar": mean_z_logvar,
        "mean_z_std": mean_z_std,
        "passed": passed,
    }


def test_unconditional_marginal(model, surfaces, n_samples=1000):
    """
    Test 2: Unconditional Marginal Match

    Generate many samples using GROUND TRUTH ctx_emb + SAMPLED z.
    Compare aggregate distribution to data distribution.

    Goal: Marginal statistics should match data (mean, std, percentiles within 10%)
    """
    print("\n" + "=" * 70)
    print("TEST 2: Unconditional Marginal (Aggregate Distribution)")
    print("=" * 70)

    config = model.config
    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon
    device = model.device

    # Sample sequences from data
    n_sequences = min(100, len(surfaces) - seq_len + 1)
    seq_indices = np.random.choice(len(surfaces) - seq_len + 1, n_sequences, replace=False)

    all_generated = []
    all_ground_truth = []

    samples_per_seq = max(1, n_samples // n_sequences)
    print(f"\nGenerating {samples_per_seq} samples each for {n_sequences} sequences...")

    with torch.no_grad():
        for idx in tqdm(seq_indices, desc="Generating samples"):
            sequence = surfaces[idx:idx+seq_len].unsqueeze(0).to(device)
            batch = {"surface": sequence}

            # Ground truth horizon surfaces
            gt_horizon = sequence[:, context_len:].cpu()
            all_ground_truth.append(gt_horizon)

            # Get ground truth ctx_emb
            ctx_emb = model.ctx_encoder(batch)

            # Get z distribution
            z_mean, z_logvar, _ = model.encoder(batch)

            # Generate samples
            for _ in range(samples_per_seq):
                eps = torch.randn_like(z_mean)
                z_sample = z_mean + torch.exp(0.5 * z_logvar) * eps
                recon = model.decoder(ctx_emb, z_sample)
                all_generated.append(recon[:, context_len:].cpu())

    # Flatten for comparison
    generated = torch.cat(all_generated, dim=0).flatten().numpy()
    ground_truth = torch.cat(all_ground_truth, dim=0).flatten().numpy()

    # Compare statistics
    print(f"\nStatistics Comparison:")
    print(f"  {'Statistic':<15} {'Generated':>12} {'Ground Truth':>12} {'Diff %':>10} {'Status':>8}")
    print(f"  {'-'*60}")

    stats = [
        ("Mean", generated.mean(), ground_truth.mean()),
        ("Std", generated.std(), ground_truth.std()),
        ("Min", generated.min(), ground_truth.min()),
        ("Max", generated.max(), ground_truth.max()),
        ("P5", np.percentile(generated, 5), np.percentile(ground_truth, 5)),
        ("P25", np.percentile(generated, 25), np.percentile(ground_truth, 25)),
        ("P50 (Median)", np.percentile(generated, 50), np.percentile(ground_truth, 50)),
        ("P75", np.percentile(generated, 75), np.percentile(ground_truth, 75)),
        ("P95", np.percentile(generated, 95), np.percentile(ground_truth, 95)),
    ]

    all_pass = True
    for name, gen, gt in stats:
        if gt != 0:
            diff_pct = abs(gen - gt) / abs(gt) * 100
        else:
            diff_pct = abs(gen - gt) * 100

        status = "PASS" if diff_pct < 10 else "FAIL"
        if diff_pct >= 10:
            all_pass = False
        print(f"  {name:<15} {gen:>12.4f} {gt:>12.4f} {diff_pct:>9.1f}% {status:>8}")

    if all_pass:
        print(f"\n  [PASS] All marginal statistics within 10% of ground truth")
    else:
        print(f"\n  [FAIL] Some marginal statistics differ by >10%")

    return {
        "generated_stats": {s[0]: float(s[1]) for s in stats},
        "ground_truth_stats": {s[0]: float(s[2]) for s in stats},
        "passed": all_pass,
    }


def test_variance_by_regime(model, surfaces, n_samples_per_seq=50):
    """
    Test 3: Regime-Specific Conditional Variance

    Check if conditional variance varies by regime (proxy: avg vol level).
    Higher vol regimes should have higher conditional variance.

    Goal: Positive correlation (>0.3) between vol level and conditional variance
    """
    print("\n" + "=" * 70)
    print("TEST 3: Regime-Specific Variance")
    print("=" * 70)

    config = model.config
    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon
    device = model.device

    # Sample sequences
    n_sequences = min(200, len(surfaces) - seq_len + 1)
    indices = np.random.choice(len(surfaces) - seq_len + 1, n_sequences, replace=False)

    vol_levels = []
    variances = []
    ctx_emb_norms = []

    print(f"\nAnalyzing {n_sequences} sequences...")

    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing regime variance"):
            sequence = surfaces[idx:idx+seq_len].unsqueeze(0).to(device)
            batch = {"surface": sequence}

            # Average vol level (proxy for regime)
            avg_vol = sequence.mean().item()
            vol_levels.append(avg_vol)

            # Get ground truth representations
            ctx_emb = model.ctx_encoder(batch)
            ctx_emb_norms.append(ctx_emb[:, context_len:].norm().item())

            z_mean, z_logvar, _ = model.encoder(batch)

            # Sample and measure variance
            samples = []
            for _ in range(n_samples_per_seq):
                eps = torch.randn_like(z_mean)
                z_sample = z_mean + torch.exp(0.5 * z_logvar) * eps
                recon = model.decoder(ctx_emb, z_sample)
                samples.append(recon[:, context_len:].cpu())

            samples = torch.stack(samples)
            variance = samples.var(dim=0).mean().item()
            variances.append(variance)

    # Convert to arrays
    vol_levels = np.array(vol_levels)
    variances = np.array(variances)
    ctx_emb_norms = np.array(ctx_emb_norms)

    # Quintile analysis
    quintiles = np.percentile(vol_levels, [0, 20, 40, 60, 80, 100])

    print(f"\nConditional Variance by Vol Regime:")
    print(f"  {'Regime':<15} {'Vol Range':>25} {'Mean Cond Var':>15} {'Mean ctx_emb norm':>18}")
    print(f"  {'-'*75}")

    for i in range(5):
        mask = (vol_levels >= quintiles[i]) & (vol_levels < quintiles[i+1] + 1e-6)
        if mask.sum() > 0:
            mean_var = variances[mask].mean()
            mean_ctx = ctx_emb_norms[mask].mean()
            print(f"  Q{i+1} ({'low' if i < 2 else 'mid' if i == 2 else 'high':>4})    "
                  f"[{quintiles[i]:.3f}, {quintiles[i+1]:.3f}]    {mean_var:.6f}         {mean_ctx:.4f}")

    # Correlations
    vol_var_corr = np.corrcoef(vol_levels, variances)[0, 1]
    ctx_var_corr = np.corrcoef(ctx_emb_norms, variances)[0, 1]

    print(f"\nCorrelations:")
    print(f"  corr(vol_level, cond_var): {vol_var_corr:.3f}")
    print(f"  corr(ctx_emb_norm, cond_var): {ctx_var_corr:.3f}")

    # Pass/Fail
    passed = vol_var_corr > 0.3
    if passed:
        print(f"\n  [PASS] Higher vol -> higher conditional variance (corr={vol_var_corr:.2f} > 0.3)")
    else:
        if vol_var_corr > 0:
            print(f"\n  [WEAK] Positive but weak relationship (corr={vol_var_corr:.2f} < 0.3)")
        else:
            print(f"\n  [FAIL] No positive relationship between vol level and conditional variance")

    return {
        "vol_variance_corr": float(vol_var_corr),
        "ctx_variance_corr": float(ctx_var_corr),
        "vol_levels": vol_levels,
        "variances": variances,
        "passed": passed,
    }


def main():
    print("=" * 70)
    print("Conditional Variance Validation")
    print("=" * 70)

    # Load model
    checkpoint_dir = Path(TwoStageConfig.checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{TwoStageConfig.checkpoint_prefix}_best.pt"

    if not checkpoint_path.exists():
        print(f"\nERROR: No checkpoint found at {checkpoint_path}")
        print("Run train_two_stage_autoencoder.py first!")
        return

    model, config = load_model(checkpoint_path)

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = torch.tensor(data["surface"], dtype=torch.float32)
    print(f"  Total surfaces: {len(surfaces)}")

    # Run tests
    results = {}

    results["conditional_variance"] = test_conditional_variance(model, surfaces, n_samples=100)
    results["unconditional_marginal"] = test_unconditional_marginal(model, surfaces, n_samples=1000)
    results["regime_variance"] = test_variance_by_regime(model, surfaces, n_samples_per_seq=50)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    test1_pass = results["conditional_variance"]["passed"]
    test2_pass = results["unconditional_marginal"]["passed"]
    test3_pass = results["regime_variance"]["passed"]

    var_ratio = results["conditional_variance"]["variance_ratio"]
    vol_corr = results["regime_variance"]["vol_variance_corr"]

    print(f"\n1. Conditional Variance: {var_ratio*100:.2f}% of data variance")
    print(f"   Status: {'PASS' if test1_pass else 'FAIL'} (threshold: >1%)")
    print(f"   Implication: {'Sufficient for CI' if test1_pass else 'Need VQ-style context'}")

    print(f"\n2. Unconditional Marginal:")
    print(f"   Status: {'PASS' if test2_pass else 'FAIL'} (threshold: stats within 10%)")
    print(f"   Implication: {'Distribution matches data' if test2_pass else 'Architecture issue'}")

    print(f"\n3. Regime-Specific Variance: corr={vol_corr:.3f}")
    print(f"   Status: {'PASS' if test3_pass else 'WEAK/FAIL'} (threshold: >0.3)")
    print(f"   Implication: {'ctx_emb captures regime' if test3_pass else 'ctx_emb may need VQ'}")

    # Overall decision
    print("\n" + "-" * 70)
    all_pass = test1_pass and test2_pass and test3_pass
    if all_pass:
        print("DECISION: Current architecture sufficient. Proceed to Stage 2 predictors.")
    elif not test1_pass:
        print("DECISION: Conditional variance insufficient. Consider VQ-style context.")
    elif not test2_pass:
        print("DECISION: Marginal mismatch. Check decoder capacity or KL weight.")
    else:
        print("DECISION: Regime-variance weak. Consider VQ-style context for discrete regimes.")
    print("-" * 70)

    # Save results
    results_dir = Path("results/two_stage_validation")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        "test1_conditional_variance_ratio": var_ratio,
        "test1_passed": test1_pass,
        "test2_passed": test2_pass,
        "test3_vol_variance_corr": vol_corr,
        "test3_passed": test3_pass,
        "all_passed": all_pass,
    }

    np.savez(results_dir / "validation_results.npz", **summary)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
