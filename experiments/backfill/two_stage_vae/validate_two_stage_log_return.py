"""
Validate Two-Stage CVAE Log-Return Model.

This script validates that the log-return model solves the horizon variance issue:
1. Reconstructs log-returns accurately
2. CI coverage improves across horizons (target: ~90%)
3. CI width grows with horizon (not flat)
4. Transformed-back IV values are always positive

Usage:
    python experiments/backfill/prior_encoder_ablation/validate_two_stage_log_return.py
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage


def to_log_returns(surfaces):
    """Transform IV surfaces to log-returns."""
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]
    return log_returns, log_surfaces


def from_log_returns(log_returns_seq, initial_log_surface):
    """Transform log-returns back to IV surfaces (always positive!)."""
    log_cumsum = np.cumsum(log_returns_seq, axis=0)
    final_log = initial_log_surface + log_cumsum
    return np.exp(final_log)


def load_model(checkpoint_path):
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = checkpoint["model_config"]
    model = CVAETwoStage(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def test_reconstruction(model, log_returns, device, config, n_samples=100):
    """Test reconstruction quality in log-return space."""
    print("=" * 70)
    print("TEST 1: Reconstruction Quality (Log-Return Space)")
    print("=" * 70)
    print()

    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon

    # Create test sequences
    test_sequences = []
    for i in range(0, len(log_returns) - seq_len, 50):
        test_sequences.append(log_returns[i:i+seq_len])
    test_sequences = np.array(test_sequences[:n_samples])

    # Reconstruct
    with torch.no_grad():
        batch = torch.tensor(test_sequences, dtype=torch.float32).to(device)
        output = model({"surface": batch}, return_full_sequence=True)
        # Output is tuple: (surface_recon, z_mean, z_logvar, z)
        recon = output[0].cpu().numpy()
        z_mean = output[1].cpu().numpy()
        z_logvar = output[2].cpu().numpy()

    # Compute MSE per position
    mse_per_pos = ((recon - test_sequences) ** 2).mean(axis=(0, 2, 3))

    print("MSE per position:")
    print(f"  Context (0-{context_len-1}): {mse_per_pos[:context_len].mean():.6f}")
    print(f"  Horizon ({context_len}-{seq_len-1}): {mse_per_pos[context_len:].mean():.6f}")
    print()
    print("z_logvar statistics:")
    print(f"  Mean: {z_logvar.mean():.4f}")
    print(f"  Std: {z_logvar.std():.4f}")
    print(f"  Range: [{z_logvar.min():.4f}, {z_logvar.max():.4f}]")
    print()

    return {
        "mse_context": mse_per_pos[:context_len].mean(),
        "mse_horizon": mse_per_pos[context_len:].mean(),
        "z_logvar_mean": z_logvar.mean(),
        "z_logvar_std": z_logvar.std(),
    }


def test_ci_coverage(model, log_returns, log_surfaces, device, config, horizons=[1, 7, 14, 30], n_samples=100, n_test=50):
    """
    Test CI coverage by sampling from model.

    For each test point:
    1. Take context (C log-returns)
    2. Sample N reconstructions for horizon positions
    3. Compute 90% CI from samples
    4. Check if ground truth falls within CI
    """
    print("=" * 70)
    print("TEST 2: CI Coverage (Log-Return Space)")
    print("=" * 70)
    print()

    context_len = config["context_len"]
    seq_len = context_len + max(horizons)

    results = {}

    for H in horizons:
        coverage_count = 0
        ci_widths = []
        n_tests = 0

        for start_idx in range(0, len(log_returns) - seq_len, max(1, (len(log_returns) - seq_len) // n_test)):
            if n_tests >= n_test:
                break

            # Ground truth sequence
            gt_seq = log_returns[start_idx:start_idx + context_len + H]

            # Sample N reconstructions
            samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    batch = torch.tensor(gt_seq[None], dtype=torch.float32).to(device)
                    output = model({"surface": batch}, return_full_sequence=True)
                    recon = output[0].cpu().numpy()[0]  # (T, 5, 5)
                    samples.append(recon[context_len + H - 1])  # H-th horizon position

            samples = np.array(samples)  # (N, 5, 5)

            # Ground truth at horizon H
            gt = gt_seq[context_len + H - 1]  # (5, 5)

            # Compute 90% CI
            p05 = np.percentile(samples, 5, axis=0)
            p95 = np.percentile(samples, 95, axis=0)

            # Check coverage
            in_ci = (gt >= p05) & (gt <= p95)
            coverage_count += in_ci.mean()

            # CI width
            ci_widths.append((p95 - p05).mean())
            n_tests += 1

        coverage = coverage_count / n_tests
        avg_ci_width = np.mean(ci_widths)

        results[H] = {
            "coverage": coverage,
            "ci_width": avg_ci_width,
        }

        status = "PASS" if abs(coverage - 0.90) < 0.15 else "FAIL"
        print(f"H={H:2d}: Coverage = {coverage*100:.1f}% (target: 90%), CI width = {avg_ci_width:.4f} [{status}]")

    print()
    return results


def test_transformed_back(model, log_returns, log_surfaces, device, config, horizons=[1, 7, 14, 30], n_samples=100, n_test=50):
    """
    Test CI coverage in transformed-back IV space.

    This verifies:
    1. CI width grows with horizon
    2. Coverage is ~90%
    3. All IV values are positive
    """
    print("=" * 70)
    print("TEST 3: CI Coverage (Transformed-Back IV Space)")
    print("=" * 70)
    print()

    context_len = config["context_len"]
    seq_len = context_len + max(horizons)
    surfaces = np.exp(log_surfaces)

    results = {}
    base_width = None
    n_negative = 0

    for H in horizons:
        coverage_count = 0
        ci_widths = []
        n_tests = 0

        for start_idx in range(0, len(log_returns) - seq_len, max(1, (len(log_returns) - seq_len) // n_test)):
            if n_tests >= n_test:
                break

            # Ground truth log-return sequence
            gt_log_seq = log_returns[start_idx:start_idx + context_len + H]
            # Initial log-surface (for reconstruction)
            initial_log = log_surfaces[start_idx]
            # Ground truth IV at horizon
            gt_iv = surfaces[start_idx + context_len + H]

            # Sample N reconstructions
            samples_iv = []
            with torch.no_grad():
                for _ in range(n_samples):
                    batch = torch.tensor(gt_log_seq[None], dtype=torch.float32).to(device)
                    output = model({"surface": batch}, return_full_sequence=True)
                    recon_log_returns = output[0].cpu().numpy()[0]  # (T, 5, 5)

                    # Transform back: sum log-returns and exp()
                    cumsum = recon_log_returns[:context_len + H].sum(axis=0)
                    final_iv = np.exp(initial_log + cumsum)
                    samples_iv.append(final_iv)

                    # Check for negatives (should be 0!)
                    if (final_iv <= 0).any():
                        n_negative += 1

            samples_iv = np.array(samples_iv)  # (N, 5, 5)

            # Compute 90% CI
            p05 = np.percentile(samples_iv, 5, axis=0)
            p95 = np.percentile(samples_iv, 95, axis=0)

            # Check coverage
            in_ci = (gt_iv >= p05) & (gt_iv <= p95)
            coverage_count += in_ci.mean()

            # CI width
            ci_widths.append((p95 - p05).mean())
            n_tests += 1

        coverage = coverage_count / n_tests
        avg_ci_width = np.mean(ci_widths)

        if base_width is None:
            base_width = avg_ci_width
        width_ratio = avg_ci_width / base_width

        results[H] = {
            "coverage": coverage,
            "ci_width": avg_ci_width,
            "width_ratio": width_ratio,
        }

        status = "PASS" if abs(coverage - 0.90) < 0.15 else "FAIL"
        print(f"H={H:2d}: Coverage = {coverage*100:.1f}%, CI width = {avg_ci_width:.4f}, "
              f"Width ratio = {width_ratio:.2f}x [{status}]")

    print()
    print(f"Negative IV values: {n_negative} (should be 0)")
    print()

    return results


def compare_with_baseline():
    """Print baseline comparison from earlier test."""
    print("=" * 70)
    print("BASELINE COMPARISON (from test_log_return_variance.py)")
    print("=" * 70)
    print()
    print("Level-based model (constant variance):")
    print("  H= 1: Coverage = 84.1%")
    print("  H= 7: Coverage = 80.7%")
    print("  H=14: Coverage = 74.5%")
    print("  H=30: Coverage = 73.4%")
    print("  CI Width: FLAT across horizons")
    print()
    print("Expected log-return model:")
    print("  H= 1: Coverage ~ 87%")
    print("  H= 7: Coverage ~ 97%")
    print("  H=14: Coverage ~ 99%")
    print("  H=30: Coverage ~ 99%")
    print("  CI Width: GROWS with horizon")
    print()


def print_summary(recon_results, log_ci_results, iv_ci_results):
    """Print summary of all tests."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Reconstruction
    print("Reconstruction Quality:")
    print(f"  MSE (context): {recon_results['mse_context']:.6f}")
    print(f"  MSE (horizon): {recon_results['mse_horizon']:.6f}")
    print(f"  z_logvar: {recon_results['z_logvar_mean']:.4f} ± {recon_results['z_logvar_std']:.4f}")
    print()

    # CI Coverage
    print("CI Coverage (IV Space):")
    for H in [1, 7, 14, 30]:
        if H in iv_ci_results:
            r = iv_ci_results[H]
            print(f"  H={H:2d}: {r['coverage']*100:.1f}% coverage, {r['width_ratio']:.2f}x width")
    print()

    # Overall assessment
    avg_coverage = np.mean([iv_ci_results[H]["coverage"] for H in [1, 7, 14, 30] if H in iv_ci_results])
    width_growth = iv_ci_results.get(30, {}).get("width_ratio", 1.0)

    print("Overall Assessment:")
    if avg_coverage > 0.85 and width_growth > 2.0:
        print("  ✓ SUCCESS: Log-return model solves horizon variance issue!")
        print(f"    - Average CI coverage: {avg_coverage*100:.1f}% (target: 90%)")
        print(f"    - CI width growth H=1→H=30: {width_growth:.1f}x (should grow)")
    elif avg_coverage > 0.75:
        print("  ~ PARTIAL: Some improvement but not fully solved")
        print(f"    - Average CI coverage: {avg_coverage*100:.1f}%")
        print(f"    - CI width growth: {width_growth:.1f}x")
    else:
        print("  ✗ FAILED: Log-return model did not improve CI coverage")
        print(f"    - Average CI coverage: {avg_coverage*100:.1f}%")

    print()


def main():
    print("=" * 70)
    print("TWO-STAGE LOG-RETURN MODEL VALIDATION")
    print("=" * 70)
    print()

    # Load model
    checkpoint_path = Path("models/backfill/two_stage/two_stage_log_return_best.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run training first: python experiments/backfill/prior_encoder_ablation/train_two_stage_log_return.py")
        return

    print(f"Loading model from {checkpoint_path}...")
    model, config, checkpoint = load_model(checkpoint_path)
    device = config["device"]
    print(f"  Device: {device}")
    print(f"  use_log_returns: {checkpoint.get('use_log_returns', False)}")
    if "log_return_stats" in checkpoint:
        stats = checkpoint["log_return_stats"]
        print(f"  Log-return stats: mean={stats['mean']:.6f}, std={stats['std']:.4f}")
    print()

    # Load data
    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)
    print(f"  Surfaces: {surfaces.shape}")
    print(f"  Log-returns: {log_returns.shape}")
    print()

    # Compare with baseline
    compare_with_baseline()

    # Test 1: Reconstruction
    recon_results = test_reconstruction(model, log_returns, device, config)

    # Test 2: CI coverage in log-return space
    log_ci_results = test_ci_coverage(model, log_returns, log_surfaces, device, config)

    # Test 3: CI coverage in transformed-back IV space
    iv_ci_results = test_transformed_back(model, log_returns, log_surfaces, device, config)

    # Summary
    print_summary(recon_results, log_ci_results, iv_ci_results)


if __name__ == "__main__":
    main()
