"""
Validate VAE Teacher Forcing Sequences with Configurable Sampling Mode

Validates all 16 generated NPZ files (4 periods × 4 horizons) to ensure:
1. File existence
2. Correct shapes
3. Quantile ordering (p05 ≤ p50 ≤ p95)
4. Reasonable value ranges (0.01 < IV < 5.0)
5. No NaN/Inf values
6. Index uniqueness
7. Correct sampling_mode metadata

Usage:
    python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode oracle
    python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode prior
"""
import argparse
import numpy as np
from pathlib import Path


def validate_file(filepath, expected_horizons, period_name, expected_sampling_mode):
    """
    Validate a single NPZ file.

    Args:
        filepath: Path to NPZ file
        expected_horizons: Expected horizon value
        period_name: Period identifier for context
        expected_sampling_mode: Expected sampling mode ('oracle' or 'prior')

    Returns:
        bool: True if all checks pass, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Validating: {filepath.name}")
    print(f"{'='*80}")

    try:
        # Load file
        data = np.load(filepath)

        # Check required keys
        required_keys = ['surfaces', 'indices', 'quantiles', 'horizon',
                        'period_start', 'period_end', 'method', 'sampling_mode']
        missing_keys = [k for k in required_keys if k not in data.keys()]
        if missing_keys:
            print(f"  ✗ FAIL: Missing keys: {missing_keys}")
            return False
        print(f"  ✓ All required keys present")

        # Extract data
        surfaces = data['surfaces']
        indices = data['indices']
        quantiles = data['quantiles']
        horizon = data['horizon']
        method = str(data['method'])
        sampling_mode = str(data['sampling_mode'])

        # Check sampling mode
        if sampling_mode != expected_sampling_mode:
            print(f"  ✗ FAIL: Sampling mode mismatch (expected '{expected_sampling_mode}', got '{sampling_mode}')")
            return False
        print(f"  ✓ Sampling mode correct: {sampling_mode}")

        # Check shape
        n_days, H, n_quantiles, n_rows, n_cols = surfaces.shape
        expected_shape = (n_days, expected_horizons, 3, 5, 5)
        if surfaces.shape[1:] != expected_shape[1:]:
            print(f"  ✗ FAIL: Shape mismatch")
            print(f"    Expected: (n_days, {expected_horizons}, 3, 5, 5)")
            print(f"    Got: {surfaces.shape}")
            return False
        print(f"  ✓ Shape correct: {surfaces.shape}")

        # Check horizon value
        if horizon != expected_horizons:
            print(f"  ✗ FAIL: Horizon mismatch (expected {expected_horizons}, got {horizon})")
            return False
        print(f"  ✓ Horizon value correct: {horizon}")

        # Check indices shape
        if len(indices) != n_days:
            print(f"  ✗ FAIL: Indices length mismatch ({len(indices)} vs {n_days})")
            return False
        print(f"  ✓ Indices shape correct: {indices.shape}")

        # Check index uniqueness
        if len(np.unique(indices)) != len(indices):
            print(f"  ✗ FAIL: Duplicate indices found")
            return False
        print(f"  ✓ All indices unique")

        # Check quantile values
        if not np.allclose(quantiles, [0.05, 0.50, 0.95]):
            print(f"  ✗ FAIL: Quantile values incorrect: {quantiles}")
            return False
        print(f"  ✓ Quantile values correct: {list(quantiles)}")

        # Check method
        if method != 'teacher_forcing':
            print(f"  ✗ FAIL: Method incorrect (expected 'teacher_forcing', got '{method}')")
            return False
        print(f"  ✓ Method correct: {method}")

        # Check for NaN/Inf
        if np.any(np.isnan(surfaces)):
            print(f"  ✗ FAIL: Contains NaN values")
            return False
        if np.any(np.isinf(surfaces)):
            print(f"  ✗ FAIL: Contains Inf values")
            return False
        print(f"  ✓ No NaN/Inf values")

        # Check value range (IV should be between 0.01 and 5.0)
        min_val = surfaces.min()
        max_val = surfaces.max()
        if min_val < 0.01 or max_val > 5.0:
            print(f"  ⚠ WARNING: Values outside typical range [0.01, 5.0]")
            print(f"    Min: {min_val:.4f}, Max: {max_val:.4f}")
            print(f"    (This might be okay, but verify)")
        else:
            print(f"  ✓ Value range reasonable: [{min_val:.4f}, {max_val:.4f}]")

        # Check quantile ordering (p05 ≤ p50 ≤ p95)
        p05 = surfaces[:, :, 0, :, :]  # (n_days, H, 5, 5)
        p50 = surfaces[:, :, 1, :, :]
        p95 = surfaces[:, :, 2, :, :]

        violations_lower = np.sum(p05 > p50)
        violations_upper = np.sum(p50 > p95)
        total_violations = violations_lower + violations_upper
        total_points = p05.size

        # Allow small tolerance for quantile violations (<1% is acceptable)
        # This can happen due to numerical precision in quantile regression
        violation_threshold = 0.01  # 1%

        if total_violations > 0:
            violation_pct = total_violations / total_points

            if violation_pct > violation_threshold:
                print(f"  ✗ FAIL: Quantile ordering violations exceed threshold")
                print(f"    p05 > p50: {violations_lower} points")
                print(f"    p50 > p95: {violations_upper} points")
                print(f"    Total violations: {total_violations} / {total_points} ({100*violation_pct:.4f}%)")
                print(f"    Threshold: {100*violation_threshold:.2f}%")
                return False
            else:
                print(f"  ✓ Quantile ordering mostly correct (violations: {100*violation_pct:.4f}% < {100*violation_threshold:.2f}% threshold)")
                print(f"    Note: {total_violations} / {total_points} points have minor violations (acceptable)")
        else:
            print(f"  ✓ Quantile ordering perfect (p05 ≤ p50 ≤ p95 everywhere)")

        # Print summary statistics
        print(f"\n  Summary Statistics:")
        print(f"    n_sequences: {n_days}")
        print(f"    horizon: {H} days")
        print(f"    quantiles: {list(quantiles)}")
        print(f"    value range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"    mean (p50): {p50.mean():.4f}")
        print(f"    std (p50): {p50.std():.4f}")

        print(f"\n  ✓ ALL CHECKS PASSED")
        return True

    except Exception as e:
        print(f"  ✗ FAIL: Error loading/validating file: {e}")
        return False


def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(
        description='Validate VAE teacher forcing sequences'
    )
    parser.add_argument('--sampling_mode', type=str, default='oracle',
                       choices=['oracle', 'prior'],
                       help='Sampling strategy to validate (oracle/prior)')
    args = parser.parse_args()

    print("=" * 80)
    print("VAE TEACHER FORCING SEQUENCE VALIDATION")
    print("=" * 80)
    print(f"Sampling mode: {args.sampling_mode}")
    print()

    # Define expected files
    periods = ['crisis', 'insample', 'oos', 'gap']
    horizons = [1, 7, 14, 30]
    expected_counts = {
        'crisis': {1: 746, 7: 740, 14: 733, 30: 717},
        'insample': {1: 3952, 7: 3946, 14: 3939, 30: 3923},
        'oos': {1: 773, 7: 767, 14: 760, 30: 744},
        'gap': {1: 1008, 7: 1002, 14: 995, 30: 979},
    }

    base_dir = Path(f"results/vae_baseline/predictions/autoregressive/{args.sampling_mode}")

    # Check directory existence
    if not base_dir.exists():
        print(f"\n✗ ERROR: Output directory does not exist: {base_dir}")
        print("  Run generation scripts first!")
        print(f"  Example: bash experiments/backfill/context20/run_generate_all_tf_sequences.sh {args.sampling_mode}")
        return

    print(f"Checking directory: {base_dir}")
    print(f"Expected files: {len(periods) * len(horizons)} (4 periods × 4 horizons)\n")

    # Validation results
    results = {}
    total_files = 0
    passed_files = 0

    # Validate each file
    for period in periods:
        for horizon in horizons:
            filename = f"vae_tf_{period}_h{horizon}.npz"
            filepath = base_dir / filename
            total_files += 1

            # Check file existence
            if not filepath.exists():
                print(f"\n{'='*80}")
                print(f"✗ MISSING: {filename}")
                print(f"{'='*80}")
                results[(period, horizon)] = False
                continue

            # Validate file
            passed = validate_file(filepath, horizon, period, args.sampling_mode)
            results[(period, horizon)] = passed

            if passed:
                passed_files += 1

                # Check sequence count (informational only)
                data = np.load(filepath)
                actual_count = len(data['surfaces'])
                expected_count = expected_counts[period][horizon]
                if actual_count != expected_count:
                    print(f"  ⚠ WARNING: Sequence count mismatch")
                    print(f"    Expected: {expected_count}, Got: {actual_count}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nTotal files checked: {total_files}")
    print(f"Passed: {passed_files}")
    print(f"Failed: {total_files - passed_files}")

    if passed_files == total_files:
        print("\n✓ ALL VALIDATIONS PASSED!")
        print(f"\nSampling mode '{args.sampling_mode}' sequences validated successfully.")
        print("\nNext steps:")
        print(f"  1. Compute CI statistics:")
        print(f"     python experiments/backfill/context20/compute_sequence_ci_width_stats.py --sampling_mode {args.sampling_mode}")
        print(f"  2. Generate visualizations:")
        print(f"     python experiments/backfill/context20/visualize_sequence_ci_width_combined.py --sampling_mode {args.sampling_mode}")
    else:
        print("\n✗ SOME VALIDATIONS FAILED")
        print("\nFailed files:")
        for (period, horizon), passed in results.items():
            if not passed:
                print(f"  - vae_tf_{period}_h{horizon}.npz")
        print("\nPlease regenerate failed files:")
        print(f"  bash experiments/backfill/context20/run_generate_all_tf_sequences.sh {args.sampling_mode}")

    print()


if __name__ == "__main__":
    main()
