"""
Validate VAE Teacher Forcing Sequences for Context60 Model (Single-Output Decoder)

Validates all 24 generated NPZ files (4 periods × 6 horizons) to ensure:
1. File existence
2. Correct shapes (n_days, H, 5, 5)
3. Reasonable value ranges (0.01 < IV < 5.0)
4. No NaN/Inf values
5. Index uniqueness
6. Correct sampling_mode metadata
7. Correct context_len metadata (60)

Usage:
    python experiments/backfill/context60_v4_full_cov/teacher_forcing/validate_vae_tf_sequences.py --sampling_mode oracle
    python experiments/backfill/context60_v4_full_cov/teacher_forcing/validate_vae_tf_sequences.py --sampling_mode prior
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
        required_keys = ['surfaces', 'indices', 'horizon',
                        'period_start', 'period_end', 'method', 'sampling_mode']
        missing_keys = [k for k in required_keys if k not in data.keys()]
        if missing_keys:
            print(f"  ✗ FAIL: Missing keys: {missing_keys}")
            return False
        print(f"  ✓ All required keys present")

        # Extract data
        surfaces = data['surfaces']
        indices = data['indices']
        horizon = data['horizon']
        method = str(data['method'])
        sampling_mode = str(data['sampling_mode'])

        # Check context_len (ADDED for context60)
        context_len = int(data.get('context_len', 20))
        if context_len != 60:
            print(f"  ✗ FAIL: Expected context_len=60, got {context_len}")
            return False
        print(f"  ✓ Context length correct: {context_len}")

        # Check sampling mode
        if sampling_mode != expected_sampling_mode:
            print(f"  ✗ FAIL: Sampling mode mismatch (expected '{expected_sampling_mode}', got '{sampling_mode}')")
            return False
        print(f"  ✓ Sampling mode correct: {sampling_mode}")

        # Check shape (single-output decoder: no quantiles)
        if len(surfaces.shape) != 4:
            print(f"  ✗ FAIL: Expected 4D array, got {len(surfaces.shape)}D")
            print(f"    Got shape: {surfaces.shape}")
            return False

        n_days, H, n_rows, n_cols = surfaces.shape
        expected_shape = (n_days, expected_horizons, 5, 5)
        if surfaces.shape[1:] != expected_shape[1:]:
            print(f"  ✗ FAIL: Shape mismatch")
            print(f"    Expected: (n_days, {expected_horizons}, 5, 5)")
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

        # Print summary statistics
        print(f"\n  Summary Statistics:")
        print(f"    n_sequences: {n_days}")
        print(f"    horizon: {H} days")
        print(f"    value range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"    mean: {surfaces.mean():.4f}")
        print(f"    std: {surfaces.std():.4f}")

        print(f"\n  ✓ ALL CHECKS PASSED")
        return True

    except Exception as e:
        print(f"  ✗ FAIL: Error loading/validating file: {e}")
        return False


def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(
        description='Validate VAE teacher forcing sequences for context60 model'
    )
    parser.add_argument('--sampling_mode', type=str, default='oracle',
                       choices=['oracle', 'prior'],
                       help='Sampling strategy to validate (oracle/prior)')
    parser.add_argument('--output_dir', type=str,
                       default='results/context60_v4_full_cov',
                       help='Base output directory for predictions')
    args = parser.parse_args()

    print("=" * 80)
    print("VAE TEACHER FORCING SEQUENCE VALIDATION (CONTEXT60)")
    print("=" * 80)
    print(f"Sampling mode: {args.sampling_mode}")
    print()

    # Define expected files
    periods = ['crisis', 'insample', 'oos', 'gap']
    horizons = [1, 7, 14, 30, 60, 90]  # CHANGED: Added 60, 90

    # CHANGED: Updated expected counts for context60 (context_len=60)
    expected_counts = {
        'crisis': {1: 706, 7: 700, 14: 693, 30: 677, 60: 647, 90: 617},
        'insample': {1: 3912, 7: 3906, 14: 3899, 30: 3883, 60: 3853, 90: 3823},
        'oos': {1: 733, 7: 727, 14: 720, 30: 704, 60: 674, 90: 644},
        'gap': {1: 968, 7: 962, 14: 955, 30: 939, 60: 909, 90: 879},
    }

    base_dir = Path(f"{args.output_dir}/predictions/teacher_forcing/{args.sampling_mode}")

    # Check directory existence
    if not base_dir.exists():
        print(f"\n✗ ERROR: Output directory does not exist: {base_dir}")
        print("  Run generation scripts first!")
        print(f"  Example: python experiments/backfill/context60_v4_full_cov/teacher_forcing/generate_vae_tf_sequences.py --period crisis --sampling_mode {args.sampling_mode}")
        return

    print(f"Checking directory: {base_dir}")
    print(f"Expected files: {len(periods) * len(horizons)} (4 periods × 6 horizons)\n")  # CHANGED: 4×6

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
        print(f"  1. Test autoregressive generation scripts")
        print(f"  2. Compare context20 vs context60 performance")
    else:
        print("\n✗ SOME VALIDATIONS FAILED")
        print("\nFailed files:")
        for (period, horizon), passed in results.items():
            if not passed:
                print(f"  - vae_tf_{period}_h{horizon}.npz")
        print("\nPlease regenerate failed files:")
        print(f"  python experiments/backfill/context60_v4_full_cov/teacher_forcing/generate_vae_tf_sequences.py --period <period> --sampling_mode {args.sampling_mode}")

    print()


if __name__ == "__main__":
    main()
