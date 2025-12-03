"""
Compute CI Width Statistics for Context60 Model

Computes min/max/avg/std/range CI width statistics across forecast horizons
for both teacher forcing (H=1,7,14,30,60,90) and autoregressive (H=180,270) predictions.

Output: sequence_ci_width_stats.npz for both oracle and prior modes

Usage:
    python experiments/backfill/context60/compute_sequence_ci_width_stats_context60.py
"""
import numpy as np
import os
from pathlib import Path


def compute_sequence_ci_width(surfaces, horizon):
    """
    Compute CI width statistics across forecast horizon.

    For each starting date, computes aggregated statistics of CI widths
    across the H-day forecast sequence.

    Args:
        surfaces: (n_dates, H, 3, 5, 5) - VAE predictions
                  3 channels: [p05, p50, p95]
        horizon: int - horizon length (for metadata/logging)

    Returns:
        min_ci_width: (n_dates, 5, 5) - min CI width across H days
        max_ci_width: (n_dates, 5, 5) - max CI width across H days
        avg_ci_width: (n_dates, 5, 5) - average CI width across H days
        std_ci_width: (n_dates, 5, 5) - std dev across H days
        range_ci_width: (n_dates, 5, 5) - range (max - min)
    """
    # Extract quantiles
    p05 = surfaces[:, :, 0, :, :]  # (n_dates, H, 5, 5)
    p95 = surfaces[:, :, 2, :, :]  # (n_dates, H, 5, 5)

    # Compute CI width for each day in sequence
    ci_width_sequence = p95 - p05  # (n_dates, H, 5, 5)

    # Aggregate across horizon dimension (axis=1)
    min_ci_width = np.min(ci_width_sequence, axis=1)    # (n_dates, 5, 5)
    max_ci_width = np.max(ci_width_sequence, axis=1)    # (n_dates, 5, 5)
    avg_ci_width = np.mean(ci_width_sequence, axis=1)   # (n_dates, 5, 5)
    std_ci_width = np.std(ci_width_sequence, axis=1)    # (n_dates, 5, 5)
    range_ci_width = max_ci_width - min_ci_width        # (n_dates, 5, 5)

    return min_ci_width, max_ci_width, avg_ci_width, std_ci_width, range_ci_width


def load_predictions(period, horizon, mode, base_dir="results/context60_baseline/predictions"):
    """
    Load prediction file for given period, horizon, and mode.

    Handles different file naming conventions for TF and AR predictions.

    Args:
        period: str - 'crisis', 'insample', 'oos', or 'gap'
        horizon: int - forecast horizon
        mode: str - 'oracle' or 'prior'
        base_dir: str - base directory for predictions

    Returns:
        data: dict - loaded NPZ file contents
    """
    if horizon in [1, 7, 14, 30, 60, 90]:  # Teacher forcing
        filepath = os.path.join(
            base_dir,
            "teacher_forcing",
            mode,
            f"vae_tf_{period}_h{horizon}.npz"
        )
    elif horizon in [180, 270]:  # Autoregressive
        filepath = os.path.join(
            base_dir,
            "autoregressive_multi_step",
            mode,
            f"vae_ar_{period}_{horizon}day.npz"
        )
    else:
        raise ValueError(f"Unknown horizon: {horizon}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prediction file not found: {filepath}")

    return np.load(filepath)


def process_mode(mode, periods, tf_horizons, ar_horizons, base_dir="results/context60_baseline/predictions"):
    """
    Process all periods and horizons for a given sampling mode.

    Args:
        mode: str - 'oracle' or 'prior'
        periods: list - period names
        tf_horizons: list - teacher forcing horizons
        ar_horizons: list - autoregressive horizons
        base_dir: str - base directory for predictions

    Returns:
        stats: dict - all computed statistics with keys like '{period}_h{horizon}_{stat_name}'
    """
    print(f"\n{'='*80}")
    print(f"Processing {mode.upper()} mode")
    print(f"{'='*80}\n")

    stats = {}
    total_files = len(periods) * (len(tf_horizons) + len(ar_horizons))
    processed = 0

    # Process TF horizons
    for period in periods:
        for h in tf_horizons:
            try:
                # Load data
                data = load_predictions(period, h, mode, base_dir)
                surfaces = data['surfaces']  # (n_dates, h, 3, 5, 5)
                indices = data['indices']     # (n_dates,)

                # Compute statistics
                min_ci, max_ci, avg_ci, std_ci, range_ci = compute_sequence_ci_width(surfaces, h)

                # Store with unified key format
                stats[f'{period}_h{h}_min_ci_width'] = min_ci
                stats[f'{period}_h{h}_max_ci_width'] = max_ci
                stats[f'{period}_h{h}_avg_ci_width'] = avg_ci
                stats[f'{period}_h{h}_std_ci_width'] = std_ci
                stats[f'{period}_h{h}_range_ci_width'] = range_ci
                stats[f'{period}_h{h}_indices'] = indices

                processed += 1
                print(f"  [{processed:2d}/{total_files}] ✓ {period.upper():8s} H={h:3d} (TF): {len(indices):4d} sequences")

            except Exception as e:
                print(f"  [{processed:2d}/{total_files}] ✗ {period.upper():8s} H={h:3d} (TF): ERROR - {e}")
                raise

    # Process AR horizons
    for period in periods:
        for h in ar_horizons:
            try:
                # Load data
                data = load_predictions(period, h, mode, base_dir)
                surfaces = data['surfaces']  # (n_dates, total_horizon, 3, 5, 5)
                indices = data['indices']     # (n_dates,)

                # Compute statistics (same function works for AR!)
                min_ci, max_ci, avg_ci, std_ci, range_ci = compute_sequence_ci_width(surfaces, h)

                # Store with unified key format
                stats[f'{period}_h{h}_min_ci_width'] = min_ci
                stats[f'{period}_h{h}_max_ci_width'] = max_ci
                stats[f'{period}_h{h}_avg_ci_width'] = avg_ci
                stats[f'{period}_h{h}_std_ci_width'] = std_ci
                stats[f'{period}_h{h}_range_ci_width'] = range_ci
                stats[f'{period}_h{h}_indices'] = indices

                processed += 1
                print(f"  [{processed:2d}/{total_files}] ✓ {period.upper():8s} H={h:3d} (AR): {len(indices):4d} sequences")

            except Exception as e:
                print(f"  [{processed:2d}/{total_files}] ✗ {period.upper():8s} H={h:3d} (AR): ERROR - {e}")
                raise

    return stats


def validate_stats(stats, periods, horizons):
    """
    Validate computed statistics.

    Args:
        stats: dict - computed statistics
        periods: list - period names
        horizons: list - all horizons (TF + AR)

    Returns:
        bool - True if valid, raises exception otherwise
    """
    print(f"\n{'='*80}")
    print("Validating Statistics")
    print(f"{'='*80}\n")

    expected_keys_per_period_horizon = 6  # min, max, avg, std, range, indices
    expected_total_keys = len(periods) * len(horizons) * expected_keys_per_period_horizon

    actual_keys = len(stats)

    print(f"  Expected keys: {expected_total_keys}")
    print(f"  Actual keys:   {actual_keys}")

    if actual_keys != expected_total_keys:
        raise ValueError(f"Key count mismatch: expected {expected_total_keys}, got {actual_keys}")

    # Check for NaN values
    nan_count = 0
    for key, value in stats.items():
        if 'indices' not in key:  # Skip index arrays
            if np.any(np.isnan(value)):
                nan_count += 1
                print(f"  ⚠ WARNING: {key} contains {np.sum(np.isnan(value))} NaN values")

    if nan_count > 0:
        print(f"\n  ⚠ Total arrays with NaNs: {nan_count}")
    else:
        print(f"\n  ✓ No NaN values detected")

    # Check shapes
    print(f"\n  Sample shapes:")
    sample_period = periods[0]
    sample_horizon = horizons[0]
    for stat_type in ['min_ci_width', 'max_ci_width', 'avg_ci_width', 'indices']:
        key = f'{sample_period}_h{sample_horizon}_{stat_type}'
        if key in stats:
            shape = stats[key].shape
            print(f"    {stat_type:15s}: {shape}")

    print(f"\n  ✓ Validation passed")
    return True


def main():
    """Main execution pipeline."""
    print("="*80)
    print("CONTEXT60 CI WIDTH STATISTICS COMPUTATION")
    print("="*80)

    # Configuration
    periods = ['crisis', 'insample', 'oos', 'gap']
    tf_horizons = [1, 7, 14, 30, 60, 90]
    ar_horizons = [180, 270]
    all_horizons = tf_horizons + ar_horizons
    modes = ['oracle', 'prior']

    print(f"\nConfiguration:")
    print(f"  Periods: {periods}")
    print(f"  TF Horizons: {tf_horizons}")
    print(f"  AR Horizons: {ar_horizons}")
    print(f"  Modes: {modes}")
    print(f"  Total files to process: {len(periods) * len(all_horizons) * len(modes)} (32 per mode)")

    # Process each mode
    for mode in modes:
        # Compute statistics
        stats = process_mode(mode, periods, tf_horizons, ar_horizons)

        # Validate
        validate_stats(stats, periods, all_horizons)

        # Save to NPZ
        output_dir = f"results/context60_baseline/analysis/{mode}/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "sequence_ci_width_stats.npz")

        print(f"\n{'='*80}")
        print(f"Saving {mode.upper()} Statistics")
        print(f"{'='*80}\n")

        np.savez(output_file, **stats)

        # Verify saved file
        saved_data = np.load(output_file)
        print(f"  Output file: {output_file}")
        print(f"  File size: {os.path.getsize(output_file) / 1024**2:.1f} MB")
        print(f"  Keys saved: {len(saved_data.files)}")
        print(f"  ✓ File saved successfully")

        saved_data.close()

    # Final summary
    print(f"\n{'='*80}")
    print("COMPUTATION COMPLETE")
    print(f"{'='*80}\n")

    print("Output files:")
    for mode in modes:
        output_file = f"results/context60_baseline/analysis/{mode}/sequence_ci_width_stats.npz"
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / 1024**2
            print(f"  ✓ {output_file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {output_file} (NOT FOUND)")

    print("\nNext steps:")
    print("  1. Run visualization: python experiments/backfill/context60/visualize_oracle_vs_prior_combined_timeseries_context60.py")
    print("  2. Run comparison: python experiments/backfill/context60/compare_oracle_vs_prior_ci_context60.py")
    print()


if __name__ == "__main__":
    main()
