"""
Bootstrap Autoregressive Sequence Generation

Generates 30-day autoregressive sequences by feeding predictions back as context.
This tests whether bootstrap's joint spatial sampling preserves co-integration
relationships over time, or if random sampling causes correlation drift.

Key difference from teacher forcing:
- Teacher forcing: Each prediction uses real historical surface
- Autoregressive: Feed median prediction back as context for next step

Usage:
    python experiments/bootstrap_baseline/generate_bootstrap_autoregressive.py --period crisis

Outputs:
    results/bootstrap_baseline/predictions/autoregressive/bootstrap_ar_crisis.npz
    Shape: (n_days, 30, 3, 5, 5) - crisis days × 30 steps × 3 quantiles × 5×5 grid
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def load_data():
    """Load volatility surface data."""
    data_path = "data/vol_surface_with_ret.npz"
    data = np.load(data_path)
    surface = data['surface']  # (N, 5, 5)
    return surface


def compute_changes(surface):
    """
    Compute first differences: Δ(IV)[t] = IV[t] - IV[t-1]

    Returns:
        changes: (N-1, 5, 5) array of daily changes
    """
    changes = surface[1:] - surface[:-1]
    return changes


def bootstrap_sample_changes(historical_changes, n_samples=1):
    """
    Sample n_samples random change vectors from historical distribution.

    Args:
        historical_changes: (N, 5, 5) historical Δ(IV) vectors
        n_samples: Number of bootstrap samples to draw

    Returns:
        sampled_changes: (n_samples, 5, 5) randomly sampled changes
    """
    n_historical = len(historical_changes)
    # Sample with replacement
    sample_indices = np.random.choice(n_historical, size=n_samples, replace=True)
    sampled_changes = historical_changes[sample_indices]
    return sampled_changes


def generate_autoregressive_sequence(
    initial_surface,
    historical_changes,
    horizon=30,
    n_samples=1000
):
    """
    Generate multi-step autoregressive sequence by feeding predictions back as context.

    Pattern mirrors VAE's generate_autoregressive_sequence() but uses bootstrap sampling:
    1. Sample Δ(IV) from historical pool
    2. Apply to current surface: IV[t+1] = IV[t] + Δ(IV)
    3. Extract p50 median as point estimate
    4. Feed median back as context for next step
    5. Repeat for 30 days

    Args:
        initial_surface: (5, 5) - starting surface (real data)
        historical_changes: (N, 5, 5) - pool of historical Δ(IV)
        horizon: int - number of days to generate (default: 30)
        n_samples: int - bootstrap samples per step (default: 1000)

    Returns:
        sequence: (horizon, n_samples, 5, 5) - 1000 samples for each of 30 days
    """
    generated_days = []
    current_surface = initial_surface.copy()  # (5, 5)

    for step in range(horizon):
        # Sample Δ(IV) from historical pool
        sampled_changes = bootstrap_sample_changes(historical_changes, n_samples)

        # Apply to current surface
        # Broadcasting: (5,5) + (n_samples, 5, 5) = (n_samples, 5, 5)
        next_surfaces = current_surface + sampled_changes

        generated_days.append(next_surfaces)

        # Extract p50 median as point estimate for next iteration
        # This is the autoregressive feedback: feed prediction back as context
        current_surface = np.percentile(next_surfaces, 50, axis=0)  # (5, 5)

    return np.array(generated_days)  # (horizon, n_samples, 5, 5)


def generate_predictions_autoregressive(
    surface,
    train_end_idx,
    forecast_start_idx,
    forecast_end_idx,
    horizon=30,
    n_samples=1000
):
    """
    Generate autoregressive predictions for all days in forecast period.

    Each day gets a full 30-day autoregressive sequence starting from that day.

    Args:
        surface: (N, 5, 5) full surface data
        train_end_idx: Last index of training data
        forecast_start_idx: First index to forecast
        forecast_end_idx: Last index to forecast
        horizon: Days to forecast per sequence (default: 30)
        n_samples: Bootstrap samples per day (default: 1000)

    Returns:
        predictions: (n_days, horizon, n_samples, 5, 5)
    """
    # Compute historical changes from training period
    train_surface = surface[:train_end_idx]
    historical_changes = compute_changes(train_surface)

    print(f"Training period: indices 0 to {train_end_idx} ({len(train_surface)} days)")
    print(f"Historical changes: {len(historical_changes)} samples")
    print(f"Forecast period: indices {forecast_start_idx} to {forecast_end_idx}")
    print(f"Generating 30-day sequences for each day...")

    n_forecast_days = forecast_end_idx - forecast_start_idx + 1
    predictions = []

    for day_idx in tqdm(range(forecast_start_idx, forecast_end_idx + 1),
                       desc="Generating AR sequences"):
        # Use previous day's surface as starting point
        # This is the KEY DIFFERENCE from teacher forcing:
        # We always start from day_idx-1 (not day_idx-horizon)
        if day_idx == 0:
            print(f"Warning: Cannot generate for day 0 (no previous day), skipping")
            continue

        initial_surface = surface[day_idx - 1]

        # Generate 30-day autoregressive sequence
        sequence = generate_autoregressive_sequence(
            initial_surface=initial_surface,
            historical_changes=historical_changes,
            horizon=horizon,
            n_samples=n_samples
        )

        predictions.append(sequence)

    # Stack all days
    predictions = np.array(predictions)  # (n_days, horizon, n_samples, 5, 5)
    print(f"Generated shape: {predictions.shape}")

    return predictions


def save_predictions_autoregressive(predictions, output_dir, period_name):
    """
    Save autoregressive predictions in format compatible with analysis scripts.

    Converts (n_days, horizon, n_samples, 5, 5) to (n_days, horizon, 3, 5, 5)
    by computing quantiles across samples.

    Args:
        predictions: (n_days, horizon, n_samples, 5, 5)
        output_dir: Path to save directory
        period_name: Name of period (e.g., 'crisis')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # predictions shape: (n_days, horizon, n_samples, 5, 5)
    n_days, horizon, n_samples, n_rows, n_cols = predictions.shape

    print(f"\nComputing quantiles...")
    print(f"  Input shape: {predictions.shape}")

    # Compute quantiles across samples (axis=2)
    p05 = np.percentile(predictions, 5, axis=2)   # (n_days, horizon, 5, 5)
    p50 = np.percentile(predictions, 50, axis=2)  # (n_days, horizon, 5, 5)
    p95 = np.percentile(predictions, 95, axis=2)  # (n_days, horizon, 5, 5)

    # Stack into (n_days, horizon, 3, 5, 5) format
    surfaces = np.stack([p05, p50, p95], axis=2)

    # Save
    output_file = output_path / f"bootstrap_ar_{period_name}.npz"
    np.savez(output_file, surfaces=surfaces)

    print(f"\nSaved: {output_file}")
    print(f"  Shape: {surfaces.shape}")
    print(f"  Format: (n_days={n_days}, horizon={horizon}, quantiles=3, rows=5, cols=5)")
    print(f"  Quantiles: [p05, p50, p95]")


def main():
    parser = argparse.ArgumentParser(
        description='Generate bootstrap autoregressive predictions'
    )
    parser.add_argument(
        '--period',
        type=str,
        choices=['crisis', 'insample', 'oos'],
        default='crisis',
        help='Which period to generate predictions for'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=30,
        help='Number of days to forecast per sequence'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Number of bootstrap samples per day'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("="*80)
    print("Bootstrap Autoregressive Sequence Generator")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Period: {args.period}")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Samples per day: {args.n_samples}")
    print(f"  Random seed: {args.seed}")

    # Load data
    print("\nLoading data...")
    surface = load_data()
    print(f"Loaded surface data: {surface.shape}")

    # Define periods
    # Based on config/backfill_config.py and previous analysis
    if args.period == 'crisis':
        # Crisis period: 2008-2010 (indices 2000-2765)
        train_end_idx = 2000  # Train on data before crisis
        forecast_start_idx = 2000
        forecast_end_idx = 2765
        period_name = 'crisis'
    elif args.period == 'insample':
        # In-sample: 2004-2019
        train_end_idx = 3900
        forecast_start_idx = 0
        forecast_end_idx = 3900
        period_name = 'insample'
    else:  # oos
        # Out-of-sample: 2019-2023
        train_end_idx = 5000
        forecast_start_idx = 5000
        forecast_end_idx = 5792
        period_name = 'oos'

    print(f"\nPeriod: {period_name}")
    print(f"  Training: indices 0-{train_end_idx}")
    print(f"  Forecast: indices {forecast_start_idx}-{forecast_end_idx}")
    print(f"  Forecast days: {forecast_end_idx - forecast_start_idx + 1}")

    # Generate predictions
    print("\n" + "="*80)
    print(f"Generating Autoregressive Predictions ({period_name.upper()})")
    print("="*80)

    predictions = generate_predictions_autoregressive(
        surface=surface,
        train_end_idx=train_end_idx,
        forecast_start_idx=forecast_start_idx,
        forecast_end_idx=forecast_end_idx,
        horizon=args.horizon,
        n_samples=args.n_samples
    )

    # Save predictions
    print("\n" + "="*80)
    print("Saving Predictions")
    print("="*80)

    save_predictions_autoregressive(
        predictions=predictions,
        output_dir="results/bootstrap_baseline/predictions/autoregressive",
        period_name=period_name
    )

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated {args.horizon}-day autoregressive sequences for {period_name} period")
    print(f"Each day has {args.n_samples} bootstrap samples")
    print("\nKey characteristics:")
    print("  - Autoregressive: Feeds p50 predictions back as context")
    print("  - Joint spatial sampling: Preserves correlations at each step")
    print("  - Random sampling: May cause correlation drift over 30 days")
    print("\nOutput location:")
    print(f"  results/bootstrap_baseline/predictions/autoregressive/bootstrap_ar_{period_name}.npz")
    print("\nNext steps:")
    print("  1. Run analyze_autoregressive_cointegration.py to measure drift")
    print("  2. Compare to econometric and VAE autoregressive sequences")
    print("  3. Test hypothesis: Does bootstrap preserve co-integration?")


if __name__ == "__main__":
    main()
