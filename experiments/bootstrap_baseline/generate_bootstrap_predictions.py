"""
Bootstrap Baseline Generator

Non-parametric bootstrap method that samples from the empirical distribution
of historical implied volatility changes. This provides a model-free benchmark
that naturally preserves fat tails and empirical correlations.

Methodology:
1. Compute Δ(IV) = IV[t] - IV[t-1] from training data
2. For each forecast day, sample random Δ(IV) from historical pool
3. Apply to previous surface: IV[t+1] = IV[t] + Δ(IV)_sampled
4. Repeat 1000 times for uncertainty quantification

Usage:
    python experiments/bootstrap_baseline/generate_bootstrap_predictions.py
"""

import numpy as np
import os
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

def generate_bootstrap_forecast(initial_surface, historical_changes, horizon=1, n_samples=1000):
    """
    Generate bootstrap forecast for given horizon.

    Args:
        initial_surface: (5, 5) starting surface
        historical_changes: (N, 5, 5) pool of historical changes
        horizon: Number of days to forecast
        n_samples: Number of bootstrap samples

    Returns:
        forecast: (n_samples, 5, 5) distribution of forecasts
    """
    forecast = np.zeros((n_samples, 5, 5))

    # Start from initial surface
    current_surfaces = np.tile(initial_surface, (n_samples, 1, 1))  # (n_samples, 5, 5)

    # Apply sequential changes for each day in horizon
    for day in range(horizon):
        # Sample changes for all samples at once
        sampled_changes = bootstrap_sample_changes(historical_changes, n_samples)
        # Apply changes
        current_surfaces = current_surfaces + sampled_changes

    return current_surfaces

def generate_predictions(surface, train_end_idx, forecast_start_idx, forecast_end_idx,
                        horizons=[1, 7, 14, 30], n_samples=1000):
    """
    Generate bootstrap predictions for all horizons.

    Args:
        surface: (N, 5, 5) full surface data
        train_end_idx: Last index of training data
        forecast_start_idx: First index to forecast
        forecast_end_idx: Last index to forecast
        horizons: List of forecast horizons
        n_samples: Number of bootstrap samples per day

    Returns:
        predictions: Dict mapping horizon -> (n_days, n_samples, 5, 5)
    """
    # Compute historical changes from training period
    train_surface = surface[:train_end_idx]
    historical_changes = compute_changes(train_surface)

    print(f"Training period: indices 0 to {train_end_idx} ({len(train_surface)} days)")
    print(f"Historical changes: {len(historical_changes)} samples")
    print(f"Forecast period: indices {forecast_start_idx} to {forecast_end_idx}")

    predictions = {}

    for horizon in horizons:
        print(f"\nGenerating predictions for H={horizon}")

        # For each forecast day, need context up to that day
        forecast_days = []

        for day_idx in tqdm(range(forecast_start_idx, forecast_end_idx + 1),
                           desc=f"H={horizon}"):
            # Get the surface from the previous day as starting point
            # For H=1: start from day_idx-1, predict day_idx
            # For H>1: start from day_idx-horizon, predict day_idx
            start_idx = day_idx - horizon

            # Ensure we have enough history
            if start_idx < 0:
                print(f"Warning: day {day_idx} has insufficient history for H={horizon}, skipping")
                continue

            initial_surface = surface[start_idx]

            # Generate bootstrap forecast
            forecast = generate_bootstrap_forecast(
                initial_surface=initial_surface,
                historical_changes=historical_changes,
                horizon=horizon,
                n_samples=n_samples
            )

            forecast_days.append(forecast)

        # Stack all days
        predictions[horizon] = np.array(forecast_days)  # (n_days, n_samples, 5, 5)
        print(f"Generated shape: {predictions[horizon].shape}")

    return predictions

def save_predictions(predictions, output_dir, period_name):
    """
    Save predictions in format compatible with other baselines.

    Saves one file per horizon with p05, p50, p95 quantiles.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for horizon, pred in predictions.items():
        # pred shape: (n_days, n_samples, 5, 5)

        # Compute quantiles across samples (axis=1)
        p05 = np.percentile(pred, 5, axis=1)   # (n_days, 5, 5)
        p50 = np.percentile(pred, 50, axis=1)  # (n_days, 5, 5)
        p95 = np.percentile(pred, 95, axis=1)  # (n_days, 5, 5)

        # Stack into (n_days, 3, 5, 5) format like quantile VAE
        surfaces = np.stack([p05, p50, p95], axis=1)

        # Save
        output_file = output_path / f"bootstrap_predictions_H{horizon}.npz"
        np.savez(output_file, surfaces=surfaces)
        print(f"Saved: {output_file}")
        print(f"  Shape: {surfaces.shape}")

def main():
    parser = argparse.ArgumentParser(description='Generate bootstrap baseline predictions')
    parser.add_argument('--period', type=str, choices=['insample', 'oos', 'both'],
                       default='both', help='Which period to generate predictions for')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of bootstrap samples per day')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("="*80)
    print("Bootstrap Baseline Generator")
    print("="*80)

    # Load data
    print("\nLoading data...")
    surface = load_data()
    print(f"Loaded surface data: {surface.shape}")

    # Define periods
    # Based on config/backfill_config.py:
    # - Training: up to index 5000 (before test set)
    # - In-sample forecast: 2004-2019 (roughly indices 0-3900)
    # - OOS forecast: 2019-2023 (roughly indices 5000+)

    train_end_idx = 5000  # Use all data before test set
    insample_start = 0
    insample_end = 3900  # Roughly end of 2019
    oos_start = 5000  # Start of test set
    oos_end = len(surface) - 31  # Leave room for H=30 horizon

    horizons = [1, 7, 14, 30]

    # Generate in-sample predictions
    if args.period in ['insample', 'both']:
        print("\n" + "="*80)
        print("Generating IN-SAMPLE predictions (2004-2019)")
        print("="*80)

        insample_predictions = generate_predictions(
            surface=surface,
            train_end_idx=insample_end,  # Use only data up to forecast point
            forecast_start_idx=insample_start,
            forecast_end_idx=insample_end,
            horizons=horizons,
            n_samples=args.n_samples
        )

        save_predictions(
            predictions=insample_predictions,
            output_dir="results/bootstrap_baseline/predictions/insample",
            period_name="insample"
        )

    # Generate OOS predictions
    if args.period in ['oos', 'both']:
        print("\n" + "="*80)
        print("Generating OUT-OF-SAMPLE predictions (2019-2023)")
        print("="*80)

        oos_predictions = generate_predictions(
            surface=surface,
            train_end_idx=train_end_idx,  # Use all training data (fixed)
            forecast_start_idx=oos_start,
            forecast_end_idx=oos_end,
            horizons=horizons,
            n_samples=args.n_samples
        )

        save_predictions(
            predictions=oos_predictions,
            output_dir="results/bootstrap_baseline/predictions/oos",
            period_name="oos"
        )

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated bootstrap predictions for comparison with:")
    print("  - Econometric baseline (co-integration + EWMA)")
    print("  - VAE Context20 model (Oracle & Prior)")
    print("\nOutput locations:")
    if args.period in ['insample', 'both']:
        print("  - results/bootstrap_baseline/predictions/insample/")
    if args.period in ['oos', 'both']:
        print("  - results/bootstrap_baseline/predictions/oos/")
    print("\nNext steps:")
    print("  1. Run RMSE comparison analysis")
    print("  2. Run CI calibration comparison")
    print("  3. Run co-integration comparison")
    print("  4. Run marginal distribution comparison (addresses shape mismatch)")

if __name__ == "__main__":
    main()
