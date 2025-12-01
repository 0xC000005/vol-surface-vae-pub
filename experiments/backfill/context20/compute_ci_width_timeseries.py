"""
Compute CI Width Time Series for VAE Context 20 Model

Extracts confidence interval (p95 - p05) widths over time from existing predictions
and correlates with market indicators (realized vol, returns, RMSE).

Output: results/backfill_16yr/analysis/ci_width_timeseries_16yr.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("COMPUTING CI WIDTH TIME SERIES - VAE Context 20")
print("=" * 80)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# 1. In-sample predictions
print("  Loading in-sample predictions...")
insample_data = np.load("results/backfill_16yr/predictions/vae_prior_insample_16yr.npz")

# 2. OOS predictions
print("  Loading OOS predictions...")
oos_data = np.load("results/backfill_16yr/predictions/vae_prior_oos_16yr.npz")

# 3. VAE health metrics (contains RMSE per sample)
print("  Loading VAE health metrics...")
health_data = np.load("results/backfill_16yr/predictions/vae_health_16yr.npz")

# 4. Date mapping
print("  Loading date mapping...")
df_dates = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")

# 5. Ground truth (for returns)
print("  Loading ground truth data...")
gt_data = np.load("data/vol_surface_with_ret.npz")
returns = gt_data['ret']  # (5822,)

print(f"  Total dates in dataset: {len(df_dates)}")
print(f"  Date range: {df_dates['date'].min()} to {df_dates['date'].max()}")
print()

# ============================================================================
# Define Horizons
# ============================================================================

horizons = [1, 7, 14, 30]
print(f"Processing horizons: {horizons}")
print()

# ============================================================================
# Compute CI Width Time Series
# ============================================================================

def compute_ci_width(predictions, horizon_key):
    """
    Compute CI width from quantile predictions.

    Args:
        predictions: dict with 'recon_h{H}' and 'indices_h{H}' keys
        horizon_key: int, e.g., 1, 7, 14, 30

    Returns:
        ci_width: (N,) - mean CI width across 5x5 grid per sample
        ci_width_per_grid: (N, 5, 5) - CI width for each grid point
        indices: (N,) - date indices
    """
    recon_key = f'recon_h{horizon_key}'
    indices_key = f'indices_h{horizon_key}'

    recons = predictions[recon_key]  # (N, 3, 5, 5)
    indices = predictions[indices_key]  # (N,)

    # Extract quantiles
    p05 = recons[:, 0, :, :]  # (N, 5, 5)
    p95 = recons[:, 2, :, :]  # (N, 5, 5)

    # Compute CI width
    ci_width_per_grid = p95 - p05  # (N, 5, 5)
    ci_width = np.mean(ci_width_per_grid, axis=(1, 2))  # (N,) - mean across grid

    return ci_width, ci_width_per_grid, indices


def compute_realized_volatility(returns, indices, window=30):
    """
    Compute rolling realized volatility from returns.

    Args:
        returns: (T,) - daily returns
        indices: (N,) - date indices to compute vol for
        window: int - rolling window size

    Returns:
        realized_vol: (N,) - 30-day realized volatility
    """
    realized_vol = np.zeros(len(indices))

    for i, idx in enumerate(indices):
        if idx < window:
            # Not enough history, use what's available
            start = 0
        else:
            start = idx - window

        window_returns = returns[start:idx]
        if len(window_returns) > 0:
            realized_vol[i] = np.std(window_returns) * np.sqrt(252)  # Annualized
        else:
            realized_vol[i] = np.nan

    return realized_vol


print("Computing CI width time series for each horizon...")
print()

results = {}

for h in horizons:
    print(f"Horizon {h}:")

    # Combine in-sample and OOS
    # Note: Some horizons might not have OOS data
    insample_ci, insample_ci_grid, insample_indices = compute_ci_width(insample_data, h)

    try:
        oos_ci, oos_ci_grid, oos_indices = compute_ci_width(oos_data, h)

        # Concatenate
        ci_width = np.concatenate([insample_ci, oos_ci])
        ci_width_per_grid = np.concatenate([insample_ci_grid, oos_ci_grid], axis=0)
        indices = np.concatenate([insample_indices, oos_indices])
    except (KeyError, FileNotFoundError):
        # OOS might not exist for all horizons
        print(f"  Note: No OOS data for horizon {h}, using in-sample only")
        ci_width = insample_ci
        ci_width_per_grid = insample_ci_grid
        indices = insample_indices

    print(f"  Total samples: {len(ci_width)}")
    print(f"  CI width range: [{ci_width.min():.4f}, {ci_width.max():.4f}]")
    print(f"  Mean CI width: {ci_width.mean():.4f}")

    # Map indices to dates
    dates = df_dates.iloc[indices]['date'].values
    dates_str = pd.to_datetime(dates).astype(str).values

    # Get RMSE from health data
    health_key = f'h{h}_rmse_per_sample'
    health_indices_key = f'h{h}_date_indices'

    if health_key in health_data.files:
        rmse_from_health = health_data[health_key]
        rmse_indices = health_data[health_indices_key]

        # Align RMSE with our indices
        # Create a mapping: index -> rmse
        rmse_map = dict(zip(rmse_indices, rmse_from_health))
        rmse = np.array([rmse_map.get(idx, np.nan) for idx in indices])
        print(f"  RMSE range: [{np.nanmin(rmse):.4f}, {np.nanmax(rmse):.4f}]")
    else:
        print(f"  Warning: No RMSE data found for horizon {h}")
        rmse = np.full(len(indices), np.nan)

    # Compute realized volatility
    realized_vol_30d = compute_realized_volatility(returns, indices, window=30)
    print(f"  Realized vol (30d) range: [{np.nanmin(realized_vol_30d):.2f}%, {np.nanmax(realized_vol_30d):.2f}%]")

    # Absolute returns
    abs_returns = np.abs(returns[indices])
    print(f"  |Returns| range: [{abs_returns.min():.4f}, {abs_returns.max():.4f}]")

    # Store results
    results[f'h{h}_ci_width'] = ci_width
    results[f'h{h}_ci_width_per_grid'] = ci_width_per_grid
    results[f'h{h}_indices'] = indices
    results[f'h{h}_dates'] = dates_str
    results[f'h{h}_rmse'] = rmse
    results[f'h{h}_realized_vol_30d'] = realized_vol_30d
    results[f'h{h}_abs_returns'] = abs_returns

    print()

# ============================================================================
# Save Results
# ============================================================================

output_dir = Path("results/backfill_16yr/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "ci_width_timeseries_16yr.npz"

print(f"Saving results to: {output_file}")
np.savez(output_file, **results)

print()
print("=" * 80)
print("CI WIDTH TIME SERIES COMPUTATION COMPLETE")
print("=" * 80)
print()
print(f"Output saved to: {output_file}")
print()
print("Next steps:")
print("  1. Run visualize_ci_width_temporal.py to create temporal plots")
print("  2. Run analyze_ci_width_statistics.py for statistical analysis")
