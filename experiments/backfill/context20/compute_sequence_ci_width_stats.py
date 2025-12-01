"""
Compute CI Width Statistics for VAE Teacher Forcing Sequences

Analyzes full H-day CI width trajectories to understand how uncertainty evolves
across entire forecasted sequences. Computes min/max/avg/std/range statistics
for CI width across each horizon.

Input: results/vae_baseline/predictions/autoregressive/{sampling_mode}/vae_tf_{period}_h{horizon}.npz
Output: results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz

Usage:
    python experiments/backfill/context20/compute_sequence_ci_width_stats.py --sampling_mode oracle
    python experiments/backfill/context20/compute_sequence_ci_width_stats.py --sampling_mode prior
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Compute CI width statistics for VAE TF sequences')
parser.add_argument('--sampling_mode', type=str, default='oracle',
                   choices=['oracle', 'prior'],
                   help='Sampling strategy (oracle/prior)')
args = parser.parse_args()

print("=" * 80)
print("COMPUTING SEQUENCE CI WIDTH STATISTICS - VAE TEACHER FORCING")
print("=" * 80)
print(f"Sampling mode: {args.sampling_mode}")
print()

# ============================================================================
# Helper Functions
# ============================================================================

def compute_sequence_ci_width(surfaces, horizon):
    """
    Compute CI width statistics across entire H-day sequence.

    Args:
        surfaces: (n_dates, H, 3, 5, 5) - VAE TF predictions
        horizon: int - horizon length H

    Returns:
        min_ci_width: (n_dates, 5, 5) - minimum CI width across H days
        max_ci_width: (n_dates, 5, 5) - maximum CI width across H days
        avg_ci_width: (n_dates, 5, 5) - average CI width across H days
        std_ci_width: (n_dates, 5, 5) - std dev of CI width across H days
        range_ci_width: (n_dates, 5, 5) - range (max - min) across H days
    """
    # Extract quantiles
    p05 = surfaces[:, :, 0, :, :]  # (n_dates, H, 5, 5)
    p95 = surfaces[:, :, 2, :, :]  # (n_dates, H, 5, 5)

    # Compute CI width for each day in sequence
    ci_width_sequence = p95 - p05  # (n_dates, H, 5, 5)

    # Aggregate across horizon (axis=1)
    min_ci_width = np.min(ci_width_sequence, axis=1)   # (n_dates, 5, 5)
    max_ci_width = np.max(ci_width_sequence, axis=1)   # (n_dates, 5, 5)
    avg_ci_width = np.mean(ci_width_sequence, axis=1)  # (n_dates, 5, 5)
    std_ci_width = np.std(ci_width_sequence, axis=1)   # (n_dates, 5, 5)
    range_ci_width = max_ci_width - min_ci_width       # (n_dates, 5, 5)

    return min_ci_width, max_ci_width, avg_ci_width, std_ci_width, range_ci_width


def compute_realized_volatility(returns, indices, window=30):
    """
    Compute rolling realized volatility from returns.

    [Reused from compute_ci_width_timeseries.py]

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


def extract_context_features(indices, gt_data):
    """
    Extract context features for given date indices.

    Args:
        indices: (n_dates,) - date indices
        gt_data: dict from vol_surface_with_ret.npz

    Returns:
        dict with:
            'returns': (n_dates,)
            'abs_returns': (n_dates,)
            'realized_vol_30d': (n_dates,)
            'skews': (n_dates,)
            'slopes': (n_dates,)
            'atm_vol': (n_dates,) - surface[:, 2, 2] (ATM 6-month)
    """
    returns = gt_data['ret'][indices]
    skews = gt_data['skews'][indices]
    slopes = gt_data['slopes'][indices]
    surfaces = gt_data['surface'][indices]

    # ATM volatility (moneyness=1.0, maturity=6M)
    atm_vol = surfaces[:, 2, 2]

    # Realized vol
    realized_vol_30d = compute_realized_volatility(gt_data['ret'], indices, window=30)

    return {
        'returns': returns,
        'abs_returns': np.abs(returns),
        'realized_vol_30d': realized_vol_30d,
        'skews': skews,
        'slopes': slopes,
        'atm_vol': atm_vol
    }


def define_market_regimes(dates):
    """
    Define market regime indicators for given dates.

    [Reused from analyze_ci_width_statistics.py]

    Returns:
        dict with boolean arrays:
            'is_crisis': (n_dates,) - 2008-2010 crisis
            'is_covid': (n_dates,) - Feb-Apr 2020
            'is_normal': (n_dates,) - all other periods
    """
    crisis_start = pd.Timestamp('2008-01-01')
    crisis_end = pd.Timestamp('2010-12-31')
    covid_start = pd.Timestamp('2020-02-15')
    covid_end = pd.Timestamp('2020-04-30')

    is_crisis = (dates >= crisis_start) & (dates <= crisis_end)
    is_covid = (dates >= covid_start) & (dates <= covid_end)
    is_normal = ~(is_crisis | is_covid)

    return {
        'is_crisis': is_crisis,
        'is_covid': is_covid,
        'is_normal': is_normal
    }


# ============================================================================
# Load Ground Truth Data
# ============================================================================

print("Loading ground truth data...")
gt_data = np.load("data/vol_surface_with_ret.npz")
print(f"  Surfaces shape: {gt_data['surface'].shape}")
print(f"  Returns shape: {gt_data['ret'].shape}")
print()

print("Loading date mapping...")
df_dates = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
print(f"  Total dates: {len(df_dates)}")
print(f"  Date range: {df_dates['date'].min()} to {df_dates['date'].max()}")
print()

# ============================================================================
# Process All VAE TF Sequences
# ============================================================================

results = {}
horizons = [1, 7, 14, 30]
periods = ['insample', 'crisis', 'oos', 'gap']

print("Processing VAE TF sequences...")
print(f"  Sampling mode: {args.sampling_mode}")
print(f"  Periods: {periods}")
print(f"  Horizons: {horizons}")
print()

total_processed = 0

for period in periods:
    print(f"{'='*80}")
    print(f"PERIOD: {period.upper()}")
    print(f"{'='*80}")
    print()

    for h in horizons:
        # Load VAE TF sequences
        filepath = f"results/vae_baseline/predictions/autoregressive/{args.sampling_mode}/vae_tf_{period}_h{h}.npz"
        filepath_obj = Path(filepath)

        if not filepath_obj.exists():
            print(f"  ⚠ WARNING: {filepath} not found, skipping")
            print()
            continue

        print(f"  Horizon H={h}:")
        data = np.load(filepath)

        surfaces = data['surfaces']  # (n_dates, H, 3, 5, 5)
        indices = data['indices']    # (n_dates,)

        print(f"    Loaded: {filepath}")
        print(f"    Shape: {surfaces.shape}")
        print(f"    Date indices: {indices.min()} to {indices.max()}")

        # Compute sequence CI width stats
        min_ci, max_ci, avg_ci, std_ci, range_ci = compute_sequence_ci_width(surfaces, h)

        print(f"    CI width stats computed:")
        print(f"      Min CI: [{min_ci.min():.4f}, {min_ci.max():.4f}]")
        print(f"      Max CI: [{max_ci.min():.4f}, {max_ci.max():.4f}]")
        print(f"      Avg CI: [{avg_ci.min():.4f}, {avg_ci.max():.4f}]")
        print(f"      Range: [{range_ci.min():.4f}, {range_ci.max():.4f}]")

        # Extract context features
        context_feats = extract_context_features(indices, gt_data)

        # Map to dates
        dates = pd.to_datetime(df_dates.iloc[indices]['date'].values)

        # Define market regimes
        regimes = define_market_regimes(dates)

        print(f"    Context features extracted:")
        print(f"      Realized vol (30d): [{np.nanmin(context_feats['realized_vol_30d']):.2f}%, {np.nanmax(context_feats['realized_vol_30d']):.2f}%]")
        print(f"      |Returns|: [{context_feats['abs_returns'].min():.4f}, {context_feats['abs_returns'].max():.4f}]")
        print(f"      Regimes - Crisis: {regimes['is_crisis'].sum()} days, COVID: {regimes['is_covid'].sum()} days, Normal: {regimes['is_normal'].sum()} days")

        # Store results
        prefix = f'{period}_h{h}'
        results[f'{prefix}_min_ci_width'] = min_ci
        results[f'{prefix}_max_ci_width'] = max_ci
        results[f'{prefix}_avg_ci_width'] = avg_ci
        results[f'{prefix}_std_ci_width'] = std_ci
        results[f'{prefix}_range_ci_width'] = range_ci
        results[f'{prefix}_indices'] = indices
        results[f'{prefix}_dates'] = dates.astype(str).values
        results[f'{prefix}_returns'] = context_feats['returns']
        results[f'{prefix}_abs_returns'] = context_feats['abs_returns']
        results[f'{prefix}_realized_vol_30d'] = context_feats['realized_vol_30d']
        results[f'{prefix}_skews'] = context_feats['skews']
        results[f'{prefix}_slopes'] = context_feats['slopes']
        results[f'{prefix}_atm_vol'] = context_feats['atm_vol']
        results[f'{prefix}_is_crisis'] = regimes['is_crisis']
        results[f'{prefix}_is_covid'] = regimes['is_covid']
        results[f'{prefix}_is_normal'] = regimes['is_normal']

        total_processed += 1
        print()

    print()

# ============================================================================
# Save Results
# ============================================================================

output_dir = Path(f"results/vae_baseline/analysis/{args.sampling_mode}")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "sequence_ci_width_stats.npz"

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()
print(f"Sampling mode: {args.sampling_mode}")
print(f"Output directory: {output_dir}")
print(f"Output file: {output_file}")
print(f"Total processed: {total_processed} period-horizon combinations")
print()

np.savez(output_file, **results)

file_size_mb = output_file.stat().st_size / (1024 ** 2)
print(f"✓ Results saved successfully")
print(f"  File size: {file_size_mb:.1f} MB")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("COMPUTATION COMPLETE")
print("=" * 80)
print()
print("Saved statistics for each period-horizon combination:")
print("  - min_ci_width, max_ci_width, avg_ci_width: (n_dates, 5, 5)")
print("  - std_ci_width, range_ci_width: (n_dates, 5, 5)")
print("  - Context features: returns, realized_vol_30d, skews, slopes, atm_vol")
print("  - Regime indicators: is_crisis, is_covid, is_normal")
print()
print("Next steps:")
print("  1. Run analyze_sequence_ci_correlations.py for statistical analysis")
print("  2. Run visualize_sequence_ci_width.py for time series plots")
print("  3. Run identify_ci_width_events.py to find extreme widening/narrowing events")
print()
