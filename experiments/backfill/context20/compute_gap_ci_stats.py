"""
Compute CI Width Statistics for Gap Period and Merge into Existing Stats File

This script processes the gap period (2015-2019) sequences and adds them
to the existing sequence_ci_width_stats.npz file.

Input: results/vae_baseline/predictions/autoregressive/{sampling_mode}/vae_tf_gap_h{horizon}.npz
Output: Merges into results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz

Usage:
    python experiments/backfill/context20/compute_gap_ci_stats.py --sampling_mode oracle
    python experiments/backfill/context20/compute_gap_ci_stats.py --sampling_mode prior
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Compute gap period CI width statistics')
parser.add_argument('--sampling_mode', type=str, default='oracle',
                   choices=['oracle', 'prior'],
                   help='Sampling strategy (oracle/prior)')
args = parser.parse_args()

print("=" * 80)
print("COMPUTING GAP PERIOD CI WIDTH STATISTICS")
print("=" * 80)
print(f"Sampling mode: {args.sampling_mode}")
print()

# ============================================================================
# Helper Functions (from compute_sequence_ci_width_stats.py)
# ============================================================================

def compute_sequence_ci_width(surfaces, horizon):
    """Compute CI width statistics across entire H-day sequence."""
    p05 = surfaces[:, :, 0, :, :]  # (n_dates, H, 5, 5)
    p95 = surfaces[:, :, 2, :, :]  # (n_dates, H, 5, 5)
    ci_width_sequence = p95 - p05  # (n_dates, H, 5, 5)

    min_ci_width = np.min(ci_width_sequence, axis=1)
    max_ci_width = np.max(ci_width_sequence, axis=1)
    avg_ci_width = np.mean(ci_width_sequence, axis=1)
    std_ci_width = np.std(ci_width_sequence, axis=1)
    range_ci_width = max_ci_width - min_ci_width

    return min_ci_width, max_ci_width, avg_ci_width, std_ci_width, range_ci_width


def compute_realized_volatility(returns, indices, window=30):
    """Compute rolling realized volatility from returns."""
    realized_vol = np.zeros(len(indices))

    for i, idx in enumerate(indices):
        if idx < window:
            start = 0
        else:
            start = idx - window

        window_returns = returns[start:idx]
        if len(window_returns) > 0:
            realized_vol[i] = np.std(window_returns) * np.sqrt(252)
        else:
            realized_vol[i] = np.nan

    return realized_vol


def extract_context_features(indices, gt_data):
    """Extract context features for given date indices."""
    returns = gt_data['ret'][indices]
    skews = gt_data['skews'][indices]
    slopes = gt_data['slopes'][indices]
    surfaces = gt_data['surface'][indices]
    atm_vol = surfaces[:, 2, 2]
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
    """Define market regime indicators for given dates."""
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
# Load Existing Statistics
# ============================================================================

print("Loading existing statistics file...")
stats_file = Path(f"results/vae_baseline/analysis/{args.sampling_mode}/sequence_ci_width_stats.npz")
existing_data = np.load(stats_file, allow_pickle=True)
existing_keys = list(existing_data.files)
print(f"  Loaded {len(existing_keys)} existing keys from {stats_file}")
print()

# ============================================================================
# Load Ground Truth Data
# ============================================================================

print("Loading ground truth data...")
gt_data = np.load("data/vol_surface_with_ret.npz")
df_dates = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
print(f"  Surfaces shape: {gt_data['surface'].shape}")
print(f"  Date range: {df_dates['date'].min()} to {df_dates['date'].max()}")
print()

# ============================================================================
# Process Gap Period Sequences
# ============================================================================

gap_results = {}
horizons = [1, 7, 14, 30]

print("Processing gap period sequences...")
print(f"  Horizons: {horizons}")
print()

for h in horizons:
    filepath = f"results/vae_baseline/predictions/autoregressive/{args.sampling_mode}/vae_tf_gap_h{h}.npz"
    filepath_obj = Path(filepath)

    if not filepath_obj.exists():
        print(f"  ⚠ WARNING: {filepath} not found, skipping")
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

    # Extract context features
    context_feats = extract_context_features(indices, gt_data)

    # Map to dates
    dates = pd.to_datetime(df_dates.iloc[indices]['date'].values)

    # Define market regimes
    regimes = define_market_regimes(dates)

    print(f"    Context features extracted")
    print(f"      Date range: {dates.min()} to {dates.max()}")
    print(f"      Regimes - Crisis: {regimes['is_crisis'].sum()}, COVID: {regimes['is_covid'].sum()}, Normal: {regimes['is_normal'].sum()}")

    # Store results with 'gap' prefix
    prefix = f'gap_h{h}'
    gap_results[f'{prefix}_min_ci_width'] = min_ci
    gap_results[f'{prefix}_max_ci_width'] = max_ci
    gap_results[f'{prefix}_avg_ci_width'] = avg_ci
    gap_results[f'{prefix}_std_ci_width'] = std_ci
    gap_results[f'{prefix}_range_ci_width'] = range_ci
    gap_results[f'{prefix}_indices'] = indices
    gap_results[f'{prefix}_dates'] = dates.astype(str).values
    gap_results[f'{prefix}_returns'] = context_feats['returns']
    gap_results[f'{prefix}_abs_returns'] = context_feats['abs_returns']
    gap_results[f'{prefix}_realized_vol_30d'] = context_feats['realized_vol_30d']
    gap_results[f'{prefix}_skews'] = context_feats['skews']
    gap_results[f'{prefix}_slopes'] = context_feats['slopes']
    gap_results[f'{prefix}_atm_vol'] = context_feats['atm_vol']
    gap_results[f'{prefix}_is_crisis'] = regimes['is_crisis']
    gap_results[f'{prefix}_is_covid'] = regimes['is_covid']
    gap_results[f'{prefix}_is_normal'] = regimes['is_normal']

    print()

# ============================================================================
# Merge and Save
# ============================================================================

print("=" * 80)
print("MERGING RESULTS")
print("=" * 80)
print()

# Combine existing and new results
merged_results = {key: existing_data[key] for key in existing_keys}
merged_results.update(gap_results)

print(f"  Existing keys: {len(existing_keys)}")
print(f"  Gap keys: {len(gap_results)}")
print(f"  Total keys: {len(merged_results)}")
print()

# Save merged results
print(f"Saving merged results to {stats_file}...")
np.savez(stats_file, **merged_results)

file_size_mb = stats_file.stat().st_size / (1024 ** 2)
print(f"✓ Merged results saved successfully")
print(f"  File size: {file_size_mb:.1f} MB")
print()

print("=" * 80)
print("GAP STATISTICS COMPUTED AND MERGED")
print("=" * 80)
print()
print("Next step: Run visualize_sequence_ci_width_combined.py")
print()
