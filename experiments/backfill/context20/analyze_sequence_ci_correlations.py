"""
Analyze Sequence CI Width Correlations with Context Features

Performs correlation analysis between CI width statistics (min/max/avg/range)
and context features (returns, volatility, skew, slope) for key grid points.
Also compares CI width across market regimes (crisis/covid/normal).

Input: results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz
Output: results/vae_baseline/analysis/{sampling_mode}/sequence_ci_correlations.csv
        results/vae_baseline/analysis/{sampling_mode}/sequence_ci_regime_comparison.csv
        results/vae_baseline/analysis/{sampling_mode}/SEQUENCE_CI_CORRELATION_SUMMARY.md

Usage:
    python experiments/backfill/context20/analyze_sequence_ci_correlations.py --sampling_mode oracle
    python experiments/backfill/context20/analyze_sequence_ci_correlations.py --sampling_mode prior
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Parse arguments
parser = argparse.ArgumentParser(description='Analyze sequence CI width correlations')
parser.add_argument('--sampling_mode', type=str, default='oracle',
                   choices=['oracle', 'prior'],
                   help='Sampling strategy (oracle/prior)')
args = parser.parse_args()

print("=" * 80)
print("SEQUENCE CI WIDTH CORRELATION ANALYSIS")
print("=" * 80)
print(f"Sampling mode: {args.sampling_mode}")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading computed sequence CI width statistics...")
data = np.load(f"results/vae_baseline/analysis/{args.sampling_mode}/sequence_ci_width_stats.npz", allow_pickle=True)
print(f"  Keys loaded: {len(data.files)} total")
print()

# ============================================================================
# Define Grid Structure
# ============================================================================

MONEYNESS_VALUES = [0.70, 0.85, 1.00, 1.15, 1.30]
MATURITY_DAYS = [30, 91, 182, 365, 730]
MATURITY_LABELS = ['1M', '3M', '6M', '1Y', '2Y']

# Key grid points for analysis (not all 25 to avoid excessive computation)
KEY_GRID_POINTS = [
    (2, 2,  'ATM 6M'),      # ATM, 6-month
    (2, 0,  'ATM 1M'),      # ATM, 1-month
    (2, 4,  'ATM 2Y'),      # ATM, 2-year
    (0, 2,  'OTM Put 6M'),  # Deep OTM put, 6-month
    (4, 2,  'OTM Call 6M'), # Deep OTM call, 6-month
]

print("Grid structure:")
print(f"  Moneyness (K/S): {MONEYNESS_VALUES}")
print(f"  Maturity: {MATURITY_LABELS}")
print()
print("Key grid points for analysis:")
for m_idx, t_idx, label in KEY_GRID_POINTS:
    print(f"  - ({m_idx}, {t_idx}): K/S={MONEYNESS_VALUES[m_idx]:.2f}, {MATURITY_LABELS[t_idx]} - {label}")
print()

# ============================================================================
# Helper Functions
# ============================================================================

def compute_correlation_per_grid_point(ci_stat, features, feature_name,
                                        period, horizon, m_idx, t_idx, grid_label):
    """
    Compute correlation between CI width statistic and feature for a single grid point.
    """
    ci_values = ci_stat[:, m_idx, t_idx]

    # Remove NaNs
    mask = ~(np.isnan(ci_values) | np.isnan(features))
    ci_clean = ci_values[mask]
    feat_clean = features[mask]

    if len(ci_clean) < 10:
        return None

    r, p = stats.pearsonr(ci_clean, feat_clean)

    return {
        'period': period,
        'horizon': horizon,
        'grid_point': grid_label,
        'moneyness_idx': m_idx,
        'maturity_idx': t_idx,
        'moneyness': MONEYNESS_VALUES[m_idx],
        'maturity': MATURITY_LABELS[t_idx],
        'feature': feature_name,
        'correlation': r,
        'p_value': p,
        'n_samples': len(ci_clean),
        'significant': p < 0.05
    }


def analyze_regime_differences(ci_stat, regime_masks, period, horizon, m_idx, t_idx, grid_label):
    """
    Compare CI width across market regimes (crisis/covid/normal).
    """
    ci_values = ci_stat[:, m_idx, t_idx]

    ci_crisis = ci_values[regime_masks['is_crisis']]
    ci_covid = ci_values[regime_masks['is_covid']]
    ci_normal = ci_values[regime_masks['is_normal']]

    # KS test: crisis vs normal
    if len(ci_crisis) > 0 and len(ci_normal) > 0:
        ks_crisis_stat, ks_crisis_pval = stats.ks_2samp(ci_crisis, ci_normal)
    else:
        ks_crisis_stat, ks_crisis_pval = np.nan, np.nan

    # KS test: covid vs normal
    if len(ci_covid) > 0 and len(ci_normal) > 0:
        ks_covid_stat, ks_covid_pval = stats.ks_2samp(ci_covid, ci_normal)
    else:
        ks_covid_stat, ks_covid_pval = np.nan, np.nan

    return {
        'period': period,
        'horizon': horizon,
        'grid_point': grid_label,
        'moneyness_idx': m_idx,
        'maturity_idx': t_idx,
        'moneyness': MONEYNESS_VALUES[m_idx],
        'maturity': MATURITY_LABELS[t_idx],
        'crisis_n': len(ci_crisis),
        'crisis_mean': ci_crisis.mean() if len(ci_crisis) > 0 else np.nan,
        'crisis_std': ci_crisis.std() if len(ci_crisis) > 0 else np.nan,
        'covid_n': len(ci_covid),
        'covid_mean': ci_covid.mean() if len(ci_covid) > 0 else np.nan,
        'covid_std': ci_covid.std() if len(ci_covid) > 0 else np.nan,
        'normal_n': len(ci_normal),
        'normal_mean': ci_normal.mean() if len(ci_normal) > 0 else np.nan,
        'normal_std': ci_normal.std() if len(ci_normal) > 0 else np.nan,
        'crisis_vs_normal_ratio': (ci_crisis.mean() / ci_normal.mean()) if len(ci_crisis) > 0 and len(ci_normal) > 0 else np.nan,
        'covid_vs_normal_ratio': (ci_covid.mean() / ci_normal.mean()) if len(ci_covid) > 0 and len(ci_normal) > 0 else np.nan,
        'ks_crisis_normal_stat': ks_crisis_stat,
        'ks_crisis_normal_pval': ks_crisis_pval,
        'ks_covid_normal_stat': ks_covid_stat,
        'ks_covid_normal_pval': ks_covid_pval
    }


# ============================================================================
# Analysis 1: Correlation Analysis
# ============================================================================

print("=" * 80)
print("ANALYSIS 1: CORRELATIONS WITH CONTEXT FEATURES")
print("=" * 80)
print()

correlation_results = []
periods = ['insample', 'crisis', 'oos']
horizons = [1, 7, 14, 30]

features_to_analyze = [
    'abs_returns',
    'realized_vol_30d',
    'skews',
    'slopes',
    'atm_vol'
]

ci_stats_to_analyze = ['min', 'max', 'avg', 'range']

total_analyses = 0

for period in periods:
    for h in horizons:
        prefix = f'{period}_h{h}'

        # Check if data exists for this combination
        if f'{prefix}_indices' not in data.files:
            print(f"  ⚠ Skipping {period} H={h} (no data)")
            continue

        indices = data[f'{prefix}_indices']

        # Skip if no data
        if len(indices) == 0:
            print(f"  ⚠ Skipping {period} H={h} (empty)")
            continue

        print(f"Processing {period} H={h} ({len(indices)} samples):")

        # Extract regime masks
        regime_masks = {
            'is_crisis': data[f'{prefix}_is_crisis'],
            'is_covid': data[f'{prefix}_is_covid'],
            'is_normal': data[f'{prefix}_is_normal']
        }

        for m_idx, t_idx, grid_label in KEY_GRID_POINTS:
            # Correlations
            for ci_stat_name in ci_stats_to_analyze:
                ci_stat = data[f'{prefix}_{ci_stat_name}_ci_width']

                for feat_name in features_to_analyze:
                    features = data[f'{prefix}_{feat_name}']

                    result = compute_correlation_per_grid_point(
                        ci_stat, features, feat_name, period, h, m_idx, t_idx, grid_label
                    )
                    if result is not None:
                        result['ci_stat'] = ci_stat_name
                        correlation_results.append(result)
                        total_analyses += 1

        print(f"  ✓ Completed {len(KEY_GRID_POINTS)} grid points × {len(ci_stats_to_analyze)} CI stats × {len(features_to_analyze)} features")

print()
print(f"Total correlation analyses: {total_analyses}")
print()

# Convert to DataFrame
df_correlations = pd.DataFrame(correlation_results)

# Display summary
if len(df_correlations) > 0:
    print("Summary of significant correlations (p < 0.05):")
    significant = df_correlations[df_correlations['significant']]
    print(f"  Total significant: {len(significant)} / {len(df_correlations)} ({100*len(significant)/len(df_correlations):.1f}%)")

    # Top 10 strongest correlations
    print("\n  Top 10 strongest correlations:")
    top_10 = df_correlations.nlargest(10, 'correlation', keep='all')[['period', 'horizon', 'grid_point', 'ci_stat', 'feature', 'correlation', 'p_value']]
    for idx, row in top_10.iterrows():
        sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"    {row['period']:9s} H={row['horizon']:2d} {row['grid_point']:12s} {row['ci_stat']:5s} vs {row['feature']:17s}: r={row['correlation']:+.3f} {sig_marker}")

print()

# ============================================================================
# Analysis 2: Regime Comparison
# ============================================================================

print("=" * 80)
print("ANALYSIS 2: REGIME COMPARISON (CRISIS / COVID / NORMAL)")
print("=" * 80)
print()

regime_results = []

for period in periods:
    for h in horizons:
        prefix = f'{period}_h{h}'

        if f'{prefix}_indices' not in data.files:
            continue

        indices = data[f'{prefix}_indices']
        if len(indices) == 0:
            continue

        print(f"Processing {period} H={h}:")

        # Extract regime masks
        regime_masks = {
            'is_crisis': data[f'{prefix}_is_crisis'],
            'is_covid': data[f'{prefix}_is_covid'],
            'is_normal': data[f'{prefix}_is_normal']
        }

        # Only analyze avg_ci_width for regime comparison
        avg_ci = data[f'{prefix}_avg_ci_width']

        for m_idx, t_idx, grid_label in KEY_GRID_POINTS:
            regime_result = analyze_regime_differences(
                avg_ci, regime_masks, period, h, m_idx, t_idx, grid_label
            )
            regime_results.append(regime_result)

        print(f"  ✓ Completed {len(KEY_GRID_POINTS)} grid points")

print()

# Convert to DataFrame
df_regimes = pd.DataFrame(regime_results)

# Display summary
if len(df_regimes) > 0:
    print("Summary of regime differences:")
    print(f"  Total comparisons: {len(df_regimes)}")

    # Crisis vs Normal significant differences
    crisis_sig = df_regimes[df_regimes['ks_crisis_normal_pval'] < 0.05]
    print(f"  Crisis vs Normal significant (p < 0.05): {len(crisis_sig)} / {len(df_regimes)} ({100*len(crisis_sig)/len(df_regimes):.1f}%)")

    # COVID vs Normal significant differences
    covid_sig = df_regimes[df_regimes['ks_covid_normal_pval'] < 0.05]
    print(f"  COVID vs Normal significant (p < 0.05): {len(covid_sig)} / {len(df_regimes)} ({100*len(covid_sig)/len(df_regimes):.1f}%)")

print()

# ============================================================================
# Save Results
# ============================================================================

output_dir = Path(f"results/vae_baseline/analysis/{args.sampling_mode}")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Save correlation results
corr_file = output_dir / "sequence_ci_correlations.csv"
df_correlations.to_csv(corr_file, index=False)
print(f"✓ Saved correlation results: {corr_file}")
print(f"  Rows: {len(df_correlations)}, Columns: {len(df_correlations.columns)}")

# Save regime comparison results
regime_file = output_dir / "sequence_ci_regime_comparison.csv"
df_regimes.to_csv(regime_file, index=False)
print(f"✓ Saved regime comparison results: {regime_file}")
print(f"  Rows: {len(df_regimes)}, Columns: {len(df_regimes.columns)}")

print()

# ============================================================================
# Generate Summary Report
# ============================================================================

print("Generating summary markdown report...")

summary_file = output_dir / "SEQUENCE_CI_CORRELATION_SUMMARY.md"

with open(summary_file, 'w') as f:
    f.write("# Sequence CI Width Correlation Analysis Summary\n\n")
    f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("---\n\n")

    # Overview
    f.write("## Overview\n\n")
    f.write(f"- **Total correlation analyses:** {total_analyses}\n")
    f.write(f"- **Key grid points analyzed:** {len(KEY_GRID_POINTS)}\n")
    f.write(f"- **Periods:** {', '.join(periods)}\n")
    f.write(f"- **Horizons:** {', '.join(map(str, horizons))}\n")
    f.write(f"- **CI statistics:** {', '.join(ci_stats_to_analyze)}\n")
    f.write(f"- **Features:** {', '.join(features_to_analyze)}\n\n")

    # Correlation results
    if len(df_correlations) > 0:
        f.write("## Correlation Analysis\n\n")
        f.write(f"**Total correlations computed:** {len(df_correlations)}\n\n")
        f.write(f"**Significant correlations (p < 0.05):** {len(significant)} ({100*len(significant)/len(df_correlations):.1f}%)\n\n")

        f.write("### Top 10 Strongest Correlations\n\n")
        f.write("| Period | H | Grid Point | CI Stat | Feature | r | p-value | Sig |\n")
        f.write("|--------|---|------------|---------|---------|-------|---------|-----|\n")

        top_10 = df_correlations.nlargest(10, 'correlation')[['period', 'horizon', 'grid_point', 'ci_stat', 'feature', 'correlation', 'p_value', 'significant']]
        for idx, row in top_10.iterrows():
            sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            f.write(f"| {row['period']} | {row['horizon']} | {row['grid_point']} | {row['ci_stat']} | {row['feature']} | {row['correlation']:+.3f} | {row['p_value']:.2e} | {sig_marker} |\n")

        f.write("\n")

    # Regime comparison results
    if len(df_regimes) > 0:
        f.write("## Regime Comparison Analysis\n\n")
        f.write(f"**Total regime comparisons:** {len(df_regimes)}\n\n")
        f.write(f"**Crisis vs Normal significant (p < 0.05):** {len(crisis_sig)} ({100*len(crisis_sig)/len(df_regimes):.1f}%)\n\n")
        f.write(f"**COVID vs Normal significant (p < 0.05):** {len(covid_sig)} ({100*len(covid_sig)/len(df_regimes):.1f}%)\n\n")

        # Find largest crisis vs normal differences
        f.write("### Largest Crisis vs Normal Differences\n\n")
        f.write("(Sorted by crisis/normal ratio)\n\n")
        f.write("| Period | H | Grid Point | Crisis Mean | Normal Mean | Ratio | KS p-value |\n")
        f.write("|--------|---|------------|-------------|-------------|-------|------------|\n")

        top_crisis = df_regimes.dropna(subset=['crisis_vs_normal_ratio']).nlargest(10, 'crisis_vs_normal_ratio')[['period', 'horizon', 'grid_point', 'crisis_mean', 'normal_mean', 'crisis_vs_normal_ratio', 'ks_crisis_normal_pval']]
        for idx, row in top_crisis.iterrows():
            f.write(f"| {row['period']} | {row['horizon']} | {row['grid_point']} | {row['crisis_mean']:.4f} | {row['normal_mean']:.4f} | {row['crisis_vs_normal_ratio']:.3f} | {row['ks_crisis_normal_pval']:.2e} |\n")

        f.write("\n")

    # Data files
    f.write("## Output Files\n\n")
    f.write(f"- **Correlation results:** `{corr_file.name}`\n")
    f.write(f"- **Regime comparison:** `{regime_file.name}`\n")
    f.write(f"- **Input data:** `sequence_ci_width_stats.npz`\n\n")

print(f"✓ Saved summary report: {summary_file}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("CORRELATION ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print("  1. Run visualize_sequence_ci_width.py for time series plots and correlation scatters")
print("  2. Run identify_ci_width_events.py to find extreme widening/narrowing events")
print()
