"""
Identify Extreme CI Width Widening/Narrowing Events

Identifies dates with the largest and smallest CI width ranges (max - min)
across H-day sequences to understand which contexts lead to widening or
narrowing confidence intervals.

Input: results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz
Output: results/vae_baseline/analysis/{sampling_mode}/CI_WIDTH_EVENTS_SUMMARY.md

Usage:
    python experiments/backfill/context20/identify_ci_width_events.py --sampling_mode oracle
    python experiments/backfill/context20/identify_ci_width_events.py --sampling_mode prior
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Identify extreme CI width events')
parser.add_argument('--sampling_mode', type=str, default='oracle',
                   choices=['oracle', 'prior'],
                   help='Sampling strategy (oracle/prior)')
args = parser.parse_args()

print("=" * 80)
print("IDENTIFYING CI WIDTH EVENTS")
print("=" * 80)
print(f"Sampling mode: {args.sampling_mode}")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading sequence CI width statistics...")
data = np.load(f"results/vae_baseline/analysis/{args.sampling_mode}/sequence_ci_width_stats.npz", allow_pickle=True)

horizons = [1, 7, 14, 30]
periods = ['insample', 'crisis', 'oos']

# ATM 6-month (benchmark grid point)
m_idx, t_idx = 2, 2
grid_label = 'ATM 6M'

print(f"Analyzing grid point: {grid_label} (K/S=1.00, 6-month)")
print()

# ============================================================================
# Identify Extreme Events
# ============================================================================

print("Identifying extreme widening/narrowing events...")
print()

widening_events = []
narrowing_events = []

for period in periods:
    for h in horizons:
        prefix = f'{period}_h{h}'

        if f'{prefix}_dates' not in data.files:
            continue

        # Load data
        range_ci = data[f'{prefix}_range_ci_width'][:, m_idx, t_idx]
        min_ci = data[f'{prefix}_min_ci_width'][:, m_idx, t_idx]
        max_ci = data[f'{prefix}_max_ci_width'][:, m_idx, t_idx]
        avg_ci = data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]
        dates = pd.to_datetime(data[f'{prefix}_dates'])
        abs_returns = data[f'{prefix}_abs_returns']
        realized_vol = data[f'{prefix}_realized_vol_30d']
        is_crisis = data[f'{prefix}_is_crisis']
        is_covid = data[f'{prefix}_is_covid']
        is_normal = data[f'{prefix}_is_normal']

        # Top-10 widening (largest range)
        top_10_idx = np.argsort(range_ci)[-10:][::-1]

        print(f"{period.upper()} H={h} - Top 10 Widening Events (Largest CI Range):")
        print("-" * 80)

        for rank, idx in enumerate(top_10_idx, 1):
            regime = ('Crisis' if is_crisis[idx] else
                     'COVID' if is_covid[idx] else
                     'Normal')

            event = {
                'period': period,
                'horizon': h,
                'event_type': 'Widening',
                'rank': rank,
                'date': str(dates[idx].date()),
                'min_ci_width': min_ci[idx],
                'max_ci_width': max_ci[idx],
                'avg_ci_width': avg_ci[idx],
                'range_ci_width': range_ci[idx],
                'abs_returns': abs_returns[idx],
                'realized_vol_30d': realized_vol[idx],
                'regime': regime
            }
            widening_events.append(event)

            print(f"  {rank:2d}. {dates[idx].date()}: "
                  f"Range={range_ci[idx]:.4f} "
                  f"[{min_ci[idx]:.4f} - {max_ci[idx]:.4f}], "
                  f"|Ret|={abs_returns[idx]:.4f}, "
                  f"RVol={realized_vol[idx]:.2f}%, "
                  f"Regime={regime}")

        print()

        # Top-10 narrowing (smallest range, excluding H=1 which has range=0)
        if h > 1:
            # Filter out zero ranges
            non_zero_mask = range_ci > 0.0001
            if non_zero_mask.sum() >= 10:
                non_zero_indices = np.where(non_zero_mask)[0]
                range_ci_filtered = range_ci[non_zero_indices]
                bottom_10_local_idx = np.argsort(range_ci_filtered)[:10]
                bottom_10_idx = non_zero_indices[bottom_10_local_idx]

                print(f"{period.upper()} H={h} - Top 10 Narrowing Events (Smallest CI Range):")
                print("-" * 80)

                for rank, idx in enumerate(bottom_10_idx, 1):
                    regime = ('Crisis' if is_crisis[idx] else
                             'COVID' if is_covid[idx] else
                             'Normal')

                    event = {
                        'period': period,
                        'horizon': h,
                        'event_type': 'Narrowing',
                        'rank': rank,
                        'date': str(dates[idx].date()),
                        'min_ci_width': min_ci[idx],
                        'max_ci_width': max_ci[idx],
                        'avg_ci_width': avg_ci[idx],
                        'range_ci_width': range_ci[idx],
                        'abs_returns': abs_returns[idx],
                        'realized_vol_30d': realized_vol[idx],
                        'regime': regime
                    }
                    narrowing_events.append(event)

                    print(f"  {rank:2d}. {dates[idx].date()}: "
                          f"Range={range_ci[idx]:.4f} "
                          f"[{min_ci[idx]:.4f} - {max_ci[idx]:.4f}], "
                          f"|Ret|={abs_returns[idx]:.4f}, "
                          f"RVol={realized_vol[idx]:.2f}%, "
                          f"Regime={regime}")

                print()

# Convert to DataFrames
df_widening = pd.DataFrame(widening_events)
df_narrowing = pd.DataFrame(narrowing_events)

# ============================================================================
# Generate Summary Report
# ============================================================================

print("=" * 80)
print("GENERATING SUMMARY REPORT")
print("=" * 80)
print()

output_dir = Path(f"results/vae_baseline/analysis/{args.sampling_mode}")
output_dir.mkdir(parents=True, exist_ok=True)

summary_file = output_dir / "CI_WIDTH_EVENTS_SUMMARY.md"

with open(summary_file, 'w') as f:
    f.write("# CI Width Events Summary - Extreme Widening/Narrowing\n\n")
    f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Grid Point:** {grid_label} (K/S=1.00, 6-month)\n\n")
    f.write("---\n\n")

    # Overview
    f.write("## Overview\n\n")
    f.write(f"- **Total widening events identified:** {len(df_widening)}\n")
    f.write(f"- **Total narrowing events identified:** {len(df_narrowing)}\n")
    f.write(f"- **Periods analyzed:** {', '.join(periods)}\n")
    f.write(f"- **Horizons analyzed:** {', '.join(map(str, horizons))}\n\n")

    # Widening events
    f.write("## Extreme Widening Events\n\n")
    f.write("**Top 10 widening events per period-horizon combination**\n\n")
    f.write("Widening indicates increasing uncertainty across the forecast sequence, "
            "typically associated with high volatility, large returns, or novel market conditions.\n\n")

    for period in periods:
        f.write(f"### {period.upper()} Period\n\n")

        for h in horizons:
            subset = df_widening[(df_widening['period'] == period) & (df_widening['horizon'] == h)]
            if len(subset) == 0:
                continue

            f.write(f"#### Horizon H={h}\n\n")
            f.write("| Rank | Date | Range | Min CI | Max CI | |Ret| | RVol(%) | Regime |\n")
            f.write("|------|------|-------|--------|--------|-------|---------|--------|\n")

            for _, row in subset.iterrows():
                f.write(f"| {row['rank']} | {row['date']} | {row['range_ci_width']:.4f} | "
                       f"{row['min_ci_width']:.4f} | {row['max_ci_width']:.4f} | "
                       f"{row['abs_returns']:.4f} | {row['realized_vol_30d']:.2f} | {row['regime']} |\n")

            f.write("\n")

        f.write("\n")

    # Narrowing events
    f.write("## Extreme Narrowing Events\n\n")
    f.write("**Top 10 narrowing events per period-horizon combination**\n\n")
    f.write("Narrowing indicates stable uncertainty across the forecast sequence, "
            "typically associated with calm markets, low volatility, and familiar patterns.\n\n")

    for period in periods:
        f.write(f"### {period.upper()} Period\n\n")

        for h in horizons:
            subset = df_narrowing[(df_narrowing['period'] == period) & (df_narrowing['horizon'] == h)]
            if len(subset) == 0:
                continue

            f.write(f"#### Horizon H={h}\n\n")
            f.write("| Rank | Date | Range | Min CI | Max CI | |Ret| | RVol(%) | Regime |\n")
            f.write("|------|------|-------|--------|--------|-------|---------|--------|\n")

            for _, row in subset.iterrows():
                f.write(f"| {row['rank']} | {row['date']} | {row['range_ci_width']:.4f} | "
                       f"{row['min_ci_width']:.4f} | {row['max_ci_width']:.4f} | "
                       f"{row['abs_returns']:.4f} | {row['realized_vol_30d']:.2f} | {row['regime']} |\n")

            f.write("\n")

        f.write("\n")

    # Key findings
    f.write("## Key Findings\n\n")

    # Regime distribution for widening
    if len(df_widening) > 0:
        regime_counts = df_widening['regime'].value_counts()
        f.write("### Widening Events by Regime\n\n")
        for regime, count in regime_counts.items():
            pct = 100 * count / len(df_widening)
            f.write(f"- **{regime}:** {count} events ({pct:.1f}%)\n")
        f.write("\n")

    # Regime distribution for narrowing
    if len(df_narrowing) > 0:
        regime_counts = df_narrowing['regime'].value_counts()
        f.write("### Narrowing Events by Regime\n\n")
        for regime, count in regime_counts.items():
            pct = 100 * count / len(df_narrowing)
            f.write(f"- **{regime}:** {count} events ({pct:.1f}%)\n")
        f.write("\n")

    # Average context features for widening vs narrowing
    if len(df_widening) > 0 and len(df_narrowing) > 0:
        f.write("### Average Context Features\n\n")
        f.write("| Event Type | Avg |Returns| | Avg RVol (%) | Avg Range |\n")
        f.write("|------------|---------------|--------------|------------|\n")
        f.write(f"| Widening | {df_widening['abs_returns'].mean():.4f} | "
               f"{df_widening['realized_vol_30d'].mean():.2f} | "
               f"{df_widening['range_ci_width'].mean():.4f} |\n")
        f.write(f"| Narrowing | {df_narrowing['abs_returns'].mean():.4f} | "
               f"{df_narrowing['realized_vol_30d'].mean():.2f} | "
               f"{df_narrowing['range_ci_width'].mean():.4f} |\n")
        f.write("\n")

    # Notes
    f.write("## Notes\n\n")
    f.write("- **Range CI Width:** max(CI width across H days) - min(CI width across H days)\n")
    f.write("- **Widening:** Large range indicates model uncertainty changes significantly across the forecast sequence\n")
    f.write("- **Narrowing:** Small range indicates stable model uncertainty throughout the sequence\n")
    f.write("- **RVol:** 30-day rolling realized volatility (annualized %)\n")
    f.write("- **|Ret|:** Absolute daily return at forecast starting date\n")
    f.write("\n")

print(f"✓ Saved summary report: {summary_file}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("EVENT IDENTIFICATION COMPLETE")
print("=" * 80)
print()
print(f"Summary report: {summary_file}")
print(f"  Widening events: {len(df_widening)}")
print(f"  Narrowing events: {len(df_narrowing)}")
print()
print("Review the markdown report for detailed analysis of extreme events.")
print()
