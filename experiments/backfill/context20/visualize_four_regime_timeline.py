"""
Visualize Four-Regime Timeline with Distinguishing Features

Creates a multi-panel time series visualization showing:
1. Where the 4 regime dates appear on the full timeline (2000-2023)
2. Feature values over time with the 4 dates highlighted

This complements the four-regime overlay plot by providing temporal context
and showing what features distinguish each regime.

**Four regimes:**
- Green (Normal): 2007-03-28 - Low Vol, Low CI
- Blue (High-Vol Low-CI): 2009-02-23 - Intelligent Confidence
- Orange (Low-Vol High-CI): 2007-10-09 - Pre-Crisis Detection
- Red (High-Vol High-CI): 2008-10-30 - Expected Crisis

Usage:
    python experiments/backfill/context20/visualize_four_regime_timeline.py

Output:
    results/vae_baseline/visualizations/comparison/four_regime_timeline.png
    results/vae_baseline/visualizations/comparison/four_regime_feature_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime


def load_all_periods_data():
    """
    Load CI width data from all periods (insample, gap, oos) and concatenate

    Returns:
        DataFrame with all dates and features
    """
    print("Loading CI width statistics from all periods...")

    ci_file = Path("results/vae_baseline/analysis/prior/sequence_ci_width_stats.npz")
    if not ci_file.exists():
        raise FileNotFoundError(f"CI width data not found: {ci_file}")

    ci_data = np.load(ci_file, allow_pickle=True)
    print(f"  ✓ Loaded: {ci_file}")
    print()

    # Extract H=30 data from all periods
    periods = ['insample', 'gap', 'oos']
    h = 30
    period_dfs = []

    for period in periods:
        prefix = f'{period}_h{h}'

        if f'{prefix}_indices' not in ci_data.files:
            print(f"  ⚠ Skipping {period} (no data)")
            continue

        # Extract data
        indices = ci_data[f'{prefix}_indices']
        dates = pd.to_datetime(ci_data[f'{prefix}_dates'])
        avg_ci = ci_data[f'{prefix}_avg_ci_width'][:, 2, 2]  # ATM 6M grid point

        # Features
        abs_returns = ci_data[f'{prefix}_abs_returns']
        realized_vol = ci_data[f'{prefix}_realized_vol_30d']
        skews = ci_data[f'{prefix}_skews']
        slopes = ci_data[f'{prefix}_slopes']
        atm_vol = ci_data[f'{prefix}_atm_vol']

        # Create DataFrame
        df_period = pd.DataFrame({
            'date': dates,
            'index': indices,
            'avg_ci_width': avg_ci,
            'abs_returns': abs_returns,
            'realized_vol_30d': realized_vol,
            'skews': skews,
            'slopes': slopes,
            'atm_vol': atm_vol,
            'period': period
        })

        period_dfs.append(df_period)
        print(f"  ✓ {period}: {len(df_period)} dates")

    # Concatenate all periods
    df = pd.concat(period_dfs, ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\nTotal data points: {len(df)} days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    return df


def find_target_date(target_date_str, df):
    """
    Find the row index for a specific target date

    Args:
        target_date_str: Date string 'YYYY-MM-DD'
        df: DataFrame with 'date' column

    Returns:
        int: Row index matching the target date
    """
    target_date = pd.Timestamp(target_date_str)

    for idx, row_date in enumerate(df['date']):
        if pd.Timestamp(row_date) == target_date:
            return idx

    raise ValueError(f"Date {target_date_str} not found in data")


def compute_z_scores(df, target_dates_info):
    """
    Compute z-scores for each feature at target dates

    Args:
        df: Full DataFrame with features
        target_dates_info: Dict with regime -> {date, idx, ...}

    Returns:
        Updated target_dates_info with z-scores
    """
    features = ['avg_ci_width', 'atm_vol', 'slopes', 'skews', 'abs_returns', 'realized_vol_30d']

    print("Computing z-scores for target dates...")

    for regime, info in target_dates_info.items():
        idx = info['idx']
        info['z_scores'] = {}

        for feat in features:
            full_mean = df[feat].mean()
            full_std = df[feat].std()
            value = df.loc[idx, feat]
            z_score = (value - full_mean) / full_std
            info['z_scores'][feat] = z_score

        print(f"  {regime} ({info['date']}):")
        for feat in features:
            print(f"    {feat}: z={info['z_scores'][feat]:.2f}")
        print()

    return target_dates_info


def create_timeline_figure(df, target_dates_info, output_path):
    """
    Create 5-panel timeline visualization with 4 regime dates highlighted

    Args:
        df: Full DataFrame with all dates and features
        target_dates_info: Dict with regime -> {date, idx, color, values}
        output_path: Where to save the figure
    """
    print("Creating 5-panel timeline visualization...")

    fig, axes = plt.subplots(5, 1, figsize=(20, 14), sharex=True)

    # Color scheme (matplotlib default colors)
    regime_colors = {
        'Green': '#2ca02c',
        'Blue': '#1f77b4',
        'Orange': '#ff7f0e',
        'Red': '#d62728'
    }

    # Extract target dates and colors for vertical lines
    target_dates_list = [info['date'] for info in target_dates_info.values()]
    target_dates_pd = [pd.Timestamp(d) for d in target_dates_list]

    # =========================================================================
    # Panel 1: CI Width
    # =========================================================================
    ax = axes[0]

    # Line plot
    ax.plot(df['date'], df['avg_ci_width'], linewidth=1, color='black', alpha=0.6)

    # 90th percentile threshold
    threshold_ci = np.percentile(df['avg_ci_width'], 90)
    ax.axhline(threshold_ci, color='gray', linestyle='--', alpha=0.5,
              label=f'90th Percentile ({threshold_ci:.4f})')

    # Mark target dates
    for regime, info in target_dates_info.items():
        idx = info['idx']
        color = regime_colors[regime]
        ax.scatter(df.loc[idx, 'date'], df.loc[idx, 'avg_ci_width'],
                  s=200, c=color, marker='o', edgecolors='black', linewidth=2,
                  label=f'{regime} ({info["date"]})', zorder=10)

    # Vertical lines at target dates
    for date_pd in target_dates_pd:
        ax.axvline(date_pd, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_ylabel('CI Width (p95 - p05)', fontsize=12, fontweight='bold')
    ax.set_title('Four-Regime Timeline: Dates and Distinguishing Features',
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: ATM Volatility
    # =========================================================================
    ax = axes[1]

    # Line plot
    ax.plot(df['date'], df['atm_vol'], linewidth=1, color='steelblue', alpha=0.7)

    # Threshold lines
    threshold_atm_low = 0.30
    threshold_atm_high = 0.40
    ax.axhline(threshold_atm_low, color='red', linestyle='--', alpha=0.5,
              label=f'Low Vol Threshold ({threshold_atm_low})')
    ax.axhline(threshold_atm_high, color='orange', linestyle='--', alpha=0.5,
              label=f'High Vol Threshold ({threshold_atm_high})')

    # Shaded regions
    ax.fill_between(df['date'], 0, threshold_atm_low, color='red', alpha=0.05)
    ax.fill_between(df['date'], threshold_atm_high, df['atm_vol'].max(),
                   color='orange', alpha=0.05)

    # Mark target dates
    for regime, info in target_dates_info.items():
        idx = info['idx']
        color = regime_colors[regime]
        ax.scatter(df.loc[idx, 'date'], df.loc[idx, 'atm_vol'],
                  s=200, c=color, marker='o', edgecolors='black', linewidth=2, zorder=10)

    # Vertical lines
    for date_pd in target_dates_pd:
        ax.axvline(date_pd, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_ylabel('ATM 6M Volatility', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Slopes (Term Structure)
    # =========================================================================
    ax = axes[2]

    # Line plot
    ax.plot(df['date'], df['slopes'], linewidth=1, color='green', alpha=0.7)

    # Zero line (flat term structure)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4, label='Flat Term Structure')

    # Mark target dates
    for regime, info in target_dates_info.items():
        idx = info['idx']
        color = regime_colors[regime]
        ax.scatter(df.loc[idx, 'date'], df.loc[idx, 'slopes'],
                  s=200, c=color, marker='o', edgecolors='black', linewidth=2, zorder=10)

    # Vertical lines
    for date_pd in target_dates_pd:
        ax.axvline(date_pd, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_ylabel('Slopes (Term Structure)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Realized Volatility 30d
    # =========================================================================
    ax = axes[3]

    # Line plot
    ax.plot(df['date'], df['realized_vol_30d'], linewidth=1, color='purple', alpha=0.7)

    # Mark target dates
    for regime, info in target_dates_info.items():
        idx = info['idx']
        color = regime_colors[regime]
        ax.scatter(df.loc[idx, 'date'], df.loc[idx, 'realized_vol_30d'],
                  s=200, c=color, marker='o', edgecolors='black', linewidth=2, zorder=10)

    # Vertical lines
    for date_pd in target_dates_pd:
        ax.axvline(date_pd, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_ylabel('Realized Vol (30d)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 5: Skews (Volatility Smile)
    # =========================================================================
    ax = axes[4]

    # Line plot
    ax.plot(df['date'], df['skews'], linewidth=1, color='darkorange', alpha=0.7)

    # Mark target dates
    for regime, info in target_dates_info.items():
        idx = info['idx']
        color = regime_colors[regime]
        ax.scatter(df.loc[idx, 'date'], df.loc[idx, 'skews'],
                  s=200, c=color, marker='o', edgecolors='black', linewidth=2, zorder=10)

    # Vertical lines
    for date_pd in target_dates_pd:
        ax.axvline(date_pd, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_ylabel('Skews', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved timeline figure: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    print()


def create_feature_summary_csv(df, target_dates_info, output_path):
    """
    Create CSV summary table with feature values and z-scores for each regime

    Args:
        df: Full DataFrame
        target_dates_info: Dict with regime information
        output_path: Where to save CSV
    """
    print("Creating feature summary CSV...")

    features = ['avg_ci_width', 'atm_vol', 'slopes', 'skews', 'abs_returns', 'realized_vol_30d']

    rows = []
    for regime in ['Green', 'Blue', 'Orange', 'Red']:
        info = target_dates_info[regime]
        idx = info['idx']

        row = {
            'Regime': regime,
            'Date': info['date'],
        }

        # Add actual values
        for feat in features:
            feat_name = feat.replace('_', ' ').title()
            row[feat_name] = df.loc[idx, feat]

        # Add z-scores
        for feat in features:
            feat_name = feat.replace('_', ' ').title() + ' (Z-Score)'
            row[feat_name] = info['z_scores'][feat]

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_path, index=False, float_format='%.4f')

    print(f"  ✓ Saved feature summary: {output_path}")
    print()


def main():
    """Main execution function"""
    print("=" * 80)
    print("FOUR-REGIME TIMELINE VISUALIZATION")
    print("=" * 80)
    print()

    # =========================================================================
    # Load data from all periods
    # =========================================================================
    df = load_all_periods_data()

    # =========================================================================
    # Define target dates and find them in data
    # =========================================================================
    target_dates = {
        'Green': '2007-03-28',      # Normal (Low-Vol Low-CI)
        'Blue': '2009-02-23',       # High-Vol Low-CI (Intelligent Confidence)
        'Orange': '2007-10-09',     # Low-Vol High-CI (Pre-Crisis Detection)
        'Red': '2008-10-30'         # High-Vol High-CI (Expected Crisis)
    }

    print("Finding target dates in data...")
    target_dates_info = {}

    for regime, date_str in target_dates.items():
        idx = find_target_date(date_str, df)
        target_dates_info[regime] = {
            'date': date_str,
            'idx': idx,
            'avg_ci_width': df.loc[idx, 'avg_ci_width'],
            'atm_vol': df.loc[idx, 'atm_vol']
        }
        print(f"  {regime} ({date_str}) → index {idx}")
        print(f"    CI width: {df.loc[idx, 'avg_ci_width']:.4f}")
        print(f"    ATM vol:  {df.loc[idx, 'atm_vol']:.3f}")

    print()

    # =========================================================================
    # Compute z-scores
    # =========================================================================
    target_dates_info = compute_z_scores(df, target_dates_info)

    # =========================================================================
    # Create visualizations
    # =========================================================================
    output_dir = Path("results/vae_baseline/visualizations/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timeline figure
    timeline_path = output_dir / "four_regime_timeline.png"
    create_timeline_figure(df, target_dates_info, timeline_path)

    # Feature summary CSV
    csv_path = output_dir / "four_regime_feature_summary.csv"
    create_feature_summary_csv(df, target_dates_info, csv_path)

    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print("Key insights:")
    print("  ✓ Timeline spans full data period (2000-2023)")
    print("  ✓ Four regime dates marked as large colored circles")
    print("  ✓ Features show distinct patterns for each regime:")
    print(f"    - Green (Normal): Baseline reference")
    print(f"    - Blue (High-Vol Low-CI): Intelligent confidence despite high vol")
    print(f"    - Orange (Low-Vol High-CI): Pre-crisis detection via unusual slopes")
    print(f"    - Red (High-Vol High-CI): Expected crisis behavior")
    print()


if __name__ == "__main__":
    main()
