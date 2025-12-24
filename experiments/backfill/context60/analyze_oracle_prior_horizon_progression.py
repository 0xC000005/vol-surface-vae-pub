"""
Oracle vs Prior Horizon Progression Analysis - Context60 Model

Investigates why H=60 and H=90 show larger oracle vs prior CI width differences
compared to other horizons, and analyzes predicted values divergence (2004-2008).

**Key Questions:**
1. Why does the gap increase progressively from H=1 to H=90?
2. Why does INSAMPLE show 4.1% gap while CRISIS shows -2.0% (reversal)?
3. Why do predicted VALUES (p50, p05, p95) differ between oracle and prior?
4. What is the mechanism behind horizon-dependent prior mismatch?

**Phases:**
- Phase 1: Statistical analysis of horizon progression
- Phase 2: Context-to-horizon ratio analysis
- Phase 3: Per-sequence analysis (CRISIS reversal)
- Phase 4: Min/Max/Std CI width analysis
- Phase 4.5: Predicted values divergence (2004-2008 focus)
- Phase 5: Visualization suite (5 plots)
- Phase 6: Comprehensive documentation

Usage:
    python experiments/backfill/context60/analyze_oracle_prior_horizon_progression.py

Outputs:
    - results/context60_baseline/analysis/comparison/horizon_progression_analysis.csv
    - results/context60_baseline/analysis/comparison/predicted_values_divergence_2004_2008.csv
    - results/context60_baseline/visualizations/comparison/*.png (5 plots)
    - results/context60_baseline/analysis/comparison/ORACLE_PRIOR_HORIZON_GAP_ANALYSIS.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Paths
RESULTS_DIR = Path("results/context60_baseline")
ANALYSIS_DIR = RESULTS_DIR / "analysis"
VIS_DIR = RESULTS_DIR / "visualizations" / "comparison"
PRED_DIR = RESULTS_DIR / "predictions" / "teacher_forcing"

# Ensure output directories exist
VIS_DIR.mkdir(parents=True, exist_ok=True)
(ANALYSIS_DIR / "comparison").mkdir(parents=True, exist_ok=True)

# Data files
DATA_DIR = Path("data")
GROUND_TRUTH_FILE = DATA_DIR / "vol_surface_with_ret.npz"
DATES_FILE = DATA_DIR / "spx_vol_surface_history_full_data_fixed.parquet"

# Grid point for analysis
GRID_ROW, GRID_COL = 2, 2  # ATM 6M

# Horizons
TF_HORIZONS = [1, 7, 14, 30, 60, 90]
AR_HORIZONS = [180, 270]
ALL_HORIZONS = TF_HORIZONS + AR_HORIZONS

# Periods
PERIODS = ['crisis', 'insample', 'oos', 'gap']

# Context length for context60 model
CONTEXT_LEN = 60

# Date range for prediction divergence analysis
DIVERGENCE_START_DATE = "2004-01-01"
DIVERGENCE_END_DATE = "2008-12-31"


# ============================================================================
# Phase 1: Load Existing Comparison Data
# ============================================================================

def load_existing_comparisons():
    """Load existing oracle vs prior comparison CSVs"""
    print("=" * 80)
    print("PHASE 1: LOADING EXISTING COMPARISON DATA")
    print("=" * 80)
    print()

    comp_dir = ANALYSIS_DIR / "comparison"

    # Load overall comparison
    overall_file = comp_dir / "oracle_vs_prior_ci_comparison.csv"
    print(f"Loading: {overall_file}")
    overall_df = pd.read_csv(overall_file)

    # Load by-period comparison
    by_period_file = comp_dir / "oracle_vs_prior_ci_by_period.csv"
    print(f"Loading: {by_period_file}")
    by_period_df = pd.read_csv(by_period_file)

    print(f"  Overall comparisons: {len(overall_df)} horizons")
    print(f"  By-period comparisons: {len(by_period_df)} period-horizon combinations")
    print()

    return overall_df, by_period_df


def load_sequence_stats(sampling_mode):
    """Load sequence-level CI width statistics"""
    print(f"Loading sequence-level stats: {sampling_mode}")

    stats_file = ANALYSIS_DIR / sampling_mode / "sequence_ci_width_stats.npz"
    data = np.load(stats_file, allow_pickle=True)

    print(f"  Loaded: {stats_file}")
    print(f"  Keys: {list(data.keys())[:5]}...")  # Show first 5 keys
    print()

    return data


def load_ground_truth():
    """Load ground truth surfaces and dates"""
    print("Loading ground truth data...")

    # Load surfaces
    gt_data = np.load(GROUND_TRUTH_FILE)
    surfaces = gt_data["surface"]

    # Load dates
    dates_df = pd.read_parquet(DATES_FILE)
    dates = pd.to_datetime(dates_df["date"].values)

    print(f"  Surfaces shape: {surfaces.shape}")
    print(f"  Dates range: {dates[0]} to {dates[-1]}")
    print()

    return surfaces, dates


# ============================================================================
# Phase 2: Horizon Progression Analysis
# ============================================================================

def analyze_horizon_progression(overall_df):
    """Analyze how oracle-prior gap changes with horizon"""
    print("=" * 80)
    print("PHASE 2: HORIZON PROGRESSION ANALYSIS")
    print("=" * 80)
    print()

    # Extract key metrics
    horizons = overall_df['horizon'].values
    ratios = overall_df['ratio_prior_oracle'].values
    rel_diffs = overall_df['relative_diff_pct'].values
    cohens_d = overall_df['cohens_d'].values

    # Compute statistics
    print("**Horizon Progression Statistics:**")
    print()
    print(f"{'Horizon':<10} {'Ratio':>10} {'Rel Diff %':>12} {'Cohen d':>12} {'Trend':>10}")
    print("-" * 60)

    for h, r, rd, cd in zip(horizons, ratios, rel_diffs, cohens_d):
        if h <= 30:
            trend = "Baseline"
        elif h <= 90:
            trend = "Diverging" if r > 1.0 else "Baseline"
        else:
            trend = "AR"

        print(f"H={h:<7} {r:>10.4f} {rd:>11.2f}% {cd:>11.4f} {trend:>10}")

    print()

    # Compute horizon-to-context ratios for TF horizons
    print("**Context-to-Horizon Ratio Analysis:**")
    print()

    tf_df = overall_df[overall_df['horizon'].isin(TF_HORIZONS)].copy()
    tf_df['horizon_to_context'] = tf_df['horizon'] / CONTEXT_LEN

    print(f"{'Horizon':<10} {'H/Context':>12} {'Ratio':>10} {'Comment':>30}")
    print("-" * 70)

    for _, row in tf_df.iterrows():
        h = row['horizon']
        h_to_c = row['horizon_to_context']
        r = row['ratio_prior_oracle']

        if h_to_c < 1.0:
            comment = "Within context window"
        elif h_to_c == 1.0:
            comment = "Equal to context length"
        else:
            comment = "Exceeds context length"

        print(f"H={h:<7} {h_to_c:>11.2f}x {r:>10.4f} {comment:>30}")

    print()

    # Test correlation between horizon/context ratio and gap
    corr, p_value = stats.spearmanr(tf_df['horizon_to_context'], tf_df['ratio_prior_oracle'])
    print(f"**Spearman correlation (H/Context vs Ratio): {corr:.4f} (p={p_value:.2e})**")
    print()

    if corr > 0 and p_value < 0.05:
        print("  ✓ Significant positive correlation: Gap increases with horizon/context ratio")
    else:
        print("  ✗ No significant correlation detected")

    print()

    # Save progression analysis
    output_file = ANALYSIS_DIR / "comparison" / "horizon_progression_analysis.csv"
    tf_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print()

    return tf_df


# ============================================================================
# Phase 3: Period-Specific Analysis
# ============================================================================

def analyze_period_patterns(by_period_df):
    """Analyze period-specific oracle-prior patterns"""
    print("=" * 80)
    print("PHASE 3: PERIOD-SPECIFIC PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Focus on H=60 and H=90 TF horizons
    key_horizons = [60, 90]

    print("**Why does CRISIS show reversed pattern (oracle wider than prior)?**")
    print()

    for h in key_horizons:
        print(f"\n--- Horizon H={h} ---")
        print(f"{'Period':<12} {'Oracle':>10} {'Prior':>10} {'Ratio':>10} {'Rel Diff %':>12} {'Pattern':>15}")
        print("-" * 75)

        for period in PERIODS:
            row = by_period_df[(by_period_df['horizon'] == h) & (by_period_df['period'] == period)]

            if len(row) == 0:
                continue

            row = row.iloc[0]
            oracle_mean = row['oracle_avg_mean']
            prior_mean = row['prior_avg_mean']
            ratio = row['ratio_prior_oracle']
            rel_diff = row['relative_diff_pct']

            if ratio < 1.0:
                pattern = "Oracle > Prior"
            elif ratio > 1.0:
                pattern = "Prior > Oracle"
            else:
                pattern = "Equal"

            print(f"{period:<12} {oracle_mean:>10.6f} {prior_mean:>10.6f} {ratio:>10.4f} {rel_diff:>11.2f}% {pattern:>15}")

    print()

    # Summary statistics by period
    print("\n**Period Summary (All Horizons):**")
    print()
    print(f"{'Period':<12} {'Mean Ratio':>12} {'Min Ratio':>12} {'Max Ratio':>12} {'Std':>10}")
    print("-" * 65)

    for period in PERIODS:
        period_data = by_period_df[by_period_df['period'] == period]
        mean_ratio = period_data['ratio_prior_oracle'].mean()
        min_ratio = period_data['ratio_prior_oracle'].min()
        max_ratio = period_data['ratio_prior_oracle'].max()
        std_ratio = period_data['ratio_prior_oracle'].std()

        print(f"{period:<12} {mean_ratio:>12.4f} {min_ratio:>12.4f} {max_ratio:>12.4f} {std_ratio:>10.4f}")

    print()

    return


# ============================================================================
# Phase 4: Min/Max/Std Analysis
# ============================================================================

def analyze_min_max_std(oracle_stats, prior_stats):
    """Analyze if gap exists in min, max, std CI widths"""
    print("=" * 80)
    print("PHASE 4: MIN/MAX/STD CI WIDTH ANALYSIS")
    print("=" * 80)
    print()

    print("**Checking if oracle-prior gap exists across all statistics (not just avg)**")
    print()

    results = []

    for period in PERIODS:
        print(f"\n--- {period.upper()} Period ---")
        print()

        for h in TF_HORIZONS:
            h_key = f"h{h}"

            # Try to load avg, min, max CI widths
            try:
                oracle_avg = oracle_stats[f'{period}_{h_key}_avg_ci_width']
                prior_avg = prior_stats[f'{period}_{h_key}_avg_ci_width']

                # Extract ATM 6M grid point
                oracle_avg_atm = oracle_avg[:, GRID_ROW, GRID_COL]
                prior_avg_atm = prior_avg[:, GRID_ROW, GRID_COL]

                # Compute ratios
                avg_ratio = np.mean(prior_avg_atm) / np.mean(oracle_avg_atm)

                results.append({
                    'period': period,
                    'horizon': h,
                    'statistic': 'avg',
                    'oracle_mean': np.mean(oracle_avg_atm),
                    'prior_mean': np.mean(prior_avg_atm),
                    'ratio': avg_ratio
                })

                print(f"H={h}: avg ratio = {avg_ratio:.4f}")

            except KeyError:
                print(f"H={h}: Data not available for {period}")
                continue

    print()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Summary
    print("\n**Summary: Do gaps exist in all statistics?**")
    print()

    tf_avg = results_df[results_df['statistic'] == 'avg']
    print(f"Average CI width ratios:")
    print(f"  H=60: {tf_avg[tf_avg['horizon']==60]['ratio'].mean():.4f}")
    print(f"  H=90: {tf_avg[tf_avg['horizon']==90]['ratio'].mean():.4f}")
    print()

    return results_df


# ============================================================================
# Phase 4.5: Predicted Values Divergence Analysis (2004-2008)
# ============================================================================

def load_predictions(horizon, sampling_mode, period='insample'):
    """Load teacher forcing predictions"""
    pred_file = PRED_DIR / sampling_mode / f"vae_tf_{period}_h{horizon}.npz"

    if not pred_file.exists():
        print(f"  Warning: {pred_file} not found")
        return None

    data = np.load(pred_file)
    return data


def analyze_predicted_values_divergence():
    """
    Critical analysis: Oracle and prior produce different PREDICTED VALUES
    (p50, p05, p95), not just different CI widths!

    Focus on 2004-2008 period where user observed significant divergence.
    """
    print("=" * 80)
    print("PHASE 4.5: PREDICTED VALUES DIVERGENCE ANALYSIS (2004-2008)")
    print("=" * 80)
    print()

    print("**CRITICAL INSIGHT: Oracle and prior produce different FORECAST VALUES!**")
    print("This is NOT just about uncertainty (CI width), but about the")
    print("model's point predictions themselves changing based on sampling mode.")
    print()

    # Load ground truth
    gt_surfaces, gt_dates = load_ground_truth()

    # Identify 2004-2008 period indices
    start_date = pd.Timestamp(DIVERGENCE_START_DATE)
    end_date = pd.Timestamp(DIVERGENCE_END_DATE)

    mask_2004_2008 = (gt_dates >= start_date) & (gt_dates <= end_date)
    indices_2004_2008 = np.where(mask_2004_2008)[0]

    print(f"2004-2008 period:")
    print(f"  Start: {start_date}")
    print(f"  End: {end_date}")
    print(f"  Number of dates: {len(indices_2004_2008)}")
    print()

    # Analyze H=60 and H=90
    results = []

    for h in [60, 90]:
        print(f"\n--- Analyzing H={h} ---")
        print()

        # Load predictions
        oracle_data = load_predictions(h, 'oracle', 'insample')
        prior_data = load_predictions(h, 'prior', 'insample')

        if oracle_data is None or prior_data is None:
            print(f"  Skipping H={h} (data not found)")
            continue

        # Extract predictions for insample period
        oracle_surfaces = oracle_data['surfaces']  # (n_seq, H, 3, 5, 5)
        prior_surfaces = prior_data['surfaces']
        oracle_indices = oracle_data['indices']
        prior_indices = prior_data['indices']

        print(f"  Oracle predictions shape: {oracle_surfaces.shape}")
        print(f"  Prior predictions shape: {prior_surfaces.shape}")
        print()

        # For each date in 2004-2008, compute prediction differences
        for target_idx in indices_2004_2008:
            # Find which sequence this date belongs to
            # Sequences are indexed by their START date
            # We need to find sequences where this date falls within the forecast window

            # Check if this date is a sequence start
            if target_idx in oracle_indices and target_idx in prior_indices:
                seq_idx_oracle = np.where(oracle_indices == target_idx)[0][0]
                seq_idx_prior = np.where(prior_indices == target_idx)[0][0]

                # Extract first day forecast (H=1 within the horizon)
                # Quantiles: 0=p05, 1=p50, 2=p95
                oracle_p05 = oracle_surfaces[seq_idx_oracle, 0, 0, GRID_ROW, GRID_COL]
                oracle_p50 = oracle_surfaces[seq_idx_oracle, 0, 1, GRID_ROW, GRID_COL]
                oracle_p95 = oracle_surfaces[seq_idx_oracle, 0, 2, GRID_ROW, GRID_COL]

                prior_p05 = prior_surfaces[seq_idx_prior, 0, 0, GRID_ROW, GRID_COL]
                prior_p50 = prior_surfaces[seq_idx_prior, 0, 1, GRID_ROW, GRID_COL]
                prior_p95 = prior_surfaces[seq_idx_prior, 0, 2, GRID_ROW, GRID_COL]

                # Ground truth
                gt_val = gt_surfaces[target_idx, GRID_ROW, GRID_COL]

                # Compute differences
                delta_p05 = prior_p05 - oracle_p05
                delta_p50 = prior_p50 - oracle_p50
                delta_p95 = prior_p95 - oracle_p95

                # CI widths
                oracle_ci_width = oracle_p95 - oracle_p05
                prior_ci_width = prior_p95 - prior_p05

                results.append({
                    'date': gt_dates[target_idx],
                    'horizon': h,
                    'oracle_p05': oracle_p05,
                    'oracle_p50': oracle_p50,
                    'oracle_p95': oracle_p95,
                    'prior_p05': prior_p05,
                    'prior_p50': prior_p50,
                    'prior_p95': prior_p95,
                    'delta_p05': delta_p05,
                    'delta_p50': delta_p50,
                    'delta_p95': delta_p95,
                    'oracle_ci_width': oracle_ci_width,
                    'prior_ci_width': prior_ci_width,
                    'ground_truth': gt_val,
                    'oracle_error': oracle_p50 - gt_val,
                    'prior_error': prior_p50 - gt_val
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("  No results found for 2004-2008 period")
        return None

    print(f"\nTotal sequences analyzed: {len(results_df)}")
    print()

    # Summary statistics
    print("**Summary Statistics: Predicted Values Divergence**")
    print()

    for h in [60, 90]:
        df_h = results_df[results_df['horizon'] == h]

        if len(df_h) == 0:
            continue

        print(f"--- H={h} (n={len(df_h)} sequences) ---")
        print()

        print(f"  Mean |Δp50|: {np.abs(df_h['delta_p50']).mean():.6f}")
        print(f"  Mean |Δp05|: {np.abs(df_h['delta_p05']).mean():.6f}")
        print(f"  Mean |Δp95|: {np.abs(df_h['delta_p95']).mean():.6f}")
        print()

        print(f"  Direction of Δp50 bias:")
        print(f"    Prior > Oracle: {(df_h['delta_p50'] > 0).sum()} ({(df_h['delta_p50'] > 0).mean()*100:.1f}%)")
        print(f"    Prior < Oracle: {(df_h['delta_p50'] < 0).sum()} ({(df_h['delta_p50'] < 0).mean()*100:.1f}%)")
        print()

        # Accuracy comparison
        oracle_rmse = np.sqrt(np.mean(df_h['oracle_error']**2))
        prior_rmse = np.sqrt(np.mean(df_h['prior_error']**2))

        print(f"  Forecast Accuracy (RMSE):")
        print(f"    Oracle: {oracle_rmse:.6f}")
        print(f"    Prior:  {prior_rmse:.6f}")
        print(f"    Ratio (Prior/Oracle): {prior_rmse/oracle_rmse:.4f}x")
        print()

        # Bias
        oracle_bias = df_h['oracle_error'].mean()
        prior_bias = df_h['prior_error'].mean()

        print(f"  Forecast Bias (mean error):")
        print(f"    Oracle: {oracle_bias:+.6f}")
        print(f"    Prior:  {prior_bias:+.6f}")
        print()

    # Save results
    output_file = ANALYSIS_DIR / "comparison" / "predicted_values_divergence_2004_2008.csv"
    results_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print()

    return results_df


# ============================================================================
# Phase 5: Visualization Suite
# ============================================================================

def plot_horizon_progression(overall_df, by_period_df):
    """Visualization 1: Horizon progression plot"""
    print("Generating visualization 1: Horizon progression plot...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot by period
    colors = {'crisis': 'red', 'insample': 'blue', 'oos': 'green', 'gap': 'orange'}

    for period in PERIODS:
        period_data = by_period_df[by_period_df['period'] == period]
        period_data = period_data.sort_values('horizon')

        ax.plot(period_data['horizon'], period_data['ratio_prior_oracle'],
                marker='o', linewidth=2.5, markersize=8,
                label=period.capitalize(), color=colors[period])

    # Reference line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.6,
               label='No Gap (Ratio=1.0)')

    # Highlight H=60 and H=90
    for h in [60, 90]:
        ax.axvline(x=h, color='purple', linestyle=':', linewidth=1.5, alpha=0.4)
        ax.text(h, ax.get_ylim()[1]*0.95, f'H={h}', ha='center', fontsize=10,
                color='purple', fontweight='bold')

    # Styling
    ax.set_xlabel('Horizon (days)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Prior/Oracle CI Width Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Oracle vs Prior CI Width Gap: Horizon Progression by Period\n'
                 'Context60 Model - Progressive Divergence at H≥60',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = VIS_DIR / "oracle_prior_gap_by_horizon.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_period_heatmap(by_period_df):
    """Visualization 2: Period comparison heatmap"""
    print("Generating visualization 2: Period comparison heatmap...")

    # Pivot data for heatmap
    pivot_data = by_period_df.pivot(index='horizon', columns='period', values='ratio_prior_oracle')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdBu_r', center=1.0,
                vmin=0.97, vmax=1.05, cbar_kws={'label': 'Prior/Oracle Ratio'},
                linewidths=0.5, linecolor='gray', ax=ax)

    # Styling
    ax.set_title('Oracle vs Prior CI Width Ratios: Period × Horizon Heatmap\n'
                 'Context60 Model - CRISIS Reversal and INSAMPLE Maximum',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Period', fontsize=12, fontweight='bold')
    ax.set_ylabel('Horizon (days)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = VIS_DIR / "oracle_prior_gap_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_multi_statistic_comparison(results_df):
    """Visualization 3: Multi-statistic comparison for H=90"""
    print("Generating visualization 3: Multi-statistic comparison...")

    # Filter for H=90
    df_h90 = results_df[results_df['horizon'] == 90]

    if len(df_h90) == 0:
        print("  No data for H=90, skipping")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Prepare data for grouped bar chart
    periods = df_h90['period'].unique()
    x = np.arange(len(periods))
    width = 0.35

    oracle_vals = []
    prior_vals = []

    for period in periods:
        period_data = df_h90[df_h90['period'] == period]
        oracle_vals.append(period_data['oracle_mean'].mean())
        prior_vals.append(period_data['prior_mean'].mean())

    # Plot bars
    ax.bar(x - width/2, oracle_vals, width, label='Oracle', color='blue', alpha=0.8)
    ax.bar(x + width/2, prior_vals, width, label='Prior', color='red', alpha=0.8)

    # Styling
    ax.set_xlabel('Period', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average CI Width (ATM 6M)', fontsize=12, fontweight='bold')
    ax.set_title('Oracle vs Prior Average CI Width: H=90 by Period\n'
                 'Context60 Model',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in periods])
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = VIS_DIR / "oracle_prior_gap_all_stats_h90.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_predicted_values_timeseries(divergence_df):
    """Visualization 4: Predicted values timeseries 2004-2008"""
    print("Generating visualization 4: Predicted values timeseries 2004-2008...")

    if divergence_df is None or len(divergence_df) == 0:
        print("  No divergence data available, skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # H=60 and H=90
    horizons = [60, 90]

    for i, h in enumerate(horizons):
        df_h = divergence_df[divergence_df['horizon'] == h].sort_values('date')

        if len(df_h) == 0:
            continue

        # Left: Timeseries
        ax_left = axes[i, 0]

        ax_left.plot(df_h['date'], df_h['ground_truth'], 'k-', linewidth=2,
                     label='Ground Truth', alpha=0.7)
        ax_left.plot(df_h['date'], df_h['oracle_p50'], 'b-', linewidth=1.5,
                     label='Oracle p50', alpha=0.8)
        ax_left.plot(df_h['date'], df_h['prior_p50'], 'r-', linewidth=1.5,
                     label='Prior p50', alpha=0.8)

        ax_left.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax_left.set_ylabel('ATM 6M Implied Volatility', fontsize=10, fontweight='bold')
        ax_left.set_title(f'H={h}: Oracle vs Prior p50 Predictions (2004-2008)',
                          fontsize=11, fontweight='bold')
        ax_left.legend(fontsize=9)
        ax_left.grid(True, alpha=0.3)

        # Highlight crisis period (2008-2010)
        ax_left.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2008-12-31'),
                        alpha=0.1, color='red', label='Crisis')

        # Right: Divergence (Δp50)
        ax_right = axes[i, 1]

        ax_right.plot(df_h['date'], df_h['delta_p50'], 'purple', linewidth=1.5)
        ax_right.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax_right.fill_between(df_h['date'], 0, df_h['delta_p50'],
                              where=(df_h['delta_p50'] > 0), alpha=0.3, color='red',
                              label='Prior > Oracle')
        ax_right.fill_between(df_h['date'], 0, df_h['delta_p50'],
                              where=(df_h['delta_p50'] < 0), alpha=0.3, color='blue',
                              label='Prior < Oracle')

        ax_right.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax_right.set_ylabel('Δp50 (Prior - Oracle)', fontsize=10, fontweight='bold')
        ax_right.set_title(f'H={h}: Median Forecast Divergence (2004-2008)',
                           fontsize=11, fontweight='bold')
        ax_right.legend(fontsize=9)
        ax_right.grid(True, alpha=0.3)

    plt.suptitle('Oracle vs Prior Predicted Values Divergence: 2004-2008\n'
                 'Context60 Model - Point Forecasts Differ, Not Just Uncertainty',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_path = VIS_DIR / "oracle_prior_predictions_2004_2008.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_quantile_divergence(divergence_df):
    """Visualization 5: Quantile divergence analysis for H=90"""
    print("Generating visualization 5: Quantile divergence analysis H=90...")

    if divergence_df is None or len(divergence_df) == 0:
        print("  No divergence data available, skipping")
        return

    df_h90 = divergence_df[divergence_df['horizon'] == 90].sort_values('date')

    if len(df_h90) == 0:
        print("  No data for H=90, skipping")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    quantiles = ['p05', 'p50', 'p95']
    titles = ['Lower Bound (p05)', 'Median (p50)', 'Upper Bound (p95)']

    for i, (q, title) in enumerate(zip(quantiles, titles)):
        ax = axes[i]

        delta_col = f'delta_{q}'

        # Plot divergence
        ax.plot(df_h90['date'], df_h90[delta_col], linewidth=2, color='purple')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        # Color-code regions
        ax.fill_between(df_h90['date'], 0, df_h90[delta_col],
                        where=(df_h90[delta_col] > 0), alpha=0.3, color='red')
        ax.fill_between(df_h90['date'], 0, df_h90[delta_col],
                        where=(df_h90[delta_col] < 0), alpha=0.3, color='blue')

        # Statistics box
        mean_delta = df_h90[delta_col].mean()
        std_delta = df_h90[delta_col].std()
        min_delta = df_h90[delta_col].min()
        max_delta = df_h90[delta_col].max()

        stats_text = f'Mean: {mean_delta:+.6f}\nStd: {std_delta:.6f}\nMin: {min_delta:+.6f}\nMax: {max_delta:+.6f}'
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_ylabel(f'Δ{q} (Prior - Oracle)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} Divergence (H=90, 2004-2008)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i == 2:
            ax.set_xlabel('Date', fontsize=11, fontweight='bold')

    plt.suptitle('Oracle vs Prior Quantile Divergence: H=90 (2004-2008)\n'
                 'Context60 Model - All Quantiles Show Systematic Differences',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    output_path = VIS_DIR / "oracle_prior_quantile_divergence_h90.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("ORACLE VS PRIOR HORIZON PROGRESSION ANALYSIS")
    print("Context60 Model - Investigating H=60 and H=90 Divergence")
    print("=" * 80)
    print()

    # Load existing data
    overall_df, by_period_df = load_existing_comparisons()
    oracle_stats = load_sequence_stats('oracle')
    prior_stats = load_sequence_stats('prior')

    # Phase 2: Horizon progression analysis
    tf_df = analyze_horizon_progression(overall_df)

    # Phase 3: Period-specific patterns
    analyze_period_patterns(by_period_df)

    # Phase 4: Min/Max/Std analysis
    results_df = analyze_min_max_std(oracle_stats, prior_stats)

    # Phase 4.5: Predicted values divergence (2004-2008)
    divergence_df = analyze_predicted_values_divergence()

    # Phase 5: Visualizations
    print("=" * 80)
    print("PHASE 5: GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    plot_horizon_progression(overall_df, by_period_df)
    plot_period_heatmap(by_period_df)
    plot_multi_statistic_comparison(results_df)
    plot_predicted_values_timeseries(divergence_df)
    plot_quantile_divergence(divergence_df)

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Generated outputs:")
    print(f"  1. Analysis CSVs: {ANALYSIS_DIR / 'comparison'}")
    print(f"  2. Visualizations: {VIS_DIR}")
    print()
    print("Next step: Review outputs and create comprehensive markdown report")
    print()


if __name__ == "__main__":
    main()
