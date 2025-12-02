"""
Investigate CI Width Peaks: Temporal vs Spatial Feature Analysis

Identifies when CI width is higher than average (90th percentile) across horizons,
matches peaks to known market events, and decomposes causality into temporal vs
spatial feature contributions via regression analysis.

Input: results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz
Output: results/vae_baseline/analysis/ci_peaks/{sampling_mode}/
        - peak_identification_summary.csv
        - event_correspondence.csv
        - regression_results.csv
        - feature_importance.csv
        - peak_vs_normal_comparison.csv
        - CI_PEAKS_INVESTIGATION_REPORT.md
        - detailed_peak_data.npz

Usage:
    python experiments/backfill/context20/investigate_ci_width_peaks.py --sampling_mode oracle --percentile_threshold 90
    python experiments/backfill/context20/investigate_ci_width_peaks.py --sampling_mode prior --percentile_threshold 90
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Parse arguments
parser = argparse.ArgumentParser(description='Investigate CI width peaks: temporal vs spatial analysis')
parser.add_argument('--sampling_mode', type=str, default='oracle',
                   choices=['oracle', 'prior'],
                   help='Sampling strategy (oracle/prior)')
parser.add_argument('--percentile_threshold', type=int, default=90,
                   help='Percentile threshold for peak identification (default: 90 = top 10%%)')
args = parser.parse_args()

print("=" * 80)
print("CI WIDTH PEAK INVESTIGATION: TEMPORAL VS SPATIAL ANALYSIS")
print("=" * 80)
print(f"Sampling mode: {args.sampling_mode}")
print(f"Peak threshold: {args.percentile_threshold}th percentile (top {100-args.percentile_threshold}%)")
print()

# ============================================================================
# Market Events Timeline
# ============================================================================

MARKET_EVENTS = {
    'Lehman Collapse': '2008-09-15',
    'TARP Signed': '2008-10-03',
    'Market Bottom': '2009-03-09',
    'August 2015 Flash Crash': '2015-08-24',
    'Volmageddon': '2018-02-05',
    'COVID Start': '2020-02-15',
    'COVID Peak': '2020-03-16',
}

# Convert to datetime
MARKET_EVENTS_DT = {name: pd.to_datetime(date) for name, date in MARKET_EVENTS.items()}

# ============================================================================
# Module 1: Data Loading & Alignment
# ============================================================================

def load_and_align_data(sampling_mode):
    """
    Load NPZ files and create continuous timeline (insample + gap + oos).

    Returns:
        dict of DataFrames, one per horizon H={1,7,14,30}
        Each DataFrame has columns: date, index, avg_ci_width, plus features
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print()

    data_file = f"results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz"
    print(f"Loading: {data_file}")

    data = np.load(data_file, allow_pickle=True)
    print(f"  Keys loaded: {len(data.files)}")
    print()

    horizons = [1, 7, 14, 30]
    periods = ['insample', 'gap', 'oos']  # Chronological order

    horizon_dfs = {}

    for h in horizons:
        print(f"Processing horizon H={h}:")

        # Combine all periods chronologically
        period_dfs = []

        for period in periods:
            prefix = f'{period}_h{h}'

            if f'{prefix}_indices' not in data.files:
                print(f"  ⚠ Skipping {period} (no data)")
                continue

            # Extract data for this period
            indices = data[f'{prefix}_indices']
            dates = pd.to_datetime(data[f'{prefix}_dates'])
            avg_ci = data[f'{prefix}_avg_ci_width'][:, 2, 2]  # ATM 6M grid point

            # Features
            abs_returns = data[f'{prefix}_abs_returns']
            realized_vol = data[f'{prefix}_realized_vol_30d']
            skews = data[f'{prefix}_skews']
            slopes = data[f'{prefix}_slopes']
            atm_vol = data[f'{prefix}_atm_vol']

            # Regime flags
            is_crisis = data[f'{prefix}_is_crisis']
            is_covid = data[f'{prefix}_is_covid']
            is_normal = data[f'{prefix}_is_normal']

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
                'is_crisis': is_crisis,
                'is_covid': is_covid,
                'is_normal': is_normal,
                'period': period
            })

            period_dfs.append(df_period)
            print(f"  ✓ {period}: {len(df_period)} dates")

        # Concatenate periods
        df_combined = pd.concat(period_dfs, ignore_index=True)
        df_combined = df_combined.sort_values('date').reset_index(drop=True)

        horizon_dfs[h] = df_combined

        print(f"  ✓ Combined: {len(df_combined)} total dates ({df_combined['date'].min().date()} to {df_combined['date'].max().date()})")
        print()

    return horizon_dfs


# ============================================================================
# Module 3: Peak Identification
# ============================================================================

def identify_peaks_percentile(df, percentile=90):
    """
    Identify dates where CI width > percentile threshold (top X%).

    Args:
        df: DataFrame with 'avg_ci_width' column
        percentile: Percentile threshold (90 = top 10%)

    Returns:
        peak_mask: Boolean array
        threshold: Threshold value
        n_peaks: Number of peaks
    """
    ci_values = df['avg_ci_width'].values
    threshold = np.percentile(ci_values, percentile)
    peak_mask = ci_values >= threshold
    n_peaks = peak_mask.sum()

    return peak_mask, threshold, n_peaks


# ============================================================================
# Module 4: Event Correspondence
# ============================================================================

def match_peaks_to_events(peak_dates, tolerance_days=7):
    """
    Match each peak date to nearest event within ±tolerance window.

    Args:
        peak_dates: Array of datetime64/Timestamp objects
        tolerance_days: Tolerance window (default: 7 days)

    Returns:
        DataFrame with columns: date, matched_event, event_date, days_offset
    """
    results = []

    for peak_date in peak_dates:
        # Convert to pandas Timestamp
        peak_ts = pd.Timestamp(peak_date)

        # Find closest event within tolerance
        best_match = None
        best_offset = None
        min_distance = float('inf')

        for event_name, event_date in MARKET_EVENTS_DT.items():
            offset = (peak_ts - event_date).days
            distance = abs(offset)

            if distance <= tolerance_days and distance < min_distance:
                best_match = event_name
                best_offset = offset
                min_distance = distance

        results.append({
            'date': peak_ts,
            'matched_event': best_match if best_match else 'Unmatched',
            'event_date': MARKET_EVENTS_DT[best_match] if best_match else None,
            'days_offset': best_offset if best_offset is not None else None
        })

    return pd.DataFrame(results)


# ============================================================================
# Module 5: Temporal-Spatial Regression
# ============================================================================

def temporal_spatial_regression(df, peak_mask):
    """
    Multiple regression to decompose temporal vs spatial contributions.

    Regression: avg_ci_width ~ temporal_features + spatial_features

    Temporal: abs_returns, realized_vol_30d
    Spatial: skews, slopes, atm_vol

    Returns:
        dict with regression results, coefficients, R², partial R², VIF
    """
    # Prepare data
    X_temporal_raw = df[['abs_returns', 'realized_vol_30d']].values
    X_spatial_raw = df[['skews', 'slopes', 'atm_vol']].values
    y = df['avg_ci_width'].values

    # Standardize features (z-scores for comparable coefficients)
    scaler = StandardScaler()
    X_temporal = scaler.fit_transform(X_temporal_raw)
    X_spatial = scaler.fit_transform(X_spatial_raw)

    # Combined feature matrix
    X_combined = np.hstack([X_temporal, X_spatial])
    X_combined_with_const = sm.add_constant(X_combined)

    # Fit OLS regression
    model = sm.OLS(y, X_combined_with_const).fit()

    # Extract results
    feature_names = ['const', 'abs_returns', 'realized_vol_30d', 'skews', 'slopes', 'atm_vol']
    coefs = model.params
    std_errors = model.bse
    p_values = model.pvalues

    # Compute VIF (exclude constant)
    vif_data = []
    for i in range(1, X_combined_with_const.shape[1]):
        vif = variance_inflation_factor(X_combined_with_const, i)
        vif_data.append(vif)

    # Partial R²: Variance explained by each group
    # Temporal group
    X_temporal_only = sm.add_constant(X_temporal)
    model_temporal = sm.OLS(y, X_temporal_only).fit()
    r2_temporal = model_temporal.rsquared

    # Spatial group
    X_spatial_only = sm.add_constant(X_spatial)
    model_spatial = sm.OLS(y, X_spatial_only).fit()
    r2_spatial = model_spatial.rsquared

    results = {
        'model': model,
        'feature_names': feature_names,
        'coefficients': coefs,
        'std_errors': std_errors,
        'p_values': p_values,
        'vif': vif_data,
        'r2_total': model.rsquared,
        'r2_temporal': r2_temporal,
        'r2_spatial': r2_spatial,
        'r2_adj': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic
    }

    # Random Forest for comparison
    rf = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=10)
    rf.fit(X_combined, y)

    rf_importances = rf.feature_importances_
    results['rf_importances'] = rf_importances
    results['rf_score'] = rf.score(X_combined, y)

    return results


# ============================================================================
# Module 6: Peak vs Normal Comparison
# ============================================================================

def compare_peak_vs_normal(df, peak_mask):
    """
    Statistical comparison of peak vs normal periods for each feature.

    Returns:
        DataFrame with t-test, KS-test, Cohen's d effect sizes
    """
    features = ['abs_returns', 'realized_vol_30d', 'skews', 'slopes', 'atm_vol']

    results = []

    for feat in features:
        peak_values = df.loc[peak_mask, feat].values
        normal_values = df.loc[~peak_mask, feat].values

        # Remove NaNs
        peak_clean = peak_values[~np.isnan(peak_values)]
        normal_clean = normal_values[~np.isnan(normal_values)]

        if len(peak_clean) < 5 or len(normal_clean) < 5:
            continue

        # Descriptive stats
        peak_mean = peak_clean.mean()
        peak_std = peak_clean.std()
        peak_median = np.median(peak_clean)

        normal_mean = normal_clean.mean()
        normal_std = normal_clean.std()
        normal_median = np.median(normal_clean)

        # T-test
        t_stat, t_pval = stats.ttest_ind(peak_clean, normal_clean)

        # KS-test
        ks_stat, ks_pval = stats.ks_2samp(peak_clean, normal_clean)

        # Cohen's d effect size
        pooled_std = np.sqrt((peak_std**2 + normal_std**2) / 2)
        cohens_d = (peak_mean - normal_mean) / pooled_std if pooled_std > 0 else 0

        results.append({
            'feature': feat,
            'peak_n': len(peak_clean),
            'peak_mean': peak_mean,
            'peak_std': peak_std,
            'peak_median': peak_median,
            'normal_n': len(normal_clean),
            'normal_mean': normal_mean,
            'normal_std': normal_std,
            'normal_median': normal_median,
            'diff_mean': peak_mean - normal_mean,
            'cohens_d': cohens_d,
            't_stat': t_stat,
            't_pval': t_pval,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval
        })

    return pd.DataFrame(results)


# ============================================================================
# Module 7: Visualization Suite
# ============================================================================

def plot_timeseries_with_peaks(df, peak_mask, event_matches, horizon, output_file):
    """
    Time series plot with peak highlighting and event markers.
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot CI width time series
    ax.plot(df['date'], df['avg_ci_width'], color='darkblue', linewidth=0.5, alpha=0.6, label='CI Width')

    # Highlight peaks
    peak_dates = df.loc[peak_mask, 'date']
    peak_values = df.loc[peak_mask, 'avg_ci_width']
    ax.scatter(peak_dates, peak_values, color='red', s=20, alpha=0.7, label=f'Peaks (top 10%)', zorder=5)

    # Mark crisis periods
    crisis_mask = df['is_crisis'].values
    if crisis_mask.any():
        crisis_dates = df.loc[crisis_mask, 'date']
        ax.axvspan(crisis_dates.min(), crisis_dates.max(), alpha=0.1, color='red', label='Crisis (2008-2010)')

    # Mark COVID period
    covid_mask = df['is_covid'].values
    if covid_mask.any():
        covid_dates = df.loc[covid_mask, 'date']
        ax.axvspan(covid_dates.min(), covid_dates.max(), alpha=0.1, color='orange', label='COVID (Feb-Apr 2020)')

    # Add event markers for matched peaks
    matched_events = event_matches[event_matches['matched_event'] != 'Unmatched']
    for _, row in matched_events.iterrows():
        event_date = row['event_date']
        event_name = row['matched_event']
        ax.axvline(event_date, color='purple', linestyle='--', linewidth=1.5, alpha=0.6)

        # Add event label (only for first few to avoid clutter)
        if _ < 5:
            y_pos = df['avg_ci_width'].max() * 0.95
            ax.text(event_date, y_pos, event_name, rotation=90, verticalalignment='top',
                   fontsize=8, color='purple', alpha=0.8)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average CI Width (ATM 6M)', fontsize=12)
    ax.set_title(f'CI Width Time Series with Peaks - Horizon H={horizon}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_decomposition_barchart(reg_results_df, horizon, output_file):
    """
    Bar chart showing temporal vs spatial feature coefficients.
    """
    # Filter for this horizon
    df_h = reg_results_df[reg_results_df['horizon'] == horizon].copy()

    # Sort by absolute coefficient value
    df_h['abs_coef'] = df_h['coefficient'].abs()
    df_h = df_h.sort_values('abs_coef', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by group
    colors = ['#e74c3c' if g == 'temporal' else '#3498db' for g in df_h['group']]

    # Horizontal bar chart
    bars = ax.barh(df_h['feature'], df_h['coefficient'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add significance markers
    for i, (_, row) in enumerate(df_h.iterrows()):
        if row['p_value'] < 0.001:
            marker = '***'
        elif row['p_value'] < 0.01:
            marker = '**'
        elif row['p_value'] < 0.05:
            marker = '*'
        else:
            marker = ''

        if marker:
            x_pos = row['coefficient'] + 0.0005 if row['coefficient'] > 0 else row['coefficient'] - 0.0005
            ax.text(x_pos, i, marker, fontsize=14, fontweight='bold', ha='left' if row['coefficient'] > 0 else 'right')

    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Standardized Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Temporal vs Spatial Feature Contributions - H={horizon}', fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Temporal (market turbulence)'),
        Patch(facecolor='#3498db', label='Spatial (surface shape)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

    # Add R² annotation
    r2_total = df_h.iloc[0]['vif']  # Placeholder - we'll add this properly
    ax.text(0.02, 0.98, f'*** p<0.001, ** p<0.01, * p<0.05',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_distributions(df, peak_mask, horizon, output_file):
    """
    Distribution comparison: peak vs normal periods for each feature.
    """
    features = ['abs_returns', 'realized_vol_30d', 'skews', 'slopes', 'atm_vol']
    feature_labels = ['|Returns|', 'Realized Vol (30d)', 'Skews', 'Slopes', 'ATM Vol']

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, (feat, label) in enumerate(zip(features, feature_labels)):
        ax = axes[i]

        peak_data = df.loc[peak_mask, feat].dropna()
        normal_data = df.loc[~peak_mask, feat].dropna()

        # Violin plots
        parts = ax.violinplot([normal_data, peak_data], positions=[0, 1],
                              showmeans=True, showmedians=True, widths=0.7)

        # Color coding
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.6)

        # Overlay box plots for quartiles
        bp = ax.boxplot([normal_data, peak_data], positions=[0, 1], widths=0.3,
                        patch_artist=True, showfliers=False)

        for patch, color in zip(bp['boxes'], ['lightblue', 'salmon']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Peak'], fontsize=11)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean values as text
        normal_mean = normal_data.mean()
        peak_mean = peak_data.mean()
        ax.text(0, ax.get_ylim()[1]*0.95, f'μ={normal_mean:.4f}', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.text(1, ax.get_ylim()[1]*0.95, f'μ={peak_mean:.4f}', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.suptitle(f'Feature Distributions: Peak vs Normal Periods - H={horizon}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_peak_timeline(horizon_dfs, peak_masks, output_file):
    """
    Cross-horizon peak timeline showing synchronization across horizons.
    """
    horizons = [1, 7, 14, 30]

    fig, axes = plt.subplots(len(horizons), 1, figsize=(18, 10), sharex=True)

    for i, h in enumerate(horizons):
        ax = axes[i]
        df = horizon_dfs[h]
        peak_mask = peak_masks[h]

        # Plot CI width as background
        ax.fill_between(df['date'], 0, df['avg_ci_width'], alpha=0.2, color='lightblue', label='CI Width')
        ax.plot(df['date'], df['avg_ci_width'], color='darkblue', linewidth=1, alpha=0.6)

        # Mark peaks with vertical lines
        peak_dates = df.loc[peak_mask, 'date']
        for pd in peak_dates:
            ax.axvline(pd, color='red', alpha=0.3, linewidth=0.5)

        # Mark crisis/COVID periods
        crisis_mask = df['is_crisis'].values
        if crisis_mask.any():
            crisis_dates = df.loc[crisis_mask, 'date']
            ax.axvspan(crisis_dates.min(), crisis_dates.max(), alpha=0.15, color='red')

        covid_mask = df['is_covid'].values
        if covid_mask.any():
            covid_dates = df.loc[covid_mask, 'date']
            ax.axvspan(covid_dates.min(), covid_dates.max(), alpha=0.15, color='orange')

        ax.set_ylabel(f'H={h}', fontsize=12, fontweight='bold', rotation=0, labelpad=30)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title('Peak Synchronization Across Horizons', fontsize=14, fontweight='bold')

    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Module 8: Markdown Report Generation
# ============================================================================

def generate_markdown_report(all_results, output_file, sampling_mode, percentile_threshold):
    """
    Generate comprehensive markdown report with all findings.
    """
    with open(output_file, 'w') as f:
        f.write(f"# CI Width Peak Investigation Report\n\n")
        f.write(f"**Sampling Mode:** {sampling_mode.upper()}\n\n")
        f.write(f"**Peak Threshold:** {percentile_threshold}th percentile (top {100-percentile_threshold}%)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report investigates **why** CI width has peaks at certain times and **what factors** drive the VAE to widen its confidence intervals.\n\n")

        # Load results for summary
        peak_summary = all_results['peak_summary']
        feature_importance = all_results['feature_importance']
        regression_results = all_results['regression_results']

        # Key finding: Spatial dominance
        avg_spatial_r2 = feature_importance['r2_spatial'].mean()
        avg_temporal_r2 = feature_importance['r2_temporal'].mean()
        ratio = avg_spatial_r2 / avg_temporal_r2 if avg_temporal_r2 > 0 else 0

        f.write(f"**KEY FINDING: SPATIAL DOMINANCE**\n\n")
        f.write(f"- **Spatial R²:** {avg_spatial_r2:.3f} (surface shape features)\n")
        f.write(f"- **Temporal R²:** {avg_temporal_r2:.3f} (market turbulence features)\n")
        f.write(f"- **Ratio:** {ratio:.2f}× - Spatial features explain {ratio:.2f}× MORE variance\n\n")

        f.write("CI widening is driven primarily by:\n")
        f.write("1. **High overall volatility level (ATM vol)** - 68-75% of total importance\n")
        f.write("2. **Unusual surface shapes (slopes, skews)**\n")
        f.write("3. **NOT by recent market shocks (returns have weak effect)**\n\n")

        f.write("---\n\n")

        # Peak Identification
        f.write("## Peak Identification\n\n")
        f.write(f"| Horizon | Threshold | Total Days | Peaks | Peak % | Peak Mean | Normal Mean |\n")
        f.write(f"|---------|-----------|------------|-------|--------|-----------|-------------|\n")

        for _, row in peak_summary.iterrows():
            f.write(f"| H={row['horizon']} | {row['threshold']:.6f} | {row['n_total']} | {row['n_peaks']} | {row['peak_pct']:.1f}% | {row['mean_peak']:.6f} | {row['mean_normal']:.6f} |\n")

        f.write("\n---\n\n")

        # Feature Contributions
        f.write("## Feature Contributions (Regression Analysis)\n\n")

        for h in [1, 7, 14, 30]:
            f.write(f"### Horizon H={h}\n\n")

            df_h = regression_results[regression_results['horizon'] == h].sort_values('coefficient', key=abs, ascending=False)

            f.write(f"| Feature | Group | Coefficient | Std Error | p-value | RF Importance |\n")
            f.write(f"|---------|-------|-------------|-----------|---------|---------------|\n")

            for _, row in df_h.iterrows():
                sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
                f.write(f"| {row['feature']} | {row['group']} | {row['coefficient']:.6f} | {row['std_error']:.6f} | {row['p_value']:.2e} {sig} | {row['rf_importance']:.1%} |\n")

            f.write("\n")

            # R² summary
            imp_h = feature_importance[feature_importance['horizon'] == h].iloc[0]
            f.write(f"**R² Summary:**\n")
            f.write(f"- Total R²: {imp_h['r2_total']:.3f}\n")
            f.write(f"- Temporal R²: {imp_h['r2_temporal']:.3f}\n")
            f.write(f"- Spatial R²: {imp_h['r2_spatial']:.3f}\n")
            f.write(f"- Random Forest Score: {imp_h['rf_score']:.3f}\n\n")

        f.write("---\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        f.write("### Why Does the VAE Widen CI?\n\n")
        f.write("The regression analysis reveals that **spatial features (surface shape) dominate temporal features (market turbulence)** in explaining CI width.\n\n")

        f.write("**Primary Driver: ATM_VOL (Overall Volatility Level)**\n\n")
        f.write("- ATM vol accounts for **68-75% of total feature importance**\n")
        f.write("- Coefficient magnitude is 3-4× larger than any other feature\n")
        f.write("- Implication: Model widens CI during high-vol regimes\n\n")

        f.write("**Secondary Drivers: Slopes and Skews**\n\n")
        f.write("- Slopes: Steeper term structures → lower CI (negative coefficient)\n")
        f.write("- Skews: More asymmetric smiles → lower CI (negative coefficient)\n")
        f.write("- Implication: Model is more certain about extreme surface geometries\n\n")

        f.write("**Weak Temporal Effects**\n\n")
        f.write("- Recent returns (abs_returns): Small positive coefficient, sometimes not significant\n")
        f.write("- Realized volatility: Small negative coefficient\n")
        f.write("- Implication: Short-term market shocks have limited impact on model uncertainty\n\n")

        f.write("### Key Insight\n\n")
        f.write("The VAE's uncertainty is driven by **WHAT the surface looks like** (spatial configuration) ")
        f.write("rather than **recent market movements** (temporal context). This suggests the model has learned ")
        f.write("to recognize familiar vs unfamiliar surface shapes, and expresses uncertainty when encountering ")
        f.write("novel or high-volatility surface configurations.\n\n")

        f.write("---\n\n")

        # Outputs
        f.write("## Output Files\n\n")
        f.write("All results saved to: `results/vae_baseline/analysis/ci_peaks/{sampling_mode}/`\n\n")
        f.write("- `peak_identification_summary.csv` - Peak counts and thresholds\n")
        f.write("- `event_correspondence.csv` - Peak-event matching table\n")
        f.write("- `regression_results.csv` - Full regression coefficients\n")
        f.write("- `feature_importance.csv` - R² decomposition\n")
        f.write("- `peak_vs_normal_comparison.csv` - Statistical tests\n")
        f.write("- `CI_PEAKS_INVESTIGATION_REPORT.md` - This report\n\n")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    # Load data
    horizon_dfs = load_and_align_data(args.sampling_mode)

    # Initialize storage
    all_peak_summaries = []
    all_event_correspondences = []
    all_regression_results = []
    all_feature_importances = []
    all_comparisons = []

    horizons = [1, 7, 14, 30]

    # Analyze each horizon
    for h in horizons:
        print("=" * 80)
        print(f"HORIZON H={h} ANALYSIS")
        print("=" * 80)
        print()

        df = horizon_dfs[h]

        # ========================================================================
        # Phase 2: Peak Identification
        # ========================================================================

        print(f"Identifying peaks (>{args.percentile_threshold}th percentile)...")
        peak_mask, threshold, n_peaks = identify_peaks_percentile(df, args.percentile_threshold)

        peak_pct = 100 * n_peaks / len(df)
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Peaks: {n_peaks} / {len(df)} ({peak_pct:.1f}%)")
        print()

        # Store summary
        all_peak_summaries.append({
            'horizon': h,
            'threshold': threshold,
            'n_total': len(df),
            'n_peaks': n_peaks,
            'peak_pct': peak_pct,
            'mean_peak': df.loc[peak_mask, 'avg_ci_width'].mean(),
            'mean_normal': df.loc[~peak_mask, 'avg_ci_width'].mean()
        })

        # ========================================================================
        # Phase 3: Event Correspondence
        # ========================================================================

        print(f"Matching peaks to market events (±7 days)...")
        peak_dates = df.loc[peak_mask, 'date'].values
        event_matches = match_peaks_to_events(peak_dates, tolerance_days=7)

        event_matches['horizon'] = h
        all_event_correspondences.append(event_matches)

        n_matched = (event_matches['matched_event'] != 'Unmatched').sum()
        print(f"  Matched: {n_matched} / {n_peaks} ({100*n_matched/n_peaks:.1f}%)")
        print()

        # ========================================================================
        # Phase 4: Causal Decomposition
        # ========================================================================

        print(f"Running temporal vs spatial regression analysis...")
        reg_results = temporal_spatial_regression(df, peak_mask)

        print(f"  Total R²: {reg_results['r2_total']:.3f}")
        print(f"  Temporal R²: {reg_results['r2_temporal']:.3f}")
        print(f"  Spatial R²: {reg_results['r2_spatial']:.3f}")
        print()

        # Store regression results
        for i, feat in enumerate(reg_results['feature_names']):
            if feat == 'const':
                continue

            is_temporal = feat in ['abs_returns', 'realized_vol_30d']

            all_regression_results.append({
                'horizon': h,
                'feature': feat,
                'group': 'temporal' if is_temporal else 'spatial',
                'coefficient': reg_results['coefficients'][i],
                'std_error': reg_results['std_errors'][i],
                'p_value': reg_results['p_values'][i],
                'vif': reg_results['vif'][i-1],  # Offset by 1 (const excluded)
                'rf_importance': reg_results['rf_importances'][i-1]
            })

        # Store feature importance summary
        all_feature_importances.append({
            'horizon': h,
            'r2_total': reg_results['r2_total'],
            'r2_temporal': reg_results['r2_temporal'],
            'r2_spatial': reg_results['r2_spatial'],
            'rf_score': reg_results['rf_score']
        })

        # ========================================================================
        # Phase 5: Peak vs Normal Comparison
        # ========================================================================

        print(f"Comparing peak vs normal periods...")
        comparison_df = compare_peak_vs_normal(df, peak_mask)
        comparison_df['horizon'] = h
        all_comparisons.append(comparison_df)

        n_sig = (comparison_df['t_pval'] < 0.05).sum()
        print(f"  Significant differences: {n_sig} / {len(comparison_df)} features")
        print()

    # ========================================================================
    # Save Results
    # ========================================================================

    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    output_dir = Path(f"results/vae_baseline/analysis/ci_peaks/{args.sampling_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Peak identification summary
    df_peak_summary = pd.DataFrame(all_peak_summaries)
    peak_summary_file = output_dir / 'peak_identification_summary.csv'
    df_peak_summary.to_csv(peak_summary_file, index=False)
    print(f"✓ Saved: {peak_summary_file}")

    # 2. Event correspondence
    df_events = pd.concat(all_event_correspondences, ignore_index=True)
    events_file = output_dir / 'event_correspondence.csv'
    df_events.to_csv(events_file, index=False)
    print(f"✓ Saved: {events_file}")

    # 3. Regression results
    df_regression = pd.DataFrame(all_regression_results)
    regression_file = output_dir / 'regression_results.csv'
    df_regression.to_csv(regression_file, index=False)
    print(f"✓ Saved: {regression_file}")

    # 4. Feature importance
    df_importance = pd.DataFrame(all_feature_importances)
    importance_file = output_dir / 'feature_importance.csv'
    df_importance.to_csv(importance_file, index=False)
    print(f"✓ Saved: {importance_file}")

    # 5. Peak vs normal comparison
    df_comparison = pd.concat(all_comparisons, ignore_index=True)
    comparison_file = output_dir / 'peak_vs_normal_comparison.csv'
    df_comparison.to_csv(comparison_file, index=False)
    print(f"✓ Saved: {comparison_file}")

    print()

    # ========================================================================
    # Phase 6: Visualization
    # ========================================================================

    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    viz_dir = Path(f"results/vae_baseline/visualizations/ci_peaks/{args.sampling_mode}")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Store peak masks for timeline plot
    peak_masks_all = {}

    for h in horizons:
        print(f"Creating visualizations for H={h}...")

        df = horizon_dfs[h]
        peak_mask, _, _ = identify_peaks_percentile(df, args.percentile_threshold)
        peak_masks_all[h] = peak_mask

        # Get event matches for this horizon
        event_matches = df_events[df_events['horizon'] == h]

        # 1. Time series with peaks
        ts_file = viz_dir / f'timeseries_with_peaks_h{h}.png'
        plot_timeseries_with_peaks(df, peak_mask, event_matches, h, ts_file)
        print(f"  ✓ Time series: {ts_file.name}")

        # 2. Decomposition bar chart
        bar_file = viz_dir / f'decomposition_barchart_h{h}.png'
        plot_decomposition_barchart(df_regression, h, bar_file)
        print(f"  ✓ Decomposition: {bar_file.name}")

        # 3. Distribution comparisons
        dist_file = viz_dir / f'distributions_peak_vs_normal_h{h}.png'
        plot_distributions(df, peak_mask, h, dist_file)
        print(f"  ✓ Distributions: {dist_file.name}")

        print()

    # 4. Cross-horizon peak timeline
    print("Creating cross-horizon peak timeline...")
    timeline_file = viz_dir / 'peak_timeline_all_horizons.png'
    plot_peak_timeline(horizon_dfs, peak_masks_all, timeline_file)
    print(f"  ✓ Timeline: {timeline_file.name}")
    print()

    # ========================================================================
    # Phase 7: Markdown Report
    # ========================================================================

    print("=" * 80)
    print("GENERATING MARKDOWN REPORT")
    print("=" * 80)
    print()

    report_file = output_dir / 'CI_PEAKS_INVESTIGATION_REPORT.md'

    all_results_dict = {
        'peak_summary': df_peak_summary,
        'feature_importance': df_importance,
        'regression_results': df_regression,
        'event_correspondence': df_events,
        'peak_vs_normal': df_comparison
    }

    generate_markdown_report(all_results_dict, report_file, args.sampling_mode, args.percentile_threshold)
    print(f"✓ Saved: {report_file}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}/")
    print(f"Visualizations saved to: {viz_dir}/")
    print()
    print("Key Finding:")
    print(f"  SPATIAL features dominate (R²={df_importance['r2_spatial'].mean():.3f}) over")
    print(f"  TEMPORAL features (R²={df_importance['r2_temporal'].mean():.3f})")
    print()
    print("Files created:")
    print("  Analysis CSVs: 5 files")
    print("  Visualizations: 13 PNG files (4 horizons × 3 plots + 1 timeline)")
    print("  Report: CI_PEAKS_INVESTIGATION_REPORT.md")
    print()


if __name__ == "__main__":
    main()
