"""
Compare Oracle vs Prior Sampling CI Width Statistics

Compares confidence interval widths between oracle (posterior) and prior (realistic)
sampling strategies to quantify the effect of VAE prior mismatch on uncertainty
quantification.

Input:
    - results/vae_baseline/analysis/oracle/sequence_ci_width_stats.npz
    - results/vae_baseline/analysis/prior/sequence_ci_width_stats.npz

Output:
    - results/vae_baseline/analysis/oracle_vs_prior_comparison.csv
    - results/vae_baseline/visualizations/sequence_ci_width/oracle_vs_prior_comparison.png
    - results/vae_baseline/analysis/ORACLE_VS_PRIOR_COMPARISON_SUMMARY.md

Usage:
    python experiments/backfill/context20/compare_oracle_vs_prior_ci.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

print("=" * 80)
print("ORACLE VS PRIOR SAMPLING CI WIDTH COMPARISON")
print("=" * 80)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading oracle sampling statistics...")
oracle_data = np.load("results/vae_baseline/analysis/oracle/sequence_ci_width_stats.npz", allow_pickle=True)
print(f"  Loaded {len(oracle_data.files)} keys")

print("Loading prior sampling statistics...")
prior_data = np.load("results/vae_baseline/analysis/prior/sequence_ci_width_stats.npz", allow_pickle=True)
print(f"  Loaded {len(prior_data.files)} keys")
print()

# ============================================================================
# Define Analysis Parameters
# ============================================================================

periods = ['insample', 'crisis', 'oos', 'gap']
horizons = [1, 7, 14, 30]

# ATM 6-month (benchmark grid point)
m_idx, t_idx = 2, 2
grid_label = 'ATM 6M'

print(f"Analyzing grid point: {grid_label} (K/S=1.00, 6-month)")
print(f"Periods: {periods}")
print(f"Horizons: {horizons}")
print()

# ============================================================================
# Comparison Analysis
# ============================================================================

print("=" * 80)
print("COMPARING CI WIDTH STATISTICS")
print("=" * 80)
print()

comparison_results = []

for period in periods:
    for h in horizons:
        prefix = f'{period}_h{h}'

        # Check if data exists in both
        oracle_key = f'{prefix}_avg_ci_width'
        prior_key = f'{prefix}_avg_ci_width'

        if oracle_key not in oracle_data.files or prior_key not in prior_data.files:
            print(f"  ⚠ Skipping {period} H={h} (missing data)")
            continue

        # Extract average CI widths for the grid point
        oracle_avg = oracle_data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]
        prior_avg = prior_data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]

        # Also get min/max for completeness
        oracle_min = oracle_data[f'{prefix}_min_ci_width'][:, m_idx, t_idx]
        oracle_max = oracle_data[f'{prefix}_max_ci_width'][:, m_idx, t_idx]
        prior_min = prior_data[f'{prefix}_min_ci_width'][:, m_idx, t_idx]
        prior_max = prior_data[f'{prefix}_max_ci_width'][:, m_idx, t_idx]

        # Compute statistics
        n_samples = len(oracle_avg)

        oracle_avg_mean = oracle_avg.mean()
        prior_avg_mean = prior_avg.mean()
        ratio = prior_avg_mean / oracle_avg_mean if oracle_avg_mean > 0 else np.nan

        # Paired t-test (same dates)
        t_stat, p_value = stats.ttest_rel(prior_avg, oracle_avg)

        # Cohen's d effect size
        diff = prior_avg - oracle_avg
        pooled_std = np.sqrt((oracle_avg.std()**2 + prior_avg.std()**2) / 2)
        cohens_d = diff.mean() / pooled_std if pooled_std > 0 else np.nan

        result = {
            'period': period,
            'horizon': h,
            'n_samples': n_samples,
            'oracle_avg_mean': oracle_avg_mean,
            'oracle_avg_std': oracle_avg.std(),
            'oracle_min_mean': oracle_min.mean(),
            'oracle_max_mean': oracle_max.mean(),
            'prior_avg_mean': prior_avg_mean,
            'prior_avg_std': prior_avg.std(),
            'prior_min_mean': prior_min.mean(),
            'prior_max_mean': prior_max.mean(),
            'ratio_prior_oracle': ratio,
            'absolute_diff': prior_avg_mean - oracle_avg_mean,
            'relative_diff_pct': 100 * (prior_avg_mean - oracle_avg_mean) / oracle_avg_mean if oracle_avg_mean > 0 else np.nan,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }

        comparison_results.append(result)

        print(f"{period.upper():9s} H={h:2d}: "
              f"Oracle={oracle_avg_mean:.4f}, Prior={prior_avg_mean:.4f}, "
              f"Ratio={ratio:.2f}x, Δ={prior_avg_mean - oracle_avg_mean:+.4f} "
              f"(p={p_value:.2e})")

print()

# Convert to DataFrame
df_comparison = pd.DataFrame(comparison_results)

# Display summary statistics
if len(df_comparison) > 0:
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    print(f"Overall statistics across all period-horizon combinations:")
    print(f"  Mean prior/oracle ratio: {df_comparison['ratio_prior_oracle'].mean():.2f}x")
    print(f"  Median prior/oracle ratio: {df_comparison['ratio_prior_oracle'].median():.2f}x")
    print(f"  Range: [{df_comparison['ratio_prior_oracle'].min():.2f}x, {df_comparison['ratio_prior_oracle'].max():.2f}x]")
    print()

    print(f"Significant differences (p < 0.05): {df_comparison['significant'].sum()} / {len(df_comparison)} "
          f"({100 * df_comparison['significant'].sum() / len(df_comparison):.1f}%)")
    print()

    # By period
    print("Mean ratios by period:")
    for period in periods:
        period_subset = df_comparison[df_comparison['period'] == period]
        if len(period_subset) > 0:
            mean_ratio = period_subset['ratio_prior_oracle'].mean()
            print(f"  {period:9s}: {mean_ratio:.2f}x")
    print()

    # By horizon
    print("Mean ratios by horizon:")
    for h in horizons:
        horizon_subset = df_comparison[df_comparison['horizon'] == h]
        if len(horizon_subset) > 0:
            mean_ratio = horizon_subset['ratio_prior_oracle'].mean()
            print(f"  H={h:2d}: {mean_ratio:.2f}x")
    print()

# ============================================================================
# Visualization: Time Series Comparison
# ============================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

output_dir = Path("results/vae_baseline/visualizations/sequence_ci_width")
output_dir.mkdir(parents=True, exist_ok=True)

# Create combined time series plot
fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
fig.suptitle(f'Oracle vs Prior Sampling: CI Width Comparison (2004-2023) - {grid_label}',
             fontsize=16, fontweight='bold')

for idx, h in enumerate(horizons):
    ax = axes[idx]

    # Combine all periods for both sampling modes
    all_dates_oracle = []
    all_avg_ci_oracle = []
    all_dates_prior = []
    all_avg_ci_prior = []

    for period in periods:
        prefix = f'{period}_h{h}'

        if f'{prefix}_dates' in oracle_data.files:
            dates = pd.to_datetime(oracle_data[f'{prefix}_dates'])
            avg_ci = oracle_data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]
            all_dates_oracle.extend(dates)
            all_avg_ci_oracle.extend(avg_ci)

        if f'{prefix}_dates' in prior_data.files:
            dates = pd.to_datetime(prior_data[f'{prefix}_dates'])
            avg_ci = prior_data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]
            all_dates_prior.extend(dates)
            all_avg_ci_prior.extend(avg_ci)

    # Convert to arrays and sort
    all_dates_oracle = np.array(all_dates_oracle)
    all_avg_ci_oracle = np.array(all_avg_ci_oracle)
    all_dates_prior = np.array(all_dates_prior)
    all_avg_ci_prior = np.array(all_avg_ci_prior)

    sort_idx_oracle = np.argsort(all_dates_oracle)
    all_dates_oracle = all_dates_oracle[sort_idx_oracle]
    all_avg_ci_oracle = all_avg_ci_oracle[sort_idx_oracle]

    sort_idx_prior = np.argsort(all_dates_prior)
    all_dates_prior = all_dates_prior[sort_idx_prior]
    all_avg_ci_prior = all_avg_ci_prior[sort_idx_prior]

    # Plot both
    ax.plot(all_dates_oracle, all_avg_ci_oracle, color='blue', linewidth=1.5,
            alpha=0.7, label='Oracle (Posterior)')
    ax.plot(all_dates_prior, all_avg_ci_prior, color='red', linewidth=1.5,
            alpha=0.7, label='Prior (Realistic)')

    # Shade crisis/COVID periods
    crisis_start = pd.Timestamp('2008-01-01')
    crisis_end = pd.Timestamp('2010-12-31')
    covid_start = pd.Timestamp('2020-02-15')
    covid_end = pd.Timestamp('2020-04-30')

    ax.axvspan(crisis_start, crisis_end, alpha=0.1, color='red', zorder=0)
    ax.axvspan(covid_start, covid_end, alpha=0.1, color='orange', zorder=0)

    # Formatting
    ax.set_ylabel(f'Avg CI Width\n(H={h})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    ratio_mean = df_comparison[df_comparison['horizon'] == h]['ratio_prior_oracle'].mean()
    stats_text = (f'Prior/Oracle Ratio: {ratio_mean:.2f}x\n'
                  f'Oracle Mean: {all_avg_ci_oracle.mean():.4f}\n'
                  f'Prior Mean: {all_avg_ci_prior.mean():.4f}')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if idx == 0:
        ax.legend(loc='upper right', fontsize=10)

# Format x-axis
axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].set_xlim(pd.Timestamp('2004-01-01'), pd.Timestamp('2024-01-01'))

plt.tight_layout()

output_file = output_dir / "oracle_vs_prior_comparison_KS1.00_6M.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved comparison plot: {output_file}")
plt.close()

# ============================================================================
# Save Results
# ============================================================================

print()
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

output_dir_analysis = Path("results/vae_baseline/analysis")
output_dir_analysis.mkdir(parents=True, exist_ok=True)

# Save comparison CSV
csv_file = output_dir_analysis / "oracle_vs_prior_comparison.csv"
df_comparison.to_csv(csv_file, index=False)
print(f"✓ Saved comparison results: {csv_file}")
print(f"  Rows: {len(df_comparison)}, Columns: {len(df_comparison.columns)}")
print()

# ============================================================================
# Generate Summary Report
# ============================================================================

print("Generating summary markdown report...")

summary_file = output_dir_analysis / "ORACLE_VS_PRIOR_COMPARISON_SUMMARY.md"

with open(summary_file, 'w') as f:
    f.write("# Oracle vs Prior Sampling CI Width Comparison\n\n")
    f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Grid Point:** {grid_label} (K/S=1.00, 6-month)\n\n")
    f.write("---\n\n")

    # Overview
    f.write("## Overview\n\n")
    f.write("This analysis compares confidence interval widths between two sampling strategies:\n\n")
    f.write("- **Oracle (Posterior):** z ~ q(z|context, target) - uses future knowledge (upper bound)\n")
    f.write("- **Prior (Realistic):** z[:,:C] = posterior_mean, z[:,C:] ~ N(0,1) - context only (deployment)\n\n")
    f.write(f"**Total comparisons:** {len(df_comparison)}\n\n")
    f.write(f"**Periods:** {', '.join(periods)}\n\n")
    f.write(f"**Horizons:** {', '.join(map(str, horizons))}\n\n")

    # Key findings
    f.write("## Key Findings\n\n")
    f.write(f"### Overall Statistics\n\n")
    f.write(f"- **Mean prior/oracle ratio:** {df_comparison['ratio_prior_oracle'].mean():.2f}x\n")
    f.write(f"- **Median prior/oracle ratio:** {df_comparison['ratio_prior_oracle'].median():.2f}x\n")
    f.write(f"- **Range:** [{df_comparison['ratio_prior_oracle'].min():.2f}x, {df_comparison['ratio_prior_oracle'].max():.2f}x]\n")
    f.write(f"- **Significant differences (p < 0.05):** {df_comparison['significant'].sum()} / {len(df_comparison)} "
            f"({100 * df_comparison['significant'].sum() / len(df_comparison):.1f}%)\n\n")

    # By period
    f.write("### Ratios by Period\n\n")
    f.write("| Period | Mean Ratio | Median Ratio | Significant |\n")
    f.write("|--------|------------|--------------|-------------|\n")
    for period in periods:
        period_subset = df_comparison[df_comparison['period'] == period]
        if len(period_subset) > 0:
            mean_ratio = period_subset['ratio_prior_oracle'].mean()
            median_ratio = period_subset['ratio_prior_oracle'].median()
            n_sig = period_subset['significant'].sum()
            f.write(f"| {period} | {mean_ratio:.2f}x | {median_ratio:.2f}x | {n_sig}/{len(period_subset)} |\n")
    f.write("\n")

    # By horizon
    f.write("### Ratios by Horizon\n\n")
    f.write("| Horizon | Mean Ratio | Median Ratio | Significant |\n")
    f.write("|---------|------------|--------------|-------------|\n")
    for h in horizons:
        horizon_subset = df_comparison[df_comparison['horizon'] == h]
        if len(horizon_subset) > 0:
            mean_ratio = horizon_subset['ratio_prior_oracle'].mean()
            median_ratio = horizon_subset['ratio_prior_oracle'].median()
            n_sig = horizon_subset['significant'].sum()
            f.write(f"| H={h} | {mean_ratio:.2f}x | {median_ratio:.2f}x | {n_sig}/{len(horizon_subset)} |\n")
    f.write("\n")

    # Detailed comparison table
    f.write("## Detailed Comparison Table\n\n")
    f.write("| Period | H | Oracle Mean | Prior Mean | Ratio | Δ | Rel. Δ (%) | p-value | Sig |\n")
    f.write("|--------|---|-------------|------------|-------|------|------------|---------|-----|\n")

    for _, row in df_comparison.iterrows():
        sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        f.write(f"| {row['period']} | {row['horizon']} | {row['oracle_avg_mean']:.4f} | "
                f"{row['prior_avg_mean']:.4f} | {row['ratio_prior_oracle']:.2f}x | "
                f"{row['absolute_diff']:+.4f} | {row['relative_diff_pct']:+.1f}% | "
                f"{row['p_value']:.2e} | {sig_marker} |\n")

    f.write("\n")

    # Interpretation
    f.write("## Interpretation\n\n")
    f.write("### VAE Prior Mismatch\n\n")
    f.write("The consistent widening of confidence intervals in prior sampling (realistic) vs oracle sampling "
            "(posterior with future knowledge) demonstrates the **VAE prior mismatch** phenomenon:\n\n")
    f.write("- **Oracle sampling:** Encoder sees full sequence (context + target), producing tight latent distribution q(z|context,target)\n")
    f.write("- **Prior sampling:** Encoder sees only context, must sample future from N(0,1) prior, producing wider predictive uncertainty\n\n")
    f.write(f"The observed {df_comparison['ratio_prior_oracle'].mean():.2f}x mean widening factor aligns with "
            "expectations for realistic deployment scenarios where future data is unavailable.\n\n")

    # Output files
    f.write("## Output Files\n\n")
    f.write(f"- **Comparison CSV:** `{csv_file.name}`\n")
    f.write(f"- **Visualization:** `oracle_vs_prior_comparison_KS1.00_6M.png`\n")
    f.write(f"- **Oracle data:** `results/vae_baseline/analysis/oracle/sequence_ci_width_stats.npz`\n")
    f.write(f"- **Prior data:** `results/vae_baseline/analysis/prior/sequence_ci_width_stats.npz`\n\n")

print(f"✓ Saved summary report: {summary_file}")
print()

# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("COMPARISON ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Key result: Prior CIs are {df_comparison['ratio_prior_oracle'].mean():.2f}x wider than oracle CIs on average")
print()
print("Output files:")
print(f"  - CSV: {csv_file}")
print(f"  - Plot: {output_file}")
print(f"  - Report: {summary_file}")
print()
