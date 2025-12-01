"""
Visualize Temporal CI Width Patterns for VAE Context 20 Model

Creates comprehensive visualizations showing when model uncertainty (CI width)
increases vs decreases over time, and how it correlates with market conditions.

Requires: results/backfill_16yr/analysis/ci_width_timeseries_16yr.npz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

print("=" * 80)
print("VISUALIZING CI WIDTH TEMPORAL PATTERNS")
print("=" * 80)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading CI width time series...")
data = np.load("results/backfill_16yr/analysis/ci_width_timeseries_16yr.npz", allow_pickle=True)

horizons = [1, 7, 14, 30]

# Create output directory
output_dir = Path("results/backfill_16yr/visualizations/ci_width_temporal")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")
print()

# ============================================================================
# Define Crisis Periods
# ============================================================================

# Crisis period: 2008-2010 (indices 2000-2765 approximately)
# We'll define it by dates for better clarity
crisis_start = pd.Timestamp('2008-01-01')
crisis_end = pd.Timestamp('2010-12-31')

covid_start = pd.Timestamp('2020-02-15')
covid_end = pd.Timestamp('2020-04-30')

print(f"Crisis period: {crisis_start} to {crisis_end}")
print(f"COVID period: {covid_start} to {covid_end}")
print()

# ============================================================================
# Visualization 1: Multi-Panel Time Series
# ============================================================================

print("Creating Visualization 1: Multi-panel time series...")

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle('VAE Model CI Width Over Time (2004-2023)', fontsize=16, fontweight='bold')

for idx, h in enumerate(horizons):
    ax = axes[idx]

    # Load data for this horizon
    ci_width = data[f'h{h}_ci_width']
    dates_str = data[f'h{h}_dates']
    dates = pd.to_datetime(dates_str)

    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'date': dates,
        'ci_width': ci_width
    })

    # Compute rolling mean (30-day)
    df['ci_width_rolling'] = df['ci_width'].rolling(window=30, center=True).mean()

    # Plot raw CI width
    ax.plot(df['date'], df['ci_width'], alpha=0.3, color='blue', linewidth=0.5, label='Daily CI width')

    # Plot rolling mean
    ax.plot(df['date'], df['ci_width_rolling'], color='darkblue', linewidth=2, label='30-day rolling mean')

    # Shade crisis period
    ax.axvspan(crisis_start, crisis_end, alpha=0.15, color='red', label='2008 Crisis')

    # Shade COVID period
    ax.axvspan(covid_start, covid_end, alpha=0.15, color='orange', label='COVID Crash')

    # Add annotations for key events
    if h == 1:  # Only on first panel to avoid clutter
        # Lehman collapse
        lehman_date = pd.Timestamp('2008-09-15')
        ax.axvline(lehman_date, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(lehman_date, ax.get_ylim()[1] * 0.95, 'Lehman\nCollapse',
                ha='right', va='top', fontsize=8, color='red')

        # COVID crash
        covid_crash = pd.Timestamp('2020-03-16')
        ax.axvline(covid_crash, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(covid_crash, ax.get_ylim()[1] * 0.95, 'COVID\nCrash',
                ha='right', va='top', fontsize=8, color='orange')

    # Formatting
    ax.set_ylabel(f'CI Width (H={h})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # Add statistics text box
    mean_ci = ci_width.mean()
    std_ci = ci_width.std()
    stats_text = f'Mean: {mean_ci:.4f}\nStd: {std_ci:.4f}'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            fontsize=9)

axes[-1].set_xlabel('Date', fontsize=12)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig(output_dir / 'ci_width_timeseries_multipanel.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved ci_width_timeseries_multipanel.png")
print()

# ============================================================================
# Visualization 2: CI Width vs Realized Volatility
# ============================================================================

print("Creating Visualization 2: CI width vs realized volatility...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('VAE CI Width vs Realized Volatility (30-day)', fontsize=14, fontweight='bold')

for idx, h in enumerate(horizons):
    ax = axes[idx // 2, idx % 2]

    ci_width = data[f'h{h}_ci_width']
    realized_vol = data[f'h{h}_realized_vol_30d']

    # Remove NaNs
    mask = ~np.isnan(realized_vol)
    ci_width_clean = ci_width[mask]
    realized_vol_clean = realized_vol[mask]

    # Scatter plot
    ax.scatter(realized_vol_clean, ci_width_clean, alpha=0.3, s=10, color='blue')

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(realized_vol_clean, ci_width_clean)
    x_line = np.array([realized_vol_clean.min(), realized_vol_clean.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'r = {r_value:.3f} (p={p_value:.2e})')

    ax.set_xlabel('Realized Volatility (30-day, annualized %)', fontsize=10)
    ax.set_ylabel('CI Width', fontsize=10)
    ax.set_title(f'Horizon {h}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'ci_width_vs_realized_vol.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved ci_width_vs_realized_vol.png")
print()

# ============================================================================
# Visualization 3: CI Width vs Forecast Error (RMSE)
# ============================================================================

print("Creating Visualization 3: CI width vs RMSE...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('VAE CI Width vs Forecast Error (RMSE)', fontsize=14, fontweight='bold')

for idx, h in enumerate(horizons):
    ax = axes[idx // 2, idx % 2]

    ci_width = data[f'h{h}_ci_width']
    rmse = data[f'h{h}_rmse']

    # Remove NaNs
    mask = ~np.isnan(rmse)
    ci_width_clean = ci_width[mask]
    rmse_clean = rmse[mask]

    # Scatter plot
    ax.scatter(rmse_clean, ci_width_clean, alpha=0.3, s=10, color='green')

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(rmse_clean, ci_width_clean)
    x_line = np.array([rmse_clean.min(), rmse_clean.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'r = {r_value:.3f} (p={p_value:.2e})')

    ax.set_xlabel('RMSE', fontsize=10)
    ax.set_ylabel('CI Width', fontsize=10)
    ax.set_title(f'Horizon {h}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # Add interpretation text
    if r_value > 0.3:
        interpretation = "Strong positive correlation:\nModel is more uncertain when errors are larger"
    elif r_value > 0.1:
        interpretation = "Weak positive correlation:\nUncertainty partially tracks error"
    else:
        interpretation = "Weak correlation:\nUncertainty doesn't track error well"

    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
            fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'ci_width_vs_rmse.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved ci_width_vs_rmse.png")
print()

# ============================================================================
# Visualization 4: Horizon Comparison
# ============================================================================

print("Creating Visualization 4: Horizon comparison...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('CI Width Across All Horizons', fontsize=14, fontweight='bold')

# Panel 1: All horizons overlaid
colors = ['blue', 'green', 'orange', 'red']
for idx, h in enumerate(horizons):
    ci_width = data[f'h{h}_ci_width']
    dates_str = data[f'h{h}_dates']
    dates = pd.to_datetime(dates_str)

    df = pd.DataFrame({'date': dates, 'ci_width': ci_width})
    df['ci_width_rolling'] = df['ci_width'].rolling(window=30, center=True).mean()

    ax1.plot(df['date'], df['ci_width_rolling'], color=colors[idx], linewidth=2,
             label=f'H={h} (mean={ci_width.mean():.4f})', alpha=0.8)

ax1.axvspan(crisis_start, crisis_end, alpha=0.1, color='red')
ax1.axvspan(covid_start, covid_end, alpha=0.1, color='orange')
ax1.set_ylabel('CI Width (30-day rolling mean)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10)
ax1.set_title('All Horizons Overlaid', fontsize=12, fontweight='bold')

# Panel 2: Ratio H30 / H1
h30_ci = data['h30_ci_width']
h30_dates = pd.to_datetime(data['h30_dates'])
h1_ci = data['h1_ci_width']
h1_dates = pd.to_datetime(data['h1_dates'])

# Align dates (H30 has fewer samples)
df_h30 = pd.DataFrame({'date': h30_dates, 'ci_h30': h30_ci})
df_h1 = pd.DataFrame({'date': h1_dates, 'ci_h1': h1_ci})
df_ratio = pd.merge(df_h30, df_h1, on='date', how='inner')
df_ratio['ratio'] = df_ratio['ci_h30'] / df_ratio['ci_h1']
df_ratio['ratio_rolling'] = df_ratio['ratio'].rolling(window=30, center=True).mean()

ax2.plot(df_ratio['date'], df_ratio['ratio'], alpha=0.3, color='purple', linewidth=0.5)
ax2.plot(df_ratio['date'], df_ratio['ratio_rolling'], color='indigo', linewidth=2,
         label=f'30-day rolling mean (overall mean={df_ratio["ratio"].mean():.3f})')
ax2.axvspan(crisis_start, crisis_end, alpha=0.1, color='red')
ax2.axvspan(covid_start, covid_end, alpha=0.1, color='orange')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('CI Width Ratio (H30 / H1)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=10)
ax2.set_title('Uncertainty Scaling: H30 vs H1', fontsize=12, fontweight='bold')

# Add horizontal line at mean ratio
mean_ratio = df_ratio['ratio'].mean()
ax2.axhline(mean_ratio, color='black', linestyle='--', alpha=0.5, linewidth=1)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig(output_dir / 'ci_width_horizon_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved ci_width_horizon_comparison.png")
print()

# ============================================================================
# Summary Statistics
# ============================================================================

print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

for h in horizons:
    ci_width = data[f'h{h}_ci_width']
    dates_str = data[f'h{h}_dates']
    dates = pd.to_datetime(dates_str)
    realized_vol = data[f'h{h}_realized_vol_30d']
    rmse = data[f'h{h}_rmse']

    print(f"Horizon {h}:")
    print(f"  Total samples: {len(ci_width)}")
    print(f"  CI width: mean={ci_width.mean():.4f}, std={ci_width.std():.4f}")
    print(f"  CI width range: [{ci_width.min():.4f}, {ci_width.max():.4f}]")

    # Find max/min CI width dates
    max_idx = np.argmax(ci_width)
    min_idx = np.argmin(ci_width)
    print(f"  Max CI width: {ci_width[max_idx]:.4f} on {dates[max_idx].date()}")
    print(f"  Min CI width: {ci_width[min_idx]:.4f} on {dates[min_idx].date()}")

    # Correlation with realized vol
    mask = ~np.isnan(realized_vol)
    if mask.sum() > 0:
        r_vol, p_vol = stats.pearsonr(ci_width[mask], realized_vol[mask])
        print(f"  Correlation with realized vol: r={r_vol:.3f} (p={p_vol:.2e})")

    # Correlation with RMSE
    mask = ~np.isnan(rmse)
    if mask.sum() > 0:
        r_rmse, p_rmse = stats.pearsonr(ci_width[mask], rmse[mask])
        print(f"  Correlation with RMSE: r={r_rmse:.3f} (p={p_rmse:.2e})")

    # Crisis vs normal
    crisis_mask = (dates >= crisis_start) & (dates <= crisis_end)
    normal_mask = ~crisis_mask

    ci_crisis = ci_width[crisis_mask]
    ci_normal = ci_width[normal_mask]

    if len(ci_crisis) > 0 and len(ci_normal) > 0:
        print(f"  Crisis period CI: mean={ci_crisis.mean():.4f}, std={ci_crisis.std():.4f}")
        print(f"  Normal period CI: mean={ci_normal.mean():.4f}, std={ci_normal.std():.4f}")
        print(f"  Difference: {ci_crisis.mean() - ci_normal.mean():.4f} ({(ci_crisis.mean() / ci_normal.mean() - 1) * 100:.1f}%)")

    print()

print("=" * 80)
print("TEMPORAL VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"All plots saved to: {output_dir}")
