"""
Visualize Multi-Horizon CI Bands Comparison

Creates a 4-panel plot showing actual confidence interval bands (p05, p50, p95)
for all horizons, allowing visual comparison of when and how uncertainty changes.

Output: results/backfill_16yr/visualizations/ci_width_temporal/ci_bands_multi_horizon.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse

print("=" * 80)
print("VISUALIZING MULTI-HORIZON CI BANDS COMPARISON")
print("=" * 80)
print()

# ============================================================================
# Parse Arguments
# ============================================================================

parser = argparse.ArgumentParser(description='Visualize CI bands across horizons')
parser.add_argument('--moneyness_idx', type=int, default=2, help='Moneyness index (0-4), default=2 (ATM)')
parser.add_argument('--maturity_idx', type=int, default=2, help='Maturity index (0-4), default=2 (6-month)')
args = parser.parse_args()

m_idx = args.moneyness_idx
t_idx = args.maturity_idx

# Define grid structure
MONEYNESS_VALUES = [0.70, 0.85, 1.00, 1.15, 1.30]
MATURITY_DAYS = [30, 91, 182, 365, 730]
MATURITY_LABELS = ['1M', '3M', '6M', '1Y', '2Y']

# Get actual values
moneyness_value = MONEYNESS_VALUES[m_idx]
maturity_days = MATURITY_DAYS[t_idx]
maturity_label = MATURITY_LABELS[t_idx]

print(f"Grid point: K/S={moneyness_value:.2f}, Maturity={maturity_label} ({maturity_days}d)")
print(f"  (Indices: moneyness_idx={m_idx}, maturity_idx={t_idx})")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Predictions
insample_data = np.load("results/backfill_16yr/predictions/vae_prior_insample_16yr.npz")
try:
    oos_data = np.load("results/backfill_16yr/predictions/vae_prior_oos_16yr.npz")
    has_oos = True
except FileNotFoundError:
    print("  Note: No OOS data found, using in-sample only")
    has_oos = False

# Ground truth
gt_data = np.load("data/vol_surface_with_ret.npz")
gt_surfaces = gt_data['surface']  # (5822, 5, 5)

# Date mapping
df_dates = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")

print(f"  Ground truth shape: {gt_surfaces.shape}")
print(f"  Date range: {df_dates['date'].min()} to {df_dates['date'].max()}")
print()

# ============================================================================
# Define Crisis Periods
# ============================================================================

crisis_start = pd.Timestamp('2008-01-01')
crisis_end = pd.Timestamp('2010-12-31')
covid_start = pd.Timestamp('2020-02-15')
covid_end = pd.Timestamp('2020-04-30')

# ============================================================================
# Extract Data for Each Horizon
# ============================================================================

horizons = [1, 7, 14, 30]
horizon_data = {}

print("Extracting CI bands for each horizon...")

for h in horizons:
    print(f"  Horizon {h}...")

    # Combine in-sample and OOS
    insample_recons = insample_data[f'recon_h{h}']  # (N, 3, 5, 5)
    insample_indices = insample_data[f'indices_h{h}']  # (N,)

    if has_oos and f'recon_h{h}' in oos_data.files:
        oos_recons = oos_data[f'recon_h{h}']
        oos_indices = oos_data[f'indices_h{h}']

        # Concatenate
        recons = np.concatenate([insample_recons, oos_recons], axis=0)
        indices = np.concatenate([insample_indices, oos_indices])
    else:
        recons = insample_recons
        indices = insample_indices

    # Extract selected grid point
    p05 = recons[:, 0, m_idx, t_idx]  # (N,)
    p50 = recons[:, 1, m_idx, t_idx]  # (N,)
    p95 = recons[:, 2, m_idx, t_idx]  # (N,)

    # Extract ground truth for the same indices
    gt = gt_surfaces[indices, m_idx, t_idx]  # (N,)

    # Map to dates
    dates = df_dates.iloc[indices]['date'].values
    dates = pd.to_datetime(dates)

    # Store
    horizon_data[h] = {
        'dates': dates,
        'indices': indices,
        'p05': p05,
        'p50': p50,
        'p95': p95,
        'gt': gt,
        'n_samples': len(dates)
    }

    print(f"    Samples: {len(dates)}, Date range: {dates.min()} to {dates.max()}")

print()

# ============================================================================
# Create Multi-Panel Plot
# ============================================================================

print("Creating multi-panel CI bands plot...")

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f'VAE Model CI Bands Across Horizons\nGrid Point: K/S={moneyness_value:.2f}, Maturity={maturity_label} ({maturity_days}d)',
             fontsize=16, fontweight='bold')

for idx, h in enumerate(horizons):
    ax = axes[idx]
    data = horizon_data[h]

    dates = data['dates']
    p05 = data['p05']
    p50 = data['p50']
    p95 = data['p95']
    gt = data['gt']

    # Sort by date for proper plotting
    sort_idx = np.argsort(dates)
    dates_sorted = dates[sort_idx]
    p05_sorted = p05[sort_idx]
    p50_sorted = p50[sort_idx]
    p95_sorted = p95[sort_idx]
    gt_sorted = gt[sort_idx]

    # Plot CI band (p05-p95)
    ax.fill_between(dates_sorted, p05_sorted, p95_sorted,
                     alpha=0.3, color='blue', label='p05-p95 CI')

    # Plot p50 prediction
    ax.plot(dates_sorted, p50_sorted, color='blue', linewidth=1.5, label='p50 (median)')

    # Plot ground truth
    ax.plot(dates_sorted, gt_sorted, color='black', linewidth=1.5, label='Ground Truth', alpha=0.7)

    # Shade crisis period
    ax.axvspan(crisis_start, crisis_end, alpha=0.1, color='red', label='2008 Crisis' if idx == 0 else '')

    # Shade COVID period
    ax.axvspan(covid_start, covid_end, alpha=0.1, color='orange', label='COVID' if idx == 0 else '')

    # Formatting
    ax.set_ylabel(f'IV (H={h})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend only on first panel
    if idx == 0:
        ax.legend(loc='upper left', fontsize=9, ncol=2)

    # Add statistics text box
    ci_width = np.mean(p95 - p05)
    violations = np.sum((gt < p05) | (gt > p95))
    violation_rate = violations / len(gt) * 100

    stats_text = (f'Mean CI Width: {ci_width:.4f}\n'
                  f'CI Violations: {violation_rate:.1f}%')

    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

    # Highlight regions where ground truth is outside CI
    outside_ci = (gt_sorted < p05_sorted) | (gt_sorted > p95_sorted)
    if outside_ci.sum() > 0:
        # Plot violations as scatter points
        ax.scatter(dates_sorted[outside_ci], gt_sorted[outside_ci],
                   color='red', s=3, alpha=0.5, zorder=10)

# Format x-axis
axes[-1].set_xlabel('Date', fontsize=12)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()

# Save
output_dir = Path("results/backfill_16yr/visualizations/ci_width_temporal")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"ci_bands_multi_horizon_KS{moneyness_value:.2f}_{maturity_label}.png"

plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved {output_file.name}")
print()

# ============================================================================
# Create Comparison Plot (All Horizons' CI Bands Overlaid)
# ============================================================================

print("Creating comparison plot (all horizons' CI bands overlaid)...")

fig, ax = plt.subplots(figsize=(18, 10))
fig.suptitle(f'VAE Model CI Bands Across All Horizons (Single Panel)\nGrid Point: K/S={moneyness_value:.2f}, Maturity={maturity_label} ({maturity_days}d)',
             fontsize=14, fontweight='bold')

colors = ['blue', 'green', 'darkorange', 'red']
alphas = [0.15, 0.15, 0.15, 0.15]  # Transparency for CI bands

# Plot ground truth first (from longest horizon for most coverage)
h_ref = 30
data_ref = horizon_data[h_ref]
dates_ref = data_ref['dates']
gt_ref = data_ref['gt']
sort_idx_ref = np.argsort(dates_ref)

ax.plot(dates_ref[sort_idx_ref], gt_ref[sort_idx_ref],
        color='black', linewidth=2, label='Ground Truth', alpha=0.8, zorder=10)

# Plot each horizon's CI bands and p50
for idx, h in enumerate(horizons):
    data = horizon_data[h]
    dates = data['dates']
    p05 = data['p05']
    p50 = data['p50']
    p95 = data['p95']

    sort_idx = np.argsort(dates)
    dates_sorted = dates[sort_idx]
    p05_sorted = p05[sort_idx]
    p50_sorted = p50[sort_idx]
    p95_sorted = p95[sort_idx]

    # Plot CI band (p05-p95)
    ax.fill_between(dates_sorted, p05_sorted, p95_sorted,
                     alpha=alphas[idx], color=colors[idx],
                     label=f'H={h} CI')

    # Plot p50
    ax.plot(dates_sorted, p50_sorted,
            color=colors[idx], linewidth=1.5, alpha=0.9,
            linestyle='--', label=f'H={h} p50')

# Shade crisis and COVID
ax.axvspan(crisis_start, crisis_end, alpha=0.08, color='gray',
           edgecolor='red', linewidth=2, linestyle='--', label='2008 Crisis')
ax.axvspan(covid_start, covid_end, alpha=0.08, color='gray',
           edgecolor='orange', linewidth=2, linestyle='--', label='COVID')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Implied Volatility', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9, ncol=3)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

# Add text box with summary
summary_text = "CI Band Interpretation:\n"
for idx, h in enumerate(horizons):
    data = horizon_data[h]
    ci_width = np.mean(data['p95'] - data['p05'])
    summary_text += f"H={h}: width={ci_width:.4f}, "

ax.text(0.02, 0.02, summary_text.strip(', '), transform=ax.transAxes,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=8)

plt.tight_layout()

output_file_overlay = output_dir / f"ci_bands_all_horizons_overlay_KS{moneyness_value:.2f}_{maturity_label}.png"
plt.savefig(output_file_overlay, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved {output_file_overlay.name}")
print()

# ============================================================================
# Summary Statistics
# ============================================================================

print("=" * 80)
print("SUMMARY STATISTICS BY HORIZON")
print("=" * 80)
print()

for h in horizons:
    data = horizon_data[h]
    p05 = data['p05']
    p50 = data['p50']
    p95 = data['p95']
    gt = data['gt']

    ci_width = p95 - p05
    violations = (gt < p05) | (gt > p95)
    violation_rate = np.mean(violations) * 100

    # RMSE
    rmse = np.sqrt(np.mean((p50 - gt) ** 2))

    # Mean absolute error
    mae = np.mean(np.abs(p50 - gt))

    print(f"Horizon {h}:")
    print(f"  Samples: {data['n_samples']}")
    print(f"  Mean CI width: {ci_width.mean():.4f} (std: {ci_width.std():.4f})")
    print(f"  CI width range: [{ci_width.min():.4f}, {ci_width.max():.4f}]")
    print(f"  CI violation rate: {violation_rate:.2f}%")
    print(f"  RMSE (p50 vs GT): {rmse:.4f}")
    print(f"  MAE (p50 vs GT): {mae:.4f}")
    print()

print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"Plots saved to: {output_dir}")
print(f"  - {output_file.name}")
print(f"  - {output_file_overlay.name}")
