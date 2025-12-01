"""
Visualize Sequence CI Width Patterns

Creates time series visualizations showing min/max/avg CI width evolution
across full H-day sequences for VAE teacher forcing predictions.

Input: results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz
Output: results/vae_baseline/visualizations/sequence_ci_width/{sampling_mode}/*.png

Usage:
    python experiments/backfill/context20/visualize_sequence_ci_width.py --period insample --sampling_mode oracle
    python experiments/backfill/context20/visualize_sequence_ci_width.py --period oos --grid_point atm_1m --sampling_mode prior
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
import argparse

print("=" * 80)
print("VISUALIZING SEQUENCE CI WIDTH PATTERNS")
print("=" * 80)
print()

# ============================================================================
# Parse Arguments
# ============================================================================

parser = argparse.ArgumentParser(description='Visualize sequence CI width patterns')
parser.add_argument('--period', type=str, default='insample',
                   choices=['insample', 'crisis', 'oos'],
                   help='Period to visualize (default: insample)')
parser.add_argument('--grid_point', type=str, default='atm_6m',
                   choices=['atm_6m', 'atm_1m', 'atm_2y', 'otm_put_6m', 'otm_call_6m'],
                   help='Grid point to visualize (default: atm_6m)')
parser.add_argument('--sampling_mode', type=str, default='oracle',
                   choices=['oracle', 'prior'],
                   help='Sampling strategy (oracle/prior)')
args = parser.parse_args()

# Map grid point names to indices
GRID_POINTS = {
    'atm_6m': (2, 2, 'ATM 6M'),
    'atm_1m': (2, 0, 'ATM 1M'),
    'atm_2y': (2, 4, 'ATM 2Y'),
    'otm_put_6m': (0, 2, 'OTM Put 6M'),
    'otm_call_6m': (4, 2, 'OTM Call 6M'),
}

m_idx, t_idx, grid_label = GRID_POINTS[args.grid_point]

print(f"Period: {args.period}")
print(f"Grid point: {grid_label} (indices: {m_idx}, {t_idx})")
print(f"Sampling mode: {args.sampling_mode}")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading sequence CI width statistics...")
data = np.load(f"results/vae_baseline/analysis/{args.sampling_mode}/sequence_ci_width_stats.npz", allow_pickle=True)
horizons = [1, 7, 14, 30]

# Create output directory
output_dir = Path(f"results/vae_baseline/visualizations/sequence_ci_width/{args.sampling_mode}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")
print()

# ============================================================================
# Define Crisis/COVID Periods
# ============================================================================

crisis_start = pd.Timestamp('2008-01-01')
crisis_end = pd.Timestamp('2010-12-31')
covid_start = pd.Timestamp('2020-02-15')
covid_end = pd.Timestamp('2020-04-30')

# ============================================================================
# Visualization 1: Multi-Panel Time Series (Min/Avg/Max CI Width)
# ============================================================================

print(f"Creating time series plot for {args.period} - {grid_label}...")

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f'Sequence CI Width Over Time - {args.period.upper()} - {grid_label}',
             fontsize=16, fontweight='bold')

for idx, h in enumerate(horizons):
    ax = axes[idx]
    prefix = f'{args.period}_h{h}'

    if f'{prefix}_dates' not in data.files:
        print(f"  ⚠ Skipping H={h} (no data)")
        continue

    # Load data
    dates = pd.to_datetime(data[f'{prefix}_dates'])
    min_ci = data[f'{prefix}_min_ci_width'][:, m_idx, t_idx]
    avg_ci = data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]
    max_ci = data[f'{prefix}_max_ci_width'][:, m_idx, t_idx]

    # Sort by date
    sort_idx = np.argsort(dates)
    dates = dates[sort_idx]
    min_ci = min_ci[sort_idx]
    avg_ci = avg_ci[sort_idx]
    max_ci = max_ci[sort_idx]

    # Plot shaded region (min-max range)
    ax.fill_between(dates, min_ci, max_ci, alpha=0.2, color='blue',
                    label='Min-Max Range')

    # Plot avg CI width (main line)
    ax.plot(dates, avg_ci, color='darkblue', linewidth=2, label='Avg CI Width')

    # Shade crisis/COVID periods
    ax.axvspan(crisis_start, crisis_end, alpha=0.1, color='red', label='Crisis (2008-2010)')
    ax.axvspan(covid_start, covid_end, alpha=0.1, color='orange', label='COVID (Feb-Apr 2020)')

    # Formatting
    ax.set_ylabel(f'CI Width\n(H={h})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = (f'Mean: {avg_ci.mean():.4f}\n'
                  f'Std: {avg_ci.std():.4f}\n'
                  f'Range: [{avg_ci.min():.4f}, {avg_ci.max():.4f}]')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

# Format x-axis
axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()

moneyness_label = f"KS{GRID_POINTS[args.grid_point][0]:.2f}".replace('KS2.', 'KS1.')
if grid_label.startswith('ATM'):
    moneyness_label = 'KS1.00'
elif 'Put' in grid_label:
    moneyness_label = 'KS0.70'
elif 'Call' in grid_label:
    moneyness_label = 'KS1.30'

maturity_label = grid_label.split()[-1]  # Extract '6M', '1M', '2Y'
output_file = output_dir / f"sequence_ci_width_timeseries_{args.period}_{moneyness_label}_{maturity_label}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_file}")
plt.close()

# ============================================================================
# Visualization 2: Correlation Scatter Plots
# ============================================================================

print(f"Creating correlation scatter plots for {args.period} - {grid_label}...")

for h in horizons:
    prefix = f'{args.period}_h{h}'

    if f'{prefix}_dates' not in data.files:
        continue

    # Load data
    min_ci = data[f'{prefix}_min_ci_width'][:, m_idx, t_idx]
    max_ci = data[f'{prefix}_max_ci_width'][:, m_idx, t_idx]
    avg_ci = data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]
    abs_returns = data[f'{prefix}_abs_returns']
    realized_vol = data[f'{prefix}_realized_vol_30d']
    atm_vol = data[f'{prefix}_atm_vol']

    # Create 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'CI Width vs Features - {args.period.upper()} H={h} - {grid_label}',
                fontsize=14, fontweight='bold')

    features = {
        'abs_returns': ('|Returns|', abs_returns),
        'realized_vol_30d': ('Realized Vol (30d, %)', realized_vol),
        'atm_vol': ('ATM Vol', atm_vol)
    }

    for col_idx, (feat_name, (feat_label, feat_values)) in enumerate(features.items()):
        # Min CI scatter (row 0)
        ax = axes[0, col_idx]
        mask = ~np.isnan(feat_values)
        ax.scatter(feat_values[mask], min_ci[mask], alpha=0.3, s=10, color='blue')

        # Regression line
        if mask.sum() > 10:
            z = np.polyfit(feat_values[mask], min_ci[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(feat_values[mask].min(), feat_values[mask].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            r, pval = stats.pearsonr(feat_values[mask], min_ci[mask])
            ax.text(0.05, 0.95, f'r={r:.3f}\np={pval:.2e}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel('Min CI Width', fontsize=10, fontweight='bold')
        ax.set_xlabel(feat_label, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Max CI scatter (row 1)
        ax = axes[1, col_idx]
        ax.scatter(feat_values[mask], max_ci[mask], alpha=0.3, s=10, color='darkblue')

        # Regression line
        if mask.sum() > 10:
            z = np.polyfit(feat_values[mask], max_ci[mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(feat_values[mask].min(), feat_values[mask].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            r, pval = stats.pearsonr(feat_values[mask], max_ci[mask])
            ax.text(0.05, 0.95, f'r={r:.3f}\np={pval:.2e}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel('Max CI Width', fontsize=10, fontweight='bold')
        ax.set_xlabel(feat_label, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"sequence_ci_width_vs_features_{args.period}_H{h}_{moneyness_label}_{maturity_label}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved H={h}: {output_file.name}")
    plt.close()

# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"Output directory: {output_dir}")
print()
print("Generated plots:")
print(f"  - Time series: sequence_ci_width_timeseries_{args.period}_{moneyness_label}_{maturity_label}.png")
print(f"  - Correlation scatters: sequence_ci_width_vs_features_{args.period}_H{{h}}_{moneyness_label}_{maturity_label}.png (4 plots)")
print()
print("To visualize other grid points, run:")
print(f"  python experiments/backfill/context20/visualize_sequence_ci_width.py --period insample --grid_point atm_1m --sampling_mode {args.sampling_mode}")
print(f"  python experiments/backfill/context20/visualize_sequence_ci_width.py --period oos --grid_point otm_put_6m --sampling_mode {args.sampling_mode}")
print()
