"""
Visualize Combined Insample + OOS Sequence CI Width Time Series

Creates a single continuous timeline showing both insample (2004-2019) and
OOS (2019-2023) periods together.

Input: results/vae_baseline/analysis/{sampling_mode}/sequence_ci_width_stats.npz
Output: results/vae_baseline/visualizations/sequence_ci_width/{sampling_mode}/sequence_ci_width_timeseries_combined_KS1.00_6M.png

Usage:
    python experiments/backfill/context20/visualize_sequence_ci_width_combined.py --sampling_mode oracle
    python experiments/backfill/context20/visualize_sequence_ci_width_combined.py --sampling_mode prior
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Visualize combined sequence CI width patterns')
parser.add_argument('--sampling_mode', type=str, default='oracle',
                   choices=['oracle', 'prior'],
                   help='Sampling strategy (oracle/prior)')
args = parser.parse_args()

print("=" * 80)
print("VISUALIZING COMBINED INSAMPLE + OOS CI WIDTH PATTERNS")
print("=" * 80)
print(f"Sampling mode: {args.sampling_mode}")
print()

# Load data
print("Loading sequence CI width statistics...")
data = np.load(f"results/vae_baseline/analysis/{args.sampling_mode}/sequence_ci_width_stats.npz", allow_pickle=True)

# ATM 6-month (benchmark grid point)
m_idx, t_idx = 2, 2
grid_label = 'ATM 6M'

horizons = [1, 7, 14, 30]

# Create output directory
output_dir = Path(f"results/vae_baseline/visualizations/sequence_ci_width/{args.sampling_mode}")
output_dir.mkdir(parents=True, exist_ok=True)

# Define crisis/COVID periods
crisis_start = pd.Timestamp('2008-01-01')
crisis_end = pd.Timestamp('2010-12-31')
covid_start = pd.Timestamp('2020-02-15')
covid_end = pd.Timestamp('2020-04-30')

# Create combined plot
fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
fig.suptitle(f'VAE Sequence CI Width Over Time (2004-2023) - {grid_label}',
             fontsize=16, fontweight='bold')

for idx, h in enumerate(horizons):
    ax = axes[idx]

    # Combine insample, gap, and OOS data
    all_dates = []
    all_min_ci = []
    all_avg_ci = []
    all_max_ci = []

    for period in ['insample', 'gap', 'oos']:
        prefix = f'{period}_h{h}'

        if f'{prefix}_dates' not in data.files:
            continue

        dates = pd.to_datetime(data[f'{prefix}_dates'])
        min_ci = data[f'{prefix}_min_ci_width'][:, m_idx, t_idx]
        avg_ci = data[f'{prefix}_avg_ci_width'][:, m_idx, t_idx]
        max_ci = data[f'{prefix}_max_ci_width'][:, m_idx, t_idx]

        all_dates.extend(dates)
        all_min_ci.extend(min_ci)
        all_avg_ci.extend(avg_ci)
        all_max_ci.extend(max_ci)

    # Convert to arrays and sort by date
    all_dates = np.array(all_dates)
    all_min_ci = np.array(all_min_ci)
    all_avg_ci = np.array(all_avg_ci)
    all_max_ci = np.array(all_max_ci)

    sort_idx = np.argsort(all_dates)
    all_dates = all_dates[sort_idx]
    all_min_ci = all_min_ci[sort_idx]
    all_avg_ci = all_avg_ci[sort_idx]
    all_max_ci = all_max_ci[sort_idx]

    # Plot shaded region (min-max range)
    ax.fill_between(all_dates, all_min_ci, all_max_ci, alpha=0.2, color='blue',
                    label='Min-Max CI Width Range')

    # Plot avg CI width (main line)
    ax.plot(all_dates, all_avg_ci, color='darkblue', linewidth=2, label='Avg CI Width')

    # Shade crisis/COVID periods
    ax.axvspan(crisis_start, crisis_end, alpha=0.1, color='red', label='Crisis (2008-2010)')
    ax.axvspan(covid_start, covid_end, alpha=0.1, color='orange', label='COVID (Feb-Apr 2020)')

    # Formatting
    ax.set_ylabel(f'CI Width\n(H={h})', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = (f'Mean: {all_avg_ci.mean():.4f}\n'
                  f'Std: {all_avg_ci.std():.4f}\n'
                  f'Range: [{all_avg_ci.min():.4f}, {all_avg_ci.max():.4f}]\n'
                  f'N samples: {len(all_avg_ci)}')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

    print(f"  H={h}: {len(all_avg_ci)} total samples ({all_dates.min().date()} to {all_dates.max().date()})")

# Format x-axis
axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].set_xlim(pd.Timestamp('2004-01-01'), pd.Timestamp('2024-01-01'))

plt.tight_layout()

output_file = output_dir / "sequence_ci_width_timeseries_combined_KS1.00_6M.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print()
print(f"✓ Saved combined plot: {output_file}")
print()
print("=" * 80)
print("COMBINED VISUALIZATION COMPLETE")
print("=" * 80)
plt.close()
