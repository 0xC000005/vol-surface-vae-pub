"""
Visualize Single AR Sequence - Context60 Model

Shows ONE AR sequence (H=180) in detail with:
- 60-day context (ground truth)
- All 3 quantile predictions (p05, p50, p95)
- Ground truth overlay for forecast period
- Discontinuity markers at chunk boundaries

Output: results/context60_baseline/visualizations/single_ar_sequence_detailed.png

Usage:
    python experiments/backfill/context60/visualize_single_ar_sequence.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("SINGLE AR SEQUENCE DETAILED VISUALIZATION")
print("=" * 80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONTEXT_LEN = 60
AR_HORIZON = 180
ATM_6M = (2, 2)  # Grid indices for ATM 6M

# Sequence to visualize (crisis period for dramatic discontinuities)
PERIOD = 'crisis'
SEQ_IDX = 200  # Representative crisis sequence

print(f"Configuration:")
print(f"  Period: {PERIOD}")
print(f"  Sequence index: {SEQ_IDX}")
print(f"  Context length: {CONTEXT_LEN} days")
print(f"  AR horizon: {AR_HORIZON} days")
print(f"  Grid point: ATM 6M (index {ATM_6M})")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading AR predictions and ground truth...")

# Load AR H=180 predictions
ar_filepath = f"results/context60_baseline/predictions/autoregressive_multi_step/oracle/vae_ar_{PERIOD}_180day.npz"
ar_data = np.load(ar_filepath)
ar_surfaces = ar_data['surfaces']  # (n_dates, 180, 3, 5, 5)
ar_indices = ar_data['indices']    # (n_dates,)
print(f"  ✓ Loaded AR predictions: {ar_surfaces.shape}")

# Load ground truth
gt_data = np.load("data/vol_surface_with_ret.npz")
gt_surface = gt_data['surface']  # (N, 5, 5)
print(f"  ✓ Loaded ground truth: {gt_surface.shape}")

# Load dates
dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
gt_dates = pd.to_datetime(dates_df["date"].values)
print(f"  ✓ Loaded {len(gt_dates)} dates")

print()

# ============================================================================
# Extract Sequence
# ============================================================================

print(f"Extracting sequence {SEQ_IDX} from {PERIOD} period...")

# Get AR forecast for selected sequence
ar_forecast = ar_surfaces[SEQ_IDX]  # (180, 3, 5, 5)
ar_start_idx = ar_indices[SEQ_IDX]

# Extract ATM 6M quantiles from AR forecast
ar_p05 = ar_forecast[:, 0, ATM_6M[0], ATM_6M[1]]  # (180,)
ar_p50 = ar_forecast[:, 1, ATM_6M[0], ATM_6M[1]]  # (180,)
ar_p95 = ar_forecast[:, 2, ATM_6M[0], ATM_6M[1]]  # (180,)

print(f"  AR start index: {ar_start_idx}")
print(f"  AR start date: {gt_dates[ar_start_idx].strftime('%Y-%m-%d')}")

# Get context from ground truth (60 days before AR start)
context_start = ar_start_idx - CONTEXT_LEN
context_gt = gt_surface[context_start:ar_start_idx, ATM_6M[0], ATM_6M[1]]  # (60,)

print(f"  Context period: {gt_dates[context_start].strftime('%Y-%m-%d')} to {gt_dates[ar_start_idx-1].strftime('%Y-%m-%d')}")

# Get forecast ground truth (180 days after AR start)
forecast_gt = gt_surface[ar_start_idx:ar_start_idx+AR_HORIZON, ATM_6M[0], ATM_6M[1]]  # (180,)

print(f"  Forecast period: {gt_dates[ar_start_idx].strftime('%Y-%m-%d')} to {gt_dates[ar_start_idx+AR_HORIZON-1].strftime('%Y-%m-%d')}")

print()

# ============================================================================
# Compute Statistics
# ============================================================================

print("Computing statistics...")

# RMSE by chunk
chunk_1_rmse = np.sqrt(np.mean((ar_p50[0:60] - forecast_gt[0:60])**2))
chunk_2_rmse = np.sqrt(np.mean((ar_p50[60:120] - forecast_gt[60:120])**2))
chunk_3_rmse = np.sqrt(np.mean((ar_p50[120:180] - forecast_gt[120:180])**2))

print(f"  RMSE by chunk:")
print(f"    Chunk 1 (days 0-59):   {chunk_1_rmse:.6f}")
print(f"    Chunk 2 (days 60-119):  {chunk_2_rmse:.6f}")
print(f"    Chunk 3 (days 120-179): {chunk_3_rmse:.6f}")

# CI violation rate
in_ci = (forecast_gt >= ar_p05) & (forecast_gt <= ar_p95)
ci_violation_rate = 100 * (1 - in_ci.mean())

print(f"  CI violation rate: {ci_violation_rate:.1f}%")

# Discontinuity magnitudes (in quantile p50)
drop_60_p50 = ar_p50[60] - ar_p50[59]
pct_60_p50 = 100 * drop_60_p50 / ar_p50[59]
drop_120_p50 = ar_p50[120] - ar_p50[119]
pct_120_p50 = 100 * drop_120_p50 / ar_p50[119]

print(f"  Discontinuities in p50:")
print(f"    Day 60:  {drop_60_p50:+.6f} ({pct_60_p50:+.1f}%)")
print(f"    Day 120: {drop_120_p50:+.6f} ({pct_120_p50:+.1f}%)")

# Discontinuity magnitudes (in CI width)
ci_width = ar_p95 - ar_p05
drop_60_ci = ci_width[60] - ci_width[59]
pct_60_ci = 100 * drop_60_ci / ci_width[59]
drop_120_ci = ci_width[120] - ci_width[119]
pct_120_ci = 100 * drop_120_ci / ci_width[119]

print(f"  Discontinuities in CI width:")
print(f"    Day 60:  {drop_60_ci:+.6f} ({pct_60_ci:+.1f}%)")
print(f"    Day 120: {drop_120_ci:+.6f} ({pct_120_ci:+.1f}%)")

print()

# ============================================================================
# Create Visualization
# ============================================================================

print("Creating visualization...")

fig, ax = plt.subplots(1, 1, figsize=(20, 9))

# Create x-axis (relative days, -60 to 180)
context_days = np.arange(-CONTEXT_LEN, 0)
forecast_days = np.arange(0, AR_HORIZON)

# 1. Context ground truth (solid black, foreground)
ax.plot(context_days, context_gt,
       color='black', linewidth=2.5, linestyle='-', zorder=6,
       label='Context: Ground Truth')

# 2. Forecast ground truth (dotted black)
ax.plot(forecast_days, forecast_gt,
       color='black', linewidth=2, linestyle=':', alpha=0.8, zorder=5,
       label='Forecast: Ground Truth')

# 3. Confidence interval band (p05-p95 fill)
ax.fill_between(forecast_days, ar_p05, ar_p95,
               alpha=0.2, color='purple', zorder=1,
               label='90% Confidence Interval')

# 4. p05 quantile (dashed blue)
ax.plot(forecast_days, ar_p05,
       color='blue', linewidth=1.5, linestyle='--', zorder=3,
       label='p05 (Lower Bound)')

# 5. p50 quantile (solid purple)
ax.plot(forecast_days, ar_p50,
       color='purple', linewidth=2, linestyle='-', zorder=4,
       label='p50 (Median Prediction)')

# 6. p95 quantile (dashed red)
ax.plot(forecast_days, ar_p95,
       color='red', linewidth=1.5, linestyle='--', zorder=3,
       label='p95 (Upper Bound)')

# Visual elements
# Background shading for context region
ax.axvspan(-CONTEXT_LEN, 0, alpha=0.05, color='gray', zorder=0,
          label='Context Region')

# Vertical lines at boundaries
ax.axvline(x=0, color='black', linewidth=2, linestyle='-',
          alpha=0.7, zorder=3)
ax.text(0, ax.get_ylim()[1]*0.95, 'AR Start (Day 0)',
       ha='center', va='top', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))

ax.axvline(x=60, color='purple', linewidth=2, linestyle='--',
          alpha=0.7, zorder=3)
ax.text(60, ax.get_ylim()[1]*0.95, f'Chunk 2 Boundary\n(Day 60, {pct_60_ci:+.1f}% CI drop)',
       ha='center', va='top', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9, edgecolor='purple', linewidth=2))

ax.axvline(x=120, color='purple', linewidth=2, linestyle='--',
          alpha=0.7, zorder=3)
ax.text(120, ax.get_ylim()[1]*0.95, f'Chunk 3 Boundary\n(Day 120, {pct_120_ci:+.1f}% CI drop)',
       ha='center', va='top', fontsize=10, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9, edgecolor='purple', linewidth=2))

# Discontinuity markers (circles at p50)
for day, label in [(60, 'Drop 1'), (120, 'Drop 2')]:
    ax.plot(day, ar_p50[day],
           marker='o', markersize=10, color='purple', markeredgecolor='black',
           markeredgewidth=2, zorder=7)

# Labels and formatting
ax.set_xlabel('Day (Relative to AR Start)', fontsize=14, fontweight='bold')
ax.set_ylabel('ATM 6M Implied Volatility', fontsize=14, fontweight='bold')
ax.set_title(f'Single AR Sequence Detailed View - Context60 Model\n'
            f'{PERIOD.title()} Period: {gt_dates[ar_start_idx].strftime("%Y-%m-%d")} (Sequence {SEQ_IDX})',
            fontsize=16, fontweight='bold', pad=20)

# Statistics box
stats_text = (
    f'Chunk RMSE: 1={chunk_1_rmse:.6f}, 2={chunk_2_rmse:.6f}, 3={chunk_3_rmse:.6f}\n'
    f'CI Violation Rate: {ci_violation_rate:.1f}%\n'
    f'p50 Discontinuities: Day 60={pct_60_p50:+.1f}%, Day 120={pct_120_p50:+.1f}%\n'
    f'CI Width Discontinuities: Day 60={pct_60_ci:+.1f}%, Day 120={pct_120_ci:+.1f}%'
)
ax.text(0.02, 0.98, stats_text,
       transform=ax.transAxes, ha='left', va='top',
       fontsize=10, fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                edgecolor='orange', linewidth=2))

# Grid and limits
ax.grid(True, alpha=0.2, linestyle='--', zorder=2)
ax.set_xlim(-CONTEXT_LEN, AR_HORIZON)

# X-ticks
ax.set_xticks(np.arange(-60, AR_HORIZON + 1, 20))

# Legend (below plot)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
         ncol=4, fontsize=10, framealpha=0.95)

plt.tight_layout()

# ============================================================================
# Save
# ============================================================================

output_dir = Path("results/context60_baseline/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "single_ar_sequence_detailed.png"

print(f"Saving plot...")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_file}")
print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

plt.close()

# ============================================================================
# Summary
# ============================================================================

print()
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print()
print(f"Output: {output_file}")
print()
print("The plot shows:")
print("  - 60-day context (solid black line)")
print("  - Ground truth forecast (dotted black line)")
print("  - All 3 quantiles (p05 blue dashed, p50 purple solid, p95 red dashed)")
print("  - 90% CI band (purple shaded region)")
print("  - Chunk boundaries at days 0, 60, 120 with discontinuity annotations")
print()
print(f"Key observations:")
print(f"  - CI width drops {pct_60_ci:+.1f}% at day 60")
print(f"  - CI width drops {pct_120_ci:+.1f}% at day 120")
print(f"  - Overall CI violation rate: {ci_violation_rate:.1f}%")
print()
