"""
Test Levels vs Changes Hypothesis - Context60 AR Discontinuities

Investigates whether AR discontinuities are caused by modeling absolute volatility
LEVELS rather than CHANGES. Tests hypothesis using post-processing only (no retraining).

Tests performed:
1. Analyze discontinuities in changes vs levels
2. Post-process level predictions to change-based (accumulated)
3. Compare loss computation methods
4. Check if accumulation reduces discontinuities
5. RMSE comparison: level-based vs change-accumulated

Usage:
    python experiments/backfill/context60/test_levels_vs_changes_hypothesis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("LEVELS VS CHANGES HYPOTHESIS TEST")
print("=" * 80)
print()

# ============================================================================
# Configuration
# ============================================================================

CONTEXT_LEN = 60
AR_HORIZON = 180
ATM_6M = (2, 2)  # Grid indices for ATM 6M

# Test on crisis period for dramatic discontinuities
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

# Load AR H=180 predictions (oracle mode)
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
# Extract Test Sequence
# ============================================================================

print(f"Extracting sequence {SEQ_IDX} from {PERIOD} period...")

# Get AR forecast for selected sequence
ar_forecast = ar_surfaces[SEQ_IDX]  # (180, 3, 5, 5)
ar_start_idx = ar_indices[SEQ_IDX]

# Extract ATM 6M median from AR forecast
ar_p50_levels = ar_forecast[:, 1, ATM_6M[0], ATM_6M[1]]  # (180,)

print(f"  AR start index: {ar_start_idx}")
print(f"  AR start date: {gt_dates[ar_start_idx].strftime('%Y-%m-%d')}")

# Get context from ground truth (60 days before AR start)
context_start = ar_start_idx - CONTEXT_LEN
context_end_level = gt_surface[ar_start_idx - 1, ATM_6M[0], ATM_6M[1]]  # Last context day

print(f"  Context end level: {context_end_level:.6f}")

# Get forecast ground truth (180 days after AR start)
gt_levels = gt_surface[ar_start_idx:ar_start_idx+AR_HORIZON, ATM_6M[0], ATM_6M[1]]  # (180,)

print(f"  Ground truth range: [{gt_levels.min():.6f}, {gt_levels.max():.6f}]")

print()

# ============================================================================
# TEST 1: Analyze Discontinuities in Changes vs Levels
# ============================================================================

print("=" * 80)
print("TEST 1: Discontinuities in Changes vs Levels")
print("=" * 80)
print()

# Compute first differences (changes) from level predictions
predicted_changes = np.diff(ar_p50_levels)  # (179,) - day-to-day changes

# Compute second differences (acceleration = change in changes)
predicted_accel = np.diff(predicted_changes)  # (178,)

# Ground truth changes
gt_changes = np.diff(gt_levels)  # (179,)
gt_accel = np.diff(gt_changes)  # (178,)

# Check discontinuities at boundaries (days 59->60 and 119->120)
# For day 60 boundary (index 59 in diff arrays)
level_jump_60 = ar_p50_levels[60] - ar_p50_levels[59]
change_jump_60 = predicted_changes[60] - predicted_changes[59]  # Change in slope
accel_60 = predicted_accel[59]  # Acceleration at boundary

# For day 120 boundary (index 119 in diff arrays)
level_jump_120 = ar_p50_levels[120] - ar_p50_levels[119]
change_jump_120 = predicted_changes[120] - predicted_changes[119]
accel_120 = predicted_accel[119]

print("Discontinuity Analysis at Day 60:")
print(f"  Level jump:        {level_jump_60:+.6f}")
print(f"  Change jump:       {change_jump_60:+.6f}")
print(f"  Acceleration:      {accel_60:+.6f}")
print()

print("Discontinuity Analysis at Day 120:")
print(f"  Level jump:        {level_jump_120:+.6f}")
print(f"  Change jump:       {change_jump_120:+.6f}")
print(f"  Acceleration:      {accel_120:+.6f}")
print()

# Compare magnitudes
print("Interpretation:")
if abs(accel_60) > 0.001:  # Significant acceleration threshold
    print("  ✓ LARGE acceleration at day 60 → discontinuity in TREND (change pattern shifts)")
else:
    print("  ✗ Small acceleration at day 60 → only level discontinuity (trend continuous)")

if abs(accel_120) > 0.001:
    print("  ✓ LARGE acceleration at day 120 → discontinuity in TREND (change pattern shifts)")
else:
    print("  ✗ Small acceleration at day 120 → only level discontinuity (trend continuous)")

print()

# ============================================================================
# TEST 2: Post-Process to Change-Based Predictions
# ============================================================================

print("=" * 80)
print("TEST 2: Convert Level Predictions to Accumulated Change-Based")
print("=" * 80)
print()

print("Converting existing level predictions to change-based...")

# Extract predicted changes from original level predictions
predicted_changes_from_levels = np.diff(ar_p50_levels)  # (179,)

# Anchor to last context day and accumulate changes
accumulated_levels = np.zeros(AR_HORIZON)
accumulated_levels[0] = context_end_level + predicted_changes_from_levels[0]

for t in range(1, AR_HORIZON - 1):
    accumulated_levels[t] = accumulated_levels[t-1] + predicted_changes_from_levels[t]

# Handle last day (no change beyond day 178)
accumulated_levels[-1] = accumulated_levels[-2] + predicted_changes_from_levels[-1]

print(f"  ✓ Accumulated levels from changes")
print(f"  Original levels range: [{ar_p50_levels.min():.6f}, {ar_p50_levels.max():.6f}]")
print(f"  Accumulated range:     [{accumulated_levels.min():.6f}, {accumulated_levels.max():.6f}]")
print()

# Check differences
diff_original_accumulated = ar_p50_levels - accumulated_levels
print(f"  Mean difference: {diff_original_accumulated.mean():.6f}")
print(f"  Max difference:  {diff_original_accumulated.max():.6f}")
print(f"  Min difference:  {diff_original_accumulated.min():.6f}")
print()

# ============================================================================
# TEST 3: Compare Loss Computation Methods
# ============================================================================

print("=" * 80)
print("TEST 3: Compare Loss Computation Methods")
print("=" * 80)
print()

# Option A: Direct change comparison
predicted_changes_trimmed = predicted_changes_from_levels[:len(gt_changes)]
loss_changes = np.mean((predicted_changes_trimmed - gt_changes)**2)
print(f"Option A - Direct Change Loss (MSE on changes):")
print(f"  Loss: {loss_changes:.9f}")
print(f"  Pros: Directly measures change prediction accuracy")
print(f"  Cons: Doesn't reflect final use case (we care about level accuracy)")
print()

# Option B: Log returns
# Add small epsilon to avoid log(0) or division by zero
epsilon = 1e-10
predicted_returns = np.log((ar_p50_levels[1:] + epsilon) / (ar_p50_levels[:-1] + epsilon))
gt_returns = np.log((gt_levels[1:] + epsilon) / (gt_levels[:-1] + epsilon))
loss_returns = np.mean((predicted_returns - gt_returns)**2)
print(f"Option B - Log Returns Loss (MSE on log returns):")
print(f"  Loss: {loss_returns:.9f}")
print(f"  Pros: Percentage changes, scale-invariant")
print(f"  Cons: Assumes multiplicative dynamics (may not fit volatility)")
print()

# Option C: Accumulated levels from changes (RECOMMENDED)
loss_accumulated = np.mean((accumulated_levels - gt_levels)**2)
print(f"Option C - Accumulated Levels Loss (MSE on final levels) [RECOMMENDED]:")
print(f"  Loss: {loss_accumulated:.9f}")
print(f"  Pros: Measures actual use case (level prediction accuracy)")
print(f"  Cons: Errors accumulate (but that's realistic!)")
print()

# Compare to original level-based loss
loss_original = np.mean((ar_p50_levels - gt_levels)**2)
print(f"Original Level-Based Loss (for comparison):")
print(f"  Loss: {loss_original:.9f}")
print()

# ============================================================================
# TEST 4: Discontinuity Impact on Accumulated Approach
# ============================================================================

print("=" * 80)
print("TEST 4: Discontinuity Impact - Original vs Accumulated")
print("=" * 80)
print()

print("Checking if accumulation reduces discontinuities...")

# Original discontinuities (from TEST 1)
original_jump_60 = ar_p50_levels[60] - ar_p50_levels[59]
original_jump_120 = ar_p50_levels[120] - ar_p50_levels[119]

# Accumulated discontinuities
accumulated_jump_60 = accumulated_levels[60] - accumulated_levels[59]
accumulated_jump_120 = accumulated_levels[120] - accumulated_levels[119]

# These should equal the predicted changes at those points
expected_jump_60 = predicted_changes_from_levels[60]
expected_jump_120 = predicted_changes_from_levels[120]

print(f"Day 60 Boundary:")
print(f"  Original discontinuity:     {original_jump_60:+.6f}")
print(f"  Accumulated discontinuity:  {accumulated_jump_60:+.6f}")
print(f"  Expected (change):          {expected_jump_60:+.6f}")
print(f"  Match: {np.isclose(accumulated_jump_60, expected_jump_60)}")
print()

print(f"Day 120 Boundary:")
print(f"  Original discontinuity:     {original_jump_120:+.6f}")
print(f"  Accumulated discontinuity:  {accumulated_jump_120:+.6f}")
print(f"  Expected (change):          {expected_jump_120:+.6f}")
print(f"  Match: {np.isclose(accumulated_jump_120, expected_jump_120)}")
print()

# Percentage comparison
pct_reduction_60 = 100 * (1 - abs(accumulated_jump_60) / abs(original_jump_60))
pct_reduction_120 = 100 * (1 - abs(accumulated_jump_120) / abs(original_jump_120))

print(f"Discontinuity Reduction:")
print(f"  Day 60:  {pct_reduction_60:+.1f}% reduction")
print(f"  Day 120: {pct_reduction_120:+.1f}% reduction")
print()

if pct_reduction_60 > 10 or pct_reduction_120 > 10:
    print("  ✓ Accumulation REDUCES discontinuities (change-based modeling could help!)")
else:
    print("  ✗ Accumulation does NOT reduce discontinuities (offset chunking is fundamental issue)")

print()

# ============================================================================
# TEST 5: RMSE Comparison
# ============================================================================

print("=" * 80)
print("TEST 5: RMSE Comparison - Level-Based vs Change-Accumulated")
print("=" * 80)
print()

# Compute RMSE for original level predictions
rmse_original = np.sqrt(np.mean((ar_p50_levels - gt_levels)**2))

# Compute RMSE for accumulated change-based predictions
rmse_accumulated = np.sqrt(np.mean((accumulated_levels - gt_levels)**2))

# Relative improvement
pct_improvement = 100 * (rmse_original - rmse_accumulated) / rmse_original

print(f"RMSE Comparison:")
print(f"  Original level-based:      {rmse_original:.6f}")
print(f"  Accumulated change-based:  {rmse_accumulated:.6f}")
print(f"  Improvement:               {pct_improvement:+.2f}%")
print()

if rmse_accumulated < rmse_original:
    print("  ✓ Change-accumulated predictions are MORE accurate!")
    print("  → Suggests change-based modeling could improve performance")
else:
    print("  ✗ Change-accumulated predictions are LESS accurate")
    print("  → Level-based modeling may be optimal despite discontinuities")

print()

# Breakdown by chunk
chunk_1_rmse_original = np.sqrt(np.mean((ar_p50_levels[0:60] - gt_levels[0:60])**2))
chunk_2_rmse_original = np.sqrt(np.mean((ar_p50_levels[60:120] - gt_levels[60:120])**2))
chunk_3_rmse_original = np.sqrt(np.mean((ar_p50_levels[120:180] - gt_levels[120:180])**2))

chunk_1_rmse_accumulated = np.sqrt(np.mean((accumulated_levels[0:60] - gt_levels[0:60])**2))
chunk_2_rmse_accumulated = np.sqrt(np.mean((accumulated_levels[60:120] - gt_levels[60:120])**2))
chunk_3_rmse_accumulated = np.sqrt(np.mean((accumulated_levels[120:180] - gt_levels[120:180])**2))

print("RMSE by Chunk (Original vs Accumulated):")
print(f"  Chunk 1 (days 0-59):    {chunk_1_rmse_original:.6f} → {chunk_1_rmse_accumulated:.6f} ({100*(chunk_1_rmse_accumulated-chunk_1_rmse_original)/chunk_1_rmse_original:+.1f}%)")
print(f"  Chunk 2 (days 60-119):  {chunk_2_rmse_original:.6f} → {chunk_2_rmse_accumulated:.6f} ({100*(chunk_2_rmse_accumulated-chunk_2_rmse_original)/chunk_2_rmse_original:+.1f}%)")
print(f"  Chunk 3 (days 120-179): {chunk_3_rmse_original:.6f} → {chunk_3_rmse_accumulated:.6f} ({100*(chunk_3_rmse_accumulated-chunk_3_rmse_original)/chunk_3_rmse_original:+.1f}%)")
print()

# ============================================================================
# Visualization: Compare Original vs Accumulated
# ============================================================================

print("=" * 80)
print("Creating Comparison Visualization")
print("=" * 80)
print()

fig, axes = plt.subplots(2, 1, figsize=(20, 12))

forecast_days = np.arange(0, AR_HORIZON)

# Panel 1: Level predictions comparison
ax1 = axes[0]

# Ground truth
ax1.plot(forecast_days, gt_levels,
        color='black', linewidth=2, linestyle=':', alpha=0.8, zorder=5,
        label='Ground Truth')

# Original level predictions
ax1.plot(forecast_days, ar_p50_levels,
        color='blue', linewidth=2, linestyle='-', zorder=4,
        label='Original (Level-Based)')

# Accumulated change predictions
ax1.plot(forecast_days, accumulated_levels,
        color='red', linewidth=2, linestyle='--', zorder=4,
        label='Accumulated (Change-Based)')

# Boundary markers
ax1.axvline(x=60, color='purple', linewidth=2, linestyle='--', alpha=0.5, zorder=3)
ax1.axvline(x=120, color='purple', linewidth=2, linestyle='--', alpha=0.5, zorder=3)

# Discontinuity markers
for day in [60, 120]:
    ax1.plot(day, ar_p50_levels[day],
            marker='o', markersize=8, color='blue', markeredgecolor='black',
            markeredgewidth=2, zorder=7)
    ax1.plot(day, accumulated_levels[day],
            marker='s', markersize=8, color='red', markeredgecolor='black',
            markeredgewidth=2, zorder=7)

ax1.set_xlabel('Day', fontsize=12, fontweight='bold')
ax1.set_ylabel('ATM 6M Implied Volatility', fontsize=12, fontweight='bold')
ax1.set_title(f'Levels Comparison: Original vs Accumulated\n{PERIOD.title()} Period - Sequence {SEQ_IDX}',
             fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.2, linestyle='--')

# Add RMSE annotation
rmse_text = f'RMSE: Original={rmse_original:.6f}, Accumulated={rmse_accumulated:.6f} ({pct_improvement:+.1f}%)'
ax1.text(0.02, 0.98, rmse_text,
        transform=ax1.transAxes, ha='left', va='top',
        fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Panel 2: Changes comparison
ax2 = axes[1]

# Ground truth changes
ax2.plot(forecast_days[:-1], gt_changes,
        color='black', linewidth=2, linestyle=':', alpha=0.8, zorder=5,
        label='Ground Truth Changes')

# Predicted changes (from original level predictions)
ax2.plot(forecast_days[:-1], predicted_changes_from_levels,
        color='blue', linewidth=2, linestyle='-', zorder=4,
        label='Predicted Changes (from levels)')

# Boundary markers
ax2.axvline(x=60, color='purple', linewidth=2, linestyle='--', alpha=0.5, zorder=3,
           label='Chunk Boundaries')
ax2.axvline(x=120, color='purple', linewidth=2, linestyle='--', alpha=0.5, zorder=3)

# Highlight change discontinuities
ax2.plot(60, predicted_changes_from_levels[60],
        marker='o', markersize=10, color='red', markeredgecolor='black',
        markeredgewidth=2, zorder=7, label='Change Discontinuities')
ax2.plot(120, predicted_changes_from_levels[120],
        marker='o', markersize=10, color='red', markeredgecolor='black',
        markeredgewidth=2, zorder=7)

ax2.set_xlabel('Day', fontsize=12, fontweight='bold')
ax2.set_ylabel('Day-to-Day Change', fontsize=12, fontweight='bold')
ax2.set_title('Changes Comparison: Do Discontinuities Exist in Changes?',
             fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.axhline(y=0, color='gray', linewidth=1, linestyle='-', alpha=0.3)

# Add discontinuity annotation
disc_text = (f'Day 60 change jump: {change_jump_60:+.6f}\n'
            f'Day 120 change jump: {change_jump_120:+.6f}')
ax2.text(0.02, 0.98, disc_text,
        transform=ax2.transAxes, ha='left', va='top',
        fontsize=10, fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()

# ============================================================================
# Save
# ============================================================================

output_dir = Path("results/context60_baseline/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "levels_vs_changes_comparison.png"

print(f"Saving visualization...")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_file}")
print(f"  File size: {output_file.stat().st_size / 1024**2:.1f} MB")

plt.close()

print()

# ============================================================================
# Final Summary
# ============================================================================

print("=" * 80)
print("HYPOTHESIS TEST SUMMARY")
print("=" * 80)
print()

print("Question: Are AR discontinuities caused by modeling LEVELS vs CHANGES?")
print()

print("Key Findings:")
print()

print("1. Discontinuities in Changes vs Levels:")
print(f"   - Level discontinuity at day 60:  {original_jump_60:+.6f}")
print(f"   - Change discontinuity at day 60: {change_jump_60:+.6f}")
print(f"   - Acceleration at day 60:         {accel_60:+.6f}")
if abs(accel_60) > 0.001:
    print("   → BOTH levels AND changes show discontinuities!")
else:
    print("   → Only levels show discontinuities (changes are smooth)")
print()

print("2. Accumulated Change-Based Predictions:")
print(f"   - Mean difference from original: {diff_original_accumulated.mean():.6f}")
print(f"   - Discontinuity reduction at day 60:  {pct_reduction_60:+.1f}%")
print(f"   - Discontinuity reduction at day 120: {pct_reduction_120:+.1f}%")
if abs(pct_reduction_60) < 5 and abs(pct_reduction_120) < 5:
    print("   → Accumulation does NOT eliminate discontinuities")
else:
    print("   → Accumulation REDUCES discontinuities")
print()

print("3. RMSE Comparison:")
print(f"   - Original level-based:      {rmse_original:.6f}")
print(f"   - Accumulated change-based:  {rmse_accumulated:.6f}")
print(f"   - Improvement:               {pct_improvement:+.2f}%")
if rmse_accumulated < rmse_original:
    print("   → Change-based approach is MORE accurate")
else:
    print("   → Level-based approach is MORE accurate")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

# Determine root cause based on evidence
if abs(accel_60) > 0.001 and abs(accel_120) > 0.001:
    print("Root Cause: BOTH level-based modeling AND offset chunking contribute")
    print()
    print("Evidence:")
    print("  - Changes show discontinuities (not just levels)")
    print("  - Acceleration spikes at boundaries indicate trend shifts")
    print("  - Offset chunking creates independent predictions at boundaries")
    print()
    print("Recommended Solutions:")
    print("  1. Change-based decoder with continuity constraints")
    print("  2. Overlapping windows with weighted averaging")
    print("  3. Continuous day-by-day AR (slower but no discontinuities)")
else:
    print("Root Cause: Offset chunking is the primary issue")
    print()
    print("Evidence:")
    print("  - Changes are smooth (only levels show discontinuities)")
    print("  - Problem is NOT inherent to level-based modeling")
    print("  - Issue is offset-based generation creating independent chunks")
    print()
    print("Recommended Solutions:")
    print("  1. Overlapping windows with weighted averaging")
    print("  2. Continuous day-by-day AR")
    print("  3. Add loss term penalizing boundary discontinuities")

print()
print(f"Visualization saved: {output_file}")
print()
