"""
Evaluate CI calibration for backfill_16yr in-sample reconstructions.

Computes CI violation rates with breakdown by:
- Overall (all 4000 training days)
- Crisis period (2008-2010)
- Normal periods (pre-2008 + post-2010)
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("CI CALIBRATION EVALUATION - backfill_16yr")
print("=" * 80)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load reconstructions
recon_file = "models/backfill/insample_reconstruction_16yr.npz"
recon_data = np.load(recon_file)

print(f"Loaded: {recon_file}")
print("Contents:")
for key in recon_data.files:
    print(f"  {key}: {recon_data[key].shape}")
print()

# Load ground truth
gt_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_gt = gt_data["surface"]

# ============================================================================
# Define Regime Periods
# ============================================================================

# Crisis period: 2008-2010 (roughly indices 2000-2765)
crisis_start = 2000
crisis_end = 2765

print("Regime definitions:")
print(f"  Crisis period: indices [{crisis_start}, {crisis_end}]")
print(f"  Normal periods: < {crisis_start} or > {crisis_end}")
print()

# ============================================================================
# Evaluate Each Horizon
# ============================================================================

horizons = [1, 7, 14, 30]
results = []

for horizon in horizons:
    print("=" * 80)
    print(f"HORIZON = {horizon} days")
    print("=" * 80)

    # Load reconstruction and indices
    recon_key = f'recon_h{horizon}'
    indices_key = f'indices_h{horizon}'

    recons = recon_data[recon_key]  # (N, 3, 5, 5)
    indices = recon_data[indices_key]  # (N,)

    num_samples = len(indices)
    print(f"Number of samples: {num_samples}")
    print(f"Date range: [{indices[0]}, {indices[-1]}]")
    print()

    # Extract quantiles
    p05 = recons[:, 0, :, :]  # (N, 5, 5)
    p50 = recons[:, 1, :, :]  # (N, 5, 5)
    p95 = recons[:, 2, :, :]  # (N, 5, 5)

    # Extract ground truth
    gt = vol_surf_gt[indices]  # (N, 5, 5)

    # ========================================================================
    # Overall Statistics
    # ========================================================================

    print("OVERALL STATISTICS")
    print("-" * 80)

    # CI violations
    below_p05 = gt < p05
    above_p95 = gt > p95
    outside_ci = below_p05 | above_p95

    num_violations = np.sum(outside_ci)
    total_points = outside_ci.size
    pct_violations = 100 * num_violations / total_points

    num_below = np.sum(below_p05)
    num_above = np.sum(above_p95)
    pct_below = 100 * num_below / total_points
    pct_above = 100 * num_above / total_points

    # RMSE
    rmse = np.sqrt(np.mean((p50 - gt) ** 2))

    # CI width
    ci_width = np.mean(p95 - p05)

    print(f"Total points: {total_points:,}")
    print(f"CI violations: {num_violations:,} ({pct_violations:.2f}%)")
    print(f"  Below p05: {num_below:,} ({pct_below:.2f}%)")
    print(f"  Above p95: {num_above:,} ({pct_above:.2f}%)")
    print(f"RMSE (p50): {rmse:.6f}")
    print(f"Mean CI width: {ci_width:.6f}")
    print()

    # ========================================================================
    # Regime Breakdown
    # ========================================================================

    print("REGIME BREAKDOWN")
    print("-" * 80)

    # Identify regime for each sample
    is_crisis = (indices >= crisis_start) & (indices <= crisis_end)
    is_normal = ~is_crisis

    num_crisis = np.sum(is_crisis)
    num_normal = np.sum(is_normal)

    print(f"Crisis samples: {num_crisis}")
    print(f"Normal samples: {num_normal}")
    print()

    # Crisis statistics
    if num_crisis > 0:
        crisis_violations = np.sum(outside_ci[is_crisis])
        crisis_total = outside_ci[is_crisis].size
        crisis_pct = 100 * crisis_violations / crisis_total
        crisis_rmse = np.sqrt(np.mean((p50[is_crisis] - gt[is_crisis]) ** 2))
        crisis_ci_width = np.mean(p95[is_crisis] - p05[is_crisis])

        print("Crisis Period:")
        print(f"  CI violations: {crisis_violations:,}/{crisis_total:,} ({crisis_pct:.2f}%)")
        print(f"  RMSE: {crisis_rmse:.6f}")
        print(f"  Mean CI width: {crisis_ci_width:.6f}")
    else:
        crisis_pct = np.nan
        crisis_rmse = np.nan
        crisis_ci_width = np.nan
        print("Crisis Period: No samples")

    print()

    # Normal statistics
    if num_normal > 0:
        normal_violations = np.sum(outside_ci[is_normal])
        normal_total = outside_ci[is_normal].size
        normal_pct = 100 * normal_violations / normal_total
        normal_rmse = np.sqrt(np.mean((p50[is_normal] - gt[is_normal]) ** 2))
        normal_ci_width = np.mean(p95[is_normal] - p05[is_normal])

        print("Normal Periods:")
        print(f"  CI violations: {normal_violations:,}/{normal_total:,} ({normal_pct:.2f}%)")
        print(f"  RMSE: {normal_rmse:.6f}")
        print(f"  Mean CI width: {normal_ci_width:.6f}")
    else:
        normal_pct = np.nan
        normal_rmse = np.nan
        normal_ci_width = np.nan
        print("Normal Periods: No samples")

    print()

    # ========================================================================
    # Per-Grid-Point Analysis
    # ========================================================================

    print("PER-GRID-POINT VIOLATIONS (Overall)")
    print("-" * 80)

    # Compute violations per grid point
    grid_violations = np.mean(outside_ci, axis=0) * 100  # (5, 5)

    # Display as table
    print("Violation % by grid point (row=moneyness, col=maturity):")
    print()
    for i in range(5):
        row_str = "  "
        for j in range(5):
            row_str += f"{grid_violations[i, j]:5.1f}% "
        print(row_str)
    print()

    # Find worst grid points
    worst_grid = np.unravel_index(np.argmax(grid_violations), grid_violations.shape)
    best_grid = np.unravel_index(np.argmin(grid_violations), grid_violations.shape)

    print(f"Worst grid point: {worst_grid} ({grid_violations[worst_grid]:.2f}%)")
    print(f"Best grid point: {best_grid} ({grid_violations[best_grid]:.2f}%)")
    print()

    # Store results
    results.append({
        'horizon': horizon,
        'overall_violations': pct_violations,
        'overall_rmse': rmse,
        'overall_ci_width': ci_width,
        'crisis_violations': crisis_pct,
        'crisis_rmse': crisis_rmse,
        'crisis_ci_width': crisis_ci_width,
        'normal_violations': normal_pct,
        'normal_rmse': normal_rmse,
        'normal_ci_width': normal_ci_width,
        'num_crisis_samples': num_crisis,
        'num_normal_samples': num_normal,
    })

# ============================================================================
# Summary Table
# ============================================================================

print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print()

df = pd.DataFrame(results)
print(df.to_string(index=False))
print()

# Save to CSV
output_csv = "models/backfill/ci_calibration_16yr.csv"
df.to_csv(output_csv, index=False)
print(f"Saved to: {output_csv}")
print()

# ============================================================================
# Final Assessment
# ============================================================================

print("=" * 80)
print("ASSESSMENT")
print("=" * 80)
print()

target_violations = 10.0  # Target CI violation rate

print("Target CI violations: ~10%")
print()

for _, row in df.iterrows():
    h = row['horizon']
    overall = row['overall_violations']
    crisis = row['crisis_violations']
    normal = row['normal_violations']

    print(f"Horizon {h}:")

    # Overall assessment
    if overall <= 15:
        status = "✓ GOOD"
    elif overall <= 20:
        status = "⚠ ACCEPTABLE"
    else:
        status = "✗ POOR"
    print(f"  Overall: {overall:.2f}% {status}")

    # Crisis assessment
    if not np.isnan(crisis):
        if crisis <= 20:
            crisis_status = "✓ GOOD"
        elif crisis <= 30:
            crisis_status = "⚠ ACCEPTABLE (crisis is harder)"
        else:
            crisis_status = "✗ POOR"
        print(f"  Crisis: {crisis:.2f}% {crisis_status}")

    # Normal assessment
    if not np.isnan(normal):
        if normal <= 12:
            normal_status = "✓ GOOD"
        elif normal <= 18:
            normal_status = "⚠ ACCEPTABLE"
        else:
            normal_status = "✗ POOR"
        print(f"  Normal: {normal:.2f}% {normal_status}")

    print()

print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
