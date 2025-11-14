"""
Evaluate RMSE for backfill_16yr in-sample reconstructions.

Computes point forecast accuracy using p50 (median) quantile with breakdown by:
- Overall (all 4000 training days)
- Crisis period (2008-2010)
- Normal periods (pre-2008 + post-2010)
- Per-grid-point analysis
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("RMSE EVALUATION - backfill_16yr")
print("=" * 80)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load reconstructions
recon_file = "models_backfill/insample_reconstruction_16yr.npz"
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

    # Extract p50 (median) and ground truth
    p50 = recons[:, 1, :, :]  # (N, 5, 5)
    gt = vol_surf_gt[indices]  # (N, 5, 5)

    # ========================================================================
    # Overall Statistics
    # ========================================================================

    print("OVERALL STATISTICS")
    print("-" * 80)

    # RMSE
    overall_rmse = np.sqrt(np.mean((p50 - gt) ** 2))

    # MAE
    overall_mae = np.mean(np.abs(p50 - gt))

    # Max absolute error
    overall_max_err = np.max(np.abs(p50 - gt))

    # Mean absolute percentage error (avoid division by zero)
    mape_mask = gt > 0.01  # Only compute MAPE where gt > 1%
    overall_mape = np.mean(np.abs((p50[mape_mask] - gt[mape_mask]) / gt[mape_mask])) * 100

    print(f"RMSE: {overall_rmse:.6f}")
    print(f"MAE: {overall_mae:.6f}")
    print(f"Max Error: {overall_max_err:.6f}")
    print(f"MAPE: {overall_mape:.2f}%")
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
        crisis_rmse = np.sqrt(np.mean((p50[is_crisis] - gt[is_crisis]) ** 2))
        crisis_mae = np.mean(np.abs(p50[is_crisis] - gt[is_crisis]))
        crisis_mape_mask = gt[is_crisis] > 0.01
        crisis_mape = np.mean(np.abs((p50[is_crisis][crisis_mape_mask] - gt[is_crisis][crisis_mape_mask]) / gt[is_crisis][crisis_mape_mask])) * 100

        print("Crisis Period:")
        print(f"  RMSE: {crisis_rmse:.6f}")
        print(f"  MAE: {crisis_mae:.6f}")
        print(f"  MAPE: {crisis_mape:.2f}%")
    else:
        crisis_rmse = np.nan
        crisis_mae = np.nan
        crisis_mape = np.nan
        print("Crisis Period: No samples")

    print()

    # Normal statistics
    if num_normal > 0:
        normal_rmse = np.sqrt(np.mean((p50[is_normal] - gt[is_normal]) ** 2))
        normal_mae = np.mean(np.abs(p50[is_normal] - gt[is_normal]))
        normal_mape_mask = gt[is_normal] > 0.01
        normal_mape = np.mean(np.abs((p50[is_normal][normal_mape_mask] - gt[is_normal][normal_mape_mask]) / gt[is_normal][normal_mape_mask])) * 100

        print("Normal Periods:")
        print(f"  RMSE: {normal_rmse:.6f}")
        print(f"  MAE: {normal_mae:.6f}")
        print(f"  MAPE: {normal_mape:.2f}%")
    else:
        normal_rmse = np.nan
        normal_mae = np.nan
        normal_mape = np.nan
        print("Normal Periods: No samples")

    print()

    # ========================================================================
    # Per-Grid-Point Analysis
    # ========================================================================

    print("PER-GRID-POINT RMSE (Overall)")
    print("-" * 80)

    # Compute RMSE per grid point
    grid_rmse = np.sqrt(np.mean((p50 - gt) ** 2, axis=0))  # (5, 5)

    # Display as table
    print("RMSE by grid point (row=moneyness, col=maturity):")
    print()
    print("       1M      3M      6M      1Y      2Y")
    moneyness_labels = ["OTM Put", "ATM Put", "ATM", "ATM Call", "OTM Call"]
    for i in range(5):
        row_str = f"{moneyness_labels[i]:8s} "
        for j in range(5):
            row_str += f"{grid_rmse[i, j]:.4f}  "
        print(row_str)
    print()

    # Find worst/best grid points
    worst_grid = np.unravel_index(np.argmax(grid_rmse), grid_rmse.shape)
    best_grid = np.unravel_index(np.argmin(grid_rmse), grid_rmse.shape)

    print(f"Worst grid point: {worst_grid} ({moneyness_labels[worst_grid[0]]}, col {worst_grid[1]}) - RMSE: {grid_rmse[worst_grid]:.6f}")
    print(f"Best grid point: {best_grid} ({moneyness_labels[best_grid[0]]}, col {best_grid[1]}) - RMSE: {grid_rmse[best_grid]:.6f}")
    print()

    # Store results
    results.append({
        'horizon': horizon,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'overall_mape': overall_mape,
        'crisis_rmse': crisis_rmse,
        'crisis_mae': crisis_mae,
        'crisis_mape': crisis_mape,
        'normal_rmse': normal_rmse,
        'normal_mae': normal_mae,
        'normal_mape': normal_mape,
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
output_csv = "models_backfill/rmse_16yr.csv"
df.to_csv(output_csv, index=False)
print(f"Saved to: {output_csv}")
print()

# ============================================================================
# Trend Analysis
# ============================================================================

print("=" * 80)
print("TREND ANALYSIS")
print("=" * 80)
print()

print("RMSE vs Horizon:")
print(f"  H1:  {df.loc[df['horizon'] == 1, 'overall_rmse'].values[0]:.6f}")
print(f"  H7:  {df.loc[df['horizon'] == 7, 'overall_rmse'].values[0]:.6f} (+{(df.loc[df['horizon'] == 7, 'overall_rmse'].values[0] / df.loc[df['horizon'] == 1, 'overall_rmse'].values[0] - 1) * 100:.1f}%)")
print(f"  H14: {df.loc[df['horizon'] == 14, 'overall_rmse'].values[0]:.6f} (+{(df.loc[df['horizon'] == 14, 'overall_rmse'].values[0] / df.loc[df['horizon'] == 1, 'overall_rmse'].values[0] - 1) * 100:.1f}%)")
print(f"  H30: {df.loc[df['horizon'] == 30, 'overall_rmse'].values[0]:.6f} (+{(df.loc[df['horizon'] == 30, 'overall_rmse'].values[0] / df.loc[df['horizon'] == 1, 'overall_rmse'].values[0] - 1) * 100:.1f}%)")
print()

print("Crisis vs Normal RMSE:")
for _, row in df.iterrows():
    h = int(row['horizon'])
    crisis_rmse = row['crisis_rmse']
    normal_rmse = row['normal_rmse']
    if not np.isnan(crisis_rmse) and not np.isnan(normal_rmse):
        diff_pct = (crisis_rmse / normal_rmse - 1) * 100
        status = "Crisis worse" if diff_pct > 0 else "Normal worse"
        print(f"  H{h:2d}: Crisis={crisis_rmse:.6f}, Normal={normal_rmse:.6f} ({status}, {abs(diff_pct):.1f}% diff)")
print()

print("=" * 80)
print("RMSE EVALUATION COMPLETE")
print("=" * 80)
