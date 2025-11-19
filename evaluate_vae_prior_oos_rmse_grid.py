"""
Evaluate VAE Prior (Oracle/Posterior) OOS RMSE with complete grid-level statistics.

Computes RMSE from existing oos_reconstruction_16yr.npz with breakdown by:
- Overall OOS period (2019-2023)
- Complete per-grid-point analysis with best/worst statistics
- Formatted for MILESTONE_PRESENTATION.md tables
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("VAE PRIOR OOS RMSE WITH GRID-LEVEL STATISTICS")
print("=" * 80)
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading OOS reconstruction data...")
recon_file = "models_backfill/oos_reconstruction_16yr.npz"
recon_data = np.load(recon_file)

print(f"Loaded: {recon_file}")
print("Contents:")
for key in recon_data.files:
    print(f"  {key}: {recon_data[key].shape}")
print()

# Load ground truth
gt_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_gt = gt_data["surface"]

print("✓ Data loaded")
print()

# ============================================================================
# Compute RMSE with Grid-Level Statistics
# ============================================================================

horizons = [1, 7, 14, 30]
summary_results = []
grid_results = []

for horizon in horizons:
    print(f"{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

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
    # Compute Grid-Level RMSE
    # ========================================================================

    # Overall grid RMSE
    grid_rmse = np.sqrt(np.mean((p50 - gt) ** 2, axis=0))  # (5, 5)

    # ========================================================================
    # Compute Statistics
    # ========================================================================

    # Average across all grid points
    average_rmse = np.mean(grid_rmse)

    # Best and worst
    best_idx = np.unravel_index(np.argmin(grid_rmse), grid_rmse.shape)
    worst_idx = np.unravel_index(np.argmax(grid_rmse), grid_rmse.shape)

    best_rmse = grid_rmse[best_idx]
    worst_rmse = grid_rmse[worst_idx]

    # Print summary
    print(f"Overall ({num_samples} samples):")
    print(f"  Average RMSE: {average_rmse:.6f}")
    print(f"  Best grid: ({best_idx[0]}, {best_idx[1]}) - RMSE: {best_rmse:.6f}")
    print(f"  Worst grid: ({worst_idx[0]}, {worst_idx[1]}) - RMSE: {worst_rmse:.6f}")
    print()

    # ========================================================================
    # Store Results
    # ========================================================================

    # Summary row for this horizon
    summary_results.append({
        'horizon': horizon,
        'period': 'OOS (2019-2023)',
        'days': num_samples,
        'average_rmse': average_rmse,
        'best_rmse': best_rmse,
        'worst_rmse': worst_rmse,
        'best_grid_row': best_idx[0],
        'best_grid_col': best_idx[1],
        'worst_grid_row': worst_idx[0],
        'worst_grid_col': worst_idx[1],
    })

    # Per-grid results for detailed analysis
    for i in range(5):
        for j in range(5):
            grid_results.append({
                'horizon': horizon,
                'grid_row': i,
                'grid_col': j,
                'rmse': grid_rmse[i, j],
            })

# ============================================================================
# Save Results to CSV
# ============================================================================

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Summary table
summary_df = pd.DataFrame(summary_results)
summary_csv = "models_backfill/vae_prior_oos_rmse_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Summary saved to: {summary_csv}")
print()
print(summary_df.to_string(index=False))
print()

# Grid-level table
grid_df = pd.DataFrame(grid_results)
grid_csv = "models_backfill/vae_prior_oos_rmse_grid.csv"
grid_df.to_csv(grid_csv, index=False)
print(f"Grid-level data saved to: {grid_csv}")
print()

# ============================================================================
# Presentation-Ready Table
# ============================================================================

print("=" * 80)
print("PRESENTATION-READY TABLE")
print("=" * 80)
print()

print("Table (OOS Portion): VAE Prior (z~N(0,1)) - RMSE by Horizon")
print()
print("| Period | H1 | H7 | H14 | H30 | Average | Best Grid | Worst Grid | Days |")
print("|--------|-----|-----|------|------|---------|-----------|------------|------|")

h1 = summary_df[summary_df['horizon'] == 1].iloc[0]
h7 = summary_df[summary_df['horizon'] == 7].iloc[0]
h14 = summary_df[summary_df['horizon'] == 14].iloc[0]
h30 = summary_df[summary_df['horizon'] == 30].iloc[0]

# Average across horizons
avg_rmse = np.mean([h1['average_rmse'], h7['average_rmse'], h14['average_rmse'], h30['average_rmse']])
avg_best = np.mean([h1['best_rmse'], h7['best_rmse'], h14['best_rmse'], h30['best_rmse']])
avg_worst = np.mean([h1['worst_rmse'], h7['worst_rmse'], h14['worst_rmse'], h30['worst_rmse']])

print(f"| OOS (2019-2023) | {h1['average_rmse']:.4f} | {h7['average_rmse']:.4f} | {h14['average_rmse']:.4f} | {h30['average_rmse']:.4f} | **{avg_rmse:.4f}** | {avg_best:.4f} | {avg_worst:.4f} | {int(h1['days'])} |")

print()
print("Note: This is Oracle (posterior sampling) data. For true VAE Prior z~N(0,1) OOS, need to regenerate.")
print()

print("=" * 80)
print("VAE PRIOR OOS RMSE GRID STATS COMPLETE")
print("=" * 80)
