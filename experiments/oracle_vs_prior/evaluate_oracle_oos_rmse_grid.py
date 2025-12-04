"""
Evaluate Oracle OOS RMSE with grid-level statistics.
Uses existing oos_reconstruction_16yr.npz (posterior sampling = Oracle).
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("ORACLE OOS RMSE WITH GRID-LEVEL STATISTICS")
print("=" * 80)
print()

# Load OOS reconstruction data (this is Oracle/posterior sampling)
recon_file = "models/backfill/oos_reconstruction_16yr.npz"
recon_data = np.load(recon_file)

# Load ground truth
gt_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_gt = gt_data["surface"]

horizons = [1, 7, 14, 30]
summary_results = []

for horizon in horizons:
    # Load reconstruction and indices
    recons = recon_data[f'recon_h{horizon}']  # (N, 3, 5, 5)
    indices = recon_data[f'indices_h{horizon}']  # (N,)

    # Extract p50 (median) and ground truth
    p50 = recons[:, 1, :, :]  # (N, 5, 5)
    gt = vol_surf_gt[indices]  # (N, 5, 5)

    # Compute grid-level RMSE
    grid_rmse = np.sqrt(np.mean((p50 - gt) ** 2, axis=0))  # (5, 5)

    # Statistics
    average_rmse = np.mean(grid_rmse)
    best_idx = np.unravel_index(np.argmin(grid_rmse), grid_rmse.shape)
    worst_idx = np.unravel_index(np.argmax(grid_rmse), grid_rmse.shape)
    best_rmse = grid_rmse[best_idx]
    worst_rmse = grid_rmse[worst_idx]

    print(f"H{horizon}: Avg={average_rmse:.6f}, Best={best_rmse:.6f}, Worst={worst_rmse:.6f}")

    summary_results.append({
        'horizon': horizon,
        'period': 'OOS (2019-2023)',
        'days': len(indices),
        'average_rmse': average_rmse,
        'best_rmse': best_rmse,
        'worst_rmse': worst_rmse,
    })

# Save
summary_df = pd.DataFrame(summary_results)
summary_csv = "models/backfill/oracle_oos_rmse_summary.csv"
summary_df.to_csv(summary_csv, index=False)

# Print table
print()
print("Table Row: Oracle OOS")
h1 = summary_df[summary_df['horizon'] == 1].iloc[0]
h7 = summary_df[summary_df['horizon'] == 7].iloc[0]
h14 = summary_df[summary_df['horizon'] == 14].iloc[0]
h30 = summary_df[summary_df['horizon'] == 30].iloc[0]

avg_rmse = np.mean([h1['average_rmse'], h7['average_rmse'], h14['average_rmse'], h30['average_rmse']])
avg_best = np.mean([h1['best_rmse'], h7['best_rmse'], h14['best_rmse'], h30['best_rmse']])
avg_worst = np.mean([h1['worst_rmse'], h7['worst_rmse'], h14['worst_rmse'], h30['worst_rmse']])

print(f"| OOS (2019-2023) | {h1['average_rmse']:.4f} | {h7['average_rmse']:.4f} | {h14['average_rmse']:.4f} | {h30['average_rmse']:.4f} | **{avg_rmse:.4f}** | {avg_best:.4f} | {avg_worst:.4f} | {int(h1['days'])} |")
print()
print(f"Saved to: {summary_csv}")
