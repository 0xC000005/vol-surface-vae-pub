"""
Evaluate Oracle RMSE for backfill_16yr in-sample reconstructions with COMPLETE grid-level statistics.

Computes point forecast accuracy using p50 (median) quantile with breakdown by:
- Overall (all 4000 training days)
- Crisis period (2008-2010)
- Normal periods (pre-2008 + post-2010)
- Complete per-grid-point analysis with best/worst statistics

Exports comprehensive CSV with format matching CI violation tables.
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("ORACLE RMSE EVALUATION WITH GRID-LEVEL STATISTICS - backfill_16yr")
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
# Evaluate Each Horizon with Complete Grid Statistics
# ============================================================================

horizons = [1, 7, 14, 30]
summary_results = []
grid_results = []

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

    # Identify regimes
    is_crisis = (indices >= crisis_start) & (indices <= crisis_end)
    is_normal = ~is_crisis

    num_crisis = np.sum(is_crisis)
    num_normal = np.sum(is_normal)

    # ========================================================================
    # Compute Grid-Level RMSE for Each Regime
    # ========================================================================

    # Overall grid RMSE
    grid_rmse_overall = np.sqrt(np.mean((p50 - gt) ** 2, axis=0))  # (5, 5)

    # Crisis grid RMSE
    if num_crisis > 0:
        grid_rmse_crisis = np.sqrt(np.mean((p50[is_crisis] - gt[is_crisis]) ** 2, axis=0))  # (5, 5)
    else:
        grid_rmse_crisis = np.full((5, 5), np.nan)

    # Normal grid RMSE
    if num_normal > 0:
        grid_rmse_normal = np.sqrt(np.mean((p50[is_normal] - gt[is_normal]) ** 2, axis=0))  # (5, 5)
    else:
        grid_rmse_normal = np.full((5, 5), np.nan)

    # ========================================================================
    # Compute Statistics for Each Regime
    # ========================================================================

    def compute_stats(grid_rmse, regime_name):
        """Compute average, best, worst for a grid RMSE matrix."""
        if np.all(np.isnan(grid_rmse)):
            return {
                'regime': regime_name,
                'average_rmse': np.nan,
                'best_rmse': np.nan,
                'worst_rmse': np.nan,
                'best_grid_row': -1,
                'best_grid_col': -1,
                'worst_grid_row': -1,
                'worst_grid_col': -1,
            }

        # Average across all grid points
        average_rmse = np.mean(grid_rmse)

        # Best and worst
        best_idx = np.unravel_index(np.argmin(grid_rmse), grid_rmse.shape)
        worst_idx = np.unravel_index(np.argmax(grid_rmse), grid_rmse.shape)

        return {
            'regime': regime_name,
            'average_rmse': average_rmse,
            'best_rmse': grid_rmse[best_idx],
            'worst_rmse': grid_rmse[worst_idx],
            'best_grid_row': best_idx[0],
            'best_grid_col': best_idx[1],
            'worst_grid_row': worst_idx[0],
            'worst_grid_col': worst_idx[1],
        }

    # Compute stats for each regime
    overall_stats = compute_stats(grid_rmse_overall, 'overall')
    crisis_stats = compute_stats(grid_rmse_crisis, 'crisis')
    normal_stats = compute_stats(grid_rmse_normal, 'normal')

    # ========================================================================
    # Print Summary
    # ========================================================================

    print(f"Overall (all {num_samples} samples):")
    print(f"  Average RMSE: {overall_stats['average_rmse']:.6f}")
    print(f"  Best grid: ({overall_stats['best_grid_row']}, {overall_stats['best_grid_col']}) - RMSE: {overall_stats['best_rmse']:.6f}")
    print(f"  Worst grid: ({overall_stats['worst_grid_row']}, {overall_stats['worst_grid_col']}) - RMSE: {overall_stats['worst_rmse']:.6f}")
    print()

    print(f"Crisis ({num_crisis} samples):")
    if num_crisis > 0:
        print(f"  Average RMSE: {crisis_stats['average_rmse']:.6f}")
        print(f"  Best grid: ({crisis_stats['best_grid_row']}, {crisis_stats['best_grid_col']}) - RMSE: {crisis_stats['best_rmse']:.6f}")
        print(f"  Worst grid: ({crisis_stats['worst_grid_row']}, {crisis_stats['worst_grid_col']}) - RMSE: {crisis_stats['worst_rmse']:.6f}")
    else:
        print("  No crisis samples")
    print()

    print(f"Normal ({num_normal} samples):")
    if num_normal > 0:
        print(f"  Average RMSE: {normal_stats['average_rmse']:.6f}")
        print(f"  Best grid: ({normal_stats['best_grid_row']}, {normal_stats['best_grid_col']}) - RMSE: {normal_stats['best_rmse']:.6f}")
        print(f"  Worst grid: ({normal_stats['worst_grid_row']}, {normal_stats['worst_grid_col']}) - RMSE: {normal_stats['worst_rmse']:.6f}")
    else:
        print("  No normal samples")
    print()

    # ========================================================================
    # Store Results
    # ========================================================================

    # Summary row for this horizon
    summary_results.append({
        'horizon': horizon,
        'period': 'In-Sample (2004-2019)',
        'days': num_samples,
        'overall_rmse': overall_stats['average_rmse'],
        'overall_best': overall_stats['best_rmse'],
        'overall_worst': overall_stats['worst_rmse'],
        'crisis_rmse': crisis_stats['average_rmse'],
        'crisis_best': crisis_stats['best_rmse'],
        'crisis_worst': crisis_stats['worst_rmse'],
        'crisis_days': num_crisis,
        'normal_rmse': normal_stats['average_rmse'],
        'normal_best': normal_stats['best_rmse'],
        'normal_worst': normal_stats['worst_rmse'],
        'normal_days': num_normal,
    })

    # Per-grid results for detailed analysis
    for i in range(5):
        for j in range(5):
            grid_results.append({
                'horizon': horizon,
                'grid_row': i,
                'grid_col': j,
                'overall_rmse': grid_rmse_overall[i, j],
                'crisis_rmse': grid_rmse_crisis[i, j],
                'normal_rmse': grid_rmse_normal[i, j],
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
summary_csv = "models_backfill/oracle_rmse_summary_with_grid_stats.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Summary saved to: {summary_csv}")
print()
print(summary_df.to_string(index=False))
print()

# Grid-level table
grid_df = pd.DataFrame(grid_results)
grid_csv = "models_backfill/oracle_rmse_grid_level.csv"
grid_df.to_csv(grid_csv, index=False)
print(f"Grid-level data saved to: {grid_csv}")
print()

# ============================================================================
# Create Presentation-Ready Table
# ============================================================================

print("=" * 80)
print("PRESENTATION-READY TABLES")
print("=" * 80)
print()

# Format for MILESTONE_PRESENTATION.md (matching CI violation tables)
print("Table 4: Oracle (Ground Truth Latent) - RMSE by Horizon")
print()
print("| Period | H1 | H7 | H14 | H30 | Average | Best Grid | Worst Grid | Days |")
print("|--------|-----|-----|------|------|---------|-----------|------------|------|")

# In-Sample Overall
h1 = summary_df[summary_df['horizon'] == 1].iloc[0]
h7 = summary_df[summary_df['horizon'] == 7].iloc[0]
h14 = summary_df[summary_df['horizon'] == 14].iloc[0]
h30 = summary_df[summary_df['horizon'] == 30].iloc[0]

avg_overall = np.mean([h1['overall_rmse'], h7['overall_rmse'], h14['overall_rmse'], h30['overall_rmse']])
avg_best = np.mean([h1['overall_best'], h7['overall_best'], h14['overall_best'], h30['overall_best']])
avg_worst = np.mean([h1['overall_worst'], h7['overall_worst'], h14['overall_worst'], h30['overall_worst']])

print(f"| In-Sample (2004-2019) | {h1['overall_rmse']:.4f} | {h7['overall_rmse']:.4f} | {h14['overall_rmse']:.4f} | {h30['overall_rmse']:.4f} | **{avg_overall:.4f}** | {avg_best:.4f} | {avg_worst:.4f} | {int(h1['days'])} |")

# Crisis
avg_crisis = np.mean([h1['crisis_rmse'], h7['crisis_rmse'], h14['crisis_rmse'], h30['crisis_rmse']])
avg_crisis_best = np.mean([h1['crisis_best'], h7['crisis_best'], h14['crisis_best'], h30['crisis_best']])
avg_crisis_worst = np.mean([h1['crisis_worst'], h7['crisis_worst'], h14['crisis_worst'], h30['crisis_worst']])

print(f"| Crisis (2008-2010) | {h1['crisis_rmse']:.4f} | {h7['crisis_rmse']:.4f} | {h14['crisis_rmse']:.4f} | {h30['crisis_rmse']:.4f} | **{avg_crisis:.4f}** | {avg_crisis_best:.4f} | {avg_crisis_worst:.4f} | {int(h1['crisis_days'])} |")

print()
print("Note: OOS data will be computed separately")
print()

print("=" * 80)
print("ORACLE RMSE EVALUATION WITH GRID STATS COMPLETE")
print("=" * 80)
