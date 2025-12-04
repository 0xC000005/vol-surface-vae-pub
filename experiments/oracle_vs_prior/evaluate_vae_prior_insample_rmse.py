"""
Evaluate VAE Prior (z~N(0,1)) In-Sample RMSE with complete grid-level statistics.

Generates In-Sample predictions using standard normal prior z ~ N(0,1) instead of
posterior sampling, matching realistic deployment conditions.

Computes RMSE breakdown by:
- Overall (all ~4000 in-sample days)
- Crisis period (2008-2010)
- Normal periods (pre-2008 + post-2010)
- Complete per-grid-point analysis with best/worst statistics
"""
import numpy as np
import pandas as pd
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds
from tqdm import tqdm

# Set seeds and dtype
set_seeds(0)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("VAE PRIOR (z~N(0,1)) IN-SAMPLE RMSE EVALUATION")
print("=" * 80)
print()

# ============================================================================
# Load Model
# ============================================================================

print("Loading model...")
model_file = "models/backfill/backfill_16yr.pt"
model_data = torch.load(model_file, weights_only=False)
model_config = model_data["model_config"]

print(f"Model config:")
print(f"  Context length: {model_config['context_len']}")
print(f"  Latent dim: {model_config['latent_dim']}")
print(f"  Horizon (training): {model_config['horizon']}")
print(f"  Quantiles: {model_config['quantiles']}")
print()

model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
print("✓ Model loaded")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
skew_data = data["skews"]
slope_data = data["slopes"]

# Concatenate extra features [return, skew, slope]
ex_data = np.concatenate([
    ret_data[..., np.newaxis],
    skew_data[..., np.newaxis],
    slope_data[..., np.newaxis]
], axis=-1)

print(f"  Total data shape: {vol_surf_data.shape}")
print(f"  Extra features shape: {ex_data.shape}")
print()

# ============================================================================
# In-Sample Configuration
# ============================================================================

# In-sample period (matches training)
insample_start = 1000
insample_end = 5000
context_len = model_config['context_len']
horizons = [1, 7, 14, 30]

# Crisis period (for breakdown)
crisis_start = 2000
crisis_end = 2765

print("In-Sample Configuration:")
print(f"  In-Sample period: indices [{insample_start}, {insample_end}]")
print(f"  Crisis period: indices [{crisis_start}, {crisis_end}]")
print(f"  Context length: {context_len}")
print(f"  Horizons to test: {horizons}")
print()

# ============================================================================
# Generate VAE Prior Predictions
# ============================================================================

print("Generating VAE Prior (z~N(0,1)) predictions...")
print()

# Storage for all horizon predictions
all_predictions = {}

for horizon in horizons:
    print(f"{'='*80}")
    print(f"HORIZON = {horizon} days")
    print(f"{'='*80}")

    # Temporarily change model horizon
    original_horizon = model.horizon
    model.horizon = horizon

    # Available in-sample days for this horizon
    min_idx = insample_start + context_len
    max_idx = insample_end - horizon + 1
    num_days = max_idx - min_idx

    print(f"  Available days: {num_days}")
    print(f"  Date range: [{min_idx}, {max_idx}]")

    # Storage for this horizon
    predictions = np.zeros((num_days, 3, 5, 5))  # (N, 3, 5, 5) - 3 quantiles
    indices = []

    with torch.no_grad():
        for i, day_idx in enumerate(tqdm(range(min_idx, max_idx), desc=f"  Generating H{horizon}")):
            # Context: [day_idx - context_len, day_idx)
            surface_ctx = vol_surf_data[day_idx - context_len : day_idx]  # (C, 5, 5)
            ex_ctx = ex_data[day_idx - context_len : day_idx]  # (C, 3)

            # Create context dict
            context = {
                "surface": torch.from_numpy(surface_ctx).unsqueeze(0),  # (1, C, 5, 5)
                "ex_feats": torch.from_numpy(ex_ctx).unsqueeze(0)  # (1, C, 3)
            }

            # Generate using VAE Prior: z ~ N(0,1) for future timesteps
            # get_surface_given_conditions with z=None samples from N(0,1)
            surf_pred, ex_pred = model.get_surface_given_conditions(context, z=None, mu=0, std=1)

            # surf_pred shape: (1, H, 3, 5, 5)
            # Extract last day prediction (H-th day)
            last_day_pred = surf_pred.cpu().numpy()[0, -1, :, :, :]  # (3, 5, 5)

            predictions[i] = last_day_pred
            indices.append(day_idx + horizon - 1)  # Index of predicted day

    # Restore original horizon
    model.horizon = original_horizon

    # Store predictions
    all_predictions[f'pred_h{horizon}'] = predictions
    all_predictions[f'indices_h{horizon}'] = np.array(indices)

    print(f"  ✓ Generated {num_days} predictions")
    print()

# ============================================================================
# Compute RMSE with Grid-Level Statistics
# ============================================================================

print("=" * 80)
print("COMPUTING RMSE WITH GRID-LEVEL STATISTICS")
print("=" * 80)
print()

summary_results = []
grid_results = []

for horizon in horizons:
    print(f"HORIZON = {horizon} days")
    print("-" * 80)

    # Load predictions and indices
    predictions = all_predictions[f'pred_h{horizon}']  # (N, 3, 5, 5)
    indices = all_predictions[f'indices_h{horizon}']  # (N,)

    # Extract p50 (median) and ground truth
    p50 = predictions[:, 1, :, :]  # (N, 5, 5)
    gt = vol_surf_data[indices]  # (N, 5, 5)

    # Identify regimes
    is_crisis = (indices >= crisis_start) & (indices <= crisis_end)
    is_normal = ~is_crisis

    num_total = len(indices)
    num_crisis = np.sum(is_crisis)
    num_normal = np.sum(is_normal)

    # ========================================================================
    # Compute Grid-Level RMSE for Each Regime
    # ========================================================================

    # Overall grid RMSE
    grid_rmse_overall = np.sqrt(np.mean((p50 - gt) ** 2, axis=0))  # (5, 5)

    # Crisis grid RMSE
    if num_crisis > 0:
        grid_rmse_crisis = np.sqrt(np.mean((p50[is_crisis] - gt[is_crisis]) ** 2, axis=0))
    else:
        grid_rmse_crisis = np.full((5, 5), np.nan)

    # Normal grid RMSE
    if num_normal > 0:
        grid_rmse_normal = np.sqrt(np.mean((p50[is_normal] - gt[is_normal]) ** 2, axis=0))
    else:
        grid_rmse_normal = np.full((5, 5), np.nan)

    # ========================================================================
    # Compute Statistics
    # ========================================================================

    def compute_stats(grid_rmse, regime_name):
        """Compute average, best, worst for a grid RMSE matrix."""
        if np.all(np.isnan(grid_rmse)):
            return {
                'regime': regime_name,
                'average_rmse': np.nan,
                'best_rmse': np.nan,
                'worst_rmse': np.nan,
            }

        average_rmse = np.mean(grid_rmse)
        best_idx = np.unravel_index(np.argmin(grid_rmse), grid_rmse.shape)
        worst_idx = np.unravel_index(np.argmax(grid_rmse), grid_rmse.shape)

        return {
            'regime': regime_name,
            'average_rmse': average_rmse,
            'best_rmse': grid_rmse[best_idx],
            'worst_rmse': grid_rmse[worst_idx],
        }

    overall_stats = compute_stats(grid_rmse_overall, 'overall')
    crisis_stats = compute_stats(grid_rmse_crisis, 'crisis')
    normal_stats = compute_stats(grid_rmse_normal, 'normal')

    # Print summary
    print(f"Overall ({num_total} samples): Avg={overall_stats['average_rmse']:.6f}, Best={overall_stats['best_rmse']:.6f}, Worst={overall_stats['worst_rmse']:.6f}")
    if num_crisis > 0:
        print(f"Crisis ({num_crisis} samples): Avg={crisis_stats['average_rmse']:.6f}, Best={crisis_stats['best_rmse']:.6f}, Worst={crisis_stats['worst_rmse']:.6f}")
    if num_normal > 0:
        print(f"Normal ({num_normal} samples): Avg={normal_stats['average_rmse']:.6f}, Best={normal_stats['best_rmse']:.6f}, Worst={normal_stats['worst_rmse']:.6f}")
    print()

    # Store results
    summary_results.append({
        'horizon': horizon,
        'period': 'In-Sample (2004-2019)',
        'days': num_total,
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

    # Per-grid results
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
# Save Results
# ============================================================================

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Summary table
summary_df = pd.DataFrame(summary_results)
summary_csv = "models/backfill/vae_prior_insample_rmse_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Summary saved to: {summary_csv}")
print()
print(summary_df.to_string(index=False))
print()

# Grid-level table
grid_df = pd.DataFrame(grid_results)
grid_csv = "models/backfill/vae_prior_insample_rmse_grid.csv"
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

print("Table 5: VAE Prior (z~N(0,1)) - RMSE by Horizon")
print()
print("| Period | H1 | H7 | H14 | H30 | Average | Best Grid | Worst Grid | Days |")
print("|--------|-----|-----|------|------|---------|-----------|------------|------|")

h1 = summary_df[summary_df['horizon'] == 1].iloc[0]
h7 = summary_df[summary_df['horizon'] == 7].iloc[0]
h14 = summary_df[summary_df['horizon'] == 14].iloc[0]
h30 = summary_df[summary_df['horizon'] == 30].iloc[0]

# Overall
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
print("=" * 80)
print("VAE PRIOR IN-SAMPLE RMSE EVALUATION COMPLETE")
print("=" * 80)
