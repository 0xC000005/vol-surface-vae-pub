"""
Improvement Heatmap: Where Does Multi-Horizon Training Help Most?

Shows improvement % across the entire 5×5 volatility surface grid.
Reveals which areas (moneyness × maturity) benefit most from horizon=5 training.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import set_seeds

# Set seeds and dtype
set_seeds(42)
torch.set_default_dtype(torch.float64)

print("=" * 80)
print("IMPROVEMENT HEATMAP: Grid-wise Analysis")
print("=" * 80)
print()

# Load test data
print("Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
test_start_idx = 4500
vol_test = vol_surf_data[test_start_idx:]
print(f"  Test data: {vol_test.shape[0]} days")

# Load models
print("\nLoading models...")
baseline_data = torch.load("test_spx/quantile_regression/no_ex.pt", weights_only=False)
baseline_model = CVAEMemRand(baseline_data["model_config"])
baseline_model.load_weights(dict_to_load=baseline_data)
baseline_model.eval()

horizon5_data = torch.load("test_horizon5/no_ex_horizon5.pt", weights_only=False)
horizon5_model = CVAEMemRand(horizon5_data["model_config"])
horizon5_model.load_weights(dict_to_load=horizon5_data)
horizon5_model.eval()

device = baseline_model.device

# Configuration
context_len = 5
horizon = 5
num_test_windows = 50

print(f"\nTesting on {num_test_windows} windows...")
print("Computing RMSE for each grid point...")

# Storage for RMSEs at each grid point
baseline_rmse_grid = np.zeros((5, 5))
horizon5_rmse_grid = np.zeros((5, 5))
improvement_grid = np.zeros((5, 5))

# Collect errors for all grid points
baseline_errors = {(i, j): [] for i in range(5) for j in range(5)}
horizon5_errors = {(i, j): [] for i in range(5) for j in range(5)}

for window_idx in range(num_test_windows):
    start_day = window_idx * 10

    if start_day + context_len + horizon >= len(vol_test):
        break

    # Get context and ground truth
    context_surfaces = vol_test[start_day:start_day+context_len]
    ground_truth_5days = vol_test[start_day+context_len:start_day+context_len+horizon]

    context = {
        "surface": torch.from_numpy(context_surfaces).unsqueeze(0).to(device)
    }

    with torch.no_grad():
        # Baseline: Autoregressive
        baseline_preds = []
        baseline_context = context.copy()

        for day in range(horizon):
            ctx_surfaces = baseline_context["surface"]
            pred_surface, _, _, _ = baseline_model.forward({
                "surface": torch.cat([ctx_surfaces,
                                    torch.zeros(1, 1, 5, 5, dtype=torch.float64).to(device)],
                                   dim=1)
            })

            baseline_preds.append(pred_surface[:, 0, 1, :, :].cpu().numpy())  # p50

            # Update context
            new_surface = pred_surface[:, 0, 1, :, :]
            new_context = torch.cat([ctx_surfaces[:, 1:, :, :],
                                    new_surface.unsqueeze(1)], dim=1)
            baseline_context = {"surface": new_context}

        baseline_preds = np.array(baseline_preds)  # (5, 1, 5, 5)
        baseline_preds = baseline_preds[:, 0, :, :]  # (5, 5, 5)

        # Horizon=5: Single-shot
        horizon5_input = torch.cat([
            context["surface"],
            torch.zeros(1, horizon, 5, 5, dtype=torch.float64).to(device)
        ], dim=1)

        horizon5_pred, _, _, _ = horizon5_model.forward({
            "surface": horizon5_input
        })

        horizon5_preds = horizon5_pred[0, :, 1, :, :].cpu().numpy()  # (5, 5, 5)

    # Accumulate errors for each grid point
    for i in range(5):
        for j in range(5):
            baseline_err = np.mean((baseline_preds[:, i, j] - ground_truth_5days[:, i, j]) ** 2)
            horizon5_err = np.mean((horizon5_preds[:, i, j] - ground_truth_5days[:, i, j]) ** 2)

            baseline_errors[(i, j)].append(baseline_err)
            horizon5_errors[(i, j)].append(horizon5_err)

print("  Completed all windows")

# Compute average RMSE for each grid point
print("\nComputing average RMSE per grid point...")

for i in range(5):
    for j in range(5):
        baseline_rmse_grid[i, j] = np.sqrt(np.mean(baseline_errors[(i, j)]))
        horizon5_rmse_grid[i, j] = np.sqrt(np.mean(horizon5_errors[(i, j)]))
        improvement_grid[i, j] = (baseline_rmse_grid[i, j] - horizon5_rmse_grid[i, j]) / baseline_rmse_grid[i, j] * 100

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Labels for axes
moneyness_labels = ['OTM-', 'OTM', 'ATM', 'ITM', 'ITM+']
maturity_labels = ['1M', '3M', '6M', '1Y', '2Y']

# Plot 1: Baseline RMSE
ax = axes[0]
sns.heatmap(baseline_rmse_grid, annot=True, fmt='.4f', cmap='Reds', ax=ax,
            xticklabels=moneyness_labels, yticklabels=maturity_labels,
            cbar_kws={'label': 'RMSE'})
ax.set_title('Baseline (Horizon=1)\nAutoregressive RMSE', fontsize=12, fontweight='bold')
ax.set_xlabel('Moneyness', fontsize=10)
ax.set_ylabel('Time to Maturity', fontsize=10)

# Plot 2: Horizon=5 RMSE
ax = axes[1]
sns.heatmap(horizon5_rmse_grid, annot=True, fmt='.4f', cmap='Blues', ax=ax,
            xticklabels=moneyness_labels, yticklabels=maturity_labels,
            cbar_kws={'label': 'RMSE'})
ax.set_title('Horizon=5\nSingle-Shot RMSE', fontsize=12, fontweight='bold')
ax.set_xlabel('Moneyness', fontsize=10)
ax.set_ylabel('Time to Maturity', fontsize=10)

# Plot 3: Improvement %
ax = axes[2]
sns.heatmap(improvement_grid, annot=True, fmt='.1f', cmap='Greens', ax=ax,
            xticklabels=moneyness_labels, yticklabels=maturity_labels,
            cbar_kws={'label': 'Improvement %'}, vmin=0)
ax.set_title('🎉 Improvement %\n(Baseline → Horizon=5)', fontsize=12, fontweight='bold')
ax.set_xlabel('Moneyness', fontsize=10)
ax.set_ylabel('Time to Maturity', fontsize=10)

plt.suptitle('Where Does Multi-Horizon Training Help Most?', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
output_file = "test_horizon5/improvement_heatmap.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Heatmap saved to: {output_file}")
plt.close()

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nBaseline RMSE:")
print(f"  Mean: {baseline_rmse_grid.mean():.5f}")
print(f"  Min:  {baseline_rmse_grid.min():.5f} at grid point {np.unravel_index(baseline_rmse_grid.argmin(), baseline_rmse_grid.shape)}")
print(f"  Max:  {baseline_rmse_grid.max():.5f} at grid point {np.unravel_index(baseline_rmse_grid.argmax(), baseline_rmse_grid.shape)}")

print(f"\nHorizon=5 RMSE:")
print(f"  Mean: {horizon5_rmse_grid.mean():.5f}")
print(f"  Min:  {horizon5_rmse_grid.min():.5f} at grid point {np.unravel_index(horizon5_rmse_grid.argmin(), horizon5_rmse_grid.shape)}")
print(f"  Max:  {horizon5_rmse_grid.max():.5f} at grid point {np.unravel_index(horizon5_rmse_grid.argmax(), horizon5_rmse_grid.shape)}")

print(f"\nImprovement %:")
print(f"  Mean: {improvement_grid.mean():.1f}%")
print(f"  Min:  {improvement_grid.min():.1f}% at grid point {np.unravel_index(improvement_grid.argmin(), improvement_grid.shape)}")
print(f"  Max:  {improvement_grid.max():.1f}% at grid point {np.unravel_index(improvement_grid.argmax(), improvement_grid.shape)}")

# Find areas with highest improvement
print(f"\nTop 3 Areas with Highest Improvement:")
flat_idx = improvement_grid.flatten().argsort()[::-1][:3]
for rank, idx in enumerate(flat_idx, 1):
    i, j = np.unravel_index(idx, improvement_grid.shape)
    print(f"  {rank}. Grid ({i},{j}) - {maturity_labels[i]} × {moneyness_labels[j]}: +{improvement_grid[i,j]:.1f}%")

# Check if improvement is uniform
print(f"\nUniformity of Improvement:")
std_improvement = improvement_grid.std()
print(f"  Std Dev: {std_improvement:.1f}%")
if std_improvement < 5:
    print(f"  ✓ Improvement is UNIFORM across the surface (low variance)")
elif std_improvement < 10:
    print(f"  ≈ Improvement is SOMEWHAT UNIFORM (moderate variance)")
else:
    print(f"  ⚠ Improvement is NON-UNIFORM (high variance)")

print("=" * 80)
