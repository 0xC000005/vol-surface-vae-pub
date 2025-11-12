import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
OUTPUT_DIR = "tables/2024_1213/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
CONTEXT_LEN = 5
START_DAY = 5
LAST_N_DAYS = 1000  # Use all available days (set to None), or limit to specific number

# Grid points to visualize (row, col) - same as teacher forcing
GRID_POINTS = {
    "ATM 3-Month": (1, 2),   # moneyness=1.0, TTM=3 months
    "ATM 1-Year": (3, 2),    # moneyness=1.0, TTM=1 year
    "OTM Put 1-Year": (3, 0) # moneyness=0.7, TTM=1 year
}

# Model configurations
MODELS = {
    "no_ex": {
        "name": "No EX\n(Surface Only)",
        "color": "#1f77b4",
        "has_returns": False
    },
    "ex_no_loss": {
        "name": "EX No Loss\n(+Features)",
        "color": "#ff7f0e",
        "has_returns": False
    },
    "ex_loss": {
        "name": "EX Loss\n(+Features+Loss)",
        "color": "#2ca02c",
        "has_returns": True
    }
}

print("Loading data...")

# Load ground truth data
ground_truth_data = np.load("data/vol_surface_with_ret.npz")
ground_truth_surfaces = ground_truth_data["surface"][START_DAY:START_DAY+5810]  # Match predictions
ground_truth_returns = ground_truth_data["ret"][START_DAY:START_DAY+5810]

# Load dates
dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
dates = pd.to_datetime(dates_df["date"].values[START_DAY:START_DAY+5810])

# Extract last N days (or use all if LAST_N_DAYS is None)
if LAST_N_DAYS is not None:
    ground_truth_surfaces = ground_truth_surfaces[-LAST_N_DAYS:]
    ground_truth_returns = ground_truth_returns[-LAST_N_DAYS:]
    dates = dates[-LAST_N_DAYS:]

num_days = len(dates)
print(f"Analyzing period: {dates[0]} to {dates[-1]}")
print(f"Total days: {num_days}")

# Load model predictions
model_data = {}
for model_key in MODELS.keys():
    print(f"Loading {model_key}...")

    # Load stochastic (1000 samples per day)
    stochastic_file = f"{BASE_MODEL_DIR}/{model_key}_gen5.npz"
    stochastic = np.load(stochastic_file)
    if LAST_N_DAYS is not None:
        stochastic_surfaces = stochastic["surfaces"][-LAST_N_DAYS:]
    else:
        stochastic_surfaces = stochastic["surfaces"]

    model_data[model_key] = {
        "stochastic": stochastic_surfaces
    }

    # Load returns if available
    if MODELS[model_key]["has_returns"]:
        if LAST_N_DAYS is not None:
            stochastic_returns = stochastic["ex_feats"][-LAST_N_DAYS:, :, 0]
        else:
            stochastic_returns = stochastic["ex_feats"][:, :, 0]
        model_data[model_key]["stochastic_returns"] = stochastic_returns

print("\nCreating Figure 1: Implied Volatility Distribution Comparison...")

# Figure 1: Implied Vol - 3 rows (grid points) x 3 columns (models)
fig1, axes = plt.subplots(3, 3, figsize=(18, 14))
total_samples = num_days * 1000
fig1.suptitle("Marginal Distribution Comparison: Historical vs Model-Generated\n" +
              f"Historical: {num_days} days | Model: {num_days} days × 1000 samples = {total_samples/1e6:.1f}M samples\n" +
              f"Time Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
              fontsize=14, fontweight='bold')

grid_point_names = list(GRID_POINTS.keys())
model_keys = list(MODELS.keys())

for row_idx, (grid_name, (grid_row, grid_col)) in enumerate(GRID_POINTS.items()):
    # Extract ground truth for this grid point (1000 values)
    gt_values = ground_truth_surfaces[:, grid_row, grid_col]

    for col_idx, model_key in enumerate(model_keys):
        ax = axes[row_idx, col_idx]

        # Extract model predictions for this grid point
        # Shape: (1000 days, 1000 samples, 5, 5) -> flatten to 1M values
        stoch_values = model_data[model_key]["stochastic"][:, :, grid_row, grid_col].flatten()

        print(f"  {grid_name} - {model_key}: GT shape={gt_values.shape}, Model shape={stoch_values.shape}")

        # Determine common bins for both histograms
        all_values = np.concatenate([gt_values, stoch_values])
        bins = np.linspace(all_values.min(), all_values.max(), 50)

        # Plot histograms
        ax.hist(gt_values, bins=bins, alpha=0.6, color='black',
                label='Historical', density=True, edgecolor='black', linewidth=0.5)

        model_color = MODELS[model_key]["color"]
        ax.hist(stoch_values, bins=bins, alpha=0.5, color=model_color,
                label='Model Generated', density=True, edgecolor=model_color, linewidth=0.5)

        # Title and labels
        if row_idx == 0:
            ax.set_title(MODELS[model_key]["name"], fontsize=12, fontweight='bold')

        if col_idx == 0:
            ax.set_ylabel(f'{grid_name}\nDensity', fontsize=10, fontweight='bold')

        if row_idx == 2:
            ax.set_xlabel('Implied Volatility', fontsize=10)

        # Legend (only for first subplot)
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='upper right', fontsize=9)

        ax.grid(True, alpha=0.3)

plt.tight_layout()
period_suffix = f"_{num_days}days" if LAST_N_DAYS is not None else "_full"
output_file_1 = f"{OUTPUT_DIR}/distribution_comparison_implied_vol{period_suffix}.png"
plt.savefig(output_file_1, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_file_1}")
plt.close()

print("\nCreating Figure 2: Returns Distribution Comparison (ex_loss model)...")

# Figure 2: Returns (only for ex_loss model)
if "stochastic_returns" in model_data["ex_loss"]:
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Historical returns (1000 values)
    gt_returns = ground_truth_returns

    # Model returns (1000 days × 1000 samples = 1M values)
    model_returns = model_data["ex_loss"]["stochastic_returns"].flatten()

    print(f"  Returns: GT shape={gt_returns.shape}, Model shape={model_returns.shape}")

    # Determine common bins
    all_returns = np.concatenate([gt_returns, model_returns])
    bins = np.linspace(all_returns.min(), all_returns.max(), 60)

    # Plot histograms
    ax.hist(gt_returns, bins=bins, alpha=0.6, color='black',
            label='Historical Returns', density=True, edgecolor='black', linewidth=0.5)

    ax.hist(model_returns, bins=bins, alpha=0.5, color=MODELS["ex_loss"]["color"],
            label='Model Generated Returns', density=True,
            edgecolor=MODELS["ex_loss"]["color"], linewidth=0.5)

    # Vertical line at zero
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Zero Return')

    # Title and labels
    ax.set_title(f"Marginal Distribution Comparison: Daily Returns (EX Loss Model)\n" +
                f"Historical: {num_days} days | Model: {num_days} days × 1000 samples = {total_samples/1e6:.1f}M samples\n" +
                f"Time Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_xlabel('Daily Log Return', fontsize=11)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file_2 = f"{OUTPUT_DIR}/distribution_comparison_returns{period_suffix}.png"
    plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file_2}")
    plt.close()

print("\n" + "="*80)
print("Distribution comparison visualization complete!")
print("="*80)
print(f"\nGenerated files:")
print(f"  1. {output_file_1}")
if "stochastic_returns" in model_data["ex_loss"]:
    print(f"  2. {output_file_2}")
print(f"\nData summary:")
print(f"  - Time period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"  - Total days: {num_days}")
print(f"  - Historical samples: {num_days}")
print(f"  - Model samples: {total_samples:,} ({num_days} days × 1000 samples)")
print(f"\nKey insights to look for:")
print(f"  - Do model distributions match historical distributions?")
print(f"  - Are the means and spreads similar?")
print(f"  - Does the model capture the full range of historical values?")
print(f"  - Are there any systematic biases (e.g., model too narrow/wide)?")
print(f"  - Does EX Loss model capture return distribution shape?")
