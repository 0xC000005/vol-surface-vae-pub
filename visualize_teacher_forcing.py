import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
OUTPUT_DIR = "tables/2024_1213/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
CONTEXT_LEN = 5
START_DAY = 5
LAST_N_DAYS = 1000  # Last 1000 days to visualize

# Grid points to visualize (row, col)
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

# Extract last N days
ground_truth_surfaces = ground_truth_surfaces[-LAST_N_DAYS:]
ground_truth_returns = ground_truth_returns[-LAST_N_DAYS:]
dates = dates[-LAST_N_DAYS:]

print(f"Visualizing period: {dates[0]} to {dates[-1]}")

# Load model predictions
model_data = {}
for model_key in MODELS.keys():
    print(f"Loading {model_key}...")

    # Load stochastic (1000 samples)
    stochastic_file = f"{BASE_MODEL_DIR}/{model_key}_gen5.npz"
    stochastic = np.load(stochastic_file)
    stochastic_surfaces = stochastic["surfaces"][-LAST_N_DAYS:]  # (1000, 1000, 5, 5)

    # Load MLE (1 sample)
    mle_file = f"{BASE_MODEL_DIR}/{model_key}_mle_gen5.npz"
    mle = np.load(mle_file)
    mle_surfaces = mle["surfaces"][-LAST_N_DAYS:, 0, :, :]  # (1000, 5, 5)

    model_data[model_key] = {
        "stochastic": stochastic_surfaces,
        "mle": mle_surfaces
    }

    # Load returns if available
    if MODELS[model_key]["has_returns"]:
        stochastic_returns = stochastic["ex_feats"][-LAST_N_DAYS:, :, 0]  # (1000, 1000)
        mle_returns = mle["ex_feats"][-LAST_N_DAYS:, 0, 0]  # (1000,)
        model_data[model_key]["stochastic_returns"] = stochastic_returns
        model_data[model_key]["mle_returns"] = mle_returns

print("Creating Figure 1: Implied Volatility Comparison...")

# Figure 1: Implied Vol - 3 rows (grid points) x 3 columns (models)
fig1, axes = plt.subplots(3, 3, figsize=(20, 14))
fig1.suptitle("Teacher Forcing Performance: Implied Volatility Predictions\n" +
              f"Context Length = {CONTEXT_LEN} days | Time Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
              fontsize=16, fontweight='bold')

grid_point_names = list(GRID_POINTS.keys())
model_keys = list(MODELS.keys())

for row_idx, (grid_name, (grid_row, grid_col)) in enumerate(GRID_POINTS.items()):
    # Extract ground truth for this grid point
    gt_surface = ground_truth_surfaces[:, grid_row, grid_col]

    for col_idx, model_key in enumerate(model_keys):
        ax = axes[row_idx, col_idx]

        # Extract model predictions for this grid point
        stoch_surface = model_data[model_key]["stochastic"][:, :, grid_row, grid_col]  # (1000, 1000)
        mle_surface = model_data[model_key]["mle"][:, grid_row, grid_col]  # (1000,)

        # Calculate percentiles (5th and 95th)
        p5 = np.percentile(stoch_surface, 5, axis=1)
        p95 = np.percentile(stoch_surface, 95, axis=1)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((mle_surface - gt_surface) ** 2))

        # Plot ground truth
        ax.plot(dates, gt_surface, 'k-', linewidth=2, label='Ground Truth', zorder=3)

        # Plot MLE prediction
        model_color = MODELS[model_key]["color"]
        ax.plot(dates, mle_surface, color=model_color, linewidth=1.5,
                label='MLE (z=0)', zorder=2)

        # Shaded uncertainty band
        ax.fill_between(dates, p5, p95, alpha=0.25, color=model_color,
                        label='90% CI (1000 samples)', zorder=1)

        # Context length indicator (vertical line at start)
        ax.axvline(dates[CONTEXT_LEN], color='red', linestyle='--',
                  alpha=0.6, linewidth=1.5, zorder=4)
        y_range = ax.get_ylim()
        y_pos = y_range[0] + (y_range[1] - y_range[0]) * 0.95
        ax.text(dates[CONTEXT_LEN], y_pos, f'  Ctx={CONTEXT_LEN}d',
               rotation=0, fontsize=9, color='red', va='top')

        # Title and labels
        if row_idx == 0:
            ax.set_title(MODELS[model_key]["name"], fontsize=13, fontweight='bold')

        if col_idx == 0:
            ax.set_ylabel(f'{grid_name}\nImplied Volatility', fontsize=11, fontweight='bold')

        if row_idx == 2:
            ax.set_xlabel('Date', fontsize=11)

        # RMSE annotation
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}',
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Legend (only for first subplot)
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='upper left', fontsize=9)

        ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file_1 = f"{OUTPUT_DIR}/teacher_forcing_implied_vol.png"
plt.savefig(output_file_1, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file_1}")
plt.close()

print("Creating Figure 2: Returns Comparison (ex_loss model)...")

# Figure 2: Returns (only for ex_loss model)
if "stochastic_returns" in model_data["ex_loss"]:
    fig2, ax = plt.subplots(1, 1, figsize=(16, 6))

    stoch_returns = model_data["ex_loss"]["stochastic_returns"]  # (1000, 1000)
    mle_returns = model_data["ex_loss"]["mle_returns"]  # (1000,)

    # Calculate percentiles
    p5_ret = np.percentile(stoch_returns, 5, axis=1)
    p95_ret = np.percentile(stoch_returns, 95, axis=1)

    # Calculate RMSE
    rmse_ret = np.sqrt(np.mean((mle_returns - ground_truth_returns) ** 2))

    # Plot ground truth
    ax.plot(dates, ground_truth_returns, 'k-', linewidth=2, label='Ground Truth Returns', zorder=3)

    # Plot MLE prediction
    ax.plot(dates, mle_returns, color=MODELS["ex_loss"]["color"], linewidth=1.5,
            label='MLE Predicted Returns (z=0)', zorder=2)

    # Shaded uncertainty band
    ax.fill_between(dates, p5_ret, p95_ret, alpha=0.25,
                    color=MODELS["ex_loss"]["color"],
                    label='90% CI (1000 samples)', zorder=1)

    # Context length indicator
    ax.axvline(dates[CONTEXT_LEN], color='red', linestyle='--',
              alpha=0.6, linewidth=1.5, zorder=4)
    y_range = ax.get_ylim()
    y_pos = y_range[0] + (y_range[1] - y_range[0]) * 0.95
    ax.text(dates[CONTEXT_LEN], y_pos, f'  Context Length={CONTEXT_LEN} days',
           rotation=0, fontsize=10, color='red', va='top')

    # Horizontal line at zero
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

    # Title and labels
    ax.set_title(f"Teacher Forcing Performance: Return Predictions (EX Loss Model)\n" +
                f"Context Length = {CONTEXT_LEN} days | Time Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Daily Log Return', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)

    # RMSE annotation
    ax.text(0.02, 0.98, f'RMSE: {rmse_ret:.6f}',
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file_2 = f"{OUTPUT_DIR}/teacher_forcing_returns.png"
    plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file_2}")
    plt.close()

print("\n" + "="*80)
print("Visualization complete!")
print("="*80)
print(f"\nGenerated files:")
print(f"  1. {output_file_1}")
print(f"  2. {output_file_2}")
print(f"\nKey insights to look for:")
print(f"  - How well does MLE (z=0) track ground truth?")
print(f"  - Are uncertainty bands well-calibrated (ground truth within 90% CI)?")
print(f"  - Do models with extra features (ex_no_loss, ex_loss) outperform no_ex?")
print(f"  - How do models perform during COVID crisis (2020)?")
print(f"  - Are ATM predictions more accurate than OTM?")
print(f"  - Does ex_loss model predict returns well?")
