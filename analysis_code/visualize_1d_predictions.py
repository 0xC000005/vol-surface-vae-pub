"""
Visualize 1D VAE predictions with teacher forcing.

Creates plots comparing all 4 models:
- Ground truth Amazon returns
- Model predictions with uncertainty bands (p05, p50, p95)
- Separate subplot for each model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
PREDICTIONS_DIR = "predictions_1d"
OUTPUT_DIR = "plots_1d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualization parameters
START_DAY = 0  # Start from beginning of test set
NUM_DAYS = 200  # Number of days to plot (or None for all)

print("=" * 80)
print("VISUALIZING 1D VAE PREDICTIONS")
print("=" * 80)
print(f"Predictions directory: {PREDICTIONS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# Model configurations
models = [
    {
        "name": "amzn_only",
        "label": "Amazon Only",
        "color": "#1f77b4",
    },
    {
        "name": "amzn_sp500",
        "label": "Amazon + SP500",
        "color": "#ff7f0e",
    },
    {
        "name": "amzn_msft",
        "label": "Amazon + MSFT",
        "color": "#2ca02c",
    },
    {
        "name": "amzn_both",
        "label": "Amazon + Both",
        "color": "#d62728",
    },
]

# Load predictions
print("Loading predictions...")
predictions = {}
for model in models:
    pred_file = os.path.join(PREDICTIONS_DIR, f"{model['name']}_predictions.npz")
    data = np.load(pred_file)
    predictions[model['name']] = {
        "stochastic": data["stochastic"],
        "mle": data["mle"],
        "ground_truth": data["ground_truth"],
        "dates": pd.to_datetime(data["dates"]),
    }
    print(f"  Loaded: {model['name']}")

print()

# Get data for plotting
ground_truth = predictions["amzn_only"]["ground_truth"]
dates = predictions["amzn_only"]["dates"]

# Limit to plotting range
if NUM_DAYS is not None:
    end_day = min(START_DAY + NUM_DAYS, len(ground_truth))
else:
    end_day = len(ground_truth)

dates_plot = dates[START_DAY:end_day]
ground_truth_plot = ground_truth[START_DAY:end_day]

print(f"Plotting {end_day - START_DAY} days")
print(f"Date range: {dates_plot[0]} to {dates_plot[-1]}")
print()

# Create figure: 4 rows (one per model)
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle(
    f"1D VAE Predictions: Amazon Returns (Teacher Forcing)\n" +
    f"Test Period: {dates_plot[0].strftime('%Y-%m-%d')} to {dates_plot[-1].strftime('%Y-%m-%d')}",
    fontsize=14,
    fontweight='bold'
)

for idx, model in enumerate(models):
    ax = axes[idx]

    # Extract predictions for this model
    stochastic = predictions[model['name']]["stochastic"][START_DAY:end_day]
    mle = predictions[model['name']]["mle"][START_DAY:end_day]

    # Compute quantiles from stochastic samples
    p05 = np.percentile(stochastic, 5, axis=1)
    p50 = np.percentile(stochastic, 50, axis=1)
    p95 = np.percentile(stochastic, 95, axis=1)

    # Plot ground truth
    ax.plot(
        dates_plot,
        ground_truth_plot,
        color='black',
        linewidth=1.5,
        label='Ground Truth',
        alpha=0.8
    )

    # Plot p50 (median prediction)
    ax.plot(
        dates_plot,
        p50,
        color=model['color'],
        linewidth=1.5,
        label='Model p50',
        alpha=0.8
    )

    # Plot uncertainty band (p05 to p95)
    ax.fill_between(
        dates_plot,
        p05,
        p95,
        color=model['color'],
        alpha=0.2,
        label='90% CI (p05-p95)'
    )

    # Formatting
    ax.set_ylabel('Return', fontsize=10)
    ax.set_title(model['label'], fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# X-axis label
axes[-1].set_xlabel('Date', fontsize=10)

plt.tight_layout()

# Save figure
output_file = os.path.join(OUTPUT_DIR, "predictions_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

print()
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
