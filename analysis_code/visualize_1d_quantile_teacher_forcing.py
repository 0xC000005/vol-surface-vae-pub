"""
Visualize 1D quantile VAE predictions with teacher forcing.

Shows:
- Ground truth vs predicted median (p50)
- Confidence intervals (p05-p95 shaded region)
- 2x4 grid: Not applicable for 1D (no "encoded latent" vs "context-only")
- Simple 1x4 grid: 4 model variants
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

# Configuration
PRED_DIR = "predictions_1d"
OUTPUT_DIR = "plots_1d"
LAST_N_DAYS = 500  # Plot last N days

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configurations
MODEL_NAMES = [
    ("amzn_only", "Amazon Only"),
    ("amzn_sp500", "Amazon + SP500"),
    ("amzn_msft", "Amazon + MSFT"),
    ("amzn_both", "Amazon + SP500 + MSFT"),
]

print("Loading predictions...")
model_data = {}
for model_key, model_label in MODEL_NAMES:
    pred_file = f"{PRED_DIR}/{model_key}_quantile_predictions.npz"
    data = np.load(pred_file)

    # Extract last N days
    p05 = data["p05"][-LAST_N_DAYS:]
    p50 = data["p50"][-LAST_N_DAYS:]
    p95 = data["p95"][-LAST_N_DAYS:]
    ground_truth = data["ground_truth"][-LAST_N_DAYS:]
    dates = pd.to_datetime(data["dates"])[-LAST_N_DAYS:]

    model_data[model_key] = {
        "label": model_label,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "ground_truth": ground_truth,
        "dates": dates,
    }

# Create visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"1D Quantile VAE Teacher Forcing Performance (Last {LAST_N_DAYS} Days)",
             fontsize=16, fontweight="bold")

for idx, (model_key, model_label) in enumerate(MODEL_NAMES):
    ax = axes[idx]
    data = model_data[model_key]

    dates = data["dates"]
    ground_truth = data["ground_truth"]
    p05 = data["p05"]
    p50 = data["p50"]
    p95 = data["p95"]

    # Plot ground truth
    ax.plot(dates, ground_truth, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.7)

    # Plot median prediction
    ax.plot(dates, p50, 'b-', linewidth=1.0, label='Predicted (p50)', alpha=0.8)

    # Plot confidence interval
    ax.fill_between(dates, p05, p95, color='blue', alpha=0.2, label='90% CI (p05-p95)')

    # Formatting
    ax.set_title(model_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Log Return', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

    # Date formatting
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Compute and display metrics
    rmse = np.sqrt(np.mean((p50 - ground_truth) ** 2))
    ci_violations = np.mean((ground_truth < p05) | (ground_truth > p95)) * 100
    ci_width = np.mean(p95 - p05)

    text = f"RMSE: {rmse:.4f}\nCI Viol: {ci_violations:.1f}%\nCI Width: {ci_width:.4f}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, "quantile_teacher_forcing_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print("\nSummary Statistics:")
for model_key, model_label in MODEL_NAMES:
    data = model_data[model_key]
    p05 = data["p05"]
    p50 = data["p50"]
    p95 = data["p95"]
    ground_truth = data["ground_truth"]

    rmse = np.sqrt(np.mean((p50 - ground_truth) ** 2))
    mae = np.mean(np.abs(p50 - ground_truth))
    ci_violations = np.mean((ground_truth < p05) | (ground_truth > p95)) * 100
    ci_width = np.mean(p95 - p05)

    print(f"\n{model_label}:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  CI Violations: {ci_violations:.2f}% (target: ~10%)")
    print(f"  Mean CI Width: {ci_width:.6f}")
