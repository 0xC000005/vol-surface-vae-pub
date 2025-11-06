"""
Evaluate 1D quantile VAE models.

Metrics:
- RMSE/MAE: Using p50 (median) as point forecast
- R²: Coefficient of determination
- Direction Accuracy: % of correct sign predictions
- CI Violation Rate: % outside 90% CI (target: ~10%)
- Mean CI Width: Average uncertainty band width
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score

# Configuration
PRED_DIR = "predictions_1d"
OUTPUT_DIR = "results_1d"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configurations
MODEL_NAMES = [
    ("amzn_only", "Amazon Only"),
    ("amzn_sp500", "Amazon + SP500"),
    ("amzn_msft", "Amazon + MSFT"),
    ("amzn_both", "Amazon + SP500 + MSFT"),
]

print("=" * 80)
print("EVALUATING 1D QUANTILE VAE MODELS")
print("=" * 80)
print()

results = []

for model_key, model_label in MODEL_NAMES:
    print(f"Evaluating: {model_label}")
    print("-" * 40)

    # Load predictions
    pred_file = f"{PRED_DIR}/{model_key}_quantile_predictions.npz"
    data = np.load(pred_file)

    p05 = data["p05"]
    p50 = data["p50"]
    p95 = data["p95"]
    ground_truth = data["ground_truth"]

    # Point forecast metrics (using p50 as point estimate)
    rmse = np.sqrt(np.mean((p50 - ground_truth) ** 2))
    mae = np.mean(np.abs(p50 - ground_truth))
    r2 = r2_score(ground_truth, p50)

    # Direction accuracy
    correct_direction = np.sum(np.sign(p50) == np.sign(ground_truth))
    direction_acc = correct_direction / len(ground_truth) * 100

    # CI calibration metrics
    below_p05 = np.sum(ground_truth < p05)
    above_p95 = np.sum(ground_truth > p95)
    ci_violations = below_p05 + above_p95
    ci_violation_rate = ci_violations / len(ground_truth) * 100

    # CI width
    mean_ci_width = np.mean(p95 - p05)

    # Print results
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  Direction Accuracy: {direction_acc:.2f}%")
    print(f"  CI Violation Rate: {ci_violation_rate:.2f}% (target: ~10%)")
    print(f"    Below p05: {below_p05} ({below_p05/len(ground_truth)*100:.2f}%)")
    print(f"    Above p95: {above_p95} ({above_p95/len(ground_truth)*100:.2f}%)")
    print(f"  Mean CI Width: {mean_ci_width:.6f}")
    print()

    # Store results
    results.append({
        "Model": model_label,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Direction Accuracy (%)": direction_acc,
        "CI Violation Rate (%)": ci_violation_rate,
        "Below p05 (%)": below_p05/len(ground_truth)*100,
        "Above p95 (%)": above_p95/len(ground_truth)*100,
        "Mean CI Width": mean_ci_width,
    })

# Create results DataFrame
df = pd.DataFrame(results)

# Save to CSV
output_file = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
df.to_csv(output_file, index=False, float_format="%.6f")
print("=" * 80)
print(f"Results saved to: {output_file}")
print("=" * 80)
print()

# Display summary table
print("Summary Table:")
print(df.to_string(index=False))
print()

# Find best model for each metric
print("=" * 80)
print("BEST MODELS BY METRIC")
print("=" * 80)
print(f"Lowest RMSE: {df.loc[df['RMSE'].idxmin(), 'Model']} ({df['RMSE'].min():.6f})")
print(f"Lowest MAE: {df.loc[df['MAE'].idxmin(), 'Model']} ({df['MAE'].min():.6f})")
print(f"Highest R²: {df.loc[df['R²'].idxmax(), 'Model']} ({df['R²'].max():.6f})")
print(f"Highest Direction Accuracy: {df.loc[df['Direction Accuracy (%)'].idxmax(), 'Model']} ({df['Direction Accuracy (%)'].max():.2f}%)")
print(f"Best CI Calibration: {df.loc[(df['CI Violation Rate (%)'] - 10).abs().idxmin(), 'Model']} ({df.loc[(df['CI Violation Rate (%)'] - 10).abs().idxmin(), 'CI Violation Rate (%)']:.2f}%)")
print()

# Analysis notes
print("=" * 80)
print("NOTES")
print("=" * 80)
print("- RMSE/MAE/R²: Based on p50 (median) as point forecast")
print("- Direction Accuracy: % of correct sign predictions (>50% = better than random)")
print("- CI Violation Rate: Target is ~10% for well-calibrated 90% CI")
print("- CI Width: Narrower is better (but must maintain calibration)")
