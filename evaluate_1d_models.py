"""
Evaluate 1D VAE models on stock return predictions.

Metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- Direction Accuracy (% correct sign predictions)
- CI Calibration (% of ground truth outside 90% CI)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Configuration
PREDICTIONS_DIR = "predictions_1d"
OUTPUT_DIR = "results_1d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("EVALUATING 1D VAE MODELS")
print("=" * 80)
print(f"Predictions directory: {PREDICTIONS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# Model configurations
models = [
    {"name": "amzn_only", "description": "Amazon Only"},
    {"name": "amzn_sp500", "description": "Amazon + SP500"},
    {"name": "amzn_msft", "description": "Amazon + MSFT"},
    {"name": "amzn_both", "description": "Amazon + Both"},
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

# Get ground truth (same for all models)
ground_truth = predictions["amzn_only"]["ground_truth"]
dates = predictions["amzn_only"]["dates"]

print(f"Test set: {len(ground_truth)} days")
print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print()


def compute_metrics(ground_truth, stochastic_preds, mle_preds):
    """Compute evaluation metrics."""
    # Compute quantiles from stochastic predictions
    p05 = np.percentile(stochastic_preds, 5, axis=1)
    p50 = np.percentile(stochastic_preds, 50, axis=1)
    p95 = np.percentile(stochastic_preds, 95, axis=1)

    # Use p50 (median) for point forecast evaluation
    pred = p50

    # RMSE
    rmse = np.sqrt(mean_squared_error(ground_truth, pred))

    # MAE
    mae = mean_absolute_error(ground_truth, pred)

    # R²
    r2 = r2_score(ground_truth, pred)

    # Direction accuracy (% correct sign)
    correct_direction = np.sum(np.sign(ground_truth) == np.sign(pred))
    direction_accuracy = correct_direction / len(ground_truth) * 100

    # CI Calibration (% outside 90% CI)
    below_ci = np.sum(ground_truth < p05)
    above_ci = np.sum(ground_truth > p95)
    outside_ci = below_ci + above_ci
    ci_violation_rate = outside_ci / len(ground_truth) * 100

    # Mean CI width
    mean_ci_width = np.mean(p95 - p05)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "ci_violation_rate": ci_violation_rate,
        "below_ci_pct": below_ci / len(ground_truth) * 100,
        "above_ci_pct": above_ci / len(ground_truth) * 100,
        "mean_ci_width": mean_ci_width,
    }


# Evaluate all models
results = {
    "model": [],
    "description": [],
    "rmse": [],
    "mae": [],
    "r2": [],
    "direction_accuracy": [],
    "ci_violation_rate": [],
    "below_ci_pct": [],
    "above_ci_pct": [],
    "mean_ci_width": [],
}

print("Computing metrics...")
print()

for model in models:
    metrics = compute_metrics(
        ground_truth,
        predictions[model['name']]["stochastic"],
        predictions[model['name']]["mle"]
    )

    results["model"].append(model['name'])
    results["description"].append(model['description'])
    results["rmse"].append(metrics["rmse"])
    results["mae"].append(metrics["mae"])
    results["r2"].append(metrics["r2"])
    results["direction_accuracy"].append(metrics["direction_accuracy"])
    results["ci_violation_rate"].append(metrics["ci_violation_rate"])
    results["below_ci_pct"].append(metrics["below_ci_pct"])
    results["above_ci_pct"].append(metrics["above_ci_pct"])
    results["mean_ci_width"].append(metrics["mean_ci_width"])

    print(f"{model['description']}:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
    print(f"  CI Violation Rate:  {metrics['ci_violation_rate']:.2f}%")
    print(f"    Below p05: {metrics['below_ci_pct']:.2f}%")
    print(f"    Above p95: {metrics['above_ci_pct']:.2f}%")
    print(f"  Mean CI Width: {metrics['mean_ci_width']:.6f}")
    print()

# Save results to CSV
results_df = pd.DataFrame(results)
results_file = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
results_df.to_csv(results_file, index=False)

print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"Results saved to: {results_file}")
print()

# Print comparison table
print("COMPARISON TABLE:")
print("=" * 80)
print(results_df.to_string(index=False))
print()

# Rank models by RMSE
print("RANKING BY RMSE (lower is better):")
print("-" * 40)
ranked = results_df.sort_values("rmse")
for idx, row in ranked.iterrows():
    print(f"  {row['description']}: {row['rmse']:.6f}")
print()

# Rank models by R²
print("RANKING BY R² (higher is better):")
print("-" * 40)
ranked = results_df.sort_values("r2", ascending=False)
for idx, row in ranked.iterrows():
    print(f"  {row['description']}: {row['r2']:.6f}")
print()

# Rank models by Direction Accuracy
print("RANKING BY DIRECTION ACCURACY (higher is better):")
print("-" * 40)
ranked = results_df.sort_values("direction_accuracy", ascending=False)
for idx, row in ranked.iterrows():
    print(f"  {row['description']}: {row['direction_accuracy']:.2f}%")
print()

print("KEY INSIGHTS:")
print("-" * 80)
print("1. Does conditioning on SP500/MSFT improve predictions?")
print("   → Compare RMSE/R² across models")
print()
print("2. Which conditioning feature is more informative?")
print("   → Compare SP500 vs MSFT vs Both")
print()
print("3. Are uncertainty estimates well-calibrated?")
print("   → Target CI violation rate: ~10% (well-calibrated)")
print("   → Actual rates shown above")
print()
print("4. Is the model better than random guessing?")
print("   → Direction accuracy > 50% indicates predictive power")
