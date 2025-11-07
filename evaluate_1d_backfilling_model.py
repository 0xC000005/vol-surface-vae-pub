"""
Evaluate 1D Backfilling Model Predictions.

Computes metrics for all 3 latent selection scenarios:
1. Oracle (upper bound with future information)
2. Mixed 80/20 (training distribution)
3. Realistic Backfilling (production scenario)

Metrics:
- RMSE/MAE: Point forecast accuracy (using p50)
- R²: Coefficient of determination
- Direction Accuracy: % correct sign predictions
- CI Violation Rate: % outside [p05, p95]
- Mean CI Width: Average (p95 - p05)
- Pinball Loss: Proper scoring rule for quantiles
"""

import numpy as np
import pandas as pd
import os

# Configuration
PREDICTIONS_FILE = "models_1d_backfilling/backfill_predictions_ctx5.npz"
OUTPUT_FILE = "models_1d_backfilling/comparison_metrics.csv"

print("=" * 80)
print("EVALUATING 1D BACKFILLING MODEL")
print("=" * 80)
print(f"Predictions file: {PREDICTIONS_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print()

# Check file exists
if not os.path.exists(PREDICTIONS_FILE):
    raise FileNotFoundError(
        f"Predictions file not found: {PREDICTIONS_FILE}\n"
        f"Run: python generate_1d_backfilling_predictions.py"
    )

# Load predictions
print("Loading predictions...")
data = np.load(PREDICTIONS_FILE)

actuals = data["actuals"]
dates = data["dates"]

# Scenario 1: Oracle
s1_predictions = {
    "p05": data["s1_p05"],
    "p50": data["s1_p50"],
    "p95": data["s1_p95"],
}

# Scenario 2: Mixed 80/20
s2_predictions = {
    "p05": data["s2_p05"],
    "p50": data["s2_p50"],
    "p95": data["s2_p95"],
}

# Scenario 3: Realistic Backfilling
s3_predictions = {
    "p05": data["s3_p05"],
    "p50": data["s3_p50"],
    "p95": data["s3_p95"],
}

print(f"  Loaded {len(actuals)} predictions")
print()


def compute_pinball_loss(predictions, actuals, quantiles=[0.05, 0.5, 0.95]):
    """
    Compute pinball loss (asymmetric quantile loss).

    Args:
        predictions: dict with keys "p05", "p50", "p95"
        actuals: (N,) array of ground truth
        quantiles: List of quantile levels [0.05, 0.5, 0.95]

    Returns:
        float: Mean pinball loss across all quantiles
    """
    losses = []
    for q, pred_key in zip(quantiles, ["p05", "p50", "p95"]):
        pred_q = predictions[pred_key]
        error = actuals - pred_q
        loss_q = np.maximum((q - 1) * error, q * error)
        losses.append(np.mean(loss_q))

    return np.mean(losses)


def compute_metrics(predictions, actuals, scenario_name):
    """
    Compute all evaluation metrics for a given scenario.

    Args:
        predictions: dict with keys "p05", "p50", "p95"
        actuals: (N,) array of ground truth
        scenario_name: String name of scenario (for display)

    Returns:
        dict: All computed metrics
    """
    p05, p50, p95 = predictions["p05"], predictions["p50"], predictions["p95"]

    metrics = {}
    metrics["scenario"] = scenario_name

    # 1. Point Forecast Accuracy (using p50 as forecast)
    metrics["rmse"] = np.sqrt(np.mean((p50 - actuals) ** 2))
    metrics["mae"] = np.mean(np.abs(p50 - actuals))

    # 2. R² (coefficient of determination)
    ss_res = np.sum((actuals - p50) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    metrics["r2"] = 1 - (ss_res / ss_tot)

    # 3. Direction Accuracy (% correct sign predictions)
    pred_signs = np.sign(p50)
    actual_signs = np.sign(actuals)
    metrics["direction_acc"] = np.mean(pred_signs == actual_signs) * 100  # Convert to %

    # 4. CI Violation Rate (% outside [p05, p95])
    violations = (actuals < p05) | (actuals > p95)
    metrics["ci_violation_rate"] = np.mean(violations) * 100  # Convert to %

    # 5. Mean CI Width (average uncertainty band)
    metrics["mean_ci_width"] = np.mean(p95 - p05)

    # 6. CI Width Std (variability in uncertainty)
    metrics["ci_width_std"] = np.std(p95 - p05)

    # 7. Pinball Loss (proper scoring rule for quantiles)
    metrics["pinball_loss"] = compute_pinball_loss(predictions, actuals)

    # 8. Separate violation types
    below_violations = np.mean(actuals < p05) * 100
    above_violations = np.mean(actuals > p95) * 100
    metrics["violations_below"] = below_violations
    metrics["violations_above"] = above_violations

    return metrics


# Compute metrics for all scenarios
print("=" * 80)
print("COMPUTING METRICS")
print("=" * 80)
print()

metrics_s1 = compute_metrics(s1_predictions, actuals, "Oracle (S1)")
print(f"✓ Scenario 1 (Oracle) metrics computed")

metrics_s2 = compute_metrics(s2_predictions, actuals, "Mixed 80/20 (S2)")
print(f"✓ Scenario 2 (Mixed 80/20) metrics computed")

metrics_s3 = compute_metrics(s3_predictions, actuals, "Realistic (S3)")
print(f"✓ Scenario 3 (Realistic Backfilling) metrics computed")

print()

# Create comparison DataFrame
df = pd.DataFrame([metrics_s1, metrics_s2, metrics_s3])
df = df.set_index("scenario")

# Reorder columns for display
column_order = [
    "rmse", "mae", "r2", "direction_acc", "ci_violation_rate",
    "violations_below", "violations_above", "mean_ci_width",
    "ci_width_std", "pinball_loss"
]
df = df[column_order]

# Save to CSV
df.to_csv(OUTPUT_FILE)
print(f"Metrics saved to: {OUTPUT_FILE}")
print()

# Print formatted table
print("=" * 80)
print("BACKFILLING MODEL COMPARISON")
print("=" * 80)
print()
print(df.round(4).to_string())
print()

# Highlight best performers
print("=" * 80)
print("BEST PERFORMERS PER METRIC")
print("=" * 80)
print()

# Metrics where lower is better
lower_is_better = ["rmse", "mae", "ci_violation_rate", "ci_width_std", "pinball_loss"]

# Metrics where higher is better
higher_is_better = ["r2", "direction_acc"]

for col in df.columns:
    if col in ["violations_below", "violations_above"]:
        continue  # Skip breakdown metrics

    if col in lower_is_better:
        best_idx = df[col].idxmin()
        best_val = df.loc[best_idx, col]
        print(f"  {col:20s}: {best_idx:20s} ({best_val:.4f}) ✓")
    elif col in higher_is_better:
        best_idx = df[col].idxmax()
        best_val = df.loc[best_idx, col]
        print(f"  {col:20s}: {best_idx:20s} ({best_val:.4f}) ✓")
    else:
        # Just show best without marker (like ci_width)
        best_idx = df[col].idxmin()
        best_val = df.loc[best_idx, col]
        print(f"  {col:20s}: {best_idx:20s} ({best_val:.4f})")

print()

# Analysis insights
print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print()

# CI Calibration
print("1. CI Calibration (Target: ~10% violations)")
print("-" * 80)
for scenario, rate in zip(df.index, df["ci_violation_rate"]):
    status = "✓ Well calibrated" if 8 <= rate <= 12 else "⚠ Needs improvement"
    print(f"  {scenario:20s}: {rate:5.2f}% {status}")
print()

# Oracle vs Realistic Gap
print("2. Oracle vs Realistic Performance Gap")
print("-" * 80)
rmse_gap = ((metrics_s3["rmse"] - metrics_s1["rmse"]) / metrics_s1["rmse"]) * 100
mae_gap = ((metrics_s3["mae"] - metrics_s1["mae"]) / metrics_s1["mae"]) * 100
ci_gap = metrics_s3["ci_violation_rate"] - metrics_s1["ci_violation_rate"]
print(f"  RMSE degradation: {rmse_gap:+.1f}%")
print(f"  MAE degradation: {mae_gap:+.1f}%")
print(f"  CI violations increase: {ci_gap:+.2f}%")
print()
if rmse_gap < 5:
    print("  → Small gap: Model relies mostly on AMZN history")
else:
    print("  → Large gap: Strong cointegration signal from MSFT/SP500")
print()

# Mixed vs Realistic
print("3. Mixed Training vs Realistic (80/20 Training Effect)")
print("-" * 80)
mixed_benefit_rmse = ((metrics_s3["rmse"] - metrics_s2["rmse"]) / metrics_s3["rmse"]) * 100
mixed_benefit_ci = metrics_s3["ci_violation_rate"] - metrics_s2["ci_violation_rate"]
print(f"  RMSE improvement: {mixed_benefit_rmse:+.1f}%")
print(f"  CI violations improvement: {mixed_benefit_ci:+.2f}%")
print()
if abs(mixed_benefit_rmse) < 1:
    print("  → Mixed training has minimal effect on accuracy")
else:
    print("  → Mixed training helps model generalize to backfilling")
print()

# Direction Accuracy
print("4. Direction Accuracy (Predictive Signal)")
print("-" * 80)
for scenario, acc in zip(df.index, df["direction_acc"]):
    if acc > 52:
        status = "✓ Strong signal"
    elif acc > 50:
        status = "✓ Weak signal"
    else:
        status = "✗ No signal (worse than random)"
    print(f"  {scenario:20s}: {acc:5.2f}% {status}")
print()

# CI Width Comparison
print("5. Confidence Interval Width (Uncertainty)")
print("-" * 80)
for scenario, width in zip(df.index, df["mean_ci_width"]):
    print(f"  {scenario:20s}: {width:.4f}")
print()
if metrics_s3["mean_ci_width"] > metrics_s1["mean_ci_width"]:
    increase_pct = ((metrics_s3["mean_ci_width"] - metrics_s1["mean_ci_width"]) / metrics_s1["mean_ci_width"]) * 100
    print(f"  → Realistic scenario has {increase_pct:.1f}% wider CIs (appropriate uncertainty)")
else:
    print(f"  → Realistic scenario has narrower CIs (may be overconfident)")
print()

print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print()
print("Summary:")
print(f"  - Best overall: {df['pinball_loss'].idxmin()}")
print(f"  - Best CI calibration: {df['ci_violation_rate'].apply(lambda x: abs(x - 10)).idxmin()}")
print(f"  - Best direction accuracy: {df['direction_acc'].idxmax()}")
print()
print("Files saved:")
print(f"  - Metrics: {OUTPUT_FILE}")
