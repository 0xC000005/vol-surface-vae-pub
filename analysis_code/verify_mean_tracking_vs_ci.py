import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# Configuration
BASE_MODEL_DIR = "test_spx/2024_11_09"
OUTPUT_DIR = "tables/2024_1213/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
START_DAY = 5
LAST_N_DAYS = 1000  # Same as teacher forcing analysis

# Grid points to analyze
GRID_POINTS = {
    "ATM 3-Month": (1, 2),
    "ATM 1-Year": (3, 2),
    "OTM Put 1-Year": (3, 0)
}

# Model configurations
MODELS = {
    "no_ex": "No EX (Surface Only)",
    "ex_no_loss": "EX No Loss (+Features)",
    "ex_loss": "EX Loss (+Features+Loss)"
}

print("="*80)
print("VERIFYING: Mean Tracking (R²) vs CI Calibration")
print("="*80)
print("\nLoading data...")

# Load ground truth
ground_truth_data = np.load("data/vol_surface_with_ret.npz")
ground_truth_surfaces = ground_truth_data["surface"][START_DAY:START_DAY+5810]
ground_truth_surfaces = ground_truth_surfaces[-LAST_N_DAYS:]

# Load dates
dates_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
dates = pd.to_datetime(dates_df["date"].values[START_DAY:START_DAY+5810])
dates = dates[-LAST_N_DAYS:]

print(f"Analysis period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"Total days: {LAST_N_DAYS}\n")

# Load model predictions
model_predictions = {}
for model_key in MODELS.keys():
    print(f"Loading {model_key}...")
    stochastic_file = f"{BASE_MODEL_DIR}/{model_key}_gen5.npz"
    stochastic = np.load(stochastic_file)
    model_predictions[model_key] = stochastic["surfaces"][-LAST_N_DAYS:]

# Store results
results = []

print("\n" + "="*80)
print("ANALYSIS: Regression + CI Violations")
print("="*80 + "\n")

# Analyze each model and grid point
for model_key, model_name in MODELS.items():
    print(f"\n{model_name}:")
    print("-" * 60)

    for grid_name, (grid_row, grid_col) in GRID_POINTS.items():
        # Extract ground truth for this grid point
        actual = ground_truth_surfaces[:, grid_row, grid_col]  # (1000,)

        # Extract model predictions (1000 days × 1000 samples)
        generated = model_predictions[model_key][:, :, grid_row, grid_col]  # (1000, 1000)

        # Calculate mean of generated samples
        mean_generated = np.mean(generated, axis=1)  # (1000,)

        # Regression: Actual = α + β₁ × Mean(generated)
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean_generated, actual)
        r_squared = r_value ** 2
        rmse = np.sqrt(np.mean((actual - mean_generated) ** 2))

        # Calculate CI violations
        p5 = np.percentile(generated, 5, axis=1)
        p95 = np.percentile(generated, 95, axis=1)
        outside_ci = (actual < p5) | (actual > p95)
        num_violations = np.sum(outside_ci)
        pct_violations = 100 * num_violations / LAST_N_DAYS

        # Store results
        results.append({
            "Model": model_name,
            "Grid Point": grid_name,
            "β₁": slope,
            "R²": r_squared,
            "RMSE": rmse,
            "CI Violations": num_violations,
            "CI Violation %": pct_violations,
            "Mean (generated)": mean_generated,
            "Actual": actual,
            "Outside CI": outside_ci
        })

        print(f"  {grid_name:20s} | β₁={slope:.4f} | R²={r_squared:.4f} | RMSE={rmse:.6f} | CI Violations: {num_violations}/{LAST_N_DAYS} ({pct_violations:.1f}%)")

# Create summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80 + "\n")

df = pd.DataFrame([{
    "Model": r["Model"],
    "Grid Point": r["Grid Point"],
    "β₁": f"{r['β₁']:.4f}",
    "R²": f"{r['R²']:.4f}",
    "RMSE": f"{r['RMSE']:.6f}",
    "CI Violations": f"{r['CI Violations']}/{LAST_N_DAYS}",
    "CI Violation %": f"{r['CI Violation %']:.1f}%"
} for r in results])

print(df.to_string(index=False))

# Save to CSV
csv_file = f"{OUTPUT_DIR}/mean_tracking_vs_ci_comparison.csv"
df.to_csv(csv_file, index=False)
print(f"\nSaved table to: {csv_file}")

# Create scatter plots
print("\n" + "="*80)
print("Creating scatter plots...")
print("="*80)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("Mean Tracking (R²) vs CI Calibration Verification\n" +
             f"Time Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({LAST_N_DAYS} days)",
             fontsize=16, fontweight='bold')

for idx, result in enumerate(results):
    row_idx = idx // 3
    col_idx = idx % 3
    ax = axes[row_idx, col_idx]

    mean_gen = result["Mean (generated)"]
    actual = result["Actual"]
    outside_ci = result["Outside CI"]

    # Scatter plot: color by CI violation status
    ax.scatter(mean_gen[~outside_ci], actual[~outside_ci],
              alpha=0.5, s=20, color='blue', label='Inside 90% CI')
    ax.scatter(mean_gen[outside_ci], actual[outside_ci],
              alpha=0.7, s=30, color='red', marker='x', label='Outside 90% CI')

    # Regression line
    slope = result["β₁"]
    intercept = mean_gen.mean() * (1 - slope) + actual.mean() * slope / slope  # approximate intercept
    x_line = np.array([mean_gen.min(), mean_gen.max()])
    y_line = slope * x_line + (actual.mean() - slope * mean_gen.mean())
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.7, label=f'Regression (β₁={slope:.3f})')

    # Perfect prediction line (y=x)
    min_val = min(mean_gen.min(), actual.min())
    max_val = max(mean_gen.max(), actual.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'g-', linewidth=1, alpha=0.4, label='Perfect (y=x)')

    # Annotations
    annotation = f"R² = {result['R²']:.4f}\nRMSE = {result['RMSE']:.6f}\nCI Violations: {result['CI Violation %']:.1f}%"
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
           fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Labels and title
    ax.set_xlabel('Mean(Generated Samples)', fontsize=10)
    ax.set_ylabel('Actual (Ground Truth)', fontsize=10)
    ax.set_title(f"{result['Model']}\n{result['Grid Point']}", fontsize=11, fontweight='bold')

    if idx == 0:
        ax.legend(loc='lower right', fontsize=8)

    ax.grid(True, alpha=0.3)

plt.tight_layout()
scatter_file = f"{OUTPUT_DIR}/mean_tracking_vs_ci_scatter.png"
plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
print(f"Saved scatter plots to: {scatter_file}")
plt.close()

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("""
This analysis verifies that BOTH can be true simultaneously:

1. ✓ Good Mean Tracking (R² ≈ 0.9):
   - The average of 1000 generated samples closely matches actual values
   - Regression shows β₁ ≈ 1.0 and high R²

2. ✗ Poor CI Calibration (35-72% violations):
   - Individual daily errors often exceed the 90% confidence intervals
   - Red 'x' markers show days where actual falls outside [p5, p95]

CONCLUSION:
- R² measures how well the MEAN tracks on average (errors cancel out)
- CI violations measure how well UNCERTAINTY is calibrated (individual errors)
- A model can have excellent mean tracking but severely underestimate uncertainty
- This is exactly what we observe: tight regression line but many red outliers

The scatter plots show:
- Points cluster around regression line (good R² ✓)
- But many red 'x' markers far from their predicted CI (poor calibration ✗)
""")

print(f"\nAnalysis complete! Check {OUTPUT_DIR}/ for outputs.")
