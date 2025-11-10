"""
Visualize 1D Backfilling Model Predictions.

Creates time series plots comparing 3 latent selection scenarios:
1. Oracle (S1): Upper bound with future information
2. Mixed 80/20 (S2): Training distribution
3. Realistic (S3): Production scenario

Plots:
- Time series with confidence intervals (full + zoomed)
- Error distribution analysis
- Metric comparison charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path

# Configuration
PREDICTIONS_FILE = "models_1d_backfilling/backfill_predictions_ctx5.npz"
METRICS_FILE = "models_1d_backfilling/comparison_metrics.csv"
OUTPUT_DIR = "models_1d_backfilling/plots"
ZOOM_N_DAYS = 100  # For zoomed plot
DPI = 300  # Output resolution

# Scenario configurations
SCENARIOS = {
    "s1": {
        "name": "Oracle (S1)",
        "color": "#2ca02c",  # Green
        "description": "Upper bound (has future information)"
    },
    "s2": {
        "name": "Mixed 80/20 (S2)",
        "color": "#ff7f0e",  # Orange
        "description": "Training distribution"
    },
    "s3": {
        "name": "Realistic (S3)",
        "color": "#1f77b4",  # Blue
        "description": "Production scenario"
    }
}

print("=" * 80)
print("VISUALIZING 1D BACKFILLING MODEL PREDICTIONS")
print("=" * 80)
print(f"Predictions file: {PREDICTIONS_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Convert dates to datetime
dates = pd.to_datetime(dates)

# Extract predictions per scenario
predictions = {}
for scenario_key in ["s1", "s2", "s3"]:
    predictions[scenario_key] = {
        "p05": data[f"{scenario_key}_p05"],
        "p50": data[f"{scenario_key}_p50"],
        "p95": data[f"{scenario_key}_p95"],
    }

print(f"  Loaded {len(actuals)} predictions")
print(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print()

# Load metrics if available
metrics_df = None
if os.path.exists(METRICS_FILE):
    print("Loading evaluation metrics...")
    metrics_df = pd.read_csv(METRICS_FILE, index_col="scenario")
    print("  Metrics loaded successfully")
    print()


def get_annotation_color(violation_rate):
    """
    Get annotation box color based on CI violation rate.

    Target: ~10% violations
    Green: 8-12% (well calibrated)
    Yellow: 5-8% or 12-20% (moderate)
    Red: <5% or >20% (poor)
    """
    if 8 <= violation_rate <= 12:
        return 'lightgreen'
    elif 5 <= violation_rate <= 20:
        return 'wheat'
    else:
        return 'lightcoral'


def compute_subplot_metrics(p05, p50, p95, actual):
    """Compute metrics for a single subplot."""
    rmse = np.sqrt(np.mean((p50 - actual) ** 2))
    mae = np.mean(np.abs(p50 - actual))

    # CI violations
    outside_ci = (actual < p05) | (actual > p95)
    num_violations = np.sum(outside_ci)
    pct_violations = (num_violations / len(actual)) * 100

    # CI width
    ci_width = np.mean(p95 - p05)

    # Direction accuracy
    pred_signs = np.sign(p50)
    actual_signs = np.sign(actual)
    direction_acc = np.mean(pred_signs == actual_signs) * 100

    return {
        "rmse": rmse,
        "mae": mae,
        "num_violations": num_violations,
        "pct_violations": pct_violations,
        "ci_width": ci_width,
        "direction_acc": direction_acc,
        "outside_ci": outside_ci
    }


def plot_timeseries_subplot(ax, dates, actuals, p05, p50, p95,
                            scenario_name, scenario_color, scenario_description, metrics):
    """
    Plot a single time series subplot with CI.

    Args:
        ax: matplotlib axis
        dates: datetime array
        actuals: ground truth
        p05, p50, p95: quantile predictions
        scenario_name: scenario label
        scenario_color: line/fill color
        scenario_description: scenario description text
        metrics: dict with computed metrics
    """
    # Plot ground truth
    ax.plot(dates, actuals, 'k-', linewidth=2.5, alpha=0.7,
            label='Ground Truth', zorder=3)

    # Plot CI violations
    outside_ci = metrics["outside_ci"]
    if metrics["num_violations"] > 0:
        ax.scatter(dates[outside_ci], actuals[outside_ci],
                  color='red', s=30, marker='o', alpha=0.8,
                  label='Outside 90% CI', zorder=4)

    # Plot median prediction
    ax.plot(dates, p50, color=scenario_color, linewidth=2,
            label='p50 (Median)', zorder=2)

    # Shaded CI
    ax.fill_between(dates, p05, p95, alpha=0.3, color=scenario_color,
                    label='90% CI [p05, p95]', zorder=1)

    # Zero line
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Labels and title
    ax.set_ylabel('AMZN Return', fontsize=11, fontweight='bold')
    ax.set_title(f'{scenario_name}: {scenario_description}',
                fontsize=12, fontweight='bold', pad=10)

    # Annotation box
    annotation_text = (
        f'RMSE: {metrics["rmse"]:.4f}\n'
        f'MAE: {metrics["mae"]:.4f}\n'
        f'CI Width: {metrics["ci_width"]:.4f}\n'
        f'Violations: {metrics["num_violations"]}/{len(actuals)} ({metrics["pct_violations"]:.1f}%)\n'
        f'Direction Acc: {metrics["direction_acc"]:.1f}%'
    )

    box_color = get_annotation_color(metrics["pct_violations"])

    ax.text(0.02, 0.98, annotation_text,
           transform=ax.transAxes, fontsize=9, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7),
           zorder=5)

    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if len(dates) <= 100:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    elif len(dates) <= 250:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)


# ============================================================================
# PLOT 1: FULL TIME SERIES COMPARISON (3 SCENARIOS)
# ============================================================================
print("Creating full time series comparison plot...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('1D Backfilling Model: Time Series Comparison (All Data)',
             fontsize=16, fontweight='bold', y=0.995)

for idx, (scenario_key, scenario_info) in enumerate(SCENARIOS.items()):
    ax = axes[idx]

    # Get predictions for this scenario
    pred = predictions[scenario_key]

    # Compute metrics
    metrics = compute_subplot_metrics(pred["p05"], pred["p50"], pred["p95"], actuals)

    # Plot
    plot_timeseries_subplot(ax, dates, actuals,
                           pred["p05"], pred["p50"], pred["p95"],
                           scenario_info["name"], scenario_info["color"],
                           scenario_info["description"], metrics)

    # Legend only on first subplot
    if idx == 0:
        ax.legend(loc='upper left', fontsize=9, ncol=4, framealpha=0.9)

# Common x-label
axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "1d_backfill_timeseries_all.png")
plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {output_path}")
print()


# ============================================================================
# PLOT 2: ZOOMED TIME SERIES (LAST N DAYS)
# ============================================================================
print(f"Creating zoomed time series plot (last {ZOOM_N_DAYS} days)...")

# Get last N days
start_idx = max(0, len(dates) - ZOOM_N_DAYS)
dates_zoom = dates[start_idx:]
actuals_zoom = actuals[start_idx:]

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f'1D Backfilling Model: Time Series Comparison (Last {ZOOM_N_DAYS} Days)',
             fontsize=16, fontweight='bold', y=0.995)

for idx, (scenario_key, scenario_info) in enumerate(SCENARIOS.items()):
    ax = axes[idx]

    # Get predictions for this scenario (zoomed)
    pred = {
        "p05": predictions[scenario_key]["p05"][start_idx:],
        "p50": predictions[scenario_key]["p50"][start_idx:],
        "p95": predictions[scenario_key]["p95"][start_idx:],
    }

    # Compute metrics
    metrics = compute_subplot_metrics(pred["p05"], pred["p50"], pred["p95"], actuals_zoom)

    # Plot
    plot_timeseries_subplot(ax, dates_zoom, actuals_zoom,
                           pred["p05"], pred["p50"], pred["p95"],
                           scenario_info["name"], scenario_info["color"],
                           scenario_info["description"], metrics)

    # Legend only on first subplot
    if idx == 0:
        ax.legend(loc='upper left', fontsize=9, ncol=4, framealpha=0.9)

# Common x-label
axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "1d_backfill_timeseries_zoom.png")
plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {output_path}")
print()


# ============================================================================
# PLOT 3: ERROR ANALYSIS
# ============================================================================
print("Creating error analysis plot...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('1D Backfilling Model: Error Analysis',
             fontsize=16, fontweight='bold', y=0.995)

# Row 1: Error histograms
for idx, (scenario_key, scenario_info) in enumerate(SCENARIOS.items()):
    ax = axes[0, idx]

    # Compute errors (actual - predicted)
    errors = actuals - predictions[scenario_key]["p50"]

    # Plot histogram
    ax.hist(errors, bins=50, color=scenario_info["color"], alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')

    # Statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    ax.set_title(f'{scenario_info["name"]}: Error Distribution',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediction Error (Actual - p50)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)

    # Annotation
    ann_text = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}'
    ax.text(0.02, 0.98, ann_text,
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8)

# Row 2: CI width evolution
for idx, (scenario_key, scenario_info) in enumerate(SCENARIOS.items()):
    ax = axes[1, idx]

    # Compute CI width
    ci_width = predictions[scenario_key]["p95"] - predictions[scenario_key]["p05"]

    # Plot time series
    ax.plot(dates, ci_width, color=scenario_info["color"], linewidth=1.5, alpha=0.8)
    ax.fill_between(dates, ci_width, alpha=0.3, color=scenario_info["color"])

    # Mean line
    mean_width = np.mean(ci_width)
    ax.axhline(mean_width, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_width:.4f}')

    ax.set_title(f'{scenario_info["name"]}: CI Width Evolution',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('CI Width (p95 - p05)', fontsize=10)

    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "1d_backfill_error_analysis.png")
plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {output_path}")
print()


# ============================================================================
# PLOT 4: METRICS COMPARISON BAR CHART
# ============================================================================
if metrics_df is not None:
    print("Creating metrics comparison bar chart...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('1D Backfilling Model: Metrics Comparison',
                 fontsize=16, fontweight='bold', y=1.0)

    # Prepare data
    scenario_names = [SCENARIOS[k]["name"] for k in ["s1", "s2", "s3"]]
    scenario_colors = [SCENARIOS[k]["color"] for k in ["s1", "s2", "s3"]]

    # Left: RMSE, MAE, R²
    ax1 = axes[0]
    x = np.arange(3)
    width = 0.25

    rmse_vals = [metrics_df.loc[name, "rmse"] for name in ["Oracle (S1)", "Mixed 80/20 (S2)", "Realistic (S3)"]]
    mae_vals = [metrics_df.loc[name, "mae"] for name in ["Oracle (S1)", "Mixed 80/20 (S2)", "Realistic (S3)"]]
    r2_vals = [metrics_df.loc[name, "r2"] * 100 for name in ["Oracle (S1)", "Mixed 80/20 (S2)", "Realistic (S3)"]]  # Scale to %

    ax1.bar(x - width, rmse_vals, width, label='RMSE', alpha=0.8, color='#1f77b4')
    ax1.bar(x, mae_vals, width, label='MAE', alpha=0.8, color='#ff7f0e')
    ax1.bar(x + width, r2_vals, width, label='R² (×100)', alpha=0.8, color='#2ca02c')

    ax1.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax1.set_title('Point Forecast Metrics', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Oracle', 'Mixed', 'Realistic'])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

    # Right: CI Violation Rate, Direction Accuracy
    ax2 = axes[1]

    ci_viol_vals = [metrics_df.loc[name, "ci_violation_rate"] for name in ["Oracle (S1)", "Mixed 80/20 (S2)", "Realistic (S3)"]]
    dir_acc_vals = [metrics_df.loc[name, "direction_acc"] for name in ["Oracle (S1)", "Mixed 80/20 (S2)", "Realistic (S3)"]]

    ax2.bar(x - width/2, ci_viol_vals, width, label='CI Violation Rate (%)', alpha=0.8, color='#d62728')
    ax2.bar(x + width/2, dir_acc_vals, width, label='Direction Accuracy (%)', alpha=0.8, color='#9467bd')

    # Target line for CI violations
    ax2.axhline(10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target: 10%')

    ax2.set_xlabel('Scenario', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('CI Calibration & Direction Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Oracle', 'Mixed', 'Realistic'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "1d_backfill_metrics_comparison.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    print()
else:
    print("⚠ Skipping metrics comparison (metrics CSV not found)")
    print()


# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print()
print("Generated plots:")
print(f"  1. {OUTPUT_DIR}/1d_backfill_timeseries_all.png")
print(f"  2. {OUTPUT_DIR}/1d_backfill_timeseries_zoom.png")
print(f"  3. {OUTPUT_DIR}/1d_backfill_error_analysis.png")
if metrics_df is not None:
    print(f"  4. {OUTPUT_DIR}/1d_backfill_metrics_comparison.png")
print()
print(f"All plots saved at {DPI} DPI")
