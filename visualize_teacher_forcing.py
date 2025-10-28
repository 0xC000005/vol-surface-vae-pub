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
LAST_N_DAYS = 60  # Last 60 days to visualize

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

# Data structure to store CI violations for reporting
violation_stats = []

# Figure 1: Implied Vol - 3 rows (grid points) x 3 columns (models)
fig1, axes = plt.subplots(3, 3, figsize=(20, 14))
fig1.suptitle("Teacher Forcing Performance: Implied Volatility Predictions (Independent One-Step-Ahead Forecasts)\n" +
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

        # Detect CI violations
        outside_ci = (gt_surface < p5) | (gt_surface > p95)
        num_violations = np.sum(outside_ci)
        pct_violations = 100 * num_violations / len(gt_surface)

        # Store violation statistics
        violation_stats.append({
            "model": MODELS[model_key]["name"].replace("\n", " "),
            "grid_point": grid_name,
            "num_violations": num_violations,
            "pct_violations": pct_violations,
            "total_days": len(gt_surface),
            "rmse": rmse,
            "violation_dates": dates[outside_ci].tolist()
        })

        # Plot ground truth (semi-transparent to see CI violations)
        ax.plot(dates, gt_surface, 'k-', linewidth=2, alpha=0.6, label='Ground Truth', zorder=3)

        # Plot CI violations as red markers
        if num_violations > 0:
            ax.scatter(dates[outside_ci], gt_surface[outside_ci],
                      color='red', s=20, marker='o', alpha=0.7,
                      label='Outside 90% CI', zorder=4)

        # Plot MLE prediction
        model_color = MODELS[model_key]["color"]
        ax.plot(dates, mle_surface, color=model_color, linewidth=1.5,
                label='MLE (z=0, independent)', zorder=2)

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

        # RMSE and violation annotation
        annotation_text = f'RMSE: {rmse:.4f}\nOutside CI: {num_violations}/{len(gt_surface)} ({pct_violations:.1f}%)'
        ax.text(0.02, 0.98, annotation_text,
               transform=ax.transAxes, fontsize=9, va='top',
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
output_file_1 = f"{OUTPUT_DIR}/teacher_forcing_implied_vol_{LAST_N_DAYS}days.png"
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

    # Detect CI violations
    outside_ci_ret = (ground_truth_returns < p5_ret) | (ground_truth_returns > p95_ret)
    num_violations_ret = np.sum(outside_ci_ret)
    pct_violations_ret = 100 * num_violations_ret / len(ground_truth_returns)

    # Store violation statistics for returns
    violation_stats.append({
        "model": "EX Loss (Returns)",
        "grid_point": "Returns",
        "num_violations": num_violations_ret,
        "pct_violations": pct_violations_ret,
        "total_days": len(ground_truth_returns),
        "rmse": rmse_ret,
        "violation_dates": dates[outside_ci_ret].tolist()
    })

    # Plot ground truth (semi-transparent to see CI violations)
    ax.plot(dates, ground_truth_returns, 'k-', linewidth=2, alpha=0.6, label='Ground Truth Returns', zorder=3)

    # Plot CI violations as red markers
    if num_violations_ret > 0:
        ax.scatter(dates[outside_ci_ret], ground_truth_returns[outside_ci_ret],
                  color='red', s=20, marker='o', alpha=0.7,
                  label='Outside 90% CI', zorder=4)

    # Plot MLE prediction
    ax.plot(dates, mle_returns, color=MODELS["ex_loss"]["color"], linewidth=1.5,
            label='MLE Predicted Returns (z=0, independent)', zorder=2)

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
    ax.set_title(f"Teacher Forcing Performance: Return Predictions (EX Loss Model) - Independent One-Step-Ahead Forecasts\n" +
                f"Context Length = {CONTEXT_LEN} days | Time Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Daily Log Return', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)

    # RMSE and violation annotation
    annotation_text_ret = f'RMSE: {rmse_ret:.6f}\nOutside CI: {num_violations_ret}/{len(ground_truth_returns)} ({pct_violations_ret:.1f}%)'
    ax.text(0.02, 0.98, annotation_text_ret,
           transform=ax.transAxes, fontsize=11, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file_2 = f"{OUTPUT_DIR}/teacher_forcing_returns_{LAST_N_DAYS}days.png"
    plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file_2}")
    plt.close()

print("\n" + "="*80)
print("CI VIOLATION ANALYSIS")
print("="*80)

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE: Confidence Interval Violations")
print("="*80)
print(f"{'Model':<30} {'Grid Point':<20} {'Violations':<15} {'Percentage':<12} {'RMSE':<10}")
print("-" * 90)

for stat in violation_stats:
    print(f"{stat['model']:<30} {stat['grid_point']:<20} "
          f"{stat['num_violations']}/{stat['total_days']:<10} "
          f"{stat['pct_violations']:>6.2f}%     {stat['rmse']:.6f}")

print("="*80)

# Print detailed violation dates for each model/grid point
print("\n" + "="*80)
print("DETAILED VIOLATION DATES")
print("="*80)

for stat in violation_stats:
    print(f"\n{stat['model']} - {stat['grid_point']}")
    print(f"  Total violations: {stat['num_violations']} out of {stat['total_days']} days ({stat['pct_violations']:.2f}%)")

    if stat['num_violations'] > 0:
        print(f"  First 10 violation dates:")
        for i, date in enumerate(stat['violation_dates'][:10]):
            print(f"    {i+1}. {date.strftime('%Y-%m-%d')}")

        if stat['num_violations'] > 10:
            print(f"    ... and {stat['num_violations'] - 10} more")
    else:
        print(f"  No violations detected!")

# Save report to file
report_file = f"{OUTPUT_DIR}/ci_violations_report_{LAST_N_DAYS}days.txt"
with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CI VIOLATION ANALYSIS REPORT\n")
    f.write("="*80 + "\n")
    f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Time Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}\n")
    f.write(f"Context Length: {CONTEXT_LEN} days\n")
    f.write(f"Total Days: {LAST_N_DAYS}\n\n")

    f.write("="*80 + "\n")
    f.write("SUMMARY TABLE: Confidence Interval Violations\n")
    f.write("="*80 + "\n")
    f.write(f"{'Model':<30} {'Grid Point':<20} {'Violations':<15} {'Percentage':<12} {'RMSE':<10}\n")
    f.write("-" * 90 + "\n")

    for stat in violation_stats:
        f.write(f"{stat['model']:<30} {stat['grid_point']:<20} "
                f"{stat['num_violations']}/{stat['total_days']:<10} "
                f"{stat['pct_violations']:>6.2f}%     {stat['rmse']:.6f}\n")

    f.write("="*80 + "\n")

    f.write("\n" + "="*80 + "\n")
    f.write("DETAILED VIOLATION DATES\n")
    f.write("="*80 + "\n")

    for stat in violation_stats:
        f.write(f"\n{stat['model']} - {stat['grid_point']}\n")
        f.write(f"  Total violations: {stat['num_violations']} out of {stat['total_days']} days ({stat['pct_violations']:.2f}%)\n")

        if stat['num_violations'] > 0:
            f.write(f"  All violation dates:\n")
            for i, date in enumerate(stat['violation_dates']):
                f.write(f"    {i+1}. {date.strftime('%Y-%m-%d')}\n")
        else:
            f.write(f"  No violations detected!\n")

print(f"\nReport saved to: {report_file}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  1. {output_file_1}")
print(f"  2. {output_file_2}")
print(f"  3. {report_file}")
print(f"\nKey insights to look for:")
print(f"  - How well does MLE (z=0) track ground truth?")
print(f"  - Are uncertainty bands well-calibrated (ground truth within 90% CI)?")
print(f"  - Do models with extra features (ex_no_loss, ex_loss) outperform no_ex?")
print(f"  - How do models perform during COVID crisis (2020)?")
print(f"  - Are ATM predictions more accurate than OTM?")
print(f"  - Does ex_loss model predict returns well?")
print(f"\nNote: Red markers on plots indicate when ground truth falls outside 90% CI")
