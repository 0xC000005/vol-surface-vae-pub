"""
Visualize empirical log variance and returns over time.

Examines whether the model learned higher uncertainty during crisis periods
(2008 financial crisis, COVID-19 crash) when returns change rapidly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

print("="*70)
print("Visualizing Empirical Uncertainty vs Returns Over Time")
print("="*70)

# Load data
print("\n[1/4] Loading data...")
data = np.load("data/vol_surface_with_ret.npz")
returns = data["ret"]  # (5822,)
print(f"  Returns shape: {returns.shape}")

# Load dates from parquet
df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
dates = pd.to_datetime(df['date'])
print(f"  Dates shape: {len(dates)}")
print(f"  Date range: {dates.iloc[0]} to {dates.iloc[-1]}")

# Model configurations
models_config = [
    {"name": "no_ex", "label": "No EX (Surface Only)"},
    {"name": "ex_no_loss", "label": "EX No Loss (Passive Features)"},
    {"name": "ex_loss", "label": "EX Loss (Joint Optimization)"}
]

base_folder = "test_spx/2024_11_09"
ctx_len = 5

# Load empirical latents for all models
print("\n[2/4] Loading empirical latents...")
latent_data = {}
for model_cfg in models_config:
    model_name = model_cfg["name"]
    latents_path = f"{base_folder}/{model_name}_empirical_latents.npz"
    latents = np.load(latents_path)

    z_mean_pool = latents["z_mean_pool"]  # (5817, 5)
    z_log_var_pool = latents["z_log_var_pool"]  # (5817, 5)

    # Compute variance: exp(log_var)
    variance_pool = np.exp(z_log_var_pool)  # (5817, 5)

    # Aggregate across latent dimensions (mean variance)
    mean_variance = np.mean(variance_pool, axis=1)  # (5817,)

    latent_data[model_name] = {
        "z_mean": z_mean_pool,
        "z_log_var": z_log_var_pool,
        "variance": variance_pool,
        "mean_variance": mean_variance,
        "label": model_cfg["label"]
    }

    print(f"  {model_name:12s}: z_log_var mean={np.mean(z_log_var_pool):.3f}, "
          f"variance mean={np.mean(mean_variance):.3f}")

# Align indices: latents start at day ctx_len (day 5)
# latent[i] corresponds to day[i + ctx_len]
latent_days = np.arange(ctx_len, ctx_len + len(latent_data["no_ex"]["mean_variance"]))
latent_dates = dates.iloc[latent_days].values  # Convert to numpy array for easier indexing
latent_returns = returns[latent_days]

print(f"\n  Aligned data: {len(latent_dates)} days from {latent_dates[0]} to {latent_dates[-1]}")

# Define crisis periods
crisis_periods = [
    {
        "name": "2008 Financial Crisis",
        "start": datetime(2008, 9, 1),
        "end": datetime(2009, 3, 31),
        "color": "red"
    },
    {
        "name": "COVID-19 Crash",
        "start": datetime(2020, 2, 15),
        "end": datetime(2020, 5, 1),
        "color": "orange"
    }
]

print("\n[3/4] Creating visualization...")

# Create figure with 3 rows (one per model) × 2 columns (returns, variance)
fig, axes = plt.subplots(3, 2, figsize=(20, 12))
fig.suptitle('Empirical Uncertainty vs Returns Over Time\n'
             'Does the model learn higher uncertainty during crisis periods?',
             fontsize=16, fontweight='bold')

for i, model_name in enumerate(["no_ex", "ex_no_loss", "ex_loss"]):
    data = latent_data[model_name]
    label = data["label"]

    # Left column: Returns
    ax_ret = axes[i, 0]
    ax_ret.plot(latent_dates, latent_returns, linewidth=0.5, color='black', alpha=0.7)
    ax_ret.set_ylabel('Daily Log Return', fontsize=10)
    ax_ret.set_title(f'{label}\nReturns', fontsize=11, fontweight='bold')
    ax_ret.grid(True, alpha=0.3)
    ax_ret.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Highlight crisis periods
    for crisis in crisis_periods:
        ax_ret.axvspan(crisis["start"], crisis["end"],
                      alpha=0.15, color=crisis["color"],
                      label=crisis["name"] if i == 0 else None)

    if i == 0:
        ax_ret.legend(loc='upper left', fontsize=8)

    # Right column: Mean variance
    ax_var = axes[i, 1]
    ax_var.plot(latent_dates, data["mean_variance"], linewidth=0.8, color='blue')
    ax_var.set_ylabel('Mean Variance\n(avg across 5 latent dims)', fontsize=10)
    ax_var.set_title(f'{label}\nMean Variance = exp(log_var)', fontsize=11, fontweight='bold')
    ax_var.grid(True, alpha=0.3)

    # Highlight crisis periods
    for crisis in crisis_periods:
        ax_var.axvspan(crisis["start"], crisis["end"],
                      alpha=0.15, color=crisis["color"])

    # Add statistics
    stats_text = (f"Mean: {np.mean(data['mean_variance']):.3f}\n"
                  f"Std: {np.std(data['mean_variance']):.3f}\n"
                  f"Min: {np.min(data['mean_variance']):.3f}\n"
                  f"Max: {np.max(data['mean_variance']):.3f}")
    ax_var.text(0.02, 0.98, stats_text, transform=ax_var.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Set x-axis label only for bottom row
    if i == 2:
        ax_ret.set_xlabel('Date', fontsize=10)
        ax_var.set_xlabel('Date', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = 'empirical_uncertainty_vs_returns.png'
print(f"\n[4/4] Saving figure to {output_path}...")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# Additional analysis: compute correlation during crisis periods
print("\n" + "="*70)
print("Analysis: Correlation between |returns| and variance during crises")
print("="*70)

for model_name in ["no_ex", "ex_no_loss", "ex_loss"]:
    data = latent_data[model_name]
    label = data["label"]

    print(f"\n{label}:")

    for crisis in crisis_periods:
        # Find indices for crisis period
        # Convert numpy datetime64 to pandas Timestamp for comparison
        latent_dates_ts = pd.to_datetime(latent_dates)
        mask = (latent_dates_ts >= crisis["start"]) & (latent_dates_ts <= crisis["end"])

        if np.sum(mask) == 0:
            print(f"  {crisis['name']}: No data in this period")
            continue

        crisis_returns = np.abs(latent_returns[mask])
        crisis_variance = data["mean_variance"][mask]

        # Compute correlation
        corr = np.corrcoef(crisis_returns, crisis_variance)[0, 1]

        print(f"  {crisis['name']}:")
        print(f"    Days: {np.sum(mask)}")
        print(f"    |Return| mean: {np.mean(crisis_returns):.5f}")
        print(f"    Variance mean: {np.mean(crisis_variance):.3f}")
        print(f"    Correlation: {corr:.3f}")

# Overall correlation
print("\n" + "="*70)
print("Overall Correlation (|returns| vs variance)")
print("="*70)

for model_name in ["no_ex", "ex_no_loss", "ex_loss"]:
    data = latent_data[model_name]
    label = data["label"]

    abs_returns = np.abs(latent_returns)
    corr = np.corrcoef(abs_returns, data["mean_variance"])[0, 1]

    print(f"{label:35s}: {corr:.3f}")

print("\n" + "="*70)
print("Visualization Complete!")
print("="*70)
print(f"\nInterpretation:")
print("- Positive correlation means higher uncertainty during volatile periods (good!)")
print("- Near-zero or negative correlation means model doesn't capture time-varying uncertainty")
print(f"\nGenerated: {output_path}")
