#!/usr/bin/env python3
"""
Hypothesis 1: Training Data Statistics (Historical Mean Reversion)

Question: Does the training data itself show mean reversion at 90 days?

Method:
1. Analyze 23 years of ground truth data (2000-2023)
2. Compute empirical transition distributions: P(vol_t+90 | vol_t, context)
3. Stratify by context endpoint percentile (low/med/high vol)
4. Check if high-vol contexts actually converge to mean after 90 days

If TRUE: Model is learning correct pattern from data (not a bug, but a feature)
If FALSE: Model is over-fitting to spurious mean reversion → Need regularization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "vol_surface_with_ret.npz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "context60_baseline" / "analysis" / "mean_reversion_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HYPOTHESIS 1: TRAINING DATA MEAN REVERSION ANALYSIS")
print("=" * 80)
print()

# Load data
print("Loading ground truth data...")
data = np.load(DATA_PATH)
surfaces = data["surface"]  # (N, 5, 5)
print(f"Loaded {len(surfaces)} days of data")
print()

# We'll focus on center ATM point (2, 2) for simplicity
atm_values = surfaces[:, 2, 2]  # (N,)

# Define context and horizon
context_len = 60
horizon = 90

# Extract valid sequences: need context_len + horizon days
max_start_idx = len(atm_values) - context_len - horizon
print(f"Can create {max_start_idx} sequences with context={context_len}, horizon={horizon}")
print()

# Build dataset of (context_endpoint, target_endpoint) pairs
context_endpoints = []
target_endpoints = []
target_paths = []  # Full 90-day paths

for start_idx in range(max_start_idx):
    context_end_idx = start_idx + context_len
    target_end_idx = context_end_idx + horizon

    context_endpoint = atm_values[context_end_idx - 1]  # Last value of context
    target_endpoint = atm_values[target_end_idx - 1]    # Value at day-90
    target_path = atm_values[context_end_idx:target_end_idx]  # Full path

    context_endpoints.append(context_endpoint)
    target_endpoints.append(target_endpoint)
    target_paths.append(target_path)

context_endpoints = np.array(context_endpoints)
target_endpoints = np.array(target_endpoints)
target_paths = np.array(target_paths)  # (N, 90)

print(f"Created {len(context_endpoints)} (context, target) pairs")
print()

# Compute historical mean (across all data)
historical_mean = np.mean(atm_values)
print(f"Historical mean IV: {historical_mean:.4f}")
print()

# Stratify by context endpoint percentile
percentiles = [0, 25, 50, 75, 100]
percentile_labels = ["Low vol (0-25%)", "Medium-low (25-50%)", "Medium-high (50-75%)", "High vol (75-100%)"]

print("=" * 80)
print("STRATIFIED ANALYSIS BY CONTEXT ENDPOINT PERCENTILE")
print("=" * 80)
print()

results = []

for i in range(len(percentiles) - 1):
    p_low = percentiles[i]
    p_high = percentiles[i + 1]
    label = percentile_labels[i]

    # Get percentile thresholds
    threshold_low = np.percentile(context_endpoints, p_low)
    threshold_high = np.percentile(context_endpoints, p_high)

    # Filter sequences in this percentile range
    mask = (context_endpoints >= threshold_low) & (context_endpoints < threshold_high)
    n_sequences = mask.sum()

    if n_sequences == 0:
        continue

    context_in_bin = context_endpoints[mask]
    target_in_bin = target_endpoints[mask]

    # Compute statistics
    mean_context = np.mean(context_in_bin)
    mean_target = np.mean(target_in_bin)
    std_context = np.std(context_in_bin)
    std_target = np.std(target_in_bin)

    # Mean reversion metric: How much does target converge toward historical mean?
    # If mean reversion: high-vol contexts should move toward mean (downward)
    #                    low-vol contexts should move toward mean (upward)
    deviation_from_mean_context = mean_context - historical_mean
    deviation_from_mean_target = mean_target - historical_mean

    # Reversion coefficient: 0 = full reversion to mean, 1 = no reversion
    if abs(deviation_from_mean_context) > 1e-6:
        reversion_coef = deviation_from_mean_target / deviation_from_mean_context
    else:
        reversion_coef = 1.0

    # Correlation between context and target
    correlation = np.corrcoef(context_in_bin, target_in_bin)[0, 1]

    print(f"{label}:")
    print(f"  N sequences: {n_sequences}")
    print(f"  Context range: [{threshold_low:.4f}, {threshold_high:.4f}]")
    print(f"  Mean context: {mean_context:.4f} (±{std_context:.4f})")
    print(f"  Mean target:  {mean_target:.4f} (±{std_target:.4f})")
    print(f"  Context deviation from mean: {deviation_from_mean_context:+.4f}")
    print(f"  Target deviation from mean:  {deviation_from_mean_target:+.4f}")
    print(f"  Reversion coefficient: {reversion_coef:.3f} (0=full reversion, 1=no reversion)")
    print(f"  Correlation(context, target): {correlation:.3f}")
    print()

    results.append({
        "percentile_range": label,
        "n_sequences": n_sequences,
        "mean_context": mean_context,
        "mean_target": mean_target,
        "std_context": std_context,
        "std_target": std_target,
        "deviation_context": deviation_from_mean_context,
        "deviation_target": deviation_from_mean_target,
        "reversion_coef": reversion_coef,
        "correlation": correlation,
    })

# Save results table
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_DIR / "mean_reversion_by_percentile.csv", index=False)
print(f"Saved results to: {OUTPUT_DIR / 'mean_reversion_by_percentile.csv'}")
print()

# Interpretation
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

# Check if mean reversion is present
high_vol_reversion = results[3]["reversion_coef"]  # High vol bin
low_vol_reversion = results[0]["reversion_coef"]   # Low vol bin

print("Mean Reversion Evidence:")
print(f"  High vol bin reversion coef: {high_vol_reversion:.3f}")
print(f"  Low vol bin reversion coef:  {low_vol_reversion:.3f}")
print()

# Strong mean reversion: reversion_coef < 0.5
# No mean reversion: reversion_coef ≈ 1.0
mean_reversion_present = False

if high_vol_reversion < 0.7:
    print("✅ STRONG MEAN REVERSION DETECTED in high-vol regime")
    print(f"   High-vol contexts move {(1 - high_vol_reversion) * 100:.1f}% toward historical mean")
    mean_reversion_present = True
elif high_vol_reversion < 0.9:
    print("⚠️  MODERATE MEAN REVERSION in high-vol regime")
    mean_reversion_present = True
else:
    print("❌ NO MEAN REVERSION detected in high-vol regime")

print()

if low_vol_reversion > 1.3:
    print("✅ MEAN REVERSION CONFIRMED in low-vol regime (moves above mean)")
    mean_reversion_present = True
elif low_vol_reversion > 1.1:
    print("⚠️  MODERATE MEAN REVERSION in low-vol regime")
    mean_reversion_present = True
else:
    print("❌ NO MEAN REVERSION in low-vol regime")

print()
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

if mean_reversion_present:
    print("✅ HYPOTHESIS CONFIRMED: Training data exhibits mean reversion")
    print()
    print("IMPLICATION:")
    print("  - Model is learning a REAL pattern from the data")
    print("  - Mean reversion is a feature, not a bug")
    print("  - BUT: Model may be over-confident in this pattern")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Accept mean reversion exists (can't fight reality)")
    print("  2. Increase aleatory uncertainty at long horizons (heteroscedastic decoder)")
    print("  3. Add diversity penalty to maintain epistemic uncertainty")
    print("  4. Use regime-weighted training (up-weight high-vol periods)")
else:
    print("❌ HYPOTHESIS REJECTED: Training data does NOT show strong mean reversion")
    print()
    print("IMPLICATION:")
    print("  - Model is over-fitting to spurious mean reversion")
    print("  - Need to investigate other causes (loss function, architecture)")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Investigate loss function bias (H2)")
    print("  2. Check decoder horizon sensitivity (H4)")
    print("  3. Consider alternative prediction targets (H5)")

print()

# Visualization 1: Context vs Target scatter by percentile
print("Generating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i in range(4):
    ax = axes[i]
    p_low = percentiles[i]
    p_high = percentiles[i + 1]
    label = percentile_labels[i]

    threshold_low = np.percentile(context_endpoints, p_low)
    threshold_high = np.percentile(context_endpoints, p_high)
    mask = (context_endpoints >= threshold_low) & (context_endpoints < threshold_high)

    context_in_bin = context_endpoints[mask]
    target_in_bin = target_endpoints[mask]

    # Scatter plot
    ax.scatter(context_in_bin, target_in_bin, alpha=0.3, s=10)

    # Reference lines
    ax.axhline(historical_mean, color='red', linestyle='--', alpha=0.7, label=f'Historical mean ({historical_mean:.4f})')
    ax.plot([threshold_low, threshold_high], [threshold_low, threshold_high],
            'k--', alpha=0.5, label='No change (45° line)')

    # Regression line
    slope, intercept = np.polyfit(context_in_bin, target_in_bin, 1)
    x_range = np.array([threshold_low, threshold_high])
    ax.plot(x_range, slope * x_range + intercept, 'g-', linewidth=2,
            label=f'Linear fit (slope={slope:.3f})')

    ax.set_xlabel('Context Endpoint (t=0)', fontsize=11)
    ax.set_ylabel('Target Endpoint (t=90)', fontsize=11)
    ax.set_title(f'{label}\nN={mask.sum()}, Corr={results[i]["correlation"]:.3f}', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'context_vs_target_by_percentile.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'context_vs_target_by_percentile.png'}")

# Visualization 2: Mean reversion coefficient bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(results))
reversion_coefs = [r["reversion_coef"] for r in results]
colors = ['blue' if r < 0.7 else 'orange' if r < 0.9 else 'red' for r in reversion_coefs]

bars = ax.bar(x_pos, reversion_coefs, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='No reversion (1.0)')
ax.axhline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='50% reversion')

ax.set_xlabel('Volatility Regime', fontsize=12)
ax.set_ylabel('Mean Reversion Coefficient', fontsize=12)
ax.set_title('Mean Reversion by Volatility Regime (90-day horizon)\n' +
             '0 = Full reversion to mean, 1 = No reversion', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([r["percentile_range"] for r in results], rotation=15, ha='right')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, reversion_coefs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mean_reversion_coefficients.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'mean_reversion_coefficients.png'}")

# Visualization 3: Path evolution for each regime
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i in range(4):
    ax = axes[i]
    p_low = percentiles[i]
    p_high = percentiles[i + 1]
    label = percentile_labels[i]

    threshold_low = np.percentile(context_endpoints, p_low)
    threshold_high = np.percentile(context_endpoints, p_high)
    mask = (context_endpoints >= threshold_low) & (context_endpoints < threshold_high)

    paths_in_bin = target_paths[mask]  # (N, 90)

    # Compute percentile bands
    p05 = np.percentile(paths_in_bin, 5, axis=0)
    p25 = np.percentile(paths_in_bin, 25, axis=0)
    p50 = np.percentile(paths_in_bin, 50, axis=0)
    p75 = np.percentile(paths_in_bin, 75, axis=0)
    p95 = np.percentile(paths_in_bin, 95, axis=0)

    horizons = np.arange(1, 91)

    # Plot bands
    ax.fill_between(horizons, p05, p95, alpha=0.2, color='blue', label='5-95%')
    ax.fill_between(horizons, p25, p75, alpha=0.3, color='blue', label='25-75%')
    ax.plot(horizons, p50, 'b-', linewidth=2, label='Median (p50)')

    # Reference: historical mean
    ax.axhline(historical_mean, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Historical mean ({historical_mean:.4f})')

    # Starting level
    mean_start = np.mean(paths_in_bin[:, 0])
    ax.axhline(mean_start, color='green', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'Mean start ({mean_start:.4f})')

    ax.set_xlabel('Horizon (days)', fontsize=11)
    ax.set_ylabel('ATM Implied Volatility', fontsize=11)
    ax.set_title(f'{label}\nN={mask.sum()} paths', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'path_evolution_by_regime.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'path_evolution_by_regime.png'}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"All results saved to: {OUTPUT_DIR}")
print()
print("Next steps:")
print("  1. Review visualizations to confirm mean reversion pattern")
print("  2. Proceed to H2 (loss function bias analysis)")
print("  3. Synthesize findings from Phase 1 experiments")
