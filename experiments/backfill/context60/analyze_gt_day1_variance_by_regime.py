#!/usr/bin/env python3
"""
Hypothesis 5: Ground Truth Variance Analysis

Question: Is GT day-1 spread (0.0248) misleadingly narrow due to regime mixing?

Logic:
- GT 0.0248 is aggregated across all regimes (crisis + calm)
- Crisis periods might have wider day-1 spreads
- Model's 0.0858 might be realistic for mixed regimes

Expected Evidence if TRUE:
- GT day-1 spread varies: crisis ~0.04-0.05, calm ~0.01-0.02
- Model's 0.0858 is reasonable average across regimes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "vol_surface_with_ret.npz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "context60_baseline" / "analysis" / "day1_over_dispersion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HYPOTHESIS 5: GROUND TRUTH DAY-1 VARIANCE BY REGIME")
print("=" * 80)
print()

# Load data
print("Loading ground truth data...")
data = np.load(DATA_PATH)
surfaces = data["surface"]  # (N, 5, 5)
atm_values = surfaces[:, 2, 2]  # (N,) ATM point
print(f"Loaded {len(surfaces)} days of data")
print()

# Define context and horizon
context_len = 60
horizon_day1 = 1

# Extract valid sequences
max_start_idx = len(atm_values) - context_len - horizon_day1
print(f"Can create {max_start_idx} sequences with context={context_len}")
print()

# Build dataset of (context_endpoint, day1_target) pairs
context_endpoints = []
day1_targets = []

for start_idx in range(max_start_idx):
    context_end_idx = start_idx + context_len
    day1_idx = context_end_idx  # Day 1 after context

    context_endpoint = atm_values[context_end_idx - 1]
    day1_target = atm_values[day1_idx]

    context_endpoints.append(context_endpoint)
    day1_targets.append(day1_target)

context_endpoints = np.array(context_endpoints)
day1_targets = np.array(day1_targets)

print(f"Created {len(context_endpoints)} (context, day1_target) pairs")
print()

# Compute day-1 changes (what we're predicting)
day1_changes = day1_targets - context_endpoints

# Overall marginal spread (baseline)
overall_p05 = np.percentile(day1_changes, 5)
overall_p50 = np.percentile(day1_changes, 50)
overall_p95 = np.percentile(day1_changes, 95)
overall_spread = overall_p95 - overall_p05

print("OVERALL GT DAY-1 STATISTICS:")
print(f"  p05: {overall_p05:.4f}")
print(f"  p50: {overall_p50:.4f}")
print(f"  p95: {overall_p95:.4f}")
print(f"  Spread (p95-p05): {overall_spread:.4f}")
print()

# Actually measure marginal distribution of day-1 p50 forecasts
# We need to look at p50s across different contexts
# Group by context endpoint percentile and compute spread of medians

# Stratify by volatility regime based on context endpoint
percentiles = np.array([stats.percentileofscore(context_endpoints, val, kind='rank')
                        for val in context_endpoints])

regimes = {
    "Low vol (0-25%)": (percentiles >= 0) & (percentiles < 25),
    "Medium vol (25-75%)": (percentiles >= 25) & (percentiles < 75),
    "High vol (75-100%)": (percentiles >= 75) & (percentiles <= 100)
}

print("=" * 80)
print("REGIME-STRATIFIED ANALYSIS")
print("=" * 80)
print()

results = []

for regime_name, mask in regimes.items():
    n_sequences = mask.sum()

    if n_sequences == 0:
        continue

    # Day-1 changes for this regime
    day1_changes_regime = day1_changes[mask]
    context_endpoints_regime = context_endpoints[mask]

    # Compute marginal distribution statistics
    p05 = np.percentile(day1_changes_regime, 5)
    p50 = np.percentile(day1_changes_regime, 50)
    p95 = np.percentile(day1_changes_regime, 95)
    spread = p95 - p05

    # Standard deviation
    std = np.std(day1_changes_regime)

    # Mean context level
    mean_context = np.mean(context_endpoints_regime)

    print(f"{regime_name}:")
    print(f"  N sequences: {n_sequences}")
    print(f"  Mean context endpoint: {mean_context:.4f}")
    print(f"  Day-1 change p05: {p05:.4f}")
    print(f"  Day-1 change p50: {p50:.4f}")
    print(f"  Day-1 change p95: {p95:.4f}")
    print(f"  Spread (p95-p05): {spread:.4f}")
    print(f"  Std dev: {std:.4f}")
    print()

    results.append({
        "regime": regime_name,
        "n_sequences": n_sequences,
        "mean_context": mean_context,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "spread": spread,
        "std": std
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_DIR / "gt_day1_variance_by_regime.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'gt_day1_variance_by_regime.csv'}")
print()

# Interpretation
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

low_vol_spread = results[0]["spread"]
high_vol_spread = results[2]["spread"]

print(f"Overall GT day-1 spread: {overall_spread:.4f}")
print(f"Low vol regime spread:   {low_vol_spread:.4f}")
print(f"High vol regime spread:  {high_vol_spread:.4f}")
print()

# Compare to model's 0.0858
model_spread = 0.0858

print(f"Model day-1 spread: {model_spread:.4f}")
print()

# Check if model spread is within range of regime variation
if low_vol_spread < model_spread < high_vol_spread:
    print("✅ Model spread is WITHIN range of GT regime variation")
    print("   This suggests model's 0.0858 is reasonable for mixed regimes")
elif model_spread > high_vol_spread:
    print("⚠️  Model spread EXCEEDS even high-vol regime")
    print(f"   Model over-dispersed by {((model_spread / high_vol_spread) - 1) * 100:.1f}%")
else:
    print("❌ Model spread is BELOW low-vol regime")
    print("   Unexpected - model should not be under-dispersed")

print()

# Visualization 1: Regime comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Spread by regime
ax = axes[0]
regime_names = [r["regime"] for r in results]
spreads = [r["spread"] for r in results]

x_pos = np.arange(len(regime_names))
bars = ax.bar(x_pos, spreads, alpha=0.7, edgecolor='black')

# Color bars
colors = ['green', 'orange', 'red']
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Add overall and model lines
ax.axhline(overall_spread, color='blue', linestyle='--', linewidth=2,
           label=f'Overall GT ({overall_spread:.4f})')
ax.axhline(model_spread, color='purple', linestyle='--', linewidth=2,
           label=f'Model ({model_spread:.4f})')

ax.set_ylabel('Day-1 Spread (p95-p05)', fontsize=12)
ax.set_title('GT Day-1 Spread by Volatility Regime', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels(regime_names, rotation=15, ha='right')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, spread) in enumerate(zip(bars, spreads)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{spread:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: Distribution histograms
ax = axes[1]

for i, (regime_name, mask) in enumerate(regimes.items()):
    changes = day1_changes[mask]
    ax.hist(changes, bins=50, alpha=0.4, label=regime_name, color=colors[i])

ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Day-1 Change in ATM IV', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Day-1 Changes by Regime', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gt_day1_variance_by_regime.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'gt_day1_variance_by_regime.png'}")

# Visualization 2: Quantile plots by regime
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (regime_name, mask) in enumerate(regimes.items()):
    ax = axes[i]

    changes = day1_changes[mask]
    contexts = context_endpoints[mask]

    # Scatter plot: context endpoint vs day-1 change
    ax.scatter(contexts, changes, alpha=0.3, s=10)

    # Add quantile lines
    p05_line = np.percentile(changes, 5)
    p50_line = np.percentile(changes, 50)
    p95_line = np.percentile(changes, 95)

    ax.axhline(p05_line, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'p05 ({p05_line:.4f})')
    ax.axhline(p50_line, color='green', linestyle='-', alpha=0.7, linewidth=2,
               label=f'p50 ({p50_line:.4f})')
    ax.axhline(p95_line, color='blue', linestyle='--', alpha=0.7, linewidth=2,
               label=f'p95 ({p95_line:.4f})')

    ax.set_xlabel('Context Endpoint (ATM IV)', fontsize=11)
    ax.set_ylabel('Day-1 Change', fontsize=11)
    ax.set_title(f'{regime_name}\nN={mask.sum()}, Spread={results[i]["spread"]:.4f}',
                 fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gt_day1_scatter_by_regime.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'gt_day1_scatter_by_regime.png'}")

print()

# Verdict
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

regime_spread_range = high_vol_spread - low_vol_spread
regime_variation_pct = (regime_spread_range / overall_spread) * 100

print(f"Regime spread variation: {low_vol_spread:.4f} to {high_vol_spread:.4f}")
print(f"Variation range: {regime_spread_range:.4f} ({regime_variation_pct:.1f}% of overall)")
print()

if regime_variation_pct > 50:  # Significant variation
    print("✅ HYPOTHESIS CONFIRMED: GT day-1 spread is regime-dependent")
    print()
    print("IMPLICATION:")
    print(f"  - Overall GT spread ({overall_spread:.4f}) masks regime differences")
    print(f"  - Low vol: {low_vol_spread:.4f}, High vol: {high_vol_spread:.4f}")

    if low_vol_spread < model_spread < high_vol_spread:
        print(f"  - Model spread ({model_spread:.4f}) is REASONABLE for mixed regimes")
        print()
        print("RECOMMENDED ACTION:")
        print("  - Use regime-conditional evaluation")
        print("  - Don't penalize model for being between low/high vol spreads")
    else:
        print(f"  - Model spread ({model_spread:.4f}) is still OUTSIDE regime range")
        print("  - Other factors contribute to over-dispersion (H3 or H4)")
else:
    print("❌ HYPOTHESIS REJECTED: GT day-1 spread is fairly uniform across regimes")
    print()
    print("IMPLICATION:")
    print(f"  - Overall GT spread ({overall_spread:.4f}) is representative")
    print(f"  - Model spread ({model_spread:.4f}) is genuinely too wide")
    print()
    print("RECOMMENDED NEXT STEPS:")
    print("  - Investigate H3 (latent bottleneck) or H4 (decoder calibration)")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
