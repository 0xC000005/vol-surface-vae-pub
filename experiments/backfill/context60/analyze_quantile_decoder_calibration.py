#!/usr/bin/env python3
"""
Hypothesis 4: Quantile Decoder Calibration Issue

Question: Does quantile regression decoder miscalibrate at short horizons?

Note: This is a theoretical analysis since we need actual predictions.
We'll analyze the implications based on model configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "context60_baseline" / "analysis" / "day1_over_dispersion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HYPOTHESIS 4: QUANTILE DECODER CALIBRATION ANALYSIS")
print("=" * 80)
print()

print("Model Configuration:")
print("  Quantile regression: True")
print("  Quantile loss weights: [5.0, 1.0, 5.0]")
print("  Interpretation: p05/p95 penalized 5× more than p50")
print()

# Based on H3 findings
print("=" * 80)
print("ANALYSIS BASED ON H3 FINDINGS")
print("=" * 80)
print()

print("From H3 (Latent Information Bottleneck):")
print("  - Marginal/Individual ratio: 2.40×")
print("  - This means marginal spread is 2.40× wider than individual bands")
print("  - Individual bands: ~0.036 (estimated from ratio)")
print("  - Marginal spread: 0.0858")
print()

# GT comparison
gt_marginal = 0.0247
individual_est = 0.0858 / 2.40
individual_over_dispersion = individual_est / gt_marginal

print("Estimated Individual Band Analysis:")
print(f"  Individual band width: ~{individual_est:.4f}")
print(f"  GT marginal spread: {gt_marginal:.4f}")
print(f"  Over-dispersion factor: {individual_over_dispersion:.2f}×")
print()

if individual_over_dispersion > 1.5:
    print("⚠️  POSSIBLE DECODER ISSUE: Individual bands > 1.5× GT")
    print("    Quantile loss weights [5.0, 1.0, 5.0] may be too aggressive")
else:
    print("✅ DECODER CALIBRATION REASONABLE: Individual bands < 1.5× GT")
    print("    Over-dispersion is primarily epistemic (from H3)")

print()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Spread decomposition
ax = axes[0]
categories = ['GT\nMarginal', 'Model\nIndividual\n(est)', 'Model\nMarginal']
spreads = [gt_marginal, individual_est, 0.0858]
colors = ['green', 'orange', 'red']

bars = ax.bar(categories, spreads, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Spread (p95-p05)', fontsize=12)
ax.set_title('Day-1 Spread Decomposition', fontsize=13)
ax.grid(axis='y', alpha=0.3)

for bar, spread in zip(bars, spreads):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{spread:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add annotations
ax.annotate('', xy=(1, individual_est), xytext=(2, 0.0858),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.text(1.5, (individual_est + 0.0858)/2, f'Epistemic\n2.40×',
        ha='center', fontsize=10, color='purple', fontweight='bold')

# Right: Over-dispersion sources
ax = axes[1]
sources = ['GT Regime\nVariation', 'Decoder\nCalibration', 'Epistemic\nUncertainty']
contributions = [1.64, 1.45, 2.40]
colors_pie = ['lightblue', 'lightcoral', 'gold']

wedges, texts, autotexts = ax.pie(contributions, labels=sources, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
ax.set_title('Over-Dispersion Sources\n(multiplicative factors)', fontsize=13)

for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'quantile_decoder_calibration.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'quantile_decoder_calibration.png'}")

print()

# === VERDICT ===
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

if individual_over_dispersion < 1.5:
    print("❌ HYPOTHESIS REJECTED: Decoder calibration is NOT the primary cause")
    print()
    print("Evidence:")
    print(f"  - Individual bands only {individual_over_dispersion:.2f}× GT (reasonable)")
    print(f"  - Marginal spread 3.47× GT (high)")
    print(f"  - Gap of 2.40× comes from epistemic uncertainty (H3)")
    print()
    print("IMPLICATION:")
    print("  - Quantile loss weights [5.0, 1.0, 5.0] are reasonable")
    print("  - Over-dispersion is primarily epistemic (latent bottleneck from H3)")
    print("  - Not a decoder calibration problem")
else:
    print("✅ HYPOTHESIS CONFIRMED: Decoder calibration contributes to over-dispersion")
    print()
    print("Evidence:")
    print(f"  - Individual bands {individual_over_dispersion:.2f}× wider than GT")
    print("  - Quantile loss weights [5.0, 1.0, 5.0] may be too aggressive")
    print()
    print("RECOMMENDED FIX:")
    print("  - Reduce quantile weights: [5.0, 1.0, 5.0] → [2.0, 1.0, 2.0]")
    print("  - Or use horizon-dependent weights (lower for short horizons)")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Note: This analysis is based on H3 findings. For precise individual band")
print("measurements, actual model predictions would be needed.")
