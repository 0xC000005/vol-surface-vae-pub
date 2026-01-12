"""
Visualization Summary: Before vs After All 3 Fixes

Shows the key metric improvements from the 3-phase fixes:
1. Phase 1: Correlation Learning (diagonal NLL → full cov NLL)
2. Phase 2: Z Usage (0% → 22.9% contribution to mean)
3. Phase 3: Fat Tails (0.6% → 64.4% kurtosis recovery)

Usage:
    python experiments/backfill/two_stage_vae/plot_improvement_summary.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def create_improvement_summary():
    """Create visual summary of all improvements."""

    # Data from experiments
    metrics = {
        "Correlation\nPreserved": {
            "before": 3.0,
            "after": 96.3,
            "unit": "%",
            "target": 15.0,
        },
        "Z Contribution\nto Mean": {
            "before": 0.0,
            "after": 22.9,
            "unit": "%",
            "target": 20.0,
        },
        "Kurtosis\nRecovery": {
            "before": 0.6,
            "after": 64.4,
            "unit": "%",
            "target": 50.0,
        },
        "Crisis CI\nViolations": {
            "before": 87.0,
            "after": 13.7,
            "unit": "%",
            "target": 20.0,
            "lower_is_better": True,
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {
        "before": "#e74c3c",  # Red
        "after": "#2ecc71",   # Green
        "target": "#3498db",  # Blue
    }

    for idx, (metric_name, data) in enumerate(metrics.items()):
        ax = axes[idx]

        before = data["before"]
        after = data["after"]
        target = data["target"]
        lower_is_better = data.get("lower_is_better", False)

        x = [0, 1]
        heights = [before, after]

        bars = ax.bar(x, heights, color=[colors["before"], colors["after"]], width=0.6, edgecolor='black')

        # Add target line
        ax.axhline(y=target, color=colors["target"], linestyle='--', linewidth=2, label=f'Target: {target}%')

        # Add value labels
        for bar, val in zip(bars, heights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

        # Calculate improvement
        if lower_is_better:
            improvement = (before - after) / before * 100
            improvement_text = f"↓ {improvement:.0f}% reduction"
        else:
            if before > 0:
                improvement = (after - before) / before * 100
                improvement_text = f"↑ {improvement:.0f}x improvement"
            else:
                improvement_text = f"↑ {after:.1f}% (from 0)"

        ax.set_xticks(x)
        ax.set_xticklabels(['Before\nFixes', 'After\nAll Fixes'], fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title(f'{metric_name}\n{improvement_text}', fontsize=14, fontweight='bold')

        # Set y-axis limits
        max_val = max(before, after, target)
        ax.set_ylim(0, max_val * 1.2)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Two-Stage VAE: 3-Phase Fix Summary\n(Full Cov NLL + Z-Dependent Mean + Student-t)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    output_path = Path("results/two_stage_analysis/improvement_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary plot saved to {output_path}")

    # Also create a text summary
    print("\n" + "=" * 70)
    print("TWO-STAGE VAE: 3-PHASE FIX SUMMARY")
    print("=" * 70)

    print("\n┌─────────────────────────────┬──────────┬──────────┬─────────────┐")
    print("│ Metric                      │  Before  │  After   │ Improvement │")
    print("├─────────────────────────────┼──────────┼──────────┼─────────────┤")

    for metric_name, data in metrics.items():
        before = data["before"]
        after = data["after"]
        lower_is_better = data.get("lower_is_better", False)

        if lower_is_better:
            improvement = f"↓ {(before - after) / before * 100:.0f}%"
        else:
            if before > 0:
                improvement = f"↑ {after / before:.0f}x"
            else:
                improvement = f"↑ ∞"

        metric_clean = metric_name.replace('\n', ' ')
        print(f"│ {metric_clean:<27} │ {before:>7.1f}% │ {after:>7.1f}% │ {improvement:>11} │")

    print("└─────────────────────────────┴──────────┴──────────┴─────────────┘")

    print("\n" + "=" * 70)
    print("FIXES APPLIED")
    print("=" * 70)
    print("""
Phase 1: Full Covariance NLL (Woodbury Identity)
  - Problem: Diagonal NLL ignores off-diagonal correlations
  - Solution: Full NLL with Σ = FF^T + D, efficient via Woodbury
  - Result: 96.3% correlation preserved (was 3%)

Phase 2: Two-Phase Training for Z Usage
  - Problem: Decoder ignores z, uses variance to explain everything
  - Solution: Pre-train mean with MSE only, then add variance
  - Result: 22.9% z contribution to mean (was 0%)

Phase 3: Student-t Decoder with Fixed ν
  - Problem: Gaussian output has near-zero kurtosis
  - Solution: Student-t with ν = 4 + 6/(GT_kurtosis - 3)
  - Result: 64.4% kurtosis recovery (was 0.6%)
""")

    return metrics


if __name__ == "__main__":
    create_improvement_summary()
