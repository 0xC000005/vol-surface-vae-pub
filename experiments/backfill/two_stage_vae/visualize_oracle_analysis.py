"""
Visualize Comprehensive Oracle Analysis Results

Creates visualizations for all analysis findings.

Usage:
    python experiments/backfill/two_stage_vae/visualize_oracle_analysis.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def load_results():
    """Load analysis results from JSON."""
    json_path = "results/two_stage_analysis/comprehensive_analysis_results.json"
    with open(json_path, "r") as f:
        return json.load(f)


def plot_bottleneck_analysis(results, output_dir):
    """Plot bottleneck capacity test results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bottleneck = results["bottleneck"]
    categories = ["Full Model\n(ctx + z)", "Ctx Only\n(z = 0)", "Z Only\n(ctx = 0)"]
    mses = [
        bottleneck["mse_full"],
        bottleneck["mse_ctx_only"],
        bottleneck["mse_z_only"]
    ]

    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(categories, mses, color=colors, edgecolor="black", linewidth=2)

    # Add value labels
    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{mse:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Bottleneck Capacity Test\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add contribution annotations
    z_contrib = bottleneck["z_contribution_pct"]
    ctx_contrib = bottleneck["ctx_contribution_pct"]
    ax.text(0.5, 0.95, f'Z Contribution: {z_contrib:.1f}%\nCtx Contribution: {ctx_contrib:.1f}%',
            transform=ax.transAxes, ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "bottleneck_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bottleneck_analysis.png")


def plot_directional_accuracy(results, output_dir):
    """Plot directional accuracy by period."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    directional = results["directional"]
    periods = [k for k in directional.keys() if k != "summary"]

    dir_accs = [directional[p]["direction_accuracy"] * 100 for p in periods]
    mag_ratios = [directional[p]["magnitude_ratio"] * 100 for p in periods]

    # Direction accuracy
    ax1 = axes[0]
    colors = ["#e74c3c" if d < 50 else "#2ecc71" for d in dir_accs]
    bars = ax1.barh(periods, dir_accs, color=colors, edgecolor="black")
    ax1.axvline(x=50, color='black', linestyle='--', linewidth=2, label='Random (50%)')
    ax1.axvline(x=60, color='green', linestyle='--', linewidth=2, label='Target (60%)')
    ax1.set_xlabel('Direction Accuracy (%)', fontsize=11)
    ax1.set_title('Direction Accuracy by Period\n(>50% means better than random)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3, axis='x')

    for bar, acc in zip(bars, dir_accs):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2., f'{acc:.0f}%',
                va='center', fontsize=10, fontweight='bold')

    # Magnitude ratio
    ax2 = axes[1]
    colors = ["#e74c3c" if m < 50 else "#2ecc71" for m in mag_ratios]
    bars = ax2.barh(periods, mag_ratios, color=colors, edgecolor="black")
    ax2.axvline(x=50, color='green', linestyle='--', linewidth=2, label='Target (50%)')
    ax2.axvline(x=100, color='gray', linestyle='--', linewidth=2, label='Perfect (100%)')
    ax2.set_xlabel('Magnitude Ratio (%)', fontsize=11)
    ax2.set_title('Predicted vs GT Magnitude\n(|pred| / |GT|)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 120)
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, ratio in zip(bars, mag_ratios):
        ax2.text(ratio + 1, bar.get_y() + bar.get_height()/2., f'{ratio:.0f}%',
                va='center', fontsize=10, fontweight='bold')

    plt.suptitle('Directional Analysis: Model Predicts Wrong Direction ~70% of Time',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "directional_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: directional_accuracy.png")


def plot_per_grid_errors(results, output_dir):
    """Plot per-grid MSE and direction accuracy heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    per_grid = results["per_grid"]
    mse_ratio = np.array(per_grid["mse_ratio_to_atm"])
    dir_acc = np.array(per_grid["grid_direction_acc"]) * 100

    # MSE ratio heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(mse_ratio, cmap='Reds', aspect='equal')
    ax1.set_title('MSE Ratio to ATM\n(ATM = 1.0)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time to Maturity')
    ax1.set_ylabel('Moneyness')
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(['1M', '3M', '6M', '1Y', '2Y'])
    ax1.set_yticklabels(['90%', '95%', 'ATM', '105%', '110%'])

    # Add text annotations
    for i in range(5):
        for j in range(5):
            color = 'white' if mse_ratio[i, j] > 50 else 'black'
            ax1.text(j, i, f'{mse_ratio[i, j]:.0f}×', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    plt.colorbar(im1, ax=ax1, label='MSE Ratio')

    # Direction accuracy heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(dir_acc, cmap='RdYlGn', aspect='equal', vmin=40, vmax=70)
    ax2.set_title('Direction Accuracy (%)\n(50% = random)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time to Maturity')
    ax2.set_ylabel('Moneyness')
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(['1M', '3M', '6M', '1Y', '2Y'])
    ax2.set_yticklabels(['90%', '95%', 'ATM', '105%', '110%'])

    # Add text annotations
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f'{dir_acc[i, j]:.0f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold')

    plt.colorbar(im2, ax=ax2, label='Accuracy (%)')

    plt.suptitle('Per-Grid Error Analysis: Corners 248× Worse Than ATM',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "per_grid_errors.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_grid_errors.png")


def plot_correlation_matrices(results, output_dir):
    """Plot GT vs Model correlation matrices side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    corr = results["correlation"]
    gt_corr = np.array(corr["gt_corr_matrix"])
    sample_corr = np.array(corr["sample_corr_matrix"])
    diff = sample_corr - gt_corr

    # GT correlation matrix
    ax1 = axes[0]
    im1 = ax1.imshow(gt_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax1.set_title('Ground Truth Correlation', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Grid Point')
    ax1.set_ylabel('Grid Point')
    plt.colorbar(im1, ax=ax1)

    # Sample correlation matrix
    ax2 = axes[1]
    im2 = ax2.imshow(sample_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax2.set_title('Model Sample Correlation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Grid Point')
    plt.colorbar(im2, ax=ax2)

    # Difference
    ax3 = axes[2]
    im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax3.set_title('Difference (Model - GT)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Grid Point')
    plt.colorbar(im3, ax=ax3)

    preserved = corr["correlation_preserved_pct"]
    corr_of_corr = corr["correlation_of_correlations"]
    plt.suptitle(f'Correlation Matrix Comparison\n'
                 f'Preserved: {preserved:.1f}% | Correlation of Correlations: {corr_of_corr:.3f}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: correlation_matrices.png")


def plot_kurtosis_comparison(results, output_dir):
    """Plot GT vs Model kurtosis heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    kurtosis = results["kurtosis"]
    gt_kurt = np.array(kurtosis["gt_kurtosis"])
    model_kurt = np.array(kurtosis["model_kurtosis"])
    recovery = np.array(kurtosis["recovery_ratio"]) * 100

    # Use log scale for better visualization
    gt_kurt_log = np.log10(np.abs(gt_kurt) + 1)
    model_kurt_log = np.log10(np.abs(model_kurt) + 1)

    vmax = max(gt_kurt_log.max(), model_kurt_log.max())

    # GT kurtosis
    ax1 = axes[0]
    im1 = ax1.imshow(gt_kurt, cmap='YlOrRd', aspect='equal')
    ax1.set_title('GT Kurtosis (Fisher)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time to Maturity')
    ax1.set_ylabel('Moneyness')
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(['1M', '3M', '6M', '1Y', '2Y'])
    ax1.set_yticklabels(['90%', '95%', 'ATM', '105%', '110%'])
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, f'{gt_kurt[i, j]:.1f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im1, ax=ax1)

    # Model kurtosis
    ax2 = axes[1]
    im2 = ax2.imshow(model_kurt, cmap='YlOrRd', aspect='equal')
    ax2.set_title('Model Kurtosis (Fisher)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time to Maturity')
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(['1M', '3M', '6M', '1Y', '2Y'])
    ax2.set_yticklabels(['90%', '95%', 'ATM', '105%', '110%'])
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f'{model_kurt[i, j]:.1f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im2, ax=ax2)

    # Recovery ratio
    ax3 = axes[2]
    im3 = ax3.imshow(recovery, cmap='RdYlGn', aspect='equal', vmin=0, vmax=200)
    ax3.set_title('Recovery Ratio (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time to Maturity')
    ax3.set_xticks(range(5))
    ax3.set_yticks(range(5))
    ax3.set_xticklabels(['1M', '3M', '6M', '1Y', '2Y'])
    ax3.set_yticklabels(['90%', '95%', 'ATM', '105%', '110%'])
    for i in range(5):
        for j in range(5):
            ax3.text(j, i, f'{recovery[i, j]:.0f}%', ha='center', va='center', fontsize=8)
    plt.colorbar(im3, ax=ax3, label='%')

    mean_recovery = kurtosis["mean_recovery"] * 100
    plt.suptitle(f'Kurtosis Analysis: Student-t Achieves {mean_recovery:.0f}% Average Recovery',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "kurtosis_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: kurtosis_comparison.png")


def plot_acf_comparison(results, output_dir):
    """Plot autocorrelation function comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    acf = results["acf"]
    gt_acf = np.array(acf["gt_acf"])
    model_acf = np.array(acf["model_acf"])
    lags = np.arange(len(gt_acf))

    ax.bar(lags - 0.15, gt_acf, width=0.3, label='Ground Truth', color='#3498db', edgecolor='black')
    ax.bar(lags + 0.15, model_acf, width=0.3, label='Model', color='#e74c3c', edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Autocorrelation Function Comparison (ATM)\n'
                 f'Lag-1 Preservation: {acf["lag1_preservation"]*100:.1f}%',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(lags)

    plt.tight_layout()
    plt.savefig(output_dir / "acf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: acf_comparison.png")


def plot_summary_dashboard(results, output_dir):
    """Create summary dashboard with all key metrics."""
    fig = plt.figure(figsize=(16, 12))

    # Define metrics
    metrics = {
        "Z Contribution": {
            "value": results["bottleneck"]["z_contribution_pct"],
            "target": 50,
            "unit": "%",
            "higher_better": True,
        },
        "Direction Accuracy": {
            "value": results["directional"]["summary"]["avg_direction_accuracy"] * 100,
            "target": 60,
            "unit": "%",
            "higher_better": True,
        },
        "Magnitude Ratio": {
            "value": results["directional"]["summary"]["avg_magnitude_ratio"] * 100,
            "target": 50,
            "unit": "%",
            "higher_better": True,
        },
        "Correlation Preserved": {
            "value": results["correlation"]["correlation_preserved_pct"],
            "target": 50,
            "unit": "%",
            "higher_better": True,
        },
        "Kurtosis Recovery": {
            "value": results["kurtosis"]["mean_recovery"] * 100,
            "target": 100,
            "unit": "%",
            "higher_better": True,
        },
        "ACF Preservation": {
            "value": abs(results["acf"]["lag1_preservation"]) * 100,
            "target": 50,
            "unit": "%",
            "higher_better": True,
        },
    }

    # Create gauge-like bars
    ax = fig.add_subplot(111)
    ax.axis('off')

    y_positions = np.arange(len(metrics)) * 1.5
    bar_height = 0.8

    for i, (name, data) in enumerate(metrics.items()):
        value = data["value"]
        target = data["target"]
        y = y_positions[i]

        # Background bar (target)
        ax.barh(y, 100, height=bar_height, color='#ecf0f1', edgecolor='black')

        # Value bar
        color = '#2ecc71' if value >= target else '#e74c3c'
        ax.barh(y, min(value, 100), height=bar_height, color=color, edgecolor='black')

        # Target line
        ax.axvline(x=target, ymin=(y - bar_height/2) / (y_positions[-1] + 1.5),
                  ymax=(y + bar_height/2) / (y_positions[-1] + 1.5),
                  color='black', linestyle='--', linewidth=2)

        # Labels
        ax.text(-5, y, name, ha='right', va='center', fontsize=12, fontweight='bold')
        ax.text(min(value, 100) + 2, y, f'{value:.1f}{data["unit"]}', ha='left', va='center', fontsize=11)
        ax.text(target, y + bar_height/2 + 0.1, f'Target: {target}%', ha='center', va='bottom', fontsize=9)

        # Status
        status = "PASS" if value >= target else "FAIL"
        status_color = '#2ecc71' if value >= target else '#e74c3c'
        ax.text(105, y, status, ha='left', va='center', fontsize=11, fontweight='bold', color=status_color)

    ax.set_xlim(-40, 120)
    ax.set_ylim(-1, y_positions[-1] + 1.5)

    plt.title('Two-Stage VAE Oracle Analysis Summary\n'
              'Key Metrics vs Targets',
              fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_dashboard.png")


def plot_issue_severity(results, output_dir):
    """Plot issue severity ranking."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define issues and their severity scores (0-100, higher = worse)
    issues = {
        "Direction Accuracy\n(29% vs 60% target)": 100 - results["directional"]["summary"]["avg_direction_accuracy"] * 100 / 0.6 * 100,
        "Magnitude Attenuation\n(19% of GT)": 100 - results["directional"]["summary"]["avg_magnitude_ratio"] * 100 / 50 * 100,
        "ACF Destruction\n(10% preservation)": 100 - abs(results["acf"]["lag1_preservation"]) * 100,
        "Correlation Gap\n(30% preserved)": 100 - results["correlation"]["correlation_preserved_pct"],
        "Z Contribution\n(40% vs 50% target)": 100 - results["bottleneck"]["z_contribution_pct"] / 50 * 100,
        "Per-Grid MSE Gap\n(248× corner/ATM)": min(results["per_grid"]["max_mse_ratio"] / 10 * 100, 100),
    }

    # Sort by severity
    sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_issues]
    severities = [x[1] for x in sorted_issues]

    colors = ['#e74c3c' if s > 70 else '#f39c12' if s > 40 else '#2ecc71' for s in severities]

    bars = ax.barh(names, severities, color=colors, edgecolor='black', linewidth=2)

    ax.set_xlabel('Severity Score (Higher = Worse)', fontsize=12)
    ax.set_title('Issue Severity Ranking\n(Prioritized by Impact)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(True, alpha=0.3, axis='x')

    # Add severity labels
    for bar, sev in zip(bars, severities):
        label = "CRITICAL" if sev > 70 else "HIGH" if sev > 40 else "MODERATE"
        ax.text(sev + 2, bar.get_y() + bar.get_height()/2., f'{sev:.0f} ({label})',
                va='center', fontsize=10, fontweight='bold')

    # Legend
    critical_patch = mpatches.Patch(color='#e74c3c', label='Critical (>70)')
    high_patch = mpatches.Patch(color='#f39c12', label='High (40-70)')
    moderate_patch = mpatches.Patch(color='#2ecc71', label='Moderate (<40)')
    ax.legend(handles=[critical_patch, high_patch, moderate_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / "issue_severity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: issue_severity.png")


def main():
    """Generate all visualizations."""
    print("="*70)
    print("Generating Oracle Analysis Visualizations")
    print("="*70)

    results = load_results()

    output_dir = Path("results/two_stage_analysis/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_bottleneck_analysis(results, output_dir)
    plot_directional_accuracy(results, output_dir)
    plot_per_grid_errors(results, output_dir)
    plot_correlation_matrices(results, output_dir)
    plot_kurtosis_comparison(results, output_dir)
    plot_acf_comparison(results, output_dir)
    plot_summary_dashboard(results, output_dir)
    plot_issue_severity(results, output_dir)

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
