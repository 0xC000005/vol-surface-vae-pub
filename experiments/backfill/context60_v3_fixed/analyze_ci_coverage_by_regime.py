"""
Analyze Individual CI Coverage by Volatility Regime

Tests whether V3 prior model's quantile outputs (p05, p95) provide good CI coverage
for individual trajectories at H=90, stratified by 4 volatility regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)
OUTPUT_DIR = Path("results/context60_latent12_v3_FIXED/analysis")


def stratify_by_regime(context_vols):
    """Split indices into 4 volatility regimes."""
    q25, q50, q75 = np.percentile(context_vols, [25, 50, 75])

    regimes = {
        'Q1 Low Vol': context_vols <= q25,
        'Q2 Med-Low Vol': (context_vols > q25) & (context_vols <= q50),
        'Q3 Med-High Vol': (context_vols > q50) & (context_vols <= q75),
        'Q4 High Vol': context_vols > q75
    }

    return regimes, (q25, q50, q75)


def compute_ci_coverage(p05, p50, p95, gt):
    """
    Compute CI coverage - % of cases where GT falls within [p05, p95].

    Args:
        p05, p50, p95: (N, 90) prediction arrays
        gt: (N, 90) ground truth array

    Returns:
        coverage_by_horizon: (90,) array of coverage rates
    """
    # Check if GT within CI at each horizon
    within_ci = (gt >= p05) & (gt <= p95)  # (N, 90)

    # Coverage rate at each horizon
    coverage_by_horizon = within_ci.mean(axis=0)  # (90,)

    return coverage_by_horizon, within_ci


def plot_coverage_by_regime(coverage_stats, output_path):
    """Bar chart of H=90 coverage by regime."""
    fig, ax = plt.subplots(figsize=(10, 6))

    regimes = list(coverage_stats.keys())
    coverages = [coverage_stats[r]['h90_coverage'] for r in regimes]

    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
    bars = ax.bar(regimes, [c*100 for c in coverages], color=colors, alpha=0.8, edgecolor='black')

    # Add target line
    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')

    # Add value labels
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cov*100:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('CI Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Volatility Regime', fontsize=12, fontweight='bold')
    ax.set_title('Individual Trajectory CI Coverage at H=90 by Regime\n(V3 Prior Mode)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_trajectories_by_regime(p05, p50, p95, gt, context_vols, regimes, quartiles, output_path):
    """4-panel plot showing sample trajectories with CI bands for each regime."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    days = np.arange(1, HORIZON + 1)
    n_samples_per_regime = 20

    regime_names = list(regimes.keys())

    for i, (regime_name, mask) in enumerate(regimes.items()):
        ax = axes[i]

        # Get indices for this regime
        regime_indices = np.where(mask)[0]

        # Sample some trajectories
        n_plot = min(n_samples_per_regime, len(regime_indices))
        plot_indices = np.random.choice(regime_indices, n_plot, replace=False)

        # Plot trajectories with CI bands
        for idx in plot_indices:
            # Check if GT falls within CI at H=90
            within_ci_h90 = (gt[idx, -1] >= p05[idx, -1]) and (gt[idx, -1] <= p95[idx, -1])
            gt_color = 'green' if within_ci_h90 else 'red'
            gt_alpha = 1.0 if not within_ci_h90 else 0.3

            # Plot CI band
            ax.fill_between(days, p05[idx], p95[idx], alpha=0.1, color='blue')

            # Plot p50
            ax.plot(days, p50[idx], color='blue', alpha=0.2, linewidth=1)

            # Plot GT
            ax.plot(days, gt[idx], color=gt_color, alpha=gt_alpha, linewidth=1.5)

        # Coverage stats
        within_ci_h90 = (gt[regime_indices, -1] >= p05[regime_indices, -1]) & \
                        (gt[regime_indices, -1] <= p95[regime_indices, -1])
        coverage = within_ci_h90.mean()

        # Regime bounds
        if i == 0:
            bounds = f"vol ≤ {quartiles[0]:.3f}"
        elif i == 1:
            bounds = f"{quartiles[0]:.3f} < vol ≤ {quartiles[1]:.3f}"
        elif i == 2:
            bounds = f"{quartiles[1]:.3f} < vol ≤ {quartiles[2]:.3f}"
        else:
            bounds = f"vol > {quartiles[2]:.3f}"

        ax.set_title(f'{regime_name}\n{bounds}\nH=90 Coverage: {coverage*100:.1f}% (n={len(regime_indices)})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('ATM 6M IV')
        ax.grid(True, alpha=0.3)

        # Add legend on first panel
        if i == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', alpha=0.2, label='p50 (median)'),
                Line2D([0], [0], color='blue', alpha=0.3, linewidth=5, label='90% CI band'),
                Line2D([0], [0], color='green', linewidth=2, label='GT (within CI)'),
                Line2D([0], [0], color='red', linewidth=2, label='GT (outside CI)')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.suptitle('Individual Trajectory CI Coverage by Volatility Regime (V3 Prior)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_coverage_by_horizon(coverage_by_regime_horizon, output_path):
    """Line plot showing how coverage degrades across horizons for each regime."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
    horizons_to_plot = [1, 7, 14, 30, 60, 90]

    for (regime_name, coverage), color in zip(coverage_by_regime_horizon.items(), colors):
        # Plot at selected horizons
        horizon_indices = [h-1 for h in horizons_to_plot]
        ax.plot(horizons_to_plot, [coverage[i]*100 for i in horizon_indices],
                marker='o', linewidth=2, label=regime_name, color=color, markersize=8)

    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_xlabel('Horizon (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CI Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('CI Coverage Degradation Across Horizons by Regime (V3 Prior)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_xticks(horizons_to_plot)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CI COVERAGE ANALYSIS BY VOLATILITY REGIME")
    print("="*80)

    # Load predictions
    pred_file = Path("results/context60_latent12_v3_FIXED/predictions/teacher_forcing/prior/vae_tf_insample_h90.npz")
    print(f"\nLoading predictions: {pred_file}")
    pred_data = np.load(pred_file)

    surfaces = pred_data['surfaces']  # (N, 90, 3, 5, 5)
    indices = pred_data['indices']

    # Extract quantiles at ATM 6M
    p05 = surfaces[:, :, 0, ATM_6M[0], ATM_6M[1]]  # (N, 90)
    p50 = surfaces[:, :, 1, ATM_6M[0], ATM_6M[1]]
    p95 = surfaces[:, :, 2, ATM_6M[0], ATM_6M[1]]

    print(f"  Loaded {len(indices)} sequences")

    # Load ground truth
    print("\nLoading ground truth...")
    gt_data = np.load("data/vol_surface_with_ret.npz")
    gt_surface = gt_data['surface'][:, ATM_6M[0], ATM_6M[1]]

    # Align GT with predictions
    gt_trajectories = []
    context_vols = []

    for idx in indices:
        # Get GT trajectory
        gt_traj = gt_surface[idx:idx+HORIZON]
        if len(gt_traj) == HORIZON:
            gt_trajectories.append(gt_traj)

            # Get context vol
            context = gt_surface[max(0, idx-CONTEXT_LEN):idx]
            context_vols.append(context.mean())

    gt_trajectories = np.array(gt_trajectories)  # (N, 90)
    context_vols = np.array(context_vols)

    # Stratify by regime
    print("\nStratifying by volatility regime...")
    regimes, quartiles = stratify_by_regime(context_vols)

    print(f"  Q1 threshold: {quartiles[0]:.4f}")
    print(f"  Q2 threshold: {quartiles[1]:.4f}")
    print(f"  Q3 threshold: {quartiles[2]:.4f}")

    # Compute coverage by regime
    print("\n" + "="*80)
    print("CI COVERAGE STATISTICS")
    print("="*80)

    coverage_stats = {}
    coverage_by_regime_horizon = {}

    for regime_name, mask in regimes.items():
        n_regime = mask.sum()

        # Coverage across all horizons
        coverage_by_horizon, within_ci = compute_ci_coverage(
            p05[mask], p50[mask], p95[mask], gt_trajectories[mask]
        )

        # H=90 coverage
        h90_coverage = coverage_by_horizon[-1]

        coverage_stats[regime_name] = {
            'n': n_regime,
            'h90_coverage': h90_coverage,
            'coverage_by_horizon': coverage_by_horizon
        }

        coverage_by_regime_horizon[regime_name] = coverage_by_horizon

        print(f"\n{regime_name}:")
        print(f"  n = {n_regime}")
        print(f"  H=90 coverage: {h90_coverage*100:.2f}% (target: 90%)")
        print(f"  H=1 coverage:  {coverage_by_horizon[0]*100:.2f}%")
        print(f"  H=30 coverage: {coverage_by_horizon[29]*100:.2f}%")

    # Overall
    print(f"\nOVERALL:")
    overall_coverage, _ = compute_ci_coverage(p05, p50, p95, gt_trajectories)
    print(f"  H=90 coverage: {overall_coverage[-1]*100:.2f}%")

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_coverage_by_regime(
        coverage_stats,
        OUTPUT_DIR / "ci_coverage_by_regime_h90.png"
    )

    plot_trajectories_by_regime(
        p05, p50, p95, gt_trajectories, context_vols, regimes, quartiles,
        OUTPUT_DIR / "ci_trajectories_by_regime.png"
    )

    plot_coverage_by_horizon(
        coverage_by_regime_horizon,
        OUTPUT_DIR / "ci_coverage_by_horizon.png"
    )

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
