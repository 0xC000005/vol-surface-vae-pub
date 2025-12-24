"""
Experiment 4: Quantile Growth Rate Analysis

Test whether the shared quantile decoder forces proportional growth across quantiles.

Hypothesis:
    Shared Conv2D decoder outputting 3 channels (p05, p50, p95) forces
    proportional growth, affecting band width dynamics but NOT p50 marginal.

Method:
    1. Extract normalized quantile trajectories (normalized to context endpoint)
    2. Compute marginal quantiles by pooling across all sequences
    3. Compute growth rates: (day_90 / day_1 - 1) for each quantile
    4. Check if p05, p50, p95 growth rates are similar (within 20%)

Decision Criteria:
    - If growth rates are similar (within 20%): ✅ Confirms proportional growth
    - Note: This affects band width, NOT p50 marginal spread
    - If we want to fix band width: Use independent quantile decoder heads
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

# Paths
ORACLE_PRED_PATH = "results/context60_baseline/predictions/teacher_forcing/oracle/vae_tf_insample_h90.npz"
PRIOR_PRED_PATH = "results/context60_baseline/predictions/teacher_forcing/prior/vae_tf_insample_h90.npz"
GT_DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_DIR = Path("results/context60_baseline/analysis/preliminary_experiments/quantile_growth_analysis")

# Grid point (ATM 6-month)
ATM_6M = (2, 2)

# Horizon
HORIZON = 90

# Context length
CONTEXT_LEN = 60


# ============================================================================
# Data Loading
# ============================================================================

def load_predictions_and_data():
    """Load oracle/prior predictions and ground truth data."""
    print("Loading predictions and ground truth...")

    # Load oracle predictions
    oracle_data = np.load(ORACLE_PRED_PATH)
    oracle_surfaces = oracle_data['surfaces']  # (N, H, 3, 5, 5)
    oracle_indices = oracle_data['indices']  # (N,)

    # Load prior predictions
    prior_data = np.load(PRIOR_PRED_PATH)
    prior_surfaces = prior_data['surfaces']  # (N, H, 3, 5, 5)

    # Load ground truth
    gt_data = np.load(GT_DATA_PATH)
    gt_surface = gt_data['surface']  # (T, 5, 5)

    print(f"  Loaded {len(oracle_indices)} prediction sequences")
    print(f"  Ground truth: {len(gt_surface)} days")

    return {
        'oracle_surfaces': oracle_surfaces,
        'prior_surfaces': prior_surfaces,
        'indices': oracle_indices,
        'gt_surface': gt_surface
    }


# ============================================================================
# Quantile Extraction and Normalization
# ============================================================================

def extract_normalized_quantiles(surfaces, gt_surface, indices, context_len=60):
    """Extract and normalize quantiles to context endpoint.

    Args:
        surfaces: (N, H, 3, 5, 5) model predictions
        gt_surface: (T, 5, 5) ground truth
        indices: (N,) sequence indices
        context_len: Context length

    Returns:
        dict with normalized trajectories for p05, p50, p95
    """
    print("Extracting and normalizing quantiles...")

    n_sequences = len(indices)
    grid_row, grid_col = ATM_6M

    # Initialize arrays for normalized trajectories
    normalized_p05 = np.zeros((n_sequences, HORIZON))
    normalized_p50 = np.zeros((n_sequences, HORIZON))
    normalized_p95 = np.zeros((n_sequences, HORIZON))

    for i, idx in enumerate(indices):
        # Get context endpoint (anchor value)
        context_start = max(0, idx - context_len)
        context_surfaces = gt_surface[context_start:idx]  # (C, 5, 5)
        anchor_value = context_surfaces[-1, grid_row, grid_col]

        # Extract quantiles from model predictions
        p05_traj = surfaces[i, :, 0, grid_row, grid_col]  # (H,)
        p50_traj = surfaces[i, :, 1, grid_row, grid_col]  # (H,)
        p95_traj = surfaces[i, :, 2, grid_row, grid_col]  # (H,)

        # Normalize to context endpoint
        normalized_p05[i] = p05_traj - anchor_value
        normalized_p50[i] = p50_traj - anchor_value
        normalized_p95[i] = p95_traj - anchor_value

    print(f"  ✓ Extracted {n_sequences} quantile trajectories")

    return {
        'p05': normalized_p05,
        'p50': normalized_p50,
        'p95': normalized_p95
    }


# ============================================================================
# Marginal Quantiles and Growth Rates
# ============================================================================

def compute_marginal_quantiles_and_growth(normalized_quantiles):
    """Compute marginal quantiles by pooling across sequences.

    Args:
        normalized_quantiles: dict with p05, p50, p95 arrays (N, H)

    Returns:
        dict with marginal statistics and growth rates
    """
    print("Computing marginal quantiles and growth rates...")

    # Extract quantiles
    p05_trajectories = normalized_quantiles['p05']  # (N, H)
    p50_trajectories = normalized_quantiles['p50']  # (N, H)
    p95_trajectories = normalized_quantiles['p95']  # (N, H)

    # Compute marginal medians (median across sequences for each day)
    marginal_p05 = np.median(p05_trajectories, axis=0)  # (H,)
    marginal_p50 = np.median(p50_trajectories, axis=0)  # (H,)
    marginal_p95 = np.median(p95_trajectories, axis=0)  # (H,)

    # Compute spreads (for band width analysis)
    # Spread = distance from p50 to p05/p95
    spread_lower = np.abs(marginal_p05)  # Distance from anchor to p05
    spread_upper = np.abs(marginal_p95)  # Distance from anchor to p95

    # Compute growth rates: (day_90 / day_1 - 1)
    # Use absolute values to compute meaningful growth rates
    growth_p05 = (np.abs(marginal_p05[-1]) / (np.abs(marginal_p05[0]) + 1e-8)) - 1
    growth_p50 = (np.abs(marginal_p50[-1]) / (np.abs(marginal_p50[0]) + 1e-8)) - 1
    growth_p95 = (np.abs(marginal_p95[-1]) / (np.abs(marginal_p95[0]) + 1e-8)) - 1

    # Check if growth rates are similar (within 20%)
    growth_rates = np.array([growth_p05, growth_p50, growth_p95])
    growth_std = growth_rates.std()
    growth_mean = growth_rates.mean()
    coefficient_of_variation = growth_std / (np.abs(growth_mean) + 1e-8)

    print(f"  Growth rates:")
    print(f"    p05: {growth_p05:+.1%}")
    print(f"    p50: {growth_p50:+.1%}")
    print(f"    p95: {growth_p95:+.1%}")
    print(f"  Coefficient of variation: {coefficient_of_variation:.3f}")

    return {
        'marginal_p05': marginal_p05,
        'marginal_p50': marginal_p50,
        'marginal_p95': marginal_p95,
        'spread_lower': spread_lower,
        'spread_upper': spread_upper,
        'growth_p05': growth_p05,
        'growth_p50': growth_p50,
        'growth_p95': growth_p95,
        'coefficient_of_variation': coefficient_of_variation
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_quantile_trajectories(oracle_stats, prior_stats, output_dir):
    """Plot marginal quantile trajectories for oracle and prior.

    Args:
        oracle_stats: Marginal statistics from oracle sampling
        prior_stats: Marginal statistics from prior sampling
        output_dir: Output directory
    """
    print("Plotting quantile trajectories...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    days = np.arange(1, HORIZON + 1)

    # Oracle plot
    ax = axes[0]
    ax.plot(days, oracle_stats['marginal_p05'], 'b-', linewidth=2, label='p05')
    ax.plot(days, oracle_stats['marginal_p50'], 'g-', linewidth=2, label='p50')
    ax.plot(days, oracle_stats['marginal_p95'], 'r-', linewidth=2, label='p95')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel("Horizon (days)", fontsize=12)
    ax.set_ylabel("Normalized IV Change", fontsize=12)
    ax.set_title(f"Oracle: Marginal Quantile Trajectories\nGrowth: p05={oracle_stats['growth_p05']:+.1%}, p50={oracle_stats['growth_p50']:+.1%}, p95={oracle_stats['growth_p95']:+.1%}",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.2)

    # Prior plot
    ax = axes[1]
    ax.plot(days, prior_stats['marginal_p05'], 'b-', linewidth=2, label='p05')
    ax.plot(days, prior_stats['marginal_p50'], 'g-', linewidth=2, label='p50')
    ax.plot(days, prior_stats['marginal_p95'], 'r-', linewidth=2, label='p95')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel("Horizon (days)", fontsize=12)
    ax.set_ylabel("Normalized IV Change", fontsize=12)
    ax.set_title(f"Prior: Marginal Quantile Trajectories\nGrowth: p05={prior_stats['growth_p05']:+.1%}, p50={prior_stats['growth_p50']:+.1%}, p95={prior_stats['growth_p95']:+.1%}",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path = output_dir / "quantile_trajectories_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_growth_rate_comparison(oracle_stats, prior_stats, output_dir):
    """Plot growth rate comparison bar chart.

    Args:
        oracle_stats: Oracle statistics
        prior_stats: Prior statistics
        output_dir: Output directory
    """
    print("Plotting growth rate comparison...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    quantiles = ['p05', 'p50', 'p95']
    oracle_growth = [oracle_stats['growth_p05'], oracle_stats['growth_p50'], oracle_stats['growth_p95']]
    prior_growth = [prior_stats['growth_p05'], prior_stats['growth_p50'], prior_stats['growth_p95']]

    x = np.arange(len(quantiles))
    width = 0.35

    ax.bar(x - width/2, oracle_growth, width, label='Oracle', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, prior_growth, width, label='Prior', color='coral', alpha=0.8)

    ax.set_xlabel("Quantile", fontsize=12)
    ax.set_ylabel("Growth Rate (day 90 / day 1 - 1)", fontsize=12)
    ax.set_title("Quantile Growth Rates: Oracle vs Prior", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Add value labels on bars
    for i, (o, p) in enumerate(zip(oracle_growth, prior_growth)):
        ax.text(i - width/2, o, f'{o:+.1%}', ha='center', va='bottom' if o > 0 else 'top', fontsize=9)
        ax.text(i + width/2, p, f'{p:+.1%}', ha='center', va='bottom' if p > 0 else 'top', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "growth_rate_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# Diagnostic Report
# ============================================================================

def generate_diagnostic_report(oracle_stats, prior_stats, output_dir):
    """Generate diagnostic report with interpretation.

    Args:
        oracle_stats: Oracle statistics
        prior_stats: Prior statistics
        output_dir: Output directory
    """
    print("Generating diagnostic report...")

    report = []
    report.append("=" * 80)
    report.append("EXPERIMENT 4: QUANTILE GROWTH RATE ANALYSIS")
    report.append("=" * 80)
    report.append("")
    report.append("HYPOTHESIS:")
    report.append("  Shared quantile decoder (single Conv2D → 3 channels) forces")
    report.append("  proportional growth across p05, p50, p95.")
    report.append("")
    report.append("METHOD:")
    report.append("  1. Extract normalized quantile trajectories (normalized to context endpoint)")
    report.append("  2. Compute marginal quantiles by pooling across sequences")
    report.append("  3. Compute growth rates: (day_90 / day_1 - 1) for each quantile")
    report.append("  4. Check if p05, p50, p95 growth rates are similar (within 20%)")
    report.append("")
    report.append("RESULTS:")
    report.append("")
    report.append("ORACLE SAMPLING:")
    report.append(f"  p05 growth: {oracle_stats['growth_p05']:+.1%}")
    report.append(f"  p50 growth: {oracle_stats['growth_p50']:+.1%}")
    report.append(f"  p95 growth: {oracle_stats['growth_p95']:+.1%}")
    report.append(f"  Coefficient of variation: {oracle_stats['coefficient_of_variation']:.3f}")
    report.append("")
    report.append("PRIOR SAMPLING:")
    report.append(f"  p05 growth: {prior_stats['growth_p05']:+.1%}")
    report.append(f"  p50 growth: {prior_stats['growth_p50']:+.1%}")
    report.append(f"  p95 growth: {prior_stats['growth_p95']:+.1%}")
    report.append(f"  Coefficient of variation: {prior_stats['coefficient_of_variation']:.3f}")
    report.append("")

    # Decision logic
    oracle_cv = oracle_stats['coefficient_of_variation']
    prior_cv = prior_stats['coefficient_of_variation']

    report.append("=" * 80)
    report.append("VERDICT:")
    report.append("")

    if oracle_cv < 0.2 and prior_cv < 0.2:
        report.append("  ✅ HYPOTHESIS CONFIRMED")
        report.append(f"  Oracle CV={oracle_cv:.3f} < 0.2, Prior CV={prior_cv:.3f} < 0.2")
        report.append("  Growth rates are similar across quantiles (proportional growth)")
        report.append("")
        report.append("IMPLICATION:")
        report.append("  Shared quantile decoder forces proportional growth.")
        report.append("  This affects BAND WIDTH, not p50 marginal spread.")
        report.append("")
        report.append("RECOMMENDATION:")
        report.append("  If band width dynamics are problematic, consider:")
        report.append("  - Independent decoder heads for each quantile")
        report.append("  - Horizon-dependent quantile loss weights")
    else:
        report.append("  ❌ HYPOTHESIS REJECTED")
        report.append(f"  Oracle CV={oracle_cv:.3f}, Prior CV={prior_cv:.3f}")
        report.append("  Growth rates differ across quantiles (non-proportional)")
        report.append("")
        report.append("IMPLICATION:")
        report.append("  Decoder allows different growth rates for different quantiles.")
        report.append("  Band width dynamics are NOT constrained by shared decoder.")

    report.append("")
    report.append("IMPORTANT NOTE:")
    report.append("  This experiment tests BAND WIDTH dynamics, NOT p50 marginal spread.")
    report.append("  The p50 marginal spread issue (day-1 over-dispersion) is caused by")
    report.append("  epistemic uncertainty, not proportional quantile growth.")
    report.append("")
    report.append("=" * 80)

    # Write report
    output_path = output_dir / "quantile_growth_diagnostics.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"  ✓ Saved: {output_path}")

    # Print to console
    print("")
    print('\n'.join(report))


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("")
    print("=" * 80)
    print("EXPERIMENT 4: QUANTILE GROWTH RATE ANALYSIS")
    print("=" * 80)
    print("")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_predictions_and_data()

    # Extract and normalize quantiles for oracle
    oracle_quantiles = extract_normalized_quantiles(
        data['oracle_surfaces'],
        data['gt_surface'],
        data['indices'],
        CONTEXT_LEN
    )

    # Extract and normalize quantiles for prior
    prior_quantiles = extract_normalized_quantiles(
        data['prior_surfaces'],
        data['gt_surface'],
        data['indices'],
        CONTEXT_LEN
    )

    # Compute marginal statistics and growth rates
    oracle_stats = compute_marginal_quantiles_and_growth(oracle_quantiles)
    prior_stats = compute_marginal_quantiles_and_growth(prior_quantiles)

    print("")
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print("")

    # Generate plots
    plot_quantile_trajectories(oracle_stats, prior_stats, OUTPUT_DIR)
    plot_growth_rate_comparison(oracle_stats, prior_stats, OUTPUT_DIR)

    # Generate diagnostic report
    generate_diagnostic_report(oracle_stats, prior_stats, OUTPUT_DIR)

    print("")
    print("=" * 80)
    print("✓ EXPERIMENT 4 COMPLETE")
    print("=" * 80)
    print("")


if __name__ == "__main__":
    main()
