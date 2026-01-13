"""
Compare Symmetric vs Skewed Student-t VAE: Conditional Fan Charts

Side-by-side comparison of ATM IV 30-day trajectories with 50 oracle samples.

Shows 4 market periods, with symmetric model (left) and skewed model (right).

Usage:
    python experiments/backfill/two_stage_vae/compare_symmetric_vs_skew_fan_chart.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP, CVAETwoStageStudentTSkew


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_symmetric_model(device: str = "cuda"):
    """Load the symmetric Student-t model."""
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentTMLP(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def load_skewed_model(device: str = "cuda"):
    """Load the skewed Student-t model (v2 with variance regularization)."""
    model_path = "models/backfill/two_stage/student_t_skew_v2/student_t_skew_v2_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentTSkew(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def get_period_indices():
    """Get indices for 4 selective periods."""
    return {
        "Vol Spike (Sep 2008)": 2100,
        "Crisis Peak (Oct 2008)": 2150,
        "Recovery (Mar 2009)": 2280,
        "Debt Ceiling (Aug 2011)": 2900,
    }


def generate_conditional_trajectories(model, surfaces, log_returns, start_idx,
                                       context_len=20, horizon=30, n_samples=50, device="cuda"):
    """
    Generate proper conditional fan chart trajectories.

    Approach (context-only, no GT leakage):
    - Feed context ONLY to the model
    - Generate n_samples by sampling different z values
    - Each sample gives a full 5x5 surface reconstruction
    - Build trajectories by accumulating sampled returns

    This shows P(trajectory | context) without any future information leakage.
    Variance will be constant across horizon (since we're not feeding back),
    but each sample path is independent.

    Returns:
        gt_iv: Ground truth IV trajectory (horizon,)
        sample_iv: Sample IV trajectories (n_samples, horizon)
        initial_iv: Initial IV level for reference
    """
    # Get context only - NO GT!
    context = log_returns[start_idx:start_idx + context_len]  # (context_len, 5, 5)

    # Get initial IV and ground truth IV trajectory for comparison
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]  # Last context day ATM IV
    gt_iv_levels = surfaces[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    # Feed context ONCE, generate many samples
    seq_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    batch = {"surface": seq_tensor}

    with torch.no_grad():
        # Generate n_samples reconstructions of the context
        # Each sample has different z, giving different reconstruction
        samples = model.sample(batch, n_samples=n_samples)
        # samples: (n_samples, 1, context_len, 5, 5)

        # Get the reconstruction of the LAST context element (ATM point)
        # This represents the model's uncertainty about the current state
        last_reconstruction = samples[:, 0, -1, 2, 2].cpu().numpy()  # (n_samples,)

    # For each sample, generate a trajectory by:
    # 1. Using the sampled reconstruction as the "return" for step 0
    # 2. For subsequent steps, sample from the same distribution (iid)
    #    This is a simplification - true AR would feed back predictions
    #
    # Alternative interpretation: Each trajectory is a random walk
    # starting from the sampled reconstruction, with steps drawn from
    # the same conditional distribution.

    sample_iv = np.zeros((n_samples, horizon))

    # Method: Sample horizon independent returns from the conditional distribution
    # and accumulate them into trajectories
    with torch.no_grad():
        for h in range(horizon):
            # For each horizon step, generate fresh samples
            samples_h = model.sample(batch, n_samples=n_samples)
            returns_h = samples_h[:, 0, -1, 2, 2].cpu().numpy()  # (n_samples,)

            # Accumulate into IV levels
            if h == 0:
                sample_iv[:, h] = initial_iv * np.exp(returns_h)
            else:
                sample_iv[:, h] = sample_iv[:, h-1] * np.exp(returns_h)

    return gt_iv_levels, sample_iv, initial_iv


def plot_single_panel(ax, gt_iv, sample_iv, initial_iv, title, color, show_ylabel=True):
    """Plot a single fan chart panel."""
    horizon = len(gt_iv)
    days = np.arange(1, horizon + 1)

    # Plot sample paths (spaghetti)
    for s in range(sample_iv.shape[0]):
        ax.plot(days, sample_iv[s], color=color, alpha=0.15, linewidth=0.5)

    # Compute and plot CI bands
    p05 = np.percentile(sample_iv, 5, axis=0)
    p50 = np.percentile(sample_iv, 50, axis=0)
    p95 = np.percentile(sample_iv, 95, axis=0)

    ax.fill_between(days, p05, p95, color=color, alpha=0.2, label='90% CI')
    ax.plot(days, p50, color=color, linewidth=2, label='Median')

    # Plot ground truth
    ax.plot(days, gt_iv, 'k-', linewidth=2.5, label='Ground Truth')

    # Mark initial IV
    ax.axhline(y=initial_iv, color='gray', linestyle='--', alpha=0.5)

    # Check CI violations
    violations = (gt_iv < p05) | (gt_iv > p95)
    violation_days = days[violations]
    violation_ivs = gt_iv[violations]

    if len(violation_days) > 0:
        ax.scatter(violation_days, violation_ivs, c='red', s=40, zorder=5,
                  marker='x', linewidths=2)

    # Formatting
    ax.set_xlabel('Horizon (days)')
    if show_ylabel:
        ax.set_ylabel('ATM IV')
    violation_pct = violations.mean() * 100
    ax.set_title(f'{title}\n(Violations: {violation_pct:.0f}%)', fontsize=10)
    ax.grid(True, alpha=0.3)

    return violation_pct


def main():
    print("=" * 70)
    print("Symmetric vs Skewed Student-t VAE: Fan Chart Comparison")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("\nLoading models...")
    sym_model, _ = load_symmetric_model(device)
    skew_model, _ = load_skewed_model(device)
    print("Models loaded.")

    # Get period indices
    periods = get_period_indices()

    context_len = 20
    horizon = 30
    n_samples = 50

    # Create figure: 4 rows (periods) x 2 columns (symmetric, skewed)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    colors = {
        "Vol Spike (Sep 2008)": "#e74c3c",
        "Crisis Peak (Oct 2008)": "#c0392b",
        "Recovery (Mar 2009)": "#27ae60",
        "Debt Ceiling (Aug 2011)": "#9b59b6",
    }

    sym_violations = []
    skew_violations = []

    for row, (period_name, start_idx) in enumerate(periods.items()):
        print(f"\nGenerating for {period_name}...")
        color = colors[period_name]

        try:
            # Generate for both models
            gt_iv_sym, sample_iv_sym, initial_iv = generate_conditional_trajectories(
                sym_model, surfaces, log_returns, start_idx,
                context_len=context_len, horizon=horizon,
                n_samples=n_samples, device=device
            )

            gt_iv_skew, sample_iv_skew, _ = generate_conditional_trajectories(
                skew_model, surfaces, log_returns, start_idx,
                context_len=context_len, horizon=horizon,
                n_samples=n_samples, device=device
            )

            # Plot symmetric (left column)
            viol_sym = plot_single_panel(
                axes[row, 0], gt_iv_sym, sample_iv_sym, initial_iv,
                f"SYMMETRIC - {period_name}", color, show_ylabel=True
            )
            sym_violations.append(viol_sym)

            # Plot skewed (right column)
            viol_skew = plot_single_panel(
                axes[row, 1], gt_iv_skew, sample_iv_skew, initial_iv,
                f"SKEWED - {period_name}", color, show_ylabel=False
            )
            skew_violations.append(viol_skew)

            print(f"  Symmetric violations: {viol_sym:.0f}%")
            print(f"  Skewed violations:    {viol_skew:.0f}%")

        except Exception as e:
            print(f"  Error: {e}")
            for col in range(2):
                axes[row, col].text(0.5, 0.5, f"Data not available\n{e}",
                                   transform=axes[row, col].transAxes, ha='center', va='center')

    # Add column headers
    axes[0, 0].annotate('SYMMETRIC Student-t', xy=(0.5, 1.15), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
    axes[0, 1].annotate('SKEWED Student-t', xy=(0.5, 1.15), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_sym = np.mean(sym_violations)
    avg_skew = np.mean(skew_violations)
    print(f"Average CI violations:")
    print(f"  Symmetric: {avg_sym:.1f}%")
    print(f"  Skewed:    {avg_skew:.1f}%")
    print(f"  Winner:    {'SKEWED' if avg_skew < avg_sym else 'SYMMETRIC'}")

    # Add summary to figure
    fig.suptitle(
        f"Symmetric vs Skewed Student-t VAE | ATM IV 30-day Trajectories | 50 Samples\n"
        f"Avg Violations: Symmetric={avg_sym:.1f}%, Skewed={avg_skew:.1f}%",
        fontsize=14, fontweight='bold', y=0.99
    )

    # Save
    output_path = "models/backfill/two_stage/student_t_skew_v2/symmetric_vs_skew_fan_chart.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    plt.close()

    return sym_violations, skew_violations


if __name__ == "__main__":
    sym_viol, skew_viol = main()
