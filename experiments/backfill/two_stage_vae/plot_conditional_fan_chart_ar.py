"""
Conditional Fan Chart: ATM IV Trajectories with DualPath+AR(1) Model

Shows 4 selective periods:
1. Vol Spike (Sep 2008) - High volatility spike
2. Crisis Peak (Oct 2008) - Lehman aftermath
3. Recovery (Mar 2009) - Market bottom recovery
4. Debt Ceiling (Aug 2011) - Debt ceiling crisis

For each period:
- Ground truth ATM IV trajectory
- 50 sample paths from oracle encoder
- 90% CI bands at each horizon
- Check if GT is within CI

Uses the new CVAETwoStageDualPathAR model which achieves:
- 35% ACF preservation (vs 15% baseline)
- 128% kurtosis recovery
- 13% context contribution

Usage:
    python experiments/backfill/two_stage_vae/plot_conditional_fan_chart_ar.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageDualPathAR


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_model(device: str = "cuda"):
    """Load the DualPath+AR(1) model."""
    model_path = "models/backfill/two_stage/dual_path_ar/dual_path_ar_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageDualPathAR(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def get_period_indices():
    """
    Get indices for 4 selective periods.

    Data starts ~2000, so:
    - 2008 crisis peak: ~day 2150 (Lehman collapse)
    - 2008 vol spike: ~day 2100 (Sept 2008 spike)
    - 2009 recovery: ~day 2280
    - 2011 debt ceiling: ~day 2900
    """
    return {
        "Vol Spike (Sep 2008)": 2100,    # Sept 2008 volatility spike
        "Crisis Peak (Oct 2008)": 2150,  # Lehman aftermath
        "Recovery (Mar 2009)": 2280,     # Market bottom recovery
        "Debt Ceiling (Aug 2011)": 2900, # 2011 debt ceiling crisis
    }


def generate_conditional_trajectories(model, surfaces, log_returns, start_idx,
                                       context_len=20, horizon=30, n_samples=50, device="cuda"):
    """
    Generate conditional trajectories from a starting point.

    Args:
        model: DualPath+AR(1) model
        surfaces: Original IV surfaces (not log-returns)
        log_returns: Log-return surfaces
        start_idx: Starting index in the data
        context_len: Context length for conditioning
        horizon: Number of days to forecast
        n_samples: Number of sample paths
        device: Device to use

    Returns:
        gt_iv: Ground truth IV trajectory (horizon,)
        sample_iv: Sample IV trajectories (n_samples, horizon)
        initial_iv: Initial IV level for reference
    """
    # Get context and ground truth
    context = log_returns[start_idx:start_idx + context_len]
    gt_returns = log_returns[start_idx + context_len:start_idx + context_len + horizon]

    # Get initial IV and ground truth IV trajectory
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]  # Last context day ATM IV
    gt_iv_levels = surfaces[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    # Prepare context tensor
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    # Generate samples for each horizon step
    sample_returns = np.zeros((n_samples, horizon))

    with torch.no_grad():
        for h in range(horizon):
            # For oracle sampling: encoder sees targets up to h-1,
            # decoder predicts target at position h-1 (which is what we want).
            #
            # Key insight: model at position t predicts input[t+1] (next-step).
            # So samples[-2] predicts the LAST INPUT element.
            # samples[-1] predicts BEYOND the input (wrong for reconstruction).
            if h == 0:
                full_seq = context.copy()  # Just context, no gt_returns
            else:
                full_seq = np.concatenate([context, gt_returns[:h]], axis=0)

            seq_tensor = torch.tensor(full_seq, dtype=torch.float32).unsqueeze(0).to(device)
            batch = {"surface": seq_tensor}

            # Generate samples - each sample is a different z draw
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, 1, T, 5, 5)

            # Get prediction for LAST INPUT element (oracle reconstruction)
            # samples[-2] predicts input[-1] = the target we want
            # samples[-1] predicts input[T] which doesn't exist (extrapolation)
            samples_h = samples[:, 0, -2, 2, 2].cpu().numpy()  # (n_samples,)
            sample_returns[:, h] = samples_h

    # Convert sample returns to IV levels
    # IV(t) = IV(t-1) * exp(return(t))
    sample_iv = np.zeros((n_samples, horizon))
    for s in range(n_samples):
        iv = initial_iv
        for h in range(horizon):
            iv = iv * np.exp(sample_returns[s, h])
            sample_iv[s, h] = iv

    return gt_iv_levels, sample_iv, initial_iv


def plot_fan_chart(output_path: str):
    """Create 4-panel fan chart visualization."""
    print("=" * 70)
    print("Conditional Fan Chart: ATM IV Trajectories (DualPath+AR(1))")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # Original IV surfaces
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("\nLoading DualPath+AR(1) model...")
    model, config = load_model(device)

    # Print AR(1) coefficient
    phi = model.get_ar_phi().item()
    print(f"Model loaded. Learned AR(1) phi = {phi:.4f}")

    # Get period indices
    periods = get_period_indices()

    # Check data bounds
    max_idx = len(log_returns) - 50  # Need room for horizon
    for name, idx in periods.items():
        if idx > max_idx:
            print(f"Warning: {name} index {idx} exceeds data bounds, adjusting...")
            periods[name] = min(idx, max_idx - 100)

    context_len = 20
    horizon = 30
    n_samples = 50

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {
        "Vol Spike (Sep 2008)": "#e74c3c",
        "Crisis Peak (Oct 2008)": "#c0392b",
        "Recovery (Mar 2009)": "#27ae60",
        "Debt Ceiling (Aug 2011)": "#9b59b6",
    }

    all_violations = []

    for idx, (period_name, start_idx) in enumerate(periods.items()):
        print(f"\nGenerating trajectories for {period_name}...")

        ax = axes[idx]
        color = colors[period_name]

        try:
            gt_iv, sample_iv, initial_iv = generate_conditional_trajectories(
                model, surfaces, log_returns, start_idx,
                context_len=context_len, horizon=horizon,
                n_samples=n_samples, device=device
            )
        except Exception as e:
            print(f"  Error: {e}")
            ax.text(0.5, 0.5, f"Data not available\nfor {period_name}",
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(period_name)
            continue

        days = np.arange(1, horizon + 1)

        # Plot sample paths (spaghetti)
        for s in range(n_samples):
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
        ax.axhline(y=initial_iv, color='gray', linestyle='--', alpha=0.5, label=f'Initial IV: {initial_iv:.3f}')

        # Check CI violations
        violations = (gt_iv < p05) | (gt_iv > p95)
        violation_days = days[violations]
        violation_ivs = gt_iv[violations]
        all_violations.append(violations.mean())

        if len(violation_days) > 0:
            ax.scatter(violation_days, violation_ivs, c='red', s=40, zorder=5,
                      marker='x', linewidths=2, label=f'Violations ({violations.mean()*100:.0f}%)')

        # Formatting
        ax.set_xlabel('Horizon (days)')
        ax.set_ylabel('ATM Implied Volatility')
        ax.set_title(f'{period_name}\n(CI violations: {violations.mean()*100:.1f}%)', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        ci_width = (p95 - p05).mean()
        ax.text(0.02, 0.98, f'Avg CI width: {ci_width:.4f}\nMedian IV: {p50.mean():.3f}',
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Conditional Fan Charts: ATM IV 30-Day Trajectories\n'
                 f'(50 Oracle Samples per Horizon, DualPath+AR(1) Model, $\\phi$={phi:.3f})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to {output_path}")
    print(f"Average CI violations across periods: {np.mean(all_violations)*100:.1f}%")

    return


if __name__ == "__main__":
    output_dir = Path("results/two_stage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fan_chart(str(output_dir / "conditional_fan_chart_iv_ar.png"))
