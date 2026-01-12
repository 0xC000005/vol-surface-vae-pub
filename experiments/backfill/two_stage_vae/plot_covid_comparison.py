"""
COVID Comparison: Why the model fails on 11-sigma events

Shows:
1. 2008 Crisis (in training) - model works well
2. COVID 2020 (out of training) - model fails on extreme moves

Demonstrates the limitation of any model on unprecedented events.

Usage:
    python experiments/backfill/two_stage_vae/plot_covid_comparison.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.backfill.two_stage_vae.exp_student_t_decoder import (
    CVAETwoStageStudentT,
    to_log_returns,
)


def load_model(device: str = "cuda"):
    """Load the Student-t model."""
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def generate_trajectories(model, surfaces, log_returns, start_idx,
                          context_len=20, horizon=30, n_samples=50, device="cuda"):
    """Generate conditional trajectories from a starting point."""
    context = log_returns[start_idx:start_idx + context_len]
    gt_returns = log_returns[start_idx + context_len:start_idx + context_len + horizon]

    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]
    gt_iv_levels = surfaces[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    sample_returns = np.zeros((n_samples, horizon))

    with torch.no_grad():
        for h in range(horizon):
            if h == 0:
                full_seq = np.concatenate([context, gt_returns[:1]], axis=0)
            else:
                full_seq = np.concatenate([context, gt_returns[:h+1]], axis=0)

            seq_tensor = torch.tensor(full_seq, dtype=torch.float32).unsqueeze(0).to(device)
            batch = {"surface": seq_tensor}

            samples = model.sample(batch, n_samples=n_samples)
            samples_h = samples[:, 0, -1, 2, 2].cpu().numpy()
            sample_returns[:, h] = samples_h

    sample_iv = np.zeros((n_samples, horizon))
    for s in range(n_samples):
        iv = initial_iv
        for h in range(horizon):
            iv = iv * np.exp(sample_returns[s, h])
            sample_iv[s, h] = iv

    return gt_iv_levels, sample_iv, initial_iv, gt_returns[:, 2, 2]


def plot_comparison(output_path: str):
    """Create comparison plot."""
    print("=" * 70)
    print("COVID Comparison: In-Sample vs Out-of-Sample Performance")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(device)

    periods = {
        "2008 Crisis (IN TRAINING)": 2150,
        "COVID 2020 (OUT OF SAMPLE)": 5050,
    }

    context_len = 20
    horizon = 30
    n_samples = 50

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (period_name, start_idx) in enumerate(periods.items()):
        print(f"\nProcessing {period_name}...")

        gt_iv, sample_iv, initial_iv, gt_returns = generate_trajectories(
            model, surfaces, log_returns, start_idx,
            context_len=context_len, horizon=horizon,
            n_samples=n_samples, device=device
        )

        days = np.arange(1, horizon + 1)

        # Left panel: IV trajectories
        ax1 = axes[row, 0]

        for s in range(n_samples):
            ax1.plot(days, sample_iv[s], color='blue' if row == 0 else 'red',
                    alpha=0.15, linewidth=0.5)

        p05 = np.percentile(sample_iv, 5, axis=0)
        p50 = np.percentile(sample_iv, 50, axis=0)
        p95 = np.percentile(sample_iv, 95, axis=0)

        color = 'blue' if row == 0 else 'red'
        ax1.fill_between(days, p05, p95, color=color, alpha=0.2, label='90% CI')
        ax1.plot(days, p50, color=color, linewidth=2, label='Median')
        ax1.plot(days, gt_iv, 'k-', linewidth=2.5, label='Ground Truth')

        violations = (gt_iv < p05) | (gt_iv > p95)
        if violations.any():
            ax1.scatter(days[violations], gt_iv[violations], c='red', s=50,
                       marker='x', linewidths=2, zorder=5)

        ax1.set_xlabel('Horizon (days)')
        ax1.set_ylabel('ATM Implied Volatility')
        ax1.set_title(f'{period_name}\nCI Violations: {violations.mean()*100:.0f}%')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Right panel: Daily returns histogram
        ax2 = axes[row, 1]

        # Training data distribution
        train_returns = log_returns[:4656, 2, 2]
        ax2.hist(train_returns * 100, bins=50, alpha=0.5, color='gray',
                label=f'Training Data\n(std={train_returns.std()*100:.1f}%)', density=True)

        # Period returns
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        for i, ret in enumerate(gt_returns):
            ax2.axvline(x=ret * 100, color=color, alpha=0.5, linewidth=1)

        # Mark extreme returns
        extreme_mask = np.abs(gt_returns) > 0.15
        if extreme_mask.any():
            for ret in gt_returns[extreme_mask]:
                ax2.axvline(x=ret * 100, color='red', linewidth=2,
                           label=f'Extreme: {ret*100:.0f}%' if ret == gt_returns[extreme_mask][0] else '')

        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Daily Returns Distribution\nMax move: {np.abs(gt_returns).max()*100:.0f}%')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_xlim(-50, 50)

    plt.suptitle('Why Model Fails on COVID: 11-Sigma Events Outside Training Distribution\n'
                 '(Training max: 33% | COVID max: 47%)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    output_dir = Path("results/two_stage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison(str(output_dir / "covid_vs_2008_comparison.png"))
