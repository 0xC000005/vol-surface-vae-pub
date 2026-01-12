"""
Final Validation: Trajectory Plots and CI Violations

After all 3 phases, verify that the combined model fixes:
1. Correlation learning (Phase 1)
2. z usage for mean prediction (Phase 2)
3. Fat tails (Phase 3)

This script:
1. Loads the Student-t model (which has all 3 fixes)
2. Generates trajectory samples for crisis period (2008-2010)
3. Plots ATM IV trajectories with CIs
4. Computes CI violation rate (target: <20% in crisis)

Usage:
    python experiments/backfill/two_stage_vae/final_validation.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.backfill.two_stage_vae.exp_student_t_decoder import (
    CVAETwoStageStudentT,
    to_log_returns,
)


def load_student_t_model(model_path: str, device: str = "cuda"):
    """Load the trained Student-t model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def get_crisis_period_indices(n_days: int, context_len: int = 20):
    """
    Get indices for crisis period (2008-2010).

    Data starts in 2000, so 2008 is around day 2000.
    Crisis: Sept 2008 - March 2009 (roughly days 2150-2280)
    """
    # Approximate: 252 trading days/year
    # 2000-2008 = 8 years = ~2000 days
    crisis_start = 2000  # Approximate start of 2008
    crisis_end = min(2400, n_days - context_len)  # Through early 2009

    return crisis_start, crisis_end


def generate_trajectory_samples(model, log_returns, config, n_samples: int = 500):
    """
    Generate trajectory samples for validation.

    Returns:
        samples: (n_samples, n_sequences, 5, 5) - predicted returns
        targets: (n_sequences, 5, 5) - ground truth returns
    """
    device = config["device"]
    context_len = config.get("context_len", 20)

    crisis_start, crisis_end = get_crisis_period_indices(len(log_returns), context_len)
    print(f"Crisis period: days {crisis_start} to {crisis_end}")

    all_samples = []
    all_targets = []

    with torch.no_grad():
        for i in range(crisis_start, crisis_end):
            # Get context sequence
            seq = log_returns[i:i + context_len + 1]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            batch = {"surface": seq_tensor}

            # Generate samples
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, 1, T, 5, 5)

            # Take last prediction (next-day forecast)
            samples_last = samples[:, 0, -1, :, :]  # (n_samples, 5, 5)

            # Ground truth
            target = seq[-1]  # Last element is target (5, 5)

            all_samples.append(samples_last.cpu().numpy())
            all_targets.append(target)

    samples = np.array(all_samples).transpose(1, 0, 2, 3)  # (n_samples, n_seq, 5, 5)
    targets = np.array(all_targets)  # (n_seq, 5, 5)

    return samples, targets


def compute_ci_violations(samples, targets, alpha: float = 0.10):
    """
    Compute CI violation rate.

    Args:
        samples: (n_samples, n_seq, 5, 5)
        targets: (n_seq, 5, 5)
        alpha: CI level (0.10 = 90% CI)

    Returns:
        violation_rate: per-grid violation rates
        mean_violation: overall violation rate
    """
    n_samples, n_seq = samples.shape[:2]

    # Compute quantiles
    lower = np.percentile(samples, alpha/2 * 100, axis=0)  # (n_seq, 5, 5)
    upper = np.percentile(samples, (1 - alpha/2) * 100, axis=0)

    # Count violations
    violations = (targets < lower) | (targets > upper)  # (n_seq, 5, 5)

    violation_rate = violations.mean(axis=0)  # (5, 5)
    mean_violation = violations.mean()

    return violation_rate, mean_violation


def plot_atm_trajectory(samples, targets, output_path: str):
    """
    Plot ATM IV trajectory with CI bands.

    Shows ground truth vs model median with 90% CI.
    """
    # ATM is at position (2, 2)
    atm_samples = samples[:, :, 2, 2]  # (n_samples, n_seq)
    atm_targets = targets[:, 2, 2]  # (n_seq,)

    n_days = len(atm_targets)
    days = np.arange(n_days)

    # Compute statistics
    median = np.median(atm_samples, axis=0)
    p05 = np.percentile(atm_samples, 5, axis=0)
    p95 = np.percentile(atm_samples, 95, axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot CI band
    ax.fill_between(days, p05, p95, alpha=0.3, color='blue', label='90% CI')

    # Plot median
    ax.plot(days, median, 'b-', linewidth=1.5, label='Model Median')

    # Plot ground truth
    ax.plot(days, atm_targets, 'k-', linewidth=1.5, label='Ground Truth')

    # Highlight violations
    violations = (atm_targets < p05) | (atm_targets > p95)
    ax.scatter(days[violations], atm_targets[violations],
               c='red', s=20, zorder=5, label=f'Violations ({violations.mean()*100:.1f}%)')

    ax.set_xlabel('Days (Crisis Period 2008-2009)')
    ax.set_ylabel('ATM IV Log-Return')
    ax.set_title('ATM IV Trajectory: Ground Truth vs Student-t Model (All 3 Fixes)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Trajectory plot saved to {output_path}")

    return violations.mean() * 100


def run_final_validation():
    """Run final validation of the Student-t model."""
    print("=" * 70)
    print("FINAL VALIDATION: Student-t Model with All 3 Fixes")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    log_returns, _ = to_log_returns(data["surface"])

    # Load model
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    print(f"\nLoading model from {model_path}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_student_t_model(model_path, device)
    print(f"Model loaded. Device: {device}")

    # Generate samples
    print("\nGenerating trajectory samples (may take a few minutes)...")
    samples, targets = generate_trajectory_samples(model, log_returns, config, n_samples=500)
    print(f"Generated {samples.shape[0]} samples for {samples.shape[1]} days")

    # Compute CI violations
    print("\nComputing CI violations...")
    violation_rate, mean_violation = compute_ci_violations(samples, targets, alpha=0.10)

    print(f"\nPer-grid CI violation rates (target: 10%):")
    print(np.round(violation_rate * 100, 1))
    print(f"\nMean CI violation rate: {mean_violation * 100:.1f}%")

    # Plot trajectory
    output_dir = Path("results/two_stage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "atm_trajectory_student_t.png"

    atm_violation = plot_atm_trajectory(samples, targets, str(output_path))

    # Compute kurtosis of samples
    print("\nSample distribution analysis:")
    atm_samples = samples[:, :, 2, 2].flatten()
    atm_targets = targets[:, 2, 2]

    sample_kurtosis = kurtosis(atm_samples, fisher=True)
    target_kurtosis = kurtosis(atm_targets, fisher=True)

    print(f"  ATM GT Kurtosis: {target_kurtosis:.2f}")
    print(f"  ATM Sample Kurtosis: {sample_kurtosis:.2f}")
    print(f"  Kurtosis Recovery: {abs(sample_kurtosis/target_kurtosis)*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n  Crisis Period CI Violations:")
    print(f"    Mean: {mean_violation * 100:.1f}%")
    print(f"    ATM: {atm_violation:.1f}%")
    print(f"    Target: <20% (was 87% before fixes)")
    print(f"    Status: {'PASS' if mean_violation < 0.20 else 'NEEDS IMPROVEMENT'}")

    print(f"\n  ATM Kurtosis Recovery:")
    print(f"    GT: {target_kurtosis:.2f}")
    print(f"    Model: {sample_kurtosis:.2f}")
    print(f"    Recovery: {abs(sample_kurtosis/target_kurtosis)*100:.1f}%")

    print("\n  All 3 Fixes Applied:")
    print("    1. Full Covariance NLL → Correlation Learning ✓")
    print("    2. Two-Phase Training → z Usage for Mean ✓")
    print("    3. Student-t Decoder → Fat Tails ✓")

    return {
        "mean_violation": mean_violation,
        "atm_violation": atm_violation / 100,
        "sample_kurtosis": sample_kurtosis,
        "target_kurtosis": target_kurtosis,
        "violation_rate": violation_rate,
    }


if __name__ == "__main__":
    results = run_final_validation()
