"""
Compare Symmetric vs Skewed Student-t VAE: Single-Step Reconstruction

This comparison shows the ACTUAL model performance without fan chart accumulation issues.
Tests single-step reconstruction uncertainty on validation data.

Usage:
    python experiments/backfill/two_stage_vae/compare_symmetric_vs_skew_single_step.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

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
    """Load the skewed Student-t model v2."""
    model_path = "models/backfill/two_stage/student_t_skew_v2/student_t_skew_v2_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentTSkew(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_single_step(model, log_returns, indices, n_samples=100, device="cuda"):
    """
    Evaluate single-step reconstruction performance.

    Returns CI violations, kurtosis recovery, RMSE for each test point.
    """
    context_len = 20
    results = {
        "ci_violations": [],
        "rmse": [],
        "bias": [],
        "sample_std": [],
        "sample_kurtosis": [],
        "gt_values": [],
    }

    with torch.no_grad():
        for idx in indices:
            context = log_returns[idx:idx + context_len]
            gt = log_returns[idx + context_len - 1, 2, 2]  # Last element, ATM

            seq_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
            batch = {"surface": seq_tensor}

            samples = model.sample(batch, n_samples=n_samples)
            atm_samples = samples[:, 0, -1, 2, 2].cpu().numpy()

            # CI violations
            p05 = np.percentile(atm_samples, 5)
            p95 = np.percentile(atm_samples, 95)
            violation = 1 if (gt < p05 or gt > p95) else 0

            results["ci_violations"].append(violation)
            results["rmse"].append((atm_samples.mean() - gt) ** 2)
            results["bias"].append(atm_samples.mean() - gt)
            results["sample_std"].append(atm_samples.std())
            results["sample_kurtosis"].append(kurtosis(atm_samples, fisher=True))
            results["gt_values"].append(gt)

    return results


def main():
    print("=" * 70)
    print("Symmetric vs Skewed Student-t VAE: Single-Step Comparison")
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

    # Test on validation set (sample every 10th point)
    val_start = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)
    test_indices = list(range(val_start, val_end - 20, 10))

    print(f"\nEvaluating on {len(test_indices)} validation points...")

    # Evaluate both models
    sym_results = evaluate_single_step(sym_model, log_returns, test_indices, n_samples=200, device=device)
    skew_results = evaluate_single_step(skew_model, log_returns, test_indices, n_samples=200, device=device)

    # Compute ground truth kurtosis
    gt_values = np.array(sym_results["gt_values"])
    gt_kurtosis = kurtosis(gt_values, fisher=True)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SINGLE-STEP RECONSTRUCTION RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Symmetric':>15} {'Skewed v2':>15} {'Target':>15}")
    print("-" * 70)

    sym_ci = np.mean(sym_results["ci_violations"]) * 100
    skew_ci = np.mean(skew_results["ci_violations"]) * 100
    print(f"{'CI Violations (%)':<25} {sym_ci:>15.1f} {skew_ci:>15.1f} {'10.0':>15}")

    sym_rmse = np.sqrt(np.mean(sym_results["rmse"]))
    skew_rmse = np.sqrt(np.mean(skew_results["rmse"]))
    print(f"{'RMSE':<25} {sym_rmse:>15.6f} {skew_rmse:>15.6f} {'-':>15}")

    sym_bias = np.mean(sym_results["bias"])
    skew_bias = np.mean(skew_results["bias"])
    print(f"{'Mean Bias':<25} {sym_bias:>15.6f} {skew_bias:>15.6f} {'0.0':>15}")

    sym_std = np.mean(sym_results["sample_std"])
    skew_std = np.mean(skew_results["sample_std"])
    print(f"{'Mean Sample Std':<25} {sym_std:>15.6f} {skew_std:>15.6f} {'-':>15}")

    sym_kurt = np.mean(sym_results["sample_kurtosis"])
    skew_kurt = np.mean(skew_results["sample_kurtosis"])
    print(f"{'Mean Sample Kurtosis':<25} {sym_kurt:>15.1f} {skew_kurt:>15.1f} {f'{gt_kurtosis:.1f}':>15}")

    sym_kurt_recovery = (sym_kurt / gt_kurtosis) * 100
    skew_kurt_recovery = (skew_kurt / gt_kurtosis) * 100
    print(f"{'Kurtosis Recovery (%)':<25} {sym_kurt_recovery:>15.1f} {skew_kurt_recovery:>15.1f} {'100.0':>15}")

    # Determine winner
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    sym_score = 0
    skew_score = 0

    # CI closer to 10%
    if abs(sym_ci - 10) < abs(skew_ci - 10):
        sym_score += 1
        print(f"CI Calibration: SYMMETRIC wins ({sym_ci:.1f}% vs {skew_ci:.1f}%)")
    else:
        skew_score += 1
        print(f"CI Calibration: SKEWED wins ({skew_ci:.1f}% vs {sym_ci:.1f}%)")

    # Lower RMSE
    if sym_rmse < skew_rmse:
        sym_score += 1
        print(f"RMSE: SYMMETRIC wins ({sym_rmse:.6f} vs {skew_rmse:.6f})")
    else:
        skew_score += 1
        print(f"RMSE: SKEWED wins ({skew_rmse:.6f} vs {sym_rmse:.6f})")

    # Lower bias
    if abs(sym_bias) < abs(skew_bias):
        sym_score += 1
        print(f"Bias: SYMMETRIC wins ({sym_bias:.6f} vs {skew_bias:.6f})")
    else:
        skew_score += 1
        print(f"Bias: SKEWED wins ({skew_bias:.6f} vs {sym_bias:.6f})")

    # Kurtosis recovery closer to 100%
    if abs(sym_kurt_recovery - 100) < abs(skew_kurt_recovery - 100):
        sym_score += 1
        print(f"Kurtosis Recovery: SYMMETRIC wins ({sym_kurt_recovery:.1f}% vs {skew_kurt_recovery:.1f}%)")
    else:
        skew_score += 1
        print(f"Kurtosis Recovery: SKEWED wins ({skew_kurt_recovery:.1f}% vs {sym_kurt_recovery:.1f}%)")

    print(f"\nFinal Score: SYMMETRIC {sym_score} - {skew_score} SKEWED")
    winner = "SYMMETRIC" if sym_score > skew_score else "SKEWED" if skew_score > sym_score else "TIE"
    print(f"Winner: {winner}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CI violations histogram
    ax = axes[0, 0]
    ax.bar(['Symmetric', 'Skewed v2'], [sym_ci, skew_ci], color=['blue', 'green'], alpha=0.7)
    ax.axhline(y=10, color='red', linestyle='--', label='Target (10%)')
    ax.set_ylabel('CI Violations (%)')
    ax.set_title('90% CI Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample std distribution
    ax = axes[0, 1]
    ax.hist(sym_results["sample_std"], bins=30, alpha=0.5, label='Symmetric', color='blue')
    ax.hist(skew_results["sample_std"], bins=30, alpha=0.5, label='Skewed v2', color='green')
    ax.set_xlabel('Sample Std')
    ax.set_ylabel('Count')
    ax.set_title('Sample Standard Deviation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bias distribution
    ax = axes[1, 0]
    ax.hist(sym_results["bias"], bins=30, alpha=0.5, label='Symmetric', color='blue')
    ax.hist(skew_results["bias"], bins=30, alpha=0.5, label='Skewed v2', color='green')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero bias')
    ax.set_xlabel('Bias (pred - GT)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Bias Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Kurtosis distribution
    ax = axes[1, 1]
    ax.hist(sym_results["sample_kurtosis"], bins=30, alpha=0.5, label='Symmetric', color='blue')
    ax.hist(skew_results["sample_kurtosis"], bins=30, alpha=0.5, label='Skewed v2', color='green')
    ax.axvline(x=gt_kurtosis, color='red', linestyle='--', label=f'GT ({gt_kurtosis:.1f})')
    ax.set_xlabel('Sample Kurtosis')
    ax.set_ylabel('Count')
    ax.set_title('Sample Kurtosis Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Single-Step Comparison: Symmetric vs Skewed v2\nWinner: {winner}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = "models/backfill/two_stage/student_t_skew_v2/single_step_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {save_path}")
    plt.close()

    return sym_results, skew_results


if __name__ == "__main__":
    sym_results, skew_results = main()
