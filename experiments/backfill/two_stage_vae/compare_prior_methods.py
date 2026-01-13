"""
Compare Prior Methods for Student-t VAE

Compares CI calibration across three sampling modes:
1. Oracle: z from posterior (encoder sees target) - upper bound
2. Prior N(0,1): z sampled from standard normal - baseline
3. Prior Predictor: z from trained LatentPredictor - our solution

Usage:
    python experiments/backfill/two_stage_vae/compare_prior_methods.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictor


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, batch_size, shuffle=False):
    """Create DataLoader from log-returns."""
    N = len(log_returns)
    sequences = []

    for i in range(N - context_len):
        seq = log_returns[i:i + context_len + 1]
        sequences.append(seq)

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_vae_and_predictor(device="cuda"):
    """Load the frozen VAE and trained predictor."""
    # Load VAE
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load predictor
    pred_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    pred_ckpt = torch.load(pred_path, map_location=device, weights_only=False)

    predictor = LatentPredictor(config)
    predictor.load_state_dict(pred_ckpt["predictor_state_dict"])
    predictor = predictor.to(device)
    predictor.eval()

    return vae, predictor, config


def generate_samples_oracle(vae, batch, n_samples=100):
    """
    Oracle mode: z sampled from posterior (encoder sees target).
    This is the upper bound - uses information that won't be available at deployment.
    """
    return vae.sample(batch, n_samples=n_samples)


def generate_samples_prior_n01(vae, batch, n_samples=100):
    """
    Prior N(0,1) mode: z sampled from standard normal.
    This is the baseline - no information about target.
    """
    surface = batch["surface"]
    B, T = surface.shape[:2]
    device = surface.device
    latent_dim = vae.config["latent_dim"]

    ctx_emb = vae.ctx_encoder({"surface": surface})

    samples = []
    for _ in range(n_samples):
        # Sample z from N(0,1) - no encoder information
        z = torch.randn(B, T, latent_dim, device=device)
        _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)
        samples.append(sample)

    return torch.stack(samples, dim=0)


def generate_samples_prior_predictor(vae, predictor, batch, n_samples=100, context_len=20):
    """
    Prior Predictor mode: z from trained LatentPredictor for future positions.
    Context positions use encoder mean, future positions use predictor.
    """
    surface = batch["surface"]
    B, T = surface.shape[:2]
    device = surface.device
    latent_dim = vae.config["latent_dim"]
    horizon = T - context_len

    ctx_emb = vae.ctx_encoder({"surface": surface})

    # Get z_mean for context from encoder
    with torch.no_grad():
        z_ctx_mean, z_ctx_logvar, _ = vae.main_encoder({"surface": surface[:, :context_len]})

    # Get predicted z distribution for future from predictor
    with torch.no_grad():
        z_future_mean, z_future_logvar = predictor(surface[:, :context_len], horizon=horizon)

    samples = []
    for _ in range(n_samples):
        # Context: sample from encoder posterior
        z_ctx = z_ctx_mean + torch.exp(0.5 * z_ctx_logvar) * torch.randn_like(z_ctx_mean)

        # Future: sample from predictor distribution
        z_future = z_future_mean + torch.exp(0.5 * z_future_logvar) * torch.randn_like(z_future_mean)

        # Concatenate
        z = torch.cat([z_ctx, z_future], dim=1)  # (B, T, latent_dim)

        _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)
        samples.append(sample)

    return torch.stack(samples, dim=0)


def compute_ci_violations(samples, gt, ci_level=0.90):
    """
    Compute CI violations.

    Args:
        samples: (n_samples, B, T, H, W)
        gt: (B, T, H, W)
        ci_level: Confidence level (default 90%)

    Returns:
        violation_rate: Fraction of GT points outside CI
        ci_width: Mean CI width
    """
    alpha = 1 - ci_level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Compute quantiles
    lower = torch.quantile(samples, lower_q, dim=0)  # (B, T, H, W)
    upper = torch.quantile(samples, upper_q, dim=0)

    # Check violations
    below = (gt < lower).float()
    above = (gt > upper).float()
    violations = below + above

    violation_rate = violations.mean().item()
    ci_width = (upper - lower).mean().item()

    return violation_rate, ci_width


def compute_kurtosis_recovery(samples, gt):
    """
    Compute kurtosis recovery.

    Args:
        samples: (n_samples, B, T, H, W)
        gt: (B, T, H, W)

    Returns:
        recovery: Model kurtosis / GT kurtosis * 100
    """
    # Pool samples for ATM point
    sample_atm = samples[:, :, -1, 2, 2].flatten().cpu().numpy()
    gt_atm = gt[:, -1, 2, 2].flatten().cpu().numpy()

    sample_kurt = kurtosis(sample_atm, fisher=True)
    gt_kurt = kurtosis(gt_atm, fisher=True)

    recovery = (sample_kurt / (gt_kurt + 1e-8)) * 100

    return recovery, sample_kurt, gt_kurt


def evaluate_mode(vae, predictor, val_loader, mode, context_len=20, n_samples=100, device="cuda"):
    """
    Evaluate a single sampling mode.

    Args:
        vae: Student-t VAE model
        predictor: Trained LatentPredictor (only used for "predictor" mode)
        val_loader: Validation data
        mode: "oracle", "prior_n01", or "prior_predictor"
        context_len: Context length
        n_samples: Number of samples for CI estimation

    Returns:
        results: dict with metrics
    """
    all_violations = []
    all_ci_widths = []
    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            B, T = surface.shape[:2]
            batch = {"surface": surface}

            # Generate samples based on mode
            if mode == "oracle":
                samples = generate_samples_oracle(vae, batch, n_samples)
            elif mode == "prior_n01":
                samples = generate_samples_prior_n01(vae, batch, n_samples)
            elif mode == "prior_predictor":
                samples = generate_samples_prior_predictor(
                    vae, predictor, batch, n_samples, context_len
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Only evaluate last position (horizon position)
            samples_last = samples[:, :, -1]  # (n_samples, B, 5, 5)
            gt_last = surface[:, -1]  # (B, 5, 5)

            # Compute CI violations
            viol, width = compute_ci_violations(samples_last.unsqueeze(2), gt_last.unsqueeze(1))
            all_violations.append(viol)
            all_ci_widths.append(width)

            all_samples.append(samples_last.cpu())
            all_gt.append(gt_last.cpu())

    # Aggregate results
    mean_violations = np.mean(all_violations) * 100  # Convert to percentage
    mean_ci_width = np.mean(all_ci_widths)

    # Compute kurtosis recovery
    all_samples_cat = torch.cat(all_samples, dim=1)  # (n_samples, total_B, 5, 5)
    all_gt_cat = torch.cat(all_gt, dim=0)  # (total_B, 5, 5)

    kurtosis_recovery, model_kurt, gt_kurt = compute_kurtosis_recovery(
        all_samples_cat.unsqueeze(2), all_gt_cat.unsqueeze(1)
    )

    return {
        "mode": mode,
        "ci_violations": mean_violations,
        "ci_width": mean_ci_width,
        "kurtosis_recovery": kurtosis_recovery,
        "model_kurtosis": model_kurt,
        "gt_kurtosis": gt_kurt,
    }


def main():
    print("=" * 70)
    print("Comparing Prior Methods for Student-t VAE")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    # Use validation set
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)
    val_data = log_returns[train_end:val_end]

    context_len = 20
    batch_size = 32
    n_samples = 200

    val_loader = create_dataloader(val_data, context_len, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Validation samples: {len(val_data) - context_len}")
    print(f"n_samples per point: {n_samples}")

    # Load models
    print("\nLoading models...")
    vae, predictor, config = load_vae_and_predictor(device)
    print("Models loaded.")

    # Evaluate all modes
    modes = ["oracle", "prior_n01", "prior_predictor"]
    results = {}

    for mode in modes:
        print(f"\nEvaluating {mode}...")
        results[mode] = evaluate_mode(
            vae, predictor, val_loader, mode,
            context_len=context_len, n_samples=n_samples, device=device
        )

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Mode':<20} {'CI Violations':<15} {'CI Width':<12} {'Kurtosis':<12}")
    print("-" * 60)

    for mode in modes:
        r = results[mode]
        print(f"{mode:<20} {r['ci_violations']:>10.1f}% {r['ci_width']:>12.6f} {r['kurtosis_recovery']:>10.1f}%")

    print("-" * 60)
    print(f"{'Target':<20} {'10.0%':<15} {'-':<12} {'100%':<12}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    oracle_viol = results["oracle"]["ci_violations"]
    n01_viol = results["prior_n01"]["ci_violations"]
    pred_viol = results["prior_predictor"]["ci_violations"]
    target = 10.0

    # Distance from target (10%)
    oracle_dist = abs(oracle_viol - target)
    n01_dist = abs(n01_viol - target)
    pred_dist = abs(pred_viol - target)

    print(f"\nDistance from 10% target:")
    print(f"  Oracle:     |{oracle_viol:.1f} - 10| = {oracle_dist:.1f}%")
    print(f"  N(0,1):     |{n01_viol:.1f} - 10| = {n01_dist:.1f}%")
    print(f"  Predictor:  |{pred_viol:.1f} - 10| = {pred_dist:.1f}%")

    # Determine winner
    print(f"\nInterpretation:")
    if n01_viol < target:
        print(f"  N(0,1): CIs TOO WIDE (only {n01_viol:.1f}% violations, need 10%)")
    else:
        print(f"  N(0,1): CIs too narrow ({n01_viol:.1f}% violations)")

    if pred_viol < target:
        print(f"  Predictor: CIs still wide ({pred_viol:.1f}% violations)")
    else:
        print(f"  Predictor: CIs slightly narrow ({pred_viol:.1f}% violations)")

    # Determine if predictor helps (closer to 10%)
    if pred_dist < n01_dist:
        improvement = (n01_dist - pred_dist) / n01_dist * 100
        print(f"\nPredictor IMPROVES calibration by {improvement:.1f}%!")
        print(f"  Moved {n01_dist - pred_dist:.1f}% closer to 10% target.")
    else:
        print(f"\nPredictor does NOT improve calibration.")

    # CI Width analysis
    oracle_width = results["oracle"]["ci_width"]
    n01_width = results["prior_n01"]["ci_width"]
    pred_width = results["prior_predictor"]["ci_width"]

    print(f"\nCI Width ratios:")
    print(f"  N(0,1) / Oracle: {n01_width / oracle_width:.2f}x")
    print(f"  Predictor / Oracle: {pred_width / oracle_width:.2f}x")

    # Save results
    save_dir = Path("models/backfill/two_stage/prior_network")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # CI Violations
    ax = axes[0]
    modes_short = ["Oracle", "Prior N(0,1)", "Prior Pred"]
    violations = [results[m]["ci_violations"] for m in modes]
    colors = ["green", "red", "blue"]
    bars = ax.bar(modes_short, violations, color=colors, alpha=0.7)
    ax.axhline(y=10, color="black", linestyle="--", label="Target (10%)")
    ax.set_ylabel("CI Violations (%)")
    ax.set_title("90% CI Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, violations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # CI Width
    ax = axes[1]
    widths = [results[m]["ci_width"] for m in modes]
    bars = ax.bar(modes_short, widths, color=colors, alpha=0.7)
    ax.set_ylabel("CI Width")
    ax.set_title("Confidence Interval Width")
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars, widths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Kurtosis Recovery
    ax = axes[2]
    kurtosis_vals = [results[m]["kurtosis_recovery"] for m in modes]
    bars = ax.bar(modes_short, kurtosis_vals, color=colors, alpha=0.7)
    ax.axhline(y=100, color="black", linestyle="--", label="Target (100%)")
    ax.set_ylabel("Kurtosis Recovery (%)")
    ax.set_title("Fat Tail Preservation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars, kurtosis_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10)

    plt.suptitle("Prior Methods Comparison: Oracle vs N(0,1) vs Predictor",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "prior_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_dir / 'prior_comparison.png'}")
    plt.close()

    # Save numerical results (convert numpy types to Python types)
    import json
    results_json = {}
    for mode, r in results.items():
        results_json[mode] = {k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
                             for k, v in r.items()}
    with open(save_dir / "comparison_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {save_dir / 'comparison_results.json'}")

    return results


if __name__ == "__main__":
    results = main()
