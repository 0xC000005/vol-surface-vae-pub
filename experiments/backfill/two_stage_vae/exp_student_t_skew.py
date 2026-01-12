"""
Skewed Student-t Decoder: Adding Learnable Skewness via Sinh-Arcsinh Transform

Problem:
    Current Student-t decoder is symmetric - only models fat tails (kurtosis).
    Real IV log-returns may have asymmetric distributions (positive or negative skew).

Solution:
    Apply sinh-arcsinh transformation to add learnable skewness:
        Y = sinh((arcsinh(X) + ε) * δ)

    Where:
        X ~ StudentT(ν)  (symmetric base distribution)
        ε: Skewness parameter (learned per grid point)
        δ: Tailweight parameter (optional, can fix to 1)

    When ε=0, δ=1: reduces to symmetric Student-t (baseline)

Bitter Lesson Approach:
    - Don't hand-engineer which grids should have positive/negative skew
    - Let the model learn ε from data through NLL optimization
    - Minimal architecture change: just add ε (and optionally δ) outputs

Expected Results:
    - Skewness captured in samples
    - Kurtosis preserved (>100% recovery)
    - CI calibration maintained (~10% violations)

Usage:
    python experiments/backfill/two_stage_vae/exp_student_t_skew.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.two_stage_config import TWO_STAGE_CONFIG
from vae.cvae_two_stage import CVAETwoStageStudentTSkew, CVAETwoStageStudentTMLP


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, batch_size, shuffle=True):
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


def compute_gt_skewness(log_returns: np.ndarray) -> np.ndarray:
    """Compute ground truth skewness per grid point."""
    N, H, W = log_returns.shape
    gt_skew = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            gt_skew[i, j] = skew(log_returns[:, i, j])

    return gt_skew


def evaluate_distribution_shape(model, val_loader, config, n_samples=50):
    """
    Evaluate kurtosis and skewness recovery of model samples vs ground truth.
    """
    model.eval()
    device = config.get("device", "cuda")

    all_gt = []
    all_samples = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            # Generate samples
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5)

            # Get last timestep predictions
            samples_last = samples[:, :, -1]  # (n_samples, B, 5, 5)

            # Ground truth (last timestep)
            gt = surface[:, -1]  # (B, 5, 5)

            all_gt.append(gt.cpu().numpy())
            all_samples.append(samples_last.cpu().numpy())

    # Stack all batches
    all_gt = np.concatenate(all_gt, axis=0)  # (N, 5, 5)
    all_samples = np.concatenate(all_samples, axis=1)  # (n_samples, N, 5, 5)

    # Compute GT kurtosis and skewness
    gt_kurtosis = np.zeros((5, 5))
    gt_skewness = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            gt_kurtosis[i, j] = kurtosis(all_gt[:, i, j], fisher=True)
            gt_skewness[i, j] = skew(all_gt[:, i, j])

    # Compute model kurtosis and skewness (average across samples)
    model_kurtosis = np.zeros((5, 5))
    model_skewness = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            # Pool all samples for this grid point
            pooled = all_samples[:, :, i, j].flatten()
            model_kurtosis[i, j] = kurtosis(pooled, fisher=True)
            model_skewness[i, j] = skew(pooled)

    # Recovery ratios
    kurtosis_recovery = np.mean(model_kurtosis / (gt_kurtosis + 1e-8)) * 100
    skewness_sign_match = np.mean(np.sign(model_skewness) == np.sign(gt_skewness)) * 100

    return {
        "gt_kurtosis": gt_kurtosis,
        "model_kurtosis": model_kurtosis,
        "kurtosis_recovery": kurtosis_recovery,
        "gt_skewness": gt_skewness,
        "model_skewness": model_skewness,
        "skewness_sign_match": skewness_sign_match,
    }


def evaluate_ci_calibration(model, val_loader, config, n_samples=100):
    """Evaluate CI calibration (90% CI should have 10% violations)."""
    model.eval()
    device = config.get("device", "cuda")

    violations = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            # Generate samples
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5)

            # Get last timestep
            samples_last = samples[:, :, -1]  # (n_samples, B, 5, 5)
            gt = surface[:, -1]  # (B, 5, 5)

            # Compute 90% CI (5th and 95th percentiles)
            p05 = torch.quantile(samples_last, 0.05, dim=0)
            p95 = torch.quantile(samples_last, 0.95, dim=0)

            # Count violations
            below = (gt < p05).float()
            above = (gt > p95).float()
            batch_violations = (below + above).mean().item()
            violations.append(batch_violations)

    ci_violations = np.mean(violations) * 100

    return {
        "ci_violations": ci_violations,
        "target": 10.0,
    }


def train_skew_student_t(
    model, train_loader, val_loader, config, epochs=100, kl_weight=0.01
):
    """Train the skewed Student-t model."""
    device = config.get("device", "cuda")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_state = None

    print(f"\nTraining Skewed Student-t VAE for {epochs} epochs")
    print("=" * 70)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_epsilons = []

        for batch_data in train_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch, kl_weight=kl_weight)

            loss = loss_dict["loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_losses.append(loss.item())
            train_epsilons.append(loss_dict["epsilon_mean"])

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_data in val_loader:
                surface = batch_data[0].to(device)
                batch = {"surface": surface}
                loss_dict = model.compute_loss(batch, kl_weight=kl_weight)
                val_losses.append(loss_dict["loss"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        epsilon_mean = np.mean(train_epsilons)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                f"ε_mean={epsilon_mean:.4f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def visualize_learned_skewness(model, val_loader, config, save_path=None):
    """Visualize the learned skewness parameters."""
    model.eval()
    device = config.get("device", "cuda")

    all_epsilon = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            skew_params = model.get_learned_skewness(batch)
            epsilon = skew_params["epsilon"].cpu().numpy()  # (B, 25)
            all_epsilon.append(epsilon)

    all_epsilon = np.concatenate(all_epsilon, axis=0)  # (N, 25)
    mean_epsilon = all_epsilon.mean(axis=0).reshape(5, 5)

    # Also compute GT skewness for comparison
    gt_skewness = None
    all_gt = []
    for batch_data in val_loader:
        surface = batch_data[0].numpy()
        all_gt.append(surface[:, -1])  # Last timestep
    all_gt = np.concatenate(all_gt, axis=0)
    gt_skewness = compute_gt_skewness(all_gt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Learned epsilon
    im1 = axes[0].imshow(mean_epsilon, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Learned ε (Skewness)")
    axes[0].set_xlabel("Maturity")
    axes[0].set_ylabel("Moneyness")
    plt.colorbar(im1, ax=axes[0])

    # GT skewness
    if gt_skewness is not None:
        vmax = np.abs(gt_skewness).max()
        im2 = axes[1].imshow(gt_skewness, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("GT Skewness")
        axes[1].set_xlabel("Maturity")
        axes[1].set_ylabel("Moneyness")
        plt.colorbar(im2, ax=axes[1])

    # Sign match
    sign_match = (np.sign(mean_epsilon) == np.sign(gt_skewness)).astype(float)
    im3 = axes[2].imshow(sign_match, cmap="RdYlGn", vmin=0, vmax=1)
    axes[2].set_title(f"Sign Match: {sign_match.mean()*100:.1f}%")
    axes[2].set_xlabel("Maturity")
    axes[2].set_ylabel("Moneyness")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.close()

    return mean_epsilon, gt_skewness


def main():
    print("=" * 70)
    print("Skewed Student-t VAE: Sinh-Arcsinh Transformation")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"Data shape: {log_returns.shape}")

    # Compute and print GT skewness
    gt_skew = compute_gt_skewness(log_returns)
    print(f"\nGT Skewness (per grid):")
    print(np.round(gt_skew, 3))
    print(f"Mean absolute skewness: {np.abs(gt_skew).mean():.3f}")

    # Split data
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)

    train_data = log_returns[:train_end]
    val_data = log_returns[train_end:val_end]

    context_len = 20
    batch_size = 64

    train_loader = create_dataloader(train_data, context_len, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, context_len, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    config = TWO_STAGE_CONFIG.copy()
    config["device"] = device
    config["learn_delta"] = False  # Fix delta=1 initially

    # =========================================================================
    # Experiment 1: Skewed Student-t (learn epsilon only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Experiment 1: Skewed Student-t (ε learned, δ=1)")
    print("=" * 70)

    model_skew = CVAETwoStageStudentTSkew(config)
    model_skew = train_skew_student_t(
        model_skew, train_loader, val_loader, config, epochs=100
    )

    # Evaluate
    print("\nEvaluating...")
    shape_results = evaluate_distribution_shape(model_skew, val_loader, config)
    ci_results = evaluate_ci_calibration(model_skew, val_loader, config)

    print(f"\n{'='*70}")
    print("SKEWED STUDENT-T RESULTS")
    print(f"{'='*70}")
    print(f"  Kurtosis Recovery:  {shape_results['kurtosis_recovery']:.1f}%")
    print(f"  Skewness Sign Match: {shape_results['skewness_sign_match']:.1f}%")
    print(f"  CI Violations:      {ci_results['ci_violations']:.1f}% (target: 10%)")

    # Visualize learned skewness
    save_dir = Path("models/backfill/two_stage/student_t_skew")
    save_dir.mkdir(parents=True, exist_ok=True)

    mean_epsilon, gt_skewness = visualize_learned_skewness(
        model_skew, val_loader, config, save_path=save_dir / "learned_skewness.png"
    )

    print(f"\nLearned ε (mean across batches):")
    print(np.round(mean_epsilon, 3))

    # Save model
    checkpoint = {
        "model_state_dict": model_skew.state_dict(),
        "model_config": config,
        "results": {
            "kurtosis_recovery": shape_results["kurtosis_recovery"],
            "skewness_sign_match": shape_results["skewness_sign_match"],
            "ci_violations": ci_results["ci_violations"],
        },
    }
    torch.save(checkpoint, save_dir / "student_t_skew_best.pt")
    print(f"\nModel saved to {save_dir / 'student_t_skew_best.pt'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Skewness Sign Match: {shape_results['skewness_sign_match']:.1f}%")
    print(f"  Kurtosis Recovery:   {shape_results['kurtosis_recovery']:.1f}%")
    print(f"  CI Violations:       {ci_results['ci_violations']:.1f}% (target: 10%)")
    print(f"\n  Note: Compare with CVAETwoStageStudentTMLP baseline using compare_all_decoders.py")

    return model_skew, shape_results, ci_results


if __name__ == "__main__":
    model, shape_results, ci_results = main()
