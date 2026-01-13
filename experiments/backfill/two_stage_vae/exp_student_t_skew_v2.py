"""
Skewed Student-t VAE v2: With Variance Regularization

This version fixes the variance collapse issue found in v1 by:
1. Adding variance regularization to prevent log_diag from hitting the -10 clamp
2. Encouraging the factor matrix to contribute to variance
3. Monitoring sigma during training

Usage:
    python experiments/backfill/two_stage_vae/exp_student_t_skew_v2.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
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


def evaluate_distribution_shape(model, val_loader, config, n_samples=50):
    """Evaluate kurtosis and skewness recovery."""
    model.eval()
    device = config.get("device", "cuda")

    all_gt = []
    all_samples = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            samples = model.sample(batch, n_samples=n_samples)
            samples_last = samples[:, :, -1]
            gt = surface[:, -1]

            all_gt.append(gt.cpu().numpy())
            all_samples.append(samples_last.cpu().numpy())

    all_gt = np.concatenate(all_gt, axis=0)
    all_samples = np.concatenate(all_samples, axis=1)

    gt_kurtosis = np.zeros((5, 5))
    gt_skewness = np.zeros((5, 5))
    model_kurtosis = np.zeros((5, 5))
    model_skewness = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_kurtosis[i, j] = kurtosis(all_gt[:, i, j], fisher=True)
            gt_skewness[i, j] = skew(all_gt[:, i, j])
            pooled = all_samples[:, :, i, j].flatten()
            model_kurtosis[i, j] = kurtosis(pooled, fisher=True)
            model_skewness[i, j] = skew(pooled)

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
    """Evaluate CI calibration."""
    model.eval()
    device = config.get("device", "cuda")

    violations = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            samples = model.sample(batch, n_samples=n_samples)
            samples_last = samples[:, :, -1]
            gt = surface[:, -1]

            p05 = torch.quantile(samples_last, 0.05, dim=0)
            p95 = torch.quantile(samples_last, 0.95, dim=0)

            below = (gt < p05).float()
            above = (gt > p95).float()
            batch_violations = (below + above).mean().item()
            violations.append(batch_violations)

    ci_violations = np.mean(violations) * 100

    return {
        "ci_violations": ci_violations,
        "target": 10.0,
    }


def train_skew_student_t_v2(
    model, train_loader, val_loader, config, epochs=150,
    kl_weight=0.01, var_reg_weight=1.0, min_log_diag=-6.0
):
    """Train the skewed Student-t model with variance regularization."""
    device = config.get("device", "cuda")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "sigma_mean": [], "log_diag_mean": []}

    print(f"\nTraining Skewed Student-t VAE v2 for {epochs} epochs")
    print(f"  kl_weight={kl_weight}, var_reg_weight={var_reg_weight}, min_log_diag={min_log_diag}")
    print("=" * 70)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_sigmas = []
        train_log_diags = []

        for batch_data in train_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            optimizer.zero_grad()
            loss_dict = model.compute_loss(
                batch, kl_weight=kl_weight,
                var_reg_weight=var_reg_weight, min_log_diag=min_log_diag
            )

            loss = loss_dict["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_sigmas.append(loss_dict["sigma_mean"])
            train_log_diags.append(loss_dict["log_diag_mean"])

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_data in val_loader:
                surface = batch_data[0].to(device)
                batch = {"surface": surface}
                loss_dict = model.compute_loss(
                    batch, kl_weight=kl_weight,
                    var_reg_weight=var_reg_weight, min_log_diag=min_log_diag
                )
                val_losses.append(loss_dict["loss"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        sigma_mean = np.mean(train_sigmas)
        log_diag_mean = np.mean(train_log_diags)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["sigma_mean"].append(sigma_mean)
        history["log_diag_mean"].append(log_diag_mean)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                f"σ_mean={sigma_mean:.4f}, log_diag={log_diag_mean:.2f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def compare_sample_variance(sym_model, skew_model, val_loader, config, n_samples=1000):
    """Compare sample variance between symmetric and skewed models."""
    device = config.get("device", "cuda")

    # Get one batch
    batch_data = next(iter(val_loader))
    surface = batch_data[0][:1].to(device)  # Just one sample
    batch = {"surface": surface}

    with torch.no_grad():
        sym_samples = sym_model.sample(batch, n_samples=n_samples)
        skew_samples = skew_model.sample(batch, n_samples=n_samples)

    sym_atm = sym_samples[:, 0, -1, 2, 2].cpu().numpy()
    skew_atm = skew_samples[:, 0, -1, 2, 2].cpu().numpy()

    print("\nSample Variance Comparison (ATM grid point):")
    print("=" * 50)
    print(f"Symmetric: std={sym_atm.std():.6f}, range=[{sym_atm.min():.4f}, {sym_atm.max():.4f}]")
    print(f"Skewed:    std={skew_atm.std():.6f}, range=[{skew_atm.min():.4f}, {skew_atm.max():.4f}]")
    print(f"Ratio (sym/skew): {sym_atm.std() / skew_atm.std():.2f}x")

    return sym_atm.std(), skew_atm.std()


def main():
    print("=" * 70)
    print("Skewed Student-t VAE v2: With Variance Regularization")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"Data shape: {log_returns.shape}")

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
    config["learn_delta"] = False

    # Load symmetric model for comparison
    print("\nLoading symmetric baseline for comparison...")
    sym_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    sym_ckpt = torch.load(sym_path, map_location=device, weights_only=False)
    sym_model = CVAETwoStageStudentTMLP(sym_ckpt["model_config"])
    sym_model.load_state_dict(sym_ckpt["model_state_dict"])
    sym_model = sym_model.to(device).eval()

    # =========================================================================
    # Train Skewed Student-t v2 with variance regularization
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training Skewed Student-t v2 (with variance regularization)")
    print("=" * 70)

    model_skew = CVAETwoStageStudentTSkew(config)
    model_skew, history = train_skew_student_t_v2(
        model_skew, train_loader, val_loader, config,
        epochs=150,
        kl_weight=0.01,
        var_reg_weight=1.0,  # Variance regularization weight
        min_log_diag=-6.0    # Target minimum log_diag (symmetric model has ~-8)
    )

    # Compare sample variance
    print("\n" + "=" * 70)
    print("Comparing Sample Variance")
    print("=" * 70)
    sym_std, skew_std = compare_sample_variance(sym_model, model_skew, val_loader, config)

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating Skewed Model v2")
    print("=" * 70)
    shape_results = evaluate_distribution_shape(model_skew, val_loader, config)
    ci_results = evaluate_ci_calibration(model_skew, val_loader, config)

    print(f"\nResults:")
    print(f"  Kurtosis Recovery:   {shape_results['kurtosis_recovery']:.1f}%")
    print(f"  Skewness Sign Match: {shape_results['skewness_sign_match']:.1f}%")
    print(f"  CI Violations:       {ci_results['ci_violations']:.1f}% (target: 10%)")
    print(f"  Variance Ratio:      {sym_std / skew_std:.2f}x (target: ~1.0x)")

    # Save model
    save_dir = Path("models/backfill/two_stage/student_t_skew_v2")
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model_skew.state_dict(),
        "model_config": config,
        "training_history": history,
        "results": {
            "kurtosis_recovery": shape_results["kurtosis_recovery"],
            "skewness_sign_match": shape_results["skewness_sign_match"],
            "ci_violations": ci_results["ci_violations"],
            "variance_ratio": sym_std / skew_std,
        },
    }
    torch.save(checkpoint, save_dir / "student_t_skew_v2_best.pt")
    print(f"\nModel saved to {save_dir / 'student_t_skew_v2_best.pt'}")

    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["sigma_mean"])
    axes[0, 1].axhline(y=0.025, color='r', linestyle='--', label='Symmetric σ')
    axes[0, 1].set_title("Mean Sigma")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["log_diag_mean"])
    axes[1, 0].axhline(y=-8.3, color='r', linestyle='--', label='Symmetric log_diag')
    axes[1, 0].axhline(y=-6.0, color='g', linestyle='--', label='Target min')
    axes[1, 0].set_title("Mean log_diag")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary text
    axes[1, 1].axis('off')
    summary = (
        f"Skewed Student-t v2 Results\n"
        f"{'='*30}\n"
        f"Kurtosis Recovery: {shape_results['kurtosis_recovery']:.1f}%\n"
        f"Skewness Sign Match: {shape_results['skewness_sign_match']:.1f}%\n"
        f"CI Violations: {ci_results['ci_violations']:.1f}%\n"
        f"Variance Ratio: {sym_std / skew_std:.2f}x\n"
        f"\nTarget:\n"
        f"  CI Violations: ~10%\n"
        f"  Variance Ratio: ~1.0x"
    )
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_dir / "training_history.png", dpi=150, bbox_inches="tight")
    print(f"Training history saved to {save_dir / 'training_history.png'}")
    plt.close()

    return model_skew, shape_results, ci_results


if __name__ == "__main__":
    model, shape_results, ci_results = main()
