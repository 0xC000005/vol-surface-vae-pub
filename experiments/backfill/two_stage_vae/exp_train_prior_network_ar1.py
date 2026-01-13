"""
Train Prior Network with AR(1) Covariance Structure

This trains LatentPredictorCov which produces temporally correlated z samples
via Cholesky factorization of an AR(1) covariance matrix.

Key difference from standard prior network:
- Learns global rho (correlation decay) and sigma (innovation std)
- sample_z() produces correlated samples to address ACF mismatch

The AR(1) covariance structure is:
    Sigma[i,j] = sigma^2 * rho^|i-j|

Sampling: z = mu + L @ eps, where L = cholesky(Sigma), eps ~ N(0,I)

Usage:
    python experiments/backfill/two_stage_vae/exp_train_prior_network_ar1.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictorCov


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


def load_frozen_vae(model_path: str, device: str = "cuda"):
    """Load the Student-t VAE and freeze all parameters."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentTMLP(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # FREEZE all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print(f"Loaded frozen VAE from {model_path}")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  Parameters frozen: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def train_prior_network_ar1(
    vae_model,
    train_loader,
    val_loader,
    config,
    epochs: int = 100,
    lr: float = 1e-3,
    context_len: int = 20,
    init_rho: float = 0.8,
    init_sigma: float = 1.0,
):
    """
    Train LatentPredictorCov with AR(1) covariance structure.

    Args:
        vae_model: Frozen Student-t VAE
        train_loader: Training data
        val_loader: Validation data
        config: Model configuration
        epochs: Number of training epochs
        lr: Learning rate
        context_len: Context length (C)
        init_rho: Initial AR(1) correlation coefficient
        init_sigma: Initial innovation standard deviation

    Returns:
        predictor: Trained LatentPredictorCov
        history: Training history
    """
    device = config.get("device", "cuda")

    # Initialize predictor with AR(1) covariance
    predictor_config = config.copy()
    predictor_config["max_horizon"] = 30
    predictor_config["init_rho"] = init_rho
    predictor_config["init_sigma"] = init_sigma

    predictor = LatentPredictorCov(predictor_config)
    predictor = predictor.to(device)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "z_target_std": [],
        "z_pred_std": [],
        "rho": [],
        "sigma": [],
    }

    print(f"\nTraining LatentPredictorCov (AR(1) Covariance) for {epochs} epochs")
    print(f"  context_len={context_len}")
    print(f"  init_rho={init_rho}, init_sigma={init_sigma}")
    print("=" * 70)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # === TRAINING ===
        predictor.train()
        train_losses = []
        train_mses = []
        z_target_stds = []
        z_pred_stds = []

        for batch_data in train_loader:
            surface = batch_data[0].to(device)  # (B, T, 5, 5)
            B, T = surface.shape[:2]
            horizon = T - context_len

            if horizon <= 0:
                continue

            batch = {"surface": surface}

            # Get target z from FROZEN encoder (sees full sequence)
            with torch.no_grad():
                z_target, _, _ = vae_model.main_encoder(batch)  # (B, T, latent_dim)
                z_target_future = z_target[:, context_len:]  # (B, horizon, latent_dim)

            # Predict z from context ONLY
            context = surface[:, :context_len]  # (B, C, 5, 5)
            z_pred, z_logvar = predictor(context, horizon=horizon)

            # MSE loss on z_mean
            loss_mse = F.mse_loss(z_pred, z_target_future.detach())
            loss = loss_mse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_mses.append(loss_mse.item())
            z_target_stds.append(z_target_future.std().item())
            z_pred_stds.append(z_pred.std().item())

        # === VALIDATION ===
        predictor.eval()
        val_losses = []
        val_mses = []

        with torch.no_grad():
            for batch_data in val_loader:
                surface = batch_data[0].to(device)
                B, T = surface.shape[:2]
                horizon = T - context_len

                if horizon <= 0:
                    continue

                batch = {"surface": surface}
                z_target, _, _ = vae_model.main_encoder(batch)
                z_target_future = z_target[:, context_len:]

                context = surface[:, :context_len]
                z_pred, z_logvar = predictor(context, horizon=horizon)

                loss_mse = F.mse_loss(z_pred, z_target_future)
                loss = loss_mse

                val_losses.append(loss.item())
                val_mses.append(loss_mse.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_mse = np.mean(train_mses)
        val_mse = np.mean(val_mses)

        # Get current AR(1) parameters
        cov_params = predictor.get_covariance_params()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["z_target_std"].append(np.mean(z_target_stds))
        history["z_pred_std"].append(np.mean(z_pred_stds))
        history["rho"].append(cov_params["rho"])
        history["sigma"].append(cov_params["sigma"])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = predictor.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                f"rho={cov_params['rho']:.4f}, sigma={cov_params['sigma']:.4f}"
            )

    # Restore best model
    if best_state is not None:
        predictor.load_state_dict(best_state)

    final_params = predictor.get_covariance_params()
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Final AR(1) params: rho={final_params['rho']:.4f}, sigma={final_params['sigma']:.4f}")

    return predictor, history


def evaluate_predictor_ar1(vae_model, predictor, val_loader, config, context_len=20):
    """Evaluate trained AR(1) predictor on validation set."""
    device = config.get("device", "cuda")
    predictor.eval()

    all_z_target = []
    all_z_pred = []

    with torch.no_grad():
        for batch_data in val_loader:
            surface = batch_data[0].to(device)
            B, T = surface.shape[:2]
            horizon = T - context_len

            if horizon <= 0:
                continue

            batch = {"surface": surface}
            z_target_full, _, _ = vae_model.main_encoder(batch)
            z_target = z_target_full[:, context_len:]

            context = surface[:, :context_len]
            z_pred, _ = predictor(context, horizon=horizon)

            all_z_target.append(z_target.cpu())
            all_z_pred.append(z_pred.cpu())

    z_target = torch.cat(all_z_target, dim=0)
    z_pred = torch.cat(all_z_pred, dim=0)

    # Compute metrics
    mse = F.mse_loss(z_pred, z_target).item()
    mae = F.l1_loss(z_pred, z_target).item()

    # Per-dimension correlation
    correlations = []
    for d in range(z_target.shape[-1]):
        target_flat = z_target[:, :, d].flatten()
        pred_flat = z_pred[:, :, d].flatten()
        corr = torch.corrcoef(torch.stack([target_flat, pred_flat]))[0, 1].item()
        correlations.append(corr)

    # AR(1) covariance parameters
    cov_params = predictor.get_covariance_params()

    print("\n" + "=" * 70)
    print("AR(1) PREDICTOR EVALUATION")
    print("=" * 70)
    print(f"MSE:              {mse:.6f}")
    print(f"MAE:              {mae:.6f}")
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Min correlation:  {np.min(correlations):.4f}")
    print(f"AR(1) rho:        {cov_params['rho']:.4f}")
    print(f"AR(1) sigma:      {cov_params['sigma']:.4f}")

    return {
        "mse": mse,
        "mae": mae,
        "correlations": correlations,
        "mean_corr": np.mean(correlations),
        "rho": cov_params["rho"],
        "sigma": cov_params["sigma"],
    }


def main():
    print("=" * 70)
    print("Training Prior Network with AR(1) Covariance Structure")
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

    # Load frozen VAE
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_model, config = load_frozen_vae(vae_path, device)

    # Train AR(1) predictor
    predictor, history = train_prior_network_ar1(
        vae_model,
        train_loader,
        val_loader,
        config,
        epochs=100,
        lr=1e-3,
        context_len=context_len,
        init_rho=0.8,  # Initialize with reasonable mean-reversion
        init_sigma=1.0,
    )

    # Evaluate
    eval_results = evaluate_predictor_ar1(
        vae_model, predictor, val_loader, config, context_len
    )

    # Save predictor
    save_dir = Path("models/backfill/two_stage/prior_network_ar1")
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "predictor_state_dict": predictor.state_dict(),
        "config": config,
        "training_history": history,
        "eval_results": eval_results,
        "context_len": context_len,
        "predictor_type": "LatentPredictorCov",
    }
    torch.save(checkpoint, save_dir / "prior_network_ar1_best.pt")
    print(f"\nPredictor saved to {save_dir / 'prior_network_ar1_best.pt'}")

    # Plot training history
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["train_mse"], label="Train")
    axes[0, 1].plot(history["val_mse"], label="Val")
    axes[0, 1].set_title("MSE Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history["z_target_std"], label="z_target")
    axes[0, 2].plot(history["z_pred_std"], label="z_pred")
    axes[0, 2].set_title("z Standard Deviation")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # AR(1) parameters over training
    axes[1, 0].plot(history["rho"], 'b-', linewidth=2)
    axes[1, 0].axhline(y=0.8, color='r', linestyle='--', label='Init (0.8)')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("rho")
    axes[1, 0].set_title("AR(1) Correlation (rho)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["sigma"], 'g-', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Init (1.0)')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("sigma")
    axes[1, 1].set_title("AR(1) Innovation Std (sigma)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Per-dimension correlation
    axes[1, 2].bar(range(len(eval_results["correlations"])), eval_results["correlations"])
    axes[1, 2].axhline(y=eval_results["mean_corr"], color='r', linestyle='--',
                       label=f'Mean={eval_results["mean_corr"]:.3f}')
    axes[1, 2].set_xlabel("Latent Dimension")
    axes[1, 2].set_ylabel("Correlation")
    axes[1, 2].set_title("z_pred vs z_target Correlation")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(
        f"AR(1) Prior Network Training | Final: rho={eval_results['rho']:.3f}, sigma={eval_results['sigma']:.3f}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "training_history.png", dpi=150, bbox_inches="tight")
    print(f"Training history saved to {save_dir / 'training_history.png'}")
    plt.close()

    return predictor, history, eval_results


if __name__ == "__main__":
    predictor, history, eval_results = main()
