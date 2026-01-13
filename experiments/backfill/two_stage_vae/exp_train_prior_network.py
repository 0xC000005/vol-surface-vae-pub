"""
Train Prior Network (LatentPredictor) for Student-t VAE

Stage 2 of Two-Stage VAE training:
- Stage 1 (DONE): Train VAE with Student-t decoder
- Stage 2 (THIS): Train LatentPredictor with FROZEN VAE

The predictor learns to predict z from context ONLY, without seeing the target.
This narrows the oracle-vs-prior gap in CI calibration.

Loss: MSE(z_predicted, z_encoder) where z_encoder is from frozen VAE

References:
- Conditional Prior Networks (ML Journal, 2022)
- CLARM (Nature, 2024) - CVAE + LSTM for forecasting

Usage:
    python experiments/backfill/two_stage_vae/exp_train_prior_network.py
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
from vae.predictors import LatentPredictor
from config.two_stage_config import TWO_STAGE_CONFIG


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


def train_prior_network(
    vae_model,
    train_loader,
    val_loader,
    config,
    epochs: int = 100,
    lr: float = 1e-3,
    context_len: int = 20,
    use_nll: bool = False,
    nll_weight: float = 0.1,
):
    """
    Train LatentPredictor with frozen VAE.

    Args:
        vae_model: Frozen Student-t VAE
        train_loader: Training data
        val_loader: Validation data
        config: Model configuration
        epochs: Number of training epochs
        lr: Learning rate
        context_len: Context length (C)
        use_nll: Whether to add NLL loss for variance learning
        nll_weight: Weight for NLL loss term

    Returns:
        predictor: Trained LatentPredictor
        history: Training history
    """
    device = config.get("device", "cuda")

    # Initialize predictor with matching config
    predictor_config = config.copy()
    predictor_config["max_horizon"] = 30  # Support up to 30-day prediction

    predictor = LatentPredictor(predictor_config)
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
    }

    print(f"\nTraining LatentPredictor for {epochs} epochs")
    print(f"  context_len={context_len}, use_nll={use_nll}")
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

            # Optional NLL loss for variance
            if use_nll:
                # NLL = 0.5 * (logvar + (target - pred)^2 / exp(logvar))
                var = torch.exp(z_logvar)
                loss_nll = 0.5 * (z_logvar + (z_target_future.detach() - z_pred).pow(2) / var)
                loss_nll = loss_nll.mean()
                loss = loss_mse + nll_weight * loss_nll
            else:
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

                if use_nll:
                    var = torch.exp(z_logvar)
                    loss_nll = 0.5 * (z_logvar + (z_target_future - z_pred).pow(2) / var)
                    loss_nll = loss_nll.mean()
                    loss = loss_mse + nll_weight * loss_nll
                else:
                    loss = loss_mse

                val_losses.append(loss.item())
                val_mses.append(loss_mse.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_mse = np.mean(train_mses)
        val_mse = np.mean(val_mses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["z_target_std"].append(np.mean(z_target_stds))
        history["z_pred_std"].append(np.mean(z_pred_stds))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = predictor.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs}: "
                f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                f"MSE={val_mse:.6f}, "
                f"z_std(target/pred)={np.mean(z_target_stds):.3f}/{np.mean(z_pred_stds):.3f}"
            )

    # Restore best model
    if best_state is not None:
        predictor.load_state_dict(best_state)

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return predictor, history


def evaluate_predictor(vae_model, predictor, val_loader, config, context_len=20):
    """Evaluate trained predictor on validation set."""
    device = config.get("device", "cuda")
    predictor.eval()

    all_z_target = []
    all_z_pred = []
    all_z_logvar = []

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
            z_pred, z_logvar = predictor(context, horizon=horizon)

            all_z_target.append(z_target.cpu())
            all_z_pred.append(z_pred.cpu())
            all_z_logvar.append(z_logvar.cpu())

    z_target = torch.cat(all_z_target, dim=0)
    z_pred = torch.cat(all_z_pred, dim=0)
    z_logvar = torch.cat(all_z_logvar, dim=0)

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

    # Variance statistics
    pred_std = torch.exp(0.5 * z_logvar).mean().item()
    target_std = z_target.std().item()

    print("\n" + "=" * 70)
    print("PREDICTOR EVALUATION")
    print("=" * 70)
    print(f"MSE:              {mse:.6f}")
    print(f"MAE:              {mae:.6f}")
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Min correlation:  {np.min(correlations):.4f}")
    print(f"z_target std:     {target_std:.4f}")
    print(f"z_pred std:       {pred_std:.4f}")

    return {
        "mse": mse,
        "mae": mae,
        "correlations": correlations,
        "mean_corr": np.mean(correlations),
        "pred_std": pred_std,
        "target_std": target_std,
    }


def main():
    print("=" * 70)
    print("Training Prior Network (LatentPredictor) for Student-t VAE")
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

    # Train predictor
    predictor, history = train_prior_network(
        vae_model,
        train_loader,
        val_loader,
        config,
        epochs=100,
        lr=1e-3,
        context_len=context_len,
        use_nll=False,  # Start with pure MSE
    )

    # Evaluate
    eval_results = evaluate_predictor(
        vae_model, predictor, val_loader, config, context_len
    )

    # Save predictor
    save_dir = Path("models/backfill/two_stage/prior_network")
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "predictor_state_dict": predictor.state_dict(),
        "config": config,
        "training_history": history,
        "eval_results": eval_results,
        "context_len": context_len,
    }
    torch.save(checkpoint, save_dir / "prior_network_best.pt")
    print(f"\nPredictor saved to {save_dir / 'prior_network_best.pt'}")

    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

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

    axes[1, 0].plot(history["z_target_std"], label="z_target")
    axes[1, 0].plot(history["z_pred_std"], label="z_pred")
    axes[1, 0].set_title("z Standard Deviation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Per-dimension correlation
    axes[1, 1].bar(range(len(eval_results["correlations"])), eval_results["correlations"])
    axes[1, 1].axhline(y=eval_results["mean_corr"], color='r', linestyle='--',
                       label=f'Mean={eval_results["mean_corr"]:.3f}')
    axes[1, 1].set_xlabel("Latent Dimension")
    axes[1, 1].set_ylabel("Correlation")
    axes[1, 1].set_title("z_pred vs z_target Correlation per Dimension")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Prior Network Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "training_history.png", dpi=150, bbox_inches="tight")
    print(f"Training history saved to {save_dir / 'training_history.png'}")
    plt.close()

    return predictor, history, eval_results


if __name__ == "__main__":
    predictor, history, eval_results = main()
