"""
Train Spatial Multi-Task Model (Option 6)

Combines the promising multi-task approach (Option 4) with CNN spatial smoothing.
The Conv2d layers enforce spatial correlation, allowing stable ATM predictions
to help stabilize OTM corners during autoregressive chaining.

Usage:
    python experiments/backfill/two_stage_vae/train_spatial_multitask.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageSpatialMultiTask


def create_sequences(data: np.ndarray, seq_len: int = 30):
    """Create overlapping sequences from time series data."""
    n_samples = len(data) - seq_len + 1
    sequences = np.zeros((n_samples, seq_len, *data.shape[1:]))

    for i in range(n_samples):
        sequences[i] = data[i:i + seq_len]

    return sequences


def main():
    print("=" * 70)
    print("Training Spatial Multi-Task Model (Option 6)")
    print("CNN + Multi-Task: Enforce spatial correlation while preserving z")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    # Convert to log-returns
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)

    # Create sequences
    seq_len = 30
    return_sequences = create_sequences(log_returns, seq_len)
    level_sequences = create_sequences(log_surfaces[1:], seq_len)  # Aligned with returns

    # Train/val split
    train_end = int(len(return_sequences) * 0.7)
    train_returns = return_sequences[:train_end]
    val_returns = return_sequences[train_end:]
    train_levels = level_sequences[:train_end]
    val_levels = level_sequences[train_end:]

    print(f"\nData shapes:")
    print(f"  Train sequences: {train_returns.shape}")
    print(f"  Val sequences: {val_returns.shape}")
    print(f"  Train levels: {train_levels.shape}")
    print(f"  Val levels: {val_levels.shape}")

    # Create dataloaders
    train_return_tensor = torch.tensor(train_returns, dtype=torch.float32)
    val_return_tensor = torch.tensor(val_returns, dtype=torch.float32)
    train_level_tensor = torch.tensor(train_levels, dtype=torch.float32)
    val_level_tensor = torch.tensor(val_levels, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_return_tensor, train_level_tensor),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_return_tensor, val_level_tensor),
        batch_size=64,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model config
    config = {
        "feat_dim": (5, 5),
        "latent_dim": 16,
        "ctx_embedding_dim": 3,
        "context_len": 20,
        "cov_rank": 4,
        "learn_ar_phi": True,
        "target_ar_phi": -0.35,
        "lambda_level": 0.1,  # Weight for level loss
        "device": device,
        # Encoder config
        "surface_hidden": [2, 4, 2],
        "mem_type": "lstm",
        "mem_hidden": 8,
        "mem_layers": 1,
        "mem_dropout": 0.2,
        # Ctx encoder
        "ctx_surface_hidden": [2, 4, 2],
        "ctx_mem_type": "lstm",
        "ctx_mem_hidden": 8,
        "ctx_mem_layers": 1,
        "ctx_mem_dropout": 0.2,
        "ctx_compress": True,
    }

    print(f"\nConfig:")
    print(f"  Lambda level: {config['lambda_level']}")
    print(f"  Latent dim: {config['latent_dim']}")
    print(f"  Cov rank: {config['cov_rank']}")

    # Create model
    model = CVAETwoStageSpatialMultiTask(config)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # Print architecture info
    print(f"\nArchitecture:")
    print(f"  z → MLP (16→128→64) → feature heads (64→25)")
    print(f"  + Conv2d spatial smoothing (1→8→4→1 channels, 3×3 kernel)")
    print(f"  + Multi-task: return + level prediction")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training
    n_epochs = 100
    best_val_loss = float("inf")
    save_dir = Path("models/backfill/two_stage/cumulative_options")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("Training Spatial Multi-Task Model")
    print("=" * 70)

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_losses = []
        train_nll = []
        train_level_mse = []

        for batch_returns, batch_levels in train_loader:
            batch_returns = batch_returns.to(device)
            batch_levels = batch_levels.to(device)
            optimizer.zero_grad()

            loss_dict = model.compute_loss(
                {"surface": batch_returns},
                log_surfaces=batch_levels,
                kl_weight=0.1
            )
            loss = loss_dict["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_nll.append(loss_dict["nll"].item())
            train_level_mse.append(loss_dict["level_mse"].item())

        # Validate
        model.eval()
        val_losses = []
        val_nll = []
        val_level_mse = []

        with torch.no_grad():
            for batch_returns, batch_levels in val_loader:
                batch_returns = batch_returns.to(device)
                batch_levels = batch_levels.to(device)

                loss_dict = model.compute_loss(
                    {"surface": batch_returns},
                    log_surfaces=batch_levels,
                    kl_weight=0.1
                )
                val_losses.append(loss_dict["loss"].item())
                val_nll.append(loss_dict["nll"].item())
                val_level_mse.append(loss_dict["level_mse"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "val_loss": val_loss,
                "epoch": epoch,
            }, save_dir / "spatial_multitask_best.pt")

        if epoch % 10 == 0:
            phi = model.get_ar_phi().item()
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, phi={phi:.4f}, "
                  f"nll={np.mean(val_nll):.4f}, level_mse={np.mean(val_level_mse):.6f}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"Model saved to: {save_dir / 'spatial_multitask_best.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
