"""
Train Per-Grid-Point Level Reversion Model

This model uses variance-scaled level reversion: OTM corners get stronger
reversion (up to 0.9) based on their empirical variance relative to ATM.

Usage:
    python experiments/backfill/two_stage_vae/train_per_grid_reversion.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStagePerGridReversion


def create_sequences(data: np.ndarray, seq_len: int = 30):
    """Create overlapping sequences from time series data."""
    n_samples = len(data) - seq_len + 1
    sequences = np.zeros((n_samples, seq_len, *data.shape[1:]))

    for i in range(n_samples):
        sequences[i] = data[i:i + seq_len]

    return sequences


def main():
    print("=" * 70)
    print("Training Per-Grid-Point Level Reversion Model")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    # Convert to log-returns
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)

    # Create sequences
    seq_len = 30
    sequences = create_sequences(log_returns, seq_len)

    # Train/val split
    train_end = int(len(sequences) * 0.7)
    train_sequences = sequences[:train_end]
    val_sequences = sequences[train_end:]

    print(f"\nData shapes:")
    print(f"  Train sequences: {train_sequences.shape}")
    print(f"  Val sequences: {val_sequences.shape}")

    # Create dataloaders
    train_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    val_tensor = torch.tensor(val_sequences, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=64,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model config (matches train_cumulative_options.py)
    config = {
        "feat_dim": (5, 5),
        "latent_dim": 16,
        "ctx_embedding_dim": 3,
        "context_len": 20,
        "cov_rank": 4,
        "learn_ar_phi": True,
        "target_ar_phi": -0.35,
        "base_reversion_strength": 0.3,  # Base for ATM; corners get up to 0.9
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
    print(f"  Base reversion strength: {config['base_reversion_strength']}")
    print(f"  (OTM corners will get up to 0.9 via variance scaling)")

    # Create model
    model = CVAETwoStagePerGridReversion(config)
    model = model.to(device)

    # Print per-grid reversion strengths
    per_grid = model.get_per_grid_reversion().cpu().numpy().reshape(5, 5)
    print(f"\nPer-grid reversion strengths:")
    for i in range(5):
        row = '  [' + ', '.join([f'{per_grid[i,j]:.2f}' for j in range(5)]) + ']'
        print(row)

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
    print("Training Per-Grid Reversion Model")
    print("=" * 70)

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_losses = []

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            loss_dict = model.compute_loss({"surface": batch}, kl_weight=0.1)
            loss = loss_dict["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                loss_dict = model.compute_loss({"surface": batch}, kl_weight=0.1)
                val_losses.append(loss_dict["loss"].item())

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
            }, save_dir / "per_grid_reversion_best.pt")

        if epoch % 10 == 0:
            phi = model.get_ar_phi().item()
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, phi={phi:.4f}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"Model saved to: {save_dir / 'per_grid_reversion_best.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
