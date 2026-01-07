"""
Train Two-Stage CVAE autoencoder (Stage 1 only).

Goal: Train autoencoder to validate conditional variance capability
before deciding on VQ-style context.

Training Parameters:
- Epochs: 100
- Batch Size: 256 (for 3070 Ti)
- Learning Rate: 1e-4
- Context: 30 days, Horizon: 30 days
- Loss Mode: "horizon" (MSE only on horizon positions)
- KL Weight: 0.001 (weak to preserve variance)

Usage:
    python experiments/backfill/prior_encoder_ablation/train_two_stage_autoencoder.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage
from config.two_stage_config import TwoStageConfig


def create_sequences(surfaces, seq_len):
    """Create overlapping sequences from surface data."""
    n_sequences = len(surfaces) - seq_len + 1
    sequences = torch.stack([surfaces[i:i+seq_len] for i in range(n_sequences)])
    return sequences


def train_epoch(model, train_sequences, optimizer, batch_size, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    # Shuffle sequences
    indices = torch.randperm(len(train_sequences))

    pbar = tqdm(range(0, len(indices), batch_size), desc="Training", leave=False)
    for i in pbar:
        batch_idx = indices[i:i+batch_size]
        batch = train_sequences[batch_idx].to(device)

        losses = model.train_step_autoencoder({"surface": batch}, optimizer)

        total_loss += losses["loss"].item()
        total_recon += losses["re_surface"].item()
        total_kl += losses["kl_loss"].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "recon": f"{losses['re_surface'].item():.6f}",
            "kl": f"{losses['kl_loss'].item():.2f}"
        })

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
    }


def validate(model, val_sequences, batch_size, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_sequences), batch_size):
            batch = val_sequences[i:i+batch_size].to(device)
            losses = model.test_step({"surface": batch})

            total_loss += losses["loss"].item()
            total_recon += losses["re_surface"].item()
            total_kl += losses["kl_loss"].item()
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
    }


def main():
    print("=" * 70)
    print("Two-Stage CVAE Autoencoder Training (Stage 1)")
    print("=" * 70)

    # Override config for this experiment
    config = TwoStageConfig.get_model_config()

    # Training parameters
    batch_size = 256  # Larger batch for 3070 Ti
    n_epochs = 100
    learning_rate = 1e-4

    # Sequence parameters from config
    context_len = config["context_len"]  # 30
    horizon = config["horizon"]  # 30
    seq_len = context_len + horizon  # 60

    print(f"\nTraining Parameters:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Context Length: {context_len}")
    print(f"  Horizon: {horizon}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Loss Mode: {config.get('loss_mode', 'horizon')}")
    print(f"  KL Weight: {config['kl_weight']}")
    print(f"  z_logvar_floor: {config.get('z_logvar_floor', None)}")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = torch.tensor(data["surface"], dtype=torch.float32)
    print(f"  Total surfaces: {len(surfaces)}")

    # Create sequences
    print(f"\nCreating sequences (length={seq_len})...")
    all_sequences = create_sequences(surfaces, seq_len)
    print(f"  Total sequences: {len(all_sequences)}")

    # Train/val split (80/20)
    n_train = int(len(all_sequences) * 0.8)
    train_sequences = all_sequences[:n_train]
    val_sequences = all_sequences[n_train:]
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences: {len(val_sequences)}")

    # Build model
    print("\nBuilding model...")
    device = config["device"]
    model = CVAETwoStage(config)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Checkpoint path
    checkpoint_dir = Path(TwoStageConfig.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{TwoStageConfig.checkpoint_prefix}_best.pt"

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_val_loss = float('inf')
    history = {"train": [], "val": []}

    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(model, train_sequences, optimizer, batch_size, device)
        history["train"].append(train_metrics)

        # Validate
        val_metrics = validate(model, val_sequences, batch_size, device)
        history["val"].append(val_metrics)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} (recon: {train_metrics['recon']:.6f}, kl: {train_metrics['kl']:.2f}) | "
              f"Val Loss: {val_metrics['loss']:.4f} (recon: {val_metrics['recon']:.6f}, kl: {val_metrics['kl']:.2f})")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "train_loss": train_metrics["loss"],
                "history": history,
            }, checkpoint_path)
            print(f"         → Saved best model (val_loss: {val_metrics['loss']:.6f})")

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Save training history
    history_path = checkpoint_dir / f"{TwoStageConfig.checkpoint_prefix}_history.npz"
    np.savez(history_path,
             train_loss=[h["loss"] for h in history["train"]],
             train_recon=[h["recon"] for h in history["train"]],
             train_kl=[h["kl"] for h in history["train"]],
             val_loss=[h["loss"] for h in history["val"]],
             val_recon=[h["recon"] for h in history["val"]],
             val_kl=[h["kl"] for h in history["val"]])
    print(f"  History: {history_path}")


if __name__ == "__main__":
    main()
