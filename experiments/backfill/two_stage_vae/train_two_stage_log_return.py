"""
Train Two-Stage CVAE on Log-Returns (Stage 1 only).

This is an enhancement of the original two-stage model that trains on
log-returns instead of IV levels. This solves the horizon variance issue
where CI coverage was ~50% instead of 90%.

Key Changes from train_two_stage_autoencoder.py:
1. Transform IV surfaces to log-returns before training
2. Model learns to predict/reconstruct log-returns (stationary)
3. Model's constant z_logvar is now CORRECT for stationary data
4. Horizon effect emerges naturally from summing predictions

Why Log-Returns Work:
- Log-returns have constant variance (stationary)
- At inference: sum log-returns, then exp() back to IV levels
- CI width grows naturally with √H
- exp() guarantees positive IV values

Usage:
    python experiments/backfill/prior_encoder_ablation/train_two_stage_log_return.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStage
from config.two_stage_config import TwoStageConfig


def to_log_returns(surfaces):
    """
    Transform IV surfaces to log-returns.

    Args:
        surfaces: (N, 5, 5) array of IV surfaces

    Returns:
        log_returns: (N-1, 5, 5) array of log-returns
        log_surfaces: (N, 5, 5) array of log(IV) for reconstruction
    """
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]
    return log_returns, log_surfaces


def from_log_returns(log_returns_seq, initial_log_surface):
    """
    Transform log-returns back to IV surfaces.

    Args:
        log_returns_seq: (T, 5, 5) sequence of log-returns
        initial_log_surface: (5, 5) starting log(IV) surface

    Returns:
        surfaces: (T+1, 5, 5) IV surfaces (always positive!)
    """
    log_cumsum = initial_log_surface + np.cumsum(log_returns_seq, axis=0)
    # Prepend initial surface
    log_surfaces = np.concatenate([initial_log_surface[None], log_cumsum], axis=0)
    return np.exp(log_surfaces)  # Always positive!


def create_sequences(data, seq_len):
    """Create overlapping sequences from data."""
    n_sequences = len(data) - seq_len + 1
    sequences = torch.stack([data[i:i+seq_len] for i in range(n_sequences)])
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
    print("Two-Stage CVAE Log-Return Training (Stage 1)")
    print("=" * 70)
    print()
    print("Training on LOG-RETURNS instead of IV levels.")
    print("This solves the horizon variance issue (CI coverage 50% → 90%).")
    print()

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

    print(f"Training Parameters:")
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
    surfaces = data["surface"]  # (N, 5, 5)
    print(f"  Total surfaces: {len(surfaces)}")
    print(f"  IV range: [{surfaces.min():.4f}, {surfaces.max():.4f}]")

    # Transform to log-returns
    print("\nTransforming to log-returns...")
    log_returns, log_surfaces = to_log_returns(surfaces)
    print(f"  Log-returns shape: {log_returns.shape}")
    print(f"  Log-return range: [{log_returns.min():.4f}, {log_returns.max():.4f}]")
    print(f"  Log-return mean: {log_returns.mean():.6f}")
    print(f"  Log-return std: {log_returns.std():.4f}")

    # Convert to tensor
    log_returns_tensor = torch.tensor(log_returns, dtype=torch.float32)

    # Create sequences from log-returns (not surfaces!)
    print(f"\nCreating sequences (length={seq_len})...")
    all_sequences = create_sequences(log_returns_tensor, seq_len)
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
    checkpoint_path = checkpoint_dir / "two_stage_log_return_best.pt"

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
                # Log-return specific metadata
                "use_log_returns": True,
                "log_return_stats": {
                    "mean": float(log_returns.mean()),
                    "std": float(log_returns.std()),
                    "min": float(log_returns.min()),
                    "max": float(log_returns.max()),
                },
            }, checkpoint_path)
            print(f"         → Saved best model (val_loss: {val_metrics['loss']:.6f})")

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print()
    print("Next steps:")
    print("  1. Run validation: python experiments/backfill/prior_encoder_ablation/validate_two_stage_log_return.py")
    print("  2. Check CI coverage improves across horizons")
    print("  3. Verify transformed-back IV values are always positive")

    # Save training history
    history_path = checkpoint_dir / "two_stage_log_return_history.npz"
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
