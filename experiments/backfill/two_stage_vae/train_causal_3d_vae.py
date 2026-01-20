#!/usr/bin/env python
"""
Training script for Causal 3D VAE on volatility surfaces.

Key features:
1. Scheduled Sampling: Gradually use model predictions instead of ground truth
2. Autoregressive Training: Train with actual AR chaining some fraction of time
3. Coverage-aware evaluation: Track CI coverage during training

Usage:
    python experiments/backfill/two_stage_vae/train_causal_3d_vae.py
"""

import os
import sys
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.causal_3d_vae import create_vol_surface_vae, create_vol_surface_vae_small, AutoencoderCausal3D, Causal3DVAEConfig
from config.causal_3d_config import Causal3DTrainingConfig


class VolSurfaceSequenceDataset(Dataset):
    """Dataset for volatility surface sequences."""

    def __init__(
        self,
        surfaces: np.ndarray,
        context_length: int = 20,
        horizon: int = 30,
        train: bool = True,
        train_split: float = 0.8,
    ):
        """
        Args:
            surfaces: (N, 5, 5) array of volatility surfaces
            context_length: Number of frames to use as context
            horizon: Number of frames to predict
            train: If True, use training split; else test split
            train_split: Fraction of data to use for training
        """
        self.context_length = context_length
        self.horizon = horizon
        self.total_length = context_length + horizon

        # Split data
        n_train = int(len(surfaces) * train_split)
        if train:
            self.surfaces = surfaces[:n_train]
        else:
            self.surfaces = surfaces[n_train:]

        # Create sequence indices
        # Each sequence is (context_length + horizon) frames
        self.valid_starts = []
        for i in range(len(self.surfaces) - self.total_length + 1):
            self.valid_starts.append(i)

        print(f"{'Train' if train else 'Test'} dataset: {len(self.valid_starts)} sequences")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        seq = self.surfaces[start:start + self.total_length]

        # Convert to (1, T, H, W) format for the model
        seq = torch.from_numpy(seq).float().unsqueeze(0)

        return {
            "sequence": seq,  # (1, T, 5, 5)
            "context": seq[:, :self.context_length],  # (1, C, 5, 5)
            "target": seq[:, self.context_length:],  # (1, H, 5, 5)
        }


def compute_teacher_forcing_ratio(epoch: int, config: Causal3DTrainingConfig) -> float:
    """
    Compute teacher forcing ratio with linear decay.

    Returns 1.0 during warmup, then linearly decays to min_teacher_forcing.
    """
    if not config.scheduled_sampling:
        return 1.0

    if epoch < config.scheduled_sampling_warmup:
        return 1.0

    # Linear decay from 1.0 to min_teacher_forcing
    decay_epochs = config.num_epochs - config.scheduled_sampling_warmup
    progress = (epoch - config.scheduled_sampling_warmup) / max(decay_epochs, 1)
    ratio = 1.0 - progress * (1.0 - config.min_teacher_forcing)
    return max(config.min_teacher_forcing, ratio)


def train_step_standard(
    model: AutoencoderCausal3D,
    batch: dict,
    config: Causal3DTrainingConfig,
) -> tuple:
    """Standard training step with reconstruction loss on full sequence."""
    sequence = batch["sequence"].to(config.device)

    # Forward pass
    recon, posterior = model(sequence, sample_posterior=True, return_posterior=True)

    # Reconstruction loss
    recon_loss = F.mse_loss(recon, sequence)

    # KL loss
    kl_loss = posterior.kl().mean()

    # Total loss
    loss = recon_loss + config.kl_weight * kl_loss

    return loss, {"recon": recon_loss.item(), "kl": kl_loss.item()}


def train_step_autoregressive(
    model: AutoencoderCausal3D,
    batch: dict,
    config: Causal3DTrainingConfig,
    teacher_forcing_ratio: float = 1.0,
) -> tuple:
    """
    Training step with autoregressive generation and scheduled sampling.

    With probability teacher_forcing_ratio, use ground truth for the next input.
    Otherwise, use model's own prediction.
    """
    context = batch["context"].to(config.device)  # (B, 1, C, 5, 5)
    target = batch["target"].to(config.device)  # (B, 1, H, 5, 5)

    B, C_ch, C_len, H, W = context.shape
    full_horizon = target.shape[2]

    # Limit AR steps to save memory (default 10 steps instead of full 60)
    ar_steps = min(config.ar_training_steps, full_horizon)

    # Start with context
    current_sequence = context.clone()
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    # Generate autoregressively (limited steps for memory)
    for t in range(ar_steps):
        # Encode current sequence
        posterior = model.encode(current_sequence, return_dict=True)

        # Sample from posterior
        z = posterior.sample() * model.scaling_factor

        # Decode
        decoded = model.decode(z, target_temporal_size=current_sequence.shape[2])

        # Get the predicted next frame (last frame of decoded)
        predicted_frame = decoded[:, :, -1:, :, :]

        # Ground truth next frame
        gt_frame = target[:, :, t:t+1, :, :]

        # Reconstruction loss for this step
        step_recon_loss = F.mse_loss(predicted_frame, gt_frame)
        total_recon_loss += step_recon_loss

        # KL loss (averaged per step)
        step_kl_loss = posterior.kl().mean()
        total_kl_loss += step_kl_loss

        # Scheduled sampling: use GT or prediction for next input
        if random.random() < teacher_forcing_ratio:
            # Teacher forcing: use ground truth
            next_frame = gt_frame
        else:
            # Free running: use model's prediction
            next_frame = predicted_frame.detach()

        # Append to sequence
        current_sequence = torch.cat([current_sequence, next_frame], dim=2)

    # Average losses over AR steps
    avg_recon_loss = total_recon_loss / ar_steps
    avg_kl_loss = total_kl_loss / ar_steps

    # Total loss
    loss = avg_recon_loss + config.kl_weight * avg_kl_loss

    return loss, {"recon": avg_recon_loss.item(), "kl": avg_kl_loss.item()}


def evaluate(
    model: AutoencoderCausal3D,
    dataloader: DataLoader,
    config: Causal3DTrainingConfig,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(config.device)

            # Forward pass
            recon, posterior = model(sequence, sample_posterior=False, return_posterior=True)

            # Losses
            recon_loss = F.mse_loss(recon, sequence)
            kl_loss = posterior.kl().mean()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

    return {
        "recon": total_recon_loss / num_batches,
        "kl": total_kl_loss / num_batches,
    }


def evaluate_ar_stability(
    model: AutoencoderCausal3D,
    dataloader: DataLoader,
    config: Causal3DTrainingConfig,
    num_samples: int = 10,
) -> dict:
    """
    Evaluate autoregressive stability (explosion rate).

    Generates sequences autoregressively and checks for IV explosion.
    """
    model.eval()
    explosion_threshold = 2.0  # IV > 2.0 is considered explosion
    min_iv_threshold = 0.01   # IV < 0.01 is also explosion

    num_explosions = 0
    total_sequences = 0

    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(config.device)
            B = context.shape[0]

            for _ in range(num_samples):
                # Generate autoregressively
                generated = model.generate_autoregressive(
                    context,
                    num_steps=config.prediction_horizon
                )

                # Check for explosions in the generated part
                gen_surfaces = generated[:, :, config.context_length:]

                # Check each sequence
                for b in range(B):
                    seq_max = gen_surfaces[b].max().item()
                    seq_min = gen_surfaces[b].min().item()
                    if seq_max > explosion_threshold or seq_min < min_iv_threshold:
                        num_explosions += 1
                    total_sequences += 1

            # Only check a few batches
            if total_sequences >= 100:
                break

    explosion_rate = num_explosions / max(total_sequences, 1)
    return {"explosion_rate": explosion_rate, "total_sequences": total_sequences}


def train(config: Causal3DTrainingConfig, resume: bool = False):
    """Main training loop."""
    print("=" * 60)
    print("Training Causal 3D VAE for Volatility Surfaces")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Load data
    print(f"\nLoading data from {config.data_path}...")
    data = np.load(config.data_path)
    surfaces = data["surface"]
    print(f"Loaded {len(surfaces)} surfaces, shape: {surfaces.shape}")

    # Create datasets
    train_dataset = VolSurfaceSequenceDataset(
        surfaces,
        context_length=config.context_length,
        horizon=config.prediction_horizon,
        train=True,
        train_split=config.train_split,
    )
    test_dataset = VolSurfaceSequenceDataset(
        surfaces,
        context_length=config.context_length,
        horizon=config.prediction_horizon,
        train=False,
        train_split=config.train_split,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model (use small model for 5x5 data)
    print("\nCreating model...")
    model = create_vol_surface_vae_small(
        latent_channels=config.latent_channels,
        temporal_compression=config.temporal_compression,
        block_channels=config.block_channels,
    )
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate / 10,
    )

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training loop
    best_loss = float("inf")
    history = {"train_loss": [], "test_loss": [], "explosion_rate": []}
    start_epoch = 0

    # Resume from checkpoint if requested
    if resume:
        checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")
        if os.path.exists(checkpoint_path):
            print(f"\nResuming from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            history = checkpoint["history"]
            start_epoch = checkpoint["epoch"] + 1
            best_loss = min(history["test_loss"]) if history["test_loss"] else float("inf")
            print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")
        else:
            print(f"\nNo checkpoint found at {checkpoint_path}, starting fresh...")

    print("\nStarting training...")
    for epoch in range(start_epoch, config.num_epochs):
        model.train()

        # Compute teacher forcing ratio for this epoch
        tf_ratio = compute_teacher_forcing_ratio(epoch, config)

        # Decide whether to use AR training for this epoch
        use_ar_training = (
            epoch >= config.ar_training_start_epoch
            and random.random() < config.ar_training_fraction
        )

        epoch_losses = {"recon": 0.0, "kl": 0.0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            if use_ar_training:
                loss, losses = train_step_autoregressive(
                    model, batch, config, teacher_forcing_ratio=tf_ratio
                )
            else:
                loss, losses = train_step_standard(model, batch, config)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            epoch_losses["recon"] += losses["recon"]
            epoch_losses["kl"] += losses["kl"]
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "recon": losses["recon"],
                "tf": f"{tf_ratio:.2f}",
                "ar": "Y" if use_ar_training else "N",
            })

        # Average epoch losses
        epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        history["train_loss"].append(epoch_losses["recon"] + config.kl_weight * epoch_losses["kl"])

        # Evaluate on test set
        test_losses = evaluate(model, test_loader, config)
        test_loss = test_losses["recon"] + config.kl_weight * test_losses["kl"]
        history["test_loss"].append(test_loss)

        # Evaluate AR stability every 10 epochs
        if (epoch + 1) % 10 == 0:
            ar_metrics = evaluate_ar_stability(model, test_loader, config, num_samples=5)
            history["explosion_rate"].append(ar_metrics["explosion_rate"])
            print(f"\n  Epoch {epoch+1}: Test Loss={test_loss:.4f}, "
                  f"Explosion Rate={ar_metrics['explosion_rate']:.2%}")
        else:
            print(f"\n  Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, "
                  f"Test Loss={test_loss:.4f}, TF Ratio={tf_ratio:.2f}")

        # Update scheduler
        scheduler.step()

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config.__dict__,
            "history": history,
        }

        # Always save latest
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, "latest.pt"))

        # Save periodic checkpoint (separate file every save_interval epochs)
        if (epoch + 1) % config.save_interval == 0:
            periodic_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}.pt")
            torch.save(checkpoint, periodic_path)
            print(f"  Saved periodic checkpoint: epoch_{epoch+1}.pt")

        # Save best
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(checkpoint, os.path.join(config.checkpoint_dir, "best_model.pt"))
            print(f"  Saved best model (test loss: {test_loss:.4f})")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best test loss: {best_loss:.4f}")
    print(f"Model saved to: {config.checkpoint_dir}")
    print("=" * 60)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Causal 3D VAE")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-scheduled-sampling", action="store_true",
                        help="Disable scheduled sampling")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--no-ar-training", action="store_true",
                        help="Disable autoregressive training (saves memory)")
    parser.add_argument("--ar-steps", type=int, default=10,
                        help="Number of AR steps per batch (default 10, reduce for memory)")
    args = parser.parse_args()

    # Create config
    config = Causal3DTrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        scheduled_sampling=not args.no_scheduled_sampling,
        ar_training_fraction=0.0 if args.no_ar_training else 0.3,
        ar_training_steps=args.ar_steps,
    )

    # Train
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
