#!/usr/bin/env python
"""
Training script for DDPM POC on volatility surfaces.

This validates that diffusion models can achieve better CI coverage than VAE.

Key metrics to track:
1. Training loss (MSE on noise prediction)
2. Sample diversity (std across generated samples)
3. CI coverage at different levels (target: match nominal)

Usage:
    python experiments/backfill/diffusion_poc/train_ddpm_poc.py
    python experiments/backfill/diffusion_poc/train_ddpm_poc.py --epochs 100
    python experiments/backfill/diffusion_poc/train_ddpm_poc.py --fast  # Quick test
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.simple_denoiser import SimpleDenoiser3D, ConditionalDDPM, DenoiserConfig
from diffusion.ddpm_scheduler import DDPMScheduler
from experiments.backfill.diffusion_poc.config_ddpm_poc import (
    DDPMPOCConfig,
    get_default_config,
    get_fast_test_config,
)

# IV normalization constants (maps [IV_MIN, IV_MAX] → [-1, 1])
# Following standard DDPM practice from Ho et al. 2020
# Empirical range from SPX data: min=0.01, max=0.9957
# Use slightly wider range [0.0, 1.0] to handle edge cases
IV_MIN = 0.0
IV_MAX = 1.0


def normalize_iv(iv: torch.Tensor) -> torch.Tensor:
    """Normalize IV from [0.05, 1.0] to [-1, 1]."""
    return 2.0 * (iv - IV_MIN) / (IV_MAX - IV_MIN) - 1.0


def denormalize_iv(iv_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize IV from [-1, 1] to [0.05, 1.0]."""
    return (iv_norm + 1.0) / 2.0 * (IV_MAX - IV_MIN) + IV_MIN


class VolSurfaceDataset(Dataset):
    """Dataset for volatility surface sequences (history -> future pairs)."""

    def __init__(
        self,
        surfaces: np.ndarray,
        history_len: int,
        future_len: int,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        """
        Args:
            surfaces: (N, 5, 5) array of volatility surfaces
            history_len: Number of frames for conditioning
            future_len: Number of frames to predict
            start_idx: Start index in surfaces array
            end_idx: End index (exclusive) in surfaces array
        """
        self.history_len = history_len
        self.future_len = future_len
        self.total_len = history_len + future_len

        # Get subset of data
        end_idx = end_idx or len(surfaces)
        self.surfaces = surfaces[start_idx:end_idx]

        # Create valid sequence starts
        self.valid_starts = list(range(len(self.surfaces) - self.total_len + 1))

        print(f"Dataset: {len(self.valid_starts)} sequences from indices {start_idx}:{end_idx}")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        seq = self.surfaces[start:start + self.total_len]

        # Convert to tensors
        history = torch.from_numpy(seq[:self.history_len]).float()  # (T_hist, 5, 5)
        future = torch.from_numpy(seq[self.history_len:]).float()  # (T_fut, 5, 5)

        # Normalize to [-1, 1] (standard DDPM practice)
        history = normalize_iv(history)
        future = normalize_iv(future)

        return {"history": history, "future": future}


def compute_ci_coverage(
    model: ConditionalDDPM,
    dataloader: DataLoader,
    n_samples: int = 50,
    device: str = 'cpu',
    max_batches: int = 20,
    sampler: str = 'ddpm',
    n_inference_steps: int = 20,
) -> dict:
    """
    Compute CI coverage on validation data.

    Args:
        model: Trained DDPM model
        dataloader: Validation dataloader
        n_samples: Number of samples per history
        device: Device to use
        max_batches: Maximum batches to evaluate (for speed)
        sampler: 'ddpm' (default, all steps) or 'ddim' (fast, skip steps)
        n_inference_steps: Number of steps for DDIM (default: 20)

    Returns:
        dict with coverage metrics
    """
    model.eval()

    all_coverages = {0.5: [], 0.8: [], 0.9: [], 0.95: []}
    all_diversity = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)  # (B, T_hist, 5, 5)
            future_gt = batch["future"].to(device)  # (B, T_fut, 5, 5)

            # Generate samples
            samples = model.sample(
                history,
                n_samples=n_samples,
                sampler=sampler,
                n_inference_steps=n_inference_steps,
            )  # (B, n_samples, T_fut, 5, 5)

            # Compute coverage for each CI level
            for level in all_coverages.keys():
                alpha = (1 - level) / 2
                lower = torch.quantile(samples, alpha, dim=1)
                upper = torch.quantile(samples, 1 - alpha, dim=1)
                covered = (future_gt >= lower) & (future_gt <= upper)
                coverage = covered.float().mean().item()
                all_coverages[level].append(coverage)

            # Sample diversity
            diversity = samples.std(dim=1).mean().item()
            all_diversity.append(diversity)

    model.train()

    return {
        "coverage_50": np.mean(all_coverages[0.5]),
        "coverage_80": np.mean(all_coverages[0.8]),
        "coverage_90": np.mean(all_coverages[0.9]),
        "coverage_95": np.mean(all_coverages[0.95]),
        "sample_diversity": np.mean(all_diversity),
    }


def train_epoch(
    model: ConditionalDDPM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        history = batch["history"].to(device)
        future = batch["future"].to(device)

        # Forward pass
        result = model(history, future)
        loss = result["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": loss.item()})

    if scheduler is not None:
        scheduler.step()

    return {"loss": total_loss / n_batches}


def validate(
    model: ConditionalDDPM,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Compute validation loss."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            result = model(history, future)
            total_loss += result["loss"].item()
            n_batches += 1

    return {"val_loss": total_loss / n_batches}


def main():
    parser = argparse.ArgumentParser(description="Train DDPM POC on volatility surfaces")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--fast", action="store_true", help="Use fast test config")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate CI coverage every N epochs")
    parser.add_argument("--n_eval_samples", type=int, default=50, help="Samples per history for CI eval")
    parser.add_argument("--noise_schedule", type=str, default=None,
                        choices=["uniform", "independent"],
                        help="Noise schedule: 'uniform' (standard DDPM) or 'independent' (Diffusion Forcing)")
    args = parser.parse_args()

    # Get config
    config = get_fast_test_config() if args.fast else get_default_config()

    # Override with command line args
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.device:
        config.device = args.device
    if args.noise_schedule:
        config.noise_schedule = args.noise_schedule

    # Auto-detect device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = "cpu"

    print("=" * 60)
    print("DDPM POC Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"History: {config.history_len} days -> Future: {config.future_len} days")
    print(f"Diffusion steps: {config.n_steps}")
    print(f"Noise schedule: {config.noise_schedule}" +
          (" (Diffusion Forcing)" if config.noise_schedule == "independent" else " (standard DDPM)"))
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}, LR: {config.lr}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data = np.load(config.data_path)
    surfaces = data["surface"]
    print(f"Loaded {len(surfaces)} surfaces with shape {surfaces.shape}")

    # Create datasets
    train_dataset = VolSurfaceDataset(
        surfaces,
        config.history_len,
        config.future_len,
        start_idx=0,
        end_idx=config.train_end,
    )
    val_dataset = VolSurfaceDataset(
        surfaces,
        config.history_len,
        config.future_len,
        start_idx=config.val_start,
        end_idx=config.val_end,
    )
    test_dataset = VolSurfaceDataset(
        surfaces,
        config.history_len,
        config.future_len,
        start_idx=config.test_start,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Create model
    print("\nCreating model...")
    denoiser_config = DenoiserConfig(
        history_len=config.history_len,
        future_len=config.future_len,
        surface_h=config.surface_h,
        surface_w=config.surface_w,
        base_channels=config.base_channels,
        n_res_blocks=config.n_res_blocks,
        condition_dim=config.condition_dim,
        time_embed_dim=config.time_embed_dim,
        groups=config.groups,
        dropout=config.dropout,
        n_steps=config.n_steps,
        noise_schedule=config.noise_schedule,
    )

    model = ConditionalDDPM(
        denoiser_config,
        scheduler_config={"schedule": config.schedule, "device": config.device},
    )
    model = model.to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr / 10,
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    best_coverage_90 = 0.0

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            config.device, config.grad_clip,
        )

        # Validate
        val_metrics = validate(model, val_loader, config.device)

        # Log
        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['val_loss']:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Evaluate CI coverage periodically
        if epoch % args.eval_every == 0 or epoch == config.epochs:
            print("  Evaluating CI coverage (DDIM)...")
            coverage_metrics = compute_ci_coverage(
                model, val_loader,
                n_samples=args.n_eval_samples,
                device=config.device,
                max_batches=10,
                sampler='ddim',
                n_inference_steps=20,
            )
            print(
                f"  Coverage: 50%={coverage_metrics['coverage_50']:.1%}, "
                f"80%={coverage_metrics['coverage_80']:.1%}, "
                f"90%={coverage_metrics['coverage_90']:.1%}, "
                f"95%={coverage_metrics['coverage_95']:.1%} | "
                f"Diversity: {coverage_metrics['sample_diversity']:.4f}"
            )

            # Save best by coverage
            if coverage_metrics['coverage_90'] > best_coverage_90:
                best_coverage_90 = coverage_metrics['coverage_90']
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": denoiser_config,
                    "metrics": {**train_metrics, **val_metrics, **coverage_metrics},
                }, f"{config.output_dir}/best_coverage_model.pt")
                print(f"  Saved best coverage model (90% CI: {best_coverage_90:.1%})")

        # Save best by validation loss
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": denoiser_config,
                "metrics": {**train_metrics, **val_metrics},
            }, f"{config.output_dir}/best_model.pt")

        # Regular checkpoint
        if epoch % config.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": denoiser_config,
            }, f"{config.output_dir}/checkpoint_epoch_{epoch}.pt")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
    )

    test_val = validate(model, test_loader, config.device)
    print(f"Test Loss: {test_val['val_loss']:.6f}")

    test_coverage = compute_ci_coverage(
        model, test_loader,
        n_samples=args.n_eval_samples,
        device=config.device,
        max_batches=20,
        sampler='ddim',
        n_inference_steps=20,
    )
    print(
        f"Test Coverage (DDIM): 50%={test_coverage['coverage_50']:.1%}, "
        f"80%={test_coverage['coverage_80']:.1%}, "
        f"90%={test_coverage['coverage_90']:.1%}, "
        f"95%={test_coverage['coverage_95']:.1%}"
    )
    print(f"Sample Diversity: {test_coverage['sample_diversity']:.4f}")

    # Save final model
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": denoiser_config,
        "test_metrics": {**test_val, **test_coverage},
    }, f"{config.output_dir}/final_model.pt")

    print(f"\nModels saved to: {config.output_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main()
