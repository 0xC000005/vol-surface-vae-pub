"""
Context=60 Conditional Prior Network Training (Latent12 V3)

This script trains a VAE with learnable conditional prior p(z|context) instead of
fixed N(0,1), eliminating VAE prior mismatch and systematic bias.

Key Innovation:
--------------
**Standard VAE:**
- Training KL: KL(q(z|context, target) || N(0,1))
- Inference: z ~ N(0,1)  [mismatch causes negative bias!]

**Conditional Prior VAE (This Script):**
- Training KL: KL(q(z|context, target) || p(z|context))
- Inference: z ~ p(z|context)  [perfect match, no bias!]

Expected Benefits:
-----------------
1. Eliminates systematic negative bias (no GMM post-processing needed)
2. Context-adaptive uncertainty (crisis → wider CIs automatically)
3. Better calibration (model learns regime-specific uncertainty)
4. Theoretically principled (proper conditional VAE)

Training Schedule:
-----------------
- Phase 1 (epochs 0-200): Teacher forcing (H=1)
- Phase 2 (epochs 201-600): Multi-horizon [1,7,14,30,60,90]

Usage:
------
# Train from scratch
python experiments/backfill/context60/train_backfill_context60_latent12_v3_conditional_prior.py

# Resume from checkpoint
python experiments/backfill/context60/train_backfill_context60_latent12_v3_conditional_prior.py \\
    --resume_from models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_phase1_ep199.pt

Output:
-------
- models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_phase1_ep199.pt
- models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_phase2_ep599.pt
- models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_best.pt
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import argparse
from collections import defaultdict
from pathlib import Path

from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_conditional_prior import CVAEMemRandConditionalPrior
from vae.conditional_prior_network import kl_divergence_gaussians
from vae.utils import set_seeds, model_eval
from config.backfill_context60_config_latent12_v3_conditional_prior import (
    BackfillContext60ConfigLatent12V3ConditionalPrior
)
from torch.cuda.amp import autocast, GradScaler

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Train Context60 Latent12 V3 model with conditional prior'
)
parser.add_argument('--resume_from', type=str, default=None,
                   help='Path to checkpoint to resume from')
args = parser.parse_args()


# ==============================================================================
# Dataset and Dataloader Creation
# ==============================================================================

def create_datasets(vol_data, ex_data, seq_len_range):
    """Create train/valid datasets with specified sequence length range."""
    min_len, max_len = seq_len_range

    # 80/20 train/valid split
    split_idx = int(0.8 * len(vol_data))

    train_dataset = VolSurfaceDataSetRand(
        (vol_data[:split_idx], ex_data[:split_idx]),
        min_seq_len=min_len,
        max_seq_len=max_len,
        dtype=torch.float32
    )

    valid_dataset = VolSurfaceDataSetRand(
        (vol_data[split_idx:], ex_data[split_idx:]),
        min_seq_len=min_len,
        max_seq_len=max_len,
        dtype=torch.float32
    )

    return train_dataset, valid_dataset


def create_dataloaders(train_dataset, valid_dataset, batch_size, valid_batch_size):
    """Create dataloaders with custom batch sampler."""
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=CustomBatchSampler(
            train_dataset,
            batch_size,
            train_dataset.seq_lens[0]
        ),
        pin_memory=True,
        num_workers=2,          # Parallel data loading
        persistent_workers=True, # Reuse workers between epochs
        prefetch_factor=2       # Prefetch 2 batches per worker
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=CustomBatchSampler(
            valid_dataset,
            valid_batch_size,
            valid_dataset.seq_lens[0]
        ),
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, valid_loader


# ==============================================================================
# Training Utilities
# ==============================================================================

def train_one_epoch_teacher_forcing(model, optimizer, train_loader, device, kl_weight, scaler):
    """
    Train one epoch with teacher forcing (horizon=1).

    Uses model's built-in train_step() which handles conditional prior KL loss.

    Returns:
        dict with training metrics
    """
    model.train()
    metrics = defaultdict(float)
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # train_step handles everything including conditional prior KL
        # Autocast is handled internally by train_step with scaler
        losses = model.train_step(batch, optimizer, scaler=scaler)

        # Update metrics
        for k, v in losses.items():
            metrics[k] += v.item() if hasattr(v, 'item') else v
        num_batches += 1

        pbar.set_postfix({'loss': f"{losses['loss'].item():.6f}"})

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return dict(metrics)


def train_one_epoch_multihorizon(model, optimizer, train_loader, device, kl_weight,
                                 horizons, weights, scaler):
    """
    Train one epoch with multiple horizons.

    Uses model's built-in train_step_multihorizon() which handles conditional prior KL loss.

    Args:
        horizons: List of horizons [1, 7, 14, 30, 60, 90]
        weights: Dict mapping horizon -> weight (ignored, uniform weighting used)

    Returns:
        dict with training metrics
    """
    model.train()
    metrics = defaultdict(float)
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # train_step_multihorizon handles everything including conditional prior KL
        # Autocast is handled internally by train_step_multihorizon with scaler
        losses = model.train_step_multihorizon(batch, optimizer, horizons=horizons, scaler=scaler)

        # Update metrics
        for k, v in losses.items():
            if k == 'horizon_losses':
                continue  # Skip horizon-specific losses
            metrics[k] += v.item() if hasattr(v, 'item') else v
        num_batches += 1

        pbar.set_postfix({'loss': f"{losses['loss'].item():.6f}"})

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return dict(metrics)


def validate(model, valid_loader, device, kl_weight, horizon=1):
    """
    Validate model.

    Uses model_eval utility which works with train_step.

    Returns:
        dict with validation metrics
    """
    return model_eval(model, valid_loader)


# ==============================================================================
# Main Training Script
# ==============================================================================

def main():
    print("=" * 80)
    print("CONTEXT=60 CONDITIONAL PRIOR NETWORK TRAINING (V3)")
    print("=" * 80)
    print()

    # Load config
    cfg = BackfillContext60ConfigLatent12V3ConditionalPrior
    cfg.summary()

    # Set random seeds
    set_seeds(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Enable TF32 for faster training on Ampere GPUs
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for CUDA")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data['surface'][cfg.train_start_idx:cfg.train_end_idx]
    ex_data = np.stack([
        data['ret'][cfg.train_start_idx:cfg.train_end_idx],
        data['skews'][cfg.train_start_idx:cfg.train_end_idx],
        data['slopes'][cfg.train_start_idx:cfg.train_end_idx]
    ], axis=1)

    print(f"Loaded {len(surfaces)} days of data")

    # Initialize model
    print("\nInitializing model...")
    model_config = {
        "seq_len": 200,  # max sequence length
        "feat_dim": (5, 5),
        "latent_dim": cfg.latent_dim,
        "kl_weight": cfg.kl_weight,
        "re_feat_weight": cfg.re_feat_weight,
        "surface_hidden": cfg.surface_hidden,
        "ctx_surface_hidden": cfg.surface_hidden,
        "ex_feats_dim": 3,
        "ex_feats_hidden": None,
        "ctx_ex_feats_hidden": None,
        "mem_type": "lstm",
        "mem_hidden": cfg.mem_hidden,
        "mem_layers": cfg.mem_layers,
        "mem_dropout": cfg.mem_dropout,
        "interaction_layers": 2,
        "use_dense_surface": False,
        "compress_context": True,
        "ex_loss_on_ret_only": cfg.ex_loss_on_ret_only,
        "ex_feats_loss_type": cfg.ex_feats_loss_type,
        "device": device,
        "num_quantiles": cfg.num_quantiles,
        "quantiles": cfg.quantiles,
        "quantile_loss_weights": cfg.quantile_loss_weights,
        "horizon": 90,  # max horizon
        "context_len": cfg.context_len,
        # CONDITIONAL PRIOR (KEY PARAMETER!)
        "use_conditional_prior": cfg.use_conditional_prior,
    }

    model = CVAEMemRandConditionalPrior(model_config)
    model = model.to(device)

    # Compile model for faster training (PyTorch 2.0+)
    # Use dynamic=True to handle variable sequence lengths without recompilation
    model = torch.compile(model, mode="default", dynamic=True)
    print("✓ Model compiled with torch.compile() (mode=default, dynamic=True)")

    print(f"✓ Model initialized")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    if model.use_conditional_prior:
        print(f"  Prior network parameters: {sum(p.numel() for p in model.prior_network.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Mixed precision training
    scaler = GradScaler()
    print("✓ Mixed precision (AMP) enabled")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from is not None:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")

    # Create output directory
    output_dir = cfg.checkpoint_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = Path(output_dir) / "context60_latent12_v3_conditional_prior_training_log.txt"

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, cfg.total_epochs):
        phase_info = cfg.get_phase_info(epoch)
        phase_num = phase_info['phase_num']
        horizon = phase_info['horizon']

        print(f"\nEpoch {epoch}/{cfg.total_epochs-1} - Phase {phase_num}: {phase_info['phase_name']}")

        # Create datasets for current phase
        train_dataset, valid_dataset = create_datasets(
            surfaces, ex_data, phase_info['seq_len']
        )
        train_loader, valid_loader = create_dataloaders(
            train_dataset, valid_dataset, cfg.batch_size, cfg.valid_batch_size
        )

        # Set model horizon for current phase
        if phase_num == 1:
            model.horizon = 1  # Teacher forcing
            train_metrics = train_one_epoch_teacher_forcing(
                model, optimizer, train_loader, device, cfg.kl_weight, scaler
            )
        else:
            model.horizon = max(cfg.phase2_horizons)  # Max horizon for Phase 2
            train_metrics = train_one_epoch_multihorizon(
                model, optimizer, train_loader, device, cfg.kl_weight,
                cfg.phase2_horizons, cfg.phase2_weights, scaler
            )

        # Validate
        val_metrics = validate(model, valid_loader, device, cfg.kl_weight, horizon=1)

        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.6f} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"KL: {val_metrics['kl_loss']:.6f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = Path(output_dir) / f"{cfg.checkpoint_prefix}_best.pt"
            model.save_model(str(best_path))
            print(f"✓ Saved best model (val_loss={best_val_loss:.6f})")

        # Save checkpoint every 50 epochs (both phases)
        if (epoch + 1) % 50 == 0:
            checkpoint_path = Path(output_dir) / f"{cfg.checkpoint_prefix}_ep{epoch}.pt"
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config,
                "epoch": epoch,
                "phase": phase_num,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Periodic checkpoint saved at epoch {epoch}")

        # Save phase checkpoints
        if epoch == cfg.phase1_end - 1 or epoch == cfg.phase2_end - 1:
            checkpoint_path = Path(output_dir) / cfg.get_checkpoint_name(epoch)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config,
                "epoch": epoch,
                "phase": phase_num,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved Phase {phase_num} checkpoint")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
