"""
Prior Encoder Ablation Training Script

Trains three variants of the VAE model for comparing prior network architectures:

1. **Baseline**: CVAEFullCovPrior (current design)
   - Prior input: context_summary (B, 12)
   - Gradient flow: Confounded (context encoder gets gradients from both recon and KL)

2. **Prior Encoder Diagonal**: CVAEWithPriorEncoderDiagonal
   - Prior input: raw_context (B, 60, 5, 5)
   - Output: Per-timestep (mu, log_var)
   - Gradient flow: Clean (context encoder from recon only, prior from KL only)

3. **Prior Encoder Full Cov**: CVAEWithPriorEncoderFullCov
   - Prior input: raw_context (B, 60, 5, 5)
   - Output: mu + AR(1) covariance
   - Gradient flow: Clean (context encoder from recon only, prior from KL only)

Usage:
------
# Train baseline
python experiments/backfill/prior_encoder_ablation/train_ablation.py --variant baseline

# Train prior encoder diagonal
python experiments/backfill/prior_encoder_ablation/train_ablation.py --variant prior_encoder_diagonal

# Train prior encoder full cov
python experiments/backfill/prior_encoder_ablation/train_ablation.py --variant prior_encoder_full_cov

# Resume from checkpoint
python experiments/backfill/prior_encoder_ablation/train_ablation.py \\
    --variant baseline --resume_from models/backfill/prior_encoder_ablation/baseline/checkpoints/...pt

Output:
-------
- models/backfill/prior_encoder_ablation/{variant}/checkpoints/*_ep99.pt
- models/backfill/prior_encoder_ablation/{variant}/checkpoints/*_ep399.pt
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
from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.cvae_prior_encoder import CVAEWithPriorEncoderDiagonal, CVAEWithPriorEncoderFullCov
from vae.utils import set_seeds, model_eval
from config.prior_encoder_ablation_config import PriorEncoderAblationConfig

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Train Prior Encoder Ablation variants'
)
parser.add_argument('--variant', type=str, required=True,
                   choices=['baseline', 'prior_encoder_diagonal', 'prior_encoder_full_cov'],
                   help='Which variant to train')
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


def create_dataloaders(train_dataset, valid_dataset, batch_size):
    """Create dataloaders with custom batch sampler."""
    train_sampler = CustomBatchSampler(
        train_dataset.lengths,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, valid_loader


# ==============================================================================
# Training Functions
# ==============================================================================

def train_one_epoch_teacher_forcing(model, optimizer, train_loader, device, kl_weight, scaler=None):
    """Train one epoch with teacher forcing (H=1)."""
    model.train()
    epoch_losses = defaultdict(float)
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        losses = model.train_step(batch, optimizer, scaler=scaler)

        # Accumulate losses
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                epoch_losses[key] += value.item()
            else:
                epoch_losses[key] += value

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses["loss"].item():.6f}',
            'recon': f'{losses["reconstruction_loss"].item():.6f}',
            'kl': f'{losses["kl_loss"].item():.6f}'
        })

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return dict(epoch_losses)


def train_one_epoch_multihorizon(model, optimizer, train_loader, device, kl_weight,
                                 horizons, weights, scaler=None):
    """Train one epoch with multi-horizon (H in [1, 30, 60, 90])."""
    model.train()
    epoch_losses = defaultdict(float)
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        losses = model.train_step_multihorizon(batch, optimizer, horizons=horizons, scaler=scaler)

        # Accumulate losses
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                epoch_losses[key] += value.item()
            elif isinstance(value, dict):
                # horizon_losses is a dict
                for h_key, h_val in value.items():
                    epoch_losses[f"horizon_{h_key}"] += h_val
            else:
                epoch_losses[key] += value

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses["loss"].item():.6f}',
            'recon': f'{losses["reconstruction_loss"].item():.6f}',
            'kl': f'{losses["kl_loss"].item():.6f}'
        })

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return dict(epoch_losses)


def validate(model, valid_loader, device, kl_weight):
    """Validate model on validation set."""
    model.eval()
    epoch_losses = defaultdict(float)
    num_batches = 0

    with torch.no_grad():
        for batch in valid_loader:
            # Validation always uses teacher forcing (H=1)
            losses = model.train_step(batch, None, scaler=None)  # No optimizer for validation

            # Accumulate losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.item()
                else:
                    epoch_losses[key] += value

            num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return dict(epoch_losses)


# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    # Set random seeds for reproducibility
    set_seeds(42)

    # Set variant-specific config
    PriorEncoderAblationConfig.variant = args.variant
    cfg = PriorEncoderAblationConfig

    # Print configuration summary
    cfg.summary()

    # Create checkpoint directory
    checkpoint_dir = Path(cfg.get_checkpoint_dir())
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    vol_data = data["vol_surface"]
    ex_data = data["ex_data"]

    # Filter to training period
    vol_data = vol_data[cfg.train_start_idx:cfg.train_end_idx]
    ex_data = ex_data[cfg.train_start_idx:cfg.train_end_idx]
    print(f"✓ Data loaded: {len(vol_data)} days")

    # Create datasets and dataloaders (Phase 1)
    print("\nCreating datasets...")
    train_dataset, valid_dataset = create_datasets(
        vol_data, ex_data,
        seq_len_range=(cfg.phase1_seq_len, cfg.phase1_seq_len)
    )
    train_loader, valid_loader = create_dataloaders(
        train_dataset, valid_dataset, cfg.batch_size
    )
    print(f"✓ Datasets created: train={len(train_dataset)}, valid={len(valid_dataset)}")

    # Build model config
    model_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "feat_dim": (5, 5),
        "context_len": cfg.context_len,
        "horizon": cfg.horizon,
        "latent_dim": cfg.latent_dim,
        "ex_feats_dim": cfg.ex_feats_dim,
        "kl_weight": cfg.kl_weight,
        "ex_loss_on_ret_only": cfg.ex_loss_on_ret_only,
        "re_feat_weight": cfg.re_feat_weight,
        "use_dense_surface": cfg.use_dense_surface,
        "surface_hidden": cfg.surface_hidden,
        "ctx_surface_hidden": cfg.ctx_surface_hidden,
        "ex_feats_hidden": cfg.ex_feats_hidden,
        "ctx_ex_feats_hidden": cfg.ctx_ex_feats_hidden,
        "interaction_layers": cfg.interaction_layers,
        "mem_type": cfg.mem_type,
        "mem_hidden": cfg.mem_hidden,
        "mem_layers": cfg.mem_layers,
        "mem_dropout": cfg.mem_dropout,
        "padding": cfg.padding,
        "compress_context": cfg.compress_context,
        "max_horizon": cfg.max_horizon,
        # Prior-specific settings
        "prior_surface_hidden": cfg.prior_surface_hidden,
        "prior_mem_hidden": cfg.prior_mem_hidden,
        "prior_mem_layers": cfg.prior_mem_layers,
        "prior_dropout": cfg.prior_dropout,
        "prior_pos_dim": cfg.prior_pos_dim,
        # Full cov prior settings
        "full_cov_pos_dim": cfg.full_cov_pos_dim,
        "full_cov_hidden_dims": cfg.full_cov_hidden_dims,
        "full_cov_dropout": cfg.full_cov_dropout,
        "full_cov_init_phi": cfg.full_cov_init_phi,
        "full_cov_init_sigma_sq": cfg.full_cov_init_sigma_sq,
        # LSTM prior network settings
        "mean_network_type": cfg.mean_network_type if hasattr(cfg, 'mean_network_type') else "mlp",
        "use_position_encoding": cfg.use_position_encoding if hasattr(cfg, 'use_position_encoding') else None,
        "rnn_hidden_dim": cfg.rnn_hidden_dim if hasattr(cfg, 'rnn_hidden_dim') else 32,
        "rnn_num_layers": cfg.rnn_num_layers if hasattr(cfg, 'rnn_num_layers') else 1,
    }

    # Initialize model based on variant
    print("\nInitializing model...")
    if args.variant == "baseline":
        model = CVAEFullCovPrior(model_config)
    elif args.variant == "prior_encoder_diagonal":
        model = CVAEWithPriorEncoderDiagonal(model_config)
    elif args.variant == "prior_encoder_full_cov":
        model = CVAEWithPriorEncoderFullCov(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # Print prior-specific info
    if args.variant == "baseline":
        print(f"    Initial φ: {model.full_cov_prior.get_phi().item():.4f}")
        print(f"    Initial σ²: {model.full_cov_prior.get_sigma_sq().item():.4f}")
    elif args.variant == "prior_encoder_full_cov":
        print(f"    Initial φ: {model.prior_encoder.get_phi().item():.4f}")
        print(f"    Initial σ²: {model.prior_encoder.get_sigma_sq().item():.4f}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Mixed precision training (BFloat16 autocast in model, no scaler needed)
    scaler = None  # GradScaler not needed for BF16 (same dynamic range as FP32)
    print("✓ Mixed precision (BFloat16) enabled - no gradient scaling needed")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from is not None:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_weights(dict_to_load=checkpoint)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✓ Resumed from epoch {checkpoint.get('epoch', 0)}")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(start_epoch, cfg.total_epochs):
        epoch_start_time = time.time()

        # Determine phase
        if epoch < cfg.phase1_end:
            phase = 1
            phase_name = "Teacher Forcing"
            print(f"\nEpoch {epoch}/{cfg.total_epochs} - Phase 1: {phase_name} (H=1)")

            # Train
            train_metrics = train_one_epoch_teacher_forcing(
                model, optimizer, train_loader, model.device, cfg.kl_weight, scaler
            )

            # Validate
            val_metrics = validate(model, valid_loader, model.device, cfg.kl_weight)

        else:
            phase = 2
            phase_name = "Multi-Horizon"

            # Switch to Phase 2 dataloader on first epoch
            if epoch == cfg.phase1_end:
                print("\n" + "=" * 80)
                print(f"SWITCHING TO PHASE 2: Multi-Horizon {cfg.phase2_horizons}")
                print("=" * 80)
                train_dataset, valid_dataset = create_datasets(
                    vol_data, ex_data,
                    seq_len_range=(cfg.phase2_seq_len, cfg.phase2_seq_len)
                )
                train_loader, valid_loader = create_dataloaders(
                    train_dataset, valid_dataset, cfg.batch_size
                )
                print(f"✓ Datasets updated: train={len(train_dataset)}, valid={len(valid_dataset)}")

            print(f"\nEpoch {epoch}/{cfg.total_epochs} - Phase 2: {phase_name} {cfg.phase2_horizons}")

            # Train
            train_metrics = train_one_epoch_multihorizon(
                model, optimizer, train_loader, model.device, cfg.kl_weight,
                cfg.phase2_horizons, cfg.phase2_weights, scaler
            )

            # Validate (always with teacher forcing)
            val_metrics = validate(model, valid_loader, model.device, cfg.kl_weight)

        # Print metrics
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_metrics['loss']:.6f} | "
              f"Recon: {train_metrics['reconstruction_loss']:.6f} | "
              f"KL: {train_metrics['kl_loss']:.6f}")
        print(f"  Valid Loss: {val_metrics['loss']:.6f} | "
              f"Recon: {val_metrics['reconstruction_loss']:.6f} | "
              f"KL: {val_metrics['kl_loss']:.6f}")

        # Save training history
        training_history.append({
            'epoch': epoch,
            'phase': phase,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_recon': train_metrics['reconstruction_loss'],
            'val_recon': val_metrics['reconstruction_loss'],
            'train_kl': train_metrics['kl_loss'],
            'val_kl': val_metrics['kl_loss'],
        })

        # Save checkpoints
        checkpoint_path = None
        if epoch == cfg.phase1_end - 1:  # End of phase 1
            checkpoint_path = checkpoint_dir / cfg.get_checkpoint_name(epoch)
        elif epoch == cfg.total_epochs - 1:  # End of training
            checkpoint_path = checkpoint_dir / cfg.get_checkpoint_name(epoch)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_checkpoint_path = checkpoint_dir / f"{cfg.get_checkpoint_prefix()}_best.pt"

            torch.save({
                'epoch': epoch,
                'model_config': model_config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'variant': args.variant,
            }, best_checkpoint_path)

        # Save phase checkpoints
        if checkpoint_path is not None:
            print(f"  Saving checkpoint: {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_config': model_config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'training_history': training_history,
                'variant': args.variant,
            }, checkpoint_path)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
