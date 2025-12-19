"""
Context=60 2-Phase Training with CORRECTED KL Weight (Latent12 V2)

This script trains the CORRECTED version of the latent12 model, fixing the
over-regularization issue discovered in V1.

V1 Problem:
- kl_weight=5e-5 caused posterior collapse (KL=0.854 < 1.0)
- Latent space not being used effectively
- PC1 variance still 99.27% (over-compressed)
- Correlation decreased 67% (0.338 → 0.113)

V2 Fix:
- kl_weight: 5e-5 → 1e-5 (reduced 5× to baseline level)
- latent_dim: 12 (kept from V1 - preserve increased capacity)
- Decouples latent capacity increase from KL regularization

Training schedule:
- Phase 1 (epochs 0-200): Teacher forcing (H=1)
- Phase 2 (epochs 201-600): Multi-horizon [1,7,14,30,60,90] - 400 EPOCHS

Expected V2 Improvements:
- KL divergence: 0.854 → 2-5 (healthy range)
- PC1 variance: 99.27% → 90-95% (better distribution)
- Correlation: 0.113 → 0.35-0.45 (recovery to baseline or better)
- Day-1 spread: 0.0404 → 0.03-0.035 (maintain improvement)

Usage:
    # Train from scratch
    python experiments/backfill/context60/train_backfill_context60_latent12_v2.py

    # Resume from Phase 1 checkpoint
    python experiments/backfill/context60/train_backfill_context60_latent12_v2.py \\
        --resume_from models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_phase1_ep199.pt

Output:
    - models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_phase1_ep199.pt
    - models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_phase2_ep599.pt
    - models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_best.pt (best validation)
    - models/backfill/context60_experiment/checkpoints/context60_latent12_v2_training_log.txt
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

from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import (
    set_seeds,
    model_eval
)
from config.backfill_context60_config_latent12_v2 import BackfillContext60ConfigLatent12V2

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train Context60 Latent12 V2 model (corrected KL weight)')
parser.add_argument('--resume_from', type=str, default=None,
                   help='Path to checkpoint to resume from (e.g., Phase 1 checkpoint)')
args = parser.parse_args()


# ==============================================================================
# Dataset and Dataloader Creation
# ==============================================================================

def create_datasets(vol_data, ex_data, seq_len_range):
    """
    Create train/valid datasets with specified sequence length range.

    Args:
        vol_data: Numpy array of volatility surfaces
        ex_data: Numpy array of extra features
        seq_len_range: Tuple of (min_len, max_len)

    Returns:
        train_dataset, valid_dataset
    """
    min_len, max_len = seq_len_range

    # 80/20 train/valid split
    split_idx = int(0.8 * len(vol_data))

    train_dataset = VolSurfaceDataSetRand(
        (vol_data[:split_idx], ex_data[:split_idx]),
        min_seq_len=min_len,
        max_seq_len=max_len,
        dtype=torch.float32  # CHANGED: Use float32 for 2× speedup
    )

    valid_dataset = VolSurfaceDataSetRand(
        (vol_data[split_idx:], ex_data[split_idx:]),
        min_seq_len=min_len,
        max_seq_len=max_len,
        dtype=torch.float32  # CHANGED: Use float32 for 2× speedup
    )

    return train_dataset, valid_dataset


def create_dataloaders(train_dataset, valid_dataset, batch_size, valid_batch_size):
    """
    Create dataloaders with custom batch sampler.

    Args:
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        batch_size: Batch size for training
        valid_batch_size: Batch size for validation

    Returns:
        train_loader, valid_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=CustomBatchSampler(
            train_dataset,
            batch_size,
            train_dataset.seq_lens[0]
        )
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=CustomBatchSampler(
            valid_dataset,
            valid_batch_size,
            valid_dataset.seq_lens[0]
        )
    )

    return train_loader, valid_loader


# ==============================================================================
# Checkpoint Saving
# ==============================================================================

def save_phase_checkpoint(model, optimizer, epoch, phase_num, phase_name,
                          val_metrics, output_dir, cfg):
    """
    Save checkpoint after phase completion.

    Args:
        model: CVAEMemRand model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        phase_num: Phase number (1 or 2)
        phase_name: Descriptive phase name
        val_metrics: Validation metrics dictionary
        output_dir: Directory to save checkpoint
        cfg: Configuration class

    Returns:
        filepath: Path to saved checkpoint
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.config,
        "epoch": epoch,
        "phase": phase_num,
        "phase_name": phase_name,
        "val_metrics": val_metrics,
        "timestamp": time.time()
    }

    filename = cfg.get_checkpoint_name(epoch)
    filepath = os.path.join(output_dir, filename)

    torch.save(checkpoint, filepath)

    print(f"\n{'='*80}")
    print(f"✅ PHASE {phase_num} CHECKPOINT SAVED")
    print(f"{'='*80}")
    print(f"File: {filepath}")
    print(f"Epoch: {epoch}")
    print(f"Phase: {phase_name}")
    print(f"Validation Loss: {val_metrics['loss']:.6f}")
    if 'reconstruction_loss' in val_metrics:
        print(f"Reconstruction Loss: {val_metrics['reconstruction_loss']:.6f}")
    if 'kl_loss' in val_metrics:
        print(f"KL Loss: {val_metrics['kl_loss']:.6f}")
    print(f"{'='*80}\n")

    return filepath


# ==============================================================================
# Phase 1: Teacher Forcing
# ==============================================================================

def train_phase1_teacher_forcing(model, optimizer, train_loader, valid_loader,
                                 start_epoch, end_epoch, output_dir, log_file, cfg):
    """
    Phase 1: Teacher Forcing (epochs 0-200)

    Standard single-step prediction with horizon=1.
    """
    print("\n" + "="*80)
    print("PHASE 1: TEACHER FORCING (H=1)")
    print("="*80)
    print(f"Epochs: {start_epoch} to {end_epoch-1}")
    print(f"Method: model.train_step()")
    print(f"Sequence length: {cfg.phase1_seq_len}")
    print("="*80 + "\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Phase1-Epoch{epoch}"):
            losses = model.train_step(batch, optimizer)
            for k, v in losses.items():
                train_losses[k] += v.item() if hasattr(v, 'item') else v
            num_batches += 1

        # Average losses
        for k in train_losses:
            train_losses[k] /= num_batches

        # Validation
        val_losses = model_eval(model, valid_loader)

        # Logging
        log_msg = f"Epoch {epoch}: Train Loss={train_losses['loss']:.6f}, Val Loss={val_losses['loss']:.6f}\n"
        print(log_msg.strip())
        log_file.write(log_msg)
        log_file.flush()

        # Save best
        if val_losses['loss'] < best_loss:
            best_loss = val_losses['loss']
            model.save_weights(optimizer, output_dir, f"{cfg.checkpoint_prefix}_best")

    # Save phase checkpoint
    save_phase_checkpoint(
        model, optimizer, end_epoch-1, 1, "Teacher Forcing",
        val_losses, output_dir, cfg
    )

    return val_losses


# ==============================================================================
# Phase 2: Multi-Horizon (EXTENDED 400 EPOCHS)
# ==============================================================================

def train_phase2_multihorizon(model, optimizer, train_loader, valid_loader,
                              start_epoch, end_epoch, horizons, weights,
                              output_dir, log_file, cfg):
    """
    Phase 2: Multi-Horizon Training (epochs 201-600) - 400 EPOCHS

    Trains on multiple horizons simultaneously with UNIFORM weighted loss.
    """
    print("\n" + "="*80)
    print(f"PHASE 2: MULTI-HORIZON {horizons} (400 EPOCHS)")
    print("="*80)
    print(f"Epochs: {start_epoch} to {end_epoch-1}")
    print(f"Method: model.train_step_multihorizon()")
    print(f"Weights: {weights} (UNIFORM)")
    print(f"Sequence length: {cfg.phase2_seq_len}")
    print("="*80 + "\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Phase2-Epoch{epoch}"):
            losses = model.train_step_multihorizon(batch, optimizer, horizons=horizons)

            for k, v in losses.items():
                if isinstance(v, dict):
                    for h, loss_val in v.items():
                        train_losses[f"{k}_h{h}"] += loss_val
                else:
                    train_losses[k] += v if isinstance(v, float) else v.item()
            num_batches += 1

        # Average losses
        for k in train_losses:
            train_losses[k] /= num_batches

        # Validation
        val_losses = model_eval(model, valid_loader)

        # Logging
        log_msg = f"Epoch {epoch}: Train Loss={train_losses['loss']:.6f}, Val Loss={val_losses['loss']:.6f}\n"
        print(log_msg.strip())
        log_file.write(log_msg)
        log_file.flush()

        # Save best
        if val_losses['loss'] < best_loss:
            best_loss = val_losses['loss']
            model.save_weights(optimizer, output_dir, f"{cfg.checkpoint_prefix}_best")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_phase_checkpoint(
                model, optimizer, epoch, 2, f"Multi-Horizon [Checkpoint ep{epoch}]",
                val_losses, output_dir, cfg
            )
            print(f"💾 Periodic checkpoint saved at epoch {epoch}")

    # Save phase checkpoint (FINAL)
    save_phase_checkpoint(
        model, optimizer, end_epoch-1, 2, f"Multi-Horizon {horizons} (FINAL)",
        val_losses, output_dir, cfg
    )

    return val_losses


# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    # Setup
    set_seeds(0)
    # NOTE: We do NOT set default dtype - let dataset handle it via dtype parameter

    cfg = BackfillContext60ConfigLatent12V2
    output_dir = cfg.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # Print header
    print("=" * 80)
    print("CONTEXT=60 LATENT12 V2 (CORRECTED) 2-PHASE TRAINING")
    print("=" * 80)
    print()

    # Print configuration
    cfg.summary()

    # Open log file
    log_file = open(f"{output_dir}/context60_latent12_v2_training_log.txt", "w")
    log_file.write("Context=60 Latent12 V2 (CORRECTED) 2-Phase Training Log\n")
    log_file.write("="*80 + "\n\n")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    ret = data["ret"]
    skew = data["skews"]
    slope = data["slopes"]
    ex_data = np.stack([ret, skew, slope], axis=-1)

    # Extract training period
    vol_train = surfaces[cfg.train_start_idx:cfg.train_end_idx]
    ex_train = ex_data[cfg.train_start_idx:cfg.train_end_idx]

    print(f"Training data: {vol_train.shape[0]} days")
    print(f"Training period: {cfg.train_period_years} years")

    # Create model
    print("\nCreating model...")
    model_config = {
        "feat_dim": (5, 5),
        "ex_feats_dim": 3,
        "latent_dim": cfg.latent_dim,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "kl_weight": cfg.kl_weight,
        "re_feat_weight": cfg.re_feat_weight,
        "surface_hidden": cfg.surface_hidden,
        "ex_feats_hidden": None,
        "mem_type": "lstm",
        "mem_hidden": cfg.mem_hidden,
        "mem_layers": cfg.mem_layers,
        "mem_dropout": cfg.mem_dropout,
        "ctx_surface_hidden": cfg.surface_hidden,
        "ctx_ex_feats_hidden": None,
        "interaction_layers": None,
        "compress_context": True,
        "use_dense_surface": False,
        "use_quantile_regression": cfg.use_quantile_regression,
        "num_quantiles": cfg.num_quantiles,
        "quantiles": cfg.quantiles,
        "quantile_loss_weights": cfg.quantile_loss_weights,
        "ex_loss_on_ret_only": cfg.ex_loss_on_ret_only,
        "ex_feats_loss_type": cfg.ex_feats_loss_type,
        "horizon": 1,  # Will be changed dynamically
        "context_len": cfg.context_len,
    }

    model = CVAEMemRand(model_config)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Load checkpoint if provided
    skip_phase1 = False
    resume_epoch = None  # Track which epoch to resume from
    if args.resume_from:
        print(f"\n{'='*80}")
        print(f"LOADING CHECKPOINT: {args.resume_from}")
        print(f"{'='*80}")
        checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        model.load_weights(dict_to_load=checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer'])

        loaded_epoch = checkpoint.get('epoch', 'unknown')
        loaded_phase = checkpoint.get('phase', 'unknown')
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Epoch: {loaded_epoch}")
        print(f"   Phase: {loaded_phase} - {checkpoint.get('phase_name', 'N/A')}")
        print(f"   Validation loss: {checkpoint.get('val_metrics', {}).get('loss', 'N/A')}")
        print(f"{'='*80}\n")

        # If loading Phase 1 checkpoint, skip Phase 1 training
        if loaded_phase == 1 or loaded_epoch >= cfg.phase1_end - 1:
            skip_phase1 = True
            print("⏭️  Phase 1 already completed - will skip to Phase 2")

        # If loading Phase 2 checkpoint, resume from next epoch
        if loaded_phase == 2 and loaded_epoch < cfg.phase2_end - 1:
            resume_epoch = loaded_epoch + 1
            print(f"📍 Will resume Phase 2 training from epoch {resume_epoch}")
    else:
        print("No checkpoint provided - training from scratch")

    model.to(model.device)
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {model.device}")
    print(f"Latent dimension: {cfg.latent_dim} (INCREASED from 5)")
    print(f"KL weight: {cfg.kl_weight} (INCREASED from 1e-5)")

    # ============================================================================
    # PHASE 1: Teacher Forcing (CONDITIONAL)
    # ============================================================================

    if not skip_phase1:
        print(f"\n{'='*80}")
        print("Creating Phase 1 datasets...")
        print(f"Sequence length: {cfg.phase1_seq_len}")
        train_ds, valid_ds = create_datasets(vol_train, ex_train, cfg.phase1_seq_len)
        train_loader, valid_loader = create_dataloaders(train_ds, valid_ds, cfg.batch_size, cfg.valid_batch_size)
        print(f"Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")
        print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

        train_phase1_teacher_forcing(
            model, optimizer, train_loader, valid_loader,
            start_epoch=0, end_epoch=cfg.phase1_end,
            output_dir=output_dir, log_file=log_file, cfg=cfg
        )
    else:
        print(f"\n{'='*80}")
        print("⏭️  SKIPPING PHASE 1 (loaded from checkpoint)")
        print(f"{'='*80}\n")

    # ============================================================================
    # PHASE 2: Multi-Horizon (400 EPOCHS)
    # ============================================================================

    print(f"\n{'='*80}")
    print("Creating Phase 2 datasets...")
    print(f"Sequence length: {cfg.phase2_seq_len}")
    train_ds, valid_ds = create_datasets(vol_train, ex_train, cfg.phase2_seq_len)
    train_loader, valid_loader = create_dataloaders(train_ds, valid_ds, cfg.batch_size, cfg.valid_batch_size)
    print(f"Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")

    # Determine start epoch for Phase 2
    phase2_start = resume_epoch if resume_epoch is not None else cfg.phase1_end

    train_phase2_multihorizon(
        model, optimizer, train_loader, valid_loader,
        start_epoch=phase2_start, end_epoch=cfg.phase2_end,
        horizons=cfg.phase2_horizons, weights=cfg.phase2_weights,
        output_dir=output_dir, log_file=log_file, cfg=cfg
    )

    # ============================================================================
    # Training Complete
    # ============================================================================

    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nCheckpoints saved:")
    print(f"  ✓ {output_dir}/{cfg.get_checkpoint_name(cfg.phase1_end-1)}")
    print(f"  ✓ {output_dir}/{cfg.get_checkpoint_name(cfg.phase2_end-1)} (FINAL)")
    print(f"  ✓ {output_dir}/{cfg.checkpoint_prefix}_best.pt (best validation)")
    print("\nTraining log saved to:")
    print(f"  {output_dir}/context60_latent12_training_log.txt")
    print("\nExpected Improvements:")
    print(f"  - Day-1 spread: 0.0858 → 0.025-0.03 (70% reduction)")
    print(f"  - Latent correlation: 0.212 → 0.4-0.5")
    print(f"  - PC1: 99.94% → <90%")
    print("\nNext steps:")
    print(f"  1. Run: python experiments/backfill/context60/analyze_latent_information_bottleneck.py")
    print(f"  2. Compare latent space metrics (correlation, PC1)")
    print(f"  3. Run: python experiments/backfill/context60/analyze_gt_day1_variance_by_regime.py")
    print(f"  4. Verify day-1 spread reduction")
    print("="*80 + "\n")

    log_file.close()


if __name__ == "__main__":
    main()
