"""
Context=60 4-Phase Curriculum Training Script (RESUME FROM PHASE 2)

This script resumes training from the completed Phase 2 checkpoint and trains
ONLY Phases 3-4 with the shortened schedule (50 epochs per phase).

Training schedule:
- Phase 1 (epochs 0-199): Teacher forcing (H=1) - ✅ ALREADY COMPLETED
- Phase 2 (epochs 200-249): Multi-horizon [1,7,14,30,60,90] - ✅ ALREADY COMPLETED
- Phase 3 (epochs 250-299): AR H=60, offsets=[30,60] - 50 epochs
- Phase 4 (epochs 300-349): AR H=90, offsets=[45,90] - 50 epochs

TOTAL: 100 epochs remaining, estimated ~20 hours on 3070 Ti

CRITICAL: Loads Phase 2 checkpoint and saves checkpoint after EACH phase for comparison.

Usage:
    python train_backfill_context60_resume_phase3.py

Output:
    - models/backfill/backfill_context60_short_phase3_ep299.pt
    - models/backfill/backfill_context60_short_phase4_ep349.pt
    - models/backfill/backfill_context60_short_best.pt (best validation)
    - models/backfill/context60_training_log_resume_phase3.txt
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from collections import defaultdict

from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import (
    set_seeds,
    model_eval,
    train_autoregressive_multi_offset
)
from config.backfill_context60_config_short import BackfillContext60ConfigShort


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
        dtype=torch.float32  # Convert to float32 to match model weights
    )

    valid_dataset = VolSurfaceDataSetRand(
        (vol_data[split_idx:], ex_data[split_idx:]),
        min_seq_len=min_len,
        max_seq_len=max_len,
        dtype=torch.float32  # Convert to float32 to match model weights
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
    """Save checkpoint after phase completion."""
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

    filename = cfg.get_checkpoint_name(phase_num, epoch)
    filepath = os.path.join(output_dir, filename)
    torch.save(checkpoint, filepath)

    print(f"\n{'='*80}")
    print(f"✅ PHASE {phase_num} CHECKPOINT SAVED")
    print(f"{'='*80}")
    print(f"File: {filename}")
    print(f"Epoch: {epoch}")
    print(f"Phase: {phase_name}")
    print(f"Val Loss: {val_metrics['loss']:.6f}")
    print(f"{'='*80}\n")


def save_best_checkpoint(model, optimizer, epoch, val_loss, output_dir, cfg):
    """Save best checkpoint based on validation loss."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": model.config,
        "epoch": epoch,
        "val_loss": val_loss,
        "timestamp": time.time()
    }

    filename = f"{cfg.checkpoint_prefix}_best.pt"
    filepath = os.path.join(output_dir, filename)
    torch.save(checkpoint, filepath)


# ==============================================================================
# Phase 3: AR H=60
# ==============================================================================

def train_phase3_ar_h60(model, optimizer, train_loader, valid_loader,
                        start_epoch, end_epoch, horizon, offsets, ar_steps,
                        output_dir, log_file, cfg):
    """Train Phase 3 with autoregressive H=60."""
    print(f"\n{'='*80}")
    print(f"PHASE 3: AUTOREGRESSIVE H={horizon}")
    print(f"{'='*80}")
    print(f"Epochs: {start_epoch} - {end_epoch-1}")
    print(f"Offsets: {offsets} (randomly sampled per batch)")
    print(f"AR Steps: {ar_steps}")
    print(f"Method: train_autoregressive_multi_offset()")
    print(f"Sequence length: {cfg.phase3_seq_len}")
    print(f"{'='*80}\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_losses = defaultdict(float)
        offset_counts = defaultdict(int)
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Phase3-Epoch{epoch}"):
            losses = train_autoregressive_multi_offset(
                model, batch, optimizer,
                horizon=horizon,
                offsets=offsets,
                ar_steps=ar_steps
            )

            for k, v in losses.items():
                train_losses[k] += v if isinstance(v, (float, int)) else v.item()
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
        model, optimizer, end_epoch-1, 3, f"AR H={horizon}, offsets={offsets}",
        val_losses, output_dir, cfg
    )


# ==============================================================================
# Phase 4: AR H=90
# ==============================================================================

def train_phase4_ar_h90(model, optimizer, train_loader, valid_loader,
                        start_epoch, end_epoch, horizon, offsets, ar_steps,
                        output_dir, log_file, cfg):
    """Train Phase 4 with autoregressive H=90."""
    print(f"\n{'='*80}")
    print(f"PHASE 4: AUTOREGRESSIVE H={horizon} (FINAL)")
    print(f"{'='*80}")
    print(f"Epochs: {start_epoch} - {end_epoch-1}")
    print(f"Offsets: {offsets} (randomly sampled per batch)")
    print(f"AR Steps: {ar_steps}")
    print(f"Method: train_autoregressive_multi_offset()")
    print(f"Sequence length: {cfg.phase4_seq_len}")
    print(f"{'='*80}\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_losses = defaultdict(float)
        offset_counts = defaultdict(int)
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Phase4-Epoch{epoch}"):
            losses = train_autoregressive_multi_offset(
                model, batch, optimizer,
                horizon=horizon,
                offsets=offsets,
                ar_steps=ar_steps
            )

            for k, v in losses.items():
                train_losses[k] += v if isinstance(v, (float, int)) else v.item()
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
        model, optimizer, end_epoch-1, 4, f"AR H={horizon}, offsets={offsets}",
        val_losses, output_dir, cfg
    )


# ==============================================================================
# Main Training Function
# ==============================================================================

def main():
    """Main training function - resumes from Phase 2 checkpoint."""
    set_seeds(42)

    cfg = BackfillContext60ConfigShort
    output_dir = cfg.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    cfg.summary()

    # Open log file
    log_file = open(f"{output_dir}/context60_training_log_resume_phase3.txt", "w")
    log_file.write("Context=60 4-Phase Training Log (RESUME FROM PHASE 2)\n")
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

    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {model.device}")

    # ============================================================================
    # LOAD PHASE 2 CHECKPOINT
    # ============================================================================

    print(f"\n{'='*80}")
    print("LOADING PHASE 2 CHECKPOINT")
    print(f"{'='*80}")

    # Find the Phase 2 checkpoint
    phase2_checkpoint_path = os.path.join(output_dir, "backfill_context60_short_phase2_ep249.pt")

    if not os.path.exists(phase2_checkpoint_path):
        raise FileNotFoundError(
            f"Phase 2 checkpoint not found: {phase2_checkpoint_path}\n"
            f"Expected location: models/backfill/backfill_context60_short_phase2_ep249.pt"
        )

    print(f"Checkpoint: {phase2_checkpoint_path}")
    checkpoint = torch.load(phase2_checkpoint_path, map_location=model.device, weights_only=False)

    # Load model and optimizer state
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"✅ Loaded Phase 2 checkpoint:")
    print(f"   - Epoch: {checkpoint['epoch']}")
    print(f"   - Phase: {checkpoint['phase']} ({checkpoint['phase_name']})")
    print(f"   - Val Loss: {checkpoint['val_metrics']['loss']:.6f}")
    print(f"{'='*80}\n")

    # ============================================================================
    # PHASE 3: AR H=60
    # ============================================================================

    print(f"\n{'='*80}")
    print("Creating Phase 3 datasets...")
    print(f"Sequence length: {cfg.phase3_seq_len}")
    train_ds, valid_ds = create_datasets(vol_train, ex_train, cfg.phase3_seq_len)
    train_loader, valid_loader = create_dataloaders(train_ds, valid_ds, cfg.batch_size, cfg.valid_batch_size)
    print(f"Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")

    train_phase3_ar_h60(
        model, optimizer, train_loader, valid_loader,
        start_epoch=cfg.phase2_end, end_epoch=cfg.phase3_end,
        horizon=cfg.phase3_horizon, offsets=cfg.phase3_offsets,
        ar_steps=cfg.phase3_ar_steps,
        output_dir=output_dir, log_file=log_file, cfg=cfg
    )

    # ============================================================================
    # PHASE 4: AR H=90
    # ============================================================================

    print(f"\n{'='*80}")
    print("Creating Phase 4 datasets...")
    print(f"Sequence length: {cfg.phase4_seq_len}")
    train_ds, valid_ds = create_datasets(vol_train, ex_train, cfg.phase4_seq_len)
    train_loader, valid_loader = create_dataloaders(train_ds, valid_ds, cfg.batch_size, cfg.valid_batch_size)
    print(f"Train samples: {len(train_ds)}, Valid samples: {len(valid_ds)}")

    train_phase4_ar_h90(
        model, optimizer, train_loader, valid_loader,
        start_epoch=cfg.phase3_end, end_epoch=cfg.phase4_end,
        horizon=cfg.phase4_horizon, offsets=cfg.phase4_offsets,
        ar_steps=cfg.phase4_ar_steps,
        output_dir=output_dir, log_file=log_file, cfg=cfg
    )

    # ============================================================================
    # Training Complete
    # ============================================================================

    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nCheckpoints saved:")
    print(f"  ✓ Phase 1: backfill_context60_phase1_ep199.pt (pre-existing)")
    print(f"  ✓ Phase 2: backfill_context60_short_phase2_ep249.pt (loaded from)")
    print(f"  ✓ {cfg.get_checkpoint_name(3, cfg.phase3_end-1)}")
    print(f"  ✓ {cfg.get_checkpoint_name(4, cfg.phase4_end-1)} (FINAL)")
    print(f"  ✓ {cfg.checkpoint_prefix}_best.pt (best validation)")
    print("\nTraining log saved to:")
    print(f"  {output_dir}/context60_training_log_resume_phase3.txt")
    print("\nNext steps:")
    print(f"  1. Compare Phase 2 vs Phase 3 vs Phase 4 checkpoints")
    print(f"  2. Analyze if 50 epochs per phase is sufficient")
    print(f"  3. Generate predictions and evaluate performance")
    print("="*80 + "\n")

    log_file.close()


if __name__ == "__main__":
    main()
