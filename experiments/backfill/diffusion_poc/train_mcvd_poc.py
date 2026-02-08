#!/usr/bin/env python
"""
Training script for MCVD POC on volatility surfaces.

Uses MCVD's 2D U-Net with frame-concatenation conditioning instead of
our 3D conv + FiLM approach. Pads 5×5 surfaces to 8×8 for proper U-Net
hierarchy (ch_mult=[1,2,2], levels 8→4→2).

Usage:
    python experiments/backfill/diffusion_poc/train_mcvd_poc.py
    python experiments/backfill/diffusion_poc/train_mcvd_poc.py --epochs 50
    python experiments/backfill/diffusion_poc/train_mcvd_poc.py --fast
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.mcvd_wrapper import MCVDModel, denormalize_iv, pad_surface, crop_surface
from diffusion.mcvd.models.ema import EMAHelper
from experiments.backfill.diffusion_poc.config_mcvd_poc import (
    MCVDPOCConfig,
    build_mcvd_config,
    get_default_config,
    get_fast_test_config,
    get_paper_aligned_config,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import (
    VolSurfaceDataset,
)


def compute_ci_coverage(
    model: MCVDModel,
    dataloader: DataLoader,
    n_samples: int = 50,
    device: str = 'cpu',
    max_batches: int = 20,
    n_inference_steps: int = 20,
) -> dict:
    """Compute CI coverage on validation data."""
    model.eval()

    all_coverages = {0.5: [], 0.8: [], 0.9: [], 0.95: []}
    all_diversity = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)   # (B, T_hist, 5, 5)
            future_gt = batch["future"].to(device)  # (B, T_fut, 5, 5)

            # Denormalize ground truth (model.sample returns denormalized)
            future_gt = denormalize_iv(future_gt)

            # Generate samples
            samples = model.sample(
                history,
                n_samples=n_samples,
                sampler='ddim',
                n_inference_steps=n_inference_steps,
            )  # (B, n_samples, T_fut, 5, 5)

            # Coverage per CI level
            for level in all_coverages.keys():
                alpha = (1 - level) / 2
                lower = torch.quantile(samples, alpha, dim=1)
                upper = torch.quantile(samples, 1 - alpha, dim=1)
                covered = (future_gt >= lower) & (future_gt <= upper)
                coverage = covered.float().mean().item()
                all_coverages[level].append(coverage)

            # Diversity
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
    model: MCVDModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    grad_clip: float = 1.0,
    ema_helper: EMAHelper = None,
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

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if ema_helper is not None:
            ema_helper.update(model)

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": loss.item()})

    if scheduler is not None:
        scheduler.step()

    return {"loss": total_loss / n_batches}


def validate(
    model: MCVDModel,
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
    parser = argparse.ArgumentParser(description="Train MCVD POC on volatility surfaces")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fast", action="store_true", help="Use fast test config")
    parser.add_argument("--paper-aligned", action="store_true",
                        help="Use paper-aligned config (5-frame blocks, ngf=64, linear T=1000)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_eval_samples", type=int, default=50)
    parser.add_argument("--ngf", type=int, default=None, help="Base feature channels")
    parser.add_argument("--prob_mask_cond", type=float, default=None,
                        help="Conditioning mask probability (0.1 recommended)")
    args = parser.parse_args()

    # Config
    if args.fast:
        config = get_fast_test_config()
    elif args.paper_aligned:
        config = get_paper_aligned_config()
    else:
        config = get_default_config()
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.device:
        config.device = args.device
    if args.ngf:
        config.ngf = args.ngf
    if args.prob_mask_cond is not None:
        config.prob_mask_cond = args.prob_mask_cond

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = "cpu"

    print("=" * 60)
    print("MCVD POC Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Architecture: {config.arch} (2D U-Net, frames as channels)")
    print(f"History: {config.history_len} days -> Future: {config.future_len} days")
    print(f"Spatial: {config.surface_h}×{config.surface_w} → padded {config.padded_h}×{config.padded_w}")
    print(f"U-Net levels: ch_mult={config.ch_mult} ({config.padded_h}→{config.padded_h//2}→{config.padded_h//4})")
    print(f"Base channels: {config.ngf}, ResBlocks/level: {config.num_res_blocks}")
    print(f"Noise steps: {config.n_steps}, Schedule: {config.schedule}")
    print(f"Conditioning mask prob: {config.prob_mask_cond}")
    print(f"Epochs: {config.epochs}, Batch: {config.batch_size}, LR: {config.lr}")
    print("=" * 60)

    # --- Load data ---
    print("\nLoading data...")
    data = np.load(config.data_path)
    surfaces = data["surface"]
    print(f"Loaded {len(surfaces)} surfaces with shape {surfaces.shape}")

    train_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=0, end_idx=config.train_end,
    )
    val_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.val_start, end_idx=config.val_end,
    )
    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=2,
    )

    # --- Build model ---
    print("\nCreating MCVD model...")
    mcvd_config = build_mcvd_config(config)
    model = MCVDModel(mcvd_config)
    model = model.to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # EMA (Exponential Moving Average)
    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    print("EMA initialized (mu=0.999)")

    # Verify pad/crop roundtrip
    test_x = torch.randn(2, config.future_len, 5, 5)
    assert torch.allclose(crop_surface(pad_surface(test_x)), test_x), "Pad/crop roundtrip failed!"
    print("Pad/crop roundtrip: OK")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr / 10,
    )

    # --- Training loop ---
    print("\nStarting training...")
    best_val_loss = float("inf")
    best_coverage_90 = 0.0
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            config.device, config.grad_clip, ema_helper,
        )

        # Use EMA weights for validation
        ema_model = MCVDModel(mcvd_config).to(config.device)
        ema_model.load_state_dict(model.state_dict())
        ema_helper.ema(ema_model)
        ema_model.eval()

        val_metrics = validate(ema_model, val_loader, config.device)

        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train: {train_metrics['loss']:.6f} | "
            f"Val: {val_metrics['val_loss']:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Evaluate CI coverage periodically
        if epoch % args.eval_every == 0 or epoch == config.epochs:
            # Use more DDIM steps for larger T schedules
            eval_ddim_steps = min(100, config.n_steps)
            print(f"  Evaluating CI coverage (DDIM {eval_ddim_steps} steps, EMA)...")
            coverage = compute_ci_coverage(
                ema_model, val_loader,
                n_samples=args.n_eval_samples,
                device=config.device,
                max_batches=10,
                n_inference_steps=eval_ddim_steps,
            )
            print(
                f"  Coverage: 50%={coverage['coverage_50']:.1%}, "
                f"80%={coverage['coverage_80']:.1%}, "
                f"90%={coverage['coverage_90']:.1%}, "
                f"95%={coverage['coverage_95']:.1%} | "
                f"Diversity: {coverage['sample_diversity']:.4f}"
            )

            if coverage['coverage_90'] > best_coverage_90:
                best_coverage_90 = coverage['coverage_90']
                _save_checkpoint(model, optimizer, epoch, config,
                                 {**train_metrics, **val_metrics, **coverage},
                                 f"{config.output_dir}/best_coverage_model.pt",
                                 ema_helper)
                print(f"  Saved best coverage model (90% CI: {best_coverage_90:.1%})")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            _save_checkpoint(model, optimizer, epoch, config,
                             {**train_metrics, **val_metrics},
                             f"{config.output_dir}/best_model.pt",
                             ema_helper)

        if epoch % config.checkpoint_every == 0:
            _save_checkpoint(model, optimizer, epoch, config, None,
                             f"{config.output_dir}/checkpoint_epoch_{epoch}.pt",
                             ema_helper)

        del ema_model  # Free memory

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=2,
    )

    # Use EMA model for final evaluation
    ema_model = MCVDModel(mcvd_config).to(config.device)
    ema_model.load_state_dict(model.state_dict())
    ema_helper.ema(ema_model)
    ema_model.eval()

    test_val = validate(ema_model, test_loader, config.device)
    print(f"Test Loss (EMA): {test_val['val_loss']:.6f}")

    eval_ddim_steps = min(100, config.n_steps)
    test_coverage = compute_ci_coverage(
        ema_model, test_loader,
        n_samples=args.n_eval_samples,
        device=config.device,
        max_batches=20,
        n_inference_steps=eval_ddim_steps,
    )
    print(
        f"Test Coverage: 50%={test_coverage['coverage_50']:.1%}, "
        f"80%={test_coverage['coverage_80']:.1%}, "
        f"90%={test_coverage['coverage_90']:.1%}, "
        f"95%={test_coverage['coverage_95']:.1%}"
    )
    print(f"Sample Diversity: {test_coverage['sample_diversity']:.4f}")

    _save_checkpoint(model, optimizer, config.epochs, config,
                     {**test_val, **test_coverage},
                     f"{config.output_dir}/final_model.pt",
                     ema_helper)

    print(f"\nModels saved to: {config.output_dir}")
    print("Training complete!")


def _save_checkpoint(model, optimizer, epoch, config, metrics, path,
                     ema_helper=None):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mcvd_config": {
            k: v for k, v in vars(config).items()
            if not k.startswith('_') and not callable(v)
        },
    }
    if metrics:
        checkpoint["metrics"] = metrics
    if ema_helper is not None:
        checkpoint["ema_state_dict"] = ema_helper.state_dict()
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
