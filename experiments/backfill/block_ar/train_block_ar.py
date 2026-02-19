"""
Training script for Block-AR DDPM on volatility surfaces.

Uses MCVD multi-task training with Diffusion Forcing noise schedules.
Reuses VolSurfaceDataset and normalization from the DDPM POC.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.block_ar_ddpm import BlockARConfig, ConditionalBlockARDDPM, denormalize_iv
from experiments.backfill.block_ar.config_block_ar import (
    BlockARPOCConfig,
    get_default_config,
    get_fast_test_config,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def train_epoch(
    model: ConditionalBlockARDDPM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    total_regime_loss = 0.0
    total_regime_acc = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        regime_ids = batch.get("regime")
        if regime_ids is not None:
            regime_ids = regime_ids.to(device)

        result = model(history, future, regime_ids=regime_ids)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        postfix = {"loss": loss.item()}
        if "regime_loss" in result:
            total_regime_loss += result["regime_loss"]
            total_regime_acc += result["regime_acc"]
            postfix["r_acc"] = f"{result['regime_acc']:.0%}"
        pbar.set_postfix(postfix)

    if scheduler is not None:
        scheduler.step()

    metrics = {"loss": total_loss / n_batches}
    if total_regime_loss > 0:
        metrics["regime_loss"] = total_regime_loss / n_batches
        metrics["regime_acc"] = total_regime_acc / n_batches
    return metrics


def validate(
    model: ConditionalBlockARDDPM,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            result = model(history, future)
            total_loss += result["loss"].item()
            n_batches += 1

    return {"val_loss": total_loss / n_batches}


def compute_ci_coverage(
    model: ConditionalBlockARDDPM,
    dataloader: DataLoader,
    n_samples: int = 10,
    device: str = "cpu",
    max_batches: int = 5,
    max_residual: int = 20,
) -> dict:
    """Compute CI coverage on validation data using staggered DDPM."""
    model.eval()

    all_coverages = {0.5: [], 0.8: [], 0.9: [], 0.95: []}
    all_diversity = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            # Denormalize ground truth
            future_gt = denormalize_iv(future_gt)

            # Generate samples (batched for GPU efficiency)
            samples = model.sample_batched(
                history, n_samples=n_samples, max_residual=max_residual
            )  # (B, n_samples, T_fut, 5, 5)

            for level in all_coverages:
                alpha = (1 - level) / 2
                lower = torch.quantile(samples, alpha, dim=1)
                upper = torch.quantile(samples, 1 - alpha, dim=1)
                covered = (future_gt >= lower) & (future_gt <= upper)
                all_coverages[level].append(covered.float().mean().item())

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


def ema_update(ema_params: dict, model: torch.nn.Module, decay: float) -> None:
    """Update EMA parameters."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ema_params:
                ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def main():
    parser = argparse.ArgumentParser(description="Train Block-AR DDPM on volatility surfaces")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fast", action="store_true", help="Use fast test config")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_eval_samples", type=int, default=None)
    parser.add_argument("--noise_rho", type=float, default=None, help="PYoCo noise correlation (0.0=independent, 0.5=default)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--block_size", type=int, default=None, help="Block size for AR generation (default: 10)")
    parser.add_argument("--jitter_std", type=float, default=None, help="DF noise jitter std (default: 0.15)")
    parser.add_argument("--checkpoint_every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--loss_type", type=str, default=None, choices=["mse", "huber"], help="Loss function (default: mse)")
    parser.add_argument("--huber_delta", type=float, default=None, help="Huber loss delta (default: 0.1)")
    parser.add_argument("--denoiser_type", type=str, default=None, choices=["bigru", "conv3d"], help="Denoiser architecture")
    parser.add_argument("--p_mask", type=float, default=None, help="MCVD mask probability (default: 0.2, uniform tasks: 0.5)")
    parser.add_argument("--use_regime", action="store_true", help="Enable hierarchical regime conditioning")
    parser.add_argument("--uniform_noise", action="store_true", help="Uniform-t training (one t per block instead of per-frame task-adaptive)")
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["pyramid", "uniform"], help="Inference sampling mode")
    args = parser.parse_args()

    config = get_fast_test_config() if args.fast else get_default_config()

    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.device:
        config.device = args.device
    if args.n_eval_samples:
        config.n_eval_samples = args.n_eval_samples
    if args.noise_rho is not None:
        config.noise_rho = args.noise_rho
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.jitter_std is not None:
        config.jitter_std = args.jitter_std
    if args.checkpoint_every is not None:
        config.checkpoint_every = args.checkpoint_every
    if args.loss_type is not None:
        config.loss_type = args.loss_type
    if args.huber_delta is not None:
        config.huber_delta = args.huber_delta
    if args.denoiser_type is not None:
        config.denoiser_type = args.denoiser_type
    if args.p_mask is not None:
        config.p_mask = args.p_mask
    if args.use_regime:
        config.use_regime_conditioning = True
    if args.uniform_noise:
        config.use_uniform_noise = True
    if args.sampling_mode is not None:
        config.sampling_mode = args.sampling_mode

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = "cpu"

    print("=" * 60)
    print("Block-AR DDPM Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"History: {config.history_len} -> Future: {config.future_len} (block_size={config.block_size})")
    print(f"Diffusion steps: {config.n_steps}, Schedule: {config.schedule}")
    print(f"MCVD p_mask: {config.p_mask}, Jitter std: {config.jitter_std}")
    print(f"Noise mode: {'uniform-t' if config.use_uniform_noise else 'task-adaptive (DF)'}")
    print(f"Sampling mode: {config.sampling_mode}")
    print(f"PYoCo noise_rho: {config.noise_rho}")
    print(f"Encoder: GRU h={config.gru_hidden_dim} -> bottleneck={config.bottleneck_dim} (attn pooling)")
    denoiser_type = getattr(config, 'denoiser_type', 'bigru')
    if denoiser_type == "conv3d":
        print(f"Denoiser: Conv3D ch={config.conv3d_base_channels} x{config.conv3d_n_res_blocks} ResBlocks")
    else:
        print(f"Denoiser: BiGRU h={config.bigru_hidden_dim}")
    print(f"Epochs: {config.epochs}, Batch: {config.batch_size}, LR: {config.lr}")
    print(f"EMA decay: {config.ema_decay}")
    if config.use_regime_conditioning:
        print(f"Regime conditioning: n_regimes={config.n_regimes}, embed_dim={config.regime_embed_dim}, loss_weight={config.regime_loss_weight}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data = np.load(config.data_path)
    surfaces = data["surface"]
    print(f"Loaded {len(surfaces)} surfaces with shape {surfaces.shape}")

    # Load regime labels if enabled
    regime_labels = None
    if config.use_regime_conditioning:
        regime_data = np.load("data/regime_labels.npz")
        regime_labels = regime_data["labels"]
        print(f"Loaded {len(regime_labels)} regime labels, {int(regime_data['n_regimes'])} regimes")

    # Create datasets
    train_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=0, end_idx=config.train_end,
        regime_labels=regime_labels, data_start_idx=0,
    )
    val_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.val_start, end_idx=config.val_end,
        regime_labels=regime_labels, data_start_idx=config.val_start,
    )
    test_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.test_start,
        regime_labels=regime_labels, data_start_idx=config.test_start,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=2,
    )

    # Create model — extract all BlockARConfig fields from experiment config.
    # This auto-copies any field shared between BlockARPOCConfig and BlockARConfig,
    # so adding a new field to BlockARConfig won't silently revert to defaults.
    print("\nCreating model...")
    import dataclasses as _dc
    _model_fields = {f.name for f in _dc.fields(BlockARConfig)}
    _model_kwargs = {
        k: getattr(config, k)
        for k in _model_fields
        if hasattr(config, k)
    }
    model_config = BlockARConfig(**_model_kwargs)
    # Fail-fast if BlockARConfig has fields not present in experiment config.
    # This prevents silent default fallback when new model fields are added.
    _missing = _model_fields - {f.name for f in _dc.fields(config)}
    if _missing:
        raise RuntimeError(
            f"BlockARConfig fields missing from BlockARPOCConfig: {_missing}. "
            f"Add these fields to BlockARPOCConfig in config_block_ar.py."
        )

    model = ConditionalBlockARDDPM(model_config)
    model = model.to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr / 10,
    )

    # EMA
    ema_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    best_coverage_90 = 0.0

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, lr_scheduler,
            config.device, config.grad_clip,
        )

        val_metrics = validate(model, val_loader, config.device)

        # EMA update
        ema_update(ema_params, model, config.ema_decay)

        epoch_msg = (
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['val_loss']:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if "regime_acc" in train_metrics:
            epoch_msg += f" | Regime Acc: {train_metrics['regime_acc']:.0%}"
        print(epoch_msg)

        # CI coverage evaluation
        if epoch % args.eval_every == 0 or epoch == config.epochs:
            print(f"  Evaluating CI coverage ({model_config.sampling_mode} sampling)...")
            coverage_metrics = compute_ci_coverage(
                model, val_loader,
                n_samples=config.n_eval_samples,
                device=config.device,
                max_batches=5,
                max_residual=config.max_residual_timestep,
            )
            print(
                f"  Coverage: 50%={coverage_metrics['coverage_50']:.1%}, "
                f"80%={coverage_metrics['coverage_80']:.1%}, "
                f"90%={coverage_metrics['coverage_90']:.1%}, "
                f"95%={coverage_metrics['coverage_95']:.1%} | "
                f"Diversity: {coverage_metrics['sample_diversity']:.4f}"
            )

            if coverage_metrics["coverage_90"] > best_coverage_90:
                best_coverage_90 = coverage_metrics["coverage_90"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": _dc.asdict(model_config),
                        "ema_params": ema_params,
                        "metrics": {**train_metrics, **val_metrics, **coverage_metrics},
                    },
                    f"{config.output_dir}/best_coverage_model.pt",
                )
                print(f"  Saved best coverage model (90% CI: {best_coverage_90:.1%})")

        # Best by val loss
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": _dc.asdict(model_config),
                    "ema_params": ema_params,
                    "metrics": {**train_metrics, **val_metrics},
                },
                f"{config.output_dir}/best_model.pt",
            )

        # Regular checkpoint
        if epoch % config.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": _dc.asdict(model_config),
                    "ema_params": ema_params,
                },
                f"{config.output_dir}/checkpoint_epoch_{epoch}.pt",
            )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=2,
    )

    test_val = validate(model, test_loader, config.device)
    print(f"Test Loss: {test_val['val_loss']:.6f}")

    test_coverage = compute_ci_coverage(
        model, test_loader,
        n_samples=config.n_eval_samples,
        device=config.device,
        max_batches=20,
        max_residual=config.max_residual_timestep,
    )
    print(
        f"Test Coverage: 50%={test_coverage['coverage_50']:.1%}, "
        f"80%={test_coverage['coverage_80']:.1%}, "
        f"90%={test_coverage['coverage_90']:.1%}, "
        f"95%={test_coverage['coverage_95']:.1%}"
    )
    print(f"Sample Diversity: {test_coverage['sample_diversity']:.4f}")

    # Save final model
    torch.save(
        {
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": _dc.asdict(model_config),
            "ema_params": ema_params,
            "test_metrics": {**test_val, **test_coverage},
        },
        f"{config.output_dir}/final_model.pt",
    )

    print(f"\nModels saved to: {config.output_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main()
