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
    parser.add_argument("--history_len", type=int, default=None, help="History context length (default: 30)")
    parser.add_argument("--jitter_std", type=float, default=None, help="DF noise jitter std (default: 0.15)")
    parser.add_argument("--checkpoint_every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--loss_type", type=str, default=None, choices=["mse", "huber", "crps"], help="Loss function (default: mse, crps requires --learn_sigma)")
    parser.add_argument("--huber_delta", type=float, default=None, help="Huber loss delta (default: 0.1)")
    parser.add_argument("--denoiser_type", type=str, default=None, choices=["bigru", "conv3d", "causal_conv3d"], help="Denoiser architecture")
    parser.add_argument("--encoder_type", type=str, default=None, choices=["gru", "conv3d"], help="Encoder architecture (gru=flat spatial, conv3d=spatial-aware)")
    parser.add_argument("--bottleneck_dim", type=int, default=None, help="Encoder bottleneck dimension (default: 64)")
    parser.add_argument("--p_mask", type=float, default=None, help="MCVD mask probability (default: 0.2, uniform tasks: 0.5)")
    parser.add_argument("--use_regime", action="store_true", help="Enable hierarchical regime conditioning")
    parser.add_argument("--uniform_noise", action="store_true", help="Uniform-t training (one t per block instead of per-frame task-adaptive)")
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["pyramid", "uniform"], help="Inference sampling mode")
    parser.add_argument("--forward_only", action="store_true", help="Disable MCVD: always FORWARD task (past visible, future masked)")
    parser.add_argument("--mcvd_task_probs", type=float, nargs=4, default=None,
                        metavar=("FWD", "BWD", "INTERP", "UNCOND"),
                        help="Explicit MCVD task probs (forward backward interpolation unconditional), must sum to 1.0. Overrides --p_mask.")
    parser.add_argument("--interp_loss_weight", type=float, default=None,
                        help="Down-weight interpolation loss (1.0=full, 0.3=30%%). Decouples task exposure from gradient pressure.")
    parser.add_argument("--conv3d_base_channels", type=int, default=None, help="Conv3D denoiser base channels (default: 32)")
    parser.add_argument("--conv3d_n_res_blocks", type=int, default=None, help="Conv3D denoiser residual blocks (default: 4)")
    parser.add_argument("--gru_hidden_dim", type=int, default=None, help="GRU encoder hidden dim (default: 64)")
    parser.add_argument("--heteroscedastic_noise", action="store_true",
                        help="Scale diffusion noise by condition IV level (wider CIs for high-IV)")
    parser.add_argument("--heteroscedastic_power", type=float, default=None,
                        help="Exponent for heteroscedastic noise (0.5=var∝IV, 1.0=std∝IV)")
    parser.add_argument("--learned_variance", action="store_true",
                        help="Diffusion2-style variance head with beta-NLL (condition-dependent uncertainty)")
    parser.add_argument("--variance_beta_nll", type=float, default=None,
                        help="beta-NLL weight for learned variance (0.5 recommended)")
    parser.add_argument("--ratio_target", action="store_true",
                        help="Ratio-space diffusion: model predicts transformed ratios instead of absolute IV")
    parser.add_argument("--ratio_target_mode", type=str, default="log", choices=["log", "logit", "vol_scaled", "vol_scaled_percell", "nsdiff", "e2e_nll", "vol_scaled_learned", "learned_percell"],
                        help="Ratio mode: 'log', 'logit', 'vol_scaled', 'vol_scaled_percell', 'nsdiff', 'e2e_nll', or 'vol_scaled_learned' (hybrid)")
    parser.add_argument("--nsdiff_sigma_lambda", type=float, default=0.1,
                        help="NLL loss weight for NSDiff/e2e learned sigma (0.1 default)")
    parser.add_argument("--e2e_sigma_reg", type=float, default=0.01,
                        help="L2 regularization on log_sigma for e2e_nll mode")
    parser.add_argument("--vol_scale_power", type=float, default=1.0,
                        help="Exponent on vol_scale: 0.5=sqrt dampening, 1.0=full (default)")
    parser.add_argument("--vol_scale_min", type=float, default=0.5,
                        help="Min clamp for vol_scale (higher = wider calm CIs)")
    parser.add_argument("--vol_scale_max", type=float, default=2.0,
                        help="Max clamp for vol_scale")
    parser.add_argument("--baseline_window", type=int, default=1,
                        help="Number of history days to average for baseline (1 = last day only)")
    parser.add_argument("--learn_sigma", action="store_true",
                        help="Nichol-Dhariwal learned variance: denoiser predicts per-element variance")
    parser.add_argument("--lambda_vlb", type=float, default=0.001,
                        help="VLB loss weight for learned variance (0.001 recommended)")
    parser.add_argument("--cond_drop_prob", type=float, default=0.0,
                        help="CFG conditioning dropout probability during training (0.0 = no CFG)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="CFG guidance scale at inference (1.0 = no guidance)")
    parser.add_argument("--crps_variance_head", action="store_true",
                        help="Enable CRPS variance head for condition-dependent posterior noise")
    parser.add_argument("--lambda_crps", type=float, default=0.1,
                        help="Weight for CRPS auxiliary loss (default: 0.1)")
    parser.add_argument("--use_mean_head", action="store_true",
                        help="Add MLP mean prediction head for bias correction")
    parser.add_argument("--mean_head_lambda", type=float, default=1.0,
                        help="Weight for mean prediction loss (default: 1.0)")
    parser.add_argument("--aux_regime_features", action="store_true",
                        help="Feed vol_of_vol and IV level as explicit conditioning features")
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
    if args.history_len is not None:
        config.history_len = args.history_len
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
    if args.encoder_type is not None:
        config.encoder_type = args.encoder_type
    if args.bottleneck_dim is not None:
        config.bottleneck_dim = args.bottleneck_dim
    if args.p_mask is not None:
        config.p_mask = args.p_mask
    if args.use_regime:
        config.use_regime_conditioning = True
    if args.uniform_noise:
        config.use_uniform_noise = True
    if args.sampling_mode is not None:
        config.sampling_mode = args.sampling_mode
    if args.forward_only:
        config.forward_only = True
    if args.mcvd_task_probs is not None:
        fwd, bwd, interp, uncond = args.mcvd_task_probs
        assert abs(fwd + bwd + interp + uncond - 1.0) < 1e-6, \
            f"--mcvd_task_probs must sum to 1.0, got {fwd + bwd + interp + uncond:.4f}"
        config.mcvd_p_forward = fwd
        config.mcvd_p_backward = bwd
        config.mcvd_p_interpolation = interp
        config.mcvd_p_unconditional = uncond
    if args.interp_loss_weight is not None:
        config.interp_loss_weight = args.interp_loss_weight
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.history_len is not None:
        config.history_len = args.history_len
    if args.conv3d_base_channels is not None:
        config.conv3d_base_channels = args.conv3d_base_channels
    if args.conv3d_n_res_blocks is not None:
        config.conv3d_n_res_blocks = args.conv3d_n_res_blocks
    if args.gru_hidden_dim is not None:
        config.gru_hidden_dim = args.gru_hidden_dim
    if args.heteroscedastic_noise:
        config.heteroscedastic_noise = True
    if args.heteroscedastic_power is not None:
        config.heteroscedastic_power = args.heteroscedastic_power
    if args.learned_variance:
        config.learned_variance = True
    if args.variance_beta_nll is not None:
        config.variance_beta_nll = args.variance_beta_nll
    if args.ratio_target:
        config.ratio_target = True
        config.ratio_target_mode = args.ratio_target_mode
        config.vol_scale_power = args.vol_scale_power
        config.vol_scale_min = args.vol_scale_min
        config.vol_scale_max = args.vol_scale_max
        config.baseline_window = args.baseline_window
        config.nsdiff_sigma_lambda = args.nsdiff_sigma_lambda
        config.e2e_sigma_reg = args.e2e_sigma_reg
    if args.learn_sigma:
        config.learn_sigma = True
        config.lambda_vlb = args.lambda_vlb
    if args.cond_drop_prob > 0:
        config.cond_drop_prob = args.cond_drop_prob
        config.guidance_scale = args.guidance_scale
    if args.crps_variance_head:
        config.crps_variance_head = True
        config.lambda_crps = args.lambda_crps
    if args.use_mean_head:
        config.use_mean_head = True
        config.mean_head_lambda = args.mean_head_lambda
    if args.aux_regime_features:
        config.aux_regime_features = True

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = "cpu"

    print("=" * 60)
    print("Block-AR DDPM Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"History: {config.history_len} -> Future: {config.future_len} (block_size={config.block_size})")
    print(f"Diffusion steps: {config.n_steps}, Schedule: {config.schedule}")
    if config.forward_only:
        mcvd_str = "DISABLED (forward-only)"
    elif config.mcvd_p_forward + config.mcvd_p_backward + config.mcvd_p_interpolation + config.mcvd_p_unconditional > 0:
        mcvd_str = (f"explicit (fwd={config.mcvd_p_forward:.0%} bwd={config.mcvd_p_backward:.0%} "
                     f"interp={config.mcvd_p_interpolation:.0%} uncond={config.mcvd_p_unconditional:.0%})")
    else:
        p = config.p_mask
        mcvd_str = (f"p_mask={p} (fwd={p*(1-p):.0%} bwd={p*(1-p):.0%} "
                     f"interp={(1-p)**2:.0%} uncond={p**2:.0%})")
    print(f"MCVD: {mcvd_str}")
    if config.interp_loss_weight < 1.0:
        print(f"Interpolation loss weight: {config.interp_loss_weight}")
    print(f"Noise mode: {'uniform-t' if config.use_uniform_noise else 'task-adaptive (DF)'}")
    print(f"Sampling mode: {config.sampling_mode}")
    print(f"PYoCo noise_rho: {config.noise_rho}")
    encoder_type = getattr(config, 'encoder_type', 'gru')
    if encoder_type == "conv3d":
        print(f"Encoder: CausalConv3D -> bottleneck={config.bottleneck_dim}")
    else:
        print(f"Encoder: GRU h={config.gru_hidden_dim} -> bottleneck={config.bottleneck_dim} (attn pooling)")
    denoiser_type = getattr(config, 'denoiser_type', 'bigru')
    if denoiser_type == "causal_conv3d":
        print(f"Denoiser: CausalConv3D ch={config.conv3d_base_channels} x{config.conv3d_n_res_blocks} ResBlocks")
    elif denoiser_type == "conv3d":
        print(f"Denoiser: Conv3D ch={config.conv3d_base_channels} x{config.conv3d_n_res_blocks} ResBlocks")
    else:
        print(f"Denoiser: BiGRU h={config.bigru_hidden_dim}")
    print(f"Epochs: {config.epochs}, Batch: {config.batch_size}, LR: {config.lr}")
    print(f"EMA decay: {config.ema_decay}")
    if config.use_regime_conditioning:
        print(f"Regime conditioning: n_regimes={config.n_regimes}, embed_dim={config.regime_embed_dim}, loss_weight={config.regime_loss_weight}")
    if config.heteroscedastic_noise:
        print(f"Heteroscedastic noise: ON (global_mean_iv={config.global_mean_iv:.4f}, power={config.heteroscedastic_power})")
    if config.learned_variance:
        print(f"Learned variance: ON (beta_nll={config.variance_beta_nll})")
    if config.ratio_target:
        print(f"Ratio target: ON (mode={config.ratio_target_mode}, conditional uncertainty via representation)")
    if getattr(config, 'learn_sigma', False):
        print(f"Learned sigma: ON (Nichol-Dhariwal, lambda_vlb={config.lambda_vlb})")
    if config.cond_drop_prob > 0:
        print(f"CFG: ON (cond_drop_prob={config.cond_drop_prob}, guidance_scale={config.guidance_scale})")
    if getattr(config, 'crps_variance_head', False):
        print(f"CRPS variance head: ON (lambda_crps={config.lambda_crps})")
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
