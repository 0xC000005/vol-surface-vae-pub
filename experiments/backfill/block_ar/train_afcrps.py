"""
Training script for Exp 89: afCRPS Single-Pass Block-AR.

Replaces DDPM's 100-step diffusion loop with a single forward pass trained
with almost-fair CRPS loss on IV-space output. Keeps encoder, Conv3D backbone,
AdaGN, and exp(z × vol_scale) denormalization.

Usage:
    # Pretrained init
    PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
        --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --no_ema --epochs 30 --batch_size 16 --noise_dim 16 --n_members 4 \
        --lr_noise 1e-3 --lr_decoder 1e-4 --lambda_vs 0.1 \
        --output_dir models/backfill/afcrps_v1_pretrained --device cuda

    # Scratch init (encoder still from pretrained)
    PYTHONPATH=. python experiments/backfill/block_ar/train_afcrps.py \
        --base_model models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --no_ema --from_scratch --epochs 30 --batch_size 16 --noise_dim 16 \
        --n_members 4 --lr 1e-3 --lambda_vs 0.1 \
        --output_dir models/backfill/afcrps_v1_scratch --device cuda
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
    load_pretrained_weights,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, n_members, lambda_vs, grad_clip, n_train_blocks=1, lambda_is=0.0, lambda_cs_reg=0.0, lambda_kurt=0.0, n_frames=0, unfreeze_encoder=False):
    model.train()
    # Keep encoder in eval mode (frozen, no dropout) unless unfrozen
    if not unfreeze_encoder:
        model.encoder.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_spread = 0.0
    total_vs = 0.0
    total_is = 0.0
    total_kurt = 0.0
    total_raw_kurt = 0.0
    total_bias = 0.0
    n_batches = 0

    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)

        result = model(history, future, n_members=n_members, lambda_vs=lambda_vs,
                       lambda_is=lambda_is, lambda_cs_reg=lambda_cs_reg,
                       lambda_kurt=lambda_kurt,
                       n_train_blocks=n_train_blocks,
                       n_frames=n_frames)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()

        total_loss += loss.item()
        total_mae += result["mae"].item()
        total_spread += result["spread"].item()
        total_vs += result["variogram"].item()
        total_is += result["interval_score"].item()
        total_kurt += result["kurt_loss"].item()
        total_raw_kurt += result["raw_kurt"].item()
        total_bias += result.get("bias_loss", torch.tensor(0.0)).item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": total_mae / max(n_batches, 1),
        "spread": total_spread / max(n_batches, 1),
        "variogram": total_vs / max(n_batches, 1),
        "interval_score": total_is / max(n_batches, 1),
        "kurt_loss": total_kurt / max(n_batches, 1),
        "raw_kurt": total_raw_kurt / max(n_batches, 1),
        "spread_mae_ratio": total_spread / max(total_mae, 1e-8),
        "bias_loss": total_bias / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, device, n_members):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_spread = 0.0
    n_batches = 0

    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        result = model(history, future, n_members=n_members)
        total_loss += result["loss"].item()
        total_mae += result["mae"].item()
        total_spread += result["spread"].item()
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_mae": total_mae / max(n_batches, 1),
        "val_spread": total_spread / max(n_batches, 1),
        "val_spread_mae_ratio": total_spread / max(total_mae, 1e-8),
    }


@torch.no_grad()
def quick_eval(model, loader, device, n_samples=50, max_batches=5):
    """Quick evaluation: CI coverage and kurtosis on val set."""
    model.eval()
    all_coverages = []
    all_gt_changes = []
    all_gen_changes = []
    all_member_samples = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        history = batch["history"].to(device)
        future = batch["future"].to(device)

        samples = model.sample(history, n_samples=n_samples)  # (B, K, 30, 5, 5)
        gt = denormalize_iv(future)  # (B, 30, 5, 5)

        # 90% CI coverage
        q05 = torch.quantile(samples, 0.05, dim=1)
        q95 = torch.quantile(samples, 0.95, dim=1)
        covered = ((gt >= q05) & (gt <= q95)).float().mean().item()
        all_coverages.append(covered)

        # Kurtosis: daily changes of ensemble mean
        gt_changes = (gt[:, 1:] - gt[:, :-1]).reshape(-1)
        gen_mean = samples.mean(dim=1)
        gen_changes = (gen_mean[:, 1:] - gen_mean[:, :-1]).reshape(-1)
        all_gt_changes.append(gt_changes.cpu())
        all_gen_changes.append(gen_changes.cpu())
        all_member_samples.append(samples.cpu())

    coverage = np.mean(all_coverages) if all_coverages else 0.0

    if all_gt_changes:
        gt_all = torch.cat(all_gt_changes)
        gen_all = torch.cat(all_gen_changes)
        gt_kurt = torch.mean((gt_all - gt_all.mean()) ** 4) / (gt_all.std() ** 4 + 1e-8)
        gen_kurt = torch.mean((gen_all - gen_all.mean()) ** 4) / (gen_all.std() ** 4 + 1e-8)
        kurtosis_ratio_mean = (gen_kurt / gt_kurt).item() if gt_kurt > 0 else 0.0

        # Also compute kurtosis from individual members (more representative)
        all_member_changes = []
        for batch_samples in all_member_samples:
            # batch_samples: (B, K, T, H, W)
            member_changes = batch_samples[:, :, 1:] - batch_samples[:, :, :-1]
            all_member_changes.append(member_changes.reshape(-1))
        if all_member_changes:
            mc = torch.cat(all_member_changes)
            member_kurt = torch.mean((mc - mc.mean()) ** 4) / (mc.std() ** 4 + 1e-8)
            kurtosis_ratio = (member_kurt / gt_kurt).item() if gt_kurt > 0 else 0.0
        else:
            kurtosis_ratio = kurtosis_ratio_mean
    else:
        kurtosis_ratio = 0.0
        kurtosis_ratio_mean = 0.0

    return {
        "coverage_90": coverage,
        "kurtosis_ratio": kurtosis_ratio,
        "kurtosis_ratio_mean": kurtosis_ratio_mean,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train afCRPS single-pass Block-AR")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to pretrained DDPM checkpoint (for encoder + optional weights)")
    parser.add_argument("--no_ema", action="store_true",
                        help="Use model_state_dict instead of ema_params")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Random init for decoder (only encoder from pretrained)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--n_members", type=int, default=4, help="K ensemble members per step")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (used for scratch init)")
    parser.add_argument("--lr_noise", type=float, default=1e-3,
                        help="LR for noise MLP (new params)")
    parser.add_argument("--lr_decoder", type=float, default=1e-4,
                        help="LR for decoder (pretrained params)")
    parser.add_argument("--lambda_vs", type=float, default=0.1,
                        help="Variogram score weight")
    parser.add_argument("--shared_noise_input", action="store_true",
                        help="Inject first noise element as shared spatial input (cross-cell correlation)")
    parser.add_argument("--lambda_is", type=float, default=0.0,
                        help="Interval score weight (CI calibration pressure)")
    parser.add_argument("--lambda_cs_reg", type=float, default=0.0,
                        help="L2 penalty pulling cell_scale toward its spatial mean")
    parser.add_argument("--lambda_kurt", type=float, default=0.0,
                        help="Kurtosis matching loss weight")
    parser.add_argument("--n_train_blocks", type=int, default=1,
                        help="Number of AR blocks to generate during training (1=block1 only, 3=full 30 frames)")
    parser.add_argument("--direct_iv", action="store_true",
                        help="Direct IV prediction (no exp/baseline transform)")
    parser.add_argument("--no_tanh", action="store_true",
                        help="Remove tanh bounding from decoder output")
    parser.add_argument("--learned_vol_scale", action="store_true",
                        help="Per-cell vol_scale from condition MLP (replaces scalar vol_scale)")
    parser.add_argument("--cond_noise_mlp", action="store_true",
                        help="Feed condition into noise MLP for regime-dependent diversity")
    parser.add_argument("--noise_dist", type=str, default="gaussian",
                        choices=["gaussian", "student_t"],
                        help="Noise distribution for ensemble diversity")
    parser.add_argument("--student_t_df", type=float, default=4.0,
                        help="Degrees of freedom for Student-t noise")
    parser.add_argument("--twcrps_beta", type=float, default=0.0,
                        help="twCRPS beta (0=standard, 2.0=3x weight at ±1 IQR)")
    parser.add_argument("--ar_frame", action="store_true",
                        help="Use per-frame AR decoder instead of Conv3D")
    parser.add_argument("--progressive_rollout", action="store_true",
                        help="Progressive training: 5→15→30 frames across epochs")
    parser.add_argument("--ar_cell_spread", action="store_true",
                        help="Learned per-cell spread scaling for AR frame decoder")
    parser.add_argument("--ar_static_cell_scale", action="store_true",
                        help="Static per-cell scale (nn.Parameter, no condition dependence)")
    parser.add_argument("--ar_bias_lambda", type=float, default=0.0,
                        help="Delta zero-mean bias loss weight")
    parser.add_argument("--ar_frame_hidden", type=int, default=128,
                        help="FrameDecoder MLP hidden dim")
    parser.add_argument("--ar_percell_vol_scale", action="store_true",
                        help="Per-cell vol_scale from history daily change std")
    parser.add_argument("--unfreeze_encoder", action="store_true",
                        help="Unfreeze GRU encoder")
    parser.add_argument("--lr_encoder", type=float, default=1e-4,
                        help="LR for encoder when unfrozen (default: 1e-4)")
    parser.add_argument("--no_pretrained_encoder", action="store_true",
                        help="Skip loading pretrained encoder weights (random init)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--n_eval_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="models/backfill/afcrps_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load base checkpoint to get encoder config
    if args.base_model:
        base_ckpt = torch.load(args.base_model, map_location="cpu", weights_only=False)
        base_cfg = base_ckpt.get("config", {})
    else:
        base_ckpt = None
        base_cfg = {}

    # Build SinglePassConfig
    config = SinglePassConfig(
        history_len=base_cfg.get("history_len", 30),
        future_len=base_cfg.get("future_len", 30),
        surface_h=base_cfg.get("surface_h", 5),
        surface_w=base_cfg.get("surface_w", 5),
        block_size=base_cfg.get("block_size", 10),
        gru_hidden_dim=base_cfg.get("gru_hidden_dim", 64),
        bottleneck_dim=base_cfg.get("bottleneck_dim", 128),
        encoder_dropout=base_cfg.get("encoder_dropout", 0.1),
        conv3d_base_channels=base_cfg.get("conv3d_base_channels", 32),
        conv3d_n_res_blocks=base_cfg.get("conv3d_n_res_blocks", 6),
        conv3d_groups=base_cfg.get("conv3d_groups", 8),
        pos_embed_dim=base_cfg.get("pos_embed_dim", 16),
        noise_dim=args.noise_dim,
        noise_embed_dim=base_cfg.get("conv3d_noise_embed_dim", 64),
        shared_noise_input=args.shared_noise_input,
        cond_noise_mlp=args.cond_noise_mlp,
        noise_dist=args.noise_dist,
        student_t_df=args.student_t_df,
        global_mean_vol=base_cfg.get("global_mean_vol", 0.0187),
        vol_scale_min=base_cfg.get("vol_scale_min", 0.5),
        vol_scale_max=base_cfg.get("vol_scale_max", 2.0),
        vol_scale_power=base_cfg.get("vol_scale_power", 1.0),
        direct_iv=args.direct_iv,
        no_tanh=args.no_tanh,
        learned_vol_scale=args.learned_vol_scale,
        twcrps_beta=args.twcrps_beta,
        ar_frame=args.ar_frame,
        ar_frame_cell_spread=args.ar_cell_spread,
        ar_frame_static_cell_scale=args.ar_static_cell_scale,
        ar_frame_hidden=args.ar_frame_hidden,
        ar_frame_bias_lambda=args.ar_bias_lambda,
        ar_frame_percell_vol_scale=args.ar_percell_vol_scale,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Create model
    model = SinglePassBlockAR(config).to(device)

    # Load pretrained weights
    if args.no_pretrained_encoder:
        print("Random encoder init (no pretrained weights loaded)")
    elif args.from_scratch:
        # Only load encoder weights
        src_state = base_ckpt["model_state_dict"] if args.no_ema else base_ckpt.get("ema_params", base_ckpt["model_state_dict"])
        tgt_state = model.state_dict()
        enc_transferred = 0
        for key, val in src_state.items():
            if key.startswith("encoder."):
                if key in tgt_state and tgt_state[key].shape == val.shape:
                    tgt_state[key] = val
                    enc_transferred += 1
        model.load_state_dict(tgt_state)
        print(f"Scratch init: transferred {enc_transferred} encoder params")
    else:
        stats = load_pretrained_weights(model, args.base_model, device="cpu", use_ema=not args.no_ema)
        print(f"Pretrained init: {stats}")
    model = model.to(device)

    # Re-init conv_out for direct IV mode (pretrained weights learned z-scores for exp())
    if args.direct_iv and hasattr(model, 'decoder'):
        nn.init.normal_(model.decoder.conv_out.weight, std=0.01)
        nn.init.zeros_(model.decoder.conv_out.bias)
        print("  Re-initialized conv_out for direct IV mode")

    # Freeze encoder (unless --unfreeze_encoder)
    if not args.unfreeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    enc_status = "encoder trainable" if args.unfreeze_encoder else "encoder frozen"
    print(f"Model: {n_total:,} total, {n_trainable:,} trainable ({enc_status})")

    # Optimizer with differential learning rates
    if args.from_scratch:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    elif args.ar_frame:
        # AR frame mode: only frame_decoder params (no NoiseMLP, no Conv3D decoder)
        decoder_params = list(model.frame_decoder.parameters())
        param_groups = [
            {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
        ]
        if args.unfreeze_encoder:
            encoder_params = list(model.encoder.parameters())
            param_groups.append(
                {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": 1e-4},
            )
        if hasattr(model, 'cell_spread_linear'):
            spread_params = list(model.cell_spread_linear.parameters())
            param_groups.append(
                {"params": spread_params, "lr": args.lr_decoder, "weight_decay": 0.1},
            )
        if hasattr(model, 'cell_scale'):
            param_groups.append(
                {"params": [model.cell_scale], "lr": 1e-3, "weight_decay": 0.0},
            )
        optimizer = torch.optim.AdamW(param_groups)
    else:
        noise_params = list(model.noise_mlp.parameters())
        spread_params = []
        if hasattr(model, 'cell_spread_mlp'):
            spread_params = list(model.cell_spread_mlp.parameters())
        decoder_params = [p for n, p in model.decoder.named_parameters()
                          if p.requires_grad]
        param_groups = [
            {"params": noise_params, "lr": args.lr_noise, "weight_decay": 1e-4},
            {"params": decoder_params, "lr": args.lr_decoder, "weight_decay": 1e-4},
        ]
        if spread_params:
            param_groups.append({"params": spread_params, "lr": args.lr_noise, "weight_decay": 0.1})
        optimizer = torch.optim.AdamW(param_groups)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Dataset
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    train_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=0, end_idx=4040,
    )
    val_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=4040, end_idx=4540,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Data: {len(train_dataset)} train, {len(val_dataset)} val")

    # Precompute per-cell median/IQR for twCRPS
    if args.twcrps_beta > 0:
        train_surfaces = surfaces[0:4040]  # raw IV [0, 1]
        cell_median = torch.from_numpy(np.median(train_surfaces, axis=0)).float()
        q75 = torch.from_numpy(np.percentile(train_surfaces, 75, axis=0)).float()
        q25 = torch.from_numpy(np.percentile(train_surfaces, 25, axis=0)).float()
        cell_iqr = (q75 - q25).clamp(min=0.01)  # prevent division by tiny IQR
        model.cell_median.copy_(cell_median.to(device))
        model.cell_iqr.copy_(cell_iqr.to(device))
        print(f"  twCRPS beta={args.twcrps_beta}")
        print(f"  cell_median: [{cell_median.min():.3f}, {cell_median.max():.3f}]")
        print(f"  cell_iqr: [{cell_iqr.min():.3f}, {cell_iqr.max():.3f}]")

    # Precompute target kurtosis for kurtosis matching loss
    if args.lambda_kurt > 0:
        train_surfaces = surfaces[0:4040]
        daily_changes = np.diff(train_surfaces, axis=0)  # (N-1, 5, 5)
        m2 = (daily_changes ** 2).mean(axis=0)
        m4 = (daily_changes ** 4).mean(axis=0)
        target_kurt = torch.from_numpy(m4 / (m2 ** 2 + 1e-8)).float()
        model.target_kurt.copy_(target_kurt.to(device))
        print(f"  Kurtosis matching: lambda={args.lambda_kurt}")
        print(f"  Target kurtosis per cell: [{target_kurt.min():.1f}, {target_kurt.max():.1f}]")

    # Training
    best_val_loss = float("inf")
    best_coverage = 0.0
    history_log = []

    print(f"\n{'='*70}")
    print(f"Training afCRPS single-pass model")
    print(f"  noise_dim={config.noise_dim}, n_members={args.n_members}, n_train_blocks={args.n_train_blocks}")
    print(f"  lambda_vs={args.lambda_vs}, from_scratch={args.from_scratch}")
    if args.ar_frame:
        print(f"  AR FRAME MODE: rho={config.ar_frame_rho}, hidden={config.ar_frame_hidden}, progressive={args.progressive_rollout}, encoder={'unfrozen' if args.unfreeze_encoder else 'frozen'}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Progressive rollout: gradually increase generated frames
        if args.progressive_rollout:
            n_frames = 5 if epoch <= 10 else 15 if epoch <= 20 else 30
        else:
            n_frames = 30 if args.ar_frame else 0  # 0 = use block-based n_frames

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            n_members=args.n_members, lambda_vs=args.lambda_vs,
            grad_clip=args.grad_clip, n_train_blocks=args.n_train_blocks,
            lambda_is=args.lambda_is, lambda_cs_reg=args.lambda_cs_reg,
            lambda_kurt=args.lambda_kurt,
            n_frames=n_frames,
            unfreeze_encoder=args.unfreeze_encoder,
        )
        lr_scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, device, n_members=args.n_members)

        # Quick eval (CI coverage, kurtosis)
        eval_metrics = {}
        if epoch % args.eval_every == 0:
            eval_metrics = quick_eval(
                model, val_loader, device,
                n_samples=args.n_eval_samples, max_batches=5,
            )

        elapsed = time.time() - t0

        # Log
        log_entry = {
            "epoch": epoch,
            "elapsed": elapsed,
            **train_metrics,
            **val_metrics,
            **eval_metrics,
        }
        history_log.append(log_entry)

        # Print
        spread_ratio = train_metrics["spread_mae_ratio"]
        eval_str = ""
        if eval_metrics:
            eval_str = f"  CI={eval_metrics['coverage_90']:.1%}  Kurt={eval_metrics['kurtosis_ratio']:.3f}(mean:{eval_metrics['kurtosis_ratio_mean']:.3f})"
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"mae={train_metrics['mae']:.4f}  "
            f"spread={train_metrics['spread']:.6f}  "
            f"s/m={spread_ratio:.4f}  "
            f"vs={train_metrics['variogram']:.4f}  "
            f"val={val_metrics['val_loss']:.4f}"
            f"{eval_str}  "
            f"({elapsed:.1f}s)"
        )

        # Log kurtosis matching if applicable
        if args.lambda_kurt > 0:
            print(f"  kurt_loss={train_metrics['kurt_loss']:.4f}  raw_kurt={train_metrics['raw_kurt']:.2f}")

        # Log frame_decoder stats if applicable
        if hasattr(model, 'frame_decoder'):
            w = model.frame_decoder.mlp[-1].weight.detach()
            print(f"  frame_decoder: w_norm={w.norm():.3f}" +
                  (f"  n_frames={n_frames}" if args.progressive_rollout else ""))

        # Log cell_scale stats if applicable (static per-cell scale)
        if hasattr(model, 'cell_scale'):
            cs = model.cell_scale.detach().clamp(0.3, 3.0)
            print(f"  cell_scale: [{cs.min():.3f}, {cs.max():.3f}] mean={cs.mean():.3f}")

        # Log bias loss if applicable
        if 'bias_loss' in train_metrics and train_metrics['bias_loss'] > 0:
            print(f"  bias_loss: {train_metrics['bias_loss']:.6f}")

        # Log cell_spread_linear stats if applicable (AR frame mode)
        if hasattr(model, 'cell_spread_linear'):
            w = model.cell_spread_linear.weight.detach()
            b = model.cell_spread_linear.bias.detach()
            base_out = F.softplus(b)
            print(f"  cell_spread: out=[{base_out.min():.3f}, {base_out.max():.3f}] w_norm={w.norm():.3f}")

        # Log cell_spread MLP stats if applicable
        if hasattr(model, 'cell_spread_mlp'):
            # Find the last Linear layer in the Sequential
            linear_layers = [m for m in model.cell_spread_mlp if isinstance(m, nn.Linear)]
            if linear_layers:
                w = linear_layers[-1].weight.detach()
                b = linear_layers[-1].bias.detach()
                base_out = F.softplus(b)
                print(f"  cell_spread: bias_out=[{base_out.min():.3f}, {base_out.max():.3f}] w_norm={w.norm():.3f}")

        # Save checkpoint dict for potential saving
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dataclasses.asdict(config),
            "metrics": log_entry,
            "training_config": {
                "base_model": args.base_model,
                "from_scratch": args.from_scratch,
                "n_members": args.n_members,
                "lambda_vs": args.lambda_vs,
                "lr_noise": args.lr_noise,
                "lr_decoder": args.lr_decoder,
                "noise_dim": args.noise_dim,
                "n_train_blocks": args.n_train_blocks,
            },
        }

        # Early stopping checks (only on member kurtosis, and only if very low)
        if eval_metrics.get("kurtosis_ratio", 1.0) < 0.1 and epoch >= 10:
            print(f"EARLY STOP: member kurtosis {eval_metrics['kurtosis_ratio']:.3f} < 0.1 at epoch {epoch}")
            break
        if epoch >= 5 and spread_ratio < 0.05:
            print(f"WARNING: spread/MAE ratio {spread_ratio:.4f} < 0.05 — noise injection may not be working")

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(save_dict, f"{args.output_dir}/best_model.pt")
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        if eval_metrics.get("coverage_90", 0) > best_coverage:
            best_coverage = eval_metrics["coverage_90"]
            torch.save(save_dict, f"{args.output_dir}/best_coverage_model.pt")
            print(f"  → Saved best coverage model (coverage={best_coverage:.1%})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(save_dict, f"{args.output_dir}/checkpoint_epoch_{epoch}.pt")

    # Save final
    torch.save(save_dict, f"{args.output_dir}/final_model.pt")

    # Save training history
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}, best coverage={best_coverage:.1%}")
    print(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
