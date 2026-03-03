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

def train_epoch(model, loader, optimizer, device, n_members, lambda_vs, grad_clip):
    model.train()
    # Keep encoder in eval mode (frozen, no dropout)
    model.encoder.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_spread = 0.0
    total_vs = 0.0
    n_batches = 0

    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)

        result = model(history, future, n_members=n_members, lambda_vs=lambda_vs)
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
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae": total_mae / max(n_batches, 1),
        "spread": total_spread / max(n_batches, 1),
        "variogram": total_vs / max(n_batches, 1),
        "spread_mae_ratio": total_spread / max(total_mae, 1e-8),
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
    parser.add_argument("--base_model", type=str, required=True,
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
    base_ckpt = torch.load(args.base_model, map_location="cpu", weights_only=False)
    base_cfg = base_ckpt.get("config", {})

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
        global_mean_vol=base_cfg.get("global_mean_vol", 0.0187),
        vol_scale_min=base_cfg.get("vol_scale_min", 0.5),
        vol_scale_max=base_cfg.get("vol_scale_max", 2.0),
        vol_scale_power=base_cfg.get("vol_scale_power", 1.0),
        output_dir=args.output_dir,
        device=args.device,
    )

    # Create model
    model = SinglePassBlockAR(config).to(device)

    # Load pretrained weights
    if args.from_scratch:
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

    # Freeze encoder
    for name, param in model.named_parameters():
        if name.startswith("encoder."):
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total:,} total, {n_trainable:,} trainable (encoder frozen)")

    # Optimizer with differential learning rates
    if args.from_scratch:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    else:
        noise_params = list(model.noise_mlp.parameters())
        decoder_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": noise_params, "lr": args.lr_noise},
            {"params": decoder_params, "lr": args.lr_decoder},
        ], weight_decay=1e-4)

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

    # Training
    best_val_loss = float("inf")
    best_coverage = 0.0
    history_log = []

    print(f"\n{'='*70}")
    print(f"Training afCRPS single-pass model")
    print(f"  noise_dim={config.noise_dim}, n_members={args.n_members}")
    print(f"  lambda_vs={args.lambda_vs}, from_scratch={args.from_scratch}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            n_members=args.n_members, lambda_vs=args.lambda_vs,
            grad_clip=args.grad_clip,
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
