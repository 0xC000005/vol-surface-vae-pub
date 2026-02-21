"""Train the learned uncertainty head on a frozen Block-AR generator.

Two-phase approach for efficiency:
  Phase 1: Generate samples from frozen generator and cache to disk.
  Phase 2: Train lightweight MLP on cached data (very fast, no diffusion sampling).

Usage:
    # Phase 1: Generate and cache samples (~2 hours)
    python experiments/backfill/block_ar/train_uncertainty_head.py cache \
        --generator_path models/backfill/block_ar_taskprob_B/best_coverage_model.pt \
        --n_samples 20 --cache_dir data/uncertainty_cache

    # Phase 2: Train head on cached data (~5 minutes)
    python experiments/backfill/block_ar/train_uncertainty_head.py train \
        --cache_dir data/uncertainty_cache \
        --epochs 100 --lr 1e-3 \
        --output_dir models/backfill/block_ar_uncertainty_head
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    UncertaintyHead,
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def ensemble_crps(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Compute CRPS for ensemble predictions (energy form).

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        samples: (B, M, ...) ensemble of M samples
        truth: (B, ...) ground truth

    Returns:
        scalar CRPS, averaged over batch and spatial dims
    """
    B, M = samples.shape[:2]

    truth_expanded = truth.unsqueeze(1)
    reliability = (samples - truth_expanded).abs().mean(dim=1)  # (B, ...)

    if M <= 50:
        diff = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs()
        resolution = diff.sum(dim=(1, 2)) / (M * M)
    else:
        idx1 = torch.randint(0, M, (M,), device=samples.device)
        idx2 = torch.randint(0, M, (M,), device=samples.device)
        resolution = (samples[:, idx1] - samples[:, idx2]).abs().mean(dim=1)

    crps = reliability - 0.5 * resolution
    return crps.mean()


def interval_score_loss(
    samples: torch.Tensor,
    truth: torch.Tensor,
    alpha: float = 0.10,
) -> torch.Tensor:
    """Interval score for alpha-level prediction interval.

    IS_α = (hi - lo) + (2/α) * [max(lo - y, 0) + max(y - hi, 0)]

    The (2/α) penalty means each miss costs 20x (for α=0.10) the width savings,
    strongly incentivizing correct coverage. This is a proper scoring rule.

    Args:
        samples: (B, M, ...) ensemble of M samples
        truth: (B, ...) ground truth
        alpha: significance level (0.10 for 90% CI)

    Returns:
        scalar interval score, averaged over batch and spatial dims
    """
    lo = samples.quantile(alpha / 2, dim=1)      # 5th percentile
    hi = samples.quantile(1 - alpha / 2, dim=1)  # 95th percentile

    width = hi - lo
    undershoot = F.relu(lo - truth)   # truth below lower bound
    overshoot = F.relu(truth - hi)    # truth above upper bound

    penalty = (2.0 / alpha) * (undershoot + overshoot)
    return (width + penalty).mean()


# ─── Phase 1: Generate and cache samples ────────────────────────────────

def cache_samples(args):
    """Generate diffusion samples and encoder conditions, save to disk."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load generator
    print(f"Loading generator from {args.generator_path}")
    checkpoint = torch.load(args.generator_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if dataclasses.is_dataclass(config):
        config_dict = dataclasses.asdict(config)
    else:
        config_dict = config

    gen_config = BlockARConfig(**{
        k: v for k, v in config_dict.items()
        if k in {f.name for f in dataclasses.fields(BlockARConfig)}
    })

    model = ConditionalBlockARDDPM(gen_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data (same split as training)
    print(f"Loading data from {args.data_path}")
    data = np.load(args.data_path)
    surfaces = data["surface"]

    from experiments.backfill.block_ar.config_block_ar import BlockARPOCConfig
    poc_config = BlockARPOCConfig()

    for split_name, start_idx, end_idx in [
        ("train", 0, poc_config.train_end),
        ("val", poc_config.val_start, poc_config.val_end),
    ]:
        dataset = VolSurfaceDataset(
            surfaces, gen_config.history_len, gen_config.future_len,
            start_idx=start_idx, end_idx=end_idx,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        all_conditions = []
        all_samples = []  # in normalized space [-1, 1]
        all_futures = []  # ground truth in normalized space

        print(f"\nGenerating {split_name} samples ({len(dataset)} sequences, "
              f"n_samples={args.n_samples})...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                history = batch["history"].to(device)
                future = batch["future"].to(device)
                B = history.shape[0]

                # Get encoder condition
                cond = model.encoder(history, mask=None)
                if model.config.forward_only:
                    cond = cond + model.encoder.null_embedding.expand(B, -1)
                cond = model._augment_condition(cond, None)

                # Generate samples (returns denormalized [0,1])
                raw_samples = model.sample_batched(
                    history, n_samples=args.n_samples, max_global_residual=0,
                )
                # Re-normalize to [-1, 1]
                raw_norm = normalize_iv(raw_samples)

                all_conditions.append(cond.cpu())
                all_samples.append(raw_norm.cpu())
                all_futures.append(future.cpu())

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"  {split_name} batch {batch_idx+1}/{len(loader)}", flush=True)

        conditions = torch.cat(all_conditions, dim=0)
        samples = torch.cat(all_samples, dim=0)
        futures = torch.cat(all_futures, dim=0)

        cache_path = os.path.join(args.cache_dir, f"{split_name}.pt")
        torch.save({
            "conditions": conditions,     # (N, bottleneck_dim)
            "samples": samples,           # (N, n_samples, future_len, 5, 5) normalized
            "futures": futures,            # (N, future_len, 5, 5) normalized
            "n_samples": args.n_samples,
            "generator_path": args.generator_path,
        }, cache_path)
        print(f"  Saved {split_name}: {conditions.shape[0]} sequences → {cache_path}")
        print(f"  Cache size: {os.path.getsize(cache_path) / 1e6:.1f} MB")

    print("\nCache generation complete.")


# ─── Phase 2: Train head on cached data ─────────────────────────────────

def train_on_cache(args):
    """Train uncertainty head using pre-cached samples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load cached data
    print(f"Loading cached data from {args.cache_dir}")
    train_data = torch.load(os.path.join(args.cache_dir, "train.pt"), map_location="cpu",
                            weights_only=True)
    val_data = torch.load(os.path.join(args.cache_dir, "val.pt"), map_location="cpu",
                          weights_only=True)

    train_cond = train_data["conditions"]
    train_samples = train_data["samples"]
    train_futures = train_data["futures"]
    n_samples = train_data["n_samples"]

    val_cond = val_data["conditions"]
    val_samples = val_data["samples"]
    val_futures = val_data["futures"]

    cond_dim = train_cond.shape[1]
    future_len = train_futures.shape[1]

    loss_type = getattr(args, "loss_type", "interval")
    target_cov = getattr(args, "target_coverage", 0.90)

    print(f"Train: {train_cond.shape[0]} sequences, Val: {val_cond.shape[0]} sequences")
    print(f"n_samples={n_samples}, cond_dim={cond_dim}, future_len={future_len}")
    print(f"Loss type: {loss_type}, target coverage: {target_cov:.0%}")

    # Create dataloaders
    train_dataset = TensorDataset(train_cond, train_samples, train_futures)
    val_dataset = TensorDataset(val_cond, val_samples, val_futures)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create head
    head = UncertaintyHead(
        cond_dim=cond_dim,
        future_len=future_len,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_crps = float("inf")
    history_log = []

    print(f"Uncertainty head params: {sum(p.numel() for p in head.parameters()):,}")
    print(f"Training with epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # ---- Training ----
        head.train()
        train_losses = []

        for cond, samples, futures in train_loader:
            cond = cond.to(device)
            samples = samples.to(device)
            futures = futures.to(device)

            # Apply uncertainty head
            scale = head(cond)  # (B, future_len)
            scale_5d = scale[:, None, :, None, None]

            mean = samples.mean(dim=1, keepdim=True)
            deviations = samples - mean
            scaled = mean + scale_5d * deviations

            # Loss in denormalized space
            scaled_d = denormalize_iv(scaled)
            futures_d = denormalize_iv(futures)
            if loss_type == "crps":
                loss = ensemble_crps(scaled_d, futures_d)
            elif loss_type == "interval":
                loss = interval_score_loss(scaled_d, futures_d, alpha=1.0 - target_cov)
            else:  # combined
                loss = ensemble_crps(scaled_d, futures_d) + interval_score_loss(
                    scaled_d, futures_d, alpha=1.0 - target_cov
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)

        # ---- Validation ----
        head.eval()
        val_losses = []
        val_coverages = []

        with torch.no_grad():
            for cond, samples, futures in val_loader:
                cond = cond.to(device)
                samples = samples.to(device)
                futures = futures.to(device)

                scale = head(cond)
                scale_5d = scale[:, None, :, None, None]
                mean = samples.mean(dim=1, keepdim=True)
                scaled = mean + scale_5d * (samples - mean)

                scaled_d = denormalize_iv(scaled)
                futures_d = denormalize_iv(futures)

                vloss = interval_score_loss(scaled_d, futures_d, alpha=1.0 - target_cov)
                val_losses.append(vloss.item())

                lo = scaled_d.quantile(0.05, dim=1)
                hi = scaled_d.quantile(0.95, dim=1)
                covered = (futures_d >= lo) & (futures_d <= hi)
                val_coverages.append(covered.float().mean().item())

        avg_val = np.mean(val_losses)
        avg_cov = np.mean(val_coverages)

        # Get scale stats
        with torch.no_grad():
            s = head(val_cond[:16].to(device))
            scale_h0 = s[:, 0].mean().item()
            scale_h29 = s[:, -1].mean().item()
            ratio = scale_h29 / max(scale_h0, 1e-8)

        print(
            f"Epoch {epoch:3d} | "
            f"Train: {avg_train:.4f} | "
            f"Val: {avg_val:.4f} | "
            f"90% CI: {avg_cov:.1%} | "
            f"Scale: {scale_h0:.2f}→{scale_h29:.2f} ({ratio:.1f}x)",
            flush=True,
        )

        history_log.append({
            "epoch": epoch,
            "train_crps": avg_train,
            "val_crps": avg_val,
            "val_coverage_90": avg_cov,
            "scale_h0": scale_h0,
            "scale_h29": scale_h29,
            "scale_ratio": ratio,
        })

        if avg_val < best_val_crps:
            best_val_crps = avg_val
            torch.save({
                "head_state_dict": head.state_dict(),
                "config": {
                    "cond_dim": cond_dim,
                    "future_len": future_len,
                    "hidden_dim": args.hidden_dim,
                },
                "epoch": epoch,
                "val_loss": avg_val,
                "val_coverage_90": avg_cov,
                "loss_type": loss_type,
                "generator_path": train_data.get("generator_path", "unknown"),
            }, os.path.join(args.output_dir, "best_uncertainty_head.pt"))
            print(f"  ↑ New best val loss: {avg_val:.4f}")

        if epoch % 10 == 0:
            torch.save({
                "head_state_dict": head.state_dict(),
                "epoch": epoch,
            }, os.path.join(args.output_dir, f"uncertainty_head_epoch_{epoch}.pt"))

    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(history_log, f, indent=2)

    print("=" * 60)
    print(f"Best val loss ({loss_type}): {best_val_crps:.4f}")
    print(f"Saved to: {args.output_dir}")

    return head


def main():
    parser = argparse.ArgumentParser(description="Train uncertainty head")
    subparsers = parser.add_subparsers(dest="command")

    # Cache subcommand
    cache_parser = subparsers.add_parser("cache", help="Generate and cache diffusion samples")
    cache_parser.add_argument("--generator_path", type=str, required=True)
    cache_parser.add_argument("--n_samples", type=int, default=20)
    cache_parser.add_argument("--batch_size", type=int, default=8)
    cache_parser.add_argument("--cache_dir", type=str, default="data/uncertainty_cache")
    cache_parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train head on cached data")
    train_parser.add_argument("--cache_dir", type=str, default="data/uncertainty_cache")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--hidden_dim", type=int, default=64)
    train_parser.add_argument("--loss_type", type=str, default="interval",
                              choices=["crps", "interval", "combined"],
                              help="Loss function: interval (default, targets coverage), "
                                   "crps (sharpness-focused), combined (both)")
    train_parser.add_argument("--target_coverage", type=float, default=0.90,
                              help="Target coverage for interval score (default: 0.90)")
    train_parser.add_argument("--output_dir", type=str,
                              default="models/backfill/block_ar_uncertainty_head")

    args = parser.parse_args()

    if args.command == "cache":
        cache_samples(args)
    elif args.command == "train":
        train_on_cache(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
