#!/usr/bin/env python
"""
Learn per-cell noise sigma for AR flow matching model.

Freeze the velocity net, optimize 25 noise parameters to maximize
CI coverage on validation set. The noise is applied per-cell in
standardized space after each ODE step.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/learn_flow_noise.py \
        --model_path models/backfill/flow_152b/best_model.pt \
        --output_dir models/backfill/flow_152b_percell \
        --iters 200 --lr 0.01 --device cuda
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig, normalize_iv, denormalize_iv,
)
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.block_ar.train_ar_flow import (
    ConditionalVelocityMLP, ARFlowMatchingModel,
)
from experiments.backfill.block_ar.eval_ar_flow import load_ar_flow_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--target_ci", type=float, default=0.90)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Learn Per-Cell Noise for AR Flow Matching")
    print("=" * 60)

    # Load frozen model
    model, ckpt = load_ar_flow_model(args.model_path, device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Per-cell log-sigma: 25 learnable parameters
    # Init at log(0.05) ≈ -3.0 (small noise to start)
    log_sigma = nn.Parameter(torch.full((25,), math.log(0.05), device=device))
    optimizer = torch.optim.Adam([log_sigma], lr=args.lr)

    # Load validation data
    config = get_default_config()
    data = np.load(config.data_path)
    surfaces = data["surface"]
    val_ds = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=4040, end_idx=4540,
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, drop_last=True)

    print(f"  Val windows: {len(val_ds)}")
    print(f"  Target CI: {args.target_ci:.0%}")
    print(f"  Initial sigma: {torch.exp(log_sigma).mean():.4f}")

    def sample_with_percell_noise(model, history, n_samples, log_sigma):
        """Generate samples with per-cell learned noise."""
        B = history.shape[0]
        device = history.device
        dt = 1.0 / model.n_steps
        sigma = torch.exp(log_sigma)  # (25,)

        condition = model.encoder(history)
        BS = B * n_samples
        cond = condition.unsqueeze(1).expand(B, n_samples, -1).reshape(BS, -1)

        history_iv = denormalize_iv(history)
        prev = history_iv[:, -1].reshape(B, 25)
        prev = prev.unsqueeze(1).expand(B, n_samples, 25).reshape(BS, 25)
        prev_std = model.standardize(prev)

        all_frames = []
        for frame_idx in range(model.future_len):
            x = torch.randn(BS, 25, device=device)
            for step in range(model.n_steps):
                t = torch.full((BS,), step * dt, device=device)
                v = model.velocity_net(x, t, cond, prev_std)
                x = x + v * dt

            # Per-cell noise in standardized space
            x = x + sigma.unsqueeze(0) * torch.randn_like(x)

            frame_iv = model.destandardize(x).clamp(0, 1)
            all_frames.append(frame_iv)
            prev_std = model.standardize(frame_iv)

        frames = torch.stack(all_frames, dim=1)
        return frames.reshape(B, n_samples, model.future_len, 5, 5)

    def ci_loss(samples, gt, target=0.90):
        """Differentiable CI coverage loss.

        Interval Score: width + (2/alpha) * overshoot
        This jointly optimizes width and coverage.
        """
        B, K, T, H, W = samples.shape
        alpha = 1.0 - target  # 0.10 for 90% CI

        # Quantiles
        lo_idx = max(0, int(K * alpha / 2) - 1)
        hi_idx = min(K - 1, int(K * (1 - alpha / 2)))

        sorted_samples, _ = samples.sort(dim=1)
        lo = sorted_samples[:, lo_idx]  # (B, T, H, W)
        hi = sorted_samples[:, hi_idx]

        width = hi - lo  # (B, T, H, W)
        overshoot_lo = torch.clamp(lo - gt, min=0)  # gt below lower bound
        overshoot_hi = torch.clamp(gt - hi, min=0)  # gt above upper bound

        penalty = (2.0 / alpha) * (overshoot_lo + overshoot_hi)
        return (width + penalty).mean()

    # Training loop
    best_loss = float("inf")
    for it in range(1, args.iters + 1):
        batch = next(iter(val_loader))
        hist = batch["history"].to(device)
        future = batch["future"].to(device)
        gt = denormalize_iv(future)

        samples = sample_with_percell_noise(model, hist, args.n_samples, log_sigma)

        loss = ci_loss(samples, gt, target=args.target_ci)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp sigma to reasonable range [0.001, 1.0]
        with torch.no_grad():
            log_sigma.clamp_(math.log(0.001), math.log(1.0))

        if it % 20 == 0 or it == 1:
            sigma = torch.exp(log_sigma).detach()
            # Quick CI check
            with torch.no_grad():
                test_samples = sample_with_percell_noise(
                    model, hist, args.n_samples, log_sigma)
                in_ci = ((gt.unsqueeze(1) >= test_samples.quantile(0.05, dim=1).unsqueeze(1)) &
                         (gt.unsqueeze(1) <= test_samples.quantile(0.95, dim=1).unsqueeze(1)))
                ci_pct = in_ci.float().mean().item()

            print(f"Iter {it:3d}/{args.iters}  loss={loss.item():.4f}  "
                  f"CI≈{ci_pct:.1%}  "
                  f"sigma=[{sigma.min():.4f}, {sigma.max():.4f}] "
                  f"mean={sigma.mean():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()

    # Save learned sigma
    sigma_final = torch.exp(log_sigma).detach().cpu()
    print(f"\nFinal per-cell sigma (5x5 grid):")
    grid = sigma_final.reshape(5, 5).numpy()
    for row in grid:
        print(f"  {' '.join(f'{v:.4f}' for v in row)}")

    torch.save({
        "log_sigma": log_sigma.detach().cpu(),
        "sigma": sigma_final,
        "sigma_grid": grid,
        "best_loss": best_loss,
        "config": {
            "iters": args.iters, "lr": args.lr,
            "target_ci": args.target_ci, "n_samples": args.n_samples,
        },
    }, f"{args.output_dir}/percell_sigma.pt")

    # Copy the base model checkpoint with sigma attached
    import shutil
    shutil.copy(args.model_path, f"{args.output_dir}/best_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump({"final_sigma": grid.tolist(), "best_loss": best_loss}, f, indent=2)

    print(f"\nSaved to {args.output_dir}")
    print(f"Best IS loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
