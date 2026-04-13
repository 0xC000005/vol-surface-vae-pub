#!/usr/bin/env python
"""
225a: Learned uncertainty head on 183c.

Adds a 2-layer MLP scale head to 183c's encoder that outputs a per-window
uncertainty multiplier. Train with coverage floor penalty + sharpness reward,
freezing the base 183c model. Only the uncertainty head is trained.

The scale head inflates the Cholesky factors of the covariance by sqrt(scale),
so the covariance is multiplied by scale. When scale > 1, the model widens;
when scale ~= 1, it stays at 183c's calibration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_183c_best_mechanism import (
    load_model as load_183c_model,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    build_rollout_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    normalize_iv,
)


class UncertaintyHead(nn.Module):
    """Predicts a per-window scalar uncertainty multiplier from encoder output."""

    def __init__(self, cond_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Init to output ~0 so softplus(0) = ln(2) ≈ 0.69, and 1 + 0.69 = 1.69
        # We want to start near scale=1, so init bias to produce softplus ≈ 0
        # softplus(x) ≈ 0 when x << 0, so init last bias to -2
        nn.init.zeros_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, -3.0)  # softplus(-3) ≈ 0.049, so 1 + 0.049 ≈ 1.05

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Returns scale multiplier >= 1.0 for covariance inflation."""
        raw = self.net(cond).squeeze(-1)  # (B,)
        return 1.0 + F.softplus(raw)  # >= 1.0, starts near 1.05


def compute_per_window_coverage(
    samples_flat: torch.Tensor,
    gt_flat: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    """Compute per-window coverage at (alpha, 1-alpha) quantiles.

    Args:
        samples_flat: (B, K, T*25) generated samples
        gt_flat: (B, T*25) ground truth

    Returns:
        coverage: (B,) fraction of (horizon, cell) pairs covered
    """
    lo = torch.quantile(samples_flat, alpha, dim=1)  # (B, T*25)
    hi = torch.quantile(samples_flat, 1 - alpha, dim=1)  # (B, T*25)
    covered = (gt_flat >= lo) & (gt_flat <= hi)
    return covered.float().mean(dim=1)  # (B,)


def main():
    parser = argparse.ArgumentParser(description="225a: Learned uncertainty head on 183c")
    parser.add_argument("--checkpoint", type=str,
                        default="models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=192)
    parser.add_argument("--n_samples", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--coverage_target", type=float, default=0.70,
                        help="Minimum per-window coverage target for floor penalty")
    parser.add_argument("--lambda_floor", type=float, default=10.0,
                        help="Weight for coverage floor penalty")
    parser.add_argument("--lambda_sharp", type=float, default=1.0,
                        help="Weight for sharpness (penalizes excessive widening)")
    parser.add_argument("--output_dir", type=str, default="models/backfill/225a_uncertainty_head")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 183c (frozen)
    model, payload = load_183c_model(args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Build validation windows
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=30, future_len=30,
        test_start=args.test_start, val_size=args.val_size,
        max_windows=None, device=device, split="val",
    )
    N = batch.history_01.shape[0]
    history_01 = batch.history_01
    future_01 = batch.future_01
    gt_flat = future_01.reshape(N, -1)  # (N, 30*5*5 = 750)

    # Pre-compute encoder outputs (frozen)
    print("Pre-computing encoder outputs...")
    with torch.no_grad():
        cond_all = model.encode(history_01)  # (N, 128)
    print(f"Encoder outputs: {cond_all.shape}")

    # Pre-generate base samples (frozen model, no scale applied)
    print(f"Pre-generating {args.n_samples} base samples...")
    base_samples = []
    for start in range(0, N, 16):
        end = min(start + 16, N)
        hist_norm = normalize_iv(history_01[start:end])
        with torch.no_grad():
            samp = model.sample_batched(hist_norm, n_samples=args.n_samples, n_steps=30, chunk_size=8)
        base_samples.append(samp)
    base_samples = torch.cat(base_samples, dim=0)  # (N, K, 30, 5, 5)
    base_samples_flat = base_samples.reshape(N, args.n_samples, -1)  # (N, K, 750)
    print(f"Base samples: {base_samples_flat.shape}")

    # Compute base sample center (median)
    base_median = base_samples_flat.median(dim=1).values  # (N, 750)

    # Uncertainty head
    head = UncertaintyHead(cond_dim=128, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    print(f"\n225a: Training uncertainty head")
    print(f"  epochs={args.epochs}, lr={args.lr}")
    print(f"  coverage_target={args.coverage_target}, lambda_floor={args.lambda_floor}, lambda_sharp={args.lambda_sharp}")
    print(f"  N={N}, n_samples={args.n_samples}")

    history_log: list[dict[str, Any]] = []
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        head.train()

        # Forward: compute scale for all windows
        scale = head(cond_all)  # (N,) >= 1.0

        # Apply scale: inflate samples around median
        # scaled_sample = median + scale * (sample - median)
        scale_expand = scale[:, None, None]  # (N, 1, 1)
        scaled_samples = base_median.unsqueeze(1) + scale_expand * (base_samples_flat - base_median.unsqueeze(1))

        # Coverage floor loss: penalize windows where coverage < target
        coverage = compute_per_window_coverage(scaled_samples, gt_flat, alpha=0.05)
        floor_loss = F.relu(args.coverage_target - coverage).mean()

        # Sharpness loss: penalize excessive scale (want scale close to 1.0)
        sharpness_loss = (scale - 1.0).pow(2).mean()

        loss = args.lambda_floor * floor_loss + args.lambda_sharp * sharpness_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            mean_cov = coverage.mean().item()
            worst_cov = coverage.min().item()
            below_70 = (coverage < 0.70).sum().item()
            below_80 = (coverage < 0.80).sum().item()
            mean_scale = scale.mean().item()
            max_scale = scale.max().item()
            min_scale = scale.min().item()

        record = {
            "epoch": epoch,
            "loss": loss.item(),
            "floor_loss": floor_loss.item(),
            "sharp_loss": sharpness_loss.item(),
            "mean_cov": mean_cov,
            "worst_cov": worst_cov,
            "below_70": int(below_70),
            "below_80": int(below_80),
            "mean_scale": mean_scale,
            "max_scale": max_scale,
            "min_scale": min_scale,
            "elapsed": time.time() - t0,
        }
        history_log.append(make_serializable(record))

        # Selection: minimize below_70 count, then maximize sharpness
        metric = below_70 * 100 + (mean_scale - 1.0) * 10
        if metric < best_metric:
            best_metric = metric
            torch.save({
                "head_state_dict": head.state_dict(),
                "epoch": epoch,
                "metrics": record,
            }, out_dir / "best_head.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={loss.item():.4f} floor={floor_loss.item():.4f} sharp={sharpness_loss.item():.4f} "
                f"cov={mean_cov:.3f} worst={worst_cov:.3f} <70%={int(below_70)} <80%={int(below_80)} "
                f"scale={mean_scale:.3f} [{min_scale:.3f}, {max_scale:.3f}]"
            )

    # Save final
    torch.save({
        "head_state_dict": head.state_dict(),
        "epoch": args.epochs,
        "metrics": history_log[-1],
        "base_checkpoint": args.checkpoint,
    }, out_dir / "final_head.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    # Final evaluation with best head
    best_payload = torch.load(out_dir / "best_head.pt", map_location=device, weights_only=False)
    head.load_state_dict(best_payload["head_state_dict"])
    head.eval()

    with torch.no_grad():
        scale_final = head(cond_all)
        scale_expand = scale_final[:, None, None]
        scaled_final = base_median.unsqueeze(1) + scale_expand * (base_samples_flat - base_median.unsqueeze(1))
        cov_final = compute_per_window_coverage(scaled_final, gt_flat, alpha=0.05)

        # Width comparison
        raw_lo = torch.quantile(base_samples_flat, 0.05, dim=1)
        raw_hi = torch.quantile(base_samples_flat, 0.95, dim=1)
        raw_width = (raw_hi - raw_lo).mean().item()

        scaled_lo = torch.quantile(scaled_final, 0.05, dim=1)
        scaled_hi = torch.quantile(scaled_final, 0.95, dim=1)
        scaled_width = (scaled_hi - scaled_lo).mean().item()

    print(f"\n=== Final Results (best head) ===")
    print(f"  Mean coverage: {cov_final.mean().item():.3f} (raw: {compute_per_window_coverage(base_samples_flat, gt_flat).mean().item():.3f})")
    print(f"  Worst window:  {cov_final.min().item():.3f}")
    print(f"  Windows < 70%: {(cov_final < 0.70).sum().item()}")
    print(f"  Windows < 80%: {(cov_final < 0.80).sum().item()}")
    print(f"  Mean scale:    {scale_final.mean().item():.3f}")
    print(f"  Scale range:   [{scale_final.min().item():.3f}, {scale_final.max().item():.3f}]")
    print(f"  Width increase:{(scaled_width/raw_width - 1)*100:+.1f}%")

    # Show which windows get highest scale
    top_k = 10
    top_idx = scale_final.topk(top_k).indices.cpu().numpy()
    print(f"\n  Top {top_k} highest-scale windows:")
    for idx in top_idx:
        print(f"    win {idx:3d}: scale={scale_final[idx].item():.3f} raw_cov={compute_per_window_coverage(base_samples_flat[idx:idx+1], gt_flat[idx:idx+1]).item():.3f} scaled_cov={cov_final[idx].item():.3f}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
