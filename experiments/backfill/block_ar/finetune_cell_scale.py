#!/usr/bin/env python
"""
Experiment 61: Fine-tune per-cell CI width via Interval Score (IS) loss.

Approach:
1. Load frozen bestval model and generate N samples per window
2. Optimize log_correction (5x5 = 25 params) to minimize IS
3. Save learned correction values for use via --cell_scale_values

The correction modifies the ratio between samples and baseline:
  corrected = baseline * (samples/baseline)^correction
  correction > 1 → wider CI (amplify deviations from baseline)
  correction < 1 → narrower CI
  correction = 1 → no change (identity)

IS = width + (2/alpha) * overshoot — directly penalizes miscoverage.
The 20x penalty for misses (at alpha=0.1) strongly incentivizes coverage.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/finetune_cell_scale.py \
        --model_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --no_ema --n_samples 50 --max_batches 20 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def interval_score(lower, upper, gt, alpha=0.1):
    """Interval Score for alpha-level CI.

    IS = (U - L) + (2/alpha) * max(0, L - y) + (2/alpha) * max(0, y - U)

    Lower IS is better. Differentiable w.r.t. lower and upper.
    """
    width = upper - lower
    penalty_below = (2.0 / alpha) * torch.relu(lower - gt)
    penalty_above = (2.0 / alpha) * torch.relu(gt - upper)
    return width + penalty_below + penalty_above


def apply_correction(samples, baselines, log_correction):
    """Apply per-cell correction to samples.

    corrected = baseline * (samples/baseline)^correction
    In log-space: log(corrected/baseline) = correction * log(samples/baseline)

    Args:
        samples: (N, S, T, 5, 5) in [0, 1]
        baselines: (N, 1, 5, 5) in [0, 1]
        log_correction: Parameter (5, 5)

    Returns:
        corrected: (N, S, T, 5, 5) in [0, 1]
    """
    correction = torch.exp(log_correction)  # (5, 5)
    baselines_expanded = baselines.unsqueeze(1)  # (N, 1, 1, 5, 5)

    # log(samples/baseline) — the log-ratio from baseline
    log_ratio = torch.log(samples / baselines_expanded.clamp(min=1e-6))

    # Apply per-cell correction: scale the log-ratio
    corrected_log_ratio = log_ratio * correction  # broadcast (5,5) over (N,S,T,5,5)

    # Back to absolute space
    corrected = baselines_expanded * torch.exp(corrected_log_ratio)
    return corrected.clamp(0.001, 1.0)


def precompute_samples(model, data_loader, n_samples, max_batches, device,
                       max_global_residual=0, split_name="val"):
    """Generate samples and cache baselines from frozen model."""
    all_samples = []
    all_gt = []
    all_baselines = []

    model.eval()
    config = model.config
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(data_loader, desc=f"Sampling {split_name}", total=max_batches)
        ):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = denormalize_iv(batch["future"].to(device))

            samples = model.sample_batched(
                history, n_samples=n_samples,
                max_global_residual=max_global_residual,
            )

            # Compute baselines (same as model internals)
            history_denorm = denormalize_iv(history)
            K = min(getattr(config, 'baseline_window', 1), history_denorm.shape[1])
            baseline = history_denorm[:, -K:].mean(dim=1).clamp(min=0.01)  # (B, 5, 5)

            all_samples.append(samples.cpu())
            all_gt.append(future_gt.cpu())
            all_baselines.append(baseline.cpu())

    samples = torch.cat(all_samples, dim=0)    # (N, S, T, 5, 5)
    gt = torch.cat(all_gt, dim=0)              # (N, T, 5, 5)
    baselines = torch.cat(all_baselines, dim=0) # (N, 5, 5)

    print(f"  Pre-computed: {samples.shape[0]} windows × {samples.shape[1]} samples")
    return samples, gt, baselines


def optimize_cell_scale(samples, gt, baselines, n_iters=500, lr=0.01,
                        alpha=0.1, device="cuda", clamp=0.5):
    """Optimize per-cell correction to minimize Interval Score.

    Args:
        samples: (N, S, T, 5, 5) model predictions in [0, 1]
        gt: (N, T, 5, 5) ground truth in [0, 1]
        baselines: (N, 5, 5) per-window baselines in [0, 1]
        n_iters: optimization steps
        lr: learning rate
        alpha: CI level (0.1 = 90% CI)
        device: torch device
        clamp: max abs value for log_correction

    Returns:
        correction: (5, 5) learned multiplicative correction
        log_correction: (5, 5) raw log values
    """
    samples = samples.to(device)
    gt = gt.to(device)
    baselines = baselines.unsqueeze(1).to(device)  # (N, 1, 5, 5)

    log_correction = nn.Parameter(torch.zeros(5, 5, device=device))
    optimizer = torch.optim.Adam([log_correction], lr=lr)

    best_loss = float('inf')
    best_correction = None
    best_log_correction = None

    for i in range(n_iters):
        optimizer.zero_grad()

        # Clamp log_correction to prevent extreme values
        with torch.no_grad():
            log_correction.data.clamp_(-clamp, clamp)

        # Apply correction
        corrected = apply_correction(samples, baselines, log_correction)

        # Compute CI (5th and 95th percentiles across samples)
        lower = torch.quantile(corrected, alpha / 2, dim=1)       # (N, T, 5, 5)
        upper = torch.quantile(corrected, 1 - alpha / 2, dim=1)   # (N, T, 5, 5)

        # Interval Score
        is_score = interval_score(lower, upper, gt, alpha)
        loss = is_score.mean()

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_correction = torch.exp(log_correction.detach().clone())
            best_log_correction = log_correction.detach().clone()

        if i % 50 == 0 or i == n_iters - 1:
            with torch.no_grad():
                covered = ((gt >= lower) & (gt <= upper)).float()
                coverage_overall = covered.mean().item()

                # Per-cell coverage
                coverage_per_cell = covered.mean(dim=(0, 1))  # (5, 5)
                corr = torch.exp(log_correction)

                width_mean = (upper - lower).mean().item()

                print(f"\nIter {i:4d}: IS={loss.item():.4f}, "
                      f"coverage={coverage_overall:.3f}, width={width_mean:.4f}")
                print(f"  correction range: [{corr.min():.3f}, {corr.max():.3f}]")
                print(f"  coverage range:   [{coverage_per_cell.min():.3f}, "
                      f"{coverage_per_cell.max():.3f}]")

    return best_correction.cpu(), best_log_correction.cpu()


def evaluate_correction(samples, gt, baselines, correction, alpha=0.1, device="cuda"):
    """Evaluate correction on a dataset and print per-cell diagnostics."""
    samples = samples.to(device)
    gt = gt.to(device)
    baselines = baselines.unsqueeze(1).to(device)

    log_correction = torch.log(correction).to(device)

    with torch.no_grad():
        corrected = apply_correction(samples, baselines, log_correction)

        lower = torch.quantile(corrected, alpha / 2, dim=1)
        upper = torch.quantile(corrected, 1 - alpha / 2, dim=1)

        covered = ((gt >= lower) & (gt <= upper)).float()
        is_score = interval_score(lower, upper, gt, alpha)

        # Per-cell metrics
        coverage_cell = covered.mean(dim=(0, 1))  # (5, 5)
        is_cell = is_score.mean(dim=(0, 1))        # (5, 5)
        width_cell = (upper - lower).mean(dim=(0, 1))  # (5, 5)

        # Per-horizon metrics
        horizons = [0, 6, 13, 29]  # h=1,7,14,30
        print("\n  Per-horizon coverage:")
        for h_idx, h in enumerate([1, 7, 14, 30]):
            if horizons[h_idx] < covered.shape[1]:
                cov_h = covered[:, horizons[h_idx]].mean().item()
                print(f"    h={h:2d}: {cov_h:.3f}")

        print(f"\n  Overall coverage: {covered.mean().item():.3f}")
        print(f"  Overall IS:       {is_score.mean().item():.4f}")
        print(f"  Mean CI width:    {(upper - lower).mean().item():.4f}")

        print("\n  Per-cell 90% coverage:")
        for r in range(5):
            print(f"    {' '.join(f'{coverage_cell[r,c]:.3f}' for c in range(5))}")

        print(f"\n  Coverage range: [{coverage_cell.min():.3f}, {coverage_cell.max():.3f}]")

        # Check gates
        n_pass = ((coverage_cell >= 0.70) & (coverage_cell <= 0.95)).sum().item()
        print(f"  Cells in [70%, 95%]: {n_pass}/25")

        return {
            "coverage_overall": covered.mean().item(),
            "is_overall": is_score.mean().item(),
            "coverage_per_cell": coverage_cell.cpu().numpy().tolist(),
            "is_per_cell": is_cell.cpu().numpy().tolist(),
            "width_per_cell": width_cell.cpu().numpy().tolist(),
        }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune per-cell CI scale via IS loss")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_batches", type=int, default=20,
                        help="Number of batches for pre-computing samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_global_residual", type=int, default=0)

    # Optimization params
    parser.add_argument("--n_iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--clamp", type=float, default=0.5,
                        help="Max abs log_correction (0.5 → [0.61, 1.65] range)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="CI level (0.1 = 90%% CI)")

    # Data split for calibration
    parser.add_argument("--cal_split", type=str, default="val",
                        choices=["train", "val"],
                        help="Data split for calibration (default: val)")

    parser.add_argument("--output_dir", type=str,
                        default="results/block_ar/cell_scale_is")
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_default_config()

    print("=" * 60)
    print("Per-Cell CI Width Calibration via Interval Score")
    print("=" * 60)
    print(f"Model:     {args.model_path}")
    print(f"Samples:   {args.n_samples} per window")
    print(f"Batches:   {args.max_batches}")
    print(f"Cal split: {args.cal_split}")
    print(f"IS alpha:  {args.alpha} ({int((1-args.alpha)*100)}% CI)")
    print(f"Iters:     {args.n_iters}, lr={args.lr}, clamp={args.clamp}")
    print(f"Output:    {output_dir}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)

    model = ConditionalBlockARDDPM(model_config)
    if "ema_params" in checkpoint and not args.no_ema:
        print("  Loading EMA parameters...")
        state_dict = model.state_dict()
        for name in state_dict:
            if name in checkpoint["ema_params"]:
                state_dict[name] = checkpoint["ema_params"][name]
        model.load_state_dict(state_dict)
    else:
        print("  Loading regular model weights...")
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    print("\nLoading data...")
    data = np.load(config.data_path)
    surfaces = data["surface"]

    if args.cal_split == "val":
        cal_dataset = VolSurfaceDataset(
            surfaces, config.history_len, config.future_len,
            start_idx=config.val_start, end_idx=config.val_end,
        )
        test_dataset = VolSurfaceDataset(
            surfaces, config.history_len, config.future_len,
            start_idx=config.test_start,
        )
    else:
        cal_dataset = VolSurfaceDataset(
            surfaces, config.history_len, config.future_len,
            end_idx=config.train_end,
        )
        test_dataset = VolSurfaceDataset(
            surfaces, config.history_len, config.future_len,
            start_idx=config.test_start,
        )

    cal_loader = DataLoader(cal_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=2)

    print(f"  Cal set ({args.cal_split}): {len(cal_dataset)} windows")
    print(f"  Test set: {len(test_dataset)} windows")

    # Pre-compute samples on calibration set
    print("\nPre-computing samples on calibration set...")
    cal_samples, cal_gt, cal_baselines = precompute_samples(
        model, cal_loader, args.n_samples, args.max_batches, device,
        max_global_residual=args.max_global_residual, split_name=args.cal_split,
    )

    # Evaluate baseline (no correction)
    print("\n" + "=" * 60)
    print("BASELINE (no correction)")
    print("=" * 60)
    identity_correction = torch.ones(5, 5)
    baseline_results = evaluate_correction(
        cal_samples, cal_gt, cal_baselines, identity_correction,
        alpha=args.alpha, device=device,
    )

    # Optimize correction
    print("\n" + "=" * 60)
    print("OPTIMIZING log_correction via IS loss")
    print("=" * 60)
    correction, log_correction = optimize_cell_scale(
        cal_samples, cal_gt, cal_baselines,
        n_iters=args.n_iters, lr=args.lr, alpha=args.alpha,
        device=device, clamp=args.clamp,
    )

    print("\n" + "=" * 60)
    print("LEARNED CORRECTION (5x5)")
    print("=" * 60)
    print("\n  Per-cell correction factor:")
    for r in range(5):
        print(f"    {' '.join(f'{correction[r,c]:.4f}' for c in range(5))}")
    print(f"\n  Range: [{correction.min():.4f}, {correction.max():.4f}]")
    print(f"  Mean:  {correction.mean():.4f}")

    # Evaluate on calibration set (in-sample)
    print("\n" + "=" * 60)
    print(f"CORRECTED — Cal set ({args.cal_split})")
    print("=" * 60)
    cal_corrected_results = evaluate_correction(
        cal_samples, cal_gt, cal_baselines, correction,
        alpha=args.alpha, device=device,
    )

    # Pre-compute samples on test set and evaluate (out-of-sample)
    print("\n" + "=" * 60)
    print("Pre-computing samples on TEST set...")
    print("=" * 60)
    test_samples, test_gt, test_baselines = precompute_samples(
        model, test_loader, args.n_samples, args.max_batches, device,
        max_global_residual=args.max_global_residual, split_name="test",
    )

    print("\n" + "=" * 60)
    print("BASELINE — Test set (no correction)")
    print("=" * 60)
    test_baseline_results = evaluate_correction(
        test_samples, test_gt, test_baselines, identity_correction,
        alpha=args.alpha, device=device,
    )

    print("\n" + "=" * 60)
    print("CORRECTED — Test set (out-of-sample)")
    print("=" * 60)
    test_corrected_results = evaluate_correction(
        test_samples, test_gt, test_baselines, correction,
        alpha=args.alpha, device=device,
    )

    # Save results
    results = {
        "correction": correction.numpy().tolist(),
        "log_correction": log_correction.numpy().tolist(),
        "config": {
            "model_path": args.model_path,
            "n_samples": args.n_samples,
            "max_batches": args.max_batches,
            "cal_split": args.cal_split,
            "alpha": args.alpha,
            "n_iters": args.n_iters,
            "lr": args.lr,
            "clamp": args.clamp,
        },
        "cal_baseline": baseline_results,
        "cal_corrected": cal_corrected_results,
        "test_baseline": test_baseline_results,
        "test_corrected": test_corrected_results,
    }

    results_path = output_dir / "cell_scale_is_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print the correction as a flat list for --cell_scale_values
    flat_correction = correction.numpy().flatten().tolist()
    print(f"\n{'=' * 60}")
    print("USE WITH TEST SUITE:")
    print(f"{'=' * 60}")
    print(f"  --cell_scale_values '{json.dumps([round(x, 4) for x in flat_correction])}'")


if __name__ == "__main__":
    main()
