#!/usr/bin/env python
"""
Measure Q5/Q1 conditional uncertainty ratio for Block-AR models.

Computes cross-sample standard deviation per window, groups by conditioning
variable quintiles, and reports Q5/Q1 ratio at each horizon.

Conditioning variables:
- vol_of_vol: std of day-to-day mean-IV changes over 30-day history (strongest GT signal)
- recent_change: abs(IV change over last 5 days)
- baseline_iv: mean IV of last history day (anti-heteroscedastic in GT due to mean reversion)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/measure_q5q1.py \
        --model_path models/backfill/block_ar_ratio_v1/best_coverage_model.pt \
        --n_samples 50 --max_windows 400
"""

import argparse
import dataclasses
import json
import sys

import numpy as np
import torch

from diffusion.block_ar.block_ar_ddpm import ConditionalBlockARDDPM, BlockARConfig, denormalize_iv


def load_model(model_path, device):
    """Load checkpoint and build model."""
    cp = torch.load(model_path, map_location=device, weights_only=False)
    c = cp["config"]
    if dataclasses.is_dataclass(c):
        c = dataclasses.asdict(c)
    config = BlockARConfig(**c)
    model = ConditionalBlockARDDPM(config)
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    return model, config


def compute_conditioning_variables(history_iv):
    """
    Compute conditioning variables from history IV surfaces.

    Args:
        history_iv: (N, 30, 5, 5) raw IV surfaces

    Returns:
        dict of variable_name -> (N,) arrays
    """
    # Mean IV per day: (N, 30)
    mean_iv = history_iv.mean(axis=(-1, -2))

    # vol_of_vol: std of day-to-day changes in mean IV
    daily_changes = np.diff(mean_iv, axis=1)  # (N, 29)
    vol_of_vol = daily_changes.std(axis=1)  # (N,)

    # recent_change: abs change in mean IV over last 5 days
    recent_change = np.abs(mean_iv[:, -1] - mean_iv[:, -6])  # (N,)

    # baseline_iv: mean IV of last history day
    baseline_iv = mean_iv[:, -1]  # (N,)

    return {
        "vol_of_vol": vol_of_vol,
        "recent_change": recent_change,
        "baseline_iv": baseline_iv,
    }


def measure_q5q1(model, config, surfaces, n_samples=50, max_windows=400, device="cuda"):
    """
    Measure Q5/Q1 conditional uncertainty ratio.

    Returns dict with per-variable, per-horizon Q5/Q1 ratios.
    """
    history_len = config.history_len
    future_len = config.future_len
    total_len = history_len + future_len

    # Build windows
    N = len(surfaces)
    n_windows = min(max_windows, N - total_len + 1)
    indices = np.linspace(0, N - total_len, n_windows, dtype=int)

    all_history = []
    all_future = []
    for idx in indices:
        all_history.append(surfaces[idx:idx + history_len])
        all_future.append(surfaces[idx + history_len:idx + total_len])

    history_arr = np.stack(all_history)  # (n_windows, 30, 5, 5)
    future_arr = np.stack(all_future)    # (n_windows, 30, 5, 5)

    # Compute conditioning variables from raw history
    cond_vars = compute_conditioning_variables(history_arr)

    # Generate samples in batches
    batch_size = 16
    all_stds = []  # (n_windows, future_len)

    for i in range(0, n_windows, batch_size):
        batch_end = min(i + batch_size, n_windows)
        hist_batch = torch.tensor(history_arr[i:batch_end], dtype=torch.float32, device=device)

        with torch.no_grad():
            samples = model.sample(hist_batch, n_samples=n_samples)
            # samples: (B, n_samples, future_len, 5, 5)

        # Denormalize
        samples_iv = denormalize_iv(samples)  # (B, n_samples, 30, 5, 5)

        # Cross-sample std per window per horizon (mean over spatial dims)
        sample_std = samples_iv.mean(dim=(-1, -2)).std(dim=1)  # (B, 30)
        all_stds.append(sample_std.cpu().numpy())

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {batch_end}/{n_windows} windows...")

    all_stds = np.concatenate(all_stds, axis=0)  # (n_windows, 30)

    # Compute Q5/Q1 for each variable at each horizon
    horizons = [0, 6, 13, 29]  # h=1, h=7, h=14, h=30
    horizon_names = ["h=1", "h=7", "h=14", "h=30"]

    results = {}
    for var_name, var_values in cond_vars.items():
        quintiles = np.percentile(var_values, [0, 20, 40, 60, 80, 100])
        q1_mask = var_values <= quintiles[1]  # bottom 20%
        q5_mask = var_values >= quintiles[4]  # top 20%

        var_results = {}
        for h_idx, h_name in zip(horizons, horizon_names):
            q1_std = all_stds[q1_mask, h_idx].mean()
            q5_std = all_stds[q5_mask, h_idx].mean()
            q5q1 = q5_std / q1_std if q1_std > 0 else float("nan")
            var_results[h_name] = {
                "q5q1": float(q5q1),
                "q5_std": float(q5_std),
                "q1_std": float(q1_std),
            }

        # Spearman correlation
        from scipy.stats import spearmanr
        spearman_h1, _ = spearmanr(var_values, all_stds[:, 0])
        var_results["spearman_h1"] = float(spearman_h1)

        results[var_name] = var_results

    # Also compute GT Q5/Q1 for reference
    gt_stds = np.std(future_arr.mean(axis=(-1, -2)), axis=0)  # This is wrong, need per-window
    # Actually, for GT we need the actual spread across windows with similar conditions
    # Skip GT - too complex for inline

    return results, n_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Override CFG guidance scale at inference")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, config = load_model(args.model_path, args.device)
    if args.guidance_scale is not None:
        config.guidance_scale = args.guidance_scale
        # Re-set on model since it reads from config at inference
        model.config.guidance_scale = args.guidance_scale
        print(f"Guidance scale override: {args.guidance_scale}")

    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)

    # Use test set
    test_start = 4540
    test_surfaces = surfaces[test_start:]
    print(f"Test surfaces: {len(test_surfaces)}")

    print(f"Generating {args.n_samples} samples per window, {args.max_windows} windows...")
    results, n_windows = measure_q5q1(
        model, config, test_surfaces,
        n_samples=args.n_samples,
        max_windows=args.max_windows,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"Q5/Q1 Conditional Uncertainty ({n_windows} windows, {args.n_samples} samples)")
    print(f"{'='*60}")

    for var_name in ["vol_of_vol", "recent_change", "baseline_iv"]:
        var_res = results[var_name]
        print(f"\n  {var_name} (Spearman h=1: {var_res['spearman_h1']:.3f}):")
        for h_name in ["h=1", "h=7", "h=14", "h=30"]:
            r = var_res[h_name]
            print(f"    {h_name}: Q5/Q1 = {r['q5q1']:.3f}x  (Q5={r['q5_std']:.4f}, Q1={r['q1_std']:.4f})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
