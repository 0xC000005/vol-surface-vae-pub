#!/usr/bin/env python
"""
Evaluate per-cell sigma head by applying it to sample generation.

Two modes:
1. "rescale": Post-hoc per-cell rescaling of model samples
   adjusted = mean + (sigma_learned / sigma_empirical) * (sample - mean) per cell

2. "coverage": Just measure per-cell coverage before and after rescaling

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/eval_percell_sigma.py \
        --model_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --sigma_path models/backfill/block_ar_percell_sigma_frozen_nll_v1/sigma_head.pt \
        --n_samples 50 --max_windows 400
"""

import argparse
import dataclasses
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_percell_sigma import (
    PerCellSigmaHead,
    get_condition_vector,
)


def load_model(model_path, device):
    cp = torch.load(model_path, map_location=device, weights_only=False)
    c = cp["config"]
    if dataclasses.is_dataclass(c):
        c = dataclasses.asdict(c)
    config = BlockARConfig(**c)
    model = ConditionalBlockARDDPM(config)
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    return model, config


def load_sigma_head(sigma_path, device):
    cp = torch.load(sigma_path, map_location=device, weights_only=False)
    head = PerCellSigmaHead(
        cond_dim=cp["cond_dim"],
        hidden_dim=cp["hidden_dim"],
    )
    head.load_state_dict(cp["sigma_head_state_dict"])
    head.to(device).eval()
    return head


def compute_conditioning_variables(history_iv):
    mean_iv = history_iv.mean(axis=(-1, -2))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)
    return {"vol_of_vol": vol_of_vol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sigma_path", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model, config = load_model(args.model_path, args.device)

    print(f"Loading sigma head: {args.sigma_path}")
    sigma_head = load_sigma_head(args.sigma_path, args.device)

    print("Loading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_start = 4540
    test_surfaces = surfaces[test_start:]

    history_len = config.history_len
    future_len = config.future_len
    total_len = history_len + future_len

    N = len(test_surfaces)
    n_windows = min(args.max_windows, N - total_len + 1)
    indices = np.linspace(0, N - total_len, n_windows, dtype=int)

    all_history = []
    all_future = []
    for idx in indices:
        all_history.append(test_surfaces[idx:idx + history_len])
        all_future.append(test_surfaces[idx + history_len:idx + total_len])

    history_arr = np.stack(all_history)  # (n_windows, 30, 5, 5)
    future_arr = np.stack(all_future)    # (n_windows, 30, 5, 5)
    cond_vars = compute_conditioning_variables(history_arr)

    # Generate samples and apply sigma head
    batch_size = 16
    all_samples_raw = []      # before rescaling
    all_samples_rescaled = [] # after per-cell sigma rescaling
    all_sigmas = []

    for i in range(0, n_windows, batch_size):
        batch_end = min(i + batch_size, n_windows)
        hist_raw = torch.tensor(history_arr[i:batch_end], dtype=torch.float32, device=args.device)
        hist_norm = normalize_iv(hist_raw)

        with torch.no_grad():
            # Get condition vector
            cond = get_condition_vector(model, hist_norm)

            # Get per-cell sigma
            log_sigma = sigma_head(cond)  # (B, 5, 5)
            sigma = torch.exp(log_sigma.clamp(-4, 4))  # (B, 5, 5)
            all_sigmas.append(sigma.cpu().numpy())

            # Generate samples
            samples = model.sample(hist_norm, n_samples=args.n_samples)
            # (B, n_samples, 30, 5, 5) in [0, 1]

        all_samples_raw.append(samples.cpu().numpy())

        # Per-cell rescaling: redistribute spread across cells based on learned sigma
        # The idea: keep the mean spread (total uncertainty budget) but reallocate
        # it so that hard cells (high sigma) get more spread, easy cells less.
        sample_mean = samples.mean(dim=1, keepdim=True)  # (B, 1, 30, 5, 5)
        sample_std = samples.std(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 30, 5, 5)

        # Learned sigma: (B, 5, 5) → (B, 1, 1, 5, 5)
        sigma_target = sigma.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 5, 5)

        # Normalize learned sigma to have same mean as empirical std per sample
        # This preserves total uncertainty budget but redistributes across cells
        mean_empirical = sample_std.mean(dim=(-1, -2), keepdim=True)  # (B, 1, 30, 1, 1)
        mean_learned = sigma_target.mean(dim=(-1, -2), keepdim=True)  # (B, 1, 1, 1, 1)
        scale = (sigma_target / mean_learned) * (mean_empirical / sample_std)
        # scale: how much to adjust each cell's spread relative to current

        rescaled = sample_mean + scale * (samples - sample_mean)
        rescaled = rescaled.clamp(0.0, 1.0)
        all_samples_rescaled.append(rescaled.cpu().numpy())

        if (i // batch_size) % 5 == 0:
            print(f"  Processed {batch_end}/{n_windows} windows...")

    all_samples_raw = np.concatenate(all_samples_raw, axis=0)
    all_samples_rescaled = np.concatenate(all_samples_rescaled, axis=0)
    all_sigmas = np.concatenate(all_sigmas, axis=0)

    # Compute coverage metrics
    horizons = [0, 6, 13, 29]
    horizon_names = ["h=1", "h=7", "h=14", "h=30"]

    print(f"\n{'='*70}")
    print(f"Per-Cell Coverage Comparison ({n_windows} windows, {args.n_samples} samples)")
    print(f"{'='*70}")

    for label, samples in [("Raw (no rescaling)", all_samples_raw),
                           ("Per-cell sigma rescaled", all_samples_rescaled)]:
        print(f"\n--- {label} ---")

        for h_idx, h_name in zip(horizons, horizon_names):
            gt = future_arr[:, h_idx]  # (n_windows, 5, 5)
            s = samples[:, :, h_idx]   # (n_windows, n_samples, 5, 5)

            # Per-cell 90% CI coverage
            q05 = np.percentile(s, 5, axis=1)   # (n_windows, 5, 5)
            q95 = np.percentile(s, 95, axis=1)   # (n_windows, 5, 5)
            covered = (gt >= q05) & (gt <= q95)

            # Overall coverage
            overall_cov = covered.mean()

            # Per-cell coverage
            percell_cov = covered.mean(axis=0)  # (5, 5)

            # Coverage std across cells (lower = more uniform = better calibrated)
            cov_std = percell_cov.std()
            cov_range = percell_cov.max() - percell_cov.min()

            print(f"  {h_name}: overall={overall_cov:.3f}  "
                  f"cell_std={cov_std:.3f}  "
                  f"range=[{percell_cov.min():.3f}, {percell_cov.max():.3f}]")

            if h_name == "h=1":
                print(f"    Per-cell coverage grid:")
                for r in range(5):
                    row = "  ".join(f"{percell_cov[r, c]:.3f}" for c in range(5))
                    print(f"      [{row}]")

    # Q5/Q1 analysis for both raw and rescaled
    print(f"\n{'='*70}")
    print(f"Q5/Q1 Conditional Uncertainty")
    print(f"{'='*70}")

    for label, samples in [("Raw", all_samples_raw),
                           ("Rescaled", all_samples_rescaled)]:
        # Cross-sample std per window (mean over cells and time)
        sample_std = samples.mean(axis=(-1, -2)).std(axis=1)  # (n_windows, 30) → std over samples dim
        # Actually: samples is (W, S, 30, 5, 5). We want std across S.
        mean_iv_per_sample = samples.mean(axis=(-1, -2))  # (W, S, 30)
        cross_sample_std = mean_iv_per_sample.std(axis=1)  # (W, 30)

        vov = cond_vars["vol_of_vol"]
        quintiles = np.percentile(vov, [0, 20, 40, 60, 80, 100])
        q1_mask = vov <= quintiles[1]
        q5_mask = vov >= quintiles[4]

        print(f"\n  {label}:")
        for h_idx, h_name in zip(horizons, horizon_names):
            q1_std = cross_sample_std[q1_mask, h_idx].mean()
            q5_std = cross_sample_std[q5_mask, h_idx].mean()
            q5q1 = q5_std / q1_std if q1_std > 0 else float('nan')
            rho, _ = spearmanr(vov, cross_sample_std[:, h_idx])
            print(f"    {h_name}: Q5/Q1={q5q1:.3f}x  Spearman={rho:.3f}")

    # Save results
    if args.output:
        results = {
            "n_windows": n_windows,
            "n_samples": args.n_samples,
            "sigma_stats": {
                "mean": float(all_sigmas.mean()),
                "std": float(all_sigmas.std()),
                "min": float(all_sigmas.min()),
                "max": float(all_sigmas.max()),
            },
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
