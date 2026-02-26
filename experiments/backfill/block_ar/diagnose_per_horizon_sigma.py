"""Diagnose per-horizon future uncertainty targets.

The key insight: total 30-day sigma is condition-independent (Q5/Q1≈1.0),
but per-horizon INCREMENTAL changes might be condition-dependent.

Compute:
1. Per-horizon incremental change magnitude |future[h] - future[h-1]|
2. Cross-sample std of these changes, grouped by vol_of_vol quintile
3. Find the right target granularity that IS condition-dependent
"""

import numpy as np
import torch
from scipy import stats

from diffusion.block_ar.block_ar_ddpm import denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from torch.utils.data import DataLoader


def main():
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    # Use test split (larger, out of sample)
    ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_vov = []
    all_baseline_iv = []
    all_incremental = []  # per-horizon changes: future[h] - future[h-1]
    all_abs_change_h1 = []  # |future[1] - baseline|
    all_cumul_sigma = []  # std of cumulative log-ratio at each horizon

    for batch in dl:
        history = batch["history"]
        future = batch["future"]
        B = history.shape[0]

        past_abs = denormalize_iv(history)  # (B, 30, 5, 5)
        fut_abs = denormalize_iv(future)  # (B, 30, 5, 5)

        # vol_of_vol
        mean_iv = past_abs.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)  # (B,)
        all_vov.append(vov.numpy())

        # baseline IV
        bl_iv = past_abs[:, -1].mean(dim=(-1, -2))  # (B,)
        all_baseline_iv.append(bl_iv.numpy())

        # Incremental changes: per-frame magnitude
        combined = torch.cat([past_abs[:, -1:], fut_abs], dim=1)  # (B, 31, 5, 5)
        incremental = (combined[:, 1:] - combined[:, :-1]).abs().mean(dim=(-1, -2))  # (B, 30)
        all_incremental.append(incremental.numpy())

        # Absolute change at h=1
        h1_change = (fut_abs[:, 0] - past_abs[:, -1]).abs().mean(dim=(-1, -2))  # (B,)
        all_abs_change_h1.append(h1_change.numpy())

        # Cumulative log-ratio sigma at each horizon
        baseline = past_abs[:, -1:].clamp(min=0.01)  # (B, 1, 5, 5)
        log_ratio = torch.log(fut_abs.clamp(min=1e-4) / baseline)  # (B, 30, 5, 5)
        # Std of log-ratio across cells at each horizon
        cumul_sigma = log_ratio.reshape(B, 30, -1).std(dim=2)  # (B, 30) std across cells
        all_cumul_sigma.append(cumul_sigma.numpy())

    vov = np.concatenate(all_vov)
    bl_iv = np.concatenate(all_baseline_iv)
    incremental = np.concatenate(all_incremental)  # (N, 30)
    abs_h1 = np.concatenate(all_abs_change_h1)
    cumul_sigma = np.concatenate(all_cumul_sigma)  # (N, 30)
    N = len(vov)

    # Quintile boundaries for vol_of_vol
    vov_q = np.quantile(vov, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    quintile_labels = np.digitize(vov, vov_q[1:-1])  # 0-4

    print(f"Test set: N={N}")
    print(f"\nvol_of_vol quintile boundaries: {[f'{q:.4f}' for q in vov_q]}")

    # Per-horizon: Q5/Q1 of incremental changes
    print(f"\n{'='*60}")
    print("Q5/Q1 of |incremental change| by vol_of_vol quintile")
    print(f"{'='*60}")
    print(f"{'Horizon':>8} {'Q1 mean':>10} {'Q5 mean':>10} {'Q5/Q1':>8} {'Spearman':>10}")
    for h in [0, 1, 2, 4, 6, 9, 14, 19, 29]:
        q1_mask = quintile_labels == 0
        q5_mask = quintile_labels == 4
        q1_mean = incremental[q1_mask, h].mean()
        q5_mean = incremental[q5_mask, h].mean()
        q5q1 = q5_mean / q1_mean
        rho = stats.spearmanr(vov, incremental[:, h])[0]
        print(f"  h={h+1:>3}   {q1_mean:.6f}   {q5_mean:.6f}   {q5q1:.3f}     {rho:.3f}")

    # Cross-sample std of changes (the actual metric for conditional uncertainty)
    print(f"\n{'='*60}")
    print("Cross-sample std of incremental changes by vol_of_vol quintile")
    print("(This is what Q5/Q1 in measure_q5q1.py computes)")
    print(f"{'='*60}")
    print(f"{'Horizon':>8} {'Q1 std':>10} {'Q5 std':>10} {'Q5/Q1':>8}")
    for h in [0, 1, 2, 4, 6, 9, 14, 19, 29]:
        q1_mask = quintile_labels == 0
        q5_mask = quintile_labels == 4
        q1_std = incremental[q1_mask, h].std()
        q5_std = incremental[q5_mask, h].std()
        q5q1 = q5_std / q1_std
        print(f"  h={h+1:>3}   {q1_std:.6f}   {q5_std:.6f}   {q5q1:.3f}")

    # Per-horizon: Spearman(vov, cumulative log-ratio)
    print(f"\n{'='*60}")
    print("Spearman(vol_of_vol, per-horizon metric)")
    print(f"{'='*60}")
    print(f"{'Horizon':>8} {'r(vov, |incr|)':>15} {'r(vov, cum_sigma)':>18}")
    for h in [0, 1, 2, 4, 6, 9, 14, 19, 29]:
        rho_incr = stats.spearmanr(vov, incremental[:, h])[0]
        rho_cum = stats.spearmanr(vov, cumul_sigma[:, h])[0]
        print(f"  h={h+1:>3}   {rho_incr:>12.3f}   {rho_cum:>15.3f}")

    # Key finding: is h=1 change correlated with vov?
    print(f"\n{'='*60}")
    print("Key correlations with vol_of_vol")
    print(f"{'='*60}")
    rho_h1 = stats.spearmanr(vov, abs_h1)[0]
    rho_mean_incr = stats.spearmanr(vov, incremental.mean(axis=1))[0]
    print(f"  |h=1 change| vs vov:  Spearman = {rho_h1:.3f}")
    print(f"  mean(|incr|) vs vov:  Spearman = {rho_mean_incr:.3f}")
    print(f"  h=1 Q5/Q1: {abs_h1[quintile_labels==4].mean() / abs_h1[quintile_labels==0].mean():.3f}")


if __name__ == "__main__":
    main()
