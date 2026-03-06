#!/usr/bin/env python
"""Comprehensive diagnosis of drift and spatial degradation in long-horizon AR generation.

Instruments the AR frame loop step-by-step to identify:
1. Per-cell drift trajectories and floor-hitting dynamics
2. MLP delta statistics evolution (does FrameDecoder go out-of-distribution?)
3. GRU condition vector drift (does the condition diverge from training distribution?)
4. Cross-cell correlation structure at different horizons
5. Spatial covariance matrix evolution (term slope, smile, full correlation)
6. Per-cell volatility of daily changes vs horizon (does it decay/explode?)
7. Absorbing barrier analysis (once near floor, escape probability)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_long_horizon.py \
        --model_path models/backfill/afcrps_90m/best_model.pt \
        --no_ema --n_samples 50 --n_windows 50 --n_frames 252 \
        --output_dir results/block_ar/long_horizon_diagnosis --device cuda
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

LABELS_K = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
LABELS_T = ["1M", "3M", "6M", "1Y", "2Y"]
CELL_NAMES = [f"{t}/{k}" for t in LABELS_T for k in LABELS_K]
HORIZONS = [30, 60, 90, 180, 252]


def load_model(model_path, device, no_ema=True):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = SinglePassConfig(**cfg)
    model = SinglePassBlockAR(cfg)
    key = "ema_state_dict" if not no_ema and "ema_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.eval().to(device)
    return model, ckpt


@torch.no_grad()
def instrumented_ar_generation(model, history, n_samples, n_frames, position_mode="native"):
    """AR frame generation with full instrumentation.

    Returns dict with per-step diagnostics:
    - samples: (B, S, T, 5, 5) generated IV surfaces
    - deltas: (B, S, T, 5, 5) raw MLP deltas (before vol_scale)
    - condition_norms: (B, S, T) L2 norm of condition vector at each step
    - condition_vectors: (B, S, T, bottleneck_dim) condition vectors (subsampled)
    - floor_hits: (B, S, T) boolean, True if clamped to floor
    - ceiling_hits: (B, S, T) boolean, True if clamped to ceiling
    """
    B = history.shape[0]
    device = history.device
    H, W = model.config.surface_h, model.config.surface_w
    rho = model.config.ar_frame_rho
    cfg = model.config

    _, vol_scale = model._compute_vol_scale(history)
    vol_scale_cell = None
    if cfg.ar_frame_percell_vol_scale:
        vol_scale_cell = model._compute_percell_vol_scale(history)

    # Storage for all samples
    all_samples = torch.zeros(B, n_samples, n_frames, H, W, device=device)
    all_deltas = torch.zeros(B, n_samples, n_frames, H, W, device=device)
    all_cond_norms = torch.zeros(B, n_samples, n_frames, device=device)
    all_floor_hits = torch.zeros(B, n_samples, n_frames, dtype=torch.bool, device=device)
    all_ceiling_hits = torch.zeros(B, n_samples, n_frames, dtype=torch.bool, device=device)
    # Subsample condition vectors to save memory (every 10 steps)
    subsample_step = 10
    n_cond_steps = (n_frames + subsample_step - 1) // subsample_step
    all_cond_vecs = torch.zeros(B, n_samples, n_cond_steps, cfg.bottleneck_dim, device=device)

    for s in range(n_samples):
        z = model._sample_noise(B, device)
        z_t = z
        gru_outputs, h_last = model._init_gru_state(history)
        condition = model.encoder(history, mask=None)
        prev_frame = denormalize_iv(history[:, -1])  # (B, H, W)

        for step_idx in range(n_frames):
            if step_idx > 0:
                eps_t = torch.randn_like(z_t)
                z_t = rho * z_t + math.sqrt(1 - rho**2) * eps_t

            local_pos, horizon_bucket = model._get_ar_frame_positions(
                step_idx=step_idx, batch_size=B, device=device,
                position_mode=position_mode,
            )

            prev_flat = prev_frame.reshape(B, H * W)
            noise_input = model._get_noise_for_decoder(z_t)
            delta = model.frame_decoder(
                prev_flat, condition, noise_input, local_pos, horizon_bucket
            ).reshape(B, H, W)

            if hasattr(model, "cell_scale"):
                cs = model.cell_scale.clamp(0.3, 3.0).view(H, W)
                delta = cs * delta
            if hasattr(model, "cell_spread_linear"):
                from torch.nn import functional as F
                cs = F.softplus(model.cell_spread_linear(condition)).view(B, H, W)
                delta = cs * delta

            vs = model._get_ar_frame_vol_scale(condition, vol_scale, vol_scale_cell)
            floor = getattr(cfg, 'ar_frame_floor_clamp', 0.001)
            if cfg.ar_frame_log_space:
                unclamped = prev_frame * torch.exp(vs * delta)
            else:
                unclamped = prev_frame + vs * delta
            iv_t = unclamped.clamp(floor, 1.0)

            # Record diagnostics
            all_samples[:, s, step_idx] = iv_t
            all_deltas[:, s, step_idx] = delta
            all_cond_norms[:, s, step_idx] = condition.norm(dim=-1)
            all_floor_hits[:, s, step_idx] = unclamped.amin(dim=(-1, -2)) <= floor
            all_ceiling_hits[:, s, step_idx] = unclamped.amax(dim=(-1, -2)) >= 1.0
            if step_idx % subsample_step == 0:
                all_cond_vecs[:, s, step_idx // subsample_step] = condition

            prev_frame = iv_t
            condition, gru_outputs, h_last = model._gru_step(
                iv_t, gru_outputs, h_last
            )

    return {
        "samples": all_samples.cpu().numpy(),
        "deltas": all_deltas.cpu().numpy(),
        "condition_norms": all_cond_norms.cpu().numpy(),
        "condition_vectors": all_cond_vecs.cpu().numpy(),
        "floor_hits": all_floor_hits.cpu().numpy(),
        "ceiling_hits": all_ceiling_hits.cpu().numpy(),
    }


def diag1_drift_trajectories(result, gt_future, output_dir):
    """Per-cell median trajectory + floor-hitting cumulative rate."""
    samples = result["samples"]  # (N, S, T, 5, 5)
    N, S, T, H, W = samples.shape

    median_path = np.median(samples, axis=1)  # (N, T, 5, 5)
    gt_T = gt_future.shape[1]

    # Floor-hitting: fraction of (window, sample) pairs where ANY cell < 0.005
    floor_threshold = 0.005
    per_cell_floor = (samples < floor_threshold)  # (N, S, T, 5, 5)
    any_cell_floor = per_cell_floor.any(axis=(-1, -2))  # (N, S, T)

    # Cumulative floor rate: at each horizon, what fraction of paths have ever hit floor
    cumulative_floor = np.zeros((N, S, T), dtype=bool)
    for t in range(T):
        if t == 0:
            cumulative_floor[:, :, t] = any_cell_floor[:, :, t]
        else:
            cumulative_floor[:, :, t] = cumulative_floor[:, :, t-1] | any_cell_floor[:, :, t]

    cum_floor_rate = cumulative_floor.mean(axis=(0, 1))  # (T,)

    # Per-cell cumulative floor rate
    per_cell_cum_floor = np.zeros((N, S, T, H, W), dtype=bool)
    for t in range(T):
        if t == 0:
            per_cell_cum_floor[:, :, t] = per_cell_floor[:, :, t]
        else:
            per_cell_cum_floor[:, :, t] = per_cell_cum_floor[:, :, t-1] | per_cell_floor[:, :, t]
    per_cell_cum_rate = per_cell_cum_floor.mean(axis=(0, 1))  # (T, 5, 5)

    # Plot 1a: Median trajectories for key cells
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    cells = [(0, 0, "1M/K=0.70"), (0, 2, "1M/ATM"), (2, 2, "6M/ATM"),
             (4, 2, "2Y/ATM"), (2, 0, "6M/K=0.70"), (2, 4, "6M/K=1.30")]
    for ax, (r, c, name) in zip(axes.flat, cells):
        # Generated: show 10th, 25th, 50th, 75th, 90th percentiles
        p10 = np.percentile(samples[:, :, :, r, c].reshape(-1, T), 10, axis=0)
        p25 = np.percentile(samples[:, :, :, r, c].reshape(-1, T), 25, axis=0)
        p50 = np.percentile(samples[:, :, :, r, c].reshape(-1, T), 50, axis=0)
        p75 = np.percentile(samples[:, :, :, r, c].reshape(-1, T), 75, axis=0)
        p90 = np.percentile(samples[:, :, :, r, c].reshape(-1, T), 90, axis=0)

        t_ax = np.arange(T)
        ax.fill_between(t_ax, p10, p90, alpha=0.15, color="blue")
        ax.fill_between(t_ax, p25, p75, alpha=0.25, color="blue")
        ax.plot(t_ax, p50, color="blue", lw=1, label="Gen median")

        # GT
        if gt_T > 0:
            gt_med = np.median(gt_future[:, :min(T, gt_T), r, c], axis=0)
            ax.plot(range(len(gt_med)), gt_med, color="red", lw=1.5, ls="--", label="GT median")

        ax.axhline(0.001, color="gray", ls=":", alpha=0.5, label="Floor")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("IV")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Drift Trajectories: Generated vs GT (10-90% fan)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/d1a_drift_trajectories.png", dpi=150)
    plt.close()

    # Plot 1b: Cumulative floor rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(T), cum_floor_rate * 100, lw=2)
    for h in HORIZONS:
        if h < T:
            ax.axvline(h, color="gray", alpha=0.3, ls="--")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Cumulative floor-hit rate (%)")
    ax.set_title("Any-cell cumulative floor rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Per-cell at key horizons
    for h in HORIZONS:
        if h <= T:
            rates = per_cell_cum_rate[h-1].flatten() * 100
            ax.bar(np.arange(25) + HORIZONS.index(h) * 0.15, rates, 0.15,
                   label=f"h={h}", alpha=0.7)
    ax.set_xlabel("Cell index")
    ax.set_ylabel("Cumulative floor-hit rate (%)")
    ax.set_title("Per-cell cumulative floor rate by horizon")
    ax.set_xticks(range(25))
    ax.set_xticklabels([CELL_NAMES[i] for i in range(25)], fontsize=5, rotation=90)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/d1b_floor_hitting.png", dpi=150)
    plt.close()

    # Print summary
    print("\n  D1: DRIFT & FLOOR-HITTING")
    print(f"  {'Horizon':>8s}  {'CumFloor%':>10s}  {'Worst Cell':>12s}  {'Worst%':>8s}")
    for h in HORIZONS:
        if h <= T:
            cf = cum_floor_rate[h-1] * 100
            cell_rates = per_cell_cum_rate[h-1]
            worst_idx = np.unravel_index(cell_rates.argmax(), (H, W))
            worst_rate = cell_rates[worst_idx] * 100
            worst_name = CELL_NAMES[worst_idx[0] * W + worst_idx[1]]
            print(f"  {h:8d}  {cf:10.1f}%  {worst_name:>12s}  {worst_rate:7.1f}%")

    return {
        "cumulative_floor_rate": {str(h): float(cum_floor_rate[h-1]) for h in HORIZONS if h <= T},
        "per_cell_floor_rate_h252": per_cell_cum_rate[-1].tolist() if T >= 252 else None,
    }


def diag2_delta_statistics(result, output_dir):
    """MLP delta statistics vs horizon: magnitude, per-cell spread, distribution shape."""
    deltas = result["deltas"]  # (N, S, T, 5, 5)
    N, S, T, H, W = deltas.shape

    # Per-step statistics across (N, S)
    delta_flat = deltas.reshape(N * S, T, H, W)

    mean_abs = np.abs(delta_flat).mean(axis=(0, 2, 3))  # (T,)
    std_delta = delta_flat.std(axis=0).mean(axis=(1, 2))  # (T,)

    # Per-cell delta mean and std at each horizon
    per_cell_mean_abs = np.abs(delta_flat).mean(axis=0)  # (T, 5, 5)
    per_cell_std = delta_flat.std(axis=0)  # (T, 5, 5)

    # Check if deltas saturate (near +-1 from tanh)
    saturation_rate = ((np.abs(delta_flat) > 0.95).mean(axis=(0, 2, 3)))  # (T,)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(range(T), mean_abs, label="Mean |delta|")
    ax.plot(range(T), std_delta, label="Std(delta)")
    for h in HORIZONS:
        if h < T:
            ax.axvline(h, color="gray", alpha=0.3, ls="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Delta statistic")
    ax.set_title("MLP delta magnitude vs step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(range(T), saturation_rate * 100)
    ax.set_xlabel("Step")
    ax.set_ylabel("Saturation rate (%)")
    ax.set_title("Tanh saturation (|delta| > 0.95)")
    ax.grid(True, alpha=0.3)

    # Per-cell delta std at h=30 vs h=252
    ax = axes[1, 0]
    h_early = min(29, T-1)
    h_late = min(251, T-1)
    x = np.arange(25)
    ax.bar(x - 0.2, per_cell_std[h_early].flatten(), 0.4, label=f"Step {h_early}", alpha=0.7)
    ax.bar(x + 0.2, per_cell_std[h_late].flatten(), 0.4, label=f"Step {h_late}", alpha=0.7)
    ax.set_xlabel("Cell")
    ax.set_ylabel("Delta std")
    ax.set_title("Per-cell delta std: early vs late")
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_NAMES[i] for i in range(25)], fontsize=5, rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Delta histogram at step 0 vs step 200
    ax = axes[1, 1]
    step0 = delta_flat[:, 0, 2, 2]  # ATM 6M at step 0
    step_late = delta_flat[:, h_late, 2, 2]  # ATM 6M at late step
    ax.hist(step0, bins=100, alpha=0.5, density=True, label=f"Step 0", range=(-1, 1))
    ax.hist(step_late, bins=100, alpha=0.5, density=True, label=f"Step {h_late}", range=(-1, 1))
    ax.set_xlabel("Delta (ATM 6M)")
    ax.set_ylabel("Density")
    ax.set_title("Delta distribution: early vs late (ATM 6M)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/d2_delta_statistics.png", dpi=150)
    plt.close()

    print("\n  D2: MLP DELTA STATISTICS")
    print(f"  {'Step':>6s}  {'Mean|d|':>8s}  {'Std(d)':>8s}  {'Satur%':>8s}")
    for h in [0, 29, 59, 89, 179, min(251, T-1)]:
        if h < T:
            print(f"  {h:6d}  {mean_abs[h]:8.4f}  {std_delta[h]:8.4f}  {saturation_rate[h]*100:7.2f}%")

    return {
        "mean_abs_delta": {str(h): float(mean_abs[min(h-1, T-1)]) for h in HORIZONS if h <= T},
        "saturation_rate": {str(h): float(saturation_rate[min(h-1, T-1)]) for h in HORIZONS if h <= T},
    }


def diag3_condition_drift(result, output_dir):
    """GRU condition vector drift: does it leave the training distribution?"""
    cond_norms = result["condition_norms"]  # (N, S, T)
    cond_vecs = result["condition_vectors"]  # (N, S, n_steps, bottleneck_dim)
    N, S, T = cond_norms.shape
    n_cond_steps = cond_vecs.shape[2]
    subsample = 10

    # Norm evolution
    mean_norm = cond_norms.mean(axis=(0, 1))  # (T,)
    std_norm = cond_norms.std(axis=(0, 1))

    # Condition similarity to initial condition (step 0)
    # Use cosine similarity
    initial_cond = cond_vecs[:, :, 0:1, :]  # (N, S, 1, D)
    cos_sim = np.zeros(n_cond_steps)
    for t in range(n_cond_steps):
        a = cond_vecs[:, :, t]  # (N, S, D)
        b = initial_cond[:, :, 0]  # (N, S, D)
        dot = (a * b).sum(axis=-1)
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        cos_sim[t] = (dot / (norm_a * norm_b + 1e-10)).mean()

    # PCA of condition vectors to check for drift
    all_conds = cond_vecs.reshape(-1, cond_vecs.shape[-1])  # (N*S*n_steps, D)
    # Subsample for efficiency
    idx = np.random.choice(len(all_conds), min(5000, len(all_conds)), replace=False)
    sub_conds = all_conds[idx]
    mean_c = sub_conds.mean(axis=0)
    centered = sub_conds - mean_c
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto first 2 PCs
    proj_2d = centered @ Vt[:2].T
    # Color by step
    step_indices = np.array([i % n_cond_steps for i in range(len(all_conds))])[idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(range(T), mean_norm, lw=1)
    ax.fill_between(range(T), mean_norm - std_norm, mean_norm + std_norm, alpha=0.3)
    for h in HORIZONS:
        if h < T:
            ax.axvline(h, color="gray", alpha=0.3, ls="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Condition L2 norm")
    ax.set_title("GRU condition norm vs step")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(np.arange(n_cond_steps) * subsample, cos_sim)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine similarity to step 0")
    ax.set_title("Condition drift from initial state")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    scatter = ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=step_indices,
                         cmap="viridis", s=3, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label="Step (x10)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Condition vector PCA (colored by step)")
    ax.grid(True, alpha=0.3)

    # Explained variance
    ax = axes[1, 1]
    explained = s**2 / (s**2).sum()
    ax.bar(range(min(20, len(explained))), explained[:20])
    ax.set_xlabel("PC index")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Condition PCA spectrum")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/d3_condition_drift.png", dpi=150)
    plt.close()

    print("\n  D3: CONDITION VECTOR DRIFT")
    print(f"  {'Step':>6s}  {'Norm':>8s}  {'CosSim':>8s}")
    for t_idx in range(0, n_cond_steps, max(1, n_cond_steps // 6)):
        step = t_idx * subsample
        print(f"  {step:6d}  {cond_norms[:, :, min(step, T-1)].mean():8.4f}  {cos_sim[t_idx]:8.4f}")

    return {
        "norm_step0": float(mean_norm[0]),
        "norm_step252": float(mean_norm[min(251, T-1)]),
        "cosine_similarity_step252": float(cos_sim[-1]),
        "explained_variance_pc1": float(explained[0]),
    }


def diag4_spatial_correlation(result, gt_future, output_dir):
    """Cross-cell correlation matrix at different horizons."""
    samples = result["samples"]  # (N, S, T, 5, 5)
    N, S, T, H, W = samples.shape
    gt_T = gt_future.shape[1]

    fig, axes = plt.subplots(2, len(HORIZONS), figsize=(4 * len(HORIZONS), 8))

    results = {}
    print("\n  D4: CROSS-CELL CORRELATION STRUCTURE")
    print(f"  {'Horizon':>8s}  {'GenLevel':>10s}  {'GenDaily':>10s}  {'GTLevel':>10s}  {'GTDaily':>10s}")

    for i, h in enumerate(HORIZONS):
        if h > T:
            continue
        # Window of daily changes around horizon h: [max(0,h-30):h]
        start = max(0, h - 30)
        end = min(h, T)

        # Generated daily changes
        gen_daily = np.diff(samples[:, :, start:end], axis=2)  # (N, S, window, 5, 5)
        gen_daily_flat = gen_daily.reshape(-1, 25)
        gen_corr = np.corrcoef(gen_daily_flat, rowvar=False)

        # Generated levels
        gen_levels = samples[:, :, start:end].reshape(-1, 25)
        gen_level_corr = np.corrcoef(gen_levels, rowvar=False)

        # GT
        gt_end = min(end, gt_T)
        gt_start = max(0, gt_end - 30)
        if gt_end > gt_start + 1:
            gt_daily = np.diff(gt_future[:, gt_start:gt_end], axis=1).reshape(-1, 25)
            gt_corr = np.corrcoef(gt_daily, rowvar=False)
            gt_levels = gt_future[:, gt_start:gt_end].reshape(-1, 25)
            gt_level_corr = np.corrcoef(gt_levels, rowvar=False)
        else:
            gt_corr = np.full((25, 25), np.nan)
            gt_level_corr = np.full((25, 25), np.nan)

        gen_upper = gen_corr[np.triu_indices(25, k=1)]
        gt_upper = gt_corr[np.triu_indices(25, k=1)]
        gen_level_upper = gen_level_corr[np.triu_indices(25, k=1)]
        gt_level_upper = gt_level_corr[np.triu_indices(25, k=1)]

        gen_daily_mean = np.nanmean(gen_upper)
        gt_daily_mean = np.nanmean(gt_upper)
        gen_level_mean = np.nanmean(gen_level_upper)
        gt_level_mean = np.nanmean(gt_level_upper)

        print(f"  {h:8d}  {gen_level_mean:10.3f}  {gen_daily_mean:10.3f}  "
              f"{gt_level_mean:10.3f}  {gt_daily_mean:10.3f}")

        results[str(h)] = {
            "gen_daily_corr": float(gen_daily_mean),
            "gt_daily_corr": float(gt_daily_mean),
            "gen_level_corr": float(gen_level_mean),
            "gt_level_corr": float(gt_level_mean),
        }

        # Plot correlation matrices
        ax = axes[0, i]
        im = ax.imshow(gen_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"Gen h={h}", fontsize=9)
        if i == 0:
            ax.set_ylabel("Generated daily-change corr")

        ax = axes[1, i]
        im = ax.imshow(gt_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"GT h={h}", fontsize=9)
        if i == 0:
            ax.set_ylabel("GT daily-change corr")

    plt.colorbar(im, ax=axes, shrink=0.6, label="Correlation")
    plt.suptitle("Cross-cell daily-change correlation by horizon", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/d4_spatial_correlation.png", dpi=150)
    plt.close()

    return results


def diag5_spatial_structure_evolution(result, gt_future, output_dir):
    """Continuous term slope and smile convexity evolution."""
    samples = result["samples"]  # (N, S, T, 5, 5)
    N, S, T, H, W = samples.shape
    gt_T = gt_future.shape[1]

    mean_surf = samples.mean(axis=1)  # (N, T, 5, 5)
    gt_mean = gt_future  # (N, T_gt, 5, 5) — already single sample

    # Term slope: ATM col=2, tenor row 4 - row 0
    gen_term_slope = (mean_surf[:, :, 4, 2] - mean_surf[:, :, 0, 2]).mean(axis=0)  # (T,)
    gt_term_slope = (gt_mean[:, :, 4, 2] - gt_mean[:, :, 0, 2]).mean(axis=0)  # (T_gt,)

    # Smile convexity: (K0 + K4)/2 - K2, averaged across tenors
    gen_smile = ((mean_surf[:, :, :, 0] + mean_surf[:, :, :, 4]) / 2 - mean_surf[:, :, :, 2]).mean(axis=(0, 2))
    gt_smile = ((gt_mean[:, :, :, 0] + gt_mean[:, :, :, 4]) / 2 - gt_mean[:, :, :, 2]).mean(axis=(0, 2))

    # Skew: K=0.70 - K=1.30 (put-call skew), averaged across tenors
    gen_skew = (mean_surf[:, :, :, 0] - mean_surf[:, :, :, 4]).mean(axis=(0, 2))
    gt_skew = (gt_mean[:, :, :, 0] - gt_mean[:, :, :, 4]).mean(axis=(0, 2))

    # Tenor spread: std across tenors at ATM (col=2)
    gen_tenor_spread = mean_surf[:, :, :, 2].std(axis=2).mean(axis=0)  # (T,)
    gt_tenor_spread = gt_mean[:, :, :, 2].std(axis=2).mean(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, gen, gt, title in [
        (axes[0, 0], gen_term_slope, gt_term_slope, "Term slope (ATM 2Y - ATM 1M)"),
        (axes[0, 1], gen_smile, gt_smile, "Smile convexity (wings avg - ATM)"),
        (axes[1, 0], gen_skew, gt_skew, "Put-call skew (K=0.70 - K=1.30)"),
        (axes[1, 1], gen_tenor_spread, gt_tenor_spread, "Tenor spread (ATM std across tenors)"),
    ]:
        ax.plot(range(len(gen)), gen, label="Generated", lw=1)
        ax.plot(range(min(len(gt), T)), gt[:T], label="GT", lw=1, ls="--", color="red")
        for h in HORIZONS:
            if h < T:
                ax.axvline(h, color="gray", alpha=0.3, ls="--")
        ax.set_xlabel("Horizon (days)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/d5_spatial_evolution.png", dpi=150)
    plt.close()

    print("\n  D5: SPATIAL STRUCTURE EVOLUTION")
    print(f"  {'Horizon':>8s}  {'TermSlope':>10s}  {'GT_TS':>8s}  {'Smile':>8s}  {'GT_Sm':>8s}  {'Skew':>8s}  {'GT_Sk':>8s}")
    for h in HORIZONS:
        if h <= T:
            gs = gen_term_slope[h-1]
            gts = gt_term_slope[min(h-1, gt_T-1)] if gt_T > 0 else float("nan")
            gsm = gen_smile[h-1]
            gtsm = gt_smile[min(h-1, gt_T-1)] if gt_T > 0 else float("nan")
            gsk = gen_skew[h-1]
            gtsk = gt_skew[min(h-1, gt_T-1)] if gt_T > 0 else float("nan")
            print(f"  {h:8d}  {gs:10.4f}  {gts:8.4f}  {gsm:8.4f}  {gtsm:8.4f}  {gsk:8.4f}  {gtsk:8.4f}")


def diag6_absorbing_barrier(result, output_dir):
    """Once a cell hits the floor, can it escape? Analyze transition dynamics."""
    samples = result["samples"]  # (N, S, T, 5, 5)
    N, S, T, H, W = samples.shape

    floor_thresh = 0.01
    escape_thresh = 0.05  # must rise above this to "escape"

    # Track per-cell: time of first floor hit, then check if it ever escapes
    n_floor_hit = 0
    n_escaped = 0
    time_to_escape = []
    stuck_duration = []

    for n in range(N):
        for s in range(S):
            for r in range(H):
                for c in range(W):
                    path = samples[n, s, :, r, c]
                    floor_times = np.where(path < floor_thresh)[0]
                    if len(floor_times) == 0:
                        continue
                    first_hit = floor_times[0]
                    n_floor_hit += 1
                    # Check if it escapes after first hit
                    post_hit = path[first_hit:]
                    escape_times = np.where(post_hit > escape_thresh)[0]
                    if len(escape_times) > 0:
                        n_escaped += 1
                        time_to_escape.append(escape_times[0])
                    else:
                        stuck_duration.append(T - first_hit)

    escape_rate = n_escaped / max(n_floor_hit, 1)
    mean_escape_time = np.mean(time_to_escape) if time_to_escape else float("nan")
    mean_stuck = np.mean(stuck_duration) if stuck_duration else float("nan")

    # Plot: histogram of time-to-escape vs stuck duration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if time_to_escape:
        ax.hist(time_to_escape, bins=50, alpha=0.7, label=f"Escaped (n={n_escaped})")
    if stuck_duration:
        ax.hist(stuck_duration, bins=50, alpha=0.7, label=f"Stuck (n={len(stuck_duration)})")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Count")
    ax.set_title(f"Absorbing barrier: escape rate={escape_rate:.1%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-cell floor absorption rate
    ax = axes[1]
    cell_floor_count = np.zeros(25)
    cell_escape_count = np.zeros(25)
    for n in range(min(N, 20)):  # Limit for speed
        for s in range(min(S, 10)):
            for ci in range(25):
                r, c = ci // 5, ci % 5
                path = samples[n, s, :, r, c]
                if (path < floor_thresh).any():
                    cell_floor_count[ci] += 1
                    first_hit = np.where(path < floor_thresh)[0][0]
                    if (path[first_hit:] > escape_thresh).any():
                        cell_escape_count[ci] += 1

    cell_escape_rate = np.where(cell_floor_count > 0,
                                cell_escape_count / cell_floor_count, 1.0)
    ax.bar(range(25), cell_escape_rate * 100, alpha=0.7)
    ax.set_xlabel("Cell")
    ax.set_ylabel("Escape rate (%)")
    ax.set_title("Per-cell escape rate after floor hit")
    ax.set_xticks(range(25))
    ax.set_xticklabels(CELL_NAMES, fontsize=5, rotation=90)
    ax.axhline(50, color="red", ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/d6_absorbing_barrier.png", dpi=150)
    plt.close()

    print(f"\n  D6: ABSORBING BARRIER ANALYSIS")
    print(f"  Floor hits: {n_floor_hit} (across all window/sample/cell combos)")
    print(f"  Escape rate: {escape_rate:.1%} (escaped above {escape_thresh})")
    print(f"  Mean escape time: {mean_escape_time:.1f} days")
    print(f"  Mean stuck duration: {mean_stuck:.1f} days")

    return {
        "n_floor_hits": n_floor_hit,
        "escape_rate": float(escape_rate),
        "mean_escape_time": float(mean_escape_time) if not np.isnan(mean_escape_time) else None,
        "mean_stuck_duration": float(mean_stuck) if not np.isnan(mean_stuck) else None,
    }


def diag7_daily_change_volatility(result, gt_future, output_dir):
    """Per-cell daily change volatility vs horizon: does it decay (dampening) or stay constant?"""
    samples = result["samples"]  # (N, S, T, 5, 5)
    N, S, T, H, W = samples.shape
    gt_T = gt_future.shape[1]

    daily_changes = np.diff(samples, axis=2)  # (N, S, T-1, 5, 5)
    gt_daily = np.diff(gt_future, axis=1)  # (N, T_gt-1, 5, 5)

    # Rolling std of daily changes per cell (window=30)
    window = 30
    n_windows_gen = (T - 1) // window
    n_windows_gt = (gt_T - 1) // window

    gen_rolling_std = np.zeros((n_windows_gen, H, W))
    gt_rolling_std = np.zeros((n_windows_gt, H, W))

    for i in range(n_windows_gen):
        start = i * window
        end = start + window
        chunk = daily_changes[:, :, start:end].reshape(-1, window, H, W)
        gen_rolling_std[i] = chunk.std(axis=(0, 1))

    for i in range(n_windows_gt):
        start = i * window
        end = start + window
        chunk = gt_daily[:, start:end].reshape(-1, window, H, W)
        gt_rolling_std[i] = chunk.std(axis=(0, 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Selected cells
    cells = [(0, 0), (0, 2), (2, 2), (4, 2), (2, 4)]
    cell_labels = ["1M/K=0.70", "1M/ATM", "6M/ATM", "2Y/ATM", "6M/K=1.30"]

    ax = axes[0]
    for (r, c), label in zip(cells, cell_labels):
        ax.plot(np.arange(n_windows_gen) * window + window // 2,
                gen_rolling_std[:, r, c], label=f"Gen {label}", marker="o", ms=3)
    ax.set_xlabel("Horizon mid-point (days)")
    ax.set_ylabel("Daily change std")
    ax.set_title("Generated: daily change volatility vs horizon")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for (r, c), label in zip(cells, cell_labels):
        gen_vals = gen_rolling_std[:, r, c]
        gt_vals = gt_rolling_std[:min(n_windows_gt, len(gen_vals)), r, c]
        ratio = gen_vals[:len(gt_vals)] / (gt_vals + 1e-10)
        ax.plot(np.arange(len(ratio)) * window + window // 2,
                ratio, label=label, marker="o", ms=3)
    ax.axhline(1.0, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Horizon mid-point (days)")
    ax.set_ylabel("Gen/GT daily change std ratio")
    ax.set_title("Daily change volatility ratio (gen/gt)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/d7_daily_change_volatility.png", dpi=150)
    plt.close()

    print("\n  D7: DAILY CHANGE VOLATILITY")
    print(f"  {'Cell':>12s}  {'Early':>8s}  {'Late':>8s}  {'Ratio':>8s}  {'GT_Early':>10s}")
    for (r, c), label in zip(cells, cell_labels):
        early = gen_rolling_std[0, r, c] if n_windows_gen > 0 else float("nan")
        late = gen_rolling_std[-1, r, c] if n_windows_gen > 0 else float("nan")
        ratio = late / (early + 1e-10)
        gt_early = gt_rolling_std[0, r, c] if n_windows_gt > 0 else float("nan")
        print(f"  {label:>12s}  {early:8.4f}  {late:8.4f}  {ratio:8.2f}  {gt_early:10.4f}")


def diag8_within_tenor_correlation(result, gt_future, output_dir):
    """Within-tenor vs across-tenor correlation to identify spatial structure breakdown."""
    samples = result["samples"]  # (N, S, T, 5, 5)
    N, S, T, H, W = samples.shape
    gt_T = gt_future.shape[1]

    print("\n  D8: WITHIN-TENOR vs ACROSS-TENOR CORRELATION")
    print(f"  {'Horizon':>8s}  {'Gen_Within':>12s}  {'Gen_Across':>12s}  {'GT_Within':>12s}  {'GT_Across':>12s}")

    results = {}
    for h in HORIZONS:
        if h > T:
            continue
        start = max(0, h - 30)
        end = min(h, T)

        gen_daily = np.diff(samples[:, :, start:end], axis=2).reshape(-1, H, W)
        gen_corr = np.zeros((25, 25))
        for i in range(25):
            ri, ci = i // 5, i % 5
            for j in range(i, 25):
                rj, cj = j // 5, j % 5
                corr = np.corrcoef(gen_daily[:, ri, ci], gen_daily[:, rj, cj])[0, 1]
                gen_corr[i, j] = corr
                gen_corr[j, i] = corr

        # Within-tenor: same row (same tenor), different strike
        within_tenor = []
        across_tenor = []
        for i in range(25):
            ri = i // 5
            for j in range(i+1, 25):
                rj = j // 5
                if np.isfinite(gen_corr[i, j]):
                    if ri == rj:
                        within_tenor.append(gen_corr[i, j])
                    else:
                        across_tenor.append(gen_corr[i, j])

        gen_within = np.mean(within_tenor) if within_tenor else 0
        gen_across = np.mean(across_tenor) if across_tenor else 0

        # GT
        gt_end = min(end, gt_T)
        gt_start = max(0, gt_end - 30)
        gt_within, gt_across = 0, 0
        if gt_end > gt_start + 1:
            gt_daily = np.diff(gt_future[:, gt_start:gt_end], axis=1).reshape(-1, H, W)
            gt_corr_mat = np.zeros((25, 25))
            for i in range(25):
                ri, ci = i // 5, i % 5
                for j in range(i, 25):
                    rj, cj = j // 5, j % 5
                    c = np.corrcoef(gt_daily[:, ri, ci], gt_daily[:, rj, cj])[0, 1]
                    gt_corr_mat[i, j] = c
                    gt_corr_mat[j, i] = c

            wt, at = [], []
            for i in range(25):
                ri = i // 5
                for j in range(i+1, 25):
                    rj = j // 5
                    if np.isfinite(gt_corr_mat[i, j]):
                        if ri == rj:
                            wt.append(gt_corr_mat[i, j])
                        else:
                            at.append(gt_corr_mat[i, j])
            gt_within = np.mean(wt) if wt else 0
            gt_across = np.mean(at) if at else 0

        print(f"  {h:8d}  {gen_within:12.3f}  {gen_across:12.3f}  {gt_within:12.3f}  {gt_across:12.3f}")
        results[str(h)] = {
            "gen_within_tenor": float(gen_within),
            "gen_across_tenor": float(gen_across),
            "gt_within_tenor": float(gt_within),
            "gt_across_tenor": float(gt_across),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/afcrps_90m/best_model.pt")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_windows", type=int, default=50)
    parser.add_argument("--n_frames", type=int, default=252)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pos_mode", type=str, default="native")
    parser.add_argument("--floor_clamp", type=float, default=None,
                        help="Override ar_frame_floor_clamp (e.g. 0.01 for Exp 91d)")
    parser.add_argument("--output_dir", type=str,
                        default="results/block_ar/long_horizon_diagnosis")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LONG-HORIZON DRIFT & SPATIAL DEGRADATION DIAGNOSIS")
    print("=" * 70)

    model, ckpt = load_model(args.model_path, device, no_ema=args.no_ema)
    if args.floor_clamp is not None:
        model.config.ar_frame_floor_clamp = args.floor_clamp
        print(f"  Floor clamp overridden: {args.floor_clamp}")
    print(f"  Model: {args.model_path}")
    print(f"  ar_dual_pos={model.config.ar_dual_pos}, "
          f"ar_frame_rho={model.config.ar_frame_rho}, "
          f"future_len={model.config.future_len}")

    # Load test data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    returns = data["ret"]
    history_len = 30
    gt_future_len = min(args.n_frames, len(surfaces) - 4540 - history_len)
    dataset = VolSurfaceDataset(surfaces, history_len, gt_future_len, start_idx=4540)
    n_windows = min(args.n_windows, len(dataset))

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=n_windows, shuffle=False)
    batch = next(iter(loader))
    history_t = batch["history"][:n_windows].to(device)
    future_t = batch["future"][:n_windows]
    history_np = denormalize_iv(history_t).cpu().numpy()
    gt_future_np = denormalize_iv(future_t).numpy()

    print(f"  Windows: {n_windows}, GT future: {gt_future_len} days")
    print(f"  Generating {args.n_samples} samples x {args.n_frames} frames...")

    # Instrumented generation — process in batches to manage GPU memory
    all_results = []
    t0 = time.time()
    for start in range(0, n_windows, args.batch_size):
        end = min(start + args.batch_size, n_windows)
        batch_hist = history_t[start:end]
        batch_result = instrumented_ar_generation(
            model, batch_hist, args.n_samples, args.n_frames,
            position_mode=args.pos_mode,
        )
        all_results.append(batch_result)
        print(f"  Batch {start}-{end} done")

    # Concatenate batch results
    result = {
        key: np.concatenate([r[key] for r in all_results], axis=0)
        for key in all_results[0].keys()
    }
    elapsed = time.time() - t0
    print(f"  Generation: {elapsed:.1f}s, shape={result['samples'].shape}")

    # Run all diagnostics
    print("\n" + "=" * 70)
    d1 = diag1_drift_trajectories(result, gt_future_np, args.output_dir)
    print("\n" + "=" * 70)
    d2 = diag2_delta_statistics(result, args.output_dir)
    print("\n" + "=" * 70)
    d3 = diag3_condition_drift(result, args.output_dir)
    print("\n" + "=" * 70)
    d4 = diag4_spatial_correlation(result, gt_future_np, args.output_dir)
    print("\n" + "=" * 70)
    diag5_spatial_structure_evolution(result, gt_future_np, args.output_dir)
    print("\n" + "=" * 70)
    d6 = diag6_absorbing_barrier(result, args.output_dir)
    print("\n" + "=" * 70)
    diag7_daily_change_volatility(result, gt_future_np, args.output_dir)
    print("\n" + "=" * 70)
    d8 = diag8_within_tenor_correlation(result, gt_future_np, args.output_dir)

    # Save all results
    summary = {
        "model_path": args.model_path,
        "n_windows": n_windows,
        "n_samples": args.n_samples,
        "n_frames": args.n_frames,
        "pos_mode": args.pos_mode,
        "generation_time_s": elapsed,
        "drift_floor": d1,
        "delta_stats": d2,
        "condition_drift": d3,
        "spatial_correlation": d4,
        "absorbing_barrier": d6,
        "within_across_tenor_corr": d8,
    }
    with open(f"{args.output_dir}/diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"All results saved to {args.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
