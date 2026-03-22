#!/usr/bin/env python
"""
Comprehensive diagnostics for Exp 144b model.

Verifies claims:
1. Delta magnitude ratio (gen/GT) = 0.993 +/- 0.214 — re-measure on 20+ batches
2. Per-cell learned scale Spearman rho = 0.854 with GT per-cell std — re-measure
3. Per-cell spread ratio: 0/25 cells under-spread (<0.5x GT) — verify
4. Per-cell spread ratio: 17/25 cells well-calibrated (0.7-1.3x GT) — verify

Missing analyses:
5. Mean reversion ACF (lag-1 ACF of daily changes)
6. Per-cell per-horizon CI breakdown (which cells/horizons are worst?)
7. Per-cell KS daily breakdown (which cells fail KS? pattern?)

Usage:
    PYTHONPATH=. python results/validations/2026-03-22/scripts/144b_diagnostics.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Imports ──────────────────────────────────────────────────────────
from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

# ── Constants ────────────────────────────────────────────────────────
MODEL_PATH = "models/backfill/afcrps_144b/best_model.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_DIR = Path("results/validations/2026-03-22/analysis/144b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCHES = 20
BATCH_SIZE = 16
N_SAMPLES = 50
TEST_START_IDX = 4540  # Standard test split

MONEYNESS_LABELS = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]


def load_model():
    """Load 144b model from checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if isinstance(cfg, dict):
        cfg = SinglePassConfig(**cfg)
    model = SinglePassBlockAR(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(DEVICE)
    return model, checkpoint


def generate_samples(model):
    """Generate samples from 20+ test batches. Return samples, GT, history."""
    data = np.load(DATA_PATH)
    surfaces = data["surface"]
    dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=TEST_START_IDX)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_samples, all_gt, all_hist = [], [], []
    for i, batch in enumerate(tqdm(loader, desc="Generating samples", total=MAX_BATCHES)):
        if i >= MAX_BATCHES:
            break
        history = batch["history"].to(DEVICE)
        future = batch["future"].to(DEVICE)
        with torch.no_grad():
            samples = model.sample(history, n_samples=N_SAMPLES)
        all_samples.append(samples.cpu().numpy())
        all_gt.append(denormalize_iv(future).cpu().numpy())
        all_hist.append(denormalize_iv(history).cpu().numpy())

    samples = np.concatenate(all_samples, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    hist = np.concatenate(all_hist, axis=0)
    return samples, gt, hist


def compute_acf_lag1(series):
    """Compute lag-1 autocorrelation of a 1D series."""
    if len(series) < 3:
        return np.nan
    mean = np.mean(series)
    var = np.var(series)
    if var < 1e-12:
        return np.nan
    cov = np.mean((series[:-1] - mean) * (series[1:] - mean))
    return cov / var


# ── Claim 1: Delta magnitude ratio ──────────────────────────────────

def verify_delta_ratio(samples, gt, hist):
    """
    Claim: Delta magnitude ratio (gen/GT) = 0.993 +/- 0.214.
    Compute per-batch delta magnitudes for generated (median) vs GT.
    """
    N, S, T, H, W = samples.shape

    # GT deltas: from history[-1] to future
    gt_first = gt[:, 0:1] - hist[:, -1:]  # (N, 1, 5, 5)
    gt_deltas = np.diff(gt, axis=1)  # (N, 29, 5, 5)
    gt_deltas = np.concatenate([gt_first, gt_deltas], axis=1)  # (N, 30, 5, 5)

    # Generated deltas (median trajectory)
    median_traj = np.median(samples, axis=1)  # (N, 30, 5, 5)
    gen_first = median_traj[:, 0:1] - hist[:, -1:]
    gen_deltas = np.diff(median_traj, axis=1)
    gen_deltas = np.concatenate([gen_first, gen_deltas], axis=1)  # (N, 30, 5, 5)

    # Per-batch ratio of mean abs delta
    batch_size = BATCH_SIZE
    n_batches = N // batch_size
    ratios = []
    for b in range(n_batches):
        s, e = b * batch_size, (b + 1) * batch_size
        gt_mag = np.mean(np.abs(gt_deltas[s:e]))
        gen_mag = np.mean(np.abs(gen_deltas[s:e]))
        if gt_mag > 1e-10:
            ratios.append(gen_mag / gt_mag)

    # Also compute overall
    overall_gt_mag = np.mean(np.abs(gt_deltas))
    overall_gen_mag = np.mean(np.abs(gen_deltas))
    overall_ratio = overall_gen_mag / overall_gt_mag

    result = {
        "claim": "Delta magnitude ratio (gen/GT) = 0.993 +/- 0.214",
        "n_batches": len(ratios),
        "per_batch_ratios": [float(r) for r in ratios],
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "overall_ratio": float(overall_ratio),
        "overall_gt_mean_abs_delta": float(overall_gt_mag),
        "overall_gen_mean_abs_delta": float(overall_gen_mag),
    }
    print(f"\n=== CLAIM 1: Delta Magnitude Ratio ===")
    print(f"  Claimed:  0.993 +/- 0.214")
    print(f"  Measured: {result['mean_ratio']:.3f} +/- {result['std_ratio']:.3f}")
    print(f"  Overall:  {result['overall_ratio']:.3f}")
    return result


# ── Claim 2: Per-cell learned scale Spearman rho ────────────────────

def verify_learned_scale_correlation(model, samples, gt, hist):
    """
    Claim: Per-cell learned scale Spearman rho = 0.854 with GT per-cell std.
    """
    # Extract learned per-cell scale from model
    if hasattr(model.frame_decoder, 'log_vol_scale'):
        import torch.nn.functional as F
        log_vs = model.frame_decoder.log_vol_scale.data.cpu()
        learned_scale = F.softplus(log_vs).numpy()  # (25,)
    elif hasattr(model, 'cell_scale'):
        learned_scale = model.cell_scale.data.cpu().clamp(0.3, 3.0).numpy()
    else:
        return {"error": "No per-cell learned scale found"}

    # GT per-cell std from test data daily changes
    gt_first = gt[:, 0:1] - hist[:, -1:]
    gt_deltas = np.diff(gt, axis=1)
    gt_deltas = np.concatenate([gt_first, gt_deltas], axis=1)  # (N, 30, 5, 5)

    # Per-cell std of GT daily changes
    gt_cell_std = gt_deltas.reshape(-1, 5, 5).std(axis=0).flatten()  # (25,)

    # Spearman correlation
    rho, pval = stats.spearmanr(learned_scale, gt_cell_std)

    result = {
        "claim": "Per-cell learned scale Spearman rho = 0.854",
        "measured_spearman_rho": float(rho),
        "measured_pvalue": float(pval),
        "learned_scale_values": learned_scale.tolist(),
        "gt_cell_std_values": gt_cell_std.tolist(),
        "learned_scale_5x5": learned_scale.reshape(5, 5).tolist(),
        "gt_cell_std_5x5": gt_cell_std.reshape(5, 5).tolist(),
    }

    print(f"\n=== CLAIM 2: Per-Cell Learned Scale Spearman Rho ===")
    print(f"  Claimed:  0.854")
    print(f"  Measured: {rho:.3f} (p={pval:.4f})")
    print(f"  Learned scale (5x5):")
    for r in range(5):
        row = learned_scale.reshape(5, 5)[r]
        print(f"    {MATURITY_LABELS[r]}: " + " ".join(f"{v:.4f}" for v in row))
    print(f"  GT cell std (5x5):")
    for r in range(5):
        row = gt_cell_std.reshape(5, 5)[r]
        print(f"    {MATURITY_LABELS[r]}: " + " ".join(f"{v:.4f}" for v in row))

    return result


# ── Claims 3-4: Per-cell spread ratio ───────────────────────────────

def verify_per_cell_spread(samples, gt, hist):
    """
    Claim 3: 0/25 cells under-spread (<0.5x GT)
    Claim 4: 17/25 cells well-calibrated (0.7-1.3x GT)
    Compute spread at h=1 and h=30.
    """
    N, S, T, H, W = samples.shape

    results_by_horizon = {}
    for h_idx, h_label in [(0, "h1"), (14, "h15"), (29, "h30")]:
        # GT std at this horizon
        gt_frame = gt[:, h_idx]  # (N, 5, 5)
        gt_daily = gt[:, h_idx] - (gt[:, h_idx - 1] if h_idx > 0 else hist[:, -1])
        gt_std = gt_daily.std(axis=0)  # (5, 5)

        # Generated spread at this horizon (IQR / 1.35 to approximate std)
        gen_frame = samples[:, :, h_idx]  # (N, S, 5, 5)
        gen_std = gen_frame.std(axis=1).mean(axis=0)  # (5, 5) — avg across windows

        # Also try: std of daily change from generated ensemble
        if h_idx > 0:
            prev_frame = samples[:, :, h_idx - 1]
        else:
            prev_frame = np.broadcast_to(hist[:, -1:], (N, S, 5, 5)).copy()
            prev_frame = prev_frame.reshape(N, S, 5, 5)
        gen_daily = gen_frame - prev_frame  # (N, S, 5, 5)
        gen_daily_std = gen_daily.std(axis=1).mean(axis=0)  # (5, 5)

        # Ratio of generated to GT std for IV levels
        # Use level-based spread (cross-sample std at each horizon)
        # vs GT level std (cross-window std at each horizon)
        gt_level_std = gt[:, h_idx].std(axis=0)  # (5, 5)
        gen_level_std = gen_frame.std(axis=1).mean(axis=0)  # (5, 5) avg per-window spread

        ratio = gen_level_std / (gt_level_std + 1e-10)

        under_spread = np.sum(ratio.flatten() < 0.5)
        well_calibrated = np.sum((ratio.flatten() >= 0.7) & (ratio.flatten() <= 1.3))

        results_by_horizon[h_label] = {
            "gt_level_std_5x5": gt_level_std.tolist(),
            "gen_level_std_5x5": gen_level_std.tolist(),
            "ratio_5x5": ratio.tolist(),
            "ratio_flat": ratio.flatten().tolist(),
            "under_spread_count": int(under_spread),
            "well_calibrated_count": int(well_calibrated),
            "min_ratio": float(ratio.min()),
            "max_ratio": float(ratio.max()),
            "median_ratio": float(np.median(ratio)),
        }

    # Use h=1 for the primary claim verification (closest to delta-level)
    h1 = results_by_horizon["h1"]
    h30 = results_by_horizon["h30"]

    result = {
        "claim_3": "0/25 cells under-spread (<0.5x GT)",
        "claim_4": "17/25 cells well-calibrated (0.7-1.3x GT)",
        "h1_under_spread": h1["under_spread_count"],
        "h1_well_calibrated": h1["well_calibrated_count"],
        "h30_under_spread": h30["under_spread_count"],
        "h30_well_calibrated": h30["well_calibrated_count"],
        "per_horizon": results_by_horizon,
    }

    print(f"\n=== CLAIMS 3-4: Per-Cell Spread Ratio ===")
    for h_label in ["h1", "h15", "h30"]:
        hr = results_by_horizon[h_label]
        print(f"  {h_label}: under-spread={hr['under_spread_count']}/25, "
              f"well-cal={hr['well_calibrated_count']}/25, "
              f"median_ratio={hr['median_ratio']:.3f}, "
              f"range=[{hr['min_ratio']:.3f}, {hr['max_ratio']:.3f}]")

    return result


# ── Analysis 5: Mean reversion ACF ──────────────────────────────────

def analyze_mean_reversion_acf(samples, gt, hist):
    """
    Measure lag-1 ACF of daily changes per cell for both GT and generated.
    Previously only measured on 143a (ACF +0.04 vs GT -0.27).
    """
    N, S, T, H, W = samples.shape

    # GT daily changes
    gt_first = gt[:, 0:1] - hist[:, -1:]
    gt_deltas = np.diff(gt, axis=1)
    gt_deltas = np.concatenate([gt_first, gt_deltas], axis=1)  # (N, 30, 5, 5)

    # Generated daily changes (per sample member, then average ACF)
    anchor = hist[:, -1:]  # (N, 1, 5, 5)
    anchor_exp = np.broadcast_to(anchor[:, np.newaxis], (N, S, 1, H, W)).copy()
    full_traj = np.concatenate([anchor_exp, samples], axis=2)  # (N, S, 31, 5, 5)
    gen_deltas = np.diff(full_traj, axis=2)  # (N, S, 30, 5, 5)

    # Per-cell ACF analysis
    gt_acf_per_cell = np.zeros((5, 5))
    gen_acf_per_cell = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            # GT: concatenate all windows' daily changes for this cell
            gt_cell_series = gt_deltas[:, :, i, j]  # (N, 30)
            # Compute ACF for each window, then average
            gt_acfs = []
            for n in range(N):
                acf = compute_acf_lag1(gt_cell_series[n])
                if not np.isnan(acf):
                    gt_acfs.append(acf)
            gt_acf_per_cell[i, j] = np.mean(gt_acfs) if gt_acfs else np.nan

            # Generated: average across samples AND windows
            gen_acfs = []
            for n in range(N):
                for s in range(min(S, 10)):  # Sample 10 members to save time
                    gen_series = gen_deltas[n, s, :, i, j]  # (30,)
                    acf = compute_acf_lag1(gen_series)
                    if not np.isnan(acf):
                        gen_acfs.append(acf)
            gen_acf_per_cell[i, j] = np.mean(gen_acfs) if gen_acfs else np.nan

    result = {
        "description": "Lag-1 ACF of daily IV changes (mean reversion indicator)",
        "gt_acf_per_cell_5x5": gt_acf_per_cell.tolist(),
        "gen_acf_per_cell_5x5": gen_acf_per_cell.tolist(),
        "gt_acf_mean": float(np.nanmean(gt_acf_per_cell)),
        "gen_acf_mean": float(np.nanmean(gen_acf_per_cell)),
        "gt_acf_min": float(np.nanmin(gt_acf_per_cell)),
        "gen_acf_min": float(np.nanmin(gen_acf_per_cell)),
        "gt_acf_max": float(np.nanmax(gt_acf_per_cell)),
        "gen_acf_max": float(np.nanmax(gen_acf_per_cell)),
        "acf_gap_mean": float(np.nanmean(gen_acf_per_cell) - np.nanmean(gt_acf_per_cell)),
    }

    print(f"\n=== ANALYSIS 5: Mean Reversion ACF (Lag-1) ===")
    print(f"  GT  ACF mean: {result['gt_acf_mean']:.3f} (range [{result['gt_acf_min']:.3f}, {result['gt_acf_max']:.3f}])")
    print(f"  Gen ACF mean: {result['gen_acf_mean']:.3f} (range [{result['gen_acf_min']:.3f}, {result['gen_acf_max']:.3f}])")
    print(f"  Gap: {result['acf_gap_mean']:.3f}")
    print(f"  GT ACF per cell:")
    for r in range(5):
        print(f"    {MATURITY_LABELS[r]}: " + " ".join(f"{gt_acf_per_cell[r, c]:+.3f}" for c in range(5)))
    print(f"  Gen ACF per cell:")
    for r in range(5):
        print(f"    {MATURITY_LABELS[r]}: " + " ".join(f"{gen_acf_per_cell[r, c]:+.3f}" for c in range(5)))

    return result


# ── Analysis 6: Per-cell per-horizon CI breakdown ────────────────────

def analyze_ci_breakdown(samples, gt, hist):
    """
    Per-cell per-horizon CI coverage.
    Which specific cells at which horizons are worst?
    """
    N, S, T, H, W = samples.shape

    # CI coverage: for each (window, horizon, cell), is GT within 90% CI of samples?
    lower = np.percentile(samples, 5, axis=1)   # (N, T, H, W)
    upper = np.percentile(samples, 95, axis=1)  # (N, T, H, W)
    covered = (gt >= lower) & (gt <= upper)      # (N, T, H, W)

    # Per-cell per-horizon coverage
    coverage_per_cell_horizon = covered.mean(axis=0)  # (T, H, W) — avg across windows

    # Per-cell coverage (averaged across horizons)
    coverage_per_cell = coverage_per_cell_horizon.mean(axis=0)  # (H, W)

    # Per-horizon coverage (averaged across cells)
    coverage_per_horizon = coverage_per_cell_horizon.reshape(T, -1).mean(axis=1)  # (T,)

    # Find worst cells at each horizon
    worst_cells_per_horizon = {}
    for t in range(T):
        cov_flat = coverage_per_cell_horizon[t].flatten()
        worst_idx = np.argmin(cov_flat)
        worst_row, worst_col = divmod(worst_idx, 5)
        worst_cells_per_horizon[f"h{t+1}"] = {
            "worst_cell": f"{MATURITY_LABELS[worst_row]}_{MONEYNESS_LABELS[worst_col]}",
            "worst_coverage": float(cov_flat[worst_idx]),
        }

    # Find worst cells overall
    cell_cov_flat = coverage_per_cell.flatten()
    sorted_idx = np.argsort(cell_cov_flat)
    worst_5_cells = []
    for idx in sorted_idx[:5]:
        r, c = divmod(idx, 5)
        worst_5_cells.append({
            "cell": f"{MATURITY_LABELS[r]}_{MONEYNESS_LABELS[c]}",
            "row": int(r),
            "col": int(c),
            "coverage": float(cell_cov_flat[idx]),
        })

    # Find worst horizons
    sorted_h = np.argsort(coverage_per_horizon)
    worst_5_horizons = [
        {"horizon": int(h + 1), "coverage": float(coverage_per_horizon[h])}
        for h in sorted_h[:5]
    ]

    # Overall coverage
    overall_coverage = float(covered.mean())

    # Key horizons
    horizon_checkpoints = {}
    for h in [0, 4, 9, 14, 19, 24, 29]:
        if h < T:
            horizon_checkpoints[f"h{h+1}"] = {
                "overall": float(coverage_per_cell_horizon[h].mean()),
                "per_cell_5x5": coverage_per_cell_horizon[h].tolist(),
                "worst_cell_coverage": float(coverage_per_cell_horizon[h].min()),
            }

    result = {
        "description": "Per-cell per-horizon 90% CI coverage breakdown",
        "overall_coverage": overall_coverage,
        "coverage_per_cell_5x5": coverage_per_cell.tolist(),
        "coverage_per_horizon_30": coverage_per_horizon.tolist(),
        "worst_5_cells_overall": worst_5_cells,
        "worst_5_horizons": worst_5_horizons,
        "horizon_checkpoints": horizon_checkpoints,
        "cells_below_80pct": int(np.sum(cell_cov_flat < 0.80)),
        "cells_below_70pct": int(np.sum(cell_cov_flat < 0.70)),
        "cells_below_60pct": int(np.sum(cell_cov_flat < 0.60)),
    }

    print(f"\n=== ANALYSIS 6: Per-Cell Per-Horizon CI Breakdown ===")
    print(f"  Overall 90% CI coverage: {overall_coverage:.3f}")
    print(f"  Cells below 80%: {result['cells_below_80pct']}/25")
    print(f"  Cells below 70%: {result['cells_below_70pct']}/25")
    print(f"  Cells below 60%: {result['cells_below_60pct']}/25")
    print(f"  Per-cell coverage (5x5):")
    for r in range(5):
        print(f"    {MATURITY_LABELS[r]}: " + " ".join(f"{coverage_per_cell[r, c]:.3f}" for c in range(5)))
    worst_summary = [(c['cell'], round(c['coverage'], 3)) for c in worst_5_cells]
    print(f"  Worst 5 cells: {worst_summary}")
    print(f"  Coverage at key horizons:")
    for h_label in ["h1", "h5", "h10", "h15", "h20", "h25", "h30"]:
        if h_label in horizon_checkpoints:
            hc = horizon_checkpoints[h_label]
            print(f"    {h_label}: overall={hc['overall']:.3f}, worst_cell={hc['worst_cell_coverage']:.3f}")

    return result


# ── Analysis 7: Per-cell KS daily breakdown ──────────────────────────

def analyze_ks_daily(samples, gt, hist):
    """
    Per-cell KS test on daily changes.
    Which cells fail? Is there a moneyness/tenor pattern?
    """
    N, S, T, H, W = samples.shape

    # GT daily changes
    gt_first = gt[:, 0:1] - hist[:, -1:]
    gt_deltas = np.diff(gt, axis=1)
    gt_deltas = np.concatenate([gt_first, gt_deltas], axis=1)  # (N, 30, 5, 5)

    # Generated daily changes (use all samples)
    anchor = hist[:, -1:]  # (N, 1, 5, 5)
    anchor_exp = np.broadcast_to(anchor[:, np.newaxis], (N, S, 1, H, W)).copy()
    full_traj = np.concatenate([anchor_exp, samples], axis=2)  # (N, S, 31, 5, 5)
    gen_deltas = np.diff(full_traj, axis=2)  # (N, S, 30, 5, 5)

    ks_results = np.zeros((5, 5))
    ks_pvalues = np.zeros((5, 5))
    ks_pass = np.zeros((5, 5), dtype=bool)

    for i in range(5):
        for j in range(5):
            # Pool all GT daily changes for this cell
            gt_pool = gt_deltas[:, :, i, j].flatten()  # (N*30,)
            # Pool generated daily changes (subsample to keep comparable size)
            gen_pool = gen_deltas[:, :, :, i, j].flatten()  # (N*S*30,)
            # Subsample generated to match GT size (avoid overwhelming the test)
            if len(gen_pool) > len(gt_pool) * 5:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(gen_pool), size=len(gt_pool) * 5, replace=False)
                gen_pool = gen_pool[idx]

            stat, pval = stats.ks_2samp(gt_pool, gen_pool)
            ks_results[i, j] = stat
            ks_pvalues[i, j] = pval
            ks_pass[i, j] = pval > 0.05

    # Also do KS on IV levels
    ks_level_results = np.zeros((5, 5))
    ks_level_pvalues = np.zeros((5, 5))
    ks_level_pass = np.zeros((5, 5), dtype=bool)

    for i in range(5):
        for j in range(5):
            gt_levels = gt[:, :, i, j].flatten()
            gen_levels = samples[:, :, :, i, j].flatten()
            if len(gen_levels) > len(gt_levels) * 5:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(gen_levels), size=len(gt_levels) * 5, replace=False)
                gen_levels = gen_levels[idx]
            stat, pval = stats.ks_2samp(gt_levels, gen_levels)
            ks_level_results[i, j] = stat
            ks_level_pvalues[i, j] = pval
            ks_level_pass[i, j] = pval > 0.05

    result = {
        "description": "Per-cell KS test on daily changes and IV levels",
        "daily_changes": {
            "ks_statistic_5x5": ks_results.tolist(),
            "ks_pvalue_5x5": ks_pvalues.tolist(),
            "ks_pass_5x5": ks_pass.tolist(),
            "n_pass": int(ks_pass.sum()),
            "n_total": 25,
            "failing_cells": [],
        },
        "iv_levels": {
            "ks_statistic_5x5": ks_level_results.tolist(),
            "ks_pvalue_5x5": ks_level_pvalues.tolist(),
            "ks_pass_5x5": ks_level_pass.tolist(),
            "n_pass": int(ks_level_pass.sum()),
            "n_total": 25,
            "failing_cells": [],
        },
    }

    # Identify failing cells
    for i in range(5):
        for j in range(5):
            if not ks_pass[i, j]:
                result["daily_changes"]["failing_cells"].append({
                    "cell": f"{MATURITY_LABELS[i]}_{MONEYNESS_LABELS[j]}",
                    "row": i, "col": j,
                    "ks_stat": float(ks_results[i, j]),
                    "pvalue": float(ks_pvalues[i, j]),
                })
            if not ks_level_pass[i, j]:
                result["iv_levels"]["failing_cells"].append({
                    "cell": f"{MATURITY_LABELS[i]}_{MONEYNESS_LABELS[j]}",
                    "row": i, "col": j,
                    "ks_stat": float(ks_level_results[i, j]),
                    "pvalue": float(ks_level_pvalues[i, j]),
                })

    print(f"\n=== ANALYSIS 7: Per-Cell KS Daily Breakdown ===")
    print(f"  Daily changes: {result['daily_changes']['n_pass']}/25 pass (alpha=0.05)")
    print(f"  IV levels:     {result['iv_levels']['n_pass']}/25 pass (alpha=0.05)")
    print(f"  Daily KS stat (5x5):")
    for r in range(5):
        print(f"    {MATURITY_LABELS[r]}: " + " ".join(f"{ks_results[r, c]:.3f}" for c in range(5)))
    print(f"  Daily KS pass/fail:")
    for r in range(5):
        print(f"    {MATURITY_LABELS[r]}: " + " ".join("PASS" if ks_pass[r, c] else "FAIL" for c in range(5)))
    if result["daily_changes"]["failing_cells"]:
        print(f"  Failing cells (daily): {[c['cell'] for c in result['daily_changes']['failing_cells']]}")
    print(f"  Level KS pass/fail:")
    for r in range(5):
        print(f"    {MATURITY_LABELS[r]}: " + " ".join("PASS" if ks_level_pass[r, c] else "FAIL" for c in range(5)))

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Exp 144b Comprehensive Diagnostics")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model()
    print(f"  Model loaded from {MODEL_PATH}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Device: {DEVICE}")

    # Generate samples
    print(f"\nGenerating samples ({MAX_BATCHES} batches, {N_SAMPLES} samples each)...")
    samples, gt, hist = generate_samples(model)
    N = samples.shape[0]
    print(f"  Total windows: {N}")
    print(f"  Samples shape: {samples.shape}")

    # Run all verifications/analyses
    results = {}

    print("\n--- Running verifications ---")
    results["claim_1_delta_ratio"] = verify_delta_ratio(samples, gt, hist)
    results["claim_2_learned_scale_rho"] = verify_learned_scale_correlation(model, samples, gt, hist)
    results["claims_3_4_spread_ratio"] = verify_per_cell_spread(samples, gt, hist)

    print("\n--- Running missing analyses ---")
    results["analysis_5_acf"] = analyze_mean_reversion_acf(samples, gt, hist)
    results["analysis_6_ci_breakdown"] = analyze_ci_breakdown(samples, gt, hist)
    results["analysis_7_ks_daily"] = analyze_ks_daily(samples, gt, hist)

    elapsed = time.time() - start_time

    # Add metadata
    results["metadata"] = {
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "device": DEVICE,
        "max_batches": MAX_BATCHES,
        "batch_size": BATCH_SIZE,
        "n_samples": N_SAMPLES,
        "n_windows": int(N),
        "test_start_idx": TEST_START_IDX,
        "elapsed_seconds": float(elapsed),
        "checkpoint_epoch": checkpoint.get("epoch", None),
    }

    # Save all results
    output_path = OUTPUT_DIR / "diagnostics_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer, np.bool_)) else x)
    print(f"\nAll results saved to {output_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Elapsed: {elapsed:.1f}s")

    c1 = results["claim_1_delta_ratio"]
    c2 = results["claim_2_learned_scale_rho"]
    c34 = results["claims_3_4_spread_ratio"]
    a5 = results["analysis_5_acf"]
    a6 = results["analysis_6_ci_breakdown"]
    a7 = results["analysis_7_ks_daily"]

    print(f"\n  Claim 1 - Delta ratio:      claimed=0.993+/-0.214  measured={c1['mean_ratio']:.3f}+/-{c1['std_ratio']:.3f}")
    print(f"  Claim 2 - Scale Spearman:   claimed=0.854          measured={c2['measured_spearman_rho']:.3f}")
    print(f"  Claim 3 - Under-spread h1:  claimed=0/25           measured={c34['h1_under_spread']}/25")
    print(f"  Claim 4 - Well-cal h1:      claimed=17/25          measured={c34['h1_well_calibrated']}/25")
    print(f"  Analysis 5 - ACF lag-1:     GT={a5['gt_acf_mean']:.3f}  Gen={a5['gen_acf_mean']:.3f}  Gap={a5['acf_gap_mean']:.3f}")
    print(f"  Analysis 6 - CI overall:    {a6['overall_coverage']:.3f}")
    print(f"  Analysis 7 - KS daily pass: {a7['daily_changes']['n_pass']}/25")
    print(f"  Analysis 7 - KS level pass: {a7['iv_levels']['n_pass']}/25")


if __name__ == "__main__":
    main()
