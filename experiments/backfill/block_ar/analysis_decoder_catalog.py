#!/usr/bin/env python
"""
Analysis E: Decoder Inductive Bias Catalog

Systematically measure and compare the inductive biases of ALL three decoder
architectures (AR MLP, One-Shot Conv3D, Attention Denoiser) on the SAME test data
with the SAME frozen encoder.

Models:
  1. AR MLP (108a): models/backfill/afcrps_108a/best_model.pt
  2. One-Shot Conv3D (111b): models/backfill/afcrps_111b/best_model.pt
  3. Attention Denoiser (118a): models/backfill/csdi_proxy_118a/best_model.pt

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/analysis_decoder_catalog.py --device cuda
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import kurtosis as sp_kurtosis
from scipy import optimize

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig
from diffusion.block_ar.attention_denoiser import AttentionDenoiser
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_csdi_proxy import ddim_sample, cosine_beta_schedule


# ─── Config ──────────────────────────────────────────────────────────────────

OUT_DIR = Path("results/investigations/decoder_catalog")
N_SAMPLES = 50
N_WINDOWS = 200
BATCH_SIZE = 8
TEST_START = 4540
HISTORY_LEN = 30
FUTURE_LEN = 30
CELL_LABELS = [f"({i},{j})" for i in range(5) for j in range(5)]
MONEYNESS = ["0.90", "0.95", "1.00", "1.05", "1.10"]
TENORS = ["30d", "60d", "90d", "180d", "365d"]


# ─── Model loading ───────────────────────────────────────────────────────────

def load_ar_model(path, device):
    """Load AR MLP model (108a)."""
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    config = SinglePassConfig(**ckpt['config'])
    config.device = str(device)
    model = SinglePassBlockAR(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def load_oneshot_model(path, device):
    """Load One-Shot Conv3D model (111b)."""
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    config = SinglePassConfig(**ckpt['config'])
    config.device = str(device)
    model = SinglePassBlockAR(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def load_attention_model(path, encoder_path, device):
    """Load Attention Denoiser (118a) + frozen GRU encoder."""
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    cfg = ckpt['config']

    model = AttentionDenoiser(
        n_cells=25, n_steps=30, channels=cfg['channels'],
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        cond_dim=128, n_diffusion_steps=cfg['T'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load encoder from AR model checkpoint (shared encoder)
    enc_ckpt = torch.load(encoder_path, weights_only=False, map_location='cpu')
    enc_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1,
    )
    encoder = GRUEncoder(enc_config).to(device)
    enc_state = {k.replace("encoder.", ""): v
                 for k, v in enc_ckpt["model_state_dict"].items()
                 if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    return model, encoder


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_test_data(device):
    """Load test data windows from vol_surface_with_ret.npz."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]

    histories = []
    futures = []
    for i in range(TEST_START, min(TEST_START + N_WINDOWS, len(surfaces) - HISTORY_LEN - FUTURE_LEN + 1)):
        histories.append(surfaces[i:i + HISTORY_LEN])
        futures.append(surfaces[i + HISTORY_LEN:i + HISTORY_LEN + FUTURE_LEN])

    hist = torch.tensor(np.array(histories), dtype=torch.float32)  # (W, 30, 5, 5)
    fut = torch.tensor(np.array(futures), dtype=torch.float32)   # (W, 30, 5, 5)

    print(f"Loaded {hist.shape[0]} test windows, surfaces[{TEST_START}:{TEST_START + hist.shape[0]}]")
    return hist, fut, surfaces


# ─── Sample generation ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples_ar(model, hist, device):
    """Generate samples from AR MLP / One-Shot Conv3D model."""
    # Input needs to be in [-1, 1] for the model
    hist_norm = hist * 2 - 1
    loader = DataLoader(TensorDataset(hist_norm), batch_size=BATCH_SIZE, shuffle=False)

    all_samples = []
    for (batch_h,) in loader:
        batch_h = batch_h.to(device)
        samples = model.sample(batch_h, n_samples=N_SAMPLES)  # (B, K, 30, 5, 5) in [0, 1]
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)  # (W, K, 30, 5, 5)


@torch.no_grad()
def generate_samples_attention(model, encoder, hist, device):
    """Generate samples from Attention Denoiser via DDIM 50 steps."""
    hist_norm = hist * 2 - 1
    loader = DataLoader(TensorDataset(hist_norm), batch_size=BATCH_SIZE, shuffle=False)

    all_samples = []
    for (batch_h,) in loader:
        batch_h = batch_h.to(device)
        samples = ddim_sample(model, encoder, batch_h,
                              n_samples=N_SAMPLES, n_steps=50, T=200,
                              device=device)  # (B, K, 30, 5, 5) in [0, 1]
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)  # (W, K, 30, 5, 5)


# ─── Metric computation ──────────────────────────────────────────────────────

def compute_all_metrics(samples, gt_futures, gt_surfaces, name):
    """Compute all 10 metric categories for one model.

    Args:
        samples: (W, K, 30, 5, 5) in [0, 1]
        gt_futures: (W, 30, 5, 5) in [0, 1]
        gt_surfaces: full surfaces array (N, 5, 5)
    Returns:
        dict of metrics
    """
    W, K, T, H, Wd = samples.shape
    samples_np = samples.numpy()
    gt_np = gt_futures.numpy()

    print(f"\n{'='*60}")
    print(f"Computing metrics for: {name}")
    print(f"Samples shape: {samples_np.shape}, GT shape: {gt_np.shape}")
    print(f"{'='*60}")

    metrics = {"name": name}

    # ─── 1. Noise effective rank ──────────────────────────────────────
    print("  [1/10] Noise effective rank...")
    # For each window, compute cross-cell covariance of ensemble at each horizon
    eff_ranks = []
    for h in range(T):
        per_win_ranks = []
        for w in range(W):
            # ensemble members at horizon h: (K, 25)
            ens = samples_np[w, :, h, :, :].reshape(K, 25)
            ens_centered = ens - ens.mean(axis=0, keepdims=True)
            cov = np.cov(ens_centered, rowvar=False)  # (25, 25)
            eigs = np.linalg.eigvalsh(cov)
            eigs = np.maximum(eigs, 0)
            total = eigs.sum()
            if total > 0:
                p = eigs / total
                p = p[p > 1e-12]
                eff_rank = np.exp(-np.sum(p * np.log(p)))
                per_win_ranks.append(eff_rank)
        eff_ranks.append(np.mean(per_win_ranks))

    metrics["eff_rank_by_horizon"] = eff_ranks
    metrics["eff_rank_mean"] = float(np.mean(eff_ranks))
    metrics["eff_rank_h1"] = float(eff_ranks[0])
    metrics["eff_rank_h30"] = float(eff_ranks[-1])
    print(f"    eff_rank: mean={metrics['eff_rank_mean']:.2f}, h1={metrics['eff_rank_h1']:.2f}, h30={metrics['eff_rank_h30']:.2f}")

    # ─── 2. Cross-cell correlation matrix ─────────────────────────────
    print("  [2/10] Cross-cell correlation matrix...")
    # GT daily changes
    gt_full_test = gt_surfaces[TEST_START + HISTORY_LEN:]
    gt_changes = np.diff(gt_full_test, axis=0).reshape(-1, 25)  # (N-1, 25)
    gt_corr = np.corrcoef(gt_changes.T)  # (25, 25)

    # Model daily changes from ensemble median
    model_median = np.median(samples_np, axis=1)  # (W, 30, 5, 5)
    model_changes_all = []
    for w in range(W):
        for t in range(1, T):
            model_changes_all.append(model_median[w, t].flatten() - model_median[w, t-1].flatten())
    model_changes = np.array(model_changes_all)  # (W*29, 25)
    model_corr = np.corrcoef(model_changes.T)  # (25, 25)

    # Eigenspectrum
    gt_eigs = np.sort(np.linalg.eigvalsh(gt_corr))[::-1]
    model_eigs = np.sort(np.linalg.eigvalsh(model_corr))[::-1]

    # Frobenius distance
    frob_dist = np.linalg.norm(model_corr - gt_corr, 'fro')

    metrics["cross_cell_corr_mean"] = float(np.mean(model_corr[np.triu_indices(25, k=1)]))
    metrics["gt_cross_cell_corr_mean"] = float(np.mean(gt_corr[np.triu_indices(25, k=1)]))
    metrics["corr_frobenius_dist"] = float(frob_dist)
    metrics["model_eigenspectrum"] = model_eigs.tolist()
    metrics["gt_eigenspectrum"] = gt_eigs.tolist()
    metrics["model_pc1_var"] = float(model_eigs[0] / model_eigs.sum() * 100)
    metrics["gt_pc1_var"] = float(gt_eigs[0] / gt_eigs.sum() * 100)
    print(f"    cross-cell corr: model={metrics['cross_cell_corr_mean']:.3f} vs GT={metrics['gt_cross_cell_corr_mean']:.3f}")
    print(f"    Frobenius dist: {frob_dist:.3f}")
    print(f"    PC1%: model={metrics['model_pc1_var']:.1f}% vs GT={metrics['gt_pc1_var']:.1f}%")

    # Store full correlation matrices for plotting
    metrics["_model_corr"] = model_corr
    metrics["_gt_corr"] = gt_corr

    # ─── 3. Temporal ACF ──────────────────────────────────────────────
    print("  [3/10] Temporal ACF...")
    max_lag = 10
    # GT ACF (from full test set daily changes)
    gt_acfs = []
    for c in range(25):
        series = gt_changes[:, c]
        acf = compute_acf_np(series, max_lag)
        gt_acfs.append(acf)
    gt_acf_mean = np.mean(gt_acfs, axis=0)

    # Model ACF (from ensemble median daily changes per window, concatenated)
    model_acfs = []
    for c in range(25):
        series = model_changes[:, c]
        acf = compute_acf_np(series, max_lag)
        model_acfs.append(acf)
    model_acf_mean = np.mean(model_acfs, axis=0)

    metrics["model_acf_profile"] = model_acf_mean.tolist()
    metrics["gt_acf_profile"] = gt_acf_mean.tolist()
    metrics["model_acf_lag1"] = float(model_acf_mean[1])
    metrics["gt_acf_lag1"] = float(gt_acf_mean[1])
    print(f"    ACF lag-1: model={metrics['model_acf_lag1']:.3f} vs GT={metrics['gt_acf_lag1']:.3f}")

    # ─── 4. Variance ratio curve ──────────────────────────────────────
    print("  [4/10] Variance ratio curve...")
    # GT variance ratio
    gt_vr = []
    for h in range(1, T + 1):
        h_changes = []
        for w in range(W):
            diff = gt_np[w, min(h, T-1)] - gt_np[w, 0]  # h-step change
            h_changes.append(diff.flatten())
        h_var = np.var(np.array(h_changes), axis=0).mean()
        d1_changes = []
        for w in range(W):
            diff = gt_np[w, min(1, T-1)] - gt_np[w, 0]
            d1_changes.append(diff.flatten())
        d1_var = np.var(np.array(d1_changes), axis=0).mean()
        if d1_var > 0:
            gt_vr.append(h_var / d1_var)
        else:
            gt_vr.append(float('nan'))

    # Model variance ratio (from ensemble)
    model_vr = []
    for h in range(1, T + 1):
        # Across-window variance of ensemble median h-step changes
        h_changes = []
        for w in range(W):
            median_traj = np.median(samples_np[w], axis=0)  # (30, 5, 5)
            diff = median_traj[min(h, T-1)] - median_traj[0]
            h_changes.append(diff.flatten())
        h_var = np.var(np.array(h_changes), axis=0).mean()
        d1_changes = []
        for w in range(W):
            median_traj = np.median(samples_np[w], axis=0)
            diff = median_traj[min(1, T-1)] - median_traj[0]
            d1_changes.append(diff.flatten())
        d1_var = np.var(np.array(d1_changes), axis=0).mean()
        if d1_var > 0:
            model_vr.append(h_var / d1_var)
        else:
            model_vr.append(float('nan'))

    # Also compute ensemble-based VR: within-ensemble variance at each horizon
    ensemble_vr = []
    for h in range(1, T + 1):
        h_idx = min(h, T-1)
        # Within-ensemble variance at horizon h (averaged across windows)
        ens_var_h = np.var(samples_np[:, :, h_idx].reshape(W, K, 25), axis=1).mean()
        ens_var_1 = np.var(samples_np[:, :, min(1, T-1)].reshape(W, K, 25), axis=1).mean()
        if ens_var_1 > 0:
            ensemble_vr.append(ens_var_h / ens_var_1)
        else:
            ensemble_vr.append(float('nan'))

    metrics["gt_variance_ratio"] = gt_vr
    metrics["model_variance_ratio"] = model_vr
    metrics["ensemble_variance_ratio"] = ensemble_vr
    print(f"    VR(h=30): GT={gt_vr[-1]:.2f}, model_median={model_vr[-1]:.2f}, ensemble={ensemble_vr[-1]:.2f}")

    # ─── 5. Per-cell std range ────────────────────────────────────────
    print("  [5/10] Per-cell std range...")
    # GT per-cell std of daily changes
    gt_cell_std = np.std(gt_changes, axis=0)  # (25,)
    gt_std_range = gt_cell_std.max() / gt_cell_std.min() if gt_cell_std.min() > 0 else float('inf')

    # Model per-cell std (ensemble spread at h=1)
    ens_at_h1 = samples_np[:, :, 0].reshape(W * K, 25)  # All ensemble members
    # Actually compute std of changes from history to h=1
    model_cell_stds = []
    for c in range(25):
        cell_changes = model_changes[:, c]
        model_cell_stds.append(np.std(cell_changes))
    model_cell_stds = np.array(model_cell_stds)
    model_std_range = model_cell_stds.max() / model_cell_stds.min() if model_cell_stds.min() > 0 else float('inf')

    # Also compute ensemble-based per-cell std
    ens_cell_std = []
    for c in range(25):
        ci, cj = c // 5, c % 5
        ens_vals = samples_np[:, :, 0, ci, cj].flatten()
        ens_cell_std.append(np.std(ens_vals))
    ens_cell_std = np.array(ens_cell_std)
    ens_std_range = ens_cell_std.max() / ens_cell_std.min() if ens_cell_std.min() > 0 else float('inf')

    metrics["gt_cell_std"] = gt_cell_std.tolist()
    metrics["model_cell_std"] = model_cell_stds.tolist()
    metrics["ens_cell_std"] = ens_cell_std.tolist()
    metrics["gt_std_range"] = float(gt_std_range)
    metrics["model_std_range"] = float(model_std_range)
    metrics["ens_std_range"] = float(ens_std_range)
    print(f"    std range: GT={gt_std_range:.1f}x, model_changes={model_std_range:.1f}x, ensemble={ens_std_range:.1f}x")

    # ─── 6. Per-cell kurtosis ─────────────────────────────────────────
    print("  [6/10] Per-cell kurtosis...")
    gt_kurt = np.array([sp_kurtosis(gt_changes[:, c], fisher=True) for c in range(25)])
    model_kurt = np.array([sp_kurtosis(model_changes[:, c], fisher=True) for c in range(25)])

    metrics["gt_kurtosis"] = gt_kurt.tolist()
    metrics["model_kurtosis"] = model_kurt.tolist()
    metrics["gt_kurtosis_mean"] = float(np.mean(gt_kurt))
    metrics["model_kurtosis_mean"] = float(np.mean(model_kurt))
    # Kurtosis ratio (model / GT), should be near 1.0
    kurt_ratio = np.mean(model_kurt) / np.mean(gt_kurt) if np.mean(gt_kurt) != 0 else float('inf')
    metrics["kurtosis_ratio"] = float(kurt_ratio)
    print(f"    kurtosis: model_mean={metrics['model_kurtosis_mean']:.2f} vs GT_mean={metrics['gt_kurtosis_mean']:.2f}, ratio={kurt_ratio:.3f}")

    # ─── 7. Surface validity ──────────────────────────────────────────
    # Uses same convention as test_block_ar_requirements.py:
    #   surf shape: (5, 5) where row=tenor (0=short, 4=long), col=moneyness (0=ITM, 4=OTM)
    #   Calendar arb: total variance TV = IV^2 * tau must increase with tenor (row index)
    #   Butterfly arb: second derivative d^2sigma/dK^2 >= 0 along moneyness (col index)
    print("  [7/10] Surface validity...")
    tenors_relative = np.array([1, 2, 4, 8, 12])  # relative tenor multiples

    # Reshape for vectorized checks: (N*K, T, 5, 5)
    all_surfs = samples_np.reshape(W * K, T, H, Wd)

    # Explosion rate
    explosion_mask = (all_surfs.max(axis=(2, 3)) > 0.99) | (all_surfs.min(axis=(2, 3)) < 0.001)
    explosion_rate = explosion_mask.mean()

    # Calendar arbitrage: TV = IV^2 * tau, tenor is row axis
    # total_var shape: (N*K, T, 5, 5), check along tenor (axis 2)
    total_var = all_surfs ** 2 * tenors_relative[None, None, :, None]
    cal_violations = (total_var[:, :, :-1, :] > total_var[:, :, 1:, :] * 1.001)  # (N*K, T, 4, 5)
    calendar_arb_rate = cal_violations.mean()

    # Butterfly arbitrage: d^2sigma/dK^2 along moneyness (col axis)
    d2 = all_surfs[:, :, :, :-2] - 2 * all_surfs[:, :, :, 1:-1] + all_surfs[:, :, :, 2:]
    butterfly_violations = (d2 < -0.005)  # (N*K, T, 5, 3)
    butterfly_arb_rate = butterfly_violations.mean()

    metrics["explosion_rate"] = float(explosion_rate)
    metrics["calendar_arb_rate"] = float(calendar_arb_rate)
    metrics["butterfly_arb_rate"] = float(butterfly_arb_rate)
    print(f"    explosions: {metrics['explosion_rate']:.3%}")
    print(f"    calendar arb: {metrics['calendar_arb_rate']:.3%}")
    print(f"    butterfly arb: {metrics['butterfly_arb_rate']:.3%}")

    # ─── 8. Conditionality ────────────────────────────────────────────
    print("  [8/10] Conditionality (turb/calm width ratio)...")
    # Classify windows by recent volatility
    rvol = []
    for w in range(W):
        # History std (proxy for recent vol-of-vol)
        hist_changes = np.diff(gt_np[w:w+1, :FUTURE_LEN].reshape(-1, 25) if w == 0
                               else gt_np[w:w+1, :FUTURE_LEN].reshape(-1, 25), axis=0)
        # Just use history variance as regime proxy
        h_std = np.std(samples_np[w, :, :, :, :].reshape(K, -1), axis=0).mean()  # ensemble spread
        rvol.append(h_std)

    rvol = np.array(rvol)
    q33 = np.percentile(rvol, 33)
    q67 = np.percentile(rvol, 67)
    calm_mask = rvol <= q33
    turb_mask = rvol >= q67

    # Width = mean 90% CI width
    ci_widths = []
    for w in range(W):
        q05 = np.percentile(samples_np[w], 5, axis=0)
        q95 = np.percentile(samples_np[w], 95, axis=0)
        ci_widths.append(np.mean(q95 - q05))
    ci_widths = np.array(ci_widths)

    calm_width = np.mean(ci_widths[calm_mask]) if calm_mask.sum() > 0 else 0
    turb_width = np.mean(ci_widths[turb_mask]) if turb_mask.sum() > 0 else 0
    turb_calm_ratio = turb_width / calm_width if calm_width > 0 else float('inf')

    # Per-cell MAE reduction (cond vs uncond)
    ensemble_median = np.median(samples_np, axis=1)  # (W, 30, 5, 5)
    cond_mae = np.mean(np.abs(ensemble_median - gt_np), axis=(0, 1))  # (5, 5)
    # "Unconditional" baseline: global mean forecast
    global_mean = gt_np.mean(axis=(0, 1), keepdims=True)  # (1, 1, 5, 5)
    uncond_mae = np.mean(np.abs(global_mean - gt_np), axis=(0, 1))  # (5, 5)
    mae_reduction = 1 - cond_mae / uncond_mae  # positive = model better than naive
    min_mae_reduction = mae_reduction.min()

    metrics["turb_calm_width_ratio"] = float(turb_calm_ratio)
    metrics["calm_width"] = float(calm_width)
    metrics["turb_width"] = float(turb_width)
    metrics["mae_reduction_grid"] = mae_reduction.tolist()
    metrics["min_mae_reduction"] = float(min_mae_reduction)
    metrics["mean_mae_reduction"] = float(mae_reduction.mean())
    print(f"    turb/calm ratio: {turb_calm_ratio:.3f}")
    print(f"    MAE reduction: mean={mae_reduction.mean():.3f}, min={min_mae_reduction:.3f}")

    # ─── 9. Mean-reversion (OU theta estimate) ────────────────────────
    print("  [9/10] Mean-reversion (OU theta)...")
    # GT OU theta per cell
    gt_thetas = estimate_ou_theta(gt_changes)

    # Model OU theta per cell (from ensemble median daily changes)
    model_thetas = estimate_ou_theta(model_changes)

    metrics["gt_ou_theta"] = gt_thetas.tolist()
    metrics["model_ou_theta"] = model_thetas.tolist()
    metrics["gt_ou_theta_mean"] = float(np.mean(gt_thetas))
    metrics["model_ou_theta_mean"] = float(np.mean(model_thetas))
    print(f"    OU theta: model_mean={metrics['model_ou_theta_mean']:.4f} vs GT_mean={metrics['gt_ou_theta_mean']:.4f}")

    # ─── 10. Coverage metrics ─────────────────────────────────────────
    print("  [10/10] CI coverage...")
    # Per-horizon 90% CI coverage
    horizon_coverage = []
    for h in range(T):
        q05 = np.percentile(samples_np[:, :, h], 5, axis=1)  # (W, 5, 5)
        q95 = np.percentile(samples_np[:, :, h], 95, axis=1)
        covered = (gt_np[:, h] >= q05) & (gt_np[:, h] <= q95)
        horizon_coverage.append(float(covered.mean()))
    metrics["ci_coverage_by_horizon"] = horizon_coverage
    metrics["ci_coverage_mean"] = float(np.mean(horizon_coverage))
    metrics["ci_coverage_h1"] = float(horizon_coverage[0])
    metrics["ci_coverage_h30"] = float(horizon_coverage[-1])

    # Per-cell coverage (averaged over horizons)
    cell_coverage = np.zeros((5, 5))
    for h in range(T):
        q05 = np.percentile(samples_np[:, :, h], 5, axis=1)
        q95 = np.percentile(samples_np[:, :, h], 95, axis=1)
        covered = (gt_np[:, h] >= q05) & (gt_np[:, h] <= q95)
        cell_coverage += covered.mean(axis=0)
    cell_coverage /= T
    metrics["cell_coverage_grid"] = cell_coverage.tolist()
    metrics["worst_cell_coverage"] = float(cell_coverage.min())

    print(f"    CI coverage: mean={metrics['ci_coverage_mean']:.3f}, worst_cell={metrics['worst_cell_coverage']:.3f}")
    print(f"    h1={metrics['ci_coverage_h1']:.3f}, h30={metrics['ci_coverage_h30']:.3f}")

    return metrics


def compute_acf_np(series, max_lag=10):
    """Compute autocorrelation function."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
        acf[lag] = cov / var
    return acf


def estimate_ou_theta(changes):
    """Estimate OU mean-reversion rate per cell from daily changes.

    For OU process: dx = -theta * (x - mu) * dt + sigma * dW
    Discrete: x_{t+1} - x_t = -theta * (x_t - mu) + eps
    So theta ≈ -regression coefficient of change on level.
    """
    thetas = np.zeros(25)
    for c in range(25):
        y = changes[:, c]  # daily changes
        n = len(y)
        if n < 10:
            continue
        # Reconstruct levels from cumulative changes
        levels = np.cumsum(y)
        # Regress change on previous level
        x = levels[:-1]
        dy = y[1:]
        if np.std(x) > 0:
            beta = np.cov(dy, x)[0, 1] / np.var(x)
            thetas[c] = -beta  # theta = -slope
    return thetas


# ─── Visualization ────────────────────────────────────────────────────────────

def create_comparison_plots(all_metrics, out_dir):
    """Create comprehensive comparison figures."""
    print("\n" + "="*60)
    print("Creating comparison figures...")
    print("="*60)

    names = [m["name"] for m in all_metrics]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

    # ─── Figure 1: Effective rank by horizon ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for i, m in enumerate(all_metrics):
        ax.plot(range(1, 31), m["eff_rank_by_horizon"], label=m["name"],
                color=colors[i], linewidth=2)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Noise Effective Rank by Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=2.6, color='gray', linestyle='--', alpha=0.5, label='GT eff_rank')

    # Eigenspectrum
    ax = axes[1]
    for i, m in enumerate(all_metrics):
        eigs = np.array(m["model_eigenspectrum"][:10])
        ax.plot(range(1, 11), eigs / eigs.sum() * 100, 'o-', label=m["name"],
                color=colors[i], linewidth=2)
    gt_eigs = np.array(all_metrics[0]["gt_eigenspectrum"][:10])
    ax.plot(range(1, 11), gt_eigs / gt_eigs.sum() * 100, 'k--o', label='GT', linewidth=2)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Cross-Cell Eigenspectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Eff rank bar chart summary
    ax = axes[2]
    x = np.arange(len(names))
    vals = [m["eff_rank_mean"] for m in all_metrics]
    bars = ax.bar(x, vals, color=colors, alpha=0.8, width=0.6)
    ax.axhline(y=2.6, color='gray', linestyle='--', alpha=0.7, label='GT ~2.6')
    ax.set_ylabel("Mean Effective Rank")
    ax.set_title("Mean Effective Rank (GT ~2.6)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / "01_effective_rank.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 2: Correlation matrices ───────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    vmin, vmax = -0.2, 1.0
    cmap = 'RdBu_r'

    # GT
    ax = axes[0]
    im = ax.imshow(all_metrics[0]["_gt_corr"], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"GT (mean corr={all_metrics[0]['gt_cross_cell_corr_mean']:.3f})")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Cell")
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i, m in enumerate(all_metrics):
        ax = axes[i + 1]
        im = ax.imshow(m["_model_corr"], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{m['name']} (corr={m['cross_cell_corr_mean']:.3f})")
        ax.set_xlabel("Cell")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_dir / "02_correlation_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 3: Temporal ACF ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    lags = list(range(0, 11))
    ax.plot(lags, all_metrics[0]["gt_acf_profile"], 'k--o', label='GT', linewidth=2)
    for i, m in enumerate(all_metrics):
        ax.plot(lags, m["model_acf_profile"], 'o-', label=m["name"],
                color=colors[i], linewidth=2)
    ax.axhline(y=0, color='gray', alpha=0.3)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF")
    ax.set_title("Mean Autocorrelation of Daily Changes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ACF lag-1 bar chart
    ax = axes[1]
    x = np.arange(len(names) + 1)
    vals = [all_metrics[0]["gt_acf_lag1"]] + [m["model_acf_lag1"] for m in all_metrics]
    bar_colors = ['gray'] + colors
    bar_labels = ['GT'] + names
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.8, width=0.6)
    ax.set_ylabel("ACF Lag-1")
    ax.set_title("Lag-1 Autocorrelation")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in bar_labels], fontsize=9)
    for bar, val in zip(bars, vals):
        y_offset = 0.02 if val >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
    ax.axhline(y=0, color='gray', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / "03_temporal_acf.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 4: Variance ratio curves ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    horizons = list(range(1, 31))

    # Median-based VR
    ax = axes[0]
    ax.plot(horizons, all_metrics[0]["gt_variance_ratio"], 'k--', label='GT', linewidth=2)
    for i, m in enumerate(all_metrics):
        ax.plot(horizons, m["model_variance_ratio"], label=m["name"],
                color=colors[i], linewidth=2)
    ax.plot(horizons, horizons, ':', color='gray', alpha=0.5, label='Random walk (VR=h)')
    ax.set_xlabel("Horizon h (days)")
    ax.set_ylabel("Variance Ratio VR(h)")
    ax.set_title("Cross-Window Variance Ratio (from median)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ensemble-based VR
    ax = axes[1]
    for i, m in enumerate(all_metrics):
        ax.plot(horizons, m["ensemble_variance_ratio"], label=m["name"],
                color=colors[i], linewidth=2)
    ax.plot(horizons, horizons, ':', color='gray', alpha=0.5, label='Random walk (VR=h)')
    ax.set_xlabel("Horizon h (days)")
    ax.set_ylabel("Ensemble Variance Ratio")
    ax.set_title("Within-Ensemble Variance Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "04_variance_ratio.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 5: Per-cell std range ─────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    gt_std = np.array(all_metrics[0]["gt_cell_std"]).reshape(5, 5)
    ax = axes[0]
    im = ax.imshow(gt_std, cmap='hot', aspect='auto')
    ax.set_title(f"GT std (range={all_metrics[0]['gt_std_range']:.0f}x)")
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Moneyness")
    ax.set_xticks(range(5)); ax.set_xticklabels(TENORS, fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels(MONEYNESS, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i, m in enumerate(all_metrics):
        ax = axes[i + 1]
        model_std = np.array(m["model_cell_std"]).reshape(5, 5)
        im = ax.imshow(model_std, cmap='hot', aspect='auto')
        ax.set_title(f"{m['name']} (range={m['model_std_range']:.0f}x)")
        ax.set_xlabel("Tenor")
        ax.set_xticks(range(5)); ax.set_xticklabels(TENORS, fontsize=7)
        ax.set_yticks(range(5)); ax.set_yticklabels(MONEYNESS, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_dir / "05_percell_std.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 6: Per-cell kurtosis heatmaps ─────────────────────────
    # Use per-panel color scales since GT has extreme outlier kurtosis (81.55 mean)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    all_kurt_data = [np.array(all_metrics[0]["gt_kurtosis"]).reshape(5, 5)]
    all_kurt_labels = [f"GT kurtosis (mean={all_metrics[0]['gt_kurtosis_mean']:.1f})"]
    for m in all_metrics:
        all_kurt_data.append(np.array(m["model_kurtosis"]).reshape(5, 5))
        all_kurt_labels.append(f"{m['name']} (mean={m['model_kurtosis_mean']:.2f})")

    for idx, (kurt_grid, label) in enumerate(zip(all_kurt_data, all_kurt_labels)):
        ax = axes[idx]
        vabs = max(abs(kurt_grid.min()), abs(kurt_grid.max()), 0.1)
        im = ax.imshow(kurt_grid, cmap='coolwarm', aspect='auto',
                       norm=TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs))
        ax.set_title(label)
        ax.set_xlabel("Tenor")
        if idx == 0:
            ax.set_ylabel("Moneyness")
        ax.set_xticks(range(5)); ax.set_xticklabels(TENORS, fontsize=7)
        ax.set_yticks(range(5)); ax.set_yticklabels(MONEYNESS, fontsize=7)
        # Annotate cells with values
        for mi in range(5):
            for ti in range(5):
                val = kurt_grid[mi, ti]
                color = 'white' if abs(val) > vabs * 0.6 else 'black'
                ax.text(ti, mi, f"{val:.1f}", ha='center', va='center',
                        fontsize=6, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_dir / "06_percell_kurtosis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 7: Surface validity comparison ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    validity_metrics = ['explosion_rate', 'calendar_arb_rate', 'butterfly_arb_rate']
    validity_labels = ['Explosions', 'Calendar Arb', 'Butterfly Arb']
    x = np.arange(len(validity_labels))
    width = 0.25

    for i, m in enumerate(all_metrics):
        vals = [m[vm] * 100 for vm in validity_metrics]
        ax.bar(x + i * width, vals, width, label=m["name"], color=colors[i], alpha=0.8)

    ax.set_ylabel("Rate (%)")
    ax.set_title("Surface Validity (lower is better)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(validity_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / "07_surface_validity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 8: Conditionality ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(len(names))
    vals = [m["turb_calm_width_ratio"] for m in all_metrics]
    bars = ax.bar(x, vals, color=colors, alpha=0.8, width=0.6)
    ax.axhline(y=1.15, color='red', linestyle='--', alpha=0.7, label='Threshold 1.15')
    ax.set_ylabel("Turb/Calm Width Ratio")
    ax.set_title("Conditionality: Turb/Calm Width Ratio")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # MAE reduction heatmap for each model
    ax = axes[1]
    x = np.arange(len(names))
    mae_means = [m["mean_mae_reduction"] for m in all_metrics]
    mae_mins = [m["min_mae_reduction"] for m in all_metrics]
    bars1 = ax.bar(x - 0.15, mae_means, 0.3, label='Mean', color=colors, alpha=0.8)
    bars2 = ax.bar(x + 0.15, mae_mins, 0.3, label='Min (worst cell)', color=colors, alpha=0.4)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("MAE Reduction vs Global Mean")
    ax.set_title("Conditionality: MAE Reduction")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / "08_conditionality.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 9: OU theta (mean-reversion) ─────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    gt_theta = np.array(all_metrics[0]["gt_ou_theta"]).reshape(5, 5)
    all_thetas = [gt_theta] + [np.array(m["model_ou_theta"]).reshape(5, 5) for m in all_metrics]
    vmin_t = min(t.min() for t in all_thetas)
    vmax_t = max(t.max() for t in all_thetas)

    ax = axes[0]
    im = ax.imshow(gt_theta, cmap='RdYlGn', aspect='auto', vmin=vmin_t, vmax=vmax_t)
    ax.set_title(f"GT OU theta (mean={all_metrics[0]['gt_ou_theta_mean']:.4f})")
    ax.set_xlabel("Tenor"); ax.set_ylabel("Moneyness")
    ax.set_xticks(range(5)); ax.set_xticklabels(TENORS, fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels(MONEYNESS, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i, m in enumerate(all_metrics):
        ax = axes[i + 1]
        model_theta = np.array(m["model_ou_theta"]).reshape(5, 5)
        im = ax.imshow(model_theta, cmap='RdYlGn', aspect='auto', vmin=vmin_t, vmax=vmax_t)
        ax.set_title(f"{m['name']} (mean={m['model_ou_theta_mean']:.4f})")
        ax.set_xlabel("Tenor")
        ax.set_xticks(range(5)); ax.set_xticklabels(TENORS, fontsize=7)
        ax.set_yticks(range(5)); ax.set_yticklabels(MONEYNESS, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_dir / "09_ou_theta.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Figure 10: CI coverage ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    horizons = list(range(1, 31))
    for i, m in enumerate(all_metrics):
        ax.plot(horizons, m["ci_coverage_by_horizon"], label=m["name"],
                color=colors[i], linewidth=2)
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Target 90%')
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("90% CI Coverage")
    ax.set_title("CI Coverage by Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cell coverage heatmaps
    ax = axes[1]
    x = np.arange(len(names))
    coverage_vals = [m["ci_coverage_mean"] for m in all_metrics]
    worst_vals = [m["worst_cell_coverage"] for m in all_metrics]
    bars1 = ax.bar(x - 0.15, coverage_vals, 0.3, label='Mean', color=colors, alpha=0.8)
    bars2 = ax.bar(x + 0.15, worst_vals, 0.3, label='Worst cell', color=colors, alpha=0.4)
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='Minimum 70%')
    ax.set_ylabel("90% CI Coverage")
    ax.set_title("CI Coverage Summary")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    for bar, val in zip(bars1, coverage_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / "10_ci_coverage.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── MASTER COMPARISON FIGURE ─────────────────────────────────────
    print("Creating master comparison table figure...")
    create_master_table(all_metrics, out_dir)


def create_master_table(all_metrics, out_dir):
    """Create the definitive comparison table as a figure."""
    names = [m["name"] for m in all_metrics]

    # Collect all metrics into rows
    rows = [
        ("Effective Rank", "eff_rank_mean", "~2.6", "higher=more diverse", True),
        ("Cross-Cell Corr", "cross_cell_corr_mean", None, "lower=better (GT ref)", False),
        ("Corr Frobenius Dist", "corr_frobenius_dist", "0.0", "lower=closer to GT", False),
        ("PC1 Variance %", "model_pc1_var", None, "~52% GT", None),
        ("ACF Lag-1", "model_acf_lag1", None, "match GT", None),
        ("Kurtosis Ratio", "kurtosis_ratio", "1.0", "closer to 1.0", None),
        ("Per-Cell Std Range", "model_std_range", None, "match GT", None),
        ("Explosion Rate %", "explosion_rate", "0%", "lower=better", False),
        ("Calendar Arb %", "calendar_arb_rate", "~7% GT", "lower=better", False),
        ("Butterfly Arb %", "butterfly_arb_rate", "~20% GT", "lower=better", False),
        ("Turb/Calm Ratio", "turb_calm_width_ratio", ">1.15", "higher=more conditional", True),
        ("Mean MAE Reduction", "mean_mae_reduction", ">0", "higher=better", True),
        ("Min MAE Reduction", "min_mae_reduction", ">0", "higher=better", True),
        ("OU Theta Mean", "model_ou_theta_mean", None, "match GT", None),
        ("CI Coverage Mean", "ci_coverage_mean", "90%", "closer to 90%", None),
        ("Worst Cell Coverage", "worst_cell_coverage", ">70%", "higher=better", True),
        ("VR(30) Ensemble", "ensemble_variance_ratio", None, "monotonic growth", None),
    ]

    # Build the table data
    col_labels = ["Metric", "GT"] + names + ["Target", "Notes"]
    table_data = []
    for label, key, target, notes, higher_better in rows:
        row = [label]

        # GT value
        gt_key = "gt_" + key.replace("model_", "")
        gt_val = None
        for m in all_metrics:
            if gt_key in m:
                gt_val = m[gt_key]
                break
        if gt_val is not None:
            row.append(f"{gt_val:.3f}" if isinstance(gt_val, float) else str(gt_val))
        elif key == "ensemble_variance_ratio":
            row.append("-")
        else:
            row.append("-")

        # Model values
        for m in all_metrics:
            val = m.get(key)
            if val is None:
                row.append("-")
            elif isinstance(val, float):
                if "rate" in key.lower():
                    row.append(f"{val:.1%}")
                elif "coverage" in key.lower():
                    row.append(f"{val:.1%}")
                else:
                    row.append(f"{val:.3f}")
            elif isinstance(val, list):
                row.append(f"{val[-1]:.2f}" if val else "-")
            else:
                row.append(str(val))

        row.append(str(target) if target else "-")
        row.append(notes)
        table_data.append(row)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(22, 12))
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style metric names
    for i in range(len(table_data)):
        table[(i + 1, 0)].set_text_props(fontweight='bold')
        table[(i + 1, 0)].set_facecolor('#D9E2F3')

    # Color-code: find best value per row
    model_cols = list(range(2, 2 + len(all_metrics)))
    for i, (label, key, target, notes, higher_better) in enumerate(rows):
        if higher_better is None:
            continue
        vals = []
        for m in all_metrics:
            v = m.get(key)
            if isinstance(v, list):
                v = v[-1] if v else None
            vals.append(v)
        valid_vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
        if not valid_vals:
            continue
        best_val = max(valid_vals) if higher_better else min(valid_vals)
        for j, v in enumerate(vals):
            if v is not None and v == best_val:
                table[(i + 1, model_cols[j])].set_facecolor('#C6EFCE')
            elif v is not None:
                table[(i + 1, model_cols[j])].set_facecolor('#FFC7CE')

    ax.set_title("Analysis E: Decoder Inductive Bias Catalog\nDefinitive 3-Architecture Comparison",
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(out_dir / "00_master_comparison_table.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also save as text table
    print("\n" + "="*120)
    print("DEFINITIVE COMPARISON TABLE")
    print("="*120)
    header = f"{'Metric':<25} {'GT':>10} "
    for n in names:
        header += f" {n:>18}"
    header += f" {'Target':>10} {'Notes':>30}"
    print(header)
    print("-"*120)
    for row in table_data:
        line = f"{row[0]:<25} {row[1]:>10} "
        for val in row[2:2+len(names)]:
            line += f" {val:>18}"
        line += f" {row[-2]:>10} {row[-1]:>30}"
        print(line)
    print("="*120)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global N_WINDOWS, N_SAMPLES

    parser = argparse.ArgumentParser(description="Analysis E: Decoder Inductive Bias Catalog")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_windows", type=int, default=N_WINDOWS)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    N_WINDOWS = args.n_windows
    N_SAMPLES = args.n_samples

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Analysis E: Decoder Inductive Bias Catalog")
    print(f"Device: {device}")
    print(f"Windows: {N_WINDOWS}, Samples: {N_SAMPLES}")
    print()

    # Load test data
    hist, fut, surfaces = load_test_data(device)

    # Define models
    model_configs = [
        ("AR MLP (108a)", "models/backfill/afcrps_108a/best_model.pt", "ar"),
        ("Conv3D (111b)", "models/backfill/afcrps_111b/best_model.pt", "oneshot"),
        ("Attention (118a)", "models/backfill/csdi_proxy_118a/best_model.pt", "attention"),
    ]

    all_metrics = []
    all_samples = {}

    for name, path, model_type in model_configs:
        print(f"\n{'='*60}")
        print(f"Loading and sampling: {name}")
        print(f"{'='*60}")

        t0 = time.time()

        if model_type == "ar":
            model = load_ar_model(path, device)
            samples = generate_samples_ar(model, hist, device)
            del model
        elif model_type == "oneshot":
            model = load_oneshot_model(path, device)
            samples = generate_samples_ar(model, hist, device)  # same interface
            del model
        elif model_type == "attention":
            encoder_path = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
            model, encoder = load_attention_model(path, encoder_path, device)
            samples = generate_samples_attention(model, encoder, hist, device)
            del model, encoder

        elapsed = time.time() - t0
        print(f"  Sampling took {elapsed:.1f}s, shape: {samples.shape}")

        torch.cuda.empty_cache()
        all_samples[name] = samples

        # Compute metrics
        metrics = compute_all_metrics(samples, fut, surfaces, name)
        all_metrics.append(metrics)

        # Free memory
        del samples
        torch.cuda.empty_cache()

    # Save metrics (without numpy arrays)
    save_metrics = []
    for m in all_metrics:
        m_save = {k: v for k, v in m.items() if not k.startswith("_")}
        save_metrics.append(m_save)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\nMetrics saved to {OUT_DIR / 'metrics.json'}")

    # Create all comparison figures
    create_comparison_plots(all_metrics, OUT_DIR)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Analysis E complete.")


if __name__ == "__main__":
    main()
