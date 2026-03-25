#!/usr/bin/env python
"""
Cross-model comparison: 153a / 154b / 154c on identical test data.

Computes all 9-suite-equivalent metrics for each model:
  1. Surface Validity (explosion, calendar arb, butterfly arb)
  2. CI Coverage (per-horizon, per-cell 90% CI, per-horizon breakdown)
  3. Conditionality (turb/calm width ratio, MAE reduction)
  4. Time Series (ACF correlation, kurtosis ratio)
  5. Growing Uncertainty (monotonic transitions, spread h1/h30)
  6. Cointegration (cell-cell pass rate)
  7. Regime Coverage (per-regime per-cell CI)
  8. Distributional (KS daily changes, KS levels, median bias)
  9. Cross-Cell Correlation (correlation ratio, effective rank ratio)

Models:
  153a: ConditionalFactoredVelocityTransformer(d=128, 4 layers, cond_dim=128) — one-shot ODE
  154b: 153a base (3-run avg) + FactoredVelocityTransformer(d=64, 2 layers) — unconditional residual FM
  154c: 153a base (3-run avg) + ConditionalFactoredVelocityTransformer(d=64, 2 layers, cond_dim=128)

Data: data/vol_surface_with_ret.npz, test_start=4540, 160 windows, 50 samples each.

Usage:
    PYTHONPATH=. python results/validations/2026-03-24/scripts/154c_crossmodel.py --device cuda
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ks_2samp, kurtosis
from statsmodels.tsa.stattools import coint

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer,
    load_encoder,
    normalize_iv,
)
from experiments.backfill.block_ar.train_oneshot_flow import (
    FactoredVelocityTransformer,
)

# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def make_serial(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serial(v) for v in obj]
    return obj


def eff_rank(corr):
    ev = np.linalg.eigvalsh(corr)[::-1]
    ev = np.maximum(ev, 0)
    p = ev / (ev.sum() + 1e-10)
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def acf(series, max_lag=10):
    m = np.mean(series)
    v = np.var(series)
    if v == 0:
        return np.zeros(max_lag + 1)
    return [1.0] + [np.mean((series[:-l] - m) * (series[l:] - m)) / v
                     for l in range(1, max_lag + 1)]


# ════════════════════════════════════════════════════════════
# Sampling functions
# ════════════════════════════════════════════════════════════

def sample_153a(base_model, encoder, hist_batch, n_samples, n_steps,
                train_mean, train_std, device):
    """Generate samples from 153a (pure conditional ODE).

    Args:
        hist_batch: (B, 30, 5, 5) raw [0,1]
    Returns:
        (B, n_samples, 30, 5, 5) in [0,1]
    """
    B = hist_batch.shape[0]
    DIM = 750
    dt = 1.0 / n_steps
    mean_t = torch.from_numpy(train_mean).float().to(device)
    std_t = torch.from_numpy(train_std).float().to(device)

    cond = encoder(normalize_iv(hist_batch))            # (B, 128)
    cond_exp = cond.repeat_interleave(n_samples, dim=0) # (B*S, 128)

    x = torch.randn(B * n_samples, DIM, device=device)
    for step in range(n_steps):
        t = torch.full((B * n_samples,), step * dt, device=device)
        x = x + base_model(x, t, cond=cond_exp) * dt

    samples = (x * std_t + mean_t).clamp(0, 1)
    return samples.reshape(B, n_samples, 30, 5, 5)


def sample_base_pred(base_model, encoder, hist_batch, n_runs, n_steps,
                     train_mean, train_std, device):
    """Generate averaged base predictions from 153a (for 154b/154c).

    Args:
        hist_batch: (B, 30, 5, 5) raw [0,1]
        n_runs: number of ODE runs to average
    Returns:
        base_pred: (B, 750) in [0,1]
        cond: (B, 128) encoder condition
    """
    B = hist_batch.shape[0]
    DIM = 750
    dt = 1.0 / n_steps
    mean_t = torch.from_numpy(train_mean).float().to(device)
    std_t = torch.from_numpy(train_std).float().to(device)

    cond = encoder(normalize_iv(hist_batch))  # (B, 128)

    preds = []
    for _ in range(n_runs):
        x = torch.randn(B, DIM, device=device)
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            x = x + base_model(x, t, cond=cond) * dt
        preds.append((x * std_t + mean_t).clamp(0, 1))
    base_pred = torch.stack(preds).mean(0)  # (B, 750)
    return base_pred, cond


def sample_154b(base_pred, cond, res_model, n_samples, n_steps,
                res_mean, res_std, device):
    """Generate samples from 154b (unconditional residual FM).

    Args:
        base_pred: (B, 750) base prediction in [0,1]
        cond: unused for 154b (unconditional)
        res_model: FactoredVelocityTransformer
    Returns:
        (B, n_samples, 30, 5, 5) in [0,1]
    """
    B = base_pred.shape[0]
    DIM = 750
    dt = 1.0 / n_steps

    all_samples = []
    for s in range(n_samples):
        x = torch.randn(B, DIM, device=device)
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            x = x + res_model(x, t) * dt
        # Denormalize residual and add to base
        combined = (base_pred + x * res_std + res_mean).clamp(0, 1)
        all_samples.append(combined)

    samples = torch.stack(all_samples, dim=1)  # (B, S, 750)
    return samples.reshape(B, n_samples, 30, 5, 5)


def sample_154c(base_pred, cond, res_model, n_samples, n_steps,
                res_mean, res_std, device):
    """Generate samples from 154c (conditional residual FM).

    Args:
        base_pred: (B, 750) base prediction in [0,1]
        cond: (B, 128) encoder condition
        res_model: ConditionalFactoredVelocityTransformer
    Returns:
        (B, n_samples, 30, 5, 5) in [0,1]
    """
    B = base_pred.shape[0]
    DIM = 750
    dt = 1.0 / n_steps

    all_samples = []
    for s in range(n_samples):
        x = torch.randn(B, DIM, device=device)
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            x = x + res_model(x, t, cond=cond) * dt
        combined = (base_pred + x * res_std + res_mean).clamp(0, 1)
        all_samples.append(combined)

    samples = torch.stack(all_samples, dim=1)  # (B, S, 750)
    return samples.reshape(B, n_samples, 30, 5, 5)


# ════════════════════════════════════════════════════════════
# Full 9-suite evaluation
# ════════════════════════════════════════════════════════════

def evaluate_all_suites(cond_samples, ground_truth, rets, test_start, n_windows):
    """Compute all 9-suite metrics.

    Args:
        cond_samples: (N, S, 30, 5, 5)
        ground_truth: (N, 30, 5, 5)
        rets: full returns array
        test_start: start index
        n_windows: number of windows

    Returns:
        dict with metrics + suite pass/fail
    """
    N, S, T = cond_samples.shape[:3]

    # ── Suite 1: Surface Validity ──
    explosion_count = 0
    cal_arb_count = 0
    but_arb_count = 0
    n_check = min(N, 50)
    for i in range(n_check):
        for s in range(S):
            for t in range(T):
                surf = cond_samples[i, s, t]
                if surf.max() > 0.99 or surf.min() < 0.001:
                    explosion_count += 1
                for row in range(5):
                    for col in range(4):
                        if surf[row, col + 1] > surf[row, col] + 0.01:
                            cal_arb_count += 1
                for col in range(5):
                    for row in range(1, 4):
                        if surf[row, col] > (surf[row - 1, col] + surf[row + 1, col]) / 2 + 0.01:
                            but_arb_count += 1

    total_checked = n_check * S * T
    expl_rate = explosion_count / total_checked
    cal_arb_rate = cal_arb_count / (total_checked * 5 * 4)
    but_arb_rate = but_arb_count / (total_checked * 5 * 3)
    s1_pass = expl_rate < 0.05 and cal_arb_rate < 0.10 and but_arb_rate < 0.25

    # ── Suite 2: CI Coverage ──
    ci_pass_horizon = 0
    worst_cell_cov = 1.0
    ci_pass_cell = 0
    ci_mean_cov = 0.0

    per_horizon_ci = {}
    for h in range(T):
        gen_h = cond_samples[:, :, h, :, :]
        gt_h = ground_truth[:, h, :, :]
        lo = np.percentile(gen_h, 5, axis=1)
        hi = np.percentile(gen_h, 95, axis=1)
        covered = (gt_h >= lo) & (gt_h <= hi)
        cov_rate = covered.mean()
        ci_mean_cov += cov_rate
        per_horizon_ci[h] = float(cov_rate)
        if cov_rate >= 0.85:
            ci_pass_horizon += 1
    ci_mean_cov /= T

    for r in range(5):
        for c in range(5):
            gen_cell = cond_samples[:, :, :, r, c]
            gt_cell = ground_truth[:, :, r, c]
            lo = np.percentile(gen_cell, 5, axis=1)
            hi = np.percentile(gen_cell, 95, axis=1)
            covered = (gt_cell >= lo) & (gt_cell <= hi)
            cell_cov = covered.mean()
            worst_cell_cov = min(worst_cell_cov, cell_cov)
            if cell_cov >= 0.85:
                ci_pass_cell += 1

    s2_pass = ci_pass_horizon >= 25 and worst_cell_cov >= 0.80

    # ── Suite 3: Conditionality ──
    H = 30
    rv = np.array([np.std(rets[i:i + H]) for i in range(test_start, test_start + n_windows)])
    turb_thresh = np.percentile(rv, 80)
    calm_thresh = np.percentile(rv, 20)
    turb_mask = rv > turb_thresh
    calm_mask = rv < calm_thresh

    if turb_mask.sum() > 0 and calm_mask.sum() > 0:
        turb_spread = cond_samples[turb_mask].std(axis=1).mean()
        calm_spread = cond_samples[calm_mask].std(axis=1).mean()
        turb_calm_ratio = float(turb_spread / (calm_spread + 1e-8))
        ensemble_mean = cond_samples.mean(axis=1)
        mae_cond = np.abs(ensemble_mean - ground_truth).mean()
        overall_mean = ground_truth.mean(axis=0)
        mae_uncond = np.abs(ground_truth - overall_mean[None]).mean()
        mae_reduction = float(1 - mae_cond / mae_uncond)
    else:
        turb_calm_ratio = 1.0
        mae_reduction = 0.0

    s3_pass = turb_calm_ratio > 1.15

    # ── Suite 4: Time Series ──
    gen_changes = np.diff(cond_samples[:, 0, :, :, :], axis=1).reshape(-1)
    gt_changes = np.diff(ground_truth, axis=1).reshape(-1)
    gen_acf = acf(gen_changes)
    gt_acf = acf(gt_changes)
    acf_corr = float(np.corrcoef(gen_acf, gt_acf)[0, 1])

    gen_ch_flat = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch_flat = np.diff(ground_truth, axis=1).reshape(-1, 25)
    gen_kurt = kurtosis(gen_ch_flat.flatten(), fisher=True)
    gt_kurt = kurtosis(gt_ch_flat.flatten(), fisher=True)
    kurt_ratio = float(gen_kurt / (gt_kurt + 1e-6))

    s4_pass = acf_corr > 0.7 and 0.5 <= kurt_ratio <= 2.0

    # ── Suite 5: Growing Uncertainty ──
    spreads = []
    for h in range(T):
        spread_h = cond_samples[:, :, h].std(axis=1).mean()
        spreads.append(float(spread_h))

    mono_count = sum(1 for i in range(len(spreads) - 1) if spreads[i + 1] >= spreads[i] * 0.97)
    s5_pass = mono_count >= 20

    # ── Suite 6: Cointegration ──
    coint_pairs = 0
    coint_total = 0
    for i in range(min(N, 30)):
        sample_path = cond_samples[i, 0, :, :, :].reshape(T, 25)
        for c1 in range(0, 25, 5):
            for c2 in range(c1 + 1, min(c1 + 5, 25)):
                try:
                    _, pval, _ = coint(sample_path[:, c1], sample_path[:, c2])
                    if pval < 0.05:
                        coint_pairs += 1
                    coint_total += 1
                except Exception:
                    coint_total += 1
    coint_rate = float(coint_pairs / max(coint_total, 1))
    s6_pass = coint_rate > 0.50

    # ── Suite 7: Regime Coverage ──
    regime_pass_count = 0
    regime_details = {}
    if turb_mask.sum() > 5 and calm_mask.sum() > 5:
        for mask, name in [(turb_mask, "turb"), (calm_mask, "calm")]:
            regime_samp = cond_samples[mask]
            regime_gt = ground_truth[mask]
            cell_pass = 0
            for r in range(5):
                for c in range(5):
                    gen_cell = regime_samp[:, :, :, r, c]
                    gt_cell = regime_gt[:, :, r, c]
                    lo = np.percentile(gen_cell, 5, axis=1)
                    hi = np.percentile(gen_cell, 95, axis=1)
                    covered = (gt_cell >= lo) & (gt_cell <= hi)
                    if covered.mean() >= 0.80:
                        cell_pass += 1
            regime_details[name] = cell_pass
            if cell_pass >= 20:
                regime_pass_count += 1
        s7_pass = regime_pass_count == 2
    else:
        s7_pass = False
        regime_details = {"turb": 0, "calm": 0}

    # ── Suite 8: Distributional ──
    gen_ch = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch = np.diff(ground_truth, axis=1).reshape(-1, 25)
    ks_daily_pass = sum(1 for c in range(25) if ks_2samp(gen_ch[:, c], gt_ch[:, c])[0] < 0.15)

    gen_levels = cond_samples[:, 0, -1].reshape(-1, 25)
    gt_levels = ground_truth[:, -1].reshape(-1, 25)
    ks_level_pass = sum(1 for c in range(25) if ks_2samp(gen_levels[:, c], gt_levels[:, c])[0] < 0.15)

    gen_median = np.median(cond_samples, axis=1)
    median_bias = float((gen_median - ground_truth).mean())

    s8_pass = ks_daily_pass >= 15 and ks_level_pass >= 15

    # ── Suite 9: Cross-Cell Correlation ──
    gen_ch = np.diff(cond_samples[:, 0], axis=1).reshape(-1, 25)
    gt_ch = np.diff(ground_truth, axis=1).reshape(-1, 25)
    gen_corr = np.corrcoef(gen_ch.T)
    gt_corr = np.corrcoef(gt_ch.T)

    er_gen = eff_rank(gen_corr)
    er_gt = eff_rank(gt_corr)
    rank_ratio = er_gen / er_gt
    corr_ratio = np.abs(gen_corr).mean() / (np.abs(gt_corr).mean() + 1e-6)
    frob = float(np.linalg.norm(gen_corr - gt_corr, 'fro'))

    gt_vecs = np.linalg.eigh(gt_corr)[1][:, ::-1]
    gen_vecs = np.linalg.eigh(gen_corr)[1][:, ::-1]
    pc_aligns = [abs(float(np.dot(gt_vecs[:, i], gen_vecs[:, i]))) for i in range(5)]

    s9_pass = rank_ratio >= 0.50 and corr_ratio >= 0.60

    # ── Summary ──
    suite_names = [
        "Surface Validity", "CI Coverage", "Conditionality", "Time Series",
        "Growing Uncertainty", "Cointegration", "Regime Coverage",
        "Distributional", "Cross-Cell Correlation",
    ]
    suite_pass = [s1_pass, s2_pass, s3_pass, s4_pass, s5_pass, s6_pass, s7_pass, s8_pass, s9_pass]
    pass_count = sum(suite_pass)

    metrics = {
        "pass_count": pass_count,
        "total_suites": 9,
        "suites": {name: bool(p) for name, p in zip(suite_names, suite_pass)},
        "metrics": {
            "explosion_rate": round(float(expl_rate), 4),
            "calendar_arb_rate": round(float(cal_arb_rate), 4),
            "butterfly_arb_rate": round(float(but_arb_rate), 4),
            "ci_horizon_pass": int(ci_pass_horizon),
            "ci_worst_cell": round(float(worst_cell_cov), 4),
            "ci_mean_coverage": round(float(ci_mean_cov), 4),
            "ci_h1": round(per_horizon_ci.get(0, 0.0), 4),
            "ci_h15": round(per_horizon_ci.get(14, 0.0), 4),
            "ci_h30": round(per_horizon_ci.get(29, 0.0), 4),
            "ci_cell_pass": int(ci_pass_cell),
            "turb_calm_ratio": round(float(turb_calm_ratio), 4),
            "mae_reduction": round(float(mae_reduction), 4),
            "acf_correlation": round(float(acf_corr), 4),
            "kurt_ratio": round(float(kurt_ratio), 4),
            "spread_h1": round(float(spreads[0]), 5),
            "spread_h30": round(float(spreads[-1]), 5),
            "monotonic_transitions": int(mono_count),
            "coint_rate": round(float(coint_rate), 4),
            "regime_turb_cells": int(regime_details.get("turb", 0)),
            "regime_calm_cells": int(regime_details.get("calm", 0)),
            "ks_daily_pass": int(ks_daily_pass),
            "ks_level_pass": int(ks_level_pass),
            "median_bias": round(float(median_bias), 5),
            "eff_rank_gen": round(float(er_gen), 4),
            "eff_rank_gt": round(float(er_gt), 4),
            "rank_ratio": round(float(rank_ratio), 4),
            "corr_ratio": round(float(corr_ratio), 4),
            "frobenius": round(float(frob), 4),
            "pc_alignments": [round(float(a), 4) for a in pc_aligns],
        },
    }
    return metrics


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-model comparison: 153a / 154b / 154c")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_windows", type=int, default=160)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # ── Paths ──
    base_dir = Path(__file__).resolve().parents[4]
    base_model_path = base_dir / "models/backfill/flow_153a/final_model.pt"
    res_b_path = base_dir / "models/backfill/flow_154b/final_model.pt"
    res_c_path = base_dir / "models/backfill/flow_154c/final_model.pt"
    encoder_path = base_dir / "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
    data_path = base_dir / "data/vol_surface_with_ret.npz"

    out_dir = base_dir / "results/validations/2026-03-24"
    result_path = out_dir / "verification_results/154c_crossmodel.json"
    table_path = out_dir / "analysis/154c_crossmodel/comparison_table.json"

    # ── Load data ──
    data = np.load(str(data_path))
    surfaces = data["surface"]
    rets = data["ret"]
    H, T = 30, 30

    test_windows = []
    for i in range(args.test_start, min(args.test_start + args.n_windows,
                                         len(surfaces) - H - T + 1)):
        test_windows.append((surfaces[i:i + H], surfaces[i + H:i + H + T]))
    n_windows = len(test_windows)
    print(f"Test data: {n_windows} windows, {args.n_samples} samples each")
    print(f"Test range: [{args.test_start}, {args.test_start + n_windows})")

    # ── Load models ──
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    # 153a base model
    base_ckpt = torch.load(str(base_model_path), weights_only=False, map_location=device)
    base_cfg = base_ckpt["config"]
    base_model = ConditionalFactoredVelocityTransformer(
        n_frames=base_cfg["n_frames"], n_cells=base_cfg["n_cells"],
        d_model=base_cfg["d_model"], n_heads=base_cfg["n_heads"],
        n_layers=base_cfg["n_layers"], cond_dim=base_cfg["cond_dim"],
    )
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.to(device).eval()
    print(f"  153a: d={base_cfg['d_model']}, layers={base_cfg['n_layers']}, "
          f"params={sum(p.numel() for p in base_model.parameters()):,}")

    # Encoder
    encoder, _ = load_encoder(str(encoder_path), device)
    for p in encoder.parameters():
        p.requires_grad = False

    train_mean = base_ckpt["train_mean"]
    train_std = base_ckpt["train_std"]
    n_steps = base_cfg.get("n_steps", 8)

    # 154b residual model (unconditional)
    res_b_ckpt = torch.load(str(res_b_path), weights_only=False, map_location=device)
    res_b_cfg = res_b_ckpt["config"]
    res_b_model = FactoredVelocityTransformer(
        n_frames=res_b_cfg["n_frames"], n_cells=res_b_cfg["n_cells"],
        d_model=res_b_cfg["d_model"], n_heads=res_b_cfg["n_heads"],
        n_layers=res_b_cfg["n_layers"],
    )
    res_b_model.load_state_dict(res_b_ckpt["model_state_dict"])
    res_b_model.to(device).eval()
    res_b_mean = torch.from_numpy(np.array(res_b_ckpt["res_mean"])).float().to(device)
    res_b_std = torch.from_numpy(np.array(res_b_ckpt["res_std"])).float().to(device)
    print(f"  154b: d={res_b_cfg['d_model']}, layers={res_b_cfg['n_layers']}, "
          f"params={sum(p.numel() for p in res_b_model.parameters()):,}")

    # 154c residual model (conditional)
    res_c_ckpt = torch.load(str(res_c_path), weights_only=False, map_location=device)
    res_c_cfg = res_c_ckpt["config"]
    res_c_model = ConditionalFactoredVelocityTransformer(
        n_frames=res_c_cfg["n_frames"], n_cells=res_c_cfg["n_cells"],
        d_model=res_c_cfg["d_model"], n_heads=res_c_cfg["n_heads"],
        n_layers=res_c_cfg["n_layers"], cond_dim=res_c_cfg["cond_dim"],
    )
    res_c_model.load_state_dict(res_c_ckpt["model_state_dict"])
    res_c_model.to(device).eval()
    res_c_mean = torch.from_numpy(np.array(res_c_ckpt["res_mean"])).float().to(device)
    res_c_std = torch.from_numpy(np.array(res_c_ckpt["res_std"])).float().to(device)
    print(f"  154c: d={res_c_cfg['d_model']}, layers={res_c_cfg['n_layers']}, "
          f"cond_dim={res_c_cfg['cond_dim']}, "
          f"params={sum(p.numel() for p in res_c_model.parameters()):,}")

    # ════════════════════════════════════════════════════════════
    # Generate samples for all three models
    # ════════════════════════════════════════════════════════════

    all_gt = []
    all_153a = []
    all_154b = []
    all_154c = []

    print("\n" + "=" * 60)
    print("Generating samples...")
    print("=" * 60)

    t0_total = time.time()

    with torch.no_grad():
        for i in range(0, n_windows, args.batch_size):
            batch_end = min(i + args.batch_size, n_windows)
            batch_hist = np.array([test_windows[j][0] for j in range(i, batch_end)],
                                   dtype=np.float32)
            batch_gt = np.array([test_windows[j][1] for j in range(i, batch_end)],
                                 dtype=np.float32)
            hist_t = torch.from_numpy(batch_hist).to(device)
            B = hist_t.shape[0]

            # === 153a ===
            # Reset seed per-window-batch for reproducible prior noise
            torch.manual_seed(args.seed + i)
            samp_153a = sample_153a(base_model, encoder, hist_t, args.n_samples,
                                     n_steps, train_mean, train_std, device)
            all_153a.append(samp_153a.cpu().numpy())

            # === Base prediction for 154b/154c (3 runs) ===
            torch.manual_seed(args.seed + i + 10000)
            base_pred, cond = sample_base_pred(base_model, encoder, hist_t, 3,
                                                n_steps, train_mean, train_std, device)

            # === 154b ===
            torch.manual_seed(args.seed + i + 20000)
            samp_154b = sample_154b(base_pred, cond, res_b_model, args.n_samples,
                                     n_steps, res_b_mean, res_b_std, device)
            all_154b.append(samp_154b.cpu().numpy())

            # === 154c ===
            torch.manual_seed(args.seed + i + 30000)
            samp_154c = sample_154c(base_pred, cond, res_c_model, args.n_samples,
                                     n_steps, res_c_mean, res_c_std, device)
            all_154c.append(samp_154c.cpu().numpy())

            all_gt.append(batch_gt)

            batch_num = i // args.batch_size + 1
            total_batches = (n_windows + args.batch_size - 1) // args.batch_size
            if batch_num % 5 == 0 or batch_num == total_batches:
                elapsed = time.time() - t0_total
                print(f"  Batch {batch_num}/{total_batches} ({elapsed:.1f}s)")

    ground_truth = np.concatenate(all_gt)
    samples_153a = np.concatenate(all_153a)
    samples_154b = np.concatenate(all_154b)
    samples_154c = np.concatenate(all_154c)

    print(f"\nGeneration complete: {time.time() - t0_total:.1f}s total")
    print(f"  153a: {samples_153a.shape}")
    print(f"  154b: {samples_154b.shape}")
    print(f"  154c: {samples_154c.shape}")
    print(f"  GT:   {ground_truth.shape}")

    # ════════════════════════════════════════════════════════════
    # Evaluate all models
    # ════════════════════════════════════════════════════════════

    results = {}
    for name, samples in [("153a", samples_153a), ("154b", samples_154b),
                           ("154c", samples_154c)]:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {name}...")
        print(f"{'=' * 60}")
        t0 = time.time()
        metrics = evaluate_all_suites(samples, ground_truth, rets,
                                       args.test_start, n_windows)
        elapsed = time.time() - t0

        results[name] = metrics
        pc = metrics["pass_count"]
        print(f"  {name}: {pc}/9 suites PASS ({elapsed:.1f}s)")
        for sname, passed in metrics["suites"].items():
            print(f"    {'PASS' if passed else 'FAIL'}: {sname}")

    # ════════════════════════════════════════════════════════════
    # Print comparison table
    # ════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON TABLE")
    print("=" * 80)

    # Define metric rows for the table
    metric_rows = [
        ("Pass Count", lambda r: f"{r['pass_count']}/9"),
        ("", lambda r: ""),
        ("--- Suite 1: Surface Validity ---", None),
        ("Explosion rate", lambda r: f"{r['metrics']['explosion_rate']:.4f}"),
        ("Calendar arb rate", lambda r: f"{r['metrics']['calendar_arb_rate']:.4f}"),
        ("Butterfly arb rate", lambda r: f"{r['metrics']['butterfly_arb_rate']:.4f}"),
        ("S1 Pass", lambda r: "PASS" if r["suites"]["Surface Validity"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 2: CI Coverage ---", None),
        ("CI worst cell", lambda r: f"{r['metrics']['ci_worst_cell']:.4f}"),
        ("CI mean coverage", lambda r: f"{r['metrics']['ci_mean_coverage']:.4f}"),
        ("CI h=1", lambda r: f"{r['metrics']['ci_h1']:.4f}"),
        ("CI h=15", lambda r: f"{r['metrics']['ci_h15']:.4f}"),
        ("CI h=30", lambda r: f"{r['metrics']['ci_h30']:.4f}"),
        ("CI horizon pass", lambda r: f"{r['metrics']['ci_horizon_pass']}/30"),
        ("CI cell pass", lambda r: f"{r['metrics']['ci_cell_pass']}/25"),
        ("S2 Pass", lambda r: "PASS" if r["suites"]["CI Coverage"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 3: Conditionality ---", None),
        ("Turb/calm ratio", lambda r: f"{r['metrics']['turb_calm_ratio']:.4f}"),
        ("MAE reduction", lambda r: f"{r['metrics']['mae_reduction']:.4f}"),
        ("S3 Pass", lambda r: "PASS" if r["suites"]["Conditionality"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 4: Time Series ---", None),
        ("Kurtosis ratio", lambda r: f"{r['metrics']['kurt_ratio']:.4f}"),
        ("ACF correlation", lambda r: f"{r['metrics']['acf_correlation']:.4f}"),
        ("S4 Pass", lambda r: "PASS" if r["suites"]["Time Series"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 5: Growing Uncertainty ---", None),
        ("Monotonic transitions", lambda r: f"{r['metrics']['monotonic_transitions']}/29"),
        ("Spread h1", lambda r: f"{r['metrics']['spread_h1']:.5f}"),
        ("Spread h30", lambda r: f"{r['metrics']['spread_h30']:.5f}"),
        ("S5 Pass", lambda r: "PASS" if r["suites"]["Growing Uncertainty"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 6: Cointegration ---", None),
        ("Cointegration rate", lambda r: f"{r['metrics']['coint_rate']:.4f}"),
        ("S6 Pass", lambda r: "PASS" if r["suites"]["Cointegration"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 7: Regime Coverage ---", None),
        ("Regime turb cells pass", lambda r: f"{r['metrics']['regime_turb_cells']}/25"),
        ("Regime calm cells pass", lambda r: f"{r['metrics']['regime_calm_cells']}/25"),
        ("S7 Pass", lambda r: "PASS" if r["suites"]["Regime Coverage"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 8: Distributional ---", None),
        ("KS daily pass", lambda r: f"{r['metrics']['ks_daily_pass']}/25"),
        ("KS levels pass", lambda r: f"{r['metrics']['ks_level_pass']}/25"),
        ("Median bias", lambda r: f"{r['metrics']['median_bias']:.5f}"),
        ("S8 Pass", lambda r: "PASS" if r["suites"]["Distributional"] else "FAIL"),
        ("", lambda r: ""),
        ("--- Suite 9: Cross-Cell Correlation ---", None),
        ("Eff rank ratio", lambda r: f"{r['metrics']['rank_ratio']:.4f}"),
        ("Corr ratio", lambda r: f"{r['metrics']['corr_ratio']:.4f}"),
        ("Frobenius", lambda r: f"{r['metrics']['frobenius']:.4f}"),
        ("Eff rank (gen)", lambda r: f"{r['metrics']['eff_rank_gen']:.4f}"),
        ("Eff rank (GT)", lambda r: f"{r['metrics']['eff_rank_gt']:.4f}"),
        ("S9 Pass", lambda r: "PASS" if r["suites"]["Cross-Cell Correlation"] else "FAIL"),
    ]

    # Print as formatted table
    col_w = 26
    header = f"{'Metric':<{col_w}} | {'153a':>12} | {'154b':>12} | {'154c':>12}"
    print(header)
    print("-" * len(header))
    for label, fn in metric_rows:
        if fn is None:
            print(f"\n{label}")
            continue
        if label == "":
            continue
        vals = []
        for model_name in ["153a", "154b", "154c"]:
            vals.append(fn(results[model_name]))
        print(f"{label:<{col_w}} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12}")

    # ════════════════════════════════════════════════════════════
    # Save results
    # ════════════════════════════════════════════════════════════

    output = {
        "description": "Cross-model comparison: 153a / 154b / 154c on identical test data",
        "test_config": {
            "test_start": args.test_start,
            "n_windows": n_windows,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "base_n_runs": 3,
            "n_steps": n_steps,
        },
        "models": {
            "153a": {
                "path": str(base_model_path),
                "type": "conditional_one_shot_ODE",
                "d_model": base_cfg["d_model"],
                "n_layers": base_cfg["n_layers"],
                "cond_dim": base_cfg["cond_dim"],
                **results["153a"],
            },
            "154b": {
                "path": str(res_b_path),
                "type": "base_153a_3run_avg + unconditional_residual_FM",
                "d_model": res_b_cfg["d_model"],
                "n_layers": res_b_cfg["n_layers"],
                **results["154b"],
            },
            "154c": {
                "path": str(res_c_path),
                "type": "base_153a_3run_avg + conditional_residual_FM",
                "d_model": res_c_cfg["d_model"],
                "n_layers": res_c_cfg["n_layers"],
                "cond_dim": res_c_cfg["cond_dim"],
                **results["154c"],
            },
        },
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(result_path), "w") as f:
        json.dump(make_serial(output), f, indent=2)
    print(f"\nResults saved to {result_path}")

    table_path.parent.mkdir(parents=True, exist_ok=True)

    # Build comparison_table.json as a concise side-by-side
    comparison = {}
    for model_name in ["153a", "154b", "154c"]:
        r = results[model_name]
        comparison[model_name] = {
            "pass_count": r["pass_count"],
            "suites": r["suites"],
            **r["metrics"],
        }
    with open(str(table_path), "w") as f:
        json.dump(make_serial(comparison), f, indent=2)
    print(f"Comparison table saved to {table_path}")


if __name__ == "__main__":
    main()
