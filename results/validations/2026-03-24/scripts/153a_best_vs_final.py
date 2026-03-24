#!/usr/bin/env python
"""
153a Best (ep28) vs Final (ep200) Comparison

Compares two checkpoints of the conditional one-shot flow matching model:
- best_model.pt (epoch 28, best val loss 0.387)
- final_model.pt (epoch 200, val loss 0.735)

Two evaluation modes:
1. Population metrics (512 samples with random training conditions)
2. Test suite metrics (160 test windows, 50 samples each)

Key question: Does the early model (ep28) have WIDER per-window spread
than the late model (ep200), potentially passing CI despite worse population metrics?
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, kurtosis

sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")

from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer,
    load_encoder,
    normalize_iv,
)
from experiments.backfill.block_ar.train_oneshot_flow import evaluate_samples


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER_PATH = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
BEST_PATH = "models/backfill/flow_153a/best_model.pt"
FINAL_PATH = "models/backfill/flow_153a/final_model.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_DIR = Path("results/validations/2026-03-24/analysis/153a_best_vs_final")
RESULT_PATH = Path("results/validations/2026-03-24/verification_results/153a_best_vs_final.json")

H = 30  # history length
F_LEN = 30  # forecast length
N_CELLS = 25
DIM = F_LEN * N_CELLS  # 750
TEST_START = 4540
N_WINDOWS = 160
N_SAMPLES = 50
N_STEPS = 8


def load_model(model_path, device):
    """Load a conditional flow matching model."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"],
        n_cells=cfg["n_cells"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    train_mean = ckpt["train_mean"]  # (1, 750) numpy
    train_std = ckpt["train_std"]    # (1, 750) numpy
    return model, train_mean, train_std, ckpt


def ode_sample(model, cond, n_samples, train_mean, train_std, device, n_steps=N_STEPS):
    """Generate samples via Euler ODE integration.

    Args:
        model: velocity network
        cond: (B, 128) conditioning vectors (one per window)
        n_samples: number of samples per window
        train_mean, train_std: denormalization stats
        device: torch device
        n_steps: number of Euler steps

    Returns:
        samples: (B, n_samples, 30, 5, 5) denormalized IV surfaces
    """
    B = cond.shape[0]
    dt = 1.0 / n_steps

    all_samples = []
    for s in range(n_samples):
        x = torch.randn(B, DIM, device=device)
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            v = model(x, t, cond=cond)
            x = x + v * dt
        all_samples.append(x.cpu().numpy())

    # (n_samples, B, 750) -> (B, n_samples, 750)
    samples = np.stack(all_samples, axis=1)

    # Denormalize
    samples = samples * train_std + train_mean
    samples = np.clip(samples, 0, 1)

    # Reshape to (B, n_samples, 30, 5, 5)
    samples = samples.reshape(B, n_samples, F_LEN, 5, 5)
    return samples


def compute_ci_coverage(samples, gt_future, horizons=[0, 4, 9, 14, 19, 24, 29]):
    """Compute per-horizon and per-cell CI coverage.

    Args:
        samples: (B, n_samples, 30, 5, 5)
        gt_future: (B, 30, 5, 5) ground truth

    Returns:
        dict with coverage metrics
    """
    B, K, T, H5, W5 = samples.shape

    # Per-horizon coverage
    horizon_coverages = {}
    for h in horizons:
        # samples at horizon h: (B, K, 5, 5)
        s_h = samples[:, :, h, :, :]
        gt_h = gt_future[:, h, :, :]  # (B, 5, 5)

        # 90% CI per cell
        lo = np.percentile(s_h, 5, axis=1)   # (B, 5, 5)
        hi = np.percentile(s_h, 95, axis=1)  # (B, 5, 5)

        covered = (gt_h >= lo) & (gt_h <= hi)  # (B, 5, 5)
        coverage = covered.mean()
        horizon_coverages[f"h{h+1}"] = float(coverage)

    # Per-cell coverage (worst cell across all horizons and windows)
    cell_coverages = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            s_cell = samples[:, :, :, r, c]  # (B, K, T)
            gt_cell = gt_future[:, :, r, c]  # (B, T)

            lo = np.percentile(s_cell, 5, axis=1)   # (B, T)
            hi = np.percentile(s_cell, 95, axis=1)   # (B, T)
            covered = (gt_cell >= lo) & (gt_cell <= hi)
            cell_coverages[r, c] = covered.mean()

    worst_cell = cell_coverages.min()
    mean_coverage = cell_coverages.mean()

    # Horizon pass count (need >85% at each horizon)
    horizon_pass = sum(1 for v in horizon_coverages.values() if v >= 0.85)

    return {
        "horizon_coverages": horizon_coverages,
        "horizon_pass_count": horizon_pass,
        "cell_coverages": cell_coverages.tolist(),
        "worst_cell_coverage": float(worst_cell),
        "mean_coverage": float(mean_coverage),
        "worst_cell_pass": worst_cell >= 0.80,
    }


def compute_conditionality(samples_turb, samples_calm, gt_turb, gt_calm):
    """Compute conditionality metrics: turb/calm width ratio.

    Args:
        samples_turb: (B_turb, K, 30, 5, 5) samples from turbulent windows
        samples_calm: (B_calm, K, 30, 5, 5) samples from calm windows

    Returns:
        dict with conditionality metrics
    """
    # Width = mean spread across horizons
    spread_turb = samples_turb.std(axis=1).mean()  # mean over all dims
    spread_calm = samples_calm.std(axis=1).mean()

    ratio = float(spread_turb / (spread_calm + 1e-10))

    # Per-cell MAE reduction vs unconditional
    # Conditional median should be closer to GT than population mean
    med_turb = np.median(samples_turb, axis=1)  # (B, 30, 5, 5)
    med_calm = np.median(samples_calm, axis=1)

    mae_turb = np.abs(med_turb - gt_turb).mean()
    mae_calm = np.abs(med_calm - gt_calm).mean()

    return {
        "turb_calm_ratio": ratio,
        "spread_turb": float(spread_turb),
        "spread_calm": float(spread_calm),
        "mae_turb": float(mae_turb),
        "mae_calm": float(mae_calm),
        "conditionality_pass": ratio > 1.15,
    }


def compute_ks_metrics(samples, gt_future):
    """Compute KS tests on daily changes and levels.

    Args:
        samples: (B, K, 30, 5, 5)
        gt_future: (B, 30, 5, 5)
    """
    B, K, T, _, _ = samples.shape

    # Flatten samples: pick one random sample per window for daily changes
    gen_flat = samples[:, 0, :, :, :].reshape(B, T, N_CELLS)  # (B, T, 25)
    gt_flat = gt_future.reshape(B, T, N_CELLS)

    # Daily changes
    gen_ch = np.diff(gen_flat, axis=1).reshape(-1, N_CELLS)
    gt_ch = np.diff(gt_flat, axis=1).reshape(-1, N_CELLS)

    ks_daily_pass = 0
    ks_daily_stats = []
    for c in range(N_CELLS):
        stat, pval = ks_2samp(gen_ch[:, c], gt_ch[:, c])
        ks_daily_stats.append(float(stat))
        if stat < 0.15:
            ks_daily_pass += 1

    # KS on levels
    gen_levels = samples[:, 0, :, :, :].reshape(-1)
    gt_levels = gt_future.reshape(-1)
    ks_level_stat, ks_level_pval = ks_2samp(gen_levels, gt_levels)

    # Kurtosis
    gen_ch_flat = gen_ch.flatten()
    gt_ch_flat = gt_ch.flatten()
    kurt_gen = kurtosis(gen_ch_flat, fisher=True)
    kurt_gt = kurtosis(gt_ch_flat, fisher=True)

    return {
        "ks_daily_pass": ks_daily_pass,
        "ks_daily_stats_mean": float(np.mean(ks_daily_stats)),
        "ks_level_stat": float(ks_level_stat),
        "kurt_ratio": float(kurt_gen / (kurt_gt + 1e-6)),
    }


def compute_cross_cell_correlation(samples, gt_future):
    """Compute cross-cell correlation metrics.

    Args:
        samples: (B, K, 30, 5, 5)
        gt_future: (B, 30, 5, 5)
    """
    B, K, T, _, _ = samples.shape

    # Use all samples for better statistics
    gen_flat = samples.reshape(B * K, T, N_CELLS)
    gt_flat = gt_future.reshape(B, T, N_CELLS)

    gen_ch = np.diff(gen_flat, axis=1).reshape(-1, N_CELLS)
    gt_ch = np.diff(gt_flat, axis=1).reshape(-1, N_CELLS)

    gen_corr = np.corrcoef(gen_ch.T)
    gt_corr = np.corrcoef(gt_ch.T)

    def eff_rank(corr):
        ev = np.linalg.eigvalsh(corr)[::-1]
        ev = np.maximum(ev, 0)
        p = ev / (ev.sum() + 1e-10)
        p = p[p > 1e-10]
        return float(np.exp(-np.sum(p * np.log(p))))

    gen_rank = eff_rank(gen_corr)
    gt_rank = eff_rank(gt_corr)
    rank_ratio = gen_rank / (gt_rank + 1e-10)

    # Correlation ratio (off-diagonal)
    mask = ~np.eye(N_CELLS, dtype=bool)
    corr_ratio = float(gen_corr[mask].mean() / (gt_corr[mask].mean() + 1e-10))

    frob = float(np.linalg.norm(gen_corr - gt_corr, 'fro'))

    return {
        "eff_rank_gen": gen_rank,
        "eff_rank_gt": gt_rank,
        "eff_rank_ratio": rank_ratio,
        "corr_ratio": corr_ratio,
        "frob": frob,
        "rank_ratio_pass": rank_ratio >= 0.50,
    }


def compute_calendar_arb(samples):
    """Check for calendar arbitrage in generated surfaces.

    Calendar arb: total variance should be monotonically increasing with tenor.
    samples: (B, K, 30, 5, 5) where last dim is tenor (5 tenors).

    IV is in [0, 1] range. Total variance = IV^2 * T.
    We approximate tenors as [0.08, 0.17, 0.25, 0.5, 1.0] (roughly 1M, 2M, 3M, 6M, 1Y).
    """
    tenors = np.array([0.08, 0.17, 0.25, 0.5, 1.0])

    B, K, T, M, Ten = samples.shape
    # For each frame, check tenor monotonicity of total variance
    n_violations = 0
    n_total = 0

    for b in range(min(B, 50)):  # subsample for speed
        for k in range(min(K, 10)):
            for t in range(T):
                for m in range(M):  # each moneyness level
                    ivs = samples[b, k, t, m, :]  # (5,) across tenors
                    total_var = ivs**2 * tenors
                    # Check monotonicity
                    for i in range(len(total_var) - 1):
                        n_total += 1
                        if total_var[i+1] < total_var[i] - 1e-8:
                            n_violations += 1

    rate = n_violations / max(n_total, 1)
    return {
        "calendar_arb_rate": float(rate),
        "n_violations": n_violations,
        "n_total": n_total,
    }


def compute_cointegration(samples, gt_future):
    """Simple cointegration proxy: correlation between sample mean and GT across windows.

    Args:
        samples: (B, K, 30, 5, 5)
        gt_future: (B, 30, 5, 5)
    """
    B, K, T, _, _ = samples.shape

    # Per-cell, compute correlation between sample median trajectory and GT
    coint_passes = 0
    total_cells = 0

    for r in range(5):
        for c in range(5):
            # Get trajectories across windows
            sample_med = np.median(samples[:, :, :, r, c], axis=1)  # (B, T)
            gt_traj = gt_future[:, :, r, c]  # (B, T)

            # Flatten across windows and time
            s_flat = sample_med.flatten()
            g_flat = gt_traj.flatten()

            corr = np.corrcoef(s_flat, g_flat)[0, 1]
            total_cells += 1
            if corr > 0.5:
                coint_passes += 1

    return {
        "coint_pass_rate": float(coint_passes / total_cells),
        "coint_passes": coint_passes,
        "total_cells": total_cells,
    }


def compute_spread_by_horizon(samples):
    """Compute spread (std across ensemble) at each horizon.

    Args:
        samples: (B, K, 30, 5, 5)
    Returns:
        dict with per-horizon spreads
    """
    B, K, T, _, _ = samples.shape

    # Spread per horizon: std across K, then mean over cells and windows
    spreads = []
    for h in range(T):
        s_h = samples[:, :, h, :, :]  # (B, K, 5, 5)
        spread = s_h.std(axis=1).mean()  # mean over B, cells
        spreads.append(float(spread))

    return {
        "spread_h1": spreads[0],
        "spread_h10": spreads[9],
        "spread_h20": spreads[19],
        "spread_h30": spreads[29],
        "spread_growth": spreads[29] / (spreads[0] + 1e-10),
        "spreads": spreads,
    }


def make_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def evaluate_model(model_name, model, encoder, train_mean, train_std,
                   surfaces, device, n_steps=N_STEPS):
    """Full evaluation of a model.

    Returns dict with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    results = {"model_name": model_name}

    # =========================================================
    # PART 1: Population metrics (512 samples, random conditions)
    # =========================================================
    print("\n[1/2] Population metrics (512 samples, random training conditions)...")
    t0 = time.time()

    # Build training data for GT comparison
    train_end = 4040
    train_futures = []
    train_histories = []
    for i in range(train_end - H - F_LEN + 1):
        train_futures.append(surfaces[i+H:i+H+F_LEN].reshape(-1))
        train_histories.append(surfaces[i:i+H])
    train_data = np.array(train_futures, dtype=np.float32)
    train_hist = np.array(train_histories, dtype=np.float32)

    # Pre-compute encoder conditions for training data
    train_conds = []
    with torch.no_grad():
        for i in range(0, len(train_hist), 256):
            batch = torch.from_numpy(train_hist[i:i+256]).to(device)
            batch_norm = normalize_iv(batch)
            cond = encoder(batch_norm)
            train_conds.append(cond.cpu().numpy())
    train_conds = np.concatenate(train_conds, axis=0)

    # Generate 512 population samples
    n_pop = 512
    pop_batch = 64
    all_pop_samples = []
    dt = 1.0 / n_steps

    with torch.no_grad():
        for si in range(0, n_pop, pop_batch):
            eb = min(pop_batch, n_pop - si)
            x = torch.randn(eb, DIM, device=device)
            idx = np.random.choice(len(train_conds), eb, replace=True)
            c = torch.from_numpy(train_conds[idx]).to(device)
            for step in range(n_steps):
                t = torch.full((eb,), step * dt, device=device)
                x = x + model(x, t, cond=c) * dt
            all_pop_samples.append(x.cpu().numpy())

    pop_samples = np.concatenate(all_pop_samples)
    pop_samples = pop_samples * train_std + train_mean
    pop_samples = np.clip(pop_samples, 0, 1)

    pop_metrics = evaluate_samples(pop_samples, train_data)
    results["population"] = pop_metrics

    elapsed = time.time() - t0
    print(f"  Population eval: {elapsed:.1f}s")
    print(f"    eff_rank: {pop_metrics['eff_rank']:.2f} (GT: {pop_metrics['gt_eff_rank']:.2f})")
    print(f"    PC1: {pop_metrics['pc1']:.3f}, PC2: {pop_metrics['pc2']:.3f}")
    print(f"    KS pass: {pop_metrics['ks_pass']}/25")
    print(f"    kurt_ratio: {pop_metrics['kurt_ratio']:.3f}")
    print(f"    frob: {pop_metrics['frob']:.2f}")
    print(f"    spread h1: {pop_metrics['spread_h1']:.4f}, h30: {pop_metrics['spread_h30']:.4f}")

    # =========================================================
    # PART 2: Test suite metrics (160 windows, 50 samples each)
    # =========================================================
    print(f"\n[2/2] Test suite metrics ({N_WINDOWS} windows, {N_SAMPLES} samples each)...")
    t0 = time.time()

    # Build test windows
    test_histories = []
    test_futures = []
    total_available = len(surfaces)

    for i in range(N_WINDOWS):
        start = TEST_START + i
        if start + H + F_LEN > total_available:
            break
        hist = surfaces[start:start+H]           # (30, 5, 5)
        fut = surfaces[start+H:start+H+F_LEN]   # (30, 5, 5)
        test_histories.append(hist)
        test_futures.append(fut)

    n_actual_windows = len(test_histories)
    print(f"  Actual test windows: {n_actual_windows}")

    test_hist = np.array(test_histories, dtype=np.float32)    # (B, 30, 5, 5)
    test_fut = np.array(test_futures, dtype=np.float32)       # (B, 30, 5, 5)

    # Compute encoder conditions for test windows
    test_conds = []
    with torch.no_grad():
        for i in range(0, n_actual_windows, 64):
            batch = torch.from_numpy(test_hist[i:i+64]).to(device)
            batch_norm = normalize_iv(batch)
            cond = encoder(batch_norm)
            test_conds.append(cond.cpu())
    test_conds = torch.cat(test_conds, dim=0)  # (B, 128)

    # Generate samples for all test windows
    all_test_samples = []
    window_batch = 16  # process 16 windows at a time

    with torch.no_grad():
        for wb_start in range(0, n_actual_windows, window_batch):
            wb_end = min(wb_start + window_batch, n_actual_windows)
            wb_size = wb_end - wb_start
            cond_batch = test_conds[wb_start:wb_end].to(device)  # (wb, 128)

            # Generate N_SAMPLES for each window
            window_samples = []
            for s in range(N_SAMPLES):
                x = torch.randn(wb_size, DIM, device=device)
                for step in range(n_steps):
                    t = torch.full((wb_size,), step * dt, device=device)
                    x = x + model(x, t, cond=cond_batch) * dt
                # Denormalize
                x_np = x.cpu().numpy()
                x_np = x_np * train_std + train_mean
                x_np = np.clip(x_np, 0, 1)
                window_samples.append(x_np)

            # (N_SAMPLES, wb, 750) -> (wb, N_SAMPLES, 750)
            batch_samples = np.stack(window_samples, axis=1)
            all_test_samples.append(batch_samples)

    # Concatenate: (B, N_SAMPLES, 750)
    test_samples = np.concatenate(all_test_samples, axis=0)
    # Reshape to (B, N_SAMPLES, 30, 5, 5)
    test_samples = test_samples.reshape(n_actual_windows, N_SAMPLES, F_LEN, 5, 5)

    elapsed_gen = time.time() - t0
    print(f"  Sample generation: {elapsed_gen:.1f}s")

    # --- CI Coverage ---
    print("  Computing CI coverage...")
    ci_results = compute_ci_coverage(test_samples, test_fut)
    results["ci_coverage"] = ci_results
    print(f"    Mean coverage: {ci_results['mean_coverage']:.3f}")
    print(f"    Worst cell: {ci_results['worst_cell_coverage']:.3f}")
    print(f"    Horizon pass: {ci_results['horizon_pass_count']}/7")
    print(f"    Worst cell pass (>=0.80): {ci_results['worst_cell_pass']}")

    # --- Conditionality ---
    print("  Computing conditionality...")
    # Classify windows as turbulent/calm based on history volatility
    hist_changes = np.diff(test_hist.reshape(n_actual_windows, H, N_CELLS), axis=1)
    hist_vol = np.abs(hist_changes).mean(axis=(1, 2))  # (B,)
    median_vol = np.median(hist_vol)
    turb_mask = hist_vol > median_vol
    calm_mask = ~turb_mask

    cond_results = compute_conditionality(
        test_samples[turb_mask], test_samples[calm_mask],
        test_fut[turb_mask], test_fut[calm_mask]
    )
    results["conditionality"] = cond_results
    print(f"    Turb/calm ratio: {cond_results['turb_calm_ratio']:.3f}")
    print(f"    Pass (>1.15): {cond_results['conditionality_pass']}")

    # --- KS metrics ---
    print("  Computing KS metrics...")
    ks_results = compute_ks_metrics(test_samples, test_fut)
    results["ks_metrics"] = ks_results
    print(f"    KS daily pass: {ks_results['ks_daily_pass']}/25")
    print(f"    KS level stat: {ks_results['ks_level_stat']:.4f}")
    print(f"    Kurt ratio: {ks_results['kurt_ratio']:.3f}")

    # --- Cross-cell correlation ---
    print("  Computing cross-cell correlation...")
    xcorr_results = compute_cross_cell_correlation(test_samples, test_fut)
    results["cross_cell_corr"] = xcorr_results
    print(f"    Eff rank: {xcorr_results['eff_rank_gen']:.2f} (GT: {xcorr_results['eff_rank_gt']:.2f})")
    print(f"    Rank ratio: {xcorr_results['eff_rank_ratio']:.3f}")
    print(f"    Corr ratio: {xcorr_results['corr_ratio']:.3f}")
    print(f"    Frob: {xcorr_results['frob']:.2f}")

    # --- Calendar arbitrage ---
    print("  Computing calendar arbitrage...")
    cal_results = compute_calendar_arb(test_samples)
    results["calendar_arb"] = cal_results
    print(f"    Calendar arb rate: {cal_results['calendar_arb_rate']:.4f}")

    # --- Cointegration ---
    print("  Computing cointegration...")
    coint_results = compute_cointegration(test_samples, test_fut)
    results["cointegration"] = coint_results
    print(f"    Coint pass rate: {coint_results['coint_pass_rate']:.2f}")

    # --- Spread by horizon ---
    print("  Computing spread by horizon...")
    spread_results = compute_spread_by_horizon(test_samples)
    results["spread_by_horizon"] = spread_results
    print(f"    Spread h1: {spread_results['spread_h1']:.5f}")
    print(f"    Spread h30: {spread_results['spread_h30']:.5f}")
    print(f"    Spread growth (h30/h1): {spread_results['spread_growth']:.3f}")

    total_elapsed = time.time() - t0
    print(f"\n  Total test suite eval: {total_elapsed:.1f}s")

    return results


def main():
    print("=" * 70)
    print("153a Best (ep28) vs Final (ep200) Comparison")
    print("=" * 70)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("\nLoading data...")
    data = np.load(DATA_PATH)
    surfaces = data["surface"]
    print(f"  Surfaces: {surfaces.shape}")

    # Load encoder
    print("Loading encoder...")
    encoder, cond_dim = load_encoder(ENCODER_PATH, DEVICE)
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  Encoder cond_dim: {cond_dim}")

    # Load both models
    print("\nLoading best model (ep28)...")
    model_best, mean_best, std_best, ckpt_best = load_model(BEST_PATH, DEVICE)
    print(f"  Epoch: {ckpt_best['epoch']}, Val loss: {ckpt_best['val_loss']:.4f}")

    print("Loading final model (ep200)...")
    model_final, mean_final, std_final, ckpt_final = load_model(FINAL_PATH, DEVICE)
    print(f"  Epoch: {ckpt_final['epoch']}, Val loss: {ckpt_final['val_loss']:.4f}")

    # Evaluate both models
    # Reset seed before each eval for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)
    results_best = evaluate_model(
        "best_model (ep28)", model_best, encoder, mean_best, std_best,
        surfaces, DEVICE
    )

    torch.manual_seed(42)
    np.random.seed(42)
    results_final = evaluate_model(
        "final_model (ep200)", model_final, encoder, mean_final, std_final,
        surfaces, DEVICE
    )

    # =========================================================
    # Side-by-side comparison
    # =========================================================
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    comparison = {
        "best_epoch": ckpt_best["epoch"],
        "final_epoch": ckpt_final["epoch"],
        "best_val_loss": ckpt_best["val_loss"],
        "final_val_loss": ckpt_final["val_loss"],
    }

    rows = [
        ("Val Loss", ckpt_best["val_loss"], ckpt_final["val_loss"]),
        ("", "", ""),
        ("--- POPULATION ---", "", ""),
        ("Eff Rank", results_best["population"]["eff_rank"], results_final["population"]["eff_rank"]),
        ("GT Eff Rank", results_best["population"]["gt_eff_rank"], results_final["population"]["gt_eff_rank"]),
        ("PC1 Alignment", results_best["population"]["pc1"], results_final["population"]["pc1"]),
        ("PC2 Alignment", results_best["population"]["pc2"], results_final["population"]["pc2"]),
        ("Frob (corr)", results_best["population"]["frob"], results_final["population"]["frob"]),
        ("KS Pass", results_best["population"]["ks_pass"], results_final["population"]["ks_pass"]),
        ("Kurt Ratio", results_best["population"]["kurt_ratio"], results_final["population"]["kurt_ratio"]),
        ("Spread h1", results_best["population"]["spread_h1"], results_final["population"]["spread_h1"]),
        ("Spread h30", results_best["population"]["spread_h30"], results_final["population"]["spread_h30"]),
        ("", "", ""),
        ("--- TEST SUITES ---", "", ""),
        ("CI: Mean Coverage", results_best["ci_coverage"]["mean_coverage"], results_final["ci_coverage"]["mean_coverage"]),
        ("CI: Worst Cell", results_best["ci_coverage"]["worst_cell_coverage"], results_final["ci_coverage"]["worst_cell_coverage"]),
        ("CI: Worst Cell Pass", results_best["ci_coverage"]["worst_cell_pass"], results_final["ci_coverage"]["worst_cell_pass"]),
        ("CI: Horizon Pass", results_best["ci_coverage"]["horizon_pass_count"], results_final["ci_coverage"]["horizon_pass_count"]),
        ("Cond: Turb/Calm", results_best["conditionality"]["turb_calm_ratio"], results_final["conditionality"]["turb_calm_ratio"]),
        ("Cond: Pass", results_best["conditionality"]["conditionality_pass"], results_final["conditionality"]["conditionality_pass"]),
        ("KS Daily Pass", results_best["ks_metrics"]["ks_daily_pass"], results_final["ks_metrics"]["ks_daily_pass"]),
        ("KS Level Stat", results_best["ks_metrics"]["ks_level_stat"], results_final["ks_metrics"]["ks_level_stat"]),
        ("Kurt Ratio (test)", results_best["ks_metrics"]["kurt_ratio"], results_final["ks_metrics"]["kurt_ratio"]),
        ("XCorr: Eff Rank", results_best["cross_cell_corr"]["eff_rank_gen"], results_final["cross_cell_corr"]["eff_rank_gen"]),
        ("XCorr: Rank Ratio", results_best["cross_cell_corr"]["eff_rank_ratio"], results_final["cross_cell_corr"]["eff_rank_ratio"]),
        ("XCorr: Corr Ratio", results_best["cross_cell_corr"]["corr_ratio"], results_final["cross_cell_corr"]["corr_ratio"]),
        ("XCorr: Frob", results_best["cross_cell_corr"]["frob"], results_final["cross_cell_corr"]["frob"]),
        ("Cal Arb Rate", results_best["calendar_arb"]["calendar_arb_rate"], results_final["calendar_arb"]["calendar_arb_rate"]),
        ("Coint Pass Rate", results_best["cointegration"]["coint_pass_rate"], results_final["cointegration"]["coint_pass_rate"]),
        ("Spread h1 (test)", results_best["spread_by_horizon"]["spread_h1"], results_final["spread_by_horizon"]["spread_h1"]),
        ("Spread h30 (test)", results_best["spread_by_horizon"]["spread_h30"], results_final["spread_by_horizon"]["spread_h30"]),
        ("Spread Growth", results_best["spread_by_horizon"]["spread_growth"], results_final["spread_by_horizon"]["spread_growth"]),
    ]

    print(f"\n{'Metric':<25} {'Best (ep28)':>15} {'Final (ep200)':>15} {'Winner':>10}")
    print("-" * 70)
    for metric, v_best, v_final in rows:
        if metric == "":
            print()
            continue
        if metric.startswith("---"):
            print(f"\n{metric}")
            continue

        if isinstance(v_best, bool):
            s_best = "PASS" if v_best else "FAIL"
            s_final = "PASS" if v_final else "FAIL"
            winner = "best" if v_best and not v_final else ("final" if v_final and not v_best else "tie")
        elif isinstance(v_best, int):
            s_best = str(v_best)
            s_final = str(v_final)
            winner = "best" if v_best > v_final else ("final" if v_final > v_best else "tie")
        else:
            s_best = f"{v_best:.4f}"
            s_final = f"{v_final:.4f}"
            winner = ""

        print(f"{metric:<25} {s_best:>15} {s_final:>15} {winner:>10}")

    # Key question answer
    print("\n" + "=" * 70)
    print("KEY QUESTION: Does ep28 have wider per-window spread than ep200?")
    print("=" * 70)
    spread_best = results_best["spread_by_horizon"]
    spread_final = results_final["spread_by_horizon"]
    print(f"  ep28  spread: h1={spread_best['spread_h1']:.5f}, h30={spread_best['spread_h30']:.5f}, growth={spread_best['spread_growth']:.3f}")
    print(f"  ep200 spread: h1={spread_final['spread_h1']:.5f}, h30={spread_final['spread_h30']:.5f}, growth={spread_final['spread_growth']:.3f}")

    ci_best = results_best["ci_coverage"]
    ci_final = results_final["ci_coverage"]
    print(f"\n  ep28  CI coverage: mean={ci_best['mean_coverage']:.3f}, worst_cell={ci_best['worst_cell_coverage']:.3f}")
    print(f"  ep200 CI coverage: mean={ci_final['mean_coverage']:.3f}, worst_cell={ci_final['worst_cell_coverage']:.3f}")

    answer = "YES" if spread_best['spread_h1'] > spread_final['spread_h1'] else "NO"
    print(f"\n  Answer: {answer} — ep28 {'has wider' if answer == 'YES' else 'has narrower'} spread than ep200")

    # =========================================================
    # Save results
    # =========================================================
    full_results = {
        "comparison_summary": comparison,
        "best_model": make_serializable(results_best),
        "final_model": make_serializable(results_final),
        "key_question": {
            "question": "Does ep28 have wider per-window spread than ep200?",
            "answer": answer,
            "best_spread_h1": spread_best["spread_h1"],
            "final_spread_h1": spread_final["spread_h1"],
            "best_ci_worst_cell": ci_best["worst_cell_coverage"],
            "final_ci_worst_cell": ci_final["worst_cell_coverage"],
        },
        "side_by_side": [
            {"metric": m, "best": make_serializable(b), "final": make_serializable(f)}
            for m, b, f in rows if m and not m.startswith("---")
        ],
    }

    # Save analysis
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "full_comparison.json", "w") as fh:
        json.dump(full_results, fh, indent=2, default=lambda x: make_serializable(x))
    print(f"\nSaved full comparison to {OUTPUT_DIR / 'full_comparison.json'}")

    # Save verification result
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    verification = {
        "task": "153a best_model (ep28) vs final_model (ep200) comparison",
        "status": "COMPLETE",
        "models": {
            "best": {"path": BEST_PATH, "epoch": ckpt_best["epoch"], "val_loss": float(ckpt_best["val_loss"])},
            "final": {"path": FINAL_PATH, "epoch": ckpt_final["epoch"], "val_loss": float(ckpt_final["val_loss"])},
        },
        "key_findings": {
            "wider_spread_early": answer == "YES",
            "best_ci_worst_cell": float(ci_best["worst_cell_coverage"]),
            "final_ci_worst_cell": float(ci_final["worst_cell_coverage"]),
            "best_ci_pass": bool(ci_best["worst_cell_pass"]),
            "final_ci_pass": bool(ci_final["worst_cell_pass"]),
            "best_conditionality_pass": bool(results_best["conditionality"]["conditionality_pass"]),
            "final_conditionality_pass": bool(results_final["conditionality"]["conditionality_pass"]),
            "best_rank_ratio": float(results_best["cross_cell_corr"]["eff_rank_ratio"]),
            "final_rank_ratio": float(results_final["cross_cell_corr"]["eff_rank_ratio"]),
        },
        "population_metrics": {
            "best": make_serializable(results_best["population"]),
            "final": make_serializable(results_final["population"]),
        },
        "test_suite_metrics": {
            "best": {
                "ci_mean": float(ci_best["mean_coverage"]),
                "ci_worst_cell": float(ci_best["worst_cell_coverage"]),
                "turb_calm_ratio": float(results_best["conditionality"]["turb_calm_ratio"]),
                "ks_daily_pass": results_best["ks_metrics"]["ks_daily_pass"],
                "eff_rank_ratio": float(results_best["cross_cell_corr"]["eff_rank_ratio"]),
                "corr_ratio": float(results_best["cross_cell_corr"]["corr_ratio"]),
                "calendar_arb_rate": float(results_best["calendar_arb"]["calendar_arb_rate"]),
                "coint_pass_rate": float(results_best["cointegration"]["coint_pass_rate"]),
                "spread_h1": float(results_best["spread_by_horizon"]["spread_h1"]),
                "spread_h30": float(results_best["spread_by_horizon"]["spread_h30"]),
            },
            "final": {
                "ci_mean": float(ci_final["mean_coverage"]),
                "ci_worst_cell": float(ci_final["worst_cell_coverage"]),
                "turb_calm_ratio": float(results_final["conditionality"]["turb_calm_ratio"]),
                "ks_daily_pass": results_final["ks_metrics"]["ks_daily_pass"],
                "eff_rank_ratio": float(results_final["cross_cell_corr"]["eff_rank_ratio"]),
                "corr_ratio": float(results_final["cross_cell_corr"]["corr_ratio"]),
                "calendar_arb_rate": float(results_final["calendar_arb"]["calendar_arb_rate"]),
                "coint_pass_rate": float(results_final["cointegration"]["coint_pass_rate"]),
                "spread_h1": float(results_final["spread_by_horizon"]["spread_h1"]),
                "spread_h30": float(results_final["spread_by_horizon"]["spread_h30"]),
            },
        },
    }

    with open(RESULT_PATH, "w") as fh:
        json.dump(verification, fh, indent=2)
    print(f"Saved verification result to {RESULT_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
