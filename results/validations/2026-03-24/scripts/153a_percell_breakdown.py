#!/usr/bin/env python
"""
153a Per-Cell Breakdown of Failing Suites
==========================================
Mechanistic analysis of WHY suites S1, S2, S6, S7, S8 fail for flow_153a.

Produces:
  - ci_coverage_grid.json (5x5 grids at h=1, h=15, h=30)
  - bias_vs_spread.json (per-cell bias and spread, correlation with CI)
  - ks_levels_percell.json (per-cell KS stats)
  - cal_arb_percell.json (per-cell arb rates)
  - coint_percell.json (per-cell-pair cointegration)
  - verification_result.json (summary)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer,
    load_encoder,
    normalize_iv,
)

# Paths
MODEL_PATH = "models/backfill/flow_153a/final_model.pt"
ENCODER_PATH = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_DIR = Path("results/validations/2026-03-24/analysis/153a_percell")
RESULT_PATH = Path("results/validations/2026-03-24/verification_results/153a_percell_breakdown.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WINDOWS = 160
N_SAMPLES = 50
TEST_START = 4540
H = 30
F_LEN = 30
N_CELLS = 25

# Moneyness labels (5 levels) x Tenor labels (5 levels)
MONEYNESS = ["0.90", "0.95", "1.00", "1.05", "1.10"]
TENORS = ["1M", "3M", "6M", "9M", "12M"]


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def load_model(device):
    """Load flow model and encoder."""
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    model = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    train_mean = ckpt["train_mean"]  # (1, 750)
    train_std = ckpt["train_std"]  # (1, 750)
    if isinstance(train_mean, np.ndarray):
        train_mean = torch.from_numpy(train_mean).to(device)
        train_std = torch.from_numpy(train_std).to(device)
    else:
        train_mean = train_mean.to(device)
        train_std = train_std.to(device)

    encoder, _ = load_encoder(ENCODER_PATH, device)
    n_steps = cfg.get("n_steps", 8)

    return model, encoder, train_mean, train_std, n_steps


@torch.no_grad()
def generate_samples(model, encoder, history_batch, train_mean, train_std, n_steps, n_samples, device):
    """
    Generate n_samples for a batch of history windows.
    history_batch: (B, 30, 5, 5) raw IV in [0,1]
    Returns: (B, n_samples, 30, 5, 5) in [0,1] IV space
    """
    B = history_batch.shape[0]
    DIM = 750
    dt = 1.0 / n_steps

    # Encode history
    hist_norm = normalize_iv(history_batch.to(device))  # [-1, 1]
    cond = encoder(hist_norm)  # (B, 128)

    all_samples = []
    # Generate n_samples per window, batching across samples
    for s in range(n_samples):
        x = torch.randn(B, DIM, device=device)
        cond_b = cond  # (B, 128)
        for step in range(n_steps):
            tt = torch.full((B,), step * dt, device=device)
            v = model(x, tt, cond=cond_b)
            x = x + v * dt
        # Denormalize
        x_denorm = x * train_std + train_mean
        x_denorm = torch.clamp(x_denorm, 0, 1)
        # Reshape to (B, 30, 5, 5)
        x_surf = x_denorm.reshape(B, F_LEN, 5, 5)
        all_samples.append(x_surf.cpu().numpy())

    # Stack: (n_samples, B, 30, 5, 5) -> (B, n_samples, 30, 5, 5)
    samples = np.stack(all_samples, axis=0).transpose(1, 0, 2, 3, 4)
    return samples


def analyze_calendar_arb(samples_all, gt_all):
    """
    Per-cell calendar arbitrage analysis.
    Calendar arb: total variance decreases with tenor at same moneyness.
    TV = IV^2 * tenor. For adjacent tenors, TV should be non-decreasing.

    samples_all: list of (n_samples, 30, 5, 5)
    gt_all: list of (30, 5, 5)
    """
    # Tenor values in years (approximate)
    tenor_years = np.array([1/12, 3/12, 6/12, 9/12, 12/12])

    # Per-cell: count violations across all windows, samples, horizons
    # Cell = (moneyness_idx, tenor_idx) for tenor_idx > 0 (comparing with previous)
    gen_violations = np.zeros((5, 5))
    gen_total = np.zeros((5, 5))
    gt_violations = np.zeros((5, 5))
    gt_total = np.zeros((5, 5))

    # Per-horizon arb rates for generated
    gen_arb_by_horizon = np.zeros(30)
    gen_total_by_horizon = np.zeros(30)

    for w in range(len(samples_all)):
        gen = samples_all[w]  # (n_samples, 30, 5, 5)
        gt = gt_all[w]  # (30, 5, 5)
        n_s = gen.shape[0]

        for h in range(30):
            for m in range(5):  # moneyness
                for t in range(1, 5):  # tenor (compare t with t-1)
                    # Generated
                    tv_prev = gen[:, h, m, t-1]**2 * tenor_years[t-1]
                    tv_curr = gen[:, h, m, t]**2 * tenor_years[t]
                    n_viol = np.sum(tv_curr < tv_prev - 1e-8)
                    gen_violations[m, t] += n_viol
                    gen_total[m, t] += n_s
                    gen_arb_by_horizon[h] += n_viol
                    gen_total_by_horizon[h] += n_s

                    # GT
                    tv_prev_gt = gt[h, m, t-1]**2 * tenor_years[t-1]
                    tv_curr_gt = gt[h, m, t]**2 * tenor_years[t]
                    if tv_curr_gt < tv_prev_gt - 1e-8:
                        gt_violations[m, t] += 1
                    gt_total[m, t] += 1

    gen_rates = np.where(gen_total > 0, gen_violations / gen_total, 0)
    gt_rates = np.where(gt_total > 0, gt_violations / gt_total, 0)
    gen_arb_horizon = np.where(gen_total_by_horizon > 0,
                                gen_arb_by_horizon / gen_total_by_horizon, 0)

    return {
        "gen_percell_arb_rate": gen_rates.tolist(),
        "gt_percell_arb_rate": gt_rates.tolist(),
        "gen_overall_arb_rate": float(gen_violations.sum() / max(gen_total.sum(), 1)),
        "gt_overall_arb_rate": float(gt_violations.sum() / max(gt_total.sum(), 1)),
        "gen_arb_by_horizon": gen_arb_horizon.tolist(),
        "moneyness_labels": MONEYNESS,
        "tenor_labels": TENORS,
        "note": "Cell (m,t) shows arb rate for tenor t vs t-1 at moneyness m. t=0 col is always 0."
    }


def analyze_ci_coverage(samples_all, gt_all):
    """
    Per-cell, per-horizon CI coverage analysis.
    Also decompose into BIAS vs SPREAD.
    """
    n_windows = len(samples_all)
    # Per-cell coverage at h=1, h=15, h=30
    ci_grids = {}
    for h_target, h_name in [(0, "h1"), (14, "h15"), (29, "h30")]:
        grid = np.zeros((5, 5))
        for m in range(5):
            for t in range(5):
                covered = 0
                for w in range(n_windows):
                    gen_vals = samples_all[w][:, h_target, m, t]  # (n_samples,)
                    gt_val = gt_all[w][h_target, m, t]
                    lo = np.percentile(gen_vals, 5)
                    hi = np.percentile(gen_vals, 95)
                    if lo <= gt_val <= hi:
                        covered += 1
                grid[m, t] = covered / n_windows
        ci_grids[h_name] = grid.tolist()

    # Per-horizon coverage (averaged over all cells)
    ci_by_horizon = np.zeros(30)
    for h in range(30):
        total_covered = 0
        total_checks = 0
        for m in range(5):
            for t in range(5):
                for w in range(n_windows):
                    gen_vals = samples_all[w][:, h, m, t]
                    gt_val = gt_all[w][h, m, t]
                    lo = np.percentile(gen_vals, 5)
                    hi = np.percentile(gen_vals, 95)
                    if lo <= gt_val <= hi:
                        total_covered += 1
                    total_checks += 1
        ci_by_horizon[h] = total_covered / total_checks

    # Worst 5 cells at h=30 (the hardest horizon)
    h30_grid = np.array(ci_grids["h30"])
    flat_idx = np.argsort(h30_grid.ravel())[:5]
    worst_cells = []
    for idx in flat_idx:
        m, t = divmod(idx, 5)
        worst_cells.append({
            "moneyness": MONEYNESS[m], "tenor": TENORS[t],
            "cell": f"({m},{t})", "coverage": float(h30_grid[m, t])
        })

    # Worst 5 horizons
    worst_horizons = np.argsort(ci_by_horizon)[:5]
    worst_h = [{"horizon": int(h+1), "coverage": float(ci_by_horizon[h])} for h in worst_horizons]

    return ci_grids, ci_by_horizon.tolist(), worst_cells, worst_h


def analyze_bias_vs_spread(samples_all, gt_all):
    """
    Per-cell: decompose CI failure into BIAS (mean shift) and SPREAD (too narrow).
    """
    n_windows = len(samples_all)
    bias_grid = np.zeros((5, 5))  # mean(gen) - mean(gt) at h=30
    spread_grid = np.zeros((5, 5))  # mean spread (hi-lo) at h=30
    gt_range_grid = np.zeros((5, 5))  # actual range of GT values
    ci_grid = np.zeros((5, 5))  # coverage at h=30

    for m in range(5):
        for t in range(5):
            gen_means = []
            gen_spreads = []
            gt_vals = []
            covered = 0
            for w in range(n_windows):
                gen_vals = samples_all[w][:, 29, m, t]  # h=30
                gt_val = gt_all[w][29, m, t]
                gen_means.append(np.mean(gen_vals))
                gen_spreads.append(np.percentile(gen_vals, 95) - np.percentile(gen_vals, 5))
                gt_vals.append(gt_val)
                lo = np.percentile(gen_vals, 5)
                hi = np.percentile(gen_vals, 95)
                if lo <= gt_val <= hi:
                    covered += 1
            bias_grid[m, t] = np.mean(gen_means) - np.mean(gt_vals)
            spread_grid[m, t] = np.mean(gen_spreads)
            gt_range_grid[m, t] = np.std(gt_vals) * 2 * 1.645  # approximate 90% CI of GT
            ci_grid[m, t] = covered / n_windows

    # Correlation of CI with bias and spread
    ci_flat = ci_grid.ravel()
    bias_flat = np.abs(bias_grid.ravel())
    spread_flat = spread_grid.ravel()
    gt_range_flat = gt_range_grid.ravel()

    # Pearson correlation
    bias_corr = float(np.corrcoef(ci_flat, bias_flat)[0, 1])
    spread_corr = float(np.corrcoef(ci_flat, spread_flat)[0, 1])
    spread_ratio = float(np.mean(spread_flat / np.maximum(gt_range_flat, 1e-6)))

    # Also compute per-horizon bias and spread
    bias_by_horizon = np.zeros(30)
    spread_by_horizon = np.zeros(30)
    for h in range(30):
        biases = []
        spreads = []
        for m in range(5):
            for t in range(5):
                gen_means_h = []
                gen_spreads_h = []
                gt_vals_h = []
                for w in range(n_windows):
                    gen_vals = samples_all[w][:, h, m, t]
                    gt_val = gt_all[w][h, m, t]
                    gen_means_h.append(np.mean(gen_vals))
                    gen_spreads_h.append(np.percentile(gen_vals, 95) - np.percentile(gen_vals, 5))
                    gt_vals_h.append(gt_val)
                biases.append(abs(np.mean(gen_means_h) - np.mean(gt_vals_h)))
                spreads.append(np.mean(gen_spreads_h))
        bias_by_horizon[h] = np.mean(biases)
        spread_by_horizon[h] = np.mean(spreads)

    return {
        "bias_grid_h30": bias_grid.tolist(),
        "abs_bias_grid_h30": np.abs(bias_grid).tolist(),
        "spread_grid_h30": spread_grid.tolist(),
        "gt_range_grid_h30": gt_range_grid.tolist(),
        "ci_grid_h30": ci_grid.tolist(),
        "spread_to_gt_range_ratio": spread_ratio,
        "bias_ci_correlation": bias_corr,
        "spread_ci_correlation": spread_corr,
        "diagnosis": "BIAS" if abs(bias_corr) > abs(spread_corr) else "SPREAD",
        "bias_by_horizon": bias_by_horizon.tolist(),
        "spread_by_horizon": spread_by_horizon.tolist(),
    }


def analyze_ks_levels(samples_all, gt_all):
    """
    Per-cell KS test on levels at h=30.
    Also compute mean shift per cell.
    """
    results = {}
    ks_grid = np.zeros((5, 5))
    pval_grid = np.zeros((5, 5))
    mean_shift_grid = np.zeros((5, 5))
    pass_grid = np.zeros((5, 5), dtype=int)

    for m in range(5):
        for t in range(5):
            gen_levels = []
            gt_levels = []
            for w in range(len(samples_all)):
                # Take first sample (or random sample) at h=30
                gen_levels.extend(samples_all[w][:, 29, m, t].tolist())
                gt_levels.append(gt_all[w][29, m, t])
            gen_arr = np.array(gen_levels)
            gt_arr = np.array(gt_levels)
            ks_stat, p_val = ks_2samp(gen_arr, gt_arr)
            ks_grid[m, t] = ks_stat
            pval_grid[m, t] = p_val
            mean_shift_grid[m, t] = np.mean(gen_arr) - np.mean(gt_arr)
            pass_grid[m, t] = 1 if p_val > 0.05 else 0

    # Also do KS on daily changes
    ks_changes_grid = np.zeros((5, 5))
    pval_changes_grid = np.zeros((5, 5))
    for m in range(5):
        for t in range(5):
            gen_changes = []
            gt_changes = []
            for w in range(len(samples_all)):
                gen = samples_all[w]  # (n_samples, 30, 5, 5)
                gt = gt_all[w]  # (30, 5, 5)
                # Daily changes for generated (use first sample for each)
                for s in range(min(5, gen.shape[0])):  # limit to 5 samples per window
                    changes = np.diff(gen[s, :, m, t])
                    gen_changes.extend(changes.tolist())
                gt_ch = np.diff(gt[:, m, t])
                gt_changes.extend(gt_ch.tolist())
            gen_arr = np.array(gen_changes)
            gt_arr = np.array(gt_changes)
            ks_stat, p_val = ks_2samp(gen_arr, gt_arr)
            ks_changes_grid[m, t] = ks_stat
            pval_changes_grid[m, t] = p_val

    return {
        "ks_levels_stat_h30": ks_grid.tolist(),
        "ks_levels_pval_h30": pval_grid.tolist(),
        "ks_levels_pass_h30": pass_grid.tolist(),
        "ks_levels_pass_count": int(pass_grid.sum()),
        "mean_shift_h30": mean_shift_grid.tolist(),
        "ks_changes_stat": ks_changes_grid.tolist(),
        "ks_changes_pval": pval_changes_grid.tolist(),
        "ks_changes_pass_count": int((pval_changes_grid > 0.05).sum()),
        "moneyness_labels": MONEYNESS,
        "tenor_labels": TENORS,
    }


def analyze_cointegration(samples_all, gt_all, max_pairs=100):
    """
    Per-cell-pair cointegration analysis.
    Test Engle-Granger on generated ensemble median paths.
    """
    # Build long time series from generated medians
    # Concatenate median paths across windows
    n_windows = len(samples_all)

    # Use median across samples for each window
    gen_medians = []  # will be (n_windows, 30, 5, 5)
    gt_series = []
    for w in range(n_windows):
        gen_medians.append(np.median(samples_all[w], axis=0))  # (30, 5, 5)
        gt_series.append(gt_all[w])  # (30, 5, 5)

    gen_medians = np.array(gen_medians)  # (n_windows, 30, 5, 5)
    gt_series = np.array(gt_series)

    # For cointegration: use concatenated paths for each cell
    # gen: (n_windows * 30, 25)
    gen_flat = gen_medians.reshape(n_windows * 30, 25)
    gt_flat = gt_series.reshape(n_windows * 30, 25)

    # Test all 300 pairs (25 choose 2)
    from itertools import combinations
    pairs = list(combinations(range(25), 2))

    coint_results = np.zeros((25, 25))
    n_tested = 0
    n_coint = 0

    # Distance matrix on 5x5 grid
    dist_matrix = np.zeros((25, 25))
    for i in range(25):
        mi, ti = divmod(i, 5)
        for j in range(25):
            mj, tj = divmod(j, 5)
            dist_matrix[i, j] = abs(mi - mj) + abs(ti - tj)  # Manhattan distance

    # Results by distance
    dist_coint = {}

    for i, j in pairs:
        # Engle-Granger: regress one on the other, test residuals for stationarity
        x = gen_flat[:, i]
        y = gen_flat[:, j]

        # Simple OLS
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue

        beta = np.cov(x, y)[0, 1] / np.var(x)
        alpha = np.mean(y) - beta * np.mean(x)
        resid = y - alpha - beta * x

        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(resid, maxlag=5)
            is_coint = adf_pval < 0.05
        except Exception:
            is_coint = False

        coint_results[i, j] = 1 if is_coint else 0
        coint_results[j, i] = coint_results[i, j]
        n_tested += 1
        if is_coint:
            n_coint += 1

        d = int(dist_matrix[i, j])
        if d not in dist_coint:
            dist_coint[d] = {"pass": 0, "total": 0}
        dist_coint[d]["total"] += 1
        if is_coint:
            dist_coint[d]["pass"] += 1

    # Compute per-cell cointegration rate (fraction of pairs containing that cell that are cointegrated)
    percell_coint = np.zeros(25)
    for c in range(25):
        mask = (coint_results[c, :] > 0)
        # Count pairs involving this cell
        total_pairs = 24  # each cell pairs with 24 others
        percell_coint[c] = mask.sum() / total_pairs if total_pairs > 0 else 0

    dist_coint_rates = {str(d): v["pass"] / max(v["total"], 1) for d, v in sorted(dist_coint.items())}

    return {
        "overall_coint_rate": float(n_coint / max(n_tested, 1)),
        "n_tested": n_tested,
        "n_cointegrated": n_coint,
        "percell_coint_rate": percell_coint.reshape(5, 5).tolist(),
        "coint_rate_by_distance": dist_coint_rates,
        "note": "Distance = Manhattan distance on 5x5 grid (moneyness + tenor steps)"
    }


def cross_suite_analysis(ci_grids, bias_spread, ks_results, cal_arb, coint):
    """
    Cross-suite correlation: are the same cells problematic across suites?
    """
    # Get per-cell metrics at h=30
    ci_h30 = np.array(ci_grids["h30"])
    bias_h30 = np.abs(np.array(bias_spread["abs_bias_grid_h30"]))
    spread_h30 = np.array(bias_spread["spread_grid_h30"])
    ks_stat_h30 = np.array(ks_results["ks_levels_stat_h30"])
    mean_shift = np.abs(np.array(ks_results["mean_shift_h30"]))
    cal_arb_gen = np.array(cal_arb["gen_percell_arb_rate"])
    coint_rate = np.array(coint["percell_coint_rate"])

    flat = {
        "ci_coverage": ci_h30.ravel(),
        "abs_bias": bias_h30.ravel(),
        "spread": spread_h30.ravel(),
        "ks_stat": ks_stat_h30.ravel(),
        "mean_shift": mean_shift.ravel(),
        "coint_rate": coint_rate.ravel(),
    }

    # Correlation matrix
    keys = list(flat.keys())
    n = len(keys)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(flat[keys[i]], flat[keys[j]])[0, 1]

    # Spatial pattern: corners vs center, short vs long tenor
    corners = [(0,0), (0,4), (4,0), (4,4)]
    center = [(2,2)]
    edges = [(i,j) for i in range(5) for j in range(5) if (i,j) not in corners and (i,j) not in center]

    corner_ci = np.mean([ci_h30[i,j] for i,j in corners])
    center_ci = float(ci_h30[2,2])
    edge_ci = np.mean([ci_h30[i,j] for i,j in edges])

    # Short tenor (col 0) vs long tenor (col 4)
    short_tenor_ci = np.mean(ci_h30[:, 0])
    long_tenor_ci = np.mean(ci_h30[:, 4])

    # ITM (row 0) vs OTM (row 4)
    itm_ci = np.mean(ci_h30[0, :])
    otm_ci = np.mean(ci_h30[4, :])

    return {
        "cross_metric_correlations": {
            "labels": keys,
            "matrix": corr.tolist()
        },
        "spatial_pattern": {
            "corner_ci": float(corner_ci),
            "center_ci": center_ci,
            "edge_ci": float(edge_ci),
            "short_tenor_ci": float(short_tenor_ci),
            "long_tenor_ci": float(long_tenor_ci),
            "itm_ci": float(itm_ci),
            "otm_ci": float(otm_ci),
        },
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("153a Per-Cell Breakdown Analysis")
    print("=" * 70)

    # Load model
    print("\n[1/5] Loading model and encoder...")
    model, encoder, train_mean, train_std, n_steps = load_model(DEVICE)
    print(f"  Model loaded, n_steps={n_steps}, device={DEVICE}")

    # Load data
    print("\n[2/5] Loading data and generating samples...")
    data = np.load(DATA_PATH)
    surfaces = data["surface"]  # (N, 5, 5)

    # Build test windows
    histories = []
    futures = []
    for i in range(TEST_START, min(TEST_START + N_WINDOWS, len(surfaces) - H - F_LEN + 1)):
        hist = surfaces[i:i+H]  # (30, 5, 5)
        fut = surfaces[i+H:i+H+F_LEN]  # (30, 5, 5)
        histories.append(hist)
        futures.append(fut)

    n_actual = len(histories)
    print(f"  Test windows: {n_actual} (from idx {TEST_START})")

    # Generate samples in batches
    BATCH_SIZE = 16
    all_samples = []  # list of (n_samples, 30, 5, 5) per window
    all_gt = futures  # list of (30, 5, 5)

    for b_start in range(0, n_actual, BATCH_SIZE):
        b_end = min(b_start + BATCH_SIZE, n_actual)
        hist_batch = torch.from_numpy(np.array(histories[b_start:b_end], dtype=np.float32))
        samples = generate_samples(model, encoder, hist_batch, train_mean, train_std,
                                   n_steps, N_SAMPLES, DEVICE)
        # samples: (B, n_samples, 30, 5, 5)
        for i in range(samples.shape[0]):
            all_samples.append(samples[i])
        progress = min(b_end, n_actual)
        print(f"  Generated {progress}/{n_actual} windows ({progress*N_SAMPLES} total samples)")

    # S1: Calendar Arbitrage
    print("\n[3/5] Analyzing calendar arbitrage (S1)...")
    cal_arb = analyze_calendar_arb(all_samples, all_gt)
    with open(OUTPUT_DIR / "cal_arb_percell.json", "w") as f:
        json.dump(make_serializable(cal_arb), f, indent=2)
    print(f"  Gen overall arb rate: {cal_arb['gen_overall_arb_rate']:.4f}")
    print(f"  GT overall arb rate: {cal_arb['gt_overall_arb_rate']:.4f}")

    # S2: CI Coverage + Bias vs Spread
    print("\n[4/5] Analyzing CI coverage, bias, and spread (S2)...")
    ci_grids, ci_horizon, worst_cells, worst_horizons = analyze_ci_coverage(all_samples, all_gt)
    bias_spread = analyze_bias_vs_spread(all_samples, all_gt)

    ci_result = {
        "ci_grids": ci_grids,
        "ci_by_horizon": ci_horizon,
        "worst_cells": worst_cells,
        "worst_horizons": worst_horizons,
        "overall_mean_ci": float(np.mean(ci_horizon)),
    }
    with open(OUTPUT_DIR / "ci_coverage_grid.json", "w") as f:
        json.dump(make_serializable(ci_result), f, indent=2)

    with open(OUTPUT_DIR / "bias_vs_spread.json", "w") as f:
        json.dump(make_serializable(bias_spread), f, indent=2)

    print(f"  Mean CI coverage: {ci_result['overall_mean_ci']:.4f}")
    print(f"  Worst cell CI (h=30): {worst_cells[0]['coverage']:.4f} at {worst_cells[0]['cell']}")
    print(f"  Diagnosis: {bias_spread['diagnosis']}")
    print(f"  Bias-CI correlation: {bias_spread['bias_ci_correlation']:.3f}")
    print(f"  Spread-CI correlation: {bias_spread['spread_ci_correlation']:.3f}")
    print(f"  Spread/GT_range ratio: {bias_spread['spread_to_gt_range_ratio']:.3f}")

    # S8: KS Levels
    print("\n[5a/5] Analyzing KS distributional (S8)...")
    ks_results = analyze_ks_levels(all_samples, all_gt)
    with open(OUTPUT_DIR / "ks_levels_percell.json", "w") as f:
        json.dump(make_serializable(ks_results), f, indent=2)
    print(f"  KS levels pass: {ks_results['ks_levels_pass_count']}/25")
    print(f"  KS changes pass: {ks_results['ks_changes_pass_count']}/25")

    # S6: Cointegration
    print("\n[5b/5] Analyzing cointegration (S6)...")
    coint = analyze_cointegration(all_samples, all_gt)
    with open(OUTPUT_DIR / "coint_percell.json", "w") as f:
        json.dump(make_serializable(coint), f, indent=2)
    print(f"  Cointegration rate: {coint['overall_coint_rate']:.4f}")
    print(f"  By distance: {coint['coint_rate_by_distance']}")

    # Cross-suite analysis
    print("\n[6/5] Cross-suite correlation analysis...")
    cross = cross_suite_analysis(ci_grids, bias_spread, ks_results, cal_arb, coint)

    elapsed = time.time() - t_start

    # Build final verification result
    verification = {
        "task": "153a_percell_breakdown",
        "model": MODEL_PATH,
        "status": "COMPLETED",
        "elapsed_seconds": round(elapsed, 1),
        "n_windows": n_actual,
        "n_samples": N_SAMPLES,
        "summary": {
            "S1_calendar_arb": {
                "gen_rate": cal_arb["gen_overall_arb_rate"],
                "gt_rate": cal_arb["gt_overall_arb_rate"],
                "worst_cell": {
                    "rate": float(np.max(cal_arb["gen_percell_arb_rate"])),
                    "location": None,  # filled below
                },
                "arb_worsens_with_horizon": bool(
                    np.corrcoef(range(30), cal_arb["gen_arb_by_horizon"])[0,1] > 0.3
                ),
            },
            "S2_ci_coverage": {
                "mean_ci": ci_result["overall_mean_ci"],
                "worst_cell_coverage": worst_cells[0]["coverage"],
                "worst_cell_location": worst_cells[0]["cell"],
                "worst_horizon": worst_horizons[0]["horizon"],
                "diagnosis": bias_spread["diagnosis"],
                "bias_ci_corr": bias_spread["bias_ci_correlation"],
                "spread_ci_corr": bias_spread["spread_ci_correlation"],
                "spread_to_gt_ratio": bias_spread["spread_to_gt_range_ratio"],
            },
            "S6_cointegration": {
                "rate": coint["overall_coint_rate"],
                "by_distance": coint["coint_rate_by_distance"],
            },
            "S8_distributional": {
                "ks_levels_pass": ks_results["ks_levels_pass_count"],
                "ks_changes_pass": ks_results["ks_changes_pass_count"],
                "worst_ks_stat": float(np.max(ks_results["ks_levels_stat_h30"])),
            },
            "cross_suite": {
                "spatial_pattern": cross["spatial_pattern"],
                "metric_correlations": cross["cross_metric_correlations"],
            },
            "investigation_answers": {
                "bias_or_spread": None,  # filled below
                "same_cells_across_suites": None,
                "spatial_pattern": None,
            },
        },
        "output_files": [
            str(OUTPUT_DIR / "ci_coverage_grid.json"),
            str(OUTPUT_DIR / "bias_vs_spread.json"),
            str(OUTPUT_DIR / "ks_levels_percell.json"),
            str(OUTPUT_DIR / "cal_arb_percell.json"),
            str(OUTPUT_DIR / "coint_percell.json"),
        ],
    }

    # Fill in worst cal arb cell location
    arb_arr = np.array(cal_arb["gen_percell_arb_rate"])
    worst_m, worst_t = np.unravel_index(np.argmax(arb_arr), arb_arr.shape)
    verification["summary"]["S1_calendar_arb"]["worst_cell"]["location"] = f"({worst_m},{worst_t})"

    # Answer investigation questions
    diag = bias_spread["diagnosis"]
    ratio = bias_spread["spread_to_gt_range_ratio"]
    if diag == "BIAS":
        verification["summary"]["investigation_answers"]["bias_or_spread"] = (
            f"BIAS-dominated. CI failure is primarily from systematic mean shift. "
            f"Bias-CI corr={bias_spread['bias_ci_correlation']:.3f}, "
            f"Spread-CI corr={bias_spread['spread_ci_correlation']:.3f}. "
            f"Spread/GT ratio={ratio:.3f}."
        )
    else:
        verification["summary"]["investigation_answers"]["bias_or_spread"] = (
            f"SPREAD-dominated. CI failure is primarily from ensemble being too narrow. "
            f"Spread-CI corr={bias_spread['spread_ci_correlation']:.3f}, "
            f"Bias-CI corr={bias_spread['bias_ci_correlation']:.3f}. "
            f"Spread/GT ratio={ratio:.3f} (need ~1.0 for proper coverage)."
        )

    # Same cells across suites
    corr_matrix = np.array(cross["cross_metric_correlations"]["matrix"])
    ci_ks_corr = corr_matrix[0, 3]  # ci_coverage vs ks_stat
    ci_bias_corr = corr_matrix[0, 1]
    verification["summary"]["investigation_answers"]["same_cells_across_suites"] = (
        f"CI-KS correlation: {ci_ks_corr:.3f}, CI-bias correlation: {ci_bias_corr:.3f}. "
        f"{'Strong' if abs(ci_ks_corr) > 0.5 else 'Weak'} overlap between CI and KS failures."
    )

    # Spatial pattern
    sp = cross["spatial_pattern"]
    verification["summary"]["investigation_answers"]["spatial_pattern"] = (
        f"Corner CI={sp['corner_ci']:.3f}, Center CI={sp['center_ci']:.3f}, "
        f"Edge CI={sp['edge_ci']:.3f}. "
        f"Short tenor CI={sp['short_tenor_ci']:.3f}, Long tenor CI={sp['long_tenor_ci']:.3f}. "
        f"ITM CI={sp['itm_ci']:.3f}, OTM CI={sp['otm_ci']:.3f}."
    )

    # Save verification result
    with open(RESULT_PATH, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results: {RESULT_PATH}")
    print(f"{'='*70}")

    return verification


if __name__ == "__main__":
    main()
