#!/usr/bin/env python
"""Long-horizon generation test: extend 90d AR frame model to 252 days (1 year).

The model generates frame-by-frame with GRU state updates. We extend the loop
from 30 to 252 frames with no retraining. Tests two position encoding strategies:
  A: pos = t (raw position, extrapolates beyond training range 0-29)
  B: pos = t % 30 (cyclic, stays in trained range)

Diagnostics at horizons h=30, 60, 90, 180, 252:
  1. Ensemble spread evolution (per-cell std vs horizon)
  2. Cointegration: IV vs EWMA realized vol (Engle-Granger)
  3. Spatial structure: term structure slope & smile convexity preserved
  4. Path stationarity: rolling std of daily changes (detect drift/explosion)
  5. ACF of daily changes at lag 1,5,10
  6. Visual: fan chart for ATM 3M cell across 252 days
  7. CI coverage at each checkpoint horizon (using GT if available)
  8. Kurtosis heatmap: time x cell daily change comparison

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/test_long_horizon.py \
        --model_path models/backfill/afcrps_90d/best_model.pt \
        --no_ema --n_samples 50 --n_windows 100 --batch_size 16 \
        --quantile_map models/backfill/afcrps_90d/quantile_map.npz --qmap_alpha 0.3 \
        --output_dir results/block_ar/long_horizon_test --device cuda
"""

import argparse
import dataclasses
import hashlib
import json
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


# ── Cell labels ──
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


def _hash_file(path: str | None) -> str | None:
    """Return a short SHA256 for provenance tracking."""
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _config_to_dict(config: SinglePassConfig) -> dict:
    return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)


def _hash_jsonable(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


@torch.no_grad()
def sample_long_horizon(model, history, n_samples, n_frames, pos_mode="native"):
    """Generate long-horizon samples through the model's native AR-frame sampler."""
    if not model.config.ar_frame:
        raise ValueError("Long-horizon generation is only implemented for ar_frame models")
    return model.sample(
        history,
        n_samples=n_samples,
        n_frames=n_frames,
        position_mode=pos_mode,
    )


def apply_quantile_map(samples_np, history_np, qmap_path, alpha):
    """Apply quantile mapping if path provided."""
    from experiments.backfill.block_ar.quantile_mapper import QuantileMapper
    qmapper = QuantileMapper(qmap_path, alpha=alpha)
    return qmapper.apply(samples_np, history_np)


def mean_pairwise_corr(flat_series: np.ndarray) -> float:
    """Mean upper-triangle correlation across cells."""
    corr = np.corrcoef(flat_series, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    return float(upper.mean()) if len(upper) else 0.0


def summarize_cross_cell_correlation(samples: np.ndarray, gt_future: np.ndarray) -> dict:
    """Summarize cross-cell correlation for levels and daily changes."""
    gen_median = np.median(samples, axis=1)  # (N, T, 5, 5)
    T = min(gen_median.shape[1], gt_future.shape[1])
    gen_levels = gen_median[:, :T].reshape(-1, 25)
    gt_levels = gt_future[:, :T].reshape(-1, 25)
    gen_daily = np.diff(gen_median[:, :T], axis=1).reshape(-1, 25)
    gt_daily = np.diff(gt_future[:, :T], axis=1).reshape(-1, 25)
    return {
        "generated": {
            "level_corr_mean": mean_pairwise_corr(gen_levels),
            "daily_change_corr_mean": mean_pairwise_corr(gen_daily),
        },
        "ground_truth": {
            "level_corr_mean": mean_pairwise_corr(gt_levels),
            "daily_change_corr_mean": mean_pairwise_corr(gt_daily),
        },
    }


def summarize_cointegration(
    samples: np.ndarray,
    gt_future: np.ndarray,
    returns: np.ndarray,
    test_start: int,
    history_len: int,
    ewma_lambda: float = 0.94,
    adf_lags: int = 3,
    adf_alpha: float = 0.10,
) -> dict:
    """Source-backed IV/EWMA structural summary for long-horizon runs."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.stattools import adfuller

    gen_median = np.median(samples, axis=1)
    T = min(gen_median.shape[1], gt_future.shape[1])
    gen_median = gen_median[:, :T]
    gt_future = gt_future[:, :T]
    N, _, H, W = gen_median.shape

    def compute_ewma_vol(ret_window):
        variance = np.zeros(len(ret_window))
        variance[0] = ret_window[0] ** 2
        for t in range(1, len(ret_window)):
            variance[t] = ewma_lambda * variance[t - 1] + (1 - ewma_lambda) * ret_window[t] ** 2
        return np.sqrt(variance * 252)

    def test_cointegration(iv_series, ewma_series):
        if len(iv_series) < 10 or np.std(iv_series) < 1e-8 or np.std(ewma_series) < 1e-8:
            return {"cointegrated": False, "adf_pvalue": 1.0, "rsquared": 0.0}
        try:
            X = add_constant(ewma_series)
            model = OLS(iv_series, X).fit()
            adf_result = adfuller(model.resid, maxlag=adf_lags, regression="c")
            return {
                "cointegrated": bool(adf_result[1] < adf_alpha),
                "adf_pvalue": float(adf_result[1]),
                "rsquared": float(model.rsquared),
            }
        except Exception:
            return {"cointegrated": False, "adf_pvalue": 1.0, "rsquared": 0.0}

    def summarize_iv_rv(iv_series, ewma_series):
        if np.std(iv_series) < 1e-8 or np.std(ewma_series) < 1e-8:
            return {"corr": 0.0, "rsquared": 0.0}
        corr = float(np.corrcoef(iv_series, ewma_series)[0, 1])
        try:
            rsquared = float(OLS(iv_series, add_constant(ewma_series)).fit().rsquared)
        except Exception:
            rsquared = 0.0
        return {"corr": corr, "rsquared": rsquared}

    gen_pass_counts = np.zeros((H, W))
    gt_pass_counts = np.zeros((H, W))
    gen_rsq_sums = np.zeros((H, W))
    gt_rsq_sums = np.zeros((H, W))
    gen_atm_corrs, gt_atm_corrs = [], []
    gen_atm_rsqs, gt_atm_rsqs = [], []
    n_valid = 0

    for win_idx in range(N):
        future_start_global = test_start + win_idx + history_len
        if future_start_global + T > len(returns):
            continue
        ret_window = returns[future_start_global:future_start_global + T]
        ewma_vol = compute_ewma_vol(ret_window)
        n_valid += 1

        gen_atm = summarize_iv_rv(gen_median[win_idx, :, 2, 2], ewma_vol)
        gt_atm = summarize_iv_rv(gt_future[win_idx, :, 2, 2], ewma_vol)
        gen_atm_corrs.append(gen_atm["corr"])
        gt_atm_corrs.append(gt_atm["corr"])
        gen_atm_rsqs.append(gen_atm["rsquared"])
        gt_atm_rsqs.append(gt_atm["rsquared"])

        for i in range(H):
            for j in range(W):
                gen_result = test_cointegration(gen_median[win_idx, :, i, j], ewma_vol)
                gt_result = test_cointegration(gt_future[win_idx, :, i, j], ewma_vol)
                if gen_result["cointegrated"]:
                    gen_pass_counts[i, j] += 1
                if gt_result["cointegrated"]:
                    gt_pass_counts[i, j] += 1
                gen_rsq_sums[i, j] += gen_result["rsquared"]
                gt_rsq_sums[i, j] += gt_result["rsquared"]

    if n_valid == 0:
        return {"n_valid_windows": 0}

    gen_pass_rates = gen_pass_counts / n_valid
    gt_pass_rates = gt_pass_counts / n_valid
    gen_mean_rsq = gen_rsq_sums / n_valid
    gt_mean_rsq = gt_rsq_sums / n_valid
    ratio_grid = np.where(
        gt_pass_rates > 0,
        gen_pass_rates / gt_pass_rates,
        np.where(gen_pass_rates > 0, np.inf, 1.0),
    )

    return {
        "n_valid_windows": n_valid,
        "horizon": T,
        "ewma_lambda": ewma_lambda,
        "generated": {
            "cointegration_pass_rate_mean": float(gen_pass_rates.mean()),
            "cointegration_rsq_mean": float(gen_mean_rsq.mean()),
            "per_cell_pass_rate": gen_pass_rates.tolist(),
            "atm6m_iv_vs_ewma_corr_mean": float(np.mean(gen_atm_corrs)),
            "atm6m_iv_vs_ewma_rsq_mean": float(np.mean(gen_atm_rsqs)),
        },
        "ground_truth": {
            "cointegration_pass_rate_mean": float(gt_pass_rates.mean()),
            "cointegration_rsq_mean": float(gt_mean_rsq.mean()),
            "per_cell_pass_rate": gt_pass_rates.tolist(),
            "atm6m_iv_vs_ewma_corr_mean": float(np.mean(gt_atm_corrs)),
            "atm6m_iv_vs_ewma_rsq_mean": float(np.mean(gt_atm_rsqs)),
        },
        "gen_vs_gt_pass_rate_ratio": float(
            gen_pass_rates.mean() / gt_pass_rates.mean()
        ) if gt_pass_rates.mean() > 0 else 0.0,
        "worst_cell_gen_vs_gt_ratio": float(np.min(ratio_grid)),
    }


# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def diag_ensemble_spread(samples, horizons, output_dir, label):
    """1. Per-cell std vs horizon."""
    # samples: (N, S, T, 5, 5)
    N, S, T, H, W = samples.shape
    std_per_h = samples.std(axis=1)  # (N, T, 5, 5)
    mean_std = std_per_h.mean(axis=0)  # (T, 5, 5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: spread over time for selected cells
    cells = [(2, 2, "ATM 6M"), (0, 2, "ATM 1M"), (4, 2, "ATM 2Y"),
             (2, 0, "6M K=0.70"), (2, 4, "6M K=1.30")]
    ax = axes[0]
    for r, c, name in cells:
        ax.plot(range(T), mean_std[:, r, c], label=name)
    for h in horizons:
        if h <= T:
            ax.axvline(h, color="gray", alpha=0.3, ls="--")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Mean ensemble std (IV pts)")
    ax.set_title(f"Ensemble spread vs horizon — {label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: spread at each checkpoint horizon (all 25 cells)
    ax = axes[1]
    horizon_stds = []
    for h in horizons:
        if h <= T:
            horizon_stds.append(mean_std[h - 1].flatten())
    x_pos = np.arange(25)
    width = 0.8 / len(horizons)
    for i, (h, vals) in enumerate(zip(horizons, horizon_stds)):
        ax.bar(x_pos + i * width, vals, width, label=f"h={h}", alpha=0.7)
    ax.set_xlabel("Cell")
    ax.set_ylabel("Mean std")
    ax.set_title(f"Per-cell spread at horizons — {label}")
    ax.set_xticks(x_pos + width * len(horizons) / 2)
    ax.set_xticklabels([f"{i}" for i in range(25)], fontsize=6)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_ensemble_spread_{label}.png", dpi=150)
    plt.close()

    # Print summary
    print(f"\n  Ensemble spread summary ({label}):")
    print(f"  {'Horizon':>8s}  {'Mean std':>10s}  {'Min cell':>10s}  {'Max cell':>10s}")
    for h in horizons:
        if h <= T:
            s = mean_std[h - 1]
            print(f"  {h:8d}  {s.mean():10.4f}  {s.min():10.4f}  {s.max():10.4f}")


def diag_spatial_structure(samples, horizons, output_dir, label):
    """3. Term structure slope & smile convexity at each horizon."""
    # samples: (N, S, T, 5, 5)
    mean_surf = samples.mean(axis=1)  # (N, T, 5, 5)

    print(f"\n  Spatial structure ({label}):")
    print(f"  {'Horizon':>8s}  {'TermSlope':>10s}  {'SmileConv':>10s}  {'Explosion':>10s}")

    results = {}
    for h in horizons:
        if h > samples.shape[2]:
            continue
        surf_h = mean_surf[:, h - 1]  # (N, 5, 5), rows=tenor, cols=moneyness

        # Term structure: ATM column (col=2), slope = tenor4 - tenor0
        atm = surf_h[:, :, 2]  # (N, 5) across tenors
        term_slope = (atm[:, 4] - atm[:, 0]).mean()

        # Smile convexity: for each tenor, convexity = (K0 + K4)/2 - K2
        convexity = ((surf_h[:, :, 0] + surf_h[:, :, 4]) / 2 - surf_h[:, :, 2]).mean()

        # Explosion check: any surface > 0.99 or < 0.01
        explosion = ((surf_h > 0.99) | (surf_h < 0.01)).any(axis=(1, 2)).mean()

        print(f"  {h:8d}  {term_slope:10.4f}  {convexity:10.4f}  {explosion:10.3%}")
        results[h] = {"term_slope": float(term_slope), "smile_conv": float(convexity),
                       "explosion_rate": float(explosion)}

    return results


def diag_path_stationarity(samples, output_dir, label):
    """4. Rolling std of daily changes — detect drift/explosion."""
    # samples: (N, S, T, 5, 5)
    daily_changes = np.diff(samples, axis=2)  # (N, S, T-1, 5, 5)
    # Average across samples and cells
    dc_mean = daily_changes.mean(axis=(1, 3, 4))  # (N, T-1)

    # Rolling std with window=20
    T = dc_mean.shape[1]
    window = 20
    rolling_std = np.zeros(T - window + 1)
    for i in range(T - window + 1):
        rolling_std[i] = dc_mean[:, i:i + window].std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(window, T + 1), rolling_std)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Rolling std of mean daily change")
    ax.set_title(f"Path stationarity — {label}")
    ax.axhline(rolling_std[:10].mean(), color="red", ls="--", alpha=0.5,
               label=f"Early mean: {rolling_std[:10].mean():.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Also check per-cell: std of daily changes in windows
    # (N, S, T-1, 5, 5) → collapse N,S → per-cell rolling std
    dc_flat = daily_changes.reshape(-1, daily_changes.shape[2], 5, 5)  # (N*S, T-1, 5, 5)
    cell_std = dc_flat.std(axis=0)  # (T-1, 5, 5)

    ax = axes[1]
    cells = [(2, 2, "ATM 6M"), (0, 0, "1M K=0.70"), (4, 4, "2Y K=1.30")]
    for r, c, name in cells:
        ax.plot(range(cell_std.shape[0]), cell_std[:, r, c], label=name)
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Daily change std across ensemble")
    ax.set_title(f"Per-cell daily change volatility — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_path_stationarity_{label}.png", dpi=150)
    plt.close()

    # Detect explosion: last 50 frames vs first 50 frames
    if T > 100:
        early_std = rolling_std[:30].mean()
        late_std = rolling_std[-30:].mean()
        ratio = late_std / (early_std + 1e-10)
        print(f"\n  Stationarity ({label}): early_std={early_std:.4f}, late_std={late_std:.4f}, ratio={ratio:.2f}")
        return ratio
    return 1.0


def diag_acf(samples, output_dir, label):
    """5. ACF of daily changes at lag 1, 5, 10."""
    daily_changes = np.diff(samples, axis=2)  # (N, S, T-1, 5, 5)
    # Focus on ATM 6M cell (2, 2)
    dc = daily_changes[:, :, :, 2, 2].reshape(-1, daily_changes.shape[2])  # (N*S, T-1)

    T = dc.shape[1]
    lags = [1, 5, 10, 20]

    print(f"\n  ACF of daily changes, ATM 6M ({label}):")
    print(f"  {'Lag':>6s}  {'ACF':>8s}  {'|ACF|':>8s}")

    acf_results = {}
    for lag in lags:
        if lag >= T:
            continue
        x = dc[:, lag:]
        y = dc[:, :-lag]
        corr = np.corrcoef(x.flatten(), y.flatten())[0, 1]
        # Also ACF of absolute changes (volatility clustering)
        abs_corr = np.corrcoef(np.abs(x).flatten(), np.abs(y).flatten())[0, 1]
        print(f"  {lag:6d}  {corr:8.4f}  {abs_corr:8.4f}")
        acf_results[lag] = {"acf": float(corr), "abs_acf": float(abs_corr)}

    return acf_results


def diag_fan_chart(samples, history, output_dir, label, window_idx=0):
    """6. Fan chart for ATM 3M cell (row=1, col=2) across all frames."""
    # samples: (N, S, T, 5, 5), history: (N, H, 5, 5)
    r, c = 1, 2  # ATM 3M
    T = samples.shape[2]
    H = history.shape[1]

    hist_vals = history[window_idx, :, r, c]  # (H,)
    gen_vals = samples[window_idx, :, :, r, c]  # (S, T)

    median = np.median(gen_vals, axis=0)
    p10 = np.percentile(gen_vals, 10, axis=0)
    p90 = np.percentile(gen_vals, 90, axis=0)
    p25 = np.percentile(gen_vals, 25, axis=0)
    p75 = np.percentile(gen_vals, 75, axis=0)

    fig, ax = plt.subplots(figsize=(16, 6))

    # History
    t_hist = np.arange(-H, 0)
    ax.plot(t_hist, hist_vals, color="black", lw=1.5, label="History")

    # Fan chart
    t_gen = np.arange(T)
    ax.fill_between(t_gen, p10, p90, alpha=0.2, color="blue", label="10-90%")
    ax.fill_between(t_gen, p25, p75, alpha=0.3, color="blue", label="25-75%")
    ax.plot(t_gen, median, color="blue", lw=1, label="Median")

    # Mark horizons
    for h in HORIZONS:
        if h <= T:
            ax.axvline(h, color="gray", alpha=0.3, ls="--")
            ax.text(h, ax.get_ylim()[1], f"h={h}", fontsize=7, ha="center", va="bottom")

    ax.axvline(0, color="gray", alpha=0.5, ls="-")
    ax.set_xlabel("Days (0 = generation start)")
    ax.set_ylabel("IV (ATM 3M)")
    ax.set_title(f"Fan chart: ATM 3M — {label} (window {window_idx})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/6_fan_chart_{label}.png", dpi=150)
    plt.close()


def diag_kurtosis_heatmap(samples, output_dir, label, window_idx=0):
    """8. Time x cell daily change heatmap comparing GT-like sample paths."""
    # samples: (N, S, T, 5, 5)
    T = samples.shape[2]

    # Pick one sample path from the specified window
    path = samples[window_idx, 0]  # (T, 5, 5)
    dc = np.diff(path, axis=0)  # (T-1, 5, 5)
    dc_flat = dc.reshape(dc.shape[0], 25).T  # (25, T-1) — cells on y, days on x

    # Pick another sample path for comparison
    path2 = samples[window_idx, 1]  # (T, 5, 5)
    dc2 = np.diff(path2, axis=0)
    dc_flat2 = dc2.reshape(dc2.shape[0], 25).T

    vmax = max(np.abs(dc_flat).max(), np.abs(dc_flat2).max()) * 0.8

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    for ax, data, title in [
        (axes[0], dc_flat, "Sample path 1"),
        (axes[1], dc_flat2, "Sample path 2"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        interpolation="nearest")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cell")
        ax.set_title(f"{title} — daily changes — {label}")
        # Mark maturity group boundaries
        for boundary in [5, 10, 15, 20]:
            ax.axhline(boundary - 0.5, color="black", lw=0.5, alpha=0.5)
        ax.set_yticks(range(25))
        ax.set_yticklabels(CELL_NAMES, fontsize=6)
        # Mark horizons
        for h in HORIZONS:
            if h - 1 < data.shape[1]:
                ax.axvline(h - 1, color="black", alpha=0.3, ls="--")
        plt.colorbar(im, ax=ax, shrink=0.6, label="Daily IV change")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/8_kurtosis_heatmap_{label}.png", dpi=150)
    plt.close()


def diag_kurtosis_by_horizon(samples, horizons, output_dir, label):
    """Kurtosis of daily changes at different horizon windows."""
    daily_changes = np.diff(samples, axis=2)  # (N, S, T-1, 5, 5)

    print(f"\n  Kurtosis of daily changes by horizon ({label}):")
    print(f"  {'Window':>12s}  {'Mean Kurt':>10s}  {'Min':>8s}  {'Max':>8s}  {'ATM6M':>8s}")

    for i in range(len(horizons)):
        start = 0 if i == 0 else horizons[i - 1]
        end = min(horizons[i], daily_changes.shape[2])
        if start >= end:
            continue
        window_dc = daily_changes[:, :, start:end, :, :]  # (N, S, window, 5, 5)
        flat = window_dc.reshape(-1, 5, 5)
        kurt_vals = np.zeros((5, 5))
        for r in range(5):
            for c in range(5):
                kurt_vals[r, c] = sp_stats.kurtosis(flat[:, r, c], fisher=True)
        window_name = f"d{start}-d{end}"
        print(f"  {window_name:>12s}  {kurt_vals.mean():10.2f}  "
              f"{kurt_vals.min():8.2f}  {kurt_vals.max():8.2f}  {kurt_vals[2, 2]:8.2f}")


def diag_ci_coverage_vs_horizon(samples, gt_future, horizons, output_dir, label):
    """7. CI coverage at each horizon using available GT."""
    # samples: (N, S, T_gen, 5, 5), gt_future: (N, T_gt, 5, 5)
    T_gt = gt_future.shape[1]
    print(f"\n  CI coverage (90%) vs horizon ({label}), GT available for {T_gt} days:")
    print(f"  {'Horizon':>8s}  {'Coverage':>10s}  {'Width':>10s}")

    results = {}
    for h in horizons:
        if h > T_gt or h > samples.shape[2]:
            continue
        gen_h = samples[:, :, h - 1, :, :]  # (N, S, 5, 5)
        gt_h = gt_future[:, h - 1, :, :]  # (N, 5, 5)
        p5 = np.percentile(gen_h, 5, axis=1)
        p95 = np.percentile(gen_h, 95, axis=1)
        covered = ((gt_h >= p5) & (gt_h <= p95)).mean()
        width = (p95 - p5).mean()
        print(f"  {h:8d}  {covered:10.3%}  {width:10.4f}")
        results[h] = {"coverage": float(covered), "width": float(width)}

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Long-horizon generation test (252 days)")
    parser.add_argument("--model_path", type=str,
                        default="models/backfill/afcrps_90d/best_model.pt")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_windows", type=int, default=100,
                        help="Number of test windows")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_frames", type=int, default=252,
                        help="Total frames to generate")
    parser.add_argument("--quantile_map", type=str, default=None)
    parser.add_argument("--qmap_alpha", type=float, default=0.3)
    parser.add_argument("--pos_mode", type=str, default="native",
                        choices=["native", "raw", "cyclic", "compare"],
                        help="Position semantics to use during long-horizon generation")
    parser.add_argument("--output_dir", type=str,
                        default="results/block_ar/long_horizon_test")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LONG-HORIZON GENERATION TEST")
    print("=" * 70)
    print(f"  Model: {args.model_path}")
    print(f"  Frames: {args.n_frames} ({args.n_frames / 252:.1f} years)")
    print(f"  Samples: {args.n_samples}, Windows: {args.n_windows}")
    print(f"  Position mode: {args.pos_mode}")
    if args.quantile_map:
        print(f"  Quantile map: {args.quantile_map} (alpha={args.qmap_alpha})")
    print()

    # Load model
    model, ckpt = load_model(args.model_path, device, no_ema=args.no_ema)
    print(f"  Model loaded: ar_frame={model.config.ar_frame}, "
          f"future_len={model.config.future_len}")

    # Load data — use test split
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    returns = data["ret"]
    history_len = 30
    # We need history_len + n_frames for GT comparison
    gt_future_len = min(args.n_frames, len(surfaces) - 4540 - history_len)
    dataset = VolSurfaceDataset(surfaces, history_len, gt_future_len, start_idx=4540)
    n_windows = min(args.n_windows, len(dataset))
    print(f"  Test windows: {n_windows} (GT future available: {gt_future_len} days)")

    # Collect test data
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_history = []
    all_future = []
    n_collected = 0
    for batch in loader:
        if n_collected >= n_windows:
            break
        take = min(batch["history"].shape[0], n_windows - n_collected)
        all_history.append(batch["history"][:take])
        all_future.append(batch["future"][:take])
        n_collected += take

    history_t = torch.cat(all_history, dim=0).to(device)  # (N, 30, 5, 5) in [-1, 1]
    future_t = torch.cat(all_future, dim=0)  # (N, gt_future_len, 5, 5) in [-1, 1]
    N = history_t.shape[0]

    # Denormalize history and GT future for comparison
    history_np = denormalize_iv(history_t).cpu().numpy()  # (N, 30, 5, 5) in [0, 1]
    gt_future_np = denormalize_iv(future_t).numpy()  # (N, gt_future_len, 5, 5) in [0, 1]

    model_config = _config_to_dict(model.config)
    model_config_hash = _hash_jsonable(model_config)
    eval_args = vars(args).copy()
    modes = ["raw", "cyclic"] if args.pos_mode == "compare" else [args.pos_mode]

    # ── Generate for requested position strategies ──
    for pos_mode in modes:
        print(f"\n{'=' * 70}")
        if pos_mode == "native":
            pos_desc = "model-native position semantics"
        elif pos_mode == "raw":
            pos_desc = "pos=t"
        else:
            pos_desc = "pos=t%period"
        print(f"POSITION MODE: {pos_mode.upper()} ({pos_desc})")
        print(f"{'=' * 70}")

        t0 = time.time()

        # Generate in batches to manage memory
        all_samples = []
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch_hist = history_t[start:end]
            batch_samples = sample_long_horizon(
                model, batch_hist, args.n_samples, args.n_frames, pos_mode=pos_mode
            )
            all_samples.append(batch_samples.cpu().numpy())
            print(f"  Generated batch {start}-{end} "
                  f"[{batch_samples.shape}, range: {batch_samples.min():.4f}-{batch_samples.max():.4f}]")

        samples = np.concatenate(all_samples, axis=0)  # (N, S, T, 5, 5)
        elapsed = time.time() - t0
        print(f"\n  Generation: {elapsed:.1f}s, shape={samples.shape}")

        # Apply quantile mapping
        if args.quantile_map:
            print(f"  Applying quantile mapping (alpha={args.qmap_alpha})...")
            samples = apply_quantile_map(samples, history_np, args.quantile_map, args.qmap_alpha)
            print(f"  Post-qmap range: [{samples.min():.4f}, {samples.max():.4f}]")

        label = pos_mode

        # ── Run diagnostics ──
        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 1: Ensemble Spread")
        print(f"{'─' * 50}")
        diag_ensemble_spread(samples, HORIZONS, args.output_dir, label)

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 3: Spatial Structure")
        print(f"{'─' * 50}")
        spatial = diag_spatial_structure(samples, HORIZONS, args.output_dir, label)
        gt_spatial = diag_spatial_structure(
            gt_future_np[:, None, :min(args.n_frames, gt_future_np.shape[1])],
            HORIZONS,
            args.output_dir,
            f"gt_{label}",
        )

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 4: Path Stationarity")
        print(f"{'─' * 50}")
        ratio = diag_path_stationarity(samples, args.output_dir, label)

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 5: ACF")
        print(f"{'─' * 50}")
        acf = diag_acf(samples, args.output_dir, label)

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 6: Fan Chart")
        print(f"{'─' * 50}")
        # Pick a turbulent window for visual interest
        hist_mean_iv = history_np.mean(axis=(2, 3))  # (N, 30)
        daily_chg = np.diff(hist_mean_iv, axis=1)
        vov = daily_chg.std(axis=1)
        turb_idx = np.argsort(vov)[-1]
        diag_fan_chart(samples, history_np, args.output_dir, label, window_idx=turb_idx)
        print(f"  Fan chart saved (turbulent window {turb_idx})")

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 7: CI Coverage vs Horizon")
        print(f"{'─' * 50}")
        ci = diag_ci_coverage_vs_horizon(
            samples, gt_future_np, HORIZONS, args.output_dir, label
        )

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 9: Cross-Cell Correlation")
        print(f"{'─' * 50}")
        cross_cell = summarize_cross_cell_correlation(samples, gt_future_np)
        print(f"  Level corr mean: gen={cross_cell['generated']['level_corr_mean']:.3f}, "
              f"gt={cross_cell['ground_truth']['level_corr_mean']:.3f}")
        print(f"  Daily-change corr mean: gen={cross_cell['generated']['daily_change_corr_mean']:.3f}, "
              f"gt={cross_cell['ground_truth']['daily_change_corr_mean']:.3f}")

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 10: Cointegration / IV-RV Structure")
        print(f"{'─' * 50}")
        cointegration = summarize_cointegration(
            samples,
            gt_future_np,
            returns=returns,
            test_start=4540,
            history_len=history_len,
        )
        if cointegration.get("n_valid_windows", 0) > 0:
            print(f"  ATM 6M IV~EWMA corr: gen={cointegration['generated']['atm6m_iv_vs_ewma_corr_mean']:.3f}, "
                  f"gt={cointegration['ground_truth']['atm6m_iv_vs_ewma_corr_mean']:.3f}")
            print(f"  Cointegration pass rate: gen={cointegration['generated']['cointegration_pass_rate_mean']:.1%}, "
                  f"gt={cointegration['ground_truth']['cointegration_pass_rate_mean']:.1%}")

        print(f"\n{'─' * 50}")
        print("DIAGNOSTIC 8: Kurtosis")
        print(f"{'─' * 50}")
        diag_kurtosis_by_horizon(samples, HORIZONS, args.output_dir, label)
        diag_kurtosis_heatmap(samples, args.output_dir, label, window_idx=turb_idx)
        print(f"  Kurtosis heatmap saved")

        # Save results
        results = {
            "provenance": {
                "model_path": str(Path(args.model_path).resolve()),
                "model_hash": _hash_file(args.model_path),
                "checkpoint_epoch": ckpt.get("epoch"),
                "quantile_map_path": (
                    str(Path(args.quantile_map).resolve()) if args.quantile_map else None
                ),
                "quantile_map_hash": _hash_file(args.quantile_map),
                "model_config_hash": model_config_hash,
                "model_config": model_config,
                "eval_args": eval_args,
            },
            "pos_mode": pos_mode,
            "n_frames": args.n_frames,
            "n_samples": args.n_samples,
            "n_windows": N,
            "generation_time_s": elapsed,
            "spatial_structure": spatial,
            "ground_truth_spatial_structure": gt_spatial,
            "stationarity_ratio": float(ratio),
            "acf": acf,
            "ci_coverage": ci,
            "cross_cell_correlation": cross_cell,
            "cointegration": cointegration,
        }
        with open(f"{args.output_dir}/results_{label}.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"{args.output_dir}/cointegration_summary_{label}.json", "w") as f:
            json.dump(cointegration, f, indent=2)

    # ── Comparison summary ──
    if len(modes) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 70}")

        for label in modes:
            with open(f"{args.output_dir}/results_{label}.json") as f:
                res = json.load(f)
            print(f"\n  {label.upper()}:")
            print(f"    Stationarity ratio (late/early std): {res['stationarity_ratio']:.2f} "
                  f"({'OK' if 0.5 < res['stationarity_ratio'] < 2.0 else 'WARN'})")
            if res.get("ci_coverage"):
                for h, ci in res["ci_coverage"].items():
                    print(f"    CI coverage h={h}: {ci['coverage']:.1%} (width={ci['width']:.4f})")
            if res.get("spatial_structure"):
                for h, sp in res["spatial_structure"].items():
                    print(f"    h={h}: term_slope={sp['term_slope']:.4f}, "
                          f"smile={sp['smile_conv']:.4f}, explosion={sp['explosion_rate']:.1%}")

    print(f"\n  Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
