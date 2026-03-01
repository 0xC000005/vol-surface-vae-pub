#!/usr/bin/env python
"""
Management Report: Block-AR Volatility Surface Scenario Generator Quality.

Generates publication-quality visualizations demonstrating:
1. Conditional variance fan charts (calm vs turbulent, per cell)
2. Cross-cell regime sensitivity comparison
3. Temporal properties (kurtosis, ACF)
4. Surface structure validity (term structure, smile)
5. Surface heatmap snapshots (GT vs generated vs uncertainty)
6. Calibration curve

All plots use the best vol-scaled model (epoch 26, 437K params).
"""

import dataclasses
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy import stats as sp_stats
from matplotlib.patches import FancyBboxPatch

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    normalize_iv,
)

# ── Constants ──
MONEYNESS = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
MATURITIES_DAYS = np.array([30, 91, 182, 365, 730])
MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
MONEYNESS_LABELS = ["0.70", "0.85", "1.00", "1.15", "1.30"]
CELL_NAMES = {}
for r in range(5):
    for c in range(5):
        CELL_NAMES[(r, c)] = f"{MATURITY_LABELS[r]} / K={MONEYNESS_LABELS[c]}"

OUTPUT_DIR = "results/block_ar/management_report_v3_conformal"
MODEL_PATH = "models/backfill/block_ar_vol_scaled_30ep/best_model.pt"
N_SAMPLES = 50
MAX_WINDOWS = 400

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

CALM_COLOR = "#2196F3"
TURB_COLOR = "#F44336"
GT_COLOR = "#2E7D32"
SCENARIO_COLOR = "#9E9E9E"
BAND_ALPHA = 0.15


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


def generate_all_data(model, config, device):
    """Generate samples for all test windows. Returns dict of arrays."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_start = 4540
    test_surfaces = surfaces[test_start:]

    history_len = config.history_len
    future_len = config.future_len
    total_len = history_len + future_len
    N = len(test_surfaces)
    n_windows = min(MAX_WINDOWS, N - total_len + 1)
    indices = np.linspace(0, N - total_len, n_windows, dtype=int)

    all_history, all_future = [], []
    for idx in indices:
        all_history.append(test_surfaces[idx:idx + history_len])
        all_future.append(test_surfaces[idx + history_len:idx + total_len])

    history_arr = np.stack(all_history)
    future_arr = np.stack(all_future)

    # vol_of_vol
    mean_iv = history_arr.mean(axis=(-1, -2))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)

    # Generate samples
    print(f"Generating {N_SAMPLES} samples for {n_windows} windows...")
    batch_size = 16
    all_samples = []
    for i in range(0, n_windows, batch_size):
        batch_end = min(i + batch_size, n_windows)
        hist = torch.tensor(history_arr[i:batch_end], dtype=torch.float32, device=device)
        hist_norm = normalize_iv(hist)
        with torch.no_grad():
            samples = model.sample(hist_norm, n_samples=N_SAMPLES)
        all_samples.append(samples.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {batch_end}/{n_windows}")

    all_samples = np.concatenate(all_samples, axis=0)

    # Apply online conformal calibration
    from experiments.backfill.block_ar.conformal_calibration import (
        online_conformal_calibration,
    )
    baselines = history_arr[:, -1, :, :]  # (W, 5, 5) last history surface
    print("Applying online conformal calibration (W=100)...")
    corrected, conf_diag = online_conformal_calibration(
        all_samples, future_arr, baselines,
        vol_of_vol[:, None],
        window_size=100,
        regime_split=True,
        per_horizon=True,
        eval_horizons=[0, 6, 13, 29],
    )
    mean_corr = np.mean(conf_diag["per_window_correction_mean"]) if conf_diag["per_window_correction_mean"] else 1.0
    print(f"  Mean conformal correction: {mean_corr:.3f}")

    return {
        "history": history_arr,      # (W, 30, 5, 5)
        "future": future_arr,        # (W, 30, 5, 5)
        "samples": corrected,        # (W, S, 30, 5, 5) — conformally calibrated
        "samples_raw": all_samples,  # (W, S, 30, 5, 5) — raw
        "vol_of_vol": vol_of_vol,    # (W,)
        "n_windows": n_windows,
    }


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Per-Cell Fan Charts — Calm vs Turbulent
# ══════════════════════════════════════════════════════════════════════
def plot_fan_charts(data):
    """Fan charts for selected cells, calm vs turbulent side by side."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    # Pick a calm and turbulent window (P10 and P90 for robustness)
    calm_idx = sorted_idx[int(0.10 * len(sorted_idx))]
    turb_idx = sorted_idx[int(0.90 * len(sorted_idx))]

    # Representative cells: short-term OTM put, ATM mid, long-term ATM, short-term OTM call
    cells = [(0, 0), (2, 2), (4, 2), (0, 4)]
    cell_labels = [
        "1M OTM Put (K=0.70)\nHigh uncertainty",
        "6M ATM (K=1.00)\nMedium uncertainty",
        "2Y ATM (K=1.00)\nLow uncertainty",
        "1M OTM Call (K=1.30)\nHigh uncertainty",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    fig.suptitle("Scenario Fan Charts: Calm vs Turbulent Market Regimes\n"
                 "Same model, same cells — uncertainty widens in turbulent conditions",
                 fontsize=15, fontweight="bold", y=0.99)

    horizons = np.arange(1, 31)

    # First pass: compute y-ranges per row (shared across calm/turbulent)
    row_ylims = []
    for row, ((r, c), label) in enumerate(zip(cells, cell_labels)):
        ymin, ymax = np.inf, -np.inf
        for win_idx in [calm_idx, turb_idx]:
            samples_cell = data["samples"][win_idx, :, :, r, c]
            gt_cell = data["future"][win_idx, :, r, c]
            ymin = min(ymin, samples_cell.min(), gt_cell.min())
            ymax = max(ymax, samples_cell.max(), gt_cell.max())
        margin = (ymax - ymin) * 0.05
        row_ylims.append((ymin - margin, ymax + margin))

    for row, ((r, c), label) in enumerate(zip(cells, cell_labels)):
        for col, (win_idx, regime, color) in enumerate([
            (calm_idx, "Calm", CALM_COLOR),
            (turb_idx, "Turbulent", TURB_COLOR),
        ]):
            ax = axes[row, col]
            samples_cell = data["samples"][win_idx, :, :, r, c]  # (S, 30)
            gt_cell = data["future"][win_idx, :, r, c]            # (30,)

            # Plot individual scenario paths (thin, transparent)
            for s in range(min(20, N_SAMPLES)):
                ax.plot(horizons, samples_cell[s], color=color, alpha=0.08, linewidth=0.5)

            # Confidence bands
            for ci, pct_hi, band_alpha in [(5, 95, 0.12), (10, 90, 0.15), (25, 75, 0.20)]:
                lo = np.percentile(samples_cell, ci, axis=0)
                hi = np.percentile(samples_cell, pct_hi, axis=0)
                ax.fill_between(horizons, lo, hi, color=color, alpha=band_alpha)

            # Median and GT
            median = np.median(samples_cell, axis=0)
            ax.plot(horizons, median, color=color, linewidth=1.5, label="Median scenario")
            ax.plot(horizons, gt_cell, color=GT_COLOR, linewidth=2.0,
                    linestyle="--", label="Ground truth", zorder=5)

            # Shared y-axis per row
            ax.set_ylim(row_ylims[row])

            # Annotate spread
            spread = samples_cell.std(axis=0).mean()
            ax.text(0.97, 0.95, f"spread={spread*100:.1f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if row == 0:
                vov_val = vov[win_idx]
                ax.set_title(f"{regime} Regime (vol-of-vol={vov_val:.4f})",
                             fontsize=12, fontweight="bold")

            if col == 0:
                ax.set_ylabel(label, fontsize=10)

            if row == 3:
                ax.set_xlabel("Forecast Horizon (days)")

            if row == 0 and col == 1:
                ax.legend(fontsize=9, loc="upper right")

            ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{OUTPUT_DIR}/fig1_fan_charts_calm_vs_turbulent.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Cross-Cell Regime Sensitivity
# ══════════════════════════════════════════════════════════════════════
def plot_cross_cell_sensitivity(data):
    """Show same window, different cells — uncertainty varies spatially.
    Top row: turbulent window. Bottom row: calm window. Same 4 cells."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)
    calm_idx = sorted_idx[int(0.10 * len(sorted_idx))]
    turb_idx = sorted_idx[int(0.90 * len(sorted_idx))]

    # 4 cells spanning the surface corners + center
    cells = [(0, 0), (0, 4), (2, 2), (4, 2)]
    col_labels = [
        "1M OTM Put\n(K=0.70)",
        "1M OTM Call\n(K=1.30)",
        "6M ATM\n(K=1.00)",
        "2Y ATM\n(K=1.00)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
    fig.suptitle(
        "Same Cells, Different Regimes: Uncertainty Responds Differently Per Cell",
        fontsize=15, fontweight="bold", y=1.01,
    )

    horizons = np.arange(1, 31)

    for row_idx, (win_idx, regime, color) in enumerate([
        (turb_idx, "Turbulent", TURB_COLOR),
        (calm_idx, "Calm", CALM_COLOR),
    ]):
        for col_idx, ((r, c), col_label) in enumerate(zip(cells, col_labels)):
            ax = axes[row_idx, col_idx]
            samples_cell = data["samples"][win_idx, :, :, r, c]
            gt_cell = data["future"][win_idx, :, r, c]

            spread = samples_cell.std(axis=0).mean()

            for s in range(min(20, N_SAMPLES)):
                ax.plot(horizons, samples_cell[s], color=color, alpha=0.08, linewidth=0.5)

            q05 = np.percentile(samples_cell, 5, axis=0)
            q95 = np.percentile(samples_cell, 95, axis=0)
            ax.fill_between(horizons, q05, q95, color=color, alpha=0.15)

            median = np.median(samples_cell, axis=0)
            ax.plot(horizons, median, color=color, linewidth=1.5)
            ax.plot(horizons, gt_cell, color=GT_COLOR, linewidth=2.0, linestyle="--", zorder=5)

            # Spread annotation
            ax.text(0.97, 0.95, f"spread={spread*100:.1f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if row_idx == 0:
                ax.set_title(col_label, fontsize=11, fontweight="bold")
            if col_idx == 0:
                vov_val = vov[win_idx]
                ax.set_ylabel(f"{regime}\n(vov={vov_val:.4f})", fontsize=10, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Horizon (days)")
            ax.tick_params(labelsize=9)

    # Share y-axis per column so spread difference is visible
    for col_idx in range(4):
        ymin = min(axes[0, col_idx].get_ylim()[0], axes[1, col_idx].get_ylim()[0])
        ymax = max(axes[0, col_idx].get_ylim()[1], axes[1, col_idx].get_ylim()[1])
        for row_idx in range(2):
            axes[row_idx, col_idx].set_ylim(ymin, ymax)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_cross_cell_sensitivity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Temporal Properties — Kurtosis & ACF
# ══════════════════════════════════════════════════════════════════════
def plot_temporal_properties(data):
    """Cross-sample distribution match + ACF overlay: generated vs ground truth."""
    future = data["future"]        # (W, 30, 5, 5)
    samples = data["samples"]      # (W, S, 30, 5, 5)
    n_windows = data["n_windows"]

    gt_daily = np.diff(future, axis=1)  # (W, 29, 5, 5)
    gen_daily_all = []
    for s in range(min(10, N_SAMPLES)):
        gen_daily = np.diff(samples[:, s, :, :, :], axis=1)
        gen_daily_all.append(gen_daily)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Temporal Properties: Generated Scenarios vs Ground Truth",
                 fontsize=15, fontweight="bold")

    # Panel 1: Cross-sample distribution at h=30 (the hardest horizon)
    # This is what the test suite measures — kurtosis of the ensemble
    ax = axes[0]
    h_idx = 29  # h=30
    # Flatten across all windows and cells
    gt_vals_h30 = future[:, h_idx, :, :].ravel()
    # Pool one sample per window for comparable size
    gen_vals_h30 = samples[:, 0, h_idx, :, :].ravel()

    # Standardize to zero-mean unit-var for shape comparison
    gt_z = (gt_vals_h30 - gt_vals_h30.mean()) / gt_vals_h30.std()
    gen_z = (gen_vals_h30 - gen_vals_h30.mean()) / gen_vals_h30.std()

    bins = np.linspace(-4, 4, 80)
    ax.hist(gt_z, bins=bins, density=True, alpha=0.5, color=GT_COLOR, label="Ground truth")
    ax.hist(gen_z, bins=bins, density=True, alpha=0.5, color=TURB_COLOR, label="Generated")

    # Normal reference
    x = np.linspace(-4, 4, 200)
    ax.plot(x, sp_stats.norm.pdf(x), "k--", alpha=0.4, linewidth=1, label="Normal")

    gt_kurt = sp_stats.kurtosis(gt_vals_h30, fisher=True)
    gen_kurt = sp_stats.kurtosis(gen_vals_h30, fisher=True)
    ratio = gen_kurt / gt_kurt if gt_kurt > 0 else float("nan")
    ax.text(0.05, 0.88,
            f"GT kurtosis: {gt_kurt:.2f}\n"
            f"Gen kurtosis: {gen_kurt:.2f}\n"
            f"Ratio: {ratio:.2f} (1.0 = perfect)",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Standardized IV Level")
    ax.set_ylabel("Density")
    ax.set_title("Cross-Sample Distribution at h=30\n(Ensemble captures fat tails)")
    ax.legend(fontsize=9)
    ax.set_xlim(-4, 4)

    # Panel 2: ACF comparison for ATM cells
    ax = axes[1]
    max_lag = 15
    lags = np.arange(1, max_lag + 1)

    for r, c, label, ls in [(2, 2, "6M ATM", "-"), (0, 2, "1M ATM", "--"), (4, 2, "2Y ATM", ":")]:
        gt_series = np.abs(gt_daily[:, :, r, c].ravel())
        gt_acf = [np.corrcoef(gt_series[:-lag], gt_series[lag:])[0, 1] for lag in lags]

        gen_series = np.abs(np.array([gd[:, :, r, c].ravel() for gd in gen_daily_all]).ravel())
        n = min(len(gt_series), len(gen_series))
        gen_series_trimmed = gen_series[:n]
        gen_acf = [np.corrcoef(gen_series_trimmed[:-lag], gen_series_trimmed[lag:])[0, 1]
                   for lag in lags]

        ax.plot(lags, gt_acf, color=GT_COLOR, linestyle=ls, linewidth=2, label=f"GT {label}")
        ax.plot(lags, gen_acf, color=TURB_COLOR, linestyle=ls, linewidth=2,
                alpha=0.7, label=f"Gen {label}")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation of |Daily Changes|")
    ax.set_title("Volatility Clustering Preserved\n(ACF of absolute daily IV changes)")
    ax.legend(fontsize=7, ncol=2)

    # Panel 3: Daily change distribution (6M ATM)
    ax = axes[2]
    r, c = 2, 2
    gt_vals = gt_daily[:, :, r, c].ravel()
    gen_vals = np.array([gd[:, :, r, c].ravel() for gd in gen_daily_all]).ravel()

    bins = np.linspace(np.percentile(gt_vals, 0.5), np.percentile(gt_vals, 99.5), 80)
    ax.hist(gt_vals, bins=bins, density=True, alpha=0.5, color=GT_COLOR, label="Ground truth")
    ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, color=TURB_COLOR, label="Generated")

    gt_kurt = sp_stats.kurtosis(gt_vals, fisher=True)
    gen_kurt = sp_stats.kurtosis(gen_vals, fisher=True)
    ax.text(0.05, 0.88,
            f"GT kurtosis: {gt_kurt:.1f}\nGen kurtosis: {gen_kurt:.1f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Daily IV Change")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Daily Changes (6M ATM)")
    ax.legend(fontsize=9)

    gt_kurt = sp_stats.kurtosis(gt_vals, fisher=True)
    gen_kurt = sp_stats.kurtosis(gen_vals, fisher=True)
    ax.text(0.05, 0.92, f"GT kurtosis: {gt_kurt:.2f}\nGen kurtosis: {gen_kurt:.2f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig3_temporal_properties.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Term Structure & Smile Validation
# ══════════════════════════════════════════════════════════════════════
def plot_surface_structure(data):
    """Term structure and smile slices with scenario CI bands."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    # Three windows: calm, median, turbulent
    window_picks = [
        (sorted_idx[int(0.10 * len(sorted_idx))], "Calm"),
        (sorted_idx[int(0.50 * len(sorted_idx))], "Medium"),
        (sorted_idx[int(0.90 * len(sorted_idx))], "Turbulent"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Surface Structure Validation: Term Structure & Volatility Smile\n"
                 "Ground truth (green dashed) should fall within scenario band",
                 fontsize=14, fontweight="bold", y=1.02)

    for row, (win_idx, regime) in enumerate(window_picks):
        # h=1 for structure check
        h_idx = 0
        gt_surface = data["future"][win_idx, h_idx]        # (5, 5)
        gen_surfaces = data["samples"][win_idx, :, h_idx]   # (S, 5, 5)

        # --- Term structure (ATM slice, column 2) ---
        ax = axes[row, 0]
        atm_col = 2
        gt_term = gt_surface[:, atm_col]
        gen_term = gen_surfaces[:, :, atm_col]  # (S, 5)

        for s in range(min(15, N_SAMPLES)):
            ax.plot(MATURITIES_DAYS, gen_term[s], color=SCENARIO_COLOR, alpha=0.15, linewidth=0.5)

        q05 = np.percentile(gen_term, 5, axis=0)
        q25 = np.percentile(gen_term, 25, axis=0)
        q75 = np.percentile(gen_term, 75, axis=0)
        q95 = np.percentile(gen_term, 95, axis=0)
        median = np.median(gen_term, axis=0)

        ax.fill_between(MATURITIES_DAYS, q05, q95, alpha=0.12, color=TURB_COLOR, label="90% CI")
        ax.fill_between(MATURITIES_DAYS, q25, q75, alpha=0.20, color=TURB_COLOR, label="50% CI")
        ax.plot(MATURITIES_DAYS, median, color=TURB_COLOR, linewidth=1.5, label="Median")
        ax.plot(MATURITIES_DAYS, gt_term, color=GT_COLOR, linewidth=2.5,
                linestyle="--", marker="o", markersize=6, label="Ground truth", zorder=5)

        ax.set_xlabel("Maturity (days)")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{regime} — ATM Term Structure (K=1.00)")
        if row == 0:
            ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(MATURITIES_DAYS)
        ax.set_xticklabels(MATURITY_LABELS)

        # --- Smile (6M slice, row 2) ---
        ax = axes[row, 1]
        tenor_row = 2
        gt_smile = gt_surface[tenor_row, :]
        gen_smile = gen_surfaces[:, tenor_row, :]  # (S, 5)

        for s in range(min(15, N_SAMPLES)):
            ax.plot(MONEYNESS, gen_smile[s], color=SCENARIO_COLOR, alpha=0.15, linewidth=0.5)

        q05 = np.percentile(gen_smile, 5, axis=0)
        q25 = np.percentile(gen_smile, 25, axis=0)
        q75 = np.percentile(gen_smile, 75, axis=0)
        q95 = np.percentile(gen_smile, 95, axis=0)
        median = np.median(gen_smile, axis=0)

        ax.fill_between(MONEYNESS, q05, q95, alpha=0.12, color=CALM_COLOR, label="90% CI")
        ax.fill_between(MONEYNESS, q25, q75, alpha=0.20, color=CALM_COLOR, label="50% CI")
        ax.plot(MONEYNESS, median, color=CALM_COLOR, linewidth=1.5, label="Median")
        ax.plot(MONEYNESS, gt_smile, color=GT_COLOR, linewidth=2.5,
                linestyle="--", marker="o", markersize=6, label="Ground truth", zorder=5)

        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{regime} — 6M Volatility Smile")
        if row == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig4_term_structure_smile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Surface Heatmap Snapshots
# ══════════════════════════════════════════════════════════════════════
def plot_surface_heatmaps(data):
    """GT surface, mean generated, and uncertainty heatmaps."""
    vov = data["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    calm_idx = sorted_idx[int(0.10 * len(sorted_idx))]
    turb_idx = sorted_idx[int(0.90 * len(sorted_idx))]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Surface Snapshots at h=1: Ground Truth vs Generated Mean vs Uncertainty Map",
                 fontsize=14, fontweight="bold")

    for row, (win_idx, regime) in enumerate([(calm_idx, "Calm"), (turb_idx, "Turbulent")]):
        h_idx = 0
        gt = data["future"][win_idx, h_idx]            # (5, 5)
        gen = data["samples"][win_idx, :, h_idx]        # (S, 5, 5)
        gen_mean = gen.mean(axis=0)
        gen_std = gen.std(axis=0)
        error = np.abs(gt - gen_mean)

        vmin = min(gt.min(), gen_mean.min())
        vmax = max(gt.max(), gen_mean.max())

        for col, (arr, title, cmap, vmin_o, vmax_o) in enumerate([
            (gt, f"{regime}: Ground Truth", "YlOrRd", vmin, vmax),
            (gen_mean, f"{regime}: Scenario Mean", "YlOrRd", vmin, vmax),
            (gen_std, f"{regime}: Uncertainty (σ)", "Reds", None, None),
            (error, f"{regime}: |Error|", "Blues", None, None),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin_o, vmax=vmax_o, aspect="auto")
            ax.set_xticks(range(5))
            ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels(MATURITY_LABELS, fontsize=8)
            ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel("Maturity")
            if row == 1:
                ax.set_xlabel("Moneyness")

            # Annotate values
            for r in range(5):
                for c in range(5):
                    val = arr[r, c]
                    text_color = "white" if val > (vmax_o or arr.max()) * 0.6 else "black"
                    if col >= 2:
                        ax.text(c, r, f"{val*100:.2f}", ha="center", va="center",
                                fontsize=7, color=text_color)
                    else:
                        ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                                fontsize=7, color=text_color)

            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig5_surface_heatmaps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Calibration Curve
# ══════════════════════════════════════════════════════════════════════
def plot_calibration_curve(data):
    """Nominal vs empirical coverage across CI levels and horizons."""
    future = data["future"]
    samples = data["samples"]

    nominal_levels = np.arange(0.1, 1.0, 0.1)  # 10%, 20%, ..., 90%

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.suptitle("Calibration Curve: Is the Model Well-Calibrated?",
                 fontsize=14, fontweight="bold")

    horizon_specs = [(0, "h=1", "-", "o"), (6, "h=7", "--", "s"),
                     (13, "h=14", "-.", "^"), (29, "h=30", ":", "D")]

    for h_idx, h_name, ls, marker in horizon_specs:
        gt = future[:, h_idx]            # (W, 5, 5)
        s = samples[:, :, h_idx]          # (W, S, 5, 5)

        empirical = []
        for nom in nominal_levels:
            alpha = (1 - nom) / 2 * 100
            lo = np.percentile(s, alpha, axis=1)
            hi = np.percentile(s, 100 - alpha, axis=1)
            covered = ((gt >= lo) & (gt <= hi)).mean()
            empirical.append(covered)

        ax.plot(nominal_levels * 100, np.array(empirical) * 100,
                linestyle=ls, marker=marker, markersize=6, linewidth=2,
                label=h_name)

    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, linewidth=1, label="Perfect calibration")
    ax.fill_between([0, 100], [0, 100], [5, 105], alpha=0.05, color="green")
    ax.fill_between([0, 100], [-5, 95], [0, 100], alpha=0.05, color="green")

    ax.set_xlabel("Nominal Coverage Level (%)", fontsize=12)
    ax.set_ylabel("Empirical Coverage (%)", fontsize=12)
    ax.set_xlim(5, 95)
    ax.set_ylim(5, 100)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_aspect("equal")

    # Add annotation
    ax.text(0.05, 0.88,
            "Points on the diagonal = perfectly calibrated\n"
            "Above = conservative (wider CIs than needed)\n"
            "Below = overconfident (CIs too narrow)",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig6_calibration_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model, config = load_model(MODEL_PATH, device)

    print("Generating data...")
    data = generate_all_data(model, config, device)

    print("\nGenerating visualizations...")

    print("\n[1/6] Fan charts: calm vs turbulent...")
    plot_fan_charts(data)

    print("[2/6] Cross-cell sensitivity...")
    plot_cross_cell_sensitivity(data)

    print("[3/6] Temporal properties...")
    plot_temporal_properties(data)

    print("[4/6] Term structure & smile...")
    plot_surface_structure(data)

    print("[5/6] Surface heatmaps...")
    plot_surface_heatmaps(data)

    print("[6/6] Calibration curve...")
    plot_calibration_curve(data)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
