#!/usr/bin/env python
"""
Management Report V2: Honest validation with per-cell detail.

Addresses issues from V1:
- Per-cell CI coverage by regime (not aggregate)
- Per-cell marginal distributions (not averaged)
- ACF computed intra-window (not pooled, which inflates GT)
- Kurtosis heatmap (dark/light daily changes, GT vs generated)
- Term structure/smile: shows coverage annotations and calm bias
- Full risk management validation
"""

import dataclasses
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from scipy import stats as sp_stats

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    normalize_iv,
)
from diffusion.block_ar.single_pass_ar import (
    SinglePassConfig,
    SinglePassBlockAR,
)

MONEYNESS = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
MATURITIES_DAYS = np.array([30, 91, 182, 365, 730])
MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y"]
MONEYNESS_LABELS = ["0.70", "0.85", "1.00", "1.15", "1.30"]

OUTPUT_DIR = "results/block_ar/management_report_99j_v3"
MODEL_PATH = "models/backfill/afcrps_99j_v3/best_model.pt"
N_SAMPLES = 50
MAX_WINDOWS = 400

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

CALM_COLOR = "#2196F3"
TURB_COLOR = "#F44336"
GT_COLOR = "#2E7D32"


def load_and_generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cp = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    c = cp["config"]
    if dataclasses.is_dataclass(c):
        c = dataclasses.asdict(c)
    # Detect model type by config keys
    if "ar_frame" in c or "noise_dim" in c:
        config = SinglePassConfig(**c)
        model = SinglePassBlockAR(config)
    else:
        config = BlockARConfig(**c)
        model = ConditionalBlockARDDPM(config)
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_surfaces = surfaces[4540:]

    hl, fl = config.history_len, config.future_len
    N = len(test_surfaces)
    n_windows = min(MAX_WINDOWS, N - hl - fl + 1)
    indices = np.linspace(0, N - hl - fl, n_windows, dtype=int)

    all_h, all_f = [], []
    for idx in indices:
        all_h.append(test_surfaces[idx:idx + hl])
        all_f.append(test_surfaces[idx + hl:idx + hl + fl])
    history_arr = np.stack(all_h)
    future_arr = np.stack(all_f)

    mean_iv = history_arr.mean(axis=(-1, -2))
    vol_of_vol = np.diff(mean_iv, axis=1).std(axis=1)

    print(f"Generating {N_SAMPLES} samples for {n_windows} windows...")
    batch_size = 16
    all_samples = []
    for i in range(0, n_windows, batch_size):
        be = min(i + batch_size, n_windows)
        hist = torch.tensor(history_arr[i:be], dtype=torch.float32, device=device)
        with torch.no_grad():
            samples = model.sample(normalize_iv(hist), n_samples=N_SAMPLES)
        all_samples.append(samples.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  {be}/{n_windows}")
    all_samples = np.concatenate(all_samples, axis=0)

    quintiles = np.percentile(vol_of_vol, [0, 20, 80, 100])
    calm_mask = vol_of_vol <= quintiles[1]
    turb_mask = vol_of_vol >= quintiles[2]

    return {
        "history": history_arr, "future": future_arr, "samples": all_samples,
        "vol_of_vol": vol_of_vol, "n_windows": n_windows,
        "calm_mask": calm_mask, "turb_mask": turb_mask,
    }


# ══════════════════════════════════════════════════════════════════════
# FIG 1: Fan charts with per-cell coverage annotations
# ══════════════════════════════════════════════════════════════════════
def fig1_fan_charts(d):
    """Fan charts using AVERAGED calm/turbulent windows (not cherry-picked)."""
    cells = [(0, 0), (0, 2), (2, 2), (4, 2)]
    cell_labels = ["1M OTM Put\n(K=0.70)", "1M ATM\n(K=1.00)",
                   "6M ATM\n(K=1.00)", "2Y ATM\n(K=1.00)"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    fig.suptitle("Scenario Fan Charts with Per-Cell Coverage\n"
                 "Red = coverage < 85%  |  Green text = GT in band",
                 fontsize=14, fontweight="bold", y=0.99)

    horizons = np.arange(1, 31)

    for row, ((r, c), label) in enumerate(zip(cells, cell_labels)):
        # Compute shared y-limits across calm/turbulent
        ymin, ymax = np.inf, -np.inf
        for mask in [d["calm_mask"], d["turb_mask"]]:
            s_all = d["samples"][mask, :, :, r, c]  # (M, S, 30)
            f_all = d["future"][mask, :, r, c]       # (M, 30)
            ymin = min(ymin, s_all.min(), f_all.min())
            ymax = max(ymax, s_all.max(), f_all.max())
        margin = (ymax - ymin) * 0.05

        for col, (mask, regime, color) in enumerate([
            (d["calm_mask"], "Calm", CALM_COLOR),
            (d["turb_mask"], "Turbulent", TURB_COLOR),
        ]):
            ax = axes[row, col]
            # Average across all windows in this regime
            s_regime = d["samples"][mask, :, :, r, c]  # (M, S, 30)
            f_regime = d["future"][mask, :, r, c]       # (M, 30)

            # Plot 5 randomly chosen windows' fan charts (not just one)
            np.random.seed(42)
            picks = np.random.choice(mask.sum(), size=min(5, mask.sum()), replace=False)

            # Aggregate percentiles across ALL windows in regime
            # For each horizon, pool samples across all windows
            all_s = s_regime.reshape(-1, s_regime.shape[-1])  # (M*S, 30)
            q05 = np.percentile(s_regime, 5, axis=1).mean(axis=0)  # avg Q5 across windows
            q95 = np.percentile(s_regime, 95, axis=1).mean(axis=0)
            q25 = np.percentile(s_regime, 25, axis=1).mean(axis=0)
            q75 = np.percentile(s_regime, 75, axis=1).mean(axis=0)
            median = np.median(s_regime, axis=1).mean(axis=0)
            gt_mean = f_regime.mean(axis=0)

            ax.fill_between(horizons, q05, q95, color=color, alpha=0.12, label="Avg 90% CI")
            ax.fill_between(horizons, q25, q75, color=color, alpha=0.20, label="Avg 50% CI")
            ax.plot(horizons, median, color=color, linewidth=1.5, label="Avg median")
            ax.plot(horizons, gt_mean, color=GT_COLOR, linewidth=2.0,
                    linestyle="--", label="Avg GT", zorder=5)

            # Per-cell coverage for this regime
            per_h_cov = []
            for h_idx in range(30):
                gt_h = f_regime[:, h_idx]
                q05_h = np.percentile(s_regime[:, :, h_idx], 5, axis=1)
                q95_h = np.percentile(s_regime[:, :, h_idx], 95, axis=1)
                covered = ((gt_h >= q05_h) & (gt_h <= q95_h)).mean()
                per_h_cov.append(covered)
            mean_cov = np.mean(per_h_cov)

            # Coverage annotation
            cov_color = "red" if mean_cov < 0.85 else "green"
            ax.text(0.97, 0.95, f"90% CI cov: {mean_cov:.1%}",
                    transform=ax.transAxes, fontsize=9, ha="right", va="top",
                    color=cov_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

            # Bias annotation
            bias = (median - gt_mean).mean()
            ax.text(0.97, 0.83, f"bias: {bias*1e3:+.1f}×10⁻³",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

            ax.set_ylim(ymin - margin, ymax + margin)
            if row == 0:
                ax.set_title(f"{regime}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 3:
                ax.set_xlabel("Forecast Horizon (days)")
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="upper left")
            ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f"{OUTPUT_DIR}/fig1_fan_charts_with_coverage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 2: Per-cell CI coverage heatmap by regime
# ══════════════════════════════════════════════════════════════════════
def fig2_coverage_heatmaps(d):
    """5x5 coverage heatmaps for calm, medium, turbulent, all."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle("Per-Cell 90% CI Coverage by Regime and Horizon\n"
                 "Target: 90%. Red cells = under-covered. Blue cells = over-covered.",
                 fontsize=14, fontweight="bold")

    all_mask = np.ones(d["n_windows"], bool)
    masks = [(d["calm_mask"], "Calm (Q1)"), (d["turb_mask"], "Turbulent (Q5)"), (all_mask, "All Windows")]
    h_specs = [(0, "h=1"), (6, "h=7"), (29, "h=30")]

    for col, (h_idx, h_name) in enumerate(h_specs):
        for row, (mask, regime) in enumerate(masks):
            ax = axes[row, col]
            gt = d["future"][mask, h_idx]
            s = d["samples"][mask, :, h_idx]
            q05 = np.percentile(s, 5, axis=1)
            q95 = np.percentile(s, 95, axis=1)
            covered = ((gt >= q05) & (gt <= q95)).mean(axis=0)

            # Diverging colormap centered at 0.90
            norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=0.90, vmax=1.0)
            im = ax.imshow(covered, cmap="RdYlGn", norm=norm, aspect="auto")

            for r in range(5):
                for c in range(5):
                    val = covered[r, c]
                    color = "white" if val < 0.65 else "black"
                    weight = "bold" if val < 0.80 else "normal"
                    ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                            fontsize=9, color=color, fontweight=weight)

            ax.set_xticks(range(5))
            ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels(MATURITY_LABELS, fontsize=8)

            if row == 0:
                ax.set_title(f"{h_name}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{regime}", fontsize=11, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Moneyness")

            plt.colorbar(im, ax=ax, shrink=0.8, label="Coverage")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig2_percell_coverage_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 3: Per-cell marginal distribution comparison
# ══════════════════════════════════════════════════════════════════════
def fig3_percell_marginals(d):
    """5x5 grid of per-cell daily change distributions, GT vs generated."""
    gt_daily = np.diff(d["future"], axis=1)       # (W, 29, 5, 5)
    gen_daily = np.diff(d["samples"][:, 0], axis=1)  # (W, 29, 5, 5) single sample

    fig, axes = plt.subplots(5, 5, figsize=(20, 18))
    fig.suptitle("Per-Cell Distribution of Daily IV Changes: Ground Truth vs Generated\n"
                 "Each cell shows the marginal at that grid point (maturity × moneyness)",
                 fontsize=14, fontweight="bold", y=1.01)

    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            gt_vals = gt_daily[:, :, r, c].ravel()
            gen_vals = gen_daily[:, :, r, c].ravel()

            # Use common bins
            pct = max(np.percentile(np.abs(gt_vals), 99),
                      np.percentile(np.abs(gen_vals), 99))
            bins = np.linspace(-pct, pct, 50)

            ax.hist(gt_vals, bins=bins, density=True, alpha=0.5,
                    color=GT_COLOR, label="GT" if r == 0 and c == 0 else None)
            ax.hist(gen_vals, bins=bins, density=True, alpha=0.5,
                    color=TURB_COLOR, label="Gen" if r == 0 and c == 0 else None)

            gt_k = sp_stats.kurtosis(gt_vals, fisher=True)
            gen_k = sp_stats.kurtosis(gen_vals, fisher=True)
            ks_stat, ks_p = sp_stats.ks_2samp(gt_vals, gen_vals)

            ax.text(0.95, 0.95,
                    f"K_gt={gt_k:.1f}\nK_gen={gen_k:.1f}\nKS={ks_stat:.3f}",
                    transform=ax.transAxes, fontsize=6, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if r == 0:
                ax.set_title(f"K={MONEYNESS_LABELS[c]}", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(MATURITY_LABELS[r], fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=6)
            ax.set_yticks([])

    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig3_percell_marginal_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 4: Kurtosis heatmap — dark/light daily changes
# ══════════════════════════════════════════════════════════════════════
def fig4_kurtosis_heatmap(d):
    """Heatmap of |daily changes| over time for GT vs generated scenarios."""
    vov = d["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    # Pick 3 windows: calm, medium, turbulent
    picks = [sorted_idx[int(0.1 * len(sorted_idx))],
             sorted_idx[int(0.5 * len(sorted_idx))],
             sorted_idx[int(0.9 * len(sorted_idx))]]
    regime_labels = ["Calm", "Medium", "Turbulent"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Volatility Clustering: |Daily IV Changes| Over Time\n"
                 "Dark = large change (fat tail event), Light = small change\n"
                 "Clustering of dark cells = vol persistence",
                 fontsize=14, fontweight="bold", y=1.02)

    for row, (win_idx, regime) in enumerate(zip(picks, regime_labels)):
        # GT daily changes for this window — flatten 5x5 to 25 cells
        gt_future = d["future"][win_idx]  # (30, 5, 5)
        gt_daily = np.abs(np.diff(gt_future, axis=0))  # (29, 5, 5)
        gt_flat = gt_daily.reshape(29, 25).T  # (25, 29)

        # Two generated samples
        gen1_future = d["samples"][win_idx, 0]  # (30, 5, 5)
        gen1_daily = np.abs(np.diff(gen1_future, axis=0))
        gen1_flat = gen1_daily.reshape(29, 25).T

        gen2_future = d["samples"][win_idx, 1]
        gen2_daily = np.abs(np.diff(gen2_future, axis=0))
        gen2_flat = gen2_daily.reshape(29, 25).T

        # Common scale
        vmax = max(gt_flat.max(), gen1_flat.max(), gen2_flat.max())

        cell_labels = [f"{MATURITY_LABELS[r]}/{MONEYNESS_LABELS[c]}"
                       for r in range(5) for c in range(5)]

        for col, (arr, title) in enumerate([
            (gt_flat, f"{regime}: Ground Truth"),
            (gen1_flat, f"{regime}: Scenario #1"),
            (gen2_flat, f"{regime}: Scenario #2"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap="hot_r", vmin=0, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Day")
            if col == 0:
                ax.set_yticks(range(25))
                ax.set_yticklabels(cell_labels, fontsize=5)
            else:
                ax.set_yticks([])
            ax.set_xticks(np.arange(0, 29, 5))

        plt.colorbar(im, ax=axes[row, 2], shrink=0.8, label="|ΔIV|")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig4_kurtosis_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 5: ACF — corrected (intra-window, not pooled)
# ══════════════════════════════════════════════════════════════════════
def fig5_acf_corrected(d):
    """Intra-window ACF of |daily changes|, properly averaged."""
    gt_daily = np.diff(d["future"], axis=1)  # (W, 29, 5, 5)
    # Use multiple samples for generated
    gen_dailies = [np.diff(d["samples"][:, s], axis=1) for s in range(min(10, N_SAMPLES))]

    max_lag = 15
    lags = np.arange(1, max_lag + 1)

    cells_to_plot = [(0, 0, "1M K=0.70"), (0, 2, "1M ATM"), (2, 2, "6M ATM"),
                     (4, 2, "2Y ATM"), (0, 4, "1M K=1.30"), (4, 4, "2Y K=1.30")]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Volatility Clustering: Intra-Window ACF of |Daily Changes|\n"
                 "(Computed within each 30-day window, then averaged — no cross-window inflation)",
                 fontsize=13, fontweight="bold")

    for idx, (r, c, label) in enumerate(cells_to_plot):
        ax = axes[idx // 3, idx % 3]

        # GT intra-window ACF
        gt_acfs = np.zeros((d["n_windows"], max_lag))
        for w in range(d["n_windows"]):
            series = np.abs(gt_daily[w, :, r, c])
            for li, lag in enumerate(lags):
                if len(series) > lag:
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    gt_acfs[w, li] = corr if not np.isnan(corr) else 0

        # Gen intra-window ACF (averaged across samples)
        gen_acfs = np.zeros((d["n_windows"], max_lag))
        for gen_d in gen_dailies:
            for w in range(d["n_windows"]):
                series = np.abs(gen_d[w, :, r, c])
                for li, lag in enumerate(lags):
                    if len(series) > lag:
                        corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                        gen_acfs[w, li] += (corr if not np.isnan(corr) else 0)
        gen_acfs /= len(gen_dailies)

        gt_mean = gt_acfs.mean(axis=0)
        gt_std = gt_acfs.std(axis=0) / np.sqrt(d["n_windows"])
        gen_mean = gen_acfs.mean(axis=0)
        gen_std = gen_acfs.std(axis=0) / np.sqrt(d["n_windows"])

        ax.plot(lags, gt_mean, color=GT_COLOR, linewidth=2, marker="o",
                markersize=4, label="GT")
        ax.fill_between(lags, gt_mean - 2 * gt_std, gt_mean + 2 * gt_std,
                         color=GT_COLOR, alpha=0.15)
        ax.plot(lags, gen_mean, color=TURB_COLOR, linewidth=2, marker="s",
                markersize=4, label="Generated")
        ax.fill_between(lags, gen_mean - 2 * gen_std, gen_mean + 2 * gen_std,
                         color=TURB_COLOR, alpha=0.15)

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lag (days)")
        if idx % 3 == 0:
            ax.set_ylabel("ACF of |ΔIV|")
        if idx == 0:
            ax.legend(fontsize=9)
        ax.tick_params(labelsize=8)

        mae = np.abs(gt_mean - gen_mean).mean()
        ax.text(0.95, 0.95, f"MAE={mae:.3f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig5_acf_intrawindow.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 6: Term structure & smile with coverage per point
# ══════════════════════════════════════════════════════════════════════
def fig6_term_structure_smile(d):
    """Term structure and smile with per-point coverage annotations."""
    vov = d["vol_of_vol"]
    sorted_idx = np.argsort(vov)

    # Use AVERAGED windows per regime (not single cherry-picked window)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Surface Structure: Term Structure & Smile with Per-Point Coverage\n"
                 "Averaged across all windows in each regime quintile",
                 fontsize=13, fontweight="bold", y=1.02)

    regimes = [(d["calm_mask"], "Calm (Q1)", CALM_COLOR),
               (np.ones(d["n_windows"], bool), "All Windows", "#9C27B0"),
               (d["turb_mask"], "Turbulent (Q5)", TURB_COLOR)]

    h_idx = 0  # h=1

    for row, (mask, regime, color) in enumerate(regimes):
        gt = d["future"][mask, h_idx]        # (M, 5, 5)
        s = d["samples"][mask, :, h_idx]     # (M, S, 5, 5)

        gt_mean = gt.mean(axis=0)
        s_mean_q05 = np.percentile(s, 5, axis=1).mean(axis=0)
        s_mean_q25 = np.percentile(s, 25, axis=1).mean(axis=0)
        s_mean_q75 = np.percentile(s, 75, axis=1).mean(axis=0)
        s_mean_q95 = np.percentile(s, 95, axis=1).mean(axis=0)
        s_median = np.median(s, axis=1).mean(axis=0)

        # Per-point coverage
        q05_per = np.percentile(s, 5, axis=1)
        q95_per = np.percentile(s, 95, axis=1)
        percell_cov = ((gt >= q05_per) & (gt <= q95_per)).mean(axis=0)

        # Term structure (ATM, col 2)
        ax = axes[row, 0]
        atm = 2
        ax.fill_between(MATURITIES_DAYS, s_mean_q05[:, atm], s_mean_q95[:, atm],
                         alpha=0.12, color=color, label="Avg 90% CI")
        ax.fill_between(MATURITIES_DAYS, s_mean_q25[:, atm], s_mean_q75[:, atm],
                         alpha=0.20, color=color, label="Avg 50% CI")
        ax.plot(MATURITIES_DAYS, s_median[:, atm], color=color, linewidth=1.5, label="Avg Median")
        ax.plot(MATURITIES_DAYS, gt_mean[:, atm], color=GT_COLOR, linewidth=2.5,
                linestyle="--", marker="o", markersize=8, label="Avg GT", zorder=5)

        # Annotate coverage per point
        for i in range(5):
            cov = percell_cov[i, atm]
            cov_color = "red" if cov < 0.85 else GT_COLOR
            ax.annotate(f"{cov:.0%}", (MATURITIES_DAYS[i], gt_mean[i, atm]),
                        textcoords="offset points", xytext=(8, -12),
                        fontsize=8, color=cov_color, fontweight="bold")

        ax.set_xscale("log")
        ax.set_xticks(MATURITIES_DAYS)
        ax.set_xticklabels(MATURITY_LABELS)
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{regime} — ATM Term Structure")
        if row == 0:
            ax.legend(fontsize=7)
        if row == 2:
            ax.set_xlabel("Maturity")

        # Smile (6M, row 2)
        ax = axes[row, 1]
        tenor = 2
        ax.fill_between(MONEYNESS, s_mean_q05[tenor, :], s_mean_q95[tenor, :],
                         alpha=0.12, color=color, label="Avg 90% CI")
        ax.fill_between(MONEYNESS, s_mean_q25[tenor, :], s_mean_q75[tenor, :],
                         alpha=0.20, color=color, label="Avg 50% CI")
        ax.plot(MONEYNESS, s_median[tenor, :], color=color, linewidth=1.5, label="Avg Median")
        ax.plot(MONEYNESS, gt_mean[tenor, :], color=GT_COLOR, linewidth=2.5,
                linestyle="--", marker="o", markersize=8, label="Avg GT", zorder=5)

        for i in range(5):
            cov = percell_cov[tenor, i]
            cov_color = "red" if cov < 0.85 else GT_COLOR
            ax.annotate(f"{cov:.0%}", (MONEYNESS[i], gt_mean[tenor, i]),
                        textcoords="offset points", xytext=(8, -12),
                        fontsize=8, color=cov_color, fontweight="bold")

        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"{regime} — 6M Smile")
        if row == 0:
            ax.legend(fontsize=7)
        if row == 2:
            ax.set_xlabel("Moneyness (K/S)")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig6_term_structure_smile_coverage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 7: Calibration curve (kept from V1, still useful)
# ══════════════════════════════════════════════════════════════════════
def fig7_calibration_curve(d):
    """Calibration curve, per-regime."""
    nominal_levels = np.arange(0.1, 1.0, 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Calibration Curves by Regime", fontsize=14, fontweight="bold")

    all_mask = np.ones(d["n_windows"], bool)
    for ax_idx, (mask, regime) in enumerate([
        (d["calm_mask"], "Calm (Q1)"),
        (d["turb_mask"], "Turbulent (Q5)"),
        (all_mask, "All Windows"),
    ]):
        ax = axes[ax_idx]
        for h_idx, h_name, ls, marker in [(0, "h=1", "-", "o"), (6, "h=7", "--", "s"),
                                           (13, "h=14", "-.", "^"), (29, "h=30", ":", "D")]:
            gt = d["future"][mask, h_idx]
            s = d["samples"][mask, :, h_idx]
            empirical = []
            for nom in nominal_levels:
                alpha = (1 - nom) / 2 * 100
                lo = np.percentile(s, alpha, axis=1)
                hi = np.percentile(s, 100 - alpha, axis=1)
                covered = ((gt >= lo) & (gt <= hi)).mean()
                empirical.append(covered)
            ax.plot(nominal_levels * 100, np.array(empirical) * 100,
                    linestyle=ls, marker=marker, markersize=5, linewidth=2, label=h_name)

        ax.plot([0, 100], [0, 100], "k--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Nominal Coverage (%)")
        ax.set_ylabel("Empirical Coverage (%)")
        ax.set_title(regime, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(5, 95)
        ax.set_ylim(5, 100)
        ax.set_aspect("equal")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig7_calibration_by_regime.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIG 8: Risk management — bias map + correlation structure
# ══════════════════════════════════════════════════════════════════════
def fig8_risk_management(d):
    """Comprehensive risk management validation."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Risk Management Validation",
                 fontsize=14, fontweight="bold")

    h_idx = 0  # h=1

    # Panel 1: Median bias heatmap (calm)
    ax = axes[0, 0]
    gt = d["future"][d["calm_mask"], h_idx]
    s = d["samples"][d["calm_mask"], :, h_idx]
    median = np.median(s, axis=1)
    bias = (median - gt).mean(axis=0)
    lim = max(np.abs(bias).max(), 0.001)
    im = ax.imshow(bias, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{bias[r,c]*1e3:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Calm: Median Bias (×1e3)")
    ax.set_xticks(range(5)); ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(MATURITY_LABELS, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: Median bias heatmap (turbulent)
    ax = axes[0, 1]
    gt = d["future"][d["turb_mask"], h_idx]
    s = d["samples"][d["turb_mask"], :, h_idx]
    median = np.median(s, axis=1)
    bias = (median - gt).mean(axis=0)
    lim = max(np.abs(bias).max(), 0.001)
    im = ax.imshow(bias, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{bias[r,c]*1e3:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Turbulent: Median Bias (×1e3)")
    ax.set_xticks(range(5)); ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(MATURITY_LABELS, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: Cross-cell correlation preservation
    ax = axes[0, 2]
    # GT cross-cell correlation at h=1
    gt_all = d["future"][:, h_idx].reshape(d["n_windows"], 25)
    gen_all = d["samples"][:, 0, h_idx].reshape(d["n_windows"], 25)
    gt_corr = np.corrcoef(gt_all.T)  # (25, 25)
    gen_corr = np.corrcoef(gen_all.T)
    # Scatter plot of correlation pairs
    triu_idx = np.triu_indices(25, k=1)
    gt_pairs = gt_corr[triu_idx]
    gen_pairs = gen_corr[triu_idx]
    ax.scatter(gt_pairs, gen_pairs, s=8, alpha=0.5, c=TURB_COLOR)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    corr_of_corr = np.corrcoef(gt_pairs, gen_pairs)[0, 1]
    ax.text(0.05, 0.92, f"Corr of corrs: {corr_of_corr:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("GT Cross-Cell Correlation")
    ax.set_ylabel("Generated Cross-Cell Correlation")
    ax.set_title("Correlation Structure Preservation")

    # Panel 4: Tail coverage (VaR-style)
    ax = axes[1, 0]
    # For each cell: what fraction of GT falls below 5th percentile of scenarios?
    # Should be 5%. If > 5% → model misses left tail.
    gt_all_h1 = d["future"][:, h_idx]  # (W, 5, 5)
    s_all_h1 = d["samples"][:, :, h_idx]  # (W, S, 5, 5)
    q05 = np.percentile(s_all_h1, 5, axis=1)  # (W, 5, 5)
    left_tail_miss = (gt_all_h1 < q05).mean(axis=0)  # (5, 5) should be ~0.05
    im = ax.imshow(left_tail_miss, cmap="Reds", vmin=0, vmax=0.20, aspect="auto")
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{left_tail_miss[r,c]:.1%}", ha="center", va="center", fontsize=8)
    ax.set_title("Left Tail Miss Rate (target: 5%)")
    ax.set_xticks(range(5)); ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(MATURITY_LABELS, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="P(GT < Q5)")

    # Panel 5: Right tail miss
    ax = axes[1, 1]
    q95 = np.percentile(s_all_h1, 95, axis=1)
    right_tail_miss = (gt_all_h1 > q95).mean(axis=0)
    im = ax.imshow(right_tail_miss, cmap="Reds", vmin=0, vmax=0.20, aspect="auto")
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{right_tail_miss[r,c]:.1%}", ha="center", va="center", fontsize=8)
    ax.set_title("Right Tail Miss Rate (target: 5%)")
    ax.set_xticks(range(5)); ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels(MATURITY_LABELS, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="P(GT > Q95)")

    # Panel 6: Growing uncertainty check
    ax = axes[1, 2]
    horizons = np.arange(1, 31)
    for mask, regime, color in [(d["calm_mask"], "Calm", CALM_COLOR),
                                 (d["turb_mask"], "Turbulent", TURB_COLOR)]:
        mean_spread = []
        for h in range(30):
            spread = d["samples"][mask, :, h].std(axis=1).mean()
            mean_spread.append(spread)
        ax.plot(horizons, mean_spread, color=color, linewidth=2, label=regime)
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Mean Scenario Spread (σ)")
    ax.set_title("Growing Uncertainty with Horizon")
    ax.legend()

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig8_risk_management.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading and generating...")
    d = load_and_generate()

    print("\n[1/8] Fan charts with coverage...")
    fig1_fan_charts(d)
    print("[2/8] Per-cell coverage heatmaps...")
    fig2_coverage_heatmaps(d)
    print("[3/8] Per-cell marginal distributions...")
    fig3_percell_marginals(d)
    print("[4/8] Kurtosis heatmap (dark/light)...")
    fig4_kurtosis_heatmap(d)
    print("[5/8] ACF corrected (intra-window)...")
    fig5_acf_corrected(d)
    print("[6/8] Term structure & smile with coverage...")
    fig6_term_structure_smile(d)
    print("[7/8] Calibration by regime...")
    fig7_calibration_curve(d)
    print("[8/8] Risk management validation...")
    fig8_risk_management(d)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
