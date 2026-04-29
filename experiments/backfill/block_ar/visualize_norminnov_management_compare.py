#!/usr/bin/env python
"""Paper-style management reports for normalized-innovation checkpoints.

The report compares two framework candidates on IV-only and anchor-only scopes.
It intentionally does not plot the native joint panel; joint visualization needs
its own dimensionality-reduction design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    reconstruct_state_from_increments,
)
import experiments.backfill.block_ar.visualize_management_report as v1  # noqa: E402
from experiments.backfill.block_ar.visualize_management_report_183c_v1 import (  # noqa: E402
    plot_burst_reversion_paths,
    plot_mean_reversion_clustering,
)


FACTOR_DISPLAY = {
    "factor:spx": "S&P 500",
    "factor:usdcad": "USD/CAD",
    "factor:usdjpy": "USD/JPY",
    "factor:dxy": "DXY",
    "factor:copper": "Copper",
    "factor:wheat": "Wheat",
    "factor:crude_oil": "Crude Oil",
    "factor:us2y": "US 2Y",
    "factor:us10y": "US 10Y",
    "factor:aaa_oas": "AAA OAS",
    "factor:bbb_oas": "BBB OAS",
    "factor:nikkei": "Nikkei",
    "factor:gold": "Gold",
    "factor:vix": "VIX",
}

ANCHOR_PREFERRED = [
    "factor:spx",
    "factor:vix",
    "factor:us10y",
    "factor:us2y",
    "factor:crude_oil",
    "factor:gold",
    "factor:aaa_oas",
    "factor:bbb_oas",
]

GT_COLOR = "#2E7D32"
GEN_COLOR = "#C62828"
CALM_COLOR = "#1976D2"
TURB_COLOR = "#D32F2F"
NEUTRAL_COLOR = "#455A64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--iv_checkpoint", required=True)
    parser.add_argument("--anchor_checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=734)
    parser.add_argument("--skip_iv", action="store_true")
    parser.add_argument("--skip_anchor", action="store_true")
    return parser.parse_args()


def default_eval_args(checkpoint: str, scope: str, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=checkpoint,
        data_path="data/vol_surface_with_ret.npz",
        state_scope=scope,
        eval_split="val",
        test_start=4511,
        val_size=441,
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
        max_windows=int(args.max_windows),
        samples=int(args.n_samples),
        n_steps=30,
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        sample_temperature=1.0,
        conditionality_samples=32,
        conditionality_max_batches=8,
        seed=int(args.seed),
        device=args.device,
        output_json="",
        output_md="",
    )


def _raw_scope(block: Any, scope: str, iv_count: int) -> tuple[np.ndarray, np.ndarray]:
    if scope == "iv_only":
        return block.history_state[..., :iv_count], block.future_state[..., :iv_count]
    if scope == "anchor_only":
        return block.history_state[..., iv_count:], block.future_state[..., iv_count:]
    raise ValueError(f"unsupported report scope {scope!r}")


@torch.no_grad()
def generate_scope_data(
    checkpoint: str,
    scope: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    eval_args = default_eval_args(checkpoint, scope, args)
    model, payload = load_model(checkpoint, device)
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        block,
    ) = build_val_block(eval_args, payload)

    n_windows = min(int(args.max_windows), int(history_level.shape[0]))
    history_level = history_level[:n_windows]
    history_norm = history_norm[:n_windows]
    center = center[:n_windows]
    scale = scale[:n_windows]
    drift_feature = drift_feature[:n_windows]
    history_raw = history_raw[:n_windows]

    samples: list[np.ndarray] = []
    print(
        f"Generating {scope} data from {checkpoint} "
        f"({n_windows} windows, {args.n_samples} samples)",
        flush=True,
    )
    for start in range(0, n_windows, int(args.batch_size)):
        stop = min(start + int(args.batch_size), n_windows)
        sampled_increment = model.sample_batched(
            torch.from_numpy(history_level[start:stop]).to(device),
            torch.from_numpy(history_norm[start:stop]).to(device),
            torch.from_numpy(center[start:stop]).to(device),
            torch.from_numpy(scale[start:stop]).to(device),
            drift_feature=torch.from_numpy(drift_feature[start:stop]).to(device),
            n_samples=int(args.n_samples),
            n_steps=30,
            chunk_size=int(args.chunk_size),
            temperature=1.0,
        )
        panel = reconstruct_state_from_increments(
            history_raw[start:stop, -1, :],
            sampled_increment.detach().cpu().numpy(),
            specs,
        )
        samples.append(panel.astype(np.float32))
        print(f"  generated {scope} windows {stop}/{n_windows}", flush=True)

    history_scope, future_scope = _raw_scope(block, scope, int(eval_args.iv_count))
    history_scope = history_scope[:n_windows].astype(np.float32)
    future_scope = future_scope[:n_windows].astype(np.float32)
    sample_arr = np.concatenate(samples, axis=0).astype(np.float32)
    spec_names = [spec.name for spec in specs]
    if scope == "iv_only":
        future_increment_scope = block.future_increment[:n_windows, :, : int(eval_args.iv_count)]
    else:
        future_increment_scope = block.future_increment[:n_windows, :, int(eval_args.iv_count) :]
    oracle_future = reconstruct_state_from_increments(
        history_raw[:n_windows, -1, :],
        future_increment_scope,
        specs,
    )
    reconstruction_error = {
        "max_abs_error": float(np.max(np.abs(oracle_future - future_scope))),
        "mean_abs_error": float(np.mean(np.abs(oracle_future - future_scope))),
    }
    print(
        f"  oracle reconstruction max_abs={reconstruction_error['max_abs_error']:.3e} "
        f"mean_abs={reconstruction_error['mean_abs_error']:.3e}",
        flush=True,
    )

    if scope == "iv_only":
        history_plot = history_scope.reshape(n_windows, 30, 5, 5)
        future_plot = future_scope.reshape(n_windows, 30, 5, 5)
        sample_plot = sample_arr.reshape(n_windows, int(args.n_samples), 30, 5, 5)
        mean_iv = history_plot.mean(axis=(-1, -2))
        vol_of_vol = np.diff(mean_iv, axis=1).std(axis=1)
        return {
            "scope": scope,
            "checkpoint": checkpoint,
            "payload": payload,
            "history": history_plot,
            "future": future_plot,
            "samples": sample_plot,
            "samples_raw": sample_plot,
            "vol_of_vol": vol_of_vol,
            "n_windows": n_windows,
            "spec_names": spec_names,
            "reconstruction_error": reconstruction_error,
        }

    history_activity = np.diff(history_scope, axis=1).std(axis=1).mean(axis=1)
    return {
        "scope": scope,
        "checkpoint": checkpoint,
        "payload": payload,
        "history": history_scope,
        "future": future_scope,
        "samples": sample_arr,
        "samples_raw": sample_arr,
        "activity": history_activity,
        "n_windows": n_windows,
        "spec_names": spec_names,
        "reconstruction_error": reconstruction_error,
    }


def _future_deltas(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    prev = np.concatenate([history[:, -1:, :], future[:, :-1, :]], axis=1)
    return future - prev


def _sample_deltas(history: np.ndarray, samples: np.ndarray) -> np.ndarray:
    prev = np.concatenate(
        [
            np.repeat(history[:, None, -1:, :], samples.shape[1], axis=1),
            samples[:, :, :-1, :],
        ],
        axis=2,
    )
    return samples - prev


def _factor_label(name: str) -> str:
    return FACTOR_DISPLAY.get(name, name.replace("factor:", ""))


def _factor_index(names: list[str], factor: str) -> int | None:
    return names.index(factor) if factor in names else None


def plot_iv_temporal_properties_paper(data: dict[str, Any], output_dir: str) -> None:
    gt_delta = _iv_future_deltas(data["history"], data["future"])
    gen_delta = _iv_sample_deltas(data["history"], data["samples"])
    horizons = np.arange(1, 31)

    gt_abs_med = np.median(np.abs(gt_delta), axis=(0, 2, 3))
    gen_abs_med = np.median(np.abs(gen_delta), axis=(0, 1, 3, 4))
    gt_tail = np.percentile(np.abs(gt_delta), 99, axis=(0, 2, 3))
    gen_tail = np.percentile(np.abs(gen_delta), 99, axis=(0, 1, 3, 4))

    def mean_acf(arr: np.ndarray, max_lag: int = 12) -> np.ndarray:
        flat = np.abs(arr).reshape(arr.shape[0], arr.shape[1], -1)
        acfs = []
        for lag in range(1, max_lag + 1):
            vals = []
            for w in range(flat.shape[0]):
                for c in range(flat.shape[2]):
                    x = flat[w, :, c]
                    if len(x) <= lag or np.std(x[:-lag]) < 1e-12 or np.std(x[lag:]) < 1e-12:
                        continue
                    vals.append(np.corrcoef(x[:-lag], x[lag:])[0, 1])
            acfs.append(np.nanmean(vals) if vals else 0.0)
        return np.asarray(acfs)

    gt_acf = mean_acf(gt_delta)
    gen_acf = mean_acf(gen_delta[:, 0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Temporal IV Realism: Scale, Clustering, and Horizon-Level Behavior",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(horizons, gt_abs_med, color=GT_COLOR, linewidth=2, label="Ground truth")
    ax.plot(horizons, gen_abs_med, color=GEN_COLOR, linewidth=2, label="Generated")
    ax.set_title("Median absolute daily IV change by horizon")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Median |daily change|")
    ax.legend()

    ax = axes[0, 1]
    lags = np.arange(1, len(gt_acf) + 1)
    ax.plot(lags, gt_acf, color=GT_COLOR, marker="o", linewidth=2, label="Ground truth")
    ax.plot(lags, gen_acf, color=GEN_COLOR, marker="s", linewidth=2, label="Generated")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Volatility clustering: ACF of |daily IV changes|")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Mean intra-window ACF")
    ax.legend()

    ax = axes[1, 0]
    for h, alpha in [(0, 0.35), (13, 0.25), (29, 0.18)]:
        gt_vals = data["future"][:, h].reshape(-1)
        gen_vals = data["samples"][:, :, h].reshape(-1)
        lo = min(np.percentile(gt_vals, 0.5), np.percentile(gen_vals, 0.5))
        hi = max(np.percentile(gt_vals, 99.5), np.percentile(gen_vals, 99.5))
        bins = np.linspace(lo, hi, 60)
        ax.hist(gt_vals, bins=bins, density=True, histtype="step", color=GT_COLOR, linewidth=1.4, alpha=alpha + 0.45, label=f"GT h{h+1}")
        ax.hist(gen_vals, bins=bins, density=True, histtype="step", color=GEN_COLOR, linewidth=1.4, alpha=alpha + 0.45, label=f"Gen h{h+1}")
    ax.set_title("Level distribution at h1 / h14 / h30")
    ax.set_xlabel("Implied volatility")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    ratio = gen_tail / np.maximum(gt_tail, 1e-8)
    ax.plot(horizons, ratio, color=NEUTRAL_COLOR, marker="o", linewidth=2)
    ax.axhspan(0.5, 2.0, color="#E8F5E9", alpha=0.6, label="risk-monitor band [0.5, 2.0]")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Tail-move scale ratio: generated / ground truth")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Q99 |daily change| ratio")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = Path(output_dir) / "fig3_temporal_properties.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _iv_future_deltas(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    prev = np.concatenate([history[:, -1:, :, :], future[:, :-1, :, :]], axis=1)
    return future - prev


def _iv_sample_deltas(history: np.ndarray, samples: np.ndarray) -> np.ndarray:
    prev = np.concatenate(
        [
            np.repeat(history[:, None, -1:, :, :], samples.shape[1], axis=1),
            samples[:, :, :-1, :, :],
        ],
        axis=2,
    )
    return samples - prev


def plot_iv_marginal_daily_changes_paper(data: dict[str, Any], output_dir: str) -> None:
    """Boundary-inclusive daily IV P&L distributions."""
    gt_daily = _iv_future_deltas(data["history"], data["future"])
    gen_daily = _iv_sample_deltas(data["history"], data["samples"][:, : min(5, data["samples"].shape[1])])
    cells_9 = [
        (0, 0), (0, 2), (0, 4),
        (2, 0), (2, 2), (2, 4),
        (4, 0), (4, 2), (4, 4),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        "Daily IV P&L Distribution: Ground Truth vs Generated\n"
        "Includes h1 move from the last observed history level.",
        fontsize=14,
        fontweight="bold",
    )
    for idx, (r, c) in enumerate(cells_9):
        ax = axes[idx // 3, idx % 3]
        gt_vals = gt_daily[:, :, r, c].ravel()
        gen_vals = gen_daily[:, :, :, r, c].ravel()
        span = max(np.percentile(np.abs(gt_vals), 99.5), np.percentile(np.abs(gen_vals), 99.5), 1e-8)
        bins = np.linspace(-span, span, 80)
        ax.hist(gt_vals, bins=bins, density=True, alpha=0.5, color=v1.GT_COLOR, label="GT")
        ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, color=v1.TURB_COLOR, label="Gen")
        gt_k = sp_stats.kurtosis(gt_vals, fisher=True)
        gen_k = sp_stats.kurtosis(gen_vals, fisher=True)
        ratio = gen_k / gt_k if gt_k > 0 else float("nan")
        ks_stat, ks_p = sp_stats.ks_2samp(gt_vals, gen_vals)
        ax.text(
            0.03,
            0.95,
            f"GT kurt={gt_k:.1f}\nGen kurt={gen_k:.1f}\nRatio={ratio:.2f}\nKS={ks_stat:.3f}",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
        ax.set_title(f"({r},{c}) {v1.CELL_NAMES[(r, c)]}", fontsize=10)
        if idx >= 6:
            ax.set_xlabel("Daily IV P&L")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        if idx == 0:
            ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = Path(output_dir) / "fig7a_marginal_daily_changes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    fig.suptitle(
        "Daily IV P&L Distribution: All 25 Cells\n"
        "Includes h1 move from history into generated/realized future.",
        fontsize=14,
        fontweight="bold",
    )
    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            gt_vals = gt_daily[:, :, r, c].ravel()
            gen_vals = gen_daily[:, :, :, r, c].ravel()
            span = max(np.percentile(np.abs(gt_vals), 99), np.percentile(np.abs(gen_vals), 99), 1e-8)
            bins = np.linspace(-span, span, 50)
            ax.hist(gt_vals, bins=bins, density=True, alpha=0.5, color=v1.GT_COLOR)
            ax.hist(gen_vals, bins=bins, density=True, alpha=0.5, color=v1.TURB_COLOR)
            gt_k = sp_stats.kurtosis(gt_vals, fisher=True)
            gen_k = sp_stats.kurtosis(gen_vals, fisher=True)
            ratio = gen_k / gt_k if gt_k > 0 else float("nan")
            ks_stat = sp_stats.ks_2samp(gt_vals, gen_vals).statistic
            ax.set_title(f"({r},{c}) r={ratio:.2f} KS={ks_stat:.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            if r < 4:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])
    for r in range(5):
        axes[r, 0].set_ylabel(v1.MATURITY_LABELS[r], fontsize=9)
    for c in range(5):
        axes[4, c].set_xlabel(f"K={v1.MONEYNESS_LABELS[c]}", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = Path(output_dir) / "fig7b_marginal_daily_changes_all25.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def write_iv_report(data: dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    v1.OUTPUT_DIR = output_dir
    v1.N_SAMPLES = int(data["samples"].shape[1])
    v1.MAX_WINDOWS = int(data["n_windows"])
    print(f"Writing IV report to {output_dir}")
    v1.plot_fan_charts(data)
    v1.plot_cross_cell_sensitivity(data)
    plot_iv_temporal_properties_paper(data, output_dir)
    v1.plot_surface_structure(data)
    v1.plot_surface_heatmaps(data)
    plot_iv_marginal_daily_changes_paper(data, output_dir)
    v1.plot_kurtosis_heatmap(data)
    plot_mean_reversion_clustering(data, output_dir, summary_path=None)
    plot_burst_reversion_paths(data, output_dir, summary_path=None)


def plot_anchor_fan_charts(data: dict[str, Any], output_dir: str) -> None:
    names = data["spec_names"]
    selected = [name for name in ANCHOR_PREFERRED if name in names][:6]
    if not selected:
        selected = names[: min(6, len(names))]
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 2.7 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = np.asarray([axes])
    fig.suptitle(
        "Anchor Factor Scenario Fan Charts: Factor-Specific Calm vs Turbulent Histories",
        fontsize=15,
        fontweight="bold",
    )

    hist_days = np.arange(-29, 1)
    fut_days = np.arange(1, 31)
    for row, name in enumerate(selected):
        c = names.index(name)
        factor_activity = np.diff(data["history"][:, :, c], axis=1).std(axis=1)
        sorted_idx = np.argsort(factor_activity)
        calm_idx = int(sorted_idx[int(0.10 * len(sorted_idx))])
        turb_idx = int(sorted_idx[int(0.90 * len(sorted_idx))])
        picks = [
            (calm_idx, "Factor-calm history", CALM_COLOR, factor_activity[calm_idx]),
            (turb_idx, "Factor-turbulent history", TURB_COLOR, factor_activity[turb_idx]),
        ]
        for col, (idx, title, color, activity_value) in enumerate(picks):
            ax = axes[row, col]
            hist = data["history"][idx, :, c]
            fut = data["future"][idx, :, c]
            samples = data["samples"][idx, :, :, c]
            q05, q25, q50, q75, q95 = np.percentile(samples, [5, 25, 50, 75, 95], axis=0)
            coverage = float(np.mean((fut >= q05) & (fut <= q95)))
            ax.plot(hist_days, hist, color="black", linewidth=1.5, label="History")
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
            ax.fill_between(fut_days, q05, q95, color=color, alpha=0.14, label="Generated 90% band")
            ax.fill_between(fut_days, q25, q75, color=color, alpha=0.22, label="Generated IQR")
            ax.plot(fut_days, q50, color=color, linewidth=1.8, label="Generated median")
            ax.plot(fut_days, fut, color=GT_COLOR, linestyle="--", linewidth=2.0, label="Ground truth")
            if row == 0:
                ax.set_title(title, fontweight="bold")
            if col == 0:
                ax.set_ylabel(_factor_label(name))
            if row == len(selected) - 1:
                ax.set_xlabel("Day")
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="best")
            ax.text(
                0.02,
                0.94,
                f"hist σ={activity_value:.3g}\n90% cov={coverage:.0%}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = Path(output_dir) / "figA1_anchor_fan_charts.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_anchor_marginal_changes(data: dict[str, Any], output_dir: str) -> None:
    names = data["spec_names"]
    gt_delta = _future_deltas(data["history"], data["future"])
    gen_delta = _sample_deltas(data["history"], data["samples"])
    n = len(names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.3 * nrows))
    axes = axes.reshape(nrows, ncols)
    fig.suptitle("Anchor Factor Marginal Daily Changes: Ground Truth vs Generated", fontsize=15, fontweight="bold")

    for i, name in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        gt = gt_delta[:, :, i].reshape(-1)
        gen = gen_delta[:, :, :, i].reshape(-1)
        span = max(np.percentile(np.abs(gt), 99), np.percentile(np.abs(gen), 99), 1e-8)
        bins = np.linspace(-span, span, 60)
        ax.hist(gt, bins=bins, density=True, alpha=0.45, color=GT_COLOR, label="GT" if i == 0 else None)
        ax.hist(gen, bins=bins, density=True, alpha=0.45, color=GEN_COLOR, label="Gen" if i == 0 else None)
        ks = sp_stats.ks_2samp(gt, gen).statistic
        q99_ratio = np.percentile(np.abs(gen), 99) / max(np.percentile(np.abs(gt), 99), 1e-8)
        zero_gt = np.mean(np.abs(gt) < 1e-10)
        zero_gen = np.mean(np.abs(gen) < 1e-10)
        ax.text(
            0.97,
            0.95,
            f"KS={ks:.3f}\nQ99x={q99_ratio:.2f}\nzero {zero_gt:.0%}/{zero_gen:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
        )
        ax.set_title(_factor_label(name), fontsize=10, fontweight="bold")
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = Path(output_dir) / "figA2_anchor_marginal_daily_changes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_anchor_factor_correlation(data: dict[str, Any], output_dir: str) -> None:
    names = data["spec_names"]
    gt_delta = _future_deltas(data["history"], data["future"]).reshape(-1, len(names))
    gen_delta = _sample_deltas(data["history"], data["samples"]).reshape(-1, len(names))
    gt_corr = np.corrcoef(gt_delta.T)
    gen_corr = np.corrcoef(gen_delta.T)
    diff = gen_corr - gt_corr

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("Anchor Factor Co-Movement: Daily-Change Correlation", fontsize=15, fontweight="bold")
    labels = [_factor_label(n) for n in names]
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], gt_corr, "Ground truth", "coolwarm", -1, 1),
        (axes[1], gen_corr, "Generated", "coolwarm", -1, 1),
        (axes[2], diff, "Generated - Ground truth", "RdBu_r", -0.5, 0.5),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    upper = np.triu_indices(len(names), k=1)
    corr_of_corr = np.corrcoef(gt_corr[upper], gen_corr[upper])[0, 1]
    axes[2].text(
        0.04,
        0.95,
        f"corr-of-corr={corr_of_corr:.3f}",
        transform=axes[2].transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = Path(output_dir) / "figA3_anchor_factor_correlation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_anchor_vix_spx_cases(data: dict[str, Any], output_dir: str) -> None:
    names = data["spec_names"]
    vix_idx = _factor_index(names, "factor:vix")
    spx_idx = _factor_index(names, "factor:spx")
    selected = [idx for idx in [vix_idx, spx_idx] if idx is not None]
    if not selected:
        return
    if vix_idx is not None:
        vix_hist_last = data["history"][:, -1, vix_idx]
        win_idx = int(np.argsort(vix_hist_last)[int(0.90 * len(vix_hist_last))])
    else:
        win_idx = int(np.argsort(data["activity"])[int(0.90 * len(data["activity"]))])

    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 4.2 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    fig.suptitle("Anchor Case Study: Volatility and Equity Paths Under a Stressed History", fontsize=15, fontweight="bold")
    hist_days = np.arange(-29, 1)
    fut_days = np.arange(1, 31)
    for ax, c in zip(axes, selected):
        hist = data["history"][win_idx, :, c]
        fut = data["future"][win_idx, :, c]
        samples = data["samples"][win_idx, :, :, c]
        q05, q25, q50, q75, q95 = np.percentile(samples, [5, 25, 50, 75, 95], axis=0)
        ax.plot(hist_days, hist, color="black", linewidth=1.6, label="History")
        ax.fill_between(fut_days, q05, q95, color=GEN_COLOR, alpha=0.14, label="Generated 90% band")
        ax.fill_between(fut_days, q25, q75, color=GEN_COLOR, alpha=0.22, label="Generated IQR")
        ax.plot(fut_days, q50, color=GEN_COLOR, linewidth=1.8, label="Generated median")
        ax.plot(fut_days, fut, color=GT_COLOR, linestyle="--", linewidth=2.2, label="Ground truth")
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(_factor_label(names[c]), fontweight="bold")
        ax.set_ylabel("Level")
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Day")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = Path(output_dir) / "figA4_anchor_vix_spx_case.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_anchor_rates_credit_diagnostics(data: dict[str, Any], output_dir: str) -> None:
    names = data["spec_names"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Anchor Sanity Checks: Rates Curve and Sticky Credit Channels", fontsize=15, fontweight="bold")

    ax = axes[0]
    us2y = _factor_index(names, "factor:us2y")
    us10y = _factor_index(names, "factor:us10y")
    if us2y is not None and us10y is not None:
        idx = int(np.argsort(data["activity"])[int(0.75 * len(data["activity"]))])
        tenors = np.asarray([2, 10])
        gt_h1 = data["future"][idx, 0, [us2y, us10y]]
        gt_h30 = data["future"][idx, -1, [us2y, us10y]]
        gen_h1 = data["samples"][idx, :, 0, :][:, [us2y, us10y]]
        gen_h30 = data["samples"][idx, :, -1, :][:, [us2y, us10y]]
        for arr, color, label, ls in [
            (gt_h1, GT_COLOR, "GT h1", "-"),
            (gt_h30, GT_COLOR, "GT h30", "--"),
            (np.median(gen_h1, axis=0), GEN_COLOR, "Gen median h1", "-"),
            (np.median(gen_h30, axis=0), GEN_COLOR, "Gen median h30", "--"),
        ]:
            ax.plot(tenors, arr, color=color, linestyle=ls, marker="o", linewidth=2, label=label)
        lo = np.percentile(gen_h30, 5, axis=0)
        hi = np.percentile(gen_h30, 95, axis=0)
        ax.fill_between(tenors, lo, hi, color=GEN_COLOR, alpha=0.15, label="Gen h30 90% band")
        ax.set_xticks(tenors)
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel("Rate level")
        ax.set_title("US rates term structure scenario")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "US2Y/US10Y not available", ha="center", va="center")
        ax.axis("off")

    ax = axes[1]
    gt_delta = _future_deltas(data["history"], data["future"])
    gen_delta = _sample_deltas(data["history"], data["samples"])
    credit = [n for n in ["factor:aaa_oas", "factor:bbb_oas"] if n in names]
    x = np.arange(len(credit))
    width = 0.35
    if credit:
        gt_zero = []
        gen_zero = []
        gt_q99 = []
        gen_q99 = []
        for name in credit:
            c = names.index(name)
            gt = gt_delta[:, :, c].reshape(-1)
            gen = gen_delta[:, :, :, c].reshape(-1)
            gt_zero.append(np.mean(np.abs(gt) < 1e-10))
            gen_zero.append(np.mean(np.abs(gen) < 1e-10))
            gt_q99.append(np.percentile(np.abs(gt), 99))
            gen_q99.append(np.percentile(np.abs(gen), 99))
        ax.bar(x - width / 2, gt_zero, width, color=GT_COLOR, alpha=0.8, label="GT zero-update rate")
        ax.bar(x + width / 2, gen_zero, width, color=GEN_COLOR, alpha=0.8, label="Gen zero-update rate")
        for i, (gq, sq) in enumerate(zip(gt_q99, gen_q99)):
            ratio = sq / max(gq, 1e-8)
            ax.text(i, max(gt_zero[i], gen_zero[i]) + 0.03, f"Q99x={ratio:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([_factor_label(n) for n in credit])
        ax.set_ylim(0, min(1.05, max(max(gt_zero), max(gen_zero)) + 0.20))
        ax.set_ylabel("Fraction of near-zero daily changes")
        ax.set_title("Credit spread stickiness diagnostic")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "AAA/BBB OAS not available", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = Path(output_dir) / "figA5_anchor_rates_credit_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def write_anchor_report(data: dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing anchor report to {output_dir}")
    plot_anchor_fan_charts(data, output_dir)
    plot_anchor_marginal_changes(data, output_dir)
    plot_anchor_factor_correlation(data, output_dir)
    plot_anchor_vix_spx_cases(data, output_dir)
    plot_anchor_rates_credit_diagnostics(data, output_dir)


def write_manifest(output_root: str, args: argparse.Namespace, iv_data: dict[str, Any] | None, anchor_data: dict[str, Any] | None) -> None:
    manifest = {
        "model_label": args.model_label,
        "iv_checkpoint": args.iv_checkpoint,
        "anchor_checkpoint": args.anchor_checkpoint,
        "n_samples": int(args.n_samples),
        "max_windows": int(args.max_windows),
        "seed": int(args.seed),
        "iv": None if iv_data is None else {
            "n_windows": int(iv_data["n_windows"]),
            "checkpoint_epoch": int(iv_data["payload"].get("epoch", -1)),
            "checkpoint_best_val": float(iv_data["payload"].get("best_val", float("nan"))),
            "reconstruction_error": iv_data["reconstruction_error"],
        },
        "anchor": None if anchor_data is None else {
            "n_windows": int(anchor_data["n_windows"]),
            "checkpoint_epoch": int(anchor_data["payload"].get("epoch", -1)),
            "checkpoint_best_val": float(anchor_data["payload"].get("best_val", float("nan"))),
            "factors": anchor_data["spec_names"],
            "reconstruction_error": anchor_data["reconstruction_error"],
        },
    }
    path = Path(output_root) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Saved: {path}")


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    iv_data = None
    anchor_data = None
    if not args.skip_iv:
        iv_data = generate_scope_data(args.iv_checkpoint, "iv_only", args)
        write_iv_report(iv_data, str(output_root / "iv_only"))
    if not args.skip_anchor:
        anchor_data = generate_scope_data(args.anchor_checkpoint, "anchor_only", args)
        write_anchor_report(anchor_data, str(output_root / "anchor_only"))
    write_manifest(str(output_root), args, iv_data, anchor_data)
    print(f"\nReport complete: {output_root}")


if __name__ == "__main__":
    main()
