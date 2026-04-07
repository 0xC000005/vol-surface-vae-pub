#!/usr/bin/env python
"""
V1-style management report for the 164a AR spatial-transformer baseline.

Purpose:
  - reuse the original V1 visual language that is easier to interpret
  - support the 164a-family AR transformer checkpoints that use `sample_batched()`
  - include the mean-reversion / burst diagnostics added for later reports
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_164a_v3_percell_bptt_softplus import (
    ARSpatialTransformerModel,
    normalize_iv,
)
import experiments.backfill.block_ar.visualize_management_report as v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1-style management report for 164a-family AR models")
    parser.add_argument(
        "--model_path",
        default="models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt",
    )
    parser.add_argument(
        "--output_dir",
        default="results/block_ar/management_report_164a_v1",
    )
    parser.add_argument(
        "--summary_path",
        default="results/block_ar/164a_v2_s3mr_full_30d/summary.json",
        help="Optional benchmark summary for selecting a representative mean-reverting cell.",
    )
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_windows", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    model_type = raw_config.get("type", "")

    if not model_type.startswith("ar_spatial_transformer_164a"):
        raise ValueError(f"Unsupported model type for this report script: {model_type}")

    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = ARSpatialTransformerModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, raw_config


def generate_all_data(model, config: dict, device: str, n_samples: int, max_windows: int, batch_size: int):
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    test_surfaces = surfaces[4540:]

    history_len = 30
    future_len = int(config.get("n_frames", 30))
    total_len = history_len + future_len
    n_total = len(test_surfaces)
    n_windows = min(max_windows, n_total - total_len + 1)
    indices = np.linspace(0, n_total - total_len, n_windows, dtype=int)

    history_arr = np.stack([test_surfaces[idx:idx + history_len] for idx in indices])
    future_arr = np.stack([test_surfaces[idx + history_len:idx + total_len] for idx in indices])

    mean_iv = history_arr.mean(axis=(-1, -2))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)

    print(f"Generating {n_samples} samples for {n_windows} windows...")
    all_samples = []
    for start in tqdm(range(0, n_windows, batch_size), desc="Sampling"):
        stop = min(start + batch_size, n_windows)
        hist = torch.tensor(history_arr[start:stop], dtype=torch.float32, device=device)
        with torch.no_grad():
            hist_norm = normalize_iv(hist)
            samples = model.sample_batched(hist_norm, n_samples=n_samples)
        all_samples.append(samples.cpu().numpy())
    all_samples = np.concatenate(all_samples, axis=0)

    return {
        "history": history_arr,
        "future": future_arr,
        "samples": all_samples,
        "samples_raw": all_samples,
        "vol_of_vol": vol_of_vol,
        "n_windows": n_windows,
    }


def choose_mean_reverting_cell(data: dict, summary_path: str | None) -> tuple[int, int]:
    if summary_path and Path(summary_path).exists():
        import json

        with open(summary_path) as f:
            summary = json.load(f)
        mr = summary["mean_reversion"]
        active = []
        for r in range(5):
            for c in range(5):
                if mr["active_cell_mask"][r][c]:
                    active.append(((r, c), mr["gt_cell_slopes"][r][c]))
        if active:
            active.sort(key=lambda item: abs(item[1]), reverse=True)
            return active[0][0]

    hist_last = data["history"][:, -1]
    fut_first = data["future"][:, 0]
    best_cell = (2, 4)
    best_slope = 0.0
    for r in range(5):
        for c in range(5):
            x = hist_last[:, r, c]
            y = fut_first[:, r, c] - x
            slope = np.polyfit(x, y, deg=1)[0]
            if slope < best_slope:
                best_slope = slope
                best_cell = (r, c)
    return best_cell


def choose_reverting_window(data: dict, cell: tuple[int, int]) -> int:
    r, c = cell
    hist = data["history"][:, :, r, c]
    fut = data["future"][:, :, r, c]
    vov = data["vol_of_vol"]

    long_run_levels = np.concatenate([hist.reshape(-1), fut.reshape(-1)])
    long_run_median = np.median(long_run_levels)
    long_run_iqr = np.subtract(*np.percentile(long_run_levels, [75, 25]))
    iqr = max(long_run_iqr, 1e-6)

    peak = fut.max(axis=1)
    peak_day = fut.argmax(axis=1)
    end = fut[:, -1]
    start = hist[:, -1]

    score = (
        (peak - long_run_median) / iqr
        + 1.5 * (peak - end) / iqr
        + 0.5 * (peak - start) / iqr
        + 0.5 * (1.0 - peak_day / max(fut.shape[1] - 1, 1))
        + 0.5 * (vov - vov.mean()) / max(vov.std(), 1e-6)
    )
    good = (peak_day >= 1) & (peak_day <= fut.shape[1] - 4) & (peak > end + 0.5 * iqr)
    if np.any(good):
        return int(np.argmax(np.where(good, score, -1e9)))
    return int(np.argmax(score))


def plot_mean_reversion_clustering(data: dict, output_dir: str, summary_path: str | None):
    cell = choose_mean_reverting_cell(data, summary_path)
    r, c = cell
    win_idx = choose_reverting_window(data, cell)

    history = data["history"][win_idx, :, r, c]
    future = data["future"][win_idx, :, r, c]
    samples = data["samples"][win_idx, :, :, r, c]
    vov = data["vol_of_vol"][win_idx]

    all_levels = np.concatenate([data["history"][:, :, r, c].reshape(-1), data["future"][:, :, r, c].reshape(-1)])
    normal_lo, normal_med, normal_hi = np.percentile(all_levels, [25, 50, 75])

    hist_days = np.arange(-len(history) + 1, 1)
    fwd_days = np.arange(1, len(future) + 1)
    full_gt = np.concatenate([history, future], axis=0)
    gt_abs_delta = np.abs(np.diff(full_gt))

    samples_full = np.concatenate(
        [np.repeat(history[None, :], samples.shape[0], axis=0), samples],
        axis=1,
    )
    sample_abs_delta = np.abs(np.diff(samples_full, axis=1))
    q05 = np.percentile(samples, 5, axis=0)
    q50 = np.percentile(samples, 50, axis=0)
    q95 = np.percentile(samples, 95, axis=0)
    delta_q05 = np.percentile(sample_abs_delta, 5, axis=0)
    delta_q25 = np.percentile(sample_abs_delta, 25, axis=0)
    delta_q75 = np.percentile(sample_abs_delta, 75, axis=0)
    delta_q95 = np.percentile(sample_abs_delta, 95, axis=0)
    delta_med = np.percentile(sample_abs_delta, 50, axis=0)

    plt.rcParams.update({"figure.dpi": 150})
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"Mean Reversion Example: {v1.CELL_NAMES[cell]}  |  window vol-of-vol={vov:.4f}\n"
        "Top: level path with return to normal band. Bottom: absolute daily changes show clustering.",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    ax.axhspan(normal_lo, normal_hi, color="#E8F5E9", alpha=0.5, label="Long-run normal band (25-75%)")
    ax.axhline(normal_med, color="#66BB6A", linestyle="--", linewidth=1.2, alpha=0.9, label="Long-run median")
    ax.plot(hist_days, history, color="black", linewidth=1.6, label="History", zorder=6)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    for s in np.linspace(0, samples.shape[0] - 1, min(12, samples.shape[0]), dtype=int):
        ax.plot(fwd_days, samples[s], color=v1.TURB_COLOR, alpha=0.10, linewidth=0.8)
    ax.fill_between(fwd_days, q05, q95, color=v1.TURB_COLOR, alpha=0.10, label="Generated 90% band")
    ax.plot(fwd_days, q50, color=v1.TURB_COLOR, linewidth=1.8, label="Generated median", zorder=4)
    ax.plot(fwd_days, future, color=v1.GT_COLOR, linewidth=2.4, linestyle="--", label="Ground truth future", zorder=7)
    peak_day = int(np.argmax(future)) + 1
    peak_val = float(np.max(future))
    end_val = float(future[-1])
    ax.scatter([peak_day], [peak_val], color=v1.GT_COLOR, s=50, zorder=8)
    ax.text(
        0.02,
        0.95,
        f"peak day={peak_day}, peak={peak_val:.3f}, end={end_val:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
    )
    ax.set_title("Cell level path: spike / elevated regime followed by reversion toward normal")
    ax.set_xlabel("Day (0 = forecast start)")
    ax.set_ylabel("Implied Volatility")
    ax.legend(fontsize=8, ncol=2, loc="best")

    ax = axes[1]
    delta_days = np.arange(-len(history) + 2, len(future) + 1)
    ax.plot(delta_days, gt_abs_delta, color="black", linewidth=1.5, label="GT |daily change|")
    ax.fill_between(delta_days, delta_q05, delta_q95, color="#FFE0B2", alpha=0.30, label="Generated 90% band |daily change|")
    ax.fill_between(delta_days, delta_q25, delta_q75, color="#FFCC80", alpha=0.45, label="Generated IQR |daily change|")
    ax.plot(delta_days, delta_med, color="#EF6C00", linewidth=1.6, label="Generated median |daily change|")
    quiet_ref = np.median(np.abs(np.diff(np.concatenate([data["history"][:, :, r, c], data["future"][:, :, r, c]], axis=1), axis=1)))
    ax.axhline(quiet_ref, color="#42A5F5", linestyle="--", linewidth=1.2, label="Long-run median |daily change|")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Absolute daily changes: 90% band, IQR, and median against GT burst clustering")
    ax.set_xlabel("Day")
    ax.set_ylabel("|Daily IV change|")
    ax.legend(fontsize=8, ncol=2, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{output_dir}/fig9_mean_reversion_clustering.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_extreme_generated_paths(data: dict, output_dir: str, summary_path: str | None):
    cell = choose_mean_reverting_cell(data, summary_path)
    r, c = cell
    win_idx = choose_reverting_window(data, cell)

    history = data["history"][win_idx, :, r, c]
    future = data["future"][win_idx, :, r, c]
    samples = data["samples"][win_idx, :, :, r, c]
    vov = data["vol_of_vol"][win_idx]

    hist_days = np.arange(-len(history) + 1, 1)
    fwd_days = np.arange(1, len(future) + 1)
    full_gt = np.concatenate([history, future], axis=0)
    gt_abs_delta = np.abs(np.diff(full_gt))
    gt_max_abs_delta = float(gt_abs_delta.max())

    samples_full = np.concatenate(
        [np.repeat(history[None, :], samples.shape[0], axis=0), samples],
        axis=1,
    )
    sample_abs_delta = np.abs(np.diff(samples_full, axis=1))
    sample_max_abs_delta = sample_abs_delta.max(axis=1)
    top_idx = np.argsort(sample_max_abs_delta)[-5:][::-1]

    plt.rcParams.update({"figure.dpi": 150})
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"Most Extreme Generated Paths vs GT: {v1.CELL_NAMES[cell]}  |  window vol-of-vol={vov:.4f}\n"
        "Generated paths are ranked by maximum absolute daily change to expose missing burst intensity.",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    ax.plot(hist_days, history, color="black", linewidth=1.6, label="History", zorder=6)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.plot(fwd_days, future, color=v1.GT_COLOR, linewidth=2.6, linestyle="--", label="Ground truth future", zorder=7)
    for rank, idx in enumerate(top_idx, start=1):
        label = f"Generated path #{rank} (max |dIV|={sample_max_abs_delta[idx]:.3f})"
        ax.plot(
            fwd_days,
            samples[idx],
            linewidth=1.4,
            alpha=0.90,
            label=label,
        )
    ax.set_title("Top generated burst paths against the GT mean-reverting spike")
    ax.set_xlabel("Day (0 = forecast start)")
    ax.set_ylabel("Implied Volatility")
    ax.legend(fontsize=8, ncol=2, loc="best")

    ax = axes[1]
    delta_days = np.arange(-len(history) + 2, len(future) + 1)
    ax.plot(delta_days, gt_abs_delta, color=v1.GT_COLOR, linewidth=2.2, linestyle="--", label=f"GT |daily change| (max={gt_max_abs_delta:.3f})")
    for rank, idx in enumerate(top_idx, start=1):
        ax.plot(
            delta_days,
            sample_abs_delta[idx],
            linewidth=1.4,
            alpha=0.90,
            label=f"Generated #{rank} |daily change|",
        )
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Absolute daily changes: generated burst paths still undershoot the GT spike")
    ax.set_xlabel("Day")
    ax.set_ylabel("|Daily IV change|")
    ax.legend(fontsize=8, ncol=2, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{output_dir}/fig10_extreme_generated_paths.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_burst_reversion_paths(data: dict, output_dir: str, summary_path: str | None):
    cell = choose_mean_reverting_cell(data, summary_path)
    r, c = cell
    win_idx = choose_reverting_window(data, cell)

    history = data["history"][win_idx, :, r, c]
    future = data["future"][win_idx, :, r, c]
    samples = data["samples"][win_idx, :, :, r, c]
    vov = data["vol_of_vol"][win_idx]

    hist_days = np.arange(-len(history) + 1, 1)
    fwd_days = np.arange(1, len(future) + 1)
    full_gt = np.concatenate([history, future], axis=0)
    gt_abs_delta = np.abs(np.diff(full_gt))

    all_levels = np.concatenate([data["history"][:, :, r, c].reshape(-1), data["future"][:, :, r, c].reshape(-1)])
    normal_lo, normal_med, normal_hi = np.percentile(all_levels, [25, 50, 75])
    iqr = max(normal_hi - normal_lo, 1e-6)

    samples_full = np.concatenate(
        [np.repeat(history[None, :], samples.shape[0], axis=0), samples],
        axis=1,
    )
    sample_abs_delta = np.abs(np.diff(samples_full, axis=1))

    peak = samples.max(axis=1)
    peak_day = samples.argmax(axis=1)
    end = samples[:, -1]
    max_abs_delta = sample_abs_delta.max(axis=1)

    post_peak_noise = np.zeros(samples.shape[0], dtype=np.float32)
    for i in range(samples.shape[0]):
        tail = sample_abs_delta[i, peak_day[i]:]
        post_peak_noise[i] = float(tail.mean()) if tail.size else 0.0

    score = (
        1.25 * (peak - normal_med) / iqr
        + 1.50 * (peak - end) / iqr
        + 0.75 * np.clip(max_abs_delta / max(float(gt_abs_delta.max()), 1e-6), 0.0, 2.0)
        - 1.00 * np.abs(end - normal_med) / iqr
        - 0.75 * post_peak_noise / max(float(np.median(gt_abs_delta)), 1e-6)
    )
    top_idx = np.argsort(score)[-5:][::-1]

    plt.rcParams.update({"figure.dpi": 150})
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"Burst-Then-Revert Generated Paths vs GT: {v1.CELL_NAMES[cell]}  |  window vol-of-vol={vov:.4f}\n"
        "Generated paths are ranked by spike size plus reversion toward the normal band, not by raw noisiness.",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    ax.axhspan(normal_lo, normal_hi, color="#E8F5E9", alpha=0.5, label="Long-run normal band (25-75%)")
    ax.axhline(normal_med, color="#66BB6A", linestyle="--", linewidth=1.2, alpha=0.9, label="Long-run median")
    ax.plot(hist_days, history, color="black", linewidth=1.6, label="History", zorder=6)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.plot(fwd_days, future, color=v1.GT_COLOR, linewidth=2.6, linestyle="--", label="Ground truth future", zorder=7)
    for rank, idx in enumerate(top_idx, start=1):
        label = (
            f"Generated path #{rank} "
            f"(peak={peak[idx]:.3f}, end={end[idx]:.3f}, max |dIV|={max_abs_delta[idx]:.3f})"
        )
        ax.plot(fwd_days, samples[idx], linewidth=1.4, alpha=0.90, label=label)
    ax.set_title("Top generated paths ranked by burst + reversion quality")
    ax.set_xlabel("Day (0 = forecast start)")
    ax.set_ylabel("Implied Volatility")
    ax.legend(fontsize=8, ncol=2, loc="best")

    ax = axes[1]
    delta_days = np.arange(-len(history) + 2, len(future) + 1)
    ax.plot(delta_days, gt_abs_delta, color=v1.GT_COLOR, linewidth=2.2, linestyle="--", label=f"GT |daily change| (max={float(gt_abs_delta.max()):.3f})")
    for rank, idx in enumerate(top_idx, start=1):
        ax.plot(
            delta_days,
            sample_abs_delta[idx],
            linewidth=1.4,
            alpha=0.90,
            label=f"Generated #{rank} |daily change|",
        )
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Absolute daily changes for the best burst-and-revert generated paths")
    ax.set_xlabel("Day")
    ax.set_ylabel("|Daily IV change|")
    ax.legend(fontsize=8, ncol=2, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = f"{output_dir}/fig11_burst_reversion_generated_paths.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading 164a-family model...")
    model, config = load_model(args.model_path, device)

    print("Generating report data...")
    data = generate_all_data(
        model=model,
        config=config,
        device=device,
        n_samples=args.n_samples,
        max_windows=args.max_windows,
        batch_size=args.batch_size,
    )

    v1.OUTPUT_DIR = args.output_dir
    v1.MODEL_PATH = args.model_path
    v1.N_SAMPLES = args.n_samples
    v1.MAX_WINDOWS = args.max_windows

    print("\nGenerating V1-style figures...")
    print("[1/11] Fan charts with history...")
    v1.plot_fan_charts(data)
    print("[2/11] Cross-cell sensitivity...")
    v1.plot_cross_cell_sensitivity(data)
    print("[3/11] Temporal properties...")
    v1.plot_temporal_properties(data)
    print("[4/11] Term structure & smile...")
    v1.plot_surface_structure(data)
    print("[5/11] Surface heatmaps...")
    v1.plot_surface_heatmaps(data)
    print("[6/11] Calibration curve...")
    v1.plot_calibration_curve(data)
    print("[7/11] Marginal daily changes...")
    v1.plot_marginal_daily_changes(data)
    print("[8/11] Kurtosis heatmap...")
    v1.plot_kurtosis_heatmap(data)
    print("[9/11] Mean reversion / clustering example...")
    plot_mean_reversion_clustering(data, args.output_dir, args.summary_path)
    print("[10/11] Extreme generated paths vs GT...")
    plot_extreme_generated_paths(data, args.output_dir, args.summary_path)
    print("[11/11] Burst-and-reversion generated paths vs GT...")
    plot_burst_reversion_paths(data, args.output_dir, args.summary_path)

    print(f"\nAll figures saved to {args.output_dir}/")
    for fname in sorted(os.listdir(args.output_dir)):
        if fname.endswith(".png"):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
