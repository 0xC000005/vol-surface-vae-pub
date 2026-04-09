#!/usr/bin/env python
"""
Plot unconditional per-cell next-day delta marginals for deterministic H=1 models
against ground truth over the validation slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_212i_h1_deterministic_direct_delta_mse_cvar import load_model as load_212i_model
from experiments.backfill.block_ar.train_212j_h1_deterministic_direct_delta_lstm_mse_cvar import load_model as load_212j_model
from experiments.backfill.block_ar.train_212k_h1_deterministic_direct_delta_mse import load_model as load_212k_model
from experiments.backfill.block_ar.train_212l_h1_deterministic_direct_delta_raw_mse import load_model as load_212l_model
from experiments.backfill.block_ar.train_212m_h1_deterministic_direct_delta_raw_unscaled_mse import load_model as load_212m_model
from experiments.backfill.block_ar.train_212n_h1_deterministic_direct_delta_fully_raw_mse import load_model as load_212n_model


LOADERS = {
    "212i": load_212i_model,
    "212j": load_212j_model,
    "212k": load_212k_model,
    "212l": load_212l_model,
    "212m": load_212m_model,
    "212n": load_212n_model,
}


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def _ks(gt: np.ndarray, pred: np.ndarray) -> float:
    xs = np.sort(np.unique(np.concatenate([gt, pred])))
    if xs.size == 0:
        return 0.0
    gt_cdf = np.searchsorted(np.sort(gt), xs, side="right") / len(gt)
    pred_cdf = np.searchsorted(np.sort(pred), xs, side="right") / len(pred)
    return float(np.max(np.abs(gt_cdf - pred_cdf)))


def _plot_hist_grid(gt: np.ndarray, pred: np.ndarray, out_path: Path, clip_quantile: float, bins: int) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(16, 14))
    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            gt_cell = gt[:, r, c]
            pred_cell = pred[:, r, c]
            xlim = float(np.quantile(np.abs(np.concatenate([gt_cell, pred_cell])), clip_quantile))
            xlim = max(xlim, 1e-6)
            edges = np.linspace(-xlim, xlim, bins + 1)
            ax.hist(gt_cell, bins=edges, density=True, histtype="step", linewidth=1.2, color="#1f77b4", label="GT")
            ax.hist(pred_cell, bins=edges, density=True, histtype="step", linewidth=1.2, color="#d62728", label="Model")
            ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.6)
            ax.set_title(f"({r},{c})", fontsize=9)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Unconditional one-day delta marginals by cell", fontsize=15)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cdf_grid(gt: np.ndarray, pred: np.ndarray, out_path: Path, clip_quantile: float) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(16, 14))
    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            gt_cell = gt[:, r, c]
            pred_cell = pred[:, r, c]
            xs_gt, ys_gt = _ecdf(gt_cell)
            xs_pred, ys_pred = _ecdf(pred_cell)
            xlim = float(np.quantile(np.abs(np.concatenate([gt_cell, pred_cell])), clip_quantile))
            xlim = max(xlim, 1e-6)
            ax.plot(xs_gt, ys_gt, color="#1f77b4", linewidth=1.1, label="GT")
            ax.plot(xs_pred, ys_pred, color="#d62728", linewidth=1.1, label="Model")
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(0.0, 1.0)
            ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.6)
            ax.set_title(f"({r},{c})", fontsize=9)
            ax.tick_params(labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Unconditional one-day delta CDFs by cell", fontsize=15)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pooled(gt: np.ndarray, pred: np.ndarray, out_path: Path, clip_quantile: float, bins: int) -> None:
    gt_flat = gt.reshape(-1)
    pred_flat = pred.reshape(-1)
    xlim = float(np.quantile(np.abs(np.concatenate([gt_flat, pred_flat])), clip_quantile))
    xlim = max(xlim, 1e-6)
    edges = np.linspace(-xlim, xlim, bins + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(gt_flat, bins=edges, density=True, histtype="step", linewidth=1.6, color="#1f77b4", label="GT")
    axes[0].hist(pred_flat, bins=edges, density=True, histtype="step", linewidth=1.6, color="#d62728", label="Model")
    axes[0].axvline(0.0, color="black", linewidth=0.7, alpha=0.6)
    axes[0].set_title("Pooled histogram")
    xs_gt, ys_gt = _ecdf(gt_flat)
    xs_pred, ys_pred = _ecdf(pred_flat)
    axes[1].plot(xs_gt, ys_gt, color="#1f77b4", linewidth=1.4, label="GT")
    axes[1].plot(xs_pred, ys_pred, color="#d62728", linewidth=1.4, label="Model")
    axes[1].set_xlim(-xlim, xlim)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].axvline(0.0, color="black", linewidth=0.7, alpha=0.6)
    axes[1].set_title("Pooled CDF")
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("Pooled unconditional one-day delta marginal", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot unconditional H=1 per-cell marginals for deterministic models")
    parser.add_argument("--model_type", type=str, required=True, choices=sorted(LOADERS))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--future_gap", type=int, default=30)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_quantile", type=float, default=0.995)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = LOADERS[args.model_type](args.checkpoint, device)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = args.test_start - args.history_len - args.future_gap
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    test_indices = np.arange(args.test_start, len(surfaces) - args.history_len - args.future_gap + 1)
    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }[args.split]
    history_01, target_01 = build_one_step_windows(split_indices, surf_tensor, args.history_len)

    preds = []
    for start in range(0, history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, history_01.shape[0])
        preds.append(model.predict_delta(history_01[start:end]).detach().cpu().numpy())
    pred_delta = np.concatenate(preds, axis=0).reshape(-1, 5, 5)

    prev = history_01[:, -1].reshape(history_01.shape[0], 25).detach().cpu().numpy()
    target_delta = (target_01.detach().cpu().numpy().reshape(history_01.shape[0], 25) - prev).reshape(-1, 5, 5)

    ks_grid = [[_ks(target_delta[:, r, c], pred_delta[:, r, c]) for c in range(5)] for r in range(5)]
    ks_arr = np.array(ks_grid)
    summary = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "split": args.split,
        "n_windows": int(history_01.shape[0]),
        "pooled": {
            "gt_mean": float(target_delta.mean()),
            "pred_mean": float(pred_delta.mean()),
            "gt_std": float(target_delta.std()),
            "pred_std": float(pred_delta.std()),
            "gt_abs_q95": float(np.quantile(np.abs(target_delta.reshape(-1)), 0.95)),
            "pred_abs_q95": float(np.quantile(np.abs(pred_delta.reshape(-1)), 0.95)),
            "gt_abs_q99": float(np.quantile(np.abs(target_delta.reshape(-1)), 0.99)),
            "pred_abs_q99": float(np.quantile(np.abs(pred_delta.reshape(-1)), 0.99)),
            "ks": _ks(target_delta.reshape(-1), pred_delta.reshape(-1)),
        },
        "ks_grid": ks_grid,
        "ks_median": float(np.median(ks_arr)),
        "ks_max": float(ks_arr.max()),
        "worst_cell": [int(x) for x in np.unravel_index(np.argmax(ks_arr), (5, 5))],
    }

    _plot_hist_grid(target_delta, pred_delta, out_dir / "per_cell_hist_grid.png", args.clip_quantile, args.bins)
    _plot_cdf_grid(target_delta, pred_delta, out_dir / "per_cell_cdf_grid.png", args.clip_quantile)
    _plot_pooled(target_delta, pred_delta, out_dir / "pooled_overlay.png", args.clip_quantile, args.bins)

    md = [
        f"# {args.model_type} Unconditional Marginal Audit",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Epoch: `{payload.get('epoch', -1)}`",
        f"- Split: `{args.split}`",
        f"- Windows: `{history_01.shape[0]}`",
        "",
        "## Pooled",
        "",
        f"- GT mean delta: `{summary['pooled']['gt_mean']:.5f}`",
        f"- Model mean delta: `{summary['pooled']['pred_mean']:.5f}`",
        f"- GT std: `{summary['pooled']['gt_std']:.5f}`",
        f"- Model std: `{summary['pooled']['pred_std']:.5f}`",
        f"- GT abs q95: `{summary['pooled']['gt_abs_q95']:.5f}`",
        f"- Model abs q95: `{summary['pooled']['pred_abs_q95']:.5f}`",
        f"- GT abs q99: `{summary['pooled']['gt_abs_q99']:.5f}`",
        f"- Model abs q99: `{summary['pooled']['pred_abs_q99']:.5f}`",
        f"- Pooled KS: `{summary['pooled']['ks']:.4f}`",
        "",
        "## Per-cell KS",
        "",
        f"- Median KS: `{summary['ks_median']:.4f}`",
        f"- Worst KS: `{summary['ks_max']:.4f}` at cell `{tuple(summary['worst_cell'])}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["pooled"], indent=2))


if __name__ == "__main__":
    main()
