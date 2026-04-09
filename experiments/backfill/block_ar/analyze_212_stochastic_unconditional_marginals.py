#!/usr/bin/env python
"""
Plot unconditional per-cell next-day delta marginals for stochastic H=1 models
against ground truth over train/validation/test splits.

For each window the model generates multiple samples. We pool those generated
samples across windows to form per-cell unconditional histograms and compare
them with the single realized GT next-day delta per window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import compute_h1_shape_stats
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import load_model as load_212b_model
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import load_model as load_212s_model
from experiments.backfill.block_ar.train_212u_h1_minimal_direct_stochastic_delta_raw_output import load_model as load_212u_model
from experiments.backfill.block_ar.train_212v_h1_minimal_direct_stochastic_delta_tail_incidence_sampling import load_model as load_212v_model
from experiments.backfill.block_ar.train_212w_h1_minimal_direct_stochastic_delta_student_t_latent import load_model as load_212w_model
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import load_model as load_212x_model
from experiments.backfill.block_ar.train_212y_h1_minimal_direct_stochastic_delta_local_scale_asinh import load_model as load_212y_model
from experiments.backfill.block_ar.train_212z_h1_minimal_direct_stochastic_delta_local_scale_asinh_crps import load_model as load_212z_model
from experiments.backfill.block_ar.train_212aa_h1_minimal_direct_stochastic_delta_local_scale_asinh_tailcrps import load_model as load_212aa_model
from experiments.backfill.block_ar.train_212ac_h1_rectified_flow_local_scale_asinh import load_model as load_212ac_model
from experiments.backfill.block_ar.train_212ad_h1_conditional_diffusion_local_scale_asinh_terminal_es import load_model as load_212ad_model
from experiments.backfill.block_ar.train_212ae_h1_conditional_flow_local_scale_asinh import load_model as load_212ae_model
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import load_model as load_212af_model
from experiments.backfill.block_ar.train_212ag_h1_conditional_flow_local_scale_asinh_mean_aux import load_model as load_212ag_model
from experiments.backfill.block_ar.train_212ah_h1_conditional_flow_local_scale_asinh_nll_mean import load_model as load_212ah_model
from experiments.backfill.block_ar.train_212ai_h1_conditional_flow_local_scale_asinh_staged_nll import load_model as load_212ai_model
from experiments.backfill.block_ar.train_212ab_h1_conditional_diffusion_local_scale_asinh import load_model as load_212ab_model


LOADERS: dict[str, Callable[[str, torch.device], tuple[torch.nn.Module, dict[str, Any]]]] = {
    "212b": load_212b_model,
    "212s": load_212s_model,
    "212u": load_212u_model,
    "212v": load_212v_model,
    "212w": load_212w_model,
    "212x": load_212x_model,
    "212y": load_212y_model,
    "212z": load_212z_model,
    "212aa": load_212aa_model,
    "212ac": load_212ac_model,
    "212ad": load_212ad_model,
    "212ae": load_212ae_model,
    "212af": load_212af_model,
    "212ag": load_212ag_model,
    "212ah": load_212ah_model,
    "212ai": load_212ai_model,
    "212ab": load_212ab_model,
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


def _pearson_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    m = x.mean()
    v = ((x - m) ** 2).mean()
    if v <= 1e-12:
        return float("nan")
    return float(((x - m) ** 4).mean() / (v ** 2))


def _plot_hist_grid(gt: np.ndarray, pred: np.ndarray, out_path: Path, clip_quantile: float, bins: int) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(16, 14))
    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            gt_cell = gt[:, r, c]
            pred_cell = pred[:, :, r, c].reshape(-1)
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
            pred_cell = pred[:, :, r, c].reshape(-1)
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
    parser = argparse.ArgumentParser(description="Plot unconditional H=1 per-cell marginals for stochastic models")
    parser.add_argument("--model_type", type=str, required=True, choices=sorted(LOADERS))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--future_gap", type=int, default=30)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_quantile", type=float, default=0.995)
    parser.add_argument("--bins", type=int, default=60)
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

    pred_batches = []
    for start in range(0, history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, history_01.shape[0])
        pred_batches.append(model.sample_delta(history_01[start:end], n_samples=args.eval_samples).detach().cpu().numpy())
    pred_delta = np.concatenate(pred_batches, axis=0).reshape(-1, args.eval_samples, 5, 5)

    prev = history_01[:, -1].reshape(history_01.shape[0], 25).detach().cpu().numpy()
    target_delta = (target_01.detach().cpu().numpy().reshape(history_01.shape[0], 25) - prev).reshape(-1, 5, 5)

    shape_stats = compute_h1_shape_stats(target_delta, pred_delta)
    ks_grid = [[_ks(target_delta[:, r, c], pred_delta[:, :, r, c].reshape(-1)) for c in range(5)] for r in range(5)]
    ks_arr = np.array(ks_grid)

    kurtosis_rows = []
    for r in range(5):
        for c in range(5):
            gt_cell = target_delta[:, r, c]
            pred_cell = pred_delta[:, :, r, c].reshape(-1)
            gt_abs = np.abs(gt_cell)
            pred_abs = np.abs(pred_cell)
            kurtosis_rows.append(
                {
                    "cell": [r, c],
                    "gt_kurtosis": _pearson_kurtosis(gt_cell),
                    "pred_kurtosis": _pearson_kurtosis(pred_cell),
                    "kurtosis_ratio": _pearson_kurtosis(pred_cell) / max(_pearson_kurtosis(gt_cell), 1e-8),
                    "move_share_ratio_le_0p005": float((pred_abs <= 0.005).mean() / max((gt_abs <= 0.005).mean(), 1e-8)),
                    "move_share_ratio_le_0p010": float((pred_abs <= 0.010).mean() / max((gt_abs <= 0.010).mean(), 1e-8)),
                    "move_share_ratio_le_0p020": float((pred_abs <= 0.020).mean() / max((gt_abs <= 0.020).mean(), 1e-8)),
                    "move_share_ratio_le_0p050": float((pred_abs <= 0.050).mean() / max((gt_abs <= 0.050).mean(), 1e-8)),
                    "ks": _ks(gt_cell, pred_cell),
                }
            )

    summary = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "split": args.split,
        "n_windows": int(history_01.shape[0]),
        "eval_samples": args.eval_samples,
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
        "shape_stats": shape_stats,
        "ks_grid": ks_grid,
        "ks_median": float(np.median(ks_arr)),
        "ks_max": float(ks_arr.max()),
        "worst_cell": [int(x) for x in np.unravel_index(np.argmax(ks_arr), (5, 5))],
        "per_cell": kurtosis_rows,
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
        f"- Samples/window: `{args.eval_samples}`",
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
        "## Shape Stats",
        "",
        f"- Quiet ratio: `{shape_stats['quiet_ratio']:.3f}`",
        f"- Shoulder ratio: `{shape_stats['shoulder_ratio']:.3f}`",
        f"- Extreme ratio: `{shape_stats['extreme_ratio']:.3f}`",
        f"- Kurtosis ratio: `{shape_stats['kurtosis_ratio']:.3f}`",
        f"- <=0.005 share ratio: `{shape_stats['very_small_0p005_ratio']:.3f}`",
        f"- <=0.010 share ratio: `{shape_stats['small_0p010_ratio']:.3f}`",
        f"- <=0.020 share ratio: `{shape_stats['moderate_0p020_ratio']:.3f}`",
        f"- <=0.050 share ratio: `{shape_stats['large_0p050_ratio']:.3f}`",
        "",
        "## Per-cell KS",
        "",
        f"- Median KS: `{summary['ks_median']:.4f}`",
        f"- Worst KS: `{summary['ks_max']:.4f}` at cell `{tuple(summary['worst_cell'])}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"pooled": summary["pooled"], "shape_stats": shape_stats}, indent=2))


if __name__ == "__main__":
    main()
