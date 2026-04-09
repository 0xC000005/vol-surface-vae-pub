#!/usr/bin/env python
"""
Focused point-forecast evaluation for deterministic H=1 direct-delta models
and a zero-delta baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows, make_serializable
from experiments.backfill.block_ar.train_212i_h1_deterministic_direct_delta_mse_cvar import load_model as load_212i_model
from experiments.backfill.block_ar.train_212j_h1_deterministic_direct_delta_lstm_mse_cvar import load_model as load_212j_model
from experiments.backfill.block_ar.train_212k_h1_deterministic_direct_delta_mse import load_model as load_212k_model
from experiments.backfill.block_ar.train_212l_h1_deterministic_direct_delta_raw_mse import load_model as load_212l_model
from experiments.backfill.block_ar.train_212m_h1_deterministic_direct_delta_raw_unscaled_mse import load_model as load_212m_model
from experiments.backfill.block_ar.train_212n_h1_deterministic_direct_delta_fully_raw_mse import load_model as load_212n_model
from experiments.backfill.block_ar.train_212o_h1_deterministic_relative_move_delta_loss import load_model as load_212o_model
from experiments.backfill.block_ar.train_212p_h1_deterministic_relative_move_dual_mse import load_model as load_212p_model


def _metrics(pred_delta: np.ndarray, target_delta: np.ndarray, q95_threshold: float, q99_threshold: float) -> dict:
    abs_err = np.abs(pred_delta - target_delta)
    sq_err = (pred_delta - target_delta) ** 2
    target_abs = np.abs(target_delta)
    q95_mask = target_abs >= q95_threshold
    q99_mask = target_abs >= q99_threshold

    per_window_mae = abs_err.mean(axis=1)
    per_cell_mae = abs_err.mean(axis=0)
    pred_sign = np.sign(pred_delta)
    target_sign = np.sign(target_delta)
    sign_acc = (pred_sign == target_sign).mean()

    all_up = (target_delta > 0).all(axis=1)
    all_down = (target_delta < 0).all(axis=1)
    same_sign_90 = ((target_delta > 0).mean(axis=1) >= 0.9) | ((target_delta < 0).mean(axis=1) >= 0.9)

    out = {
        "overall_mae": float(abs_err.mean()),
        "overall_rmse": float(np.sqrt(sq_err.mean())),
        "overall_sign_acc": float(sign_acc),
        "q95_mae": float(abs_err[q95_mask].mean()) if q95_mask.any() else None,
        "q99_mae": float(abs_err[q99_mask].mean()) if q99_mask.any() else None,
        "q95_sign_acc": float((pred_sign[q95_mask] == target_sign[q95_mask]).mean()) if q95_mask.any() else None,
        "q99_sign_acc": float((pred_sign[q99_mask] == target_sign[q99_mask]).mean()) if q99_mask.any() else None,
        "worst_window_idx": int(per_window_mae.argmax()),
        "worst_window_mae": float(per_window_mae.max()),
        "best_window_mae": float(per_window_mae.min()),
        "worst_cell": list(np.unravel_index(per_cell_mae.argmax(), (5, 5))),
        "worst_cell_mae": float(per_cell_mae.max()),
        "all_up_count": int(all_up.sum()),
        "all_down_count": int(all_down.sum()),
        "same_sign_90_count": int(same_sign_90.sum()),
    }

    if all_up.any():
        out["all_up_mae"] = float(abs_err[all_up].mean())
        out["all_up_sign_acc"] = float((pred_sign[all_up] == target_sign[all_up]).mean())
        out["all_up_pred_mean_delta"] = float(pred_delta[all_up].mean())
        out["all_up_realized_mean_delta"] = float(target_delta[all_up].mean())
    if all_down.any():
        out["all_down_mae"] = float(abs_err[all_down].mean())
        out["all_down_sign_acc"] = float((pred_sign[all_down] == target_sign[all_down]).mean())
        out["all_down_pred_mean_delta"] = float(pred_delta[all_down].mean())
        out["all_down_realized_mean_delta"] = float(target_delta[all_down].mean())
    if same_sign_90.any():
        out["same_sign_90_mae"] = float(abs_err[same_sign_90].mean())
        out["same_sign_90_sign_acc"] = float((pred_sign[same_sign_90] == target_sign[same_sign_90]).mean())

    top_pos_idx = np.argsort(target_delta.mean(axis=1))[-5:][::-1]
    top_neg_idx = np.argsort(target_delta.mean(axis=1))[:5]
    out["top_positive_windows"] = [
        {
            "idx": int(i),
            "realized_mean_delta": float(target_delta[i].mean()),
            "pred_mean_delta": float(pred_delta[i].mean()),
            "window_mae": float(per_window_mae[i]),
            "positive_share": float((target_delta[i] > 0).mean()),
        }
        for i in top_pos_idx
    ]
    out["top_negative_windows"] = [
        {
            "idx": int(i),
            "realized_mean_delta": float(target_delta[i].mean()),
            "pred_mean_delta": float(pred_delta[i].mean()),
            "window_mae": float(per_window_mae[i]),
            "negative_share": float((target_delta[i] < 0).mean()),
        }
        for i in top_neg_idx
    ]
    return out


def _to_md(model_name: str, metrics: dict) -> str:
    lines = [
        f"## {model_name}",
        "",
        f"- Overall MAE: `{metrics['overall_mae']:.4f}`",
        f"- Overall RMSE: `{metrics['overall_rmse']:.4f}`",
        f"- Overall sign accuracy: `{metrics['overall_sign_acc']:.3f}`",
        f"- q95 MAE: `{metrics['q95_mae']:.4f}`" if metrics.get("q95_mae") is not None else "- q95 MAE: `n/a`",
        f"- q99 MAE: `{metrics['q99_mae']:.4f}`" if metrics.get("q99_mae") is not None else "- q99 MAE: `n/a`",
        f"- q95 sign accuracy: `{metrics['q95_sign_acc']:.3f}`" if metrics.get("q95_sign_acc") is not None else "- q95 sign accuracy: `n/a`",
        f"- q99 sign accuracy: `{metrics['q99_sign_acc']:.3f}`" if metrics.get("q99_sign_acc") is not None else "- q99 sign accuracy: `n/a`",
        f"- Worst window idx: `{metrics['worst_window_idx']}` with MAE `{metrics['worst_window_mae']:.4f}`",
        f"- Worst cell: `{tuple(metrics['worst_cell'])}` with MAE `{metrics['worst_cell_mae']:.4f}`",
        f"- All-up windows: `{metrics['all_up_count']}`",
        f"- All-down windows: `{metrics['all_down_count']}`",
        f"- >=90% same-sign windows: `{metrics['same_sign_90_count']}`",
    ]
    if "all_up_mae" in metrics:
        lines.extend([
            f"- All-up MAE: `{metrics['all_up_mae']:.4f}`",
            f"- All-up sign accuracy: `{metrics['all_up_sign_acc']:.3f}`",
            f"- All-up mean predicted delta: `{metrics['all_up_pred_mean_delta']:.4f}` vs realized `{metrics['all_up_realized_mean_delta']:.4f}`",
        ])
    if "all_down_mae" in metrics:
        lines.extend([
            f"- All-down MAE: `{metrics['all_down_mae']:.4f}`",
            f"- All-down sign accuracy: `{metrics['all_down_sign_acc']:.3f}`",
            f"- All-down mean predicted delta: `{metrics['all_down_pred_mean_delta']:.4f}` vs realized `{metrics['all_down_realized_mean_delta']:.4f}`",
        ])
    if "same_sign_90_mae" in metrics:
        lines.extend([
            f"- >=90% same-sign MAE: `{metrics['same_sign_90_mae']:.4f}`",
            f"- >=90% same-sign sign accuracy: `{metrics['same_sign_90_sign_acc']:.3f}`",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze deterministic H=1 point forecast")
    parser.add_argument("--model_type", type=str, default="212i", choices=["212i", "212j", "212k", "212l", "212m", "212n", "212o", "212p"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loader = {
        "212i": load_212i_model,
        "212j": load_212j_model,
        "212k": load_212k_model,
        "212l": load_212l_model,
        "212m": load_212m_model,
        "212n": load_212n_model,
        "212o": load_212o_model,
        "212p": load_212p_model,
    }[args.model_type]
    model, payload = loader(args.checkpoint, device)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = args.test_start - args.history_len - 30
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    history_01, target_01 = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    q95_threshold = float(np.quantile(np.abs(np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0).reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0).reshape(-1)), 0.99))

    preds = []
    for start in range(0, history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, history_01.shape[0])
        batch = history_01[start:end]
        pred_delta = model.predict_delta(batch).detach().cpu().numpy()
        preds.append(pred_delta)
    pred_delta = np.concatenate(preds, axis=0)

    prev = history_01[:, -1].reshape(history_01.shape[0], 25).detach().cpu().numpy()
    target_delta = target_01.detach().cpu().numpy().reshape(history_01.shape[0], 25) - prev
    zero_delta = np.zeros_like(target_delta)

    model_metrics = _metrics(pred_delta, target_delta, q95_threshold, q99_threshold)
    zero_metrics = _metrics(zero_delta, target_delta, q95_threshold, q99_threshold)
    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "model": make_serializable(model_metrics),
        "zero_delta_baseline": make_serializable(zero_metrics),
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2))
    model_label = args.model_type
    md = [
        f"# {model_label} Point Forecast Evaluation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Epoch: `{payload.get('epoch', -1)}`",
        "",
        _to_md(model_label, model_metrics),
        "",
        _to_md("Zero-Delta Baseline", zero_metrics),
    ]
    Path(args.output_md).write_text("\n".join(md))
    print(json.dumps({
        "model_overall_mae": model_metrics["overall_mae"],
        "baseline_overall_mae": zero_metrics["overall_mae"],
        "model_q99_mae": model_metrics["q99_mae"],
        "baseline_q99_mae": zero_metrics["q99_mae"],
        "model_worst_window_mae": model_metrics["worst_window_mae"],
        "baseline_worst_window_mae": zero_metrics["worst_window_mae"],
    }, indent=2))


if __name__ == "__main__":
    main()
