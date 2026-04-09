#!/usr/bin/env python
"""
Tail-incidence diagnostics for 212v.

Measures:
  - q95 / q99 window-level AUC
  - Brier scores
  - probability-decile calibration
  - generated width by predicted-risk decile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows, make_serializable
from experiments.backfill.block_ar.train_212v_h1_minimal_direct_stochastic_delta_tail_incidence_sampling import (
    load_model,
)


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score, method="average")
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def decile_table(y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    out = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        out.append(
            {
                "decile": i,
                "count": int(len(idx)),
                "mean_pred": float(np.mean(p[idx])),
                "emp_rate": float(np.mean(y[idx])),
                "min_pred": float(np.min(p[idx])),
                "max_pred": float(np.max(p[idx])),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 212v tail-incidence head")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    q95_threshold = float(payload["thresholds"]["q95"])
    q99_threshold = float(payload["thresholds"]["q99"])

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    max_train_idx = args.test_start - args.history_len - 30
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, target_01 = build_one_step_windows(val_indices, surf_tensor, args.history_len)
    prev = history_01[:, -1].reshape(history_01.shape[0], 25)
    target_delta = target_01.reshape(history_01.shape[0], 25) - prev
    realized_max_abs = target_delta.abs().amax(dim=1).detach().cpu().numpy()
    y95 = (realized_max_abs >= q95_threshold).astype(np.float32)
    y99 = (realized_max_abs >= q99_threshold).astype(np.float32)

    p_all = []
    width_all = []
    bs = args.batch_size
    with torch.no_grad():
        for start in range(0, history_01.shape[0], bs):
            end = min(start + bs, history_01.shape[0])
            batch = history_01[start:end]
            probs = model.predict_tail_probs(batch).detach().cpu().numpy()
            sample_iv = model.sample_next_iv(batch, n_samples=args.eval_samples).detach().cpu().numpy().reshape(batch.shape[0], args.eval_samples, 5, 5)
            q05 = np.quantile(sample_iv, 0.05, axis=1)
            q95 = np.quantile(sample_iv, 0.95, axis=1)
            width = (q95 - q05).mean(axis=(1, 2))
            p_all.append(probs)
            width_all.append(width)

    probs = np.concatenate(p_all, axis=0)
    width = np.concatenate(width_all, axis=0)
    p95 = probs[:, 0]
    p99 = probs[:, 1]

    risk_deciles = decile_table(y99, p99)
    width_by_decile = []
    order = np.argsort(p99)
    bins = np.array_split(order, 10)
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        width_by_decile.append(
            {
                "decile": i,
                "count": int(len(idx)),
                "mean_p99": float(np.mean(p99[idx])),
                "mean_width_90": float(np.mean(width[idx])),
                "mean_realized_max_abs": float(np.mean(realized_max_abs[idx])),
                "emp_q99_rate": float(np.mean(y99[idx])),
            }
        )

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        "base_rates": {"q95": float(y95.mean()), "q99": float(y99.mean())},
        "q95_head": {
            "auc": binary_auc(y95, p95),
            "brier": float(np.mean((p95 - y95) ** 2)),
            "calibration_deciles": decile_table(y95, p95),
        },
        "q99_head": {
            "auc": binary_auc(y99, p99),
            "brier": float(np.mean((p99 - y99) ** 2)),
            "calibration_deciles": risk_deciles,
        },
        "width_by_q99_risk_decile": width_by_decile,
        "width_vs_predicted_q99_corr": float(np.corrcoef(width, p99)[0, 1]) if np.std(width) > 1e-12 and np.std(p99) > 1e-12 else float("nan"),
        "realized_max_vs_predicted_q99_corr": float(np.corrcoef(realized_max_abs, p99)[0, 1]) if np.std(realized_max_abs) > 1e-12 and np.std(p99) > 1e-12 else float("nan"),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(out), indent=2))

    lines = [
        "# 212v Tail-Incidence Diagnostics",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Epoch: `{out['epoch']}`",
        "",
        "## Headline",
        "",
        f"- q95 AUC: `{out['q95_head']['auc']:.3f}`",
        f"- q99 AUC: `{out['q99_head']['auc']:.3f}`",
        f"- q95 Brier: `{out['q95_head']['brier']:.4f}`",
        f"- q99 Brier: `{out['q99_head']['brier']:.4f}`",
        f"- Width vs predicted q99 corr: `{out['width_vs_predicted_q99_corr']:.3f}`",
        f"- Realized max |Δ| vs predicted q99 corr: `{out['realized_max_vs_predicted_q99_corr']:.3f}`",
        "",
        "## Interpretation",
        "",
        "- The tail head is useful if AUC is materially above 0.5 and generated width rises by predicted-risk decile.",
        "- If the head learns risk but width does not rise with it, the bottleneck is decoder usage rather than representation.",
        "",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n")

    print(json.dumps(make_serializable({
        "q95_auc": out["q95_head"]["auc"],
        "q99_auc": out["q99_head"]["auc"],
        "q95_brier": out["q95_head"]["brier"],
        "q99_brier": out["q99_head"]["brier"],
        "width_vs_predicted_q99_corr": out["width_vs_predicted_q99_corr"],
    }), indent=2))


if __name__ == "__main__":
    main()
