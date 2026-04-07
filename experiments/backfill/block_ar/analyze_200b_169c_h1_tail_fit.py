#!/usr/bin/env python
"""
Pretest the H=1 local tail fit of the clean AR baseline (169c).

Questions:
  1. Is immediate next-step fit already good overall?
  2. Does the model generate rare next-step moves at roughly the right rate?
  3. How does next-step fit behave on realized q95/q99 tail cell-days?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_h7_mechanisms import (
    build_test_subset,
    load_model,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv


def safe_mean(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def summarize_subset(
    name: str,
    window_mask: np.ndarray,
    inside90: np.ndarray,
    width90: np.ndarray,
    actual_abs_delta: np.ndarray,
    sample_abs_delta: np.ndarray,
    q95: float,
    q99: float,
) -> dict[str, Any]:
    realized_q95 = actual_abs_delta >= q95
    realized_q99 = actual_abs_delta >= q99
    sample_q95_prob = (sample_abs_delta >= q95).mean(axis=1)
    sample_q99_prob = (sample_abs_delta >= q99).mean(axis=1)
    q95_mask_2d = window_mask[:, None] & realized_q95
    q99_mask_2d = window_mask[:, None] & realized_q99

    out = {
        "name": name,
        "windows": int(window_mask.sum()),
        "cov90": float(inside90[window_mask].mean()) if np.any(window_mask) else float("nan"),
        "width90": float(width90[window_mask].mean()) if np.any(window_mask) else float("nan"),
        "empirical_q95_rate": float(realized_q95[window_mask].mean()) if np.any(window_mask) else float("nan"),
        "empirical_q99_rate": float(realized_q99[window_mask].mean()) if np.any(window_mask) else float("nan"),
        "sample_q95_rate": float(sample_q95_prob[window_mask].mean()) if np.any(window_mask) else float("nan"),
        "sample_q99_rate": float(sample_q99_prob[window_mask].mean()) if np.any(window_mask) else float("nan"),
        "coverage90_on_realized_q95_cells": safe_mean(q95_mask_2d, inside90),
        "coverage90_on_realized_q99_cells": safe_mean(q99_mask_2d, inside90),
        "width90_on_realized_q95_cells": safe_mean(q95_mask_2d, width90),
        "width90_on_realized_q99_cells": safe_mean(q99_mask_2d, width90),
        "sample_q95_prob_on_realized_q95_cells": safe_mean(q95_mask_2d, sample_q95_prob),
        "sample_q99_prob_on_realized_q99_cells": safe_mean(q99_mask_2d, sample_q99_prob),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="H=1 tail-fit probe for 169c")
    parser.add_argument("--ckpt", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/200_design/200b_169c_h1_tail_fit",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_type, _payload = load_model(args.ckpt, str(device))
    if model_type != "multi_step_student_t_169c":
        raise ValueError(f"Expected 169c checkpoint, got {model_type}")

    history_norm, future_norm = build_test_subset(
        args.data_path,
        args.history_len,
        args.future_len,
        args.test_start,
        args.max_windows,
    )
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    prev = history_01[:, -1].reshape(history_01.shape[0], -1).numpy()
    target_h1 = future_01[:, 0].reshape(future_01.shape[0], -1).numpy()
    actual_delta = target_h1 - prev
    actual_abs_delta = np.abs(actual_delta)

    raw = np.load(args.data_path)
    train_surface = raw["surface"][: args.test_start].astype(np.float32)
    train_abs_delta = np.abs(np.diff(train_surface, axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    _vov, q20, q80 = regime_masks_from_history(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2)).numpy()
    hist_vov = np.diff(mean_iv, axis=1).std(axis=1)
    calm = hist_vov <= q20
    turb = hist_vov >= q80

    sample_chunks = []
    inside_chunks = []
    width_chunks = []
    for start in range(0, history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, history_01.shape[0])
        hist_batch = history_01[start:end].to(device)
        samples = model.sample_next_iv(hist_batch, n_samples=args.samples).detach().cpu().numpy()
        prev_batch = prev[start:end]
        lo = np.quantile(samples, 0.05, axis=1)
        hi = np.quantile(samples, 0.95, axis=1)
        inside_chunks.append((target_h1[start:end] >= lo) & (target_h1[start:end] <= hi))
        width_chunks.append(hi - lo)
        sample_chunks.append(np.abs(samples - prev_batch[:, None, :]))

    inside90 = np.concatenate(inside_chunks, axis=0)
    width90 = np.concatenate(width_chunks, axis=0)
    sample_abs_delta = np.concatenate(sample_chunks, axis=0)

    summary = {
        "config": {
            "ckpt": args.ckpt,
            "max_windows": int(history_01.shape[0]),
            "samples": args.samples,
            "batch_size": args.batch_size,
        },
        "train_thresholds": {
            "abs_q95": q95,
            "abs_q99": q99,
        },
        "overall": summarize_subset(
            "all",
            np.ones(history_01.shape[0], dtype=bool),
            inside90,
            width90,
            actual_abs_delta,
            sample_abs_delta,
            q95,
            q99,
        ),
        "calm": summarize_subset("calm", calm, inside90, width90, actual_abs_delta, sample_abs_delta, q95, q99),
        "turb": summarize_subset("turb", turb, inside90, width90, actual_abs_delta, sample_abs_delta, q95, q99),
        "interpretation": {
            "read": (
                "If H=1 overall fit is already solid but realized q99 cell-days remain weakly covered or weakly "
                "generated, the sparse-case problem already exists locally before long rollout."
            )
        },
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
