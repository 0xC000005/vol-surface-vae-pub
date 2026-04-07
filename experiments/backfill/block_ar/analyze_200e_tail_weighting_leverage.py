#!/usr/bin/env python
"""
Quantify how much generic tail-aware weighting can increase training mass on sparse cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    w = weights.astype(np.float64)
    return float(w[mask].sum() / np.maximum(w.sum(), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail-weighting leverage audit")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/200_design/200e_tail_weighting_leverage",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surf = np.load(args.data_path)["surface"].astype(np.float32)
    train_surface = surf[: args.test_start]
    abs_train_delta = np.abs(np.diff(train_surface, axis=0))
    q95 = float(np.quantile(abs_train_delta.reshape(-1), 0.95))
    q99 = float(np.quantile(abs_train_delta.reshape(-1), 0.99))

    max_train_idx = args.test_start - args.history_len - args.future_len
    starts = np.arange(max_train_idx)

    window_vov = []
    hard_score = []
    flat_future_abs_delta = []
    for s in starts:
        hist = surf[s : s + args.history_len]
        fut = surf[s + args.history_len : s + args.history_len + args.future_len]
        prev = hist[-1]
        hist_mean = hist.mean(axis=(1, 2))
        window_vov.append(float(np.diff(hist_mean).std()))
        hard_score.append(float(np.abs(fut[-1] - prev).sum()))
        future_abs_delta = np.abs(np.diff(np.concatenate([prev[None], fut], axis=0), axis=0)).reshape(-1)
        flat_future_abs_delta.append(future_abs_delta)

    window_vov = np.asarray(window_vov)
    hard_score = np.asarray(hard_score)
    future_abs_delta = np.stack(flat_future_abs_delta, axis=0)

    q80_vov = float(np.quantile(window_vov, 0.8))
    turb = window_vov >= q80_vov
    q80_hard_within_turb = float(np.quantile(hard_score[turb], 0.8))
    hard_late = turb & (hard_score >= q80_hard_within_turb)

    q95_cells = future_abs_delta >= q95
    q99_cells = future_abs_delta >= q99

    uniform_window = np.ones_like(window_vov)
    hard_window_weight = 1.0 + args.alpha * hard_late.astype(np.float32)
    vov_window_weight = 1.0 + args.alpha * np.clip(window_vov / np.maximum(q80_vov, 1e-12), 0.0, 2.0)

    uniform_cell = np.ones_like(future_abs_delta, dtype=np.float32)
    q95_cell_weight = 1.0 + args.alpha * q95_cells.astype(np.float32)
    magnitude_cell_weight = 1.0 + args.alpha * np.clip(future_abs_delta / np.maximum(q95, 1e-12), 0.0, 3.0)

    summary = {
        "config": {
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": int(len(starts)),
            "alpha": args.alpha,
            "q95_abs_delta": q95,
            "q99_abs_delta": q99,
            "q80_vov": q80_vov,
            "q80_hard_within_turb": q80_hard_within_turb,
        },
        "base_rates": {
            "hard_late_window_fraction": float(hard_late.mean()),
            "future_q95_cell_fraction": float(q95_cells.mean()),
            "future_q99_cell_fraction": float(q99_cells.mean()),
        },
        "window_weighting": {
            "uniform_hard_late_share": weighted_share(hard_late, uniform_window),
            "hard_indicator_hard_late_share": weighted_share(hard_late, hard_window_weight),
            "vov_weight_hard_late_share": weighted_share(hard_late, vov_window_weight),
        },
        "cell_weighting": {
            "uniform_q95_share": weighted_share(q95_cells, uniform_cell),
            "uniform_q99_share": weighted_share(q99_cells, uniform_cell),
            "q95_indicator_q95_share": weighted_share(q95_cells, q95_cell_weight),
            "q95_indicator_q99_share": weighted_share(q99_cells, q95_cell_weight),
            "magnitude_q95_share": weighted_share(q95_cells, magnitude_cell_weight),
            "magnitude_q99_share": weighted_share(q99_cells, magnitude_cell_weight),
        },
        "interpretation": {
            "read": (
                "These are generic training-side weights, not benchmark masks. They quantify whether tail-aware "
                "scoring can materially shift optimization mass onto the sparse windows/cell-days that uniform losses underweight."
            )
        },
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
