#!/usr/bin/env python
"""605a: hard-window support audit for sparse stress failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402


FACTOR_KEYS = ("ret", "price", "slopes", "skews", "levels")


def percentile_rank(value: float, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if ref.size == 0:
        return float("nan")
    return float(np.mean(ref <= float(value)))


def nearest_distance(reference: np.ndarray, query: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    qry = np.asarray(query, dtype=np.float64)
    if ref.ndim != 2 or qry.ndim != 2:
        raise ValueError("reference and query must be 2D")
    if ref.shape[1] != qry.shape[1]:
        raise ValueError("feature dimensions must match")
    ref_norm = np.sum(ref * ref, axis=1)
    out = np.empty(qry.shape[0], dtype=np.float64)
    for start in range(0, qry.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), qry.shape[0])
        block = qry[start:end]
        dist2 = np.sum(block * block, axis=1, keepdims=True) + ref_norm[None, :] - 2.0 * block @ ref.T
        out[start:end] = np.sqrt(np.maximum(np.min(dist2, axis=1), 0.0))
    return out


def nearest_self_distance(reference: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    """Nearest-neighbor distance within one matrix, excluding the same row."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("reference must be 2D")
    ref_norm = np.sum(ref * ref, axis=1)
    out = np.empty(ref.shape[0], dtype=np.float64)
    for start in range(0, ref.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), ref.shape[0])
        block = ref[start:end]
        dist2 = np.sum(block * block, axis=1, keepdims=True) + ref_norm[None, :] - 2.0 * block @ ref.T
        rows = np.arange(end - start)
        cols = np.arange(start, end)
        dist2[rows, cols] = np.inf
        out[start:end] = np.sqrt(np.maximum(np.min(dist2, axis=1), 0.0))
    return out


def future_cell_delta(
    history: np.ndarray,
    future: np.ndarray,
    *,
    horizon: int,
    row: int,
    col: int,
) -> np.ndarray:
    hist = np.asarray(history)
    fut = np.asarray(future)
    return fut[:, int(horizon) - 1, int(row), int(col)] - hist[:, -1, int(row), int(col)]


def split_indices(test_start: int, history_len: int, future_len: int, val_size: int) -> tuple[np.ndarray, np.ndarray]:
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    train_indices = np.arange(0, max_train_idx - int(val_size))
    val_indices = np.arange(max_train_idx - int(val_size), max_train_idx)
    return train_indices, val_indices


def build_windows(series: np.ndarray, indices: np.ndarray, history_len: int, future_len: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(series)
    history = np.stack([x[idx : idx + history_len] for idx in indices], axis=0)
    future = np.stack([x[idx + history_len : idx + history_len + future_len] for idx in indices], axis=0)
    return history, future


def build_iv_features(history: np.ndarray) -> np.ndarray:
    hist = np.asarray(history, dtype=np.float64)
    flat = hist.reshape(hist.shape[0], hist.shape[1], -1)
    diffs = np.diff(flat, axis=1)
    blocks = [
        flat[:, -1],
        flat.mean(axis=1),
        flat.std(axis=1),
        flat[:, -1] - flat.mean(axis=1),
        diffs[:, -1] if diffs.shape[1] else np.zeros_like(flat[:, -1]),
        np.quantile(np.abs(diffs), 0.90, axis=1) if diffs.shape[1] else np.zeros_like(flat[:, -1]),
    ]
    return np.concatenate(blocks, axis=1)


def build_factor_features(raw: dict[str, np.ndarray], indices: np.ndarray, history_len: int) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for key in FACTOR_KEYS:
        series = np.asarray(raw[key], dtype=np.float64)
        hist = np.stack([series[idx : idx + history_len] for idx in indices], axis=0)
        diffs = np.diff(hist, axis=1)
        blocks.extend(
            [
                hist[:, -1:],
                hist.mean(axis=1, keepdims=True),
                hist.std(axis=1, keepdims=True),
                (hist[:, -1] - hist.mean(axis=1))[:, None],
                (diffs[:, -1] if diffs.shape[1] else np.zeros(hist.shape[0]))[:, None],
                np.quantile(np.abs(diffs), 0.90, axis=1, keepdims=True)
                if diffs.shape[1]
                else np.zeros((hist.shape[0], 1), dtype=np.float64),
            ]
        )
    return np.concatenate(blocks, axis=1)


def standardize(train: np.ndarray, other: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True) + float(eps)
    return (train - mean) / scale, (other - mean) / scale


def support_report(
    *,
    train_history: np.ndarray,
    train_future: np.ndarray,
    val_history: np.ndarray,
    val_future: np.ndarray,
    train_iv_features: np.ndarray,
    val_iv_features: np.ndarray,
    train_factor_features: np.ndarray,
    val_factor_features: np.ndarray,
    hard_windows: list[int],
    hard_row: int,
    hard_col: int,
    hard_horizon: int,
) -> dict[str, Any]:
    train_iv_z, val_iv_z = standardize(train_iv_features, val_iv_features)
    train_factor_z, val_factor_z = standardize(train_factor_features, val_factor_features)
    train_joint_z, val_joint_z = standardize(
        np.concatenate([train_iv_features, train_factor_features], axis=1),
        np.concatenate([val_iv_features, val_factor_features], axis=1),
    )

    iv_nn = nearest_distance(train_iv_z, val_iv_z)
    factor_nn = nearest_distance(train_factor_z, val_factor_z)
    joint_nn = nearest_distance(train_joint_z, val_joint_z)
    train_iv_self = nearest_self_distance(train_iv_z)
    train_factor_self = nearest_self_distance(train_factor_z)
    train_joint_self = nearest_self_distance(train_joint_z)

    train_delta = future_cell_delta(
        train_history,
        train_future,
        horizon=hard_horizon,
        row=hard_row,
        col=hard_col,
    )
    val_delta = future_cell_delta(
        val_history,
        val_future,
        horizon=hard_horizon,
        row=hard_row,
        col=hard_col,
    )
    train_abs_path = np.max(np.abs(train_future - train_history[:, -1:, :, :]), axis=(1, 2, 3))
    val_abs_path = np.max(np.abs(val_future - val_history[:, -1:, :, :]), axis=(1, 2, 3))

    hard_rows = []
    for idx in hard_windows:
        hard_rows.append(
            {
                "val_window": int(idx),
                "iv_nn_distance": float(iv_nn[idx]),
                "iv_nn_percentile_vs_val": percentile_rank(iv_nn[idx], iv_nn),
                "iv_nn_percentile_vs_train_self": percentile_rank(iv_nn[idx], train_iv_self),
                "factor_nn_distance": float(factor_nn[idx]),
                "factor_nn_percentile_vs_val": percentile_rank(factor_nn[idx], factor_nn),
                "factor_nn_percentile_vs_train_self": percentile_rank(factor_nn[idx], train_factor_self),
                "joint_nn_distance": float(joint_nn[idx]),
                "joint_nn_percentile_vs_val": percentile_rank(joint_nn[idx], joint_nn),
                "joint_nn_percentile_vs_train_self": percentile_rank(joint_nn[idx], train_joint_self),
                "hard_cell_delta": float(val_delta[idx]),
                "hard_cell_delta_train_percentile": percentile_rank(val_delta[idx], train_delta),
                "abs_path_severity": float(val_abs_path[idx]),
                "abs_path_severity_train_percentile": percentile_rank(val_abs_path[idx], train_abs_path),
            }
        )

    return {
        "config": {
            "hard_windows": hard_windows,
            "hard_cell": [int(hard_row), int(hard_col)],
            "hard_horizon": int(hard_horizon),
        },
        "global": {
            "n_train": int(train_history.shape[0]),
            "n_val": int(val_history.shape[0]),
            "iv_nn_median": float(np.median(iv_nn)),
            "factor_nn_median": float(np.median(factor_nn)),
            "joint_nn_median": float(np.median(joint_nn)),
            "train_hard_cell_delta_p05_p50_p95": [
                float(x) for x in np.quantile(train_delta, [0.05, 0.50, 0.95])
            ],
            "train_abs_path_severity_p50_p90_p95_p99": [
                float(x) for x in np.quantile(train_abs_path, [0.50, 0.90, 0.95, 0.99])
            ],
        },
        "hard_windows": hard_rows,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 605a Hard-Window Support Audit",
        "",
        "## Summary",
        "",
        f"- train windows: `{report['global']['n_train']}`",
        f"- validation windows: `{report['global']['n_val']}`",
        f"- hard cell: `{report['config']['hard_cell']}`",
        f"- hard horizon: `{report['config']['hard_horizon']}`",
        f"- train hard-cell delta p05/p50/p95: `{[round(x, 5) for x in report['global']['train_hard_cell_delta_p05_p50_p95']]}`",
        f"- train max-path severity p50/p90/p95/p99: `{[round(x, 5) for x in report['global']['train_abs_path_severity_p50_p90_p95_p99']]}`",
        "",
        "## Hard Windows",
        "",
        "| val window | IV NN pct val/train | factor NN pct val/train | joint NN pct val/train | cell delta | cell delta pct | path severity | path severity pct |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["hard_windows"]:
        lines.append(
            f"| {row['val_window']} | {row['iv_nn_percentile_vs_val']:.3f} "
            f"/ {row['iv_nn_percentile_vs_train_self']:.3f} "
            f"| {row['factor_nn_percentile_vs_val']:.3f} "
            f"/ {row['factor_nn_percentile_vs_train_self']:.3f} "
            f"| {row['joint_nn_percentile_vs_val']:.3f} "
            f"/ {row['joint_nn_percentile_vs_train_self']:.3f} "
            f"| {row['hard_cell_delta']:.5f} "
            f"| {row['hard_cell_delta_train_percentile']:.3f} "
            f"| {row['abs_path_severity']:.5f} "
            f"| {row['abs_path_severity_train_percentile']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--hard_windows", default="420,421,423")
    parser.add_argument("--hard_row", type=int, default=3)
    parser.add_argument("--hard_col", type=int, default=3)
    parser.add_argument("--hard_horizon", type=int, default=30)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    raw_npz = np.load(args.data_path)
    raw = {key: raw_npz[key].astype(np.float64) for key in raw_npz.files}
    train_idx, val_idx = split_indices(args.test_start, args.history_len, args.future_len, args.val_size)
    train_history, train_future = build_windows(raw["surface"], train_idx, args.history_len, args.future_len)
    val_history, val_future = build_windows(raw["surface"], val_idx, args.history_len, args.future_len)
    train_iv = build_iv_features(train_history)
    val_iv = build_iv_features(val_history)
    train_factor = build_factor_features(raw, train_idx, args.history_len)
    val_factor = build_factor_features(raw, val_idx, args.history_len)
    hard_windows = [int(x) for x in args.hard_windows.split(",") if x.strip()]
    report = support_report(
        train_history=train_history,
        train_future=train_future,
        val_history=val_history,
        val_future=val_future,
        train_iv_features=train_iv,
        val_iv_features=val_iv,
        train_factor_features=train_factor,
        val_factor_features=val_factor,
        hard_windows=hard_windows,
        hard_row=args.hard_row,
        hard_col=args.hard_col,
        hard_horizon=args.hard_horizon,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(json.dumps(make_serializable(report["hard_windows"]), indent=2))


if __name__ == "__main__":
    main()
