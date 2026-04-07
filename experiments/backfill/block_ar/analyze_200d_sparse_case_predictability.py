#!/usr/bin/env python
"""
Low-cost predictability probe for sparse AR cases from history alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, ".")


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def build_window_arrays(surface: np.ndarray, history_len: int, future_len: int, starts: np.ndarray):
    histories = []
    hard_late_score = []
    nextday_any_q99 = []
    nextday_mean_abs = []
    hist_vov = []
    hist_last_mean = []
    hist_trend = []
    for s in starts:
        hist = surface[s : s + history_len]
        fut = surface[s + history_len : s + history_len + future_len]
        prev = hist[-1]
        histories.append(hist.reshape(-1))
        hist_mean = hist.mean(axis=(1, 2))
        hist_vov.append(float(np.diff(hist_mean).std()))
        hist_last_mean.append(float(hist_mean[-1]))
        hist_trend.append(float(hist_mean[-1] - hist_mean[0]))
        next_delta = np.abs(fut[0] - prev)
        nextday_mean_abs.append(float(next_delta.mean()))
        hard_late_score.append(float(np.abs(fut[-1] - prev).sum()))
    return (
        np.stack(histories, axis=0),
        np.asarray(hist_vov),
        np.asarray(hist_last_mean),
        np.asarray(hist_trend),
        np.asarray(hard_late_score),
        np.asarray(nextday_mean_abs),
    )


def fit_and_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(32, x_train.shape[1]))),
            ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    clf.fit(x_train, y_train)
    score = clf.predict_proba(x_test)[:, 1]
    return {
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc": safe_auc(y_test, score),
        "average_precision": safe_ap(y_test, score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse-case predictability from history alone")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/200_design/200d_sparse_case_predictability",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surf = np.load(args.data_path)["surface"].astype(np.float32)
    changes = np.abs(np.diff(surf[: args.test_start], axis=0).reshape(-1))
    q99 = float(np.quantile(changes, 0.99))

    max_train_idx = args.test_start - args.history_len - args.future_len
    train_starts = np.arange(max_train_idx)
    test_dataset_len = surf.shape[0] - args.test_start - args.history_len - args.future_len + 1
    test_starts = args.test_start + np.arange(test_dataset_len)

    x_train_flat, vov_train, last_mean_train, trend_train, hard_score_train, next_mean_abs_train = build_window_arrays(
        surf, args.history_len, args.future_len, train_starts
    )
    x_test_flat, vov_test, last_mean_test, trend_test, hard_score_test, next_mean_abs_test = build_window_arrays(
        surf, args.history_len, args.future_len, test_starts
    )

    q80_vov_train = float(np.quantile(vov_train, 0.8))
    turb_train = vov_train >= q80_vov_train
    q80_hard_within_turb = float(np.quantile(hard_score_train[turb_train], 0.8))
    hard_late_train = turb_train & (hard_score_train >= q80_hard_within_turb)
    hard_late_test = (vov_test >= q80_vov_train) & (hard_score_test >= q80_hard_within_turb)

    # H=1 any-cell next-day q99 event.
    def any_next_q99(starts: np.ndarray) -> np.ndarray:
        labels = []
        for s in starts:
            prev = surf[s + args.history_len - 1]
            fut0 = surf[s + args.history_len]
            labels.append(bool((np.abs(fut0 - prev) >= q99).any()))
        return np.asarray(labels, dtype=np.int64)

    next_q99_train = any_next_q99(train_starts)
    next_q99_test = any_next_q99(test_starts)

    simple_train = np.stack([vov_train, last_mean_train, trend_train], axis=1)
    simple_test = np.stack([vov_test, last_mean_test, trend_test], axis=1)

    summary: dict[str, Any] = {
        "config": {
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": int(len(train_starts)),
            "test_windows": int(len(test_starts)),
            "train_abs_q99_threshold": q99,
            "train_q80_vov": q80_vov_train,
            "train_q80_hard_within_turb": q80_hard_within_turb,
        },
        "hard_late_window_task": {
            "simple_features": fit_and_score(simple_train, hard_late_train.astype(int), simple_test, hard_late_test.astype(int)),
            "flattened_history": fit_and_score(x_train_flat, hard_late_train.astype(int), x_test_flat, hard_late_test.astype(int)),
        },
        "nextday_any_q99_task": {
            "simple_features": fit_and_score(simple_train, next_q99_train, simple_test, next_q99_test),
            "flattened_history": fit_and_score(x_train_flat, next_q99_train, x_test_flat, next_q99_test),
        },
        "interpretation": {
            "read": (
                "If flattened history has real AUC/AP on hard_late or nextday_any_q99 tasks, the sparse case has "
                "conditional signal in history and the main bottleneck is modeling/training, not pure randomness."
            )
        },
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
