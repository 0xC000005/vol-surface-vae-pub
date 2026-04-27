#!/usr/bin/env python
"""599a: audit whether anchor factors explain hard 510a IV failure windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


def validation_indices(
    *,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    max_windows: int,
) -> np.ndarray:
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    indices = np.arange(max_train_idx - int(val_size), max_train_idx)
    return indices[: int(max_windows)]


def build_history_summary_features(history: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Build simple last/mean/std/change summaries from history paths."""
    arr = np.asarray(history, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("history must have shape [N,H,C]")
    last = arr[:, -1]
    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    last_minus_mean = last - mean
    last_delta = arr[:, -1] - arr[:, -2] if arr.shape[1] >= 2 else np.zeros_like(last)
    blocks = [
        ("last", last),
        ("mean", mean),
        ("std", std),
        ("last_minus_mean", last_minus_mean),
        ("last_delta", last_delta),
    ]
    features = np.concatenate([values for _name, values in blocks], axis=1)
    names: list[str] = []
    for block_name, values in blocks:
        names.extend(f"{prefix}_{block_name}_{idx}" for idx in range(values.shape[1]))
    return features.astype(np.float64), names


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC for binary labels using average ranks for ties."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("y_true and score must have the same shape")
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_s = s[order]
    start = 0
    while start < len(sorted_s):
        end = start + 1
        while end < len(sorted_s) and sorted_s[end] == sorted_s[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = float(ranks[pos].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def ridge_oos_score(
    features: np.ndarray,
    target: np.ndarray,
    *,
    train_frac: float,
    alpha: float,
) -> dict[str, float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be [N,D] and target must be [N]")
    n_train = int(round(x.shape[0] * float(train_frac)))
    n_train = min(max(n_train, 2), x.shape[0] - 1)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    y_mean = float(y_train.mean())
    y_center = y_train - y_mean
    xtx = x_train.T @ x_train
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    beta = np.linalg.solve(xtx + reg, x_train.T @ y_center)
    pred = x_test @ beta + y_mean
    mse = float(np.mean((pred - y_test) ** 2))
    base_mse = float(np.mean((y_mean - y_test) ** 2))
    r2 = 1.0 - mse / max(base_mse, 1e-12)
    corr = float(np.corrcoef(pred, y_test)[0, 1]) if np.std(pred) > 1e-12 and np.std(y_test) > 1e-12 else 0.0
    auc = auc_score(y_test, pred) if set(np.unique(y_test)).issubset({0.0, 1.0}) else float("nan")
    return {
        "n_train": int(n_train),
        "n_test": int(x.shape[0] - n_train),
        "mse": mse,
        "baseline_mse": base_mse,
        "r2": float(r2),
        "corr": corr,
        "auc": float(auc),
    }


@torch.no_grad()
def sample_native_model(
    *,
    checkpoint: str,
    batch,
    samples: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model, _payload = load_one_day_kernel("340c", checkpoint, device)
    model.eval()
    outputs: list[np.ndarray] = []
    n_steps = int(batch.future_01.shape[1])
    for start in range(0, batch.history_01.shape[0], int(batch_size)):
        end = min(start + int(batch_size), batch.history_01.shape[0])
        hist_batch = normalize_iv(batch.history_01[start:end])
        sampled = model.sample_batched(
            hist_batch,
            n_samples=int(samples),
            n_steps=n_steps,
            chunk_size=int(chunk_size),
        )
        outputs.append(sampled.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_failure_targets(samples: np.ndarray, future_01: np.ndarray) -> dict[str, np.ndarray]:
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    median = np.median(samples, axis=1)
    gt = np.asarray(future_01, dtype=np.float64)
    inside = (gt >= q05) & (gt <= q95)
    window_cov = inside.mean(axis=(1, 2, 3))
    cell_cov = inside.mean(axis=1)
    persistent_under_count = (cell_cov < 0.30).sum(axis=(1, 2)).astype(np.float64)
    median_bias_frac = (median > gt).mean(axis=(1, 2, 3))
    level_abs_error = np.abs(median - gt).mean(axis=(1, 2, 3))
    return {
        "bad_window_lt50": (window_cov < 0.50).astype(np.float64),
        "window_coverage": window_cov.astype(np.float64),
        "persistent_under_count": persistent_under_count,
        "median_bias_frac": median_bias_frac.astype(np.float64),
        "level_abs_error": level_abs_error.astype(np.float64),
    }


def build_factor_history(panel: np.ndarray, indices: np.ndarray, history_len: int, factor_start: int = 25) -> np.ndarray:
    histories = [panel[int(idx) : int(idx) + int(history_len), factor_start:] for idx in indices]
    return np.asarray(histories, dtype=np.float64)


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# 599a Factor Signal Audit",
        "",
        f"- checkpoint: `{results['config']['checkpoint']}`",
        f"- windows: `{results['config']['n_windows']}`",
        f"- samples: `{results['config']['samples']}`",
        f"- train/test split: `{results['config']['train_frac']:.2f}`",
        "",
        "## Target Summary",
    ]
    for name, stats in results["target_summary"].items():
        lines.append(
            f"- `{name}`: mean `{stats['mean']:.4f}`, std `{stats['std']:.4f}`, "
            f"min `{stats['min']:.4f}`, max `{stats['max']:.4f}`"
        )
    lines.extend(["", "## Lift vs IV-Only"])
    for target, lift in results["lift_summary"].items():
        lines.append(
            f"- `{target}`: factor R2 lift `{lift['factor_minus_iv_r2']:.4f}`, "
            f"IV+factor R2 lift `{lift['iv_factor_minus_iv_r2']:.4f}`, "
            f"factor AUC lift `{lift['factor_minus_iv_auc']:.4f}`, "
            f"IV+factor AUC lift `{lift['iv_factor_minus_iv_auc']:.4f}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=599)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    samples = sample_native_model(
        checkpoint=args.checkpoint,
        batch=batch,
        samples=args.samples,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
        seed=args.seed,
    )
    indices = validation_indices(
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
    )
    panel, columns, _dates = load_aligned_iv_factor_panel(iv_path=args.data_path)
    factor_history = build_factor_history(panel, indices, args.history_len)
    iv_history = batch.history_01.detach().cpu().numpy().reshape(batch.history_01.shape[0], args.history_len, -1)
    iv_features, iv_feature_names = build_history_summary_features(iv_history, "iv")
    factor_features, factor_feature_names = build_history_summary_features(factor_history, "factor")
    combined_features = np.concatenate([iv_features, factor_features], axis=1)
    targets = build_failure_targets(samples, batch.future_01.detach().cpu().numpy())
    scores: dict[str, dict[str, dict[str, float]]] = {}
    lift_summary: dict[str, dict[str, float]] = {}
    for target_name, target in targets.items():
        target_scores = {
            "iv": ridge_oos_score(iv_features, target, train_frac=args.train_frac, alpha=args.ridge_alpha),
            "factor": ridge_oos_score(factor_features, target, train_frac=args.train_frac, alpha=args.ridge_alpha),
            "iv_factor": ridge_oos_score(combined_features, target, train_frac=args.train_frac, alpha=args.ridge_alpha),
        }
        scores[target_name] = target_scores
        iv_score = target_scores["iv"]
        factor_score = target_scores["factor"]
        both_score = target_scores["iv_factor"]
        lift_summary[target_name] = {
            "factor_minus_iv_r2": float(factor_score["r2"] - iv_score["r2"]),
            "iv_factor_minus_iv_r2": float(both_score["r2"] - iv_score["r2"]),
            "factor_minus_iv_auc": float(factor_score["auc"] - iv_score["auc"])
            if np.isfinite(factor_score["auc"]) and np.isfinite(iv_score["auc"])
            else float("nan"),
            "iv_factor_minus_iv_auc": float(both_score["auc"] - iv_score["auc"])
            if np.isfinite(both_score["auc"]) and np.isfinite(iv_score["auc"])
            else float("nan"),
        }
    target_summary = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in targets.items()
    }
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "n_windows": int(batch.history_01.shape[0]),
            "samples": int(args.samples),
            "train_frac": float(args.train_frac),
            "ridge_alpha": float(args.ridge_alpha),
            "seed": int(args.seed),
            "factor_columns": columns[25:],
            "n_iv_features": int(iv_features.shape[1]),
            "n_factor_features": int(factor_features.shape[1]),
            "iv_feature_names_head": iv_feature_names[:8],
            "factor_feature_names_head": factor_feature_names[:8],
        },
        "target_summary": target_summary,
        "scores": scores,
        "lift_summary": lift_summary,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), results)
    print(json.dumps(make_serializable({"target_summary": target_summary, "lift_summary": lift_summary}), indent=2))


if __name__ == "__main__":
    main()
