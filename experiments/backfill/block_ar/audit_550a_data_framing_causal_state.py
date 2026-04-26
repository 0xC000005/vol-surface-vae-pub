#!/usr/bin/env python
"""550a: data-framing and causal-state audit for deployable scenario generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    load_one_day_kernel,
    make_serializable,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (  # noqa: E402
    normalize_iv,
)


def effective_independent_windows(n_windows: int, future_len: int) -> float:
    """Approximate non-overlapping-equivalent windows for rolling future paths."""
    if future_len <= 0:
        raise ValueError("future_len must be positive")
    return round(float(n_windows) / float(future_len), 2)


def split_geometry(
    n_total_days: int,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    max_windows: int,
) -> dict[str, int | float]:
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    base_train_windows = max_train_idx - int(val_size)
    calibration_start = max(0, base_train_windows - int(val_size))
    calibration_end = base_train_windows - 1
    official_val_start = base_train_windows
    official_val_end = max_train_idx - 1
    eval_subset_end = official_val_start + int(max_windows) - 1
    return {
        "n_total_days": int(n_total_days),
        "history_len": int(history_len),
        "future_len": int(future_len),
        "test_start": int(test_start),
        "max_train_idx": int(max_train_idx),
        "base_train_windows": int(base_train_windows),
        "official_val_windows": int(val_size),
        "calibration_start": int(calibration_start),
        "calibration_end": int(calibration_end),
        "calibration_windows": int(calibration_end - calibration_start + 1),
        "official_val_start": int(official_val_start),
        "official_val_end": int(official_val_end),
        "eval_subset_start": int(official_val_start),
        "eval_subset_end": int(eval_subset_end),
        "eval_subset_windows": int(max_windows),
        "future_overlap_days_between_adjacent_windows": int(max(0, future_len - 1)),
        "official_val_effective_independent_windows": effective_independent_windows(
            val_size,
            future_len,
        ),
        "eval_subset_effective_independent_windows": effective_independent_windows(
            max_windows,
            future_len,
        ),
    }


def make_windows_np(
    surfaces: np.ndarray,
    indices: np.ndarray,
    history_len: int,
    future_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_offsets = np.arange(history_len)[None, :]
    future_offsets = history_len + np.arange(future_len)[None, :]
    history = surfaces[indices[:, None] + history_offsets]
    future = surfaces[indices[:, None] + future_offsets]
    return history.astype(np.float32), future.astype(np.float32)


def causal_state_features(history_01: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Causal state features from the history window only."""
    if history_01.ndim != 4:
        raise ValueError("history_01 must have shape (windows, history, rows, cols)")
    last = history_01[:, -1]
    first = history_01[:, 0]
    daily_mean = history_01.mean(axis=(2, 3))
    diffs = np.diff(history_01, axis=1)
    mean_diffs = np.diff(daily_mean, axis=1)
    center = last[:, last.shape[1] // 2, last.shape[2] // 2]
    edge_mean = (
        last[:, 0, :].mean(axis=1)
        + last[:, -1, :].mean(axis=1)
        + last[:, :, 0].mean(axis=1)
        + last[:, :, -1].mean(axis=1)
    ) / 4.0
    names = [
        "last_mean",
        "last_std",
        "last_min",
        "last_max",
        "surface_range",
        "history_vov",
        "history_abs_move_mean",
        "history_abs_move_q90",
        "history_trend_mean",
        "term_slope",
        "smile_slope",
        "center_minus_edge",
    ]
    features = np.stack(
        [
            last.mean(axis=(1, 2)),
            last.std(axis=(1, 2)),
            last.min(axis=(1, 2)),
            last.max(axis=(1, 2)),
            last.max(axis=(1, 2)) - last.min(axis=(1, 2)),
            mean_diffs.std(axis=1),
            np.abs(diffs).mean(axis=(1, 2, 3)),
            np.quantile(np.abs(diffs).reshape(history_01.shape[0], -1), 0.90, axis=1),
            (last - first).mean(axis=(1, 2)),
            last[:, -1, :].mean(axis=1) - last[:, 0, :].mean(axis=1),
            last[:, :, -1].mean(axis=1) - last[:, :, 0].mean(axis=1),
            center - edge_mean,
        ],
        axis=1,
    )
    return features.astype(np.float64), names


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return 0.0
    x_finite = x[finite]
    y_finite = y[finite]
    if np.ptp(x_finite) <= 0.0 or np.ptp(y_finite) <= 0.0:
        return 0.0
    xr = _ranks(x_finite)
    yr = _ranks(y_finite)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = float(np.sqrt((xr * xr).sum() * (yr * yr).sum()))
    if denom <= 0.0:
        return 0.0
    return float((xr * yr).sum() / denom)


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return 0.0
    values = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, values, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, values, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def feature_drift_table(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(feature_names):
        train_col = train_features[:, idx]
        eval_col = eval_features[:, idx]
        train_std = float(np.std(train_col))
        eval_mean = float(np.mean(eval_col))
        train_mean = float(np.mean(train_col))
        denom = max(train_std, 1e-12)
        rows.append(
            {
                "feature": name,
                "train_mean": train_mean,
                "eval_mean": eval_mean,
                "standardized_mean_shift": float((eval_mean - train_mean) / denom),
                "ks_stat": ks_statistic(train_col, eval_col),
            }
        )
    return sorted(
        rows,
        key=lambda row: max(abs(float(row["standardized_mean_shift"])), float(row["ks_stat"])),
        reverse=True,
    )


@torch.no_grad()
def sample_model(
    model: torch.nn.Module,
    history_01: np.ndarray,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    history = torch.from_numpy(history_01).to(device)
    history_norm = normalize_iv(history)
    outputs: list[np.ndarray] = []
    for start in range(0, history_norm.shape[0], batch_size):
        end = min(start + batch_size, history_norm.shape[0])
        samples = model.sample_batched(
            history_norm[start:end],
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
        )
        outputs.append(samples.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def window_failure_metrics(
    samples: np.ndarray,
    future_01: np.ndarray,
    train_future_01: np.ndarray,
) -> dict[str, np.ndarray]:
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    median = np.median(samples, axis=1)
    covered = (future_01 >= q05) & (future_01 <= q95)
    coverage = covered.mean(axis=(1, 2, 3))
    width = (q95 - q05).mean(axis=(1, 2, 3))
    model_mae = np.abs(median - future_01).mean(axis=(1, 2, 3))
    uncond_median = np.median(train_future_01, axis=0)
    uncond_mae = np.abs(uncond_median[None] - future_01).mean(axis=(1, 2, 3))
    mae_reduction = (uncond_mae - model_mae) / np.maximum(uncond_mae, 1e-8)
    gen_jumps = np.abs(np.diff(samples, axis=2)).max(axis=(2, 3, 4)).mean(axis=1)
    gt_jumps = np.abs(np.diff(future_01, axis=1)).max(axis=(1, 2, 3))
    future_move = np.abs(np.diff(future_01, axis=1)).mean(axis=(1, 2, 3))
    future_level_mean = future_01.mean(axis=(1, 2, 3))
    return {
        "coverage90": coverage,
        "width90": width,
        "model_median_mae": model_mae,
        "unconditional_median_mae": uncond_mae,
        "mae_reduction": mae_reduction,
        "gen_path_max_jump_mean": gen_jumps,
        "gt_path_max_jump": gt_jumps,
        "future_abs_move_mean": future_move,
        "future_level_mean": future_level_mean,
    }


def feature_metric_correlations(
    features: np.ndarray,
    feature_names: list[str],
    metrics: dict[str, np.ndarray],
) -> list[dict[str, float | str]]:
    target_names = [
        "coverage90",
        "width90",
        "model_median_mae",
        "mae_reduction",
        "future_abs_move_mean",
        "gt_path_max_jump",
    ]
    rows: list[dict[str, float | str]] = []
    for feature_idx, feature_name in enumerate(feature_names):
        for target_name in target_names:
            corr = rank_corr(features[:, feature_idx], metrics[target_name])
            rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "spearman": corr,
                    "abs_spearman": abs(corr),
                }
            )
    return sorted(rows, key=lambda row: float(row["abs_spearman"]), reverse=True)


def binned_state_summary(
    features: np.ndarray,
    feature_names: list[str],
    metrics: dict[str, np.ndarray],
    selected_features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in selected_features:
        if name not in feature_names:
            continue
        idx = feature_names.index(name)
        values = features[:, idx]
        q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        masks = [
            ("low", values <= q1),
            ("mid", (values > q1) & (values <= q2)),
            ("high", values > q2),
        ]
        for bucket, mask in masks:
            if int(mask.sum()) == 0:
                continue
            rows.append(
                {
                    "feature": name,
                    "bucket": bucket,
                    "n": int(mask.sum()),
                    "feature_mean": float(values[mask].mean()),
                    "coverage90": float(metrics["coverage90"][mask].mean()),
                    "mae_reduction": float(metrics["mae_reduction"][mask].mean()),
                    "median_mae": float(metrics["model_median_mae"][mask].mean()),
                    "width90": float(metrics["width90"][mask].mean()),
                    "future_abs_move_mean": float(metrics["future_abs_move_mean"][mask].mean()),
                }
            )
    return rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    split = report["split_geometry"]
    metrics = report["window_metric_summary"]
    drift = report["calibration_to_eval_drift_top"]
    corrs = report["top_feature_metric_correlations"]
    lines = [
        "# 550a Data-Framing and Causal-State Audit",
        "",
        "## Split Geometry",
        "",
        f"- base train windows: `{split['base_train_windows']}`",
        f"- calibration block: `{split['calibration_start']}..{split['calibration_end']}`",
        f"- official validation block: `{split['official_val_start']}..{split['official_val_end']}`",
        f"- eval subset: `{split['eval_subset_start']}..{split['eval_subset_end']}`",
        f"- eval subset non-overlap equivalent windows: `{split['eval_subset_effective_independent_windows']}`",
        f"- adjacent 30-day future windows overlap by `{split['future_overlap_days_between_adjacent_windows']}` days",
        "",
        "## Frontier Sample Diagnostics",
        "",
        f"- mean 90% window coverage: `{metrics['coverage90_mean']:.3f}`",
        f"- mean conditional MAE reduction proxy: `{metrics['mae_reduction_mean']:.3%}`",
        f"- mean 90% width: `{metrics['width90_mean']:.4f}`",
        f"- mean median MAE: `{metrics['model_median_mae_mean']:.4f}`",
        "",
        "## Calibration-To-Eval State Drift",
        "",
    ]
    for row in drift[:8]:
        lines.append(
            f"- `{row['feature']}`: KS `{row['ks_stat']:.3f}`, "
            f"standardized mean shift `{row['standardized_mean_shift']:.2f}`"
        )
    lines.extend(["", "## Strongest Causal-State Associations", ""])
    for row in corrs[:12]:
        lines.append(
            f"- `{row['feature']}` vs `{row['target']}`: Spearman `{row['spearman']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    geom = split_geometry(
        n_total_days=surfaces.shape[0],
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
    )

    train_indices = np.arange(0, int(geom["base_train_windows"]))
    calib_indices = np.arange(int(geom["calibration_start"]), int(geom["calibration_end"]) + 1)
    eval_indices = np.arange(int(geom["eval_subset_start"]), int(geom["eval_subset_end"]) + 1)
    train_history, train_future = make_windows_np(
        surfaces,
        train_indices,
        args.history_len,
        args.future_len,
    )
    calib_history, _ = make_windows_np(
        surfaces,
        calib_indices,
        args.history_len,
        args.future_len,
    )
    eval_history, eval_future = make_windows_np(
        surfaces,
        eval_indices,
        args.history_len,
        args.future_len,
    )

    train_features, feature_names = causal_state_features(train_history)
    calib_features, _ = causal_state_features(calib_history)
    eval_features, _ = causal_state_features(eval_history)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    model.eval()
    samples = sample_model(
        model=model,
        history_01=eval_history,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    metrics = window_failure_metrics(samples, eval_future, train_future)
    correlations = feature_metric_correlations(eval_features, feature_names, metrics)
    binned = binned_state_summary(
        eval_features,
        feature_names,
        metrics,
        selected_features=[
            "last_mean",
            "history_vov",
            "history_abs_move_mean",
            "history_trend_mean",
            "term_slope",
            "smile_slope",
        ],
    )
    drift_calib_eval = feature_drift_table(calib_features, eval_features, feature_names)
    drift_train_eval = feature_drift_table(train_features, eval_features, feature_names)

    window_metric_summary = {
        f"{name}_mean": float(np.mean(value))
        for name, value in metrics.items()
    }
    window_metric_summary.update(
        {
            f"{name}_p10": float(np.quantile(value, 0.10))
            for name, value in metrics.items()
        }
    )

    decision = (
        "The next deployable improvement should be a training-only calibration/state "
        "protocol, not another base architecture. The audit should guide which causal "
        "state variables define calibration strata and whether the calibration block is "
        "state-compatible with the evaluation block."
    )
    report: dict[str, Any] = {
        "context": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "samples": int(args.samples),
            "max_windows": int(args.max_windows),
        },
        "split_geometry": geom,
        "feature_names": feature_names,
        "window_metric_summary": window_metric_summary,
        "calibration_to_eval_drift_top": drift_calib_eval[:12],
        "train_to_eval_drift_top": drift_train_eval[:12],
        "top_feature_metric_correlations": correlations[:24],
        "binned_state_summary": binned,
        "decision": decision,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(json.dumps(make_serializable(report["window_metric_summary"]), indent=2))
    print(json.dumps(make_serializable(report["split_geometry"]), indent=2))


if __name__ == "__main__":
    main()
