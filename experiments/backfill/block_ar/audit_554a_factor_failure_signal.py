#!/usr/bin/env python
"""554a: audit whether factor history predicts frontier model failure.

This is a deployability diagnostic, not a new generator. It asks whether the
local non-IV factors identified in 553a predict the current IV-only frontier's
actual validation under-inclusion errors after fitting only on pre-validation
calibration windows.
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

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    load_one_day_kernel,
    make_serializable,
)
from experiments.backfill.block_ar.audit_550a_data_framing_causal_state import (  # noqa: E402
    make_windows_np,
    sample_model,
    split_geometry,
)
from experiments.backfill.block_ar.audit_553a_factor_state_signal import (  # noqa: E402
    FACTOR_KEYS,
    factor_history_features,
    iv_history_features,
    rank_corr,
)


FAILURE_TARGETS = [
    "coverage_under_target",
    "upper_miss_rate",
    "lower_miss_rate",
    "model_median_mae",
    "future_abs_move_mean",
    "gt_path_max_jump",
]


def window_interval_miss_metrics(
    samples: np.ndarray,
    future_01: np.ndarray,
    target_coverage: float = 0.90,
) -> dict[str, np.ndarray]:
    """Compute window-level interval misses and stress proxies.

    `upper_miss_rate` is important for IV stress scenarios: it measures realized
    future IV cells above the generated 95th percentile envelope.
    """
    samples = np.asarray(samples, dtype=np.float64)
    future = np.asarray(future_01, dtype=np.float64)
    if samples.ndim != 5:
        raise ValueError("samples must have shape (windows, samples, horizon, rows, cols)")
    if future.ndim != 4:
        raise ValueError("future_01 must have shape (windows, horizon, rows, cols)")
    if samples.shape[0] != future.shape[0] or samples.shape[2:] != future.shape[1:]:
        raise ValueError("samples and future_01 have incompatible window/horizon/cell shapes")

    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    median = np.median(samples, axis=1)
    covered = (future >= q05) & (future <= q95)
    lower_miss = future < q05
    upper_miss = future > q95
    coverage = covered.mean(axis=(1, 2, 3))
    width = (q95 - q05).mean(axis=(1, 2, 3))
    model_mae = np.abs(median - future).mean(axis=(1, 2, 3))
    future_diffs = np.diff(future, axis=1)
    if future_diffs.shape[1] == 0:
        future_abs_move = np.zeros(future.shape[0], dtype=np.float64)
        future_max_jump = np.zeros(future.shape[0], dtype=np.float64)
    else:
        future_abs_move = np.abs(future_diffs).mean(axis=(1, 2, 3))
        future_max_jump = np.abs(future_diffs).max(axis=(1, 2, 3))
    return {
        "coverage90": coverage,
        "coverage_under_target": np.maximum(0.0, float(target_coverage) - coverage),
        "upper_miss_rate": upper_miss.mean(axis=(1, 2, 3)),
        "lower_miss_rate": lower_miss.mean(axis=(1, 2, 3)),
        "width90": width,
        "model_median_mae": model_mae,
        "future_abs_move_mean": future_abs_move,
        "gt_path_max_jump": future_max_jump,
        "future_level_mean": future.mean(axis=(1, 2, 3)),
    }


def top_feature_failure_correlations(
    features: np.ndarray,
    feature_names: list[str],
    metrics: dict[str, np.ndarray],
    target_names: list[str] | None = None,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """Rank history-only feature associations with model failure metrics."""
    if target_names is None:
        target_names = [name for name in FAILURE_TARGETS if name in metrics]
    rows: list[dict[str, Any]] = []
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
    return sorted(rows, key=lambda row: float(row["abs_spearman"]), reverse=True)[:top_k]


def fit_ridge_score(
    train_features: np.ndarray,
    train_target: np.ndarray,
    eval_features: np.ndarray,
    l2: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a small chronological ridge score on calibration features only."""
    train_x = np.asarray(train_features, dtype=np.float64)
    eval_x = np.asarray(eval_features, dtype=np.float64)
    y = np.asarray(train_target, dtype=np.float64)
    if train_x.ndim != 2 or eval_x.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("train and eval feature dimensions differ")
    if train_x.shape[0] != y.shape[0]:
        raise ValueError("train target length does not match train features")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    xz = (train_x - mean) / std
    ez = (eval_x - mean) / std
    design = np.concatenate([np.ones((xz.shape[0], 1)), xz], axis=1)
    penalty = np.eye(design.shape[1]) * float(l2)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    eval_design = np.concatenate([np.ones((ez.shape[0], 1)), ez], axis=1)
    score = eval_design @ beta
    return score.astype(np.float64), {
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "intercept": float(beta[0]),
        "coef": beta[1:].tolist(),
        "l2": float(l2),
        "train_target_mean": float(y.mean()),
    }


def score_alignment(
    score: np.ndarray,
    metrics: dict[str, np.ndarray],
    target_name: str = "coverage_under_target",
) -> dict[str, Any]:
    """Summarize out-of-sample score alignment with a failure target."""
    target = np.asarray(metrics[target_name], dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if score.shape[0] != target.shape[0]:
        raise ValueError("score and metric target lengths differ")
    q1, q2 = np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])
    low = score <= q1
    high = score > q2
    low_mean = float(target[low].mean()) if int(low.sum()) else 0.0
    high_mean = float(target[high].mean()) if int(high.sum()) else 0.0
    return {
        "target": target_name,
        "spearman": rank_corr(score, target),
        "low_score_target_mean": low_mean,
        "high_score_target_mean": high_mean,
        "high_low_lift": float(high_mean - low_mean),
        "low_n": int(low.sum()),
        "high_n": int(high.sum()),
    }


def failure_signal_summary(
    factor_rows: list[dict[str, Any]],
    min_abs_spearman: float = 0.25,
) -> dict[str, Any]:
    """Read whether factor histories carry material model-failure signal."""
    failure_rows = [
        row
        for row in factor_rows
        if row["target"] in {"coverage_under_target", "upper_miss_rate", "lower_miss_rate"}
    ]
    best = max(failure_rows, key=lambda row: float(row["abs_spearman"]), default=None)
    best_abs = float(best["abs_spearman"]) if best else 0.0
    return {
        "best_failure_signal": best,
        "best_failure_abs_spearman": best_abs,
        "min_abs_spearman": float(min_abs_spearman),
        "factor_failure_signal_material": bool(best_abs >= float(min_abs_spearman)),
    }


def _metric_summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        f"{name}_mean": float(np.mean(value))
        for name, value in metrics.items()
        if np.asarray(value).ndim == 1
    } | {
        f"{name}_p90": float(np.quantile(value, 0.90))
        for name, value in metrics.items()
        if np.asarray(value).ndim == 1
    }


def _prepare_features(
    raw: np.lib.npyio.NpzFile,
    surfaces: np.ndarray,
    indices: np.ndarray,
    history_len: int,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    factor_keys = [key for key in FACTOR_KEYS if key in raw.files]
    factors = {key: raw[key].astype(np.float64) for key in factor_keys}
    factor_features, factor_names = factor_history_features(factors, indices, history_len)
    iv_features, iv_names = iv_history_features(surfaces, indices, history_len)
    return factor_features, factor_names, iv_features, iv_names


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    factor_summary = report["factor_failure_signal"]
    factor_align = report["oos_factor_score_alignment"]
    iv_align = report["oos_iv_score_alignment"]
    lines = [
        "# 554a Factor Failure-Signal Audit",
        "",
        "## Question",
        "",
        "Do observed non-IV factor histories predict where the current IV-only frontier actually under-includes future scenarios?",
        "",
        "## Setup",
        "",
        f"- model: `{report['context']['model_type']}`",
        f"- checkpoint: `{report['context']['checkpoint']}`",
        f"- calibration windows: `{report['context']['calibration_windows']}`",
        f"- validation windows: `{report['context']['eval_windows']}`",
        f"- samples per window: `{report['context']['samples']}`",
        "",
        "## Frontier Failure Geometry",
        "",
        f"- validation mean 90% coverage: `{report['eval_metric_summary']['coverage90_mean']:.3f}`",
        f"- validation mean upper stress miss rate: `{report['eval_metric_summary']['upper_miss_rate_mean']:.3f}`",
        f"- validation mean lower miss rate: `{report['eval_metric_summary']['lower_miss_rate_mean']:.3f}`",
        "",
        "## Factor Failure Signal",
        "",
        f"- material factor failure signal: `{factor_summary['factor_failure_signal_material']}`",
        f"- best failure abs Spearman: `{factor_summary['best_failure_abs_spearman']:.3f}`",
        f"- best failure row: `{factor_summary['best_failure_signal']}`",
        f"- OOS factor score Spearman to undercoverage: `{factor_align['spearman']:.3f}`",
        f"- OOS factor score high-low undercoverage lift: `{factor_align['high_low_lift']:.4f}`",
        f"- OOS IV-only score Spearman to undercoverage: `{iv_align['spearman']:.3f}`",
        f"- OOS IV-only score high-low undercoverage lift: `{iv_align['high_low_lift']:.4f}`",
        "",
        "## Top Factor Associations",
        "",
    ]
    for row in report["top_factor_failure_correlations"][:12]:
        lines.append(
            f"- `{row['feature']}` vs `{row['target']}`: Spearman `{row['spearman']:.3f}`"
        )
    lines.extend(["", "## Top IV-History Associations", ""])
    for row in report["top_iv_failure_correlations"][:8]:
        lines.append(
            f"- `{row['feature']}` vs `{row['target']}`: Spearman `{row['spearman']:.3f}`"
        )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--target_coverage", type=float, default=0.90)
    parser.add_argument("--ridge_l2", type=float, default=1.0)
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
    calib_end = int(geom["calibration_end"])
    calib_start = max(int(geom["calibration_start"]), calib_end - int(args.max_windows) + 1)
    eval_start = int(geom["official_val_start"])
    n_eval = min(int(args.max_windows), int(args.val_size))
    calib_indices = np.arange(calib_start, calib_start + n_eval)
    eval_indices = np.arange(eval_start, eval_start + n_eval)
    calib_history, calib_future = make_windows_np(
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
    calib_factor, factor_names, calib_iv, iv_names = _prepare_features(
        raw,
        surfaces,
        calib_indices,
        args.history_len,
    )
    eval_factor, _, eval_iv, _ = _prepare_features(raw, surfaces, eval_indices, args.history_len)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    model.eval()
    calib_samples = sample_model(
        model=model,
        history_01=calib_history,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    eval_samples = sample_model(
        model=model,
        history_01=eval_history,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    calib_metrics = window_interval_miss_metrics(
        calib_samples,
        calib_future,
        target_coverage=args.target_coverage,
    )
    eval_metrics = window_interval_miss_metrics(
        eval_samples,
        eval_future,
        target_coverage=args.target_coverage,
    )

    factor_rows = top_feature_failure_correlations(eval_factor, factor_names, eval_metrics)
    iv_rows = top_feature_failure_correlations(eval_iv, iv_names, eval_metrics)
    factor_failure = failure_signal_summary(factor_rows)
    factor_score, factor_fit = fit_ridge_score(
        calib_factor,
        calib_metrics["coverage_under_target"],
        eval_factor,
        l2=args.ridge_l2,
    )
    iv_score, iv_fit = fit_ridge_score(
        calib_iv,
        calib_metrics["coverage_under_target"],
        eval_iv,
        l2=args.ridge_l2,
    )
    factor_align = score_alignment(factor_score, eval_metrics)
    iv_align = score_alignment(iv_score, eval_metrics)

    factor_better_oos = (
        float(factor_align["spearman"]) > float(iv_align["spearman"])
        and float(factor_align["high_low_lift"]) > 0.0
    )
    if factor_failure["factor_failure_signal_material"] and factor_better_oos:
        decision = (
            "Factor history predicts the frozen IV-only frontier's failure out of sample. "
            "The next deployable research step should build a factor-conditioned generator "
            "or a factor-conditioned stress envelope, not another IV-only calibration knob."
        )
    elif factor_failure["factor_failure_signal_material"]:
        decision = (
            "Factor history has validation failure signal, but the pre-validation ridge score "
            "does not cleanly dominate IV-history scoring out of sample. The next step should "
            "train a factor-conditioned core rather than rely on post-hoc factor calibration."
        )
    else:
        decision = (
            "Factor histories do not materially identify the current frontier's failure windows. "
            "The factor-conditioned route is not yet justified without broader data or a new law."
        )

    report: dict[str, Any] = {
        "context": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "calibration_windows": int(calib_indices.shape[0]),
            "eval_windows": int(eval_indices.shape[0]),
            "samples": int(args.samples),
            "target_coverage": float(args.target_coverage),
            "ridge_l2": float(args.ridge_l2),
        },
        "split_geometry": geom,
        "factor_names": factor_names,
        "iv_feature_names": iv_names,
        "calib_metric_summary": _metric_summary(calib_metrics),
        "eval_metric_summary": _metric_summary(eval_metrics),
        "top_factor_failure_correlations": factor_rows,
        "top_iv_failure_correlations": iv_rows,
        "factor_failure_signal": factor_failure,
        "oos_factor_score_alignment": factor_align,
        "oos_iv_score_alignment": iv_align,
        "factor_ridge_fit": factor_fit,
        "iv_ridge_fit": iv_fit,
        "decision": decision,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(
        json.dumps(
            make_serializable(
                {
                    "eval_coverage90_mean": report["eval_metric_summary"]["coverage90_mean"],
                    "factor_failure_signal": factor_failure,
                    "oos_factor_score_alignment": factor_align,
                    "oos_iv_score_alignment": iv_align,
                    "decision": decision,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
