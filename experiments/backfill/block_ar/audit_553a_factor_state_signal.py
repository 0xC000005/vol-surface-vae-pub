#!/usr/bin/env python
"""553a: audit non-IV factor signal for future IV stress states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402


FACTOR_KEYS = ["ret", "price", "slopes", "skews", "levels"]


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return 0.0
    x_f = x[finite]
    y_f = y[finite]
    if np.ptp(x_f) <= 0.0 or np.ptp(y_f) <= 0.0:
        return 0.0
    xr = _rankdata(x_f)
    yr = _rankdata(y_f)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = float(np.sqrt(np.sum(xr * xr) * np.sum(yr * yr)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(xr * yr) / denom)


def factor_history_features(
    factors: dict[str, np.ndarray],
    indices: np.ndarray,
    history_len: int,
) -> tuple[np.ndarray, list[str]]:
    """Build simple history-only summaries for each available non-IV factor."""
    feature_blocks: list[np.ndarray] = []
    names: list[str] = []
    offsets = np.arange(int(history_len))[None, :]
    for key in FACTOR_KEYS:
        if key not in factors:
            continue
        series = np.asarray(factors[key], dtype=np.float64)
        window = series[indices[:, None] + offsets]
        diffs = np.diff(window, axis=1)
        abs_diffs = np.abs(diffs)
        block = np.stack(
            [
                window[:, -1],
                window.mean(axis=1),
                window.std(axis=1),
                window[:, -1] - window[:, 0],
                abs_diffs.mean(axis=1) if abs_diffs.shape[1] else np.zeros(indices.shape[0]),
                np.quantile(abs_diffs, 0.90, axis=1) if abs_diffs.shape[1] else np.zeros(indices.shape[0]),
            ],
            axis=1,
        )
        feature_blocks.append(block)
        names.extend(
            [
                f"{key}_hist_last",
                f"{key}_hist_mean",
                f"{key}_hist_std",
                f"{key}_hist_trend",
                f"{key}_hist_abs_move_mean",
                f"{key}_hist_abs_q90",
            ]
        )
    if not feature_blocks:
        return np.empty((indices.shape[0], 0), dtype=np.float64), []
    return np.concatenate(feature_blocks, axis=1), names


def iv_history_features(
    surfaces: np.ndarray,
    indices: np.ndarray,
    history_len: int,
) -> tuple[np.ndarray, list[str]]:
    offsets = np.arange(int(history_len))[None, :]
    history = np.asarray(surfaces, dtype=np.float64)[indices[:, None] + offsets]
    daily_mean = history.mean(axis=(2, 3))
    diffs = np.diff(history, axis=1)
    abs_diffs = np.abs(diffs)
    last = history[:, -1]
    block = np.stack(
        [
            last.mean(axis=(1, 2)),
            last.std(axis=(1, 2)),
            last.max(axis=(1, 2)) - last.min(axis=(1, 2)),
            daily_mean[:, -1] - daily_mean[:, 0],
            abs_diffs.mean(axis=(1, 2, 3)),
            np.quantile(abs_diffs.reshape(history.shape[0], -1), 0.90, axis=1),
        ],
        axis=1,
    )
    return block, [
        "iv_hist_last_mean",
        "iv_hist_last_std",
        "iv_hist_surface_range",
        "iv_hist_trend_mean",
        "iv_hist_abs_move_mean",
        "iv_hist_abs_move_q90",
    ]


def future_stress_targets(
    surfaces: np.ndarray,
    factors: dict[str, np.ndarray],
    indices: np.ndarray,
    history_len: int,
    future_len: int,
) -> tuple[np.ndarray, list[str]]:
    future_offsets = int(history_len) + np.arange(int(future_len))[None, :]
    surface_arr = np.asarray(surfaces, dtype=np.float64)
    if surface_arr.ndim == 4 and surface_arr.shape[1] == 1:
        surface_arr = surface_arr[:, 0]
    future = surface_arr[indices[:, None] + future_offsets]
    iv_diffs = np.diff(future, axis=1)
    iv_abs = np.abs(iv_diffs)
    targets = [
        future.mean(axis=(1, 2, 3)),
        future[:, -1].mean(axis=(1, 2)),
        iv_abs.mean(axis=(1, 2, 3)),
        np.quantile(iv_abs.reshape(future.shape[0], -1), 0.90, axis=1),
        iv_abs.max(axis=(1, 2, 3)),
    ]
    names = [
        "future_iv_level_mean",
        "future_iv_horizon_end_mean",
        "future_iv_abs_move_mean",
        "future_iv_abs_move_q90",
        "future_iv_max_jump",
    ]
    if "ret" in factors:
        ret_future = np.asarray(factors["ret"], dtype=np.float64)[indices[:, None] + future_offsets]
        abs_ret = np.abs(ret_future)
        targets.extend(
            [
                ret_future.sum(axis=1),
                abs_ret.mean(axis=1),
                np.quantile(abs_ret, 0.90, axis=1),
                abs_ret.max(axis=1),
            ]
        )
        names.extend(
            [
                "future_ret_sum",
                "future_ret_abs_mean",
                "future_ret_abs_q90",
                "future_ret_abs_max",
            ]
        )
    return np.stack(targets, axis=1), names


def top_feature_targets(
    features: np.ndarray,
    feature_names: list[str],
    targets: np.ndarray,
    target_names: list[str],
    top_k: int = 30,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_names):
        for target_idx, target_name in enumerate(target_names):
            corr = rank_corr(features[:, feature_idx], targets[:, target_idx])
            rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "spearman": corr,
                    "abs_spearman": abs(corr),
                }
            )
    return sorted(rows, key=lambda row: float(row["abs_spearman"]), reverse=True)[:top_k]


def incremental_signal_summary(
    iv_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best_iv = max((float(row["abs_spearman"]) for row in iv_rows), default=0.0)
    best_factor = max((float(row["abs_spearman"]) for row in factor_rows), default=0.0)
    return {
        "best_iv_abs_spearman": best_iv,
        "best_factor_abs_spearman": best_factor,
        "factor_signal_beats_iv_signal": bool(best_factor > best_iv),
        "factor_signal_material": bool(best_factor >= 0.25),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 553a Factor-State Signal Audit",
        "",
        "## Question",
        "",
        "Do the existing non-IV factors contain history-only signal for future IV stress states?",
        "",
        "## Data",
        "",
        f"- windows: `{report['n_windows']}`",
        f"- history length: `{report['history_len']}`",
        f"- future length: `{report['future_len']}`",
        f"- factor keys: `{', '.join(report['factor_keys'])}`",
        "",
        "## Incremental Signal",
        "",
        f"- best IV-history absolute Spearman: `{report['incremental_signal']['best_iv_abs_spearman']:.3f}`",
        f"- best factor-history absolute Spearman: `{report['incremental_signal']['best_factor_abs_spearman']:.3f}`",
        f"- factor signal material: `{report['incremental_signal']['factor_signal_material']}`",
        f"- factor beats IV-only signal: `{report['incremental_signal']['factor_signal_beats_iv_signal']}`",
        "",
        "## Strongest Factor Signals",
        "",
    ]
    for row in report["top_factor_signal"][:12]:
        lines.append(
            f"- `{row['feature']}` vs `{row['target']}`: Spearman `{row['spearman']:.3f}`"
        )
    lines.extend(["", "## Strongest IV-History Signals", ""])
    for row in report["top_iv_signal"][:10]:
        lines.append(
            f"- `{row['feature']}` vs `{row['target']}`: Spearman `{row['spearman']:.3f}`"
        )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float64)
    factor_keys = [key for key in FACTOR_KEYS if key in raw.files]
    factors = {key: raw[key].astype(np.float64) for key in factor_keys}
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_start = max_train_idx - args.val_size
    n_windows = min(int(args.max_windows), int(args.val_size))
    indices = np.arange(val_start, val_start + n_windows)

    iv_features, iv_names = iv_history_features(surfaces, indices, args.history_len)
    factor_features, factor_names = factor_history_features(factors, indices, args.history_len)
    targets, target_names = future_stress_targets(
        surfaces=surfaces,
        factors=factors,
        indices=indices,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    top_iv = top_feature_targets(iv_features, iv_names, targets, target_names)
    top_factor = top_feature_targets(factor_features, factor_names, targets, target_names)
    signal = incremental_signal_summary(top_iv, top_factor)
    if signal["factor_signal_material"]:
        decision = (
            "Existing local non-IV factors contain material stress-state signal. The next "
            "model direction should be a factor-conditioned scenario generator or joint "
            "IV+factor path model, not another IV-only width/calibration wrapper."
        )
    else:
        decision = (
            "Existing local non-IV factors do not provide enough signal by themselves. "
            "The next direction would require broader panel data rather than another local wrapper."
        )
    report = {
        "n_windows": int(n_windows),
        "history_len": int(args.history_len),
        "future_len": int(args.future_len),
        "factor_keys": factor_keys,
        "target_names": target_names,
        "top_factor_signal": top_factor,
        "top_iv_signal": top_iv,
        "incremental_signal": signal,
        "decision": decision,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(json.dumps(make_serializable(report["incremental_signal"]), indent=2))


if __name__ == "__main__":
    main()
