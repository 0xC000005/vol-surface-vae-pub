#!/usr/bin/env python
"""700a: tri-scope conditional-signal audit for normalized innovations.

This is a data/objective diagnostic, not a model architecture. It asks whether
future normalized innovations are detectably tied to matched history for
`iv_only`, `anchor_only`, and `joint38`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    make_serializable,
    select_rollout_indices,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    clean_nonpositive_log_level_factors,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    build_increment_coordinate_block,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    select_normalized_innovation_scope,
)


def ordinal_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size < 3 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    rx = np.argsort(np.argsort(x_arr)).astype(np.float64)
    ry = np.argsort(np.argsort(y_arr)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_abs_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def _improvement(reference_mae: float, candidate_mae: float) -> float:
    if not np.isfinite(reference_mae) or abs(reference_mae) < 1e-12:
        return 0.0
    return float((reference_mae - candidate_mae) / reference_mae * 100.0)


def conditional_signal_metrics(
    history_norm: np.ndarray,
    future_norm: np.ndarray,
    *,
    nearest_neighbor_prediction: np.ndarray | None = None,
) -> dict[str, float]:
    history = np.asarray(history_norm, dtype=np.float32)
    future = np.asarray(future_norm, dtype=np.float32)
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history_norm and future_norm must have rank 3")
    if history.shape[0] != future.shape[0] or history.shape[2] != future.shape[2]:
        raise ValueError("history_norm and future_norm dimensions are inconsistent")

    zero_pred = np.zeros_like(future)
    last_step_pred = np.repeat(history[:, -1:, :], future.shape[1], axis=1)
    hist_mean_pred = np.repeat(history.mean(axis=1, keepdims=True), future.shape[1], axis=1)
    rolled_future_pred = np.roll(future, shift=1, axis=0)
    global_median_pred = np.broadcast_to(
        np.median(future, axis=0, keepdims=True),
        future.shape,
    )

    zero_mae = mean_abs_error(zero_pred, future)
    last_step_mae = mean_abs_error(last_step_pred, future)
    hist_mean_mae = mean_abs_error(hist_mean_pred, future)
    rolled_mae = mean_abs_error(rolled_future_pred, future)
    global_median_mae = mean_abs_error(global_median_pred, future)

    history_activity = np.mean(history * history, axis=(1, 2))
    future_activity = np.mean(future * future, axis=(1, 2))

    out: dict[str, float] = {
        "zero_mae": zero_mae,
        "last_step_mae": last_step_mae,
        "history_mean_mae": hist_mean_mae,
        "rolled_future_mae": rolled_mae,
        "global_median_mae": global_median_mae,
        "last_step_improvement_vs_rolled_pct": _improvement(rolled_mae, last_step_mae),
        "history_mean_improvement_vs_rolled_pct": _improvement(rolled_mae, hist_mean_mae),
        "zero_improvement_vs_rolled_pct": _improvement(rolled_mae, zero_mae),
        "global_median_improvement_vs_rolled_pct": _improvement(
            rolled_mae,
            global_median_mae,
        ),
        "history_future_activity_spearman": ordinal_spearman(
            history_activity,
            future_activity,
        ),
    }
    if nearest_neighbor_prediction is not None:
        nn_mae = mean_abs_error(nearest_neighbor_prediction, future)
        out["nearest_neighbor_mae"] = nn_mae
        out["nearest_neighbor_improvement_vs_rolled_pct"] = _improvement(
            rolled_mae,
            nn_mae,
        )
        out["nearest_neighbor_improvement_vs_global_median_pct"] = _improvement(
            global_median_mae,
            nn_mae,
        )
    return out


def nearest_neighbor_future_prediction(
    eval_history: np.ndarray,
    reference_history: np.ndarray,
    reference_future: np.ndarray,
    *,
    eval_indices: np.ndarray,
    reference_indices: np.ndarray,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    eval_flat = np.asarray(eval_history, dtype=np.float32).reshape(eval_history.shape[0], -1)
    ref_flat = np.asarray(reference_history, dtype=np.float32).reshape(reference_history.shape[0], -1)
    ref_future = np.asarray(reference_future, dtype=np.float32)
    eval_idx = np.asarray(eval_indices)
    ref_idx = np.asarray(reference_indices)
    k_eff = max(1, min(int(k), int(ref_flat.shape[0])))
    preds: list[np.ndarray] = []
    ref_norm = np.sum(ref_flat * ref_flat, axis=1)[None, :]
    chunk = max(1, int(chunk_size))
    for start in range(0, eval_flat.shape[0], chunk):
        end = min(start + chunk, eval_flat.shape[0])
        q = eval_flat[start:end]
        dist = np.sum(q * q, axis=1, keepdims=True) + ref_norm - 2.0 * q @ ref_flat.T
        same_index = eval_idx[start:end, None] == ref_idx[None, :]
        dist[same_index] = np.inf
        order = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        preds.append(np.median(ref_future[order], axis=1).astype(np.float32))
    return np.concatenate(preds, axis=0)


def _load_panel(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    if args.clean_nonpositive_log_levels:
        panel, _report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
            positive_level_policy=args.positive_level_policy,
        )
    return panel, columns


def _scope_arrays(
    panel: np.ndarray,
    columns: list[str],
    args: argparse.Namespace,
    *,
    scope: str,
    split: str,
    max_windows: int | None,
) -> dict[str, Any]:
    indices = select_rollout_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        max_windows=max_windows,
        split=split,
    )
    block = build_increment_coordinate_block(
        panel,
        columns,
        indices,
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        iv_count=int(args.iv_count),
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=float(args.iv_lower_bound),
        iv_upper_bound=float(args.iv_upper_bound),
    )
    scale_half_life = None if float(args.scale_half_life) <= 0.0 else float(args.scale_half_life)
    (
        history_level,
        history_norm,
        future_level,
        future_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
    ) = select_normalized_innovation_scope(
        block,
        scope,
        int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=float(args.scale_floor),
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    return {
        "indices": indices,
        "history_level": history_level,
        "history_norm": history_norm,
        "future_level": future_level,
        "future_norm": future_norm,
        "center": center,
        "scale": scale,
        "drift_feature": drift_feature,
        "history_raw": history_raw,
        "spec_names": [spec.name for spec in specs],
    }


def audit_scope_split(
    panel: np.ndarray,
    columns: list[str],
    args: argparse.Namespace,
    *,
    scope: str,
    split: str,
) -> dict[str, Any]:
    eval_arrays = _scope_arrays(
        panel,
        columns,
        args,
        scope=scope,
        split=split,
        max_windows=int(args.max_windows),
    )
    ref_arrays = _scope_arrays(
        panel,
        columns,
        args,
        scope=scope,
        split="train",
        max_windows=None,
    )
    max_ref = int(args.max_reference_windows)
    if max_ref > 0 and ref_arrays["indices"].shape[0] > max_ref:
        for key in ["indices", "history_norm", "future_norm"]:
            ref_arrays[key] = ref_arrays[key][-max_ref:]

    nn_pred = nearest_neighbor_future_prediction(
        eval_arrays["history_norm"],
        ref_arrays["history_norm"],
        ref_arrays["future_norm"],
        eval_indices=eval_arrays["indices"],
        reference_indices=ref_arrays["indices"],
        k=int(args.nn_k),
        chunk_size=int(args.nn_chunk_size),
    )
    metrics = conditional_signal_metrics(
        eval_arrays["history_norm"],
        eval_arrays["future_norm"],
        nearest_neighbor_prediction=nn_pred,
    )
    return {
        "scope": scope,
        "split": split,
        "n_eval_windows": int(eval_arrays["indices"].shape[0]),
        "n_reference_windows": int(ref_arrays["indices"].shape[0]),
        "n_channels": int(eval_arrays["history_norm"].shape[-1]),
        "history_len": int(eval_arrays["history_norm"].shape[1]),
        "future_len": int(eval_arrays["future_norm"].shape[1]),
        "index_start": int(eval_arrays["indices"][0]),
        "index_end": int(eval_arrays["indices"][-1]),
        "metrics": metrics,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# 700a Tri-Scope Conditional-Signal Audit",
        "",
        f"- scopes: `{', '.join(result['config']['scopes'])}`",
        f"- splits: `{', '.join(result['config']['splits'])}`",
        f"- nearest-neighbor k/reference windows: `{result['config']['nn_k']}` / `{result['config']['max_reference_windows']}`",
        "",
        "| scope | split | channels | eval windows | NN imp vs rolled | NN imp vs global | last-step imp | hist/future activity rho |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["audits"]:
        m = row["metrics"]
        lines.append(
            f"| {row['scope']} | {row['split']} | {row['n_channels']} | {row['n_eval_windows']} | "
            f"{m['nearest_neighbor_improvement_vs_rolled_pct']:.2f}% | "
            f"{m['nearest_neighbor_improvement_vs_global_median_pct']:.2f}% | "
            f"{m['last_step_improvement_vs_rolled_pct']:.2f}% | "
            f"{m['history_future_activity_spearman']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopes", nargs="+", choices=["iv_only", "anchor_only", "joint38"], default=["iv_only", "anchor_only", "joint38"])
    parser.add_argument("--splits", nargs="+", choices=["val", "train_tail", "train"], default=["val", "train_tail"])
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--max_reference_windows", type=int, default=1500)
    parser.add_argument("--nn_k", type=int, default=25)
    parser.add_argument("--nn_chunk_size", type=int, default=32)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    panel, columns = _load_panel(args)
    audits = [
        audit_scope_split(panel, columns, args, scope=scope, split=split)
        for scope in args.scopes
        for split in args.splits
    ]
    result = {
        "config": {
            "scopes": args.scopes,
            "splits": args.splits,
            "test_start": int(args.test_start),
            "val_size": int(args.val_size),
            "history_len": int(args.history_len),
            "future_len": int(args.future_len),
            "max_windows": int(args.max_windows),
            "max_reference_windows": int(args.max_reference_windows),
            "nn_k": int(args.nn_k),
            "coordinate": "history-normalized encoded innovation",
            "center_mode": args.center_mode,
            "scale_half_life": None if float(args.scale_half_life) <= 0.0 else float(args.scale_half_life),
            "iv_transform": args.iv_transform,
        },
        "audits": audits,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(result), indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps(make_serializable(result), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
