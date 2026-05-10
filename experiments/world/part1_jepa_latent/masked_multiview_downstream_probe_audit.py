from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.world_data import (  # noqa: E402
    WorldWindowBatch,
    build_iv_world_windows,
)
from experiments.world.evaluation.masked_multiview_data import (
    build_masked_multiview_batch,
)  # noqa: E402
from experiments.world.evaluation.part1_metrics import (
    representation_health_metrics,
)  # noqa: E402
from experiments.world.part1_jepa_latent.context_probe_audit import (
    ridge_probe_predict,
)  # noqa: E402
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    concatenate_feature_blocks,
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
    regression_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_mask_artifact_audit import (  # noqa: E402
    classification_metrics,
    fit_predict_multiclass_ridge,
)


def make_extended_future_targets(
    past_surface: np.ndarray,
    future_surface: np.ndarray,
    *,
    regime_labels: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    past = np.asarray(past_surface, dtype=np.float32)
    future = np.asarray(future_surface, dtype=np.float32)
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError("past_surface and future_surface must have shape (N, T, C)")
    if past.shape[0] != future.shape[0] or past.shape[2] != future.shape[2]:
        raise ValueError("past and future must share sample count and channel count")
    last = past[:, -1, :]
    path = np.concatenate([last[:, None, :], future], axis=1)
    step = np.diff(path, axis=1)
    running_peak = np.maximum.accumulate(path, axis=1)
    drawdown = np.max(running_peak - path, axis=1)
    regression = {
        "future_mean_delta": (future.mean(axis=1) - last).astype(np.float32),
        "future_range": (future.max(axis=1) - future.min(axis=1)).astype(np.float32),
        "future_terminal_delta": (future[:, -1, :] - last).astype(np.float32),
        "future_max_abs_step": np.max(np.abs(step), axis=1).astype(np.float32),
        "future_drawdown": drawdown.astype(np.float32),
    }
    classification: dict[str, np.ndarray] = {}
    if regime_labels is not None:
        classification["regime_label"] = np.asarray(regime_labels, dtype=np.int64)
    return {"regression": regression, "classification": classification}


def _feature_sets(
    encoded: np.ndarray,
    iv_batch: WorldWindowBatch,
) -> dict[str, np.ndarray]:
    barlow_last = encoded[:, -1, :]
    barlow_mean = encoded.mean(axis=1)
    raw_surface_last = iv_batch.past_window[:, -1, :]
    raw_surface_flat = iv_batch.past_window.reshape(iv_batch.past_window.shape[0], -1)
    return {
        "barlow_clean_last": barlow_last,
        "barlow_clean_mean": barlow_mean,
        "raw_surface_last": raw_surface_last,
        "raw_surface_flat": raw_surface_flat,
        "raw_surface_last_plus_barlow_clean_last": concatenate_feature_blocks(
            raw_surface_last,
            barlow_last,
        ),
    }


def _probe_regression_targets(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature_name, train_x in train_features.items():
        rows: dict[str, Any] = {}
        for target_name, train_y in train_targets.items():
            pred = ridge_probe_predict(
                train_x,
                train_y,
                val_features[feature_name],
                alpha=alpha,
            )
            rows[target_name] = regression_metrics(pred, val_targets[target_name])
        out[feature_name] = {
            "feature_shape": {
                "train": list(train_x.shape),
                "val": list(val_features[feature_name].shape),
            },
            "health": representation_health_metrics(val_features[feature_name]),
            "targets": rows,
        }
    return out


def _probe_classification_targets(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature_name, train_x in train_features.items():
        rows: dict[str, Any] = {}
        for target_name, train_y in train_targets.items():
            pred = fit_predict_multiclass_ridge(
                train_x,
                train_y,
                val_features[feature_name],
                alpha=alpha,
            )
            rows[target_name] = classification_metrics(pred, val_targets[target_name])
        out[feature_name] = rows
    return out


def _mean_target_baselines(
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target_name, train_y in train_targets.items():
        pred = np.repeat(
            train_y.mean(axis=0, keepdims=True),
            val_targets[target_name].shape[0],
            axis=0,
        )
        out[target_name] = regression_metrics(pred, val_targets[target_name])
    return out


def audit_downstream_probes(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = load_direct_barlow_checkpoint(args.checkpoint, device=device)
    train_masked = build_masked_multiview_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
        normalize=True,
    )
    val_masked = build_masked_multiview_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
        normalize=True,
    )
    train_iv = build_iv_world_windows(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        normalize=True,
    )
    val_iv = build_iv_world_windows(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        normalize=True,
    )
    train_encoded = encode_clean_masked_windows(
        model,
        train_masked,
        batch_size=args.batch_size,
        device=device,
    )
    val_encoded = encode_clean_masked_windows(
        model,
        val_masked,
        batch_size=args.batch_size,
        device=device,
    )
    train_features = _feature_sets(train_encoded, train_iv)
    val_features = _feature_sets(val_encoded, val_iv)
    train_targets = make_extended_future_targets(
        train_iv.past_window,
        train_iv.future_window,
        regime_labels=train_iv.regime_label,
    )
    val_targets = make_extended_future_targets(
        val_iv.past_window,
        val_iv.future_window,
        regime_labels=val_iv.regime_label,
    )
    return {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "train_shape": {
            "masked": list(train_masked.clean_values.shape),
            "iv_past": list(train_iv.past_window.shape),
            "iv_future": list(train_iv.future_window.shape),
            "encoded": list(train_encoded.shape),
        },
        "val_shape": {
            "masked": list(val_masked.clean_values.shape),
            "iv_past": list(val_iv.past_window.shape),
            "iv_future": list(val_iv.future_window.shape),
            "encoded": list(val_encoded.shape),
        },
        "regression_probe_metrics": _probe_regression_targets(
            train_features,
            val_features,
            train_targets["regression"],
            val_targets["regression"],
            alpha=args.ridge_alpha,
        ),
        "classification_probe_metrics": _probe_classification_targets(
            train_features,
            val_features,
            train_targets["classification"],
            val_targets["classification"],
            alpha=args.ridge_alpha,
        ),
        "mean_target_baseline": _mean_target_baselines(
            train_targets["regression"],
            val_targets["regression"],
        ),
    }


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    features = (
        "barlow_clean_last",
        "raw_surface_last",
        "raw_surface_last_plus_barlow_clean_last",
    )
    targets = (
        "future_mean_delta",
        "future_range",
        "future_terminal_delta",
        "future_max_abs_step",
        "future_drawdown",
    )
    lines = [
        f"# {title}",
        "",
        "## Objective Family",
        "",
        "`downstream_probe` audit for frozen Part 1 representations.",
        "",
        "## Regression Probes",
        "",
        "| feature | target | MSE | R2 |",
        "| --- | --- | ---: | ---: |",
    ]
    for feature_name in features:
        feature = result["regression_probe_metrics"][feature_name]
        for target_name in targets:
            metrics = feature["targets"][target_name]
            lines.append(
                f"| {feature_name} | {target_name} | {_fmt(metrics['mse'])} | {_fmt(metrics['r2'])} |"
            )
    lines.extend(
        [
            "",
            "## Regime Classification Probe",
            "",
            "| feature | target | accuracy | majority | lift | macro recall |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for feature_name in features:
        target_rows = result["classification_probe_metrics"].get(feature_name, {})
        for target_name, metrics in target_rows.items():
            lines.append(
                "| {feature} | {target} | {acc} | {maj} | {lift} | {macro} |".format(
                    feature=feature_name,
                    target=target_name,
                    acc=_fmt(metrics["accuracy"]),
                    maj=_fmt(metrics["majority_accuracy"]),
                    lift=_fmt(metrics["accuracy_lift"]),
                    macro=_fmt(metrics["macro_recall"]),
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is downstream evaluation only. It should not become a Part 1",
            "pretraining objective.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extended frozen downstream probe audit for HEAD070"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/masked_multiview_downstream_probe_head085.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head085_downstream_probe_coverage.md"
        ),
    )
    parser.add_argument(
        "--report-title", default="World Model HEAD085: Downstream Probe Coverage"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = audit_downstream_probes(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        render_markdown(result, title=args.report_title), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
