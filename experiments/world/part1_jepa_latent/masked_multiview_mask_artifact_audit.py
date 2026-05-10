from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_smoke import (  # noqa: E402
    DirectMaskedMultiviewBarlowModel,
    encode_direct_barlow_split,
)


def _standardize_train_val(
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_features, dtype=np.float64)
    val = np.asarray(val_features, dtype=np.float64)
    if train.ndim != 2 or val.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if train.shape[1] != val.shape[1]:
        raise ValueError("train and val feature dimensions must match")
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (train - mean) / scale, (val - mean) / scale


def fit_predict_multiclass_ridge(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    train_x, val_x = _standardize_train_val(train_features, val_features)
    train_y = np.asarray(train_labels, dtype=object).reshape(-1)
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train_features and train_labels must share row count")
    classes = np.asarray(sorted({str(x) for x in train_y.tolist()}), dtype=object)
    if classes.size < 2:
        raise ValueError("need at least two classes for multiclass ridge")
    class_index = {label: idx for idx, label in enumerate(classes.tolist())}
    y = np.zeros((train_y.shape[0], classes.size), dtype=np.float64)
    for row, label in enumerate(train_y.tolist()):
        y[row, class_index[str(label)]] = 1.0

    ones_train = np.ones((train_x.shape[0], 1), dtype=np.float64)
    ones_val = np.ones((val_x.shape[0], 1), dtype=np.float64)
    x_train_aug = np.concatenate([train_x, ones_train], axis=1)
    x_val_aug = np.concatenate([val_x, ones_val], axis=1)
    regularizer = float(alpha) * np.eye(x_train_aug.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    gram = x_train_aug.T @ x_train_aug + regularizer
    rhs = x_train_aug.T @ y
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ rhs
    pred_idx = np.argmax(x_val_aug @ weights, axis=1)
    return classes[pred_idx]


def classification_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=object).reshape(-1)
    y = np.asarray(truth, dtype=object).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError("predicted and truth must have the same shape")
    if y.size == 0:
        raise ValueError("truth must be non-empty")
    labels, counts = np.unique(y.astype(str), return_counts=True)
    accuracy = float(np.mean(pred.astype(str) == y.astype(str)))
    majority = float(np.max(counts) / y.size)
    per_class_recall = {}
    for label in labels.tolist():
        mask = y.astype(str) == label
        per_class_recall[label] = float(np.mean(pred.astype(str)[mask] == label))
    macro_recall = float(np.mean(list(per_class_recall.values())))
    return {
        "accuracy": accuracy,
        "majority_accuracy": majority,
        "accuracy_lift": accuracy - majority,
        "macro_recall": macro_recall,
        "n": int(y.size),
        "n_classes": int(labels.size),
        "class_counts": {str(label): int(count) for label, count in zip(labels, counts)},
        "per_class_recall": per_class_recall,
    }


def _pool_sequence(encoded: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(encoded, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"encoded must have shape (N, T, D), got {arr.shape}")
    if mode == "mean":
        return arr.mean(axis=1)
    if mode == "last":
        return arr[:, -1, :]
    raise ValueError(f"unknown pool mode: {mode}")


def _probe_one(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    pred = fit_predict_multiclass_ridge(
        train_features,
        train_labels,
        val_features,
        alpha=alpha,
    )
    return classification_metrics(pred, val_labels)


@torch.no_grad()
def encode_audit_features(
    model: DirectMaskedMultiviewBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
    pool: str,
) -> dict[str, np.ndarray]:
    encoded = encode_direct_barlow_split(
        model,
        batch,
        batch_size=batch_size,
        device=device,
    )
    clean = encode_clean_masked_windows(
        model,
        batch,
        batch_size=batch_size,
        device=device,
    )
    return {
        "view_a": _pool_sequence(encoded["view_a"], pool),
        "view_b": _pool_sequence(encoded["view_b"], pool),
        "clean": _pool_sequence(clean, pool),
    }


def audit_mask_artifact_leakage(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_direct_barlow_checkpoint(args.checkpoint, device=device)
    train = build_masked_multiview_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
        normalize=True,
    )
    val = build_masked_multiview_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
        normalize=True,
    )
    train_features = encode_audit_features(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
        pool=args.pool,
    )
    val_features = encode_audit_features(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
        pool=args.pool,
    )
    probes = {
        "view_a_to_mask_family_a": _probe_one(
            train_features["view_a"],
            train.mask_family_a,
            val_features["view_a"],
            val.mask_family_a,
            alpha=args.alpha,
        ),
        "view_b_to_mask_family_b": _probe_one(
            train_features["view_b"],
            train.mask_family_b,
            val_features["view_b"],
            val.mask_family_b,
            alpha=args.alpha,
        ),
        "clean_to_mask_family_a": _probe_one(
            train_features["clean"],
            train.mask_family_a,
            val_features["clean"],
            val.mask_family_a,
            alpha=args.alpha,
        ),
        "clean_to_mask_family_b": _probe_one(
            train_features["clean"],
            train.mask_family_b,
            val_features["clean"],
            val.mask_family_b,
            alpha=args.alpha,
        ),
    }
    return {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "pool": args.pool,
        "alpha": float(args.alpha),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "probes": probes,
        "decision_hint": _decision_hint(probes),
    }


def _decision_hint(probes: dict[str, dict[str, Any]]) -> str:
    corrupted_lifts = [
        float(probes[name]["accuracy_lift"])
        for name in ("view_a_to_mask_family_a", "view_b_to_mask_family_b")
    ]
    clean_lifts = [
        float(probes[name]["accuracy_lift"])
        for name in ("clean_to_mask_family_a", "clean_to_mask_family_b")
    ]
    if max(corrupted_lifts) <= 0.10:
        return "no_large_mask_family_leakage"
    if max(corrupted_lifts) > max(clean_lifts) + 0.10:
        return "possible_mask_family_leakage"
    return "ambiguous_mask_family_signal"


def render_markdown(
    result: dict[str, Any],
    *,
    title: str,
    candidate_label: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Objective Family",
        "",
        "`masked_multiview_invariance` mask-artifact diagnostic.",
        "",
        "## Hypothesis",
        "",
        f"If {candidate_label} learned market-state structure rather than corruption artifacts,",
        "a simple frozen-embedding probe should not predict synthetic mask family",
        "far above the majority-class baseline.",
        "",
        "## Probe Results",
        "",
        "| probe | accuracy | majority | lift | macro recall | classes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["probes"].items():
        lines.append(
            "| {name} | {acc:.6f} | {maj:.6f} | {lift:.6f} | {macro:.6f} | {classes} |".format(
                name=name,
                acc=float(metrics["accuracy"]),
                maj=float(metrics["majority_accuracy"]),
                lift=float(metrics["accuracy_lift"]),
                macro=float(metrics["macro_recall"]),
                classes=int(metrics["n_classes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{result['decision_hint']}`.",
            "",
            "This is a diagnostic, not a training objective. A positive leakage",
            "result should trigger mask-policy or representation-surface analysis",
            "before any new model knobs.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit mask-family leakage in frozen Part 1 embeddings")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt"),
    )
    parser.add_argument("--output-json", type=Path, default=Path("results/world/masked_multiview_mask_artifact_head083.json"))
    parser.add_argument("--output-md", type=Path, default=Path("experiments/world/reports/world_model_head083_mask_artifact_audit.md"))
    parser.add_argument("--report-title", default="World Model HEAD083: Mask-Artifact Leakage Audit")
    parser.add_argument("--candidate-label", default="HEAD070")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=384)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=680)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--pool", choices=("mean", "last"), default="mean")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = audit_mask_artifact_leakage(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        render_markdown(
            result,
            title=args.report_title,
            candidate_label=args.candidate_label,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
