from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.supervised_horizon_frame import (  # noqa: E402
    SupervisedHorizonConfig,
    SupervisedHorizonFrameModel,
    make_horizon_frame_targets,
    predict_horizon_frames,
)


def ridge_probe_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    *,
    alpha: float = 1e-3,
) -> np.ndarray:
    x_train = np.asarray(train_features, dtype=np.float64)
    y_train = np.asarray(train_targets, dtype=np.float64)
    x_val = np.asarray(val_features, dtype=np.float64)
    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if y_train.ndim != 2:
        raise ValueError("targets must have shape (N, C)")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("train_features and train_targets must share sample count")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("train and validation features must share feature dimension")

    ones_train = np.ones((x_train.shape[0], 1), dtype=np.float64)
    ones_val = np.ones((x_val.shape[0], 1), dtype=np.float64)
    x_train_aug = np.concatenate([x_train, ones_train], axis=1)
    x_val_aug = np.concatenate([x_val, ones_val], axis=1)
    regularizer = float(alpha) * np.eye(x_train_aug.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    gram = x_train_aug.T @ x_train_aug + regularizer
    rhs = x_train_aug.T @ y_train
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ rhs
    return x_val_aug @ weights


def horizon_target_metrics(
    predicted_targets: np.ndarray,
    targets: np.ndarray,
    *,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    predicted = np.asarray(predicted_targets, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(f"predicted and target shapes must match: {predicted.shape} != {truth.shape}")
    if predicted.ndim != 3:
        raise ValueError("predicted and targets must have shape (N, H, C)")
    if predicted.shape[1] != len(horizons):
        raise ValueError("horizon count does not match target shape")

    per_horizon: dict[str, object] = {}
    for horizon_idx, horizon in enumerate(horizons):
        per_horizon[str(horizon)] = {
            "prediction": latent_prediction_metrics(
                predicted[:, horizon_idx, :],
                truth[:, horizon_idx, :],
            ),
            "retrieval": retrieval_metrics(
                predicted[:, horizon_idx, :],
                truth[:, horizon_idx, :],
                top_k=(1, 5, 10),
            ),
        }
    pred_flat = predicted.reshape(-1, predicted.shape[-1])
    truth_flat = truth.reshape(-1, truth.shape[-1])
    return {
        "overall_prediction": latent_prediction_metrics(pred_flat, truth_flat),
        "overall_retrieval": {
            "mrr_mean": float(
                np.mean([per_horizon[str(h)]["retrieval"]["mrr"] for h in horizons])
            ),
            "top1_mean": float(
                np.mean([per_horizon[str(h)]["retrieval"]["top1"] for h in horizons])
            ),
            "top5_mean": float(
                np.mean([per_horizon[str(h)]["retrieval"]["top5"] for h in horizons])
            ),
            "top10_mean": float(
                np.mean([per_horizon[str(h)]["retrieval"]["top10"] for h in horizons])
            ),
        },
        "predicted_health": representation_health_metrics(pred_flat),
        "target_health": representation_health_metrics(truth_flat),
        "per_horizon": per_horizon,
    }


def ridge_probe_metrics(
    train_contexts: np.ndarray,
    val_contexts: np.ndarray,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    *,
    horizons: tuple[int, ...],
    alpha: float = 1e-3,
) -> dict[str, object]:
    train_y = np.asarray(train_targets, dtype=np.float64)
    val_y = np.asarray(val_targets, dtype=np.float64)
    if train_y.ndim != 3 or val_y.ndim != 3:
        raise ValueError("targets must have shape (N, H, C)")
    if train_y.shape[1:] != val_y.shape[1:]:
        raise ValueError("train and validation targets must share horizon and channel shape")
    if train_y.shape[1] != len(horizons):
        raise ValueError("horizon count does not match target shape")

    predictions = []
    for horizon_idx in range(len(horizons)):
        predictions.append(
            ridge_probe_predict(
                train_contexts,
                train_y[:, horizon_idx, :],
                val_contexts,
                alpha=alpha,
            )
        )
    predicted = np.stack(predictions, axis=1)
    return horizon_target_metrics(predicted, val_y, horizons=horizons)


@torch.no_grad()
def encode_contexts(
    model: SupervisedHorizonFrameModel,
    past: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(past)), batch_size=batch_size, shuffle=False)
    encoded = []
    for (past_batch,) in loader:
        context = model.context_encoder(past_batch.to(device))
        encoded.append(context.cpu().numpy())
    return np.concatenate(encoded, axis=0)


@torch.no_grad()
def make_target_arrays(
    model: SupervisedHorizonFrameModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(past), torch.from_numpy(future)),
        batch_size=batch_size,
        shuffle=False,
    )
    targets = []
    frames = []
    for past_batch, future_batch in loader:
        target, frame = make_horizon_frame_targets(
            past_batch.to(device),
            future_batch.to(device),
            horizons=model.horizons,
            target_mode=model.cfg.target_mode,
        )
        targets.append(target.cpu().numpy())
        frames.append(frame.cpu().numpy())
    return {
        "target": np.concatenate(targets, axis=0),
        "frame": np.concatenate(frames, axis=0),
    }


def load_supervised_horizon_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> SupervisedHorizonFrameModel:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = SupervisedHorizonConfig(**checkpoint["config"])
    model = SupervisedHorizonFrameModel(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _serializable(obj):
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def audit_context_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = load_supervised_horizon_checkpoint(args.checkpoint, device=device)
    train = build_iv_world_windows(
        split="train",
        max_windows=args.max_train_windows,
        normalize=True,
    )
    val = build_iv_world_windows(
        split="val",
        max_windows=args.max_val_windows,
        normalize=True,
    )
    train_contexts = encode_contexts(
        model,
        train.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    val_contexts = encode_contexts(
        model,
        val.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    train_targets = make_target_arrays(
        model,
        train.past_window,
        train.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    val_targets = make_target_arrays(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    ridge_metrics = ridge_probe_metrics(
        train_contexts,
        val_contexts,
        train_targets["target"],
        val_targets["target"],
        horizons=model.horizons,
        alpha=args.ridge_alpha,
    )
    zero_metrics = horizon_target_metrics(
        np.zeros_like(val_targets["target"]),
        val_targets["target"],
        horizons=model.horizons,
    )
    head_preds = predict_horizon_frames(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    head_target_metrics = horizon_target_metrics(
        head_preds["predicted_target"],
        head_preds["target"],
        horizons=model.horizons,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "config": asdict(model.cfg),
        "device": str(device),
        "ridge_alpha": args.ridge_alpha,
        "train_shape": {
            "past": list(train.past_window.shape),
            "future": list(train.future_window.shape),
            "context": list(train_contexts.shape),
        },
        "val_shape": {
            "past": list(val.past_window.shape),
            "future": list(val.future_window.shape),
            "context": list(val_contexts.shape),
        },
        "train_context_health": representation_health_metrics(train_contexts),
        "val_context_health": representation_health_metrics(val_contexts),
        "ridge_probe_target_metrics": ridge_metrics,
        "zero_delta_target_baseline": zero_metrics,
        "trained_head_target_metrics": head_target_metrics,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_serializable(result), indent=2), encoding="utf-8")
    print(json.dumps(_serializable(result["val_context_health"]), indent=2))
    print(json.dumps(_serializable(result["ridge_probe_target_metrics"]["overall_prediction"]), indent=2))
    print(json.dumps(_serializable(result["ridge_probe_target_metrics"]["overall_retrieval"]), indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen context embeddings for Part 1 JEPA work")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_contrastive_mrr_head009.pt",
    )
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_context_probe_audit_head010.json",
    )
    args = parser.parse_args()
    audit_context_checkpoint(args)


if __name__ == "__main__":
    main()
