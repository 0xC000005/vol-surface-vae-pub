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

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)
from experiments.world.evaluation.part1_metrics import representation_health_metrics  # noqa: E402
from experiments.world.evaluation.world_data import (  # noqa: E402
    WorldWindowBatch,
    build_iv_world_windows,
)
from experiments.world.part1_jepa_latent.context_probe_audit import ridge_probe_predict  # noqa: E402
from experiments.world.part1_jepa_latent.masked_multiview_barlow_smoke import (  # noqa: E402
    DirectMaskedMultiviewBarlowConfig,
    DirectMaskedMultiviewBarlowModel,
)
from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (  # noqa: E402
    make_masked_view_features,
)


def make_future_summary_targets(
    past_surface: np.ndarray,
    future_surface: np.ndarray,
) -> dict[str, np.ndarray]:
    past = np.asarray(past_surface, dtype=np.float32)
    future = np.asarray(future_surface, dtype=np.float32)
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError("past_surface and future_surface must have shape (N, T, C)")
    if past.shape[0] != future.shape[0] or past.shape[2] != future.shape[2]:
        raise ValueError("past and future must share sample count and channel count")
    last = past[:, -1, :]
    return {
        "future_mean_delta": (future.mean(axis=1) - last).astype(np.float32),
        "future_range": (future.max(axis=1) - future.min(axis=1)).astype(np.float32),
    }


def regression_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"predicted and target shapes must match: {pred.shape} != {truth.shape}")
    err = pred - truth
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    centered = truth - truth.mean(axis=0, keepdims=True)
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum(centered * centered))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": float(r2)}


def load_direct_barlow_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> DirectMaskedMultiviewBarlowModel:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = DirectMaskedMultiviewBarlowConfig(**checkpoint["config"])
    model = DirectMaskedMultiviewBarlowModel(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def encode_clean_masked_windows(
    model: DirectMaskedMultiviewBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(batch.clean_values),
        torch.from_numpy(batch.observed_mask),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    for clean_values, observed in loader:
        synthetic = torch.ones_like(observed, dtype=torch.bool)
        features = make_masked_view_features(
            clean_values.to(device),
            observed.to(device),
            synthetic.to(device),
        )
        rows.append(model.encoder(features).cpu().numpy())
    return np.concatenate(rows, axis=0)


def _feature_sets(
    encoded: np.ndarray,
    masked_batch: MaskedMultiviewBatch,
    iv_batch: WorldWindowBatch,
) -> dict[str, np.ndarray]:
    return {
        "barlow_clean_last": encoded[:, -1, :],
        "barlow_clean_mean": encoded.mean(axis=1),
        "raw_geometry_last": masked_batch.clean_values[:, -1, :],
        "raw_geometry_flat": masked_batch.clean_values.reshape(masked_batch.clean_values.shape[0], -1),
        "raw_surface_last": iv_batch.past_window[:, -1, :],
        "raw_surface_flat": iv_batch.past_window.reshape(iv_batch.past_window.shape[0], -1),
    }


def probe_feature_sets(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for feature_name, train_x in train_features.items():
        val_x = val_features[feature_name]
        target_rows: dict[str, object] = {}
        for target_name, train_y in train_targets.items():
            val_y = val_targets[target_name]
            pred = ridge_probe_predict(train_x, train_y, val_x, alpha=alpha)
            target_rows[target_name] = regression_metrics(pred, val_y)
        out[feature_name] = {
            "feature_shape": {
                "train": list(train_x.shape),
                "val": list(val_x.shape),
            },
            "health": representation_health_metrics(val_x),
            "targets": target_rows,
        }
    return out


def mean_target_baselines(
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for target_name, train_y in train_targets.items():
        pred = np.repeat(train_y.mean(axis=0, keepdims=True), val_targets[target_name].shape[0], axis=0)
        out[target_name] = regression_metrics(pred, val_targets[target_name])
    return out


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def audit_barlow_probe(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
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
    train_features = _feature_sets(train_encoded, train_masked, train_iv)
    val_features = _feature_sets(val_encoded, val_masked, val_iv)
    train_targets = make_future_summary_targets(train_iv.past_window, train_iv.future_window)
    val_targets = make_future_summary_targets(val_iv.past_window, val_iv.future_window)
    probes = probe_feature_sets(
        train_features,
        val_features,
        train_targets,
        val_targets,
        alpha=args.ridge_alpha,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "config": asdict(model.cfg),
        "device": str(device),
        "ridge_alpha": args.ridge_alpha,
        "target_names": sorted(train_targets),
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
        "probe_metrics": probes,
        "mean_target_baseline": mean_target_baselines(train_targets, val_targets),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_serializable(result), indent=2), encoding="utf-8")
    print(json.dumps(_serializable(result["probe_metrics"]["barlow_clean_last"]["targets"]), indent=2))
    print(json.dumps(_serializable(result["mean_target_baseline"]), indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen probe audit for direct Barlow masked-multiview embeddings")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/masked_multiview_barlow_probe_head072.json",
    )
    audit_barlow_probe(parser.parse_args())


if __name__ == "__main__":
    main()
