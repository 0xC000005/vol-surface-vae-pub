from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import (  # noqa: E402
    validate_horizons,
)
from experiments.world.part1_jepa_latent.jepa_smoke import SequenceEncoder  # noqa: E402
from experiments.world.part1_jepa_latent.supervised_horizon_frame import (  # noqa: E402
    raw_horizon_persistence_baseline,
)


@dataclass(frozen=True)
class FixedDeltaPCATarget:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray

    @property
    def target_dim(self) -> int:
        return int(self.components.shape[1])


@dataclass(frozen=True)
class FixedDeltaPCAConfig:
    input_dim: int = 25
    hidden_dim: int = 64
    context_dim: int = 32
    target_dim: int = 8
    predictor_hidden_dim: int = 128
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)


class FixedDeltaPCAWorldModel(nn.Module):
    def __init__(self, cfg: FixedDeltaPCAConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        self.context_encoder = SequenceEncoder(
            argparse.Namespace(
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                latent_dim=cfg.context_dim,
            )
        )
        self.horizon_embedding = nn.Embedding(len(self.horizons), cfg.context_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.context_dim * 2),
            nn.Linear(cfg.context_dim * 2, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.target_dim),
        )

    def forward(
        self,
        past: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        context = self.context_encoder(past)
        batch_size = past.shape[0]
        rows = []
        for horizon_idx in range(len(self.horizons)):
            horizon_ids = torch.full(
                (batch_size,),
                horizon_idx,
                dtype=torch.long,
                device=past.device,
            )
            horizon_emb = self.horizon_embedding(horizon_ids)
            rows.append(self.predictor(torch.cat([context, horizon_emb], dim=1)))
        predicted = torch.stack(rows, dim=1)
        if return_context:
            return predicted, context
        return predicted


def make_horizon_delta_matrix(
    past: np.ndarray,
    future: np.ndarray,
    *,
    horizons: tuple[int, ...],
) -> np.ndarray:
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError("past and future must have shape (B, T, C)")
    if past.shape[0] != future.shape[0] or past.shape[2] != future.shape[2]:
        raise ValueError("past and future must share batch and channel dimensions")
    validate_horizons(horizons, future_len=future.shape[1])
    frames = np.stack([future[:, h - 1, :] for h in horizons], axis=1)
    return (frames - past[:, -1:, :]).astype(np.float32)


def fit_delta_pca_target(deltas: np.ndarray, *, target_dim: int = 8) -> FixedDeltaPCATarget:
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("deltas must have shape (B, H, C)")
    if target_dim <= 0 or target_dim > arr.shape[2]:
        raise ValueError(f"target_dim must be in [1, {arr.shape[2]}]")
    flat = arr.reshape(-1, arr.shape[-1])
    mean = flat.mean(axis=0)
    centered = flat - mean
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:target_dim].T
    projected = centered @ components
    scale = projected.std(axis=0).clip(min=1e-8)
    return FixedDeltaPCATarget(
        mean=mean.astype(np.float64),
        components=components.astype(np.float64),
        scale=scale.astype(np.float64),
    )


def transform_delta_targets(deltas: np.ndarray, target: FixedDeltaPCATarget) -> np.ndarray:
    arr = np.asarray(deltas, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    z = ((flat - target.mean) @ target.components) / target.scale
    return z.reshape(arr.shape[0], arr.shape[1], target.target_dim).astype(np.float32)


def inverse_transform_delta_targets(z: np.ndarray, target: FixedDeltaPCATarget) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    deltas = (flat * target.scale) @ target.components.T + target.mean
    return deltas.reshape(arr.shape[0], arr.shape[1], target.mean.shape[0]).astype(np.float32)


def fixed_delta_pca_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.mse_loss(predicted, target)
    return loss, {"target_mse": float(loss.detach().cpu()), "loss": float(loss.detach().cpu())}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader_from_arrays(
    past: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(past), torch.from_numpy(target))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _parameter_groups(model: FixedDeltaPCAWorldModel) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.horizon_embedding.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def predict_fixed_targets(
    model: FixedDeltaPCAWorldModel,
    past: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_rows = []
    context_rows = []
    loader = DataLoader(torch.from_numpy(past), batch_size=batch_size, shuffle=False)
    for past_batch in loader:
        predicted, context = model(past_batch.to(device), return_context=True)
        pred_rows.append(predicted.detach().cpu().numpy())
        context_rows.append(context.detach().cpu().numpy())
    return np.concatenate(pred_rows, axis=0), np.concatenate(context_rows, axis=0)


def evaluate_fixed_delta_pca(
    model: FixedDeltaPCAWorldModel,
    past: np.ndarray,
    target_z: np.ndarray,
    truth_delta: np.ndarray,
    pca_target: FixedDeltaPCATarget,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    predicted_z, context = predict_fixed_targets(model, past, batch_size=batch_size, device=device)
    decoded_delta = inverse_transform_delta_targets(predicted_z, pca_target)

    per_horizon: dict[str, object] = {}
    for horizon_idx, horizon in enumerate(model.horizons):
        per_horizon[str(horizon)] = {
            "prediction": latent_prediction_metrics(
                predicted_z[:, horizon_idx, :],
                target_z[:, horizon_idx, :],
            ),
            "retrieval": retrieval_metrics(
                predicted_z[:, horizon_idx, :],
                target_z[:, horizon_idx, :],
                top_k=(1, 5, 10),
            ),
            "delta_decode": latent_prediction_metrics(
                decoded_delta[:, horizon_idx, :],
                truth_delta[:, horizon_idx, :],
            ),
            "predicted_health": representation_health_metrics(predicted_z[:, horizon_idx, :]),
            "target_health": representation_health_metrics(target_z[:, horizon_idx, :]),
        }

    return {
        "overall_prediction": latent_prediction_metrics(predicted_z, target_z),
        "overall_retrieval": {
            "mrr_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["mrr"] for h in model.horizons])),
            "top1_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top1"] for h in model.horizons])),
            "top5_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top5"] for h in model.horizons])),
            "top10_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top10"] for h in model.horizons])),
        },
        "overall_delta_decode": {
            "mse_mean": float(np.mean([per_horizon[str(h)]["delta_decode"]["mse"] for h in model.horizons])),
            "cosine_mean": float(
                np.mean([per_horizon[str(h)]["delta_decode"]["cosine_mean"] for h in model.horizons])
            ),
        },
        "context_health": representation_health_metrics(context),
        "predicted_health": representation_health_metrics(predicted_z.reshape(-1, predicted_z.shape[-1])),
        "target_health": representation_health_metrics(target_z.reshape(-1, target_z.shape[-1])),
        "per_horizon": per_horizon,
    }


def _serializable(obj):
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def train_smoke(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    horizons = validate_horizons(tuple(args.horizons), future_len=30)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

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
    train_delta = make_horizon_delta_matrix(train.past_window, train.future_window, horizons=horizons)
    val_delta = make_horizon_delta_matrix(val.past_window, val.future_window, horizons=horizons)
    pca_target = fit_delta_pca_target(train_delta, target_dim=args.target_dim)
    train_z = transform_delta_targets(train_delta, pca_target)
    val_z = transform_delta_targets(val_delta, pca_target)

    cfg = FixedDeltaPCAConfig(
        input_dim=25,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        target_dim=args.target_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        horizons=horizons,
    )
    model = FixedDeltaPCAWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader_from_arrays(
        train.past_window,
        train_z,
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_summary: dict[str, object] | None = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for past_batch, target_batch in train_loader:
            past_batch = past_batch.to(device)
            target_batch = target_batch.to(device)
            opt.zero_grad(set_to_none=True)
            predicted = model(past_batch)
            loss, _parts = fixed_delta_pca_loss(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_fixed_delta_pca(
            model,
            val.past_window,
            val_z,
            val_delta,
            pca_target,
            batch_size=args.batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_mse": val_metrics["overall_prediction"]["mse"],
            "val_mrr_mean": val_metrics["overall_retrieval"]["mrr_mean"],
            "val_top1_mean": val_metrics["overall_retrieval"]["top1_mean"],
            "val_top5_mean": val_metrics["overall_retrieval"]["top5_mean"],
            "val_top10_mean": val_metrics["overall_retrieval"]["top10_mean"],
            "val_delta_decode_mse": val_metrics["overall_delta_decode"]["mse_mean"],
            "val_predicted_effective_rank": val_metrics["predicted_health"]["effective_rank"],
        }
        history.append(row)
        if best_summary is None or row["val_mrr_mean"] > best_summary["val_mrr_mean"]:
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row))

    final_metrics = evaluate_fixed_delta_pca(
        model,
        val.past_window,
        val_z,
        val_delta,
        pca_target,
        batch_size=args.batch_size,
        device=device,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = evaluate_fixed_delta_pca(
        model,
        val.past_window,
        val_z,
        val_delta,
        pca_target,
        batch_size=args.batch_size,
        device=device,
    )
    baseline = raw_horizon_persistence_baseline(
        val.past_window,
        val.future_window,
        horizons=horizons,
    )
    result = {
        "config": asdict(cfg),
        "args": vars(args),
        "target_contract": {
            "kind": "fixed_delta_pca",
            "target_dim": args.target_dim,
            "mean": pca_target.mean,
            "components": pca_target.components,
            "scale": pca_target.scale,
        },
        "device": str(device),
        "train_shape": {
            "past": list(train.past_window.shape),
            "future": list(train.future_window.shape),
            "target_z": list(train_z.shape),
        },
        "val_shape": {
            "past": list(val.past_window.shape),
            "future": list(val.future_window.shape),
            "target_z": list(val_z.shape),
        },
        "history": history,
        "best_epoch_summary": best_summary,
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "raw_horizon_frame_baseline": baseline,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_serializable(result), indent=2), encoding="utf-8")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "result": _serializable(result),
        },
        checkpoint,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed delta-PCA target Part 1 diagnostic")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--context_dim", type=int, default=32)
    parser.add_argument("--target_dim", type=int, default=8)
    parser.add_argument("--predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7706)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_fixed_delta_pca_jepa_head028.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/fixed_delta_pca_jepa_head028.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
