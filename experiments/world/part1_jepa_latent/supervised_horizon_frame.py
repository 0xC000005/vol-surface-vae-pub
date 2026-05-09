from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

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
from experiments.world.part1_jepa_latent.jepa_smoke import (  # noqa: E402
    covariance_loss,
    retrieval_contrastive_loss,
    variance_loss,
)


TargetMode = Literal["frame", "delta"]


@dataclass(frozen=True)
class SupervisedHorizonConfig:
    input_dim: int = 25
    hidden_dim: int = 64
    context_dim: int = 32
    predictor_hidden_dim: int = 128
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)
    target_mode: TargetMode = "delta"


class GRUContextEncoder(nn.Module):
    def __init__(self, cfg: SupervisedHorizonConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.input_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.context_dim),
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        _seq, h_n = self.gru(past)
        return self.head(h_n[-1])


class SupervisedHorizonFrameModel(nn.Module):
    def __init__(self, cfg: SupervisedHorizonConfig):
        super().__init__()
        if cfg.target_mode not in {"frame", "delta"}:
            raise ValueError(f"Unknown target_mode: {cfg.target_mode!r}")
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        self.context_encoder = GRUContextEncoder(cfg)
        self.horizon_embedding = nn.Embedding(len(self.horizons), cfg.context_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.context_dim * 2),
            nn.Linear(cfg.context_dim * 2, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.input_dim),
        )

    def forward(
        self,
        past: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        context = self.context_encoder(past)
        rows = []
        batch_size = past.shape[0]
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


def make_horizon_frame_targets(
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    horizons: tuple[int, ...],
    target_mode: TargetMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError("past and future must have shape (B, T, C)")
    validate_horizons(horizons, future_len=future.shape[1])
    frames = torch.stack([future[:, h - 1, :] for h in horizons], dim=1)
    if target_mode == "frame":
        return frames, frames
    if target_mode == "delta":
        return frames - past[:, -1:, :], frames
    raise ValueError(f"Unknown target_mode: {target_mode!r}")


def decode_horizon_prediction(
    past: torch.Tensor,
    predicted_target: torch.Tensor,
    *,
    target_mode: TargetMode,
) -> torch.Tensor:
    if target_mode == "frame":
        return predicted_target
    if target_mode == "delta":
        return past[:, -1:, :] + predicted_target
    raise ValueError(f"Unknown target_mode: {target_mode!r}")


def context_correlation_loss(context: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if context.ndim != 2:
        raise ValueError("context must have shape (B, D)")
    centered = context - context.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    normalized = centered / std
    denom = max(normalized.shape[0] - 1, 1)
    corr = normalized.T @ normalized / denom
    offdiag = corr - torch.diag(torch.diag(corr))
    return (offdiag * offdiag).sum() / context.shape[1]


def soft_neighborhood_contrastive_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.1,
    target_temperature: float = 0.2,
    detach_target: bool = True,
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must match, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.ndim != 2:
        raise ValueError("predicted and target must have shape (B, D)")
    if predicted.shape[0] < 2:
        raise ValueError("soft_neighborhood_contrastive_loss needs at least two samples")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if target_temperature <= 0.0:
        raise ValueError("target_temperature must be positive")

    pred_norm = F.normalize(predicted, dim=1)
    target_for_scores = target.detach() if detach_target else target
    target_norm = F.normalize(target_for_scores, dim=1)
    logits = pred_norm @ target_norm.T / temperature
    with torch.no_grad():
        neighbor_probs = F.softmax((target_norm @ target_norm.T) / target_temperature, dim=1)
    row_loss = -(neighbor_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    col_loss = -(neighbor_probs.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (row_loss + col_loss)


def supervised_horizon_loss(
    predicted_target: torch.Tensor,
    target: torch.Tensor,
    frame: torch.Tensor,
    past: torch.Tensor,
    *,
    target_mode: TargetMode = "delta",
    frame_weight: float = 0.25,
    retrieval_weight: float = 0.0,
    retrieval_temperature: float = 0.1,
    neighborhood_weight: float = 0.0,
    neighborhood_temperature: float = 0.1,
    neighborhood_target_temperature: float = 0.2,
    context: torch.Tensor | None = None,
    context_variance_weight: float = 0.0,
    context_covariance_weight: float = 0.0,
    context_variance_gamma: float = 0.1,
    context_correlation_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_frame = decode_horizon_prediction(
        past,
        predicted_target,
        target_mode=target_mode,
    )
    target_mse = F.mse_loss(predicted_target, target)
    frame_mse = F.mse_loss(predicted_frame, frame)
    if retrieval_weight > 0.0:
        retrieval_terms = [
            retrieval_contrastive_loss(
                predicted_target[:, horizon_idx, :],
                target[:, horizon_idx, :],
                temperature=retrieval_temperature,
            )
            for horizon_idx in range(predicted_target.shape[1])
        ]
        retrieval = torch.stack(retrieval_terms).mean()
    else:
        retrieval = predicted_target.new_tensor(0.0)
    if neighborhood_weight > 0.0:
        neighborhood_terms = [
            soft_neighborhood_contrastive_loss(
                predicted_target[:, horizon_idx, :],
                target[:, horizon_idx, :],
                temperature=neighborhood_temperature,
                target_temperature=neighborhood_target_temperature,
            )
            for horizon_idx in range(predicted_target.shape[1])
        ]
        neighborhood = torch.stack(neighborhood_terms).mean()
    else:
        neighborhood = predicted_target.new_tensor(0.0)
    if context is not None:
        context_variance = variance_loss(context, gamma=context_variance_gamma)
        context_covariance = covariance_loss(context)
        context_correlation = context_correlation_loss(context)
    elif (
        context_variance_weight > 0.0
        or context_covariance_weight > 0.0
        or context_correlation_weight > 0.0
    ):
        raise ValueError("context must be provided when context regularization weights are positive")
    else:
        context_variance = predicted_target.new_tensor(0.0)
        context_covariance = predicted_target.new_tensor(0.0)
        context_correlation = predicted_target.new_tensor(0.0)
    loss = (
        target_mse
        + frame_weight * frame_mse
        + retrieval_weight * retrieval
        + neighborhood_weight * neighborhood
        + context_variance_weight * context_variance
        + context_covariance_weight * context_covariance
        + context_correlation_weight * context_correlation
    )
    return loss, {
        "target_mse": float(target_mse.detach().cpu()),
        "frame_mse": float(frame_mse.detach().cpu()),
        "retrieval": float(retrieval.detach().cpu()),
        "neighborhood": float(neighborhood.detach().cpu()),
        "context_variance": float(context_variance.detach().cpu()),
        "context_covariance": float(context_covariance.detach().cpu()),
        "context_correlation": float(context_correlation.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader_from_arrays(
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(past), torch.from_numpy(future))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _parameter_groups(model: SupervisedHorizonFrameModel) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.horizon_embedding.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def predict_horizon_frames(
    model: SupervisedHorizonFrameModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    pred_targets = []
    pred_frames = []
    true_frames = []
    targets = []
    loader = _loader_from_arrays(past, future, batch_size=batch_size, shuffle=False)
    for past_batch, future_batch in loader:
        past_batch = past_batch.to(device)
        future_batch = future_batch.to(device)
        target, frame = make_horizon_frame_targets(
            past_batch,
            future_batch,
            horizons=model.horizons,
            target_mode=model.cfg.target_mode,
        )
        pred_target = model(past_batch)
        pred_frame = decode_horizon_prediction(
            past_batch,
            pred_target,
            target_mode=model.cfg.target_mode,
        )
        pred_targets.append(pred_target.cpu().numpy())
        pred_frames.append(pred_frame.cpu().numpy())
        true_frames.append(frame.cpu().numpy())
        targets.append(target.cpu().numpy())
    return {
        "predicted_target": np.concatenate(pred_targets, axis=0),
        "predicted_frame": np.concatenate(pred_frames, axis=0),
        "true_frame": np.concatenate(true_frames, axis=0),
        "target": np.concatenate(targets, axis=0),
    }


def evaluate_supervised_horizon(
    model: SupervisedHorizonFrameModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    preds = predict_horizon_frames(
        model,
        past,
        future,
        batch_size=batch_size,
        device=device,
    )
    predicted = preds["predicted_frame"]
    truth = preds["true_frame"]
    per_horizon: dict[str, object] = {}
    for horizon_idx, horizon in enumerate(model.horizons):
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
                np.mean([per_horizon[str(h)]["retrieval"]["mrr"] for h in model.horizons])
            ),
            "top1_mean": float(
                np.mean([per_horizon[str(h)]["retrieval"]["top1"] for h in model.horizons])
            ),
            "top5_mean": float(
                np.mean([per_horizon[str(h)]["retrieval"]["top5"] for h in model.horizons])
            ),
            "top10_mean": float(
                np.mean([per_horizon[str(h)]["retrieval"]["top10"] for h in model.horizons])
            ),
        },
        "predicted_health": representation_health_metrics(pred_flat),
        "truth_health": representation_health_metrics(truth_flat),
        "per_horizon": per_horizon,
    }


def raw_horizon_persistence_baseline(
    past: np.ndarray,
    future: np.ndarray,
    *,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    predicted = np.stack([past[:, -1, :] for _h in horizons], axis=1)
    truth = np.stack([future[:, h - 1, :] for h in horizons], axis=1)
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
    return {
        "prediction": {
            "mse_mean": float(np.mean([per_horizon[str(h)]["prediction"]["mse"] for h in horizons])),
            "cosine_mean": float(
                np.mean([per_horizon[str(h)]["prediction"]["cosine_mean"] for h in horizons])
            ),
        },
        "retrieval": {
            "mrr_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["mrr"] for h in horizons])),
            "top1_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top1"] for h in horizons])),
            "top5_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top5"] for h in horizons])),
            "top10_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top10"] for h in horizons])),
        },
        "per_horizon": per_horizon,
    }


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


def checkpoint_selection_score(row: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "mse":
        return -float(row["val_mse"])
    if selection_metric == "mrr":
        return float(row["val_mrr_mean"])
    if selection_metric == "top5":
        return float(row["val_top5_mean"])
    raise ValueError(f"Unknown selection_metric: {selection_metric!r}")


def epoch_checkpoint_path(checkpoint_dir: str | Path, epoch: int) -> Path:
    return Path(checkpoint_dir) / f"epoch_{int(epoch):03d}.pt"


def save_epoch_checkpoint(
    checkpoint_dir: str | Path,
    *,
    epoch: int,
    model: SupervisedHorizonFrameModel,
    cfg: SupervisedHorizonConfig,
    epoch_summary: dict[str, object],
    args: dict[str, object],
) -> Path:
    path = epoch_checkpoint_path(checkpoint_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "config": asdict(cfg),
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "epoch_summary": _serializable(epoch_summary),
            "args": _serializable(args),
        },
        path,
    )
    return path


def train_smoke(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    horizons = validate_horizons(tuple(args.horizons), future_len=30)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    cfg = SupervisedHorizonConfig(
        input_dim=25,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        horizons=horizons,
        target_mode=args.target_mode,
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

    model = SupervisedHorizonFrameModel(cfg).to(device)
    opt = torch.optim.AdamW(_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader_from_arrays(
        train.past_window,
        train.future_window,
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_summary: dict[str, object] | None = None
    best_score: float | None = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts_accum: dict[str, list[float]] = {}
        for past_batch, future_batch in train_loader:
            past_batch = past_batch.to(device)
            future_batch = future_batch.to(device)
            target, frame = make_horizon_frame_targets(
                past_batch,
                future_batch,
                horizons=horizons,
                target_mode=args.target_mode,
            )
            opt.zero_grad(set_to_none=True)
            pred_target, context = model(past_batch, return_context=True)
            loss, parts = supervised_horizon_loss(
                pred_target,
                target,
                frame,
                past_batch,
                target_mode=args.target_mode,
                frame_weight=args.frame_weight,
                retrieval_weight=args.retrieval_weight,
                retrieval_temperature=args.retrieval_temperature,
                neighborhood_weight=args.neighborhood_weight,
                neighborhood_temperature=args.neighborhood_temperature,
                neighborhood_target_temperature=args.neighborhood_target_temperature,
                context=context,
                context_variance_weight=args.context_variance_weight,
                context_covariance_weight=args.context_covariance_weight,
                context_variance_gamma=args.context_variance_gamma,
                context_correlation_weight=args.context_correlation_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                parts_accum.setdefault(key, []).append(value)

        val_metrics = evaluate_supervised_horizon(
            model,
            val.past_window,
            val.future_window,
            batch_size=args.batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{f"{k}_mean": float(np.mean(v)) for k, v in parts_accum.items()},
            "val_mse": val_metrics["overall_prediction"]["mse"],
            "val_mrr_mean": val_metrics["overall_retrieval"]["mrr_mean"],
            "val_top1_mean": val_metrics["overall_retrieval"]["top1_mean"],
            "val_top5_mean": val_metrics["overall_retrieval"]["top5_mean"],
        }
        history.append(row)
        if args.epoch_checkpoint_dir:
            save_epoch_checkpoint(
                args.epoch_checkpoint_dir,
                epoch=epoch,
                model=model,
                cfg=cfg,
                epoch_summary=row,
                args=vars(args),
            )
        score = checkpoint_selection_score(row, args.selection_metric)
        if best_score is None or score > best_score:
            best_score = score
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row))

    final_metrics = evaluate_supervised_horizon(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = evaluate_supervised_horizon(
        model,
        val.past_window,
        val.future_window,
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
        "device": str(device),
        "train_shape": {
            "past": list(train.past_window.shape),
            "future": list(train.future_window.shape),
        },
        "val_shape": {
            "past": list(val.past_window.shape),
            "future": list(val.future_window.shape),
        },
        "history": history,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "epoch_checkpoint_dir": args.epoch_checkpoint_dir or None,
        "best_epoch_summary": best_summary,
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "raw_persistence_baseline": baseline,
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
    parser = argparse.ArgumentParser(description="Supervised horizon-frame Part 1 lower bound")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--context_dim", type=int, default=32)
    parser.add_argument("--predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--target_mode", choices=("frame", "delta"), default="delta")
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--frame_weight", type=float, default=0.25)
    parser.add_argument("--retrieval_weight", type=float, default=0.0)
    parser.add_argument("--retrieval_temperature", type=float, default=0.1)
    parser.add_argument("--neighborhood_weight", type=float, default=0.0)
    parser.add_argument("--neighborhood_temperature", type=float, default=0.1)
    parser.add_argument("--neighborhood_target_temperature", type=float, default=0.2)
    parser.add_argument("--context_variance_weight", type=float, default=0.0)
    parser.add_argument("--context_covariance_weight", type=float, default=0.0)
    parser.add_argument("--context_variance_gamma", type=float, default=0.1)
    parser.add_argument("--context_correlation_weight", type=float, default=0.0)
    parser.add_argument("--selection_metric", choices=("mse", "mrr", "top5"), default="mse")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7705)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_supervised_horizon_delta_head007.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_head007.pt",
    )
    parser.add_argument("--epoch_checkpoint_dir", type=str, default="")
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
