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
from experiments.world.part1_jepa_latent.jepa_smoke import (  # noqa: E402
    SequenceEncoder,
    covariance_loss,
    retrieval_contrastive_loss,
    update_ema,
    variance_loss,
)


@dataclass(frozen=True)
class HorizonJEPAConfig:
    input_dim: int = 25
    hidden_dim: int = 64
    latent_dim: int = 16
    predictor_hidden_dim: int = 64
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)
    ema_decay: float = 0.99


def validate_horizons(horizons: tuple[int, ...], future_len: int) -> tuple[int, ...]:
    if not horizons:
        raise ValueError("At least one horizon is required")
    normalized = tuple(int(h) for h in horizons)
    if any(h <= 0 for h in normalized):
        raise ValueError("Horizons must be positive one-based indices")
    if any(h > future_len for h in normalized):
        raise ValueError(f"Horizons {normalized} exceed future_len={future_len}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Horizons must be unique, got {normalized}")
    return normalized


def select_horizon_prefix(future: torch.Tensor, *, horizon: int) -> torch.Tensor:
    if future.ndim != 3:
        raise ValueError(f"future must have shape (B, T, C), got {tuple(future.shape)}")
    if horizon <= 0 or horizon > future.shape[1]:
        raise ValueError(f"horizon={horizon} is outside future length {future.shape[1]}")
    return future[:, :horizon, :]


def select_horizon_target(
    future: torch.Tensor,
    *,
    horizon: int,
    target_mode: Literal["prefix", "frame"] = "prefix",
) -> torch.Tensor:
    if target_mode == "prefix":
        return select_horizon_prefix(future, horizon=horizon)
    if target_mode == "frame":
        if future.ndim != 3:
            raise ValueError(f"future must have shape (B, T, C), got {tuple(future.shape)}")
        if horizon <= 0 or horizon > future.shape[1]:
            raise ValueError(f"horizon={horizon} is outside future length {future.shape[1]}")
        return future[:, horizon - 1 : horizon, :]
    raise ValueError(f"Unknown target_mode: {target_mode!r}")


class HorizonJEPAWorldModel(nn.Module):
    def __init__(
        self,
        cfg: HorizonJEPAConfig,
        *,
        target_mode: Literal["prefix", "frame"] = "prefix",
        target_encoder_mode: Literal["ema", "trainable"] = "ema",
    ):
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        self.target_mode = target_mode
        self.target_encoder_mode = target_encoder_mode
        if target_encoder_mode not in {"ema", "trainable"}:
            raise ValueError(f"Unknown target_encoder_mode: {target_encoder_mode!r}")
        self.context_encoder = SequenceEncoder(cfg)
        self.target_encoder = SequenceEncoder(cfg)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        if target_encoder_mode == "ema":
            for param in self.target_encoder.parameters():
                param.requires_grad = False

        self.horizon_embedding = nn.Embedding(len(self.horizons), cfg.latent_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim * 2),
            nn.Linear(cfg.latent_dim * 2, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.latent_dim),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        validate_horizons(self.horizons, future.shape[1])
        context = self.context_encoder(past)
        predicted_rows = []
        target_rows = []
        batch_size = past.shape[0]
        for horizon_idx, horizon in enumerate(self.horizons):
            horizon_ids = torch.full(
                (batch_size,),
                horizon_idx,
                device=past.device,
                dtype=torch.long,
            )
            horizon_emb = self.horizon_embedding(horizon_ids)
            pred = self.predictor(torch.cat([context, horizon_emb], dim=1))
            target_input = select_horizon_target(
                future,
                horizon=horizon,
                target_mode=self.target_mode,
            )
            if self.target_encoder_mode == "ema":
                with torch.no_grad():
                    target = self.target_encoder(target_input)
            else:
                target = self.target_encoder(target_input)
            predicted_rows.append(pred)
            target_rows.append(target)
        predicted = torch.stack(predicted_rows, dim=1)
        target = torch.stack(target_rows, dim=1)
        return {"context": context, "target": target, "predicted": predicted}


def horizon_jepa_loss(
    outputs: dict[str, torch.Tensor],
    *,
    variance_weight: float = 0.05,
    covariance_weight: float = 0.005,
    retrieval_weight: float = 0.0,
    retrieval_temperature: float = 0.1,
    target_regularization_grad: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    context = outputs["context"]
    predicted = outputs["predicted"]
    target_raw = outputs["target"]
    target = target_raw.detach()
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must match, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )

    pred_flat = predicted.reshape(-1, predicted.shape[-1])
    target_health = target_raw if target_regularization_grad else target
    target_flat = target.reshape(-1, target.shape[-1])
    target_health_flat = target_health.reshape(-1, target_health.shape[-1])
    prediction = F.mse_loss(pred_flat, target_flat)
    var = (
        variance_loss(context)
        + variance_loss(pred_flat)
        + variance_loss(target_health_flat)
    ) / 3.0
    cov = (
        covariance_loss(context)
        + covariance_loss(pred_flat)
        + covariance_loss(target_health_flat)
    ) / 3.0
    if retrieval_weight > 0.0:
        retrieval_terms = [
            retrieval_contrastive_loss(
                predicted[:, horizon_idx, :],
                target_raw[:, horizon_idx, :],
                temperature=retrieval_temperature,
                detach_target=not target_regularization_grad,
            )
            for horizon_idx in range(predicted.shape[1])
        ]
        retrieval = torch.stack(retrieval_terms).mean()
    else:
        retrieval = prediction.new_tensor(0.0)
    loss = (
        prediction
        + variance_weight * var
        + covariance_weight * cov
        + retrieval_weight * retrieval
    )
    return loss, {
        "prediction": float(prediction.detach().cpu()),
        "variance": float(var.detach().cpu()),
        "covariance": float(cov.detach().cpu()),
        "retrieval": float(retrieval.detach().cpu()),
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


def _parameter_groups(model: HorizonJEPAWorldModel) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from (param for param in model.target_encoder.parameters() if param.requires_grad)
    yield from model.horizon_embedding.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def encode_horizon_split(
    model: HorizonJEPAWorldModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    rows = {"context": [], "target": [], "predicted": []}
    loader = _loader_from_arrays(past, future, batch_size=batch_size, shuffle=False)
    for past_batch, future_batch in loader:
        out = model(past_batch.to(device), future_batch.to(device))
        for key in rows:
            rows[key].append(out[key].detach().cpu().numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in rows.items()}


def evaluate_horizon_part1(
    model: HorizonJEPAWorldModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_horizon_split(model, past, future, batch_size=batch_size, device=device)
    predicted = encoded["predicted"]
    target = encoded["target"]

    per_horizon: dict[str, object] = {}
    for horizon_idx, horizon in enumerate(model.horizons):
        per_horizon[str(horizon)] = {
            "prediction": latent_prediction_metrics(
                predicted[:, horizon_idx, :],
                target[:, horizon_idx, :],
            ),
            "retrieval": retrieval_metrics(
                predicted[:, horizon_idx, :],
                target[:, horizon_idx, :],
                top_k=(1, 5, 10),
            ),
            "target_health": representation_health_metrics(target[:, horizon_idx, :]),
            "predicted_health": representation_health_metrics(predicted[:, horizon_idx, :]),
        }

    pred_flat = predicted.reshape(-1, predicted.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    return {
        "overall_prediction": latent_prediction_metrics(pred_flat, target_flat),
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
        "context_health": representation_health_metrics(encoded["context"]),
        "target_health": representation_health_metrics(target_flat),
        "predicted_health": representation_health_metrics(pred_flat),
        "per_horizon": per_horizon,
    }


def raw_horizon_frame_baseline(
    past: np.ndarray,
    future: np.ndarray,
    *,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    per_horizon: dict[str, object] = {}
    for horizon in horizons:
        predicted = past[:, -1, :]
        target = future[:, horizon - 1, :]
        per_horizon[str(horizon)] = {
            "prediction": latent_prediction_metrics(predicted, target),
            "retrieval": retrieval_metrics(predicted, target, top_k=(1, 5, 10)),
            "target_health": representation_health_metrics(target),
        }
    return {
        "per_horizon": per_horizon,
        "retrieval": {
            "mrr_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["mrr"] for h in horizons])),
            "top1_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top1"] for h in horizons])),
            "top5_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top5"] for h in horizons])),
            "top10_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top10"] for h in horizons])),
        },
        "prediction": {
            "mse_mean": float(np.mean([per_horizon[str(h)]["prediction"]["mse"] for h in horizons])),
            "cosine_mean": float(
                np.mean([per_horizon[str(h)]["prediction"]["cosine_mean"] for h in horizons])
            ),
        },
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


def train_smoke(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    horizons = validate_horizons(tuple(args.horizons), future_len=30)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    cfg = HorizonJEPAConfig(
        input_dim=25,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        horizons=horizons,
        ema_decay=args.ema_decay,
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

    model = HorizonJEPAWorldModel(
        cfg,
        target_mode=args.target_mode,
        target_encoder_mode=args.target_encoder_mode,
    ).to(device)
    opt = torch.optim.AdamW(_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader_from_arrays(
        train.past_window,
        train.future_window,
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_summary: dict[str, object] | None = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts_accum: dict[str, list[float]] = {}
        for past_batch, future_batch in train_loader:
            past_batch = past_batch.to(device)
            future_batch = future_batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(past_batch, future_batch)
            loss, parts = horizon_jepa_loss(
                out,
                variance_weight=args.variance_weight,
                covariance_weight=args.covariance_weight,
                retrieval_weight=args.retrieval_weight,
                retrieval_temperature=args.retrieval_temperature,
                target_regularization_grad=args.target_encoder_mode == "trainable",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            if args.target_encoder_mode == "ema":
                update_ema(model.context_encoder, model.target_encoder, cfg.ema_decay)
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                parts_accum.setdefault(key, []).append(value)

        val_metrics = evaluate_horizon_part1(
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
            "val_mrr_mean": val_metrics["overall_retrieval"]["mrr_mean"],
            "val_top1_mean": val_metrics["overall_retrieval"]["top1_mean"],
            "val_top5_mean": val_metrics["overall_retrieval"]["top5_mean"],
            "val_mse": val_metrics["overall_prediction"]["mse"],
            "val_predicted_effective_rank": val_metrics["predicted_health"]["effective_rank"],
        }
        history.append(row)
        if best_summary is None or row["val_mrr_mean"] > best_summary["val_mrr_mean"]:
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row))

    final_metrics = evaluate_horizon_part1(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = evaluate_horizon_part1(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    baseline = raw_horizon_frame_baseline(
        val.past_window,
        val.future_window,
        horizons=horizons,
    )
    result = {
        "config": asdict(cfg),
        "args": vars(args),
        "target_mode": args.target_mode,
        "target_encoder_mode": args.target_encoder_mode,
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
    parser = argparse.ArgumentParser(description="Horizon-specific IV-only JEPA smoke")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--predictor_hidden_dim", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--target_mode", choices=("prefix", "frame"), default="prefix")
    parser.add_argument("--target_encoder_mode", choices=("ema", "trainable"), default="ema")
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--variance_weight", type=float, default=0.2)
    parser.add_argument("--covariance_weight", type=float, default=0.005)
    parser.add_argument("--retrieval_weight", type=float, default=0.1)
    parser.add_argument("--retrieval_temperature", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7704)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_horizon_jepa_head006.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/horizon_jepa_head006.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
