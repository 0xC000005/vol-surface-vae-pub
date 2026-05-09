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


@dataclass(frozen=True)
class JEPAConfig:
    input_dim: int = 25
    hidden_dim: int = 64
    latent_dim: int = 16
    predictor_hidden_dim: int = 64
    ema_decay: float = 0.99


class SequenceEncoder(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.input_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _seq, h_n = self.gru(x)
        return self.head(h_n[-1])


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = SequenceEncoder(cfg)
        self.target_encoder = SequenceEncoder(cfg)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.latent_dim),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.context_encoder(past)
        with torch.no_grad():
            target = self.target_encoder(future)
        predicted = self.predictor(context)
        return {"context": context, "target": target, "predicted": predicted}


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, decay: float) -> None:
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1]")
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.mul_(decay).add_(source_param, alpha=1.0 - decay)


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    denom = max(z.shape[0] - 1, 1)
    cov = z.T @ z / denom
    offdiag = cov - torch.diag(torch.diag(cov))
    return (offdiag * offdiag).sum() / z.shape[1]


def retrieval_contrastive_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must match, got {tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[0] < 2:
        raise ValueError("retrieval_contrastive_loss needs at least two samples")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    pred_norm = F.normalize(predicted, dim=1)
    target_norm = F.normalize(target.detach(), dim=1)
    logits = pred_norm @ target_norm.T / temperature
    labels = torch.arange(predicted.shape[0], device=predicted.device)
    row_loss = F.cross_entropy(logits, labels)
    col_loss = F.cross_entropy(logits.T, labels)
    return 0.5 * (row_loss + col_loss)


def jepa_loss(
    outputs: dict[str, torch.Tensor],
    *,
    variance_weight: float = 0.05,
    covariance_weight: float = 0.005,
    retrieval_weight: float = 0.0,
    retrieval_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = outputs["predicted"]
    target = outputs["target"].detach()
    context = outputs["context"]

    prediction = F.mse_loss(predicted, target)
    var = (
        variance_loss(context)
        + variance_loss(predicted)
        + variance_loss(target)
    ) / 3.0
    cov = (
        covariance_loss(context)
        + covariance_loss(predicted)
        + covariance_loss(target)
    ) / 3.0
    retrieval = (
        retrieval_contrastive_loss(
            predicted,
            target,
            temperature=retrieval_temperature,
        )
        if retrieval_weight > 0.0
        else prediction.new_tensor(0.0)
    )
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


def _parameter_groups(model: JEPAWorldModel) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def encode_split(
    model: JEPAWorldModel,
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


def evaluate_part1(
    model: JEPAWorldModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_split(model, past, future, batch_size=batch_size, device=device)
    return {
        "prediction": latent_prediction_metrics(encoded["predicted"], encoded["target"]),
        "retrieval": retrieval_metrics(encoded["predicted"], encoded["target"], top_k=(1, 5, 10)),
        "context_health": representation_health_metrics(encoded["context"]),
        "target_health": representation_health_metrics(encoded["target"]),
        "predicted_health": representation_health_metrics(encoded["predicted"]),
    }


def raw_placeholder_baseline(past: np.ndarray, future: np.ndarray) -> dict[str, object]:
    predicted = past[:, -1, :]
    target = future.mean(axis=1)
    return {
        "prediction": latent_prediction_metrics(predicted, target),
        "retrieval": retrieval_metrics(predicted, target, top_k=(1, 5, 10)),
        "target_health": representation_health_metrics(target),
    }


def _serializable(obj):
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def train_smoke(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    cfg = JEPAConfig(
        input_dim=25,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
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

    model = JEPAWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader_from_arrays(
        train.past_window,
        train.future_window,
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts_accum: dict[str, list[float]] = {}
        for past_batch, future_batch in train_loader:
            past_batch = past_batch.to(device)
            future_batch = future_batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(past_batch, future_batch)
            loss, parts = jepa_loss(
                out,
                variance_weight=args.variance_weight,
                covariance_weight=args.covariance_weight,
                retrieval_weight=args.retrieval_weight,
                retrieval_temperature=args.retrieval_temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            update_ema(model.context_encoder, model.target_encoder, cfg.ema_decay)
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                parts_accum.setdefault(key, []).append(value)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{f"{k}_mean": float(np.mean(v)) for k, v in parts_accum.items()},
        }
        history.append(row)
        print(json.dumps(row))

    val_metrics = evaluate_part1(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    baseline = raw_placeholder_baseline(val.past_window, val.future_window)
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
        "val_metrics": val_metrics,
        "raw_placeholder_baseline": baseline,
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
    parser = argparse.ArgumentParser(description="Minimal IV-only JEPA world-model smoke")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--predictor_hidden_dim", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--variance_weight", type=float, default=0.05)
    parser.add_argument("--covariance_weight", type=float, default=0.005)
    parser.add_argument("--retrieval_weight", type=float, default=0.0)
    parser.add_argument("--retrieval_temperature", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7703)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_jepa_smoke_head003.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/jepa_smoke_head003.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
