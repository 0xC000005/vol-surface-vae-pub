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
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_metrics,
)
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (  # noqa: E402
    fit_delta_pca_target,
    make_horizon_delta_matrix,
    transform_delta_targets,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import validate_horizons  # noqa: E402
from experiments.world.part1_jepa_latent.jepa_smoke import (  # noqa: E402
    SequenceEncoder,
    update_ema,
)


@dataclass(frozen=True)
class FusedContextEMAJEPAConfig:
    input_dim: int = 25
    flat_input_dim: int = 750
    hidden_dim: int = 64
    latent_dim: int = 16
    predictor_hidden_dim: int = 128
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)
    ema_decay: float = 0.99


def relative_to_last_observation(past: torch.Tensor) -> torch.Tensor:
    if past.ndim != 3:
        raise ValueError(f"past must have shape (B, T, C), got {tuple(past.shape)}")
    return past - past[:, -1:, :]


def horizon_delta_frames(
    past: torch.Tensor,
    future: torch.Tensor,
    *,
    horizons: tuple[int, ...],
) -> torch.Tensor:
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError("past and future must have shape (B, T, C)")
    if past.shape[0] != future.shape[0] or past.shape[2] != future.shape[2]:
        raise ValueError("past and future must share batch and channel dimensions")
    validate_horizons(horizons, future_len=future.shape[1])
    frames = [future[:, horizon - 1 : horizon, :] - past[:, -1:, :] for horizon in horizons]
    return torch.cat(frames, dim=1)


class FusedContextEMAJEPAWorldModel(nn.Module):
    def __init__(self, cfg: FusedContextEMAJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        encoder_cfg = argparse.Namespace(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
        )
        self.sequence_encoder = SequenceEncoder(encoder_cfg)
        self.target_encoder = SequenceEncoder(encoder_cfg)
        self.target_encoder.load_state_dict(self.sequence_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.direct_encoder = nn.Sequential(
            nn.LayerNorm(cfg.flat_input_dim),
            nn.Linear(cfg.flat_input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim * 2),
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.SiLU(),
        )
        self.horizon_embedding = nn.Embedding(len(self.horizons), cfg.latent_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim * 2),
            nn.Linear(cfg.latent_dim * 2, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.latent_dim),
        )

    def encode_context(self, past: torch.Tensor) -> torch.Tensor:
        rel_past = relative_to_last_observation(past)
        flat = rel_past.reshape(rel_past.shape[0], -1)
        if flat.shape[1] != self.cfg.flat_input_dim:
            raise ValueError(f"flattened past dimension {flat.shape[1]} != {self.cfg.flat_input_dim}")
        seq_context = self.sequence_encoder(rel_past)
        direct_context = self.direct_encoder(flat)
        return self.fusion(torch.cat([seq_context, direct_context], dim=1))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode_context(past)
        delta_targets = horizon_delta_frames(past, future, horizons=self.horizons)
        predicted_rows = []
        target_rows = []
        batch_size = past.shape[0]
        for horizon_idx in range(len(self.horizons)):
            horizon_ids = torch.full(
                (batch_size,),
                horizon_idx,
                device=past.device,
                dtype=torch.long,
            )
            horizon_emb = self.horizon_embedding(horizon_ids)
            predicted_rows.append(self.predictor(torch.cat([context, horizon_emb], dim=1)))
            with torch.no_grad():
                target_rows.append(self.target_encoder(delta_targets[:, horizon_idx : horizon_idx + 1, :]))
        return {
            "context": context,
            "predicted": torch.stack(predicted_rows, dim=1),
            "target": torch.stack(target_rows, dim=1),
        }


def fused_context_ema_jepa_loss(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = outputs["predicted"]
    target = outputs["target"].detach()
    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target must match, got {tuple(predicted.shape)} != {tuple(target.shape)}")
    loss = F.mse_loss(predicted.reshape(-1, predicted.shape[-1]), target.reshape(-1, target.shape[-1]))
    return loss, {"prediction": float(loss.detach().cpu()), "loss": float(loss.detach().cpu())}


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


def _parameter_groups(model: FusedContextEMAJEPAWorldModel) -> Iterator[nn.Parameter]:
    yield from model.sequence_encoder.parameters()
    yield from model.direct_encoder.parameters()
    yield from model.fusion.parameters()
    yield from model.horizon_embedding.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def encode_fused_context_ema_split(
    model: FusedContextEMAJEPAWorldModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    rows = {"context": [], "predicted": [], "target": []}
    loader = _loader_from_arrays(past, future, batch_size=batch_size, shuffle=False)
    for past_batch, future_batch in loader:
        out = model(past_batch.to(device), future_batch.to(device))
        for key in rows:
            rows[key].append(out[key].detach().cpu().numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in rows.items()}


def evaluate_fused_context_ema_jepa(
    model: FusedContextEMAJEPAWorldModel,
    past: np.ndarray,
    future: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_fused_context_ema_split(model, past, future, batch_size=batch_size, device=device)
    predicted = encoded["predicted"]
    target = encoded["target"]
    per_horizon: dict[str, object] = {}
    for horizon_idx, horizon in enumerate(model.horizons):
        per_horizon[str(horizon)] = {
            "prediction": latent_prediction_metrics(predicted[:, horizon_idx, :], target[:, horizon_idx, :]),
            "retrieval": retrieval_metrics(predicted[:, horizon_idx, :], target[:, horizon_idx, :], top_k=(1, 5, 10)),
            "predicted_health": representation_health_metrics(predicted[:, horizon_idx, :]),
            "target_health": representation_health_metrics(target[:, horizon_idx, :]),
        }
    return {
        "overall_prediction": latent_prediction_metrics(predicted, target),
        "overall_retrieval": {
            "mrr_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["mrr"] for h in model.horizons])),
            "top1_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top1"] for h in model.horizons])),
            "top5_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top5"] for h in model.horizons])),
            "top10_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top10"] for h in model.horizons])),
        },
        "context_health": representation_health_metrics(encoded["context"]),
        "predicted_health": representation_health_metrics(predicted.reshape(-1, predicted.shape[-1])),
        "target_health": representation_health_metrics(target.reshape(-1, target.shape[-1])),
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
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train = build_iv_world_windows(split="train", max_windows=args.max_train_windows, normalize=True)
    val = build_iv_world_windows(split="val", max_windows=args.max_val_windows, normalize=True)
    train_delta = make_horizon_delta_matrix(train.past_window, train.future_window, horizons=horizons)
    val_delta = make_horizon_delta_matrix(val.past_window, val.future_window, horizons=horizons)
    pca_target = fit_delta_pca_target(train_delta, target_dim=args.pca_target_dim)
    train_z = transform_delta_targets(train_delta, pca_target)
    val_z = transform_delta_targets(val_delta, pca_target)

    cfg = FusedContextEMAJEPAConfig(
        input_dim=train.past_window.shape[-1],
        flat_input_dim=train.past_window.shape[1] * train.past_window.shape[2],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        horizons=horizons,
        ema_decay=args.ema_decay,
    )
    model = FusedContextEMAJEPAWorldModel(cfg).to(device)
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
        for past_batch, future_batch in train_loader:
            past_batch = past_batch.to(device)
            future_batch = future_batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(past_batch, future_batch)
            loss, _parts = fused_context_ema_jepa_loss(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            update_ema(model.sequence_encoder, model.target_encoder, cfg.ema_decay)
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_fused_context_ema_jepa(
            model,
            val.past_window,
            val.future_window,
            batch_size=args.batch_size,
            device=device,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_mse": val_metrics["overall_prediction"]["mse"],
            "val_mrr_mean": val_metrics["overall_retrieval"]["mrr_mean"],
            "val_top5_mean": val_metrics["overall_retrieval"]["top5_mean"],
            "val_predicted_effective_rank": val_metrics["predicted_health"]["effective_rank"],
            "val_target_effective_rank": val_metrics["target_health"]["effective_rank"],
            "val_context_effective_rank": val_metrics["context_health"]["effective_rank"],
        }
        history.append(row)
        if best_summary is None or row["val_mrr_mean"] > best_summary["val_mrr_mean"]:
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row))

    final_metrics = evaluate_fused_context_ema_jepa(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = evaluate_fused_context_ema_jepa(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    train_encoded = encode_fused_context_ema_split(
        model,
        train.past_window,
        train.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    val_encoded = encode_fused_context_ema_split(
        model,
        val.past_window,
        val.future_window,
        batch_size=args.batch_size,
        device=device,
    )
    ridge_fixed_pca_metrics = ridge_probe_metrics(
        train_encoded["context"],
        val_encoded["context"],
        train_z,
        val_z,
        horizons=horizons,
        alpha=args.ridge_alpha,
    )
    ridge_delta_metrics = ridge_probe_metrics(
        train_encoded["context"],
        val_encoded["context"],
        train_delta,
        val_delta,
        horizons=horizons,
        alpha=args.ridge_alpha,
    )

    result = {
        "config": asdict(cfg),
        "args": vars(args),
        "literature_status": "canonical_jepa_adaptation",
        "literature_rationale": (
            "EMA target encoder and context-to-target latent prediction follow I-JEPA/A-JEPA; "
            "relative-past and horizon-delta targets are the time-series target-block adaptation."
        ),
        "device": str(device),
        "train_shape": {
            "past": list(train.past_window.shape),
            "future": list(train.future_window.shape),
            "delta": list(train_delta.shape),
        },
        "val_shape": {
            "past": list(val.past_window.shape),
            "future": list(val.future_window.shape),
            "delta": list(val_delta.shape),
        },
        "history": history,
        "best_epoch_summary": best_summary,
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "ridge_probe_fixed_pca_metrics": ridge_fixed_pca_metrics,
        "ridge_probe_delta_metrics": ridge_delta_metrics,
        "reference_thresholds": {
            "head038_primary_fixed_pca_mrr": 0.103763,
            "head038_primary_fixed_pca_top5": 0.136719,
            "head038_primary_raw_delta_mse": 0.015701,
            "head038_primary_raw_delta_mrr": 0.096147,
            "head038_primary_raw_delta_top5": 0.128125,
        },
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
    parser = argparse.ArgumentParser(description="Fused-context EMA JEPA with relative-past/delta-frame targets")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--pca_target_dim", type=int, default=8)
    parser.add_argument("--ridge_alpha", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7712)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_fused_context_ema_jepa_head056.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/fused_context_ema_jepa_head056.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
