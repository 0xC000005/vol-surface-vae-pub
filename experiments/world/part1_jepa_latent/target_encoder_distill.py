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
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (  # noqa: E402
    FixedDeltaPCATarget,
    fit_delta_pca_target,
    inverse_transform_delta_targets,
    make_horizon_delta_matrix,
    transform_delta_targets,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import (  # noqa: E402
    validate_horizons,
)


@dataclass(frozen=True)
class TargetEncoderDistillConfig:
    input_dim: int = 25
    hidden_dim: int = 64
    target_dim: int = 8
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)


class DeltaTargetEncoder(nn.Module):
    def __init__(self, cfg: TargetEncoderDistillConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        self.delta_projection = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.horizon_embedding = nn.Embedding(len(self.horizons), cfg.hidden_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.target_dim),
        )

    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        if deltas.ndim != 3:
            raise ValueError(f"deltas must have shape (B, H, C), got {tuple(deltas.shape)}")
        if deltas.shape[1] != len(self.horizons):
            raise ValueError(
                f"deltas horizon dimension {deltas.shape[1]} does not match {len(self.horizons)}"
            )
        if deltas.shape[2] != self.cfg.input_dim:
            raise ValueError(f"deltas feature dimension {deltas.shape[2]} != {self.cfg.input_dim}")

        horizon_ids = torch.arange(len(self.horizons), device=deltas.device)
        horizon_emb = self.horizon_embedding(horizon_ids).unsqueeze(0)
        hidden = self.delta_projection(deltas) + horizon_emb
        return self.encoder(hidden)


def target_encoder_distill_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.mse_loss(predicted, target)
    value = float(loss.detach().cpu())
    return loss, {"target_mse": value, "loss": value}


def evaluate_distilled_targets(
    *,
    predicted_z: np.ndarray,
    target_z: np.ndarray,
    truth_delta: np.ndarray,
    pca_target: FixedDeltaPCATarget,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    decoded_delta = inverse_transform_delta_targets(predicted_z, pca_target)

    per_horizon: dict[str, object] = {}
    for horizon_idx, horizon in enumerate(horizons):
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
            "mrr_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["mrr"] for h in horizons])),
            "top1_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top1"] for h in horizons])),
            "top5_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top5"] for h in horizons])),
            "top10_mean": float(np.mean([per_horizon[str(h)]["retrieval"]["top10"] for h in horizons])),
        },
        "overall_delta_decode": {
            "mse_mean": float(np.mean([per_horizon[str(h)]["delta_decode"]["mse"] for h in horizons])),
            "cosine_mean": float(
                np.mean([per_horizon[str(h)]["delta_decode"]["cosine_mean"] for h in horizons])
            ),
        },
        "predicted_health": representation_health_metrics(predicted_z.reshape(-1, predicted_z.shape[-1])),
        "target_health": representation_health_metrics(target_z.reshape(-1, target_z.shape[-1])),
        "per_horizon": per_horizon,
    }


def pca_oracle_delta_mse(
    target_z: np.ndarray,
    truth_delta: np.ndarray,
    pca_target: FixedDeltaPCATarget,
) -> float:
    decoded = inverse_transform_delta_targets(target_z, pca_target)
    return float(np.mean((decoded - truth_delta) ** 2))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader_from_arrays(
    deltas: np.ndarray,
    target_z: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(deltas), torch.from_numpy(target_z))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _parameter_groups(model: DeltaTargetEncoder) -> Iterator[nn.Parameter]:
    yield from model.parameters()


@torch.no_grad()
def predict_distilled_targets(
    model: DeltaTargetEncoder,
    deltas: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows = []
    loader = DataLoader(torch.from_numpy(deltas), batch_size=batch_size, shuffle=False)
    for batch in loader:
        rows.append(model(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


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

    cfg = TargetEncoderDistillConfig(
        input_dim=train_delta.shape[-1],
        hidden_dim=args.hidden_dim,
        target_dim=args.target_dim,
        horizons=horizons,
    )
    model = DeltaTargetEncoder(cfg).to(device)
    opt = torch.optim.AdamW(_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader_from_arrays(
        train_delta,
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
        for delta_batch, target_batch in train_loader:
            delta_batch = delta_batch.to(device)
            target_batch = target_batch.to(device)
            opt.zero_grad(set_to_none=True)
            predicted = model(delta_batch)
            loss, _parts = target_encoder_distill_loss(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict_distilled_targets(
            model,
            val_delta,
            batch_size=args.batch_size,
            device=device,
        )
        val_metrics = evaluate_distilled_targets(
            predicted_z=val_pred,
            target_z=val_z,
            truth_delta=val_delta,
            pca_target=pca_target,
            horizons=horizons,
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
        if best_summary is None or row["val_mse"] < best_summary["val_mse"]:
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row))

    final_pred = predict_distilled_targets(
        model,
        val_delta,
        batch_size=args.batch_size,
        device=device,
    )
    final_metrics = evaluate_distilled_targets(
        predicted_z=final_pred,
        target_z=val_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_pred = predict_distilled_targets(
        model,
        val_delta,
        batch_size=args.batch_size,
        device=device,
    )
    best_metrics = evaluate_distilled_targets(
        predicted_z=best_pred,
        target_z=val_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
    )
    result = {
        "config": asdict(cfg),
        "args": vars(args),
        "target_contract": {
            "kind": "fixed_delta_pca_teacher",
            "target_dim": args.target_dim,
            "mean": pca_target.mean,
            "components": pca_target.components,
            "scale": pca_target.scale,
        },
        "pca_oracle_delta_mse": pca_oracle_delta_mse(val_z, val_delta, pca_target),
        "reference_thresholds": {
            "head028_context_predictor_val_mse": 0.9871298567806953,
            "head028_context_predictor_delta_decode_mse": 0.015368715906110709,
            "head025_predicted_effective_rank": 1.262683,
        },
        "device": str(device),
        "train_shape": {
            "delta": list(train_delta.shape),
            "target_z": list(train_z.shape),
        },
        "val_shape": {
            "delta": list(val_delta.shape),
            "target_z": list(val_z.shape),
        },
        "history": history,
        "best_epoch_summary": best_summary,
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
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
    parser = argparse.ArgumentParser(description="Target encoder distillation to fixed delta-PCA targets")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--target_dim", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7707)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_target_encoder_distill_head031.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/target_encoder_distill_head031.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
