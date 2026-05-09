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

from experiments.world.evaluation.part1_metrics import representation_health_metrics  # noqa: E402
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (  # noqa: E402
    fit_delta_pca_target,
    make_horizon_delta_matrix,
    transform_delta_targets,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import validate_horizons  # noqa: E402
from experiments.world.part1_jepa_latent.supervised_horizon_frame import (  # noqa: E402
    raw_horizon_persistence_baseline,
)
from experiments.world.part1_jepa_latent.target_encoder_distill import (  # noqa: E402
    evaluate_distilled_targets,
    pca_oracle_delta_mse,
)


@dataclass(frozen=True)
class DirectDeltaPCAConfig:
    input_dim: int = 750
    hidden_dim: int = 128
    target_dim: int = 8
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)


def flatten_past_window(past: torch.Tensor) -> torch.Tensor:
    if past.ndim != 3:
        raise ValueError(f"past must have shape (B, T, C), got {tuple(past.shape)}")
    return past.reshape(past.shape[0], past.shape[1] * past.shape[2])


class DirectDeltaPCAPredictor(nn.Module):
    def __init__(self, cfg: DirectDeltaPCAConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.horizon_embedding = nn.Embedding(len(self.horizons), cfg.hidden_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.target_dim),
        )

    def forward(
        self,
        past: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        flat = flatten_past_window(past)
        if flat.shape[1] != self.cfg.input_dim:
            raise ValueError(f"flattened past dimension {flat.shape[1]} != {self.cfg.input_dim}")
        context = self.trunk(flat)
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


def direct_delta_pca_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.mse_loss(predicted, target)
    value = float(loss.detach().cpu())
    return loss, {"target_mse": value, "loss": value}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader_from_arrays(
    past: np.ndarray,
    target_z: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(past), torch.from_numpy(target_z))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _parameter_groups(model: DirectDeltaPCAPredictor) -> Iterator[nn.Parameter]:
    yield from model.parameters()


@torch.no_grad()
def predict_direct_delta_pca(
    model: DirectDeltaPCAPredictor,
    past: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_rows = []
    context_rows = []
    loader = DataLoader(torch.from_numpy(past), batch_size=batch_size, shuffle=False)
    for batch in loader:
        predicted, context = model(batch.to(device), return_context=True)
        pred_rows.append(predicted.detach().cpu().numpy())
        context_rows.append(context.detach().cpu().numpy())
    return np.concatenate(pred_rows, axis=0), np.concatenate(context_rows, axis=0)


def evaluate_direct_delta_pca(
    *,
    predicted_z: np.ndarray,
    target_z: np.ndarray,
    truth_delta: np.ndarray,
    pca_target,
    horizons: tuple[int, ...],
    context: np.ndarray,
) -> dict[str, object]:
    metrics = evaluate_distilled_targets(
        predicted_z=predicted_z,
        target_z=target_z,
        truth_delta=truth_delta,
        pca_target=pca_target,
        horizons=horizons,
    )
    metrics["context_health"] = representation_health_metrics(context)
    return metrics


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

    cfg = DirectDeltaPCAConfig(
        input_dim=train.past_window.shape[1] * train.past_window.shape[2],
        hidden_dim=args.hidden_dim,
        target_dim=args.target_dim,
        horizons=horizons,
    )
    model = DirectDeltaPCAPredictor(cfg).to(device)
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
            loss, _parts = direct_delta_pca_loss(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred, val_context = predict_direct_delta_pca(
            model,
            val.past_window,
            batch_size=args.batch_size,
            device=device,
        )
        val_metrics = evaluate_direct_delta_pca(
            predicted_z=val_pred,
            target_z=val_z,
            truth_delta=val_delta,
            pca_target=pca_target,
            horizons=horizons,
            context=val_context,
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
            "val_context_effective_rank": val_metrics["context_health"]["effective_rank"],
        }
        history.append(row)
        if best_summary is None or row["val_mse"] < best_summary["val_mse"]:
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row))

    final_pred, final_context = predict_direct_delta_pca(
        model,
        val.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    final_metrics = evaluate_direct_delta_pca(
        predicted_z=final_pred,
        target_z=val_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=final_context,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_pred, best_context = predict_direct_delta_pca(
        model,
        val.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    best_metrics = evaluate_direct_delta_pca(
        predicted_z=best_pred,
        target_z=val_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=best_context,
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
        "pca_oracle_delta_mse": pca_oracle_delta_mse(val_z, val_delta, pca_target),
        "reference_thresholds": {
            "head028_fixed_pca_mrr": 0.09637384278162231,
            "head028_fixed_pca_top5": 0.13203125,
            "head028_fixed_pca_top10": 0.203125,
            "head028_fixed_pca_val_mse": 0.9871298567806953,
            "head028_fixed_pca_delta_decode_mse": 0.015368715906110709,
            "head032_frozen_target_mrr": 0.0851397462581691,
            "head032_frozen_target_val_mse": 0.9792700251917029,
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
    parser = argparse.ArgumentParser(description="Direct flattened-past predictor to fixed delta-PCA targets")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--target_dim", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7709)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_direct_delta_pca_predictor_head034.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/direct_delta_pca_predictor_head034.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
