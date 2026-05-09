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
    representation_health_metrics,
)
from experiments.world.evaluation.world_data import build_iv_world_windows  # noqa: E402
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (  # noqa: E402
    fit_delta_pca_target,
    make_horizon_delta_matrix,
    transform_delta_targets,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import (  # noqa: E402
    validate_horizons,
)
from experiments.world.part1_jepa_latent.jepa_smoke import SequenceEncoder  # noqa: E402
from experiments.world.part1_jepa_latent.supervised_horizon_frame import (  # noqa: E402
    raw_horizon_persistence_baseline,
)
from experiments.world.part1_jepa_latent.target_encoder_distill import (  # noqa: E402
    DeltaTargetEncoder,
    TargetEncoderDistillConfig,
    evaluate_distilled_targets,
    pca_oracle_delta_mse,
    predict_distilled_targets,
    target_encoder_distill_loss,
)


@dataclass(frozen=True)
class FrozenTargetJEPAConfig:
    input_dim: int = 25
    hidden_dim: int = 64
    context_dim: int = 32
    target_dim: int = 8
    predictor_hidden_dim: int = 128
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)


class FrozenTargetJEPAWorldModel(nn.Module):
    def __init__(self, cfg: FrozenTargetJEPAConfig):
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


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def frozen_target_jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.mse_loss(predicted, target.detach())
    value = float(loss.detach().cpu())
    return loss, {"prediction": value, "loss": value}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def _target_parameters(model: DeltaTargetEncoder) -> Iterator[nn.Parameter]:
    yield from model.parameters()


def _jepa_parameters(model: FrozenTargetJEPAWorldModel) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.horizon_embedding.parameters()
    yield from model.predictor.parameters()


@torch.no_grad()
def predict_frozen_target_jepa(
    model: FrozenTargetJEPAWorldModel,
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


def evaluate_frozen_target_jepa(
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


def train_target_encoder_stage(
    *,
    train_delta: np.ndarray,
    train_z: np.ndarray,
    val_delta: np.ndarray,
    val_z: np.ndarray,
    pca_target,
    cfg: TargetEncoderDistillConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
) -> tuple[DeltaTargetEncoder, list[dict[str, float]], dict[str, object]]:
    target_encoder = DeltaTargetEncoder(cfg).to(device)
    opt = torch.optim.AdamW(_target_parameters(target_encoder), lr=lr, weight_decay=weight_decay)
    loader = _loader_from_arrays(train_delta, train_z, batch_size=batch_size, shuffle=True)
    history: list[dict[str, float]] = []
    best_state = None
    best_summary: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        target_encoder.train()
        losses = []
        for delta_batch, target_batch in loader:
            delta_batch = delta_batch.to(device)
            target_batch = target_batch.to(device)
            opt.zero_grad(set_to_none=True)
            predicted = target_encoder(delta_batch)
            loss, _parts = target_encoder_distill_loss(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_target_parameters(target_encoder)), grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict_distilled_targets(
            target_encoder,
            val_delta,
            batch_size=batch_size,
            device=device,
        )
        val_metrics = evaluate_distilled_targets(
            predicted_z=val_pred,
            target_z=val_z,
            truth_delta=val_delta,
            pca_target=pca_target,
            horizons=cfg.horizons,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_mse": val_metrics["overall_prediction"]["mse"],
            "val_mrr_mean": val_metrics["overall_retrieval"]["mrr_mean"],
            "val_delta_decode_mse": val_metrics["overall_delta_decode"]["mse_mean"],
            "val_predicted_effective_rank": val_metrics["predicted_health"]["effective_rank"],
        }
        history.append(row)
        if best_summary is None or row["val_mse"] < best_summary["val_mse"]:
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in target_encoder.state_dict().items()}
        print(json.dumps({"stage": "target_encoder", **row}))

    if best_state is not None:
        target_encoder.load_state_dict(best_state)
    target_encoder = freeze_module(target_encoder)
    best_pred = predict_distilled_targets(
        target_encoder,
        val_delta,
        batch_size=batch_size,
        device=device,
    )
    best_metrics = evaluate_distilled_targets(
        predicted_z=best_pred,
        target_z=val_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=cfg.horizons,
    )
    return target_encoder, history, best_metrics


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
    train_pca_z = transform_delta_targets(train_delta, pca_target)
    val_pca_z = transform_delta_targets(val_delta, pca_target)

    target_cfg = TargetEncoderDistillConfig(
        input_dim=train_delta.shape[-1],
        hidden_dim=args.hidden_dim,
        target_dim=args.target_dim,
        horizons=horizons,
    )
    target_encoder, target_history, target_quality = train_target_encoder_stage(
        train_delta=train_delta,
        train_z=train_pca_z,
        val_delta=val_delta,
        val_z=val_pca_z,
        pca_target=pca_target,
        cfg=target_cfg,
        epochs=args.target_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=device,
    )

    train_target_z = predict_distilled_targets(
        target_encoder,
        train_delta,
        batch_size=args.batch_size,
        device=device,
    )
    val_target_z = predict_distilled_targets(
        target_encoder,
        val_delta,
        batch_size=args.batch_size,
        device=device,
    )

    jepa_cfg = FrozenTargetJEPAConfig(
        input_dim=train.past_window.shape[-1],
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        target_dim=args.target_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        horizons=horizons,
    )
    model = FrozenTargetJEPAWorldModel(jepa_cfg).to(device)
    opt = torch.optim.AdamW(_jepa_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader_from_arrays(
        train.past_window,
        train_target_z,
        batch_size=args.batch_size,
        shuffle=True,
    )

    history = []
    best_summary: dict[str, object] | None = None
    best_state = None
    for epoch in range(1, args.predictor_epochs + 1):
        model.train()
        losses = []
        for past_batch, target_batch in train_loader:
            past_batch = past_batch.to(device)
            target_batch = target_batch.to(device)
            opt.zero_grad(set_to_none=True)
            predicted = model(past_batch)
            loss, _parts = frozen_target_jepa_loss(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_jepa_parameters(model)), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred, val_context = predict_frozen_target_jepa(
            model,
            val.past_window,
            batch_size=args.batch_size,
            device=device,
        )
        val_metrics = evaluate_frozen_target_jepa(
            predicted_z=val_pred,
            target_z=val_target_z,
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
        print(json.dumps({"stage": "frozen_target_jepa", **row}))

    final_pred, final_context = predict_frozen_target_jepa(
        model,
        val.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    final_metrics = evaluate_frozen_target_jepa(
        predicted_z=final_pred,
        target_z=val_target_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=final_context,
    )
    final_vs_pca_metrics = evaluate_frozen_target_jepa(
        predicted_z=final_pred,
        target_z=val_pca_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=final_context,
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    best_pred, best_context = predict_frozen_target_jepa(
        model,
        val.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    best_metrics = evaluate_frozen_target_jepa(
        predicted_z=best_pred,
        target_z=val_target_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=best_context,
    )
    best_vs_pca_metrics = evaluate_frozen_target_jepa(
        predicted_z=best_pred,
        target_z=val_pca_z,
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
        "config": {
            "target_encoder": asdict(target_cfg),
            "frozen_target_jepa": asdict(jepa_cfg),
        },
        "args": vars(args),
        "target_contract": {
            "kind": "distilled_fixed_delta_pca_target_encoder",
            "target_dim": args.target_dim,
            "mean": pca_target.mean,
            "components": pca_target.components,
            "scale": pca_target.scale,
        },
        "pca_oracle_delta_mse": pca_oracle_delta_mse(val_pca_z, val_delta, pca_target),
        "reference_thresholds": {
            "head028_fixed_pca_mrr": 0.09637384278162231,
            "head028_fixed_pca_val_mse": 0.9871298567806953,
            "head028_fixed_pca_delta_decode_mse": 0.015368715906110709,
            "head031_target_encoder_val_mse": 0.008652427596281726,
            "head031_target_encoder_mrr": 0.9984375,
            "head031_target_encoder_delta_decode_mse": 0.0019901794836017276,
        },
        "device": str(device),
        "train_shape": {
            "past": list(train.past_window.shape),
            "delta": list(train_delta.shape),
            "target_z": list(train_target_z.shape),
        },
        "val_shape": {
            "past": list(val.past_window.shape),
            "delta": list(val_delta.shape),
            "target_z": list(val_target_z.shape),
        },
        "target_encoder_history": target_history,
        "target_encoder_val_vs_pca": target_quality,
        "history": history,
        "best_epoch_summary": best_summary,
        "best_val_metrics": best_metrics,
        "best_val_vs_fixed_pca_metrics": best_vs_pca_metrics,
        "final_val_metrics": final_metrics,
        "final_val_vs_fixed_pca_metrics": final_vs_pca_metrics,
        "raw_horizon_frame_baseline": baseline,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_serializable(result), indent=2), encoding="utf-8")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": result["config"],
            "target_encoder_state_dict": target_encoder.state_dict(),
            "state_dict": model.state_dict(),
            "result": _serializable(result),
        },
        checkpoint,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="JEPA predictor against a frozen distilled target encoder")
    parser.add_argument("--target_epochs", type=int, default=25)
    parser.add_argument("--predictor_epochs", type=int, default=25)
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
    parser.add_argument("--seed", type=int, default=7708)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_frozen_target_jepa_head032.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/frozen_target_jepa_head032.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
