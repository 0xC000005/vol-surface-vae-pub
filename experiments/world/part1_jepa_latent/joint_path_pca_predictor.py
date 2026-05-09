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
    horizon_target_metrics,
    ridge_probe_metrics,
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.direct_delta_pca_predictor import (  # noqa: E402
    flatten_past_window,
)
from experiments.world.part1_jepa_latent.fixed_delta_pca_jepa import (  # noqa: E402
    make_horizon_delta_matrix,
)
from experiments.world.part1_jepa_latent.horizon_jepa_smoke import validate_horizons  # noqa: E402
from experiments.world.part1_jepa_latent.jepa_smoke import SequenceEncoder  # noqa: E402


@dataclass(frozen=True)
class JointDeltaPathPCATarget:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray
    horizons: tuple[int, ...]
    input_dim: int

    @property
    def target_dim(self) -> int:
        return int(self.components.shape[1])


@dataclass(frozen=True)
class FusedContextJointPathPCAConfig:
    input_dim: int = 25
    flat_input_dim: int = 750
    hidden_dim: int = 64
    context_dim: int = 32
    target_dim: int = 16
    predictor_hidden_dim: int = 128
    horizons: tuple[int, ...] = (1, 5, 10, 20, 30)


class FusedContextJointPathPCAPredictor(nn.Module):
    def __init__(self, cfg: FusedContextJointPathPCAConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        self.sequence_encoder = SequenceEncoder(
            argparse.Namespace(
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                latent_dim=cfg.context_dim,
            )
        )
        self.direct_encoder = nn.Sequential(
            nn.LayerNorm(cfg.flat_input_dim),
            nn.Linear(cfg.flat_input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.context_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(cfg.context_dim * 2),
            nn.Linear(cfg.context_dim * 2, cfg.context_dim),
            nn.SiLU(),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.context_dim),
            nn.Linear(cfg.context_dim, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.target_dim),
        )

    def encode_context(self, past: torch.Tensor) -> torch.Tensor:
        flat = flatten_past_window(past)
        if flat.shape[1] != self.cfg.flat_input_dim:
            raise ValueError(f"flattened past dimension {flat.shape[1]} != {self.cfg.flat_input_dim}")
        seq_context = self.sequence_encoder(past)
        direct_context = self.direct_encoder(flat)
        return self.fusion(torch.cat([seq_context, direct_context], dim=1))

    def forward(
        self,
        past: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(past)
        predicted = self.predictor(context)
        if return_context:
            return predicted, context
        return predicted


def fit_joint_delta_path_pca_target(
    deltas: np.ndarray,
    *,
    horizons: tuple[int, ...],
    target_dim: int = 16,
) -> JointDeltaPathPCATarget:
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("deltas must have shape (B, H, C)")
    if arr.shape[1] != len(horizons):
        raise ValueError("horizon count does not match delta matrix")
    flat = arr.reshape(arr.shape[0], -1)
    if target_dim <= 0 or target_dim > flat.shape[1]:
        raise ValueError(f"target_dim must be in [1, {flat.shape[1]}]")
    mean = flat.mean(axis=0)
    centered = flat - mean
    _u, _singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:target_dim].T
    projected = centered @ components
    scale = projected.std(axis=0).clip(min=1e-8)
    return JointDeltaPathPCATarget(
        mean=mean.astype(np.float64),
        components=components.astype(np.float64),
        scale=scale.astype(np.float64),
        horizons=tuple(int(h) for h in horizons),
        input_dim=int(arr.shape[-1]),
    )


def transform_joint_delta_path_targets(
    deltas: np.ndarray,
    target: JointDeltaPathPCATarget,
) -> np.ndarray:
    arr = np.asarray(deltas, dtype=np.float64)
    flat = arr.reshape(arr.shape[0], -1)
    z = ((flat - target.mean) @ target.components) / target.scale
    return z.astype(np.float32)


def inverse_transform_joint_delta_path_targets(
    z: np.ndarray,
    target: JointDeltaPathPCATarget,
) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    flat = (arr * target.scale) @ target.components.T + target.mean
    return flat.reshape(arr.shape[0], len(target.horizons), target.input_dim).astype(np.float32)


def joint_path_pca_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.mse_loss(predicted, target)
    value = float(loss.detach().cpu())
    return loss, {"target_mse": value, "loss": value}


def evaluate_joint_path_pca(
    *,
    predicted_z: np.ndarray,
    target_z: np.ndarray,
    truth_delta: np.ndarray,
    pca_target: JointDeltaPathPCATarget,
    horizons: tuple[int, ...],
    context: np.ndarray | None = None,
) -> dict[str, object]:
    decoded_delta = inverse_transform_joint_delta_path_targets(predicted_z, pca_target)
    out = {
        "path_prediction": latent_prediction_metrics(predicted_z, target_z),
        "path_retrieval": retrieval_metrics(predicted_z, target_z, top_k=(1, 5, 10)),
        "delta_decode": horizon_target_metrics(decoded_delta, truth_delta, horizons=horizons),
        "predicted_health": representation_health_metrics(predicted_z),
        "target_health": representation_health_metrics(target_z),
    }
    if context is not None:
        out["context_health"] = representation_health_metrics(context)
    return out


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


def _parameter_groups(model: FusedContextJointPathPCAPredictor) -> Iterator[nn.Parameter]:
    yield from model.parameters()


@torch.no_grad()
def predict_joint_path_pca(
    model: FusedContextJointPathPCAPredictor,
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
    pca_target = fit_joint_delta_path_pca_target(train_delta, horizons=horizons, target_dim=args.target_dim)
    train_z = transform_joint_delta_path_targets(train_delta, pca_target)
    val_z = transform_joint_delta_path_targets(val_delta, pca_target)

    cfg = FusedContextJointPathPCAConfig(
        input_dim=train.past_window.shape[-1],
        flat_input_dim=train.past_window.shape[1] * train.past_window.shape[2],
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        target_dim=args.target_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        horizons=horizons,
    )
    model = FusedContextJointPathPCAPredictor(cfg).to(device)
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
            loss, _parts = joint_path_pca_loss(predicted, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred, val_context = predict_joint_path_pca(
            model,
            val.past_window,
            batch_size=args.batch_size,
            device=device,
        )
        val_metrics = evaluate_joint_path_pca(
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
            "val_path_mse": val_metrics["path_prediction"]["mse"],
            "val_path_mrr": val_metrics["path_retrieval"]["mrr"],
            "val_path_top5": val_metrics["path_retrieval"]["top5"],
            "val_delta_decode_mse": val_metrics["delta_decode"]["overall_prediction"]["mse"],
            "val_delta_decode_mrr": val_metrics["delta_decode"]["overall_retrieval"]["mrr_mean"],
            "val_delta_decode_top5": val_metrics["delta_decode"]["overall_retrieval"]["top5_mean"],
            "val_predicted_effective_rank": val_metrics["predicted_health"]["effective_rank"],
            "val_context_effective_rank": val_metrics["context_health"]["effective_rank"],
        }
        history.append(row)
        if best_summary is None or row["val_delta_decode_mse"] < best_summary["val_delta_decode_mse"]:
            best_summary = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row))

    final_pred, final_context = predict_joint_path_pca(
        model,
        val.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    final_metrics = evaluate_joint_path_pca(
        predicted_z=final_pred,
        target_z=val_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=final_context,
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    best_pred, best_context = predict_joint_path_pca(
        model,
        val.past_window,
        batch_size=args.batch_size,
        device=device,
    )
    best_metrics = evaluate_joint_path_pca(
        predicted_z=best_pred,
        target_z=val_z,
        truth_delta=val_delta,
        pca_target=pca_target,
        horizons=horizons,
        context=best_context,
    )
    train_context = predict_joint_path_pca(model, train.past_window, batch_size=args.batch_size, device=device)[1]
    ridge_path_pred = ridge_probe_predict(train_context, train_z, best_context, alpha=args.ridge_alpha)
    ridge_path_metrics = {
        "prediction": latent_prediction_metrics(ridge_path_pred, val_z),
        "retrieval": retrieval_metrics(ridge_path_pred, val_z, top_k=(1, 5, 10)),
    }
    ridge_delta_metrics = ridge_probe_metrics(
        train_context,
        best_context,
        train_delta,
        val_delta,
        horizons=horizons,
        alpha=args.ridge_alpha,
    )

    result = {
        "config": asdict(cfg),
        "args": vars(args),
        "literature_status": "supported_adjacent",
        "target_contract": {
            "kind": "joint_whitened_horizon_delta_path_pca",
            "target_dim": args.target_dim,
            "horizons": list(horizons),
            "mean": pca_target.mean,
            "components": pca_target.components,
            "scale": pca_target.scale,
        },
        "device": str(device),
        "train_shape": {
            "past": list(train.past_window.shape),
            "future": list(train.future_window.shape),
            "delta": list(train_delta.shape),
            "target_z": list(train_z.shape),
        },
        "val_shape": {
            "past": list(val.past_window.shape),
            "future": list(val.future_window.shape),
            "delta": list(val_delta.shape),
            "target_z": list(val_z.shape),
        },
        "history": history,
        "best_epoch_summary": best_summary,
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "ridge_probe_path_metrics": ridge_path_metrics,
        "ridge_probe_delta_metrics": ridge_delta_metrics,
        "reference_thresholds": {
            "head038_primary_decoded_delta_mse": 0.015176,
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
    parser = argparse.ArgumentParser(description="Fused-context predictor to joint horizon-delta path PCA target")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--context_dim", type=int, default=32)
    parser.add_argument("--target_dim", type=int, default=16)
    parser.add_argument("--predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--ridge_alpha", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7714)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/part1_joint_path_pca_head060.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/joint_path_pca_head060.pt",
    )
    args = parser.parse_args()
    train_smoke(args)


if __name__ == "__main__":
    main()
