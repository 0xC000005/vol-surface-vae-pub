from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from experiments.world.evaluation.context_target_jepa_data import (
    ContextTargetJepaBatch,
    build_context_target_jepa_batch,
)
from experiments.world.evaluation.part1_metrics import representation_health_metrics
from experiments.world.part1_jepa_latent.jepa_smoke import update_ema
from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (
    make_masked_view_features,
    torch_barlow_cross_correlation_loss,
)


@dataclass(frozen=True)
class ContextTargetJEPAConfig:
    token_dim: int = 58
    input_dim: int = 174
    hidden_dim: int = 128
    latent_dim: int = 64
    predictor_hidden_dim: int = 128
    ema_decay: float = 0.99


class ContextTargetSequenceEncoder(nn.Module):
    def __init__(self, cfg: ContextTargetJEPAConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.input_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        seq, _h_n = self.gru(features)
        return self.head(seq)


class ContextTargetJEPAModel(nn.Module):
    def __init__(self, cfg: ContextTargetJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = ContextTargetSequenceEncoder(cfg)
        self.target_encoder = ContextTargetSequenceEncoder(cfg)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.predictor_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.latent_dim),
        )

    def forward(
        self,
        context_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(context_features)
        predicted = self.predictor(context)
        with torch.no_grad():
            target = self.target_encoder(target_features)
        return {"context": context, "predicted": predicted, "target": target}


def make_context_target_features(
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    return make_masked_view_features(values, observed_mask, visible_mask)


def _target_time_mask(target_mask: torch.Tensor) -> torch.Tensor:
    if target_mask.ndim != 3:
        raise ValueError(
            f"target_mask must have shape (B, T, N), got {tuple(target_mask.shape)}"
        )
    return target_mask.bool().any(dim=-1)


def _select_target_time_rows(
    values: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError(f"values must have shape (B, T, D), got {tuple(values.shape)}")
    time_mask = _target_time_mask(target_mask)
    if values.shape[:2] != time_mask.shape:
        raise ValueError(
            "values and target_mask must share batch/time shape, "
            f"got {tuple(values.shape[:2])} and {tuple(time_mask.shape)}"
        )
    selected = values[time_mask]
    if selected.shape[0] < 2:
        raise ValueError(
            "Need at least two target time rows for context-to-target loss"
        )
    return selected


def context_target_jepa_loss(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    *,
    barlow_weight: float = 0.05,
    offdiag_weight: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = _select_target_time_rows(outputs["predicted"], target_mask)
    target = _select_target_time_rows(outputs["target"].detach(), target_mask)
    alignment = F.mse_loss(predicted, target)
    barlow, parts = torch_barlow_cross_correlation_loss(
        predicted[:, None, :],
        target[:, None, :],
        offdiag_weight=offdiag_weight,
    )
    loss = alignment + barlow_weight * barlow
    return loss, {
        "alignment": float(alignment.detach().cpu()),
        "barlow": float(barlow.detach().cpu()),
        **parts,
        "loss": float(loss.detach().cpu()),
        "target_time_rows": int(predicted.shape[0]),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parameter_groups(model: ContextTargetJEPAModel) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.predictor.parameters()


def _loader_from_batch(
    batch: ContextTargetJepaBatch,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(batch.context_values),
        torch.from_numpy(batch.target_values),
        torch.from_numpy(batch.observed_mask),
        torch.from_numpy(batch.context_mask),
        torch.from_numpy(batch.target_mask),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
    )


def _features_from_tensors(
    values: torch.Tensor,
    observed: torch.Tensor,
    visible: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    return make_context_target_features(
        values.to(device),
        observed.to(device),
        visible.to(device),
    )


def _run_epoch(
    model: ContextTargetJEPAModel,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    barlow_weight: float,
    offdiag_weight: float,
    ema_decay: float,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: dict[str, list[float]] = {}
    for context_values, target_values, observed, context_mask, target_mask in loader:
        context_features = _features_from_tensors(
            context_values,
            observed,
            context_mask,
            device=device,
        )
        target_features = _features_from_tensors(
            target_values,
            observed,
            target_mask,
            device=device,
        )
        target_mask_device = target_mask.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out = model(context_features, target_features)
            loss, parts = context_target_jepa_loss(
                out,
                target_mask_device,
                barlow_weight=barlow_weight,
                offdiag_weight=offdiag_weight,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(_parameter_groups(model)),
                    grad_clip,
                )
                optimizer.step()
                update_ema(model.context_encoder, model.target_encoder, ema_decay)
        for key, value in parts.items():
            rows.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in rows.items()}


@torch.no_grad()
def encode_clean_context(
    model: ContextTargetJEPAModel,
    batch: ContextTargetJepaBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    clean_visible = batch.observed_mask
    dataset = TensorDataset(
        torch.from_numpy(batch.clean_values),
        torch.from_numpy(batch.observed_mask),
        torch.from_numpy(clean_visible),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for values, observed, visible in loader:
        features = _features_from_tensors(
            values,
            observed,
            visible,
            device=device,
        )
        rows.append(model.context_encoder(features).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def _evaluate(
    model: ContextTargetJEPAModel,
    batch: ContextTargetJepaBatch,
    *,
    batch_size: int,
    device: torch.device,
    barlow_weight: float,
    offdiag_weight: float,
) -> dict[str, object]:
    loader = _loader_from_batch(batch, batch_size=batch_size, shuffle=False)
    loss_parts = _run_epoch(
        model,
        loader,
        device=device,
        optimizer=None,
        barlow_weight=barlow_weight,
        offdiag_weight=offdiag_weight,
        ema_decay=model.cfg.ema_decay,
        grad_clip=1.0,
    )
    encoded = encode_clean_context(
        model,
        batch,
        batch_size=batch_size,
        device=device,
    )
    return {
        **loss_parts,
        "clean_last_health": representation_health_metrics(encoded[:, -1, :]),
        "clean_flat_health": representation_health_metrics(
            encoded.reshape(encoded.shape[0] * encoded.shape[1], encoded.shape[2])
        ),
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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_context_target_smoke(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train = build_context_target_jepa_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    val = build_context_target_jepa_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
    )
    cfg = ContextTargetJEPAConfig(
        token_dim=train.token_metadata.n_tokens,
        input_dim=train.token_metadata.n_tokens * 3,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        ema_decay=args.ema_decay,
    )
    model = ContextTargetJEPAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loader = _loader_from_batch(
        train,
        batch_size=args.batch_size,
        shuffle=True,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        row = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            barlow_weight=args.barlow_weight,
            offdiag_weight=args.barlow_offdiag_weight,
            ema_decay=args.ema_decay,
            grad_clip=args.grad_clip,
        )
        row = {"epoch": epoch, **row}
        history.append(row)
        print(json.dumps(row))

    train_metrics = _evaluate(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
        barlow_weight=args.barlow_weight,
        offdiag_weight=args.barlow_offdiag_weight,
    )
    val_metrics = _evaluate(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
        barlow_weight=args.barlow_weight,
        offdiag_weight=args.barlow_offdiag_weight,
    )
    result = {
        "objective_family": "context_to_target_jepa",
        "literature_status": "canonical_jepa_same_window_masked_state_prediction",
        "uses_future_targets": False,
        "uses_value_reconstruction": False,
        "config": asdict(cfg),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_serializable(result), indent=2),
        encoding="utf-8",
    )

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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Same-window context-to-target JEPA smoke"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--max_train_windows", type=int, default=384)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--barlow_weight", type=float, default=0.05)
    parser.add_argument("--barlow_offdiag_weight", type=float, default=0.005)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2140)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/world/context_target_jepa_smoke_head140.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/world/checkpoints/part1_jepa_latent/context_target_jepa_smoke_head140.pt"
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_context_target_smoke(args)


if __name__ == "__main__":
    main()
