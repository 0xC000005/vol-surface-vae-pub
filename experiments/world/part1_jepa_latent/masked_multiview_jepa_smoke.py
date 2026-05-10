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

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)
from experiments.world.evaluation.masked_multiview_metrics import (  # noqa: E402
    flattened_time_rows,
    mask_visibility_summary,
    same_state_multiview_metrics,
)
from experiments.world.part1_jepa_latent.jepa_smoke import update_ema  # noqa: E402


@dataclass(frozen=True)
class MaskedMultiviewJEPAConfig:
    token_dim: int = 58
    input_dim: int = 174
    hidden_dim: int = 64
    latent_dim: int = 16
    predictor_hidden_dim: int = 64
    ema_decay: float = 0.99


class MaskedMultiviewSequenceEncoder(nn.Module):
    def __init__(self, cfg: MaskedMultiviewJEPAConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.input_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        seq, _h_n = self.gru(features)
        return self.head(seq)


class MaskedMultiviewJEPAWorldModel(nn.Module):
    def __init__(self, cfg: MaskedMultiviewJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = MaskedMultiviewSequenceEncoder(cfg)
        self.target_encoder = MaskedMultiviewSequenceEncoder(cfg)
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


def make_masked_view_features(
    view_values: torch.Tensor,
    observed_mask: torch.Tensor,
    synthetic_mask: torch.Tensor,
) -> torch.Tensor:
    if view_values.shape != observed_mask.shape or view_values.shape != synthetic_mask.shape:
        raise ValueError(
            "view_values, observed_mask, and synthetic_mask must share shape, "
            f"got {tuple(view_values.shape)}, {tuple(observed_mask.shape)}, "
            f"{tuple(synthetic_mask.shape)}"
        )
    return torch.cat(
        [
            view_values.float(),
            observed_mask.float(),
            synthetic_mask.float(),
        ],
        dim=-1,
    )


def _flatten_time(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 3:
        raise ValueError(f"Expected shape (B, T, D), got {tuple(z.shape)}")
    return z.reshape(z.shape[0] * z.shape[1], z.shape[2])


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("matrix must be square")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def torch_barlow_cross_correlation_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    *,
    offdiag_weight: float = 0.005,
    canonical_mean_scale: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    a = _flatten_time(view_a)
    b = _flatten_time(view_b)
    if a.shape != b.shape:
        raise ValueError(f"view_a and view_b must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[0] < 2:
        raise ValueError("Need at least two time rows for Barlow loss")

    a = (a - a.mean(dim=0, keepdim=True)) / torch.sqrt(a.var(dim=0, unbiased=False, keepdim=True) + eps)
    b = (b - b.mean(dim=0, keepdim=True)) / torch.sqrt(b.var(dim=0, unbiased=False, keepdim=True) + eps)
    corr = a.T @ b / a.shape[0]
    diag = torch.diagonal(corr)
    offdiag = _off_diagonal(corr)
    diag_loss = torch.mean((diag - 1.0) ** 2)
    offdiag_loss = torch.mean(offdiag * offdiag) if offdiag.numel() else corr.new_tensor(0.0)
    effective_offdiag_weight = offdiag_weight * (a.shape[1] - 1 if canonical_mean_scale else 1.0)
    loss = diag_loss + effective_offdiag_weight * offdiag_loss
    return loss, {
        "barlow_diag_loss": float(diag_loss.detach().cpu()),
        "barlow_offdiag_loss": float(offdiag_loss.detach().cpu()),
    }


def masked_multiview_jepa_loss(
    outputs: dict[str, torch.Tensor],
    *,
    barlow_weight: float = 0.05,
    offdiag_weight: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = outputs["predicted"]
    target = outputs["target"].detach()
    alignment = F.mse_loss(predicted, target)
    barlow, parts = torch_barlow_cross_correlation_loss(
        predicted,
        target,
        offdiag_weight=offdiag_weight,
    )
    loss = alignment + barlow_weight * barlow
    return loss, {
        "alignment": float(alignment.detach().cpu()),
        "barlow": float(barlow.detach().cpu()),
        **parts,
        "loss": float(loss.detach().cpu()),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parameter_groups(model: MaskedMultiviewJEPAWorldModel) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.predictor.parameters()


def _loader_from_batch(
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(batch.view_a_values),
        torch.from_numpy(batch.view_b_values),
        torch.from_numpy(batch.observed_mask),
        torch.from_numpy(batch.synthetic_mask_a),
        torch.from_numpy(batch.synthetic_mask_b),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def encode_masked_multiview_split(
    model: MaskedMultiviewJEPAWorldModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = {"context": [], "predicted": [], "target": []}
    loader = _loader_from_batch(batch, batch_size=batch_size, shuffle=False)
    for view_a, view_b, observed, synth_a, synth_b in loader:
        context_features = make_masked_view_features(
            view_a.to(device),
            observed.to(device),
            synth_a.to(device),
        )
        target_features = make_masked_view_features(
            view_b.to(device),
            observed.to(device),
            synth_b.to(device),
        )
        out = model(context_features, target_features)
        for key in rows:
            rows[key].append(out[key].detach().cpu().numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in rows.items()}


def evaluate_masked_multiview_part1(
    model: MaskedMultiviewJEPAWorldModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_masked_multiview_split(
        model,
        batch,
        batch_size=batch_size,
        device=device,
    )
    return {
        "predicted_target": same_state_multiview_metrics(
            flattened_time_rows(encoded["predicted"]),
            flattened_time_rows(encoded["target"]),
        ),
        "context_target": same_state_multiview_metrics(
            flattened_time_rows(encoded["context"]),
            flattened_time_rows(encoded["target"]),
        ),
        "visibility": mask_visibility_summary(batch),
    }


def raw_masked_view_baseline(batch: MaskedMultiviewBatch) -> dict[str, object]:
    return same_state_multiview_metrics(
        flattened_time_rows(batch.view_a_values),
        flattened_time_rows(batch.view_b_values),
    )


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


def train_smoke(args: argparse.Namespace) -> dict[str, object]:
    _set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train = build_masked_multiview_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
        normalize=True,
    )
    val = build_masked_multiview_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
        normalize=True,
    )
    cfg = MaskedMultiviewJEPAConfig(
        token_dim=train.token_metadata.n_tokens,
        input_dim=train.token_metadata.n_tokens * 3,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        ema_decay=args.ema_decay,
    )
    model = MaskedMultiviewJEPAWorldModel(cfg).to(device)
    opt = torch.optim.AdamW(_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    loader = _loader_from_batch(train, batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts_accum: dict[str, list[float]] = {}
        for view_a, view_b, observed, synth_a, synth_b in loader:
            context_features = make_masked_view_features(
                view_a.to(device),
                observed.to(device),
                synth_a.to(device),
            )
            target_features = make_masked_view_features(
                view_b.to(device),
                observed.to(device),
                synth_b.to(device),
            )
            opt.zero_grad(set_to_none=True)
            out = model(context_features, target_features)
            loss, parts = masked_multiview_jepa_loss(
                out,
                barlow_weight=args.barlow_weight,
                offdiag_weight=args.barlow_offdiag_weight,
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
            **{f"{k}_mean": float(np.mean(values)) for k, values in parts_accum.items()},
        }
        history.append(row)
        print(json.dumps(row))

    train_metrics = evaluate_masked_multiview_part1(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
    )
    val_metrics = evaluate_masked_multiview_part1(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    result = {
        "literature_status": "canonical_jepa_ema_stopgrad_with_supported_adjacent_barlow_redundancy_control",
        "config": asdict(cfg),
        "args": vars(args),
        "device": str(device),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "raw_train_baseline": raw_masked_view_baseline(train),
        "raw_val_baseline": raw_masked_view_baseline(val),
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
    parser = argparse.ArgumentParser(description="Masked-multiview JEPA Part 1 smoke")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--predictor_hidden_dim", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--barlow_weight", type=float, default=0.05)
    parser.add_argument("--barlow_offdiag_weight", type=float, default=0.005)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=660)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/masked_multiview_jepa_head066.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/masked_multiview_jepa_head066.pt",
    )
    train_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
