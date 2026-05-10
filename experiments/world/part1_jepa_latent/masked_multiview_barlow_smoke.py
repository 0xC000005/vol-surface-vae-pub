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
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)
from experiments.world.evaluation.hard_mask_presets import (  # noqa: E402
    build_hard_masked_batch,
)
from experiments.world.evaluation.masked_multiview_metrics import (  # noqa: E402
    flattened_time_rows,
    mask_visibility_summary,
    same_state_multiview_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (  # noqa: E402
    make_masked_view_features,
    torch_barlow_cross_correlation_loss,
)


@dataclass(frozen=True)
class DirectMaskedMultiviewBarlowConfig:
    token_dim: int = 58
    input_dim: int = 174
    hidden_dim: int = 128
    latent_dim: int = 64


class DirectMaskedMultiviewEncoder(nn.Module):
    def __init__(self, cfg: DirectMaskedMultiviewBarlowConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.input_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        seq, _h_n = self.gru(features)
        return self.head(seq)


class DirectMaskedMultiviewBarlowModel(nn.Module):
    def __init__(self, cfg: DirectMaskedMultiviewBarlowConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = DirectMaskedMultiviewEncoder(cfg)

    def forward(
        self,
        view_a_features: torch.Tensor,
        view_b_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "view_a": self.encoder(view_a_features),
            "view_b": self.encoder(view_b_features),
        }


def direct_masked_multiview_barlow_loss(
    outputs: dict[str, torch.Tensor],
    *,
    offdiag_weight: float = 0.005,
    canonical_mean_scale: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    barlow, parts = torch_barlow_cross_correlation_loss(
        outputs["view_a"],
        outputs["view_b"],
        offdiag_weight=offdiag_weight,
        canonical_mean_scale=canonical_mean_scale,
    )
    return barlow, {
        "barlow": float(barlow.detach().cpu()),
        **parts,
        "loss": float(barlow.detach().cpu()),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parameter_groups(
    model: DirectMaskedMultiviewBarlowModel,
) -> Iterator[nn.Parameter]:
    yield from model.encoder.parameters()


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
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle
    )


@torch.no_grad()
def encode_direct_barlow_split(
    model: DirectMaskedMultiviewBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = {"view_a": [], "view_b": []}
    loader = _loader_from_batch(batch, batch_size=batch_size, shuffle=False)
    for view_a, view_b, observed, synth_a, synth_b in loader:
        features_a = make_masked_view_features(
            view_a.to(device),
            observed.to(device),
            synth_a.to(device),
        )
        features_b = make_masked_view_features(
            view_b.to(device),
            observed.to(device),
            synth_b.to(device),
        )
        out = model(features_a, features_b)
        for key in rows:
            rows[key].append(out[key].detach().cpu().numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in rows.items()}


def evaluate_direct_barlow_part1(
    model: DirectMaskedMultiviewBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_direct_barlow_split(
        model,
        batch,
        batch_size=batch_size,
        device=device,
    )
    return {
        "view_alignment": same_state_multiview_metrics(
            flattened_time_rows(encoded["view_a"]),
            flattened_time_rows(encoded["view_b"]),
        ),
        "visibility": mask_visibility_summary(batch),
    }


def raw_masked_view_baseline(batch: MaskedMultiviewBatch) -> dict[str, object]:
    return same_state_multiview_metrics(
        flattened_time_rows(batch.view_a_values),
        flattened_time_rows(batch.view_b_values),
    )


def build_batch_for_mask_preset(
    *,
    split: str,
    history_len: int,
    future_len: int,
    max_windows: int,
    seed: int,
    mask_preset: str = "default",
) -> MaskedMultiviewBatch:
    if mask_preset == "default":
        return build_masked_multiview_batch(
            split=split,
            history_len=history_len,
            future_len=future_len,
            max_windows=max_windows,
            seed=seed,
            normalize=True,
        )
    if mask_preset == "hard_head122":
        return build_hard_masked_batch(
            split=split,
            history_len=history_len,
            future_len=future_len,
            max_windows=max_windows,
            seed=seed,
        )
    raise ValueError(f"unknown mask_preset: {mask_preset!r}")


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
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train = build_batch_for_mask_preset(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
        mask_preset=args.mask_preset,
    )
    val = build_batch_for_mask_preset(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
        mask_preset=args.mask_preset,
    )
    cfg = DirectMaskedMultiviewBarlowConfig(
        token_dim=train.token_metadata.n_tokens,
        input_dim=train.token_metadata.n_tokens * 3,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    model = DirectMaskedMultiviewBarlowModel(cfg).to(device)
    opt = torch.optim.AdamW(
        _parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = _loader_from_batch(train, batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts_accum: dict[str, list[float]] = {}
        for view_a, view_b, observed, synth_a, synth_b in loader:
            features_a = make_masked_view_features(
                view_a.to(device),
                observed.to(device),
                synth_a.to(device),
            )
            features_b = make_masked_view_features(
                view_b.to(device),
                observed.to(device),
                synth_b.to(device),
            )
            opt.zero_grad(set_to_none=True)
            out = model(features_a, features_b)
            loss, parts = direct_masked_multiview_barlow_loss(
                out,
                offdiag_weight=args.barlow_offdiag_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(_parameter_groups(model)), args.grad_clip
            )
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                parts_accum.setdefault(key, []).append(value)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{
                f"{k}_mean": float(np.mean(values)) for k, values in parts_accum.items()
            },
        }
        history.append(row)
        print(json.dumps(row))

    train_metrics = evaluate_direct_barlow_part1(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
    )
    val_metrics = evaluate_direct_barlow_part1(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    result = {
        "literature_status": "supported_adjacent_direct_barlow_twins_for_same_state_masked_multiview",
        "loss_scaling": "canonical_mean_scaled_barlow",
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
    output_json.write_text(
        json.dumps(_serializable(result), indent=2), encoding="utf-8"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct Barlow masked-multiview Part 1 smoke"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=384)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--barlow_offdiag_weight", type=float, default=0.005)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=680)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mask_preset",
        choices=("default", "hard_head122"),
        default="default",
        help="Named structured masking preset. hard_head122 is a diagnostic branch, not the active reference.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/masked_multiview_barlow_head070.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt",
    )
    train_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
