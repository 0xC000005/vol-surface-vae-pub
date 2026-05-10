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
    GeometryTokenMetadata,
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)
from experiments.world.evaluation.masked_multiview_metrics import (  # noqa: E402
    flattened_time_rows,
    mask_visibility_summary,
    same_state_multiview_metrics,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_smoke import (  # noqa: E402
    raw_masked_view_baseline,
)
from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (  # noqa: E402
    torch_barlow_cross_correlation_loss,
)


GEOMETRY_IDS = ("iv_surface", "vol_side_channel", "factor_level", "factor_return")


@dataclass(frozen=True)
class GeometryAwareDirectBarlowConfig:
    n_tokens: int = 58
    token_descriptor_dim: int = 13
    token_hidden_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64


def _one_hot(labels: np.ndarray, vocabulary: tuple[str, ...]) -> np.ndarray:
    out = np.zeros((labels.shape[0], len(vocabulary)), dtype=np.float32)
    index = {name: i for i, name in enumerate(vocabulary)}
    for row, label in enumerate(labels.tolist()):
        if str(label) in index:
            out[row, index[str(label)]] = 1.0
    return out


def build_token_descriptor_matrix(metadata: GeometryTokenMetadata) -> np.ndarray:
    coord = np.asarray(metadata.geometry_coord, dtype=np.float32)
    coord = np.where(coord >= 0.0, coord / 4.0, -1.0).astype(np.float32)
    geometry = _one_hot(metadata.geometry_id, GEOMETRY_IDS)
    families = tuple(sorted({str(x) for x in metadata.factor_family.tolist()}))
    family = _one_hot(metadata.factor_family, families)
    return np.concatenate([coord, geometry, family], axis=1).astype(np.float32)


class GeometryAwareDailyEncoder(nn.Module):
    def __init__(
        self,
        cfg: GeometryAwareDirectBarlowConfig,
        *,
        token_descriptors: np.ndarray,
    ):
        super().__init__()
        descriptors = torch.as_tensor(token_descriptors, dtype=torch.float32)
        if descriptors.shape != (cfg.n_tokens, cfg.token_descriptor_dim):
            raise ValueError(
                "token_descriptors shape must match config, got "
                f"{tuple(descriptors.shape)} vs {(cfg.n_tokens, cfg.token_descriptor_dim)}"
            )
        self.cfg = cfg
        self.register_buffer("token_descriptors", descriptors)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(3 + cfg.token_descriptor_dim),
            nn.Linear(3 + cfg.token_descriptor_dim, cfg.token_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.token_hidden_dim, cfg.token_hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(cfg.token_hidden_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(
        self,
        view_values: torch.Tensor,
        observed_mask: torch.Tensor,
        synthetic_mask: torch.Tensor,
    ) -> torch.Tensor:
        if view_values.shape != observed_mask.shape or view_values.shape != synthetic_mask.shape:
            raise ValueError("view_values, observed_mask, and synthetic_mask must share shape")
        if view_values.ndim != 3 or view_values.shape[-1] != self.cfg.n_tokens:
            raise ValueError(f"Expected shape (B, T, {self.cfg.n_tokens}), got {tuple(view_values.shape)}")
        batch, time, _tokens = view_values.shape
        descriptors = self.token_descriptors.view(1, 1, self.cfg.n_tokens, -1).expand(
            batch,
            time,
            -1,
            -1,
        )
        token_inputs = torch.cat(
            [
                view_values.float().unsqueeze(-1),
                observed_mask.float().unsqueeze(-1),
                synthetic_mask.float().unsqueeze(-1),
                descriptors,
            ],
            dim=-1,
        )
        token_emb = self.token_mlp(token_inputs)
        day_emb = token_emb.mean(dim=2)
        seq, _h_n = self.gru(day_emb)
        return self.head(seq)


class GeometryAwareDirectBarlowModel(nn.Module):
    def __init__(
        self,
        cfg: GeometryAwareDirectBarlowConfig,
        *,
        token_descriptors: np.ndarray,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = GeometryAwareDailyEncoder(cfg, token_descriptors=token_descriptors)

    def forward(
        self,
        view_a_values: torch.Tensor,
        view_b_values: torch.Tensor,
        observed_mask: torch.Tensor,
        synthetic_mask_a: torch.Tensor,
        synthetic_mask_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "view_a": self.encoder(view_a_values, observed_mask, synthetic_mask_a),
            "view_b": self.encoder(view_b_values, observed_mask, synthetic_mask_b),
        }


def geometry_masked_multiview_barlow_loss(
    outputs: dict[str, torch.Tensor],
    *,
    offdiag_weight: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    barlow, parts = torch_barlow_cross_correlation_loss(
        outputs["view_a"],
        outputs["view_b"],
        offdiag_weight=offdiag_weight,
        canonical_mean_scale=True,
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


def _parameter_groups(model: GeometryAwareDirectBarlowModel) -> Iterator[nn.Parameter]:
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def encode_geometry_barlow_split(
    model: GeometryAwareDirectBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = {"view_a": [], "view_b": []}
    loader = _loader_from_batch(batch, batch_size=batch_size, shuffle=False)
    for view_a, view_b, observed, synth_a, synth_b in loader:
        out = model(
            view_a.to(device),
            view_b.to(device),
            observed.to(device),
            synth_a.to(device),
            synth_b.to(device),
        )
        for key in rows:
            rows[key].append(out[key].detach().cpu().numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in rows.items()}


def evaluate_geometry_barlow_part1(
    model: GeometryAwareDirectBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_geometry_barlow_split(
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
    descriptors = build_token_descriptor_matrix(train.token_metadata)
    cfg = GeometryAwareDirectBarlowConfig(
        n_tokens=train.token_metadata.n_tokens,
        token_descriptor_dim=descriptors.shape[1],
        token_hidden_dim=args.token_hidden_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    model = GeometryAwareDirectBarlowModel(cfg, token_descriptors=descriptors).to(device)
    opt = torch.optim.AdamW(_parameter_groups(model), lr=args.lr, weight_decay=args.weight_decay)
    loader = _loader_from_batch(train, batch_size=args.batch_size, shuffle=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts_accum: dict[str, list[float]] = {}
        for view_a, view_b, observed, synth_a, synth_b in loader:
            opt.zero_grad(set_to_none=True)
            out = model(
                view_a.to(device),
                view_b.to(device),
                observed.to(device),
                synth_a.to(device),
                synth_b.to(device),
            )
            loss, parts = geometry_masked_multiview_barlow_loss(
                out,
                offdiag_weight=args.barlow_offdiag_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(_parameter_groups(model)), args.grad_clip)
            opt.step()
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

    train_metrics = evaluate_geometry_barlow_part1(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
    )
    val_metrics = evaluate_geometry_barlow_part1(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    result = {
        "literature_status": "supported_adjacent_direct_barlow_with_geometry_aware_encoder",
        "loss_scaling": "canonical_mean_scaled_barlow",
        "config": asdict(cfg),
        "args": vars(args),
        "device": str(device),
        "token_descriptor_shape": list(descriptors.shape),
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
            "token_descriptors": descriptors,
            "state_dict": model.state_dict(),
            "result": _serializable(result),
        },
        checkpoint,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-aware direct Barlow masked-multiview smoke")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--token_hidden_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=384)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--barlow_offdiag_weight", type=float, default=0.005)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=760)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/masked_multiview_geometry_barlow_head076.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/masked_multiview_geometry_barlow_head076.pt",
    )
    train_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
