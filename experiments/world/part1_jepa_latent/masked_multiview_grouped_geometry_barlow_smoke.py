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
from torch.utils.data import DataLoader, TensorDataset

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
from experiments.world.part1_jepa_latent.masked_multiview_geometry_barlow_smoke import (  # noqa: E402
    build_token_descriptor_matrix,
)
from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (  # noqa: E402
    torch_barlow_cross_correlation_loss,
)


GEOMETRY_GROUPS = ("iv_surface", "vol_side_channel", "factor_level", "factor_return")


@dataclass(frozen=True)
class GroupedGeometryDirectBarlowConfig:
    n_tokens: int = 58
    token_descriptor_dim: int = 13
    token_hidden_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64


def build_geometry_group_indices(
    metadata: GeometryTokenMetadata,
    *,
    groups: tuple[str, ...] = GEOMETRY_GROUPS,
) -> dict[str, np.ndarray]:
    out = {}
    assigned = np.zeros(metadata.n_tokens, dtype=bool)
    for group in groups:
        idx = np.where(metadata.geometry_id == group)[0].astype(np.int64)
        if idx.size == 0:
            raise ValueError(f"geometry group has no tokens: {group}")
        out[group] = idx
        assigned[idx] = True
    if not bool(np.all(assigned)):
        missing = np.where(~assigned)[0].tolist()
        raise ValueError(f"tokens not assigned to a geometry group: {missing}")
    return out


class GroupedGeometryDailyEncoder(nn.Module):
    def __init__(
        self,
        cfg: GroupedGeometryDirectBarlowConfig,
        *,
        token_descriptors: np.ndarray,
        group_indices: dict[str, np.ndarray],
    ):
        super().__init__()
        descriptors = torch.as_tensor(token_descriptors, dtype=torch.float32)
        if descriptors.shape != (cfg.n_tokens, cfg.token_descriptor_dim):
            raise ValueError(
                "token_descriptors shape must match config, got "
                f"{tuple(descriptors.shape)} vs {(cfg.n_tokens, cfg.token_descriptor_dim)}"
            )
        self.cfg = cfg
        self.group_names = tuple(group_indices)
        self.daily_input_dim = len(self.group_names) * cfg.token_hidden_dim
        self.register_buffer("token_descriptors", descriptors)
        for name, idx in group_indices.items():
            self.register_buffer(
                f"group_{name}", torch.as_tensor(idx, dtype=torch.long)
            )
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(3 + cfg.token_descriptor_dim),
            nn.Linear(3 + cfg.token_descriptor_dim, cfg.token_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.token_hidden_dim, cfg.token_hidden_dim),
            nn.SiLU(),
        )
        self.daily_fusion = nn.Sequential(
            nn.LayerNorm(self.daily_input_dim),
            nn.Linear(self.daily_input_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def _group_index(self, name: str) -> torch.Tensor:
        return getattr(self, f"group_{name}")

    def forward(
        self,
        view_values: torch.Tensor,
        observed_mask: torch.Tensor,
        synthetic_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            view_values.shape != observed_mask.shape
            or view_values.shape != synthetic_mask.shape
        ):
            raise ValueError(
                "view_values, observed_mask, and synthetic_mask must share shape"
            )
        if view_values.ndim != 3 or view_values.shape[-1] != self.cfg.n_tokens:
            raise ValueError(
                f"Expected shape (B, T, {self.cfg.n_tokens}), got {tuple(view_values.shape)}"
            )
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
        group_rows = []
        for name in self.group_names:
            idx = self._group_index(name)
            group_rows.append(token_emb.index_select(dim=2, index=idx).mean(dim=2))
        day_emb = self.daily_fusion(torch.cat(group_rows, dim=-1))
        seq, _h_n = self.gru(day_emb)
        return self.head(seq)


class GroupedGeometryDirectBarlowModel(nn.Module):
    def __init__(
        self,
        cfg: GroupedGeometryDirectBarlowConfig,
        *,
        token_descriptors: np.ndarray,
        group_indices: dict[str, np.ndarray],
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = GroupedGeometryDailyEncoder(
            cfg,
            token_descriptors=token_descriptors,
            group_indices=group_indices,
        )

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


def grouped_geometry_barlow_loss(
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


def _parameter_groups(
    model: GroupedGeometryDirectBarlowModel,
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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
    )


@torch.no_grad()
def encode_grouped_geometry_barlow_split(
    model: GroupedGeometryDirectBarlowModel,
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


@torch.no_grad()
def encode_clean_grouped_geometry_windows(
    model: GroupedGeometryDirectBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(batch.clean_values),
        torch.from_numpy(batch.observed_mask),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    for clean_values, observed in loader:
        synthetic = torch.ones_like(observed, dtype=torch.bool)
        rows.append(
            model.encoder(
                clean_values.to(device),
                observed.to(device),
                synthetic.to(device),
            )
            .detach()
            .cpu()
            .numpy()
        )
    return np.concatenate(rows, axis=0)


def evaluate_grouped_geometry_barlow_part1(
    model: GroupedGeometryDirectBarlowModel,
    batch: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_grouped_geometry_barlow_split(
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


def load_grouped_geometry_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> GroupedGeometryDirectBarlowModel:
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = GroupedGeometryDirectBarlowConfig(**checkpoint["config"])
    model = GroupedGeometryDirectBarlowModel(
        cfg,
        token_descriptors=np.asarray(checkpoint["token_descriptors"], dtype=np.float32),
        group_indices={
            key: np.asarray(value, dtype=np.int64)
            for key, value in checkpoint["group_indices"].items()
        },
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


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
    group_indices = build_geometry_group_indices(train.token_metadata)
    cfg = GroupedGeometryDirectBarlowConfig(
        n_tokens=train.token_metadata.n_tokens,
        token_descriptor_dim=descriptors.shape[1],
        token_hidden_dim=args.token_hidden_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    model = GroupedGeometryDirectBarlowModel(
        cfg,
        token_descriptors=descriptors,
        group_indices=group_indices,
    ).to(device)
    opt = torch.optim.AdamW(
        _parameter_groups(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
            loss, parts = grouped_geometry_barlow_loss(
                out,
                offdiag_weight=args.barlow_offdiag_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(_parameter_groups(model)),
                args.grad_clip,
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

    train_metrics = evaluate_grouped_geometry_barlow_part1(
        model,
        train,
        batch_size=args.batch_size,
        device=device,
    )
    val_metrics = evaluate_grouped_geometry_barlow_part1(
        model,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    result = {
        "literature_status": "supported_adjacent_direct_barlow_with_grouped_geometry_encoder",
        "loss_scaling": "canonical_mean_scaled_barlow",
        "config": asdict(cfg),
        "args": vars(args),
        "device": str(device),
        "token_descriptor_shape": list(descriptors.shape),
        "geometry_groups": {
            key: value.tolist() for key, value in group_indices.items()
        },
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
        json.dumps(_serializable(result), indent=2),
        encoding="utf-8",
    )

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "token_descriptors": descriptors,
            "group_indices": {
                key: value.tolist() for key, value in group_indices.items()
            },
            "state_dict": model.state_dict(),
            "result": _serializable(result),
        },
        checkpoint,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped-geometry direct Barlow masked-multiview smoke"
    )
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
    parser.add_argument("--seed", type=int, default=780)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/world/masked_multiview_grouped_geometry_barlow_head126.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/world/checkpoints/part1_jepa_latent/masked_multiview_grouped_geometry_barlow_head126.pt",
    )
    train_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
