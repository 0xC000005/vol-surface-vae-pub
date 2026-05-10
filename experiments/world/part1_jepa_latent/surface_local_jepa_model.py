from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.world.part1_jepa_latent.masked_multiview_jepa_smoke import (
    torch_barlow_cross_correlation_loss,
)


@dataclass(frozen=True)
class SurfaceLocalTokenJepaConfig:
    n_tokens: int = 58
    token_descriptor_dim: int = 13
    token_hidden_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64
    predictor_hidden_dim: int = 128


def select_target_token_rows(
    token_embeddings: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    if token_embeddings.ndim != 4:
        raise ValueError(
            "token_embeddings must have shape (B, T, N, D), "
            f"got {tuple(token_embeddings.shape)}"
        )
    if target_positions.ndim != 2 or target_positions.shape[1] != 3:
        raise ValueError(
            "target_positions must have shape (K, 3), "
            f"got {tuple(target_positions.shape)}"
        )
    if target_positions.shape[0] < 2:
        raise ValueError("Need at least two target token rows")

    positions = target_positions.to(device=token_embeddings.device, dtype=torch.long)
    max_values = torch.tensor(
        token_embeddings.shape[:3],
        device=token_embeddings.device,
        dtype=torch.long,
    )
    if torch.any(positions < 0) or torch.any(positions >= max_values.view(1, 3)):
        raise ValueError("target_positions contains an out-of-bounds index")
    return token_embeddings[positions[:, 0], positions[:, 1], positions[:, 2]]


class SurfaceLocalTokenEncoder(nn.Module):
    def __init__(
        self,
        cfg: SurfaceLocalTokenJepaConfig,
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
            nn.LayerNorm(4 + cfg.token_descriptor_dim),
            nn.Linear(4 + cfg.token_descriptor_dim, cfg.token_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.token_hidden_dim, cfg.token_hidden_dim),
            nn.SiLU(),
        )
        self.temporal_gru = nn.GRU(
            cfg.token_hidden_dim,
            cfg.hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != observed_mask.shape or values.shape != visible_mask.shape:
            raise ValueError("values, observed_mask, and visible_mask must share shape")
        if values.ndim != 3 or values.shape[-1] != self.cfg.n_tokens:
            raise ValueError(
                f"Expected shape (B, T, {self.cfg.n_tokens}), got {tuple(values.shape)}"
            )

        batch, time, _tokens = values.shape
        descriptors = self.token_descriptors.view(1, 1, self.cfg.n_tokens, -1).expand(
            batch,
            time,
            -1,
            -1,
        )
        if time > 1:
            rel = torch.linspace(
                -1.0,
                1.0,
                steps=time,
                device=values.device,
                dtype=values.dtype,
            )
        else:
            rel = torch.zeros(1, device=values.device, dtype=values.dtype)
        relative_time = rel.view(1, time, 1, 1).expand(batch, -1, self.cfg.n_tokens, -1)
        token_inputs = torch.cat(
            [
                values.float().unsqueeze(-1),
                observed_mask.float().unsqueeze(-1),
                visible_mask.float().unsqueeze(-1),
                relative_time.float(),
                descriptors.to(device=values.device),
            ],
            dim=-1,
        )
        token_hidden = self.token_mlp(token_inputs)
        seq_in = token_hidden.permute(0, 2, 1, 3).reshape(
            batch * self.cfg.n_tokens,
            time,
            self.cfg.token_hidden_dim,
        )
        seq, _hidden = self.temporal_gru(seq_in)
        seq = seq.reshape(batch, self.cfg.n_tokens, time, self.cfg.hidden_dim).permute(
            0,
            2,
            1,
            3,
        )
        return self.head(seq)


class SurfaceLocalTokenJepaModel(nn.Module):
    def __init__(
        self,
        cfg: SurfaceLocalTokenJepaConfig,
        *,
        token_descriptors: np.ndarray,
    ):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = SurfaceLocalTokenEncoder(
            cfg,
            token_descriptors=token_descriptors,
        )
        self.target_encoder = SurfaceLocalTokenEncoder(
            cfg,
            token_descriptors=token_descriptors,
        )
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
        *,
        context_values: torch.Tensor,
        clean_values: torch.Tensor,
        observed_mask: torch.Tensor,
        context_mask: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_tokens = self.context_encoder(
            context_values,
            observed_mask,
            context_mask,
        )
        clean_visible = torch.ones_like(observed_mask, dtype=torch.bool)
        with torch.no_grad():
            target_grid = self.target_encoder(
                clean_values,
                observed_mask,
                clean_visible,
            )
        context_rows = select_target_token_rows(context_tokens, target_positions)
        target_rows = select_target_token_rows(target_grid, target_positions).detach()
        return {
            "context_tokens": context_tokens,
            "predicted_target_tokens": self.predictor(context_rows),
            "target_tokens": target_rows,
            "target_positions": target_positions,
        }


def surface_local_context_target_loss(
    outputs: dict[str, torch.Tensor],
    *,
    barlow_weight: float = 0.05,
    offdiag_weight: float = 0.005,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = outputs["predicted_target_tokens"]
    target = outputs["target_tokens"].detach()
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted_target_tokens and target_tokens must match, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if predicted.shape[0] < 2:
        raise ValueError("Need at least two target token rows for loss")
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
        "target_token_rows": int(predicted.shape[0]),
    }


def surface_local_parameter_groups(
    model: SurfaceLocalTokenJepaModel,
) -> Iterator[nn.Parameter]:
    yield from model.context_encoder.parameters()
    yield from model.predictor.parameters()
