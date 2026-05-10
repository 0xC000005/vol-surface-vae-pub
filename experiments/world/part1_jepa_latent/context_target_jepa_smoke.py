from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        raise ValueError(f"target_mask must have shape (B, T, N), got {tuple(target_mask.shape)}")
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
        raise ValueError("Need at least two target time rows for context-to-target loss")
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
