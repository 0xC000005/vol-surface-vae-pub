from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.future_scalar_ar_mixture_density_model import (
    FutureScalarARMixtureDensityModel,
    load_model as load_scalar_ar_model,
)
from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder
from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class ConditionalMarginalCopulaConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25

    context_dim: int = 192
    history_hidden: int = 128
    encoder_dropout: float = 0.1
    marginal_hidden: int = 256

    logit_eps: float = 1e-4
    logit_std_floor: float = 1e-3
    scale_floor: float = 1e-3
    sample_temperature: float = 1.0
    max_sample_chunk: int = 8
    base_checkpoint: str = "models/backfill/321c_v0_s42/best_model.pt"


class ConditionalMarginalCopulaModel(nn.Module):
    """323a: conditional scalar marginals plus an empirical neural copula.

    The marginal law is learned directly from history. Dependence is supplied by
    the rank ordering of a clean scalar AR model, so sampling uses the identity
    "joint law = marginals + copula" rather than an additive correction path.
    """

    def __init__(
        self,
        cfg: ConditionalMarginalCopulaConfig,
        base_model: FutureScalarARMixtureDensityModel | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        hist_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            gru_hidden_dim=cfg.history_hidden,
            bottleneck_dim=cfg.context_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=0.0,
        )
        self.history_encoder = GRUEncoder(hist_cfg)
        self.context_proj = nn.Linear(cfg.context_dim, cfg.marginal_hidden)
        self.day_embed = nn.Embedding(cfg.future_len, cfg.marginal_hidden)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.marginal_hidden)
        self.last_level_proj = nn.Linear(1, cfg.marginal_hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.marginal_hidden, cfg.marginal_hidden),
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.marginal_hidden, 2),
        )
        token_idx = torch.arange(cfg.future_len * cfg.n_cells)
        self.register_buffer("token_day", token_idx // cfg.n_cells)
        self.register_buffer("token_cell", token_idx % cfg.n_cells)
        self.register_buffer("cell_logit_mean", torch.zeros(cfg.n_cells))
        self.register_buffer("cell_logit_std", torch.ones(cfg.n_cells))
        # Keep the copula supplier out of this module's state_dict. The 323a
        # checkpoint stores its path and only learns the conditional marginals.
        self.__dict__["base_model"] = base_model

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_logit_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != (self.cfg.n_cells,) or std.shape != (self.cfg.n_cells,):
            raise ValueError("Expected per-cell logit stats with shape (n_cells,)")
        self.cell_logit_mean.copy_(mean.to(self.cell_logit_mean))
        self.cell_logit_std.copy_(
            std.clamp_min(self.cfg.logit_std_floor).to(self.cell_logit_std)
        )

    def to_logits(self, levels_norm: torch.Tensor) -> torch.Tensor:
        levels_norm = self._flatten(levels_norm)
        levels_01 = denormalize_iv(levels_norm)
        return iv_to_logit(levels_01, self.cfg.logit_eps)

    def _to_model_coord(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits - self.cell_logit_mean) / self.cell_logit_std

    def _from_model_coord(self, coord: torch.Tensor) -> torch.Tensor:
        return coord * self.cell_logit_std + self.cell_logit_mean

    def marginal_params(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_norm = self._flatten(history_norm)
        context = self.history_encoder(history_norm)
        history_coord = self._to_model_coord(self.to_logits(history_norm))
        last_cell = history_coord[:, -1, self.token_cell].unsqueeze(-1)
        token_state = self.context_proj(context)[:, None]
        token_state = token_state + self.day_embed(self.token_day)[None]
        token_state = token_state + self.cell_embed(self.token_cell)[None]
        token_state = token_state + self.last_level_proj(last_cell)
        raw = self.head(token_state)
        mean = raw[..., 0].view(history_norm.shape[0], self.cfg.future_len, self.cfg.n_cells)
        scale = (F.softplus(raw[..., 1]) + self.cfg.scale_floor).view_as(mean)
        return mean, scale

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_norm = self._flatten(history_norm)
        future_norm = self._flatten(future_norm)
        target = self._to_model_coord(self.to_logits(future_norm))
        mean, scale = self.marginal_params(history_norm)
        z = (target - mean) / scale
        nll_grid = 0.5 * z.pow(2) + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)
        nll = nll_grid.mean()
        metrics = {
            "nll": nll.detach(),
            "target_std": target.std(unbiased=False).detach(),
            "scale_mean": scale.mean().detach(),
            "mean_abs_err": (mean - target).abs().mean().detach(),
        }
        return nll, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        temperature: float | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        if self.base_model is None:
            raise RuntimeError("ConditionalMarginalCopulaModel requires a base copula model")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_norm = self._flatten(history_norm)
        mean, scale = self.marginal_params(history_norm)
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        base_samples = self.base_model.sample_batched(
            history_norm,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
            **kwargs,
        )
        base_flat = base_samples.view(
            base_samples.shape[0],
            base_samples.shape[1],
            self.cfg.future_len,
            self.cfg.n_cells,
        )
        base_coord = self._to_model_coord(
            iv_to_logit(base_flat, self.cfg.logit_eps)
        )
        ranks = torch.argsort(torch.argsort(base_coord, dim=1), dim=1).to(base_coord)
        u = (ranks + 0.5) / float(n_samples)
        eps = 1.0 / (2.0 * float(n_samples) + 2.0)
        z = torch.special.ndtri(u.clamp(eps, 1.0 - eps))
        future_coord = mean[:, None] + temp * scale[:, None] * z
        future_01 = logit_to_iv(self._from_model_coord(future_coord))
        if self.cfg.n_cells == 25:
            return future_01.view(
                history_norm.shape[0],
                n_samples,
                self.cfg.future_len,
                5,
                5,
            )
        return future_01


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ConditionalMarginalCopulaModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ConditionalMarginalCopulaConfig(**payload["config"])
    base_model, _ = load_scalar_ar_model(cfg.base_checkpoint, device)
    model = ConditionalMarginalCopulaModel(cfg, base_model=base_model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    model.base_model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: ConditionalMarginalCopulaModel,
    cfg: ConditionalMarginalCopulaConfig,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "config": asdict(cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
