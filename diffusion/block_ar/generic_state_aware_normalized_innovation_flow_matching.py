from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.causal_future_memory_transition_flow_matching import (
    CausalFutureMemoryTransitionFMConfig,
    MemoryConditionedTokenTransitionVelocity,
)


@dataclass
class GenericStateAwareNormalizedInnovationFMConfig(CausalFutureMemoryTransitionFMConfig):
    """Flow for normalized innovations conditioned on encoded state."""

    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "scale"


class GenericStateAwareNormalizedInnovationFlowMatching(nn.Module):
    """Generate normalized innovations while remaining level-aware.

    Sampling returns unnormalized encoded increments for compatibility with the
    existing reconstruction and audit stack.
    """

    def __init__(self, cfg: GenericStateAwareNormalizedInnovationFMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        feature_mult = 6 if cfg.prefix_feature_mode == "scale" else 4
        self.feature_proj = nn.Linear(feature_mult * cfg.n_cells, cfg.memory_dim)
        self.pos_embed = nn.Embedding(cfg.history_len + cfg.future_len, cfg.memory_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.memory_dim,
            nhead=cfg.memory_heads,
            dim_feedforward=cfg.memory_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory = nn.TransformerEncoder(layer, num_layers=cfg.memory_layers)
        self.memory_norm = nn.LayerNorm(cfg.memory_dim)
        self.velocity = MemoryConditionedTokenTransitionVelocity(cfg)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("level_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    def set_level_quantiles(
        self,
        level_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if level_quantiles.shape != expected:
            raise ValueError(f"level_quantiles must have shape {expected}, got {tuple(level_quantiles.shape)}")
        self.level_quantiles.copy_(level_quantiles.to(self.level_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("quantile_levels must have shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("level quantiles must be set before use")

    def level_values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        if values.shape[-1] != self.cfg.n_cells:
            raise ValueError(f"last dimension must be {self.cfg.n_cells}, got {values.shape[-1]}")
        levels = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        table = self.level_quantiles.to(device=values.device, dtype=values.dtype)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_cells):
            q = table[var]
            flat = values[..., var].reshape(-1)
            idx = torch.searchsorted(q.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            alpha = (flat - q_lo) / (q_hi - q_lo).clamp_min(1e-12)
            u = u_lo + alpha.clamp(0.0, 1.0) * (u_hi - u_lo)
            u = torch.where(flat <= q[0], levels[0], u)
            u = torch.where(flat >= q[-1], levels[-1], u)
            eps = float(self.cfg.cdf_eps)
            score = torch.special.ndtri(u.clamp(eps, 1.0 - eps))
            cols.append(score.view(values.shape[:-1]))
        return torch.stack(cols, dim=-1)

    @staticmethod
    def _scale_features(center: torch.Tensor, scale: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        safe_scale = scale.clamp_min(1e-8)
        center_feature = center / safe_scale
        log_scale = torch.log(safe_scale)
        return (
            center_feature[:, None, :].expand(-1, seq_len, -1),
            log_scale[:, None, :].expand(-1, seq_len, -1),
        )

    def _prefix_features(
        self,
        level_scores: torch.Tensor,
        normalized_innovation: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        center_feature, log_scale = self._scale_features(center, scale, level_scores.shape[1])
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat(
                [
                    level_scores,
                    normalized_innovation,
                    normalized_innovation.abs(),
                    normalized_innovation.square(),
                    center_feature,
                    log_scale,
                ],
                dim=-1,
            )
        return torch.cat([level_scores, normalized_innovation, center_feature, log_scale], dim=-1)

    def _encode_prefix(
        self,
        level_scores: torch.Tensor,
        normalized_innovation: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        if level_scores.shape != normalized_innovation.shape:
            raise ValueError("level_scores and normalized_innovation must match")
        seq_len = level_scores.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(f"prefix length {seq_len} exceeds configured maximum")
        pos = torch.arange(seq_len, device=level_scores.device)
        x = self.feature_proj(self._prefix_features(level_scores, normalized_innovation, center, scale))
        x = x + self.pos_embed(pos)[None, :, :]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=level_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.memory_norm(self.memory(x, mask=mask))

    def training_loss(
        self,
        history_level_values: torch.Tensor,
        history_normalized_innovation: torch.Tensor,
        future_level_values: torch.Tensor,
        future_normalized_innovation: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_level_scores = self.level_values_to_scores(history_level_values)
        future_level_scores = self.level_values_to_scores(future_level_values)
        prefix_level_scores = torch.cat([history_level_scores, future_level_scores[:, :-1]], dim=1)
        prefix_norm = torch.cat(
            [history_normalized_innovation, future_normalized_innovation[:, :-1]],
            dim=1,
        )
        hidden = self._encode_prefix(prefix_level_scores, prefix_norm, center, scale)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_level_scores = prefix_level_scores[:, start : start + self.cfg.future_len]

        x1 = future_normalized_innovation
        x0 = torch.randn_like(x1)
        bsz, horizon, n_vars = x1.shape
        t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.velocity(
            x_t.reshape(bsz * horizon, n_vars),
            current_level_scores.reshape(bsz * horizon, n_vars),
            memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
            t.reshape(bsz * horizon),
        ).view_as(x1)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        return fm_loss, {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "target_norm_std": x1.std(unbiased=False).detach(),
            "target_norm_abs": x1.abs().mean().detach(),
            "local_scale_mean": scale.mean().detach(),
            "local_scale_min": scale.min().detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }

    @torch.no_grad()
    def sample_batched(
        self,
        history_level_values: torch.Tensor,
        history_normalized_innovation: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        level_scores = self.level_values_to_scores(history_level_values)
        bsz = int(level_scores.shape[0])
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)
        outs: list[torch.Tensor] = []
        for start in range(0, int(n_samples), chunk_size):
            k = min(chunk_size, int(n_samples) - start)
            prefix_level_values = (
                history_level_values.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            prefix_level_scores = (
                level_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            prefix_norm = (
                history_normalized_innovation.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            center_rep = center.unsqueeze(1).expand(bsz, k, self.cfg.n_cells).reshape(bsz * k, self.cfg.n_cells)
            scale_rep = scale.unsqueeze(1).expand(bsz, k, self.cfg.n_cells).reshape(bsz * k, self.cfg.n_cells)
            frames: list[torch.Tensor] = []
            for _step in range(int(n_steps)):
                memory_state = self._encode_prefix(prefix_level_scores, prefix_norm, center_rep, scale_rep)[:, -1]
                current_level_score = prefix_level_scores[:, -1]
                x = temp * torch.randn(
                    bsz * k,
                    self.cfg.n_cells,
                    device=level_scores.device,
                    dtype=level_scores.dtype,
                )
                for flow_step in range(int(self.cfg.flow_steps)):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=level_scores.device,
                        dtype=level_scores.dtype,
                    )
                    x = x + dt * self.velocity(x, current_level_score, memory_state, t)
                next_norm = x
                next_increment = next_norm * scale_rep + center_rep
                next_level_value = prefix_level_values[:, -1] + next_increment
                next_level_score = self.level_values_to_scores(next_level_value)
                frames.append(next_increment.view(bsz, k, self.cfg.n_cells))
                prefix_level_values = torch.cat([prefix_level_values, next_level_value[:, None, :]], dim=1)
                prefix_level_scores = torch.cat([prefix_level_scores, next_level_score[:, None, :]], dim=1)
                prefix_norm = torch.cat([prefix_norm, next_norm[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericStateAwareNormalizedInnovationFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericStateAwareNormalizedInnovationFMConfig(**payload["config"])
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    cfg: GenericStateAwareNormalizedInnovationFMConfig,
    epoch: int,
    best_val: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "config": asdict(cfg),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
