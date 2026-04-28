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
    velocity_readout_mode: str = "shared"
    readout_iv_count: int = 25


class GroupResidualTokenTransitionVelocity(nn.Module):
    """Shared velocity with zero-initialized IV/anchor residual readouts."""

    def __init__(
        self,
        cfg: GenericStateAwareNormalizedInnovationFMConfig,
        base: MemoryConditionedTokenTransitionVelocity | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.base = base if base is not None else MemoryConditionedTokenTransitionVelocity(cfg)
        self.iv_head = self._make_residual_head(cfg.token_dim)
        self.anchor_head = self._make_residual_head(cfg.token_dim)
        iv_count = min(max(int(cfg.readout_iv_count), 0), int(cfg.n_cells))
        group_ids = torch.zeros(int(cfg.n_cells), dtype=torch.long)
        if iv_count < int(cfg.n_cells):
            group_ids[iv_count:] = 1
        self.register_buffer("group_ids", group_ids)

    @staticmethod
    def _make_residual_head(token_dim: int) -> nn.Sequential:
        head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.base.hidden_tokens(x_t, current_logit, memory_state, t)
        out = self.base.out(hidden).squeeze(-1)
        residual = torch.zeros_like(out)
        iv_mask = self.group_ids.to(device=out.device) == 0
        anchor_mask = ~iv_mask
        if bool(iv_mask.any()):
            residual[:, iv_mask] = self.iv_head(hidden[:, iv_mask, :]).squeeze(-1)
        if bool(anchor_mask.any()):
            residual[:, anchor_mask] = self.anchor_head(hidden[:, anchor_mask, :]).squeeze(-1)
        return out + residual


def enable_group_residual_velocity_readout(
    model: "GenericStateAwareNormalizedInnovationFlowMatching",
    *,
    iv_count: int,
) -> None:
    """Upgrade a loaded shared-readout model without changing its initial outputs."""
    if isinstance(model.velocity, GroupResidualTokenTransitionVelocity):
        model.cfg.velocity_readout_mode = "group_residual"
        model.cfg.readout_iv_count = int(iv_count)
        return
    if not isinstance(model.velocity, MemoryConditionedTokenTransitionVelocity):
        raise TypeError(f"unsupported velocity module {type(model.velocity).__name__}")
    model.cfg.velocity_readout_mode = "group_residual"
    model.cfg.readout_iv_count = int(iv_count)
    upgraded = GroupResidualTokenTransitionVelocity(model.cfg, base=model.velocity)
    model.velocity = upgraded.to(next(model.parameters()).device)


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
        if cfg.velocity_readout_mode == "shared":
            self.velocity = MemoryConditionedTokenTransitionVelocity(cfg)
        elif cfg.velocity_readout_mode == "group_residual":
            self.velocity = GroupResidualTokenTransitionVelocity(cfg)
        else:
            raise ValueError("velocity_readout_mode must be 'shared' or 'group_residual'")
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
        condition_contrast_weight: float = 0.0,
        condition_contrast_margin: float = 0.0,
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
        pos_loss_per_window = (pred_velocity - target_velocity).square().mean(dim=(1, 2))
        fm_loss = pos_loss_per_window.mean()
        contrast_weight = float(condition_contrast_weight)
        contrast_loss = fm_loss.new_zeros(())
        neg_loss = fm_loss.new_zeros(())
        if contrast_weight > 0.0 and bsz > 1:
            perm = torch.roll(torch.arange(bsz, device=x1.device), shifts=1)
            neg_prefix_level_scores = torch.cat(
                [history_level_scores[perm], future_level_scores[:, :-1]],
                dim=1,
            )
            neg_prefix_norm = torch.cat(
                [history_normalized_innovation[perm], future_normalized_innovation[:, :-1]],
                dim=1,
            )
            neg_hidden = self._encode_prefix(
                neg_prefix_level_scores,
                neg_prefix_norm,
                center[perm],
                scale[perm],
            )
            neg_memory_states = neg_hidden[:, start : start + self.cfg.future_len]
            neg_current_level_scores = neg_prefix_level_scores[:, start : start + self.cfg.future_len]
            neg_pred_velocity = self.velocity(
                x_t.reshape(bsz * horizon, n_vars),
                neg_current_level_scores.reshape(bsz * horizon, n_vars),
                neg_memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
                t.reshape(bsz * horizon),
            ).view_as(x1)
            neg_loss_per_window = (neg_pred_velocity - target_velocity).square().mean(dim=(1, 2))
            neg_loss = neg_loss_per_window.mean()
            contrast_loss = F.softplus(
                pos_loss_per_window - neg_loss_per_window + float(condition_contrast_margin)
            ).mean()
        total_loss = fm_loss + contrast_weight * contrast_loss
        return total_loss, {
            "total": total_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "condition_contrast_loss": contrast_loss.detach(),
            "condition_contrast_neg_loss": neg_loss.detach(),
            "condition_contrast_weight": torch.as_tensor(contrast_weight, device=x1.device, dtype=x1.dtype),
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
