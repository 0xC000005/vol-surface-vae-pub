from __future__ import annotations

import copy
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
    base_noise_rho: float = 0.0
    conditional_base_noise_scale: bool = False
    base_noise_scale_min: float = 0.5
    base_noise_scale_max: float = 2.0
    innovation_coordinate: str = "normalized"
    risk_state_dim: int = 0


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


class GroupHeadTokenTransitionVelocity(nn.Module):
    """Shared token mixer with separate IV/anchor output heads."""

    def __init__(
        self,
        cfg: GenericStateAwareNormalizedInnovationFMConfig,
        base: MemoryConditionedTokenTransitionVelocity | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.base = base if base is not None else MemoryConditionedTokenTransitionVelocity(cfg)
        self.iv_head = copy.deepcopy(self.base.out)
        self.anchor_head = copy.deepcopy(self.base.out)
        iv_count = min(max(int(cfg.readout_iv_count), 0), int(cfg.n_cells))
        group_ids = torch.zeros(int(cfg.n_cells), dtype=torch.long)
        if iv_count < int(cfg.n_cells):
            group_ids[iv_count:] = 1
        self.register_buffer("group_ids", group_ids)

    def forward(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.base.hidden_tokens(x_t, current_logit, memory_state, t)
        out = torch.empty(x_t.shape, device=x_t.device, dtype=x_t.dtype)
        iv_mask = self.group_ids.to(device=out.device) == 0
        anchor_mask = ~iv_mask
        if bool(iv_mask.any()):
            out[:, iv_mask] = self.iv_head(hidden[:, iv_mask, :]).squeeze(-1)
        if bool(anchor_mask.any()):
            out[:, anchor_mask] = self.anchor_head(hidden[:, anchor_mask, :]).squeeze(-1)
        return out


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


def enable_group_head_velocity_readout(
    model: "GenericStateAwareNormalizedInnovationFlowMatching",
    *,
    iv_count: int,
) -> None:
    """Upgrade a loaded shared-readout model to separate group heads as a no-op."""
    if isinstance(model.velocity, GroupHeadTokenTransitionVelocity):
        model.cfg.velocity_readout_mode = "group_head"
        model.cfg.readout_iv_count = int(iv_count)
        return
    if not isinstance(model.velocity, MemoryConditionedTokenTransitionVelocity):
        raise TypeError(f"unsupported velocity module {type(model.velocity).__name__}")
    model.cfg.velocity_readout_mode = "group_head"
    model.cfg.readout_iv_count = int(iv_count)
    upgraded = GroupHeadTokenTransitionVelocity(model.cfg, base=model.velocity)
    model.velocity = upgraded.to(next(model.parameters()).device)


def _make_base_noise_scale_head(cfg: GenericStateAwareNormalizedInnovationFMConfig) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(cfg.memory_dim),
        nn.Linear(cfg.memory_dim, cfg.memory_dim),
        nn.GELU(),
        nn.Linear(cfg.memory_dim, cfg.n_cells),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def _make_risk_state_head(cfg: GenericStateAwareNormalizedInnovationFMConfig) -> nn.Sequential:
    head = nn.Sequential(
        nn.LayerNorm(cfg.memory_dim),
        nn.Linear(cfg.memory_dim, cfg.memory_dim),
        nn.GELU(),
        nn.Linear(cfg.memory_dim, cfg.risk_state_dim),
    )
    return head


def _make_risk_context_proj(cfg: GenericStateAwareNormalizedInnovationFMConfig) -> nn.Sequential:
    proj = nn.Sequential(
        nn.LayerNorm(cfg.risk_state_dim),
        nn.Linear(cfg.risk_state_dim, cfg.memory_dim),
    )
    nn.init.zeros_(proj[-1].weight)
    nn.init.zeros_(proj[-1].bias)
    return proj


def enable_conditional_base_noise_scale(
    model: "GenericStateAwareNormalizedInnovationFlowMatching",
    *,
    scale_min: float,
    scale_max: float,
) -> None:
    """Attach a unit-initialized conditional base-noise scale head."""
    model.cfg.conditional_base_noise_scale = True
    model.cfg.base_noise_scale_min = float(scale_min)
    model.cfg.base_noise_scale_max = float(scale_max)
    if model.base_noise_log_scale is None:
        model.base_noise_log_scale = _make_base_noise_scale_head(model.cfg).to(next(model.parameters()).device)


class GenericStateAwareNormalizedInnovationFlowMatching(nn.Module):
    """Generate normalized innovations while remaining level-aware.

    Sampling returns unnormalized encoded increments for compatibility with the
    existing reconstruction and audit stack.
    """

    def __init__(self, cfg: GenericStateAwareNormalizedInnovationFMConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale", "scale_drift"}:
            raise ValueError("prefix_feature_mode must be 'basic', 'scale', or 'scale_drift'")
        if cfg.innovation_coordinate not in {"normalized", "score", "hybrid_sticky_score"}:
            raise ValueError(
                "innovation_coordinate must be 'normalized', 'score', or 'hybrid_sticky_score'"
            )
        feature_mult = 7 if cfg.prefix_feature_mode == "scale_drift" else 6 if cfg.prefix_feature_mode == "scale" else 4
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
        elif cfg.velocity_readout_mode == "group_head":
            self.velocity = GroupHeadTokenTransitionVelocity(cfg)
        else:
            raise ValueError("velocity_readout_mode must be 'shared', 'group_residual', or 'group_head'")
        self.base_noise_log_scale = (
            _make_base_noise_scale_head(cfg) if bool(cfg.conditional_base_noise_scale) else None
        )
        self.risk_state_head = (
            _make_risk_state_head(cfg) if int(cfg.risk_state_dim) > 0 else None
        )
        self.risk_context_proj = (
            _make_risk_context_proj(cfg) if int(cfg.risk_state_dim) > 0 else None
        )
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("level_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("innovation_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_innovation_quantiles_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("innovation_score_mask", torch.zeros(cfg.n_cells, dtype=torch.bool))

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

    def set_innovation_quantiles(
        self,
        innovation_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if innovation_quantiles.shape != expected:
            raise ValueError(
                f"innovation_quantiles must have shape {expected}, got {tuple(innovation_quantiles.shape)}"
            )
        self.innovation_quantiles.copy_(innovation_quantiles.to(self.innovation_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("quantile_levels must have shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._innovation_quantiles_ready.fill_(True)

    def set_innovation_score_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != (self.cfg.n_cells,):
            raise ValueError(f"innovation score mask must have shape ({self.cfg.n_cells},)")
        self.innovation_score_mask.copy_(mask.to(device=self.innovation_score_mask.device, dtype=torch.bool))

    def _check_innovation_quantiles(self) -> None:
        if not bool(self._innovation_quantiles_ready.item()):
            raise RuntimeError("innovation quantiles must be set before score-coordinate use")

    def _values_to_scores_with_table(self, values: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.cfg.n_cells:
            raise ValueError(f"last dimension must be {self.cfg.n_cells}, got {values.shape[-1]}")
        levels = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        table = table.to(device=values.device, dtype=values.dtype)
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

    def _scores_to_values_with_table(self, scores: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        if scores.shape[-1] != self.cfg.n_cells:
            raise ValueError(f"last dimension must be {self.cfg.n_cells}, got {scores.shape[-1]}")
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = table.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        flat = u_all.reshape(-1, self.cfg.n_cells)
        idx = torch.searchsorted(levels.contiguous(), flat.contiguous(), right=False)
        idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
        idx_lo = idx_hi - 1
        u_lo = levels[idx_lo]
        u_hi = levels[idx_hi]
        cell_idx = torch.arange(self.cfg.n_cells, device=scores.device)[None, :]
        q_lo = table[cell_idx, idx_lo]
        q_hi = table[cell_idx, idx_hi]
        alpha = (flat - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
        value = q_lo + alpha.clamp(0.0, 1.0) * (q_hi - q_lo)
        value = torch.where(flat <= levels[0], table[:, 0][None, :], value)
        value = torch.where(flat >= levels[-1], table[:, -1][None, :], value)
        return value.view_as(scores)

    def level_values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        return self._values_to_scores_with_table(values, self.level_quantiles)

    def normalized_innovations_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        self._check_innovation_quantiles()
        return self._values_to_scores_with_table(values, self.innovation_quantiles)

    def scores_to_normalized_innovations(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_innovation_quantiles()
        return self._scores_to_values_with_table(scores, self.innovation_quantiles)

    def _to_flow_coordinate(self, normalized_innovation: torch.Tensor) -> torch.Tensor:
        if self.cfg.innovation_coordinate == "score":
            return self.normalized_innovations_to_scores(normalized_innovation)
        if self.cfg.innovation_coordinate == "hybrid_sticky_score":
            scores = self.normalized_innovations_to_scores(normalized_innovation)
            mask = self.innovation_score_mask.to(device=normalized_innovation.device)
            mask = mask.view(*([1] * (normalized_innovation.ndim - 1)), self.cfg.n_cells)
            return torch.where(mask, scores, normalized_innovation)
        return normalized_innovation

    def _from_flow_coordinate(self, flow_coordinate: torch.Tensor) -> torch.Tensor:
        if self.cfg.innovation_coordinate == "score":
            return self.scores_to_normalized_innovations(flow_coordinate)
        if self.cfg.innovation_coordinate == "hybrid_sticky_score":
            values = self.scores_to_normalized_innovations(flow_coordinate)
            mask = self.innovation_score_mask.to(device=flow_coordinate.device)
            mask = mask.view(*([1] * (flow_coordinate.ndim - 1)), self.cfg.n_cells)
            return torch.where(mask, values, flow_coordinate)
        return flow_coordinate

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
        drift_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        center_feature, log_scale = self._scale_features(center, scale, level_scores.shape[1])
        if self.cfg.prefix_feature_mode == "scale_drift":
            if drift_feature is None:
                drift_feature = torch.zeros_like(center)
            drift_scaled, _unused = self._scale_features(drift_feature, scale, level_scores.shape[1])
            return torch.cat(
                [
                    level_scores,
                    normalized_innovation,
                    normalized_innovation.abs(),
                    normalized_innovation.square(),
                    center_feature,
                    log_scale,
                    drift_scaled,
                ],
                dim=-1,
            )
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
        drift_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if level_scores.shape != normalized_innovation.shape:
            raise ValueError("level_scores and normalized_innovation must match")
        seq_len = level_scores.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(f"prefix length {seq_len} exceeds configured maximum")
        pos = torch.arange(seq_len, device=level_scores.device)
        x = self.feature_proj(
            self._prefix_features(level_scores, normalized_innovation, center, scale, drift_feature)
        )
        x = x + self.pos_embed(pos)[None, :, :]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=level_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.memory_norm(self.memory(x, mask=mask))

    def _base_noise_like(self, reference: torch.Tensor, rho: float | None = None) -> torch.Tensor:
        """Sample iid or AR(1)-correlated Gaussian base noise over the horizon axis."""
        if reference.ndim != 3:
            raise ValueError("reference must have shape [B,T,C]")
        rho_value = float(self.cfg.base_noise_rho if rho is None else rho)
        if abs(rho_value) < 1e-8:
            return torch.randn_like(reference)
        rho_value = max(-0.99, min(0.99, rho_value))
        eps = torch.randn_like(reference)
        frames = [eps[:, 0]]
        innovation_scale = math.sqrt(max(1.0 - rho_value * rho_value, 1e-8))
        prev = frames[0]
        for step in range(1, reference.shape[1]):
            prev = rho_value * prev + innovation_scale * eps[:, step]
            frames.append(prev)
        return torch.stack(frames, dim=1)

    def _conditional_base_noise_scale(self, memory_state: torch.Tensor) -> torch.Tensor | None:
        if self.base_noise_log_scale is None:
            return None
        lo = math.log(float(self.cfg.base_noise_scale_min))
        hi = math.log(float(self.cfg.base_noise_scale_max))
        return torch.exp(self.base_noise_log_scale(memory_state).clamp(lo, hi))

    def _risk_state_from_history(
        self,
        history_level_scores: torch.Tensor,
        history_flow_coordinate: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        drift_feature: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.risk_state_head is None or self.risk_context_proj is None:
            return None, None
        history_hidden = self._encode_prefix(
            history_level_scores,
            history_flow_coordinate,
            center,
            scale,
            drift_feature,
        )
        predicted = self.risk_state_head(history_hidden[:, -1])
        context = self.risk_context_proj(predicted)
        return predicted, context

    @staticmethod
    def _future_risk_targets(future_flow_coordinate: torch.Tensor) -> torch.Tensor:
        abs_x = future_flow_coordinate.abs()
        activity = future_flow_coordinate.square().mean(dim=(1, 2))
        mean_abs = abs_x.mean(dim=(1, 2))
        max_abs = abs_x.amax(dim=(1, 2))
        per_step = abs_x.mean(dim=2)
        temporal_peak = per_step.amax(dim=1) / per_step.mean(dim=1).clamp_min(1e-8)
        raw = torch.stack([activity, mean_abs, max_abs, temporal_peak], dim=1)
        return torch.log1p(raw)

    def _risk_state_losses(
        self,
        risk_prediction: torch.Tensor | None,
        future_flow_coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if risk_prediction is None:
            zero = future_flow_coordinate.new_zeros(())
            return zero, zero, zero
        target = self._future_risk_targets(future_flow_coordinate)
        target = target[:, : risk_prediction.shape[1]]
        target_z = (target - target.mean(dim=0, keepdim=True)) / target.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)
        regression = F.mse_loss(risk_prediction, target_z.detach())
        if risk_prediction.shape[0] < 2:
            rank_loss = regression.new_zeros(())
        else:
            pred_score = risk_prediction[:, 0]
            target_score = target[:, 0].detach()
            pred_diff = pred_score[:, None] - pred_score[None, :]
            target_diff = target_score[:, None] - target_score[None, :]
            sign = target_diff.sign()
            mask = target_diff.abs() > 1e-6
            if bool(mask.any()):
                rank_loss = F.softplus(-sign[mask] * pred_diff[mask]).mean()
            else:
                rank_loss = regression.new_zeros(())
        rho = torch.zeros((), device=future_flow_coordinate.device, dtype=future_flow_coordinate.dtype)
        if risk_prediction.shape[0] >= 3 and torch.std(risk_prediction[:, 0], unbiased=False) > 1e-8:
            pred_rank = torch.argsort(torch.argsort(risk_prediction[:, 0])).to(future_flow_coordinate.dtype)
            target_rank = torch.argsort(torch.argsort(target[:, 0])).to(future_flow_coordinate.dtype)
            pred_rank = pred_rank - pred_rank.mean()
            target_rank = target_rank - target_rank.mean()
            rho = (pred_rank * target_rank).mean() / (
                pred_rank.std(unbiased=False).clamp_min(1e-8)
                * target_rank.std(unbiased=False).clamp_min(1e-8)
            )
        return regression, rank_loss, rho.detach()

    def training_loss(
        self,
        history_level_values: torch.Tensor,
        history_normalized_innovation: torch.Tensor,
        future_level_values: torch.Tensor,
        future_normalized_innovation: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        drift_feature: torch.Tensor | None = None,
        condition_contrast_weight: float = 0.0,
        condition_contrast_margin: float = 0.0,
        risk_state_weight: float = 0.0,
        risk_state_rank_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_level_scores = self.level_values_to_scores(history_level_values)
        future_level_scores = self.level_values_to_scores(future_level_values)
        history_flow_coordinate = self._to_flow_coordinate(history_normalized_innovation)
        future_flow_coordinate = self._to_flow_coordinate(future_normalized_innovation)
        prefix_level_scores = torch.cat([history_level_scores, future_level_scores[:, :-1]], dim=1)
        prefix_norm = torch.cat(
            [history_flow_coordinate, future_flow_coordinate[:, :-1]],
            dim=1,
        )
        hidden = self._encode_prefix(prefix_level_scores, prefix_norm, center, scale, drift_feature)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        risk_prediction, risk_context = self._risk_state_from_history(
            history_level_scores,
            history_flow_coordinate,
            center,
            scale,
            drift_feature,
        )
        if risk_context is not None:
            memory_states = memory_states + risk_context[:, None, :]
        current_level_scores = prefix_level_scores[:, start : start + self.cfg.future_len]

        x1 = future_flow_coordinate
        x0 = self._base_noise_like(x1)
        base_noise_scale = self._conditional_base_noise_scale(memory_states)
        if base_noise_scale is not None:
            x0 = x0 * base_noise_scale
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
                [history_flow_coordinate[perm], future_flow_coordinate[:, :-1]],
                dim=1,
            )
            neg_hidden = self._encode_prefix(
                neg_prefix_level_scores,
                neg_prefix_norm,
                center[perm],
                scale[perm],
                None if drift_feature is None else drift_feature[perm],
            )
            neg_memory_states = neg_hidden[:, start : start + self.cfg.future_len]
            _neg_risk_prediction, neg_risk_context = self._risk_state_from_history(
                history_level_scores[perm],
                history_flow_coordinate[perm],
                center[perm],
                scale[perm],
                None if drift_feature is None else drift_feature[perm],
            )
            if neg_risk_context is not None:
                neg_memory_states = neg_memory_states + neg_risk_context[:, None, :]
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
        risk_loss, risk_rank_loss, risk_rank_rho = self._risk_state_losses(
            risk_prediction,
            future_flow_coordinate,
        )
        total_loss = (
            fm_loss
            + contrast_weight * contrast_loss
            + float(risk_state_weight) * risk_loss
            + float(risk_state_rank_weight) * risk_rank_loss
        )
        if base_noise_scale is None:
            base_noise_scale_enabled = torch.zeros((), device=x1.device, dtype=x1.dtype)
            base_noise_scale_mean = torch.ones((), device=x1.device, dtype=x1.dtype)
            base_noise_scale_std = torch.zeros((), device=x1.device, dtype=x1.dtype)
            base_noise_scale_min = torch.ones((), device=x1.device, dtype=x1.dtype)
            base_noise_scale_max = torch.ones((), device=x1.device, dtype=x1.dtype)
        else:
            base_noise_scale_enabled = torch.ones((), device=x1.device, dtype=x1.dtype)
            base_noise_scale_mean = base_noise_scale.mean().detach()
            base_noise_scale_std = base_noise_scale.std(unbiased=False).detach()
            base_noise_scale_min = base_noise_scale.min().detach()
            base_noise_scale_max = base_noise_scale.max().detach()
        return total_loss, {
            "total": total_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "condition_contrast_loss": contrast_loss.detach(),
            "condition_contrast_neg_loss": neg_loss.detach(),
            "condition_contrast_weight": torch.as_tensor(contrast_weight, device=x1.device, dtype=x1.dtype),
            "base_noise_rho": torch.as_tensor(float(self.cfg.base_noise_rho), device=x1.device, dtype=x1.dtype),
            "base_noise_scale_enabled": base_noise_scale_enabled,
            "base_noise_scale_mean": base_noise_scale_mean,
            "base_noise_scale_std": base_noise_scale_std,
            "base_noise_scale_min": base_noise_scale_min,
            "base_noise_scale_max": base_noise_scale_max,
            "risk_state_enabled": torch.as_tensor(
                1.0 if risk_prediction is not None else 0.0,
                device=x1.device,
                dtype=x1.dtype,
            ),
            "risk_state_loss": risk_loss.detach(),
            "risk_state_rank_loss": risk_rank_loss.detach(),
            "risk_state_rank_rho": risk_rank_rho,
            "risk_state_weight": torch.as_tensor(float(risk_state_weight), device=x1.device, dtype=x1.dtype),
            "risk_state_rank_weight": torch.as_tensor(float(risk_state_rank_weight), device=x1.device, dtype=x1.dtype),
            "target_norm_std": x1.std(unbiased=False).detach(),
            "target_flow_std": x1.std(unbiased=False).detach(),
            "target_norm_abs": x1.abs().mean().detach(),
            "local_scale_mean": scale.mean().detach(),
            "local_scale_min": scale.min().detach(),
            "drift_feature_abs": (
                torch.zeros((), device=x1.device, dtype=x1.dtype)
                if drift_feature is None
                else drift_feature.abs().mean().detach()
            ),
            "memory_abs": memory_states.abs().mean().detach(),
        }

    @torch.no_grad()
    def sample_batched(
        self,
        history_level_values: torch.Tensor,
        history_normalized_innovation: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        drift_feature: torch.Tensor | None = None,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        level_scores = self.level_values_to_scores(history_level_values)
        history_flow_coordinate = self._to_flow_coordinate(history_normalized_innovation)
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
                history_flow_coordinate.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            center_rep = center.unsqueeze(1).expand(bsz, k, self.cfg.n_cells).reshape(bsz * k, self.cfg.n_cells)
            scale_rep = scale.unsqueeze(1).expand(bsz, k, self.cfg.n_cells).reshape(bsz * k, self.cfg.n_cells)
            if drift_feature is None:
                drift_rep = None
            else:
                drift_rep = (
                    drift_feature.unsqueeze(1)
                    .expand(bsz, k, self.cfg.n_cells)
                    .reshape(bsz * k, self.cfg.n_cells)
                )
            _risk_prediction, risk_context = self._risk_state_from_history(
                prefix_level_scores,
                prefix_norm,
                center_rep,
                scale_rep,
                drift_rep,
            )
            base_noise = temp * self._base_noise_like(
                torch.empty(
                    bsz * k,
                    int(n_steps),
                    self.cfg.n_cells,
                    device=level_scores.device,
                    dtype=level_scores.dtype,
                )
            )
            frames: list[torch.Tensor] = []
            for _step in range(int(n_steps)):
                memory_state = self._encode_prefix(prefix_level_scores, prefix_norm, center_rep, scale_rep, drift_rep)[
                    :, -1
                ]
                if risk_context is not None:
                    memory_state = memory_state + risk_context
                current_level_score = prefix_level_scores[:, -1]
                x = base_noise[:, _step]
                base_noise_scale = self._conditional_base_noise_scale(memory_state)
                if base_noise_scale is not None:
                    x = x * base_noise_scale
                for flow_step in range(int(self.cfg.flow_steps)):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=level_scores.device,
                        dtype=level_scores.dtype,
                    )
                    x = x + dt * self.velocity(x, current_level_score, memory_state, t)
                next_flow_coordinate = x
                next_norm = self._from_flow_coordinate(next_flow_coordinate)
                next_increment = next_norm * scale_rep + center_rep
                next_level_value = prefix_level_values[:, -1] + next_increment
                next_level_score = self.level_values_to_scores(next_level_value)
                frames.append(next_increment.view(bsz, k, self.cfg.n_cells))
                prefix_level_values = torch.cat([prefix_level_values, next_level_value[:, None, :]], dim=1)
                prefix_level_scores = torch.cat([prefix_level_scores, next_level_score[:, None, :]], dim=1)
                prefix_norm = torch.cat([prefix_norm, next_flow_coordinate[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericStateAwareNormalizedInnovationFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericStateAwareNormalizedInnovationFMConfig(**payload["config"])
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    incompat = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {"innovation_quantiles", "_innovation_quantiles_ready", "innovation_score_mask"}
    if cfg.innovation_coordinate == "score":
        allowed_missing = {"innovation_score_mask"}
    if cfg.innovation_coordinate == "hybrid_sticky_score":
        allowed_missing = set()
    unexpected = set(incompat.unexpected_keys)
    missing = set(incompat.missing_keys)
    if unexpected or missing.difference(allowed_missing):
        raise RuntimeError(
            "checkpoint state dict mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
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
