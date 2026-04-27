from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GenericGaussianTransitionConfig:
    """Likelihood-trained AR transition law for generic financial state panels."""

    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    prefix_feature_mode: str = "basic"
    memory_dim: int = 128
    memory_layers: int = 3
    memory_heads: int = 4
    memory_ff: int = 256
    head_hidden: int = 256
    model_dropout: float = 0.05
    diag_floor: float = 0.03
    diag_max: float = 3.0
    offdiag_scale: float = 0.25
    sample_temperature: float = 1.0


class GenericGaussianTransitionLaw(nn.Module):
    """Autoregressive conditional Gaussian law over empirical-score increments.

    This is a density model, not a flow-matching velocity model. It maximizes the
    exact one-step conditional likelihood of the next score increment, then rolls
    that transition law forward autoregressively.
    """

    def __init__(self, cfg: GenericGaussianTransitionConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prefix_feature_mode not in {"basic", "scale"}:
            raise ValueError("prefix_feature_mode must be 'basic' or 'scale'")
        feature_mult = 4 if cfg.prefix_feature_mode == "scale" else 2
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
        self.tril_size = cfg.n_cells * (cfg.n_cells - 1) // 2
        out_dim = cfg.n_cells + cfg.n_cells + self.tril_size
        self.param_head = nn.Sequential(
            nn.LayerNorm(cfg.memory_dim + cfg.n_cells),
            nn.Linear(cfg.memory_dim + cfg.n_cells, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.model_dropout),
            nn.Linear(cfg.head_hidden, out_dim),
        )
        self._init_param_head()
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("value_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))

    def _init_param_head(self) -> None:
        last = self.param_head[-1]
        if not isinstance(last, nn.Linear):
            return
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # Start from a moderately broad score-increment scale. The model can
        # shrink or expand it by likelihood, but does not get a degenerate source
        # distribution to optimize away.
        diag_bias = math.log(math.exp(0.55) - 1.0)
        with torch.no_grad():
            start = self.cfg.n_cells
            end = start + self.cfg.n_cells
            last.bias[start:end].fill_(diag_bias)

    def set_empirical_quantiles(
        self,
        value_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if value_quantiles.shape != expected:
            raise ValueError(f"expected value_quantiles shape {expected}, got {tuple(value_quantiles.shape)}")
        self.value_quantiles.copy_(value_quantiles.to(self.value_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("quantile_levels must have shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("empirical quantiles must be set before use")

    def values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        if values.ndim != 3 or values.shape[-1] != self.cfg.n_cells:
            raise ValueError(
                f"values must have shape (batch,time,{self.cfg.n_cells}), got {tuple(values.shape)}"
            )
        levels = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        table = self.value_quantiles.to(device=values.device, dtype=values.dtype)
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

    def scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        if scores.ndim != 3 and scores.ndim != 2:
            raise ValueError("scores must have shape (batch,time,vars) or (batch,vars)")
        if scores.shape[-1] != self.cfg.n_cells:
            raise ValueError(f"last dimension must be {self.cfg.n_cells}")
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.value_quantiles.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_cells):
            q = table[var]
            flat = u_all[..., var].reshape(-1)
            idx = torch.searchsorted(levels.contiguous(), flat.contiguous(), right=False)
            idx_hi = idx.clamp(1, self.cfg.n_quantiles - 1)
            idx_lo = idx_hi - 1
            u_lo = levels[idx_lo]
            u_hi = levels[idx_hi]
            q_lo = q[idx_lo]
            q_hi = q[idx_hi]
            alpha = (flat - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
            value = q_lo + alpha.clamp(0.0, 1.0) * (q_hi - q_lo)
            value = torch.where(flat <= levels[0], q[0], value)
            value = torch.where(flat >= levels[-1], q[-1], value)
            cols.append(value.view(scores.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _score_features(self, scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        if self.cfg.prefix_feature_mode == "scale":
            return torch.cat([scores, deltas, deltas.abs(), deltas.square()], dim=-1)
        return torch.cat([scores, deltas], dim=-1)

    def _encode_prefix_scores(self, prefix_scores: torch.Tensor) -> torch.Tensor:
        seq_len = prefix_scores.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(
                f"prefix length {seq_len} exceeds {self.cfg.history_len + self.cfg.future_len}"
            )
        pos = torch.arange(seq_len, device=prefix_scores.device)
        x = self.feature_proj(self._score_features(prefix_scores))
        x = x + self.pos_embed(pos)[None, :, :]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=prefix_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.memory_norm(self.memory(x, mask=mask))

    def _params(
        self,
        memory_state: torch.Tensor,
        current_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.param_head(torch.cat([memory_state, current_score], dim=-1))
        d = self.cfg.n_cells
        mean = raw[..., :d]
        raw_diag = raw[..., d : 2 * d]
        raw_offdiag = raw[..., 2 * d :]
        diag = F.softplus(raw_diag).clamp_max(float(self.cfg.diag_max)) + float(self.cfg.diag_floor)
        tril = torch.zeros(*raw.shape[:-1], d, d, device=raw.device, dtype=raw.dtype)
        idx = torch.tril_indices(d, d, offset=-1, device=raw.device)
        tril[..., idx[0], idx[1]] = torch.tanh(raw_offdiag) * float(self.cfg.offdiag_scale)
        diag_idx = torch.arange(d, device=raw.device)
        tril[..., diag_idx, diag_idx] = diag
        return mean, tril, diag

    def training_loss(
        self,
        history_values: torch.Tensor,
        future_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.values_to_scores(history_values)
        future_scores = self.values_to_scores(future_values)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        hidden = self._encode_prefix_scores(prefix_scores)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_scores = prefix_scores[:, start : start + self.cfg.future_len]
        target = future_scores - current_scores
        bsz, horizon, n_vars = target.shape
        mean, tril, diag = self._params(
            memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
            current_scores.reshape(bsz * horizon, n_vars),
        )
        target_flat = target.reshape(bsz * horizon, n_vars)
        dist = torch.distributions.MultivariateNormal(mean, scale_tril=tril)
        nll = -dist.log_prob(target_flat).mean()
        pred_error = target_flat - mean
        metrics = {
            "total": nll.detach(),
            "nll": nll.detach(),
            "transition_std": target.std(unbiased=False).detach(),
            "transition_abs": target.abs().mean().detach(),
            "pred_error_abs": pred_error.abs().mean().detach(),
            "diag_mean": diag.mean().detach(),
            "diag_std": diag.std(unbiased=False).detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }
        return nll, metrics

    @torch.no_grad()
    def sample_batched(
        self,
        history_values: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_scores = self.values_to_scores(history_values)
        bsz = history_scores.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        outs: list[torch.Tensor] = []
        for start in range(0, int(n_samples), chunk_size):
            k = min(chunk_size, int(n_samples) - start)
            prefix = (
                history_scores.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                memory_state = self._encode_prefix_scores(prefix)[:, -1]
                current_score = prefix[:, -1]
                mean, tril, _diag = self._params(memory_state, current_score)
                eps = torch.randn(
                    mean.shape[0],
                    self.cfg.n_cells,
                    1,
                    device=mean.device,
                    dtype=mean.dtype,
                )
                transition = mean + temp * torch.bmm(tril, eps).squeeze(-1)
                next_score = current_score + transition
                next_values = self.scores_to_values(next_score)
                frames.append(next_values.view(bsz, k, self.cfg.n_cells))
                prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GenericGaussianTransitionLaw, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GenericGaussianTransitionConfig(**payload["config"])
    model = GenericGaussianTransitionLaw(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: GenericGaussianTransitionLaw,
    cfg: GenericGaussianTransitionConfig,
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
