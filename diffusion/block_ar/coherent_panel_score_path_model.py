from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoherentPanelScorePathConfig:
    history_len: int = 30
    future_len: int = 30
    n_vars: int = 51
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    history_hidden: int = 192
    token_hidden: int = 256
    encoder_layers: int = 1
    dropout: float = 0.05
    scale_floor: float = 1e-3
    scale_max: float = 5.0
    sample_temperature: float = 1.0


class CoherentPanelScorePathModel(nn.Module):
    """535a: one coherent Gaussian score-path density for a raw financial panel."""

    def __init__(self, cfg: CoherentPanelScorePathConfig):
        super().__init__()
        self.cfg = cfg
        self.path_dim = int(cfg.future_len * cfg.n_vars)
        self.history_encoder = nn.GRU(
            input_size=2 * cfg.n_vars,
            hidden_size=cfg.history_hidden,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )
        self.context_norm = nn.LayerNorm(cfg.history_hidden)
        self.context_proj = nn.Linear(cfg.history_hidden, cfg.token_hidden)
        self.day_embed = nn.Embedding(cfg.future_len, cfg.token_hidden)
        self.var_embed = nn.Embedding(cfg.n_vars, cfg.token_hidden)
        self.last_score_proj = nn.Linear(1, cfg.token_hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_hidden, cfg.token_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.token_hidden, 2),
        )
        token_idx = torch.arange(self.path_dim)
        self.register_buffer("token_day", token_idx // cfg.n_vars)
        self.register_buffer("token_var", token_idx % cfg.n_vars)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("value_quantiles", torch.zeros(cfg.n_vars, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("residual_cholesky", torch.eye(self.path_dim))
        self.register_buffer("residual_cholesky_logdet", torch.tensor(0.0))

    def set_empirical_quantiles(
        self,
        value_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_vars, self.cfg.n_quantiles)
        if value_quantiles.shape != expected:
            raise ValueError(f"Expected value_quantiles with shape {expected}")
        self.value_quantiles.copy_(value_quantiles.to(self.value_quantiles))
        if quantile_levels is not None:
            if quantile_levels.shape != (self.cfg.n_quantiles,):
                raise ValueError("Expected quantile_levels with shape (n_quantiles,)")
            self.quantile_levels.copy_(quantile_levels.to(self.quantile_levels))
        self._quantiles_ready.fill_(True)

    def set_residual_cholesky(self, chol: torch.Tensor) -> None:
        expected = (self.path_dim, self.path_dim)
        if chol.shape != expected:
            raise ValueError(f"Expected Cholesky factor with shape {expected}")
        if not torch.allclose(chol, torch.tril(chol)):
            raise ValueError("Cholesky factor must be lower triangular")
        diag = torch.diagonal(chol)
        if torch.any(diag <= 0):
            raise ValueError("Cholesky diagonal must be positive")
        self.residual_cholesky.copy_(chol.to(self.residual_cholesky))
        self.residual_cholesky_logdet.copy_(
            torch.log(diag.to(self.residual_cholesky_logdet)).sum()
        )

    def _check_quantiles(self) -> None:
        if not bool(self._quantiles_ready.item()):
            raise RuntimeError("Empirical quantiles must be set before use")

    def values_to_scores(self, values: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        levels = self.quantile_levels.to(device=values.device, dtype=values.dtype)
        table = self.value_quantiles.to(device=values.device, dtype=values.dtype)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_vars):
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
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.value_quantiles.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for var in range(self.cfg.n_vars):
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

    @staticmethod
    def _score_features(scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        return torch.cat([scores, deltas], dim=-1)

    def conditional_params(self, history_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_scores = self.values_to_scores(history_values)
        _, hidden = self.history_encoder(self._score_features(history_scores))
        context = self.context_norm(hidden[-1])
        last_score = history_scores[:, -1, self.token_var].unsqueeze(-1)
        token_state = self.context_proj(context)[:, None, :]
        token_state = token_state + self.day_embed(self.token_day)[None, :, :]
        token_state = token_state + self.var_embed(self.token_var)[None, :, :]
        token_state = token_state + self.last_score_proj(last_score)
        raw = self.head(token_state)
        mean = raw[..., 0].view(history_values.shape[0], self.cfg.future_len, self.cfg.n_vars)
        scale = (
            F.softplus(raw[..., 1]).clamp_max(float(self.cfg.scale_max))
            + float(self.cfg.scale_floor)
        ).view_as(mean)
        return mean, scale

    def training_loss(
        self,
        history_values: torch.Tensor,
        future_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self.values_to_scores(future_values)
        mean, scale = self.conditional_params(history_values)
        residual = ((target - mean) / scale).reshape(target.shape[0], self.path_dim)
        chol = self.residual_cholesky.to(device=residual.device, dtype=residual.dtype)
        whitened = torch.linalg.solve_triangular(chol, residual.T, upper=False).T
        log_scale = torch.log(scale.reshape(scale.shape[0], self.path_dim))
        logdet = self.residual_cholesky_logdet.to(device=residual.device, dtype=residual.dtype)
        nll_per = (
            0.5 * whitened.square().sum(dim=1)
            + log_scale.sum(dim=1)
            + logdet
            + 0.5 * self.path_dim * math.log(2.0 * math.pi)
        ) / float(self.path_dim)
        loss = nll_per.mean()
        metrics = {
            "nll": loss.detach(),
            "total": loss.detach(),
            "mean_abs_err": (target - mean).abs().mean().detach(),
            "scale_mean": scale.mean().detach(),
            "target_std": target.std(unbiased=False).detach(),
            "residual_std": residual.std(unbiased=False).detach(),
            "residual_cholesky_logdet": logdet.detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample_scores_batched(
        self,
        history_values: torch.Tensor,
        n_samples: int = 50,
        chunk_size: int = 8,
        temperature: float | None = None,
    ) -> torch.Tensor:
        mean, scale = self.conditional_params(history_values)
        bsz = mean.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        mean_flat = mean.reshape(bsz, self.path_dim)
        scale_flat = scale.reshape(bsz, self.path_dim)
        chol = self.residual_cholesky.to(device=mean.device, dtype=mean.dtype)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            eps = torch.randn(bsz, k, self.path_dim, device=mean.device, dtype=mean.dtype)
            correlated = torch.matmul(eps, chol.T)
            scores = mean_flat[:, None, :] + temp * scale_flat[:, None, :] * correlated
            outs.append(scores.view(bsz, k, self.cfg.future_len, self.cfg.n_vars))
        return torch.cat(outs, dim=1)

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
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        scores = self.sample_scores_batched(
            history_values,
            n_samples=n_samples,
            chunk_size=chunk_size,
            temperature=temperature,
        )
        return self.scores_to_values(scores)


def save_checkpoint(
    path: str,
    model: CoherentPanelScorePathModel,
    epoch: int,
    best_val: float,
    panel_columns: list[str],
) -> None:
    torch.save(
        {
            "config": asdict(model.cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "panel_columns": list(panel_columns),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[CoherentPanelScorePathModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CoherentPanelScorePathConfig(**payload["config"])
    model = CoherentPanelScorePathModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
