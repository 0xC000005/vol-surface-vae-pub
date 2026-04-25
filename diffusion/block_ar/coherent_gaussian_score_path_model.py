from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class CoherentGaussianScorePathConfig:
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    history_hidden: int = 192
    token_hidden: int = 256
    encoder_layers: int = 1
    dropout: float = 0.05
    scale_floor: float = 1e-3
    scale_max: float = 5.0
    sample_temperature: float = 1.0


class CoherentGaussianScorePathModel(nn.Module):
    """532a: one-shot conditional Gaussian path density in empirical score space."""

    def __init__(self, cfg: CoherentGaussianScorePathConfig):
        super().__init__()
        self.cfg = cfg
        self.path_dim = int(cfg.future_len * cfg.n_cells)
        self.history_encoder = nn.GRU(
            input_size=2 * cfg.n_cells,
            hidden_size=cfg.history_hidden,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )
        self.context_norm = nn.LayerNorm(cfg.history_hidden)
        self.context_proj = nn.Linear(cfg.history_hidden, cfg.token_hidden)
        self.day_embed = nn.Embedding(cfg.future_len, cfg.token_hidden)
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_hidden)
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
        self.register_buffer("token_day", token_idx // cfg.n_cells)
        self.register_buffer("token_cell", token_idx % cfg.n_cells)
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("level_quantiles", torch.zeros(cfg.n_cells, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("residual_cholesky", torch.eye(self.path_dim))
        self.register_buffer("residual_cholesky_logdet", torch.tensor(0.0))

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def set_empirical_quantiles(
        self,
        level_quantiles: torch.Tensor,
        quantile_levels: torch.Tensor | None = None,
    ) -> None:
        expected = (self.cfg.n_cells, self.cfg.n_quantiles)
        if level_quantiles.shape != expected:
            raise ValueError(f"Expected level_quantiles with shape {expected}")
        self.level_quantiles.copy_(level_quantiles.to(self.level_quantiles))
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

    def _values_to_scores(self, values_01: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        values_01 = self._flatten(values_01)
        levels = self.quantile_levels.to(device=values_01.device, dtype=values_01.dtype)
        table = self.level_quantiles.to(device=values_01.device, dtype=values_01.dtype)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = values_01[..., cell].reshape(-1)
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
            cols.append(score.view(values_01.shape[:-1]))
        return torch.stack(cols, dim=-1)

    def _scores_to_values(self, scores: torch.Tensor) -> torch.Tensor:
        self._check_quantiles()
        levels = self.quantile_levels.to(device=scores.device, dtype=scores.dtype)
        table = self.level_quantiles.to(device=scores.device, dtype=scores.dtype)
        eps = float(self.cfg.cdf_eps)
        u_all = (0.5 * (1.0 + torch.erf(scores / math.sqrt(2.0)))).clamp(eps, 1.0 - eps)
        cols: list[torch.Tensor] = []
        for cell in range(self.cfg.n_cells):
            q = table[cell]
            flat = u_all[..., cell].reshape(-1)
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
        return torch.stack(cols, dim=-1).clamp(0.0, 1.0)

    def history_scores(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_01 = denormalize_iv(self._flatten(history_norm))
        return self._values_to_scores(history_01)

    def target_future_scores(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_01 = denormalize_iv(self._flatten(future_norm))
        return self._values_to_scores(future_01)

    @staticmethod
    def _score_features(scores: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(scores)
        deltas[:, 1:] = scores[:, 1:] - scores[:, :-1]
        return torch.cat([scores, deltas], dim=-1)

    def conditional_params(self, history_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history_scores = self.history_scores(history_norm)
        _, hidden = self.history_encoder(self._score_features(history_scores))
        context = self.context_norm(hidden[-1])
        last_score = history_scores[:, -1, self.token_cell].unsqueeze(-1)
        token_state = self.context_proj(context)[:, None, :]
        token_state = token_state + self.day_embed(self.token_day)[None, :, :]
        token_state = token_state + self.cell_embed(self.token_cell)[None, :, :]
        token_state = token_state + self.last_score_proj(last_score)
        raw = self.head(token_state)
        mean = raw[..., 0].view(history_scores.shape[0], self.cfg.future_len, self.cfg.n_cells)
        scale = (
            F.softplus(raw[..., 1]).clamp_max(float(self.cfg.scale_max))
            + float(self.cfg.scale_floor)
        ).view_as(mean)
        return mean, scale

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self.target_future_scores(future_norm)
        mean, scale = self.conditional_params(history_norm)
        residual = ((target - mean) / scale).reshape(target.shape[0], self.path_dim)
        chol = self.residual_cholesky.to(device=residual.device, dtype=residual.dtype)
        whitened = torch.linalg.solve_triangular(
            chol,
            residual.T,
            upper=False,
        ).T
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
            "mean_abs_err": (target - mean).abs().mean().detach(),
            "scale_mean": scale.mean().detach(),
            "target_std": target.std(unbiased=False).detach(),
            "residual_std": residual.std(unbiased=False).detach(),
            "residual_cholesky_logdet": logdet.detach(),
            "total": loss.detach(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample_scores_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        temperature: float | None = None,
    ) -> torch.Tensor:
        history_norm = history if history_is_normalized else normalize_iv(history)
        mean, scale = self.conditional_params(history_norm)
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
            outs.append(scores.view(bsz, k, self.cfg.future_len, self.cfg.n_cells))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        temperature: float | None = None,
        **_: object,
    ) -> torch.Tensor:
        if n_steps != self.cfg.future_len:
            raise ValueError(f"Expected n_steps={self.cfg.future_len}, got {n_steps}")
        scores = self.sample_scores_batched(
            history,
            n_samples=n_samples,
            chunk_size=chunk_size,
            history_is_normalized=history_is_normalized,
            temperature=temperature,
        )
        future_01 = self._scores_to_values(scores)
        if self.cfg.n_cells == 25:
            return future_01.view(history.shape[0], n_samples, self.cfg.future_len, 5, 5)
        return future_01


def save_checkpoint(
    path: str,
    model: CoherentGaussianScorePathModel,
    epoch: int,
    best_val: float,
) -> None:
    torch.save(
        {
            "config": asdict(model.cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[CoherentGaussianScorePathModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CoherentGaussianScorePathConfig(**payload["config"])
    model = CoherentGaussianScorePathModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload
