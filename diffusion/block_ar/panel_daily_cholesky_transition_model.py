from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PanelDailyCholeskyTransitionConfig:
    history_len: int = 30
    future_len: int = 30
    n_vars: int = 51
    n_quantiles: int = 401
    cdf_eps: float = 1e-4
    history_hidden: int = 192
    context_hidden: int = 256
    encoder_layers: int = 1
    dropout: float = 0.05
    diag_floor: float = 1e-3
    diag_max: float = 5.0
    sample_temperature: float = 1.0


class PanelDailyCholeskyTransitionModel(nn.Module):
    """537a: causal daily panel transition density in empirical normal-score space."""

    def __init__(self, cfg: PanelDailyCholeskyTransitionConfig):
        super().__init__()
        self.cfg = cfg
        self.n_tril = cfg.n_vars * (cfg.n_vars + 1) // 2
        self.encoder = nn.GRU(
            input_size=2 * cfg.n_vars,
            hidden_size=cfg.history_hidden,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )
        self.context_norm = nn.LayerNorm(cfg.history_hidden)
        self.context_proj = nn.Linear(cfg.history_hidden, cfg.context_hidden)
        self.day_embed = nn.Embedding(cfg.future_len, cfg.context_hidden)
        self.var_embed = nn.Embedding(cfg.n_vars, cfg.context_hidden)
        self.current_score_proj = nn.Linear(1, cfg.context_hidden)
        self.mean_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.context_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, 1),
        )
        self.chol_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.context_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, self.n_tril),
        )
        levels = (torch.arange(cfg.n_quantiles, dtype=torch.float32) + 0.5) / float(
            cfg.n_quantiles
        )
        self.register_buffer("quantile_levels", levels)
        self.register_buffer("value_quantiles", torch.zeros(cfg.n_vars, cfg.n_quantiles))
        self.register_buffer("_quantiles_ready", torch.tensor(False, dtype=torch.bool))
        tril = torch.tril_indices(cfg.n_vars, cfg.n_vars)
        self.register_buffer("tril_row", tril[0])
        self.register_buffer("tril_col", tril[1])
        self.register_buffer("diag_index", torch.arange(cfg.n_vars))

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

    def _encode_prefix(self, prefix_scores: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self._score_features(prefix_scores))
        return self.context_norm(encoded)

    def _params_from_context(
        self,
        context: torch.Tensor,
        current_scores: torch.Tensor,
        day_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.context_proj(context) + self.day_embed(day_idx)
        var_ids = torch.arange(self.cfg.n_vars, device=context.device)
        token_state = base.unsqueeze(-2) + self.var_embed(var_ids)
        token_state = token_state + self.current_score_proj(current_scores.unsqueeze(-1))
        mean = self.mean_head(token_state).squeeze(-1)

        raw = self.chol_head(base)
        chol = torch.zeros(
            *raw.shape[:-1],
            self.cfg.n_vars,
            self.cfg.n_vars,
            device=raw.device,
            dtype=raw.dtype,
        )
        chol[..., self.tril_row, self.tril_col] = raw
        raw_diag = chol[..., self.diag_index, self.diag_index]
        diag = F.softplus(raw_diag).clamp_max(float(self.cfg.diag_max)) + float(
            self.cfg.diag_floor
        )
        chol[..., self.diag_index, self.diag_index] = diag
        return mean, chol

    def teacher_forced_params(
        self,
        history_values: torch.Tensor,
        future_prefix_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_scores = self.values_to_scores(history_values)
        if future_prefix_values is None or future_prefix_values.shape[1] == 0:
            future_prefix_scores = history_scores[:, :0]
        else:
            future_prefix_scores = self.values_to_scores(future_prefix_values)
        max_future = min(future_prefix_scores.shape[1] + 1, self.cfg.future_len)
        prefix_scores = torch.cat([history_scores, future_prefix_scores], dim=1)
        encoded = self._encode_prefix(prefix_scores)
        context = encoded[:, self.cfg.history_len - 1 : self.cfg.history_len - 1 + max_future]
        current_scores = prefix_scores[:, self.cfg.history_len - 1 : self.cfg.history_len - 1 + max_future]
        day_idx = torch.arange(max_future, device=history_values.device)
        return self._params_from_context(context, current_scores, day_idx)

    def training_loss(
        self,
        history_values: torch.Tensor,
        future_values: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        history_scores = self.values_to_scores(history_values)
        future_scores = self.values_to_scores(future_values)
        prefix_scores = torch.cat([history_scores, future_scores[:, :-1]], dim=1)
        encoded = self._encode_prefix(prefix_scores)
        context = encoded[
            :,
            self.cfg.history_len - 1 : self.cfg.history_len - 1 + self.cfg.future_len,
        ]
        current_scores = prefix_scores[
            :,
            self.cfg.history_len - 1 : self.cfg.history_len - 1 + self.cfg.future_len,
        ]
        day_idx = torch.arange(self.cfg.future_len, device=history_values.device)
        mean, chol = self._params_from_context(context, current_scores, day_idx)
        target_delta = future_scores - current_scores
        centered = target_delta - mean
        whitened = torch.linalg.solve_triangular(
            chol,
            centered.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        logdet = torch.log(diag).sum(dim=-1)
        nll_day = (
            0.5 * whitened.square().sum(dim=-1)
            + logdet
            + 0.5 * self.cfg.n_vars * math.log(2.0 * math.pi)
        )
        nll = (nll_day / float(self.cfg.n_vars)).mean()
        offdiag = chol - torch.diag_embed(diag)
        metrics = {
            "nll": nll.detach(),
            "total": nll.detach(),
            "innovation_mae": centered.abs().mean().detach(),
            "mean_abs_delta": mean.abs().mean().detach(),
            "diag_mean": diag.mean().detach(),
            "offdiag_abs": offdiag.abs().mean().detach(),
            "innovation_std": target_delta.std(unbiased=False).detach(),
            "whitened_std": whitened.std(unbiased=False).detach(),
        }
        return nll, metrics

    @torch.no_grad()
    def sample_scores_batched(
        self,
        history_values: torch.Tensor,
        n_samples: int = 50,
        chunk_size: int = 8,
        temperature: float | None = None,
    ) -> torch.Tensor:
        history_scores = self.values_to_scores(history_values)
        bsz = history_scores.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            prefix = history_scores[:, None, :, :].repeat(1, k, 1, 1)
            prefix = prefix.reshape(bsz * k, self.cfg.history_len, self.cfg.n_vars)
            future_scores = []
            for day in range(self.cfg.future_len):
                encoded = self._encode_prefix(prefix)
                context = encoded[:, -1]
                current_scores = prefix[:, -1]
                day_idx = torch.full(
                    (prefix.shape[0],),
                    day,
                    device=history_values.device,
                    dtype=torch.long,
                )
                mean, chol = self._params_from_context(context, current_scores, day_idx)
                eps = torch.randn(
                    prefix.shape[0],
                    self.cfg.n_vars,
                    1,
                    device=history_values.device,
                    dtype=history_values.dtype,
                )
                delta = mean + temp * torch.matmul(chol, eps).squeeze(-1)
                next_score = current_scores + delta
                future_scores.append(next_score)
                prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
            stacked = torch.stack(future_scores, dim=1)
            outs.append(stacked.view(bsz, k, self.cfg.future_len, self.cfg.n_vars))
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
    model: PanelDailyCholeskyTransitionModel,
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
) -> tuple[PanelDailyCholeskyTransitionModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = PanelDailyCholeskyTransitionConfig(**payload["config"])
    model = PanelDailyCholeskyTransitionModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload

