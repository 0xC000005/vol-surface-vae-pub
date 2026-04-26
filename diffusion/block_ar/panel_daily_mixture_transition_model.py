from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.panel_daily_cholesky_transition_model import (
    PanelDailyCholeskyTransitionConfig,
    PanelDailyCholeskyTransitionModel,
)


@dataclass
class PanelDailyMixtureTransitionConfig(PanelDailyCholeskyTransitionConfig):
    n_components: int = 3
    scale_floor: float = 1e-3
    scale_max: float = 5.0


class PanelDailyMixtureTransitionModel(PanelDailyCholeskyTransitionModel):
    """538a: causal daily panel transition density with non-Gaussian innovations."""

    def __init__(self, cfg: PanelDailyMixtureTransitionConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.mean_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.context_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.n_components),
        )
        self.scale_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.context_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.n_components),
        )
        self.weight_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.context_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.context_hidden, cfg.n_components),
        )

    def _params_from_context(
        self,
        context: torch.Tensor,
        current_scores: torch.Tensor,
        day_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.context_proj(context) + self.day_embed(day_idx)
        var_ids = torch.arange(self.cfg.n_vars, device=context.device)
        token_state = base.unsqueeze(-2) + self.var_embed(var_ids)
        token_state = token_state + self.current_score_proj(current_scores.unsqueeze(-1))
        mean = self.mean_head(token_state).movedim(-1, -2)
        scale = (
            F.softplus(self.scale_head(token_state)).clamp_max(float(self.cfg.scale_max))
            + float(self.cfg.scale_floor)
        ).movedim(-1, -2)
        logits = self.weight_head(base)

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
        return logits, mean, scale, chol

    def teacher_forced_params(
        self,
        history_values: torch.Tensor,
        future_prefix_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        logits, mean, scale, chol = self._params_from_context(context, current_scores, day_idx)
        target_delta = future_scores - current_scores
        residual = (target_delta.unsqueeze(-2) - mean) / scale
        chol_k = chol.unsqueeze(-3).expand(*residual.shape[:-1], self.cfg.n_vars, self.cfg.n_vars)
        whitened = torch.linalg.solve_triangular(
            chol_k,
            residual.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        corr_logdet = torch.log(diag).sum(dim=-1)
        scale_logdet = torch.log(scale).sum(dim=-1)
        log_prob = -(
            0.5 * whitened.square().sum(dim=-1)
            + scale_logdet
            + corr_logdet.unsqueeze(-1)
            + 0.5 * self.cfg.n_vars * math.log(2.0 * math.pi)
        )
        log_weight = F.log_softmax(logits, dim=-1)
        log_mix = torch.logsumexp(log_weight + log_prob, dim=-1)
        nll = -(log_mix / float(self.cfg.n_vars)).mean()
        probs = log_weight.exp()
        entropy = -(probs * log_weight).sum(dim=-1).mean()
        offdiag = chol - torch.diag_embed(diag)
        metrics = {
            "nll": nll.detach(),
            "total": nll.detach(),
            "innovation_mae": (target_delta.unsqueeze(-2) - mean).abs().mean().detach(),
            "diag_mean": diag.mean().detach(),
            "offdiag_abs": offdiag.abs().mean().detach(),
            "component_scale_mean": scale.mean().detach(),
            "innovation_std": target_delta.std(unbiased=False).detach(),
            "whitened_std": whitened.std(unbiased=False).detach(),
            "mixture_entropy": entropy.detach(),
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
                logits, mean, scale, chol = self._params_from_context(
                    context,
                    current_scores,
                    day_idx,
                )
                probs = logits.softmax(dim=-1)
                comp = torch.multinomial(probs, num_samples=1)
                gather_idx = comp.unsqueeze(-1).expand(-1, 1, self.cfg.n_vars)
                chosen_mean = mean.gather(dim=1, index=gather_idx).squeeze(1)
                chosen_scale = scale.gather(dim=1, index=gather_idx).squeeze(1)
                eps = torch.randn(
                    prefix.shape[0],
                    self.cfg.n_vars,
                    1,
                    device=history_values.device,
                    dtype=history_values.dtype,
                )
                correlated = torch.matmul(chol, eps).squeeze(-1)
                delta = chosen_mean + temp * chosen_scale * correlated
                next_score = current_scores + delta
                future_scores.append(next_score)
                prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
            stacked = torch.stack(future_scores, dim=1)
            outs.append(stacked.view(bsz, k, self.cfg.future_len, self.cfg.n_vars))
        return torch.cat(outs, dim=1)


def save_checkpoint(
    path: str,
    model: PanelDailyMixtureTransitionModel,
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
) -> tuple[PanelDailyMixtureTransitionModel, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = PanelDailyMixtureTransitionConfig(**payload["config"])
    model = PanelDailyMixtureTransitionModel(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload

