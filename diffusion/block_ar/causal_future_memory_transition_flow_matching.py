from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.logit_level_flow_matching import iv_to_logit, logit_to_iv
from diffusion.block_ar.recurrent_logit_transition_flow_matching import _time_features
from diffusion.block_ar.recurrent_logit_transition_token_flow_matching import (
    RecurrentLogitTransitionTokenFMConfig,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)


@dataclass
class CausalFutureMemoryTransitionFMConfig(RecurrentLogitTransitionTokenFMConfig):
    memory_dim: int = 128
    memory_layers: int = 3
    memory_heads: int = 4
    memory_ff: int = 256
    conditioning_mode: str = "additive"


class MemoryConditionedTokenTransitionVelocity(nn.Module):
    def __init__(self, cfg: CausalFutureMemoryTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.cell_embed = nn.Embedding(cfg.n_cells, cfg.token_dim)
        self.memory_proj = nn.Linear(cfg.memory_dim, cfg.token_dim)
        self.time_proj = nn.Linear(cfg.time_dim, cfg.token_dim)
        self.value_proj = nn.Linear(2, cfg.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.token_heads,
            dim_feedforward=cfg.token_ff,
            dropout=cfg.model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=cfg.token_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            nn.Linear(cfg.token_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_cells = x_t.shape
        cell_ids = torch.arange(n_cells, device=x_t.device)
        cell = self.cell_embed(cell_ids)[None, :, :].expand(bsz, -1, -1)
        values = torch.stack([x_t, current_logit], dim=-1)
        token = self.value_proj(values) + cell
        memory = self.memory_proj(memory_state)[:, None, :]
        time = self.time_proj(_time_features(t, self.cfg.time_dim))[:, None, :]
        if self.cfg.conditioning_mode == "additive":
            hidden = token + memory + time
            hidden = self.mixer(hidden)
        elif self.cfg.conditioning_mode == "prefix":
            hidden = torch.cat([memory, time, token], dim=1)
            hidden = self.mixer(hidden)[:, 2:]
        else:
            raise ValueError(f"Unknown conditioning_mode={self.cfg.conditioning_mode!r}")
        return self.out(hidden).squeeze(-1)


class CausalFutureMemoryTransitionFlowMatching(nn.Module):
    """330a: autoregressive logit-transition flow with causal future-prefix memory."""

    def __init__(self, cfg: CausalFutureMemoryTransitionFMConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_proj = nn.Linear(2 * cfg.n_cells, cfg.memory_dim)
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

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.view(x.shape[0], x.shape[1], -1)
        return x

    def _logit_features_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        deltas = torch.zeros_like(logits)
        deltas[:, 1:] = logits[:, 1:] - logits[:, :-1]
        return torch.cat([logits, deltas], dim=-1)

    def _encode_prefix_logits(self, prefix_logits: torch.Tensor) -> torch.Tensor:
        seq_len = prefix_logits.shape[1]
        if seq_len > self.cfg.history_len + self.cfg.future_len:
            raise ValueError(
                f"Prefix length {seq_len} exceeds configured max "
                f"{self.cfg.history_len + self.cfg.future_len}"
            )
        pos = torch.arange(seq_len, device=prefix_logits.device)
        x = self.feature_proj(self._logit_features_from_logits(prefix_logits))
        x = x + self.pos_embed(pos)[None, :, :]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=prefix_logits.device, dtype=torch.bool),
            diagonal=1,
        )
        return self.memory_norm(self.memory(x, mask=mask))

    def target_future_logits(self, future_norm: torch.Tensor) -> torch.Tensor:
        future_norm = self._flatten(future_norm)
        return iv_to_logit(denormalize_iv(future_norm), self.cfg.logit_eps)

    def history_logits(self, history_norm: torch.Tensor) -> torch.Tensor:
        history_norm = self._flatten(history_norm)
        return iv_to_logit(denormalize_iv(history_norm), self.cfg.logit_eps)

    def teacher_forced_memory(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_logits = self.history_logits(history_norm)
        future_logits = self.target_future_logits(future_norm)
        prefix_logits = torch.cat([history_logits, future_logits[:, :-1]], dim=1)
        hidden = self._encode_prefix_logits(prefix_logits)
        start = self.cfg.history_len - 1
        memory_states = hidden[:, start : start + self.cfg.future_len]
        current_logits = prefix_logits[:, start : start + self.cfg.future_len]
        return memory_states, current_logits, future_logits

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        current_logit: torch.Tensor,
        memory_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(x_t, current_logit, memory_state, t)

    def training_loss(
        self,
        history_norm: torch.Tensor,
        future_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        memory_states, current_logits, future_logits = self.teacher_forced_memory(
            history_norm,
            future_norm,
        )
        x1 = future_logits - current_logits
        x0 = torch.randn_like(x1)
        bsz, horizon, n_cells = x1.shape
        t = torch.rand(bsz, horizon, device=x1.device, dtype=x1.dtype)
        x_t = (1.0 - t[..., None]) * x0 + t[..., None] * x1
        target_velocity = x1 - x0
        pred_velocity = self.predict_velocity(
            x_t.reshape(bsz * horizon, n_cells),
            current_logits.reshape(bsz * horizon, n_cells),
            memory_states.reshape(bsz * horizon, self.cfg.memory_dim),
            t.reshape(bsz * horizon),
        ).view_as(x1)
        fm_loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "total": fm_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "transition_std": x1.std(unbiased=False).detach(),
            "transition_abs": x1.abs().mean().detach(),
            "target_velocity_std": target_velocity.std(unbiased=False).detach(),
            "memory_abs": memory_states.abs().mean().detach(),
        }
        return fm_loss, metrics

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
        if n_steps < 1 or n_steps > self.cfg.future_len:
            raise ValueError(f"Expected n_steps in [1,{self.cfg.future_len}], got {n_steps}")
        history_norm = history if history_is_normalized else normalize_iv(history)
        history_logits = self.history_logits(history_norm)
        bsz = history_logits.shape[0]
        chunk_size = max(1, min(int(chunk_size), int(n_samples)))
        temp = float(self.cfg.sample_temperature if temperature is None else temperature)
        dt = 1.0 / float(self.cfg.flow_steps)

        outs: list[torch.Tensor] = []
        for start in range(0, n_samples, chunk_size):
            k = min(chunk_size, n_samples - start)
            prefix = (
                history_logits.unsqueeze(1)
                .expand(bsz, k, self.cfg.history_len, self.cfg.n_cells)
                .reshape(bsz * k, self.cfg.history_len, self.cfg.n_cells)
                .clone()
            )
            frames: list[torch.Tensor] = []
            for _step in range(n_steps):
                memory_state = self._encode_prefix_logits(prefix)[:, -1]
                current_logit = prefix[:, -1]
                x = temp * torch.randn_like(current_logit)
                for flow_step in range(self.cfg.flow_steps):
                    t = torch.full(
                        (bsz * k,),
                        (flow_step + 0.5) * dt,
                        device=history_logits.device,
                        dtype=history_logits.dtype,
                    )
                    x = x + dt * self.predict_velocity(
                        x,
                        current_logit,
                        memory_state,
                        t,
                    )
                next_logit = current_logit + x
                next_iv = logit_to_iv(next_logit)
                frames.append(next_iv.view(bsz, k, 5, 5))
                prefix = torch.cat([prefix, next_logit[:, None, :]], dim=1)
            outs.append(torch.stack(frames, dim=2))
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def sample_next_iv(
        self,
        history_01: torch.Tensor,
        n_samples: int,
        **kwargs: object,
    ) -> torch.Tensor:
        history_norm = normalize_iv(history_01)
        return self.sample_batched(
            history_norm,
            n_samples=n_samples,
            n_steps=1,
            history_is_normalized=True,
            **kwargs,
        )[:, :, 0]


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[CausalFutureMemoryTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CausalFutureMemoryTransitionFMConfig(**payload["config"])
    model = CausalFutureMemoryTransitionFlowMatching(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload


def save_checkpoint(
    path: str,
    model: CausalFutureMemoryTransitionFlowMatching,
    cfg: CausalFutureMemoryTransitionFMConfig,
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
