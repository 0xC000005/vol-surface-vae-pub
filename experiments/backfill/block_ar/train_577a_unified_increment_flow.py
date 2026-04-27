#!/usr/bin/env python
"""577a: train a minimal shared flow over unified IV+factor increments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    UnifiedIncrementBlock,
    build_unified_increment_block,
    clean_nonpositive_log_level_factors,
    decode_state,
    encode_state,
)


@dataclass(frozen=True)
class UnifiedIncrementFlowConfig:
    history_len: int
    future_len: int
    n_vars: int
    hidden_dim: int = 256
    time_embed_dim: int = 32
    depth: int = 4
    dropout: float = 0.05
    source_mode: str = "independent"
    conditional_source_topk: int = 64
    source_loc_clip: float = 5.0
    source_log_scale_min: float = -3.0
    source_log_scale_max: float = 2.0
    source_latent_dim: int = 32


class UnifiedIncrementFlow(nn.Module):
    """One shared conditional flow for the full future increment tensor."""

    def __init__(self, cfg: UnifiedIncrementFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.path_dim = int(cfg.future_len * cfg.n_vars)
        self.history_encoder = nn.GRU(
            input_size=cfg.n_vars,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, cfg.time_embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
            nn.SiLU(),
        )
        layers: list[nn.Module] = []
        in_dim = self.path_dim + cfg.hidden_dim + cfg.time_embed_dim
        for layer_idx in range(int(cfg.depth)):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else cfg.hidden_dim, cfg.hidden_dim))
            layers.append(nn.SiLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(cfg.hidden_dim, self.path_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("source_mean", torch.zeros(self.path_dim), persistent=True)
        self.register_buffer("source_cholesky", torch.eye(self.path_dim), persistent=True)
        self.register_buffer("source_bank", torch.empty(0, self.path_dim), persistent=True)
        self.register_buffer("source_keys", torch.empty(0, cfg.n_vars), persistent=True)
        self.source_temperature = 1.0
        if cfg.source_mode == "conditional_affine":
            self.source_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, 2 * self.path_dim),
            )
            final = self.source_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.source_head = None
        if cfg.source_mode == "conditional_latent":
            self.latent_param_head = nn.Linear(cfg.hidden_dim, 2 * int(cfg.source_latent_dim))
            self.latent_path_loc = nn.Linear(cfg.hidden_dim, self.path_dim)
            self.latent_path_decoder = nn.Sequential(
                nn.Linear(cfg.hidden_dim + int(cfg.source_latent_dim), cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, self.path_dim),
            )
            nn.init.zeros_(self.latent_param_head.weight)
            nn.init.zeros_(self.latent_param_head.bias)
            nn.init.zeros_(self.latent_path_loc.weight)
            nn.init.zeros_(self.latent_path_loc.bias)
        else:
            self.latent_param_head = None
            self.latent_path_loc = None
            self.latent_path_decoder = None

    def set_source_gaussian(self, mean: torch.Tensor, cholesky: torch.Tensor) -> None:
        if mean.shape != (self.path_dim,):
            raise ValueError(f"mean shape must be ({self.path_dim},), got {tuple(mean.shape)}")
        if cholesky.shape != (self.path_dim, self.path_dim):
            raise ValueError(
                f"cholesky shape must be ({self.path_dim}, {self.path_dim}), got {tuple(cholesky.shape)}"
            )
        self.source_mean.copy_(mean.to(device=self.source_mean.device, dtype=self.source_mean.dtype))
        self.source_cholesky.copy_(cholesky.to(device=self.source_cholesky.device, dtype=self.source_cholesky.dtype))

    def set_source_bank(self, bank: torch.Tensor, keys: torch.Tensor | None = None) -> None:
        if bank.ndim != 2 or bank.shape[1] != self.path_dim:
            raise ValueError(f"bank shape must be (n, {self.path_dim}), got {tuple(bank.shape)}")
        self.source_bank = bank.to(device=self.source_mean.device, dtype=self.source_mean.dtype).contiguous()
        if keys is not None:
            if keys.ndim != 2 or keys.shape[0] != bank.shape[0] or keys.shape[1] != self.cfg.n_vars:
                raise ValueError(f"keys shape must be ({bank.shape[0]}, {self.cfg.n_vars}), got {tuple(keys.shape)}")
            self.source_keys = keys.to(device=self.source_mean.device, dtype=self.source_mean.dtype).contiguous()

    def _history_context(self, history: torch.Tensor) -> torch.Tensor:
        _, h_n = self.history_encoder(history)
        return h_n[-1]

    def conditional_source_params(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.source_head is None:
            raise RuntimeError("conditional_source_params requires source_mode='conditional_affine'")
        context = self._history_context(history)
        loc_raw, log_scale_raw = self.source_head(context).chunk(2, dim=-1)
        loc_clip = float(max(self.cfg.source_loc_clip, 1e-6))
        loc = loc_clip * torch.tanh(loc_raw / loc_clip)
        log_scale = torch.clamp(
            log_scale_raw,
            min=float(self.cfg.source_log_scale_min),
            max=float(self.cfg.source_log_scale_max),
        )
        return loc, log_scale

    def _draw_conditional_affine_flat(
        self,
        history: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc, log_scale = self.conditional_source_params(history.to(device=device, dtype=dtype))
        eps = torch.randn(loc.shape, device=device, dtype=dtype)
        flat = loc + float(self.source_temperature) * eps * torch.exp(log_scale)
        return flat, loc, log_scale

    def conditional_latent_params(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.latent_param_head is None or self.latent_path_loc is None:
            raise RuntimeError("conditional_latent_params requires source_mode='conditional_latent'")
        context = self._history_context(history)
        latent_loc, latent_log_scale = self.latent_param_head(context).chunk(2, dim=-1)
        latent_log_scale = torch.clamp(
            latent_log_scale,
            min=float(self.cfg.source_log_scale_min),
            max=float(self.cfg.source_log_scale_max),
        )
        path_loc = self.latent_path_loc(context)
        return path_loc, latent_loc, latent_log_scale

    def _draw_conditional_latent_flat(
        self,
        history: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.latent_path_decoder is None:
            raise RuntimeError("conditional_latent source requires latent_path_decoder")
        context = self._history_context(history.to(device=device, dtype=dtype))
        latent_loc, latent_log_scale = self.latent_param_head(context).chunk(2, dim=-1)
        latent_log_scale = torch.clamp(
            latent_log_scale,
            min=float(self.cfg.source_log_scale_min),
            max=float(self.cfg.source_log_scale_max),
        )
        z = latent_loc + torch.randn_like(latent_loc) * torch.exp(latent_log_scale)
        path_loc = self.latent_path_loc(context)
        residual = self.latent_path_decoder(torch.cat([context, z], dim=-1))
        return path_loc + float(self.source_temperature) * residual

    def draw_source(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cfg.source_mode == "conditional_affine":
            if history is None:
                raise RuntimeError("conditional_affine source requires history")
            flat, _loc, _log_scale = self._draw_conditional_affine_flat(
                history,
                device=device,
                dtype=dtype,
            )
            if flat.shape[0] != int(batch_size):
                raise ValueError(f"history batch {flat.shape[0]} does not match batch_size {batch_size}")
            return flat.reshape(int(batch_size), self.cfg.future_len, self.cfg.n_vars)
        if self.cfg.source_mode == "conditional_latent":
            if history is None:
                raise RuntimeError("conditional_latent source requires history")
            flat = self._draw_conditional_latent_flat(
                history,
                device=device,
                dtype=dtype,
            )
            if flat.shape[0] != int(batch_size):
                raise ValueError(f"history batch {flat.shape[0]} does not match batch_size {batch_size}")
            return flat.reshape(int(batch_size), self.cfg.future_len, self.cfg.n_vars)
        if self.cfg.source_mode in {"empirical_path", "empirical_conditional"}:
            if self.source_bank.numel() == 0:
                raise RuntimeError(f"{self.cfg.source_mode} source requires a non-empty source_bank")
            if self.cfg.source_mode == "empirical_conditional":
                if history is None:
                    raise RuntimeError("empirical_conditional source requires history")
                if self.source_keys.numel() == 0:
                    raise RuntimeError("empirical_conditional source requires source_keys")
                query = history[:, -1, :].to(device=device, dtype=dtype)
                keys = self.source_keys.to(device=device, dtype=dtype)
                distance = torch.cdist(query, keys)
                k = min(int(self.cfg.conditional_source_topk), int(keys.shape[0]))
                nearest = torch.topk(distance, k=k, largest=False).indices
                pick = torch.randint(0, k, (int(batch_size),), device=device)
                indices = nearest[torch.arange(int(batch_size), device=device), pick]
            else:
                indices = torch.randint(
                    0,
                    self.source_bank.shape[0],
                    (int(batch_size),),
                    device=device,
                )
            flat = self.source_bank.to(device=device, dtype=dtype).index_select(0, indices)
            return flat.reshape(int(batch_size), self.cfg.future_len, self.cfg.n_vars)
        eps = torch.randn(int(batch_size), self.path_dim, device=device, dtype=dtype)
        flat = self.source_mean.to(device=device, dtype=dtype) + float(self.source_temperature) * (eps @ self.source_cholesky.to(
            device=device,
            dtype=dtype,
        ).T)
        return flat.reshape(int(batch_size), self.cfg.future_len, self.cfg.n_vars)

    def forward(self, history: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or x_t.ndim != 3:
            raise ValueError("history and x_t must have shape (batch, time, vars)")
        context = self._history_context(history)
        t_embed = self.time_embed(t.reshape(-1, 1))
        flat_x = x_t.reshape(x_t.shape[0], -1)
        velocity = self.net(torch.cat([flat_x, context, t_embed], dim=1))
        return velocity.reshape_as(x_t)

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        *,
        fm_loss_mode: str = "velocity",
        source_prior_nll_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_prior_nll = target.new_tensor(0.0)
        if self.cfg.source_mode == "conditional_affine":
            flat_noise, loc, log_scale = self._draw_conditional_affine_flat(
                history,
                device=target.device,
                dtype=target.dtype,
            )
            noise = flat_noise.reshape_as(target)
            target_flat = target.reshape(target.shape[0], -1)
            inv_scale = torch.exp(-log_scale)
            source_prior_nll = 0.5 * (
                ((target_flat - loc) * inv_scale).square()
                + 2.0 * log_scale
                + float(np.log(2.0 * np.pi))
            ).mean()
        else:
            noise = self.draw_source(target.shape[0], device=target.device, dtype=target.dtype, history=history)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        shape = (target.shape[0],) + (1,) * (target.ndim - 1)
        x_t = (1.0 - t.reshape(shape)) * noise + t.reshape(shape) * target
        target_velocity = target - noise
        pred_velocity = self.forward(history, x_t, t)
        velocity_residual = pred_velocity - target_velocity
        velocity_mse = torch.mean(velocity_residual.square())
        cumulative_mse = torch.mean(torch.cumsum(velocity_residual, dim=1).square())
        if fm_loss_mode == "velocity":
            fm_loss = velocity_mse
        elif fm_loss_mode == "cumulative_state":
            fm_loss = cumulative_mse
        else:
            raise ValueError(f"unknown fm_loss_mode {fm_loss_mode}")
        loss = fm_loss + float(source_prior_nll_weight) * source_prior_nll
        return loss, {
            "loss": loss.detach(),
            "fm_loss": fm_loss.detach(),
            "velocity_mse": velocity_mse.detach(),
            "cumulative_mse": cumulative_mse.detach(),
            "source_prior_nll": source_prior_nll.detach(),
            "target_std": target.detach().std(),
            "velocity_std": target_velocity.detach().std(),
            "pred_velocity_std": pred_velocity.detach().std(),
        }

    @torch.no_grad()
    def sample(self, history: torch.Tensor, *, n_samples: int, n_steps: int) -> torch.Tensor:
        self.eval()
        bsz = history.shape[0]
        repeated_history = history.repeat_interleave(int(n_samples), dim=0)
        x = self.draw_source(
            bsz * int(n_samples),
            device=history.device,
            dtype=history.dtype,
            history=repeated_history,
        )
        dt = 1.0 / float(n_steps)
        for step in range(int(n_steps)):
            t_value = torch.full((x.shape[0],), (step + 0.5) * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.forward(repeated_history, x, t_value)
        return x.reshape(bsz, int(n_samples), self.cfg.future_len, self.cfg.n_vars)


def _fit_mean_std(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def fit_path_gaussian(
    standardized_increment: np.ndarray,
    *,
    shrinkage: float,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(standardized_increment, dtype=np.float64).reshape(
        standardized_increment.shape[0],
        -1,
    )
    mean = flat.mean(axis=0)
    centered = flat - mean
    cov = (centered.T @ centered) / max(flat.shape[0] - 1, 1)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(np.diag(cov))
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    cov = (1.0 - shrink) * cov + shrink * diag
    eye = np.eye(cov.shape[0], dtype=np.float64)
    for scale in [1.0, 3.0, 10.0, 30.0, 100.0]:
        try:
            chol = np.linalg.cholesky(cov + float(jitter) * scale * eye)
            return mean.astype(np.float32), chol.astype(np.float32)
        except np.linalg.LinAlgError:
            continue
    chol = np.linalg.cholesky(cov + max(float(jitter), 1e-3) * 1000.0 * eye)
    return mean.astype(np.float32), chol.astype(np.float32)


def _standardize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(array, dtype=np.float32) - mean) / std).astype(np.float32)


def _normal_score_levels(n_quantiles: int, cdf_eps: float) -> tuple[np.ndarray, np.ndarray]:
    levels = np.linspace(float(cdf_eps), 1.0 - float(cdf_eps), int(n_quantiles), dtype=np.float64)
    normal = torch.distributions.Normal(0.0, 1.0).icdf(torch.from_numpy(levels)).numpy()
    return levels.astype(np.float32), normal.astype(np.float32)


def fit_increment_normal_score(
    increments: np.ndarray,
    *,
    n_quantiles: int,
    cdf_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels, normal_levels = _normal_score_levels(n_quantiles, cdf_eps)
    flat = np.asarray(increments, dtype=np.float64).reshape(-1, increments.shape[-1])
    quantiles = np.quantile(flat, levels.astype(np.float64), axis=0).astype(np.float64)
    for col in range(quantiles.shape[1]):
        quantiles[:, col] = np.maximum.accumulate(quantiles[:, col])
        for row in range(1, quantiles.shape[0]):
            if quantiles[row, col] <= quantiles[row - 1, col]:
                quantiles[row, col] = quantiles[row - 1, col] + 1e-8
    return quantiles.astype(np.float32), levels, normal_levels


def normal_score_transform(
    increments: np.ndarray,
    quantiles: np.ndarray,
    normal_levels: np.ndarray,
) -> np.ndarray:
    values = np.asarray(increments, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty_like(flat, dtype=np.float64)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(
            flat[:, col],
            quantiles[:, col],
            normal_levels,
            left=normal_levels[0],
            right=normal_levels[-1],
        )
    return out.reshape(values.shape).astype(np.float32)


def normal_score_inverse(
    scores: np.ndarray,
    quantiles: np.ndarray,
    normal_levels: np.ndarray,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty_like(flat, dtype=np.float64)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(
            flat[:, col],
            normal_levels,
            quantiles[:, col],
            left=quantiles[0, col],
            right=quantiles[-1, col],
        )
    return out.reshape(values.shape).astype(np.float32)


def _make_train_val_blocks(args: argparse.Namespace) -> tuple[np.ndarray, list[str], UnifiedIncrementBlock, UnifiedIncrementBlock]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    if getattr(args, "clean_nonpositive_log_levels", False):
        panel, cleaning_report = clean_nonpositive_log_level_factors(panel, columns, iv_count=args.iv_count)
        print(json.dumps({"panel_cleaning": cleaning_report}, indent=2))
    train_indices, val_indices = official_train_val_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    if args.max_train_windows > 0:
        train_indices = train_indices[-int(args.max_train_windows) :]
    train_block = build_unified_increment_block(
        panel,
        columns,
        train_indices,
        history_len=args.history_len,
        future_len=args.future_len,
        iv_count=args.iv_count,
    )
    val_block = build_unified_increment_block(
        panel,
        columns,
        val_indices,
        history_len=args.history_len,
        future_len=args.future_len,
        iv_count=args.iv_count,
    )
    return panel, columns, train_block, val_block


def _reconstruct_samples(
    history_state: np.ndarray,
    transformed_samples: np.ndarray,
    *,
    increment_transform: str,
    inc_mean: np.ndarray,
    inc_std: np.ndarray,
    inc_quantiles: np.ndarray | None,
    normal_levels: np.ndarray | None,
    specs: list,
) -> np.ndarray:
    if increment_transform == "standard":
        samples = np.asarray(transformed_samples, dtype=np.float64) * inc_std + inc_mean
    elif increment_transform == "normal_score":
        if inc_quantiles is None or normal_levels is None:
            raise ValueError("normal_score inverse requires quantiles and normal_levels")
        samples = normal_score_inverse(transformed_samples, inc_quantiles, normal_levels).astype(np.float64)
    else:
        raise ValueError(f"unknown increment_transform {increment_transform}")
    history_encoded = encode_state(np.asarray(history_state, dtype=np.float64), specs)
    last_encoded = history_encoded[:, -1, :][:, None, None, :]
    future_encoded = last_encoded + np.cumsum(samples, axis=2)
    return decode_state(future_encoded, specs).astype(np.float32)


def _sample_audit(
    model: UnifiedIncrementFlow,
    val_hist_std: torch.Tensor,
    val_block: UnifiedIncrementBlock,
    *,
    hist_mean: np.ndarray,
    hist_std: np.ndarray,
    inc_mean: np.ndarray,
    inc_std: np.ndarray,
    increment_transform: str,
    inc_quantiles: np.ndarray | None,
    normal_levels: np.ndarray | None,
    sample_windows: int,
    n_samples: int,
    n_steps: int,
) -> dict[str, Any]:
    n = min(int(sample_windows), int(val_hist_std.shape[0]))
    samples_std = model.sample(val_hist_std[:n], n_samples=n_samples, n_steps=n_steps)
    samples_np = samples_std.detach().cpu().numpy()
    samples_state = _reconstruct_samples(
        val_block.history_state[:n],
        samples_np,
        increment_transform=increment_transform,
        inc_mean=inc_mean,
        inc_std=inc_std,
        inc_quantiles=inc_quantiles,
        normal_levels=normal_levels,
        specs=val_block.specs,
    )
    if increment_transform == "standard":
        target_inc_transformed = _standardize(val_block.future_increment[:n], inc_mean, inc_std)
    else:
        if inc_quantiles is None or normal_levels is None:
            raise ValueError("normal_score transform requires quantiles and normal_levels")
        target_inc_transformed = normal_score_transform(
            val_block.future_increment[:n],
            inc_quantiles,
            normal_levels,
        )
    sample_inc_std = samples_np.reshape(-1, samples_np.shape[-2], samples_np.shape[-1])
    return {
        "sample_windows": int(n),
        "n_samples": int(n_samples),
        "n_steps": int(n_steps),
        "finite_sample_state_rate": float(np.isfinite(samples_state).mean()),
        "finite_sample_increment_rate": float(np.isfinite(samples_np).mean()),
        "standardized_increment_std_ratio": float(
            np.std(sample_inc_std) / max(float(np.std(target_inc_transformed)), 1e-12)
        ),
        "iv_min": float(np.nanmin(samples_state[..., :25])),
        "iv_max": float(np.nanmax(samples_state[..., :25])),
        "factor_min": float(np.nanmin(samples_state[..., 25:])),
        "factor_max": float(np.nanmax(samples_state[..., 25:])),
        "history_mean_abs": float(np.mean(np.abs(hist_mean))),
        "history_std_min": float(np.min(hist_std)),
        "increment_std_min": float(np.min(inc_std)),
    }


def save_checkpoint(
    path: Path,
    model: UnifiedIncrementFlow,
    *,
    epoch: int,
    best_val: float,
    hist_mean: np.ndarray,
    hist_std: np.ndarray,
    inc_mean: np.ndarray,
    inc_std: np.ndarray,
    increment_transform: str,
    inc_quantiles: np.ndarray | None,
    normal_levels: np.ndarray | None,
    specs: list,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.cfg),
            "epoch": int(epoch),
            "best_val": float(best_val),
            "model_state_dict": model.state_dict(),
            "history_mean": hist_mean.astype(np.float32),
            "history_std": hist_std.astype(np.float32),
            "increment_mean": inc_mean.astype(np.float32),
            "increment_std": inc_std.astype(np.float32),
            "increment_transform": increment_transform,
            "increment_quantiles": None if inc_quantiles is None else inc_quantiles.astype(np.float32),
            "normal_score_levels": None if normal_levels is None else normal_levels.astype(np.float32),
            "state_variables": [asdict(spec) for spec in specs],
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true")
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_embed_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--source_mode",
        choices=[
            "independent",
            "path_gaussian",
            "empirical_path",
            "empirical_conditional",
            "conditional_affine",
            "conditional_latent",
        ],
        default="independent",
    )
    parser.add_argument("--conditional_source_topk", type=int, default=64)
    parser.add_argument("--source_prior_nll_weight", type=float, default=0.0)
    parser.add_argument("--fm_loss_mode", choices=["velocity", "cumulative_state"], default="velocity")
    parser.add_argument("--source_loc_clip", type=float, default=5.0)
    parser.add_argument("--source_log_scale_min", type=float, default=-3.0)
    parser.add_argument("--source_log_scale_max", type=float, default=2.0)
    parser.add_argument("--source_latent_dim", type=int, default=32)
    parser.add_argument("--source_cov_shrinkage", type=float, default=0.05)
    parser.add_argument("--source_cov_jitter", type=float, default=1e-4)
    parser.add_argument("--increment_transform", choices=["standard", "normal_score"], default="standard")
    parser.add_argument("--n_increment_quantiles", type=int, default=501)
    parser.add_argument("--increment_cdf_eps", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--sample_windows", type=int, default=96)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=577)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    _panel, _columns, train_block, val_block = _make_train_val_blocks(args)
    hist_mean, hist_std = _fit_mean_std(train_block.history_state)
    inc_mean, inc_std = _fit_mean_std(train_block.future_increment)
    inc_quantiles = None
    normal_levels = None
    train_hist = torch.from_numpy(_standardize(train_block.history_state, hist_mean, hist_std)).to(device)
    if args.increment_transform == "standard":
        train_inc_np = _standardize(train_block.future_increment, inc_mean, inc_std)
        val_inc_np = _standardize(val_block.future_increment, inc_mean, inc_std)
    else:
        inc_quantiles, _levels, normal_levels = fit_increment_normal_score(
            train_block.future_increment,
            n_quantiles=args.n_increment_quantiles,
            cdf_eps=args.increment_cdf_eps,
        )
        train_inc_np = normal_score_transform(train_block.future_increment, inc_quantiles, normal_levels)
        val_inc_np = normal_score_transform(val_block.future_increment, inc_quantiles, normal_levels)
        inc_mean = np.zeros_like(inc_mean, dtype=np.float32)
        inc_std = np.ones_like(inc_std, dtype=np.float32)
    train_inc = torch.from_numpy(train_inc_np).to(device)
    val_hist = torch.from_numpy(_standardize(val_block.history_state, hist_mean, hist_std)).to(device)
    val_inc = torch.from_numpy(val_inc_np).to(device)

    cfg = UnifiedIncrementFlowConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_vars=train_inc.shape[-1],
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        depth=args.depth,
        dropout=args.dropout,
        source_mode=args.source_mode,
        conditional_source_topk=args.conditional_source_topk,
        source_loc_clip=args.source_loc_clip,
        source_log_scale_min=args.source_log_scale_min,
        source_log_scale_max=args.source_log_scale_max,
        source_latent_dim=args.source_latent_dim,
    )
    model = UnifiedIncrementFlow(cfg).to(device)
    source_diagnostics: dict[str, Any] = {"source_mode": args.source_mode}
    source_diagnostics["fm_loss_mode"] = args.fm_loss_mode
    if args.source_mode == "path_gaussian":
        source_mean, source_chol = fit_path_gaussian(
            train_inc.detach().cpu().numpy(),
            shrinkage=args.source_cov_shrinkage,
            jitter=args.source_cov_jitter,
        )
        model.set_source_gaussian(
            torch.from_numpy(source_mean).to(device),
            torch.from_numpy(source_chol).to(device),
        )
        source_diagnostics.update(
            {
                "source_mean_abs": float(np.mean(np.abs(source_mean))),
                "source_cholesky_diag_min": float(np.min(np.diag(source_chol))),
                "source_cholesky_diag_max": float(np.max(np.diag(source_chol))),
                "source_cov_shrinkage": float(args.source_cov_shrinkage),
                "source_cov_jitter": float(args.source_cov_jitter),
            }
        )
    elif args.source_mode in {"empirical_path", "empirical_conditional"}:
        source_bank = train_inc.reshape(train_inc.shape[0], -1).detach().cpu()
        source_keys = train_hist[:, -1, :].detach().cpu() if args.source_mode == "empirical_conditional" else None
        model.set_source_bank(source_bank.to(device), None if source_keys is None else source_keys.to(device))
        source_diagnostics.update(
            {
                "source_bank_rows": int(source_bank.shape[0]),
                "source_bank_dim": int(source_bank.shape[1]),
                "source_bank_std": float(source_bank.std().item()),
                "conditional_source_topk": int(args.conditional_source_topk)
                if args.source_mode == "empirical_conditional"
                else None,
            }
        )
    elif args.source_mode == "conditional_affine":
        source_diagnostics.update(
            {
                "source_prior_nll_weight": float(args.source_prior_nll_weight),
                "source_loc_clip": float(args.source_loc_clip),
                "source_log_scale_min": float(args.source_log_scale_min),
                "source_log_scale_max": float(args.source_log_scale_max),
            }
        )
    elif args.source_mode == "conditional_latent":
        source_diagnostics.update(
            {
                "source_latent_dim": int(args.source_latent_dim),
                "source_log_scale_min": float(args.source_log_scale_min),
                "source_log_scale_max": float(args.source_log_scale_max),
            }
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(
        TensorDataset(train_hist, train_inc),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_inc),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for history, target in loader:
            with torch.set_grad_enabled(train_mode):
                loss, metrics = model.training_loss(
                    history,
                    target,
                    fm_loss_mode=args.fm_loss_mode,
                    source_prior_nll_weight=float(args.source_prior_nll_weight),
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Device: {device}")
    print(f"Train windows: {train_hist.shape[0]}  Val windows: {val_hist.shape[0]}")
    print(f"Target shape: {tuple(train_inc.shape[1:])}  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(json.dumps(source_diagnostics, indent=2))
    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_loss": train_avg["loss"],
            "val_loss": val_avg["loss"],
            "train_fm_loss": train_avg.get("fm_loss", train_avg["loss"]),
            "val_fm_loss": val_avg.get("fm_loss", val_avg["loss"]),
            "train_velocity_mse": train_avg.get("velocity_mse", train_avg["loss"]),
            "val_velocity_mse": val_avg.get("velocity_mse", val_avg["loss"]),
            "train_cumulative_mse": train_avg.get("cumulative_mse", train_avg["loss"]),
            "val_cumulative_mse": val_avg.get("cumulative_mse", val_avg["loss"]),
            "train_source_prior_nll": train_avg.get("source_prior_nll", 0.0),
            "val_source_prior_nll": val_avg.get("source_prior_nll", 0.0),
            "val_target_std": val_avg["target_std"],
            "val_velocity_std": val_avg["velocity_std"],
            "val_pred_velocity_std": val_avg["pred_velocity_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_loss']:.5f} val={rec['val_loss']:.5f} "
            f"target_std={rec['val_target_std']:.3f} pred_v_std={rec['val_pred_velocity_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s",
            flush=True,
        )
        if rec["val_loss"] < best_val:
            best_val = rec["val_loss"]
            best_epoch = epoch
            save_checkpoint(
                out_dir / "best_model.pt",
                model,
                epoch=epoch,
                best_val=best_val,
                hist_mean=hist_mean,
                hist_std=hist_std,
                inc_mean=inc_mean,
                inc_std=inc_std,
                increment_transform=args.increment_transform,
                inc_quantiles=inc_quantiles,
                normal_levels=normal_levels,
                specs=train_block.specs,
            )

    save_checkpoint(
        out_dir / "final_model.pt",
        model,
        epoch=args.epochs,
        best_val=best_val,
        hist_mean=hist_mean,
        hist_std=hist_std,
        inc_mean=inc_mean,
        inc_std=inc_std,
        increment_transform=args.increment_transform,
        inc_quantiles=inc_quantiles,
        normal_levels=normal_levels,
        specs=train_block.specs,
    )
    best_path = out_dir / "best_model.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state_dict"])
    sample_audit = _sample_audit(
        model,
        val_hist,
        val_block,
        hist_mean=hist_mean,
        hist_std=hist_std,
        inc_mean=inc_mean,
        inc_std=inc_std,
        increment_transform=args.increment_transform,
        inc_quantiles=inc_quantiles,
        normal_levels=normal_levels,
        sample_windows=args.sample_windows,
        n_samples=args.n_samples,
        n_steps=args.sample_steps,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "params": int(sum(p.numel() for p in model.parameters())),
        "train_windows": int(train_hist.shape[0]),
        "val_windows": int(val_hist.shape[0]),
        "target_shape": list(train_inc.shape[1:]),
        "config": asdict(cfg),
        "source_diagnostics": source_diagnostics,
        "sample_audit": sample_audit,
    }
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(make_serializable(summary), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
