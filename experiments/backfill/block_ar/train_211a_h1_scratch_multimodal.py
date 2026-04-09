#!/usr/bin/env python
"""
211a: fresh minimal scratch H=1 multimodal models.

This file intentionally avoids the old 210c/210e/210g model classes and decoder
scaffolding. It provides three minimal variants under one harness:

  - mdn: soft mixture of K conditional Student-t heads
  - cat: categorical latent model with exact discrete ELBO
  - vq: VQ/codebook latent model

Shared assumptions:
  - H=1 only
  - same 30-day rolling history input
  - same bounded-support transform and smoke-gate metrics
  - no warm starts
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    inverse_softplus,
    iv_to_unconstrained,
    make_serializable,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import (
    compute_h1_shape_stats,
)


def build_sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ScratchHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 25,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        bottleneck_dim: int = 96,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer("positional_encoding", build_sinusoidal_encoding(max_len, d_model), persistent=False)
        self.summary_proj = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        self.pool_proj = nn.Linear(d_model, 1)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model + d_model // 2),
            nn.Linear(d_model + d_model // 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, bottleneck_dim),
        )

    def forward(self, history_01: torch.Tensor) -> torch.Tensor:
        history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
        history_norm = normalize_iv(history_flat)
        t = history_norm.shape[1]
        x = self.input_proj(history_norm) + self.positional_encoding[:t].unsqueeze(0)
        h = self.encoder(x)
        attn = torch.softmax(self.pool_proj(h).squeeze(-1), dim=1)
        pooled = (attn.unsqueeze(-1) * h).sum(dim=1)

        hist_mean = history_01.mean(dim=(-1, -2))
        vov = (hist_mean[:, 1:] - hist_mean[:, :-1]).std(dim=1)
        last_mean = hist_mean[:, -1]
        trend = hist_mean[:, -1] - hist_mean[:, 0]
        summary = self.summary_proj(torch.stack([vov, last_mean, trend], dim=-1))
        return self.bottleneck(torch.cat([pooled, summary], dim=-1))


class ScratchTargetEncoder(nn.Module):
    def __init__(self, n_cells: int = 25, hidden_dim: int = 96, out_dim: int = 48, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_cells * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, prev_u: torch.Tensor, target_u: torch.Tensor) -> torch.Tensor:
        delta = target_u - prev_u
        feat = torch.cat([prev_u, target_u, delta, delta.abs()], dim=-1)
        return self.net(feat)


class ScratchSpatialStudentTDecoder(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        cond_dim: int = 96,
        latent_dim: int = 48,
        rank: int = 4,
        dropout: float = 0.1,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        nu_floor: float = 2.1,
        nu_max: float = 100.0,
        init_diag: float = 0.08,
        init_scale: float = 0.08,
        fixed_nu: float | None = 8.0,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.rank = rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.nu_floor = nu_floor
        self.nu_max = nu_max
        self.fixed_nu = fixed_nu

        self.prev_proj = nn.Linear(1, d_model)
        self.cell_embed = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.latent_scale = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.Tanh(),
        )
        self.latent_shift = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, 1)
        self.factor_head = nn.Linear(d_model, rank)
        self.diag_head = nn.Linear(d_model, 1)
        self.scale_head = nn.Linear(d_model, 1)
        if self.fixed_nu is None:
            self.nu_head = nn.Linear(d_model, 1)
        else:
            fixed = float(np.clip(self.fixed_nu, self.nu_floor + 1e-6, self.nu_max))
            self.register_buffer("fixed_nu_value", torch.tensor(fixed, dtype=torch.float32))

        self._init_parameters(init_diag=init_diag, init_scale=init_scale)

    def _init_parameters(self, init_diag: float, init_scale: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(h in name for h in ("mean_head", "factor_head", "diag_head", "scale_head", "nu_head")):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.normal_(self.factor_head.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.factor_head.bias, mean=0.0, std=1e-3)
        nn.init.zeros_(self.diag_head.weight)
        nn.init.constant_(self.diag_head.bias, inverse_softplus(max(init_diag - self.diag_floor, 1e-6)))
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, inverse_softplus(max(init_scale - self.scale_floor, 1e-6)))
        if self.fixed_nu is None:
            nn.init.zeros_(self.nu_head.weight)
            nn.init.constant_(self.nu_head.bias, inverse_softplus(max(8.0 - self.nu_floor, 1e-6)))

    def forward(
        self,
        cond: torch.Tensor,
        prev_u: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent is None:
            latent = cond.new_zeros(cond.shape[0], 0)
            cond_lat = cond
            lat_scale = cond.new_zeros(cond.shape[0], self.cell_embed.shape[-1])
            lat_shift = cond.new_zeros(cond.shape[0], self.cell_embed.shape[-1])
        else:
            cond_lat = torch.cat([cond, latent], dim=-1)
            lat_scale = self.latent_scale(latent)
            lat_shift = self.latent_shift(latent)

        h = self.prev_proj(prev_u.unsqueeze(-1))
        h = h + self.cell_embed
        ctx = self.cond_proj(cond_lat).unsqueeze(1)
        h = (h + ctx) * (1.0 + lat_scale.unsqueeze(1)) + lat_shift.unsqueeze(1)
        h = self.encoder(h)
        h = self.out_norm(h)
        pooled = h.mean(dim=1)

        mu = prev_u + self.mean_head(h).squeeze(-1)
        factor = self.factor_head(h)
        diag = F.softplus(self.diag_head(h).squeeze(-1)) + self.diag_floor
        scale = F.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor
        if self.fixed_nu is None:
            nu = F.softplus(self.nu_head(pooled).squeeze(-1)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(h.shape[0]).to(h.dtype)
        return mu, factor, diag, scale, nu


class ScratchStudentTBase(nn.Module):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        support_lo: float = 0.01,
        support_hi: float = 1.0,
        support_eps: float = 1e-5,
        cov_jitter: float = 1e-4,
    ):
        super().__init__()
        self.encoder = ScratchHistoryEncoder(**encoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        return self.encoder(history_01)

    def _prev_u(self, history_01: torch.Tensor) -> torch.Tensor:
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        return iv_to_unconstrained(prev_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)

    def normalized_components(
        self, factor: torch.Tensor, diag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_diag = factor.pow(2).sum(dim=-1) + diag.pow(2) + self.cov_jitter
        avg_var = raw_diag.mean(dim=-1).clamp_min(self.cov_jitter)
        norm = avg_var.sqrt().unsqueeze(-1)
        factor_norm = factor / norm.unsqueeze(-1)
        diag_norm = diag / norm
        return factor_norm, diag_norm, avg_var

    def covariance(self, factor: torch.Tensor, diag: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        cov = factor_norm @ factor_norm.transpose(-1, -2)
        cov = cov + torch.diag_embed(diag_norm.pow(2) + self.cov_jitter)
        cov = cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
        return cov

    def student_t_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        cov = self.covariance(factor, diag, scale)
        chol = torch.linalg.cholesky(cov)
        diff = (target_u - mu).unsqueeze(-1)
        solved = torch.cholesky_solve(diff, chol).squeeze(-1)
        mahal = (diff.squeeze(-1) * solved).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        d = target_u.shape[-1]
        nu = nu.clamp_min(2.1 + 1e-6)
        pi = target_u.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (d * torch.log(nu * pi) + logdet)
        )
        log_kernel = -0.5 * (nu + d) * torch.log1p(mahal / nu)
        return log_norm + log_kernel

    def nll_from_params(
        self,
        target_01: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        return -self.student_t_log_prob(target_u, mu, factor, diag, scale, nu)

    def sample_from_params(
        self,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        factor_norm, diag_norm, _ = self.normalized_components(factor, diag)
        eps_lowrank = torch.randn(mu.shape[0], factor.shape[-1], device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(mu.shape[0], mu.shape[-1], device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bcr,br->bc", factor_norm, eps_lowrank)
        diag_noise = diag_norm * eps_diag
        gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
        mix = gamma.sample().clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)
        return mu + (lowrank_noise + diag_noise) * scale.unsqueeze(-1) * t_scale

    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        samples_u = self.sample_next_u(history_01, n_samples=n_samples)
        return unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)


class ScratchMDNModel(ScratchStudentTBase):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        n_components: int = 4,
        latent_dim: int = 48,
        **kwargs,
    ):
        super().__init__(encoder_config=encoder_config, decoder_config=decoder_config, **kwargs)
        self.n_components = n_components
        self.latent_dim = latent_dim
        self.decoder = ScratchSpatialStudentTDecoder(**decoder_config)
        self.mix_head = nn.Linear(encoder_config["bottleneck_dim"], n_components)
        self.component_embed = nn.Embedding(n_components, latent_dim)
        nn.init.normal_(self.component_embed.weight, mean=0.0, std=0.05)

    def decode_all(self, cond: torch.Tensor, prev_u: torch.Tensor):
        batch = cond.shape[0]
        comp_ids = torch.arange(self.n_components, device=cond.device)
        lat = self.component_embed(comp_ids)
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_components, cond.shape[-1]).reshape(batch * self.n_components, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_components, prev_u.shape[-1]).reshape(batch * self.n_components, prev_u.shape[-1])
        lat_rep = lat.unsqueeze(0).expand(batch, self.n_components, self.latent_dim).reshape(batch * self.n_components, self.latent_dim)
        params = self.decoder(cond_rep, prev_rep, lat_rep)
        return tuple(x.view(batch, self.n_components, *x.shape[1:]) for x in params)

    def train_objective(self, history_01: torch.Tensor, target_01: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        mix_logits = self.mix_head(cond)
        mu, factor, diag, scale, nu = self.decode_all(cond, prev_u)
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        target_rep = target_u.unsqueeze(1).expand(-1, self.n_components, -1).reshape(-1, target_u.shape[-1])
        log_prob = self.student_t_log_prob(
            target_rep,
            mu.reshape(-1, mu.shape[-1]),
            factor.reshape(-1, factor.shape[-2], factor.shape[-1]),
            diag.reshape(-1, diag.shape[-1]),
            scale.reshape(-1),
            nu.reshape(-1),
        ).view(target_u.shape[0], self.n_components)
        log_mix = torch.log_softmax(mix_logits, dim=-1)
        nll = -torch.logsumexp(log_mix + log_prob, dim=-1)
        resp = torch.softmax(log_mix + log_prob, dim=-1)
        mix_probs = torch.softmax(mix_logits, dim=-1)
        marginal = mix_probs.mean(dim=0)
        total = nll.mean()
        metrics = {
            "nll": nll.mean().detach(),
            "mix_top1": mix_probs.max(dim=-1).values.mean().detach(),
            "mix_entropy": (-(mix_probs * torch.log(mix_probs.clamp_min(1e-8))).sum(dim=-1)).mean().detach(),
            "resp_top1": resp.max(dim=-1).values.mean().detach(),
            "active_components": torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum()).detach(),
        }
        return total, metrics

    @torch.no_grad()
    def exact_marginal_nll(self, history_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        mix_logits = self.mix_head(cond)
        mu, factor, diag, scale, nu = self.decode_all(cond, prev_u)
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        target_rep = target_u.unsqueeze(1).expand(-1, self.n_components, -1).reshape(-1, target_u.shape[-1])
        log_prob = self.student_t_log_prob(
            target_rep,
            mu.reshape(-1, mu.shape[-1]),
            factor.reshape(-1, factor.shape[-2], factor.shape[-1]),
            diag.reshape(-1, diag.shape[-1]),
            scale.reshape(-1),
            nu.reshape(-1),
        ).view(target_u.shape[0], self.n_components)
        return -torch.logsumexp(torch.log_softmax(mix_logits, dim=-1) + log_prob, dim=-1)

    @torch.no_grad()
    def prior_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        probs = torch.softmax(self.mix_head(cond), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        marginal = probs.mean(dim=0)
        active = torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum())
        return {
            "prior_top1_mean": probs.max(dim=-1).values.mean(),
            "prior_entropy_mean": entropy.mean(),
            "prior_active": active,
            "prior_marginal_probs": marginal,
        }

    @torch.no_grad()
    def effect_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        mu, _factor, _diag, scale, _nu = self.decode_all(cond, prev_u)
        mu_disp = (mu - mu.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2)).sqrt().mean()
        scale_disp = (scale - scale.mean(dim=1, keepdim=True)).pow(2).mean(dim=1).sqrt().mean()
        return {"mu_dispersion": mu_disp, "scale_dispersion": scale_disp}

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        probs = torch.softmax(self.mix_head(cond), dim=-1)
        comp_idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = history_01.shape[0]
        cond_rep = cond.unsqueeze(1).expand(batch, n_samples, cond.shape[-1]).reshape(batch * n_samples, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, n_samples, prev_u.shape[-1]).reshape(batch * n_samples, prev_u.shape[-1])
        lat = self.component_embed(comp_idx.reshape(-1))
        mu, factor, diag, scale, nu = self.decoder(cond_rep, prev_rep, lat)
        samples = self.sample_from_params(mu, factor, diag, scale, nu)
        return samples.view(batch, n_samples, -1)


class ScratchCategoricalModel(ScratchStudentTBase):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        n_codes: int = 8,
        latent_dim: int = 48,
        posterior_hidden_dim: int = 96,
        posterior_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(encoder_config=encoder_config, decoder_config=decoder_config, **kwargs)
        self.n_codes = n_codes
        self.latent_dim = latent_dim
        self.decoder = ScratchSpatialStudentTDecoder(**decoder_config)
        self.target_encoder = ScratchTargetEncoder(n_cells=25, hidden_dim=posterior_hidden_dim, out_dim=latent_dim, dropout=posterior_dropout)
        self.prior_head = nn.Linear(encoder_config["bottleneck_dim"], n_codes)
        self.posterior_head = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] + latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, n_codes),
        )
        self.code_embed = nn.Embedding(n_codes, latent_dim)
        nn.init.normal_(self.code_embed.weight, mean=0.0, std=0.05)

    def decode_all(self, cond: torch.Tensor, prev_u: torch.Tensor):
        batch = cond.shape[0]
        code_ids = torch.arange(self.n_codes, device=cond.device)
        lat = self.code_embed(code_ids)
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_codes, cond.shape[-1]).reshape(batch * self.n_codes, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_codes, prev_u.shape[-1]).reshape(batch * self.n_codes, prev_u.shape[-1])
        lat_rep = lat.unsqueeze(0).expand(batch, self.n_codes, self.latent_dim).reshape(batch * self.n_codes, self.latent_dim)
        params = self.decoder(cond_rep, prev_rep, lat_rep)
        return tuple(x.view(batch, self.n_codes, *x.shape[1:]) for x in params)

    def train_objective(self, history_01: torch.Tensor, target_01: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        prior_logits = self.prior_head(cond)
        post_feat = self.target_encoder(prev_u, target_u)
        post_logits = self.posterior_head(torch.cat([cond, post_feat], dim=-1))
        q = torch.softmax(post_logits, dim=-1)
        log_q = torch.log(q.clamp_min(1e-8))
        log_p = torch.log_softmax(prior_logits, dim=-1)

        mu, factor, diag, scale, nu = self.decode_all(cond, prev_u)
        target_rep = target_u.unsqueeze(1).expand(-1, self.n_codes, -1).reshape(-1, target_u.shape[-1])
        nll_z = self.nll_from_params(
            unconstrained_to_iv(target_rep, lo=self.support_lo, hi=self.support_hi),
            mu.reshape(-1, mu.shape[-1]),
            factor.reshape(-1, factor.shape[-2], factor.shape[-1]),
            diag.reshape(-1, diag.shape[-1]),
            scale.reshape(-1),
            nu.reshape(-1),
        ).view(target_u.shape[0], self.n_codes)
        exp_nll = (q * nll_z).sum(dim=-1)
        kl = (q * (log_q - log_p)).sum(dim=-1)
        total = exp_nll.mean() + kl_weight * kl.mean()
        marginal = torch.softmax(prior_logits, dim=-1).mean(dim=0)
        metrics = {
            "nll": exp_nll.mean().detach(),
            "kl": kl.mean().detach(),
            "prior_top1": torch.softmax(prior_logits, dim=-1).max(dim=-1).values.mean().detach(),
            "prior_entropy": (-(torch.softmax(prior_logits, dim=-1) * log_p).sum(dim=-1)).mean().detach(),
            "post_top1": q.max(dim=-1).values.mean().detach(),
            "post_entropy": (-(q * log_q).sum(dim=-1)).mean().detach(),
            "active_codes": torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum()).detach(),
        }
        return total, metrics

    @torch.no_grad()
    def exact_marginal_nll(self, history_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        prior_logits = self.prior_head(cond)
        mu, factor, diag, scale, nu = self.decode_all(cond, prev_u)
        target_rep = target_u.unsqueeze(1).expand(-1, self.n_codes, -1).reshape(-1, target_u.shape[-1])
        log_prob = self.student_t_log_prob(
            target_rep,
            mu.reshape(-1, mu.shape[-1]),
            factor.reshape(-1, factor.shape[-2], factor.shape[-1]),
            diag.reshape(-1, diag.shape[-1]),
            scale.reshape(-1),
            nu.reshape(-1),
        ).view(target_u.shape[0], self.n_codes)
        return -torch.logsumexp(torch.log_softmax(prior_logits, dim=-1) + log_prob, dim=-1)

    @torch.no_grad()
    def prior_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        probs = torch.softmax(self.prior_head(cond), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        marginal = probs.mean(dim=0)
        active = torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum())
        return {
            "prior_top1_mean": probs.max(dim=-1).values.mean(),
            "prior_entropy_mean": entropy.mean(),
            "prior_active": active,
            "prior_marginal_probs": marginal,
        }

    @torch.no_grad()
    def effect_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        mu, _factor, _diag, scale, _nu = self.decode_all(cond, prev_u)
        mu_disp = (mu - mu.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2)).sqrt().mean()
        scale_disp = (scale - scale.mean(dim=1, keepdim=True)).pow(2).mean(dim=1).sqrt().mean()
        return {"mu_dispersion": mu_disp, "scale_dispersion": scale_disp}

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        probs = torch.softmax(self.prior_head(cond), dim=-1)
        code_idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = history_01.shape[0]
        cond_rep = cond.unsqueeze(1).expand(batch, n_samples, cond.shape[-1]).reshape(batch * n_samples, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, n_samples, prev_u.shape[-1]).reshape(batch * n_samples, prev_u.shape[-1])
        lat = self.code_embed(code_idx.reshape(-1))
        mu, factor, diag, scale, nu = self.decoder(cond_rep, prev_rep, lat)
        samples = self.sample_from_params(mu, factor, diag, scale, nu)
        return samples.view(batch, n_samples, -1)


class ScratchVQModel(ScratchStudentTBase):
    def __init__(
        self,
        encoder_config: dict[str, Any],
        decoder_config: dict[str, Any],
        n_codes: int = 8,
        latent_dim: int = 48,
        posterior_hidden_dim: int = 96,
        posterior_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(encoder_config=encoder_config, decoder_config=decoder_config, **kwargs)
        self.n_codes = n_codes
        self.latent_dim = latent_dim
        self.decoder = ScratchSpatialStudentTDecoder(**decoder_config)
        self.target_encoder = ScratchTargetEncoder(n_cells=25, hidden_dim=posterior_hidden_dim, out_dim=latent_dim, dropout=posterior_dropout)
        self.posterior_proj = nn.Sequential(
            nn.Linear(encoder_config["bottleneck_dim"] + latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prior_head = nn.Linear(encoder_config["bottleneck_dim"], n_codes)
        self.codebook = nn.Embedding(n_codes, latent_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.05)

    def posterior_embedding(self, cond: torch.Tensor, prev_u: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        feat = self.target_encoder(prev_u, target_u)
        return self.posterior_proj(torch.cat([cond, feat], dim=-1))

    def code_distances(self, post_embed: torch.Tensor) -> torch.Tensor:
        codes = self.codebook.weight
        post_sq = post_embed.pow(2).sum(dim=-1, keepdim=True)
        code_sq = codes.pow(2).sum(dim=-1).unsqueeze(0)
        cross = post_embed @ codes.t()
        return post_sq + code_sq - 2.0 * cross

    def quantize(self, post_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = self.code_distances(post_embed)
        idx = distances.argmin(dim=-1)
        return self.codebook(idx), idx

    def train_objective(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        prior_ce_weight: float,
        codebook_weight: float,
        commitment_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        prior_logits = self.prior_head(cond)
        post_embed = self.posterior_embedding(cond, prev_u, target_01)
        quantized, code_idx = self.quantize(post_embed)
        quantized_st = post_embed + (quantized - post_embed).detach()
        mu, factor, diag, scale, nu = self.decoder(cond, prev_u, quantized_st)
        nll = self.nll_from_params(target_01, mu, factor, diag, scale, nu)
        prior_ce = F.cross_entropy(prior_logits, code_idx)
        codebook_loss = F.mse_loss(quantized, post_embed.detach())
        commitment_loss = F.mse_loss(post_embed, quantized.detach())
        total = nll.mean() + prior_ce_weight * prior_ce + codebook_weight * codebook_loss + commitment_weight * commitment_loss
        prior_probs = torch.softmax(prior_logits, dim=-1)
        marginal = prior_probs.mean(dim=0)
        metrics = {
            "nll": nll.mean().detach(),
            "prior_ce": prior_ce.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "prior_top1": prior_probs.max(dim=-1).values.mean().detach(),
            "prior_entropy": (-(prior_probs * torch.log(prior_probs.clamp_min(1e-8))).sum(dim=-1)).mean().detach(),
            "post_top1": torch.ones_like(prior_probs[:, 0]).mean().detach(),
            "active_codes": torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum()).detach(),
        }
        return total, metrics

    @torch.no_grad()
    def exact_marginal_nll(self, history_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        prior_logits = self.prior_head(cond)
        batch = cond.shape[0]
        code_ids = torch.arange(self.n_codes, device=cond.device)
        lat = self.codebook(code_ids)
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_codes, cond.shape[-1]).reshape(batch * self.n_codes, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_codes, prev_u.shape[-1]).reshape(batch * self.n_codes, prev_u.shape[-1])
        lat_rep = lat.unsqueeze(0).expand(batch, self.n_codes, self.latent_dim).reshape(batch * self.n_codes, self.latent_dim)
        mu, factor, diag, scale, nu = self.decoder(cond_rep, prev_rep, lat_rep)
        target_u = iv_to_unconstrained(target_01, lo=self.support_lo, hi=self.support_hi, eps=self.support_eps)
        target_rep = target_u.unsqueeze(1).expand(-1, self.n_codes, -1).reshape(-1, target_u.shape[-1])
        log_prob = self.student_t_log_prob(
            target_rep,
            mu.reshape(-1, mu.shape[-1]),
            factor.reshape(-1, factor.shape[-2], factor.shape[-1]),
            diag.reshape(-1, diag.shape[-1]),
            scale.reshape(-1),
            nu.reshape(-1),
        ).view(batch, self.n_codes)
        return -torch.logsumexp(torch.log_softmax(prior_logits, dim=-1) + log_prob, dim=-1)

    @torch.no_grad()
    def prior_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        probs = torch.softmax(self.prior_head(cond), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        marginal = probs.mean(dim=0)
        active = torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum())
        return {
            "prior_top1_mean": probs.max(dim=-1).values.mean(),
            "prior_entropy_mean": entropy.mean(),
            "prior_active": active,
            "prior_marginal_probs": marginal,
        }

    @torch.no_grad()
    def effect_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        batch = cond.shape[0]
        code_ids = torch.arange(self.n_codes, device=cond.device)
        lat = self.codebook(code_ids)
        cond_rep = cond.unsqueeze(1).expand(batch, self.n_codes, cond.shape[-1]).reshape(batch * self.n_codes, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, self.n_codes, prev_u.shape[-1]).reshape(batch * self.n_codes, prev_u.shape[-1])
        lat_rep = lat.unsqueeze(0).expand(batch, self.n_codes, self.latent_dim).reshape(batch * self.n_codes, self.latent_dim)
        mu, _factor, _diag, scale, _nu = self.decoder(cond_rep, prev_rep, lat_rep)
        mu = mu.view(batch, self.n_codes, -1)
        scale = scale.view(batch, self.n_codes)
        mu_disp = (mu - mu.mean(dim=1, keepdim=True)).pow(2).mean(dim=(1, 2)).sqrt().mean()
        scale_disp = (scale - scale.mean(dim=1, keepdim=True)).pow(2).mean(dim=1).sqrt().mean()
        return {"mu_dispersion": mu_disp, "scale_dispersion": scale_disp}

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        cond = self.encode(history_01)
        prev_u = self._prev_u(history_01)
        probs = torch.softmax(self.prior_head(cond), dim=-1)
        code_idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch = history_01.shape[0]
        cond_rep = cond.unsqueeze(1).expand(batch, n_samples, cond.shape[-1]).reshape(batch * n_samples, cond.shape[-1])
        prev_rep = prev_u.unsqueeze(1).expand(batch, n_samples, prev_u.shape[-1]).reshape(batch * n_samples, prev_u.shape[-1])
        lat = self.codebook(code_idx.reshape(-1))
        mu, factor, diag, scale, nu = self.decoder(cond_rep, prev_rep, lat)
        samples = self.sample_from_params(mu, factor, diag, scale, nu)
        return samples.view(batch, n_samples, -1)


def build_model(variant: str, args: argparse.Namespace):
    encoder_config = dict(
        input_dim=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        max_len=max(args.history_len, 64),
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.cell_layers,
        cond_dim=args.bottleneck_dim,
        latent_dim=args.latent_dim,
        rank=args.rank,
        dropout=args.dropout,
        fixed_nu=args.fixed_nu,
    )
    common = dict(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    )
    if variant == "mdn":
        return ScratchMDNModel(n_components=args.n_components, latent_dim=args.latent_dim, **common)
    if variant == "cat":
        return ScratchCategoricalModel(
            n_codes=args.n_codes,
            latent_dim=args.latent_dim,
            posterior_hidden_dim=args.posterior_hidden_dim,
            posterior_dropout=args.posterior_dropout,
            **common,
        )
    if variant == "vq":
        return ScratchVQModel(
            n_codes=args.n_codes,
            latent_dim=args.latent_dim,
            posterior_hidden_dim=args.posterior_hidden_dim,
            posterior_dropout=args.posterior_dropout,
            **common,
        )
    raise ValueError(f"Unknown variant: {variant}")


@torch.no_grad()
def evaluate_h1(model: nn.Module, loader: DataLoader, q95_threshold: float, q99_threshold: float, eval_samples: int) -> dict[str, float]:
    model.eval()
    totals = {
        "val_nll": 0.0,
        "val_mae": 0.0,
        "val_coverage_90": 0.0,
        "val_width_90": 0.0,
        "val_prior_top1_mean": 0.0,
        "val_prior_entropy_mean": 0.0,
        "val_mu_dispersion": 0.0,
        "val_scale_dispersion": 0.0,
    }
    q95_cover_sum = 0.0
    q99_cover_sum = 0.0
    q95_count = 0
    q99_count = 0
    total_count = 0
    gt_delta_all = []
    sample_delta_all = []
    marginals = []

    for history_01, target_01 in loader:
        nll = model.exact_marginal_nll(history_01, target_01)
        samples = model.sample_next_iv(history_01, n_samples=eval_samples)
        stats = model.prior_statistics(history_01)
        disp = model.effect_statistics(history_01)

        q05 = samples.quantile(0.05, dim=1)
        q95 = samples.quantile(0.95, dim=1)
        mean_pred = samples.mean(dim=1)

        prev = history_01[:, -1].reshape(history_01.shape[0], -1)
        target_abs = (target_01 - prev).abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold

        coverage = ((target_01 >= q05) & (target_01 <= q95)).float().mean()
        mae = (mean_pred - target_01).abs().mean()
        width = (q95 - q05).mean()
        q95_cov = ((target_01[q95_mask] >= q05[q95_mask]) & (target_01[q95_mask] <= q95[q95_mask])).float().mean() if q95_mask.any() else target_01.new_tensor(0.0)
        q99_cov = ((target_01[q99_mask] >= q05[q99_mask]) & (target_01[q99_mask] <= q95[q99_mask])).float().mean() if q99_mask.any() else target_01.new_tensor(0.0)

        batch_size = history_01.shape[0]
        totals["val_nll"] += float(nll.mean().item()) * batch_size
        totals["val_mae"] += float(mae.item()) * batch_size
        totals["val_coverage_90"] += float(coverage.item()) * batch_size
        totals["val_width_90"] += float(width.item()) * batch_size
        totals["val_prior_top1_mean"] += float(stats["prior_top1_mean"].item()) * batch_size
        totals["val_prior_entropy_mean"] += float(stats["prior_entropy_mean"].item()) * batch_size
        totals["val_mu_dispersion"] += float(disp["mu_dispersion"].item()) * batch_size
        totals["val_scale_dispersion"] += float(disp["scale_dispersion"].item()) * batch_size
        if q95_mask.any():
            q95_cover_sum += float(q95_cov.item()) * int(q95_mask.sum().item())
            q95_count += int(q95_mask.sum().item())
        if q99_mask.any():
            q99_cover_sum += float(q99_cov.item()) * int(q99_mask.sum().item())
            q99_count += int(q99_mask.sum().item())

        gt_delta_all.append((target_01 - prev).detach().cpu().numpy())
        sample_delta_all.append((samples - prev.unsqueeze(1)).detach().cpu().numpy())
        marginals.append(stats["prior_marginal_probs"].detach().cpu().numpy())
        total_count += batch_size

    gt_delta = np.concatenate(gt_delta_all, axis=0)
    sample_delta = np.concatenate(sample_delta_all, axis=0)
    shape = compute_h1_shape_stats(gt_delta, sample_delta)
    marginal = np.mean(np.stack(marginals, axis=0), axis=0)
    active = float(np.exp(-(marginal * np.log(np.clip(marginal, 1e-8, None))).sum()))

    metrics = {k: v / max(total_count, 1) for k, v in totals.items()}
    metrics.update(
        {
            "val_realized_q95_coverage_90": q95_cover_sum / max(q95_count, 1),
            "val_realized_q99_coverage_90": q99_cover_sum / max(q99_count, 1),
            "val_q95_cell_count": q95_count,
            "val_q99_cell_count": q99_count,
            "val_h1_quiet_ratio": shape["quiet_ratio"],
            "val_h1_shoulder_ratio": shape["shoulder_ratio"],
            "val_h1_extreme_ratio": shape["extreme_ratio"],
            "val_h1_kurtosis_ratio": shape["kurtosis_ratio"],
            "val_prior_active": active,
        }
    )
    return metrics


def load_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    variant = cfg["variant"]
    ns = argparse.Namespace(**cfg["args"])
    model = build_model(variant, ns)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="211a fresh scratch H=1 multimodal models")
    parser.add_argument("--variant", type=str, required=True, choices=["mdn", "cat", "vq"])
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--val_samples", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--bottleneck_dim", type=int, default=96)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--cell_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
    parser.add_argument("--latent_dim", type=int, default=48)
    parser.add_argument("--n_components", type=int, default=4)
    parser.add_argument("--n_codes", type=int, default=8)
    parser.add_argument("--posterior_hidden_dim", type=int, default=96)
    parser.add_argument("--posterior_dropout", type=float, default=0.1)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--prior_ce_weight", type=float, default=0.5)
    parser.add_argument("--codebook_weight", type=float, default=1.0)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)
    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    model = build_model(args.variant, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"211a scratch H=1 variant={args.variant}")
    print(f"  Train windows: {train_hist.shape[0]}")
    print(f"  Val windows:   {val_hist.shape[0]}")
    print(f"  Params:        {n_params:,}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep: dict[str, float] = {}
        nb = 0
        for history_01, target_01 in train_loader:
            optimizer.zero_grad()
            if args.variant == "mdn":
                loss, metrics = model.train_objective(history_01, target_01)
            elif args.variant == "cat":
                loss, metrics = model.train_objective(history_01, target_01, kl_weight=args.kl_weight)
            else:
                loss, metrics = model.train_objective(
                    history_01,
                    target_01,
                    prior_ce_weight=args.prior_ce_weight,
                    codebook_weight=args.codebook_weight,
                    commitment_weight=args.commitment_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            ep["train_loss"] = ep.get("train_loss", 0.0) + float(loss.item())
            for k, v in metrics.items():
                ep[f"train_{k}"] = ep.get(f"train_{k}", 0.0) + float(v.item())
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in ep.items()}
        val_metrics = evaluate_h1(model, val_loader, q95_threshold=q95_threshold, q99_threshold=q99_threshold, eval_samples=args.val_samples)

        gap = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, val_metrics["val_coverage_90"] - 0.93)
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
        )
        selection_score = gap + 0.01 * val_metrics["val_nll"]
        val_metrics["selection_gap"] = gap
        val_metrics["selection_score"] = selection_score
        row = {"epoch": epoch, **train_metrics, **val_metrics, "elapsed_sec": time.time() - t0}
        history.append(make_serializable(row))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "type": f"scratch_h1_multimodal_{args.variant}_211a",
                "variant": args.variant,
                "args": vars(args),
            },
            "metrics": make_serializable(row),
        }
        torch.save(ckpt, output_dir / "final_model.pt")
        if selection_score < best_score:
            best_score = selection_score
            best_metrics = dict(row)
            torch.save(ckpt, output_dir / "best_model.pt")
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        line = (
            f"Ep {epoch:>3}  train_loss={train_metrics['train_loss']:.4f}  "
            f"val_nll={val_metrics['val_nll']:.4f}  cov90={val_metrics['val_coverage_90']:.4f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.4f}  quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"ptop={val_metrics['val_prior_top1_mean']:.3f}  pact={val_metrics['val_prior_active']:.3f}  "
            f"disp={val_metrics['val_mu_dispersion']:.4f}  ({row['elapsed_sec']:.1f}s)"
        )
        if selection_score <= best_score:
            line += "  *best"
        print(line)

    summary = {
        "best_score": best_score,
        "best_metrics": make_serializable(best_metrics),
        "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        "variant": args.variant,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(make_serializable(summary), f, indent=2)

    print("\nBest metrics:")
    for key, value in (best_metrics or {}).items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
