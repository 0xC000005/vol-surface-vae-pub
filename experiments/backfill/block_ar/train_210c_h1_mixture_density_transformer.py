#!/usr/bin/env python
"""
210c: H=1 mixture-density transformer.

Direct one-step multimodal baseline:
  - trusted transformer history encoder from 201a/201b
  - direct K-component mixture of multivariate Student-t heads
  - one-step likelihood only
  - fixed H=1 smoke setup before any larger run
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
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    TemporalTransformerHistoryEncoder,
)
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import (
    compute_h1_shape_stats,
)


class MixtureSpatialShapeScaleStudentTDecoder(nn.Module):
    def __init__(
        self,
        n_components: int = 4,
        n_cells: int = 25,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        cond_dim: int = 128,
        rank: int = 5,
        diag_floor: float = 1e-3,
        scale_floor: float = 1e-4,
        nu_floor: float = 2.1,
        nu_max: float = 100.0,
        init_diag: float = 0.10,
        init_scale: float = 0.10,
        fixed_nu: float | None = 8.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_components = n_components
        self.n_cells = n_cells
        self.rank = rank
        self.diag_floor = diag_floor
        self.scale_floor = scale_floor
        self.nu_floor = nu_floor
        self.nu_max = nu_max
        self.fixed_nu = fixed_nu

        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.ls_params = nn.ParameterList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn_norm": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout),
                        "ff_norm": nn.LayerNorm(d_model),
                        "ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 4, d_model),
                        ),
                    }
                )
            )
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))
            self.ls_params.append(nn.Parameter(torch.ones(d_model) * 0.1))

        self.out_norm = nn.LayerNorm(d_model)
        self.mix_logit_head = nn.Linear(d_model, n_components)
        self.mean_head = nn.Linear(d_model, n_components)
        self.factor_head = nn.Linear(d_model, n_components * rank)
        self.diag_head = nn.Linear(d_model, n_components)
        self.scale_head = nn.Linear(d_model, n_components)
        if self.fixed_nu is None:
            self.nu_head = nn.Linear(d_model, n_components)
        else:
            fixed = float(np.clip(self.fixed_nu, self.nu_floor + 1e-6, self.nu_max))
            self.register_buffer("fixed_nu_value", torch.tensor(fixed, dtype=torch.float32))

        self._init_parameters(init_diag=init_diag, init_scale=init_scale)

    def _init_parameters(self, init_diag: float, init_scale: float) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if any(
                    head_name in name
                    for head_name in (
                        "mix_logit_head",
                        "mean_head",
                        "factor_head",
                        "diag_head",
                        "scale_head",
                        "nu_head",
                    )
                ):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.mix_logit_head.weight)
        nn.init.zeros_(self.mix_logit_head.bias)

        nn.init.normal_(self.mean_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mean_head.bias)

        nn.init.normal_(self.factor_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.factor_head.bias)

        nn.init.zeros_(self.diag_head.weight)
        nn.init.constant_(
            self.diag_head.bias,
            inverse_softplus(max(init_diag - self.diag_floor, 1e-6)),
        )

        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(
            self.scale_head.bias,
            inverse_softplus(max(init_scale - self.scale_floor, 1e-6)),
        )

        if self.fixed_nu is None:
            nn.init.zeros_(self.nu_head.weight)
            nn.init.constant_(
                self.nu_head.bias,
                inverse_softplus(max(8.0 - self.nu_floor, 1e-6)),
            )

    def forward(
        self, cond: torch.Tensor, prev_u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(prev_u.unsqueeze(-1))
        h = h + self.cond_proj(cond).unsqueeze(1)
        h = h + self.spatial_pos
        h = self.dropout(h)

        for li, layer in enumerate(self.layers):
            ls_a = self.ls_params[2 * li]
            ls_f = self.ls_params[2 * li + 1]
            h_norm = layer["attn_norm"](h)
            attn_out, _ = layer["attn"](h_norm, h_norm, h_norm)
            h = h + ls_a * attn_out
            h = h + ls_f * layer["ff"](layer["ff_norm"](h))

        h = self.out_norm(h)
        pooled = h.mean(dim=1)
        batch = h.shape[0]

        mix_logits = self.mix_logit_head(pooled)

        mu_delta = self.mean_head(h).permute(0, 2, 1).contiguous()
        mu = prev_u.unsqueeze(1) + mu_delta

        factor = self.factor_head(h).view(batch, self.n_cells, self.n_components, self.rank)
        factor = factor.permute(0, 2, 1, 3).contiguous()

        diag = self.diag_head(h).permute(0, 2, 1).contiguous()
        diag = F.softplus(diag) + self.diag_floor

        scale = F.softplus(self.scale_head(pooled)) + self.scale_floor

        if self.fixed_nu is None:
            nu = F.softplus(self.nu_head(pooled)) + self.nu_floor
            nu = torch.clamp(nu, max=self.nu_max)
        else:
            nu = self.fixed_nu_value.expand(batch, self.n_components).to(h.dtype)
        return mix_logits, mu, factor, diag, scale, nu


class H1MixtureDensityTransformer(nn.Module):
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
        self.encoder = TemporalTransformerHistoryEncoder(**encoder_config)
        self.decoder = MixtureSpatialShapeScaleStudentTDecoder(**decoder_config)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.support_lo = support_lo
        self.support_hi = support_hi
        self.support_eps = support_eps
        self.cov_jitter = cov_jitter

    def encode(self, history_01: torch.Tensor, return_attention: bool = False):
        history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1)
        history_norm = normalize_iv(history_flat)
        cond, attn = self.encoder(history_norm)
        if return_attention:
            return cond, attn
        return cond

    def forward_from_history(self, history_01: torch.Tensor, return_attention: bool = False):
        cond, attn = self.encode(history_01, return_attention=True)
        prev_01 = history_01[:, -1].reshape(history_01.shape[0], -1)
        prev_u = iv_to_unconstrained(
            prev_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        outputs = self.decoder(cond, prev_u)
        if return_attention:
            return (*outputs, attn)
        return outputs

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

    def component_log_prob(
        self,
        target_u: torch.Tensor,
        mu: torch.Tensor,
        factor: torch.Tensor,
        diag: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_components, n_cells = mu.shape
        flat_target = target_u.unsqueeze(1).expand(batch, n_components, n_cells).reshape(batch * n_components, n_cells)
        flat_mu = mu.reshape(batch * n_components, n_cells)
        flat_factor = factor.reshape(batch * n_components, n_cells, factor.shape[-1])
        flat_diag = diag.reshape(batch * n_components, n_cells)
        flat_scale = scale.reshape(batch * n_components)
        flat_nu = nu.reshape(batch * n_components)

        cov = self.covariance(flat_factor, flat_diag, flat_scale)
        chol = torch.linalg.cholesky(cov)
        diff = (flat_target - flat_mu).unsqueeze(-1)
        solved = torch.cholesky_solve(diff, chol).squeeze(-1)
        mahal = (diff.squeeze(-1) * solved).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        d = target_u.shape[-1]
        flat_nu = flat_nu.clamp_min(self.decoder.nu_floor + 1e-6)
        pi = flat_target.new_tensor(math.pi)
        log_norm = (
            torch.lgamma((flat_nu + d) / 2.0)
            - torch.lgamma(flat_nu / 2.0)
            - 0.5 * (d * torch.log(flat_nu * pi) + logdet)
        )
        log_kernel = -0.5 * (flat_nu + d) * torch.log1p(mahal / flat_nu)
        return (log_norm + log_kernel).view(batch, n_components)

    def nll(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mix_logits, mu, factor, diag, scale, nu = self.forward_from_history(history_01)
        target_u = iv_to_unconstrained(
            target_01,
            lo=self.support_lo,
            hi=self.support_hi,
            eps=self.support_eps,
        )
        comp_logp = self.component_log_prob(target_u, mu, factor, diag, scale, nu)
        log_mix = torch.log_softmax(mix_logits, dim=-1)
        joint = log_mix + comp_logp
        total_logp = torch.logsumexp(joint, dim=-1)
        nll = -total_logp
        responsibilities = torch.softmax(joint, dim=-1)
        probs = torch.softmax(mix_logits, dim=-1)
        metrics = {
            "nll": nll.mean().detach(),
            "mix_entropy": (-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)).mean().detach(),
            "mix_top1": probs.max(dim=-1).values.mean().detach(),
            "resp_top1": responsibilities.max(dim=-1).values.mean().detach(),
        }
        return nll.mean(), metrics

    @torch.no_grad()
    def mixture_statistics(self, history_01: torch.Tensor) -> dict[str, torch.Tensor]:
        mix_logits, mu, factor, diag, scale, nu = self.forward_from_history(history_01)
        probs = torch.softmax(mix_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        marginal = probs.mean(dim=0)
        active = torch.exp(-(marginal * torch.log(marginal.clamp_min(1e-8))).sum())
        return {
            "mix_top1_mean": probs.max(dim=-1).values.mean(),
            "mix_entropy_mean": entropy.mean(),
            "mix_active_components": active,
            "mix_marginal_probs": marginal,
        }

    @torch.no_grad()
    def sample_next_u(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        mix_logits, mu, factor, diag, scale, nu = self.forward_from_history(history_01)
        batch, n_components, n_cells = mu.shape
        probs = torch.softmax(mix_logits, dim=-1)
        comp_idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)
        batch_idx = torch.arange(batch, device=mu.device)[:, None]

        chosen_mu = mu[batch_idx, comp_idx]
        chosen_factor = factor[batch_idx, comp_idx]
        chosen_diag = diag[batch_idx, comp_idx]
        chosen_scale = scale[batch_idx, comp_idx]
        chosen_nu = nu[batch_idx, comp_idx]

        factor_norm, diag_norm, _ = self.normalized_components(
            chosen_factor.reshape(batch * n_samples, n_cells, factor.shape[-1]),
            chosen_diag.reshape(batch * n_samples, n_cells),
        )
        factor_norm = factor_norm.reshape(batch, n_samples, n_cells, factor.shape[-1])
        diag_norm = diag_norm.reshape(batch, n_samples, n_cells)
        eps_lowrank = torch.randn(batch, n_samples, factor.shape[-1], device=mu.device, dtype=mu.dtype)
        eps_diag = torch.randn(batch, n_samples, n_cells, device=mu.device, dtype=mu.dtype)
        lowrank_noise = torch.einsum("bncr,bnr->bnc", factor_norm, eps_lowrank)
        diag_noise = diag_norm * eps_diag
        gamma = torch.distributions.Gamma(chosen_nu / 2.0, chosen_nu / 2.0)
        mix = gamma.sample().clamp_min(1e-6)
        t_scale = torch.rsqrt(mix).unsqueeze(-1)
        total_noise = (lowrank_noise + diag_noise) * chosen_scale.unsqueeze(-1)
        return chosen_mu + total_noise * t_scale

    @torch.no_grad()
    def sample_next_iv(self, history_01: torch.Tensor, n_samples: int) -> torch.Tensor:
        samples_u = self.sample_next_u(history_01, n_samples=n_samples)
        return unconstrained_to_iv(samples_u, lo=self.support_lo, hi=self.support_hi)


def warm_start_from_201b(model: H1MixtureDensityTransformer, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        print("No warm start")
        return
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Warm start missing: {checkpoint_path}")
        return
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    model_state = model.state_dict()
    loaded = 0

    for key, value in state.items():
        if key.startswith("encoder.") and key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1

    shared_decoder_keys = [
        "decoder.input_proj.weight",
        "decoder.input_proj.bias",
        "decoder.cond_proj.0.weight",
        "decoder.cond_proj.0.bias",
        "decoder.cond_proj.2.weight",
        "decoder.cond_proj.2.bias",
        "decoder.spatial_pos",
        "decoder.out_norm.weight",
        "decoder.out_norm.bias",
    ]
    for key in shared_decoder_keys:
        if key in state and key in model_state and model_state[key].shape == state[key].shape:
            model_state[key] = state[key]
            loaded += 1

    for i in range(len(model.decoder.layers)):
        for sub in ("attn_norm", "ff_norm"):
            for suffix in ("weight", "bias"):
                key = f"decoder.layers.{i}.{sub}.{suffix}"
                if key in state and key in model_state and model_state[key].shape == state[key].shape:
                    model_state[key] = state[key]
                    loaded += 1
        for suffix in ("in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"):
            key = f"decoder.layers.{i}.attn.{suffix}"
            if key in state and key in model_state and model_state[key].shape == state[key].shape:
                model_state[key] = state[key]
                loaded += 1
        for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
            key = f"decoder.layers.{i}.ff.{suffix}"
            if key in state and key in model_state and model_state[key].shape == state[key].shape:
                model_state[key] = state[key]
                loaded += 1
        for ls_idx in (2 * i, 2 * i + 1):
            key = f"decoder.ls_params.{ls_idx}"
            if key in state and key in model_state and model_state[key].shape == state[key].shape:
                model_state[key] = state[key]
                loaded += 1

    # Seed component 0 with the old single-head decoder.
    def copy_first_component(new_key: str, old_key: str, rows: int = 1) -> None:
        nonlocal loaded
        if new_key not in model_state or old_key not in state:
            return
        new_v = model_state[new_key].clone()
        old_v = state[old_key]
        if new_v.dim() == 2:
            new_v[:rows] = old_v
        else:
            new_v[:rows] = old_v
        model_state[new_key] = new_v
        loaded += 1

    copy_first_component("decoder.mean_head.weight", "decoder.mean_head.weight", rows=1)
    copy_first_component("decoder.mean_head.bias", "decoder.mean_head.bias", rows=1)
    copy_first_component("decoder.factor_head.weight", "decoder.factor_head.weight", rows=model.decoder.rank)
    copy_first_component("decoder.factor_head.bias", "decoder.factor_head.bias", rows=model.decoder.rank)
    copy_first_component("decoder.diag_head.weight", "decoder.diag_head.weight", rows=1)
    copy_first_component("decoder.diag_head.bias", "decoder.diag_head.bias", rows=1)
    copy_first_component("decoder.scale_head.weight", "decoder.scale_head.weight", rows=1)
    copy_first_component("decoder.scale_head.bias", "decoder.scale_head.bias", rows=1)

    model.load_state_dict(model_state, strict=False)
    print(f"Loaded warm start from {checkpoint_path} ({loaded} tensors)")


def component_balance_penalty(model: H1MixtureDensityTransformer, history_01: torch.Tensor) -> torch.Tensor:
    mix_logits, *_ = model.forward_from_history(history_01)
    probs = torch.softmax(mix_logits, dim=-1)
    marginal = probs.mean(dim=0)
    target = torch.full_like(marginal, 1.0 / marginal.numel())
    return F.kl_div(marginal.clamp_min(1e-8).log(), target, reduction="batchmean")


@torch.no_grad()
def evaluate_h1(
    model: H1MixtureDensityTransformer,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    eval_samples: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "val_nll": 0.0,
        "val_mae": 0.0,
        "val_coverage_90": 0.0,
        "val_width_90": 0.0,
        "val_mix_top1_mean": 0.0,
        "val_mix_entropy_mean": 0.0,
        "val_resp_top1_mean": 0.0,
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
        loss, metrics = model.nll(history_01, target_01)
        samples = model.sample_next_iv(history_01, n_samples=eval_samples)
        stats = model.mixture_statistics(history_01)
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
        totals["val_nll"] += float(loss.item()) * batch_size
        totals["val_mae"] += float(mae.item()) * batch_size
        totals["val_coverage_90"] += float(coverage.item()) * batch_size
        totals["val_width_90"] += float(width.item()) * batch_size
        totals["val_mix_top1_mean"] += float(metrics["mix_top1"].item()) * batch_size
        totals["val_mix_entropy_mean"] += float(metrics["mix_entropy"].item()) * batch_size
        totals["val_resp_top1_mean"] += float(metrics["resp_top1"].item()) * batch_size

        if q95_mask.any():
            q95_cover_sum += float(q95_cov.item()) * int(q95_mask.sum().item())
            q95_count += int(q95_mask.sum().item())
        if q99_mask.any():
            q99_cover_sum += float(q99_cov.item()) * int(q99_mask.sum().item())
            q99_count += int(q99_mask.sum().item())

        gt_delta_all.append((target_01 - prev).detach().cpu().numpy())
        sample_delta_all.append((samples - prev.unsqueeze(1)).detach().cpu().numpy())
        marginals.append(stats["mix_marginal_probs"].detach().cpu().numpy())
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
            "val_mix_active_components": active,
        }
    )
    return metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[H1MixtureDensityTransformer, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_h1_mixture_density_student_t_210c":
        raise ValueError(f"Expected 210c checkpoint, got {raw_config['type']}")
    model = H1MixtureDensityTransformer(
        encoder_config=raw_config["encoder"],
        decoder_config=raw_config["decoder"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-4),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="210c H=1 mixture density transformer")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--usage_balance_weight", type=float, default=0.0)
    parser.add_argument("--val_samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_components", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
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

    encoder_config = dict(
        input_dim=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        bottleneck_dim=args.d_model,
        max_len=max(args.history_len, 64),
    )
    decoder_config = dict(
        n_components=args.n_components,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=args.d_model,
        rank=args.rank,
        fixed_nu=args.fixed_nu,
        dropout=args.dropout,
    )

    model = H1MixtureDensityTransformer(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    warm_start_from_201b(model, args.base_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)

    n_params = sum(p.numel() for p in model.parameters())
    print("210c H=1 mixture density transformer")
    print(f"  Train windows: {train_hist.shape[0]}")
    print(f"  Val windows:   {val_hist.shape[0]}")
    print(f"  Params:        {n_params:,}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(f"  Components={args.n_components} rank={args.rank}")

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_nll = 0.0
        ep_balance = 0.0
        ep_mix_top1 = 0.0
        ep_mix_entropy = 0.0
        nb = 0

        for history_01, target_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = model.nll(history_01, target_01)
            balance = component_balance_penalty(model, history_01) if args.usage_balance_weight > 0 else loss.new_tensor(0.0)
            total = loss + args.usage_balance_weight * balance
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            ep_nll += float(loss.item())
            ep_balance += float(balance.item())
            ep_mix_top1 += float(metrics["mix_top1"].item())
            ep_mix_entropy += float(metrics["mix_entropy"].item())
            nb += 1

        scheduler.step()
        train_metrics = {
            "train_nll": ep_nll / max(nb, 1),
            "train_balance": ep_balance / max(nb, 1),
            "train_mix_top1": ep_mix_top1 / max(nb, 1),
            "train_mix_entropy": ep_mix_entropy / max(nb, 1),
        }
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.val_samples,
        )

        gap = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, val_metrics["val_coverage_90"] - 0.93)
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
            + max(0.0, 1.50 - val_metrics["val_mix_active_components"])
        )
        selection_score = gap + 0.01 * val_metrics["val_nll"]
        val_metrics["selection_score"] = selection_score
        val_metrics["selection_gap"] = gap

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "elapsed_sec": time.time() - t0,
        }
        history.append(make_serializable(row))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "type": "transformer_h1_mixture_density_student_t_210c",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
            },
            "metrics": make_serializable(row),
        }
        torch.save(ckpt, output_dir / "final_model.pt")

        if selection_score < best_score:
            best_score = selection_score
            best_metrics = row
            torch.save(ckpt, output_dir / "best_model.pt")
            best_flag = "  *best"
        else:
            best_flag = ""

        print(
            f"Ep {epoch:>3d}  "
            f"train_nll={train_metrics['train_nll']:.4f}  "
            f"val_nll={val_metrics['val_nll']:.4f}  "
            f"cov90={val_metrics['val_coverage_90']:.4f}  "
            f"q99cov={val_metrics['val_realized_q99_coverage_90']:.4f}  "
            f"quiet={val_metrics['val_h1_quiet_ratio']:.3f}  "
            f"shoulder={val_metrics['val_h1_shoulder_ratio']:.3f}  "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f}  "
            f"mix_top1={val_metrics['val_mix_top1_mean']:.3f}  "
            f"mix_active={val_metrics['val_mix_active_components']:.3f}  "
            f"({row['elapsed_sec']:.1f}s){best_flag}"
        )

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    if best_metrics is not None:
        print("\nBest metrics:")
        for k, v in best_metrics.items():
            if isinstance(v, (float, int)):
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
