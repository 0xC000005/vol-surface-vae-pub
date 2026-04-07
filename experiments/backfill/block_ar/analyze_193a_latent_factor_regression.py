#!/usr/bin/env python
"""
Focused mechanistic review of 193a regression against the 169c AR baseline.

Questions:
  1. Does 193a overcover already under teacher forcing, or mainly under rollout?
  2. Is the excess width coming from shared latent factors, diagonal noise, or both?
  3. Is 193a living in its high-variance mixture component too often?
  4. Does 193a preserve cross-cell structure under rollout better than 169c?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    ShapeScaleStudentTARModel,
)
from experiments.backfill.block_ar.train_193a_graph_ar_latent_factor_innovation import (
    LatentFactorInnovationARModel,
    compute_cond_from_outputs,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


SELECT_HORIZONS = [1, 7, 14, 30]
SELECT_HIDX = [h - 1 for h in SELECT_HORIZONS]


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return make_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def mean_offdiag_corr(cov: np.ndarray, eps: float = 1e-10) -> float:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(corr[mask].mean())


def corr_from_cov(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def eff_rank_from_matrix(mat: np.ndarray, eps: float = 1e-10) -> float:
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.maximum(eigvals, eps)
    probs = eigvals / np.maximum(eigvals.sum(), eps)
    probs = probs[probs > eps]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def build_test_subset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(data_path)
    dataset = VolSurfaceDataset(
        data["surface"],
        history_len,
        future_len,
        start_idx=test_start,
    )
    n = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    histories = []
    futures = []
    for i in range(n):
        item = dataset[i]
        histories.append(item["history"])
        futures.append(item["future"])
    return torch.stack(histories, dim=0), torch.stack(futures, dim=0)


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, float, float]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    return vov, q20, q80


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    model_type = raw_config["type"]
    enc_cfg = EncoderConfig(**raw_config["encoder"])

    if model_type == "multi_step_student_t_169c":
        model = ShapeScaleStudentTARModel(
            encoder_config=enc_cfg,
            decoder_config=raw_config["decoder"],
            support_lo=raw_config.get("support_lo", 0.01),
            support_hi=raw_config.get("support_hi", 1.0),
            support_eps=raw_config.get("support_eps", 1e-5),
        )
    elif model_type == "graph_ar_latent_factor_innovation_193a":
        model = LatentFactorInnovationARModel(
            encoder_config=enc_cfg,
            decoder_config=raw_config["decoder"],
            support_lo=raw_config.get("support_lo", 0.01),
            support_hi=raw_config.get("support_hi", 1.0),
            support_eps=raw_config.get("support_eps", 1e-5),
            cov_jitter=raw_config.get("cov_jitter", 1e-4),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, model_type, checkpoint


def sample_169c_from_params(
    model: ShapeScaleStudentTARModel,
    mu: torch.Tensor,
    factor: torch.Tensor,
    diag: torch.Tensor,
    scale: torch.Tensor,
    nu: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    batch_size, n_cells = mu.shape
    rank = factor.shape[-1]

    factor_norm, diag_norm, _ = model.normalized_components(factor, diag)
    eps_lowrank = torch.randn(batch_size, n_samples, rank, device=mu.device, dtype=mu.dtype)
    eps_diag = torch.randn(batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype)
    lowrank_noise = torch.einsum("bcr,bnr->bnc", factor_norm, eps_lowrank)
    diag_noise = diag_norm.unsqueeze(1) * eps_diag

    gamma = torch.distributions.Gamma(nu / 2.0, nu / 2.0)
    mix = gamma.sample((n_samples,)).transpose(0, 1).to(mu.dtype).clamp_min(1e-6)
    t_scale = torch.rsqrt(mix).unsqueeze(-1)
    total_noise = (lowrank_noise + diag_noise) * scale.unsqueeze(1).unsqueeze(-1)
    return mu.unsqueeze(1) + total_noise * t_scale


def sample_193a_from_params(
    model: LatentFactorInnovationARModel,
    mu: torch.Tensor,
    base_factor: torch.Tensor,
    diag: torch.Tensor,
    factor_scales: torch.Tensor,
    diag_component_mod: torch.Tensor,
    mix_logits: torch.Tensor,
    component_nus: torch.Tensor,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_cells = mu.shape
    num_factors = base_factor.shape[-1]

    mix_probs = F.softmax(mix_logits, dim=-1)
    comp_idx = torch.multinomial(mix_probs, n_samples, replacement=True)

    eps_factor = torch.randn(batch_size, n_samples, num_factors, device=mu.device, dtype=mu.dtype)
    eps_diag = torch.randn(batch_size, n_samples, n_cells, device=mu.device, dtype=mu.dtype)

    factor_scales_sel = torch.gather(
        factor_scales.unsqueeze(1).expand(-1, n_samples, -1, -1),
        2,
        comp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, num_factors),
    ).squeeze(2)
    diag_mod_sel = torch.gather(
        diag_component_mod.unsqueeze(1).expand(-1, n_samples, -1),
        2,
        comp_idx.unsqueeze(-1),
    ).squeeze(-1)
    nu_sel = torch.gather(
        component_nus.unsqueeze(1).expand(-1, n_samples, -1),
        2,
        comp_idx.unsqueeze(-1),
    ).squeeze(-1)

    lowrank_noise = torch.einsum("bnk,bsk->bsn", base_factor, factor_scales_sel * eps_factor)
    diag_noise = diag.unsqueeze(1) * diag_mod_sel.unsqueeze(-1) * eps_diag

    gamma = torch.distributions.Gamma(nu_sel / 2.0, nu_sel / 2.0)
    mix = gamma.sample().to(mu.dtype).clamp_min(1e-6)
    t_scale = torch.rsqrt(mix).unsqueeze(-1)
    return mu.unsqueeze(1) + (lowrank_noise + diag_noise) * t_scale, comp_idx


def update_delta_stats(
    target_flat: torch.Tensor,
    sample_flat: torch.Tensor,
    gt_delta_sum: np.ndarray,
    gt_delta_outer: np.ndarray,
    gt_delta_count: list[int],
    ro_delta_sum: np.ndarray,
    ro_delta_outer: np.ndarray,
    ro_delta_count: list[int],
) -> None:
    gt_delta = target_flat[:, 29, :] - target_flat[:, 28, :]
    gen_delta = sample_flat[:, :, 29, :] - sample_flat[:, :, 28, :]
    gt_np = gt_delta.detach().cpu().numpy()
    gen_np = gen_delta.reshape(-1, gen_delta.shape[-1]).detach().cpu().numpy()
    gt_delta_sum += gt_np.sum(axis=0)
    gt_delta_outer += gt_np.T @ gt_np
    gt_delta_count[0] += gt_np.shape[0]
    ro_delta_sum += gen_np.sum(axis=0)
    ro_delta_outer += gen_np.T @ gen_np
    ro_delta_count[0] += gen_np.shape[0]


def finalize_delta_summary(
    gt_delta_sum: np.ndarray,
    gt_delta_outer: np.ndarray,
    gt_delta_count: int,
    ro_delta_sum: np.ndarray,
    ro_delta_outer: np.ndarray,
    ro_delta_count: int,
) -> dict[str, float]:
    gt_mean = gt_delta_sum / max(gt_delta_count, 1)
    gt_cov = gt_delta_outer / max(gt_delta_count, 1) - np.outer(gt_mean, gt_mean)
    ro_mean = ro_delta_sum / max(ro_delta_count, 1)
    ro_cov = ro_delta_outer / max(ro_delta_count, 1) - np.outer(ro_mean, ro_mean)
    gt_corr = mean_offdiag_corr(gt_cov)
    ro_corr = mean_offdiag_corr(ro_cov)
    gt_rank = eff_rank_from_matrix(corr_from_cov(gt_cov))
    ro_rank = eff_rank_from_matrix(corr_from_cov(ro_cov))
    return {
        "corr_ratio_h30": float(ro_corr / gt_corr) if abs(gt_corr) > 1e-8 else float("nan"),
        "rank_ratio_h30": float(ro_rank / gt_rank) if gt_rank > 1e-8 else float("nan"),
    }


def analyze_model(
    model,
    model_type: str,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    tf_samples: int,
    rollout_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01_all = denormalize_iv(history_norm)
    future_01_all = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    n_windows, future_len, n_cells = future_01_all.shape
    calm_mask = torch.from_numpy(vov <= q20)
    turb_mask = torch.from_numpy(vov >= q80)

    tf_cov90_sum = 0.0
    tf_width90_sum = 0.0
    tf_mae_sum = 0.0
    tf_total_var_sum = 0.0
    tf_factor_share_sum = 0.0
    tf_high_var_prob_sum = 0.0
    tf_mix_entropy_sum = 0.0
    tf_step_count = 0
    tf_h30_widths = []
    tf_window_widths = []
    tf_h30_cov90 = 0.0
    tf_h30_count = 0

    ro_cov90_sum = 0.0
    ro_width90_sum = 0.0
    ro_mae_sum = 0.0
    ro_total_var_sum = 0.0
    ro_factor_share_sum = 0.0
    ro_high_var_prob_sum = 0.0
    ro_mix_entropy_sum = 0.0
    ro_sampled_high_var_rate_sum = 0.0
    ro_step_count = 0
    ro_h30_widths = []
    ro_window_widths = []
    ro_h30_cov90 = 0.0
    ro_h30_count = 0

    gt_delta_sum_tf = np.zeros(n_cells, dtype=np.float64)
    gt_delta_outer_tf = np.zeros((n_cells, n_cells), dtype=np.float64)
    gt_delta_count_tf = [0]
    ro_delta_sum_tf = np.zeros(n_cells, dtype=np.float64)
    ro_delta_outer_tf = np.zeros((n_cells, n_cells), dtype=np.float64)
    ro_delta_count_tf = [0]

    gt_delta_sum_ro = np.zeros(n_cells, dtype=np.float64)
    gt_delta_outer_ro = np.zeros((n_cells, n_cells), dtype=np.float64)
    gt_delta_count_ro = [0]
    ro_delta_sum_ro = np.zeros(n_cells, dtype=np.float64)
    ro_delta_outer_ro = np.zeros((n_cells, n_cells), dtype=np.float64)
    ro_delta_count_ro = [0]

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            history_01 = history_01_all[start:end]
            future_01 = future_01_all[start:end]
            batch = end - start

            # Teacher-forced local law
            hist_norm = normalize_iv(history_01).reshape(batch, history_01.shape[1], n_cells)
            if model_type == "multi_step_student_t_169c":
                gru_outputs, gru_state = model.encoder.gru(hist_norm)
                prev_01 = history_01[:, -1].reshape(batch, n_cells)
                tf_samples_all = []
                for step in range(future_len):
                    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
                    attn_weights = F.softmax(attn_logits, dim=1)
                    pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
                    cond = model.encoder.bottleneck(pooled)
                    prev_u = iv_to_unconstrained(prev_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
                    mu, factor, diag, scale, nu = model.decoder(cond, prev_u)
                    cov = model.covariance(factor, diag, scale)
                    factor_norm, diag_norm, _ = model.normalized_components(factor, diag)
                    factor_cov = factor_norm @ factor_norm.transpose(-1, -2)
                    factor_cov = factor_cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
                    total_var = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
                    factor_var = torch.diagonal(factor_cov, dim1=-2, dim2=-1).sum(dim=-1)
                    factor_share = factor_var / total_var.clamp_min(1e-8)

                    samples_u = sample_169c_from_params(model, mu, factor, diag, scale, nu, tf_samples)
                    samples_iv = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
                    target_t = future_01[:, step, :]
                    lo = samples_iv.quantile(0.05, dim=1)
                    hi = samples_iv.quantile(0.95, dim=1)
                    median = samples_iv.median(dim=1).values
                    coverage = ((target_t >= lo) & (target_t <= hi)).float().mean(dim=1)
                    width = (hi - lo).mean(dim=1)

                    tf_cov90_sum += float(coverage.mean().item()) * batch
                    tf_width90_sum += float(width.mean().item()) * batch
                    tf_mae_sum += float((median - target_t).abs().mean().item()) * batch
                    tf_total_var_sum += float((total_var / n_cells).mean().item()) * batch
                    tf_factor_share_sum += float(factor_share.mean().item()) * batch
                    tf_step_count += batch

                    if step == 29:
                        tf_h30_widths.append(width.detach().cpu())
                        tf_h30_cov90 += float(coverage.mean().item()) * batch
                        tf_h30_count += batch

                    if step == 29:
                        tf_samples_all.append(samples_iv.detach().cpu())

                    next_t = target_t
                    next_norm = normalize_iv(next_t).unsqueeze(1)
                    next_out, gru_state = model.encoder.gru(next_norm, gru_state)
                    gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
                    prev_01 = next_t

            else:
                hist_norm = normalize_iv(history_01).reshape(batch, history_01.shape[1], n_cells)
                gru_outputs, gru_state = model.encoder.gru(hist_norm)
                prev_01 = history_01[:, -1].reshape(batch, n_cells)
                tf_samples_all = []
                for step in range(future_len):
                    cond, _attn_weights = compute_cond_from_outputs(model, gru_outputs)
                    prev_u = iv_to_unconstrained(prev_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
                    mu, base_factor, diag, factor_scales, diag_component_mod, mix_logits, component_nus = model.decoder(cond, prev_u)
                    cov, loadings, diag_per_component = model.mixture_covariances(base_factor, diag, factor_scales, diag_component_mod)
                    expected_cov = model.expected_covariance(cov, mix_logits)
                    total_var = torch.diagonal(expected_cov, dim1=-2, dim2=-1).sum(dim=-1)
                    factor_cov = torch.einsum("bcik,bcjk->bcij", loadings, loadings)
                    factor_var_comp = torch.diagonal(factor_cov, dim1=-2, dim2=-1).sum(dim=-1)
                    total_var_comp = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-8)
                    mix_probs = F.softmax(mix_logits, dim=-1)
                    factor_share = (mix_probs * (factor_var_comp / total_var_comp)).sum(dim=-1)
                    comp_trace = total_var_comp
                    high_var_idx = comp_trace.argmax(dim=-1)
                    high_var_prob = mix_probs.gather(1, high_var_idx.unsqueeze(-1)).squeeze(-1)
                    mix_entropy = -(mix_probs * mix_probs.clamp_min(1e-8).log()).sum(dim=-1)

                    samples_u, _comp_idx = sample_193a_from_params(
                        model, mu, base_factor, diag, factor_scales, diag_component_mod, mix_logits, component_nus, tf_samples
                    )
                    samples_iv = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
                    target_t = future_01[:, step, :]
                    lo = samples_iv.quantile(0.05, dim=1)
                    hi = samples_iv.quantile(0.95, dim=1)
                    median = samples_iv.median(dim=1).values
                    coverage = ((target_t >= lo) & (target_t <= hi)).float().mean(dim=1)
                    width = (hi - lo).mean(dim=1)

                    tf_cov90_sum += float(coverage.mean().item()) * batch
                    tf_width90_sum += float(width.mean().item()) * batch
                    tf_mae_sum += float((median - target_t).abs().mean().item()) * batch
                    tf_total_var_sum += float((total_var / n_cells).mean().item()) * batch
                    tf_factor_share_sum += float(factor_share.mean().item()) * batch
                    tf_high_var_prob_sum += float(high_var_prob.mean().item()) * batch
                    tf_mix_entropy_sum += float(mix_entropy.mean().item()) * batch
                    tf_step_count += batch

                    if step == 29:
                        tf_h30_widths.append(width.detach().cpu())
                        tf_h30_cov90 += float(coverage.mean().item()) * batch
                        tf_h30_count += batch

                    if step == 29:
                        tf_samples_all.append(samples_iv.detach().cpu())

                    next_t = target_t
                    next_norm = normalize_iv(next_t).unsqueeze(1)
                    next_out, gru_state = model.encoder.gru(next_norm, gru_state)
                    gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
                    prev_01 = next_t

            if tf_samples_all:
                tf_samples_path = torch.zeros(batch, tf_samples, future_len, n_cells)
                tf_samples_path[:, :, 29, :] = tf_samples_all[0]
                update_delta_stats(
                    future_01,
                    tf_samples_path,
                    gt_delta_sum_tf,
                    gt_delta_outer_tf,
                    gt_delta_count_tf,
                    ro_delta_sum_tf,
                    ro_delta_outer_tf,
                    ro_delta_count_tf,
                )

            # Rollout loop with internal stats
            hist_k = history_01.unsqueeze(1).expand(batch, rollout_samples, -1, -1, -1)
            hist_k = hist_k.reshape(batch * rollout_samples, history_01.shape[1], 5, 5).clone()
            rollout_frames = []
            for step in range(future_len):
                if model_type == "multi_step_student_t_169c":
                    mu, factor, diag, scale, nu = model.forward_from_history(hist_k)
                    cov = model.covariance(factor, diag, scale)
                    factor_norm, diag_norm, _ = model.normalized_components(factor, diag)
                    factor_cov = factor_norm @ factor_norm.transpose(-1, -2)
                    factor_cov = factor_cov * scale.pow(2).unsqueeze(-1).unsqueeze(-1)
                    total_var = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1)
                    factor_var = torch.diagonal(factor_cov, dim1=-2, dim2=-1).sum(dim=-1)
                    factor_share = factor_var / total_var.clamp_min(1e-8)

                    samples_u = sample_169c_from_params(model, mu, factor, diag, scale, nu, n_samples=1).squeeze(1)
                    next_iv = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)

                    ro_total_var_sum += float((total_var / n_cells).mean().item()) * (batch * rollout_samples)
                    ro_factor_share_sum += float(factor_share.mean().item()) * (batch * rollout_samples)
                    ro_step_count += batch * rollout_samples
                else:
                    mu, base_factor, diag, factor_scales, diag_component_mod, mix_logits, component_nus, _ = model.forward_from_history(hist_k)
                    cov, loadings, diag_per_component = model.mixture_covariances(base_factor, diag, factor_scales, diag_component_mod)
                    expected_cov = model.expected_covariance(cov, mix_logits)
                    total_var = torch.diagonal(expected_cov, dim1=-2, dim2=-1).sum(dim=-1)
                    factor_cov = torch.einsum("bcik,bcjk->bcij", loadings, loadings)
                    factor_var_comp = torch.diagonal(factor_cov, dim1=-2, dim2=-1).sum(dim=-1)
                    total_var_comp = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-8)
                    mix_probs = F.softmax(mix_logits, dim=-1)
                    factor_share = (mix_probs * (factor_var_comp / total_var_comp)).sum(dim=-1)
                    high_var_idx = total_var_comp.argmax(dim=-1)
                    high_var_prob = mix_probs.gather(1, high_var_idx.unsqueeze(-1)).squeeze(-1)
                    mix_entropy = -(mix_probs * mix_probs.clamp_min(1e-8).log()).sum(dim=-1)

                    samples_u, comp_idx = sample_193a_from_params(
                        model, mu, base_factor, diag, factor_scales, diag_component_mod, mix_logits, component_nus, n_samples=1
                    )
                    next_iv = unconstrained_to_iv(samples_u.squeeze(1), lo=model.support_lo, hi=model.support_hi)
                    sampled_high_var = (comp_idx.squeeze(1) == high_var_idx).float()

                    ro_total_var_sum += float((total_var / n_cells).mean().item()) * (batch * rollout_samples)
                    ro_factor_share_sum += float(factor_share.mean().item()) * (batch * rollout_samples)
                    ro_high_var_prob_sum += float(high_var_prob.mean().item()) * (batch * rollout_samples)
                    ro_mix_entropy_sum += float(mix_entropy.mean().item()) * (batch * rollout_samples)
                    ro_sampled_high_var_rate_sum += float(sampled_high_var.mean().item()) * (batch * rollout_samples)
                    ro_step_count += batch * rollout_samples

                rollout_frames.append(next_iv.view(batch, rollout_samples, 5, 5))
                hist_k = torch.cat([hist_k[:, 1:], next_iv.view(batch * rollout_samples, 1, 5, 5)], dim=1)

            rollout_samples_path = torch.stack(rollout_frames, dim=2)
            future_grid = future_01.view(batch, future_len, 5, 5)
            lo = rollout_samples_path.quantile(0.05, dim=1)
            hi = rollout_samples_path.quantile(0.95, dim=1)
            median = rollout_samples_path.median(dim=1).values
            coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean(dim=(1, 2, 3))
            width = (hi - lo).mean(dim=(1, 2, 3))

            ro_cov90_sum += float(coverage.mean().item()) * batch
            ro_width90_sum += float(width.mean().item()) * batch
            ro_mae_sum += float((median - future_grid).abs().mean().item()) * batch
            ro_window_widths.append(width.detach().cpu())

            h30_target = future_grid[:, 29]
            h30_lo = lo[:, 29]
            h30_hi = hi[:, 29]
            h30_cov = ((h30_target >= h30_lo) & (h30_target <= h30_hi)).float().mean(dim=(1, 2))
            h30_width = (h30_hi - h30_lo).mean(dim=(1, 2))
            ro_h30_cov90 += float(h30_cov.mean().item()) * batch
            ro_h30_count += batch
            ro_h30_widths.append(h30_width.detach().cpu())

            rollout_flat = rollout_samples_path.reshape(batch, rollout_samples, future_len, n_cells)
            update_delta_stats(
                future_01,
                rollout_flat,
                gt_delta_sum_ro,
                gt_delta_outer_ro,
                gt_delta_count_ro,
                ro_delta_sum_ro,
                ro_delta_outer_ro,
                ro_delta_count_ro,
            )

    tf_h30_widths_t = torch.cat(tf_h30_widths) if tf_h30_widths else torch.empty(0)
    ro_h30_widths_t = torch.cat(ro_h30_widths) if ro_h30_widths else torch.empty(0)
    ro_window_widths_t = torch.cat(ro_window_widths) if ro_window_widths else torch.empty(0)
    calm_h30 = calm_mask[: tf_h30_widths_t.shape[0]] if tf_h30_widths_t.numel() else torch.empty(0, dtype=torch.bool)
    turb_h30 = turb_mask[: tf_h30_widths_t.shape[0]] if tf_h30_widths_t.numel() else torch.empty(0, dtype=torch.bool)
    calm_ro = calm_mask[: ro_window_widths_t.shape[0]] if ro_window_widths_t.numel() else torch.empty(0, dtype=torch.bool)
    turb_ro = turb_mask[: ro_window_widths_t.shape[0]] if ro_window_widths_t.numel() else torch.empty(0, dtype=torch.bool)

    result = {
        "teacher_forced": {
            "coverage90": tf_cov90_sum / max(tf_step_count, 1),
            "width90": tf_width90_sum / max(tf_step_count, 1),
            "median_mae": tf_mae_sum / max(tf_step_count, 1),
            "total_var_per_cell": tf_total_var_sum / max(tf_step_count, 1),
            "factor_var_share": tf_factor_share_sum / max(tf_step_count, 1),
            "h30_coverage90": tf_h30_cov90 / max(tf_h30_count, 1),
            "h30_width90": float(tf_h30_widths_t.mean().item()) if tf_h30_widths_t.numel() else float("nan"),
            "h30_turb_calm_width_ratio": float(
                (tf_h30_widths_t[turb_h30].mean() / tf_h30_widths_t[calm_h30].mean()).item()
            ) if tf_h30_widths_t.numel() and calm_h30.any() and turb_h30.any() else float("nan"),
            "delta_corr_rank": finalize_delta_summary(
                gt_delta_sum_tf,
                gt_delta_outer_tf,
                gt_delta_count_tf[0],
                ro_delta_sum_tf,
                ro_delta_outer_tf,
                ro_delta_count_tf[0],
            ),
        },
        "rollout": {
            "coverage90": ro_cov90_sum / max(n_windows, 1),
            "width90": ro_width90_sum / max(n_windows, 1),
            "median_mae": ro_mae_sum / max(n_windows, 1),
            "total_var_per_cell": ro_total_var_sum / max(ro_step_count, 1),
            "factor_var_share": ro_factor_share_sum / max(ro_step_count, 1),
            "h30_coverage90": ro_h30_cov90 / max(ro_h30_count, 1),
            "h30_width90": float(ro_h30_widths_t.mean().item()) if ro_h30_widths_t.numel() else float("nan"),
            "turb_calm_width_ratio": float(
                (ro_window_widths_t[turb_ro].mean() / ro_window_widths_t[calm_ro].mean()).item()
            ) if ro_window_widths_t.numel() and calm_ro.any() and turb_ro.any() else float("nan"),
            "delta_corr_rank": finalize_delta_summary(
                gt_delta_sum_ro,
                gt_delta_outer_ro,
                gt_delta_count_ro[0],
                ro_delta_sum_ro,
                ro_delta_outer_ro,
                ro_delta_count_ro[0],
            ),
        },
    }
    if model_type == "graph_ar_latent_factor_innovation_193a":
        result["teacher_forced"].update({
            "high_var_component_prob": tf_high_var_prob_sum / max(tf_step_count, 1),
            "mix_entropy": tf_mix_entropy_sum / max(tf_step_count, 1),
        })
        result["rollout"].update({
            "high_var_component_prob": ro_high_var_prob_sum / max(ro_step_count, 1),
            "mix_entropy": ro_mix_entropy_sum / max(ro_step_count, 1),
            "sampled_high_var_component_rate": ro_sampled_high_var_rate_sum / max(ro_step_count, 1),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused mechanistic review for 193a vs 169c")
    parser.add_argument("--model_169c", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--model_193a_best", type=str, default="models/backfill/graph_ar_latent_factor_innovation_193a/best_model.pt")
    parser.add_argument("--model_193a_final", type=str, default="models/backfill/graph_ar_latent_factor_innovation_193a/final_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--teacher_forced_samples", type=int, default=16)
    parser.add_argument("--rollout_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-06/analysis/193a_latent_factor_mechanistic",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_169c, type_169c, ckpt_169c = load_model(args.model_169c, args.device)
    model_193a_best, type_193a_best, _ = load_model(args.model_193a_best, args.device)
    model_193a_final, type_193a_final, _ = load_model(args.model_193a_final, args.device)

    history_len = ckpt_169c["config"]["history_len"]
    future_len = ckpt_169c["config"]["future_len"]
    history_norm, future_norm = build_test_subset(
        args.data_path,
        history_len=history_len,
        future_len=future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    vov, q20, q80 = regime_masks_from_history(history_norm)

    result = {
        "config": {
            "test_start": args.test_start,
            "max_windows": int(history_norm.shape[0]),
            "teacher_forced_samples": args.teacher_forced_samples,
            "rollout_samples": args.rollout_samples,
            "batch_size": args.batch_size,
            "q20_vov": q20,
            "q80_vov": q80,
            "calm_windows": int((vov <= q20).sum()),
            "turb_windows": int((vov >= q80).sum()),
        }
    }

    print(f"Analyzing 169c best on {history_norm.shape[0]} windows...")
    result["model_169c_best"] = analyze_model(
        model_169c,
        type_169c,
        history_norm,
        future_norm,
        vov,
        q20,
        q80,
        device=args.device,
        batch_size=args.batch_size,
        tf_samples=args.teacher_forced_samples,
        rollout_samples=args.rollout_samples,
    )

    print(f"Analyzing 193a best on {history_norm.shape[0]} windows...")
    result["model_193a_best"] = analyze_model(
        model_193a_best,
        type_193a_best,
        history_norm,
        future_norm,
        vov,
        q20,
        q80,
        device=args.device,
        batch_size=args.batch_size,
        tf_samples=args.teacher_forced_samples,
        rollout_samples=args.rollout_samples,
    )

    print(f"Analyzing 193a final on {history_norm.shape[0]} windows...")
    result["model_193a_final"] = analyze_model(
        model_193a_final,
        type_193a_final,
        history_norm,
        future_norm,
        vov,
        q20,
        q80,
        device=args.device,
        batch_size=args.batch_size,
        tf_samples=args.teacher_forced_samples,
        rollout_samples=args.rollout_samples,
    )

    summary = {
        "teacher_forced_overcoverage": {
            "169c_cov90": result["model_169c_best"]["teacher_forced"]["coverage90"],
            "193a_best_cov90": result["model_193a_best"]["teacher_forced"]["coverage90"],
            "193a_final_cov90": result["model_193a_final"]["teacher_forced"]["coverage90"],
            "169c_width90": result["model_169c_best"]["teacher_forced"]["width90"],
            "193a_best_width90": result["model_193a_best"]["teacher_forced"]["width90"],
            "193a_final_width90": result["model_193a_final"]["teacher_forced"]["width90"],
        },
        "rollout_overcoverage": {
            "169c_cov90": result["model_169c_best"]["rollout"]["coverage90"],
            "193a_best_cov90": result["model_193a_best"]["rollout"]["coverage90"],
            "193a_final_cov90": result["model_193a_final"]["rollout"]["coverage90"],
            "169c_width90": result["model_169c_best"]["rollout"]["width90"],
            "193a_best_width90": result["model_193a_best"]["rollout"]["width90"],
            "193a_final_width90": result["model_193a_final"]["rollout"]["width90"],
        },
        "193a_factor_usage": {
            "best_teacher_factor_share": result["model_193a_best"]["teacher_forced"]["factor_var_share"],
            "best_rollout_factor_share": result["model_193a_best"]["rollout"]["factor_var_share"],
            "best_teacher_high_var_prob": result["model_193a_best"]["teacher_forced"].get("high_var_component_prob"),
            "best_rollout_high_var_prob": result["model_193a_best"]["rollout"].get("high_var_component_prob"),
            "best_rollout_sampled_high_var_rate": result["model_193a_best"]["rollout"].get("sampled_high_var_component_rate"),
            "final_teacher_factor_share": result["model_193a_final"]["teacher_forced"]["factor_var_share"],
            "final_rollout_factor_share": result["model_193a_final"]["rollout"]["factor_var_share"],
            "final_teacher_high_var_prob": result["model_193a_final"]["teacher_forced"].get("high_var_component_prob"),
            "final_rollout_high_var_prob": result["model_193a_final"]["rollout"].get("high_var_component_prob"),
            "final_rollout_sampled_high_var_rate": result["model_193a_final"]["rollout"].get("sampled_high_var_component_rate"),
        },
        "rollout_structure": {
            "169c_corr_ratio_h30": result["model_169c_best"]["rollout"]["delta_corr_rank"]["corr_ratio_h30"],
            "169c_rank_ratio_h30": result["model_169c_best"]["rollout"]["delta_corr_rank"]["rank_ratio_h30"],
            "193a_best_corr_ratio_h30": result["model_193a_best"]["rollout"]["delta_corr_rank"]["corr_ratio_h30"],
            "193a_best_rank_ratio_h30": result["model_193a_best"]["rollout"]["delta_corr_rank"]["rank_ratio_h30"],
            "193a_final_corr_ratio_h30": result["model_193a_final"]["rollout"]["delta_corr_rank"]["corr_ratio_h30"],
            "193a_final_rank_ratio_h30": result["model_193a_final"]["rollout"]["delta_corr_rank"]["rank_ratio_h30"],
        },
    }

    payload = {"summary": summary, "details": result}
    out_path = output_dir / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(payload), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
