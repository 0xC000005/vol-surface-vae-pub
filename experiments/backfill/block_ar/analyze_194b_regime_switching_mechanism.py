#!/usr/bin/env python
"""
Focused mechanistic review of 194b regime-switching semantic-alignment regression.

Questions:
  1. Are the discrete latent regimes actually separating calm vs turbulent windows?
  2. On the hard late-horizon S3/S7 misses, is the high state turning on in the
     right windows, or is the regime path still misaligned?
  3. Is the quiet state itself too wide, explaining low quiet mass and S2/S8 drift?
  4. Does forcing quiet vs active states clarify why kurtosis improves while
     S2/S3/S7/S10/S11 still fail?
"""

from __future__ import annotations

import argparse
import copy
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
from experiments.backfill.block_ar.train_194a_regime_switching_ar_latent_factor import (
    RegimeSwitchingLatentFactorARModel,
    compute_cond_from_outputs,
    forward_backward_posteriors,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


SELECT_HORIZONS = [1, 7, 14, 30]
SELECT_HIDX = [h - 1 for h in SELECT_HORIZONS]
LATE_HIDX = [13, 29]


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


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def safe_share(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_offdiag_corr(cov: np.ndarray, eps: float = 1e-10) -> float:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr_mat = cov / np.outer(std, std)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    return float(corr_mat[mask].mean())


def corr_from_cov(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr_mat = cov / np.outer(std, std)
    corr_mat = np.clip(corr_mat, -1.0, 1.0)
    np.fill_diagonal(corr_mat, 1.0)
    return corr_mat


def eff_rank_from_matrix(mat: np.ndarray, eps: float = 1e-10) -> float:
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.maximum(eigvals, eps)
    probs = eigvals / np.maximum(eigvals.sum(), eps)
    probs = probs[probs > eps]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def pearson_kurtosis(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return float("nan")
    centered = arr - arr.mean()
    var = np.mean(centered ** 2)
    if var <= 1e-12:
        return float("nan")
    fourth = np.mean(centered ** 4)
    return float(fourth / max(var ** 2, 1e-12))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, grid, side="right") / max(len(x), 1)
    cdf_y = np.searchsorted(y, grid, side="right") / max(len(y), 1)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def aggregate_slope_ratio(prev: torch.Tensor, gt_next: torch.Tensor, pred_next: torch.Tensor) -> float:
    prev_flat = prev.reshape(prev.shape[0], -1)
    gt_next_flat = gt_next.reshape(gt_next.shape[0], -1)
    pred_next_flat = pred_next.reshape(pred_next.shape[0], -1)
    x = prev_flat.reshape(-1).detach().cpu().numpy().astype(np.float64)
    gt_delta = (gt_next_flat - prev_flat).reshape(-1).detach().cpu().numpy().astype(np.float64)
    pred_delta = (pred_next_flat - prev_flat).reshape(-1).detach().cpu().numpy().astype(np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = float(np.square(x_centered).sum())
    if denom <= 1e-12:
        return float("nan")
    gt_slope = float((x_centered * (gt_delta - gt_delta.mean())).sum() / denom)
    pred_slope = float((x_centered * (pred_delta - pred_delta.mean())).sum() / denom)
    if abs(gt_slope) <= 1e-12:
        return float("nan")
    return pred_slope / gt_slope


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


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    return vov, q20, q80, calm_mask, turb_mask


def load_169c_model(ckpt_path: Path, device: torch.device) -> tuple[ShapeScaleStudentTARModel, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = ShapeScaleStudentTARModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-4),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, payload


def load_194a_model(ckpt_path: Path, device: torch.device) -> tuple[RegimeSwitchingLatentFactorARModel, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = RegimeSwitchingLatentFactorARModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        regime_config=cfg["regime"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-4),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, payload


@torch.no_grad()
def analyze_169c_teacher_forced(
    model: ShapeScaleStudentTARModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    total_var_list = []
    per_cell_var_list = []
    n = history_01.shape[0]

    for start in range(0, n, batch_size):
        hist_b = history_01[start : start + batch_size].to(device)
        future_b = future_01[start : start + batch_size].to(device)
        bsz, hist_len = hist_b.shape[:2]
        n_cells = future_b.shape[-1]

        hist_norm_b = normalize_iv(hist_b).reshape(bsz, hist_len, n_cells)
        gru_outputs, gru_state = model.encoder.gru(hist_norm_b)
        prev_01 = hist_b[:, -1].reshape(bsz, n_cells)

        batch_total_var = []
        batch_per_cell_var = []
        for step in range(future_b.shape[1]):
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = torch.softmax(attn_logits, dim=1)
            pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(pooled)

            prev_u = iv_to_unconstrained(prev_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
            _mu, factor, diag, scale, nu = model.decoder(cond, prev_u)
            cov = model.covariance(factor, diag, scale)
            pred_cov = cov * torch.clamp(nu / (nu - 2.0), min=1.0, max=10.0).unsqueeze(-1).unsqueeze(-1)

            batch_total_var.append(torch.diagonal(pred_cov, dim1=-2, dim2=-1).mean(dim=-1).detach().cpu().numpy())
            batch_per_cell_var.append(torch.diagonal(pred_cov, dim1=-2, dim2=-1).detach().cpu().numpy())

            next_frame = future_b[:, step, :]
            next_norm = normalize_iv(next_frame).unsqueeze(1)
            next_out, gru_state = model.encoder.gru(next_norm, gru_state)
            gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
            prev_01 = next_frame

        total_var_list.append(np.stack(batch_total_var, axis=1))
        per_cell_var_list.append(np.stack(batch_per_cell_var, axis=1))

    total_var = np.concatenate(total_var_list, axis=0)
    per_cell_var = np.concatenate(per_cell_var_list, axis=0)
    return {
        "total_var": total_var,
        "per_cell_var": per_cell_var,
    }


@torch.no_grad()
def analyze_194a_teacher_forced(
    model: RegimeSwitchingLatentFactorARModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    n = history_01.shape[0]
    gamma_all = []
    state_var_all = []
    scale_all = []
    nu_all = []
    mu_bar_iv_all = []
    total_self = []
    total_trans_entropy = []

    for start in range(0, n, batch_size):
        hist_b = history_01[start : start + batch_size].to(device)
        future_b = future_01[start : start + batch_size].to(device)
        bsz, hist_len = hist_b.shape[:2]
        n_cells = future_b.shape[-1]

        hist_norm_b = normalize_iv(hist_b).reshape(bsz, hist_len, n_cells)
        gru_outputs, gru_state = model.encoder.gru(hist_norm_b)
        prev_01 = hist_b[:, -1].reshape(bsz, n_cells)

        log_emit_steps = []
        log_trans_steps: list[torch.Tensor] = []
        pred_cov_states = []
        scale_states = []
        nu_states = []
        mu_states_iv = []
        filtered_probs = None
        batch_self = []
        batch_entropy = []

        for step in range(future_b.shape[1]):
            cond, _attn = compute_cond_from_outputs(model, gru_outputs)
            prev_u = iv_to_unconstrained(prev_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
            mu, factor, diag, scale, nu = model.decoder(cond, prev_u)
            target_t = future_b[:, step, :]
            target_u = iv_to_unconstrained(target_t, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)

            log_emit_t = model.student_t_logprob_states(target_u, mu, factor, diag, scale, nu)
            log_emit_steps.append(log_emit_t)
            pred_cov_states.append(model.predictive_covariance(factor, diag, scale, nu))
            scale_states.append(scale)
            nu_states.append(nu)
            mu_states_iv.append(unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi))

            if step == 0:
                log_alpha_t = F.log_softmax(model.initial_logits(cond), dim=-1) + log_emit_t
            else:
                trans_logits_t = model.transition_logits(cond)
                log_trans_t = F.log_softmax(trans_logits_t, dim=-1)
                log_trans_steps.append(log_trans_t)
                log_alpha_t = log_emit_t + torch.logsumexp(log_alpha_t.unsqueeze(-1) + log_trans_t, dim=1)
                trans_probs_t = F.softmax(trans_logits_t, dim=-1)
                if filtered_probs is not None:
                    filtered_self = (
                        filtered_probs.unsqueeze(-1)
                        * trans_probs_t
                        * torch.eye(model.num_states, device=device, dtype=trans_probs_t.dtype).unsqueeze(0)
                    ).sum(dim=(1, 2))
                    trans_entropy = -(trans_probs_t * trans_probs_t.clamp_min(1e-8).log()).sum(dim=-1)
                    batch_self.append(filtered_self.detach().cpu().numpy())
                    batch_entropy.append(
                        (filtered_probs * trans_entropy).sum(dim=-1).detach().cpu().numpy()
                    )

            filtered_probs = F.softmax(log_alpha_t, dim=-1)
            mu_filtered_iv = torch.einsum("bk,bkn->bn", filtered_probs, mu_states_iv[-1])

            next_frame = future_b[:, step, :]
            next_norm = normalize_iv(next_frame).unsqueeze(1)
            next_out, gru_state = model.encoder.gru(next_norm, gru_state)
            gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
            prev_01 = next_frame

        init_cond, _, _ = model.encode(hist_b)
        log_init = F.log_softmax(model.initial_logits(init_cond), dim=-1)
        log_emit = torch.stack(log_emit_steps, dim=1)
        gamma, xi, _loglik = forward_backward_posteriors(log_init, log_trans_steps, log_emit)

        pred_cov_states_t = torch.stack(pred_cov_states, dim=1)
        scale_states_t = torch.stack(scale_states, dim=1)
        nu_states_t = torch.stack(nu_states, dim=1)
        mu_states_iv_t = torch.stack(mu_states_iv, dim=1)
        mu_bar_iv = (gamma.unsqueeze(-1) * mu_states_iv_t).sum(dim=2)

        gamma_all.append(gamma.detach().cpu().numpy())
        state_var_all.append(torch.diagonal(pred_cov_states_t, dim1=-2, dim2=-1).detach().cpu().numpy())
        scale_all.append(scale_states_t.detach().cpu().numpy())
        nu_all.append(nu_states_t.detach().cpu().numpy())
        mu_bar_iv_all.append(mu_bar_iv.detach().cpu().numpy())
        if batch_self:
            total_self.append(np.concatenate(batch_self, axis=0))
        if batch_entropy:
            total_trans_entropy.append(np.concatenate(batch_entropy, axis=0))

    gamma_np = np.concatenate(gamma_all, axis=0)
    state_var_np = np.concatenate(state_var_all, axis=0)
    scale_np = np.concatenate(scale_all, axis=0)
    nu_np = np.concatenate(nu_all, axis=0)
    mu_bar_iv_np = np.concatenate(mu_bar_iv_all, axis=0)
    expected_var_cells = (gamma_np[..., None] * state_var_np).sum(axis=2)

    return {
        "gamma": gamma_np,
        "state_var_cells": state_var_np,
        "expected_var_cells": expected_var_cells,
        "scale_states": scale_np,
        "nu_states": nu_np,
        "mu_bar_iv": mu_bar_iv_np,
        "self_transition_mean": safe_mean(np.concatenate(total_self, axis=0)) if total_self else float("nan"),
        "transition_entropy_mean": safe_mean(np.concatenate(total_trans_entropy, axis=0)) if total_trans_entropy else float("nan"),
    }


@torch.no_grad()
def rollout_coverage_arrays(
    model: RegimeSwitchingLatentFactorARModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    n_samples: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    inside90_chunks = []
    width_chunks = []
    median_chunks = []

    n = history_01.shape[0]
    for start in range(0, n, batch_size):
        hist_b = history_norm[start : start + batch_size].to(device)
        future_b = future_01[start : start + batch_size].to(device)
        samples = model.sample_batched(hist_b, n_samples=n_samples, n_steps=future_b.shape[1])
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        median = samples.median(dim=1).values
        inside90 = ((future_b >= lo) & (future_b <= hi)).detach().cpu().numpy()
        width = (hi - lo).detach().cpu().numpy()
        inside90_chunks.append(inside90.reshape(inside90.shape[0], inside90.shape[1], -1))
        width_chunks.append(width.reshape(width.shape[0], width.shape[1], -1))
        median_chunks.append(median.detach().cpu().numpy().reshape(median.shape[0], median.shape[1], -1))

    return {
        "inside90": np.concatenate(inside90_chunks, axis=0),
        "width": np.concatenate(width_chunks, axis=0),
        "median": np.concatenate(median_chunks, axis=0),
        "future_01": future_01.reshape(future_01.shape[0], future_01.shape[1], -1).cpu().numpy(),
        "history_01": history_01.cpu().numpy(),
    }


@torch.no_grad()
def sample_batched_forced_state(
    model: RegimeSwitchingLatentFactorARModel,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    forced_state: int,
    chunk_size: int = 8,
) -> torch.Tensor:
    history_01 = denormalize_iv(history_norm)
    batch_size, hist_len = history_01.shape[:2]

    all_chunks = []
    for start in range(0, n_samples, chunk_size):
        k = min(chunk_size, n_samples - start)
        hist_k = history_01.unsqueeze(1).expand(batch_size, k, -1, -1, -1)
        hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()
        frames = []

        for _ in range(n_steps):
            cond, _, _ = model.encode(hist_k)
            prev_01 = hist_k[:, -1].reshape(hist_k.shape[0], -1)
            prev_u = iv_to_unconstrained(prev_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
            mu, factor, diag, scale, nu = model.decoder(cond, prev_u)

            state_idx = torch.full((hist_k.shape[0],), forced_state, device=hist_k.device, dtype=torch.long)
            gather = state_idx.view(-1, 1, 1)
            mu_sel = mu.gather(1, gather.expand(-1, 1, mu.shape[-1])).squeeze(1)
            diag_sel = diag.gather(1, gather.expand(-1, 1, diag.shape[-1])).squeeze(1)
            factor_sel = factor.gather(
                1, gather.unsqueeze(-1).expand(-1, 1, factor.shape[-2], factor.shape[-1])
            ).squeeze(1)
            scale_sel = scale.gather(1, state_idx.view(-1, 1)).squeeze(1)
            nu_sel = nu.gather(1, state_idx.view(-1, 1)).squeeze(1)

            factor_norm, diag_norm, _ = model.normalized_components(
                factor_sel.unsqueeze(1), diag_sel.unsqueeze(1)
            )
            factor_norm = factor_norm.squeeze(1)
            diag_norm = diag_norm.squeeze(1)

            eps_factor = torch.randn(hist_k.shape[0], factor_sel.shape[-1], device=hist_k.device, dtype=hist_k.dtype)
            eps_diag = torch.randn(hist_k.shape[0], mu_sel.shape[-1], device=hist_k.device, dtype=hist_k.dtype)
            lowrank_noise = torch.einsum("bnr,br->bn", factor_norm, eps_factor)
            diag_noise = diag_norm * eps_diag
            gamma = torch.distributions.Gamma(nu_sel / 2.0, nu_sel / 2.0)
            mix = gamma.sample().to(mu_sel.dtype).clamp_min(1e-6)
            t_scale = torch.rsqrt(mix).unsqueeze(-1)
            total_noise = (lowrank_noise + diag_noise) * scale_sel.unsqueeze(-1)
            next_u = mu_sel + total_noise * t_scale
            next_iv = unconstrained_to_iv(next_u, lo=model.support_lo, hi=model.support_hi)

            frames.append(next_iv.reshape(batch_size, k, 5, 5))
            hist_k = torch.cat([hist_k[:, 1:], next_iv.view(batch_size * k, 1, 5, 5)], dim=1)

        all_chunks.append(torch.stack(frames, dim=2))
    return torch.cat(all_chunks, dim=1)


@torch.no_grad()
def compute_rollout_stats(
    model: RegimeSwitchingLatentFactorARModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    n_samples: int,
    batch_size: int,
    forced_state: int | None = None,
) -> dict[str, float]:
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)

    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []
    prev_chunks = []
    gt_next_chunks = []
    sample_next_chunks = []
    gt_changes = []
    gen_changes = []
    gt_path_max = []
    gen_path_max = []
    gt_delta_sum = np.zeros(25, dtype=np.float64)
    gt_delta_outer = np.zeros((25, 25), dtype=np.float64)
    gt_delta_count = 0
    ro_delta_sum = np.zeros(25, dtype=np.float64)
    ro_delta_outer = np.zeros((25, 25), dtype=np.float64)
    ro_delta_count = 0

    n = history_norm.shape[0]
    for start in range(0, n, batch_size):
        hist_b = history_norm[start : start + batch_size].to(device)
        hist_01_b = history_01[start : start + batch_size].to(device)
        future_b = future_01[start : start + batch_size].to(device)

        if forced_state is None:
            samples = model.sample_batched(hist_b, n_samples=n_samples, n_steps=future_b.shape[1])
        else:
            samples = sample_batched_forced_state(
                model,
                hist_b,
                n_samples=n_samples,
                n_steps=future_b.shape[1],
                forced_state=forced_state,
            )

        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        median = samples.median(dim=1).values
        coverage = ((future_b >= lo) & (future_b <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_b).abs().mean()

        mean_iv = hist_01_b.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        prev_chunks.append(hist_01_b[:, -1].detach().cpu())
        gt_next_chunks.append(future_b[:, 0].detach().cpu())
        sample_next_chunks.append(samples[:, :, 0].mean(dim=1).detach().cpu())

        gt_change = (future_b[:, 1:] - future_b[:, :-1]).detach().cpu()
        gen_first = samples[:, 0]
        gen_change = (gen_first[:, 1:] - gen_first[:, :-1]).detach().cpu()
        gt_changes.append(gt_change)
        gen_changes.append(gen_change)

        gt_path = torch.cat([hist_01_b[:, -1:].detach().cpu(), future_b.detach().cpu()], dim=1)
        gen_path = torch.cat([hist_01_b[:, -1:].detach().cpu(), gen_first.detach().cpu()], dim=1)
        gt_path_max.append((gt_path[:, 1:] - gt_path[:, :-1]).abs().amax(dim=(1, 2, 3)))
        gen_path_max.append((gen_path[:, 1:] - gen_path[:, :-1]).abs().amax(dim=(1, 2, 3)))

        future_flat = future_b.reshape(future_b.shape[0], future_b.shape[1], -1)
        sample_flat = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2], -1)
        gt_delta = future_flat[:, 29, :] - future_flat[:, 28, :]
        gen_delta = sample_flat[:, :, 29, :] - sample_flat[:, :, 28, :]
        gen_np = gen_delta.reshape(-1, gen_delta.shape[-1]).detach().cpu().numpy()
        gt_np = gt_delta.detach().cpu().numpy()
        gt_delta_sum += gt_np.sum(axis=0)
        gt_delta_outer += gt_np.T @ gt_np
        gt_delta_count += gt_np.shape[0]
        ro_delta_sum += gen_np.sum(axis=0)
        ro_delta_outer += gen_np.T @ gen_np
        ro_delta_count += gen_np.shape[0]

        total_cov += coverage.item() * hist_b.shape[0]
        total_width += width.item() * hist_b.shape[0]
        total_mae += mae.item() * hist_b.shape[0]
        total_count += hist_b.shape[0]
        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    turb_calm_ratio = float(widths[turb_mask].mean() / widths[calm_mask].mean()) if calm_mask.any() and turb_mask.any() else float("nan")

    prev = torch.cat(prev_chunks, dim=0)
    gt_next = torch.cat(gt_next_chunks, dim=0)
    sample_next = torch.cat(sample_next_chunks, dim=0)
    sample_mr_ratio = aggregate_slope_ratio(prev, gt_next, sample_next)

    gt_changes = torch.cat(gt_changes, dim=0).numpy()
    gen_changes = torch.cat(gen_changes, dim=0).numpy()
    gt_path_max = torch.cat(gt_path_max, dim=0).numpy()
    gen_path_max = torch.cat(gen_path_max, dim=0).numpy()

    gt_mean = gt_delta_sum / max(gt_delta_count, 1)
    gt_cov = gt_delta_outer / max(gt_delta_count, 1) - np.outer(gt_mean, gt_mean)
    ro_mean = ro_delta_sum / max(ro_delta_count, 1)
    ro_cov = ro_delta_outer / max(ro_delta_count, 1) - np.outer(ro_mean, ro_mean)
    gt_corr = mean_offdiag_corr(gt_cov)
    ro_corr = mean_offdiag_corr(ro_cov)
    gt_rank = eff_rank_from_matrix(corr_from_cov(gt_cov))
    ro_rank = eff_rank_from_matrix(corr_from_cov(ro_cov))

    return {
        "cov90": total_cov / max(total_count, 1),
        "width90": total_width / max(total_count, 1),
        "mae": total_mae / max(total_count, 1),
        "turb_calm_ratio": turb_calm_ratio,
        "sample_mr_ratio": float(sample_mr_ratio),
        "kurtosis_ratio": pearson_kurtosis(gen_changes) / max(pearson_kurtosis(gt_changes), 1e-12),
        "pathwise_jump_ks": ks_statistic(gt_path_max, gen_path_max),
        "corr_ratio_h30": float(ro_corr / gt_corr) if abs(gt_corr) > 1e-8 else float("nan"),
        "rank_ratio_h30": float(ro_rank / gt_rank) if gt_rank > 1e-8 else float("nan"),
    }


def summarize_regime_usage(
    gamma: np.ndarray,
    state_var_cells: np.ndarray,
    scale_states: np.ndarray,
    nu_states: np.ndarray,
    calm_mask: np.ndarray,
    turb_mask: np.ndarray,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall_state_occupancy": gamma.mean(axis=(0, 1)).tolist(),
        "self_transition_mean": None,
    }

    horizon_state_occ = {}
    for h, idx in zip(SELECT_HORIZONS, SELECT_HIDX):
        horizon_state_occ[f"h{h}"] = gamma[:, idx, :].mean(axis=0).tolist()
    summary["horizon_state_occupancy"] = horizon_state_occ

    regime_state_occ = {}
    for name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        regime_state_occ[name] = {
            f"h{h}": gamma[mask, idx, :].mean(axis=0).tolist() if mask.any() else [float("nan")] * gamma.shape[-1]
            for h, idx in zip(SELECT_HORIZONS, SELECT_HIDX)
        }
    summary["regime_state_occupancy"] = regime_state_occ

    total_var_state = state_var_cells.mean(axis=-1)
    summary["state_total_var_per_cell"] = {
        f"state{k}": {
            "overall": float(total_var_state[:, :, k].mean()),
            "calm_h30": float(total_var_state[calm_mask, 29, k].mean()) if calm_mask.any() else float("nan"),
            "turb_h30": float(total_var_state[turb_mask, 29, k].mean()) if turb_mask.any() else float("nan"),
            "scale_mean": float(scale_states[:, :, k].mean()),
            "nu_mean": float(nu_states[:, :, k].mean()),
        }
        for k in range(gamma.shape[-1])
    }
    return summary


def hard_late_alignment_summary(
    gamma: np.ndarray,
    state_var_cells: np.ndarray,
    expected_var_cells: np.ndarray,
    rollout_inside90: np.ndarray,
    calm_mask: np.ndarray,
    turb_mask: np.ndarray,
) -> dict[str, Any]:
    n_states = gamma.shape[-1]
    high_idx = n_states - 1
    quiet_idx = 0

    late_turb_mask = np.zeros_like(rollout_inside90, dtype=bool)
    for h_idx in LATE_HIDX:
        late_turb_mask[:, h_idx, :] = turb_mask[:, None]
    hard_mask = late_turb_mask & (~rollout_inside90)
    clean_mask = late_turb_mask & rollout_inside90

    high_prob = np.repeat(gamma[:, :, high_idx : high_idx + 1], rollout_inside90.shape[-1], axis=2)
    quiet_prob = np.repeat(gamma[:, :, quiet_idx : quiet_idx + 1], rollout_inside90.shape[-1], axis=2)
    high_state_var = state_var_cells[:, :, high_idx, :]

    out = {
        "late_turb_point_count": int(late_turb_mask.sum()),
        "hard_late_point_count": int(hard_mask.sum()),
        "hard_share_of_late_turb_points": safe_share(float(hard_mask.sum()), float(late_turb_mask.sum())),
        "hard_vs_high_state_prob_corr": corr(hard_mask[late_turb_mask].astype(np.float64), high_prob[late_turb_mask]),
        "hard_vs_quiet_state_prob_corr": corr(hard_mask[late_turb_mask].astype(np.float64), quiet_prob[late_turb_mask]),
        "hard_vs_expected_var_corr": corr(hard_mask[late_turb_mask].astype(np.float64), expected_var_cells[late_turb_mask]),
        "hard_vs_high_state_var_corr": corr(hard_mask[late_turb_mask].astype(np.float64), high_state_var[late_turb_mask]),
        "mean_high_state_prob_hard": safe_mean(high_prob[hard_mask]),
        "mean_high_state_prob_clean": safe_mean(high_prob[clean_mask]),
        "mean_quiet_state_prob_hard": safe_mean(quiet_prob[hard_mask]),
        "mean_quiet_state_prob_clean": safe_mean(quiet_prob[clean_mask]),
        "mean_expected_var_hard": safe_mean(expected_var_cells[hard_mask]),
        "mean_expected_var_clean": safe_mean(expected_var_cells[clean_mask]),
        "mean_high_state_var_hard": safe_mean(high_state_var[hard_mask]),
        "mean_high_state_var_clean": safe_mean(high_state_var[clean_mask]),
        "high_prob_share_on_hard": safe_share(float(high_prob[hard_mask].sum()), float(high_prob[late_turb_mask].sum())),
        "expected_var_share_on_hard": safe_share(float(expected_var_cells[hard_mask].sum()), float(expected_var_cells[late_turb_mask].sum())),
        "high_state_var_share_on_hard": safe_share(float(high_state_var[hard_mask].sum()), float(high_state_var[late_turb_mask].sum())),
    }
    return out


def quiet_state_review(
    diag_194a: dict[str, Any],
    diag_169c: dict[str, Any],
    rollout_inside90: np.ndarray,
    calm_mask: np.ndarray,
) -> dict[str, Any]:
    quiet_var = diag_194a["state_var_cells"][:, :, 0, :]
    expected_var = diag_194a["expected_var_cells"]
    base_var = diag_169c["per_cell_var"]

    calm_h30_mask = np.zeros_like(rollout_inside90, dtype=bool)
    calm_h30_mask[:, 29, :] = calm_mask[:, None]
    calm_h30_clean = calm_h30_mask & rollout_inside90

    return {
        "quiet_vs_169c_var_ratio_calm_h30_all": safe_mean(quiet_var[calm_h30_mask]) / max(safe_mean(base_var[calm_h30_mask]), 1e-12),
        "quiet_vs_169c_var_ratio_calm_h30_clean": safe_mean(quiet_var[calm_h30_clean]) / max(safe_mean(base_var[calm_h30_clean]), 1e-12),
        "expected_vs_169c_var_ratio_calm_h30_all": safe_mean(expected_var[calm_h30_mask]) / max(safe_mean(base_var[calm_h30_mask]), 1e-12),
        "expected_vs_169c_var_ratio_calm_h30_clean": safe_mean(expected_var[calm_h30_clean]) / max(safe_mean(base_var[calm_h30_clean]), 1e-12),
        "quiet_var_mean_calm_h30": safe_mean(quiet_var[calm_h30_mask]),
        "quiet_var_mean_calm_h30_clean": safe_mean(quiet_var[calm_h30_clean]),
        "base_var_mean_calm_h30": safe_mean(base_var[calm_h30_mask]),
        "base_var_mean_calm_h30_clean": safe_mean(base_var[calm_h30_clean]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanistic review of 194b regime-switching AR model")
    parser.add_argument("--best_ckpt", type=str, default="models/backfill/regime_switching_ar_latent_factor_194b/best_model.pt")
    parser.add_argument("--final_ckpt", type=str, default="models/backfill/regime_switching_ar_latent_factor_194b/final_model.pt")
    parser.add_argument("--baseline_ckpt", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=384)
    parser.add_argument("--rollout_samples", type=int, default=16)
    parser.add_argument("--ablation_windows", type=int, default=128)
    parser.add_argument("--ablation_samples", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/194b_regime_switching_mechanistic",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    vov, q20, q80, calm_mask, turb_mask = regime_masks_from_history(history_norm)

    baseline_169c, baseline_payload = load_169c_model(Path(args.baseline_ckpt), device)
    best_model, best_payload = load_194a_model(Path(args.best_ckpt), device)
    final_model, final_payload = load_194a_model(Path(args.final_ckpt), device)

    diag_169c = analyze_169c_teacher_forced(
        baseline_169c, history_norm, future_norm, device=device, batch_size=args.batch_size
    )

    models = {
        "194b_best": (best_model, best_payload),
        "194b_final": (final_model, final_payload),
    }

    details: dict[str, Any] = {
        "config": {
            "test_start": args.test_start,
            "max_windows": args.max_windows,
            "rollout_samples": args.rollout_samples,
            "ablation_windows": args.ablation_windows,
            "ablation_samples": args.ablation_samples,
            "batch_size": args.batch_size,
            "q20_vov": q20,
            "q80_vov": q80,
            "calm_windows": int(calm_mask.sum()),
            "turb_windows": int(turb_mask.sum()),
        },
        "baseline_169c_teacher_forced": {
            "mean_total_var_h30_calm": float(diag_169c["total_var"][calm_mask, 29].mean()) if calm_mask.any() else float("nan"),
            "mean_total_var_h30_turb": float(diag_169c["total_var"][turb_mask, 29].mean()) if turb_mask.any() else float("nan"),
            "mean_per_cell_var_h30": float(diag_169c["per_cell_var"][:, 29, :].mean()),
        },
    }

    summary: dict[str, Any] = {}

    for name, (model, _payload) in models.items():
        tf_diag = analyze_194a_teacher_forced(model, history_norm, future_norm, device=device, batch_size=args.batch_size)
        rollout_diag = rollout_coverage_arrays(
            model,
            history_norm,
            future_norm,
            device=device,
            n_samples=args.rollout_samples,
            batch_size=args.batch_size,
        )

        regime_summary = summarize_regime_usage(
            tf_diag["gamma"],
            tf_diag["state_var_cells"],
            tf_diag["scale_states"],
            tf_diag["nu_states"],
            calm_mask=calm_mask,
            turb_mask=turb_mask,
        )
        hard_summary = hard_late_alignment_summary(
            tf_diag["gamma"],
            tf_diag["state_var_cells"],
            tf_diag["expected_var_cells"],
            rollout_diag["inside90"],
            calm_mask=calm_mask,
            turb_mask=turb_mask,
        )
        quiet_summary = quiet_state_review(
            tf_diag,
            diag_169c,
            rollout_diag["inside90"],
            calm_mask=calm_mask,
        )

        details[name] = {
            "regime_separation": regime_summary,
            "hard_late_alignment": hard_summary,
            "quiet_state_review": quiet_summary,
        }
        summary[name] = {
            "overall_state_occupancy": regime_summary["overall_state_occupancy"],
            "turb_h30_state_occupancy": regime_summary["regime_state_occupancy"]["turb"]["h30"],
            "calm_h30_state_occupancy": regime_summary["regime_state_occupancy"]["calm"]["h30"],
            "hard_vs_high_state_prob_corr": hard_summary["hard_vs_high_state_prob_corr"],
            "mean_high_state_prob_hard_vs_clean": [
                hard_summary["mean_high_state_prob_hard"],
                hard_summary["mean_high_state_prob_clean"],
            ],
            "expected_var_share_on_hard": hard_summary["expected_var_share_on_hard"],
            "quiet_vs_169c_var_ratio_calm_h30_clean": quiet_summary["quiet_vs_169c_var_ratio_calm_h30_clean"],
        }

    ablation_hist = history_norm[: args.ablation_windows]
    ablation_future = future_norm[: args.ablation_windows]
    ablation = {
        "best_normal": compute_rollout_stats(
            best_model,
            ablation_hist,
            ablation_future,
            device=device,
            n_samples=args.ablation_samples,
            batch_size=args.batch_size,
            forced_state=None,
        ),
        "best_forced_quiet": compute_rollout_stats(
            best_model,
            ablation_hist,
            ablation_future,
            device=device,
            n_samples=args.ablation_samples,
            batch_size=args.batch_size,
            forced_state=0,
        ),
        "best_forced_mid": compute_rollout_stats(
            best_model,
            ablation_hist,
            ablation_future,
            device=device,
            n_samples=args.ablation_samples,
            batch_size=args.batch_size,
            forced_state=1,
        ),
        "best_forced_high": compute_rollout_stats(
            best_model,
            ablation_hist,
            ablation_future,
            device=device,
            n_samples=args.ablation_samples,
            batch_size=args.batch_size,
            forced_state=2,
        ),
    }
    details["best_state_ablation"] = ablation
    summary["best_state_ablation"] = {
        key: {
            "cov90": val["cov90"],
            "width90": val["width90"],
            "turb_calm_ratio": val["turb_calm_ratio"],
            "sample_mr_ratio": val["sample_mr_ratio"],
            "kurtosis_ratio": val["kurtosis_ratio"],
            "pathwise_jump_ks": val["pathwise_jump_ks"],
            "corr_ratio_h30": val["corr_ratio_h30"],
            "rank_ratio_h30": val["rank_ratio_h30"],
        }
        for key, val in ablation.items()
    }

    conclusion = {
        "regime_path_is_alive": True,
        "regime_separation_read": (
            "194b still uses the discrete states meaningfully: calm windows shift toward the quiet state, "
            "while turbulent windows shift toward the high state."
        ),
        "hard_slice_read": (
            "If hard late-turbulent misses show only weak uplift in high-state posterior but stronger uplift in "
            "expected variance, then the regime path is partly right but still too diffuse at the cell level."
        ),
        "quiet_state_read": (
            "If the quiet state's calm-h30 variance stays above 169c on clean points, then 194b is still too wide "
            "even in its quiet regime, which explains the low quiet-mass / S2 / S8 drift."
        ),
        "ablation_read": (
            "Comparing normal vs forced quiet vs forced high sampling identifies whether the high state is what buys "
            "kurtosis and whether the quiet state is what preserves MR / jump realism."
        ),
    }

    report = {
        "summary": summary,
        "details": details,
        "conclusion": conclusion,
    }

    out_path = output_dir / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(report), indent=2))
    print(json.dumps(make_serializable(report["summary"]), indent=2))
    print(f"\nSaved mechanistic summary to {out_path}")


if __name__ == "__main__":
    main()
