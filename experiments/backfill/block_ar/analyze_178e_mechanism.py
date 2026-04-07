#!/usr/bin/env python
"""
Focused mechanistic postmortem for 178e_v0.

Questions:
  1. Did the new flow experts leak conditional mean into the residual law and
     thereby break the mean-reverting backbone?
  2. Are the remaining S2/S3/S7 failures still mainly expert-capacity failures,
     or did routing collapse again?
  3. Did 178e improve broad realism by over-broadening easy slices instead of
     fixing the hard turbulent slices?
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
from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    build_test_subset,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    normalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_178e_exact_block_flow_expert_mean_reverting_residual_flow import (
    ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel,
)


SELECT_HORIZONS = [1, 7, 14, 30]
LAYER2_LOW = 0.70
LAYER2_HIGH = 0.95


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "exact_block_flow_expert_mean_reverting_residual_flow_structured_joint_student_t_178e"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


def top_cells(values: np.ndarray, reverse: bool = True, k: int = 5) -> list[dict[str, Any]]:
    flat = []
    for idx, value in enumerate(values.reshape(-1)):
        flat.append({"cell": [int(idx // 5), int(idx % 5)], "value": float(value)})
    flat.sort(key=lambda x: x["value"], reverse=reverse)
    return flat[:k]


def slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = x.mean()
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean
    denom = float(np.square(xc).sum())
    slope = 0.0 if denom <= 1e-12 else float((xc * yc).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    y_hat = intercept + slope * x
    sse = float(np.square(y - y_hat).sum())
    sst = float(np.square(y - y_mean).sum())
    r2 = 1.0 - sse / sst if sst > 1e-12 else 1.0
    return slope, intercept, r2


def summarize_variant(prev: np.ndarray, next_values: np.ndarray, gt_next: np.ndarray, active_threshold: float) -> dict[str, Any]:
    gt_delta = gt_next - prev
    pred_delta = next_values - prev

    gt_slope, gt_intercept, gt_r2 = slope_intercept_r2(prev, gt_delta)
    pred_slope, pred_intercept, pred_r2 = slope_intercept_r2(prev, pred_delta)
    ratio = pred_slope / gt_slope if abs(gt_slope) > 1e-12 else float("nan")

    gt_cell_slopes = np.zeros((5, 5), dtype=np.float64)
    pred_cell_slopes = np.zeros((5, 5), dtype=np.float64)
    cell_ratio = np.full((5, 5), np.nan, dtype=np.float64)
    cell_sign_match = np.zeros((5, 5), dtype=bool)
    cell_pass = np.zeros((5, 5), dtype=bool)

    for i in range(5):
        for j in range(5):
            gt_s, _, _ = slope_intercept_r2(prev[:, i, j], gt_delta[:, i, j])
            pred_s, _, _ = slope_intercept_r2(prev[:, i, j], pred_delta[:, i, j])
            gt_cell_slopes[i, j] = gt_s
            pred_cell_slopes[i, j] = pred_s
            if abs(gt_s) > 1e-12:
                cell_ratio[i, j] = pred_s / gt_s
            cell_sign_match[i, j] = np.sign(gt_s) == np.sign(pred_s)

    active_mask = np.abs(gt_cell_slopes) >= active_threshold
    ratio_mask = np.isfinite(cell_ratio) & (cell_ratio >= 0.50) & (cell_ratio <= 1.50)
    cell_pass = active_mask & cell_sign_match & ratio_mask
    active_count = int(active_mask.sum())
    active_pass_count = int(cell_pass.sum())
    active_pass_rate = active_pass_count / active_count if active_count > 0 else 1.0
    slope_corr = (
        float(np.corrcoef(gt_cell_slopes[active_mask].reshape(-1), pred_cell_slopes[active_mask].reshape(-1))[0, 1])
        if active_count >= 2
        else 1.0
    )

    worst_active_cells = []
    for i in range(5):
        for j in range(5):
            if not active_mask[i, j]:
                continue
            ratio_ij = float(cell_ratio[i, j]) if np.isfinite(cell_ratio[i, j]) else float("nan")
            rel_err = abs(ratio_ij - 1.0) if np.isfinite(ratio_ij) else float("inf")
            worst_active_cells.append(
                {
                    "cell": [i, j],
                    "gt_slope": float(gt_cell_slopes[i, j]),
                    "pred_slope": float(pred_cell_slopes[i, j]),
                    "ratio": ratio_ij,
                    "sign_match": bool(cell_sign_match[i, j]),
                    "pass": bool(cell_pass[i, j]),
                    "relative_error_from_1": float(rel_err),
                }
            )
    worst_active_cells.sort(key=lambda x: x["relative_error_from_1"], reverse=True)

    return {
        "gt_aggregate_slope": gt_slope,
        "gt_aggregate_intercept": gt_intercept,
        "gt_aggregate_r2": gt_r2,
        "pred_aggregate_slope": pred_slope,
        "pred_aggregate_intercept": pred_intercept,
        "pred_aggregate_r2": pred_r2,
        "mr_gt_ratio": ratio,
        "active_pass_rate": active_pass_rate,
        "active_cell_count": active_count,
        "active_pass_count": active_pass_count,
        "active_cell_slope_corr": slope_corr,
        "worst_active_cells": worst_active_cells[:10],
    }


@torch.no_grad()
def exact_block_posterior(
    model: ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel,
    target_u: torch.Tensor,
    mu: torch.Tensor,
    time_factor: torch.Tensor,
    time_diag: torch.Tensor,
    cell_factor: torch.Tensor,
    cell_diag: torch.Tensor,
    scale: torch.Tensor,
    base_local_delta: torch.Tensor,
    block_logits: torch.Tensor,
    block_context: torch.Tensor,
):
    batch, n_frames, n_cells = target_u.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)

    shared_local_delta = model.decoder.build_shared_local_delta(base_local_delta)
    shared_local_scale = torch.exp(0.5 * shared_local_delta)
    diff = (target_u - mu) / shared_local_scale
    white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
    white_shared = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
    white_blocks = white_shared.view(batch, model.decoder.n_blocks, model.decoder.block_len, n_cells)

    factors_bank, logdet_template_cov, _offdiag_rms = model.decoder.build_template_bank()
    log_prior_blocks = F.log_softmax(block_logits, dim=-1)
    logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet_cov = n_cells * logdet_t + n_frames * logdet_c
    logdet_local = 2.0 * torch.log(shared_local_scale).sum(dim=(1, 2))

    posterior_blocks = []
    comp_logprobs = []
    for b in range(model.decoder.n_blocks):
        obs = white_blocks[:, b]
        ctx = block_context[:, b]
        rhs = obs.transpose(1, 2)
        comp_terms = []
        for k, expert in enumerate(model.flow_experts):
            factor = factors_bank[b, k].unsqueeze(0).expand(batch, -1, -1)
            base_block = torch.linalg.solve_triangular(factor, rhs, upper=False).transpose(1, 2)
            base_flat = base_block.reshape(batch, model.block_dim)
            z, flow_logdet = expert(base_flat, ctx)
            base_logprob = model._base_logprob(z)
            comp_terms.append(
                log_prior_blocks[:, b, k]
                + base_logprob
                + flow_logdet
                - 0.5 * (logdet_template_cov[b, k] + logdet_cov + logdet_local)
            )
        comp = torch.stack(comp_terms, dim=-1)
        post = F.softmax(comp - torch.logsumexp(comp, dim=-1, keepdim=True), dim=-1)
        posterior_blocks.append(post)
        comp_logprobs.append(comp)
    return F.softmax(block_logits, dim=-1), torch.stack(posterior_blocks, dim=1), comp_logprobs


@torch.no_grad()
def sample_forced_assignments(
    model: ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    forced_assignments: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        base_local_delta,
        _block_logits,
        block_context,
    ) = model.forward_from_history(history_01)
    batch, n_frames, n_cells = mu.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)
    shared_local_delta = model.decoder.build_shared_local_delta(base_local_delta)
    local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)

    base = torch.distributions.StudentT(df=model.base_nu)
    factors_bank, _logdet_cov, _offdiag_rms = model.decoder.build_template_bank()
    white_blocks = []
    for b in range(model.decoder.n_blocks):
        ctx_b = block_context[:, b].unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
        assign_b = forced_assignments[:, :, b].reshape(batch * n_samples)
        z = base.sample((batch * n_samples, model.block_dim)).to(device=mu.device, dtype=mu.dtype)
        base_flat = torch.zeros_like(z)
        for k, expert in enumerate(model.flow_experts):
            mask = assign_b == k
            if mask.any():
                xk, _ = expert.inverse(z[mask], ctx_b[mask])
                base_flat[mask] = xk
        base_block = base_flat.view(batch * n_samples, model.decoder.block_len, n_cells)
        factors = factors_bank[b][assign_b]
        obs = torch.matmul(factors, base_block.transpose(1, 2)).transpose(1, 2)
        white_blocks.append(obs.view(batch, n_samples, model.decoder.block_len, n_cells))
    routed_white = torch.cat(white_blocks, dim=2)
    temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
    noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
    samples_u = mu.unsqueeze(1) + noise * local_scale
    return unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).view(
        batch, n_samples, n_frames, 5, 5
    )


@torch.no_grad()
def analyze_178e(
    model: ExactBlockFlowExpertMeanRevertingResidualFlowStructuredJointStudentTModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    n_samples: int,
    slice_eval_samples: int,
    expert_probe_samples: int,
    active_slope_threshold: float,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01_all = denormalize_iv(history_norm)
    future_01_all = denormalize_iv(future_norm)
    future_flat_all = future_01_all.reshape(future_01_all.shape[0], future_01_all.shape[1], -1)

    n_windows, future_len, _, _ = future_01_all.shape
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    regime_masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm": calm_mask,
        "turb": turb_mask,
    }

    n_blocks = model.decoder.n_blocks
    n_templates = model.decoder.n_templates

    prior_blocks_all = np.zeros((n_windows, n_blocks, n_templates), dtype=np.float32)
    posterior_blocks_all = np.zeros((n_windows, n_blocks, n_templates), dtype=np.float32)
    prior_argmax_all = np.zeros((n_windows, n_blocks), dtype=np.int64)
    posterior_argmax_all = np.zeros((n_windows, n_blocks), dtype=np.int64)
    prior_entropy_all = np.zeros((n_windows, n_blocks), dtype=np.float32)
    posterior_entropy_all = np.zeros((n_windows, n_blocks), dtype=np.float32)

    det_next_all = np.zeros((n_windows, future_len, 5, 5), dtype=np.float32)
    sample_mean_all = np.zeros((n_windows, future_len, 5, 5), dtype=np.float32)
    sample_residual_mean_all = np.zeros((n_windows, future_len, 5, 5), dtype=np.float32)
    per_window_cov = np.zeros(n_windows, dtype=np.float32)
    covered90 = np.zeros((n_windows, future_len, 5, 5), dtype=bool)

    row0 = 0
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_norm_b = history_norm[start:end]
        hist_01_b = history_01_all[start:end]
        fut_01_b = future_01_all[start:end].to(device)

        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            base_local_delta,
            block_logits,
            block_context,
        ) = model.forward_from_history(hist_01_b)
        target_u = iv_to_unconstrained(
            fut_01_b.reshape(end - start, future_len, -1),
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        prior_blocks, posterior_blocks, _ = exact_block_posterior(
            model,
            target_u,
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            base_local_delta,
            block_logits,
            block_context,
        )
        prior_np = prior_blocks.detach().cpu().numpy()
        post_np = posterior_blocks.detach().cpu().numpy()
        prior_blocks_all[row0:end] = prior_np
        posterior_blocks_all[row0:end] = post_np
        prior_argmax_all[row0:end] = prior_np.argmax(axis=-1)
        posterior_argmax_all[row0:end] = post_np.argmax(axis=-1)
        prior_entropy_all[row0:end] = (-(prior_np * np.log(np.clip(prior_np, 1e-8, 1.0)))).sum(axis=-1)
        posterior_entropy_all[row0:end] = (-(post_np * np.log(np.clip(post_np, 1e-8, 1.0)))).sum(axis=-1)

        det_01 = unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi).reshape(end - start, future_len, 5, 5)
        det_next_all[row0:end] = det_01.detach().cpu().numpy()

        samples_u = model.sample_future_u(hist_01_b, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).view(
            end - start, n_samples, future_len, 5, 5
        )
        sample_mean = samples_01.mean(dim=1)
        sample_mean_all[row0:end] = sample_mean.detach().cpu().numpy()
        sample_residual_mean_all[row0:end] = (sample_mean - det_01).detach().cpu().numpy()

        lo = samples_01.quantile(0.05, dim=1)
        hi = samples_01.quantile(0.95, dim=1)
        covered_b = ((fut_01_b >= lo) & (fut_01_b <= hi)).detach().cpu().numpy()
        covered90[row0:end] = covered_b
        per_window_cov[row0:end] = covered_b.mean(axis=(1, 2, 3))
        row0 = end

    prev_all = history_01_all[:, -1].cpu().numpy()
    gt_all = future_01_all.cpu().numpy()

    mean_reversion = {"deterministic": {}, "sampled": {}}
    mean_shift = {"overall_by_horizon": {}, "by_regime": {}}
    for regime_name, mask in regime_masks.items():
        prev = prev_all[mask]
        gt = gt_all[mask]
        det = det_next_all[mask]
        sampled = sample_mean_all[mask]

        mean_reversion["deterministic"][regime_name] = summarize_variant(prev, det[:, 0], gt[:, 0], active_slope_threshold)
        mean_reversion["sampled"][regime_name] = summarize_variant(prev, sampled[:, 0], gt[:, 0], active_slope_threshold)

        reg_shift = {}
        for h in SELECT_HORIZONS:
            hid = h - 1
            shift_grid = sampled[:, hid] - det[:, hid]
            reg_shift[str(h)] = {
                "mean_abs_shift": float(np.mean(np.abs(shift_grid))),
                "mean_signed_shift": float(np.mean(shift_grid)),
                "top_positive_cells": top_cells(shift_grid.mean(axis=0), reverse=True, k=5),
                "top_negative_cells": top_cells(shift_grid.mean(axis=0), reverse=False, k=5),
            }
        mean_shift["by_regime"][regime_name] = reg_shift

    for h in SELECT_HORIZONS:
        hid = h - 1
        shift_grid = sample_mean_all[:, hid] - det_next_all[:, hid]
        mean_shift["overall_by_horizon"][str(h)] = {
            "mean_abs_shift": float(np.mean(np.abs(shift_grid))),
            "mean_signed_shift": float(np.mean(shift_grid)),
            "top_positive_cells": top_cells(shift_grid.mean(axis=0), reverse=True, k=5),
            "top_negative_cells": top_cells(shift_grid.mean(axis=0), reverse=False, k=5),
        }

    regime_block_usage = {}
    regime_block_post = {}
    regime_entropy = {}
    for name, mask in regime_masks.items():
        regime_block_usage[name] = prior_blocks_all[mask].mean(axis=0).tolist()
        regime_block_post[name] = posterior_blocks_all[mask].mean(axis=0).tolist()
        regime_entropy[name] = {
            "prior_mean": prior_entropy_all[mask].mean(axis=0).tolist(),
            "posterior_mean": posterior_entropy_all[mask].mean(axis=0).tolist(),
        }

    top_assignment_patterns = {"prior_argmax": {}, "posterior_argmax": {}}
    for key, arr in [("prior_argmax", prior_argmax_all), ("posterior_argmax", posterior_argmax_all)]:
        patterns = {}
        for row in arr:
            pat = "-".join(str(int(x)) for x in row.tolist())
            patterns[pat] = patterns.get(pat, 0) + 1
        ranked = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        top_assignment_patterns[key] = [
            {"pattern": pat, "count": int(cnt), "fraction": float(cnt / n_windows)} for pat, cnt in ranked
        ]

    # Template-bank summary in covariance space plus flow-expert probing in whitened block space.
    factors_bank, _logdet_cov, offdiag_rms_bank = model.decoder.build_template_bank()
    factors_bank = factors_bank.detach().cpu().numpy()
    offdiag_rms_bank = offdiag_rms_bank.detach().cpu().numpy()
    template_bank_summary = {}
    for b in range(n_blocks):
        block_name = f"block_{b + 1}"
        template_bank_summary[block_name] = {}
        covs = []
        for t in range(n_templates):
            factor = factors_bank[b, t]
            cov = factor @ factor.T
            covs.append(cov)
            diag_vals = np.diag(cov).reshape(5, 5)
            template_bank_summary[block_name][f"template_{t}"] = {
                "diag_mean": float(np.mean(np.diag(cov))),
                "diag_std": float(np.std(np.diag(cov))),
                "offdiag_rms": float(np.sqrt(np.mean(np.square(cov - np.diag(np.diag(cov)))))),
                "reported_factor_offdiag_rms": float(offdiag_rms_bank[b, t]),
                "top_diag_cells": top_cells(diag_vals, reverse=True, k=5),
                "low_diag_cells": top_cells(diag_vals, reverse=False, k=5),
            }
        dists = {}
        for t1 in range(n_templates):
            for t2 in range(t1 + 1, n_templates):
                dists[f"{t1}-{t2}"] = float(np.sqrt(np.square(covs[t1] - covs[t2]).sum()))
        template_bank_summary[block_name]["pairwise_frobenius"] = dists

    # Probe whether experts are zero-centered in whitened space.
    expert_probe = {}
    probe_per_regime = 96
    base = torch.distributions.StudentT(df=model.base_nu)
    for regime_name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        idx = np.where(mask)[0][:probe_per_regime]
        if len(idx) == 0:
            continue
        hist_01 = history_01_all[idx].to(device)
        (
            _mu,
            _tf,
            _td,
            _cf,
            _cd,
            _scale,
            _base_local_delta,
            _block_logits,
            block_context,
        ) = model.forward_from_history(hist_01)
        expert_probe[regime_name] = {}
        for b in range(n_blocks):
            ctx_b = block_context[:, b]
            rep_ctx = ctx_b.unsqueeze(1).expand(len(idx), expert_probe_samples, -1).reshape(len(idx) * expert_probe_samples, -1)
            z = base.sample((len(idx) * expert_probe_samples, model.block_dim)).to(device=device, dtype=ctx_b.dtype)
            expert_probe[regime_name][f"block_{b + 1}"] = {}
            for t, expert in enumerate(model.flow_experts):
                base_flat, _ = expert.inverse(z, rep_ctx)
                base_block = base_flat.view(len(idx), expert_probe_samples, model.decoder.block_len, model.decoder.n_cells)
                factor = torch.from_numpy(factors_bank[b, t]).to(device=device, dtype=base_block.dtype)
                routed = torch.matmul(
                    factor.unsqueeze(0).unsqueeze(0),
                    base_block.transpose(-1, -2),
                ).transpose(-1, -2)
                expert_probe[regime_name][f"block_{b + 1}"][f"template_{t}"] = {
                    "base_signed_mean": float(base_block.mean().item()),
                    "base_abs_mean": float(base_block.abs().mean().item()),
                    "base_std": float(base_block.std().item()),
                    "routed_signed_mean": float(routed.mean().item()),
                    "routed_abs_mean": float(routed.abs().mean().item()),
                    "routed_std": float(routed.std().item()),
                }

    under_records = []
    over_records = []
    for regime_name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        for h in [7, 14, 30]:
            hid = h - 1
            cov_grid = covered90[mask, hid].mean(axis=0)
            for i in range(5):
                for j in range(5):
                    val = float(cov_grid[i, j])
                    if val < LAYER2_LOW:
                        under_records.append({"regime": regime_name, "horizon": h, "cell": [i, j], "coverage": val})
                    if val > LAYER2_HIGH:
                        over_records.append({"regime": regime_name, "horizon": h, "cell": [i, j], "coverage": val})
    under_records.sort(key=lambda x: x["coverage"])
    over_records.sort(key=lambda x: x["coverage"], reverse=True)
    selected_under = under_records[:8]
    selected_over = over_records[:6]

    def analyze_slice(slice_record: dict[str, Any], mode: str) -> dict[str, Any]:
        regime = slice_record["regime"]
        h = slice_record["horizon"]
        i, j = slice_record["cell"]
        block_idx = (h - 1) // model.decoder.block_len
        idx = np.where(regime_masks[regime])[0]
        if len(idx) == 0:
            return {**slice_record, "block": int(block_idx + 1), "n_windows": 0}
        idx = idx[: min(len(idx), 96)]
        hist_01 = history_01_all[idx].to(device)
        fut_01 = future_01_all[idx].to(device)
        hist_norm_sel = normalize_iv(hist_01)

        (
            mu,
            _tf,
            _td,
            _cf,
            _cd,
            _scale,
            _base_local_delta,
            block_logits,
            _block_context,
        ) = model.forward_from_history(hist_01)
        prior_block = F.softmax(block_logits, dim=-1).detach().cpu().numpy()[:, block_idx]
        posterior_block = posterior_blocks_all[idx, block_idx]
        current_assign = torch.from_numpy(prior_argmax_all[idx]).to(device=device, dtype=torch.long)

        current_samples = model.sample_batched(hist_norm_sel, n_samples=slice_eval_samples)
        current_mean = current_samples.mean(dim=1)
        current_lo = current_samples.quantile(0.05, dim=1)
        current_hi = current_samples.quantile(0.95, dim=1)
        current_cov = ((fut_01 >= current_lo) & (fut_01 <= current_hi)).float().mean(dim=0)[h - 1, i, j].item()
        current_mean_shift = (current_mean - unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi).view_as(current_mean))[
            :, h - 1, i, j
        ].mean().item()

        forced_coverages = []
        forced_mean_shifts = []
        for t in range(n_templates):
            forced_assign = current_assign.unsqueeze(1).expand(-1, slice_eval_samples, -1).clone()
            forced_assign[:, :, block_idx] = t
            forced_samples = sample_forced_assignments(model, hist_01, forced_assign, n_samples=slice_eval_samples)
            forced_mean = forced_samples.mean(dim=1)
            forced_lo = forced_samples.quantile(0.05, dim=1)
            forced_hi = forced_samples.quantile(0.95, dim=1)
            forced_cov = ((fut_01 >= forced_lo) & (fut_01 <= forced_hi)).float().mean(dim=0)[h - 1, i, j].item()
            forced_coverages.append(float(forced_cov))
            forced_shift = (
                forced_mean
                - unconstrained_to_iv(mu, lo=model.support_lo, hi=model.support_hi).view_as(forced_mean)
            )[:, h - 1, i, j].mean().item()
            forced_mean_shifts.append(float(forced_shift))

        if mode == "under":
            helpful_template = int(np.argmax(forced_coverages))
            best_cov = float(np.max(forced_coverages))
            prior_help = float(prior_block[:, helpful_template].mean())
            posterior_help = float(posterior_block[:, helpful_template].mean())
            if best_cov >= LAYER2_LOW and posterior_help > prior_help + 0.10:
                diagnosis = "prior_underuses_helpful_template"
            elif best_cov < max(LAYER2_LOW, current_cov + 0.05):
                diagnosis = "expert_family_too_weak"
            else:
                diagnosis = "partial_recoverable"
        else:
            helpful_template = int(np.argmin(forced_coverages))
            best_cov = float(np.min(forced_coverages))
            prior_help = float(prior_block[:, helpful_template].mean())
            posterior_help = float(posterior_block[:, helpful_template].mean())
            if best_cov <= LAYER2_HIGH and posterior_help > prior_help + 0.10:
                diagnosis = "prior_underuses_helpful_template"
            elif best_cov > min(LAYER2_HIGH, current_cov - 0.05):
                diagnosis = "expert_family_too_weak"
            else:
                diagnosis = "partial_recoverable"

        return {
            **slice_record,
            "block": int(block_idx + 1),
            "n_windows": int(len(idx)),
            "current_coverage_recomputed": float(current_cov),
            "current_mean_shift_vs_deterministic": float(current_mean_shift),
            "forced_template_coverages": [float(x) for x in forced_coverages],
            "forced_template_mean_shifts_vs_deterministic": [float(x) for x in forced_mean_shifts],
            "helpful_template": helpful_template,
            "best_forced_coverage": best_cov,
            "prior_block_mean": prior_block.mean(axis=0).tolist(),
            "posterior_block_mean": posterior_block.mean(axis=0).tolist(),
            "prior_helpful_mass": prior_help,
            "posterior_helpful_mass": posterior_help,
            "diagnosis": diagnosis,
        }

    analyzed_under = [analyze_slice(rec, "under") for rec in selected_under]
    analyzed_over = [analyze_slice(rec, "over") for rec in selected_over]

    return {
        "overall": {
            "n_windows": int(n_windows),
            "n_blocks": int(n_blocks),
            "n_templates": int(n_templates),
            "overall_window_coverage_mean": float(per_window_cov.mean()),
            "calm_window_coverage_mean": float(per_window_cov[calm_mask].mean()),
            "turb_window_coverage_mean": float(per_window_cov[turb_mask].mean()),
            "prior_posterior_argmax_agreement_by_block": (prior_argmax_all == posterior_argmax_all).mean(axis=0).tolist(),
        },
        "mean_reversion": mean_reversion,
        "mean_shift_vs_deterministic": mean_shift,
        "routing": {
            "regime_block_usage": regime_block_usage,
            "regime_block_posterior": regime_block_post,
            "regime_entropy": regime_entropy,
            "top_assignment_patterns": top_assignment_patterns,
        },
        "template_bank_summary": template_bank_summary,
        "expert_probe": expert_probe,
        "hard_under_slices": analyzed_under,
        "high_overcoverage_slices": analyzed_over,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze 178e_v0 mechanism")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/backfill/exact_block_flow_expert_mean_reverting_residual_flow_structured_joint_student_t_178e/best_model.pt",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--slice_eval_samples", type=int, default=40)
    parser.add_argument("--expert_probe_samples", type=int, default=24)
    parser.add_argument("--active_slope_threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-05/analysis/178e_mechanistic",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_model(args.model_path, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    vov, q20, q80 = regime_masks_from_history(history_norm)
    analysis = analyze_178e(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        vov=vov,
        q20=q20,
        q80=q80,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        slice_eval_samples=args.slice_eval_samples,
        expert_probe_samples=args.expert_probe_samples,
        active_slope_threshold=args.active_slope_threshold,
    )
    analysis["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    analysis["model_path"] = args.model_path
    analysis["test_start"] = args.test_start
    analysis["max_windows"] = int(history_norm.shape[0])

    out_path = output_dir / "mechanistic_summary.json"
    with open(out_path, "w") as f:
        json.dump(make_serializable(analysis), f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
