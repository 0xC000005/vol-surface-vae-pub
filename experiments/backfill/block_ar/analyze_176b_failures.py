#!/usr/bin/env python
"""
Focused failure decomposition for 176b after the corrected S3 gate.

Questions:
  1. Which exact slices still drive the corrected S3 and S7 failures?
  2. Are those failures mean bias, underwidth, overwidth, or mixed?
  3. Can any single latent template fix the hard slices if routed differently?
  4. Is the next step better routing or a richer local-law family?
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
    classify_cell_failure,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_176b_shared_local_template_mixture import (
    SharedLocalTemplateMixtureStudentTModel,
)


SELECT_HORIZONS = [1, 7, 14, 30]
REGIME_NAMES = ["all", "calm", "turb"]
LAYER2_LOW = 0.70
LAYER2_HIGH = 0.95
LAYER3_GATE = 0.05
MAX_UNCOND_BATCHES = 5


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    if raw_config["type"] != "shared_local_template_mixture_residual_flow_structured_joint_student_t_176b":
        raise ValueError(
            "Expected shared_local_template_mixture_residual_flow_structured_joint_student_t_176b, "
            f"got {raw_config['type']}"
        )
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = SharedLocalTemplateMixtureStudentTModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


def summarize_grid_extremes(
    grid: np.ndarray,
    top_k: int = 10,
    reverse: bool = True,
) -> list[dict[str, Any]]:
    flat = []
    for idx, value in enumerate(grid.reshape(-1)):
        flat.append(
            {
                "cell": [int(idx // 5), int(idx % 5)],
                "value": float(value),
            }
        )
    flat.sort(key=lambda x: x["value"], reverse=reverse)
    return flat[:top_k]


@torch.no_grad()
def sample_forced_component(
    model: SharedLocalTemplateMixtureStudentTModel,
    mu: torch.Tensor,
    time_factor: torch.Tensor,
    time_diag: torch.Tensor,
    cell_factor: torch.Tensor,
    cell_diag: torch.Tensor,
    scale: torch.Tensor,
    flow_context: torch.Tensor,
    local_delta_components: torch.Tensor,
    component_idx: int,
    n_samples: int,
) -> torch.Tensor:
    batch, n_frames, n_cells = mu.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)

    base = torch.distributions.StudentT(df=model.base_nu)
    z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
    ctx = flow_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
    white_flat, _ = model.flow.inverse(z, ctx)
    white = white_flat.view(batch, n_samples, n_frames, n_cells)

    temp = torch.einsum("bij,bsjk->bsik", chol_t, white)
    noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
    local_scale = torch.exp(0.5 * local_delta_components[:, component_idx])
    samples_u = mu.unsqueeze(1) + noise * local_scale.unsqueeze(1)
    return unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)


def grid_records_from_mask(
    grid: np.ndarray,
    mask_fn,
    extra: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for i in range(5):
        for j in range(5):
            value = float(grid[i, j])
            if mask_fn(value):
                record = {"cell": [i, j], "value": value}
                if extra is not None:
                    for key, arr in extra.items():
                        record[key] = float(arr[i, j])
                records.append(record)
    return records


@torch.no_grad()
def analyze_176b_failures(
    model: SharedLocalTemplateMixtureStudentTModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    n_samples: int,
    max_batches: int,
) -> dict[str, Any]:
    n_total = history_norm.shape[0]
    n_batches = min(max_batches, int(np.ceil(n_total / batch_size)))
    n_eval = min(n_total, n_batches * batch_size)
    history_norm = history_norm[:n_eval].to(device)
    future_norm = future_norm[:n_eval].to(device)
    vov = vov[:n_eval]

    history_01_all = denormalize_iv(history_norm).reshape(n_eval, history_norm.shape[1], -1)
    future_01_all = denormalize_iv(future_norm).reshape(n_eval, future_norm.shape[1], -1)
    n_windows, future_len, n_cells = future_01_all.shape

    calm_mask = vov <= q20
    turb_mask = vov >= q80
    masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm": calm_mask,
        "turb": turb_mask,
    }

    (
        _mu0,
        _tf0,
        _td0,
        _cf0,
        _cd0,
        _scale0,
        _ctx0,
        _base0,
        local_delta_components0,
        gate_logits0,
    ) = model.forward_from_history(history_01_all[:1])
    n_components = gate_logits0.shape[1]

    mix_mean_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    mix_std_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    mix_width_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    mix_covered90 = np.zeros((n_windows, future_len, n_cells), dtype=bool)
    mix_prior = np.zeros((n_windows, n_components), dtype=np.float32)
    mix_post = np.zeros((n_windows, n_components), dtype=np.float32)
    sampled_component_frac = np.zeros((n_windows, n_components), dtype=np.float32)
    weighted_local_scale = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    forced_covered = np.zeros((n_components, n_windows, future_len, n_cells), dtype=bool)
    forced_width = np.zeros((n_components, n_windows, future_len, n_cells), dtype=np.float32)

    cond_width_sum = np.zeros((5, 5), dtype=np.float64)
    uncond_width_sum = np.zeros((5, 5), dtype=np.float64)
    comp_cond_width_sum = np.zeros((n_components, 5, 5), dtype=np.float64)
    comp_uncond_width_sum = np.zeros((n_components, 5, 5), dtype=np.float64)
    n_cond_batches = 0
    n_uncond_batches = 0

    row0 = 0
    for batch_idx, start in enumerate(range(0, n_eval, batch_size)):
        if batch_idx >= n_batches:
            break
        end = min(start + batch_size, n_eval)
        hist_norm_b = history_norm[start:end]
        fut_norm_b = future_norm[start:end]
        hist_01_b = denormalize_iv(hist_norm_b)
        fut_01_b = denormalize_iv(fut_norm_b).reshape(end - start, future_len, n_cells)

        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            _base_local_delta,
            local_delta_components,
            gate_logits,
        ) = model.forward_from_history(hist_01_b)

        target_u = iv_to_unconstrained(
            fut_01_b,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        _mix_logprob, posterior, _aux = model.mixture_log_prob(
            target_u=target_u,
            mu=mu,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            flow_context=flow_context,
            local_delta_components=local_delta_components,
            gate_logits=gate_logits,
        )

        prior = F.softmax(gate_logits, dim=-1)
        local_scale_components = torch.exp(0.5 * local_delta_components)
        weighted_scale_b = (prior.unsqueeze(-1).unsqueeze(-1) * local_scale_components).sum(dim=1)

        mix_samples_u, mix_component_idx = model.sample_future_u(hist_01_b, n_samples=n_samples)
        mix_samples_iv = unconstrained_to_iv(
            mix_samples_u,
            lo=model.support_lo,
            hi=model.support_hi,
        )
        mix_mean_b = mix_samples_iv.mean(dim=1)
        mix_std_b = mix_samples_iv.std(dim=1, unbiased=False)
        mix_lower_b = torch.quantile(mix_samples_iv, 0.05, dim=1)
        mix_upper_b = torch.quantile(mix_samples_iv, 0.95, dim=1)
        mix_width_b = mix_upper_b - mix_lower_b
        mix_covered_b = ((fut_01_b >= mix_lower_b) & (fut_01_b <= mix_upper_b))

        mix_mean_iv[row0:end] = mix_mean_b.detach().cpu().numpy()
        mix_std_iv[row0:end] = mix_std_b.detach().cpu().numpy()
        mix_width_iv[row0:end] = mix_width_b.detach().cpu().numpy()
        mix_covered90[row0:end] = mix_covered_b.detach().cpu().numpy()
        mix_prior[row0:end] = prior.detach().cpu().numpy()
        mix_post[row0:end] = posterior.detach().cpu().numpy()
        weighted_local_scale[row0:end] = weighted_scale_b.detach().cpu().numpy()
        sampled_component_frac[row0:end] = (
            F.one_hot(mix_component_idx, num_classes=n_components).float().mean(dim=1).cpu().numpy()
        )

        cond_width_sum += mix_width_b.mean(dim=(0, 1)).detach().cpu().numpy().reshape(5, 5)
        n_cond_batches += 1

        for k in range(n_components):
            comp_samples_iv = sample_forced_component(
                model=model,
                mu=mu,
                time_factor=time_factor,
                time_diag=time_diag,
                cell_factor=cell_factor,
                cell_diag=cell_diag,
                scale=scale,
                flow_context=flow_context,
                local_delta_components=local_delta_components,
                component_idx=k,
                n_samples=n_samples,
            )
            comp_lower = torch.quantile(comp_samples_iv, 0.05, dim=1)
            comp_upper = torch.quantile(comp_samples_iv, 0.95, dim=1)
            comp_width_b = comp_upper - comp_lower
            comp_covered_b = ((fut_01_b >= comp_lower) & (fut_01_b <= comp_upper))
            forced_covered[k, row0:end] = comp_covered_b.detach().cpu().numpy()
            forced_width[k, row0:end] = comp_width_b.detach().cpu().numpy()
            comp_cond_width_sum[k] += comp_width_b.mean(dim=(0, 1)).detach().cpu().numpy().reshape(5, 5)

        if batch_idx < MAX_UNCOND_BATCHES:
            zero_hist_norm = torch.zeros_like(hist_norm_b)
            zero_hist_01 = denormalize_iv(zero_hist_norm)
            (
                mu0,
                tf0,
                td0,
                cf0,
                cd0,
                scale0,
                ctx0,
                _base0,
                local_delta_components0,
                _gate_logits0,
            ) = model.forward_from_history(zero_hist_01)

            uncond_samples_u, _ = model.sample_future_u(zero_hist_01, n_samples=n_samples)
            uncond_samples_iv = unconstrained_to_iv(
                uncond_samples_u,
                lo=model.support_lo,
                hi=model.support_hi,
            )
            uncond_lower = torch.quantile(uncond_samples_iv, 0.05, dim=1)
            uncond_upper = torch.quantile(uncond_samples_iv, 0.95, dim=1)
            uncond_width_b = uncond_upper - uncond_lower
            uncond_width_sum += uncond_width_b.mean(dim=(0, 1)).detach().cpu().numpy().reshape(5, 5)
            n_uncond_batches += 1

            for k in range(n_components):
                comp_uncond_iv = sample_forced_component(
                    model=model,
                    mu=mu0,
                    time_factor=tf0,
                    time_diag=td0,
                    cell_factor=cf0,
                    cell_diag=cd0,
                    scale=scale0,
                    flow_context=ctx0,
                    local_delta_components=local_delta_components0,
                    component_idx=k,
                    n_samples=n_samples,
                )
                comp_uncond_lower = torch.quantile(comp_uncond_iv, 0.05, dim=1)
                comp_uncond_upper = torch.quantile(comp_uncond_iv, 0.95, dim=1)
                comp_uncond_width_b = comp_uncond_upper - comp_uncond_lower
                comp_uncond_width_sum[k] += comp_uncond_width_b.mean(dim=(0, 1)).detach().cpu().numpy().reshape(5, 5)

        row0 = end

    target_iv = future_01_all.detach().cpu().numpy()
    resid_iv = target_iv - mix_mean_iv
    z_iv = resid_iv / np.clip(mix_std_iv, 1e-6, None)

    cell_width_ratio = cond_width_sum / max(n_cond_batches, 1)
    cell_width_ratio /= np.maximum(uncond_width_sum / max(n_uncond_batches, 1), 1e-8)
    component_cell_width_ratio = np.zeros((n_components, 5, 5), dtype=np.float64)
    for k in range(n_components):
        component_cell_width_ratio[k] = comp_cond_width_sum[k] / max(n_cond_batches, 1)
        component_cell_width_ratio[k] /= np.maximum(comp_uncond_width_sum[k] / max(n_uncond_batches, 1), 1e-8)

    top_s3_cells = []
    flat_idx = np.argsort(cell_width_ratio.reshape(-1))[::-1]
    for idx in flat_idx[:12]:
        r, c = divmod(int(idx), 5)
        comp_vals = component_cell_width_ratio[:, r, c]
        best_k = int(np.argmin(comp_vals))
        top_s3_cells.append(
            {
                "cell": [r, c],
                "mixture_width_ratio": float(cell_width_ratio[r, c]),
                "component_width_ratios": comp_vals.tolist(),
                "best_component": best_k,
                "best_component_width_ratio": float(comp_vals[best_k]),
                "best_component_can_pass": bool(comp_vals[best_k] < 1.20),
            }
        )

    layer2_breakdown: dict[str, dict[str, Any]] = {}
    failing_slices = []
    recoverable_under = 0
    total_under = 0
    recoverable_over = 0
    total_over = 0

    for regime_name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        layer2_breakdown[regime_name] = {}
        for h in SELECT_HORIZONS:
            hidx = h - 1
            cell_cov = mix_covered90[mask, hidx].mean(axis=0).reshape(5, 5)
            cell_z_mean = z_iv[mask, hidx].mean(axis=0).reshape(5, 5)
            cell_z_std = z_iv[mask, hidx].std(axis=0).reshape(5, 5)
            cell_pred_std = mix_std_iv[mask, hidx].mean(axis=0).reshape(5, 5)
            cell_local = weighted_local_scale[mask, hidx].mean(axis=0).reshape(5, 5)
            cell_comp_cov = forced_covered[:, mask, hidx].mean(axis=1).reshape(n_components, 5, 5)
            low_cells = []
            high_cells = []
            for i in range(5):
                for j in range(5):
                    coverage = float(cell_cov[i, j])
                    comp_covs = cell_comp_cov[:, i, j]
                    if coverage < LAYER2_LOW:
                        total_under += 1
                        best_k = int(np.argmax(comp_covs))
                        recoverable = bool(comp_covs[best_k] >= LAYER2_LOW)
                        recoverable_under += int(recoverable)
                        record = {
                            "regime": regime_name,
                            "horizon": h,
                            "cell": [i, j],
                            "coverage_90": coverage,
                            "z_mean": float(cell_z_mean[i, j]),
                            "z_std": float(cell_z_std[i, j]),
                            "pred_std_mean": float(cell_pred_std[i, j]),
                            "local_scale_mean": float(cell_local[i, j]),
                            "classification": classify_cell_failure(
                                coverage,
                                float(cell_z_mean[i, j]),
                                float(cell_z_std[i, j]),
                            ),
                            "prior_mean": mix_prior[mask].mean(axis=0).tolist(),
                            "posterior_mean": mix_post[mask].mean(axis=0).tolist(),
                            "component_coverages": comp_covs.tolist(),
                            "best_component": best_k,
                            "best_component_coverage": float(comp_covs[best_k]),
                            "recoverable_by_component": recoverable,
                        }
                        low_cells.append(record)
                        failing_slices.append(record)
                    elif coverage > LAYER2_HIGH:
                        total_over += 1
                        best_k = int(np.argmin(comp_covs))
                        recoverable = bool(comp_covs[best_k] <= LAYER2_HIGH)
                        recoverable_over += int(recoverable)
                        record = {
                            "regime": regime_name,
                            "horizon": h,
                            "cell": [i, j],
                            "coverage_90": coverage,
                            "z_mean": float(cell_z_mean[i, j]),
                            "z_std": float(cell_z_std[i, j]),
                            "pred_std_mean": float(cell_pred_std[i, j]),
                            "local_scale_mean": float(cell_local[i, j]),
                            "classification": classify_cell_failure(
                                coverage,
                                float(cell_z_mean[i, j]),
                                float(cell_z_std[i, j]),
                            ),
                            "prior_mean": mix_prior[mask].mean(axis=0).tolist(),
                            "posterior_mean": mix_post[mask].mean(axis=0).tolist(),
                            "component_coverages": comp_covs.tolist(),
                            "best_component": best_k,
                            "best_component_coverage": float(comp_covs[best_k]),
                            "recoverable_by_component": recoverable,
                        }
                        high_cells.append(record)
                        failing_slices.append(record)

            layer2_breakdown[regime_name][str(h)] = {
                "n_low": len(low_cells),
                "n_high": len(high_cells),
                "worst": float(cell_cov.min()),
                "best": float(cell_cov.max()),
                "low_cells": sorted(low_cells, key=lambda x: x["coverage_90"])[:10],
                "high_cells": sorted(high_cells, key=lambda x: x["coverage_90"], reverse=True)[:10],
                "top_local_scale_cells": summarize_grid_extremes(cell_local, top_k=8),
                "top_pred_std_cells": summarize_grid_extremes(cell_pred_std, top_k=8),
            }

    window_cell_cov = mix_covered90.mean(axis=1).reshape(n_windows, 5, 5)
    catastrophic = window_cell_cov < 0.30
    cats_per_window = catastrophic.sum(axis=(1, 2))
    top_windows = np.argsort(cats_per_window)[-10:][::-1]
    top_window_records = []
    for w in top_windows:
        if cats_per_window[w] <= 0:
            continue
        bad_cells = [(int(i), int(j)) for i, j in zip(*np.where(catastrophic[w].reshape(5, 5)))]
        top_window_records.append(
            {
                "window": int(w),
                "n_catastrophic_cells": int(cats_per_window[w]),
                "vol_of_vol": float(vov[w]),
                "regime": "turb" if turb_mask[w] else ("calm" if calm_mask[w] else "mid"),
                "prior": mix_prior[w].tolist(),
                "posterior": mix_post[w].tolist(),
                "sampled_component_frac": sampled_component_frac[w].tolist(),
                "bad_cells": bad_cells[:10],
            }
        )

    cell_fail_frequency = []
    catastrophic_cell_rate = catastrophic.mean(axis=0).reshape(5, 5)
    for i in range(5):
        for j in range(5):
            cell_fail_frequency.append(
                {
                    "cell": [i, j],
                    "catastrophic_rate": float(catastrophic_cell_rate[i, j]),
                }
            )
    cell_fail_frequency.sort(key=lambda x: x["catastrophic_rate"], reverse=True)

    overall_prior = mix_prior.mean(axis=0)
    prior_bad_windows = mix_prior[cats_per_window > 0].mean(axis=0) if np.any(cats_per_window > 0) else overall_prior
    posterior_bad_windows = mix_post[cats_per_window > 0].mean(axis=0) if np.any(cats_per_window > 0) else mix_post.mean(axis=0)

    next_step_inference = {
        "s3_failure_is_real_overwide_cluster": bool(float(cell_width_ratio.max()) >= 1.20),
        "s7_failure_concentrates_in_turbulent_late_horizons": bool(
            layer2_breakdown["turb"]["14"]["n_low"] > 0 or layer2_breakdown["turb"]["30"]["n_low"] > 0
        ),
        "at_least_one_component_can_fix_some_undercovered_slices": bool(recoverable_under > 0),
        "most_undercovered_slices_not_fixable_by_component_choice": bool(
            total_under > 0 and recoverable_under < 0.5 * total_under
        ),
        "most_overcovered_slices_not_fixable_by_component_choice": bool(
            total_over > 0 and recoverable_over < 0.5 * total_over
        ),
        "routing_problem_dominates": bool(total_under > 0 and recoverable_under >= 0.5 * total_under),
        "template_capacity_problem_dominates": bool(total_under > 0 and recoverable_under < 0.5 * total_under),
    }

    return {
        "metadata": {
            "n_windows": n_windows,
            "future_len": future_len,
            "n_cells": n_cells,
            "n_samples": n_samples,
            "batch_size": batch_size,
            "max_batches": n_batches,
            "max_uncond_batches": MAX_UNCOND_BATCHES,
            "q20_vov": q20,
            "q80_vov": q80,
        },
        "s3_failure": {
            "mixture_cell_width_ratio": cell_width_ratio.tolist(),
            "mixture_worst_cell_width_ratio": float(cell_width_ratio.max()),
            "top_failing_cells": top_s3_cells,
            "component_cell_width_ratio": {
                f"component_{k}": component_cell_width_ratio[k].tolist()
                for k in range(n_components)
            },
        },
        "s7_failure": {
            "layer2_breakdown": layer2_breakdown,
            "recoverable_undercovered_slices": {
                "recoverable": recoverable_under,
                "total": total_under,
            },
            "recoverable_overcovered_slices": {
                "recoverable": recoverable_over,
                "total": total_over,
            },
            "top_failing_slices": sorted(
                failing_slices,
                key=lambda x: (
                    x["coverage_90"] if x["coverage_90"] < LAYER2_LOW else -x["coverage_90"]
                ),
            )[:24],
            "layer3_summary": {
                "catastrophic_rate": float(catastrophic.mean()),
                "gate": LAYER3_GATE,
                "top_windows": top_window_records,
                "top_cells": cell_fail_frequency[:15],
                "prior_mean_all_windows": overall_prior.tolist(),
                "prior_mean_bad_windows": prior_bad_windows.tolist(),
                "posterior_mean_bad_windows": posterior_bad_windows.tolist(),
            },
        },
        "routing_summary": {
            "prior_by_regime": {
                name: mix_prior[mask].mean(axis=0).tolist() if np.any(mask) else [0.0] * n_components
                for name, mask in masks.items()
            },
            "posterior_by_regime": {
                name: mix_post[mask].mean(axis=0).tolist() if np.any(mask) else [0.0] * n_components
                for name, mask in masks.items()
            },
            "sampled_component_frac_by_regime": {
                name: sampled_component_frac[mask].mean(axis=0).tolist() if np.any(mask) else [0.0] * n_components
                for name, mask in masks.items()
            },
        },
        "next_step_inference": next_step_inference,
    }


def main():
    parser = argparse.ArgumentParser(description="Focused failure decomposition for 176b")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, checkpoint = load_model(args.model_path, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=None,
    )
    vov, q20, q80 = regime_masks_from_history(history_norm)
    analysis = analyze_176b_failures(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        vov=vov,
        q20=q20,
        q80=q80,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        max_batches=args.max_batches,
    )
    analysis["checkpoint_epoch"] = checkpoint.get("epoch")
    analysis["model_path"] = args.model_path

    out_path = Path(args.output_dir) / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(analysis), indent=2))
    print(f"Saved analysis to {out_path}")
    print(json.dumps(make_serializable(analysis["next_step_inference"]), indent=2))


if __name__ == "__main__":
    main()
