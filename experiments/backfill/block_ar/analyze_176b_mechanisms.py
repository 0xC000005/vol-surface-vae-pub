#!/usr/bin/env python
"""
Focused mechanistic analysis for 176b.

Questions:
  1. Are the local-template mixture components actually active on held-out data?
  2. Does the prior respond to condition/regime?
  3. Do different templates trade off width vs hard-slice coverage in a meaningful way?
  4. Is the S2 improvement plausibly real, or just a disguised global widening?
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
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_176b_shared_local_template_mixture import (
    SharedLocalTemplateMixtureStudentTModel,
)


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


def per_window_coverage(samples_iv: torch.Tensor, target_iv: torch.Tensor) -> np.ndarray:
    lo = torch.quantile(samples_iv, 0.05, dim=1)
    hi = torch.quantile(samples_iv, 0.95, dim=1)
    cov = ((target_iv >= lo) & (target_iv <= hi)).float().mean(dim=(1, 2)).cpu().numpy()
    return cov


@torch.no_grad()
def analyze_176b(
    model: SharedLocalTemplateMixtureStudentTModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    future_01_all = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    n_windows, future_len, n_cells = future_01_all.shape
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm": calm_mask,
        "turb": turb_mask,
    }

    prior_probs = []
    posterior_probs = []
    gate_max = []
    prior_entropy = []
    post_entropy = []
    base_local_rms = []
    component_local_rms = []
    per_window_cov = np.zeros(n_windows, dtype=np.float32)

    hist_batches = []
    fut_batches = []

    row0 = 0
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_norm_b = history_norm[start:end]
        hist_01_b = denormalize_iv(hist_norm_b)
        fut_01_b = denormalize_iv(future_norm[start:end]).reshape(end - start, future_len, n_cells)

        (
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            local_delta_components,
            gate_logits,
        ) = model.forward_from_history(hist_01_b)

        target_u = iv_to_unconstrained(
            fut_01_b,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        mix_log_prob, posterior, aux = model.mixture_log_prob(
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
        prior_probs.append(prior.detach().cpu())
        posterior_probs.append(posterior.detach().cpu())
        gate_max.append(prior.max(dim=-1).values.detach().cpu())
        prior_entropy.append((-(prior * torch.log(prior.clamp_min(1e-8))).sum(dim=-1)).detach().cpu())
        post_entropy.append((-(posterior * torch.log(posterior.clamp_min(1e-8))).sum(dim=-1)).detach().cpu())
        base_local_rms.append(base_local_delta.pow(2).mean(dim=(1, 2)).sqrt().detach().cpu())
        component_local_rms.append(aux["local_delta_rms"].detach().cpu())

        samples_01 = model.sample_batched(hist_norm_b, n_samples=n_samples)
        samples_iv = samples_01.reshape(end - start, n_samples, future_len, n_cells)
        per_window_cov[row0:end] = per_window_coverage(samples_iv, fut_01_b)
        row0 = end

        hist_batches.append(hist_norm_b.detach().cpu())
        fut_batches.append(fut_01_b.detach().cpu())

    prior_probs = torch.cat(prior_probs).numpy()
    posterior_probs = torch.cat(posterior_probs).numpy()
    gate_max = torch.cat(gate_max).numpy()
    prior_entropy = torch.cat(prior_entropy).numpy()
    post_entropy = torch.cat(post_entropy).numpy()
    base_local_rms = torch.cat(base_local_rms).numpy()
    component_local_rms = torch.cat(component_local_rms).numpy()

    n_components = prior_probs.shape[1]
    arg_post = posterior_probs.argmax(axis=1)
    hard_window_mask = per_window_cov < np.quantile(per_window_cov, 0.1)
    very_hard_mask = per_window_cov < 0.5

    prior_by_regime = {}
    posterior_assign_by_regime = {}
    for name, mask in masks.items():
        prior_by_regime[name] = {
            "mean_probs": prior_probs[mask].mean(axis=0).tolist(),
            "mean_entropy": float(prior_entropy[mask].mean()),
            "mean_max_prob": float(gate_max[mask].mean()),
        }
        counts = np.bincount(arg_post[mask], minlength=n_components).astype(float)
        posterior_assign_by_regime[name] = {
            "counts": counts.astype(int).tolist(),
            "fractions": (counts / max(counts.sum(), 1.0)).tolist(),
        }

    prior_condition_correlation = {
        f"component_{k}_vs_vov": float(np.corrcoef(prior_probs[:, k], vov)[0, 1]) if np.std(prior_probs[:, k]) > 1e-8 else 0.0
        for k in range(n_components)
    }

    component_specialization = {}
    for k in range(n_components):
        k_mask = arg_post == k
        component_specialization[f"component_{k}"] = {
            "posterior_fraction": float(k_mask.mean()),
            "prior_mean": float(prior_probs[:, k].mean()),
            "prior_mean_calm": float(prior_probs[calm_mask, k].mean()),
            "prior_mean_turb": float(prior_probs[turb_mask, k].mean()),
            "avg_component_local_delta_rms": float(component_local_rms[:, k].mean()),
            "avg_base_local_delta_rms": float(base_local_rms.mean()),
            "hard_window_fraction": float(hard_window_mask[k_mask].mean()) if np.any(k_mask) else 0.0,
            "very_hard_window_fraction": float(very_hard_mask[k_mask].mean()) if np.any(k_mask) else 0.0,
        }

    history_norm_all_cpu = torch.cat(hist_batches, dim=0)
    future_01_all_cpu = torch.cat(fut_batches, dim=0)
    forced_component_cov = {}
    forced_component_width = {}
    forced_component_tc = {}
    forced_component_h30_worst = {}

    for k in range(n_components):
        all_samples = []
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            hist_norm_b = history_norm_all_cpu[start:end].to(device)
            hist_01_b = denormalize_iv(hist_norm_b)
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
                _gate_logits,
            ) = model.forward_from_history(hist_01_b)

            batch = hist_01_b.shape[0]
            n_frames = mu.shape[1]
            chol_t = torch.linalg.cholesky(model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)[0])
            chol_c = torch.linalg.cholesky(model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)[1])
            local_scale = torch.exp(0.5 * local_delta_components[:, k])
            ctx = flow_context

            base = torch.distributions.StudentT(df=model.base_nu)
            z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
            white_flat, _ = model.flow.inverse(z, ctx.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1))
            white = white_flat.view(batch, n_samples, n_frames, n_cells)
            temp = torch.einsum("bij,bsjk->bsik", chol_t, white)
            noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
            samples_u = mu.unsqueeze(1) + noise * local_scale.unsqueeze(1)
            samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
            all_samples.append(samples_01.detach().cpu())

        samples = torch.cat(all_samples, dim=0).reshape(n_windows, n_samples, future_len, 5, 5)
        future_grid = future_01_all_cpu.reshape(n_windows, future_len, 5, 5)
        lo = torch.quantile(samples, 0.05, dim=1)
        hi = torch.quantile(samples, 0.95, dim=1)
        cov = ((future_grid >= lo) & (future_grid <= hi)).float()
        width = (hi - lo)
        forced_component_cov[f"component_{k}"] = float(cov.mean().item())
        forced_component_width[f"component_{k}"] = float(width.mean().item())

        mean_iv = denormalize_iv(history_norm_all_cpu).mean(dim=(-1, -2))
        daily = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov_tensor = daily.std(dim=1)
        q20_t = torch.quantile(vov_tensor, 0.2)
        q80_t = torch.quantile(vov_tensor, 0.8)
        calm = vov_tensor <= q20_t
        turb = vov_tensor >= q80_t
        window_width = width.mean(dim=(1, 2, 3))
        forced_component_tc[f"component_{k}"] = float((window_width[turb].mean() / window_width[calm].mean()).item())
        h30_cov = cov[:, 29].reshape(n_windows, -1).mean(dim=0)
        forced_component_h30_worst[f"component_{k}"] = float(h30_cov.min().item())

    return {
        "metadata": {
            "n_windows": int(n_windows),
            "n_components": int(n_components),
            "n_samples": int(n_samples),
            "q20_vov": float(q20),
            "q80_vov": float(q80),
            "hard_window_threshold": float(np.quantile(per_window_cov, 0.1)),
        },
        "global_gate_summary": {
            "prior_entropy_mean": float(prior_entropy.mean()),
            "prior_max_prob_mean": float(gate_max.mean()),
            "posterior_entropy_mean": float(post_entropy.mean()),
            "hard_window_fraction": float(hard_window_mask.mean()),
            "very_hard_window_fraction": float(very_hard_mask.mean()),
        },
        "prior_condition_correlation": prior_condition_correlation,
        "prior_by_regime": prior_by_regime,
        "posterior_assign_by_regime": posterior_assign_by_regime,
        "component_specialization": component_specialization,
        "forced_component_overall_coverage_90": forced_component_cov,
        "forced_component_avg_width90": forced_component_width,
        "forced_component_turb_calm_ratio": forced_component_tc,
        "forced_component_h30_worst_cell_coverage": forced_component_h30_worst,
    }


def main():
    parser = argparse.ArgumentParser(description="Mechanistic analysis for 176b")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.model_path, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    vov, q20, q80 = regime_masks_from_history(history_norm)
    analysis = analyze_176b(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        vov=vov,
        q20=q20,
        q80=q80,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    analysis["checkpoint_epoch"] = checkpoint.get("epoch")
    analysis["model_path"] = args.model_path

    out_path = Path(args.output_dir) / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(analysis), indent=2))
    print(f"Saved analysis to {out_path}")


if __name__ == "__main__":
    main()
