#!/usr/bin/env python
"""
Focused template / assignment analysis for 178c.

Questions:
  1. Are the now-active blockwise latent templates actually distinct?
  2. Does the prior route different blocks differently across calm vs turbulence?
  3. On the exact failing S2/S3/S7 slices, does the posterior know a better
     template than the prior, or are the templates themselves too weak?
  4. Does the evidence imply better routing, richer templates, or a selection
     issue next?
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
)
from experiments.backfill.block_ar.train_178c_exact_block_mixture_mean_reverting_residual_flow import (
    ExactBlockMixtureMeanRevertingResidualFlowStructuredJointStudentTModel,
)


SELECT_HORIZONS = [1, 7, 14, 30]
LAYER2_LOW = 0.70
LAYER2_HIGH = 0.95


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    if raw_config["type"] != "exact_block_mixture_mean_reverting_residual_flow_structured_joint_student_t_178c":
        raise ValueError(
            "Expected exact_block_mixture_mean_reverting_residual_flow_structured_joint_student_t_178c, "
            f"got {raw_config['type']}"
        )
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = ExactBlockMixtureMeanRevertingResidualFlowStructuredJointStudentTModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
        mix_chunk_size=raw_config.get("mix_chunk_size", 27),
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


@torch.no_grad()
def exact_block_posterior(
    model: ExactBlockMixtureMeanRevertingResidualFlowStructuredJointStudentTModel,
    target_u: torch.Tensor,
    mu: torch.Tensor,
    time_factor: torch.Tensor,
    time_diag: torch.Tensor,
    cell_factor: torch.Tensor,
    cell_diag: torch.Tensor,
    scale: torch.Tensor,
    flow_context: torch.Tensor,
    base_local_delta: torch.Tensor,
    block_logits: torch.Tensor,
):
    batch, n_frames, n_cells = target_u.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)
    assignments = model.decoder.assignment_index.to(target_u.device)
    n_assign = assignments.shape[0]

    log_prior_blocks = F.log_softmax(block_logits, dim=-1)
    log_prior = target_u.new_zeros(batch, n_assign)
    for b in range(model.decoder.n_blocks):
        idx = assignments[:, b].unsqueeze(0).expand(batch, -1)
        log_prior = log_prior + log_prior_blocks[:, b].gather(1, idx)

    local_delta_all = model.decoder.build_local_delta_components(base_local_delta, assignments)
    logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet_cov = n_cells * logdet_t + n_frames * logdet_c

    log_joint_chunks = []
    for start in range(0, n_assign, model.mix_chunk_size):
        end = min(start + model.mix_chunk_size, n_assign)
        local_delta = local_delta_all[:, start:end]
        local_scale = torch.exp(0.5 * local_delta)
        diff = (target_u.unsqueeze(1) - mu.unsqueeze(1)) / local_scale
        white_t = torch.linalg.solve_triangular(chol_t.unsqueeze(1), diff, upper=False)
        white = torch.linalg.solve_triangular(
            chol_c.unsqueeze(1), white_t.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
        white_flat = white.reshape(batch * (end - start), n_frames * n_cells)
        ctx = flow_context.unsqueeze(1).expand(batch, end - start, -1).reshape(batch * (end - start), -1)
        z, flow_logdet = model.flow(white_flat, ctx)
        base_logprob = model._base_logprob(z).view(batch, end - start)
        flow_logdet = flow_logdet.view(batch, end - start)
        logdet_local = 2.0 * torch.log(local_scale).sum(dim=(2, 3))
        comp_logprob = base_logprob + flow_logdet - 0.5 * (logdet_cov.unsqueeze(1) + logdet_local)
        log_joint_chunks.append(log_prior[:, start:end] + comp_logprob)

    log_joint = torch.cat(log_joint_chunks, dim=1)
    posterior_assign = F.softmax(log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True), dim=1)
    assignment_onehot = F.one_hot(assignments, num_classes=model.decoder.n_templates).float()
    posterior_blocks = torch.einsum("ba,akd->bkd", posterior_assign, assignment_onehot)
    prior_blocks = F.softmax(block_logits, dim=-1)
    return prior_blocks, posterior_blocks, posterior_assign, assignments


@torch.no_grad()
def analyze_178c_templates(
    model: ExactBlockMixtureMeanRevertingResidualFlowStructuredJointStudentTModel,
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
    history_01_all = denormalize_iv(history_norm)
    future_01_all = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)
    n_windows, future_len, n_cells = future_01_all.shape
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
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(hist_01_b)
        target_u = iv_to_unconstrained(
            fut_01_b,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        prior_blocks, posterior_blocks, _posterior_assign, _assignments = exact_block_posterior(
            model,
            target_u,
            mu,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        )
        prior_blocks_np = prior_blocks.detach().cpu().numpy()
        posterior_blocks_np = posterior_blocks.detach().cpu().numpy()
        prior_blocks_all[row0:end] = prior_blocks_np
        posterior_blocks_all[row0:end] = posterior_blocks_np
        prior_argmax_all[row0:end] = prior_blocks_np.argmax(axis=-1)
        posterior_argmax_all[row0:end] = posterior_blocks_np.argmax(axis=-1)
        prior_entropy_all[row0:end] = (-(prior_blocks_np * np.log(np.clip(prior_blocks_np, 1e-8, 1.0)))).sum(axis=-1)
        posterior_entropy_all[row0:end] = (-(posterior_blocks_np * np.log(np.clip(posterior_blocks_np, 1e-8, 1.0)))).sum(axis=-1)

        samples = model.sample_batched(hist_norm_b, n_samples=n_samples)
        future_grid = hist_01_b.new_tensor(fut_01_b).view(end - start, future_len, 5, 5)
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        covered_b = ((future_grid >= lo) & (future_grid <= hi)).detach().cpu().numpy()
        covered90[row0:end] = covered_b
        per_window_cov[row0:end] = covered_b.mean(axis=(1, 2, 3))
        row0 = end

    template_bank = model.decoder.block_template_bank.detach().cpu().numpy()  # [T,B,C]
    static_field = model.decoder.static_local_logvar.detach().cpu().numpy().reshape(model.decoder.n_frames, 5, 5)

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

    template_bank_summary = {}
    for b in range(n_blocks):
        block_name = f"block_{b + 1}"
        template_bank_summary[block_name] = {}
        for t in range(n_templates):
            vals = template_bank[t, b].reshape(5, 5)
            template_bank_summary[block_name][f"template_{t}"] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "top_boosted_cells": top_cells(vals, reverse=True, k=5),
                "top_suppressed_cells": top_cells(vals, reverse=False, k=5),
            }
        dists = {}
        for t1 in range(n_templates):
            for t2 in range(t1 + 1, n_templates):
                dists[f"{t1}-{t2}"] = float(np.sqrt(np.square(template_bank[t1, b] - template_bank[t2, b]).sum()))
        template_bank_summary[block_name]["pairwise_l2"] = dists

    under_records = []
    over_records = []
    for h in [7, 14, 30]:
        hid = h - 1
        turb_cov = covered90[turb_mask, hid].mean(axis=0)
        calm_cov = covered90[calm_mask, hid].mean(axis=0)
        for i in range(5):
            for j in range(5):
                if turb_cov[i, j] < LAYER2_LOW:
                    under_records.append(
                        {"regime": "turb", "horizon": h, "cell": [i, j], "coverage": float(turb_cov[i, j])}
                    )
                if calm_cov[i, j] > LAYER2_HIGH:
                    over_records.append(
                        {"regime": "calm", "horizon": h, "cell": [i, j], "coverage": float(calm_cov[i, j])}
                    )
                if turb_cov[i, j] > LAYER2_HIGH:
                    over_records.append(
                        {"regime": "turb", "horizon": h, "cell": [i, j], "coverage": float(turb_cov[i, j])}
                    )

    under_records.sort(key=lambda x: x["coverage"])
    over_records.sort(key=lambda x: x["coverage"], reverse=True)
    selected_under = under_records[:8]
    selected_over = over_records[:6]

    def analyze_slice(slice_record: dict[str, Any], mode: str) -> dict[str, Any]:
        regime = slice_record["regime"]
        h = slice_record["horizon"]
        i, j = slice_record["cell"]
        block_idx = (h - 1) // model.decoder.block_len
        base_mask = regime_masks[regime]
        if mode == "under":
            win_mask = base_mask & (~covered90[:, h - 1, i, j])
        else:
            win_mask = base_mask
        idx = np.where(win_mask)[0]
        if len(idx) == 0:
            return {
                **slice_record,
                "block": int(block_idx + 1),
                "n_windows": 0,
            }
        idx = idx[: min(len(idx), 128)]
        hist_01 = history_01_all[idx].to(device)
        fut_01 = future_01_all[idx].to(device)
        (
            _mu,
            _tf,
            _td,
            _cf,
            _cd,
            _scale,
            _ctx,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(hist_01)

        prior_block = prior_blocks_all[idx, block_idx]
        posterior_block = posterior_blocks_all[idx, block_idx]
        current_assign = torch.from_numpy(prior_argmax_all[idx]).to(device=device, dtype=torch.long)

        candidate_scales = []
        for t in range(n_templates):
            cand_assign = current_assign.clone()
            cand_assign[:, block_idx] = t
            local_delta = model.decoder.build_local_delta_for_batch_assignments(base_local_delta, cand_assign)
            scale_val = torch.exp(0.5 * local_delta[:, h - 1, i * 5 + j]).detach().cpu().numpy()
            candidate_scales.append(scale_val)
        candidate_scales = np.stack(candidate_scales, axis=1)
        current_template = prior_argmax_all[idx, block_idx]
        current_scale = candidate_scales[np.arange(len(idx)), current_template]

        if mode == "under":
            helpful_template = int(candidate_scales.mean(axis=0).argmax())
            helpful_scale = candidate_scales[:, helpful_template]
            multiplier = float(np.mean(helpful_scale / np.clip(current_scale, 1e-8, None)))
        else:
            helpful_template = int(candidate_scales.mean(axis=0).argmin())
            helpful_scale = candidate_scales[:, helpful_template]
            multiplier = float(np.mean(current_scale / np.clip(helpful_scale, 1e-8, None)))

        prior_help = float(prior_block[:, helpful_template].mean())
        posterior_help = float(posterior_block[:, helpful_template].mean())
        if mode == "under":
            if multiplier > 1.15 and posterior_help > prior_help + 0.10:
                diagnosis = "prior_underuses_helpful_template"
            elif multiplier <= 1.10:
                diagnosis = "template_family_too_weak"
            elif posterior_help >= 0.50 and multiplier <= 1.20:
                diagnosis = "posterior_already_prefers_helpful_template"
            else:
                diagnosis = "mixed"
        else:
            if multiplier > 1.15 and posterior_help > prior_help + 0.10:
                diagnosis = "prior_underuses_helpful_narrow_template"
            elif multiplier <= 1.10:
                diagnosis = "template_family_too_weak"
            else:
                diagnosis = "mixed"

        return {
            **slice_record,
            "block": int(block_idx + 1),
            "n_windows": int(len(idx)),
            "prior_block_mean": prior_block.mean(axis=0).tolist(),
            "posterior_block_mean": posterior_block.mean(axis=0).tolist(),
            "current_template_mode": int(np.bincount(current_template, minlength=n_templates).argmax()),
            "helpful_template": helpful_template,
            "helpful_scale_multiplier": multiplier,
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
            "turb_window_coverage_mean": float(per_window_cov[turb_mask].mean()),
            "calm_window_coverage_mean": float(per_window_cov[calm_mask].mean()),
            "prior_posterior_argmax_agreement_by_block": (
                prior_argmax_all == posterior_argmax_all
            ).mean(axis=0).tolist(),
        },
        "regime_block_usage": regime_block_usage,
        "regime_block_posterior": regime_block_post,
        "regime_entropy": regime_entropy,
        "top_assignment_patterns": top_assignment_patterns,
        "template_bank_summary": template_bank_summary,
        "static_field_extremes": {
            "top_boosted_cells": top_cells(static_field.mean(axis=0), reverse=True, k=8),
            "top_suppressed_cells": top_cells(static_field.mean(axis=0), reverse=False, k=8),
        },
        "hard_under_slices": analyzed_under,
        "high_overcoverage_slices": analyzed_over,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze 178c templates and assignment behavior")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/backfill/exact_block_mixture_mean_reverting_residual_flow_structured_joint_student_t_178c/best_model.pt",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-05/analysis/178c_mechanistic",
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
    analysis = analyze_178c_templates(
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
