#!/usr/bin/env python
"""
Focused jump-mechanism review for 180a.

Questions:
  1. Is the explicit jump branch actually active on the saved checkpoints?
  2. Does the jump branch materially change sampled paths relative to smooth-only sampling?
  3. If we force nonzero jumps, do pathwise jump metrics improve, or are the fixed atoms too coarse?
  4. Is 180a failing because the jump branch is dead, or because the smooth and jump branches fight over the same residual mass?
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
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    effective_rank,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_177a_mean_reverting_local_template_mixture import (
    aggregate_slope_ratio,
)
from experiments.backfill.block_ar.train_180a_centered_smooth_jump_residual import (
    CenteredSmoothJumpResidualMeanRevertingCovarianceMixtureModel,
)


SUITE_KEYS = [
    "surface",
    "coverage",
    "conditionality",
    "time_series",
    "block_ar",
    "cointegration",
    "regime_coverage",
    "distributional",
    "cross_cell_correlation",
    "mean_reversion",
    "pathwise_jump_realism",
]


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def count_passes(summary: dict[str, Any]) -> tuple[int, int]:
    passed = 0
    total = 0
    for key in SUITE_KEYS:
        block = summary.get(key, {})
        if "overall_pass" in block:
            total += 1
            if block["overall_pass"]:
                passed += 1
    return passed, total


def quantile_profile(x: np.ndarray, qs: tuple[float, ...] = (0.5, 0.75, 0.9, 0.99)) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {str(q): float(np.quantile(x, q)) for q in qs}


def ks_detail(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if len(x) == 0 or len(y) == 0:
        return {"ks_stat": float("nan"), "peak_x": float("nan"), "cdf_gt": float("nan"), "cdf_gen": float("nan")}
    grid = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    diff = np.abs(cdf_x - cdf_y)
    idx = int(np.argmax(diff))
    return {
        "ks_stat": float(diff[idx]),
        "peak_x": float(grid[idx]),
        "cdf_gt": float(cdf_x[idx]),
        "cdf_gen": float(cdf_y[idx]),
    }


def pearson_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan")
    xc = x - x.mean()
    var = np.mean(np.square(xc))
    if var <= 1e-12:
        return float("nan")
    return float(np.mean(np.power(xc, 4)) / (var * var))


def summarize_distribution(gt: np.ndarray, gen: np.ndarray) -> dict[str, Any]:
    q_gt = quantile_profile(gt)
    q_gen = quantile_profile(gen)
    ratios = {k: float(q_gen[k] / max(q_gt[k], 1e-12)) for k in q_gt}
    return {
        "gt_quantiles": q_gt,
        "gen_quantiles": q_gen,
        "ratio_quantiles": ratios,
        "ks": ks_detail(gt, gen),
        "gt_kurtosis": pearson_kurtosis(gt),
        "gen_kurtosis": pearson_kurtosis(gen),
        "kurtosis_ratio": float(pearson_kurtosis(gen) / max(pearson_kurtosis(gt), 1e-12)),
    }


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float, float]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80
    return calm, turb, q20, q80


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "centered_smooth_jump_residual_mean_reverting_covariance_mixture_structured_joint_student_t_180a"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = CenteredSmoothJumpResidualMeanRevertingCovarianceMixtureModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        jump_config=raw_config["jump"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
        mix_chunk_size=raw_config.get("mix_chunk_size", 27),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def sample_future_mode(
    model: CenteredSmoothJumpResidualMeanRevertingCovarianceMixtureModel,
    history_01: torch.Tensor,
    n_samples: int,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
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
    ) = model.forward_from_history(history_01)
    batch, n_frames, n_cells = mu.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)

    torch.manual_seed(seed)
    base = torch.distributions.StudentT(df=model.base_nu)
    z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
    ctx = flow_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
    smooth_basis_flat, _ = model.flow.inverse_basis_flat(z, ctx)
    smooth_basis_flat_3d = smooth_basis_flat.view(batch, n_samples, -1)

    probs, scales = model.jump.mixture_params(flow_context)
    if mode == "default":
        jump_shift, jump_stats = model.jump.sample_shifts(flow_context, n_samples=n_samples)
    elif mode == "zero":
        jump_shift = smooth_basis_flat_3d.new_zeros(batch, n_samples, smooth_basis_flat_3d.shape[-1])
        jump_stats = {
            "jump_zero_prob": probs[:, 0],
            "jump_nonzero_prob": 1.0 - probs[:, 0],
            "jump_scale_mean": scales.mean(dim=-1),
            "jump_scale_max": scales.max(dim=-1).values,
        }
    elif mode == "force_nonzero":
        nonzero_probs = probs[:, 1:].clamp_min(1e-8)
        nonzero_probs = nonzero_probs / nonzero_probs.sum(dim=-1, keepdim=True)
        sampled = 1 + torch.multinomial(nonzero_probs, num_samples=n_samples, replacement=True)
        signs = torch.where(
            torch.randint(0, 2, (batch, n_samples), device=mu.device) > 0,
            torch.ones(batch, n_samples, device=mu.device),
            -torch.ones(batch, n_samples, device=mu.device),
        )
        jump_shift = smooth_basis_flat_3d.new_zeros(batch, n_samples, smooth_basis_flat_3d.shape[-1])
        atom_idx = (sampled - 1).long()
        amps = scales.unsqueeze(1).expand(-1, n_samples, -1).gather(2, atom_idx.unsqueeze(-1)).squeeze(-1)
        atoms = model.jump.atom_bank[atom_idx]
        jump_shift = signs.unsqueeze(-1) * amps.unsqueeze(-1) * atoms
        jump_stats = {
            "jump_zero_prob": probs[:, 0],
            "jump_nonzero_prob": 1.0 - probs[:, 0],
            "jump_scale_mean": scales.mean(dim=-1),
            "jump_scale_max": scales.max(dim=-1).values,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    block_probs = F.softmax(block_logits, dim=-1)
    torch.manual_seed(seed + 1)
    sampled_blocks = []
    for b in range(model.decoder.n_blocks):
        sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
        sampled_blocks.append(sampled)
    sampled_assign = torch.stack(sampled_blocks, dim=-1)
    assign_flat = sampled_assign.reshape(batch * n_samples, model.decoder.n_blocks)
    sampled_factors, _logdet_cov, _offdiag_rms = model.decoder.build_template_factors(assign_flat)

    basis_with_jump = smooth_basis_flat_3d + jump_shift
    base_white_flat = model.flow.from_basis_flat(basis_with_jump.reshape(batch * n_samples, -1))
    base_white = base_white_flat.view(batch * n_samples, model.decoder.n_blocks, model.decoder.block_len, n_cells)
    lhs = base_white.permute(0, 1, 3, 2).reshape(batch * n_samples * model.decoder.n_blocks, n_cells, model.decoder.block_len)
    factor_flat = sampled_factors.reshape(batch * n_samples * model.decoder.n_blocks, n_cells, n_cells)
    routed_white = torch.matmul(factor_flat, lhs)
    routed_white = routed_white.reshape(batch * n_samples, model.decoder.n_blocks, n_cells, model.decoder.block_len).permute(0, 1, 3, 2)
    routed_white = routed_white.reshape(batch, n_samples, n_frames, n_cells)

    temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
    noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
    shared_local_delta = model.decoder.build_shared_local_delta(base_local_delta)
    local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)
    u_samples = mu.unsqueeze(1) + noise * local_scale
    iv_samples = unconstrained_to_iv(u_samples, lo=model.support_lo, hi=model.support_hi).reshape(batch, n_samples, n_frames, 5, 5)

    jump_l2 = jump_shift.pow(2).sum(dim=-1).sqrt()
    smooth_l2 = smooth_basis_flat_3d.pow(2).sum(dim=-1).sqrt()
    high_mask = model.flow.geometry.high_band_mask().to(jump_shift.device, jump_shift.dtype).view(1, 1, -1)
    jump_high_share = (jump_shift.pow(2) * high_mask).sum(dim=-1) / jump_shift.pow(2).sum(dim=-1).clamp_min(1e-12)

    meta = {
        "jump_stats": jump_stats,
        "sampled_assign": sampled_assign,
        "jump_shift_l2": jump_l2,
        "smooth_basis_l2": smooth_l2,
        "jump_high_share": jump_high_share,
        "jump_nonzero_realized": (jump_l2 > 1e-8).float(),
        "block_probs": block_probs,
    }
    return iv_samples, meta


def summarize_mode_vs_gt(
    gt_future_01: torch.Tensor,
    prev_01: torch.Tensor,
    samples_01: torch.Tensor,
    geometry,
) -> dict[str, Any]:
    sample_mean = samples_01.mean(dim=1)
    gt_path = torch.cat([prev_01[:, None], gt_future_01], dim=1)
    gen_path = torch.cat([prev_01[:, None, None].expand(-1, samples_01.shape[1], -1, -1, -1), samples_01], dim=2)
    gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
    gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]

    pooled_abs_delta_gt = gt_diff.abs().reshape(-1).cpu().numpy()
    pooled_abs_delta_gen = gen_diff.abs().reshape(-1).cpu().numpy()
    gt_path_max = gt_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy()
    gen_path_max = gen_diff.abs().amax(dim=(2, 3, 4)).reshape(-1).cpu().numpy()

    gt_coeff = geometry.to_basis(gt_diff.reshape(gt_diff.shape[0], geometry.n_frames, geometry.n_cells))
    gen_coeff = geometry.to_basis(gen_diff.reshape(gen_diff.shape[0] * gen_diff.shape[1], geometry.n_frames, geometry.n_cells))
    high_mask = geometry.high_band_mask().to(gt_coeff.device, gt_coeff.dtype).reshape(1, geometry.n_frames, geometry.n_cells)
    gt_high_max = geometry.from_basis(gt_coeff * high_mask).abs().amax(dim=(1, 2)).cpu().numpy()
    gen_high_max = geometry.from_basis(gen_coeff * high_mask).abs().amax(dim=(1, 2)).cpu().numpy()

    prev_flat = prev_01.reshape(prev_01.shape[0], -1)
    gt_next_flat = gt_future_01[:, 0].reshape(gt_future_01.shape[0], -1)
    pred_next_flat = sample_mean[:, 0].reshape(sample_mean.shape[0], -1)

    total_var_gt = gt_diff.abs().sum(dim=(1, 2, 3)).cpu().numpy()
    total_var_gen = gen_diff.abs().sum(dim=(2, 3, 4)).reshape(-1).cpu().numpy()

    return {
        "daily_abs_delta": summarize_distribution(pooled_abs_delta_gt, pooled_abs_delta_gen),
        "pathwise_max_jump": summarize_distribution(gt_path_max, gen_path_max),
        "pathwise_high_band_max_jump": summarize_distribution(gt_high_max, gen_high_max),
        "path_total_variation": summarize_distribution(total_var_gt, total_var_gen),
        "first_step_mean_reversion_ratio": float(aggregate_slope_ratio(prev_flat, gt_next_flat, pred_next_flat)),
    }


def regime_slice_summary(values: np.ndarray, calm_mask: np.ndarray, turb_mask: np.ndarray) -> dict[str, Any]:
    return {
        "all": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
            "max": float(values.max()),
        },
        "calm": {
            "mean": float(values[calm_mask].mean()),
            "median": float(np.median(values[calm_mask])),
            "q90": float(np.quantile(values[calm_mask], 0.9)),
            "max": float(values[calm_mask].max()),
        },
        "turb": {
            "mean": float(values[turb_mask].mean()),
            "median": float(np.median(values[turb_mask])),
            "q90": float(np.quantile(values[turb_mask], 0.9)),
            "max": float(values[turb_mask].max()),
        },
    }


@torch.no_grad()
def analyze_checkpoint(
    model: CenteredSmoothJumpResidualMeanRevertingCovarianceMixtureModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: str,
    batch_size: int,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    prev_01 = history_01[:, -1]
    calm_mask, turb_mask, q20, q80 = regime_masks_from_history(history_norm.cpu())

    jump_zero_probs = []
    jump_nonzero_probs = []
    jump_scale_means = []
    jump_scale_maxes = []
    jump_entropies = []
    atom_prob_means = []
    nonzero_prob_by_atom = []
    block_gate_entropy = []
    block_gate_max = []

    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist_b = history_01[start:end]
        outputs = model.forward_from_history(hist_b)
        flow_context = outputs[6]
        block_logits = outputs[8]
        probs, scales = model.jump.mixture_params(flow_context)
        jump_zero_probs.append(probs[:, 0].cpu().numpy())
        jump_nonzero_probs.append((1.0 - probs[:, 0]).cpu().numpy())
        jump_scale_means.append(scales.mean(dim=-1).cpu().numpy())
        jump_scale_maxes.append(scales.max(dim=-1).values.cpu().numpy())
        jump_entropies.append((-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)).cpu().numpy())
        atom_prob_means.append(probs.mean(dim=0).cpu().numpy())
        nonzero_prob_by_atom.append(probs[:, 1:].cpu().numpy())
        block_probs = F.softmax(block_logits, dim=-1)
        block_gate_entropy.append((-(block_probs * torch.log(block_probs.clamp_min(1e-8))).sum(dim=-1)).mean(dim=-1).cpu().numpy())
        block_gate_max.append(block_probs.max(dim=-1).values.mean(dim=-1).cpu().numpy())

    jump_zero_probs = np.concatenate(jump_zero_probs)
    jump_nonzero_probs = np.concatenate(jump_nonzero_probs)
    jump_scale_means = np.concatenate(jump_scale_means)
    jump_scale_maxes = np.concatenate(jump_scale_maxes)
    jump_entropies = np.concatenate(jump_entropies)
    atom_prob_means = np.stack(atom_prob_means, axis=0).mean(axis=0)
    nonzero_prob_by_atom = np.concatenate(nonzero_prob_by_atom, axis=0)
    block_gate_entropy = np.concatenate(block_gate_entropy)
    block_gate_max = np.concatenate(block_gate_max)

    atom_meta = []
    for i, meta in enumerate(model.jump_atom_metadata):
        atom_meta.append(
            {
                "atom_index": int(i),
                "block": int(meta["block"]),
                "band": str(meta["band"]),
                "mean_prob": float(atom_prob_means[i + 1]),
                "mean_nonzero_prob_mass": float(nonzero_prob_by_atom[:, i].mean()),
            }
        )

    band_mass = {}
    for band in ["low", "mid", "high"]:
        idx = [i for i, meta in enumerate(model.jump_atom_metadata) if meta["band"] == band]
        band_mass[band] = float(nonzero_prob_by_atom[:, idx].sum(axis=1).mean()) if idx else 0.0

    mode_results = {}
    batch_modes = {k: [] for k in ["default", "zero", "force_nonzero"]}
    direct_jump_effect = {k: [] for k in ["mean_abs_iv_shift", "max_abs_iv_shift", "jump_l2", "smooth_l2", "jump_high_share", "realized_nonzero_rate"]}

    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist_b = history_01[start:end]
        fut_b = future_01[start:end]
        samples_default, meta_default = sample_future_mode(model, hist_b, n_samples=n_samples, mode="default", seed=seed + start)
        samples_zero, _meta_zero = sample_future_mode(model, hist_b, n_samples=n_samples, mode="zero", seed=seed + start)
        samples_force, _meta_force = sample_future_mode(model, hist_b, n_samples=n_samples, mode="force_nonzero", seed=seed + start)

        batch_modes["default"].append(samples_default.cpu())
        batch_modes["zero"].append(samples_zero.cpu())
        batch_modes["force_nonzero"].append(samples_force.cpu())

        diff = (samples_default - samples_zero).abs()
        direct_jump_effect["mean_abs_iv_shift"].append(diff.mean(dim=(1, 2, 3, 4)).cpu().numpy())
        direct_jump_effect["max_abs_iv_shift"].append(diff.amax(dim=(1, 2, 3, 4)).cpu().numpy())
        direct_jump_effect["jump_l2"].append(meta_default["jump_shift_l2"].mean(dim=1).cpu().numpy())
        direct_jump_effect["smooth_l2"].append(meta_default["smooth_basis_l2"].mean(dim=1).cpu().numpy())
        direct_jump_effect["jump_high_share"].append(meta_default["jump_high_share"].mean(dim=1).cpu().numpy())
        direct_jump_effect["realized_nonzero_rate"].append(meta_default["jump_nonzero_realized"].mean(dim=1).cpu().numpy())

    for key in batch_modes:
        batch_modes[key] = torch.cat(batch_modes[key], dim=0)
    for key in direct_jump_effect:
        direct_jump_effect[key] = np.concatenate(direct_jump_effect[key], axis=0)

    geometry = model.flow.geometry
    for mode, samples in batch_modes.items():
        mode_results[mode] = summarize_mode_vs_gt(
            gt_future_01=future_01,
            prev_01=prev_01,
            samples_01=samples.to(future_01.device),
            geometry=geometry,
        )

    return {
        "jump_prior_behavior": {
            "zero_prob": regime_slice_summary(jump_zero_probs, calm_mask, turb_mask),
            "nonzero_prob": regime_slice_summary(jump_nonzero_probs, calm_mask, turb_mask),
            "scale_mean": regime_slice_summary(jump_scale_means, calm_mask, turb_mask),
            "scale_max": regime_slice_summary(jump_scale_maxes, calm_mask, turb_mask),
            "entropy": regime_slice_summary(jump_entropies, calm_mask, turb_mask),
            "band_mass_mean": band_mass,
            "atom_probability_table": atom_meta,
            "regime_thresholds": {"vov_q20": q20, "vov_q80": q80},
        },
        "shared_router_behavior": {
            "block_gate_entropy": regime_slice_summary(block_gate_entropy, calm_mask, turb_mask),
            "block_gate_max": regime_slice_summary(block_gate_max, calm_mask, turb_mask),
        },
        "direct_jump_effect": {
            "mean_abs_iv_shift": regime_slice_summary(direct_jump_effect["mean_abs_iv_shift"], calm_mask, turb_mask),
            "max_abs_iv_shift": regime_slice_summary(direct_jump_effect["max_abs_iv_shift"], calm_mask, turb_mask),
            "jump_l2": regime_slice_summary(direct_jump_effect["jump_l2"], calm_mask, turb_mask),
            "smooth_l2": regime_slice_summary(direct_jump_effect["smooth_l2"], calm_mask, turb_mask),
            "jump_to_smooth_l2_ratio": regime_slice_summary(
                direct_jump_effect["jump_l2"] / np.maximum(direct_jump_effect["smooth_l2"], 1e-12),
                calm_mask,
                turb_mask,
            ),
            "jump_high_share": regime_slice_summary(direct_jump_effect["jump_high_share"], calm_mask, turb_mask),
            "realized_nonzero_rate": regime_slice_summary(direct_jump_effect["realized_nonzero_rate"], calm_mask, turb_mask),
        },
        "mode_counterfactuals": mode_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Focused jump mechanism review for 180a")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--summary_path", type=str, required=True)
    parser.add_argument("--comparison_summary_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, checkpoint = load_model(args.model_path, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )

    summary = load_json(args.summary_path)
    comparison = load_json(args.comparison_summary_path)
    chosen_passed, chosen_total = count_passes(summary)
    comp_passed, comp_total = count_passes(comparison)

    analysis = analyze_checkpoint(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    analysis["checkpoint_review"] = {
        "chosen_model": {
            "path": args.model_path,
            "epoch": int(checkpoint.get("epoch", -1)),
            "passed_suites": chosen_passed,
            "total_suites": chosen_total,
            "coverage_pass": bool(summary["coverage"]["overall_pass"]),
            "conditionality_pass": bool(summary["conditionality"]["overall_pass"]),
            "time_series_pass": bool(summary["time_series"]["overall_pass"]),
            "distributional_pass": bool(summary["distributional"]["overall_pass"]),
            "mean_reversion_pass": bool(summary["mean_reversion"]["overall_pass"]),
            "pathwise_jump_pass": bool(summary["pathwise_jump_realism"]["overall_pass"]),
            "mean_reversion_ratio": float(summary["mean_reversion"]["mr_gt_ratio"]),
            "pathwise_jump_ks": float(summary["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
            "kurtosis_ratio": float(summary["time_series"]["kurtosis"]["kurtosis_ratio"]),
        },
        "comparison_checkpoint": {
            "path": args.comparison_summary_path,
            "passed_suites": comp_passed,
            "total_suites": comp_total,
            "coverage_pass": bool(comparison["coverage"]["overall_pass"]),
            "conditionality_pass": bool(comparison["conditionality"]["overall_pass"]),
            "time_series_pass": bool(comparison["time_series"]["overall_pass"]),
            "distributional_pass": bool(comparison["distributional"]["overall_pass"]),
            "mean_reversion_pass": bool(comparison["mean_reversion"]["overall_pass"]),
            "pathwise_jump_pass": bool(comparison["pathwise_jump_realism"]["overall_pass"]),
            "mean_reversion_ratio": float(comparison["mean_reversion"]["mr_gt_ratio"]),
            "pathwise_jump_ks": float(comparison["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
            "kurtosis_ratio": float(comparison["time_series"]["kurtosis"]["kurtosis_ratio"]),
        },
    }
    analysis["benchmark_anchor"] = {
        "model_type": checkpoint["config"]["type"],
        "summary_path": args.summary_path,
        "comparison_summary_path": args.comparison_summary_path,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_serializable(analysis), indent=2))
    print(f"Saved review to {output_path}")


if __name__ == "__main__":
    main()
