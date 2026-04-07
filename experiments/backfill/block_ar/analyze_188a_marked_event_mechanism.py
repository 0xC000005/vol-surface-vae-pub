#!/usr/bin/env python
"""
Focused mechanistic review of 188a_v0.

Questions:
  1. Do the learned marked-event objects overlap the exact hard S3/S7 late-turbulent
     slices, or are they active elsewhere?
  2. Are the event slots selective in time/node space, or effectively always-on and
     diffuse?
  3. Does turning the event branch off help or hurt broad fidelity, kurtosis, and
     hard-slice coverage?
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
from experiments.backfill.block_ar.train_177a_mean_reverting_local_template_mixture import aggregate_slope_ratio
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import ks_statistic, pearson_kurtosis
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_188a_graph_group_marked_event_residual import (
    MarkedEventResidualModel,
)


SELECT_HORIZONS = [13, 29]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def safe_share(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def build_model(ckpt_path: Path, device: torch.device) -> tuple[MarkedEventResidualModel, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = MarkedEventResidualModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        slot_config=cfg["event_slots"],
        event_config=cfg["event"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, payload


def decode_anchor_field(model: MarkedEventResidualModel, slot_dec: dict[str, torch.Tensor]) -> torch.Tensor:
    time_frame_probs = slot_dec["time_probs"][:, :, model.event_block_ids]
    anchor = (
        slot_dec["active_prob"].unsqueeze(-1).unsqueeze(-1)
        * time_frame_probs.unsqueeze(-1)
        * slot_dec["node_probs"].unsqueeze(-2)
    ).sum(dim=1)
    return anchor


def teacher_slot_dec(model: MarkedEventResidualModel, teacher_slots: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "active_prob": teacher_slots["active"].to(dtype=dtype),
        "time_probs": F.one_hot(teacher_slots["time_idx"], num_classes=model.n_event_blocks).to(dtype=dtype),
        "node_probs": F.one_hot(teacher_slots["node_idx"], num_classes=model.n_nodes).to(dtype=dtype),
        "amp": teacher_slots["amp"].to(dtype=dtype),
        "radius": teacher_slots["radius"].to(dtype=dtype),
        "duration": teacher_slots["duration"].to(dtype=dtype),
        "band": torch.zeros(
            teacher_slots["active"].shape[0],
            teacher_slots["active"].shape[1],
            3,
            device=teacher_slots["active"].device,
            dtype=dtype,
        ),
    }


@torch.no_grad()
def sample_eval_arrays(
    model: MarkedEventResidualModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    n_samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], 5, 5)

    n = history_norm.shape[0]
    future_len = future_01.shape[1]
    samples_all = []
    for start in range(0, n, batch_size):
        h = history_01[start : start + batch_size]
        samples_u = model.sample_future_u(h, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(
            samples_u,
            lo=model.support_lo,
            hi=model.support_hi,
        ).view(h.shape[0], n_samples, future_len, 5, 5)
        samples_all.append(samples_01.detach().cpu())
    cond_samples = torch.cat(samples_all, dim=0).numpy()
    ground_truth = future_01.detach().cpu().numpy()
    history = history_01.detach().cpu().numpy()

    lo90 = np.quantile(cond_samples, 0.05, axis=1)
    hi90 = np.quantile(cond_samples, 0.95, axis=1)
    coverage = ((ground_truth >= lo90) & (ground_truth <= hi90)).astype(np.float32)
    window_cov = coverage.mean(axis=(1, 2, 3))
    width = (hi90 - lo90).astype(np.float32)

    gt_path = np.concatenate([history[:, -1:, :, :], ground_truth], axis=1)
    gen_path = np.concatenate(
        [np.repeat(history[:, None, -1:, :, :], cond_samples.shape[1], axis=1), cond_samples],
        axis=2,
    )
    gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
    gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]
    pooled_gt = np.abs(gt_diff).reshape(-1)
    pooled_gen = np.abs(gen_diff).reshape(-1)
    q50 = float(np.quantile(pooled_gt, 0.50))
    q95 = float(np.quantile(pooled_gt, 0.95))
    q99 = float(np.quantile(pooled_gt, 0.99))
    quiet_ratio = float((pooled_gen <= q50).mean() / max((pooled_gt <= q50).mean(), 1e-12))
    shoulder_ratio = float(
        (((pooled_gen > q50) & (pooled_gen <= q95)).mean())
        / max((((pooled_gt > q50) & (pooled_gt <= q95)).mean()), 1e-12)
    )
    extreme_ratio = float((pooled_gen > q99).mean() / max((pooled_gt > q99).mean(), 1e-12))

    prev = torch.from_numpy(history[:, -1])
    gt_next = torch.from_numpy(ground_truth[:, 0])
    sample_next = torch.from_numpy(cond_samples[:, :, 0].mean(axis=1))
    mr_ratio = aggregate_slope_ratio(prev, gt_next, sample_next)
    gt_path_max = np.abs(gt_path[:, 1:] - gt_path[:, :-1]).max(axis=(1, 2, 3))
    gen_path_max = np.abs(gen_path[:, :, 1:] - gen_path[:, :, :-1]).max(axis=(1, 2, 3))[:, 0]
    jump_ks = ks_statistic(gt_path_max, gen_path_max)
    kurt_ratio = float(
        pearson_kurtosis(torch.from_numpy(gen_diff[:, 0]))
        / max(pearson_kurtosis(torch.from_numpy(gt_diff)), 1e-12)
    )

    return {
        "history": history,
        "ground_truth": ground_truth,
        "coverage": coverage,
        "window_coverage": window_cov,
        "bad_window_rate": float((window_cov < 0.50).mean()),
        "mr_ratio": float(mr_ratio),
        "jump_ks": float(jump_ks),
        "kurtosis_ratio": kurt_ratio,
        "quiet_ratio": quiet_ratio,
        "shoulder_ratio": shoulder_ratio,
        "extreme_ratio": extreme_ratio,
        "mean_width": float(width.mean()),
        "overall_cov90": float(coverage.mean()),
    }


@torch.no_grad()
def marked_event_state_analysis(
    model: MarkedEventResidualModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)

    target_event_abs = []
    teacher_event_abs = []
    prior_event_abs = []
    post_event_abs = []
    teacher_anchor = []
    prior_anchor = []
    post_anchor = []
    teacher_active = []
    prior_active = []
    post_active = []
    prior_slot_count = []
    post_slot_count = []
    prior_time_top1 = []
    prior_node_top1 = []
    post_time_top1 = []
    post_node_top1 = []
    prior_amp_abs = []
    post_amp_abs = []
    prior_radius = []
    post_radius = []
    prior_duration = []
    post_duration = []
    prior_event_band_abs = []
    post_event_band_abs = []

    for start in range(0, history_norm.shape[0], batch_size):
        h_norm = history_norm[start : start + batch_size]
        f_norm = future_norm[start : start + batch_size]
        h_01 = denormalize_iv(h_norm)
        f_01 = denormalize_iv(f_norm).reshape(h_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        target_u = iv_to_unconstrained(f_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)

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
        ) = model.forward_from_history(h_01)
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=mu,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            base_local_delta=base_local_delta,
            block_logits=block_logits,
        ).view(h_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, target_band_log = compute_control_targets(model, target_basis)
        z_t = target_basis.reshape(target_basis.shape[0], -1)
        t = torch.ones(target_basis.shape[0], device=device, dtype=target_basis.dtype)

        (
            _raw_local,
            _raw_band,
            quiet_metric_local,
            quiet_metric_band,
            _local_gate,
            _band_gate,
        ) = super(MarkedEventResidualModel, model).build_state_metric_controls(z_t, t, path_context)

        target_event_local = (target_local_log - quiet_metric_local).clamp(
            min=-model.event_config["event_local_clip"],
            max=model.event_config["event_local_clip"],
        )
        target_event_band = (target_band_log - quiet_metric_band).clamp(
            min=-model.event_config["event_band_clip"],
            max=model.event_config["event_band_clip"],
        )
        _ = target_event_band
        teacher_slots = model.extract_teacher_slots(target_event_local.detach())
        slot_pack = model.infer_slots(path_context, teacher_slots=teacher_slots)
        prior_dec = model._decode_slots(slot_pack["prior"])
        post_dec = model._decode_slots(slot_pack["post"])
        teacher_dec = teacher_slot_dec(model, teacher_slots, target_basis.dtype)

        prior_event_local, _ = model.decode_event_field(prior_dec)
        post_event_local, post_event_band = model.decode_event_field(post_dec)
        teacher_event_local, _ = model.decode_event_field(teacher_dec)
        _, prior_event_band = model.decode_event_field(prior_dec)

        target_event_abs.append(target_event_local.abs().detach().cpu().numpy())
        teacher_event_abs.append(teacher_event_local.abs().detach().cpu().numpy())
        prior_event_abs.append(prior_event_local.abs().detach().cpu().numpy())
        post_event_abs.append(post_event_local.abs().detach().cpu().numpy())
        teacher_anchor.append(decode_anchor_field(model, teacher_dec).detach().cpu().numpy())
        prior_anchor.append(decode_anchor_field(model, prior_dec).detach().cpu().numpy())
        post_anchor.append(decode_anchor_field(model, post_dec).detach().cpu().numpy())
        teacher_active.append(teacher_slots["active"].mean(dim=1).detach().cpu().numpy())
        prior_active.append(prior_dec["active_prob"].mean(dim=1).detach().cpu().numpy())
        post_active.append(post_dec["active_prob"].mean(dim=1).detach().cpu().numpy())
        prior_slot_count.append((prior_dec["active_prob"] > 0.5).float().sum(dim=1).detach().cpu().numpy())
        post_slot_count.append((post_dec["active_prob"] > 0.5).float().sum(dim=1).detach().cpu().numpy())
        prior_time_top1.append(prior_dec["time_probs"].amax(dim=-1).mean(dim=1).detach().cpu().numpy())
        prior_node_top1.append(prior_dec["node_probs"].amax(dim=-1).mean(dim=1).detach().cpu().numpy())
        post_time_top1.append(post_dec["time_probs"].amax(dim=-1).mean(dim=1).detach().cpu().numpy())
        post_node_top1.append(post_dec["node_probs"].amax(dim=-1).mean(dim=1).detach().cpu().numpy())
        prior_amp_abs.append(prior_dec["amp"].abs().mean(dim=1).detach().cpu().numpy())
        post_amp_abs.append(post_dec["amp"].abs().mean(dim=1).detach().cpu().numpy())
        prior_radius.append(prior_dec["radius"].mean(dim=1).detach().cpu().numpy())
        post_radius.append(post_dec["radius"].mean(dim=1).detach().cpu().numpy())
        prior_duration.append(prior_dec["duration"].mean(dim=1).detach().cpu().numpy())
        post_duration.append(post_dec["duration"].mean(dim=1).detach().cpu().numpy())
        prior_event_band_abs.append(prior_event_band.abs().mean(dim=1).detach().cpu().numpy())
        post_event_band_abs.append(post_event_band.abs().mean(dim=1).detach().cpu().numpy())

    return {
        "target_event_abs": np.concatenate(target_event_abs, axis=0),
        "teacher_event_abs": np.concatenate(teacher_event_abs, axis=0),
        "prior_event_abs": np.concatenate(prior_event_abs, axis=0),
        "post_event_abs": np.concatenate(post_event_abs, axis=0),
        "teacher_anchor": np.concatenate(teacher_anchor, axis=0),
        "prior_anchor": np.concatenate(prior_anchor, axis=0),
        "post_anchor": np.concatenate(post_anchor, axis=0),
        "teacher_active": np.concatenate(teacher_active, axis=0),
        "prior_active": np.concatenate(prior_active, axis=0),
        "post_active": np.concatenate(post_active, axis=0),
        "prior_slot_count": np.concatenate(prior_slot_count, axis=0),
        "post_slot_count": np.concatenate(post_slot_count, axis=0),
        "prior_time_top1": np.concatenate(prior_time_top1, axis=0),
        "prior_node_top1": np.concatenate(prior_node_top1, axis=0),
        "post_time_top1": np.concatenate(post_time_top1, axis=0),
        "post_node_top1": np.concatenate(post_node_top1, axis=0),
        "prior_amp_abs": np.concatenate(prior_amp_abs, axis=0),
        "post_amp_abs": np.concatenate(post_amp_abs, axis=0),
        "prior_radius": np.concatenate(prior_radius, axis=0),
        "post_radius": np.concatenate(post_radius, axis=0),
        "prior_duration": np.concatenate(prior_duration, axis=0),
        "post_duration": np.concatenate(post_duration, axis=0),
        "prior_event_band_abs": np.concatenate(prior_event_band_abs, axis=0),
        "post_event_band_abs": np.concatenate(post_event_band_abs, axis=0),
    }


def summarize_overlap(
    coverage: np.ndarray,
    turb_mask: np.ndarray,
    stats: dict[str, np.ndarray],
) -> dict[str, Any]:
    n, t_len, h, w = coverage.shape
    coverage_flat = coverage.reshape(n, t_len, h * w)
    hard = np.zeros_like(coverage_flat, dtype=bool)
    easy = np.zeros_like(coverage_flat, dtype=bool)
    overwide = np.zeros_like(coverage_flat, dtype=bool)
    for h_idx in SELECT_HORIZONS:
        if h_idx >= t_len:
            continue
        hard[:, h_idx] = turb_mask[:, None] & (coverage_flat[:, h_idx] < 0.70)
        easy[:, h_idx] = turb_mask[:, None] & (coverage_flat[:, h_idx] > 0.85)
        overwide[:, h_idx] = turb_mask[:, None] & (coverage_flat[:, h_idx] > 0.95)

    late_turb_mask = np.zeros_like(coverage_flat, dtype=bool)
    for h_idx in SELECT_HORIZONS:
        if h_idx < t_len:
            late_turb_mask[:, h_idx] = turb_mask[:, None]

    hard_float = hard.astype(np.float32)
    out = {
        "hard_point_count": int(hard.sum()),
        "easy_point_count": int(easy.sum()),
        "overwide_point_count": int(overwide.sum()),
        "late_turb_point_count": int(late_turb_mask.sum()),
    }

    for prefix in ["teacher_anchor", "prior_anchor", "post_anchor", "target_event_abs", "teacher_event_abs", "prior_event_abs", "post_event_abs"]:
        arr = stats[prefix]
        out[f"{prefix}_mean_hard"] = safe_mean(arr[hard])
        out[f"{prefix}_mean_easy"] = safe_mean(arr[easy])
        out[f"{prefix}_mean_overwide"] = safe_mean(arr[overwide])
        out[f"hard_vs_{prefix}_corr_late_turb"] = corr(hard_float[late_turb_mask], arr[late_turb_mask])
        out[f"{prefix}_share_on_hard_late_turb"] = safe_share(float(arr[hard].sum()), float(arr[late_turb_mask].sum()))

    hard_windows = hard.any(axis=(1, 2))
    clean_turb_windows = turb_mask & (~hard_windows)
    out.update(
        {
            "teacher_active_all_windows": safe_mean(stats["teacher_active"]),
            "prior_active_all_windows": safe_mean(stats["prior_active"]),
            "post_active_all_windows": safe_mean(stats["post_active"]),
            "teacher_active_hard_windows": safe_mean(stats["teacher_active"][hard_windows]),
            "teacher_active_clean_turb_windows": safe_mean(stats["teacher_active"][clean_turb_windows]),
            "prior_active_hard_windows": safe_mean(stats["prior_active"][hard_windows]),
            "prior_active_clean_turb_windows": safe_mean(stats["prior_active"][clean_turb_windows]),
            "post_active_hard_windows": safe_mean(stats["post_active"][hard_windows]),
            "post_active_clean_turb_windows": safe_mean(stats["post_active"][clean_turb_windows]),
            "prior_amp_abs_all_windows": safe_mean(stats["prior_amp_abs"]),
            "post_amp_abs_all_windows": safe_mean(stats["post_amp_abs"]),
            "prior_slot_count_hard_windows": safe_mean(stats["prior_slot_count"][hard_windows]),
            "prior_slot_count_clean_turb_windows": safe_mean(stats["prior_slot_count"][clean_turb_windows]),
            "post_slot_count_hard_windows": safe_mean(stats["post_slot_count"][hard_windows]),
            "post_slot_count_clean_turb_windows": safe_mean(stats["post_slot_count"][clean_turb_windows]),
            "prior_time_top1_hard_windows": safe_mean(stats["prior_time_top1"][hard_windows]),
            "prior_time_top1_clean_turb_windows": safe_mean(stats["prior_time_top1"][clean_turb_windows]),
            "prior_node_top1_hard_windows": safe_mean(stats["prior_node_top1"][hard_windows]),
            "prior_node_top1_clean_turb_windows": safe_mean(stats["prior_node_top1"][clean_turb_windows]),
            "post_time_top1_hard_windows": safe_mean(stats["post_time_top1"][hard_windows]),
            "post_time_top1_clean_turb_windows": safe_mean(stats["post_time_top1"][clean_turb_windows]),
            "post_node_top1_hard_windows": safe_mean(stats["post_node_top1"][hard_windows]),
            "post_node_top1_clean_turb_windows": safe_mean(stats["post_node_top1"][clean_turb_windows]),
            "prior_amp_abs_hard_windows": safe_mean(stats["prior_amp_abs"][hard_windows]),
            "prior_amp_abs_clean_turb_windows": safe_mean(stats["prior_amp_abs"][clean_turb_windows]),
            "post_amp_abs_hard_windows": safe_mean(stats["post_amp_abs"][hard_windows]),
            "post_amp_abs_clean_turb_windows": safe_mean(stats["post_amp_abs"][clean_turb_windows]),
            "prior_radius_hard_windows": safe_mean(stats["prior_radius"][hard_windows]),
            "prior_radius_clean_turb_windows": safe_mean(stats["prior_radius"][clean_turb_windows]),
            "prior_duration_hard_windows": safe_mean(stats["prior_duration"][hard_windows]),
            "prior_duration_clean_turb_windows": safe_mean(stats["prior_duration"][clean_turb_windows]),
            "prior_event_band_abs_all_windows": safe_mean(stats["prior_event_band_abs"]),
            "post_event_band_abs_all_windows": safe_mean(stats["post_event_band_abs"]),
            "prior_event_band_abs_hard_windows": safe_mean(stats["prior_event_band_abs"][hard_windows]),
            "prior_event_band_abs_clean_turb_windows": safe_mean(stats["prior_event_band_abs"][clean_turb_windows]),
            "post_event_band_abs_hard_windows": safe_mean(stats["post_event_band_abs"][hard_windows]),
            "post_event_band_abs_clean_turb_windows": safe_mean(stats["post_event_band_abs"][clean_turb_windows]),
        }
    )
    return out


def build_event_off_model(model: MarkedEventResidualModel) -> MarkedEventResidualModel:
    off = copy.deepcopy(model)
    off.event_scale_logit.data.fill_(-40.0)
    off.eval()
    return off


@torch.no_grad()
def review_checkpoint(
    checkpoint_path: Path,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    n_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    model, payload = build_model(checkpoint_path, device)
    vov, _q20, q80 = regime_masks_from_history(history_norm.cpu())
    turb_mask = vov >= q80

    on_eval = sample_eval_arrays(
        model,
        history_norm,
        future_norm,
        device,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=1234,
    )
    mech_stats = marked_event_state_analysis(model, history_norm, future_norm, device, batch_size=batch_size)
    overlap = summarize_overlap(on_eval["coverage"], turb_mask, mech_stats)

    off_model = build_event_off_model(model)
    off_eval = sample_eval_arrays(
        off_model,
        history_norm,
        future_norm,
        device,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=1234,
    )

    coverage_flat = on_eval["coverage"].reshape(on_eval["coverage"].shape[0], on_eval["coverage"].shape[1], -1)
    off_coverage_flat = off_eval["coverage"].reshape(off_eval["coverage"].shape[0], off_eval["coverage"].shape[1], -1)
    hard_mask = np.zeros_like(coverage_flat, dtype=bool)
    for h_idx in SELECT_HORIZONS:
        if h_idx < coverage_flat.shape[1]:
            hard_mask[:, h_idx] = turb_mask[:, None] & (coverage_flat[:, h_idx] < 0.70)

    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(payload.get("epoch", -1)),
        "subset_windows": int(history_norm.shape[0]),
        "hard_slice_review": overlap,
        "event_off_ablation": {
            "event_on_bad_window_rate": float(on_eval["bad_window_rate"]),
            "event_off_bad_window_rate": float(off_eval["bad_window_rate"]),
            "bad_window_rate_delta_off_minus_on": float(off_eval["bad_window_rate"] - on_eval["bad_window_rate"]),
            "event_on_cov90": float(on_eval["overall_cov90"]),
            "event_off_cov90": float(off_eval["overall_cov90"]),
            "cov90_delta_off_minus_on": float(off_eval["overall_cov90"] - on_eval["overall_cov90"]),
            "event_on_mr_ratio": float(on_eval["mr_ratio"]),
            "event_off_mr_ratio": float(off_eval["mr_ratio"]),
            "mr_ratio_delta_off_minus_on": float(off_eval["mr_ratio"] - on_eval["mr_ratio"]),
            "event_on_jump_ks": float(on_eval["jump_ks"]),
            "event_off_jump_ks": float(off_eval["jump_ks"]),
            "jump_ks_delta_off_minus_on": float(off_eval["jump_ks"] - on_eval["jump_ks"]),
            "event_on_kurtosis_ratio": float(on_eval["kurtosis_ratio"]),
            "event_off_kurtosis_ratio": float(off_eval["kurtosis_ratio"]),
            "kurtosis_ratio_delta_off_minus_on": float(off_eval["kurtosis_ratio"] - on_eval["kurtosis_ratio"]),
            "event_on_spectrum": {
                "quiet": float(on_eval["quiet_ratio"]),
                "shoulder": float(on_eval["shoulder_ratio"]),
                "extreme": float(on_eval["extreme_ratio"]),
            },
            "event_off_spectrum": {
                "quiet": float(off_eval["quiet_ratio"]),
                "shoulder": float(off_eval["shoulder_ratio"]),
                "extreme": float(off_eval["extreme_ratio"]),
            },
            "event_on_hard_late_coverage": safe_mean(coverage_flat[hard_mask]),
            "event_off_hard_late_coverage": safe_mean(off_coverage_flat[hard_mask]),
            "hard_late_coverage_delta_off_minus_on": float(
                safe_mean(off_coverage_flat[hard_mask]) - safe_mean(coverage_flat[hard_mask])
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused mechanistic review of 188a marked-event residual model")
    parser.add_argument(
        "--best_ckpt",
        default=(
            "models/backfill/"
            "graph_group_marked_event_residual_pathwise_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_188a/best_model.pt"
        ),
    )
    parser.add_argument(
        "--final_ckpt",
        default=(
            "models/backfill/"
            "graph_group_marked_event_residual_pathwise_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_188a/final_model.pt"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--max_windows", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        default="results/validations/2026-04-06/analysis/188a_marked_event_mechanistic",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    history_norm, future_norm = build_test_subset(
        data_path="data/vol_surface_with_ret.npz",
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )

    best_review = review_checkpoint(
        checkpoint_path=Path(args.best_ckpt),
        history_norm=history_norm,
        future_norm=future_norm,
        device=device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )
    final_review = review_checkpoint(
        checkpoint_path=Path(args.final_ckpt),
        history_norm=history_norm,
        future_norm=future_norm,
        device=device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    out = {
        "subset_config": {
            "max_windows": args.max_windows,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "test_start": args.test_start,
        },
        "best_checkpoint_review": best_review,
        "final_checkpoint_review": final_review,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(out), indent=2))
    print(json.dumps(make_serializable(out), indent=2))


if __name__ == "__main__":
    main()
