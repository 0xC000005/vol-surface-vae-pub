"""
Focused mechanistic review of 185a_v0.

This script answers three narrow questions before any 185b build:
1. Does the latent event path overlap the exact hard late-horizon turbulent slices?
2. Does the prior know the same slices as the posterior, or is the event path only
   useful under teacher forcing?
3. Does the event path help or hurt broad calibration / mean-reversion when we
   ablate it at sampling time?
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    iv_to_unconstrained,
    make_serializable,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_177a_mean_reverting_local_template_mixture import aggregate_slope_ratio
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import (
    ks_statistic,
    pearson_kurtosis,
)
from experiments.backfill.block_ar.train_182b_width_tail_control import evaluate_frontier_subset
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import compute_control_targets
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel
from experiments.backfill.block_ar.train_185a_graph_group_latent_event_path import (
    GraphGroupLatentEventPathModel,
)


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text())


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


def build_model(ckpt_path: Path, device: torch.device) -> tuple[GraphGroupLatentEventPathModel, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw = payload["config"]
    model = GraphGroupLatentEventPathModel(
        encoder_config=EncoderConfig(**raw["encoder"]),
        decoder_config=raw["decoder"],
        flow_config=raw["flow"],
        path_config=raw["path"],
        prior_config=raw["prior"],
        integrated_config=raw["integrated"],
        state_config=raw["state"],
        metric_config=raw["metric"],
        activity_config=raw["activity"],
        event_config=raw["event"],
        support_lo=raw.get("support_lo", 0.01),
        support_hi=raw.get("support_hi", 1.0),
        support_eps=raw.get("support_eps", 1e-5),
        base_nu=raw.get("base_nu", 8.0),
        mix_chunk_size=raw.get("mix_chunk_size", 27),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, payload


def build_val_tensors(device: torch.device, history_len: int, future_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    test_start = 4511
    max_train_idx = test_start - history_len - future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    return val_hist, val_future


def make_subset_loader(history: torch.Tensor, future: torch.Tensor, batch_size: int) -> DataLoader:
    ds = TensorDataset(history.detach().cpu(), future.detach().cpu())
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def compute_window_stats(
    model: GraphGroupLatentEventPathModel,
    history: torch.Tensor,
    future: torch.Tensor,
    n_samples: int,
    batch_size: int,
) -> dict[str, float]:
    sample_batches = []
    gt_batches = []
    hist_batches = []
    for start in range(0, history.shape[0], batch_size):
        h = history[start : start + batch_size].to(next(model.parameters()).device)
        f = future[start : start + batch_size].to(next(model.parameters()).device)
        samples_u = model.sample_future_u(h, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(
            samples_u,
            lo=model.support_lo,
            hi=model.support_hi,
        ).view(h.shape[0], n_samples, model.decoder.n_frames, 5, 5)
        future_grid = f.view(h.shape[0], model.decoder.n_frames, 5, 5)
        sample_batches.append(samples_01.detach().cpu())
        gt_batches.append(future_grid.detach().cpu())
        hist_batches.append(h.detach().cpu())

    cond_samples = torch.cat(sample_batches, dim=0).numpy()
    ground_truth = torch.cat(gt_batches, dim=0).numpy()
    history_np = torch.cat(hist_batches, dim=0).numpy()

    q05 = np.percentile(cond_samples, 5, axis=1)
    q95 = np.percentile(cond_samples, 95, axis=1)
    covered = (ground_truth >= q05) & (ground_truth <= q95)
    per_window_cov = covered.mean(axis=(1, 2, 3))
    mean_iv = history_np.mean(axis=(2, 3))
    vov = np.diff(mean_iv, axis=1).std(axis=1)
    q20 = np.quantile(vov, 0.2)
    q80 = np.quantile(vov, 0.8)
    window_width = (q95 - q05).mean(axis=(1, 2, 3))
    calm = vov <= q20
    turb = vov >= q80
    turb_calm = float(window_width[turb].mean() / max(window_width[calm].mean(), 1e-12))
    return {
        "overall_cov90": float(covered.mean()),
        "pct_bad_windows_under_50": float((per_window_cov < 0.50).mean()),
        "worst_window_cov": float(per_window_cov.min()),
        "p10_window_cov": float(np.quantile(per_window_cov, 0.10)),
        "turb_calm_width_ratio": turb_calm,
    }


@torch.no_grad()
def compute_joint_like_stats(
    model: GraphGroupLatentEventPathModel,
    history: torch.Tensor,
    future: torch.Tensor,
    n_samples: int,
    batch_size: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
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

    for start in range(0, history.shape[0], batch_size):
        h = history[start : start + batch_size].to(device)
        f = future[start : start + batch_size].to(device)
        future_grid = f.view(h.shape[0], model.decoder.n_frames, 5, 5)
        samples_u = model.sample_future_u(h, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).view(
            h.shape[0], n_samples, model.decoder.n_frames, 5, 5
        )
        lo = samples_01.quantile(0.05, dim=1)
        hi = samples_01.quantile(0.95, dim=1)
        median = samples_01.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float().mean()
        width = (hi - lo).mean()
        mae = (median - future_grid).abs().mean()

        mean_iv = h.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = (hi - lo).mean(dim=(1, 2, 3))

        prev_chunks.append(h[:, -1].detach().cpu())
        gt_next_chunks.append(future_grid[:, 0].detach().cpu())
        sample_next_chunks.append(samples_01[:, :, 0].mean(dim=1).detach().cpu())

        gt_change = (future_grid[:, 1:] - future_grid[:, :-1]).detach().cpu()
        gen_first = samples_01[:, 0]
        gen_change = (gen_first[:, 1:] - gen_first[:, :-1]).detach().cpu()
        gt_changes.append(gt_change)
        gen_changes.append(gen_change)

        gt_path = torch.cat([h[:, -1:].detach().cpu(), future_grid.detach().cpu()], dim=1)
        gen_path = torch.cat([h[:, -1:].detach().cpu(), gen_first.detach().cpu()], dim=1)
        gt_path_max.append((gt_path[:, 1:] - gt_path[:, :-1]).abs().amax(dim=(1, 2, 3)))
        gen_path_max.append((gen_path[:, 1:] - gen_path[:, :-1]).abs().amax(dim=(1, 2, 3)))

        total_cov += coverage.item() * h.shape[0]
        total_width += width.item() * h.shape[0]
        total_mae += mae.item() * h.shape[0]
        total_count += h.shape[0]
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

    gt_changes = torch.cat(gt_changes, dim=0)
    gen_changes = torch.cat(gen_changes, dim=0)
    gt_path_max = torch.cat(gt_path_max, dim=0).numpy()
    gen_path_max = torch.cat(gen_path_max, dim=0).numpy()
    kurt_ratio = pearson_kurtosis(gen_changes) / max(pearson_kurtosis(gt_changes), 1e-12)
    jump_ks = ks_statistic(gt_path_max, gen_path_max)

    return {
        "joint_cov90": total_cov / max(total_count, 1),
        "joint_width90": total_width / max(total_count, 1),
        "joint_mae": total_mae / max(total_count, 1),
        "joint_turb_calm_ratio": turb_calm_ratio,
        "joint_sample_mr_ratio": float(sample_mr_ratio),
        "joint_kurtosis_ratio": float(kurt_ratio),
        "joint_pathwise_jump_ks": float(jump_ks),
    }


@torch.no_grad()
def review_checkpoint(
    model: GraphGroupLatentEventPathModel,
    history: torch.Tensor,
    future: torch.Tensor,
    n_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    all_target_event = []
    all_prior_event = []
    all_post_event = []
    all_prior_support = []
    all_post_support = []
    all_prior_gate = []
    all_post_gate = []
    all_prior_budget = []
    all_post_budget = []
    all_event_amp = []
    all_inside90 = []
    all_turb = []
    all_event_peak_hit = []
    all_support_peak_hit = []
    all_post_event_peak_hit = []
    all_target_peak_hit = []
    all_event_top3_hit = []
    all_support_top3_hit = []
    all_post_event_top3_hit = []
    all_target_top3_hit = []

    for start in range(0, history.shape[0], batch_size):
        h = history[start : start + batch_size].to(device)
        f = future[start : start + batch_size].to(device)
        bsz = h.shape[0]
        future_grid = f.view(bsz, model.decoder.n_frames, 5, 5)

        samples_u = model.sample_future_u(h, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).view(
            bsz, n_samples, model.decoder.n_frames, 5, 5
        )
        lo90 = torch.quantile(samples_01, 0.05, dim=1)
        hi90 = torch.quantile(samples_01, 0.95, dim=1)
        inside90 = ((future_grid >= lo90) & (future_grid <= hi90)).detach().cpu().numpy()

        mean_iv = h.mean(dim=(2, 3))
        vov = torch.diff(mean_iv, dim=1).std(dim=1)

        target_u = iv_to_unconstrained(f, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
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
        ) = model.forward_from_history(h)
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
        ).view(bsz, model.decoder.n_frames, model.decoder.n_cells)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, _target_band_log = compute_control_targets(model, target_basis)

        z_t = target_basis.reshape(bsz, -1)
        t = torch.ones(bsz, device=device, dtype=z_t.dtype)
        state_local, state_band = model.build_state_features(z_t)
        (
            _raw_local,
            _raw_band,
            quiet_metric_local,
            _quiet_metric_band,
            _local_gate,
            _band_gate,
        ) = StateMetricTransportModel.build_state_metric_controls(model, z_t, t, path_context)
        target_event_local = (target_local_log - quiet_metric_local).clamp(
            min=-model.event_config["event_local_clip"],
            max=model.event_config["event_local_clip"],
        )
        activity_pack = model.infer_activity_process(
            path_context=path_context,
            state_local=state_local,
            state_band=state_band,
            quiet_metric_local=quiet_metric_local,
            target_local_log=target_event_local,
        )

        prior_pack = model.build_event_path_controls(
            z_t, t, path_context, activity_pack=activity_pack, use_posterior=False
        )
        post_pack = model.build_event_path_controls(
            z_t, t, path_context, activity_pack=activity_pack, use_posterior=True
        )

        prior_event_local = prior_pack[4]
        post_event_local = post_pack[4]
        prior_activity_gate = prior_pack[11]
        post_activity_gate = post_pack[11]
        prior_activity_budget = prior_pack[12]
        post_activity_budget = post_pack[12]
        prior_event_amp = prior_pack[17]
        prior_support = prior_pack[18]
        post_support = post_pack[18]

        all_target_event.append(target_event_local.detach().cpu().numpy())
        all_prior_event.append(prior_event_local.detach().cpu().numpy())
        all_post_event.append(post_event_local.detach().cpu().numpy())
        all_prior_support.append(prior_support.detach().cpu().numpy())
        all_post_support.append(post_support.detach().cpu().numpy())
        all_prior_gate.append(prior_activity_gate.detach().cpu().numpy())
        all_post_gate.append(post_activity_gate.detach().cpu().numpy())
        all_prior_budget.append(prior_activity_budget.detach().cpu().numpy())
        all_post_budget.append(post_activity_budget.detach().cpu().numpy())
        all_event_amp.append(prior_event_amp.detach().cpu().numpy())
        all_inside90.append(inside90)
        all_turb.append(vov.detach().cpu().numpy())

        inside_late = inside90[:, [13, 29]].reshape(bsz, 2, 25)
        hard_any = ~inside_late.all(axis=2)
        target_abs = target_event_local[:, [13, 29]].abs().detach().cpu().numpy().reshape(bsz, 2, 25)
        prior_event_abs = prior_event_local[:, [13, 29]].abs().detach().cpu().numpy().reshape(bsz, 2, 25)
        post_event_abs = post_event_local[:, [13, 29]].abs().detach().cpu().numpy().reshape(bsz, 2, 25)
        prior_support_abs = prior_support[:, [13, 29]].abs().detach().cpu().numpy().reshape(bsz, 2, 25)
        for i in range(bsz):
            for j in range(2):
                if not hard_any[i, j]:
                    continue
                hard_set = (~inside_late[i, j]).reshape(-1)
                k = min(3, hard_set.size)
                target_arg = int(np.argmax(target_abs[i, j]))
                prior_event_arg = int(np.argmax(prior_event_abs[i, j]))
                post_event_arg = int(np.argmax(post_event_abs[i, j]))
                prior_support_arg = int(np.argmax(prior_support_abs[i, j]))
                all_target_peak_hit.append(bool(hard_set[target_arg]))
                all_event_peak_hit.append(bool(hard_set[prior_event_arg]))
                all_post_event_peak_hit.append(bool(hard_set[post_event_arg]))
                all_support_peak_hit.append(bool(hard_set[prior_support_arg]))
                target_top = set(np.argpartition(target_abs[i, j], -k)[-k:])
                prior_event_top = set(np.argpartition(prior_event_abs[i, j], -k)[-k:])
                post_event_top = set(np.argpartition(post_event_abs[i, j], -k)[-k:])
                prior_support_top = set(np.argpartition(prior_support_abs[i, j], -k)[-k:])
                hard_idx = set(np.flatnonzero(hard_set))
                all_target_top3_hit.append(len(target_top & hard_idx) / max(len(target_top), 1))
                all_event_top3_hit.append(len(prior_event_top & hard_idx) / max(len(prior_event_top), 1))
                all_post_event_top3_hit.append(len(post_event_top & hard_idx) / max(len(post_event_top), 1))
                all_support_top3_hit.append(len(prior_support_top & hard_idx) / max(len(prior_support_top), 1))

    target_event = np.concatenate(all_target_event, axis=0)
    prior_event = np.concatenate(all_prior_event, axis=0)
    post_event = np.concatenate(all_post_event, axis=0)
    prior_support = np.concatenate(all_prior_support, axis=0)
    post_support = np.concatenate(all_post_support, axis=0)
    prior_gate = np.concatenate(all_prior_gate, axis=0)
    post_gate = np.concatenate(all_post_gate, axis=0)
    prior_budget = np.concatenate(all_prior_budget, axis=0)
    post_budget = np.concatenate(all_post_budget, axis=0)
    event_amp = np.concatenate(all_event_amp, axis=0)
    inside90 = np.concatenate(all_inside90, axis=0)
    vov = np.concatenate(all_turb, axis=0)

    q80 = float(np.quantile(vov, 0.8))
    turb_win = vov >= q80
    late_mask = np.zeros((target_event.shape[0], target_event.shape[1], target_event.shape[2]), dtype=bool)
    late_mask[:, 13, :] = True
    late_mask[:, 29, :] = True
    turb_late = late_mask & turb_win[:, None, None]
    inside90_flat = inside90.reshape(inside90.shape[0], inside90.shape[1], -1)
    hard_slices = turb_late & (~inside90_flat)
    clean_slices = turb_late & inside90_flat

    target_abs = np.abs(target_event)
    prior_event_abs = np.abs(prior_event)
    post_event_abs = np.abs(post_event)
    prior_support_abs = np.abs(prior_support)
    post_support_abs = np.abs(post_support)

    hard_windows = hard_slices.any(axis=(1, 2))
    clean_turb_windows = turb_win & (~hard_windows)

    overlap = {
        "n_windows": int(target_event.shape[0]),
        "n_turb_windows": int(turb_win.sum()),
        "n_hard_turb_windows": int(hard_windows.sum()),
        "n_clean_turb_windows": int(clean_turb_windows.sum()),
        "slice_mass_share_on_hard": {
            "target_event": safe_share(float(target_abs[hard_slices].sum()), float(target_abs[turb_late].sum())),
            "prior_event": safe_share(float(prior_event_abs[hard_slices].sum()), float(prior_event_abs[turb_late].sum())),
            "post_event": safe_share(float(post_event_abs[hard_slices].sum()), float(post_event_abs[turb_late].sum())),
            "prior_support": safe_share(float(prior_support_abs[hard_slices].sum()), float(prior_support_abs[turb_late].sum())),
            "post_support": safe_share(float(post_support_abs[hard_slices].sum()), float(post_support_abs[turb_late].sum())),
        },
        "mean_abs_on_slices": {
            "hard_target_event": safe_mean(target_abs[hard_slices]),
            "hard_prior_event": safe_mean(prior_event_abs[hard_slices]),
            "hard_post_event": safe_mean(post_event_abs[hard_slices]),
            "hard_prior_support": safe_mean(prior_support_abs[hard_slices]),
            "hard_post_support": safe_mean(post_support_abs[hard_slices]),
            "hard_event_amp": safe_mean(np.abs(event_amp[hard_slices])),
            "clean_target_event": safe_mean(target_abs[clean_slices]),
            "clean_prior_event": safe_mean(prior_event_abs[clean_slices]),
            "clean_post_event": safe_mean(post_event_abs[clean_slices]),
            "clean_prior_support": safe_mean(prior_support_abs[clean_slices]),
            "clean_post_support": safe_mean(post_support_abs[clean_slices]),
            "clean_event_amp": safe_mean(np.abs(event_amp[clean_slices])),
        },
        "window_level_activity": {
            "prior_gate_hard_turb": safe_mean(prior_gate[hard_windows]),
            "post_gate_hard_turb": safe_mean(post_gate[hard_windows]),
            "prior_budget_hard_turb": safe_mean(prior_budget[hard_windows]),
            "post_budget_hard_turb": safe_mean(post_budget[hard_windows]),
            "prior_gate_clean_turb": safe_mean(prior_gate[clean_turb_windows]),
            "post_gate_clean_turb": safe_mean(post_gate[clean_turb_windows]),
            "prior_budget_clean_turb": safe_mean(prior_budget[clean_turb_windows]),
            "post_budget_clean_turb": safe_mean(post_budget[clean_turb_windows]),
        },
        "alignment": {
            "corr_target_vs_prior_event_turb_late": corr(target_abs[turb_late], prior_event_abs[turb_late]),
            "corr_target_vs_post_event_turb_late": corr(target_abs[turb_late], post_event_abs[turb_late]),
            "corr_target_vs_prior_support_turb_late": corr(target_abs[turb_late], prior_support_abs[turb_late]),
            "corr_miss_vs_prior_event_turb_late": corr(hard_slices[turb_late].astype(np.float64), prior_event_abs[turb_late]),
            "corr_miss_vs_post_event_turb_late": corr(hard_slices[turb_late].astype(np.float64), post_event_abs[turb_late]),
            "corr_miss_vs_prior_support_turb_late": corr(hard_slices[turb_late].astype(np.float64), prior_support_abs[turb_late]),
        },
        "peak_hit_rate_on_hard_window_h": {
            "target_peak_hits_hard": safe_mean(np.asarray(all_target_peak_hit, dtype=np.float64)),
            "prior_event_peak_hits_hard": safe_mean(np.asarray(all_event_peak_hit, dtype=np.float64)),
            "post_event_peak_hits_hard": safe_mean(np.asarray(all_post_event_peak_hit, dtype=np.float64)),
            "prior_support_peak_hits_hard": safe_mean(np.asarray(all_support_peak_hit, dtype=np.float64)),
            "target_top3_hard_share": safe_mean(np.asarray(all_target_top3_hit, dtype=np.float64)),
            "prior_event_top3_hard_share": safe_mean(np.asarray(all_event_top3_hit, dtype=np.float64)),
            "post_event_top3_hard_share": safe_mean(np.asarray(all_post_event_top3_hit, dtype=np.float64)),
            "prior_support_top3_hard_share": safe_mean(np.asarray(all_support_top3_hit, dtype=np.float64)),
        },
    }
    return overlap


@torch.no_grad()
def event_off_ablation(
    model: GraphGroupLatentEventPathModel,
    history: torch.Tensor,
    future: torch.Tensor,
    joint_val_samples: int,
    eval_limit: int,
    batch_size: int,
) -> dict[str, Any]:
    loader = make_subset_loader(history[:eval_limit], future[:eval_limit], batch_size=batch_size)
    full_frontier = evaluate_frontier_subset(model, loader, joint_val_samples=joint_val_samples, eval_limit=eval_limit)
    full_joint = compute_joint_like_stats(
        model, history[:eval_limit], future[:eval_limit], n_samples=joint_val_samples, batch_size=batch_size
    )
    full_window = compute_window_stats(model, history[:eval_limit], future[:eval_limit], n_samples=joint_val_samples, batch_size=batch_size)

    saved = model.event_scale_logit.detach().clone()
    model.event_scale_logit.data.fill_(-20.0)
    off_frontier = evaluate_frontier_subset(model, loader, joint_val_samples=joint_val_samples, eval_limit=eval_limit)
    off_joint = compute_joint_like_stats(
        model, history[:eval_limit], future[:eval_limit], n_samples=joint_val_samples, batch_size=batch_size
    )
    off_window = compute_window_stats(model, history[:eval_limit], future[:eval_limit], n_samples=joint_val_samples, batch_size=batch_size)
    model.event_scale_logit.data.copy_(saved)

    return {
        "full": {
            "frontier": full_frontier,
            "joint": full_joint,
            "window": full_window,
        },
        "event_off": {
            "frontier": off_frontier,
            "joint": off_joint,
            "window": off_window,
        },
        "delta_event_off_minus_full": {
            "frontier_turb_late_worst_cov": off_frontier["frontier_turb_late_worst_cov"] - full_frontier["frontier_turb_late_worst_cov"],
            "frontier_pooled_kurt_ratio": off_frontier["frontier_pooled_kurt_ratio"] - full_frontier["frontier_pooled_kurt_ratio"],
            "joint_sample_mr_ratio": off_joint["joint_sample_mr_ratio"] - full_joint["joint_sample_mr_ratio"],
            "joint_pathwise_jump_ks": off_joint["joint_pathwise_jump_ks"] - full_joint["joint_pathwise_jump_ks"],
            "joint_kurtosis_ratio": off_joint["joint_kurtosis_ratio"] - full_joint["joint_kurtosis_ratio"],
            "pct_bad_windows_under_50": off_window["pct_bad_windows_under_50"] - full_window["pct_bad_windows_under_50"],
            "overall_cov90": off_window["overall_cov90"] - full_window["overall_cov90"],
            "turb_calm_width_ratio": off_window["turb_calm_width_ratio"] - full_window["turb_calm_width_ratio"],
        },
    }


def find_epoch_record(history: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    for rec in history:
        if rec.get("epoch") == epoch:
            return rec
    raise KeyError(f"epoch {epoch} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused mechanistic review of 185a event-path behavior")
    parser.add_argument(
        "--best_ckpt",
        default=(
            "models/backfill/"
            "graph_group_latent_event_path_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_185a/best_model.pt"
        ),
    )
    parser.add_argument(
        "--final_ckpt",
        default=(
            "models/backfill/"
            "graph_group_latent_event_path_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_185a/final_model.pt"
        ),
    )
    parser.add_argument(
        "--training_history",
        default=(
            "models/backfill/"
            "graph_group_latent_event_path_residual_law_mean_reverting_covariance_mixture_"
            "structured_joint_student_t_185a/training_history.json"
        ),
    )
    parser.add_argument(
        "--anchor_summary",
        default="results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json",
    )
    parser.add_argument(
        "--best_summary",
        default="results/block_ar/185a_best_v2_s3mrjspec_full_30d/summary.json",
    )
    parser.add_argument(
        "--final_summary",
        default="results/block_ar/185a_final_v2_s3mrjspec_full_30d/summary.json",
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        default="results/validations/2026-04-06/analysis/185a_event_mechanistic",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_model, best_payload = build_model(Path(args.best_ckpt), device)
    final_model, final_payload = build_model(Path(args.final_ckpt), device)
    history = load_json(Path(args.training_history))
    assert isinstance(history, list)

    history_len = int(best_payload["config"]["history_len"])
    future_len = int(best_payload["config"]["future_len"])
    val_hist, val_future = build_val_tensors(device, history_len=history_len, future_len=future_len)
    val_hist = val_hist[: args.eval_limit]
    val_future = val_future[: args.eval_limit]

    best_epoch = int(best_payload.get("epoch", 0))
    final_epoch = int(final_payload.get("epoch", 0))
    best_row = find_epoch_record(history, best_epoch)
    final_row = find_epoch_record(history, final_epoch)

    summary_out = {
        "anchor_summary": load_json(Path(args.anchor_summary)),
        "best_summary": load_json(Path(args.best_summary)),
        "final_summary": load_json(Path(args.final_summary)),
        "best_epoch_record": {
            k: best_row.get(k)
            for k in [
                "epoch",
                "stage",
                "val_post_activity_gate_mean",
                "val_prior_activity_gate_mean",
                "val_activity_budget_mean",
                "val_event_scale",
                "val_event_local_abs_mean",
                "val_event_amp_abs_mean",
                "val_event_support_top1_mean",
                "joint_turb_calm_ratio",
                "joint_kurtosis_ratio",
                "joint_sample_mr_ratio",
                "joint_pathwise_jump_ks",
                "frontier_turb_late_worst_cov",
                "frontier_turb_late_best_cov",
            ]
        },
        "final_epoch_record": {
            k: final_row.get(k)
            for k in [
                "epoch",
                "stage",
                "val_post_activity_gate_mean",
                "val_prior_activity_gate_mean",
                "val_activity_budget_mean",
                "val_event_scale",
                "val_event_local_abs_mean",
                "val_event_amp_abs_mean",
                "val_event_support_top1_mean",
                "joint_turb_calm_ratio",
                "joint_kurtosis_ratio",
                "joint_sample_mr_ratio",
                "joint_pathwise_jump_ks",
                "frontier_turb_late_worst_cov",
                "frontier_turb_late_best_cov",
            ]
        },
    }

    best_overlap = review_checkpoint(best_model, val_hist, val_future, n_samples=args.n_samples, batch_size=args.batch_size)
    best_ablation = event_off_ablation(
        best_model,
        val_hist,
        val_future,
        joint_val_samples=args.n_samples,
        eval_limit=min(args.eval_limit, val_hist.shape[0]),
        batch_size=args.batch_size,
    )
    final_overlap = review_checkpoint(final_model, val_hist, val_future, n_samples=args.n_samples, batch_size=args.batch_size)
    final_ablation = event_off_ablation(
        final_model,
        val_hist,
        val_future,
        joint_val_samples=args.n_samples,
        eval_limit=min(args.eval_limit, val_hist.shape[0]),
        batch_size=args.batch_size,
    )

    diagnosis = {
        "best_checkpoint": (
            "185a_best should be read as an event-path model whose gate remains alive, but whose "
            "event amplitude may be too weak or too poorly aligned to materially fix the hard late-horizon "
            "turbulent slices."
        ),
        "key_questions": [
            "Does the posterior concentrate event support on the exact missed S3/S7 slices more than the prior?",
            "Does the prior event path put enough mass on hard slices relative to clean turbulent slices?",
            "Does disabling the event path improve S8/S10, which would indicate interference with the quiet backbone?",
        ],
    }

    out = {
        "review_context": {
            "device": str(device),
            "eval_limit": int(args.eval_limit),
            "n_samples": int(args.n_samples),
            "best_ckpt": args.best_ckpt,
            "final_ckpt": args.final_ckpt,
            "training_history": args.training_history,
        },
        "checkpoint_summary_context": summary_out,
        "best_checkpoint_review": {
            "event_support_overlap": best_overlap,
            "event_off_ablation": best_ablation,
        },
        "final_checkpoint_review": {
            "event_support_overlap": final_overlap,
            "event_off_ablation": final_ablation,
        },
        "diagnosis": diagnosis,
    }
    (out_dir / "mechanistic_summary.json").write_text(json.dumps(make_serializable(out), indent=2))
    print(json.dumps(make_serializable(out), indent=2))


if __name__ == "__main__":
    main()
