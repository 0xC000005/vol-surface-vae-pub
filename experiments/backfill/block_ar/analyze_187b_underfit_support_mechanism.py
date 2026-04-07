#!/usr/bin/env python
"""
Focused mechanistic review of 187b_v0.

Questions:
  1. Is learned support now aligned with the exact hard S3/S7 late-turbulent miss
     slices, or is it still pointed elsewhere?
  2. Is support now aligned but too diffuse, or is amplitude still too weak?
  3. Does turning the event branch off help or hurt broad behavior?
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from experiments.backfill.block_ar.train_187b_underfit_support_residual import SparseSupportResidualModel


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


def binary_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.mean(ent))


def load_model(ckpt_path: Path, device: torch.device) -> tuple[SparseSupportResidualModel, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = SparseSupportResidualModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        support_config=cfg["support_model"],
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


@torch.no_grad()
def sample_eval_arrays(
    model: SparseSupportResidualModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    n_samples: int,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
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
    gen_path_max = np.abs(gen_path[:, :, 1:] - gen_path[:, :, :-1]).max(axis=(1, 2, 3))
    gen_path_max = gen_path_max[:, 0]
    jump_ks = ks_statistic(gt_path_max, gen_path_max)
    kurt_ratio = float(
        pearson_kurtosis(torch.from_numpy(gen_diff[:, 0]))
        / max(pearson_kurtosis(torch.from_numpy(gt_diff)), 1e-12)
    )

    window_cov = coverage.mean(axis=(1, 2, 3))
    bad_window_rate = float((window_cov < 0.50).mean())

    return {
        "history": history,
        "ground_truth": ground_truth,
        "coverage": coverage,
        "width": width,
        "window_coverage": window_cov,
        "bad_window_rate": bad_window_rate,
        "mr_ratio": mr_ratio,
        "jump_ks": float(jump_ks),
        "kurtosis_ratio": kurt_ratio,
        "quiet_ratio": quiet_ratio,
        "shoulder_ratio": shoulder_ratio,
        "extreme_ratio": extreme_ratio,
    }


@torch.no_grad()
def target_state_support_analysis(
    model: SparseSupportResidualModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)

    prior_support = []
    post_support = []
    teacher_support = []
    target_event_abs = []
    event_local_prior_abs = []
    event_local_post_abs = []
    event_amp_prior_abs = []
    event_amp_post_abs = []
    quiet_local_abs = []
    target_local_abs = []

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
        ) = model.build_quiet_controls(z_t, t, path_context)
        (
            teacher_sup,
            target_event_local,
            _target_event_basis,
            _target_event_band,
        ) = model.build_teacher_event_components(
            target_basis=target_basis,
            quiet_metric_local=quiet_metric_local,
            quiet_metric_band=quiet_metric_band,
            target_local_log=target_local_log,
            target_band_log=target_band_log,
        )
        teacher_abs = target_event_local.abs()
        state_local, _state_band = model.build_state_features(z_t)
        support_pack = model.infer_support_process(
            path_context=path_context,
            state_local=state_local,
            quiet_metric_local=quiet_metric_local,
            t=t,
            teacher_abs=teacher_abs,
        )
        prior_controls = model.build_sparse_support_controls(
            z_t, t, path_context, support_pack=support_pack, use_posterior=False
        )
        post_controls = model.build_sparse_support_controls(
            z_t, t, path_context, support_pack=support_pack, use_posterior=True
        )

        prior_support.append(prior_controls[9].detach().cpu().numpy())
        post_support.append(post_controls[9].detach().cpu().numpy())
        teacher_support.append(teacher_sup.detach().cpu().numpy())
        target_event_abs.append(teacher_abs.detach().cpu().numpy())
        event_local_prior_abs.append(prior_controls[4].abs().detach().cpu().numpy())
        event_local_post_abs.append(post_controls[4].abs().detach().cpu().numpy())
        event_amp_prior_abs.append(prior_controls[11].abs().detach().cpu().numpy())
        event_amp_post_abs.append(post_controls[11].abs().detach().cpu().numpy())
        quiet_local_abs.append(quiet_metric_local.abs().detach().cpu().numpy())
        target_local_abs.append(target_local_log.abs().detach().cpu().numpy())

    return {
        "prior_support": np.concatenate(prior_support, axis=0),
        "post_support": np.concatenate(post_support, axis=0),
        "teacher_support": np.concatenate(teacher_support, axis=0),
        "target_event_abs": np.concatenate(target_event_abs, axis=0),
        "event_local_prior_abs": np.concatenate(event_local_prior_abs, axis=0),
        "event_local_post_abs": np.concatenate(event_local_post_abs, axis=0),
        "event_amp_prior_abs": np.concatenate(event_amp_prior_abs, axis=0),
        "event_amp_post_abs": np.concatenate(event_amp_post_abs, axis=0),
        "quiet_local_abs": np.concatenate(quiet_local_abs, axis=0),
        "target_local_abs": np.concatenate(target_local_abs, axis=0),
    }


def summarize_support_overlap(
    coverage: np.ndarray,
    turb_mask: np.ndarray,
    support_stats: dict[str, np.ndarray],
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

    prior_support = support_stats["prior_support"]
    post_support = support_stats["post_support"]
    teacher_support = support_stats["teacher_support"]
    target_event_abs = support_stats["target_event_abs"]
    event_prior_abs = support_stats["event_local_prior_abs"]
    event_post_abs = support_stats["event_local_post_abs"]
    event_amp_prior_abs = support_stats["event_amp_prior_abs"]
    event_amp_post_abs = support_stats["event_amp_post_abs"]
    quiet_local_abs = support_stats["quiet_local_abs"]
    target_local_abs = support_stats["target_local_abs"]

    late_turb_mask = np.zeros_like(coverage_flat, dtype=bool)
    for h_idx in SELECT_HORIZONS:
        if h_idx < t_len:
            late_turb_mask[:, h_idx] = turb_mask[:, None]

    hard_float = hard.astype(np.float32)
    summary = {
        "hard_point_count": int(hard.sum()),
        "easy_point_count": int(easy.sum()),
        "overwide_point_count": int(overwide.sum()),
        "late_turb_point_count": int(late_turb_mask.sum()),
        "prior_support_mean_all_late_turb": safe_mean(prior_support[late_turb_mask]),
        "post_support_mean_all_late_turb": safe_mean(post_support[late_turb_mask]),
        "teacher_support_mean_all_late_turb": safe_mean(teacher_support[late_turb_mask]),
        "prior_support_entropy_all_late_turb": binary_entropy(prior_support[late_turb_mask]),
        "post_support_entropy_all_late_turb": binary_entropy(post_support[late_turb_mask]),
        "prior_support_mean_hard": safe_mean(prior_support[hard]),
        "prior_support_mean_easy": safe_mean(prior_support[easy]),
        "prior_support_mean_overwide": safe_mean(prior_support[overwide]),
        "post_support_mean_hard": safe_mean(post_support[hard]),
        "post_support_mean_easy": safe_mean(post_support[easy]),
        "post_support_mean_overwide": safe_mean(post_support[overwide]),
        "teacher_support_mean_hard": safe_mean(teacher_support[hard]),
        "teacher_support_mean_easy": safe_mean(teacher_support[easy]),
        "teacher_support_mean_overwide": safe_mean(teacher_support[overwide]),
        "target_event_abs_mean_hard": safe_mean(target_event_abs[hard]),
        "target_event_abs_mean_easy": safe_mean(target_event_abs[easy]),
        "event_local_prior_abs_mean_hard": safe_mean(event_prior_abs[hard]),
        "event_local_prior_abs_mean_easy": safe_mean(event_prior_abs[easy]),
        "event_local_post_abs_mean_hard": safe_mean(event_post_abs[hard]),
        "event_local_post_abs_mean_easy": safe_mean(event_post_abs[easy]),
        "event_amp_prior_abs_mean_hard": safe_mean(event_amp_prior_abs[hard]),
        "event_amp_prior_abs_mean_easy": safe_mean(event_amp_prior_abs[easy]),
        "event_amp_post_abs_mean_hard": safe_mean(event_amp_post_abs[hard]),
        "event_amp_post_abs_mean_easy": safe_mean(event_amp_post_abs[easy]),
        "quiet_local_abs_mean_hard": safe_mean(quiet_local_abs[hard]),
        "quiet_local_abs_mean_easy": safe_mean(quiet_local_abs[easy]),
        "target_local_abs_mean_hard": safe_mean(target_local_abs[hard]),
        "target_local_abs_mean_easy": safe_mean(target_local_abs[easy]),
        "hard_vs_prior_support_corr_late_turb": corr(hard_float[late_turb_mask], prior_support[late_turb_mask]),
        "hard_vs_post_support_corr_late_turb": corr(hard_float[late_turb_mask], post_support[late_turb_mask]),
        "hard_vs_teacher_support_corr_late_turb": corr(hard_float[late_turb_mask], teacher_support[late_turb_mask]),
        "hard_vs_target_event_abs_corr_late_turb": corr(hard_float[late_turb_mask], target_event_abs[late_turb_mask]),
        "hard_vs_event_local_prior_corr_late_turb": corr(hard_float[late_turb_mask], event_prior_abs[late_turb_mask]),
        "hard_vs_event_local_post_corr_late_turb": corr(hard_float[late_turb_mask], event_post_abs[late_turb_mask]),
        "hard_vs_event_amp_prior_corr_late_turb": corr(hard_float[late_turb_mask], event_amp_prior_abs[late_turb_mask]),
        "hard_vs_event_amp_post_corr_late_turb": corr(hard_float[late_turb_mask], event_amp_post_abs[late_turb_mask]),
    }
    return summary


def build_event_off_model(model: SparseSupportResidualModel) -> SparseSupportResidualModel:
    off = copy.deepcopy(model)
    off.event_scale_logit.data.fill_(-40.0)
    off.eval()
    return off


def review_checkpoint(
    checkpoint_path: Path,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    n_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    model, payload = load_model(checkpoint_path, device)
    history_norm = history_norm.clone()
    future_norm = future_norm.clone()
    vov, _q20, q80 = regime_masks_from_history(history_norm.cpu())
    turb_mask = vov >= q80

    on_eval = sample_eval_arrays(model, history_norm, future_norm, device, n_samples=n_samples, batch_size=batch_size)
    support_stats = target_state_support_analysis(model, history_norm, future_norm, device, batch_size=batch_size)
    overlap = summarize_support_overlap(on_eval["coverage"], turb_mask, support_stats)

    off_model = build_event_off_model(model)
    off_eval = sample_eval_arrays(off_model, history_norm, future_norm, device, n_samples=n_samples, batch_size=batch_size)

    coverage_flat = on_eval["coverage"].reshape(on_eval["coverage"].shape[0], on_eval["coverage"].shape[1], -1)
    late_h = [h for h in SELECT_HORIZONS if h < coverage_flat.shape[1]]
    hard_mask = np.zeros_like(coverage_flat, dtype=bool)
    for h_idx in late_h:
        hard_mask[:, h_idx] = turb_mask[:, None] & (coverage_flat[:, h_idx] < 0.70)
    off_coverage_flat = off_eval["coverage"].reshape(off_eval["coverage"].shape[0], off_eval["coverage"].shape[1], -1)

    review = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(payload.get("epoch", -1)),
        "subset_windows": int(history_norm.shape[0]),
        "hard_slice_review": overlap,
        "event_off_ablation": {
            "event_on_bad_window_rate": float(on_eval["bad_window_rate"]),
            "event_off_bad_window_rate": float(off_eval["bad_window_rate"]),
            "bad_window_rate_delta_off_minus_on": float(off_eval["bad_window_rate"] - on_eval["bad_window_rate"]),
            "event_on_mr_ratio": float(on_eval["mr_ratio"]),
            "event_off_mr_ratio": float(off_eval["mr_ratio"]),
            "mr_ratio_delta_off_minus_on": float(off_eval["mr_ratio"] - on_eval["mr_ratio"]),
            "event_on_jump_ks": float(on_eval["jump_ks"]),
            "event_off_jump_ks": float(off_eval["jump_ks"]),
            "jump_ks_delta_off_minus_on": float(off_eval["jump_ks"] - on_eval["jump_ks"]),
            "event_on_kurtosis_ratio": float(on_eval["kurtosis_ratio"]),
            "event_off_kurtosis_ratio": float(off_eval["kurtosis_ratio"]),
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
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused mechanistic review of 187b underfit-support residual model")
    parser.add_argument(
        "--best_ckpt",
        default="models/backfill/underfit_support_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_187b/best_model.pt",
    )
    parser.add_argument(
        "--final_ckpt",
        default="models/backfill/underfit_support_residual_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_187b/final_model.pt",
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
        default="results/validations/2026-04-06/analysis/187b_underfit_support_mechanistic",
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

    summary = {
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
    (output_dir / "mechanistic_summary.json").write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
