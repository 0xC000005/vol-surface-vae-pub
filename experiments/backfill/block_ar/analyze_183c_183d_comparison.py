#!/usr/bin/env python
"""
Comparative mechanism review for 183c_best vs 183d_best.

Focus:
  1. Did 183d put more mass on the true hard slices than 183c?
  2. Did 183d become broadly active instead of selectively concentrated?
  3. Did 183d improve hard-slice coverage enough to justify the loss in S4/S8?
  4. What is the narrowest next principled fix after 183d?
"""

from __future__ import annotations

import argparse
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
from experiments.backfill.block_ar.analyze_183a_final_mechanism import (
    distribution_compare,
    quantile_profile,
)
from experiments.backfill.block_ar.analyze_183c_best_mechanism import (
    BANDS,
    exceedance_spectrum,
    grid_neighbors,
    load_json,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel,
)
from experiments.backfill.block_ar.train_183d_sparse_concentration_transport import (
    SparseConcentrationTransportModel,
)


def load_183c_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = StateMetricTransportModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        path_config=raw_config["path"],
        prior_config=raw_config["prior"],
        integrated_config=raw_config["integrated"],
        state_config=raw_config["state"],
        metric_config=raw_config["metric"],
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


def load_183d_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "sparse_concentration_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183d"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = SparseConcentrationTransportModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        path_config=raw_config["path"],
        prior_config=raw_config["prior"],
        integrated_config=raw_config["integrated"],
        state_config=raw_config["state"],
        metric_config=raw_config["metric"],
        concentration_config=raw_config["concentration"],
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


def build_hard_mask(hard_slices: list[dict[str, Any]], future_len: int, n_cells: int) -> np.ndarray:
    mask = np.zeros((future_len, n_cells), dtype=bool)
    for item in hard_slices:
        h_idx = int(item["horizon"]) - 1
        c = int(item["cell"][0]) * 5 + int(item["cell"][1])
        mask[h_idx, c] = True
    return mask


@torch.no_grad()
def run_common_backbone(
    model: StateMetricTransportModel,
    history_01_b: torch.Tensor,
    fut_01_b: torch.Tensor,
):
    batch = history_01_b.shape[0]
    future_len = fut_01_b.shape[1]
    n_cells = fut_01_b.shape[2]
    (
        mu_u,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        flow_context,
        base_local_delta,
        block_logits,
    ) = model.forward_from_history(history_01_b)
    det_iv = unconstrained_to_iv(mu_u, lo=model.support_lo, hi=model.support_hi)
    target_u = iv_to_unconstrained(
        fut_01_b,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    target_basis = model.teacher_basis_flat_from_outputs(
        target_u=target_u,
        mu=mu_u,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        base_local_delta=base_local_delta,
        block_logits=block_logits,
    ).view(batch, future_len, n_cells)
    target_basis_flat = target_basis.reshape(batch, -1)
    path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
    target_local_log, _target_band_log = compute_control_targets(model, target_basis)
    return det_iv, target_basis, target_basis_flat, path_context, target_local_log


@torch.no_grad()
def collect_stats_for_model(
    model: StateMetricTransportModel,
    model_kind: str,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    hard_slices: list[dict[str, Any]],
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)
    n_windows, future_len, n_cells = future_01.shape

    vov, q20, q80 = regime_masks_from_history(history_norm.cpu())
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    hard_mask = build_hard_mask(hard_slices, future_len, n_cells)

    sample_std = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    lo90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    hi90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    det_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    target_local_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    metric_local_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    metric_local_zero_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    local_gate_all = np.zeros((n_windows, 1), dtype=np.float32)

    extra = {
        "event_gate": np.zeros((n_windows, 1), dtype=np.float32),
        "event_budget": np.zeros((n_windows, 1), dtype=np.float32),
        "event_pos_top1": np.zeros((n_windows,), dtype=np.float32),
        "event_neg_top1": np.zeros((n_windows,), dtype=np.float32),
    }

    pooled_abs_delta_gt = []
    pooled_abs_delta_gen = []
    path_max_jump_gt = []
    path_max_jump_gen = []
    path_q99_exceed_count_gt = []
    path_q99_exceed_count_gen = []
    high_band_coeff_gt = []
    high_band_coeff_gen = []

    geometry = model.path_geometry
    high_band_mask = geometry.high_band_mask().reshape(1, future_len, n_cells).to(device)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_01_b = history_01[start:end]
        fut_01_b = future_01[start:end].to(device)
        batch = end - start

        det_iv, target_basis, target_basis_flat, path_context, target_local_log = run_common_backbone(
            model, hist_01_b, fut_01_b
        )
        det_mean[start:end] = det_iv.detach().cpu().numpy()
        target_local_all[start:end] = target_local_log.detach().cpu().numpy()

        t = torch.full((batch,), 0.5, device=device, dtype=target_basis.dtype)
        zero_state = torch.zeros_like(target_basis_flat)
        if model_kind == "183c":
            _raw0, _rawb0, metric_local_zero, _metric_band_zero, local_gate, _band_gate = model.build_state_metric_controls(
                zero_state, t, path_context
            )
            _rawt, _rawbt, metric_local_t, _metric_band_t, _local_gate_t, _band_gate_t = model.build_state_metric_controls(
                target_basis_flat, t, path_context
            )
            local_gate_all[start:end] = local_gate.detach().cpu().numpy()
        elif model_kind == "183d":
            (
                _raw0,
                _rawb0,
                metric_local_zero,
                _metric_band_zero,
                local_gate,
                _band_gate,
                event_gate0,
                event_budget0,
                alloc_pos0,
                alloc_neg0,
            ) = model.build_sparse_metric_controls(zero_state, t, path_context)
            (
                _rawt,
                _rawbt,
                metric_local_t,
                _metric_band_t,
                _local_gate_t,
                _band_gate_t,
                event_gate_t,
                event_budget_t,
                alloc_pos_t,
                alloc_neg_t,
            ) = model.build_sparse_metric_controls(target_basis_flat, t, path_context)
            local_gate_all[start:end] = local_gate.detach().cpu().numpy()
            extra["event_gate"][start:end] = event_gate_t.detach().cpu().numpy()
            extra["event_budget"][start:end] = event_budget_t.detach().cpu().numpy()
            extra["event_pos_top1"][start:end] = (
                alloc_pos_t.reshape(batch, -1).max(dim=1).values.detach().cpu().numpy()
            )
            extra["event_neg_top1"][start:end] = (
                alloc_neg_t.reshape(batch, -1).max(dim=1).values.detach().cpu().numpy()
            )
        else:
            raise ValueError(model_kind)

        metric_local_zero_all[start:end] = metric_local_zero.detach().cpu().numpy()
        metric_local_all[start:end] = metric_local_t.detach().cpu().numpy()

        samples_u = model.sample_future_u(hist_01_b, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
        samples_iv = samples_01.detach().cpu().numpy()
        sample_std[start:end] = samples_iv.std(axis=1)
        lo90[start:end] = np.quantile(samples_iv, 0.05, axis=1)
        hi90[start:end] = np.quantile(samples_iv, 0.95, axis=1)

        gt_path = torch.cat(
            [hist_01_b[:, -1:].reshape(batch, 1, 5, 5), fut_01_b.view(batch, future_len, 5, 5)],
            dim=1,
        )
        gen_path = torch.cat(
            [
                hist_01_b[:, -1:].reshape(batch, 1, 1, 5, 5).expand(-1, n_samples, -1, -1, -1),
                samples_01.view(batch, n_samples, future_len, 5, 5),
            ],
            dim=2,
        )
        gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
        gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]

        pooled_abs_delta_gt.append(gt_diff.abs().reshape(-1).cpu().numpy())
        pooled_abs_delta_gen.append(gen_diff.abs().reshape(-1).cpu().numpy())
        path_max_jump_gt.append(gt_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy())
        path_max_jump_gen.append(gen_diff.abs().amax(dim=(2, 3, 4)).reshape(-1).cpu().numpy())

        gt_q99 = torch.quantile(gt_diff.abs().reshape(-1), 0.99)
        path_q99_exceed_count_gt.append((gt_diff.abs().reshape(batch, -1) > gt_q99).sum(dim=1).cpu().numpy())
        path_q99_exceed_count_gen.append((gen_diff.abs().reshape(batch * n_samples, -1) > gt_q99).sum(dim=1).cpu().numpy())

        gt_coeff = geometry.to_basis(gt_diff.reshape(batch, future_len, n_cells))
        gen_coeff = geometry.to_basis(gen_diff.reshape(batch * n_samples, future_len, n_cells))
        gt_high = (gt_coeff * high_band_mask.to(dtype=gt_coeff.dtype)).abs().reshape(-1).cpu().numpy()
        gen_high = (gen_coeff * high_band_mask.to(dtype=gen_coeff.dtype)).abs().reshape(-1).cpu().numpy()
        high_band_coeff_gt.append(gt_high)
        high_band_coeff_gen.append(gen_high)

    future_np = future_01.cpu().numpy()
    inside90 = (future_np >= lo90) & (future_np <= hi90)
    det_resid = future_np - det_mean
    z_mean = det_resid / np.maximum(sample_std, 1e-6)

    turb_cov = inside90[turb_mask]
    turb_std = sample_std[turb_mask]
    turb_gt = future_np[turb_mask]
    turb_det = det_mean[turb_mask]
    turb_det_resid = turb_gt - turb_det

    hard_slice_rows = []
    for item in hard_slices:
        h = int(item["horizon"])
        h_idx = h - 1
        c = int(item["cell"][0]) * 5 + int(item["cell"][1])
        cov = float(turb_cov[:, h_idx, c].mean())
        gen_std = float(turb_std[:, h_idx, c].mean())
        gt_std = float(np.std(turb_det_resid[:, h_idx, c]))
        ratio = gen_std / max(gt_std, 1e-6)
        hard_slice_rows.append(
            {
                "horizon": h,
                "cell": item["cell"],
                "coverage_90": cov,
                "std_ratio_gen_to_gt": ratio,
                "z_mean": float(z_mean[turb_mask, h_idx, c].mean()),
                "target_local_log": float(target_local_all[turb_mask, h_idx, c].mean()),
                "metric_local_target": float(metric_local_all[turb_mask, h_idx, c].mean()),
                "metric_local_zero": float(metric_local_zero_all[turb_mask, h_idx, c].mean()),
            }
        )

    hard_positive_share_target = float(
        np.maximum(target_local_all[turb_mask], 0.0)[:, hard_mask].sum() / max(np.maximum(target_local_all[turb_mask], 0.0).sum(), 1e-12)
    )
    hard_positive_share_metric = float(
        np.maximum(metric_local_all[turb_mask], 0.0)[:, hard_mask].sum() / max(np.maximum(metric_local_all[turb_mask], 0.0).sum(), 1e-12)
    )

    neighbor_reviews = []
    for item in hard_slices:
        h_idx = int(item["horizon"]) - 1
        cell = int(item["cell"][0]) * 5 + int(item["cell"][1])
        neigh = grid_neighbors(cell)
        if not neigh:
            continue
        target_cell = target_local_all[turb_mask, h_idx, cell]
        target_neigh = target_local_all[turb_mask, h_idx][:, neigh].mean(axis=1)
        metric_cell = metric_local_all[turb_mask, h_idx, cell]
        metric_neigh = metric_local_all[turb_mask, h_idx][:, neigh].mean(axis=1)
        neighbor_reviews.append(
            {
                "horizon": int(item["horizon"]),
                "cell": item["cell"],
                "target_contrast": float((target_cell - target_neigh).mean()),
                "metric_contrast": float((metric_cell - metric_neigh).mean()),
                "contrast_ratio_metric_to_target": float(
                    ((metric_cell - metric_neigh).mean()) / max(abs((target_cell - target_neigh).mean()), 1e-6)
                ),
            }
        )

    pooled_gt = np.concatenate(pooled_abs_delta_gt)
    pooled_gen = np.concatenate(pooled_abs_delta_gen)
    gt_path_max = np.concatenate(path_max_jump_gt)
    gen_path_max = np.concatenate(path_max_jump_gen)
    gt_q99_exceed = np.concatenate(path_q99_exceed_count_gt)
    gen_q99_exceed = np.concatenate(path_q99_exceed_count_gen)
    high_band_abs_gt = np.concatenate(high_band_coeff_gt)
    high_band_abs_gen = np.concatenate(high_band_coeff_gen)

    out = {
        "control": {
            "hard_positive_share_target": hard_positive_share_target,
            "hard_positive_share_metric": hard_positive_share_metric,
            "mean_metric_local_hard": float(metric_local_all[turb_mask][:, hard_mask].mean()) if hard_mask.any() else float("nan"),
            "mean_target_local_hard": float(target_local_all[turb_mask][:, hard_mask].mean()) if hard_mask.any() else float("nan"),
            "local_gate_mean_turb": float(local_gate_all[turb_mask].mean()),
            "local_gate_mean_calm": float(local_gate_all[calm_mask].mean()),
        },
        "width": {
            "hard_slices": hard_slice_rows,
            "neighbor_contrast": neighbor_reviews,
        },
        "tail": {
            "pooled_abs_delta": distribution_compare(pooled_gt, pooled_gen),
            "pathwise_max_jump": distribution_compare(gt_path_max, gen_path_max),
            "gt_q99_exceed_count": quantile_profile(gt_q99_exceed),
            "gen_q99_exceed_count": quantile_profile(gen_q99_exceed),
            "pooled_abs_delta_exceedance_spectrum": exceedance_spectrum(pooled_gt, pooled_gen),
            "high_band_abs_exceedance_spectrum": exceedance_spectrum(high_band_abs_gt, high_band_abs_gen),
        },
    }
    if model_kind == "183d":
        hard_window_indicator = np.zeros((n_windows,), dtype=bool)
        for item in hard_slices:
            h_idx = int(item["horizon"]) - 1
            c = int(item["cell"][0]) * 5 + int(item["cell"][1])
            hard_window_indicator |= ~inside90[:, h_idx, c]
        hard_turb_windows = turb_mask & hard_window_indicator
        clean_turb_windows = turb_mask & ~hard_window_indicator
        out["event"] = {
            "event_gate_mean_all": float(extra["event_gate"].mean()),
            "event_budget_mean_all": float(extra["event_budget"].mean()),
            "event_pos_top1_mean_all": float(extra["event_pos_top1"].mean()),
            "event_neg_top1_mean_all": float(extra["event_neg_top1"].mean()),
            "event_gate_mean_hard_turb": float(extra["event_gate"][hard_turb_windows].mean()) if hard_turb_windows.any() else float("nan"),
            "event_gate_mean_clean_turb": float(extra["event_gate"][clean_turb_windows].mean()) if clean_turb_windows.any() else float("nan"),
            "event_budget_mean_hard_turb": float(extra["event_budget"][hard_turb_windows].mean()) if hard_turb_windows.any() else float("nan"),
            "event_budget_mean_clean_turb": float(extra["event_budget"][clean_turb_windows].mean()) if clean_turb_windows.any() else float("nan"),
            "event_pos_top1_mean_hard_turb": float(extra["event_pos_top1"][hard_turb_windows].mean()) if hard_turb_windows.any() else float("nan"),
            "event_pos_top1_mean_clean_turb": float(extra["event_pos_top1"][clean_turb_windows].mean()) if clean_turb_windows.any() else float("nan"),
        }
    return out


def hard_slice_delta(rows_c: list[dict[str, Any]], rows_d: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_c = {(r["horizon"], tuple(r["cell"])): r for r in rows_c}
    index_d = {(r["horizon"], tuple(r["cell"])): r for r in rows_d}
    out = []
    for key, rc in index_c.items():
        if key not in index_d:
            continue
        rd = index_d[key]
        out.append(
            {
                "horizon": rc["horizon"],
                "cell": list(rc["cell"]),
                "coverage_183c": rc["coverage_90"],
                "coverage_183d": rd["coverage_90"],
                "coverage_delta": rd["coverage_90"] - rc["coverage_90"],
                "std_ratio_183c": rc["std_ratio_gen_to_gt"],
                "std_ratio_183d": rd["std_ratio_gen_to_gt"],
                "std_ratio_delta": rd["std_ratio_gen_to_gt"] - rc["std_ratio_gen_to_gt"],
                "metric_local_183c": rc["metric_local_target"],
                "metric_local_183d": rd["metric_local_target"],
                "metric_delta": rd["metric_local_target"] - rc["metric_local_target"],
            }
        )
    out.sort(key=lambda x: x["coverage_delta"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 183c_best vs 183d_best mechanisms")
    parser.add_argument("--checkpoint_183c", type=str, required=True)
    parser.add_argument("--checkpoint_183d", type=str, required=True)
    parser.add_argument("--review_183c", type=str, required=True)
    parser.add_argument("--summary_183c", type=str, required=True)
    parser.add_argument("--summary_183d", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    review_183c = load_json(args.review_183c)
    summary_183c = load_json(args.summary_183c)
    summary_183d = load_json(args.summary_183d)
    hard_slices = review_183c["width_allocation_review"]["top_turbulent_hard_slices"][:12]

    model_183c, _ = load_183c_model(args.checkpoint_183c, args.device)
    model_183d, _ = load_183d_model(args.checkpoint_183d, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    stats_183c = collect_stats_for_model(
        model=model_183c,
        model_kind="183c",
        history_norm=history_norm,
        future_norm=future_norm,
        hard_slices=hard_slices,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    stats_183d = collect_stats_for_model(
        model=model_183d,
        model_kind="183d",
        history_norm=history_norm,
        future_norm=future_norm,
        hard_slices=hard_slices,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )

    compare = {
        "suite_context": {
            "183c_best_passes": [k for k, v in summary_183c.items() if isinstance(v, dict) and v.get("overall_pass") is True],
            "183d_best_passes": [k for k, v in summary_183d.items() if isinstance(v, dict) and v.get("overall_pass") is True],
            "183c_best_fails": [k for k, v in summary_183c.items() if isinstance(v, dict) and v.get("overall_pass") is False],
            "183d_best_fails": [k for k, v in summary_183d.items() if isinstance(v, dict) and v.get("overall_pass") is False],
        },
        "control_comparison": {
            "hard_positive_share_metric_183c": stats_183c["control"]["hard_positive_share_metric"],
            "hard_positive_share_metric_183d": stats_183d["control"]["hard_positive_share_metric"],
            "hard_positive_share_delta": stats_183d["control"]["hard_positive_share_metric"] - stats_183c["control"]["hard_positive_share_metric"],
            "mean_metric_local_hard_183c": stats_183c["control"]["mean_metric_local_hard"],
            "mean_metric_local_hard_183d": stats_183d["control"]["mean_metric_local_hard"],
            "mean_metric_local_hard_delta": stats_183d["control"]["mean_metric_local_hard"] - stats_183c["control"]["mean_metric_local_hard"],
            "local_gate_mean_turb_183c": stats_183c["control"]["local_gate_mean_turb"],
            "local_gate_mean_turb_183d": stats_183d["control"]["local_gate_mean_turb"],
        },
        "hard_slice_progress": hard_slice_delta(
            stats_183c["width"]["hard_slices"],
            stats_183d["width"]["hard_slices"],
        ),
        "neighbor_contrast_comparison": {
            "mean_abs_contrast_ratio_183c": float(np.mean([abs(x["contrast_ratio_metric_to_target"]) for x in stats_183c["width"]["neighbor_contrast"]])),
            "mean_abs_contrast_ratio_183d": float(np.mean([abs(x["contrast_ratio_metric_to_target"]) for x in stats_183d["width"]["neighbor_contrast"]])),
        },
        "tail_comparison": {
            "pooled_quiet_mass_ratio_183c": stats_183c["tail"]["pooled_abs_delta_exceedance_spectrum"]["quiet_mass"]["ratio"],
            "pooled_quiet_mass_ratio_183d": stats_183d["tail"]["pooled_abs_delta_exceedance_spectrum"]["quiet_mass"]["ratio"],
            "pooled_shoulder_mass_ratio_183c": stats_183c["tail"]["pooled_abs_delta_exceedance_spectrum"]["shoulder_mass"]["ratio"],
            "pooled_shoulder_mass_ratio_183d": stats_183d["tail"]["pooled_abs_delta_exceedance_spectrum"]["shoulder_mass"]["ratio"],
            "pooled_extreme_mass_ratio_183c": stats_183c["tail"]["pooled_abs_delta_exceedance_spectrum"]["extreme_mass"]["ratio"],
            "pooled_extreme_mass_ratio_183d": stats_183d["tail"]["pooled_abs_delta_exceedance_spectrum"]["extreme_mass"]["ratio"],
            "high_band_extreme_mass_ratio_183c": stats_183c["tail"]["high_band_abs_exceedance_spectrum"]["extreme_mass"]["ratio"],
            "high_band_extreme_mass_ratio_183d": stats_183d["tail"]["high_band_abs_exceedance_spectrum"]["extreme_mass"]["ratio"],
        },
        "event_review_183d": stats_183d.get("event", {}),
        "diagnosis": {
            "183d": "183d activates the event path broadly rather than selectively. It increases local concentration only modestly on the true hard slices, while giving back quiet-mass discipline and broad distributional fidelity.",
            "why_no_frontier_gain": "The sparse patch was not actually sparse enough. It turned on in many windows, failed to sharpen target-vs-neighbor contrast enough on the hard S3/S7 slices, and pushed the residual law back toward shoulder-heavy behavior that hurt S4 and S8.",
            "next_principled_step": "If continuing, the next model should not be another always-on event patch. It should use an explicit latent quiet-vs-event residual state so event transport only activates on the right windows before allocating mass within those windows.",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_serializable(compare), indent=2))
    print(f"Wrote comparison review to {output_path}")


if __name__ == "__main__":
    main()
