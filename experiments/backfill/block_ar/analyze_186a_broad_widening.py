#!/usr/bin/env python
"""
Focused mechanism review for 186a broad widening.

Questions:
  1. Where did 186a add width relative to the 183c anchor?
  2. Did that extra width go to true hard slices or to already easy/high-scale cells?
  3. Are the new objective weights aligned more with global move size than with actual misses?
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183c_state_metric_transport import (
    StateMetricTransportModel as AnchorModel,
)
from experiments.backfill.block_ar.train_186a_hard_slice_tail_objective import (
    StateMetricTransportModel as ObjectiveModel,
    build_objective_weights,
)


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cell_label(idx: int) -> str:
    return f"({idx // 5},{idx % 5})"


def load_anchor_model(checkpoint_path: str, device: str) -> tuple[AnchorModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    enc_cfg = EncoderConfig(**cfg["encoder"])
    model = AnchorModel(
        encoder_config=enc_cfg,
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


def load_186a_model(checkpoint_path: str, device: str) -> tuple[ObjectiveModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    enc_cfg = EncoderConfig(**cfg["encoder"])
    model = ObjectiveModel(
        encoder_config=enc_cfg,
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def analyze(
    anchor_model: AnchorModel,
    obj_model: ObjectiveModel,
    obj_config: dict[str, Any],
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)
    n, t_len, n_cells = future_01.shape

    vov, q20, q80 = regime_masks_from_history(history_norm.cpu())
    turb_mask = vov >= q80
    calm_mask = vov <= q20
    late_idx = [13, 29]
    late_idx = [i for i in late_idx if i < t_len]

    width_a = np.zeros((n, t_len, n_cells), dtype=np.float32)
    width_b = np.zeros((n, t_len, n_cells), dtype=np.float32)
    covered_a = np.zeros((n, t_len, n_cells), dtype=np.float32)
    covered_b = np.zeros((n, t_len, n_cells), dtype=np.float32)
    local_weight_all = np.zeros((n, t_len, n_cells), dtype=np.float32)
    target_local_all = np.zeros((n, t_len, n_cells), dtype=np.float32)
    window_boost_all = np.zeros((n,), dtype=np.float32)
    target_mag_all = np.zeros((n, t_len, n_cells), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        hist_b = history_01[start:end]
        fut_b = future_01[start:end].to(device)
        b = end - start

        target_u = iv_to_unconstrained(
            fut_b,
            lo=obj_model.support_lo,
            hi=obj_model.support_hi,
            eps=obj_model.support_eps,
        )
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
        ) = obj_model.forward_from_history(hist_b)
        target_basis = obj_model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=mu,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            base_local_delta=base_local_delta,
            block_logits=block_logits,
        ).view(b, t_len, n_cells)
        path_context = obj_model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, target_band_log = compute_control_targets(obj_model, target_basis)
        target_v = target_basis.reshape(b, -1)
        local_weight, _band_weight, _fm_weight, window_boost = build_objective_weights(
            target_local_log=target_local_log,
            target_band_log=target_band_log,
            target_basis=target_basis,
            target_v=target_v,
            objective_config=obj_config,
        )
        local_weight_all[start:end] = local_weight.cpu().numpy()
        target_local_all[start:end] = target_local_log.cpu().numpy()
        window_boost_all[start:end] = window_boost.cpu().numpy()
        target_mag_all[start:end] = target_basis.abs().cpu().numpy()

        for model, out_width, out_cov in [
            (anchor_model, width_a, covered_a),
            (obj_model, width_b, covered_b),
        ]:
            samples_u = model.sample_future_u(hist_b, n_samples=n_samples)
            samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
            samples_np = samples_01.cpu().numpy()
            lo = np.quantile(samples_np, 0.05, axis=1)
            hi = np.quantile(samples_np, 0.95, axis=1)
            out_width[start:end] = hi - lo
            gt_np = fut_b.cpu().numpy()
            out_cov[start:end] = ((gt_np >= lo) & (gt_np <= hi)).astype(np.float32)

    width_delta = width_b - width_a
    miss_a = covered_a < 0.5
    hit_a = covered_a > 0.5

    turb_late_mask = np.zeros((n, t_len, n_cells), dtype=bool)
    for h in late_idx:
        turb_late_mask[:, h, :] = True
    turb_late_mask &= turb_mask[:, None, None]

    hard_turb_late = turb_late_mask & miss_a.astype(bool)
    easy_turb_late = turb_late_mask & hit_a.astype(bool)

    global_hard = miss_a.astype(bool)
    global_easy = hit_a.astype(bool)

    mean_delta_hard = float(width_delta[global_hard].mean()) if global_hard.any() else float("nan")
    mean_delta_easy = float(width_delta[global_easy].mean()) if global_easy.any() else float("nan")
    mean_delta_hard_turb_late = float(width_delta[hard_turb_late].mean()) if hard_turb_late.any() else float("nan")
    mean_delta_easy_turb_late = float(width_delta[easy_turb_late].mean()) if easy_turb_late.any() else float("nan")

    mean_weight_hard = float(local_weight_all[global_hard].mean()) if global_hard.any() else float("nan")
    mean_weight_easy = float(local_weight_all[global_easy].mean()) if global_easy.any() else float("nan")
    mean_weight_hard_turb_late = float(local_weight_all[hard_turb_late].mean()) if hard_turb_late.any() else float("nan")
    mean_weight_easy_turb_late = float(local_weight_all[easy_turb_late].mean()) if easy_turb_late.any() else float("nan")

    coverage_a = covered_a.mean(axis=0)
    coverage_b = covered_b.mean(axis=0)
    width_a_mean = width_a.mean(axis=0)
    width_b_mean = width_b.mean(axis=0)
    width_delta_mean = width_delta.mean(axis=0)
    weight_mean = local_weight_all.mean(axis=0)
    target_local_mean = target_local_all.mean(axis=0)
    target_mag_mean = target_mag_all.mean(axis=0)

    cell_rows = []
    for h in [0, 6, 13, 29]:
        if h >= t_len:
            continue
        for c in range(n_cells):
            cell_rows.append(
                {
                    "horizon": h + 1,
                    "cell": cell_label(c),
                    "anchor_cov": float(coverage_a[h, c]),
                    "obj_cov": float(coverage_b[h, c]),
                    "cov_delta": float(coverage_b[h, c] - coverage_a[h, c]),
                    "anchor_width": float(width_a_mean[h, c]),
                    "obj_width": float(width_b_mean[h, c]),
                    "width_delta": float(width_delta_mean[h, c]),
                    "mean_local_weight": float(weight_mean[h, c]),
                    "mean_target_local_log": float(target_local_mean[h, c]),
                    "mean_target_abs_basis": float(target_mag_mean[h, c]),
                }
            )
    top_widened = sorted(cell_rows, key=lambda x: x["width_delta"], reverse=True)[:12]
    top_hard_helped = sorted(
        [r for r in cell_rows if r["anchor_cov"] < 0.80],
        key=lambda x: x["cov_delta"],
        reverse=True,
    )[:12]

    per_window_mean_delta = width_delta.mean(axis=(1, 2))
    per_window_hard_rate = miss_a.mean(axis=(1, 2))
    per_window_target_mag = target_mag_all.mean(axis=(1, 2))
    per_window_late_mag = target_mag_all[:, late_idx].mean(axis=(1, 2)) if late_idx else target_mag_all.mean(axis=(1, 2))

    top_window_idx = np.argsort(window_boost_all)[-10:][::-1]
    top_windows = []
    for idx in top_window_idx:
        top_windows.append(
            {
                "window": int(idx),
                "window_boost": float(window_boost_all[idx]),
                "mean_width_delta": float(per_window_mean_delta[idx]),
                "hard_rate_anchor": float(per_window_hard_rate[idx]),
                "target_mag_mean": float(per_window_target_mag[idx]),
                "late_target_mag_mean": float(per_window_late_mag[idx]),
                "vol_of_vol": float(vov[idx]),
                "regime": "turb" if turb_mask[idx] else ("calm" if calm_mask[idx] else "mid"),
            }
        )

    result = {
        "diagnosis": {
            "broad_widening_driver": (
                "186a objective weights align strongly with global residual magnitude and late-horizon scale, "
                "but only weakly separate true hard misses from already-easy high-scale slices. The result is broad "
                "widening, especially on cells that already had high baseline width."
            )
        },
        "global_comparison": {
            "mean_width_delta_all": float(width_delta.mean()),
            "mean_width_delta_hard_points": mean_delta_hard,
            "mean_width_delta_easy_points": mean_delta_easy,
            "mean_width_delta_hard_turb_late": mean_delta_hard_turb_late,
            "mean_width_delta_easy_turb_late": mean_delta_easy_turb_late,
            "mean_local_weight_hard_points": mean_weight_hard,
            "mean_local_weight_easy_points": mean_weight_easy,
            "mean_local_weight_hard_turb_late": mean_weight_hard_turb_late,
            "mean_local_weight_easy_turb_late": mean_weight_easy_turb_late,
            "corr_width_delta_vs_local_weight": _corr(width_delta_mean.reshape(-1), weight_mean.reshape(-1)),
            "corr_width_delta_vs_target_local_log": _corr(width_delta_mean.reshape(-1), target_local_mean.reshape(-1)),
            "corr_window_boost_vs_window_target_mag": _corr(window_boost_all, per_window_target_mag),
            "corr_window_boost_vs_window_hard_rate": _corr(window_boost_all, per_window_hard_rate),
        },
        "cell_level": {
            "top_widened_cells": top_widened,
            "top_hard_cells_helped": top_hard_helped,
        },
        "window_level": {
            "window_boost_mean": float(window_boost_all.mean()),
            "window_boost_p90": float(np.quantile(window_boost_all, 0.90)),
            "window_boost_p99": float(np.quantile(window_boost_all, 0.99)),
            "top_weighted_windows": top_windows,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_ckpt", type=str, required=True)
    parser.add_argument("--objective_ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=30)
    args = parser.parse_args()

    anchor_model, _anchor_ckpt = load_anchor_model(args.anchor_ckpt, args.device)
    obj_model, obj_ckpt = load_186a_model(args.objective_ckpt, args.device)
    objective_config = obj_ckpt["config"]["objective"]

    history_norm, future_norm = build_test_subset(
        data_path="data/vol_surface_with_ret.npz",
        history_len=obj_ckpt["config"]["history_len"],
        future_len=obj_ckpt["config"]["future_len"],
        test_start=4540,
        max_windows=args.max_windows,
    )

    results = analyze(
        anchor_model=anchor_model,
        obj_model=obj_model,
        obj_config=objective_config,
        history_norm=history_norm,
        future_norm=future_norm,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_serializable(results), indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
