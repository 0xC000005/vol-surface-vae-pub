#!/usr/bin/env python
"""
Focused mean-reversion diagnosis for 176b.

Questions:
  1. Is under-reversion mainly in the deterministic mean path or only after sampling?
  2. Which cells and regimes drive the Suite 10 failure?
  3. Does sampling help enough to rescue the mean path, or is the shared mean law itself too weak?
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
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_176b_shared_local_template_mixture import (
    SharedLocalTemplateMixtureStudentTModel,
)


REGIME_NAMES = ["all", "calm", "turb"]


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


def slope_intercept_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = float(np.square(x_centered).sum())
    if denom <= 1e-12:
        slope = 0.0
    else:
        slope = float((x_centered * y_centered).sum() / denom)
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
        "gt_cell_slopes": gt_cell_slopes,
        "pred_cell_slopes": pred_cell_slopes,
        "cell_ratio": cell_ratio,
        "active_cell_mask": active_mask,
        "active_cell_pass": cell_pass,
        "active_pass_count": active_pass_count,
        "active_cell_count": active_count,
        "active_pass_rate": active_pass_rate,
        "active_cell_slope_corr": slope_corr,
        "worst_active_cells": worst_active_cells[:10],
    }


@torch.no_grad()
def analyze_mean_reversion(
    model: SharedLocalTemplateMixtureStudentTModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    n_samples: int,
    active_threshold: float,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    n_windows = history_01.shape[0]

    prev_all = history_01[:, -1].cpu().numpy()
    gt_next_all = future_01[:, 0].cpu().numpy()
    det_next_all = np.zeros_like(gt_next_all)
    sampled_next_all = np.zeros_like(gt_next_all)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_01_b = history_01[start:end]
        (
            mu,
            _time_factor,
            _time_diag,
            _cell_factor,
            _cell_diag,
            _scale,
            _flow_context,
            _base_local_delta,
            _local_delta_components,
            _gate_logits,
        ) = model.forward_from_history(hist_01_b)
        det_next_all[start:end] = unconstrained_to_iv(
            mu[:, 0],
            lo=model.support_lo,
            hi=model.support_hi,
        ).cpu().numpy().reshape(end - start, 5, 5)

        hist_norm_b = history_norm[start:end]
        samples = model.sample_batched(hist_norm_b, n_samples=n_samples)
        sampled_next_all[start:end] = samples[:, :, 0].mean(dim=1).cpu().numpy()

    calm_mask = vov <= q20
    turb_mask = vov >= q80
    masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm": calm_mask,
        "turb": turb_mask,
    }

    summary: dict[str, Any] = {
        "methodology": {
            "description": "176b deterministic vs sampled first-step mean reversion diagnosis",
            "n_windows": int(n_windows),
            "n_samples_for_sampled_mean": int(n_samples),
            "active_slope_threshold": float(active_threshold),
        },
        "regime_thresholds": {
            "vol_of_vol_q20": float(q20),
            "vol_of_vol_q80": float(q80),
        },
        "regime_counts": {
            "all": int(masks["all"].sum()),
            "calm": int(calm_mask.sum()),
            "turb": int(turb_mask.sum()),
        },
        "by_regime": {},
    }

    for regime_name, mask in masks.items():
        prev = prev_all[mask]
        gt_next = gt_next_all[mask]
        det_next = det_next_all[mask]
        sampled_next = sampled_next_all[mask]

        det_stats = summarize_variant(prev, det_next, gt_next, active_threshold)
        sampled_stats = summarize_variant(prev, sampled_next, gt_next, active_threshold)

        gain_grid = sampled_stats["cell_ratio"] - det_stats["cell_ratio"]
        gain_records = []
        for i in range(5):
            for j in range(5):
                if not sampled_stats["active_cell_mask"][i, j]:
                    continue
                gain_records.append(
                    {
                        "cell": [i, j],
                        "det_ratio": float(det_stats["cell_ratio"][i, j]),
                        "sampled_ratio": float(sampled_stats["cell_ratio"][i, j]),
                        "sample_gain": float(gain_grid[i, j]),
                    }
                )
        gain_records.sort(key=lambda x: abs(x["sample_gain"]), reverse=True)

        summary["by_regime"][regime_name] = {
            "deterministic": det_stats,
            "sampled_mean": sampled_stats,
            "sample_gain_top": gain_records[:10],
        }

    return make_serializable(summary)


def main():
    parser = argparse.ArgumentParser(description="Analyze 176b mean-reversion failure slices")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/backfill/shared_local_template_mixture_residual_flow_structured_joint_student_t_176b/best_model.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-05/analysis/176b_mean_reversion_mechanistic",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--active_slope_threshold", type=float, default=0.05)
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

    summary = analyze_mean_reversion(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        vov=vov,
        q20=q20,
        q80=q80,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        active_threshold=args.active_slope_threshold,
    )
    summary["model_path"] = args.model_path
    summary["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))

    out_path = Path(args.output_dir) / "mechanistic_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved analysis to {out_path}")


if __name__ == "__main__":
    main()
