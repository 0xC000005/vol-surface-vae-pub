#!/usr/bin/env python
"""
Focused mechanistic study for the H7 density branch.

Compares 169c vs 170b on five questions:
  1. Teacher-forced vs free-rollout error growth by horizon and cell
  2. Mean-vs-variance decomposition of S2/S7 failures
  3. Predicted covariance vs empirical residual covariance by regime
  4. Whitened-residual diagnostics for 170b
  5. Cross-cell spectrum by horizon (where correlation magnitude is lost / rank inflates)

This is diagnostic-only. It does not change model weights.
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    ShapeScaleStudentTARModel,
)
from experiments.backfill.block_ar.train_170b_whitened_flow import (
    WhitenedFlowARModel,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


SELECT_HORIZONS = [1, 7, 14, 30]
SELECT_HIDX = [h - 1 for h in SELECT_HORIZONS]
REGIME_NAMES = ["all", "calm", "turb"]


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return make_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def eff_rank_from_matrix(mat: np.ndarray, eps: float = 1e-10) -> float:
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.maximum(eigvals, eps)
    probs = eigvals / np.maximum(eigvals.sum(), eps)
    probs = probs[probs > eps]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def mean_offdiag_corr(cov: np.ndarray, eps: float = 1e-10) -> float:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return float(corr[mask].mean())


def corr_from_cov(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), eps, None))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def sample_cov(samples: torch.Tensor) -> torch.Tensor:
    """
    Args:
        samples: (B, S, C)
    Returns:
        cov: (B, C, C)
    """
    centered = samples - samples.mean(dim=1, keepdim=True)
    denom = max(samples.shape[1] - 1, 1)
    return torch.matmul(centered.transpose(1, 2), centered) / denom


def summarize_z(arr: np.ndarray) -> dict[str, float]:
    flat_mean = float(arr.mean())
    flat_std = float(arr.std())
    centered = arr - arr.mean(axis=0, keepdims=True)
    var = centered.var(axis=0)
    var_safe = np.clip(var, 1e-8, None)
    kurt = ((centered ** 4).mean(axis=0) / (var_safe ** 2)).mean()
    cov = np.cov(arr, rowvar=False, bias=True)
    return {
        "mean": flat_mean,
        "std": flat_std,
        "mean_abs": float(np.abs(arr).mean()),
        "mean_offdiag_corr": mean_offdiag_corr(cov),
        "eff_rank": eff_rank_from_matrix(cov),
        "avg_kurtosis": float(kurt),
    }


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    model_type = raw_config["type"]
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    common = dict(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
    )
    if model_type == "multi_step_student_t_169c":
        model = ShapeScaleStudentTARModel(**common)
    elif model_type == "whitened_flow_170b":
        model = WhitenedFlowARModel(flow_config=raw_config["flow"], **common)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, model_type, checkpoint


def build_test_subset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(data_path)
    dataset = VolSurfaceDataset(
        data["surface"],
        history_len,
        future_len,
        start_idx=test_start,
    )
    n = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    histories = []
    futures = []
    for i in range(n):
        item = dataset[i]
        histories.append(item["history"])
        futures.append(item["future"])
    return torch.stack(histories, dim=0), torch.stack(futures, dim=0)


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, float, float]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    return vov, q20, q80


def analyze_model(
    model,
    model_type: str,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    tf_samples: int,
    rollout_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01_all = denormalize_iv(history_norm).reshape(history_norm.shape[0], history_norm.shape[1], -1)
    future_01_all = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    n_windows, future_len, n_cells = future_01_all.shape
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm": calm_mask,
        "turb": turb_mask,
    }

    tf_mean_chunks = []
    tf_std_chunks = []
    tf_lo_chunks = []
    tf_hi_chunks = []
    tf_scale_chunks = []
    tf_shape_var_chunks = []

    ro_mean_chunks = []
    ro_std_chunks = []
    ro_lo_chunks = []
    ro_hi_chunks = []

    residuals_tf = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pred_cov_sum = {name: np.zeros((future_len, n_cells, n_cells), dtype=np.float64) for name in REGIME_NAMES}
    pred_cov_count = {name: np.zeros(future_len, dtype=np.int64) for name in REGIME_NAMES}

    gt_delta_sum = np.zeros((len(SELECT_HIDX), n_cells), dtype=np.float64)
    gt_delta_outer = np.zeros((len(SELECT_HIDX), n_cells, n_cells), dtype=np.float64)
    gt_delta_count = np.zeros(len(SELECT_HIDX), dtype=np.int64)

    ro_delta_sum = np.zeros((len(SELECT_HIDX), n_cells), dtype=np.float64)
    ro_delta_outer = np.zeros((len(SELECT_HIDX), n_cells, n_cells), dtype=np.float64)
    ro_delta_count = np.zeros(len(SELECT_HIDX), dtype=np.int64)

    z_stats_store: dict[str, list[np.ndarray]] | None = None
    if model_type == "whitened_flow_170b":
        z_stats_store = {"white": [], "z": []}

    with torch.no_grad():
        row0 = 0
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            hist_norm_b = history_norm[start:end]
            fut_norm_b = future_norm[start:end]
            hist_01_b = denormalize_iv(hist_norm_b)
            fut_01_b = denormalize_iv(fut_norm_b).reshape(end - start, future_len, n_cells)
            row1 = row0 + (end - start)

            current_hist_01 = hist_01_b.clone()
            batch_tf_mean = []
            batch_tf_std = []
            batch_tf_lo = []
            batch_tf_hi = []
            batch_tf_scale = []
            batch_tf_shape_var = []
            batch_white = []
            batch_z = []

            for t in range(future_len):
                pred_samples = model.sample_next_iv(current_hist_01, n_samples=tf_samples).float()
                pred_mean = pred_samples.mean(dim=1)
                pred_std = pred_samples.std(dim=1, unbiased=False)
                pred_lo = torch.quantile(pred_samples, 0.05, dim=1)
                pred_hi = torch.quantile(pred_samples, 0.95, dim=1)
                pred_cov = sample_cov(pred_samples).detach().cpu().numpy()

                target_t = fut_01_b[:, t, :]

                if model_type == "multi_step_student_t_169c":
                    mu, factor, diag, scale, _nu = model.forward_from_history(current_hist_01)
                    _, _, avg_var = model.normalized_components(factor, diag)
                else:
                    mu, factor, diag, scale, flow_context = model.forward_from_history(current_hist_01)
                    _, _, avg_var = model.normalized_components(factor, diag)
                    target_u = iv_to_unconstrained(
                        target_t,
                        lo=model.support_lo,
                        hi=model.support_hi,
                        eps=model.support_eps,
                    )
                    cov = model.covariance(factor, diag, scale)
                    chol = torch.linalg.cholesky(cov)
                    diff = (target_u - mu).unsqueeze(-1)
                    white = torch.linalg.solve_triangular(chol, diff, upper=False).squeeze(-1)
                    z, _ = model.flow(white, flow_context)
                    batch_white.append(white.detach().cpu())
                    batch_z.append(z.detach().cpu())

                batch_tf_mean.append(pred_mean.detach().cpu())
                batch_tf_std.append(pred_std.detach().cpu())
                batch_tf_lo.append(pred_lo.detach().cpu())
                batch_tf_hi.append(pred_hi.detach().cpu())
                batch_tf_scale.append(scale.detach().cpu())
                batch_tf_shape_var.append(avg_var.detach().cpu())

                residuals_tf[row0:row1, t] = (target_t - pred_mean).detach().cpu().numpy()
                batch_regime = {
                    "all": np.ones(end - start, dtype=bool),
                    "calm": calm_mask[start:end],
                    "turb": turb_mask[start:end],
                }
                for name, mask in batch_regime.items():
                    if not mask.any():
                        continue
                    pred_cov_sum[name][t] += pred_cov[mask].sum(axis=0)
                    pred_cov_count[name][t] += int(mask.sum())

                current_hist_01 = torch.cat(
                    [current_hist_01[:, 1:], target_t.view(end - start, 1, 5, 5)],
                    dim=1,
                )

            tf_mean_chunks.append(torch.stack(batch_tf_mean, dim=1).numpy())
            tf_std_chunks.append(torch.stack(batch_tf_std, dim=1).numpy())
            tf_lo_chunks.append(torch.stack(batch_tf_lo, dim=1).numpy())
            tf_hi_chunks.append(torch.stack(batch_tf_hi, dim=1).numpy())
            tf_scale_chunks.append(torch.stack(batch_tf_scale, dim=1).numpy())
            tf_shape_var_chunks.append(torch.stack(batch_tf_shape_var, dim=1).numpy())
            if z_stats_store is not None:
                z_stats_store["white"].append(torch.stack(batch_white, dim=1).numpy())
                z_stats_store["z"].append(torch.stack(batch_z, dim=1).numpy())

            rollout = model.sample_batched(
                hist_norm_b,
                n_samples=rollout_samples,
                n_steps=future_len,
            ).view(end - start, rollout_samples, future_len, n_cells)
            ro_mean = rollout.mean(dim=1)
            ro_std = rollout.std(dim=1, unbiased=False)
            ro_lo = torch.quantile(rollout, 0.05, dim=1)
            ro_hi = torch.quantile(rollout, 0.95, dim=1)

            ro_mean_chunks.append(ro_mean.detach().cpu().numpy())
            ro_std_chunks.append(ro_std.detach().cpu().numpy())
            ro_lo_chunks.append(ro_lo.detach().cpu().numpy())
            ro_hi_chunks.append(ro_hi.detach().cpu().numpy())

            hist_last = hist_01_b[:, -1].reshape(end - start, n_cells)
            gt_batch = fut_01_b.detach().cpu().numpy()
            rollout_np = rollout.detach().cpu().numpy()
            hist_last_np = hist_last.detach().cpu().numpy()

            for j, hidx in enumerate(SELECT_HIDX):
                if hidx == 0:
                    gt_delta = gt_batch[:, 0, :] - hist_last_np
                    gen_delta = rollout_np[:, :, 0, :] - hist_last_np[:, None, :]
                else:
                    gt_delta = gt_batch[:, hidx, :] - gt_batch[:, hidx - 1, :]
                    gen_delta = rollout_np[:, :, hidx, :] - rollout_np[:, :, hidx - 1, :]
                gen_flat = gen_delta.reshape(-1, n_cells)
                gt_delta_sum[j] += gt_delta.sum(axis=0)
                gt_delta_outer[j] += gt_delta.T @ gt_delta
                gt_delta_count[j] += gt_delta.shape[0]
                ro_delta_sum[j] += gen_flat.sum(axis=0)
                ro_delta_outer[j] += gen_flat.T @ gen_flat
                ro_delta_count[j] += gen_flat.shape[0]

            row0 = row1

    target = future_01_all.detach().cpu().numpy()
    tf_mean = np.concatenate(tf_mean_chunks, axis=0)
    tf_std = np.concatenate(tf_std_chunks, axis=0)
    tf_lo = np.concatenate(tf_lo_chunks, axis=0)
    tf_hi = np.concatenate(tf_hi_chunks, axis=0)
    tf_scale = np.concatenate(tf_scale_chunks, axis=0)
    tf_shape_var = np.concatenate(tf_shape_var_chunks, axis=0)
    ro_mean = np.concatenate(ro_mean_chunks, axis=0)
    ro_std = np.concatenate(ro_std_chunks, axis=0)
    ro_lo = np.concatenate(ro_lo_chunks, axis=0)
    ro_hi = np.concatenate(ro_hi_chunks, axis=0)

    def by_horizon_summary():
        rows = []
        for hidx, horizon in zip(SELECT_HIDX, SELECT_HORIZONS):
            tf_cov = ((target[:, hidx] >= tf_lo[:, hidx]) & (target[:, hidx] <= tf_hi[:, hidx])).mean()
            ro_cov = ((target[:, hidx] >= ro_lo[:, hidx]) & (target[:, hidx] <= ro_hi[:, hidx])).mean()
            rows.append({
                "horizon": horizon,
                "teacher_forced_mae": float(np.abs(tf_mean[:, hidx] - target[:, hidx]).mean()),
                "rollout_mae": float(np.abs(ro_mean[:, hidx] - target[:, hidx]).mean()),
                "mae_gap": float(np.abs(ro_mean[:, hidx] - target[:, hidx]).mean() - np.abs(tf_mean[:, hidx] - target[:, hidx]).mean()),
                "teacher_forced_cov90": float(tf_cov),
                "rollout_cov90": float(ro_cov),
                "teacher_forced_width90": float((tf_hi[:, hidx] - tf_lo[:, hidx]).mean()),
                "rollout_width90": float((ro_hi[:, hidx] - ro_lo[:, hidx]).mean()),
            })
        return rows

    def top_rollout_gap_cells(hidx: int = 29, topk: int = 5):
        gap = np.abs(ro_mean[:, hidx] - target[:, hidx]).mean(axis=0) - np.abs(tf_mean[:, hidx] - target[:, hidx]).mean(axis=0)
        order = np.argsort(gap)[::-1][:topk]
        out = []
        for idx in order:
            out.append({
                "cell": [int(idx // 5), int(idx % 5)],
                "mae_gap": float(gap[idx]),
                "teacher_forced_mae": float(np.abs(tf_mean[:, hidx, idx] - target[:, hidx, idx]).mean()),
                "rollout_mae": float(np.abs(ro_mean[:, hidx, idx] - target[:, hidx, idx]).mean()),
                "teacher_forced_cov90": float(((target[:, hidx, idx] >= tf_lo[:, hidx, idx]) & (target[:, hidx, idx] <= tf_hi[:, hidx, idx])).mean()),
                "rollout_cov90": float(((target[:, hidx, idx] >= ro_lo[:, hidx, idx]) & (target[:, hidx, idx] <= ro_hi[:, hidx, idx])).mean()),
            })
        return out

    def mean_variance_decomp(pred_mean: np.ndarray, pred_std: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, Any]:
        eps = 1e-6
        out = {}
        for regime_name, mask in masks.items():
            regime_rows = []
            if not mask.any():
                out[regime_name] = regime_rows
                continue
            for hidx, horizon in zip(SELECT_HIDX, SELECT_HORIZONS):
                err = target[mask, hidx] - pred_mean[mask, hidx]
                z = err / np.clip(pred_std[mask, hidx], eps, None)
                below = (target[mask, hidx] < lo[mask, hidx]).mean()
                above = (target[mask, hidx] > hi[mask, hidx]).mean()
                regime_rows.append({
                    "horizon": horizon,
                    "coverage90": float(((target[mask, hidx] >= lo[mask, hidx]) & (target[mask, hidx] <= hi[mask, hidx])).mean()),
                    "lower_miss_rate": float(below),
                    "upper_miss_rate": float(above),
                    "mean_signed_error": float(err.mean()),
                    "mean_abs_error": float(np.abs(err).mean()),
                    "pred_std_mean": float(pred_std[mask, hidx].mean()),
                    "resid_rmse": float(np.sqrt(np.mean(err ** 2))),
                    "mean_signed_z": float(z.mean()),
                    "mean_abs_z": float(np.abs(z).mean()),
                })
            out[regime_name] = regime_rows
        h30 = 29
        cov_h30 = ((target[:, h30] >= lo[:, h30]) & (target[:, h30] <= hi[:, h30])).mean(axis=0)
        under_idx = int(np.argmin(cov_h30))
        over_idx = int(np.argmax(cov_h30))
        extremes = {}
        for label, idx in [("worst_undercovered_h30", under_idx), ("most_overcovered_h30", over_idx)]:
            err = target[:, h30, idx] - pred_mean[:, h30, idx]
            z = err / np.clip(pred_std[:, h30, idx], eps, None)
            extremes[label] = {
                "cell": [int(idx // 5), int(idx % 5)],
                "coverage90": float(cov_h30[idx]),
                "lower_miss_rate": float((target[:, h30, idx] < lo[:, h30, idx]).mean()),
                "upper_miss_rate": float((target[:, h30, idx] > hi[:, h30, idx]).mean()),
                "mean_signed_error": float(err.mean()),
                "mean_abs_error": float(np.abs(err).mean()),
                "pred_std_mean": float(pred_std[:, h30, idx].mean()),
                "resid_rmse": float(np.sqrt(np.mean(err ** 2))),
                "mean_signed_z": float(z.mean()),
                "mean_abs_z": float(np.abs(z).mean()),
            }
        out["h30_extremes"] = extremes
        return out

    def covariance_regime_summary() -> dict[str, Any]:
        out = {}
        for regime_name, mask in masks.items():
            regime_out = []
            if not mask.any():
                out[regime_name] = regime_out
                continue
            for hidx, horizon in zip(SELECT_HIDX, SELECT_HORIZONS):
                if pred_cov_count[regime_name][hidx] == 0:
                    continue
                pred_cov = pred_cov_sum[regime_name][hidx] / pred_cov_count[regime_name][hidx]
                resid = residuals_tf[mask, hidx]
                emp_cov = np.cov(resid, rowvar=False, bias=True)
                regime_out.append({
                    "horizon": horizon,
                    "pred_mean_corr": mean_offdiag_corr(pred_cov),
                    "emp_mean_corr": mean_offdiag_corr(emp_cov),
                    "pred_eff_rank": eff_rank_from_matrix(pred_cov),
                    "emp_eff_rank": eff_rank_from_matrix(emp_cov),
                    "pred_trace": float(np.trace(pred_cov)),
                    "emp_trace": float(np.trace(emp_cov)),
                })
            out[regime_name] = regime_out
        return out

    def regime_width_response(pred_std: np.ndarray) -> list[dict[str, float]]:
        rows = []
        for hidx, horizon in zip(SELECT_HIDX, SELECT_HORIZONS):
            calm = pred_std[calm_mask, hidx].mean() if calm_mask.any() else np.nan
            turb = pred_std[turb_mask, hidx].mean() if turb_mask.any() else np.nan
            rows.append({
                "horizon": horizon,
                "calm_pred_std": float(calm),
                "turb_pred_std": float(turb),
                "turb_calm_ratio": float(turb / calm) if calm and np.isfinite(calm) else float("nan"),
            })
        return rows

    def spectrum_by_horizon() -> list[dict[str, float]]:
        rows = []
        for j, horizon in enumerate(SELECT_HORIZONS):
            gt_mean = gt_delta_sum[j] / max(gt_delta_count[j], 1)
            gt_cov = gt_delta_outer[j] / max(gt_delta_count[j], 1) - np.outer(gt_mean, gt_mean)
            gen_mean = ro_delta_sum[j] / max(ro_delta_count[j], 1)
            gen_cov = ro_delta_outer[j] / max(ro_delta_count[j], 1) - np.outer(gen_mean, gen_mean)
            gt_corr = mean_offdiag_corr(gt_cov)
            gen_corr = mean_offdiag_corr(gen_cov)
            gt_rank = eff_rank_from_matrix(corr_from_cov(gt_cov))
            gen_rank = eff_rank_from_matrix(corr_from_cov(gen_cov))
            rows.append({
                "horizon": horizon,
                "gt_mean_corr": gt_corr,
                "gen_mean_corr": gen_corr,
                "corr_ratio": float(gen_corr / gt_corr) if abs(gt_corr) > 1e-8 else float("nan"),
                "gt_eff_rank": gt_rank,
                "gen_eff_rank": gen_rank,
                "rank_ratio": float(gen_rank / gt_rank) if gt_rank > 1e-8 else float("nan"),
            })
        return rows

    out = {
        "model_type": model_type,
        "n_windows": int(n_windows),
        "teacher_forced_vs_rollout": by_horizon_summary(),
        "h30_top_rollout_gap_cells": top_rollout_gap_cells(),
        "teacher_forced_mean_variance": mean_variance_decomp(tf_mean, tf_std, tf_lo, tf_hi),
        "rollout_mean_variance": mean_variance_decomp(ro_mean, ro_std, ro_lo, ro_hi),
        "teacher_forced_covariance_vs_empirical": covariance_regime_summary(),
        "teacher_forced_regime_width_response": regime_width_response(tf_std),
        "rollout_regime_width_response": regime_width_response(ro_std),
        "cross_cell_spectrum_by_horizon": spectrum_by_horizon(),
        "scale_summary": {
            "teacher_forced_scale_mean_by_horizon": [
                {
                    "horizon": horizon,
                    "all": float(tf_scale[:, hidx].mean()),
                    "calm": float(tf_scale[calm_mask, hidx].mean()) if calm_mask.any() else float("nan"),
                    "turb": float(tf_scale[turb_mask, hidx].mean()) if turb_mask.any() else float("nan"),
                }
                for hidx, horizon in zip(SELECT_HIDX, SELECT_HORIZONS)
            ],
            "teacher_forced_shape_var_mean_by_horizon": [
                {
                    "horizon": horizon,
                    "all": float(tf_shape_var[:, hidx].mean()),
                    "calm": float(tf_shape_var[calm_mask, hidx].mean()) if calm_mask.any() else float("nan"),
                    "turb": float(tf_shape_var[turb_mask, hidx].mean()) if turb_mask.any() else float("nan"),
                }
                for hidx, horizon in zip(SELECT_HIDX, SELECT_HORIZONS)
            ],
        },
    }

    if z_stats_store is not None:
        white = np.concatenate(z_stats_store["white"], axis=0)
        z = np.concatenate(z_stats_store["z"], axis=0)
        out["whitened_residual_diagnostics"] = {
            regime_name: {
                str(horizon): {
                    "white": summarize_z(white[masks[regime_name], hidx]),
                    "z": summarize_z(z[masks[regime_name], hidx]),
                }
                for hidx, horizon in zip(SELECT_HIDX, SELECT_HORIZONS)
                if masks[regime_name].any()
            }
            for regime_name in REGIME_NAMES
        }

    return out


def main():
    parser = argparse.ArgumentParser(description="Mechanistic comparison for H7 density models")
    parser.add_argument("--model_169c", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--model_170b", type=str, default="models/backfill/whitened_flow_170b/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--teacher_forced_samples", type=int, default=32)
    parser.add_argument("--rollout_samples", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-04/analysis/h7_mechanistic",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_169c, type_169c, ckpt_169c = load_model(args.model_169c, args.device)
    model_170b, type_170b, ckpt_170b = load_model(args.model_170b, args.device)

    hist_len = ckpt_169c["config"]["history_len"]
    fut_len = ckpt_169c["config"]["future_len"]
    history_norm, future_norm = build_test_subset(
        args.data_path,
        history_len=hist_len,
        future_len=fut_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    vov, q20, q80 = regime_masks_from_history(history_norm)

    result = {
        "config": {
            "test_start": args.test_start,
            "max_windows": history_norm.shape[0],
            "teacher_forced_samples": args.teacher_forced_samples,
            "rollout_samples": args.rollout_samples,
            "batch_size": args.batch_size,
            "q20_vov": q20,
            "q80_vov": q80,
            "calm_windows": int((vov <= q20).sum()),
            "turb_windows": int((vov >= q80).sum()),
        }
    }

    print(f"Analyzing 169c on {history_norm.shape[0]} test windows...")
    result["model_169c"] = analyze_model(
        model_169c,
        type_169c,
        history_norm,
        future_norm,
        vov,
        q20,
        q80,
        device=args.device,
        batch_size=args.batch_size,
        tf_samples=args.teacher_forced_samples,
        rollout_samples=args.rollout_samples,
    )

    print(f"Analyzing 170b on {history_norm.shape[0]} test windows...")
    result["model_170b"] = analyze_model(
        model_170b,
        type_170b,
        history_norm,
        future_norm,
        vov,
        q20,
        q80,
        device=args.device,
        batch_size=args.batch_size,
        tf_samples=args.teacher_forced_samples,
        rollout_samples=args.rollout_samples,
    )

    with open(output_dir / "mechanistic_summary.json", "w") as f:
        json.dump(make_serializable(result), f, indent=2)
    print(f"Saved to {output_dir / 'mechanistic_summary.json'}")


if __name__ == "__main__":
    main()
