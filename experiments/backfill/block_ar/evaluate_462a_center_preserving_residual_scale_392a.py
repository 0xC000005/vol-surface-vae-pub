#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    HistoryFutureDictDataset,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (
    suite_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


def _generate_native_samples(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist_norm = normalize_iv(history_01[start:end])
        with torch.no_grad():
            samples = model.sample_batched(
                hist_norm,
                n_samples=n_samples,
                n_steps=n_steps,
                chunk_size=chunk_size,
                history_is_normalized=True,
            )
        outputs.append(samples.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _build_recent_prevalidation_windows(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    calibration_windows: int,
    device: torch.device,
):
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    calib_end = max_train_idx - val_size
    calib_start = max(0, calib_end - calibration_windows)
    indices = np.arange(calib_start, calib_end)
    history_01, future_01 = build_multistep_windows(
        indices,
        surf_tensor,
        history_len,
        future_len,
    )
    future_01 = future_01.view(history_01.shape[0], future_len, 5, 5)
    return history_01, future_01, int(calib_start), int(calib_end)


def fit_center_residual_scales(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    candidates: np.ndarray,
    target_coverage: float,
    scale_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    center = np.median(samples, axis=1)
    lower = np.quantile(samples, 0.05, axis=1)
    upper = np.quantile(samples, 0.95, axis=1)
    horizon, height, width = ground_truth.shape[1:]
    scales = np.ones((horizon, height, width), dtype=np.float32)
    cov_before = np.zeros((horizon, height, width), dtype=np.float32)
    cov_after = np.zeros((horizon, height, width), dtype=np.float32)

    for t in range(horizon):
        for i in range(height):
            for j in range(width):
                gt = ground_truth[:, t, i, j]
                c = center[:, t, i, j]
                lo0 = lower[:, t, i, j]
                hi0 = upper[:, t, i, j]
                before = ((gt >= lo0) & (gt <= hi0)).mean()
                best_scale = 1.0
                best_score = float("inf")
                best_cov = before
                for scale in candidates:
                    lo = c + scale * (lo0 - c)
                    hi = c + scale * (hi0 - c)
                    cov = ((gt >= lo) & (gt <= hi)).mean()
                    score = (cov - target_coverage) ** 2
                    score += scale_penalty * (scale - 1.0) ** 2
                    if score < best_score:
                        best_score = float(score)
                        best_scale = float(scale)
                        best_cov = float(cov)
                scales[t, i, j] = best_scale
                cov_before[t, i, j] = before
                cov_after[t, i, j] = best_cov

    fit_summary = {
        "target_coverage": float(target_coverage),
        "scale_penalty": float(scale_penalty),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_mean": float(scales.mean()),
        "coverage_before_mean": float(cov_before.mean()),
        "coverage_after_mean": float(cov_after.mean()),
        "coverage_before_min": float(cov_before.min()),
        "coverage_after_min": float(cov_after.min()),
        "coverage_before_max": float(cov_before.max()),
        "coverage_after_max": float(cov_after.max()),
    }
    return scales, fit_summary


def _history_variance(history_01: np.ndarray) -> np.ndarray:
    dhist = np.diff(history_01, axis=1)
    return (dhist**2).mean(axis=(1, 2, 3))


def _assign_bins(history_01: np.ndarray, thresholds: tuple[float, float] | None) -> np.ndarray:
    if thresholds is None:
        return np.zeros(history_01.shape[0], dtype=np.int64)
    vov = _history_variance(history_01)
    lo, hi = thresholds
    return np.where(vov <= lo, 0, np.where(vov >= hi, 2, 1)).astype(np.int64)


def fit_binned_center_residual_scales(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    candidates: np.ndarray,
    target_coverage: float,
    scale_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    global_scales, global_summary = fit_center_residual_scales(
        samples,
        ground_truth,
        candidates=candidates,
        target_coverage=target_coverage,
        scale_penalty=scale_penalty,
    )
    scales = np.repeat(global_scales[None], n_bins, axis=0)
    bin_summaries: list[dict[str, Any]] = []
    for bin_idx in range(n_bins):
        mask = bins == bin_idx
        if int(mask.sum()) < 16:
            bin_summaries.append(
                {
                    "bin": bin_idx,
                    "n_windows": int(mask.sum()),
                    "fallback": "global",
                }
            )
            continue
        bin_scales, bin_summary = fit_center_residual_scales(
            samples[mask],
            ground_truth[mask],
            candidates=candidates,
            target_coverage=target_coverage,
            scale_penalty=scale_penalty,
        )
        scales[bin_idx] = bin_scales
        bin_summary = dict(bin_summary)
        bin_summary.update({"bin": bin_idx, "n_windows": int(mask.sum())})
        bin_summaries.append(bin_summary)
    fit_summary = {
        "target_coverage": float(target_coverage),
        "scale_penalty": float(scale_penalty),
        "n_bins": int(n_bins),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_mean": float(scales.mean()),
        "global_summary": global_summary,
        "bin_summaries": bin_summaries,
    }
    return scales.astype(np.float32), fit_summary


def fit_binned_abs_residual_quantiles(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    quantile_levels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    center = np.median(samples, axis=1)
    gen_abs = np.abs(samples - center[:, None])
    real_abs = np.abs(ground_truth - center)
    horizon, height, width = ground_truth.shape[1:]
    gen_q = np.zeros((n_bins, horizon, height, width, len(quantile_levels)), dtype=np.float32)
    real_q = np.zeros_like(gen_q)
    bin_summaries: list[dict[str, Any]] = []
    for bin_idx in range(n_bins):
        mask = bins == bin_idx
        if int(mask.sum()) < 16:
            mask = np.ones_like(bins, dtype=bool)
            fallback = "global"
        else:
            fallback = None
        for t in range(horizon):
            for i in range(height):
                for j in range(width):
                    gen_vals = gen_abs[mask, :, t, i, j].reshape(-1)
                    real_vals = real_abs[mask, t, i, j].reshape(-1)
                    gq = np.maximum.accumulate(np.quantile(gen_vals, quantile_levels))
                    rq = np.maximum.accumulate(np.quantile(real_vals, quantile_levels))
                    gq = gq + np.arange(len(gq), dtype=np.float32) * 1e-8
                    gen_q[bin_idx, t, i, j] = gq
                    real_q[bin_idx, t, i, j] = rq
        bin_summaries.append(
            {
                "bin": bin_idx,
                "n_windows": int((bins == bin_idx).sum()),
                "fallback": fallback,
                "gen_abs_q90_mean": float(gen_q[bin_idx, ..., int(0.9 * (len(quantile_levels) - 1))].mean()),
                "real_abs_q90_mean": float(real_q[bin_idx, ..., int(0.9 * (len(quantile_levels) - 1))].mean()),
            }
        )
    fit_summary = {
        "calibration_mode": "abs_residual_quantile",
        "n_bins": int(n_bins),
        "n_quantiles": int(len(quantile_levels)),
        "gen_abs_q90_mean": float(gen_q[..., int(0.9 * (len(quantile_levels) - 1))].mean()),
        "real_abs_q90_mean": float(real_q[..., int(0.9 * (len(quantile_levels) - 1))].mean()),
        "bin_summaries": bin_summaries,
    }
    return {"gen_q": gen_q, "real_q": real_q, "levels": quantile_levels.astype(np.float32)}, fit_summary


def apply_center_residual_scales(
    samples: np.ndarray,
    scales: np.ndarray,
    bins: np.ndarray | None = None,
) -> np.ndarray:
    center = np.median(samples, axis=1, keepdims=True)
    if scales.ndim == 3:
        scale_arr = scales[None, None]
    else:
        if bins is None:
            raise ValueError("bins are required for binned scales")
        scale_arr = scales[bins][:, None]
    scaled = center + scale_arr * (samples - center)
    return np.clip(scaled, 0.0, 1.0)


def apply_abs_residual_quantiles(
    samples: np.ndarray,
    quantile_map: dict[str, np.ndarray],
    bins: np.ndarray,
) -> np.ndarray:
    center = np.median(samples, axis=1, keepdims=True)
    residual = samples - center
    sign = np.sign(residual)
    magnitude = np.abs(residual)
    mapped = np.empty_like(magnitude)
    gen_q = quantile_map["gen_q"]
    real_q = quantile_map["real_q"]
    n_bins = gen_q.shape[0]
    for bin_idx in range(n_bins):
        row_mask = bins == bin_idx
        if not np.any(row_mask):
            continue
        for t in range(samples.shape[2]):
            for i in range(samples.shape[3]):
                for j in range(samples.shape[4]):
                    x = magnitude[row_mask, :, t, i, j].reshape(-1)
                    xp = gen_q[bin_idx, t, i, j]
                    fp = real_q[bin_idx, t, i, j]
                    mapped_vals = np.interp(x, xp, fp, left=fp[0], right=fp[-1])
                    mapped[row_mask, :, t, i, j] = mapped_vals.reshape(
                        magnitude[row_mask, :, t, i, j].shape
                    )
    calibrated = center + sign * mapped
    return np.clip(calibrated, 0.0, 1.0)


class CenterResidualScaleWrapper:
    def __init__(
        self,
        base_model: torch.nn.Module,
        scales: np.ndarray,
        thresholds: tuple[float, float] | None = None,
        quantile_map: dict[str, np.ndarray] | None = None,
    ):
        self.base_model = base_model
        self.scales_np = scales.astype(np.float32)
        self.thresholds = thresholds
        self.quantile_map = quantile_map
        self.scales: torch.Tensor | None = None

    def eval(self):
        self.base_model.eval()
        return self

    def train(self, mode: bool = True):
        self.base_model.train(mode)
        return self

    def _scales_for(self, samples: torch.Tensor) -> torch.Tensor:
        if self.scales is None or self.scales.device != samples.device:
            self.scales = torch.from_numpy(self.scales_np).to(samples.device)
        return self.scales.to(dtype=samples.dtype)

    def _bins_for_history(self, history: torch.Tensor, history_is_normalized: bool) -> torch.Tensor:
        if self.thresholds is None:
            return torch.zeros(history.shape[0], device=history.device, dtype=torch.long)
        history_01 = denormalize_iv(history) if history_is_normalized else history
        dhist = history_01[:, 1:] - history_01[:, :-1]
        vov = dhist.square().mean(dim=(1, 2, 3))
        lo, hi = self.thresholds
        return torch.where(
            vov <= lo,
            torch.zeros_like(vov, dtype=torch.long),
            torch.where(vov >= hi, torch.full_like(vov, 2, dtype=torch.long), torch.ones_like(vov, dtype=torch.long)),
        )

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        samples = self.base_model.sample_batched(
            history,
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=history_is_normalized,
            **kwargs,
        )
        if self.quantile_map is not None:
            history_01 = denormalize_iv(history) if history_is_normalized else history
            bins = _assign_bins(history_01.detach().cpu().numpy(), self.thresholds)
            calibrated = apply_abs_residual_quantiles(
                samples.detach().cpu().numpy(),
                self.quantile_map,
                bins=bins,
            )
            return torch.from_numpy(calibrated).to(device=samples.device, dtype=samples.dtype)

        scales = self._scales_for(samples)[: samples.shape[2]]
        center = samples.median(dim=1, keepdim=True).values
        if scales.ndim == 3:
            scale_arr = scales[None, None]
        else:
            bins = self._bins_for_history(history, history_is_normalized)
            scale_arr = scales[bins, : samples.shape[2]][:, None]
        return (center + scale_arr * (samples - center)).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="462a center-preserving residual scale calibration for 392a"
    )
    parser.add_argument("--base_model_type", type=str, default="340c")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--calibration_windows", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--calibration_samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--target_coverage", type=float, default=0.88)
    parser.add_argument("--min_scale", type=float, default=0.65)
    parser.add_argument("--max_scale", type=float, default=1.35)
    parser.add_argument("--n_scale_candidates", type=int, default=29)
    parser.add_argument("--scale_penalty", type=float, default=0.002)
    parser.add_argument(
        "--calibration_mode",
        choices=["scale", "abs_quantile"],
        default="scale",
    )
    parser.add_argument("--n_abs_quantiles", type=int, default=101)
    parser.add_argument(
        "--regime_bins",
        choices=["none", "history_vov_q20_q80"],
        default="none",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--scale_map_json", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, payload = load_one_day_kernel(args.base_model_type, args.checkpoint, device)
    base_model.eval()

    calib_hist, calib_future, calib_start, calib_end = _build_recent_prevalidation_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        calibration_windows=args.calibration_windows,
        device=device,
    )
    calib_samples = _generate_native_samples(
        base_model,
        calib_hist,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    candidates = np.linspace(args.min_scale, args.max_scale, args.n_scale_candidates)
    calib_history_np = calib_hist.detach().cpu().numpy()
    thresholds = None
    if args.regime_bins == "history_vov_q20_q80":
        calib_vov = _history_variance(calib_history_np)
        thresholds = (float(np.quantile(calib_vov, 0.2)), float(np.quantile(calib_vov, 0.8)))
        calib_bins = _assign_bins(calib_history_np, thresholds)
        n_bins = 3
    else:
        calib_bins = np.zeros(calib_history_np.shape[0], dtype=np.int64)
        n_bins = 1

    quantile_map = None
    if args.calibration_mode == "abs_quantile":
        quantile_levels = np.linspace(
            0.005,
            0.995,
            args.n_abs_quantiles,
            dtype=np.float32,
        )
        quantile_map, fit_summary = fit_binned_abs_residual_quantiles(
            calib_samples,
            calib_future.detach().cpu().numpy(),
            bins=calib_bins,
            n_bins=n_bins,
            quantile_levels=quantile_levels,
        )
        scales = np.ones((n_bins, args.future_len, 5, 5), dtype=np.float32)
    elif n_bins > 1:
        scales, fit_summary = fit_binned_center_residual_scales(
            calib_samples,
            calib_future.detach().cpu().numpy(),
            bins=calib_bins,
            n_bins=n_bins,
            candidates=candidates,
            target_coverage=args.target_coverage,
            scale_penalty=args.scale_penalty,
        )
    else:
        scales, fit_summary = fit_center_residual_scales(
            calib_samples,
            calib_future.detach().cpu().numpy(),
            candidates=candidates,
            target_coverage=args.target_coverage,
            scale_penalty=args.scale_penalty,
        )

    scale_path = Path(args.scale_map_json)
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    scale_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "checkpoint_epoch": int(payload.get("epoch", -1)),
                "calibration_start": calib_start,
                "calibration_end": calib_end,
                "regime_thresholds": list(thresholds) if thresholds is not None else None,
                "fit_summary": fit_summary,
                "scales": scales.tolist(),
                "quantile_map": {
                    key: value.tolist() for key, value in quantile_map.items()
                } if quantile_map is not None else None,
            },
            indent=2,
        )
    )

    wrapper = CenterResidualScaleWrapper(
        base_model,
        scales,
        thresholds=thresholds,
        quantile_map=quantile_map,
    ).eval()
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    val_base_samples = _generate_native_samples(
        base_model,
        batch.history_01,
        n_samples=args.samples,
        n_steps=batch.future_01.shape[1],
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    val_bins = _assign_bins(batch.history_01.detach().cpu().numpy(), thresholds)
    if quantile_map is not None:
        cond_samples = apply_abs_residual_quantiles(
            val_base_samples,
            quantile_map,
            bins=val_bins,
        )
    else:
        cond_samples = apply_center_residual_scales(val_base_samples, scales, bins=val_bins)
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    conditionality = run_conditionality_tests(
        wrapper,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    cointegration = run_cointegration_tests(
        cond_samples,
        ground_truth,
        returns=returns,
        test_start=rollout_start,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    regime_coverage = run_regime_coverage_tests(cond_samples, ground_truth, history_01)
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    results = {
        "config": {
            "model_type": "462a_center_residual_scale_392a",
            "base_model_type": args.base_model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
            "calibration_windows": int(calib_hist.shape[0]),
            "calibration_start": calib_start,
            "calibration_end": calib_end,
            "fit_summary": fit_summary,
            "regime_thresholds": list(thresholds) if thresholds is not None else None,
            "calibration_mode": args.calibration_mode,
            "rollout_start": int(rollout_start),
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "time_series": time_series,
        "block_ar": block_ar,
        "cointegration": cointegration,
        "regime_coverage": regime_coverage,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {
        "n_pass": n_pass,
        "n_total": 11,
        "failed_suites": failed,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    fit_cov_before = fit_summary.get(
        "coverage_before_mean",
        fit_summary.get("global_summary", {}).get("coverage_before_mean", float("nan")),
    )
    fit_cov_after = fit_summary.get(
        "coverage_after_mean",
        fit_summary.get("global_summary", {}).get("coverage_after_mean", float("nan")),
    )
    lines = [
        "- model: `462a_center_residual_scale_392a`",
        f"- base checkpoint: `{args.checkpoint}`",
        f"- calibration windows: `{calib_hist.shape[0]}` recent pre-validation",
        f"- validation windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Calibration Fit**",
        f"- mode: `{args.calibration_mode}`",
        f"- scale range: `{fit_summary.get('scale_min', float('nan')):.3f}` to `{fit_summary.get('scale_max', float('nan')):.3f}`",
        f"- scale mean: `{fit_summary.get('scale_mean', float('nan')):.3f}`",
        f"- calibration coverage mean: `{fit_cov_before:.3f}` -> `{fit_cov_after:.3f}`",
        "",
        "**Key Metrics**",
        f"- coverage90: `{coverage['overall'][0.9]:.3f}`",
        f"- conditionality MAE reduction: `{conditionality['mae_reduction_pct']:.2f}%`",
        f"- regime layer2: `{regime_coverage['layer2_n_passing']}/{regime_coverage['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- mean-reversion active pass rate: `{mean_reversion['full_horizon']['mean_active_pass_rate']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "462a Center-Preserving Residual Scale Calibration", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
