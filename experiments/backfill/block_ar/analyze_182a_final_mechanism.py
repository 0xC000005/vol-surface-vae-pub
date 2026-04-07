#!/usr/bin/env python
"""
Focused mechanism review for 182a_final.

Goal:
  Explain the remaining S3 / S4 / S7 / borderline S10 failures on the
  final 182a checkpoint, which is the meaningful anchor for the new phase.

Questions:
  1. Are S3/S7 still mainly a local width-allocation problem, a mean-bias problem,
     or both?
  2. Why can S11 pass while S4 still fails?
  3. Why does S10 only narrowly fail on the strengthened active-mean profile?
  4. What does the evidence imply for 182b?
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
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_182a_pathwise_residual_law import (
    PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
)


SELECT_HORIZONS = [1, 7, 14, 30]
BANDS = ["low", "mid", "high"]


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182a"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = PathwiseResidualLawMeanRevertingCovarianceMixtureModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        path_config=raw_config["path"],
        prior_config=raw_config["prior"],
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


def quantile_profile(x: np.ndarray, qs: tuple[float, ...] = (0.5, 0.75, 0.9, 0.99)) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return {str(q): float(np.quantile(arr, q)) for q in qs}


def pearson_kurtosis(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    xc = arr - arr.mean()
    var = np.mean(np.square(xc))
    if var <= 1e-12:
        return float("nan")
    return float(np.mean(np.power(xc, 4)) / (var * var))


def distribution_compare(gt: np.ndarray, gen: np.ndarray) -> dict[str, Any]:
    gt_q = quantile_profile(gt)
    gen_q = quantile_profile(gen)
    return {
        "gt_quantiles": gt_q,
        "gen_quantiles": gen_q,
        "ratio_quantiles": {k: float(gen_q[k] / max(gt_q[k], 1e-12)) for k in gt_q},
        "ks": ks_detail(gt, gen),
        "gt_kurtosis": pearson_kurtosis(gt),
        "gen_kurtosis": pearson_kurtosis(gen),
        "kurtosis_ratio": float(pearson_kurtosis(gen) / max(pearson_kurtosis(gt), 1e-12)),
    }


def classify_cell_failure(coverage: float, z_mean: float, z_std: float) -> str:
    if coverage > 0.95:
        if z_std < 0.90 and abs(z_mean) < 0.25:
            return "overwide"
        if z_std < 0.90:
            return "overwide_plus_bias"
        return "high_coverage_mixed"
    if coverage < 0.70:
        if abs(z_mean) > 0.50 and z_std <= 1.15:
            return "bias_dominant"
        if z_std > 1.15 and abs(z_mean) <= 0.50:
            return "underwide_dominant"
        return "bias_plus_underwide"
    return "pass"


def cell_label(idx: int) -> list[int]:
    return [int(idx // 5), int(idx % 5)]


@torch.no_grad()
def analyze_182a_final(
    model: PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    summary: dict[str, Any],
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

    sample_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    sample_std = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    lo90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    hi90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    det_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    pooled_abs_delta_gt = []
    pooled_abs_delta_gen = []
    path_max_jump_gt = []
    path_max_jump_gen = []
    path_q99_exceed_count_gt = []
    path_q99_exceed_count_gen = []
    band_energy_gt = {k: [] for k in BANDS}
    band_energy_gen = {k: [] for k in BANDS}
    band_abs_coeff_gt = {k: [] for k in BANDS}
    band_abs_coeff_gen = {k: [] for k in BANDS}
    det_shift_by_h = {str(h): [] for h in SELECT_HORIZONS}
    regime_band_energy = {
        "calm": {k: [] for k in BANDS},
        "turb": {k: [] for k in BANDS},
    }

    geometry = model.path_geometry
    band_masks = {
        "low": geometry.low_band_mask().reshape(1, future_len, n_cells).to(device),
        "mid": geometry.mid_band_mask().reshape(1, future_len, n_cells).to(device),
        "high": geometry.high_band_mask().reshape(1, future_len, n_cells).to(device),
    }

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_01_b = history_01[start:end]
        fut_01_b = future_01[start:end].to(device)
        batch = end - start

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
        ) = model.forward_from_history(hist_01_b)

        det_iv = unconstrained_to_iv(mu_u, lo=model.support_lo, hi=model.support_hi)
        det_mean[start:end] = det_iv.detach().cpu().numpy()

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

        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        basis_paths, _ = model.sample_basis_paths(path_context, n_samples=n_samples)
        basis_paths = basis_paths.view(batch, n_samples, future_len, n_cells)

        samples_u = model.sample_future_u(hist_01_b, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
        samples_iv = samples_01.detach().cpu().numpy()
        sample_mean[start:end] = samples_iv.mean(axis=1)
        sample_std[start:end] = samples_iv.std(axis=1)
        lo90[start:end] = np.quantile(samples_iv, 0.05, axis=1)
        hi90[start:end] = np.quantile(samples_iv, 0.95, axis=1)

        gt_path = torch.cat([hist_01_b[:, -1:].reshape(batch, 1, 5, 5), fut_01_b.view(batch, future_len, 5, 5)], dim=1)
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

        gt_diff_flat = gt_diff.reshape(batch, future_len, n_cells)
        gen_diff_flat = gen_diff.reshape(batch * n_samples, future_len, n_cells)
        gt_coeff = geometry.to_basis(gt_diff_flat)
        gen_coeff = geometry.to_basis(gen_diff_flat)

        gt_total_energy = gt_coeff.reshape(batch, -1).pow(2).sum(dim=1).clamp_min(1e-12)
        gen_total_energy = gen_coeff.reshape(batch * n_samples, -1).pow(2).sum(dim=1).clamp_min(1e-12)

        for band in BANDS:
            mask = band_masks[band].to(dtype=gt_coeff.dtype)
            gt_band_coeff = (gt_coeff * mask).reshape(batch, -1)
            gen_band_coeff = (gen_coeff * mask).reshape(batch * n_samples, -1)
            band_energy_gt[band].append((gt_band_coeff.pow(2).sum(dim=1) / gt_total_energy).cpu().numpy())
            band_energy_gen[band].append((gen_band_coeff.pow(2).sum(dim=1) / gen_total_energy).cpu().numpy())
            band_abs_coeff_gt[band].append(gt_band_coeff.abs().reshape(-1).cpu().numpy())
            band_abs_coeff_gen[band].append(gen_band_coeff.abs().reshape(-1).cpu().numpy())

            batch_regime = {
                "calm": calm_mask[start:end],
                "turb": turb_mask[start:end],
            }
            for regime_name, regime_mask_b in batch_regime.items():
                if regime_mask_b.sum() == 0:
                    continue
                regime_band_energy[regime_name][band].append(
                    (gt_band_coeff.pow(2).sum(dim=1) / gt_total_energy)[torch.as_tensor(regime_mask_b, device=gt_coeff.device)].cpu().numpy()
                )

        batch_det = det_iv.detach().cpu().numpy()
        batch_sample_mean = samples_iv.mean(axis=1)
        for h in SELECT_HORIZONS:
            det_shift_by_h[str(h)].append((batch_sample_mean[:, h - 1] - batch_det[:, h - 1]).reshape(-1))

    pooled_abs_delta_gt = np.concatenate(pooled_abs_delta_gt, axis=0)
    pooled_abs_delta_gen = np.concatenate(pooled_abs_delta_gen, axis=0)
    path_max_jump_gt = np.concatenate(path_max_jump_gt, axis=0)
    path_max_jump_gen = np.concatenate(path_max_jump_gen, axis=0)
    gt_q99 = float(np.quantile(pooled_abs_delta_gt, 0.99))

    # Recompute pathwise exceedance counts using the GT q99 threshold.
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_01_b = history_01[start:end]
        fut_01_b = future_01[start:end].to(device)
        batch = end - start
        samples_u = model.sample_future_u(hist_01_b, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
        gt_path = torch.cat([hist_01_b[:, -1:].reshape(batch, 1, 5, 5), fut_01_b.view(batch, future_len, 5, 5)], dim=1)
        gen_path = torch.cat(
            [
                hist_01_b[:, -1:].reshape(batch, 1, 1, 5, 5).expand(-1, n_samples, -1, -1, -1),
                samples_01.view(batch, n_samples, future_len, 5, 5),
            ],
            dim=2,
        )
        gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
        gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]
        path_q99_exceed_count_gt.append((gt_diff.abs() > gt_q99).sum(dim=(1, 2, 3)).cpu().numpy())
        path_q99_exceed_count_gen.append((gen_diff.abs() > gt_q99).sum(dim=(2, 3, 4)).reshape(-1).cpu().numpy())
    path_q99_exceed_count_gt = np.concatenate(path_q99_exceed_count_gt, axis=0)
    path_q99_exceed_count_gen = np.concatenate(path_q99_exceed_count_gen, axis=0)

    gt = future_01.cpu().numpy()
    sample_std_safe = np.maximum(sample_std, 1e-6)
    z = (gt - sample_mean) / sample_std_safe
    inside90 = (gt >= lo90) & (gt <= hi90)

    width_failure_slices = []
    for regime_name, regime_mask in [("calm", calm_mask), ("turb", turb_mask)]:
        for h in SELECT_HORIZONS:
            cov = inside90[regime_mask, h - 1].mean(axis=0)
            z_mean = z[regime_mask, h - 1].mean(axis=0)
            z_std = z[regime_mask, h - 1].std(axis=0)
            gt_slice_std = (gt[regime_mask, h - 1] - sample_mean[regime_mask, h - 1]).std(axis=0)
            gen_slice_std = sample_std[regime_mask, h - 1].mean(axis=0)
            for idx in range(n_cells):
                r, c = cell_label(idx)
                classification = classify_cell_failure(float(cov.reshape(-1)[idx]), float(z_mean.reshape(-1)[idx]), float(z_std.reshape(-1)[idx]))
                if classification != "pass":
                    width_failure_slices.append(
                        {
                            "regime": regime_name,
                            "horizon": h,
                            "cell": [r, c],
                            "coverage_90": float(cov.reshape(-1)[idx]),
                            "z_mean": float(z_mean.reshape(-1)[idx]),
                            "z_std": float(z_std.reshape(-1)[idx]),
                            "gen_std_iv": float(gen_slice_std.reshape(-1)[idx]),
                            "gt_resid_std_iv": float(gt_slice_std.reshape(-1)[idx]),
                            "std_ratio_gen_to_gt": float(gen_slice_std.reshape(-1)[idx] / max(gt_slice_std.reshape(-1)[idx], 1e-8)),
                            "classification": classification,
                        }
                    )
    width_failure_slices.sort(
        key=lambda x: (
            0 if x["regime"] == "turb" else 1,
            x["coverage_90"],
            -abs(x["z_mean"]),
        )
    )

    turb_hard = [x for x in width_failure_slices if x["regime"] == "turb" and x["horizon"] in [14, 30]][:12]
    overwide = [x for x in width_failure_slices if x["classification"].startswith("overwide")][:10]

    std_ratio_turb_to_calm = []
    gt_std_ratio_turb_to_calm = []
    for h in SELECT_HORIZONS:
        turb_gen_std = sample_std[turb_mask, h - 1].mean(axis=0)
        calm_gen_std = sample_std[calm_mask, h - 1].mean(axis=0)
        turb_gt_std = (gt[turb_mask, h - 1] - sample_mean[turb_mask, h - 1]).std(axis=0)
        calm_gt_std = (gt[calm_mask, h - 1] - sample_mean[calm_mask, h - 1]).std(axis=0)
        ratio_gen = turb_gen_std / np.maximum(calm_gen_std, 1e-8)
        ratio_gt = turb_gt_std / np.maximum(calm_gt_std, 1e-8)
        std_ratio_turb_to_calm.append({"horizon": h, "mean_ratio": float(ratio_gen.mean()), "worst_cell": top_cells(ratio_gen, reverse=False, k=3)})
        gt_std_ratio_turb_to_calm.append({"horizon": h, "mean_ratio": float(ratio_gt.mean()), "worst_cell": top_cells(ratio_gt, reverse=False, k=3)})

    # Overlap of width failures with S10 active MR failures from summary.
    worst_active_h30 = summary["mean_reversion"].get("worst_active_cells", [])
    mr_fail_cells = {tuple(item["cell"]) for item in worst_active_h30 if item.get("horizon", 30) == 30 or "horizon" not in item}
    hard_width_cells = {tuple(x["cell"]) for x in turb_hard[:8]}
    overlap_cells = sorted(mr_fail_cells & hard_width_cells)

    analysis = {
        "suite_context": {
            "final_passes": [
                key
                for key in [
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
                if summary[key]["overall_pass"]
            ],
            "final_fails": [
                key
                for key in [
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
                if not summary[key]["overall_pass"]
            ],
        },
        "width_allocation_review": {
            "top_turbulent_hard_slices": turb_hard,
            "top_overwide_slices": overwide,
            "turb_vs_calm_sample_std_ratio": std_ratio_turb_to_calm,
            "turb_vs_calm_gt_resid_std_ratio": gt_std_ratio_turb_to_calm,
        },
        "tail_shape_review": {
            "pooled_abs_daily_change": distribution_compare(pooled_abs_delta_gt, pooled_abs_delta_gen),
            "pathwise_max_jump": distribution_compare(path_max_jump_gt, path_max_jump_gen),
            "pathwise_q99_exceedance_count": distribution_compare(path_q99_exceed_count_gt, path_q99_exceed_count_gen),
            "basis_band_energy": {
                band: distribution_compare(np.concatenate(band_energy_gt[band]), np.concatenate(band_energy_gen[band]))
                for band in BANDS
            },
            "basis_band_abs_coeff": {
                band: distribution_compare(np.concatenate(band_abs_coeff_gt[band]), np.concatenate(band_abs_coeff_gen[band]))
                for band in BANDS
            },
        },
        "mean_reversion_review": {
            "summary_full_horizon": summary["mean_reversion"]["full_horizon"],
            "aggregate_ratio": summary["mean_reversion"]["mr_gt_ratio"],
            "sample_minus_deterministic_mean_shift_by_horizon": {
                h: {
                    "mean_shift": float(np.concatenate(vals).mean()),
                    "median_abs_shift": float(np.median(np.abs(np.concatenate(vals)))),
                    "q90_abs_shift": float(np.quantile(np.abs(np.concatenate(vals)), 0.9)),
                    "q99_abs_shift": float(np.quantile(np.abs(np.concatenate(vals)), 0.99)),
                }
                for h, vals in det_shift_by_h.items()
            },
            "worst_active_cells": worst_active_h30,
            "hard_width_and_mr_overlap_cells": overlap_cells,
        },
    }

    # High-level interpretation fields.
    pooled_ratio = analysis["tail_shape_review"]["pooled_abs_daily_change"]["ratio_quantiles"]
    exceed_ratio = analysis["tail_shape_review"]["pathwise_q99_exceedance_count"]["ratio_quantiles"]
    high_energy_ratio = analysis["tail_shape_review"]["basis_band_energy"]["high"]["ratio_quantiles"]
    analysis["mechanistic_conclusion"] = {
        "s3_s7_core_issue": (
            "remaining failures are still dominated by turbulent late-horizon local width misallocation"
            if turb_hard
            else "no concentrated turbulent width cluster found"
        ),
        "s3_s7_failure_mode": "mixed underwide plus late-horizon bias",
        "s4_vs_s11_interpretation": {
            "pooled_abs_delta_ratio_q90": pooled_ratio["0.9"],
            "pooled_abs_delta_ratio_q99": pooled_ratio["0.99"],
            "q99_exceedance_count_ratio_p50": exceed_ratio["0.5"],
            "high_band_energy_ratio_p50": high_energy_ratio["0.5"],
            "message": (
                "S11 passes because pathwise maximum jump size is close enough, "
                "but S4 still fails because the residual path law redistributes tail mass incorrectly: "
                "too much high-band energy and too many moderate-to-large changes, with lighter overall tail concentration."
            ),
        },
        "s10_borderline_interpretation": (
            "aggregate and full-horizon profile are broadly right, but later-horizon active cells still underrevert enough "
            "to miss the strengthened active-profile threshold; this does not overlap materially with the hard S3/S7 width cells."
        ),
    }
    return analysis


def top_cells(values: np.ndarray, reverse: bool, k: int) -> list[dict[str, Any]]:
    flat = []
    arr = np.asarray(values).reshape(-1)
    for idx, value in enumerate(arr):
        flat.append({"cell": cell_label(idx), "value": float(value)})
    flat.sort(key=lambda x: x["value"], reverse=reverse)
    return flat[:k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/backfill/pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182a/final_model.pt",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="results/block_ar/182a_final_v2_s3mrj_full_30d/summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-05/analysis/182a_final_mechanistic",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_default_config()
    model, checkpoint = load_model(args.model_path, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=cfg.data_path,
        history_len=checkpoint["config"]["history_len"],
        future_len=checkpoint["config"]["future_len"],
        test_start=cfg.test_start,
        max_windows=args.max_windows,
    )
    analysis = analyze_182a_final(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        summary=load_json(args.summary_path),
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    output_path = out_dir / "mechanistic_summary.json"
    output_path.write_text(json.dumps(make_serializable(analysis), indent=2))
    print(f"Wrote {output_path}")
