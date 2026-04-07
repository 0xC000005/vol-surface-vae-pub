#!/usr/bin/env python
"""
Focused tail/jump mechanism review for 179b.

Questions:
  1. Was benchmarking only the epoch-2 best-loss checkpoint premature?
  2. Why can broad marginal/KS tests pass while kurtosis and pathwise max-jump fail?
  3. Is the geometry-aware basis transport suppressing true tail behavior,
     redistributing it across bands, or creating the wrong pathwise jump shape?
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
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_179b_basis_centered_residual_transport import (
    BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel,
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
BANDS = ["low", "mid", "high"]


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


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "basis_centered_residual_transport_mean_reverting_covariance_mixture_structured_joint_student_t_179b"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
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


def quantile_profile(x: np.ndarray, qs: tuple[float, ...] = (0.5, 0.75, 0.9, 0.99, 0.999)) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {str(q): float(np.quantile(x, q)) for q in qs}


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


@torch.no_grad()
def analyze_179b_tail_jump(
    model: BasisCenteredResidualTransportMeanRevertingCovarianceMixtureModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    geometry = model.flow.geometry
    band_masks = {
        "low": geometry.low_band_mask().reshape(1, geometry.n_frames, geometry.n_cells),
        "mid": geometry.mid_band_mask().reshape(1, geometry.n_frames, geometry.n_cells),
        "high": geometry.high_band_mask().reshape(1, geometry.n_frames, geometry.n_cells),
    }
    band_masks = {k: v.to(device=device, dtype=history_norm.dtype) for k, v in band_masks.items()}

    pooled_abs_delta_gt = []
    pooled_abs_delta_gen = []
    path_max_jump_gt = []
    path_max_jump_gen = []
    path_extreme_count_gt = []
    path_extreme_count_gen = []
    band_energy_gt = {k: [] for k in BANDS}
    band_energy_gen = {k: [] for k in BANDS}
    band_abs_coeff_gt = {k: [] for k in BANDS}
    band_abs_coeff_gen = {k: [] for k in BANDS}
    band_path_max_gt = {k: [] for k in BANDS}
    band_path_max_gen = {k: [] for k in BANDS}

    flow_abs_logscale = []
    high_band_neg_logscale = []
    high_band_abs_logscale = []
    basis_coeff_std = []
    high_band_coeff_std = []
    block_gate_entropy = []
    block_gate_max = []

    n_windows = history_norm.shape[0]
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_norm_b = history_norm[start:end]
        fut_norm_b = future_norm[start:end]

        hist_01_b = denormalize_iv(hist_norm_b)
        fut_01_b = denormalize_iv(fut_norm_b)
        samples_u = model.sample_future_u(hist_01_b, n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).reshape(
            end - start, n_samples, geometry.n_frames, 5, 5
        )

        gt_path = torch.cat([hist_01_b[:, -1:].reshape(end - start, 1, 5, 5), fut_01_b], dim=1)
        gen_path = torch.cat(
            [hist_01_b[:, -1:].reshape(end - start, 1, 1, 5, 5).expand(-1, n_samples, -1, -1, -1), samples_01],
            dim=2,
        )
        gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
        gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]

        gt_diff_flat = gt_diff.reshape(end - start, geometry.n_frames, geometry.n_cells)
        gen_diff_flat = gen_diff.reshape((end - start) * n_samples, geometry.n_frames, geometry.n_cells)
        gt_coeff = geometry.to_basis(gt_diff_flat)
        gen_coeff = geometry.to_basis(gen_diff_flat)

        pooled_abs_delta_gt.append(gt_diff.abs().reshape(-1).cpu().numpy())
        pooled_abs_delta_gen.append(gen_diff.abs().reshape(-1).cpu().numpy())
        path_max_jump_gt.append(gt_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy())
        path_max_jump_gen.append(gen_diff.abs().amax(dim=(2, 3, 4)).reshape(-1).cpu().numpy())

        for band in BANDS:
            mask = band_masks[band]
            gt_band_coeff = (gt_coeff * mask).reshape(end - start, -1)
            gen_band_coeff = (gen_coeff * mask).reshape((end - start) * n_samples, -1)
            gt_total_energy = gt_coeff.reshape(end - start, -1).pow(2).sum(dim=1).clamp_min(1e-12)
            gen_total_energy = gen_coeff.reshape((end - start) * n_samples, -1).pow(2).sum(dim=1).clamp_min(1e-12)
            gt_band_energy = gt_band_coeff.pow(2).sum(dim=1) / gt_total_energy
            gen_band_energy = gen_band_coeff.pow(2).sum(dim=1) / gen_total_energy
            band_energy_gt[band].append(gt_band_energy.cpu().numpy())
            band_energy_gen[band].append(gen_band_energy.cpu().numpy())
            band_abs_coeff_gt[band].append(gt_band_coeff.abs().reshape(-1).cpu().numpy())
            band_abs_coeff_gen[band].append(gen_band_coeff.abs().reshape(-1).cpu().numpy())

            gt_band_diff = geometry.from_basis(gt_coeff * mask).reshape(end - start, geometry.n_frames, 5, 5)
            gen_band_diff = geometry.from_basis(gen_coeff * mask).reshape((end - start) * n_samples, geometry.n_frames, 5, 5)
            band_path_max_gt[band].append(gt_band_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy())
            band_path_max_gen[band].append(gen_band_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy())

        gt_target_u = iv_to_unconstrained(
            fut_01_b.reshape(end - start, geometry.n_frames, geometry.n_cells),
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        outputs = model.forward_from_history(hist_01_b)
        _logprob, aux = model.log_prob_future(gt_target_u, *outputs)
        flow_abs_logscale.append(aux["flow_abs_logscale"].cpu().numpy())
        high_band_neg_logscale.append(aux["high_band_neg_logscale"].cpu().numpy())
        high_band_abs_logscale.append(aux["high_band_abs_logscale"].cpu().numpy())
        basis_coeff_std.append(aux["basis_coeff_std"].cpu().numpy())
        high_band_coeff_std.append(aux["high_band_coeff_std"].cpu().numpy())
        block_gate_entropy.append(aux["block_gate_entropy"].cpu().numpy())
        block_gate_max.append(aux["block_gate_max"].cpu().numpy())

    pooled_abs_delta_gt = np.concatenate(pooled_abs_delta_gt, axis=0)
    pooled_abs_delta_gen = np.concatenate(pooled_abs_delta_gen, axis=0)
    path_max_jump_gt = np.concatenate(path_max_jump_gt, axis=0)
    path_max_jump_gen = np.concatenate(path_max_jump_gen, axis=0)

    gt_extreme_threshold = float(np.quantile(pooled_abs_delta_gt, 0.99))
    path_extreme_count_gt = []
    path_extreme_count_gen = []
    # Recompute count distributions from cached path maxima sources is not enough; use quantile threshold from pooled diffs.
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_norm_b = history_norm[start:end]
        fut_norm_b = future_norm[start:end]
        hist_01_b = denormalize_iv(hist_norm_b)
        fut_01_b = denormalize_iv(fut_norm_b)
        samples_u = model.sample_future_u(hist_01_b, n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).reshape(
            end - start, n_samples, geometry.n_frames, 5, 5
        )
        gt_path = torch.cat([hist_01_b[:, -1:].reshape(end - start, 1, 5, 5), fut_01_b], dim=1)
        gen_path = torch.cat(
            [hist_01_b[:, -1:].reshape(end - start, 1, 1, 5, 5).expand(-1, n_samples, -1, -1, -1), samples_01],
            dim=2,
        )
        gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
        gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]
        path_extreme_count_gt.append((gt_diff.abs() >= gt_extreme_threshold).sum(dim=(1, 2, 3)).cpu().numpy())
        path_extreme_count_gen.append((gen_diff.abs() >= gt_extreme_threshold).sum(dim=(2, 3, 4)).reshape(-1).cpu().numpy())
    path_extreme_count_gt = np.concatenate(path_extreme_count_gt, axis=0)
    path_extreme_count_gen = np.concatenate(path_extreme_count_gen, axis=0)

    basis_band_summary = {}
    for band in BANDS:
        gt_energy = np.concatenate(band_energy_gt[band], axis=0)
        gen_energy = np.concatenate(band_energy_gen[band], axis=0)
        gt_abs_coeff = np.concatenate(band_abs_coeff_gt[band], axis=0)
        gen_abs_coeff = np.concatenate(band_abs_coeff_gen[band], axis=0)
        gt_path_jump = np.concatenate(band_path_max_gt[band], axis=0)
        gen_path_jump = np.concatenate(band_path_max_gen[band], axis=0)
        basis_band_summary[band] = {
            "energy_share": {
                "gt_mean": float(gt_energy.mean()),
                "gen_mean": float(gen_energy.mean()),
                "mean_ratio": float(gen_energy.mean() / max(gt_energy.mean(), 1e-12)),
            },
            "abs_coeff_distribution": summarize_distribution(gt_abs_coeff, gen_abs_coeff),
            "pathwise_band_max_jump": summarize_distribution(gt_path_jump, gen_path_jump),
        }

    teacher_forced_stats = {
        "flow_abs_logscale_mean": float(np.concatenate(flow_abs_logscale, axis=0).mean()),
        "high_band_neg_logscale_mean": float(np.concatenate(high_band_neg_logscale, axis=0).mean()),
        "high_band_abs_logscale_mean": float(np.concatenate(high_band_abs_logscale, axis=0).mean()),
        "basis_coeff_std_mean": float(np.concatenate(basis_coeff_std, axis=0).mean()),
        "high_band_coeff_std_mean": float(np.concatenate(high_band_coeff_std, axis=0).mean()),
        "block_gate_entropy_mean": float(np.concatenate(block_gate_entropy, axis=0).mean()),
        "block_gate_max_mean": float(np.concatenate(block_gate_max, axis=0).mean()),
    }

    return {
        "bulk_vs_tail_daily_changes": summarize_distribution(pooled_abs_delta_gt, pooled_abs_delta_gen),
        "pathwise_max_jump": summarize_distribution(path_max_jump_gt, path_max_jump_gen),
        "pathwise_extreme_jump_count": summarize_distribution(path_extreme_count_gt, path_extreme_count_gen),
        "basis_band_profile": basis_band_summary,
        "teacher_forced_flow_stats": teacher_forced_stats,
        "gt_extreme_jump_threshold_q99": gt_extreme_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Focused tail/jump review for 179b")
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
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

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

    analysis = analyze_179b_tail_jump(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
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
            "mr_first_step_ratio": float(summary["mean_reversion"]["mr_gt_ratio"]),
            "mr_full_horizon_pass": bool(summary["mean_reversion"]["full_horizon"]["overall_pass"]),
            "kurtosis_ratio": float(summary["time_series"]["kurtosis"]["kurtosis_ratio"]),
            "max_jump_ks": float(summary["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
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
            "mr_first_step_ratio": float(comparison["mean_reversion"]["mr_gt_ratio"]),
            "mr_full_horizon_pass": bool(comparison["mean_reversion"]["full_horizon"]["overall_pass"]),
            "kurtosis_ratio": float(comparison["time_series"]["kurtosis"]["kurtosis_ratio"]),
            "max_jump_ks": float(comparison["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
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
