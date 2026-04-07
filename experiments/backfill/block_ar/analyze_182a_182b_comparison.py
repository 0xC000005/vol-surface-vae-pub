#!/usr/bin/env python
"""
Narrow comparative review of 182a_final vs 182b_final.

Goal:
  Identify exactly what 182b damaged relative to 182a_final on the same
  windows/slices, so 183a can move the useful width/tail controls inside
  the residual path law instead of applying them afterward.
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
from experiments.backfill.block_ar.train_182a_pathwise_residual_law import (
    PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
)
from experiments.backfill.block_ar.train_182b_width_tail_control import (
    WidthTailControlledPathwiseResidualLawModel,
)


SELECT_HORIZONS = [1, 7, 14, 30]
BANDS = ["low", "mid", "high"]


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model_type = cfg["type"]
    enc_cfg = EncoderConfig(**cfg["encoder"])
    common_kwargs = dict(
        encoder_config=enc_cfg,
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    )
    if model_type == "pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182a":
        model = PathwiseResidualLawMeanRevertingCovarianceMixtureModel(**common_kwargs)
    elif model_type == "width_tail_controlled_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_182b":
        model = WidthTailControlledPathwiseResidualLawModel(
            **common_kwargs,
            amplitude_config=cfg["amplitude"],
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


def quantiles(arr: np.ndarray, qs: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {str(q): float(np.quantile(flat, q)) for q in qs}


def ks_detail(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if len(x) == 0 or len(y) == 0:
        return {"ks_stat": float("nan"), "peak_x": float("nan"), "cdf_x": float("nan"), "cdf_y": float("nan")}
    grid = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    diff = np.abs(cdf_x - cdf_y)
    idx = int(np.argmax(diff))
    return {
        "ks_stat": float(diff[idx]),
        "peak_x": float(grid[idx]),
        "cdf_x": float(cdf_x[idx]),
        "cdf_y": float(cdf_y[idx]),
    }


def pearson_kurtosis(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    xc = x - x.mean()
    var = np.mean(np.square(xc))
    if var <= 1e-12:
        return float("nan")
    return float(np.mean(np.power(xc, 4)) / (var * var))


def summarize_benchmark_delta(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, Any]:
    def fetch(summary: dict[str, Any], *keys: str) -> float:
        cur: Any = summary
        for key in keys:
            cur = cur[key]
        return float(cur)

    keys = {
        "s2_cov90": ("coverage", "overall", "0.9"),
        "s3_turb_calm": ("conditionality", "turb_calm_ratio"),
        "s3_worst_width_ratio": ("conditionality", "worst_cell_width_ratio"),
        "s4_kurtosis_ratio": ("time_series", "kurtosis", "kurtosis_ratio"),
        "s7_layer2": ("regime_coverage", "layer2_n_passing"),
        "s7_catastrophic": ("regime_coverage", "layer3_catastrophic_rate"),
        "s8_bad_window_rate": ("distributional", "window_floor", "pct_bad"),
        "s10_active_pass_rate": ("mean_reversion", "active_pass_rate"),
        "s11_jump_ks": ("pathwise_jump_realism", "pathwise_max_jump", "ks_stat"),
    }
    out = {}
    for name, path in keys.items():
        a = fetch(summary_a, *path)
        b = fetch(summary_b, *path)
        out[name] = {"182a_final": a, "182b_final": b, "delta_b_minus_a": b - a}
    return out


@torch.no_grad()
def analyze_model(
    model: PathwiseResidualLawMeanRevertingCovarianceMixtureModel,
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
    n_windows, future_len, n_cells = future_01.shape

    vov, q20, q80 = regime_masks_from_history(history_norm.cpu())
    turb_mask = vov >= q80

    sample_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    sample_std = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    lo90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    hi90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    det_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pooled_abs_delta_gt = []
    pooled_abs_delta_gen = []
    gt_path_max = []
    gen_path_max = []
    gt_q99_exceed = []
    gen_q99_exceed = []
    band_energy_gt = {k: [] for k in BANDS}
    band_energy_gen = {k: [] for k in BANDS}

    geometry = model.path_geometry
    band_masks = {
        "low": geometry.low_band_mask().reshape(1, future_len, n_cells).to(device),
        "mid": geometry.mid_band_mask().reshape(1, future_len, n_cells).to(device),
        "high": geometry.high_band_mask().reshape(1, future_len, n_cells).to(device),
    }

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch = end - start
        hist_01_b = history_01[start:end]
        fut_01_b = future_01[start:end].to(device)

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
        gt_path_max.append(gt_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy())
        gen_path_max.append(gen_diff.abs().amax(dim=(2, 3, 4)).reshape(-1).cpu().numpy())

        gt_q99 = torch.quantile(gt_diff.abs().reshape(-1), 0.99)
        gt_q99_exceed.append((gt_diff.abs().reshape(batch, -1) > gt_q99).sum(dim=1).cpu().numpy())
        gen_q99_exceed.append((gen_diff.abs().reshape(batch * n_samples, -1) > gt_q99).sum(dim=1).cpu().numpy())

        gt_diff_flat = gt_diff.reshape(batch, future_len, n_cells)
        gen_diff_flat = gen_diff.reshape(batch * n_samples, future_len, n_cells)
        gt_coeff = geometry.to_basis(gt_diff_flat)
        gen_coeff = geometry.to_basis(gen_diff_flat)
        gt_total = gt_coeff.reshape(batch, -1).pow(2).sum(dim=1).clamp_min(1e-12)
        gen_total = gen_coeff.reshape(batch * n_samples, -1).pow(2).sum(dim=1).clamp_min(1e-12)
        for band in BANDS:
            mask = band_masks[band].to(dtype=gt_coeff.dtype)
            gt_band = (gt_coeff * mask).reshape(batch, -1)
            gen_band = (gen_coeff * mask).reshape(batch * n_samples, -1)
            band_energy_gt[band].append((gt_band.pow(2).sum(dim=1) / gt_total).cpu().numpy())
            band_energy_gen[band].append((gen_band.pow(2).sum(dim=1) / gen_total).cpu().numpy())

    inside90 = (future_01.cpu().numpy() >= lo90) & (future_01.cpu().numpy() <= hi90)
    det_resid = future_01.cpu().numpy() - det_mean
    z_mean = det_resid / np.maximum(sample_std, 1e-6)

    return {
        "turb_mask": turb_mask,
        "coverage90": inside90.astype(np.float32),
        "sample_std": sample_std,
        "z_mean": z_mean,
        "det_resid": det_resid,
        "sample_mean": sample_mean,
        "det_mean": det_mean,
        "pooled_abs_delta_gt": np.concatenate(pooled_abs_delta_gt),
        "pooled_abs_delta_gen": np.concatenate(pooled_abs_delta_gen),
        "gt_path_max": np.concatenate(gt_path_max),
        "gen_path_max": np.concatenate(gen_path_max),
        "gt_q99_exceed": np.concatenate(gt_q99_exceed),
        "gen_q99_exceed": np.concatenate(gen_q99_exceed),
        "band_energy_gt": {k: np.concatenate(v) for k, v in band_energy_gt.items()},
        "band_energy_gen": {k: np.concatenate(v) for k, v in band_energy_gen.items()},
    }


def top_slice_deltas(
    review_182a: dict[str, Any],
    res_a: dict[str, Any],
    res_b: dict[str, Any],
) -> dict[str, Any]:
    hard_slices = review_182a["width_allocation_review"]["top_turbulent_hard_slices"][:12]
    over_slices = review_182a["width_allocation_review"]["top_overwide_slices"][:8]

    def collect(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for item in entries:
            h = int(item["horizon"])
            cell_r, cell_c = item["cell"]
            h_idx = SELECT_HORIZONS.index(h)
            c_idx = cell_r * 5 + cell_c
            turb = res_a["turb_mask"]
            cov_a = float(res_a["coverage90"][turb, h_idx, c_idx].mean())
            cov_b = float(res_b["coverage90"][turb, h_idx, c_idx].mean())
            gt_std_a = float(res_a["det_resid"][turb, h_idx, c_idx].std())
            gt_std_b = float(res_b["det_resid"][turb, h_idx, c_idx].std())
            std_a = float(res_a["sample_std"][turb, h_idx, c_idx].mean() / max(gt_std_a, 1e-6))
            std_b = float(res_b["sample_std"][turb, h_idx, c_idx].mean() / max(gt_std_b, 1e-6))
            z_a = float(res_a["z_mean"][turb, h_idx, c_idx].mean())
            z_b = float(res_b["z_mean"][turb, h_idx, c_idx].mean())
            out.append(
                {
                    "horizon": h,
                    "cell": [cell_r, cell_c],
                    "coverage90_182a": cov_a,
                    "coverage90_182b": cov_b,
                    "coverage_delta_b_minus_a": cov_b - cov_a,
                    "std_ratio_182a": std_a,
                    "std_ratio_182b": std_b,
                    "std_ratio_delta_b_minus_a": std_b - std_a,
                    "z_mean_182a": z_a,
                    "z_mean_182b": z_b,
                }
            )
        return out

    return {
        "hard_slice_deltas": collect(hard_slices),
        "overwide_slice_deltas": collect(over_slices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 182a_final vs 182b_final")
    parser.add_argument("--checkpoint_182a", type=str, required=True)
    parser.add_argument("--checkpoint_182b", type=str, required=True)
    parser.add_argument("--summary_182a", type=str, required=True)
    parser.add_argument("--summary_182b", type=str, required=True)
    parser.add_argument("--review_182a", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--max_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    summary_a = load_json(args.summary_182a)
    summary_b = load_json(args.summary_182b)
    review_a = load_json(args.review_182a)
    model_a, _ = load_model(args.checkpoint_182a, args.device)
    model_b, _ = load_model(args.checkpoint_182b, args.device)

    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )

    res_a = analyze_model(model_a, history_norm, future_norm, args.device, args.batch_size, args.n_samples)
    res_b = analyze_model(model_b, history_norm, future_norm, args.device, args.batch_size, args.n_samples)

    out = {
        "benchmark_delta": summarize_benchmark_delta(summary_a, summary_b),
        "top_slice_deltas": top_slice_deltas(review_a, res_a, res_b),
        "pooled_abs_delta_compare": {
            "182a_gen_to_gt_quantiles": {
                k: float(v / max(quantiles(res_a["pooled_abs_delta_gt"])[k], 1e-12))
                for k, v in quantiles(res_a["pooled_abs_delta_gen"]).items()
            },
            "182b_gen_to_gt_quantiles": {
                k: float(v / max(quantiles(res_b["pooled_abs_delta_gt"])[k], 1e-12))
                for k, v in quantiles(res_b["pooled_abs_delta_gen"]).items()
            },
            "182a_kurtosis_ratio": float(
                pearson_kurtosis(res_a["pooled_abs_delta_gen"]) / max(pearson_kurtosis(res_a["pooled_abs_delta_gt"]), 1e-12)
            ),
            "182b_kurtosis_ratio": float(
                pearson_kurtosis(res_b["pooled_abs_delta_gen"]) / max(pearson_kurtosis(res_b["pooled_abs_delta_gt"]), 1e-12)
            ),
        },
        "pathwise_jump_compare": {
            "182a_jump_ks": ks_detail(res_a["gt_path_max"], res_a["gen_path_max"]),
            "182b_jump_ks": ks_detail(res_b["gt_path_max"], res_b["gen_path_max"]),
            "182a_jump_q99_exceed_quantiles": {
                "gt": quantiles(res_a["gt_q99_exceed"]),
                "gen": quantiles(res_a["gen_q99_exceed"]),
            },
            "182b_jump_q99_exceed_quantiles": {
                "gt": quantiles(res_b["gt_q99_exceed"]),
                "gen": quantiles(res_b["gen_q99_exceed"]),
            },
        },
        "band_energy_compare": {
            band: {
                "182a_gen_to_gt_p50": float(
                    np.quantile(res_a["band_energy_gen"][band], 0.5)
                    / max(np.quantile(res_a["band_energy_gt"][band], 0.5), 1e-12)
                ),
                "182b_gen_to_gt_p50": float(
                    np.quantile(res_b["band_energy_gen"][band], 0.5)
                    / max(np.quantile(res_b["band_energy_gt"][band], 0.5), 1e-12)
                ),
            }
            for band in BANDS
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_serializable(out), indent=2))
    print(f"Wrote comparison to {output_path}")


if __name__ == "__main__":
    main()
