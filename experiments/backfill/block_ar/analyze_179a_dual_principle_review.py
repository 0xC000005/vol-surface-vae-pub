#!/usr/bin/env python
"""
Dual-principle targeted review for 179a.

Goal:
  - check whether 179a still produces individually plausible scenarios from
    both a spatial and temporal perspective
  - distinguish broad structural realism from pathwise realism regressions
  - verify whether the centered residual transport is actually preserving the
    mean backbone in sampled paths
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
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_179a_centered_residual_transport_mean_reverting_covariance_mixture import (
    CenteredResidualTransportMeanRevertingCovarianceMixtureModel,
)


SELECT_HORIZONS = [1, 7, 14, 30]


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def window_slopes(prev: np.ndarray, next_values: np.ndarray) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    delta = next_values - prev
    xc = prev - prev.mean(axis=1, keepdims=True)
    yc = delta - delta.mean(axis=1, keepdims=True)
    denom = np.square(xc).sum(axis=1)
    numer = (xc * yc).sum(axis=1)
    out = np.zeros(prev.shape[0], dtype=np.float64)
    mask = denom > 1e-12
    out[mask] = numer[mask] / denom[mask]
    return out


def surface_roughness(surfaces: np.ndarray) -> np.ndarray:
    surfaces = np.asarray(surfaces, dtype=np.float64)
    d2_row = surfaces[:, 2:, :] - 2.0 * surfaces[:, 1:-1, :] + surfaces[:, :-2, :]
    d2_col = surfaces[:, :, 2:] - 2.0 * surfaces[:, :, 1:-1] + surfaces[:, :, :-2]
    return np.mean(np.abs(d2_row), axis=(1, 2)) + np.mean(np.abs(d2_col), axis=(1, 2))


def path_total_variation(paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(paths, dtype=np.float64)
    diff = np.diff(paths, axis=1)
    return np.mean(np.abs(diff), axis=(1, 2, 3))


def path_max_jump(paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(paths, dtype=np.float64)
    diff = np.diff(paths, axis=1)
    return np.max(np.abs(diff), axis=(1, 2, 3))


def top_cells(values: np.ndarray, reverse: bool = True, k: int = 5) -> list[dict[str, Any]]:
    flat = []
    for idx, value in enumerate(values.reshape(-1)):
        flat.append({"cell": [int(idx // 5), int(idx % 5)], "value": float(value)})
    flat.sort(key=lambda x: x["value"], reverse=reverse)
    return flat[:k]


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "centered_residual_transport_mean_reverting_covariance_mixture_structured_joint_student_t_179a"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = CenteredResidualTransportMeanRevertingCovarianceMixtureModel(
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


@torch.no_grad()
def analyze_179a(
    model: CenteredResidualTransportMeanRevertingCovarianceMixtureModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], 5, 5)
    prev_01 = history_01[:, -1]

    shift_u_sum = {h: 0.0 for h in SELECT_HORIZONS}
    shift_u_abs_sum = {h: 0.0 for h in SELECT_HORIZONS}
    shift_iv_sum = {h: 0.0 for h in SELECT_HORIZONS}
    shift_iv_abs_sum = {h: 0.0 for h in SELECT_HORIZONS}
    shift_count = 0
    h30_shift_iv_sum = np.zeros((5, 5), dtype=np.float64)

    mr_gt = {h: [] for h in SELECT_HORIZONS}
    mr_gen = {h: [] for h in SELECT_HORIZONS}
    rough_gt = {h: [] for h in SELECT_HORIZONS}
    rough_gen = {h: [] for h in SELECT_HORIZONS}
    gate_entropy = []
    gate_max = []
    tv_gt_all = []
    tv_gen_all = []
    jump_gt_all = []
    jump_gen_all = []

    n_windows = history_norm.shape[0]
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_b = history_01[start:end]
        fut_b = future_01[start:end]
        prev_b = prev_01[start:end].reshape(end - start, 25)

        (
            mu_u,
            _time_factor,
            _time_diag,
            _cell_factor,
            _cell_diag,
            _scale,
            _flow_context,
            _base_local_delta,
            block_logits,
        ) = model.forward_from_history(hist_b)
        samples_u = model.sample_future_u(hist_b, n_samples)

        mu_iv = unconstrained_to_iv(mu_u, lo=model.support_lo, hi=model.support_hi).reshape(end - start, 30, 5, 5)
        sample_mean_u = samples_u.mean(dim=1)
        sample_mean_iv = unconstrained_to_iv(sample_mean_u, lo=model.support_lo, hi=model.support_hi).reshape(end - start, 30, 5, 5)
        samples_iv = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi).reshape(end - start, n_samples, 30, 5, 5)

        probs = torch.softmax(block_logits, dim=-1)
        gate_entropy.append((-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)).mean().item())
        gate_max.append(probs.max(dim=-1).values.mean().item())

        for h in SELECT_HORIZONS:
            det_u_h = mu_u[:, h - 1].reshape(end - start, 5, 5)
            mean_u_h = sample_mean_u[:, h - 1].reshape(end - start, 5, 5)
            det_iv_h = mu_iv[:, h - 1]
            mean_iv_h = sample_mean_iv[:, h - 1]

            shift_u = (mean_u_h - det_u_h).cpu().numpy()
            shift_iv = (mean_iv_h - det_iv_h).cpu().numpy()
            shift_u_sum[h] += float(shift_u.mean())
            shift_u_abs_sum[h] += float(np.abs(shift_u).mean())
            shift_iv_sum[h] += float(shift_iv.mean())
            shift_iv_abs_sum[h] += float(np.abs(shift_iv).mean())

            gt_h = fut_b[:, h - 1].cpu().numpy()
            gen_h = samples_iv[:, :, h - 1].cpu().numpy().reshape((end - start) * n_samples, 5, 5)
            rough_gt[h].append(surface_roughness(gt_h))
            rough_gen[h].append(surface_roughness(gen_h))

            gt_slope = window_slopes(prev_b.cpu().numpy(), gt_h.reshape(end - start, 25))
            gen_slope = window_slopes(
                np.repeat(prev_b.cpu().numpy(), n_samples, axis=0),
                gen_h.reshape((end - start) * n_samples, 25),
            )
            mr_gt[h].append(gt_slope)
            mr_gen[h].append(gen_slope)

            if h == 30:
                h30_shift_iv_sum += shift_iv.mean(axis=0)

        gt_path = torch.cat([hist_b[:, -1:].cpu(), fut_b.cpu().reshape(end - start, 30, 5, 5)], dim=1).numpy()
        gen_path = torch.cat(
            [
                hist_b[:, -1:].unsqueeze(1).cpu().expand(end - start, n_samples, 1, 5, 5),
                samples_iv.cpu(),
            ],
            dim=2,
        ).numpy().reshape((end - start) * n_samples, 31, 5, 5)
        tv_gt_all.append(path_total_variation(gt_path))
        tv_gen_all.append(path_total_variation(gen_path))
        jump_gt_all.append(path_max_jump(gt_path))
        jump_gen_all.append(path_max_jump(gen_path))

        shift_count += 1

    roughness_summary = {}
    mean_reversion_summary = {}
    for h in SELECT_HORIZONS:
        gt_r = np.concatenate(rough_gt[h], axis=0)
        gen_r = np.concatenate(rough_gen[h], axis=0)
        gt_s = np.concatenate(mr_gt[h], axis=0)
        gen_s = np.concatenate(mr_gen[h], axis=0)
        roughness_summary[str(h)] = {
            "gt_mean": float(gt_r.mean()),
            "gen_mean": float(gen_r.mean()),
            "mean_ratio": float(gen_r.mean() / max(gt_r.mean(), 1e-12)),
            "gt_q90": float(np.quantile(gt_r, 0.90)),
            "gen_q90": float(np.quantile(gen_r, 0.90)),
            "q90_ratio": float(np.quantile(gen_r, 0.90) / max(np.quantile(gt_r, 0.90), 1e-12)),
            "ks_stat": ks_statistic(gt_r, gen_r),
        }
        mean_reversion_summary[str(h)] = {
            "gt_mean_slope": float(gt_s.mean()),
            "gen_mean_slope": float(gen_s.mean()),
            "mean_ratio": float(gen_s.mean() / min(gt_s.mean(), -1e-12)) if gt_s.mean() < 0 else float("nan"),
            "gt_q10": float(np.quantile(gt_s, 0.10)),
            "gen_q10": float(np.quantile(gen_s, 0.10)),
            "gt_q50": float(np.quantile(gt_s, 0.50)),
            "gen_q50": float(np.quantile(gen_s, 0.50)),
            "gt_q90": float(np.quantile(gt_s, 0.90)),
            "gen_q90": float(np.quantile(gen_s, 0.90)),
            "ks_stat": ks_statistic(gt_s, gen_s),
        }

    tv_gt = np.concatenate(tv_gt_all, axis=0)
    tv_gen = np.concatenate(tv_gen_all, axis=0)
    jump_gt = np.concatenate(jump_gt_all, axis=0)
    jump_gen = np.concatenate(jump_gen_all, axis=0)

    return {
        "centered_transport_review": {
            "mean_signed_shift_u_by_horizon": {str(h): shift_u_sum[h] / shift_count for h in SELECT_HORIZONS},
            "mean_abs_shift_u_by_horizon": {str(h): shift_u_abs_sum[h] / shift_count for h in SELECT_HORIZONS},
            "mean_signed_shift_iv_by_horizon": {str(h): shift_iv_sum[h] / shift_count for h in SELECT_HORIZONS},
            "mean_abs_shift_iv_by_horizon": {str(h): shift_iv_abs_sum[h] / shift_count for h in SELECT_HORIZONS},
            "largest_positive_h30_shift_cells_iv": top_cells(h30_shift_iv_sum / max(shift_count, 1), reverse=True, k=5),
            "largest_negative_h30_shift_cells_iv": top_cells(h30_shift_iv_sum / max(shift_count, 1), reverse=False, k=5),
        },
        "spatial_individual_realism": {
            "surface_roughness_by_horizon": roughness_summary,
            "gate_entropy_mean": float(np.mean(gate_entropy)),
            "gate_max_mean": float(np.mean(gate_max)),
        },
        "temporal_individual_realism": {
            "scenario_level_mean_reversion_by_horizon": mean_reversion_summary,
            "path_total_variation": {
                "gt_mean": float(tv_gt.mean()),
                "gen_mean": float(tv_gen.mean()),
                "mean_ratio": float(tv_gen.mean() / max(tv_gt.mean(), 1e-12)),
                "gt_q90": float(np.quantile(tv_gt, 0.90)),
                "gen_q90": float(np.quantile(tv_gen, 0.90)),
                "q90_ratio": float(np.quantile(tv_gen, 0.90) / max(np.quantile(tv_gt, 0.90), 1e-12)),
                "ks_stat": ks_statistic(tv_gt, tv_gen),
            },
            "path_max_jump": {
                "gt_mean": float(jump_gt.mean()),
                "gen_mean": float(jump_gen.mean()),
                "mean_ratio": float(jump_gen.mean() / max(jump_gt.mean(), 1e-12)),
                "gt_q90": float(np.quantile(jump_gt, 0.90)),
                "gen_q90": float(np.quantile(jump_gen, 0.90)),
                "q90_ratio": float(np.quantile(jump_gen, 0.90) / max(np.quantile(jump_gt, 0.90), 1e-12)),
                "ks_stat": ks_statistic(jump_gt, jump_gen),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Dual-principle mechanism review for 179a")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--summary_path", type=str, required=True)
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
    with open(args.summary_path) as f:
        suite_summary = json.load(f)

    analysis = analyze_179a(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    analysis["benchmark_anchor"] = {
        "model_type": checkpoint["config"]["type"],
        "surface_pass": suite_summary["surface"]["overall_pass"],
        "time_series_pass": suite_summary["time_series"]["overall_pass"],
        "cross_cell_pass": suite_summary["cross_cell_correlation"]["overall_pass"],
        "mean_reversion_pass": suite_summary["mean_reversion"]["overall_pass"],
        "pathwise_jump_pass": suite_summary["pathwise_jump_realism"]["overall_pass"],
        "distributional_pass": suite_summary["distributional"]["overall_pass"],
        "coverage_pass": suite_summary["coverage"]["overall_pass"],
        "conditionality_pass": suite_summary["conditionality"]["overall_pass"],
        "regime_coverage_pass": suite_summary["regime_coverage"]["overall_pass"],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_serializable(analysis), indent=2))
    print(f"Saved review to {output_path}")


if __name__ == "__main__":
    main()
