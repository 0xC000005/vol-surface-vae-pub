#!/usr/bin/env python
"""
220m: Oracle slow-state sufficiency test for multi-day rollout.

This is not a deployable model. It keeps the frozen one-day kernel and injects
the ground-truth slow level component during rollout updates, while letting the
kernel generate the fast residual around that oracle slow path.
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

from experiments.backfill.block_ar._rollout_220_utils import (
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_block_ar_tests,
    run_ci_coverage_tests,
    run_cointegration_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_regime_coverage_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)


def _coerce_next_iv(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 3 and samples.shape[-1] == 25:
        return samples.view(samples.shape[0], samples.shape[1], 5, 5)
    if samples.ndim == 4 and samples.shape[-2:] == (5, 5):
        return samples
    raise ValueError(f"Unexpected sample_next_iv shape: {tuple(samples.shape)}")


def compute_slow_surface_series(surfaces: np.ndarray, alpha: float) -> np.ndarray:
    slow = np.empty_like(surfaces)
    slow[0] = surfaces[0]
    for t in range(1, surfaces.shape[0]):
        slow[t] = (1.0 - alpha) * slow[t - 1] + alpha * surfaces[t]
    return slow


@torch.no_grad()
def oracle_rollout_samples(
    model: torch.nn.Module,
    history_norm: torch.Tensor,
    oracle_prev_slow: np.ndarray,
    oracle_future_slow: np.ndarray,
    n_samples: int,
    chunk_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    history_01 = denormalize_iv(history_norm.to(device))
    batch_size, hist_len = history_01.shape[:2]
    future_len = oracle_future_slow.shape[1]
    chunk_size = max(1, min(int(chunk_size), int(n_samples)))

    all_chunks: list[np.ndarray] = []
    for start in range(0, n_samples, chunk_size):
        k = min(chunk_size, n_samples - start)
        hist_k = history_01.unsqueeze(1).expand(batch_size, k, hist_len, 5, 5)
        hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()
        slow_prev = (
            torch.from_numpy(oracle_prev_slow)
            .to(device=device, dtype=history_01.dtype)
            .unsqueeze(1)
            .expand(batch_size, k, 5, 5)
            .reshape(batch_size * k, 5, 5)
            .clone()
        )
        oracle_future = (
            torch.from_numpy(oracle_future_slow)
            .to(device=device, dtype=history_01.dtype)
            .unsqueeze(1)
            .expand(batch_size, k, future_len, 5, 5)
            .reshape(batch_size * k, future_len, 5, 5)
        )

        frames: list[torch.Tensor] = []
        for step in range(future_len):
            next_iv = _coerce_next_iv(model.sample_next_iv(hist_k, n_samples=1)).squeeze(1)
            model_slow_next = (1.0 - args.oracle_alpha) * slow_prev + args.oracle_alpha * next_iv
            residual_next = next_iv - model_slow_next
            hybrid_next = (oracle_future[:, step] + residual_next).clamp(0.0, 1.0)
            frames.append(hybrid_next.view(batch_size, k, 5, 5))
            hist_k = torch.cat([hist_k[:, 1:], hybrid_next.unsqueeze(1)], dim=1)
            slow_prev = oracle_future[:, step]
        all_chunks.append(torch.stack(frames, dim=2).cpu().numpy())
    return np.concatenate(all_chunks, axis=1)


@torch.no_grad()
def unconditional_oracle_rollout_samples(
    model: torch.nn.Module,
    history_shape: torch.Size,
    oracle_prev_slow: np.ndarray,
    oracle_future_slow: np.ndarray,
    n_samples: int,
    chunk_size: int,
) -> np.ndarray:
    zero_history_norm = torch.zeros(history_shape, dtype=torch.float32)
    return oracle_rollout_samples(
        model=model,
        history_norm=zero_history_norm,
        oracle_prev_slow=oracle_prev_slow,
        oracle_future_slow=oracle_future_slow,
        n_samples=n_samples,
        chunk_size=chunk_size,
    )


def run_custom_conditionality(
    cond_samples: np.ndarray,
    uncond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history_01: np.ndarray,
) -> dict[str, Any]:
    cond_lower = np.quantile(cond_samples, 0.05, axis=1)
    cond_upper = np.quantile(cond_samples, 0.95, axis=1)
    cond_median = np.median(cond_samples, axis=1)
    uncond_lower = np.quantile(uncond_samples, 0.05, axis=1)
    uncond_upper = np.quantile(uncond_samples, 0.95, axis=1)
    uncond_median = np.median(uncond_samples, axis=1)

    cond_width = float((cond_upper - cond_lower).mean())
    uncond_width = float((uncond_upper - uncond_lower).mean())
    cond_mae = float(np.abs(cond_median - ground_truth).mean())
    uncond_mae = float(np.abs(uncond_median - ground_truth).mean())
    mae_reduction_pct = ((uncond_mae - cond_mae) / max(uncond_mae, 1e-8)) * 100.0

    mean_iv_hist = history_01.mean(axis=(2, 3))
    vov = np.diff(mean_iv_hist, axis=1).std(axis=1)
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80
    window_width = (cond_upper - cond_lower).mean(axis=(1, 2, 3))
    turb_calm_ratio = float(window_width[turb].mean() / max(window_width[calm].mean(), 1e-8))

    per_h = {}
    for h in [1, 7, 14, 30]:
        if h <= cond_samples.shape[2]:
            t = h - 1
            cw = float((cond_upper[:, t] - cond_lower[:, t]).mean())
            uw = float((uncond_upper[:, t] - uncond_lower[:, t]).mean())
            cmae = float(np.abs(cond_median[:, t] - ground_truth[:, t]).mean())
            umae = float(np.abs(uncond_median[:, t] - ground_truth[:, t]).mean())
            per_h[h] = {
                "width_ratio": cw / max(uw, 1e-8),
                "mae_reduction_pct": ((umae - cmae) / max(umae, 1e-8)) * 100.0,
            }

    var_by_h = {}
    for h in [1, 10, 20, 30]:
        if h <= cond_samples.shape[2]:
            var_by_h[h] = float(cond_samples[:, :, h - 1].var(axis=1).mean())
    monotonic = True
    ordered = [h for h in [1, 10, 20, 30] if h in var_by_h]
    for a, b in zip(ordered[:-1], ordered[1:]):
        monotonic = monotonic and (var_by_h[a] <= var_by_h[b] + 1e-12)

    overall_pass = (mae_reduction_pct > 5.0) and (turb_calm_ratio > 1.15)
    return {
        "cond_width": cond_width,
        "uncond_width": uncond_width,
        "width_ratio": cond_width / max(uncond_width, 1e-8),
        "cond_mae": cond_mae,
        "uncond_mae": uncond_mae,
        "mae_reduction_pct": mae_reduction_pct,
        "turb_calm_ratio": turb_calm_ratio,
        "variance_by_horizon": var_by_h,
        "monotonic_variance": monotonic,
        "per_horizon": per_h,
        "overall_pass": overall_pass,
    }


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    ordered = [
        ("surface", results["surface"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("conditionality", results["conditionality"]["overall_pass"]),
        ("time_series", results["time_series"]["overall_pass"]),
        ("block_ar", results["block_ar"]["overall_pass"]),
        ("cointegration", results["cointegration"]["overall_pass"]),
        ("regime_coverage", results["regime_coverage"]["overall_pass"]),
        ("distributional_fidelity", results["distributional_fidelity"]["overall_pass"]),
        ("cross_cell_correlation", results["cross_cell_correlation"]["overall_pass"]),
        ("mean_reversion", results["mean_reversion"]["overall_pass"]),
        ("pathwise_jump_realism", results["pathwise_jump_realism"]["overall_pass"]),
    ]
    failed = [name for name, passed in ordered if not passed]
    return sum(int(passed) for _name, passed in ordered), failed


def main() -> None:
    parser = argparse.ArgumentParser(description="220m oracle slow-state sufficiency test")
    parser.add_argument("--base_model_type", type=str, default="212ai")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--oracle_alpha", type=float, default=0.08)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    globals()["args"] = args
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.base_model_type, args.checkpoint, device)
    model.eval()

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
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    returns = raw["ret"].astype(np.float64)
    slow_series = compute_slow_surface_series(surfaces, alpha=args.oracle_alpha)

    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_windows]
    oracle_prev_slow = slow_series[val_indices + args.history_len - 1]
    oracle_future_slow = np.stack(
        [slow_series[idx + args.history_len : idx + args.history_len + args.future_len] for idx in val_indices],
        axis=0,
    )

    cond_samples = oracle_rollout_samples(
        model=model,
        history_norm=batch.history_norm,
        oracle_prev_slow=oracle_prev_slow,
        oracle_future_slow=oracle_future_slow,
        n_samples=args.samples,
        chunk_size=args.chunk_size,
    )
    uncond_samples = unconditional_oracle_rollout_samples(
        model=model,
        history_shape=batch.history_norm.shape,
        oracle_prev_slow=oracle_prev_slow,
        oracle_future_slow=oracle_future_slow,
        n_samples=min(args.samples, 32),
        chunk_size=args.chunk_size,
    )

    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    conditionality = run_custom_conditionality(cond_samples, uncond_samples, ground_truth, history_01)
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
            "oracle_type": "220m_oracle_slow_state",
            "base_model_type": args.base_model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "oracle_alpha": args.oracle_alpha,
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

    lines = [
        f"- oracle base model: `{args.base_model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- oracle slow alpha: `{args.oracle_alpha}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- oracle turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- MR ratio h1: `{mean_reversion.get('mr_gt_ratio', float('nan')):.3f}`",
        f"- MR ratio h30: `{mean_reversion.get('full_horizon', {}).get('per_horizon', {}).get(30, {}).get('ratio', float('nan')):.3f}`",
        "",
        "**Fidelity / Structure**",
        f"- time-series ACF corr: `{time_series['acf']['acf_correlation']:.3f}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "220m Oracle Slow-State Sufficiency Test", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
