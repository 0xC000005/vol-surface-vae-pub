#!/usr/bin/env python
"""759a: state-local residual support calibration around a normalized-innovation generator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import load_model  # noqa: E402
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    alignment_diagnostics,
    build_val_block,
    generate_iv_samples,
)


def split_namespace(args: argparse.Namespace, split: str) -> argparse.Namespace:
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        state_scope=args.state_scope,
        eval_split=split,
        test_start=args.test_start,
        val_size=args.val_size,
        iv_count=args.iv_count,
        clean_nonpositive_log_levels=True,
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=args.iv_lower_bound,
        iv_upper_bound=args.iv_upper_bound,
        scale_half_life=args.scale_half_life,
        scale_floor=args.scale_floor,
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
        max_windows=args.max_windows,
    )


def generate_split(
    args: argparse.Namespace,
    model: Any,
    payload: dict[str, Any],
    split: str,
    device: torch.device,
) -> tuple[np.ndarray, Any, dict[str, float]]:
    split_args = split_namespace(args, split)
    history_level, history_norm, center, scale, drift_feature, history_raw, specs, block = build_val_block(
        split_args,
        payload,
    )
    n_windows = min(int(args.max_windows), int(history_level.shape[0]))
    history_level = history_level[:n_windows]
    history_norm = history_norm[:n_windows]
    center = center[:n_windows]
    scale = scale[:n_windows]
    drift_feature = drift_feature[:n_windows]
    history_raw = history_raw[:n_windows]
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=n_windows,
        device=device,
        split=split,
    )
    alignment = alignment_diagnostics(block, batch, n_windows)
    if alignment["history_max_abs_error"] > 1e-6 or alignment["future_max_abs_error"] > 1e-6:
        raise RuntimeError(f"split alignment failed for {split}: {alignment}")
    print(f"Generating {args.samples} samples for {n_windows} {split} windows", flush=True)
    samples = generate_iv_samples(
        model,
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
        sample_temperature=float(args.sample_temperature),
    )
    return samples, batch, alignment


def state_bucket_edges(current_level: np.ndarray, n_buckets: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(n_buckets) + 1)[1:-1]
    edges = np.quantile(current_level, quantiles, axis=0)
    return np.asarray(edges, dtype=np.float64)


def bucketize_current(current_level: np.ndarray, edges: np.ndarray) -> np.ndarray:
    buckets = np.zeros_like(current_level, dtype=np.int64)
    for cell in range(current_level.shape[1]):
        buckets[:, cell] = np.searchsorted(edges[:, cell], current_level[:, cell], side="right")
    return buckets


def fit_state_local_calibration(
    samples: np.ndarray,
    target: np.ndarray,
    current_level: np.ndarray,
    *,
    n_buckets: int,
    quantile: float,
    min_scale: float,
    max_scale: float,
    location_shrink: float,
    eps: float,
) -> dict[str, Any]:
    flat_samples = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2], -1)
    flat_target = target.reshape(target.shape[0], target.shape[1], -1)
    median = np.median(flat_samples, axis=1)
    q05 = np.quantile(flat_samples, 0.05, axis=1)
    q95 = np.quantile(flat_samples, 0.95, axis=1)
    half_width = np.maximum((q95 - q05) * 0.5, float(eps))
    residual = flat_target - median
    flat_current = current_level.reshape(current_level.shape[0], -1)
    edges = state_bucket_edges(flat_current, int(n_buckets))
    buckets = bucketize_current(flat_current, edges)
    horizon = residual.shape[1]
    n_cells = residual.shape[2]
    loc = np.zeros((int(n_buckets), horizon, n_cells), dtype=np.float32)
    scale = np.ones((int(n_buckets), horizon, n_cells), dtype=np.float32)
    counts = np.zeros((int(n_buckets), n_cells), dtype=np.int64)
    for bucket in range(int(n_buckets)):
        for cell in range(n_cells):
            mask = buckets[:, cell] == bucket
            counts[bucket, cell] = int(mask.sum())
            if int(mask.sum()) < 8:
                continue
            bucket_residual = residual[mask, :, cell]
            bucket_half_width = half_width[mask, :, cell]
            bucket_loc = float(location_shrink) * np.median(bucket_residual, axis=0)
            required_scale = np.quantile(
                np.abs(bucket_residual - bucket_loc[None, :]) / np.maximum(bucket_half_width, float(eps)),
                float(quantile),
                axis=0,
            )
            loc[bucket, :, cell] = bucket_loc.astype(np.float32)
            scale[bucket, :, cell] = np.clip(required_scale, float(min_scale), float(max_scale)).astype(np.float32)
    return {
        "edges": edges.astype(np.float32),
        "loc": loc,
        "scale": scale,
        "counts": counts,
        "n_buckets": int(n_buckets),
        "quantile": float(quantile),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "location_shrink": float(location_shrink),
    }


def apply_state_local_calibration(
    samples: np.ndarray,
    current_level: np.ndarray,
    calibration: dict[str, Any],
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    original_shape = samples.shape
    flat = samples.reshape(samples.shape[0], samples.shape[1], samples.shape[2], -1)
    median = np.median(flat, axis=1, keepdims=True)
    flat_current = current_level.reshape(current_level.shape[0], -1)
    buckets = bucketize_current(flat_current, np.asarray(calibration["edges"]))
    loc = np.asarray(calibration["loc"])
    scale = np.asarray(calibration["scale"])
    calibrated = np.empty_like(flat)
    for window in range(flat.shape[0]):
        for cell in range(flat.shape[-1]):
            bucket = int(buckets[window, cell])
            calibrated[window, :, :, cell] = (
                median[window, :, :, cell]
                + loc[bucket, :, cell][None, :]
                + scale[bucket, :, cell][None, :] * (flat[window, :, :, cell] - median[window, :, :, cell])
            )
    return np.clip(calibrated.reshape(original_shape), float(lower), float(upper))


def summarize(results: dict[str, Any], config: dict[str, Any]) -> list[str]:
    coverage = results["coverage"]
    distributional = results["distributional_fidelity"]
    return [
        "- source: `759a state-local support calibration`",
        f"- base checkpoint: `{config['checkpoint']}`",
        f"- calibration split: `{config['calibration_split']}`",
        f"- eval split: `{config['eval_split']}`",
        f"- suite score: `{results['summary']['n_pass']}/11`",
        f"- failed suites: `{', '.join(results['summary']['failed_suites']) if results['summary']['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- calibration error: `{coverage['calibration_error']:.3f}`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- median-bias cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- cointegration ratio: `{results['cointegration']['gen_gt_ratio']:.3f}`",
        f"- worst-cell cointegration ratio: `{results['cointegration']['worst_cell_ratio']:.3f}`",
        f"- regime layer2: `{results['regime_coverage']['layer2_n_passing']}/8`",
        f"- mean reversion pass: `{results['mean_reversion']['overall_pass']}`",
        f"- pathwise max-jump KS: `{results['pathwise_jump_realism']['pathwise_max_jump']['ks_stat']:.3f}`",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--state_scope", choices=["iv_only"], default="iv_only")
    parser.add_argument("--calibration_split", choices=["train", "train_tail"], default="train_tail")
    parser.add_argument("--eval_split", choices=["val", "train_tail"], default="val")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--n_buckets", type=int, default=3)
    parser.add_argument("--calibration_quantile", type=float, default=0.90)
    parser.add_argument("--min_scale", type=float, default=1.0)
    parser.add_argument("--max_scale", type=float, default=1.5)
    parser.add_argument("--location_shrink", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=759)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    t0 = time.time()
    cal_samples, cal_batch, cal_alignment = generate_split(args, model, payload, args.calibration_split, device)
    eval_samples, eval_batch, eval_alignment = generate_split(args, model, payload, args.eval_split, device)
    cal_current = cal_batch.history_01.detach().cpu().numpy()[:, -1]
    eval_current = eval_batch.history_01.detach().cpu().numpy()[:, -1]
    cal_target = cal_batch.future_01.detach().cpu().numpy()
    calibration = fit_state_local_calibration(
        cal_samples,
        cal_target,
        cal_current,
        n_buckets=int(args.n_buckets),
        quantile=float(args.calibration_quantile),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        location_shrink=float(args.location_shrink),
        eps=float(args.eps),
    )
    calibrated_samples = apply_state_local_calibration(
        eval_samples,
        eval_current,
        calibration,
        lower=float(args.iv_lower_bound),
        upper=float(args.iv_upper_bound),
    )
    hist_norm_np = eval_batch.history_norm.detach().cpu().numpy()[: calibrated_samples.shape[0]]
    samples_by_key = {history_key(hist_norm_np[i]): calibrated_samples[i] for i in range(calibrated_samples.shape[0])}
    fixed_model = FixedDeployableSampler(samples_by_key).eval()
    results = run_suite(
        cond_samples=calibrated_samples,
        batch=eval_batch,
        model=fixed_model,
        data_path=args.data_path,
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        batch_size=int(args.batch_size),
        conditionality_samples=int(args.conditionality_samples),
        conditionality_max_batches=int(args.conditionality_max_batches),
        device=device,
        eval_split=args.eval_split,
    )
    config = {
        "mode": "759a_state_local_support_calibration",
        "checkpoint": args.checkpoint,
        "calibration_split": args.calibration_split,
        "eval_split": args.eval_split,
        "n_windows": int(calibrated_samples.shape[0]),
        "samples": int(args.samples),
        "n_steps": int(args.n_steps),
        "n_buckets": int(args.n_buckets),
        "calibration_quantile": float(args.calibration_quantile),
        "min_scale": float(args.min_scale),
        "max_scale": float(args.max_scale),
        "location_shrink": float(args.location_shrink),
        "calibration_scale_mean": float(np.mean(calibration["scale"])),
        "calibration_scale_max": float(np.max(calibration["scale"])),
        "calibration_abs_loc_mean": float(np.mean(np.abs(calibration["loc"]))),
        "generation_time_s": float(time.time() - t0),
        "calibration_alignment": cal_alignment,
        "eval_alignment": eval_alignment,
    }
    results["config"] = config
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown_summary(out_md, "759a State-Local Support Calibration", summarize(results, config))
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
