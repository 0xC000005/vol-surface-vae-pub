#!/usr/bin/env python
"""499a: deployable center-occupancy plus asymmetric tail policy.

498a showed that median-locked residual calibration preserves conditionality but cannot
fix level KS because it leaves the 392a center law unchanged. This evaluator adds one
history-only policy component: a pre-validation quantile map from 392a sample medians to
realized future levels, then reattaches asymmetric residual tails around the shifted
center.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_403a_calibrated_risk_system import (  # noqa: E402
    build_recent_calibration_batch,
    sample_native,
    strictly_increasing,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.evaluate_498a_asymmetric_tail_policy import (  # noqa: E402
    GRID,
    HORIZON,
    apply_asymmetric_tail_policy,
    fit_asymmetric_tail_policy,
)


@dataclass(frozen=True)
class CenterQuantileMap:
    levels: np.ndarray
    center_quantiles: np.ndarray
    target_quantiles: np.ndarray
    alpha: float


def fit_center_quantile_map(
    samples: np.ndarray,
    future: np.ndarray,
    n_quantiles: int,
    alpha: float,
) -> tuple[CenterQuantileMap, dict[str, Any]]:
    levels = np.linspace(0.001, 0.999, int(n_quantiles), dtype=np.float64)
    center = np.median(samples, axis=1)
    center_q = np.zeros((HORIZON, GRID, GRID, len(levels)), dtype=np.float32)
    target_q = np.zeros_like(center_q)
    for t in range(HORIZON):
        for i in range(GRID):
            for j in range(GRID):
                center_q[t, i, j] = strictly_increasing(
                    np.quantile(center[:, t, i, j], levels)
                )
                target_q[t, i, j] = strictly_increasing(
                    np.quantile(future[:, t, i, j], levels)
                )
    mapped = apply_center_quantile_map_to_center(center, CenterQuantileMap(levels, center_q, target_q, alpha))
    summary = {
        "alpha": float(alpha),
        "n_quantiles": int(n_quantiles),
        "center_mean_abs_shift": float(np.abs(mapped - center).mean()),
        "center_p95_abs_shift": float(np.quantile(np.abs(mapped - center), 0.95)),
    }
    return CenterQuantileMap(levels, center_q, target_q, float(alpha)), summary


def apply_center_quantile_map_to_center(
    center: np.ndarray,
    qmap: CenterQuantileMap,
) -> np.ndarray:
    mapped = np.empty_like(center, dtype=np.float32)
    for t in range(center.shape[1]):
        for i in range(center.shape[2]):
            for j in range(center.shape[3]):
                values = center[:, t, i, j].astype(np.float64)
                src = qmap.center_quantiles[t, i, j]
                tgt = qmap.target_quantiles[t, i, j]
                u = np.interp(values, src, qmap.levels, left=qmap.levels[0], right=qmap.levels[-1])
                out = np.interp(u, qmap.levels, tgt)
                mapped[:, t, i, j] = ((1.0 - qmap.alpha) * values + qmap.alpha * out).astype(np.float32)
    return np.clip(mapped, 0.0, 1.0)


def shift_samples_to_center(samples: np.ndarray, new_center: np.ndarray) -> np.ndarray:
    old_center = np.median(samples, axis=1, keepdims=True)
    shifted = new_center[:, None] + (samples - old_center)
    shifted_median = np.median(shifted, axis=1, keepdims=True)
    shifted = shifted - shifted_median + new_center[:, None]
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--calibration_windows", type=int, default=441)
    parser.add_argument("--calibration_samples", type=int, default=48)
    parser.add_argument("--center_quantiles", type=int, default=101)
    parser.add_argument("--center_alpha", type=float, default=1.0)
    parser.add_argument("--min_scale", type=float, default=0.70)
    parser.add_argument("--max_scale", type=float, default=1.45)
    parser.add_argument("--n_scale_candidates", type=int, default=31)
    parser.add_argument("--target_tail", type=float, default=0.05)
    parser.add_argument("--scale_penalty", type=float, default=0.001)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--seed", type=int, default=499)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--policy_json", required=True)
    args = parser.parse_args()

    if args.future_len != HORIZON:
        raise ValueError(f"499a currently expects future_len={HORIZON}")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    base_model.eval()

    calib_hist_01, calib_hist_norm, calib_future = build_recent_calibration_batch(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        calibration_windows=args.calibration_windows,
        device=device,
    )
    print("Sampling pre-validation calibration forecasts")
    calib_samples = sample_native(
        model=base_model,
        history_norm=calib_hist_norm,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    calib_future_np = calib_future.detach().cpu().numpy()
    center_map, center_summary = fit_center_quantile_map(
        calib_samples,
        calib_future_np,
        n_quantiles=args.center_quantiles,
        alpha=args.center_alpha,
    )
    shifted_calib_center = apply_center_quantile_map_to_center(
        np.median(calib_samples, axis=1),
        center_map,
    )
    shifted_calib_samples = shift_samples_to_center(calib_samples, shifted_calib_center)
    tail_policy, tail_summary = fit_asymmetric_tail_policy(
        calib_samples=shifted_calib_samples,
        calib_future=calib_future_np,
        calib_history_01=calib_hist_01.detach().cpu().numpy(),
        candidates=np.linspace(args.min_scale, args.max_scale, args.n_scale_candidates, dtype=np.float64),
        target_tail=args.target_tail,
        scale_penalty=args.scale_penalty,
        regime_bins=bool(args.regime_bins),
        min_bin_windows=args.min_bin_windows,
        lock_median=True,
    )

    policy_path = Path(args.policy_json)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(
            make_serializable(
                {
                    "config": vars(args),
                    "checkpoint_epoch": int(payload.get("epoch", -1)),
                    "center_summary": center_summary,
                    "tail_summary": tail_summary,
                    "center_levels": center_map.levels.tolist(),
                    "center_quantiles": center_map.center_quantiles.tolist(),
                    "target_quantiles": center_map.target_quantiles.tolist(),
                    "lower_scale": tail_policy.lower_scale.tolist(),
                    "upper_scale": tail_policy.upper_scale.tolist(),
                    "vov_q20": tail_policy.vov_q20,
                    "vov_q80": tail_policy.vov_q80,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

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
    print("Sampling validation learned core")
    val_base_samples = sample_native(
        model=base_model,
        history_norm=batch.history_norm,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    shifted_val_center = apply_center_quantile_map_to_center(
        np.median(val_base_samples, axis=1),
        center_map,
    )
    shifted_val_samples = shift_samples_to_center(val_base_samples, shifted_val_center)
    cond_samples = apply_asymmetric_tail_policy(
        shifted_val_samples,
        batch.history_01.detach().cpu().numpy(),
        tail_policy,
    )

    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(hist_norm_np.shape[0])}
    model = FixedDeployableSampler(samples_by_key).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=model,
        data_path=args.data_path,
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
        batch_size=args.batch_size,
        conditionality_samples=args.conditionality_samples,
        conditionality_max_batches=args.conditionality_max_batches,
        device=device,
    )

    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size
    results["config"] = {
        "model_type": "499a_center_occupancy_tail_policy",
        "base_model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "deployability": {
            "kind": "center_occupancy_quantile_map_plus_asymmetric_tail_policy",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_validation_tuned_weights": False,
            "calibration_split": "pre_validation",
            "calibration_windows": int(calib_hist_01.shape[0]),
            "calibration_samples": int(args.calibration_samples),
            "risk_policy_not_base_learned_law": True,
            "report_base_model_separately": True,
        },
        "center_summary": center_summary,
        "tail_summary": tail_summary,
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": int(args.samples),
        "conditionality_samples": int(args.conditionality_samples),
        "rollout_start": int(rollout_start),
        "seed": int(args.seed),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")

    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    distributional = results["distributional_fidelity"]
    regime = results["regime_coverage"]
    cross_cell = results["cross_cell_correlation"]
    mean_rev = results["mean_reversion"]
    pathwise = results["pathwise_jump_realism"]
    cointegration = results["cointegration"]
    lines = [
        "- policy: `center occupancy quantile map + asymmetric tail residual scales`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- uses validation future: `{results['config']['deployability']['uses_validation_future']}`",
        f"- center alpha: `{args.center_alpha:.3f}`",
        f"- center mean/p95 abs shift: `{center_summary['center_mean_abs_shift']:.4f}` / `{center_summary['center_p95_abs_shift']:.4f}`",
        f"- regime tail bins: `{bool(args.regime_bins)}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- regime layer2: `{regime['layer2_n_passing']}/{regime['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- median-bias cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- cointegration worst-cell ratio: `{cointegration.get('worst_cell_ratio', float('nan')):.3f}`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "499a Center Occupancy + Tail Policy", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
