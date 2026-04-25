#!/usr/bin/env python
"""501a: state-conditional ridge center residual policy plus tail calibration."""

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
from experiments.backfill.block_ar.evaluate_499a_center_occupancy_tail_policy import (  # noqa: E402
    shift_samples_to_center,
)


@dataclass(frozen=True)
class RidgeCenterPolicy:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    alpha: float
    ridge_lambda: float
    clip_abs: float


def build_history_features(history_01: np.ndarray) -> np.ndarray:
    """Low-dimensional history summary features for center residual prediction."""
    last = history_01[:, -1].reshape(history_01.shape[0], -1)
    mean = history_01.mean(axis=1).reshape(history_01.shape[0], -1)
    trend = (history_01[:, -1] - history_01[:, 0]).reshape(history_01.shape[0], -1)
    diff_std = np.diff(history_01, axis=1).std(axis=1).reshape(history_01.shape[0], -1)
    global_mean = history_01.mean(axis=(2, 3))
    global_diff = np.diff(global_mean, axis=1)
    global_feats = np.stack(
        [
            global_mean[:, -1],
            global_mean.mean(axis=1),
            global_mean[:, -1] - global_mean[:, 0],
            global_diff.std(axis=1),
            global_diff.mean(axis=1),
        ],
        axis=1,
    )
    return np.concatenate([last, mean, trend, diff_std, global_feats], axis=1).astype(np.float64)


def _standardize_features(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    z = (features - mean) / std
    return z, mean, std


def _fit_ridge(features: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, mean, std = _standardize_features(features)
    x = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ targets)
    return weights.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _predict_ridge(features: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z, _, _ = _standardize_features(features, mean.astype(np.float64), std.astype(np.float64))
    x = np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)
    return (x @ weights.astype(np.float64)).astype(np.float32)


def _parse_alpha_candidates(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def fit_ridge_center_policy(
    calib_history_01: np.ndarray,
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    ridge_lambda: float,
    alpha_candidates: list[float],
    holdout_frac: float,
    clip_quantile: float,
) -> tuple[RidgeCenterPolicy, dict[str, Any]]:
    features = build_history_features(calib_history_01)
    base_center = np.median(calib_samples, axis=1)
    target_residual = (calib_future - base_center).reshape(calib_future.shape[0], -1)
    n = features.shape[0]
    holdout_n = max(32, int(round(n * float(holdout_frac))))
    split = n - holdout_n
    if split <= 32:
        raise ValueError("not enough calibration windows for chronological holdout")

    weights_h, mean_h, std_h = _fit_ridge(features[:split], target_residual[:split], ridge_lambda)
    pred_h = _predict_ridge(features[split:], weights_h, mean_h, std_h).reshape(
        holdout_n, HORIZON, GRID, GRID
    )
    hold_center = base_center[split:]
    hold_future = calib_future[split:]
    holdout_scores = {}
    best_alpha = 0.0
    best_mae = float("inf")
    for alpha in alpha_candidates:
        pred_center = np.clip(hold_center + float(alpha) * pred_h, 0.0, 1.0)
        mae = float(np.abs(pred_center - hold_future).mean())
        holdout_scores[str(float(alpha))] = mae
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    weights, mean, std = _fit_ridge(features, target_residual, ridge_lambda)
    pred_all = _predict_ridge(features, weights, mean, std)
    clip_abs = float(np.quantile(np.abs(pred_all), float(clip_quantile)))
    policy = RidgeCenterPolicy(
        feature_mean=mean,
        feature_std=std,
        weights=weights,
        alpha=best_alpha,
        ridge_lambda=float(ridge_lambda),
        clip_abs=clip_abs,
    )
    summary = {
        "ridge_lambda": float(ridge_lambda),
        "holdout_frac": float(holdout_frac),
        "holdout_n": int(holdout_n),
        "alpha_candidates": alpha_candidates,
        "selected_alpha": float(best_alpha),
        "holdout_mae_by_alpha": holdout_scores,
        "holdout_base_mae": holdout_scores.get("0.0", float("nan")),
        "holdout_best_mae": float(best_mae),
        "clip_quantile": float(clip_quantile),
        "clip_abs": clip_abs,
        "pred_abs_mean": float(np.abs(pred_all).mean()),
        "pred_abs_p95": float(np.quantile(np.abs(pred_all), 0.95)),
        "n_features": int(features.shape[1]),
    }
    return policy, summary


def apply_ridge_center_policy(
    history_01: np.ndarray,
    samples: np.ndarray,
    policy: RidgeCenterPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    features = build_history_features(history_01)
    pred = _predict_ridge(features, policy.weights, policy.feature_mean, policy.feature_std)
    pred = pred.reshape(history_01.shape[0], HORIZON, GRID, GRID)
    pred = np.clip(pred, -policy.clip_abs, policy.clip_abs)
    base_center = np.median(samples, axis=1)
    shifted_center = np.clip(base_center + float(policy.alpha) * pred, 0.0, 1.0)
    shifted_samples = shift_samples_to_center(samples, shifted_center)
    return shifted_samples, shifted_center


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
    parser.add_argument("--ridge_lambda", type=float, default=10.0)
    parser.add_argument("--alpha_candidates", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--holdout_frac", type=float, default=0.30)
    parser.add_argument("--clip_quantile", type=float, default=0.99)
    parser.add_argument("--min_scale", type=float, default=0.70)
    parser.add_argument("--max_scale", type=float, default=1.45)
    parser.add_argument("--n_scale_candidates", type=int, default=31)
    parser.add_argument("--target_tail", type=float, default=0.05)
    parser.add_argument("--scale_penalty", type=float, default=0.001)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--policy_json", required=True)
    args = parser.parse_args()

    if args.future_len != HORIZON:
        raise ValueError(f"501a currently expects future_len={HORIZON}")

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
    calib_hist_np = calib_hist_01.detach().cpu().numpy()
    calib_future_np = calib_future.detach().cpu().numpy()
    center_policy, center_summary = fit_ridge_center_policy(
        calib_history_01=calib_hist_np,
        calib_samples=calib_samples,
        calib_future=calib_future_np,
        ridge_lambda=args.ridge_lambda,
        alpha_candidates=_parse_alpha_candidates(args.alpha_candidates),
        holdout_frac=args.holdout_frac,
        clip_quantile=args.clip_quantile,
    )
    shifted_calib_samples, shifted_calib_center = apply_ridge_center_policy(
        calib_hist_np,
        calib_samples,
        center_policy,
    )
    tail_policy, tail_summary = fit_asymmetric_tail_policy(
        calib_samples=shifted_calib_samples,
        calib_future=calib_future_np,
        calib_history_01=calib_hist_np,
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
                    "center_feature_mean": center_policy.feature_mean.tolist(),
                    "center_feature_std": center_policy.feature_std.tolist(),
                    "center_weights": center_policy.weights.tolist(),
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
    shifted_val_samples, shifted_val_center = apply_ridge_center_policy(
        batch.history_01.detach().cpu().numpy(),
        val_base_samples,
        center_policy,
    )
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
    val_shift = shifted_val_center - np.median(val_base_samples, axis=1)
    results["config"] = {
        "model_type": "501a_state_conditional_center_policy",
        "base_model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "deployability": {
            "kind": "state_conditional_ridge_center_residual_plus_asymmetric_tail_policy",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_validation_tuned_weights": False,
            "calibration_split": "pre_validation",
            "shrinkage_selection": "chronological_pre_validation_holdout_mae",
            "risk_policy_not_base_learned_law": True,
            "report_base_model_separately": True,
        },
        "center_summary": {
            **center_summary,
            "validation_shift_abs_mean": float(np.abs(val_shift).mean()),
            "validation_shift_abs_p95": float(np.quantile(np.abs(val_shift), 0.95)),
        },
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
        "- policy: `state-conditional ridge center residual + asymmetric tail scales`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- selected center alpha: `{center_summary['selected_alpha']:.3f}`",
        f"- holdout base/best MAE: `{center_summary['holdout_base_mae']:.5f}` / `{center_summary['holdout_best_mae']:.5f}`",
        f"- validation shift mean/p95 abs: `{results['config']['center_summary']['validation_shift_abs_mean']:.4f}` / `{results['config']['center_summary']['validation_shift_abs_p95']:.4f}`",
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
    write_markdown_summary(args.output_md, "501a State-Conditional Center Policy", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
