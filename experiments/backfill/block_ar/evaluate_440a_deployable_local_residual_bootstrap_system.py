#!/usr/bin/env python
"""440a: deployable history-local residual-error bootstrap risk system.

This is the local version of 438a. It keeps the same deployability rule but samples
forecast-error paths from calibration histories nearest to the current history in a
generic panel-history feature space.
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
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)


@dataclass(frozen=True)
class LocalResidualBank:
    errors: np.ndarray
    features_z: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    neighbor_count: int
    feature_recent: int


def build_history_features(history_01: np.ndarray, recent: int) -> np.ndarray:
    """Generic panel-history features, with no IV-specific labels."""
    recent = max(2, min(int(recent), int(history_01.shape[1])))
    last = history_01[:, -1].reshape(history_01.shape[0], -1)
    recent_mean = history_01[:, -recent:].mean(axis=1).reshape(history_01.shape[0], -1)
    recent_change = (history_01[:, -1] - history_01[:, -recent]).reshape(history_01.shape[0], -1)
    diffs = np.diff(history_01[:, -recent:], axis=1)
    realized_var = (diffs**2).mean(axis=(1, 2, 3), keepdims=False)[:, None]
    features = np.concatenate([last, recent_mean, recent_change, realized_var], axis=1)
    return features.astype(np.float64)


def standardize_features(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = features.mean(axis=0, keepdims=True)
    if std is None:
        std = features.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def fit_local_residual_bank(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    calib_history_01: np.ndarray,
    neighbor_count: int,
    feature_recent: int,
) -> tuple[LocalResidualBank, dict[str, Any]]:
    calib_median = np.median(calib_samples, axis=1)
    errors = (calib_future - calib_median).astype(np.float32)
    features = build_history_features(calib_history_01, feature_recent)
    features_z, mean, std = standardize_features(features)
    k = max(1, min(int(neighbor_count), int(calib_history_01.shape[0])))
    bank = LocalResidualBank(
        errors=errors,
        features_z=features_z,
        feature_mean=mean,
        feature_std=std,
        neighbor_count=k,
        feature_recent=int(feature_recent),
    )
    meta = {
        "forecast_error_abs_mean": float(np.abs(errors).mean()),
        "forecast_error_abs_p90": float(np.quantile(np.abs(errors), 0.90)),
        "forecast_error_abs_p99": float(np.quantile(np.abs(errors), 0.99)),
        "forecast_error_std": float(errors.std()),
        "feature_dim": int(features_z.shape[1]),
        "neighbor_count": int(k),
        "feature_recent": int(feature_recent),
    }
    return bank, meta


def nearest_indices(
    bank: LocalResidualBank,
    val_features_z: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    # Squared Euclidean distances in standardized generic history-feature space.
    distances = (
        (val_features_z[:, None, :] - bank.features_z[None, :, :]) ** 2
    ).mean(axis=2)
    k = int(bank.neighbor_count)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    chosen_dist = np.take_along_axis(distances, neighbors, axis=1)
    meta = {
        "neighbor_distance_mean": float(np.sqrt(chosen_dist).mean()),
        "neighbor_distance_p90": float(np.quantile(np.sqrt(chosen_dist), 0.90)),
        "neighbor_distance_max": float(np.sqrt(chosen_dist).max()),
    }
    return neighbors.astype(np.int64), meta


def build_local_deployable_samples(
    val_base_samples: np.ndarray,
    val_history_01: np.ndarray,
    bank: LocalResidualBank,
    residual_shape_scale: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    val_features = build_history_features(val_history_01, bank.feature_recent)
    val_features_z, _, _ = standardize_features(val_features, bank.feature_mean, bank.feature_std)
    neighbors, neighbor_meta = nearest_indices(bank, val_features_z)

    val_median = np.median(val_base_samples, axis=1)
    val_residual_shape = val_base_samples - val_median[:, None]
    n_windows, n_samples, horizon, rows, cols = val_base_samples.shape
    out = np.empty_like(val_base_samples, dtype=np.float32)
    unique_neighbor_counts = []
    for w in range(n_windows):
        pool = neighbors[w]
        picks = rng.choice(pool, size=n_samples, replace=True)
        unique_neighbor_counts.append(int(np.unique(picks).shape[0]))
        sampled_errors = bank.errors[picks, :horizon, :rows, :cols]
        shaped = val_median[w, None] + sampled_errors
        shaped = shaped + float(residual_shape_scale) * val_residual_shape[w]
        out[w] = np.clip(shaped, 0.0, 1.0).astype(np.float32)

    meta: dict[str, Any] = {
        "residual_shape_scale": float(residual_shape_scale),
        "unique_neighbors_per_window_mean": float(np.mean(unique_neighbor_counts)),
        "unique_neighbors_per_window_min": int(np.min(unique_neighbor_counts)),
        "unique_neighbors_per_window_max": int(np.max(unique_neighbor_counts)),
        "sample_min": float(out.min()),
        "sample_median": float(np.median(out)),
        "sample_max": float(out.max()),
        "sample_std": float(out.std()),
        **neighbor_meta,
    }
    return out, meta


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
    parser.add_argument("--neighbor_count", type=int, default=64)
    parser.add_argument("--feature_recent", type=int, default=5)
    parser.add_argument("--residual_shape_scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=440)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

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
    print("Sampling calibration forecasts")
    calib_samples = sample_native(
        model=base_model,
        history_norm=calib_hist_norm,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    bank, bank_meta = fit_local_residual_bank(
        calib_samples=calib_samples,
        calib_future=calib_future.detach().cpu().numpy(),
        calib_history_01=calib_hist_01.detach().cpu().numpy(),
        neighbor_count=args.neighbor_count,
        feature_recent=args.feature_recent,
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
    cond_samples, policy_meta = build_local_deployable_samples(
        val_base_samples=val_base_samples,
        val_history_01=batch.history_01.detach().cpu().numpy(),
        bank=bank,
        residual_shape_scale=args.residual_shape_scale,
        seed=args.seed,
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
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "deployability": {
            "kind": "pre_validation_history_local_residual_error_bootstrap",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_per_window_oracle_miss_placement": False,
            "calibration_split": "pre_validation",
            "calibration_windows": int(args.calibration_windows),
            "calibration_samples": int(args.calibration_samples),
            "base_center": "validation_392a_sample_median",
            "residual_error_source": "nearest_pre_validation_realized_future_minus_392a_sample_median",
            "history_feature_policy": "last_surface + recent_mean + recent_change + realized_history_variance",
            "not_base_learned_law": True,
            "calibrated_risk_policy": True,
            **bank_meta,
            **policy_meta,
        },
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
    lines = [
        "- policy: `pre-validation history-local residual-error bootstrap around frozen 392a median`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- uses validation future: `{results['config']['deployability']['uses_validation_future']}`",
        f"- neighbor count: `{args.neighbor_count}`",
        f"- feature recent window: `{args.feature_recent}`",
        f"- calibration windows/samples: `{args.calibration_windows}` / `{args.calibration_samples}`",
        f"- residual shape scale: `{args.residual_shape_scale:.3f}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- regime layer2: `{regime['layer2_n_passing']}/{regime['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "440a Deployable Local Residual Bootstrap Risk System", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
