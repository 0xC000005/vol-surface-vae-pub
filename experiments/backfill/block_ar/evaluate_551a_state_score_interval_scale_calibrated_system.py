#!/usr/bin/env python
"""551a: history-state interval scaling for the frozen learned-law frontier."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    HistoryFutureDictDataset,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_220h_full_multihorizon_v2_suite import (  # noqa: E402
    suite_summary,
)
from experiments.backfill.block_ar.evaluate_403a_calibrated_risk_system import (  # noqa: E402
    build_recent_calibration_batch,
    sample_native,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (  # noqa: E402
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv  # noqa: E402


@dataclass
class StateScaleTables:
    scales: np.ndarray
    low_q: float
    high_q: float
    bin_low_quantile: float
    bin_high_quantile: float
    target_coverage: float
    coverage_lo: float
    coverage_hi: float
    scale_min: float
    scale_max: float
    scale_steps: int
    min_bin_windows: int
    state_feature: str = "history_abs_move_q90"


def state_score_from_history(history_01: np.ndarray) -> np.ndarray:
    """Risk-state score from history only: 90th percentile absolute daily move."""
    history_01 = np.asarray(history_01, dtype=np.float64)
    if history_01.ndim != 4:
        raise ValueError("history_01 must have shape (windows, history, rows, cols)")
    diffs = np.abs(np.diff(history_01, axis=1))
    if diffs.shape[1] == 0:
        return np.zeros(history_01.shape[0], dtype=np.float64)
    return np.quantile(diffs.reshape(history_01.shape[0], -1), 0.90, axis=1)


def assign_state_bins(score: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    bins = np.ones(score.shape[0], dtype=np.int64)
    bins[score <= float(low_q)] = 0
    bins[score >= float(high_q)] = 2
    return bins


def interval_coverage_at_scale(samples: np.ndarray, target: np.ndarray, scale: float) -> float:
    median = np.median(samples, axis=1)
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    lo = median + float(scale) * (q05 - median)
    hi = median + float(scale) * (q95 - median)
    return float(((target >= lo) & (target <= hi)).mean())


def _scale_objective(cov: float, scale: float, target: float, lo: float, hi: float) -> float:
    obj = abs(float(cov) - float(target))
    obj += 0.35 * max(0.0, float(cov) - float(hi))
    obj += 0.35 * max(0.0, float(lo) - float(cov))
    obj += 0.002 * abs(float(scale) - 1.0)
    return float(obj)


def fit_state_horizon_scale_tables(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    calib_history: np.ndarray,
    target_coverage: float,
    coverage_lo: float,
    coverage_hi: float,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    min_bin_windows: int,
    bin_low_quantile: float = 1.0 / 3.0,
    bin_high_quantile: float = 2.0 / 3.0,
) -> StateScaleTables:
    """Fit one residual scale per state bin and horizon on calibration windows."""
    if calib_samples.ndim != 5:
        raise ValueError("calib_samples must have shape (windows, samples, horizon, rows, cols)")
    if calib_future.ndim != 4:
        raise ValueError("calib_future must have shape (windows, horizon, rows, cols)")
    if calib_samples.shape[0] != calib_future.shape[0] or calib_samples.shape[0] != calib_history.shape[0]:
        raise ValueError("calibration arrays must share the same window dimension")
    if not 0.0 < bin_low_quantile < bin_high_quantile < 1.0:
        raise ValueError("state bin quantiles must satisfy 0 < low < high < 1")

    score = state_score_from_history(calib_history)
    low_q = float(np.quantile(score, bin_low_quantile))
    high_q = float(np.quantile(score, bin_high_quantile))
    bins = assign_state_bins(score, low_q=low_q, high_q=high_q)
    horizon = int(calib_future.shape[1])
    scales = np.ones((3, horizon), dtype=np.float32)
    candidates = np.linspace(float(scale_min), float(scale_max), int(scale_steps), dtype=np.float64)
    all_idx = np.arange(calib_history.shape[0])

    for bin_id in range(3):
        idx = np.where(bins == bin_id)[0]
        if idx.shape[0] < int(min_bin_windows):
            idx = all_idx
        for t in range(horizon):
            samples_t = calib_samples[idx, :, t]
            target_t = calib_future[idx, t]
            best_scale = 1.0
            best_obj = float("inf")
            for scale in candidates:
                cov = interval_coverage_at_scale(samples_t, target_t, float(scale))
                obj = _scale_objective(
                    cov=cov,
                    scale=float(scale),
                    target=float(target_coverage),
                    lo=float(coverage_lo),
                    hi=float(coverage_hi),
                )
                if obj < best_obj:
                    best_obj = obj
                    best_scale = float(scale)
            scales[bin_id, t] = best_scale

    return StateScaleTables(
        scales=scales,
        low_q=low_q,
        high_q=high_q,
        bin_low_quantile=float(bin_low_quantile),
        bin_high_quantile=float(bin_high_quantile),
        target_coverage=float(target_coverage),
        coverage_lo=float(coverage_lo),
        coverage_hi=float(coverage_hi),
        scale_min=float(scale_min),
        scale_max=float(scale_max),
        scale_steps=int(scale_steps),
        min_bin_windows=int(min_bin_windows),
    )


def apply_state_interval_scaling(
    samples: np.ndarray,
    history_01: np.ndarray,
    tables: StateScaleTables,
    alpha: float,
) -> np.ndarray:
    """Scale ensemble residuals around each window's sample median by state bin."""
    score = state_score_from_history(history_01)
    bins = assign_state_bins(score, low_q=tables.low_q, high_q=tables.high_q)
    median = np.median(samples, axis=1, keepdims=True)
    calibrated = np.empty_like(samples, dtype=np.float32)
    for window_idx in range(samples.shape[0]):
        bin_id = int(bins[window_idx])
        horizon_scale = tables.scales[bin_id].reshape(1, -1, 1, 1)
        eff_scale = 1.0 + float(alpha) * (horizon_scale - 1.0)
        out = median[window_idx] + eff_scale * (samples[window_idx] - median[window_idx])
        calibrated[window_idx] = np.clip(out, 0.0, 1.0).astype(np.float32)
    return calibrated


class StateIntervalScaleSampler:
    def __init__(
        self,
        base_model: torch.nn.Module,
        tables: StateScaleTables,
        alpha: float,
        device: torch.device,
    ):
        self.base_model = base_model
        self.tables = tables
        self.alpha = float(alpha)
        self.device = device

    def eval(self) -> "StateIntervalScaleSampler":
        self.base_model.eval()
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 8,
        history_is_normalized: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        history_device = history.device
        history_base = history.to(self.device)
        with torch.no_grad():
            raw = self.base_model.sample_batched(
                history_base,
                n_samples=n_samples,
                n_steps=n_steps,
                chunk_size=chunk_size,
                history_is_normalized=history_is_normalized,
                **kwargs,
            )
        if history_is_normalized:
            history_01 = denormalize_iv(history_base).detach().cpu().numpy()
        else:
            history_01 = history_base.detach().cpu().numpy()
        calibrated = apply_state_interval_scaling(
            samples=raw.detach().cpu().numpy(),
            history_01=history_01,
            tables=self.tables,
            alpha=self.alpha,
        )
        return torch.from_numpy(calibrated).to(history_device)


def _bin_counts(history_01: np.ndarray, tables: StateScaleTables) -> dict[str, int]:
    bins = assign_state_bins(
        state_score_from_history(history_01),
        low_q=tables.low_q,
        high_q=tables.high_q,
    )
    return {str(bin_id): int((bins == bin_id).sum()) for bin_id in range(3)}


def _evaluate_suite(
    model: StateIntervalScaleSampler,
    batch: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    outputs: list[np.ndarray] = []
    for start in range(0, batch.history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, batch.history_01.shape[0])
        with torch.no_grad():
            samples = model.sample_batched(
                batch.history_norm[start:end],
                n_samples=args.samples,
                n_steps=args.future_len,
                chunk_size=args.chunk_size,
                history_is_normalized=True,
            )
        outputs.append(samples.detach().cpu().numpy())
    cond_samples = np.concatenate(outputs, axis=0)
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
        model,
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

    return {
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
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--calibration_windows", type=int, default=441)
    parser.add_argument("--calibration_samples", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--target_coverage", type=float, default=0.88)
    parser.add_argument("--coverage_lo", type=float, default=0.70)
    parser.add_argument("--coverage_hi", type=float, default=0.95)
    parser.add_argument("--scale_min", type=float, default=0.75)
    parser.add_argument("--scale_max", type=float, default=1.45)
    parser.add_argument("--scale_steps", type=int, default=29)
    parser.add_argument("--min_bin_windows", type=int, default=80)
    parser.add_argument("--bin_low_quantile", type=float, default=1.0 / 3.0)
    parser.add_argument("--bin_high_quantile", type=float, default=2.0 / 3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

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
    print("Sampling frozen base model on pre-validation calibration block")
    calib_samples = sample_native(
        model=base_model,
        history_norm=calib_hist_norm,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    tables = fit_state_horizon_scale_tables(
        calib_samples=calib_samples,
        calib_future=calib_future.detach().cpu().numpy(),
        calib_history=calib_hist_01.detach().cpu().numpy(),
        target_coverage=args.target_coverage,
        coverage_lo=args.coverage_lo,
        coverage_hi=args.coverage_hi,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_steps=args.scale_steps,
        min_bin_windows=args.min_bin_windows,
        bin_low_quantile=args.bin_low_quantile,
        bin_high_quantile=args.bin_high_quantile,
    )
    print(
        "State-scale summary:",
        float(tables.scales.min()),
        float(np.median(tables.scales)),
        float(tables.scales.max()),
    )

    model = StateIntervalScaleSampler(base_model, tables, alpha=args.alpha, device=device).eval()
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
    results = _evaluate_suite(model, batch, args, device)
    n_pass, failed = suite_summary(results)
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}
    results["config"] = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "calibration": {
            "kind": "pre_validation_state_score_interval_scale",
            "state_feature": tables.state_feature,
            "calibration_windows": args.calibration_windows,
            "calibration_samples": args.calibration_samples,
            "alpha": args.alpha,
            "target_coverage": args.target_coverage,
            "coverage_lo": args.coverage_lo,
            "coverage_hi": args.coverage_hi,
            "bin_low_quantile": tables.bin_low_quantile,
            "bin_high_quantile": tables.bin_high_quantile,
            "low_q": tables.low_q,
            "high_q": tables.high_q,
            "calibration_bin_counts": _bin_counts(calib_hist_01.detach().cpu().numpy(), tables),
            "validation_bin_counts": _bin_counts(batch.history_01.detach().cpu().numpy(), tables),
            "scale_min": float(tables.scales.min()),
            "scale_median": float(np.median(tables.scales)),
            "scale_max": float(tables.scales.max()),
            "scale_by_bin_mean": [float(x) for x in tables.scales.mean(axis=1)],
        },
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": args.samples,
        "conditionality_samples": args.conditionality_samples,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    lines = [
        f"- base model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        "- calibration: `pre-validation state-score interval scale`",
        f"- state feature: `{tables.state_feature}`",
        f"- alpha / target coverage: `{args.alpha:.2f}` / `{args.target_coverage:.2f}`",
        f"- state thresholds: `{tables.low_q:.5f}` / `{tables.high_q:.5f}`",
        f"- scale range: `{float(tables.scales.min()):.3f} / {float(np.median(tables.scales)):.3f} / {float(tables.scales.max()):.3f}`",
        f"- scale mean by state bin: `{[round(float(x), 3) for x in tables.scales.mean(axis=1)]}`",
        f"- validation bin counts: `{_bin_counts(batch.history_01.detach().cpu().numpy(), tables)}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{results['coverage']['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{results['coverage']['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- turb/calm ratio: `{results['conditionality'].get('turb_calm_ratio', float('nan')):.3f}`",
        f"- conditional MAE reduction: `{results['conditionality'].get('mae_reduction_pct', float('nan')):.3f}`",
        "",
        "**Hard Suites**",
        f"- coverage overall: `{results['coverage']['overall_pass']}`",
        f"- regime coverage layer2: `{results['regime_coverage']['layer2_n_passing']}/{results['regime_coverage']['layer2_n_total']}`",
        f"- level KS pass cells: `{results['distributional_fidelity']['ks_level_test']['n_pass']}/25`",
        f"- max-jump KS: `{results['pathwise_jump_realism']['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "551a State-Score Interval-Scale Calibrated Validation", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
