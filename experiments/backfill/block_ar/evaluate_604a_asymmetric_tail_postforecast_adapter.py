#!/usr/bin/env python
"""604a: asymmetric postforecast tail adapter for frozen 510a samples."""

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
class AsymmetricTailTables:
    lower_scales: np.ndarray
    upper_scales: np.ndarray
    lower_tail_target: float
    upper_tail_target: float
    scale_min: float
    scale_max: float
    scale_steps: int
    penalty: float


def _as_scale_array(scale: float | np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(scale, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(target_shape, float(arr), dtype=np.float64)
    return np.broadcast_to(arr, target_shape).astype(np.float64)


def tail_miss_rates(
    samples: np.ndarray,
    target: np.ndarray,
    *,
    lower_scale: float | np.ndarray,
    upper_scale: float | np.ndarray,
) -> dict[str, float]:
    """Return lower/upper miss rates after asymmetric scaling around the median."""
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 5:
        raise ValueError("samples must have shape [windows, samples, horizon, rows, cols]")
    if y.shape != x.shape[:1] + x.shape[2:]:
        raise ValueError("target must have shape [windows, horizon, rows, cols]")
    median = np.median(x, axis=1)
    q05 = np.quantile(x, 0.05, axis=1)
    q95 = np.quantile(x, 0.95, axis=1)
    lower = median + _as_scale_array(lower_scale, median.shape) * (q05 - median)
    upper = median + _as_scale_array(upper_scale, median.shape) * (q95 - median)
    lower_miss = y < lower
    upper_miss = y > upper
    covered = ~(lower_miss | upper_miss)
    return {
        "lower_miss": float(np.mean(lower_miss)),
        "upper_miss": float(np.mean(upper_miss)),
        "coverage": float(np.mean(covered)),
    }


def _tail_objective(miss_rate: float, scale: float, target: float, penalty: float) -> float:
    return max(0.0, float(miss_rate) - float(target)) ** 2 + float(penalty) * abs(float(scale) - 1.0)


def fit_asymmetric_tail_scale_tables(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    *,
    lower_tail_target: float,
    upper_tail_target: float,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    penalty: float = 0.001,
) -> AsymmetricTailTables:
    """Fit independent lower/upper widening scales per horizon/cell."""
    x = np.asarray(calib_samples, dtype=np.float64)
    y = np.asarray(calib_future, dtype=np.float64)
    if x.ndim != 5:
        raise ValueError("calib_samples must have shape [windows, samples, horizon, rows, cols]")
    if y.shape != x.shape[:1] + x.shape[2:]:
        raise ValueError("calib_future must have shape [windows, horizon, rows, cols]")
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("scale bounds must satisfy 0 < scale_min <= scale_max")
    candidates = np.linspace(float(scale_min), float(scale_max), int(scale_steps), dtype=np.float64)
    horizon, rows, cols = y.shape[1:]
    lower_scales = np.ones((horizon, rows, cols), dtype=np.float32)
    upper_scales = np.ones((horizon, rows, cols), dtype=np.float32)

    median = np.median(x, axis=1)
    q05 = np.quantile(x, 0.05, axis=1)
    q95 = np.quantile(x, 0.95, axis=1)
    for t in range(horizon):
        for row in range(rows):
            for col in range(cols):
                med = median[:, t, row, col]
                low_base = q05[:, t, row, col] - med
                high_base = q95[:, t, row, col] - med
                target = y[:, t, row, col]
                best_lower = 1.0
                best_lower_obj = float("inf")
                best_upper = 1.0
                best_upper_obj = float("inf")
                for scale in candidates:
                    lower = med + float(scale) * low_base
                    lower_miss = float(np.mean(target < lower))
                    lower_obj = _tail_objective(lower_miss, float(scale), lower_tail_target, penalty)
                    if lower_obj < best_lower_obj:
                        best_lower_obj = lower_obj
                        best_lower = float(scale)

                    upper = med + float(scale) * high_base
                    upper_miss = float(np.mean(target > upper))
                    upper_obj = _tail_objective(upper_miss, float(scale), upper_tail_target, penalty)
                    if upper_obj < best_upper_obj:
                        best_upper_obj = upper_obj
                        best_upper = float(scale)
                lower_scales[t, row, col] = best_lower
                upper_scales[t, row, col] = best_upper

    return AsymmetricTailTables(
        lower_scales=lower_scales,
        upper_scales=upper_scales,
        lower_tail_target=float(lower_tail_target),
        upper_tail_target=float(upper_tail_target),
        scale_min=float(scale_min),
        scale_max=float(scale_max),
        scale_steps=int(scale_steps),
        penalty=float(penalty),
    )


def apply_asymmetric_tail_scaling(
    samples: np.ndarray,
    lower_scales: np.ndarray,
    upper_scales: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Median-preserving asymmetric residual scaling."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 5:
        raise ValueError("samples must have shape [windows, samples, horizon, rows, cols]")
    lower = np.asarray(lower_scales, dtype=np.float64)
    upper = np.asarray(upper_scales, dtype=np.float64)
    if lower.shape != x.shape[2:] or upper.shape != x.shape[2:]:
        raise ValueError("scale tables must have shape [horizon, rows, cols]")
    med = np.median(x, axis=1, keepdims=True)
    lower_eff = 1.0 + float(alpha) * (lower[None, None] - 1.0)
    upper_eff = 1.0 + float(alpha) * (upper[None, None] - 1.0)
    scale = np.where(x < med, lower_eff, upper_eff)
    out = med + scale * (x - med)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class AsymmetricTailSampler:
    def __init__(
        self,
        base_model: torch.nn.Module,
        tables: AsymmetricTailTables,
        alpha: float,
        device: torch.device,
    ):
        self.base_model = base_model
        self.tables = tables
        self.alpha = float(alpha)
        self.device = device

    def eval(self) -> "AsymmetricTailSampler":
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
        calibrated = apply_asymmetric_tail_scaling(
            raw.detach().cpu().numpy(),
            lower_scales=self.tables.lower_scales,
            upper_scales=self.tables.upper_scales,
            alpha=self.alpha,
        )
        return torch.from_numpy(calibrated).to(history_device)


def _evaluate_suite(
    model: AsymmetricTailSampler,
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

    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return {
        "surface": run_surface_validity_tests(cond_samples, ground_truth),
        "coverage": run_ci_coverage_tests(cond_samples, ground_truth),
        "conditionality": run_conditionality_tests(
            model,
            cond_loader,
            n_samples=args.conditionality_samples,
            max_batches=args.conditionality_max_batches,
            device=str(device),
        ),
        "time_series": run_time_series_tests(cond_samples, ground_truth),
        "block_ar": run_block_ar_tests(cond_samples),
        "cointegration": run_cointegration_tests(
            cond_samples,
            ground_truth,
            returns=returns,
            test_start=rollout_start,
            history_len=args.history_len,
            future_len=args.future_len,
        ),
        "regime_coverage": run_regime_coverage_tests(cond_samples, ground_truth, history_01),
        "distributional_fidelity": run_distributional_fidelity_tests(cond_samples, ground_truth, history_01),
        "cross_cell_correlation": run_cross_cell_correlation_tests(cond_samples, ground_truth),
        "mean_reversion": run_mean_reversion_tests(cond_samples, ground_truth, history_01),
        "pathwise_jump_realism": run_pathwise_jump_realism_tests(cond_samples, ground_truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type", default="340c")
    parser.add_argument("--checkpoint", default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--calibration_windows", type=int, default=441)
    parser.add_argument("--calibration_samples", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lower_tail_target", type=float, default=0.025)
    parser.add_argument("--upper_tail_target", type=float, default=0.025)
    parser.add_argument("--scale_min", type=float, default=1.0)
    parser.add_argument("--scale_max", type=float, default=2.5)
    parser.add_argument("--scale_steps", type=int, default=31)
    parser.add_argument("--penalty", type=float, default=0.001)
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
    tables = fit_asymmetric_tail_scale_tables(
        calib_samples=calib_samples,
        calib_future=calib_future.detach().cpu().numpy(),
        lower_tail_target=args.lower_tail_target,
        upper_tail_target=args.upper_tail_target,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_steps=args.scale_steps,
        penalty=args.penalty,
    )
    print(
        "Asymmetric scale summary:",
        float(tables.lower_scales.min()),
        float(np.median(tables.lower_scales)),
        float(tables.lower_scales.max()),
        "|",
        float(tables.upper_scales.min()),
        float(np.median(tables.upper_scales)),
        float(tables.upper_scales.max()),
    )

    model = AsymmetricTailSampler(base_model, tables, alpha=args.alpha, device=device).eval()
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
            "kind": "pre_validation_asymmetric_tail_postforecast_adapter",
            "calibration_windows": args.calibration_windows,
            "calibration_samples": args.calibration_samples,
            "alpha": args.alpha,
            "lower_tail_target": args.lower_tail_target,
            "upper_tail_target": args.upper_tail_target,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "scale_steps": args.scale_steps,
            "penalty": args.penalty,
            "lower_scale_min": float(tables.lower_scales.min()),
            "lower_scale_median": float(np.median(tables.lower_scales)),
            "lower_scale_max": float(tables.lower_scales.max()),
            "upper_scale_min": float(tables.upper_scales.min()),
            "upper_scale_median": float(np.median(tables.upper_scales)),
            "upper_scale_max": float(tables.upper_scales.max()),
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
        "- calibration: `pre-validation asymmetric tail postforecast adapter`",
        f"- alpha / lower target / upper target: `{args.alpha:.2f}` / `{args.lower_tail_target:.3f}` / `{args.upper_tail_target:.3f}`",
        f"- lower scale range: `{float(tables.lower_scales.min()):.3f} / {float(np.median(tables.lower_scales)):.3f} / {float(tables.lower_scales.max()):.3f}`",
        f"- upper scale range: `{float(tables.upper_scales.min()):.3f} / {float(np.median(tables.upper_scales)):.3f} / {float(tables.upper_scales.max()):.3f}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{results['coverage']['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{results['coverage']['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditional MAE reduction: `{results['conditionality'].get('mae_reduction_pct', float('nan')):.3f}`",
        f"- turb/calm ratio: `{results['conditionality'].get('turb_calm_ratio', float('nan')):.3f}`",
        "",
        "**Hard Suites**",
        f"- regime coverage layer2: `{results['regime_coverage']['layer2_n_passing']}/{results['regime_coverage']['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{results['distributional_fidelity']['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{results['distributional_fidelity']['ks_level_test']['n_pass']}/25`",
        f"- max-jump KS: `{results['pathwise_jump_realism']['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "604a Asymmetric Tail Postforecast Adapter", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
