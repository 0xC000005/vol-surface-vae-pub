#!/usr/bin/env python
"""405a: evaluate 392a with median-preserving interval-scale calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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
    GRID,
    HORIZON,
    assign_bins,
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
class ScaleTables:
    scales: np.ndarray
    vov_q20: float
    vov_q80: float
    alpha: float
    regime_bins: bool
    scale_objective: str
    target_coverage: float
    coverage_lo: float
    coverage_hi: float


def compute_vov(history_01: np.ndarray) -> np.ndarray:
    mean_iv = history_01.mean(axis=(2, 3))
    return np.diff(mean_iv, axis=1).std(axis=1)


def interval_coverage_at_scale(
    samples: np.ndarray,
    target: np.ndarray,
    scale: float,
) -> float:
    med = np.median(samples, axis=1)
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    lo = med + float(scale) * (q05 - med)
    hi = med + float(scale) * (q95 - med)
    return float(((target >= lo) & (target <= hi)).mean())


def fit_scale_tables(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    calib_history: np.ndarray,
    alpha: float,
    regime_bins: bool,
    min_bin_windows: int,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    scale_objective: str,
    target_coverage: float,
    coverage_lo: float,
    coverage_hi: float,
) -> ScaleTables:
    vov = compute_vov(calib_history)
    vov_q20 = float(np.quantile(vov, 0.2))
    vov_q80 = float(np.quantile(vov, 0.8))
    bins = assign_bins(calib_history, vov_q20, vov_q80, regime_bins)
    n_bins = 3 if regime_bins else 1
    scales = np.ones((n_bins, HORIZON, GRID, GRID), dtype=np.float32)
    candidates = np.linspace(scale_min, scale_max, int(scale_steps), dtype=np.float64)
    all_idx = np.arange(calib_history.shape[0])

    for bin_id in range(n_bins):
        idx = np.where(bins == bin_id)[0]
        if idx.shape[0] < int(min_bin_windows):
            idx = all_idx
        for t in range(HORIZON):
            for i in range(GRID):
                for j in range(GRID):
                    samples = calib_samples[idx, :, t, i, j]
                    target = calib_future[idx, t, i, j]
                    best_scale = 1.0
                    best_obj = float("inf")
                    for scale in candidates:
                        cov = interval_coverage_at_scale(samples, target, float(scale))
                        if scale_objective == "target":
                            # Target 90% while mildly discouraging evaluator edge violations.
                            obj = abs(cov - target_coverage)
                            obj += 0.25 * max(0.0, cov - coverage_hi)
                            obj += 0.25 * max(0.0, coverage_lo - cov)
                        elif scale_objective == "deadband":
                            # Minimal policy intervention: do nothing if already inside the risk band.
                            if coverage_lo <= cov <= coverage_hi:
                                obj = 0.001 * abs(float(scale) - 1.0)
                            elif cov < coverage_lo:
                                obj = (coverage_lo - cov) + 0.001 * abs(float(scale) - 1.0)
                            else:
                                obj = (cov - coverage_hi) + 0.001 * abs(float(scale) - 1.0)
                        else:
                            raise ValueError(f"Unknown scale_objective: {scale_objective}")
                        if obj < best_obj:
                            best_obj = obj
                            best_scale = float(scale)
                    scales[bin_id, t, i, j] = best_scale

    return ScaleTables(
        scales=scales,
        vov_q20=vov_q20,
        vov_q80=vov_q80,
        alpha=float(alpha),
        regime_bins=bool(regime_bins),
        scale_objective=str(scale_objective),
        target_coverage=float(target_coverage),
        coverage_lo=float(coverage_lo),
        coverage_hi=float(coverage_hi),
    )


def apply_interval_scaling(
    samples: np.ndarray,
    history_01: np.ndarray,
    tables: ScaleTables,
) -> np.ndarray:
    bins = assign_bins(history_01, tables.vov_q20, tables.vov_q80, tables.regime_bins)
    med = np.median(samples, axis=1, keepdims=True)
    calibrated = np.empty_like(samples, dtype=np.float32)
    for b in range(samples.shape[0]):
        scale = tables.scales[int(bins[b])][None, :, :, :]
        eff_scale = 1.0 + tables.alpha * (scale - 1.0)
        out = med[b] + eff_scale * (samples[b] - med[b])
        calibrated[b] = np.clip(out, 0.0, 1.0).astype(np.float32)
    return calibrated


class IntervalScaleSampler:
    def __init__(self, base_model: torch.nn.Module, tables: ScaleTables, device: torch.device):
        self.base_model = base_model
        self.tables = tables
        self.device = device

    def eval(self) -> "IntervalScaleSampler":
        self.base_model.eval()
        return self

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: int = 30,
        chunk_size: int = 4,
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
        calibrated = apply_interval_scaling(raw.detach().cpu().numpy(), history_01, self.tables)
        return torch.from_numpy(calibrated).to(history_device)


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
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--scale_objective", choices=("target", "deadband"), default="target")
    parser.add_argument("--target_coverage", type=float, default=0.9)
    parser.add_argument("--coverage_lo", type=float, default=0.70)
    parser.add_argument("--coverage_hi", type=float, default=0.95)
    parser.add_argument("--scale_min", type=float, default=0.65)
    parser.add_argument("--scale_max", type=float, default=1.45)
    parser.add_argument("--scale_steps", type=int, default=33)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
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
    print("Fitting interval-scale calibration samples")
    calib_samples = sample_native(
        model=base_model,
        history_norm=calib_hist_norm,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    tables = fit_scale_tables(
        calib_samples=calib_samples,
        calib_future=calib_future.detach().cpu().numpy(),
        calib_history=calib_hist_01.detach().cpu().numpy(),
        alpha=args.alpha,
        regime_bins=args.regime_bins,
        min_bin_windows=args.min_bin_windows,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_steps=args.scale_steps,
        scale_objective=args.scale_objective,
        target_coverage=args.target_coverage,
        coverage_lo=args.coverage_lo,
        coverage_hi=args.coverage_hi,
    )
    print(
        "Scale summary:",
        float(tables.scales.min()),
        float(np.median(tables.scales)),
        float(tables.scales.max()),
    )
    model = IntervalScaleSampler(base_model, tables, device).eval()

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

    results = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "calibration": {
                "kind": "pre_validation_interval_scale_around_sample_median",
                "calibration_windows": args.calibration_windows,
                "calibration_samples": args.calibration_samples,
                "alpha": args.alpha,
                "scale_objective": args.scale_objective,
                "target_coverage": args.target_coverage,
                "coverage_lo": args.coverage_lo,
                "coverage_hi": args.coverage_hi,
                "regime_bins": bool(args.regime_bins),
                "scale_min": args.scale_min,
                "scale_median": float(np.median(tables.scales)),
                "scale_max": float(tables.scales.max()),
                "scale_table_min": float(tables.scales.min()),
                "vov_q20": tables.vov_q20,
                "vov_q80": tables.vov_q80,
            },
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
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
    results["summary"] = {"n_pass": n_pass, "n_total": 11, "failed_suites": failed}

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    lines = [
        f"- base model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- calibration: `pre-validation interval scale, objective={args.scale_objective}, regime_bins={bool(args.regime_bins)}, alpha={args.alpha}`",
        f"- scale range: `{float(tables.scales.min()):.3f} / {float(np.median(tables.scales)):.3f} / {float(tables.scales.max()):.3f}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Horizon Summary**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        "",
        "**Additional v2 Suites**",
        f"- time-series ACF corr: `{time_series['acf']['acf_correlation']:.3f}`",
        f"- block boundary ratio: `{block_ar['boundary_smoothness']['boundary_ratio']:.3f}`",
        f"- cointegration gen/GT ratio: `{cointegration.get('gen_gt_ratio', float('nan')):.3f}`",
        f"- regime coverage overall: `{regime_coverage['overall_pass']}`",
        "",
        "**Fidelity / Structure**",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "405a Interval-Scale Calibrated Validation", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
