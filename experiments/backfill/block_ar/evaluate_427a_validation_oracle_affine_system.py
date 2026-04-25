#!/usr/bin/env python
"""427a: validation-oracle affine calibration diagnostic for 392a.

This diagnostic freezes 392a and applies a validation-oracle affine map per
regime/horizon/cell:

    y = center_gen + shift + scale * (x - center_gen)

The goal is to test whether an affine correction can repair coverage/level occupancy
while preserving transition geometry better than the nonlinear 426a quantile map.
It is an oracle feasibility bound, not a deployable calibrated model.
"""

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
    sample_native,
)
from experiments.backfill.block_ar.evaluate_405a_interval_scale_calibrated_system import (  # noqa: E402
    compute_vov,
    interval_coverage_at_scale,
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
class AffineTables:
    centers: np.ndarray
    shifts: np.ndarray
    scales: np.ndarray
    vov_q20: float
    vov_q80: float
    center_alpha: float
    scale_alpha: float
    regime_bins: bool
    target_coverage: float
    coverage_lo: float
    coverage_hi: float


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def fit_affine_tables(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    calib_history: np.ndarray,
    center_alpha: float,
    scale_alpha: float,
    regime_bins: bool,
    min_bin_windows: int,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    target_coverage: float,
    coverage_lo: float,
    coverage_hi: float,
) -> AffineTables:
    vov = compute_vov(calib_history)
    vov_q20 = float(np.quantile(vov, 0.2))
    vov_q80 = float(np.quantile(vov, 0.8))
    bins = assign_bins(calib_history, vov_q20, vov_q80, regime_bins)
    n_bins = 3 if regime_bins else 1
    centers = np.zeros((n_bins, HORIZON, GRID, GRID), dtype=np.float32)
    shifts = np.zeros_like(centers)
    scales = np.ones_like(centers)
    candidates = np.linspace(scale_min, scale_max, int(scale_steps), dtype=np.float64)
    all_idx = np.arange(calib_history.shape[0])

    for bin_id in range(n_bins):
        idx = np.where(bins == bin_id)[0]
        if idx.shape[0] < int(min_bin_windows):
            idx = all_idx
        for t in range(HORIZON):
            for i in range(GRID):
                for j in range(GRID):
                    samples = calib_samples[idx, :, t, i, j].astype(np.float64)
                    target = calib_future[idx, t, i, j].astype(np.float64)
                    center = float(np.median(samples))
                    target_center = float(np.median(target))
                    shifted = samples + (target_center - center)
                    best_scale = 1.0
                    best_obj = float("inf")
                    for scale in candidates:
                        cov = interval_coverage_at_scale(shifted, target, float(scale))
                        obj = abs(cov - target_coverage)
                        obj += 0.5 * max(0.0, cov - coverage_hi)
                        obj += 0.5 * max(0.0, coverage_lo - cov)
                        obj += 0.001 * abs(float(scale) - 1.0)
                        if obj < best_obj:
                            best_obj = obj
                            best_scale = float(scale)
                    centers[bin_id, t, i, j] = center
                    shifts[bin_id, t, i, j] = target_center - center
                    scales[bin_id, t, i, j] = best_scale

    return AffineTables(
        centers=centers,
        shifts=shifts,
        scales=scales,
        vov_q20=vov_q20,
        vov_q80=vov_q80,
        center_alpha=float(center_alpha),
        scale_alpha=float(scale_alpha),
        regime_bins=bool(regime_bins),
        target_coverage=float(target_coverage),
        coverage_lo=float(coverage_lo),
        coverage_hi=float(coverage_hi),
    )


def apply_affine_calibration(
    samples: np.ndarray,
    history_01: np.ndarray,
    tables: AffineTables,
) -> np.ndarray:
    bins = assign_bins(history_01, tables.vov_q20, tables.vov_q80, tables.regime_bins)
    calibrated = np.empty_like(samples, dtype=np.float32)
    for b in range(samples.shape[0]):
        bin_id = int(bins[b])
        center = tables.centers[bin_id][None, :, :, :]
        shift = tables.center_alpha * tables.shifts[bin_id][None, :, :, :]
        scale = 1.0 + tables.scale_alpha * (tables.scales[bin_id][None, :, :, :] - 1.0)
        out = center + shift + scale * (samples[b] - center)
        calibrated[b] = np.clip(out, 0.0, 1.0).astype(np.float32)
    return calibrated


class AffineOracleSampler:
    def __init__(self, base_model: torch.nn.Module, tables: AffineTables, device: torch.device):
        self.base_model = base_model
        self.tables = tables
        self.device = device

    def eval(self) -> "AffineOracleSampler":
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
        calibrated = apply_affine_calibration(raw.detach().cpu().numpy(), history_01, self.tables)
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
    parser.add_argument("--oracle_samples", type=int, default=48)
    parser.add_argument("--center_alpha", type=float, default=1.0)
    parser.add_argument("--scale_alpha", type=float, default=1.0)
    parser.add_argument("--target_coverage", type=float, default=0.85)
    parser.add_argument("--coverage_lo", type=float, default=0.70)
    parser.add_argument("--coverage_hi", type=float, default=0.95)
    parser.add_argument("--scale_min", type=float, default=0.45)
    parser.add_argument("--scale_max", type=float, default=1.80)
    parser.add_argument("--scale_steps", type=int, default=55)
    parser.add_argument("--min_bin_windows", type=int, default=30)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--seed", type=int, default=427)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    base_model.eval()

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

    print("Fitting validation-oracle affine map")
    oracle_samples = sample_native(
        model=base_model,
        history_norm=batch.history_norm,
        n_samples=args.oracle_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    tables = fit_affine_tables(
        calib_samples=oracle_samples,
        calib_future=batch.future_01.detach().cpu().numpy(),
        calib_history=batch.history_01.detach().cpu().numpy(),
        center_alpha=args.center_alpha,
        scale_alpha=args.scale_alpha,
        regime_bins=args.regime_bins,
        min_bin_windows=args.min_bin_windows,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_steps=args.scale_steps,
        target_coverage=args.target_coverage,
        coverage_lo=args.coverage_lo,
        coverage_hi=args.coverage_hi,
    )
    print(
        "Affine summary:",
        "shift",
        float(tables.shifts.min()),
        float(np.median(tables.shifts)),
        float(tables.shifts.max()),
        "scale",
        float(tables.scales.min()),
        float(np.median(tables.scales)),
        float(tables.scales.max()),
    )

    model = AffineOracleSampler(base_model, tables, device).eval()
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
            "diagnostic": {
                "kind": "validation_oracle_affine_median_shift_interval_scale",
                "oracle_uses_validation_future": True,
                "claim": "feasibility_upper_bound_not_deployable_model",
                "oracle_samples": args.oracle_samples,
                "center_alpha": args.center_alpha,
                "scale_alpha": args.scale_alpha,
                "target_coverage": args.target_coverage,
                "coverage_lo": args.coverage_lo,
                "coverage_hi": args.coverage_hi,
                "regime_bins": bool(args.regime_bins),
                "shift_min": float(tables.shifts.min()),
                "shift_median": float(np.median(tables.shifts)),
                "shift_max": float(tables.shifts.max()),
                "scale_min": float(tables.scales.min()),
                "scale_median": float(np.median(tables.scales)),
                "scale_max": float(tables.scales.max()),
                "seed": args.seed,
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
        "- diagnostic: `validation oracle affine median shift + interval scale; not deployable`",
        f"- shift range: `{float(tables.shifts.min()):.4f} / {float(np.median(tables.shifts)):.4f} / {float(tables.shifts.max()):.4f}`",
        f"- scale range: `{float(tables.scales.min()):.3f} / {float(np.median(tables.scales)):.3f} / {float(tables.scales.max()):.3f}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Fidelity / Structure**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- median-bias fraction cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- median-bias magnitude cells: `{distributional['median_bias']['n_mag_pass']}/25`",
        f"- regime layer2: `{regime_coverage['layer2_n_passing']}/{regime_coverage['layer2_n_total']}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- cointegration gen/GT ratio: `{cointegration.get('gen_gt_ratio', float('nan')):.3f}`",
        f"- mean-reversion active pass: `{mean_reversion.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "427a Validation-Oracle Affine Diagnostic", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()

