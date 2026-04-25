#!/usr/bin/env python
"""446a: stationary history-marginal quantile policy around 392a."""

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
    GRID,
    HORIZON,
    assign_bins,
    build_recent_calibration_batch,
    compute_vov,
    sample_native,
    strictly_increasing,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)


@dataclass(frozen=True)
class StationaryHistoryTables:
    levels: np.ndarray
    gen_quantiles: np.ndarray
    target_history_quantiles: np.ndarray
    vov_q20: float
    vov_q80: float
    alpha: float
    regime_bins: bool
    min_bin_windows: int


def fit_stationary_history_tables(
    calib_samples: np.ndarray,
    calib_history_01: np.ndarray,
    n_quantiles: int,
    alpha: float,
    regime_bins: bool,
    min_bin_windows: int,
) -> tuple[StationaryHistoryTables, dict[str, Any]]:
    levels = np.linspace(0.001, 0.999, int(n_quantiles), dtype=np.float64)
    vov = compute_vov(calib_history_01)
    vov_q20 = float(np.quantile(vov, 0.20))
    vov_q80 = float(np.quantile(vov, 0.80))
    bins = assign_bins(calib_history_01, vov_q20, vov_q80, regime_bins)
    n_bins = 3 if regime_bins else 1
    all_idx = np.arange(calib_history_01.shape[0])
    gen_q = np.zeros((n_bins, HORIZON, GRID, GRID, len(levels)), dtype=np.float64)
    hist_q = np.zeros((n_bins, GRID, GRID, len(levels)), dtype=np.float64)
    counts = {}

    for bin_id in range(n_bins):
        idx = np.where(bins == bin_id)[0]
        if idx.shape[0] < int(min_bin_windows):
            idx = all_idx
        counts[str(bin_id)] = int(idx.shape[0])
        for i in range(GRID):
            for j in range(GRID):
                hist_values = calib_history_01[idx, :, i, j].reshape(-1)
                hist_q[bin_id, i, j] = strictly_increasing(np.quantile(hist_values, levels))
        for t in range(HORIZON):
            for i in range(GRID):
                for j in range(GRID):
                    gen_values = calib_samples[idx, :, t, i, j].reshape(-1)
                    gen_q[bin_id, t, i, j] = strictly_increasing(np.quantile(gen_values, levels))

    tables = StationaryHistoryTables(
        levels=levels,
        gen_quantiles=gen_q,
        target_history_quantiles=hist_q,
        vov_q20=vov_q20,
        vov_q80=vov_q80,
        alpha=float(alpha),
        regime_bins=bool(regime_bins),
        min_bin_windows=int(min_bin_windows),
    )
    meta = {
        "vov_q20": vov_q20,
        "vov_q80": vov_q80,
        "calibration_bin_counts": counts,
        "target_history_median": float(np.median(calib_history_01)),
        "target_history_std": float(np.std(calib_history_01)),
    }
    return tables, meta


def apply_stationary_history_map(
    samples: np.ndarray,
    history_01: np.ndarray,
    tables: StationaryHistoryTables,
) -> np.ndarray:
    calibrated = np.empty_like(samples, dtype=np.float32)
    bins = assign_bins(history_01, tables.vov_q20, tables.vov_q80, tables.regime_bins)
    for b in range(samples.shape[0]):
        bin_id = int(bins[b])
        for t in range(samples.shape[2]):
            for i in range(GRID):
                for j in range(GRID):
                    values = samples[b, :, t, i, j].astype(np.float64)
                    gen_q = tables.gen_quantiles[bin_id, t, i, j]
                    target_q = tables.target_history_quantiles[bin_id, i, j]
                    u = np.interp(
                        values,
                        gen_q,
                        tables.levels,
                        left=tables.levels[0],
                        right=tables.levels[-1],
                    )
                    mapped = np.interp(u, tables.levels, target_q)
                    out = (1.0 - tables.alpha) * values + tables.alpha * mapped
                    calibrated[b, :, t, i, j] = np.clip(out, 0.0, 1.0).astype(np.float32)
    return calibrated


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
    parser.add_argument("--n_quantiles", type=int, default=101)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--seed", type=int, default=446)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    base_model.eval()

    calib_hist_01, calib_hist_norm, _calib_future = build_recent_calibration_batch(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        calibration_windows=args.calibration_windows,
        device=device,
    )
    print("Sampling calibration generated levels")
    calib_samples = sample_native(
        model=base_model,
        history_norm=calib_hist_norm,
        n_samples=args.calibration_samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    tables, table_meta = fit_stationary_history_tables(
        calib_samples=calib_samples,
        calib_history_01=calib_hist_01.detach().cpu().numpy(),
        n_quantiles=args.n_quantiles,
        alpha=args.alpha,
        regime_bins=bool(args.regime_bins),
        min_bin_windows=args.min_bin_windows,
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
    raw_samples = sample_native(
        model=base_model,
        history_norm=batch.history_norm,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
    )
    cond_samples = apply_stationary_history_map(
        samples=raw_samples,
        history_01=batch.history_01.detach().cpu().numpy(),
        tables=tables,
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
            "kind": "stationary_history_marginal_quantile_map",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_per_window_oracle_miss_placement": False,
            "calibration_split": "pre_validation",
            "calibration_windows": int(args.calibration_windows),
            "calibration_samples": int(args.calibration_samples),
            "target_marginal_source": "pre_validation_observed_history_levels",
            "base_samples_source": "392a_validation_history_only",
            "alpha": float(args.alpha),
            "n_quantiles": int(args.n_quantiles),
            "regime_bins": bool(args.regime_bins),
            "calibrated_risk_policy": True,
            **table_meta,
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
        "- policy: `stationary history-marginal quantile map around frozen 392a`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- uses validation future: `{results['config']['deployability']['uses_validation_future']}`",
        f"- alpha: `{args.alpha:.3f}`",
        f"- regime bins: `{bool(args.regime_bins)}`",
        f"- calibration windows/samples: `{args.calibration_windows}` / `{args.calibration_samples}`",
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
        f"- median-bias fraction/magnitude: `{distributional['median_bias']['n_pass']}/25` / `{distributional['median_bias']['n_mag_pass']}/25`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "446a Stationary History-Marginal Quantile Policy", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
