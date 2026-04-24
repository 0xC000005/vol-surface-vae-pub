#!/usr/bin/env python
"""374a: frozen-340c calibration feasibility ladder.

This is an analysis/falsifier, not a new learned generator. It freezes a 340c
checkpoint, generates the standard validation sample array once, then applies
small cross-fitted logit-space calibration maps to test whether the revised
remaining failures are calibration-feasible.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (  # noqa: E402
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


EPS = 1e-5
HORIZONS = [1, 7, 14, 30]
SUITES_10 = [
    "surface",
    "coverage",
    "time_series",
    "block_ar",
    "cointegration",
    "regime_coverage",
    "distributional_fidelity",
    "cross_cell_correlation",
    "mean_reversion",
    "pathwise_jump_realism",
]


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def vol_of_vol(history: np.ndarray) -> np.ndarray:
    mean_iv = history.mean(axis=(2, 3))
    return np.diff(mean_iv, axis=1).std(axis=1)


@dataclass
class AffineParams:
    loc: np.ndarray
    scale: np.ndarray


def fit_global_affine(gen_logits: np.ndarray, gt_logits: np.ndarray) -> AffineParams:
    gen_mu = gen_logits.mean(keepdims=True)
    gen_std = gen_logits.std(keepdims=True)
    gt_mu = gt_logits.mean(keepdims=True)
    gt_std = gt_logits.std(keepdims=True)
    scale = np.clip(gt_std / np.maximum(gen_std, 1e-6), 0.25, 4.0)
    loc = gt_mu - scale * gen_mu
    return AffineParams(loc=loc, scale=scale)


def fit_horizon_cell_affine(gen_logits: np.ndarray, gt_logits: np.ndarray) -> AffineParams:
    """Match logit mean/std by future day and cell.

    Generated logits have shape [N, S, T, 5, 5]; realized logits have shape
    [N, T, 5, 5]. The map is deliberately small: no learned network, only
    cross-fitted per-horizon/cell location and scale.
    """
    gen_mu = gen_logits.mean(axis=(0, 1), keepdims=True)
    gen_std = gen_logits.std(axis=(0, 1), keepdims=True)
    gt_mu = gt_logits.mean(axis=0, keepdims=True)[:, None, ...]
    gt_std = gt_logits.std(axis=0, keepdims=True)[:, None, ...]
    scale = np.clip(gt_std / np.maximum(gen_std, 1e-6), 0.25, 4.0)
    loc = gt_mu - scale * gen_mu
    return AffineParams(loc=loc, scale=scale)


def apply_affine(gen_logits: np.ndarray, params: AffineParams) -> np.ndarray:
    return params.loc + params.scale * gen_logits


def crossfit_affine(
    samples: np.ndarray,
    gt: np.ndarray,
    mode: str,
    history: np.ndarray | None = None,
) -> np.ndarray:
    """Apply two-fold cross-fitted logit-space affine calibration."""
    gen_logits = logit(samples)
    gt_logits = logit(gt)
    out = np.empty_like(gen_logits)
    n = samples.shape[0]
    folds = [np.arange(n) % 2 == 0, np.arange(n) % 2 == 1]

    for eval_mask in folds:
        fit_mask = ~eval_mask
        if mode == "global":
            params = fit_global_affine(gen_logits[fit_mask], gt_logits[fit_mask])
            out[eval_mask] = apply_affine(gen_logits[eval_mask], params)
        elif mode == "horizon_cell":
            params = fit_horizon_cell_affine(gen_logits[fit_mask], gt_logits[fit_mask])
            out[eval_mask] = apply_affine(gen_logits[eval_mask], params)
        elif mode == "regime_horizon_cell":
            if history is None:
                raise ValueError("history is required for regime_horizon_cell")
            vov = vol_of_vol(history)
            q20 = np.quantile(vov[fit_mask], 0.20)
            q80 = np.quantile(vov[fit_mask], 0.80)
            eval_labels = np.full(n, 1, dtype=np.int64)
            fit_labels = np.full(n, 1, dtype=np.int64)
            eval_labels[vov <= q20] = 0
            eval_labels[vov >= q80] = 2
            fit_labels[vov <= q20] = 0
            fit_labels[vov >= q80] = 2
            global_params = fit_horizon_cell_affine(gen_logits[fit_mask], gt_logits[fit_mask])
            for label in [0, 1, 2]:
                fit_regime = fit_mask & (fit_labels == label)
                eval_regime = eval_mask & (eval_labels == label)
                if not eval_regime.any():
                    continue
                if fit_regime.sum() < 12:
                    params = global_params
                else:
                    params = fit_horizon_cell_affine(gen_logits[fit_regime], gt_logits[fit_regime])
                out[eval_regime] = apply_affine(gen_logits[eval_regime], params)
        else:
            raise ValueError(f"unknown mode: {mode}")
    return sigmoid(out)


def conditionality_proxy(samples: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """Sample-array proxy for official model-vs-shuffled-history conditionality."""
    cond_width = np.quantile(samples, 0.95, axis=1) - np.quantile(samples, 0.05, axis=1)
    cond_med = np.median(samples, axis=1)
    cond_mae = np.abs(cond_med - gt)

    perm = np.roll(np.arange(samples.shape[0]), samples.shape[0] // 2)
    uncond = samples[perm]
    uncond_width = np.quantile(uncond, 0.95, axis=1) - np.quantile(uncond, 0.05, axis=1)
    uncond_med = np.median(uncond, axis=1)
    uncond_mae = np.abs(uncond_med - gt)

    avg_cond_mae = float(cond_mae.mean())
    avg_uncond_mae = float(uncond_mae.mean())
    mae_reduction = (avg_uncond_mae - avg_cond_mae) / max(avg_uncond_mae, 1e-12) * 100.0
    cell_mae_reduction = np.where(
        uncond_mae.mean(axis=(0, 1)) > 1e-12,
        (uncond_mae.mean(axis=(0, 1)) - cond_mae.mean(axis=(0, 1)))
        / uncond_mae.mean(axis=(0, 1))
        * 100.0,
        0.0,
    )
    cell_width_ratio = cond_width.mean(axis=(0, 1)) / np.maximum(uncond_width.mean(axis=(0, 1)), 1e-12)
    mae_pass = mae_reduction > 5.0
    worst_cell_mae_pass = float(cell_mae_reduction.min()) > -10.0
    worst_cell_wr_pass = float(cell_width_ratio.max()) < 1.20
    return {
        "mae_reduction_pct": float(mae_reduction),
        "avg_cond_mae": avg_cond_mae,
        "avg_uncond_mae": avg_uncond_mae,
        "worst_cell_mae_reduction": float(cell_mae_reduction.min()),
        "worst_cell_mae_pass": bool(worst_cell_mae_pass),
        "worst_cell_width_ratio": float(cell_width_ratio.max()),
        "worst_cell_wr_pass": bool(worst_cell_wr_pass),
        "mae_pass": bool(mae_pass),
        "overall_pass_proxy": bool(mae_pass and worst_cell_mae_pass and worst_cell_wr_pass),
        "note": "sample-array proxy using a deterministic window roll as unconditional baseline",
    }


def suite_summary(results: dict[str, Any], cond_proxy: dict[str, Any]) -> tuple[int, list[str]]:
    ordered = [(name, bool(results[name]["overall_pass"])) for name in SUITES_10]
    ordered.insert(2, ("conditionality_proxy", bool(cond_proxy["overall_pass_proxy"])))
    failed = [name for name, ok in ordered if not ok]
    return sum(int(ok) for _name, ok in ordered), failed


def evaluate_samples(
    samples: np.ndarray,
    gt: np.ndarray,
    history: np.ndarray,
    returns: np.ndarray,
    rollout_start: int,
) -> tuple[dict[str, Any], dict[str, Any], int, list[str], str]:
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        results = {
            "surface": run_surface_validity_tests(samples, gt),
            "coverage": run_ci_coverage_tests(samples, gt),
            "time_series": run_time_series_tests(samples, gt),
            "block_ar": run_block_ar_tests(samples),
            "cointegration": run_cointegration_tests(
                samples,
                gt,
                returns=returns,
                test_start=rollout_start,
                history_len=30,
                future_len=30,
            ),
            "regime_coverage": run_regime_coverage_tests(samples, gt, history),
            "distributional_fidelity": run_distributional_fidelity_tests(samples, gt, history),
            "cross_cell_correlation": run_cross_cell_correlation_tests(samples, gt),
            "mean_reversion": run_mean_reversion_tests(samples, gt, history),
            "pathwise_jump_realism": run_pathwise_jump_realism_tests(samples, gt),
        }
    cond_proxy = conditionality_proxy(samples, gt)
    n_pass, failed = suite_summary(results, cond_proxy)
    return results, cond_proxy, n_pass, failed, logs.getvalue()


def get_metric(mapping: dict[Any, Any], key: Any) -> Any:
    if key in mapping:
        return mapping[key]
    str_key = str(key)
    if str_key in mapping:
        return mapping[str_key]
    raise KeyError(key)


def metric_digest(results: dict[str, Any], cond_proxy: dict[str, Any]) -> dict[str, Any]:
    return {
        "coverage90": get_metric(results["coverage"]["overall"], 0.9),
        "worst_h30_cell_coverage": get_metric(results["coverage"]["worst_cell_per_horizon"], 30),
        "best_h30_cell_coverage": get_metric(results["coverage"]["best_cell_per_horizon"], 30),
        "conditionality_proxy_mae_reduction": cond_proxy["mae_reduction_pct"],
        "regime_layer2": [
            results["regime_coverage"]["layer2_n_passing"],
            results["regime_coverage"]["layer2_n_total"],
        ],
        "daily_ks_cells": results["distributional_fidelity"]["ks_test"]["n_pass"],
        "level_ks_cells": results["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "median_bias_cells": results["distributional_fidelity"]["median_bias"]["n_pass"],
        "cointegration_worst_cell": results["cointegration"]["worst_cell_ratio"],
        "corr_ratio": results["cross_cell_correlation"]["corr_ratio"],
        "rank_ratio": results["cross_cell_correlation"]["rank_ratio"],
        "mean_reversion_pass": results["mean_reversion"]["overall_pass"],
        "pathwise_maxjump_ks": results["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/340c_v0_s42/best_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", default="results/validations/2026-04-24/analysis/374a_frozen_340c_calibration_ladder")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel("340c", args.checkpoint, device)
    model.eval()

    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=30,
        future_len=30,
        test_start=4511,
        val_size=441,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    outputs = []
    for start in range(0, batch.history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, batch.history_01.shape[0])
        hist_batch = normalize_iv(batch.history_01[start:end])
        with torch.no_grad():
            samples = model.sample_batched(
                hist_batch,
                n_samples=args.samples,
                n_steps=batch.future_01.shape[1],
                chunk_size=args.chunk_size,
            )
        outputs.append(samples.cpu().numpy())
    raw_samples = np.concatenate(outputs, axis=0)
    gt = batch.future_01.detach().cpu().numpy()
    history = batch.history_01.detach().cpu().numpy()

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    rollout_start = 4511 - 30 - 30 - 441

    variants = {
        "raw": raw_samples,
        "global_logit_affine": crossfit_affine(raw_samples, gt, "global"),
        "horizon_cell_logit_affine": crossfit_affine(raw_samples, gt, "horizon_cell"),
        "regime_horizon_cell_logit_affine": crossfit_affine(
            raw_samples,
            gt,
            "regime_horizon_cell",
            history=history,
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "n_windows": int(gt.shape[0]),
        "samples": int(args.samples),
        "variants": {},
    }

    for name, samples in variants.items():
        results, cond_proxy, n_pass, failed, logs = evaluate_samples(
            samples,
            gt,
            history,
            returns,
            rollout_start,
        )
        summary["variants"][name] = {
            "n_pass_proxy11": n_pass,
            "failed_proxy11": failed,
            "conditionality_proxy": cond_proxy,
            "digest": metric_digest(results, cond_proxy),
            "official_sample_array_results": results,
        }
        (out_dir / f"{name}_suite_stdout.txt").write_text(logs)

    best_name = max(summary["variants"], key=lambda k: summary["variants"][k]["n_pass_proxy11"])
    summary["best_variant"] = best_name
    summary["best_n_pass_proxy11"] = summary["variants"][best_name]["n_pass_proxy11"]
    (out_dir / "summary.json").write_text(json.dumps(make_serializable(summary), indent=2))

    lines = [
        "# 374a Frozen 340c Calibration Ladder",
        "",
        "This is a frozen-model feasibility analysis. Scores use official sample-array suites plus a sample-array conditionality proxy.",
        "",
        "| variant | proxy score | failed | coverage90 | h30 worst/best | cond MAE red | regime L2 | level KS | daily KS | maxjump KS |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for name, item in summary["variants"].items():
        d = item["digest"]
        lines.append(
            f"| `{name}` | {item['n_pass_proxy11']}/11 | {', '.join(item['failed_proxy11']) or 'none'} "
            f"| {d['coverage90']:.3f} | {d['worst_h30_cell_coverage']:.3f}/{d['best_h30_cell_coverage']:.3f} "
            f"| {d['conditionality_proxy_mae_reduction']:.1f}% | {d['regime_layer2'][0]}/{d['regime_layer2'][1]} "
            f"| {d['level_ks_cells']}/25 | {d['daily_ks_cells']}/25 | {d['pathwise_maxjump_ks']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Best variant: `{best_name}` with proxy score `{summary['best_n_pass_proxy11']}/11`.",
            "",
            "Interpretation guard: this is a calibration feasibility ladder, not a final deployable calibration protocol. A deployable protocol needs a fixed calibration split or cross-fitting policy and official model-wrapper evaluation.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in ["best_variant", "best_n_pass_proxy11"]}, indent=2))


if __name__ == "__main__":
    main()
