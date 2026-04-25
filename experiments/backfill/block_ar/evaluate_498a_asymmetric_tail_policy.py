#!/usr/bin/env python
"""498a: median-locked asymmetric tail policy around frozen 392a.

This is a deployable risk-policy layer, not a new learned conditional law. It fits
separate lower/upper residual scales on pre-validation windows, applies them around the
392a sample median, and locks the transformed sample median back to the original 392a
median to protect conditionality.
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
    assign_bins,
    build_recent_calibration_batch,
    compute_vov,
    sample_native,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)


HORIZON = 30
GRID = 5


@dataclass(frozen=True)
class AsymmetricTailPolicy:
    lower_scale: np.ndarray
    upper_scale: np.ndarray
    vov_q20: float
    vov_q80: float
    regime_bins: bool
    target_tail: float
    scale_penalty: float
    lock_median: bool


def _fit_one_scale(
    center: np.ndarray,
    bound_q: np.ndarray,
    gt: np.ndarray,
    candidates: np.ndarray,
    target_tail: float,
    scale_penalty: float,
    side: str,
) -> tuple[float, float, float]:
    before_bound = center + (bound_q - center)
    if side == "lower":
        before = float((gt < before_bound).mean())
    elif side == "upper":
        before = float((gt > before_bound).mean())
    else:
        raise ValueError(f"unknown side: {side}")

    best_scale = 1.0
    best_score = float("inf")
    best_miss = before
    for scale in candidates:
        bound = center + float(scale) * (bound_q - center)
        if side == "lower":
            miss = float((gt < bound).mean())
        else:
            miss = float((gt > bound).mean())
        score = (miss - target_tail) ** 2 + float(scale_penalty) * (float(scale) - 1.0) ** 2
        if score < best_score:
            best_score = score
            best_scale = float(scale)
            best_miss = miss
    return best_scale, before, best_miss


def _fit_scales_for_subset(
    samples: np.ndarray,
    future: np.ndarray,
    candidates: np.ndarray,
    target_tail: float,
    scale_penalty: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    center = np.median(samples, axis=1)
    q05 = np.quantile(samples, 0.05, axis=1)
    q95 = np.quantile(samples, 0.95, axis=1)
    lower_scale = np.ones((HORIZON, GRID, GRID), dtype=np.float32)
    upper_scale = np.ones_like(lower_scale)
    lower_before = np.zeros_like(lower_scale)
    lower_after = np.zeros_like(lower_scale)
    upper_before = np.zeros_like(lower_scale)
    upper_after = np.zeros_like(lower_scale)

    for t in range(HORIZON):
        for i in range(GRID):
            for j in range(GRID):
                c = center[:, t, i, j]
                y = future[:, t, i, j]
                lower_scale[t, i, j], lower_before[t, i, j], lower_after[t, i, j] = _fit_one_scale(
                    c,
                    q05[:, t, i, j],
                    y,
                    candidates,
                    target_tail,
                    scale_penalty,
                    side="lower",
                )
                upper_scale[t, i, j], upper_before[t, i, j], upper_after[t, i, j] = _fit_one_scale(
                    c,
                    q95[:, t, i, j],
                    y,
                    candidates,
                    target_tail,
                    scale_penalty,
                    side="upper",
                )

    summary = {
        "lower_scale_min": float(lower_scale.min()),
        "lower_scale_max": float(lower_scale.max()),
        "lower_scale_mean": float(lower_scale.mean()),
        "upper_scale_min": float(upper_scale.min()),
        "upper_scale_max": float(upper_scale.max()),
        "upper_scale_mean": float(upper_scale.mean()),
        "lower_miss_before_mean": float(lower_before.mean()),
        "lower_miss_after_mean": float(lower_after.mean()),
        "upper_miss_before_mean": float(upper_before.mean()),
        "upper_miss_after_mean": float(upper_after.mean()),
    }
    return lower_scale, upper_scale, summary


def fit_asymmetric_tail_policy(
    calib_samples: np.ndarray,
    calib_future: np.ndarray,
    calib_history_01: np.ndarray,
    candidates: np.ndarray,
    target_tail: float,
    scale_penalty: float,
    regime_bins: bool,
    min_bin_windows: int,
    lock_median: bool,
) -> tuple[AsymmetricTailPolicy, dict[str, Any]]:
    vov = compute_vov(calib_history_01)
    vov_q20 = float(np.quantile(vov, 0.20))
    vov_q80 = float(np.quantile(vov, 0.80))
    bins = assign_bins(calib_history_01, vov_q20, vov_q80, regime_bins)
    n_bins = 3 if regime_bins else 1

    global_lower, global_upper, global_summary = _fit_scales_for_subset(
        calib_samples,
        calib_future,
        candidates,
        target_tail,
        scale_penalty,
    )
    lower = np.repeat(global_lower[None], n_bins, axis=0)
    upper = np.repeat(global_upper[None], n_bins, axis=0)
    bin_summaries: list[dict[str, Any]] = []

    for bin_id in range(n_bins):
        mask = bins == bin_id
        if int(mask.sum()) < int(min_bin_windows):
            bin_summaries.append(
                {"bin": int(bin_id), "n_windows": int(mask.sum()), "fallback": "global"}
            )
            continue
        lower_b, upper_b, summary_b = _fit_scales_for_subset(
            calib_samples[mask],
            calib_future[mask],
            candidates,
            target_tail,
            scale_penalty,
        )
        lower[bin_id] = lower_b
        upper[bin_id] = upper_b
        summary_b = dict(summary_b)
        summary_b.update({"bin": int(bin_id), "n_windows": int(mask.sum()), "fallback": None})
        bin_summaries.append(summary_b)

    policy = AsymmetricTailPolicy(
        lower_scale=lower.astype(np.float32),
        upper_scale=upper.astype(np.float32),
        vov_q20=vov_q20,
        vov_q80=vov_q80,
        regime_bins=bool(regime_bins),
        target_tail=float(target_tail),
        scale_penalty=float(scale_penalty),
        lock_median=bool(lock_median),
    )
    summary = {
        "target_tail": float(target_tail),
        "scale_penalty": float(scale_penalty),
        "regime_bins": bool(regime_bins),
        "n_bins": int(n_bins),
        "min_bin_windows": int(min_bin_windows),
        "global_summary": global_summary,
        "bin_summaries": bin_summaries,
        "lower_scale_min": float(lower.min()),
        "lower_scale_max": float(lower.max()),
        "lower_scale_mean": float(lower.mean()),
        "upper_scale_min": float(upper.min()),
        "upper_scale_max": float(upper.max()),
        "upper_scale_mean": float(upper.mean()),
    }
    return policy, summary


def apply_asymmetric_tail_policy(
    samples: np.ndarray,
    history_01: np.ndarray,
    policy: AsymmetricTailPolicy,
) -> np.ndarray:
    center = np.median(samples, axis=1, keepdims=True)
    residual = samples - center
    bins = assign_bins(history_01, policy.vov_q20, policy.vov_q80, policy.regime_bins)
    lower = policy.lower_scale[bins][:, None, : samples.shape[2]]
    upper = policy.upper_scale[bins][:, None, : samples.shape[2]]
    scaled = np.where(residual < 0.0, center + lower * residual, center + upper * residual)
    if policy.lock_median:
        shifted_median = np.median(scaled, axis=1, keepdims=True)
        scaled = scaled - shifted_median + center
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


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
    parser.add_argument("--min_scale", type=float, default=0.70)
    parser.add_argument("--max_scale", type=float, default=1.45)
    parser.add_argument("--n_scale_candidates", type=int, default=31)
    parser.add_argument("--target_tail", type=float, default=0.05)
    parser.add_argument("--scale_penalty", type=float, default=0.001)
    parser.add_argument("--min_bin_windows", type=int, default=50)
    parser.add_argument("--regime_bins", action="store_true")
    parser.add_argument("--no_lock_median", action="store_true")
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--policy_json", required=True)
    args = parser.parse_args()

    if args.future_len != HORIZON:
        raise ValueError(f"498a currently expects future_len={HORIZON}")

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
    candidates = np.linspace(args.min_scale, args.max_scale, args.n_scale_candidates, dtype=np.float64)
    policy, policy_summary = fit_asymmetric_tail_policy(
        calib_samples=calib_samples,
        calib_future=calib_future.detach().cpu().numpy(),
        calib_history_01=calib_hist_01.detach().cpu().numpy(),
        candidates=candidates,
        target_tail=args.target_tail,
        scale_penalty=args.scale_penalty,
        regime_bins=bool(args.regime_bins),
        min_bin_windows=args.min_bin_windows,
        lock_median=not bool(args.no_lock_median),
    )

    policy_path = Path(args.policy_json)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(
            make_serializable(
                {
                    "config": vars(args),
                    "checkpoint_epoch": int(payload.get("epoch", -1)),
                    "policy_summary": policy_summary,
                    "vov_q20": policy.vov_q20,
                    "vov_q80": policy.vov_q80,
                    "lower_scale": policy.lower_scale.tolist(),
                    "upper_scale": policy.upper_scale.tolist(),
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
    cond_samples = apply_asymmetric_tail_policy(
        val_base_samples,
        batch.history_01.detach().cpu().numpy(),
        policy,
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
        "model_type": "498a_asymmetric_tail_policy",
        "base_model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "deployability": {
            "kind": "median_locked_asymmetric_tail_policy",
            "deployable": True,
            "uses_validation_future": False,
            "uses_validation_future_errors": False,
            "uses_per_window_oracle_center": False,
            "uses_validation_tuned_weights": False,
            "calibration_split": "pre_validation",
            "calibration_windows": int(calib_hist_01.shape[0]),
            "calibration_samples": int(args.calibration_samples),
            "base_center": "validation_392a_sample_median",
            "risk_policy_not_base_learned_law": True,
            "report_base_model_separately": True,
        },
        "policy_summary": policy_summary,
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
        "- policy: `median-locked asymmetric lower/upper tail residual scales`",
        f"- deployable: `{results['config']['deployability']['deployable']}`",
        f"- uses validation future: `{results['config']['deployability']['uses_validation_future']}`",
        f"- regime bins: `{bool(args.regime_bins)}`",
        f"- calibration windows/samples: `{calib_hist_01.shape[0]}` / `{args.calibration_samples}`",
        f"- scale range: `{args.min_scale:.2f}` to `{args.max_scale:.2f}`",
        f"- lock median: `{not bool(args.no_lock_median)}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Policy Fit**",
        f"- lower scale mean/range: `{policy_summary['lower_scale_mean']:.3f}` "
        f"[`{policy_summary['lower_scale_min']:.3f}`, `{policy_summary['lower_scale_max']:.3f}`]",
        f"- upper scale mean/range: `{policy_summary['upper_scale_mean']:.3f}` "
        f"[`{policy_summary['upper_scale_min']:.3f}`, `{policy_summary['upper_scale_max']:.3f}`]",
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
    write_markdown_summary(args.output_md, "498a Median-Locked Asymmetric Tail Policy", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
