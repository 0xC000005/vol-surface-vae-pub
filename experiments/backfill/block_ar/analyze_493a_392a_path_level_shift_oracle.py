#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv


def _np_logit(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _oracle_window_cell_logit_shift(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Oracle constant future-path shift per validation window/cell.

    This is intentionally non-deployable. It tests whether the remaining 392a
    failures are compatible with preserving each generated path's within-horizon
    shape while moving its level location.
    """

    sample_logit = _np_logit(samples, eps)
    gt_logit = _np_logit(ground_truth, eps)
    gen_center = np.median(sample_logit, axis=(1, 2))
    gt_center = np.median(gt_logit, axis=1)
    shift = (gt_center - gen_center)[:, None, None]
    return _np_sigmoid(sample_logit + shift)


def _evaluate_samples(
    *,
    samples: np.ndarray,
    ground_truth: np.ndarray,
    history_01: np.ndarray,
    returns: np.ndarray,
    rollout_start: int,
    history_len: int,
    future_len: int,
) -> dict[str, Any]:
    results = {
        "surface": run_surface_validity_tests(samples, ground_truth),
        "coverage": run_ci_coverage_tests(samples, ground_truth),
        "time_series": run_time_series_tests(samples, ground_truth),
        "block_ar": run_block_ar_tests(samples),
        "cointegration": run_cointegration_tests(
            samples,
            ground_truth,
            returns=returns,
            test_start=rollout_start,
            history_len=history_len,
            future_len=future_len,
        ),
        "regime_coverage": run_regime_coverage_tests(samples, ground_truth, history_01),
        "distributional_fidelity": run_distributional_fidelity_tests(
            samples,
            ground_truth,
            history_01,
        ),
        "cross_cell_correlation": run_cross_cell_correlation_tests(samples, ground_truth),
        "mean_reversion": run_mean_reversion_tests(samples, ground_truth, history_01),
        "pathwise_jump_realism": run_pathwise_jump_realism_tests(samples, ground_truth),
    }
    ordered = [
        ("surface", results["surface"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("time_series", results["time_series"]["overall_pass"]),
        ("block_ar", results["block_ar"]["overall_pass"]),
        ("cointegration", results["cointegration"]["overall_pass"]),
        ("regime_coverage", results["regime_coverage"]["overall_pass"]),
        ("distributional_fidelity", results["distributional_fidelity"]["overall_pass"]),
        ("cross_cell_correlation", results["cross_cell_correlation"]["overall_pass"]),
        ("mean_reversion", results["mean_reversion"]["overall_pass"]),
        ("pathwise_jump_realism", results["pathwise_jump_realism"]["overall_pass"]),
    ]
    failed = [name for name, passed in ordered if not passed]
    n_pass = sum(int(passed) for _name, passed in ordered)
    results["summary"] = {"n_pass": n_pass, "n_total": 10, "failed_suites": failed}
    return results


def _selected_metrics(results: dict[str, Any]) -> dict[str, Any]:
    coverage = results["coverage"]["overall"]
    return {
        "n_pass_without_conditionality": results["summary"]["n_pass"],
        "failed": results["summary"]["failed_suites"],
        "coverage90": coverage.get(0.9, coverage.get("0.9")),
        "daily_ks_pass": results["distributional_fidelity"]["ks_test"]["n_pass"],
        "level_ks_pass": results["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "median_bias_pass": results["distributional_fidelity"]["median_bias"]["n_pass"],
        "bias_magnitude_pass": results["distributional_fidelity"]["median_bias"][
            "n_mag_pass"
        ],
        "regime_layer2_pass": results["regime_coverage"]["layer2_n_passing"],
        "mean_reversion_ratio": results["mean_reversion"]["mr_gt_ratio"],
        "mean_reversion_active_rate": results["mean_reversion"]["active_pass_rate"],
        "pathwise_max_jump_ks": results["pathwise_jump_realism"]["pathwise_max_jump"][
            "ks_stat"
        ],
        "pathwise_q99_pass": results["pathwise_jump_realism"]["per_cell_q99"]["n_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="493a oracle audit for path-level constant shifts of 392a samples"
    )
    parser.add_argument("--model_type", type=str, default="340c")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--logit_eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    model.eval()
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
        hist_batch = normalize_iv(batch.history_01[start:end])
        with torch.no_grad():
            samp = model.sample_batched(
                hist_batch,
                n_samples=args.samples,
                n_steps=args.future_len,
                chunk_size=args.chunk_size,
            )
        outputs.append(samp.detach().cpu().numpy())
    base_samples = np.concatenate(outputs, axis=0)
    shifted_samples = _oracle_window_cell_logit_shift(
        base_samples,
        batch.future_01.detach().cpu().numpy(),
        eps=args.logit_eps,
    )

    raw = np.load(args.data_path)
    returns = raw["ret"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    rollout_start = max_train_idx - args.val_size
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    variants = {
        "base_392a_resample": base_samples,
        "oracle_window_cell_logit_shift": shifted_samples,
    }
    results = {
        "config": vars(args) | {"checkpoint_epoch": int(payload.get("epoch", -1))},
        "variants": {},
    }
    for name, samples in variants.items():
        variant_results = _evaluate_samples(
            samples=samples,
            ground_truth=ground_truth,
            history_01=history_01,
            returns=returns,
            rollout_start=rollout_start,
            history_len=args.history_len,
            future_len=args.future_len,
        )
        results["variants"][name] = variant_results
        results["variants"][name]["selected_metrics"] = _selected_metrics(
            variant_results
        )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_01.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        "",
    ]
    for name, variant_results in results["variants"].items():
        metrics = variant_results["selected_metrics"]
        lines.extend(
            [
                f"**{name}**",
                f"- score without conditionality rerun: `{metrics['n_pass_without_conditionality']}/10`",
                f"- failed suites: `{', '.join(metrics['failed']) if metrics['failed'] else 'none'}`",
                f"- coverage90: `{metrics['coverage90']:.4f}`",
                f"- level KS pass cells: `{metrics['level_ks_pass']}/25`",
                f"- daily-change KS pass cells: `{metrics['daily_ks_pass']}/25`",
                f"- regime layer2 pass: `{metrics['regime_layer2_pass']}/8`",
                f"- MR ratio: `{metrics['mean_reversion_ratio']:.3f}`",
                f"- pathwise max-jump KS: `{metrics['pathwise_max_jump_ks']:.3f}`",
                f"- pathwise q99 pass cells: `{metrics['pathwise_q99_pass']}/25`",
                "",
            ]
        )
    write_markdown_summary(
        args.output_md,
        "493a 392a Path-Level Shift Oracle",
        lines,
    )
    print(json.dumps(make_serializable({k: v["selected_metrics"] for k, v in results["variants"].items()}), indent=2))


if __name__ == "__main__":
    main()
