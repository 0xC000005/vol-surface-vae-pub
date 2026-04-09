#!/usr/bin/env python
"""
220o1 evaluation: recursive multi-day rollout for the jointly trained slow-fast
support-aware state-space model.
"""

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
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_220m_oracle_slow_state import run_custom_conditionality
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
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
from experiments.backfill.block_ar.train_220o1_joint_slow_fast_support_state_space import (
    build_slow_factor_params,
    compute_slow_surface_series,
    load_model,
    slow_surfaces_to_factors,
)


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    ordered = [
        ("surface", results["surface"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("conditionality", results["conditionality"]["overall_pass"]),
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
    return sum(int(passed) for _name, passed in ordered), failed


@torch.no_grad()
def rollout_samples(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    slow_history_01: torch.Tensor,
    slow_hist_factor: torch.Tensor,
    n_samples: int,
    future_len: int,
    chunk_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    batch_size, hist_len = history_01.shape[:2]
    chunk_size = max(1, min(int(chunk_size), int(n_samples)))

    all_chunks: list[np.ndarray] = []
    for start in range(0, n_samples, chunk_size):
        k = min(chunk_size, n_samples - start)
        hist_k = history_01.unsqueeze(1).expand(batch_size, k, hist_len, 5, 5)
        hist_k = hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()
        slow_hist_k = slow_history_01.unsqueeze(1).expand(batch_size, k, hist_len, 5, 5)
        slow_hist_k = slow_hist_k.reshape(batch_size * k, hist_len, 5, 5).clone()
        slow_factor_hist_k = slow_hist_factor.unsqueeze(1).expand(batch_size, k, hist_len, slow_hist_factor.shape[-1])
        slow_factor_hist_k = slow_factor_hist_k.reshape(batch_size * k, hist_len, slow_hist_factor.shape[-1]).clone()

        frames: list[torch.Tensor] = []
        for _step in range(future_len):
            next_iv, pred_slow_iv, pred_factor = model.sample_next_iv(
                history_01=hist_k,
                slow_history_01=slow_hist_k,
                slow_hist_factor=slow_factor_hist_k,
                n_samples=1,
            )
            next_iv = next_iv.squeeze(1)
            frames.append(next_iv.view(batch_size, k, 5, 5))
            hist_k = torch.cat([hist_k[:, 1:], next_iv.unsqueeze(1)], dim=1)
            slow_hist_k = torch.cat([slow_hist_k[:, 1:], pred_slow_iv.unsqueeze(1)], dim=1)
            slow_factor_hist_k = torch.cat([slow_factor_hist_k[:, 1:], pred_factor.unsqueeze(1)], dim=1)
        all_chunks.append(torch.stack(frames, dim=2).cpu().numpy())
    return np.concatenate(all_chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 220o1 joint slow-fast support-aware state-space")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
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
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    returns = raw["ret"].astype(np.float64)
    slow_surfaces = compute_slow_surface_series(surfaces, alpha=payload["config"]["slow_alpha"])
    max_train_idx = args.test_start - args.history_len - args.future_len
    fit_end_idx = max_train_idx - args.val_size + args.history_len
    factor_params = build_slow_factor_params(
        slow_surfaces=slow_surfaces,
        fit_end_idx=fit_end_idx,
        factor_dim=payload["config"]["factor_dim"],
        eps=payload["config"]["support_eps"],
    )
    slow_factors = slow_surfaces_to_factors(slow_surfaces, factor_params)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_windows]
    slow_history_01 = np.stack([slow_surfaces[idx : idx + args.history_len] for idx in val_indices], axis=0)
    slow_hist_factor = np.stack([slow_factors[idx : idx + args.history_len] for idx in val_indices], axis=0)

    cond_samples = rollout_samples(
        model=model,
        history_01=batch.history_01,
        slow_history_01=torch.from_numpy(slow_history_01).to(device=device, dtype=batch.history_01.dtype),
        slow_hist_factor=torch.from_numpy(slow_hist_factor).to(device=device, dtype=torch.float32),
        n_samples=args.samples,
        future_len=args.future_len,
        chunk_size=args.chunk_size,
    )
    uncond_samples = rollout_samples(
        model=model,
        history_01=torch.zeros_like(batch.history_01),
        slow_history_01=torch.from_numpy(slow_history_01).to(device=device, dtype=batch.history_01.dtype),
        slow_hist_factor=torch.from_numpy(slow_hist_factor).to(device=device, dtype=torch.float32),
        n_samples=min(args.samples, 32),
        future_len=args.future_len,
        chunk_size=args.chunk_size,
    )

    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()
    rollout_start = max_train_idx - args.val_size

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)
    conditionality = run_custom_conditionality(cond_samples, uncond_samples, ground_truth, history_01)
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
            "model_type": "220o1_joint_slow_fast_support_state_space",
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "slow_alpha": float(payload["config"]["slow_alpha"]),
            "n_windows": int(batch.history_01.shape[0]),
            "samples": args.samples,
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
    floor_day_incidence = float(np.any(cond_samples <= 0.001, axis=(2, 3, 4)).mean())
    overall_coverage_90 = coverage["overall"].get(0.9, coverage["overall"].get("0.9", float("nan")))
    results["summary"] = {
        "n_pass": n_pass,
        "n_total": 11,
        "failed_suites": failed,
        "mean_reversion_ratio_h1": float(mean_reversion.get("mr_gt_ratio", float("nan"))),
        "mean_reversion_ratio_h30": float(
            mean_reversion.get("full_horizon", {}).get("per_horizon", {}).get(30, {}).get("ratio", float("nan"))
        ),
        "turb_calm_width_ratio_h30": float(conditionality.get("turb_calm_ratio", float("nan"))),
        "h30_delta_corr_ratio": float(cross_cell.get("corr_ratio", float("nan"))),
        "h30_level_corr_ratio": float(cross_cell.get("rank_ratio", float("nan"))),
        "level_ks_pass_cells_h30": int(distributional["ks_level_test"]["n_pass"]),
        "daily_change_ks_pass_cells_h30": int(distributional["ks_test"]["n_pass"]),
        "floor_day_incidence": floor_day_incidence,
        "max_jump_ks_h30": float(pathwise["pathwise_max_jump"]["ks_stat"]),
        "coverage_90": float(overall_coverage_90),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_01.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/11`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        f"- h1/h30 MR ratio: `{results['summary']['mean_reversion_ratio_h1']:.3f} / {results['summary']['mean_reversion_ratio_h30']:.3f}`",
        f"- turb/calm ratio: `{results['summary']['turb_calm_width_ratio_h30']:.3f}`",
        f"- floor-day incidence: `{floor_day_incidence:.3%}`",
        f"- max-jump KS: `{results['summary']['max_jump_ks_h30']:.3f}`",
        f"- level KS pass cells: `{results['summary']['level_ks_pass_cells_h30']}/25`",
        f"- daily-change KS pass cells: `{results['summary']['daily_change_ks_pass_cells_h30']}/25`",
    ]
    write_markdown_summary(args.output_md, "220o1 Joint Slow-Fast Support State-Space Validation", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
