#!/usr/bin/env python
"""
220m oracle slow-state persistence diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from experiments.backfill.block_ar.analyze_220_persistence_diagnostics import (
    _boundary_persistence,
    _corr_summary,
    _jump_cluster_stats,
    _label_regimes,
    _mean_acf,
    _sign_run_stats,
    _state_conditioned_profiles,
    _transition_stats,
)
from experiments.backfill.block_ar.evaluate_220m_oracle_slow_state import (
    compute_slow_surface_series,
    oracle_rollout_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="220m oracle slow-state persistence diagnostics")
    parser.add_argument("--base_model_type", type=str, default="212ai")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--oracle_alpha", type=float, default=0.08)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    import experiments.backfill.block_ar.evaluate_220m_oracle_slow_state as oracle_mod

    oracle_mod.args = args

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _payload = load_one_day_kernel(args.base_model_type, args.checkpoint, device)
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
    slow_series = compute_slow_surface_series(surfaces, alpha=args.oracle_alpha)

    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_windows]
    oracle_prev_slow = slow_series[val_indices + args.history_len - 1]
    oracle_future_slow = np.stack(
        [slow_series[idx + args.history_len : idx + args.history_len + args.future_len] for idx in val_indices],
        axis=0,
    )

    cond_samples = oracle_rollout_samples(
        model=model,
        history_norm=batch.history_norm,
        oracle_prev_slow=oracle_prev_slow,
        oracle_future_slow=oracle_future_slow,
        n_samples=args.samples,
        chunk_size=args.chunk_size,
    )
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    gt_levels = np.concatenate([history_01[:, -1:, :, :], ground_truth], axis=1)
    prev_expanded = np.broadcast_to(
        history_01[:, None, -1:, :, :],
        (history_01.shape[0], cond_samples.shape[1], 1, 5, 5),
    )
    gen_levels = np.concatenate([prev_expanded, cond_samples], axis=2)

    gt_mean = gt_levels.mean(axis=(2, 3))
    gen_mean = gen_levels.mean(axis=(-1, -2))
    gt_delta_mean = np.diff(gt_mean, axis=1)
    gen_delta_mean = np.diff(gen_mean, axis=2)

    gt_move_score = np.abs(gt_delta_mean)
    q20 = float(np.quantile(gt_move_score, 0.2))
    q80 = float(np.quantile(gt_move_score, 0.8))
    gt_states = _label_regimes(gt_move_score, q20, q80)
    gen_states = _label_regimes(np.abs(gen_delta_mean), q20, q80)

    gt_delta_h30 = ground_truth[:, 29].reshape(ground_truth.shape[0], -1) - ground_truth[:, 28].reshape(ground_truth.shape[0], -1)
    gen_delta_h30 = cond_samples[:, :, 29].reshape(cond_samples.shape[0] * cond_samples.shape[1], -1) - cond_samples[:, :, 28].reshape(cond_samples.shape[0] * cond_samples.shape[1], -1)
    gt_level_h30 = ground_truth[:, 29].reshape(ground_truth.shape[0], -1)
    gen_level_h30 = cond_samples[:, :, 29].reshape(cond_samples.shape[0] * cond_samples.shape[1], -1)

    jump_threshold = float(np.quantile(gt_move_score.reshape(-1), 0.95))

    results = {
        "model_type": "220m_oracle",
        "base_model_type": args.base_model_type,
        "oracle_alpha": args.oracle_alpha,
        "acf": {
            "gt_mean_level": _mean_acf(gt_mean, 10).tolist(),
            "gen_mean_level": _mean_acf(gen_mean.reshape(-1, gen_mean.shape[-1]), 10).tolist(),
            "mean_level_corr": float(np.corrcoef(_mean_acf(gt_mean, 10), _mean_acf(gen_mean.reshape(-1, gen_mean.shape[-1]), 10))[0, 1]),
            "gt_mean_delta": _mean_acf(gt_delta_mean, 10).tolist(),
            "gen_mean_delta": _mean_acf(gen_delta_mean.reshape(-1, gen_delta_mean.shape[-1]), 10).tolist(),
            "mean_delta_corr": float(np.corrcoef(_mean_acf(gt_delta_mean, 10), _mean_acf(gen_delta_mean.reshape(-1, gen_delta_mean.shape[-1]), 10))[0, 1]),
            "gt_abs_mean_delta": _mean_acf(np.abs(gt_delta_mean), 10).tolist(),
            "gen_abs_mean_delta": _mean_acf(np.abs(gen_delta_mean).reshape(-1, gen_delta_mean.shape[-1]), 10).tolist(),
            "abs_mean_delta_corr": float(np.corrcoef(_mean_acf(np.abs(gt_delta_mean), 10), _mean_acf(np.abs(gen_delta_mean).reshape(-1, gen_delta_mean.shape[-1]), 10))[0, 1]),
        },
        "regime_persistence": {
            "thresholds": {"q20": q20, "q80": q80},
            "gt": _transition_stats(gt_states),
            "gen": _transition_stats(gen_states.reshape(-1, gen_states.shape[-1])),
        },
        "h30_structure": {
            "gt_delta": _corr_summary(gt_delta_h30),
            "gen_delta": _corr_summary(gen_delta_h30),
            "gt_level": _corr_summary(gt_level_h30),
            "gen_level": _corr_summary(gen_level_h30),
        },
        "path_realism": {
            "sign_runs": {
                "gt": _sign_run_stats(gt_delta_mean),
                "gen": _sign_run_stats(gen_delta_mean.reshape(-1, gen_delta_mean.shape[-1])),
            },
            "jump_cluster": {
                "gt": _jump_cluster_stats(gt_move_score, jump_threshold),
                "gen": _jump_cluster_stats(np.abs(gen_delta_mean).reshape(-1, gen_delta_mean.shape[-1]), jump_threshold),
            },
            "boundary": {
                "gt": _boundary_persistence(gt_levels[:, 1:]),
                "gen": _boundary_persistence(cond_samples.reshape(-1, cond_samples.shape[2], 5, 5)),
            },
        },
        "state_conditioned": _state_conditioned_profiles(history_01, ground_truth, cond_samples, [1, 5, 10, 20, 30]),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- oracle base model: `{args.base_model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{args.max_windows}`",
        f"- samples: `{args.samples}`",
        f"- oracle alpha: `{args.oracle_alpha}`",
        "",
        "**ACF**",
        f"- mean-level ACF corr: `{results['acf']['mean_level_corr']:.3f}`",
        f"- mean-delta ACF corr: `{results['acf']['mean_delta_corr']:.3f}`",
        f"- abs-mean-delta ACF corr: `{results['acf']['abs_mean_delta_corr']:.3f}`",
        "",
        "**Regime Persistence**",
        f"- GT calm self-transition: `{results['regime_persistence']['gt']['self_transition']['calm']:.3f}`",
        f"- Oracle gen calm self-transition: `{results['regime_persistence']['gen']['self_transition']['calm']:.3f}`",
        f"- GT turb self-transition: `{results['regime_persistence']['gt']['self_transition']['turb']:.3f}`",
        f"- Oracle gen turb self-transition: `{results['regime_persistence']['gen']['self_transition']['turb']:.3f}`",
        "",
        "**Temporal-Spatial h30**",
        f"- delta corr ratio h30: `{results['h30_structure']['gen_delta']['mean_corr'] / max(results['h30_structure']['gt_delta']['mean_corr'], 1e-8):.3f}`",
        f"- delta rank ratio h30: `{results['h30_structure']['gen_delta']['eff_rank'] / max(results['h30_structure']['gt_delta']['eff_rank'], 1e-8):.3f}`",
        f"- level corr ratio h30: `{results['h30_structure']['gen_level']['mean_corr'] / max(results['h30_structure']['gt_level']['mean_corr'], 1e-8):.3f}`",
        "",
        "**Path Realism**",
        f"- jump cluster lag1 GT/gen: `{results['path_realism']['jump_cluster']['gt']['lag1']:.3f}` / `{results['path_realism']['jump_cluster']['gen']['lag1']:.3f}`",
        f"- floor day incidence GT/gen: `{results['path_realism']['boundary']['gt']['floor_day_incidence']:.3%}` / `{results['path_realism']['boundary']['gen']['floor_day_incidence']:.3%}`",
        "",
        "**State Conditioned**",
        f"- h30 level-vs-drift Spearman GT/gen: `{results['state_conditioned']['per_horizon']['30']['spearman']['level_vs_gt_drift']:.3f}` / `{results['state_conditioned']['per_horizon']['30']['spearman']['level_vs_gen_drift']:.3f}`",
        f"- h30 vov-vs-width Spearman gen: `{results['state_conditioned']['per_horizon']['30']['spearman']['vov_vs_gen_width']:.3f}`",
    ]
    write_markdown_summary(args.output_md, "220m Oracle Slow-State Persistence Diagnostics", lines)
    print(json.dumps({"status": "ok", "model_type": "220m_oracle"}, indent=2))


if __name__ == "__main__":
    main()
