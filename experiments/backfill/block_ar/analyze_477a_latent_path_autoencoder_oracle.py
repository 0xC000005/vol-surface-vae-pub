#!/usr/bin/env python
"""477a: reconstruction-oracle diagnostic for the 476a latent path autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.latent_path_manifold_flow import load_model  # noqa: E402
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (  # noqa: E402
    run_block_ar_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_surface_validity_tests,
    run_time_series_tests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
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

    recons: list[np.ndarray] = []
    latent_stds: list[float] = []
    target_stds: list[float] = []
    recon_stds: list[float] = []
    for start in range(0, batch.future_norm.shape[0], args.batch_size):
        end = min(start + args.batch_size, batch.future_norm.shape[0])
        fut_norm = batch.future_norm[start:end].to(device)
        with torch.no_grad():
            future_scores = model.target_future_scores(fut_norm)
            latent = model.encode_future_scores(future_scores)
            recon_scores = model.decode_future_scores(latent)
            flat = recon_scores.reshape(-1, model.cfg.n_cells)
            recon_01 = model._scores_to_values(flat).view(
                recon_scores.shape[0],
                model.cfg.future_len,
                5,
                5,
            )
        recons.append(recon_01.cpu().numpy())
        latent_stds.append(float(latent.std(unbiased=False).item()))
        target_stds.append(float(future_scores.std(unbiased=False).item()))
        recon_stds.append(float(recon_scores.std(unbiased=False).item()))

    recon = np.concatenate(recons, axis=0)
    ground_truth = batch.future_01.detach().cpu().numpy()
    history = batch.history_01.detach().cpu().numpy()

    # Repeat deterministic reconstructions only for APIs that expect sample axis.
    cond_samples = np.repeat(recon[:, None], int(args.n_repeats), axis=1)

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    time_series = run_time_series_tests(cond_samples, ground_truth)
    block_ar = run_block_ar_tests(cond_samples)
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)

    abs_err = np.abs(recon - ground_truth)
    diff_gt = np.diff(ground_truth, axis=1)
    diff_recon = np.diff(recon, axis=1)
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_stage": payload.get("stage"),
        "n_windows": int(ground_truth.shape[0]),
        "n_repeats": int(args.n_repeats),
        "iv_mae": float(abs_err.mean()),
        "iv_p95_abs_error": float(np.quantile(abs_err, 0.95)),
        "iv_max_abs_error": float(abs_err.max()),
        "score_latent_std_mean": float(np.mean(latent_stds)),
        "target_score_std_mean": float(np.mean(target_stds)),
        "recon_score_std_mean": float(np.mean(recon_stds)),
        "score_std_ratio": float(np.mean(recon_stds) / max(np.mean(target_stds), 1e-12)),
        "daily_change_std_ratio": float(diff_recon.std() / max(diff_gt.std(), 1e-12)),
        "daily_change_mae": float(np.abs(diff_recon - diff_gt).mean()),
        "geometry_passes": {
            "surface": bool(surface["overall_pass"]),
            "time_series": bool(time_series["overall_pass"]),
            "block_ar": bool(block_ar["overall_pass"]),
            "distributional_fidelity": bool(distributional["overall_pass"]),
            "cross_cell_correlation": bool(cross_cell["overall_pass"]),
            "mean_reversion": bool(mean_reversion["overall_pass"]),
            "pathwise_jump_realism": bool(pathwise["overall_pass"]),
        },
        "key_metrics": {
            "daily_ks_pass_cells": int(distributional["ks_test"]["n_pass"]),
            "level_ks_pass_cells": int(distributional["ks_level_test"]["n_pass"]),
            "median_bias_pass_cells": int(distributional["median_bias"]["n_pass"]),
            "cross_cell_corr_ratio": float(cross_cell["corr_ratio"]),
            "cross_cell_rank_ratio": float(cross_cell["rank_ratio"]),
            "mean_reversion_ratio": float(mean_reversion["mr_gt_ratio"]),
            "mean_reversion_active_pass_rate": float(mean_reversion["active_pass_rate"]),
            "pathwise_max_jump_ks": float(pathwise["pathwise_max_jump"]["ks_stat"]),
        },
    }
    results = {
        "summary": summary,
        "surface": surface,
        "time_series": time_series,
        "block_ar": block_ar,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
