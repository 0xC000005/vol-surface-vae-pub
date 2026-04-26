#!/usr/bin/env python
"""Evaluate 555a local-factor conditioned core fine-tune on the full suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._local_factor_conditioning_555_utils import (  # noqa: E402
    build_local_factor_history_block,
    tensor_dict_summary,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.evaluate_525a_factor_conditioned_surface_fm import (  # noqa: E402
    HistoryKeyedFactorLiveSampler,
    sample_factor_conditioned,
)


def summary_lines(results: dict[str, Any], block_summary: dict[str, Any], checkpoint: str) -> list[str]:
    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    regime = results["regime_coverage"]
    distributional = results["distributional_fidelity"]
    pathwise = results["pathwise_jump_realism"]
    mean_rev = results["mean_reversion"]
    cointegration = results["cointegration"]
    return [
        f"- checkpoint: `{checkpoint}`",
        f"- windows: `{block_summary['n_windows']}`",
        f"- local factor columns: `{', '.join(block_summary['factor_columns'])}`",
        f"- conditionality mode: `{results['config']['conditionality_mode']}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- cointegration ratio: `{cointegration.get('gen_gt_ratio', float('nan')):.3f}`",
        f"- regime layer2: `{regime['layer2_n_passing']}/{regime['layer2_n_total']}`",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument(
        "--conditionality_mode",
        choices=["live", "fixed"],
        default="live",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
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

    factor_dim = int(getattr(model.cfg, "factor_dim", 0))
    if factor_dim <= 0:
        raise ValueError("555a checkpoints must have factor_dim > 0")
    factor_mean = payload.get("factor_mean")
    factor_std = payload.get("factor_std")
    if factor_mean is None or factor_std is None:
        raise ValueError("555a checkpoints must contain factor_mean and factor_std")
    _, val_indices = official_train_val_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    if args.max_windows is not None:
        val_indices = val_indices[: args.max_windows]
    factor_block = build_local_factor_history_block(
        data_path=args.data_path,
        indices=val_indices,
        history_len=args.history_len,
        future_len=args.future_len,
        device=device,
        factor_mean=factor_mean,
        factor_std=factor_std,
    )
    max_hist_err = float(
        torch.max(torch.abs(factor_block.history_01 - batch.history_01)).detach().cpu().item()
    )
    max_future_err = float(
        torch.max(torch.abs(factor_block.future_01 - batch.future_01)).detach().cpu().item()
    )
    if max_hist_err > 1e-6 or max_future_err > 1e-6:
        raise RuntimeError(
            f"Official/local factor block alignment failed: history={max_hist_err}, future={max_future_err}"
        )
    block_summary = tensor_dict_summary(factor_block)

    t0 = time.time()
    cond_samples = sample_factor_conditioned(
        model=model,
        history_norm=batch.history_norm,
        factor_history=factor_block.factor_history,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    sample_time = time.time() - t0
    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    if args.conditionality_mode == "fixed":
        samples_by_key = {
            history_key(hist_norm_np[i]): cond_samples[i]
            for i in range(hist_norm_np.shape[0])
        }
        suite_model = FixedDeployableSampler(samples_by_key).eval()
    else:
        suite_model = HistoryKeyedFactorLiveSampler(
            model=model,
            history_norm_np=hist_norm_np,
            factor_history=factor_block.factor_history,
        ).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=suite_model,
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
    results["config"] = {
        "model_type": "555a",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": int(args.samples),
        "conditionality_samples": int(args.conditionality_samples),
        "sample_time_s": float(sample_time),
        "conditionality_mode": args.conditionality_mode,
        "factor_block": block_summary,
        "alignment": {
            "history_max_abs_error": max_hist_err,
            "future_max_abs_error": max_future_err,
        },
        "seed": int(args.seed),
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown_summary(
        out_md,
        "555a Local-Factor Conditioned Core Fine-Tune",
        summary_lines(results, block_summary, args.checkpoint),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
