#!/usr/bin/env python
"""Evaluate 525a factor-conditioned surface-level FM on the official full suite."""

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

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (
    load_model,
)
from experiments.backfill.block_ar._factor_conditioning_525_utils import (
    build_factor_history_block,
    official_train_val_indices,
    tensor_dict_summary,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)


def sample_factor_conditioned(
    model: Any,
    history_norm: torch.Tensor,
    factor_history: torch.Tensor | None,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
) -> np.ndarray:
    outs = []
    for start in range(0, history_norm.shape[0], batch_size):
        end = min(start + batch_size, history_norm.shape[0])
        samples = model.sample_batched(
            history_norm[start:end],
            n_samples=n_samples,
            n_steps=n_steps,
            chunk_size=chunk_size,
            history_is_normalized=True,
            factor_history=None if factor_history is None else factor_history[start:end],
        )
        outs.append(samples.detach().cpu().numpy())
        print(f"  sampled windows {end}/{history_norm.shape[0]}", flush=True)
    return np.concatenate(outs, axis=0).astype(np.float32)


def summary_lines(results: dict[str, Any], block_summary: dict[str, Any], checkpoint: str) -> list[str]:
    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    regime = results["regime_coverage"]
    distributional = results["distributional_fidelity"]
    pathwise = results["pathwise_jump_realism"]
    mean_rev = results["mean_reversion"]
    return [
        f"- checkpoint: `{checkpoint}`",
        f"- windows: `{block_summary['n_windows']}`",
        f"- factor dim: `{block_summary['factor_history_shape'][-1]}`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        "",
        "**Key Metrics**",
        f"- cov90 overall: `{coverage['overall'][0.9]:.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- conditionality MAE reduction: `{conditionality.get('mae_reduction_pct', float('nan')):.2f}%`",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=525)
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
    factor_history = None
    block_summary = {
        "n_windows": int(batch.history_norm.shape[0]),
        "history_shape": list(batch.history_01.shape),
        "future_shape": list(batch.future_01.shape),
        "factor_history_shape": [int(batch.history_01.shape[0]), args.history_len, 0],
        "index_start": None,
        "index_end": None,
        "factor_dim": 0,
    }
    max_hist_err = 0.0
    max_future_err = 0.0
    if factor_dim > 0:
        factor_mean = payload.get("factor_mean")
        factor_std = payload.get("factor_std")
        if factor_mean is None or factor_std is None:
            raise ValueError("factor-conditioned checkpoints must contain factor_mean and factor_std")
        _, val_indices = official_train_val_indices(
            test_start=args.test_start,
            val_size=args.val_size,
            history_len=args.history_len,
            future_len=args.future_len,
        )
        if args.max_windows is not None:
            val_indices = val_indices[: args.max_windows]
        factor_block = build_factor_history_block(
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
                f"Official/factor block alignment failed: history={max_hist_err}, future={max_future_err}"
            )
        factor_history = factor_block.factor_history
        block_summary = tensor_dict_summary(factor_block)

    t0 = time.time()
    cond_samples = sample_factor_conditioned(
        model=model,
        history_norm=batch.history_norm,
        factor_history=factor_history,
        n_samples=args.samples,
        n_steps=args.future_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    sample_time = time.time() - t0
    hist_norm_np = batch.history_norm.detach().cpu().numpy()
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(hist_norm_np.shape[0])}
    fixed_model = FixedDeployableSampler(samples_by_key).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=fixed_model,
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
        "model_type": "525a",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "n_windows": int(batch.history_norm.shape[0]),
        "samples": int(args.samples),
        "conditionality_samples": int(args.conditionality_samples),
        "sample_time_s": float(sample_time),
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
        "525a Factor-Conditioned Surface-Level FM",
        summary_lines(results, block_summary, args.checkpoint),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
