#!/usr/bin/env python
"""588a: score unified increment-flow checkpoints on the official IV 11-suite."""

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

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.audit_583a_unified_flow_sample_quality import (  # noqa: E402
    load_model_for_audit,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.train_577a_unified_increment_flow import (  # noqa: E402
    _reconstruct_samples,
)


def alignment_diagnostics(val_block: Any, batch: Any, n_windows: int) -> dict[str, float]:
    history_iv = val_block.history_state[:n_windows, :, :25].reshape(n_windows, -1)
    future_iv = val_block.future_state[:n_windows, :, :25].reshape(n_windows, -1)
    batch_history = batch.history_01.detach().cpu().numpy()[:n_windows].reshape(n_windows, -1)
    batch_future = batch.future_01.detach().cpu().numpy()[:n_windows].reshape(n_windows, -1)
    return {
        "history_max_abs_error": float(np.max(np.abs(history_iv - batch_history))),
        "future_max_abs_error": float(np.max(np.abs(future_iv - batch_future))),
        "history_mean_abs_error": float(np.mean(np.abs(history_iv - batch_history))),
        "future_mean_abs_error": float(np.mean(np.abs(future_iv - batch_future))),
    }


@torch.no_grad()
def generate_iv_samples(
    model: Any,
    payload: dict[str, Any],
    val_block: Any,
    val_hist: torch.Tensor,
    *,
    n_windows: int,
    samples: int,
    sample_steps: int,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    common = {
        "increment_transform": payload.get("increment_transform", "standard"),
        "inc_mean": payload["increment_mean"],
        "inc_std": payload["increment_std"],
        "inc_quantiles": payload.get("increment_quantiles"),
        "normal_levels": payload.get("normal_score_levels"),
        "specs": val_block.specs,
    }
    for start in range(0, int(n_windows), int(batch_size)):
        end = min(start + int(batch_size), int(n_windows))
        inc = model.sample(
            val_hist[start:end],
            n_samples=int(samples),
            n_steps=int(sample_steps),
        )
        state = _reconstruct_samples(
            val_block.history_state[start:end],
            inc.detach().cpu().numpy(),
            **common,
        )
        iv = state[..., :25].reshape(end - start, int(samples), model.cfg.future_len, 5, 5)
        chunks.append(iv.astype(np.float32))
        print(f"  generated windows {end}/{n_windows}", flush=True)
    return np.concatenate(chunks, axis=0)


def summarize_results(results: dict[str, Any], alignment: dict[str, float]) -> list[str]:
    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    regime = results["regime_coverage"]
    distributional = results["distributional_fidelity"]
    pathwise = results["pathwise_jump_realism"]
    cross_cell = results["cross_cell_correlation"]
    mean_rev = results["mean_reversion"]
    return [
        "- source: `unified IV-plus-anchor-factor increment-flow checkpoint`",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        f"- history alignment max error: `{alignment['history_max_abs_error']:.3e}`",
        f"- future alignment max error: `{alignment['future_max_abs_error']:.3e}`",
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
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/587a_conditional_affine_cumulative_flow_s587/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--source_temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=588)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload, _train_block, val_block, val_hist = load_model_for_audit(
        args.checkpoint,
        args=args,
        device=device,
    )
    model.source_temperature = float(args.source_temperature)
    cfg = payload["config"]
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(args.max_windows),
        device=device,
        split="val",
    )
    n_windows = min(int(args.max_windows), int(val_hist.shape[0]), int(batch.history_01.shape[0]))
    alignment = alignment_diagnostics(val_block, batch, n_windows)
    if alignment["history_max_abs_error"] > 1e-6 or alignment["future_max_abs_error"] > 1e-6:
        raise RuntimeError(f"Unified flow/full-suite alignment failed: {alignment}")

    print(f"Generating {args.samples} samples for {n_windows} windows")
    t0 = time.time()
    cond_samples = generate_iv_samples(
        model,
        payload,
        val_block,
        val_hist,
        n_windows=n_windows,
        samples=int(args.samples),
        sample_steps=int(args.sample_steps),
        batch_size=int(args.batch_size),
    )
    generation_time = time.time() - t0
    hist_norm_np = batch.history_norm.detach().cpu().numpy()[:n_windows]
    samples_by_key = {history_key(hist_norm_np[i]): cond_samples[i] for i in range(n_windows)}
    fixed_model = FixedDeployableSampler(samples_by_key).eval()
    results = run_suite(
        cond_samples=cond_samples,
        batch=batch,
        model=fixed_model,
        data_path=args.data_path,
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        batch_size=int(args.batch_size),
        conditionality_samples=int(args.conditionality_samples),
        conditionality_max_batches=int(args.conditionality_max_batches),
        device=device,
    )
    results["config"] = {
        "mode": "588a_unified_flow_full11_bridge",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
        "checkpoint_config": cfg,
        "n_windows": int(n_windows),
        "samples": int(args.samples),
        "sample_steps": int(args.sample_steps),
        "source_temperature": float(args.source_temperature),
        "generation_time_s": float(generation_time),
        "device": str(device),
        "seed": int(args.seed),
        "alignment": alignment,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown_summary(
        out_md,
        "588a Unified Flow Official Full 11 Bridge",
        summarize_results(results, alignment),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
