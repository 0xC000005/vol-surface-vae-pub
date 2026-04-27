#!/usr/bin/env python
"""614a: score generic likelihood-trained AR transition checkpoints on the IV suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_gaussian_transition_law import load_model  # noqa: E402
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import load_aligned_iv_factor_panel  # noqa: E402
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    build_unified_increment_block,
    clean_nonpositive_log_level_factors,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    FixedDeployableSampler,
    history_key,
    run_suite,
    set_seed,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import select_scope  # noqa: E402


def build_val_history(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Any]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    if args.clean_nonpositive_log_levels:
        panel, _cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
        )
    _train_indices, val_indices = official_train_val_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
    )
    if args.max_windows is not None and args.max_windows > 0:
        val_indices = val_indices[: int(args.max_windows)]
    block = build_unified_increment_block(
        panel,
        columns,
        val_indices,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        iv_count=int(args.iv_count),
    )
    scope = payload.get("state_scope", args.state_scope)
    history, future, specs = select_scope(block, scope, int(args.iv_count))
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    actual = [spec.name for spec in specs]
    if expected and expected != actual:
        raise RuntimeError("checkpoint state specs do not match rebuilt validation specs")
    return history.astype(np.float32), future.astype(np.float32), block


def alignment_diagnostics(block: Any, batch: Any, n_windows: int) -> dict[str, float]:
    history_iv = block.history_state[:n_windows, :, :25].reshape(n_windows, -1)
    future_iv = block.future_state[:n_windows, :, :25].reshape(n_windows, -1)
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
    history: np.ndarray,
    *,
    samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    iv_count: int,
    sample_temperature: float,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, int(history.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(history.shape[0]))
        hist = torch.from_numpy(history[start:end]).to(device)
        panel_samples = model.sample_batched(
            hist,
            n_samples=int(samples),
            n_steps=int(n_steps),
            chunk_size=int(chunk_size),
            temperature=float(sample_temperature),
        )
        arr = panel_samples.detach().cpu().numpy()[..., :iv_count]
        chunks.append(arr.reshape(end - start, int(samples), int(n_steps), 5, 5).astype(np.float32))
        print(f"  generated windows {end}/{history.shape[0]}", flush=True)
    return np.concatenate(chunks, axis=0)


def summarize_results(results: dict[str, Any], alignment: dict[str, float]) -> list[str]:
    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    regime = results["regime_coverage"]
    distributional = results["distributional_fidelity"]
    cross_cell = results["cross_cell_correlation"]
    mean_rev = results["mean_reversion"]
    pathwise = results["pathwise_jump_realism"]
    return [
        "- source: `614a generic Gaussian AR transition likelihood`",
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
        f"- median-bias cells: `{distributional['median_bias']['n_pass']}/25`",
        f"- corr ratio/rank ratio: `{cross_cell['corr_ratio']:.3f}` / `{cross_cell['rank_ratio']:.3f}`",
        f"- mean-reversion active pass: `{mean_rev.get('active_pass_rate', float('nan')):.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--state_scope", choices=["iv_only", "joint38"], default="joint38")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=614)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    history, _future, block = build_val_history(args, payload)
    cfg = payload["config"]
    n_windows = min(int(args.max_windows), int(history.shape[0]))
    history = history[:n_windows]
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(n_windows),
        device=device,
        split="val",
    )
    alignment = alignment_diagnostics(block, batch, n_windows)
    if alignment["history_max_abs_error"] > 1e-6 or alignment["future_max_abs_error"] > 1e-6:
        raise RuntimeError(f"614a/full-suite alignment failed: {alignment}")

    print(f"Generating {args.samples} samples for {n_windows} windows")
    t0 = time.time()
    cond_samples = generate_iv_samples(
        model,
        history,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
        sample_temperature=float(args.sample_temperature),
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
        "mode": "614a_unified_ar_gaussian_likelihood",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
        "checkpoint_config": cfg,
        "state_scope": payload.get("state_scope", args.state_scope),
        "n_windows": int(n_windows),
        "samples": int(args.samples),
        "n_steps": int(args.n_steps),
        "sample_temperature": float(args.sample_temperature),
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
        "614a Unified AR Gaussian Likelihood Official IV 11-Suite",
        summarize_results(results, alignment),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
