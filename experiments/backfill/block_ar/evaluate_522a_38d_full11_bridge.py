#!/usr/bin/env python
"""522a: score existing 38-d joint baselines on the official IV full 11-suite.

The older 38-d benchmark used daily-change windows whose first generated IV
surface is one day after the official full-suite h1 target. This bridge builds
official-aligned 38-d conditioning windows:

    official history surfaces: s ... s+H-1
    38-d observed changes:     s-1 ... s+H-2
    forecast anchor:           surface s+H-1
    generated changes:         s+H-1 ... s+H+F-2

Reconstructing generated IV changes from that anchor gives official future
surfaces s+H ... s+H+F-1 exactly.
"""

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

from experiments.backfill.baselines.data_loader_38d import load_aligned_38d_data
from experiments.backfill.baselines.evaluate_baselines_38d import (
    CLASSICAL_BASELINES,
    DEEP_BASELINES,
    create_classical_baseline,
    load_deep_baseline,
    reconstruct_iv_surfaces,
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


def official_rollout_start(
    test_start: int,
    val_size: int,
    history_len: int,
    future_len: int,
) -> int:
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    return max_train_idx - int(val_size)


def build_official_aligned_38d_windows(
    data: dict[str, Any],
    test_start: int,
    val_size: int,
    history_len: int = 30,
    future_len: int = 30,
    max_windows: int | None = None,
) -> dict[str, np.ndarray]:
    """Build 38-d windows aligned to ``build_rollout_windows(..., split="val")``."""
    rollout_start = official_rollout_start(test_start, val_size, history_len, future_len)
    max_train_idx = int(test_start) - int(history_len) - int(future_len)
    indices = np.arange(rollout_start, max_train_idx, dtype=np.int64)
    if max_windows is not None:
        indices = indices[: int(max_windows)]
    if len(indices) == 0:
        raise ValueError("No official validation indices selected")
    if int(indices[0]) < 1:
        raise ValueError("Official-aligned 38-d windows require one pre-history change")

    surfaces = data["surfaces"]
    joint_changes = data["joint_changes_38"]
    factor_levels = data["factor_levels_13"]

    history_changes = []
    future_changes = []
    anchor_surfaces = []
    history_surfaces = []
    history_factor_levels = []
    for s in indices:
        history_changes.append(joint_changes[s - 1 : s - 1 + history_len])
        future_changes.append(
            joint_changes[s + history_len - 1 : s + history_len - 1 + future_len]
        )
        anchor_surfaces.append(surfaces[s + history_len - 1])
        history_surfaces.append(surfaces[s : s + history_len])
        history_factor_levels.append(factor_levels[s : s + history_len])

    return {
        "indices": indices,
        "history_changes": np.asarray(history_changes, dtype=np.float32),
        "future_changes": np.asarray(future_changes, dtype=np.float32),
        "anchor_surfaces": np.asarray(anchor_surfaces, dtype=np.float32),
        "history_surfaces": np.asarray(history_surfaces, dtype=np.float32),
        "history_factor_levels": np.asarray(history_factor_levels, dtype=np.float32),
    }


def alignment_diagnostics(
    windows: dict[str, np.ndarray],
    batch: Any,
) -> dict[str, float]:
    reconstructed_future = reconstruct_iv_surfaces(
        windows["future_changes"][:, None, :, :25],
        windows["anchor_surfaces"],
    ).squeeze(1)
    history_01 = batch.history_01.detach().cpu().numpy()
    future_01 = batch.future_01.detach().cpu().numpy()
    return {
        "history_max_abs_error": float(np.max(np.abs(windows["history_surfaces"] - history_01))),
        "future_reconstruction_max_abs_error": float(np.max(np.abs(reconstructed_future - future_01))),
        "history_mean_abs_error": float(np.mean(np.abs(windows["history_surfaces"] - history_01))),
        "future_reconstruction_mean_abs_error": float(np.mean(np.abs(reconstructed_future - future_01))),
    }


def load_baseline(name: str, data: dict[str, Any], device: torch.device, train_stop: int):
    if name in CLASSICAL_BASELINES:
        train_changes = data["joint_changes_38"][:train_stop]
        return create_classical_baseline(name, data, train_changes)
    if name in DEEP_BASELINES:
        return load_deep_baseline(name, str(device))
    raise ValueError(f"Unknown 38-d baseline: {name}")


def generate_joint_samples(
    name: str,
    model: Any,
    windows: dict[str, np.ndarray],
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    all_samples = []
    n_windows = int(windows["history_changes"].shape[0])
    for start in range(0, n_windows, int(batch_size)):
        end = min(start + int(batch_size), n_windows)
        kwargs: dict[str, np.ndarray] = {}
        if name == "historical_sim":
            kwargs["surfaces"] = windows["history_surfaces"][start:end]
            kwargs["factor_levels"] = windows["history_factor_levels"][start:end]
        batch_samples = model.sample_joint(
            windows["history_changes"][start:end],
            n_samples=int(n_samples),
            **kwargs,
        )
        all_samples.append(batch_samples)
        print(f"  generated windows {end}/{n_windows}", flush=True)
    return np.concatenate(all_samples, axis=0)


def summarize_results(results: dict[str, Any], baseline: str, alignment: dict[str, float]) -> list[str]:
    summary = results["summary"]
    coverage = results["coverage"]
    conditionality = results["conditionality"]
    regime = results["regime_coverage"]
    distributional = results["distributional_fidelity"]
    pathwise = results["pathwise_jump_realism"]
    cross_cell = results["cross_cell_correlation"]
    mean_rev = results["mean_reversion"]
    return [
        f"- baseline: `{baseline}`",
        "- source: existing 38-d joint IV+factor baseline, official-aligned to full 11-suite",
        f"- suite score: `{summary['n_pass']}/11`",
        f"- failed suites: `{', '.join(summary['failed_suites']) if summary['failed_suites'] else 'none'}`",
        f"- history alignment max error: `{alignment['history_max_abs_error']:.3e}`",
        f"- future reconstruction max error: `{alignment['future_reconstruction_max_abs_error']:.3e}`",
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
    parser.add_argument("--baseline", default="csdi", choices=CLASSICAL_BASELINES + DEEP_BASELINES)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=522)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_json",
        default="results/autoresearch/522a_38d_full11_bridge/csdi_full11.json",
    )
    parser.add_argument(
        "--output_md",
        default="results/autoresearch/522a_38d_full11_bridge/csdi_full11.md",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = load_aligned_38d_data()
    windows = build_official_aligned_38d_windows(
        data=data,
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_windows,
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
    alignment = alignment_diagnostics(windows, batch)
    if alignment["history_max_abs_error"] > 1e-6 or alignment["future_reconstruction_max_abs_error"] > 1e-6:
        raise RuntimeError(f"Official alignment failed: {alignment}")

    rollout_start = official_rollout_start(
        args.test_start, args.val_size, args.history_len, args.future_len
    )
    train_stop = args.test_start - args.history_len - args.future_len - args.val_size - 1
    print(f"Loading baseline `{args.baseline}` on {device}")
    model = load_baseline(args.baseline, data, device, train_stop=max(1, train_stop))
    if hasattr(model, "eval"):
        model.eval()

    print(f"Generating {args.samples} samples for {windows['history_changes'].shape[0]} windows")
    t0 = time.time()
    joint_samples = generate_joint_samples(
        name=args.baseline,
        model=model,
        windows=windows,
        n_samples=args.samples,
        batch_size=args.batch_size,
    )
    gen_time = time.time() - t0
    cond_samples = reconstruct_iv_surfaces(
        joint_samples[:, :, :, :25],
        windows["anchor_surfaces"],
    ).astype(np.float32)

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
        "baseline": args.baseline,
        "mode": "522a_official_aligned_38d_full11_bridge",
        "n_windows": int(windows["history_changes"].shape[0]),
        "samples": int(args.samples),
        "generation_time_s": float(gen_time),
        "device": str(device),
        "seed": int(args.seed),
        "rollout_start": int(rollout_start),
        "train_stop_changes_for_classical": int(max(1, train_stop)),
        "alignment": alignment,
        "deployability_note": (
            "The bridge uses pre-existing 38-d baseline checkpoints when a deep baseline "
            "is selected; it is a mechanics/frontier diagnostic, not a new deployable claim."
        ),
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown_summary(
        out_md,
        "522a 38-d Baseline Official Full 11 Bridge",
        summarize_results(results, args.baseline, alignment),
    )
    print(json.dumps(make_serializable(results["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
