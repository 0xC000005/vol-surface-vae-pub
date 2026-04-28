#!/usr/bin/env python
"""664a condition-use and same-history diversity diagnostics for 662/663 checkpoints."""

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

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_662a_normalized_innovation_diagnostics import (  # noqa: E402
    build_scope,
    generate_increments,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    reconstruct_state_from_increments,
)


def make_deranged_permutation(n: int, *, seed: int) -> np.ndarray:
    """Return a deterministic permutation with no fixed points when possible."""
    n_int = int(n)
    if n_int < 2:
        return np.arange(n_int, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    base = np.arange(n_int, dtype=np.int64)
    for _ in range(128):
        perm = rng.permutation(n_int).astype(np.int64)
        if np.all(perm != base):
            return perm
    return np.roll(base, 1)


def condition_diversity_summary(
    target: np.ndarray,
    original_samples: np.ndarray,
    shuffled_samples: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    target_arr = np.asarray(target, dtype=np.float64)
    original_arr = np.asarray(original_samples, dtype=np.float64)
    shuffled_arr = np.asarray(shuffled_samples, dtype=np.float64)
    original_mean = np.mean(original_arr, axis=1)
    shuffled_mean = np.mean(shuffled_arr, axis=1)
    original_mae = float(np.mean(np.abs(original_mean - target_arr)))
    shuffled_mae = float(np.mean(np.abs(shuffled_mean - target_arr)))
    within_std = float(np.mean(np.std(original_arr, axis=1)))
    paired_effect = float(np.mean(np.abs(original_arr - shuffled_arr)))
    return {
        "original_mean_mae": original_mae,
        "shuffled_mean_mae": shuffled_mae,
        "shuffle_mae_ratio": float(shuffled_mae / max(original_mae, eps)),
        "within_sample_std": within_std,
        "paired_condition_effect": paired_effect,
        "paired_condition_effect_over_std": float(paired_effect / max(within_std, eps)),
    }


def subset_summary(
    target: np.ndarray,
    original: np.ndarray,
    shuffled: np.ndarray,
    start: int,
    end: int,
) -> dict[str, float]:
    return condition_diversity_summary(
        target[..., start:end],
        original[..., start:end],
        shuffled[..., start:end],
    )


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# 664a Condition-Diversity Diagnostics",
        "",
        f"- scope: `{summary['scope']}`",
        f"- windows/samples/steps: `{summary['n_windows']}` / `{summary['n_samples']}` / `{summary['n_steps']}`",
        f"- permutation fixed-point rate: `{summary['permutation_fixed_point_rate']:.4f}`",
        "",
        "**Encoded Increment Law**",
    ]
    for key in [
        "original_mean_mae",
        "shuffled_mean_mae",
        "shuffle_mae_ratio",
        "within_sample_std",
        "paired_condition_effect",
        "paired_condition_effect_over_std",
    ]:
        lines.append(f"- {key}: `{summary['encoded_increment'][key]:.6g}`")
    lines.extend(["", "**Raw State Law**"])
    for key in [
        "original_mean_mae",
        "shuffled_mean_mae",
        "shuffle_mae_ratio",
        "within_sample_std",
        "paired_condition_effect",
        "paired_condition_effect_over_std",
    ]:
        lines.append(f"- {key}: `{summary['raw_state'][key]:.6g}`")
    if "iv_raw" in summary:
        lines.extend(["", "**IV Raw Subset**"])
        for key in [
            "original_mean_mae",
            "shuffled_mean_mae",
            "shuffle_mae_ratio",
            "within_sample_std",
            "paired_condition_effect",
            "paired_condition_effect_over_std",
        ]:
            lines.append(f"- {key}: `{summary['iv_raw'][key]:.6g}`")
    if "factor_raw" in summary:
        lines.extend(["", "**Factor Raw Subset**"])
        for key in [
            "original_mean_mae",
            "shuffled_mean_mae",
            "shuffle_mae_ratio",
            "within_sample_std",
            "paired_condition_effect",
            "paired_condition_effect_over_std",
        ]:
            lines.append(f"- {key}: `{summary['factor_raw'][key]:.6g}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state_scope", choices=["iv_only", "anchor_only", "joint38"], default="joint38")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=664)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.eval()
    (
        scope,
        history_level,
        history_norm,
        future_norm,
        center,
        scale,
        raw_history,
        raw_future,
        specs,
    ) = build_scope(args, payload)
    n = min(int(args.max_windows), int(history_level.shape[0]))
    history_level = history_level[:n]
    history_norm = history_norm[:n]
    future_norm = future_norm[:n, : int(args.n_steps), :]
    center = center[:n]
    scale = scale[:n]
    raw_history = raw_history[:n]
    raw_future = raw_future[:n, : int(args.n_steps), :]
    perm = make_deranged_permutation(n, seed=int(args.seed) + 1009)

    t0 = time.time()
    sample_seed = int(args.seed) + 2003
    set_seed(sample_seed)
    original_increment = generate_increments(
        model,
        history_level,
        history_norm,
        center,
        scale,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        temperature=float(args.sample_temperature),
    )
    set_seed(sample_seed)
    shuffled_increment = generate_increments(
        model,
        history_level[perm],
        history_norm[perm],
        center[perm],
        scale[perm],
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        temperature=float(args.sample_temperature),
    )
    gt_increment = future_norm * np.maximum(scale[:, None, :], 1e-12)
    original_raw = reconstruct_state_from_increments(raw_history[:, -1, :], original_increment, specs)
    shuffled_raw = reconstruct_state_from_increments(raw_history[perm, -1, :], shuffled_increment, specs)
    fixed_rate = float(np.mean(perm == np.arange(n))) if n > 0 else 0.0
    effective_iv_count = int(args.iv_count) if scope in {"iv_only", "joint38"} else 0
    summary: dict[str, Any] = {
        "scope": scope,
        "n_windows": int(n),
        "n_samples": int(args.samples),
        "n_steps": int(args.n_steps),
        "n_channels": int(original_increment.shape[-1]),
        "permutation_fixed_point_rate": fixed_rate,
        "encoded_increment": condition_diversity_summary(
            gt_increment,
            original_increment,
            shuffled_increment,
        ),
        "raw_state": condition_diversity_summary(
            raw_future,
            original_raw,
            shuffled_raw,
        ),
    }
    if effective_iv_count > 0:
        summary["iv_raw"] = subset_summary(
            raw_future,
            original_raw,
            shuffled_raw,
            0,
            effective_iv_count,
        )
    if original_raw.shape[-1] > effective_iv_count:
        summary["factor_raw"] = subset_summary(
            raw_future,
            original_raw,
            shuffled_raw,
            effective_iv_count,
            int(original_raw.shape[-1]),
        )
    result = {
        "summary": summary,
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
            "state_scope": scope,
            "sample_temperature": float(args.sample_temperature),
            "seed": int(args.seed),
            "sample_seed": sample_seed,
            "generation_time_s": float(time.time() - t0),
        },
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(result), indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
