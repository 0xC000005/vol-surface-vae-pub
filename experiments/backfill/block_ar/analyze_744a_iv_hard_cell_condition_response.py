#!/usr/bin/env python
"""744a: audit incumbent hard-cell response to current-level state."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import load_model  # noqa: E402
from experiments.backfill.block_ar.analyze_743a_iv_hard_cell_shift import (  # noqa: E402
    parse_hard_keys,
    split_block,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import set_seed  # noqa: E402
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
    generate_iv_samples,
)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def tertile_indices(x: np.ndarray) -> dict[str, np.ndarray]:
    q1, q2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "low": np.where(x <= q1)[0],
        "mid": np.where((x > q1) & (x <= q2))[0],
        "high": np.where(x > q2)[0],
    }


def summarize_hard_cell(
    *,
    key: dict[str, Any],
    current_level: np.ndarray,
    realized: np.ndarray,
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    train_current_level: np.ndarray,
) -> dict[str, Any]:
    width = q95 - q05
    inside = (realized >= q05) & (realized <= q95)
    lower_miss = realized < q05
    upper_miss = realized > q95
    train_q05, train_q50, train_q95 = np.quantile(train_current_level, [0.05, 0.50, 0.95])
    current_std = float(np.std(train_current_level))
    row: dict[str, Any] = {
        **key,
        "current_level_train_tail_q05": float(train_q05),
        "current_level_train_tail_q50": float(train_q50),
        "current_level_train_tail_q95": float(train_q95),
        "current_level_val_q05": float(np.quantile(current_level, 0.05)),
        "current_level_val_q50": float(np.quantile(current_level, 0.50)),
        "current_level_val_q95": float(np.quantile(current_level, 0.95)),
        "current_level_val_mean_shift_train_std": float(
            (np.mean(current_level) - np.mean(train_current_level)) / max(current_std, 1e-8)
        ),
        "realized_vs_current_slope": slope(current_level, realized),
        "generated_median_vs_current_slope": slope(current_level, q50),
        "width_vs_current_corr": corr(current_level, width),
        "coverage90": float(np.mean(inside)),
        "lower_miss_rate": float(np.mean(lower_miss)),
        "upper_miss_rate": float(np.mean(upper_miss)),
        "median_bias_mean": float(np.mean(q50 - realized)),
        "width_mean": float(np.mean(width)),
    }
    for bucket, idx in tertile_indices(current_level).items():
        row[f"{bucket}_n"] = int(idx.size)
        row[f"{bucket}_coverage90"] = float(np.mean(inside[idx])) if idx.size else float("nan")
        row[f"{bucket}_lower_miss_rate"] = float(np.mean(lower_miss[idx])) if idx.size else float("nan")
        row[f"{bucket}_upper_miss_rate"] = float(np.mean(upper_miss[idx])) if idx.size else float("nan")
        row[f"{bucket}_median_bias_mean"] = float(np.mean(q50[idx] - realized[idx])) if idx.size else float("nan")
        row[f"{bucket}_width_mean"] = float(np.mean(width[idx])) if idx.size else float("nan")
    return row


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 744a IV Hard-Cell Conditioning Response Audit",
        "",
        "## Summary",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Hard Cells", ""])
    lines.append(
        "| horizon | cell | cov90 | lower miss | upper miss | current shift | real slope | gen slope | width/current corr | low cov | low bias |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["hard_cells"]:
        lines.append(
            "| {horizon} | ({row},{col}) | {coverage90:.3f} | {lower_miss_rate:.3f} | "
            "{upper_miss_rate:.3f} | {current_level_val_mean_shift_train_std:.3f} | "
            "{realized_vs_current_slope:.3f} | {generated_median_vs_current_slope:.3f} | "
            "{width_vs_current_corr:.3f} | {low_coverage90:.3f} | {low_median_bias_mean:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Mechanism Read",
            "",
            report["mechanism_read"],
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/674a_iv_channel_level_alltrain_w005_e3_s6731/best_model.pt")
    parser.add_argument("--localization_json", default="results/block_ar/741a_iv_coverage_localization/summary.json")
    parser.add_argument("--output_dir", default="results/block_ar/744a_iv_hard_cell_condition_response")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7441)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    localization = json.loads(Path(args.localization_json).read_text(encoding="utf-8"))
    hard_keys = parse_hard_keys(localization)

    val_args = SimpleNamespace(
        eval_split="val",
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        max_windows=int(args.max_windows),
        iv_count=25,
        state_scope="iv_only",
        clean_nonpositive_log_levels=True,
        positive_level_policy=args.positive_level_policy,
        iv_transform=args.iv_transform,
        iv_lower_bound=float(args.iv_lower_bound),
        iv_upper_bound=float(args.iv_upper_bound),
        scale_half_life=float(args.scale_half_life),
        scale_floor=float(args.scale_floor),
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
    )
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        val_block,
    ) = build_val_block(val_args, payload)
    n_windows = min(int(args.max_windows), int(history_level.shape[0]))
    train_tail = split_block("train_tail", payload, args)

    t0 = time.time()
    samples = generate_iv_samples(
        model,
        history_level[:n_windows],
        history_norm[:n_windows],
        center[:n_windows],
        scale[:n_windows],
        drift_feature[:n_windows],
        history_raw[:n_windows],
        specs,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=25,
        sample_temperature=float(args.sample_temperature),
    )
    generation_time_s = time.time() - t0

    hard_rows: list[dict[str, Any]] = []
    for key in hard_keys:
        h0 = int(key["horizon"]) - 1
        c = int(key["cell_index"])
        q05, q50, q95 = np.quantile(samples[:, :, h0, :, :].reshape(n_windows, int(args.samples), 25)[:, :, c], [0.05, 0.50, 0.95], axis=1)
        current_level = val_block.history_state[:n_windows, -1, c]
        realized = val_block.future_state[:n_windows, h0, c]
        train_current = train_tail.history_state[:, -1, c]
        hard_rows.append(
            summarize_hard_cell(
                key=key,
                current_level=current_level,
                realized=realized,
                q05=q05,
                q50=q50,
                q95=q95,
                train_current_level=train_current,
            )
        )

    low_cov = [row["low_coverage90"] for row in hard_rows]
    slope_gap = [
        abs(row["generated_median_vs_current_slope"] - row["realized_vs_current_slope"])
        for row in hard_rows
        if np.isfinite(row["generated_median_vs_current_slope"])
        and np.isfinite(row["realized_vs_current_slope"])
    ]
    lower_miss = [row["lower_miss_rate"] for row in hard_rows]
    summary = {
        "n_windows": n_windows,
        "samples": int(args.samples),
        "generation_time_s": round(float(generation_time_s), 3),
        "median_low_tertile_coverage90": round(float(np.median(low_cov)), 6),
        "median_hard_lower_miss_rate": round(float(np.median(lower_miss)), 6),
        "median_abs_slope_gap": round(float(np.median(slope_gap)), 6) if slope_gap else None,
    }

    median_low_cov = float(np.median(low_cov))
    median_lower_miss = float(np.median(lower_miss))
    if median_low_cov < 0.70 and median_lower_miss > 0.20:
        mechanism_read = (
            "The incumbent undercovers the shifted low-level validation region mostly through lower-tail misses. "
            "This means the model is not just too narrow globally; it is not translating the current level geometry "
            "far enough into the late-horizon lower tail for the hard cells."
        )
        decision = (
            "Next experiment should target generic local-geometry conditioning or robustness of the state encoder. "
            "Do not use scalar temperature or interval-score losses, because the miss is state-local and directional."
        )
    else:
        mechanism_read = (
            "The hard-cell misses are not concentrated in the low-level region; this favors a split-shift/test limitation "
            "or broad temporal factorization issue over a simple conditioning-response repair."
        )
        decision = (
            "Next step should be research ideation before any architecture change; avoid adding a local-geometry module "
            "unless the conditioning-response failure is directional and repeatable."
        )

    report = {
        "config": {
            "checkpoint": args.checkpoint,
            "localization_json": args.localization_json,
            "samples": int(args.samples),
            "n_steps": int(args.n_steps),
            "sample_temperature": float(args.sample_temperature),
            "seed": int(args.seed),
            "device": str(device),
        },
        "summary": summary,
        "hard_cells": hard_rows,
        "mechanism_read": mechanism_read,
        "decision": decision,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
