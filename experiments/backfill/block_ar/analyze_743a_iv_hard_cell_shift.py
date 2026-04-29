#!/usr/bin/env python
"""743a: audit whether IV hard-cell coverage failures are train/val shift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)


def ks_2samp_stat(x: np.ndarray, y: np.ndarray) -> float:
    """Small dependency-free two-sample KS statistic."""
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([x, y]))
    cx = np.searchsorted(x, grid, side="right") / float(x.size)
    cy = np.searchsorted(y, grid, side="right") / float(y.size)
    return float(np.max(np.abs(cx - cy)))


def parse_hard_keys(localization: dict[str, Any]) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    for item in localization.get("stable_coverage_under70_all_temperatures", []):
        horizon, cell = item["key"]
        row, col = cell
        keys.append(
            {
                "horizon": int(horizon),
                "row": int(row),
                "col": int(col),
                "cell_index": int(row) * 5 + int(col),
                "baseline_coverage": float(item["coverages"]["temp1000_baseline"]),
            }
        )
    return keys


def split_block(split: str, payload: dict[str, Any], args: argparse.Namespace) -> Any:
    ns = SimpleNamespace(
        eval_split=split,
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
    *_, block = build_val_block(ns, payload)
    return block


def quantile_summary(x: np.ndarray) -> dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def interval_containment(train: np.ndarray, val: np.ndarray, lo: float, hi: float) -> float:
    qlo, qhi = np.quantile(train, [lo, hi])
    val = np.asarray(val)
    return float(np.mean((val >= qlo) & (val <= qhi)))


def audit_pair(
    train_block: Any,
    val_block: Any,
    hard_keys: list[dict[str, Any]],
    *,
    iv_count: int = 25,
) -> dict[str, Any]:
    train_level = train_block.future_state[:, :, :iv_count]
    val_level = val_block.future_state[:, :, :iv_count]
    train_last = train_block.history_state[:, -1:, :iv_count]
    val_last = val_block.history_state[:, -1:, :iv_count]
    train_cum = train_level - train_last
    val_cum = val_level - val_last
    train_inc = train_block.future_increment[:, :, :iv_count]
    val_inc = val_block.future_increment[:, :, :iv_count]

    level_ks = np.zeros((30, iv_count), dtype=np.float64)
    cum_ks = np.zeros((30, iv_count), dtype=np.float64)
    inc_ks = np.zeros((30, iv_count), dtype=np.float64)
    for h in range(30):
        for c in range(iv_count):
            level_ks[h, c] = ks_2samp_stat(train_level[:, h, c], val_level[:, h, c])
            cum_ks[h, c] = ks_2samp_stat(train_cum[:, h, c], val_cum[:, h, c])
            inc_ks[h, c] = ks_2samp_stat(train_inc[:, h, c], val_inc[:, h, c])

    hard_rows: list[dict[str, Any]] = []
    for key in hard_keys:
        h0 = int(key["horizon"]) - 1
        c = int(key["cell_index"])
        tr_level = train_level[:, h0, c]
        va_level = val_level[:, h0, c]
        tr_cum = train_cum[:, h0, c]
        va_cum = val_cum[:, h0, c]
        tr_inc_path = train_inc[:, : h0 + 1, c].reshape(-1)
        va_inc_path = val_inc[:, : h0 + 1, c].reshape(-1)
        tr_std = float(np.std(tr_level))
        hard_rows.append(
            {
                **key,
                "level_ks_train_tail_vs_val": float(level_ks[h0, c]),
                "cum_change_ks_train_tail_vs_val": float(cum_ks[h0, c]),
                "path_increment_ks_train_tail_vs_val": ks_2samp_stat(tr_inc_path, va_inc_path),
                "level_train_tail": quantile_summary(tr_level),
                "level_val": quantile_summary(va_level),
                "cum_change_train_tail": quantile_summary(tr_cum),
                "cum_change_val": quantile_summary(va_cum),
                "val_in_train_tail_80_interval_level": interval_containment(tr_level, va_level, 0.10, 0.90),
                "val_in_train_tail_90_interval_level": interval_containment(tr_level, va_level, 0.05, 0.95),
                "val_in_train_tail_95_interval_level": interval_containment(tr_level, va_level, 0.025, 0.975),
                "val_mean_shift_in_train_tail_std_level": float(
                    (np.mean(va_level) - np.mean(tr_level)) / max(tr_std, 1e-8)
                ),
            }
        )

    def grid_summary(grid: np.ndarray) -> dict[str, float | int]:
        flat = grid.reshape(-1)
        return {
            "median": float(np.median(flat)),
            "p90": float(np.quantile(flat, 0.90)),
            "max": float(np.max(flat)),
            "n_gt_020": int(np.sum(flat > 0.20)),
            "n_total": int(flat.size),
        }

    return {
        "n_train_tail_windows": int(train_level.shape[0]),
        "n_val_windows": int(val_level.shape[0]),
        "grid_level_ks": grid_summary(level_ks),
        "grid_cum_change_ks": grid_summary(cum_ks),
        "grid_increment_ks": grid_summary(inc_ks),
        "hard_cells": hard_rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 743a IV Hard-Cell Train/Validation Shift Audit",
        "",
        "## Summary",
    ]
    summary = report["summary"]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Hard Cells", ""])
    lines.append(
        "| horizon | cell | base cov | level KS | cum KS | inc KS | val in train 90 | mean shift std |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in report["audit"]["hard_cells"]:
        lines.append(
            "| {horizon} | ({row},{col}) | {baseline_coverage:.3f} | "
            "{level_ks_train_tail_vs_val:.3f} | {cum_change_ks_train_tail_vs_val:.3f} | "
            "{path_increment_ks_train_tail_vs_val:.3f} | {val_in_train_tail_90_interval_level:.3f} | "
            "{val_mean_shift_in_train_tail_std_level:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Grid Shift",
            "",
            f"- level KS train-tail vs val: `{report['audit']['grid_level_ks']}`",
            f"- cumulative-change KS train-tail vs val: `{report['audit']['grid_cum_change_ks']}`",
            f"- one-day-increment KS train-tail vs val: `{report['audit']['grid_increment_ks']}`",
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
    parser.add_argument("--output_dir", default="results/block_ar/743a_iv_hard_cell_shift")
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

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    localization = json.loads(Path(args.localization_json).read_text(encoding="utf-8"))
    hard_keys = parse_hard_keys(localization)

    train_tail = split_block("train_tail", payload, args)
    val = split_block("val", payload, args)
    audit = audit_pair(train_tail, val, hard_keys)

    hard_level_ks = [row["level_ks_train_tail_vs_val"] for row in audit["hard_cells"]]
    hard_containment = [row["val_in_train_tail_90_interval_level"] for row in audit["hard_cells"]]
    shifted_hard = sum(ks > 0.20 for ks in hard_level_ks)
    high_containment = sum(cov >= 0.80 for cov in hard_containment)
    mechanism_read = (
        "The hard cells show meaningful train-tail/validation distribution shift if their level KS "
        "exceeds 0.20, but validation targets remain mostly inside train-tail empirical intervals "
        "if the 90% containment is high. That pattern separates data drift from pure impossibility."
    )
    if shifted_hard and high_containment >= max(1, len(hard_containment) // 2):
        decision = (
            "Treat the strict coverage defect as a mixed problem: there is real local validation shift, "
            "but not enough to declare the target impossible. The next model-side experiment should target "
            "local state/cell/horizon uncertainty allocation, not global widening or another scalar loss."
        )
    elif shifted_hard:
        decision = (
            "Treat the defect primarily as a split-shift/oracle limitation before adding model complexity. "
            "A model-side fix is unlikely to be clean unless it uses a generic robustness/local-geometry mechanism."
        )
    else:
        decision = (
            "Treat the defect primarily as model-side uncertainty allocation failure because the hard cells "
            "are not strongly shifted from train-tail targets."
        )

    report = {
        "config": {
            "checkpoint": args.checkpoint,
            "localization_json": args.localization_json,
            "test_start": args.test_start,
            "val_size": args.val_size,
            "max_windows": args.max_windows,
        },
        "hard_key_source": hard_keys,
        "audit": audit,
        "summary": {
            "hard_cells_with_level_ks_gt_020": shifted_hard,
            "hard_cells_total": len(hard_level_ks),
            "hard_cells_with_val_inside_train_tail_90_ge_080": high_containment,
            "median_hard_level_ks": round(float(np.median(hard_level_ks)), 6) if hard_level_ks else None,
            "median_hard_val_in_train_tail_90": round(float(np.median(hard_containment)), 6)
            if hard_containment
            else None,
        },
        "mechanism_read": mechanism_read,
        "decision": decision,
    }

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "summary.md")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
