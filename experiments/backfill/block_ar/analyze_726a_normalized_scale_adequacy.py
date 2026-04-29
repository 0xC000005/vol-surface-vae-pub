#!/usr/bin/env python
"""726a: check whether normalized innovations remove OAS scale shift.

725a showed raw OAS validation tails are much calmer than the training tail.
This diagnostic asks whether the 662a normalized-innovation coordinate already
removes that shift. If it does, the failure is conditional generation. If it
does not, the coordinate itself is not stationary enough for sticky channels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    panel_daily_changes,
    select_raw_state_scope,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import build_blocks  # noqa: E402
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    select_normalized_innovation_scope,
)


FOCUS_NAMES = [
    "factor:spx",
    "factor:us2y",
    "factor:us10y",
    "factor:aaa_oas",
    "factor:bbb_oas",
]


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"q50": 0.0, "q90": 0.0, "q95": 0.0, "q99": 0.0}
    return {
        "q50": float(np.quantile(finite, 0.50)),
        "q90": float(np.quantile(finite, 0.90)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
    }


def scope_stats(
    block: Any,
    scope: str,
    *,
    iv_count: int,
    scale_half_life: float | None,
    scale_floor: float,
    center_mode: str,
    drift_feature_mode: str,
    zero_eps: float,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    (
        _level,
        _history_norm,
        _future_level,
        future_norm,
        _center,
        scale,
        _drift,
        _raw,
        specs,
    ) = select_normalized_innovation_scope(
        block,
        scope,
        iv_count,
        scale_half_life=scale_half_life,
        scale_floor=scale_floor,
        center_mode=center_mode,
        drift_feature_mode=drift_feature_mode,
    )
    raw_history, raw_future = select_raw_state_scope(block, scope, iv_count)
    raw_delta = panel_daily_changes(raw_history, raw_future)
    names = [spec.name for spec in specs]
    rows: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(names):
        raw = np.asarray(raw_delta[..., idx], dtype=np.float64).reshape(-1)
        zero = np.abs(raw) <= float(zero_eps)
        raw_abs_nonzero = np.abs(raw[~zero])
        norm_abs_nonzero = np.abs(np.asarray(future_norm[..., idx], dtype=np.float64).reshape(-1)[~zero])
        scale_values = np.asarray(scale[:, idx], dtype=np.float64).reshape(-1)
        rows[name] = {
            "zero_rate": float(np.mean(zero)),
            "raw_abs": quantiles(raw_abs_nonzero),
            "normalized_abs": quantiles(norm_abs_nonzero),
            "history_scale": quantiles(scale_values),
        }
    return rows, names


def build_ratio_rows(
    train_stats: dict[str, dict[str, Any]],
    val_stats: dict[str, dict[str, Any]],
    names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        train = train_stats[name]
        val = val_stats[name]
        rows.append(
            {
                "name": name,
                "train_zero": train["zero_rate"],
                "val_zero": val["zero_rate"],
                "raw_train_q99": train["raw_abs"]["q99"],
                "raw_val_q99": val["raw_abs"]["q99"],
                "raw_val_to_train_q99": val["raw_abs"]["q99"] / max(train["raw_abs"]["q99"], 1e-12),
                "norm_train_q99": train["normalized_abs"]["q99"],
                "norm_val_q99": val["normalized_abs"]["q99"],
                "norm_val_to_train_q99": val["normalized_abs"]["q99"] / max(train["normalized_abs"]["q99"], 1e-12),
                "scale_train_q90": train["history_scale"]["q90"],
                "scale_val_q90": val["history_scale"]["q90"],
                "scale_val_to_train_q90": val["history_scale"]["q90"] / max(train["history_scale"]["q90"], 1e-12),
            }
        )
    return rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 726a Normalized-Scale Adequacy",
        "",
        "## Anchor Scope",
        "",
        "| name | raw q99 train/val | raw val/train | norm q99 train/val | norm val/train | scale q90 train/val | scale val/train |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["anchor_rows"]:
        if row["name"] not in FOCUS_NAMES:
            continue
        lines.append(
            f"| `{row['name']}` | {row['raw_train_q99']:.3f}/{row['raw_val_q99']:.3f} | "
            f"{row['raw_val_to_train_q99']:.2f} | {row['norm_train_q99']:.2f}/{row['norm_val_q99']:.2f} | "
            f"{row['norm_val_to_train_q99']:.2f} | {row['scale_train_q90']:.4f}/{row['scale_val_q90']:.4f} | "
            f"{row['scale_val_to_train_q90']:.2f} |"
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
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    parser.add_argument("--zero_eps", type=float, default=1e-10)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    scale_half_life = None if float(args.scale_half_life) <= 0.0 else float(args.scale_half_life)
    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    train_stats, names = scope_stats(
        train_block,
        "anchor_only",
        iv_count=int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=float(args.scale_floor),
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
        zero_eps=float(args.zero_eps),
    )
    val_stats, val_names = scope_stats(
        val_block,
        "anchor_only",
        iv_count=int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=float(args.scale_floor),
        center_mode=args.center_mode,
        drift_feature_mode=args.drift_feature_mode,
        zero_eps=float(args.zero_eps),
    )
    if names != val_names:
        raise RuntimeError("train/validation specs differ")
    rows = build_ratio_rows(train_stats, val_stats, names)
    row_by_name = {row["name"]: row for row in rows}
    aaa = row_by_name["factor:aaa_oas"]
    bbb = row_by_name["factor:bbb_oas"]
    spx = row_by_name["factor:spx"]
    mechanism_read = (
        "Current history-RMS normalization removes most raw scale shift for ordinary channels, "
        f"for example SPX normalized q99 is {spx['norm_train_q99']:.2f} train versus {spx['norm_val_q99']:.2f} validation. "
        "It does not fully stationarize sticky OAS: "
        f"AAA normalized q99 remains {aaa['norm_train_q99']:.2f} train versus {aaa['norm_val_q99']:.2f} validation, "
        f"and BBB remains {bbb['norm_train_q99']:.2f} versus {bbb['norm_val_q99']:.2f}. "
        "The history scale itself is lower in validation, but the normalized target tail is still regime-shifted, so raw scale alone is not enough."
    )
    decision = (
        "The next repair should target the coordinate/objective for mixed-frequency channels, not the atom gate. "
        "A principled candidate is a deterministic frequency-aware tail coordinate or loss balance that makes nonzero updates comparable across regimes while preserving one shared AR flow framework."
    )
    report = {
        "iteration": "726a",
        "purpose": "normalized coordinate adequacy for sticky OAS nonzero updates",
        "panel_metadata": panel_metadata,
        "focus_names": FOCUS_NAMES,
        "anchor_rows": rows,
        "mechanism_read": mechanism_read,
        "decision": decision,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(output_md, report)
    print(json.dumps(make_serializable({"mechanism_read": mechanism_read, "decision": decision}), indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
