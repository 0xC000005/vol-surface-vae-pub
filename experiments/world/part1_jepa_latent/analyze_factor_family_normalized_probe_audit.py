from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.world.evaluation.factor_panel_data import (  # noqa: E402
    build_factor_panel_world_windows,
    make_factor_panel_future_targets,
)
from experiments.world.part1_jepa_latent.analyze_factor_panel_probe_bakeoff import (  # noqa: E402
    DEFAULT_SCALE_CHECKPOINT,
    FACTOR_TARGETS,
    RAW_FACTOR_FLAT,
    RAW_FACTOR_LAST,
    SCALE,
    SCALE_PLUS,
    _encode_optional_scale,
    _factor_feature_sets,
    _fmt,
    _serializable,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    concatenate_feature_blocks,
    regression_metrics,
)

FACTOR_FAMILIES = ("factor_level", "factor_return")


def _family_indices(columns: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {family: [] for family in FACTOR_FAMILIES}
    for idx, column in enumerate(columns):
        family = column.split(":", 1)[0]
        if family in out:
            out[family].append(idx)
    return out


def _standardize_targets(
    train_y: np.ndarray,
    val_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_y.mean(axis=0, keepdims=True)
    std = train_y.std(axis=0, keepdims=True)
    safe_std = np.maximum(std, 1e-6)
    return (
        ((train_y - mean) / safe_std).astype(np.float32),
        ((val_y - mean) / safe_std).astype(np.float32),
    )


def _standardize_feature_surfaces(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_out: dict[str, np.ndarray] = {}
    val_out: dict[str, np.ndarray] = {}
    for feature_name, train_x in train_features.items():
        val_x = val_features[feature_name]
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        safe_std = np.maximum(std, 1e-6)
        train_out[feature_name] = ((train_x - mean) / safe_std).astype(np.float32)
        val_out[feature_name] = ((val_x - mean) / safe_std).astype(np.float32)
    return train_out, val_out


def _probe_normalized_family(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    columns: list[str],
    alpha: float,
) -> dict[str, Any]:
    indices_by_family = _family_indices(columns)
    family_rows: dict[str, Any] = {}
    for family, indices in indices_by_family.items():
        if not indices:
            continue
        target_rows: dict[str, Any] = {}
        for target_name in FACTOR_TARGETS:
            train_y, val_y = _standardize_targets(
                train_targets[target_name][:, indices],
                val_targets[target_name][:, indices],
            )
            feature_rows: dict[str, Any] = {}
            for feature_name, train_x in train_features.items():
                pred = ridge_probe_predict(
                    train_x,
                    train_y,
                    val_features[feature_name],
                    alpha=alpha,
                )
                feature_rows[feature_name] = regression_metrics(pred, val_y)
            target_rows[target_name] = feature_rows
        family_rows[family] = target_rows
    return family_rows


def _cell_summary(
    family_probe: dict[str, Any],
    *,
    comparison_features: list[str],
) -> dict[str, dict[str, dict[str, float | str]]]:
    out: dict[str, dict[str, dict[str, float | str]]] = {}
    for family, target_rows in family_probe.items():
        out[family] = {}
        for target_name in FACTOR_TARGETS:
            feature_rows = target_rows[target_name]
            raw_last = float(feature_rows[RAW_FACTOR_LAST]["mse"])
            raw_flat = float(feature_rows[RAW_FACTOR_FLAT]["mse"])
            best_raw = min(raw_last, raw_flat)
            row: dict[str, float | str] = {
                "family": family,
                "target": target_name,
                "raw_factor_last_normalized_mse": raw_last,
                "raw_factor_flat_normalized_mse": raw_flat,
                "best_raw_normalized_mse": best_raw,
            }
            for feature in comparison_features:
                mse = float(feature_rows[feature]["mse"])
                row[f"{feature}_normalized_mse"] = mse
                row[f"{feature}_to_raw_factor_last_ratio"] = mse / raw_last
                row[f"{feature}_to_best_raw_ratio"] = mse / best_raw
            out[family][target_name] = row
    return out


def _count_cells(
    rows: dict[str, dict[str, dict[str, float | str]]],
    *,
    feature: str,
    baseline_key: str,
) -> int:
    wins = 0
    for target_rows in rows.values():
        for row in target_rows.values():
            wins += int(
                float(row[f"{feature}_normalized_mse"]) < float(row[baseline_key])
            )
    return wins


def run_factor_family_normalized_probe_audit(
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train_factor = build_factor_panel_world_windows(
        data_path=args.factor_data_path,
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_train_windows,
        normalize=False,
    )
    val_factor = build_factor_panel_world_windows(
        data_path=args.factor_data_path,
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_val_windows,
        normalize=False,
    )
    train_features = _factor_feature_sets(train_factor)
    val_features = _factor_feature_sets(val_factor)
    skipped_feature_surfaces: dict[str, str] = {}

    scale_train, scale_val, scale_skip = _encode_optional_scale(
        args.scale_checkpoint,
        args=args,
        device=device,
    )
    if scale_train is not None and scale_val is not None:
        if scale_train.shape[0] != train_factor.past_panel.shape[0]:
            raise ValueError("scale train rows do not match factor train rows")
        if scale_val.shape[0] != val_factor.past_panel.shape[0]:
            raise ValueError("scale val rows do not match factor val rows")
        train_features[SCALE] = scale_train
        val_features[SCALE] = scale_val
        train_features[SCALE_PLUS] = concatenate_feature_blocks(
            train_features[RAW_FACTOR_LAST],
            scale_train,
        )
        val_features[SCALE_PLUS] = concatenate_feature_blocks(
            val_features[RAW_FACTOR_LAST],
            scale_val,
        )
    else:
        skipped_feature_surfaces[SCALE] = str(scale_skip)
        skipped_feature_surfaces[SCALE_PLUS] = str(scale_skip)
    probe_train_features, probe_val_features = _standardize_feature_surfaces(
        train_features,
        val_features,
    )

    train_targets = make_factor_panel_future_targets(
        train_factor.past_panel,
        train_factor.future_panel,
        columns=train_factor.columns,
    )["regression"]
    val_targets = make_factor_panel_future_targets(
        val_factor.past_panel,
        val_factor.future_panel,
        columns=val_factor.columns,
    )["regression"]
    family_probe = _probe_normalized_family(
        probe_train_features,
        probe_val_features,
        train_targets,
        val_targets,
        columns=train_factor.columns,
        alpha=args.ridge_alpha,
    )
    comparison_features = [
        feature for feature in (SCALE, SCALE_PLUS) if feature in train_features
    ]
    family_summary = _cell_summary(
        family_probe,
        comparison_features=comparison_features,
    )
    family_column_counts = {
        family: len(indices)
        for family, indices in _family_indices(train_factor.columns).items()
    }
    target_family_cells = sum(
        len(target_rows) for target_rows in family_summary.values()
    )
    summary_counts: dict[str, int | None | str] = {
        "target_family_cells": int(target_family_cells),
        "scale_learned_best_raw_wins": (
            _count_cells(
                family_summary,
                feature=SCALE,
                baseline_key="best_raw_normalized_mse",
            )
            if SCALE in train_features
            else None
        ),
        "raw_factor_plus_scale_improvements": (
            _count_cells(
                family_summary,
                feature=SCALE_PLUS,
                baseline_key="raw_factor_last_normalized_mse",
            )
            if SCALE_PLUS in train_features
            else None
        ),
        "raw_factor_plus_scale_best_raw_wins": (
            _count_cells(
                family_summary,
                feature=SCALE_PLUS,
                baseline_key="best_raw_normalized_mse",
            )
            if SCALE_PLUS in train_features
            else None
        ),
    }
    decision_summary = {
        "probe_status": "factor_family_normalized_signal_audited",
        "scale_standalone_family_wins": (
            f"{summary_counts['scale_learned_best_raw_wins']}/{target_family_cells}"
            if summary_counts["scale_learned_best_raw_wins"] is not None
            else "n/a"
        ),
        "raw_plus_scale_improves_raw_last": (
            f"{summary_counts['raw_factor_plus_scale_improvements']}/{target_family_cells}"
            if summary_counts["raw_factor_plus_scale_improvements"] is not None
            else "n/a"
        ),
        "raw_plus_scale_beats_best_raw": (
            f"{summary_counts['raw_factor_plus_scale_best_raw_wins']}/{target_family_cells}"
            if summary_counts["raw_factor_plus_scale_best_raw_wins"] is not None
            else "n/a"
        ),
        "caveats": [
            "smoke_scale_128_train_64_val",
            "targets_standardized_with_train_family_statistics",
            "still_downstream_probe_only",
        ],
        "next_step": "interpret_family_normalized_audit_against_part1_gate",
    }
    result: dict[str, Any] = {
        "analysis": "world_model_factor_family_normalized_probe_audit",
        "date": "2026-05-11",
        "iteration": 180,
        "iteration_type": "experiment",
        "objective_family": "downstream_probe_factor_family_normalized",
        "target_metric_units": "train_standardized_per_factor_family",
        "feature_metric_units": "train_standardized_per_feature_surface",
        "uses_future_targets_as_pretraining": False,
        "uses_decoder": False,
        "device": str(device),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_shape": {
            "factor_past": list(train_factor.past_panel.shape),
            "factor_future": list(train_factor.future_panel.shape),
        },
        "val_shape": {
            "factor_past": list(val_factor.past_panel.shape),
            "factor_future": list(val_factor.future_panel.shape),
        },
        "factor_columns": train_factor.columns,
        "family_column_counts": family_column_counts,
        "feature_surfaces": list(train_features.keys()),
        "skipped_feature_surfaces": skipped_feature_surfaces,
        "family_probe_rows": family_probe,
        "family_summary": family_summary,
        "summary_counts": summary_counts,
        "decision_summary": decision_summary,
        "promotion_decision": "PROBE_ONLY_DO_NOT_PROMOTE",
        "part_b_blocked": True,
    }
    output_json = getattr(args, "output_json", None)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(
            json.dumps(_serializable(result), indent=2) + "\n",
            encoding="utf-8",
        )
    report_md = getattr(args, "report_md", None)
    if report_md:
        Path(report_md).parent.mkdir(parents=True, exist_ok=True)
        Path(report_md).write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# World Model HEAD180: Factor-Family Normalized Probe Audit",
        "",
        "Date: 2026-05-11",
        "",
        "## Iteration Type",
        "",
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`downstream_probe_factor_family_normalized`; future factor-panel targets",
        "are frozen evaluation targets only and are not Part 1 pretraining losses.",
        "",
        "## Metric Units",
        "",
        f"`{result['target_metric_units']}`",
        "",
        "## Feature Probe Scaling",
        "",
        f"`{result['feature_metric_units']}`",
        "",
        "## Feature Surfaces",
        "",
    ]
    for feature in result["feature_surfaces"]:
        lines.append(f"- `{feature}`")
    if result["skipped_feature_surfaces"]:
        lines.append("")
        lines.append("Skipped surfaces:")
        for feature, reason in result["skipped_feature_surfaces"].items():
            lines.append(f"- `{feature}`: {reason}")
    lines.extend(
        [
            "",
            "## Family-Normalized Summary",
            "",
            "| family | target | raw-last norm MSE | raw-flat norm MSE | scale/best-raw | raw+scale/best-raw |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, target_rows in result["family_summary"].items():
        for target in FACTOR_TARGETS:
            row = target_rows[target]
            lines.append(
                "| {family} | {target} | {raw_last} | {raw_flat} | {scale} | {scale_plus} |".format(
                    family=family,
                    target=target,
                    raw_last=_fmt(row["raw_factor_last_normalized_mse"]),
                    raw_flat=_fmt(row["raw_factor_flat_normalized_mse"]),
                    scale=_fmt(row.get(f"{SCALE}_to_best_raw_ratio")),
                    scale_plus=_fmt(row.get(f"{SCALE_PLUS}_to_best_raw_ratio")),
                )
            )
    counts = result["summary_counts"]
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Target-family cells: `{counts['target_family_cells']}`.",
            f"- Scale learned standalone best-raw wins: `{counts['scale_learned_best_raw_wins']}`.",
            f"- Raw-factor-last plus scale improvements: `{counts['raw_factor_plus_scale_improvements']}`.",
            f"- Raw-factor-last plus scale best-raw wins: `{counts['raw_factor_plus_scale_best_raw_wins']}`.",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in result["decision_summary"]["caveats"]:
        lines.append(f"- `{caveat}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Promotion decision: `{result['promotion_decision']}`.",
            f"Probe status: `{result['decision_summary']['probe_status']}`.",
            f"Next step: `{result['decision_summary']['next_step']}`.",
            "",
            "This audit removes raw-unit target-scale dominance from the factor",
            "probe comparison, but remains downstream evidence only. It does not",
            "promote Part 1 or authorize Part B.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Factor-family normalized downstream probe audit"
    )
    parser.add_argument(
        "--factor_data_path", type=Path, default=Path("data/multi_factor_data.npz")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=128)
    parser.add_argument("--max_val_windows", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--scale_checkpoint", default=DEFAULT_SCALE_CHECKPOINT)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/world/factor_family_normalized_probe_head180.json"),
    )
    parser.add_argument(
        "--report_md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head180_factor_family_normalized_probe_audit.md"
        ),
    )
    return parser


def main() -> int:
    result = run_factor_family_normalized_probe_audit(_build_parser().parse_args())
    print(json.dumps(_serializable(result["summary_counts"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
