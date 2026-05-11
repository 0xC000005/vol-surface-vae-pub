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
    FactorPanelWindowBatch,
    build_factor_panel_world_windows,
    make_factor_panel_future_targets,
)
from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    build_masked_multiview_batch,
)
from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    representation_health_metrics,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    concatenate_feature_blocks,
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
    regression_metrics,
)


RAW_FACTOR_LAST = "raw_factor_last"
RAW_FACTOR_FLAT = "raw_factor_flat"
SCALE = "scale_barlow_last"
SCALE_PLUS = "raw_factor_last_plus_scale_barlow_last"
FACTOR_TARGETS = (
    "factor_future_mean_delta",
    "factor_future_range",
    "factor_future_terminal_delta",
    "factor_future_max_abs_step",
)
DEFAULT_SCALE_CHECKPOINT = (
    "models/world/checkpoints/part1_jepa_latent/"
    "masked_multiview_barlow_scale_head127.pt"
)


def _factor_feature_sets(batch: FactorPanelWindowBatch) -> dict[str, np.ndarray]:
    return {
        RAW_FACTOR_LAST: batch.past_panel[:, -1, :],
        RAW_FACTOR_FLAT: batch.past_panel.reshape(batch.past_panel.shape[0], -1),
    }


def _probe_regression(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature_name, train_x in train_features.items():
        rows: dict[str, Any] = {}
        for target_name, train_y in train_targets.items():
            pred = ridge_probe_predict(
                train_x,
                train_y,
                val_features[feature_name],
                alpha=alpha,
            )
            rows[target_name] = regression_metrics(pred, val_targets[target_name])
        out[feature_name] = {
            "feature_shape": {
                "train": list(train_x.shape),
                "val": list(val_features[feature_name].shape),
            },
            "health": representation_health_metrics(val_features[feature_name]),
            "targets": rows,
        }
    return out


def _encode_optional_scale(
    checkpoint: str | Path,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if not str(checkpoint):
        return None, None, "scale checkpoint disabled"
    path = Path(checkpoint)
    if not path.exists():
        return None, None, f"missing scale checkpoint: {path}"
    model = load_direct_barlow_checkpoint(path, device=device)
    train_masked = build_masked_multiview_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=680,
        normalize=True,
    )
    val_masked = build_masked_multiview_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=1680,
        normalize=True,
    )
    train_z = encode_clean_masked_windows(
        model,
        train_masked,
        batch_size=args.batch_size,
        device=device,
    )[:, -1, :]
    val_z = encode_clean_masked_windows(
        model,
        val_masked,
        batch_size=args.batch_size,
        device=device,
    )[:, -1, :]
    return train_z, val_z, None


def _mse(probe: dict[str, Any], feature: str, target: str) -> float:
    return float(probe[feature]["targets"][target]["mse"])


def _target_summary(
    future_probe: dict[str, Any],
    *,
    features: list[str],
) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    for target in FACTOR_TARGETS:
        raw_last = _mse(future_probe, RAW_FACTOR_LAST, target)
        raw_flat = _mse(future_probe, RAW_FACTOR_FLAT, target)
        best_raw = min(raw_last, raw_flat)
        row: dict[str, float | str] = {
            "target": target,
            "raw_factor_last_mse": raw_last,
            "raw_factor_flat_mse": raw_flat,
            "best_raw_mse": best_raw,
        }
        for feature in features:
            mse = _mse(future_probe, feature, target)
            row[f"{feature}_mse"] = mse
            row[f"{feature}_to_raw_factor_last_ratio"] = mse / raw_last
            row[f"{feature}_to_best_raw_ratio"] = mse / best_raw
        rows[target] = row
    return rows


def _count_improvements(
    rows: dict[str, dict[str, float | str]],
    *,
    feature: str,
    baseline_key: str,
) -> int:
    return int(
        sum(
            float(row[f"{feature}_mse"]) < float(row[baseline_key])
            for row in rows.values()
        )
    )


def _serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_factor_panel_probe_bakeoff(args: argparse.Namespace) -> dict[str, Any]:
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
        train_features[SCALE_PLUS] = concatenate_feature_blocks(
            train_features[RAW_FACTOR_LAST],
            scale_train,
        )
        val_features[SCALE] = scale_val
        val_features[SCALE_PLUS] = concatenate_feature_blocks(
            val_features[RAW_FACTOR_LAST],
            scale_val,
        )
    else:
        skipped_feature_surfaces[SCALE] = str(scale_skip)
        skipped_feature_surfaces[SCALE_PLUS] = str(scale_skip)

    train_targets = make_factor_panel_future_targets(
        train_factor.past_panel,
        train_factor.future_panel,
        columns=train_factor.columns,
    )
    val_targets = make_factor_panel_future_targets(
        val_factor.past_panel,
        val_factor.future_panel,
        columns=val_factor.columns,
    )
    future_probe = _probe_regression(
        train_features,
        val_features,
        train_targets["regression"],
        val_targets["regression"],
        alpha=args.ridge_alpha,
    )
    comparison_features = [
        feature for feature in (SCALE, SCALE_PLUS) if feature in train_features
    ]
    future_summary = _target_summary(future_probe, features=comparison_features)
    summary_counts: dict[str, int | None | str] = {
        "factor_targets": len(FACTOR_TARGETS),
        "scale_learned_best_raw_wins": (
            _count_improvements(
                future_summary,
                feature=SCALE,
                baseline_key="best_raw_mse",
            )
            if SCALE in train_features
            else None
        ),
        "raw_factor_plus_scale_improvements": (
            _count_improvements(
                future_summary,
                feature=SCALE_PLUS,
                baseline_key="raw_factor_last_mse",
            )
            if SCALE_PLUS in train_features
            else None
        ),
        "raw_factor_plus_scale_best_raw_wins": (
            _count_improvements(
                future_summary,
                feature=SCALE_PLUS,
                baseline_key="best_raw_mse",
            )
            if SCALE_PLUS in train_features
            else None
        ),
    }
    decision_summary = {
        "probe_status": "factor_panel_signal_present_smoke_only",
        "scale_standalone_factor_wins": (
            f"{summary_counts['scale_learned_best_raw_wins']}/{len(FACTOR_TARGETS)}"
            if summary_counts["scale_learned_best_raw_wins"] is not None
            else "n/a"
        ),
        "raw_plus_scale_improves_raw_last": (
            f"{summary_counts['raw_factor_plus_scale_improvements']}/{len(FACTOR_TARGETS)}"
            if summary_counts["raw_factor_plus_scale_improvements"] is not None
            else "n/a"
        ),
        "raw_plus_scale_beats_best_raw": (
            f"{summary_counts['raw_factor_plus_scale_best_raw_wins']}/{len(FACTOR_TARGETS)}"
            if summary_counts["raw_factor_plus_scale_best_raw_wins"] is not None
            else "n/a"
        ),
        "caveats": [
            "smoke_scale_128_train_64_val",
            "raw_unit_mse_not_column_standardized",
            "target_family_breakdown_missing",
        ],
        "next_step": "factor_family_normalized_probe_audit_before_promotion",
    }

    result: dict[str, Any] = {
        "analysis": "world_model_factor_panel_probe_bakeoff",
        "date": "2026-05-11",
        "iteration": 179,
        "iteration_type": "experiment",
        "objective_family": "downstream_probe_factor_panel_bakeoff",
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
        "feature_surfaces": list(train_features.keys()),
        "skipped_feature_surfaces": skipped_feature_surfaces,
        "future_probe_rows": future_probe,
        "future_summary": future_summary,
        "summary_counts": summary_counts,
        "decision_summary": decision_summary,
        "target_scope": "factor_panel_future_downstream_probe_only",
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


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# World Model HEAD179: Factor-Panel Probe Bakeoff",
        "",
        "Date: 2026-05-11",
        "",
        "## Iteration Type",
        "",
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`downstream_probe_factor_panel_bakeoff`; future factor-panel targets",
        "are frozen evaluation targets only and are not Part 1 pretraining losses.",
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
            "## Future Factor-Panel Summary",
            "",
            "| target | raw-last MSE | raw-flat MSE | scale/best-raw | raw+scale/raw-last |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in FACTOR_TARGETS:
        row = result["future_summary"][target]
        lines.append(
            "| {target} | {raw_last} | {raw_flat} | {scale} | {scale_plus} |".format(
                target=target,
                raw_last=_fmt(row["raw_factor_last_mse"]),
                raw_flat=_fmt(row["raw_factor_flat_mse"]),
                scale=_fmt(row.get(f"{SCALE}_to_best_raw_ratio")),
                scale_plus=_fmt(row.get(f"{SCALE_PLUS}_to_raw_factor_last_ratio")),
            )
        )
    counts = result["summary_counts"]
    lines.extend(
        [
            "",
            "## Counts",
            "",
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
            "This is downstream coverage evidence only. It does not promote Part 1",
            "or authorize Part B.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Factor-panel downstream probe bakeoff"
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
        default=Path("results/world/factor_panel_probe_bakeoff_head179.json"),
    )
    parser.add_argument(
        "--report_md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head179_factor_panel_probe_bakeoff.md"
        ),
    )
    return parser


def main() -> int:
    result = run_factor_panel_probe_bakeoff(_build_parser().parse_args())
    print(json.dumps(_serializable(result["summary_counts"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
