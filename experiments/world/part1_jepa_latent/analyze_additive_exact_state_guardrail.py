from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.masked_multiview_data import (  # noqa: E402
    build_masked_multiview_batch,
)
from experiments.world.part1_jepa_latent.analyze_present_state_probe import (  # noqa: E402
    _geometry_last,
    _surface_last,
    _target_groups,
)
from experiments.world.part1_jepa_latent.analyze_scale_exact_state_gap import (  # noqa: E402
    _encode_scaled_barlow,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    regression_metrics,
)


TARGETS = (
    "iv_surface",
    "vol_side_channel",
    "factor_level",
    "factor_return",
    "all_geometry",
)


def _mse(probe_metrics: dict[str, Any], feature: str, target: str) -> float:
    return float(probe_metrics[feature]["targets"][target]["mse"])


def summarize_exact_state_guardrail(
    probe_metrics: dict[str, Any],
) -> dict[str, Any]:
    raw_iv = _mse(probe_metrics, "raw_surface", "iv_surface")
    learned_iv = _mse(probe_metrics, "scale_barlow", "iv_surface")
    raw_plus_iv = _mse(probe_metrics, "raw_surface_plus_scale_barlow", "iv_surface")
    upper_iv = _mse(probe_metrics, "raw_geometry_upper", "iv_surface")
    return {
        "status": "PASS" if raw_plus_iv <= raw_iv else "FAIL",
        "raw_only_iv_mse": raw_iv,
        "learned_only_iv_mse": learned_iv,
        "raw_plus_learned_iv_mse": raw_plus_iv,
        "raw_geometry_upper_iv_mse": upper_iv,
        "learned_to_raw_ratio": learned_iv / raw_iv,
        "raw_plus_to_raw_ratio": raw_plus_iv / raw_iv,
        "raw_plus_delta_vs_raw": raw_plus_iv - raw_iv,
    }


def summarize_iv_cell_deltas(
    raw_mse: Any,
    raw_plus_mse: Any,
) -> dict[str, Any]:
    raw = np.asarray(raw_mse, dtype=np.float64)
    raw_plus = np.asarray(raw_plus_mse, dtype=np.float64)
    delta = raw_plus - raw
    return {
        "n_surface_cells": int(delta.size),
        "raw_plus_worse_cells": int(np.sum(delta > 0.0)),
        "raw_plus_better_cells": int(np.sum(delta < 0.0)),
        "mean_raw_plus_minus_raw": float(np.mean(delta)),
        "max_raw_plus_minus_raw": round(float(np.max(delta)), 12),
        "min_raw_plus_minus_raw": round(float(np.min(delta)), 12),
    }


def _per_dim_mse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_arr = np.asarray(pred, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    return np.mean((pred_arr - truth_arr) ** 2, axis=0)


def _probe_group_targets(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    out = {}
    predictions = {}
    for feature_name, train_x in train_features.items():
        rows = {}
        predictions[feature_name] = {}
        for target_name, train_y in train_targets.items():
            pred = ridge_probe_predict(
                train_x,
                train_y,
                val_features[feature_name],
                alpha=alpha,
            )
            predictions[feature_name][target_name] = pred
            rows[target_name] = regression_metrics(pred, val_targets[target_name])
        out[feature_name] = {
            "feature_shape": {
                "train": list(train_x.shape),
                "val": list(val_features[feature_name].shape),
            },
            "targets": rows,
        }
    return out, predictions


def _target_rows(probe_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        raw = _mse(probe_metrics, "raw_surface", target)
        learned = _mse(probe_metrics, "scale_barlow", target)
        raw_plus = _mse(probe_metrics, "raw_surface_plus_scale_barlow", target)
        upper = _mse(probe_metrics, "raw_geometry_upper", target)
        rows.append(
            {
                "target": target,
                "raw_only_mse": raw,
                "learned_only_mse": learned,
                "raw_plus_learned_mse": raw_plus,
                "raw_geometry_upper_mse": upper,
                "raw_plus_delta_vs_raw": raw_plus - raw,
                "raw_plus_to_raw_ratio": raw_plus / raw,
                "learned_to_raw_ratio": learned / raw,
            }
        )
    return rows


def _iv_cell_rows(
    meta: Any,
    raw_mse: np.ndarray,
    learned_mse: np.ndarray,
    raw_plus_mse: np.ndarray,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    surface_idx = np.flatnonzero(meta.geometry_id == "iv_surface")
    rows = []
    for local_idx, token_idx in enumerate(surface_idx.tolist()):
        coord = meta.geometry_coord[token_idx]
        raw = float(raw_mse[local_idx])
        learned = float(learned_mse[local_idx])
        raw_plus = float(raw_plus_mse[local_idx])
        rows.append(
            {
                "factor_id": str(meta.factor_id[token_idx]),
                "moneyness_index": int(coord[0]),
                "maturity_index": int(coord[1]),
                "raw_only_mse": raw,
                "learned_only_mse": learned,
                "raw_plus_learned_mse": raw_plus,
                "raw_plus_minus_raw_mse": raw_plus - raw,
            }
        )
    rows.sort(key=lambda row: float(row["raw_plus_minus_raw_mse"]), reverse=True)
    return rows[:limit]


def analyze_additive_exact_state_guardrail(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train = build_masked_multiview_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
        normalize=True,
    )
    val = build_masked_multiview_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
        normalize=True,
    )
    train_z, val_z = _encode_scaled_barlow(
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    train_surface = _surface_last(train)
    val_surface = _surface_last(val)
    train_features = {
        "raw_surface": train_surface,
        "scale_barlow": train_z,
        "raw_surface_plus_scale_barlow": np.concatenate(
            [train_surface, train_z], axis=1
        ),
        "raw_geometry_upper": _geometry_last(train),
    }
    val_features = {
        "raw_surface": val_surface,
        "scale_barlow": val_z,
        "raw_surface_plus_scale_barlow": np.concatenate([val_surface, val_z], axis=1),
        "raw_geometry_upper": _geometry_last(val),
    }
    train_targets = _target_groups(train)
    val_targets = _target_groups(val)
    probe_metrics, predictions = _probe_group_targets(
        train_features,
        val_features,
        train_targets,
        val_targets,
        alpha=args.ridge_alpha,
    )
    raw_iv_dim = _per_dim_mse(predictions["raw_surface"]["iv_surface"], val_targets["iv_surface"])
    learned_iv_dim = _per_dim_mse(
        predictions["scale_barlow"]["iv_surface"],
        val_targets["iv_surface"],
    )
    raw_plus_iv_dim = _per_dim_mse(
        predictions["raw_surface_plus_scale_barlow"]["iv_surface"],
        val_targets["iv_surface"],
    )
    guardrail = summarize_exact_state_guardrail(probe_metrics)
    return {
        "analysis": "world_model_additive_exact_state_guardrail",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_additive_signal_gate",
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "probe_metrics": probe_metrics,
        "target_rows": _target_rows(probe_metrics),
        "iv_cell_delta_summary": summarize_iv_cell_deltas(
            raw_iv_dim,
            raw_plus_iv_dim,
        ),
        "largest_raw_plus_degradation_cells": _iv_cell_rows(
            val.token_metadata,
            raw_iv_dim,
            learned_iv_dim,
            raw_plus_iv_dim,
            limit=args.top_cells,
        ),
        "guardrail": guardrail,
        "decision": {
            "raw_plus_exact_state_guardrail": guardrail["status"],
            "promotion_decision": "DO_NOT_PROMOTE",
            "part_b_blocked": True,
            "interpretation": (
                "Raw-plus-learned is the correct exact-state guardrail for an "
                "additive embedding. This check does not promote Part 1 by itself; "
                "it only verifies whether adding the embedding harms or helps the "
                "explicit raw-state floor."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    guardrail = result["guardrail"]
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`post_experiment_analysis`",
        "",
        "## Objective Family",
        "",
        "`downstream_probe_additive_signal_gate`; frozen current-state probe.",
        "",
        "## Hypothesis",
        "",
        "If raw exact state remains explicit, the relevant guardrail is whether",
        "`raw_surface_plus_scale_barlow` preserves or improves raw-surface",
        "current-state probes, not whether the learned embedding alone replaces",
        "raw state.",
        "",
        "## Falsifier",
        "",
        "The additive framing would be unsafe if adding the frozen embedding to",
        "raw current-state features materially worsens exact-state probes.",
        "",
        "## IV Exact-State Guardrail",
        "",
        "| feature surface | IV MSE | ratio to raw |",
        "| --- | ---: | ---: |",
        f"| raw-only | {_fmt(guardrail['raw_only_iv_mse'])} | 1.000000 |",
        f"| learned-only | {_fmt(guardrail['learned_only_iv_mse'])} | {_fmt(guardrail['learned_to_raw_ratio'])} |",
        f"| raw-plus-learned | {_fmt(guardrail['raw_plus_learned_iv_mse'])} | {_fmt(guardrail['raw_plus_to_raw_ratio'])} |",
        f"| raw-geometry upper | {_fmt(guardrail['raw_geometry_upper_iv_mse'])} | n/a |",
        "",
        "## Target Rows",
        "",
        "| target | raw-only MSE | learned-only MSE | raw+learned MSE | raw+learned delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["target_rows"]:
        lines.append(
            "| {target} | {raw} | {learned} | {raw_plus} | {delta} |".format(
                target=row["target"],
                raw=_fmt(row["raw_only_mse"]),
                learned=_fmt(row["learned_only_mse"]),
                raw_plus=_fmt(row["raw_plus_learned_mse"]),
                delta=_fmt(row["raw_plus_delta_vs_raw"]),
            )
        )
    cell_summary = result.get("iv_cell_delta_summary", {})
    lines.extend(
        [
            "",
            "## IV Cell Delta Topology",
            "",
            f"- Raw-plus worse cells: `{cell_summary.get('raw_plus_worse_cells')}/{cell_summary.get('n_surface_cells')}`.",
            f"- Raw-plus better cells: `{cell_summary.get('raw_plus_better_cells')}/{cell_summary.get('n_surface_cells')}`.",
            f"- Mean raw-plus minus raw MSE: `{_fmt(cell_summary.get('mean_raw_plus_minus_raw'))}`.",
            "",
            "| cell | moneyness | maturity | raw MSE | learned MSE | raw+learned MSE | raw+learned minus raw |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.get("largest_raw_plus_degradation_cells", []):
        lines.append(
            "| {cell} | {mon} | {mat} | {raw} | {learned} | {raw_plus} | {delta} |".format(
                cell=row["factor_id"],
                mon=row["moneyness_index"],
                mat=row["maturity_index"],
                raw=_fmt(row["raw_only_mse"]),
                learned=_fmt(row["learned_only_mse"]),
                raw_plus=_fmt(row["raw_plus_learned_mse"]),
                delta=_fmt(row["raw_plus_minus_raw_mse"]),
            )
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Raw-plus exact-state guardrail: `{decision['raw_plus_exact_state_guardrail']}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            f"- Part B blocked: `{decision['part_b_blocked']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit raw-plus-learned exact-state guardrail"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history-len", type=int, default=30)
    parser.add_argument("--future-len", type=int, default=30)
    parser.add_argument("--max-train-windows", type=int, default=1024)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--top-cells", type=int, default=10)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/additive_exact_state_guardrail_head168.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head168_additive_exact_state_guardrail.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD168: Additive Exact-State Guardrail",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_additive_exact_state_guardrail(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(
        render_markdown(result, title=args.report_title),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
