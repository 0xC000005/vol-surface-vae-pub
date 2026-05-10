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
from experiments.world.part1_jepa_latent.analyze_additive_exact_state_guardrail import (  # noqa: E402
    _target_rows,
    summarize_exact_state_guardrail,
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


DEFAULT_PLAIN_GUARDRAIL = Path("results/world/additive_exact_state_topology_head169.json")


def standardize_train_val(
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_features, dtype=np.float64)
    val = np.asarray(val_features, dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (train - mean) / std, (val - mean) / std


def _standardized_probe_group_targets(
    train_features: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, Any]:
    out = {}
    for feature_name, train_x in train_features.items():
        rows = {}
        train_std, val_std = standardize_train_val(train_x, val_features[feature_name])
        for target_name, train_y in train_targets.items():
            pred = ridge_probe_predict(train_std, train_y, val_std, alpha=alpha)
            rows[target_name] = regression_metrics(pred, val_targets[target_name])
        out[feature_name] = {
            "feature_shape": {
                "train": list(train_x.shape),
                "val": list(val_features[feature_name].shape),
            },
            "targets": rows,
        }
    return out


def analyze_additive_probe_standardization(args: argparse.Namespace) -> dict[str, Any]:
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
    standardized_probe_metrics = _standardized_probe_group_targets(
        train_features,
        val_features,
        train_targets,
        val_targets,
        alpha=args.ridge_alpha,
    )
    standardized_guardrail = summarize_exact_state_guardrail(
        standardized_probe_metrics
    )
    plain = json.loads(args.plain_guardrail_json.read_text(encoding="utf-8"))
    plain_guardrail = plain["guardrail"]
    fixes_guardrail = standardized_guardrail["status"] == "PASS"
    return {
        "analysis": "world_model_additive_probe_standardization",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_additive_signal_gate",
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "plain_guardrail_source": str(args.plain_guardrail_json),
        "plain_guardrail": plain_guardrail,
        "standardized_guardrail": standardized_guardrail,
        "standardized_target_rows": _target_rows(standardized_probe_metrics),
        "decision": {
            "standardization_fixes_iv_guardrail": fixes_guardrail,
            "promotion_decision": "DO_NOT_PROMOTE",
            "part_b_blocked": True,
            "interpretation": (
                "Feature standardization checks whether the raw-plus IV miss is "
                "mainly a ridge-probe scale artifact. This is evaluation hygiene, "
                "not a Part 1 objective change."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    plain = result["plain_guardrail"]
    standard = result["standardized_guardrail"]
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
        "`downstream_probe_additive_signal_gate`; probe-hygiene diagnostic.",
        "",
        "## Hypothesis",
        "",
        "If the raw-plus-learned IV guardrail failure is mostly caused by feature",
        "scale differences under one ridge penalty, then standardizing each",
        "feature surface with train-split statistics should remove or materially",
        "reduce the failure.",
        "",
        "## Falsifier",
        "",
        "The probe-scale artifact explanation is weak if standardization does not",
        "fix the IV guardrail.",
        "",
        "## Guardrail Comparison",
        "",
        "| probe surface | raw-only IV MSE | raw+learned IV MSE | raw+learned/raw | status |",
        "| --- | ---: | ---: | ---: | --- |",
        "| unstandardized | {raw} | {raw_plus} | {ratio} | {status} |".format(
            raw=_fmt(plain["raw_only_iv_mse"]),
            raw_plus=_fmt(plain["raw_plus_learned_iv_mse"]),
            ratio=_fmt(plain["raw_plus_to_raw_ratio"]),
            status=plain["status"],
        ),
        "| standardized | {raw} | {raw_plus} | {ratio} | {status} |".format(
            raw=_fmt(standard["raw_only_iv_mse"]),
            raw_plus=_fmt(standard["raw_plus_learned_iv_mse"]),
            ratio=_fmt(standard["raw_plus_to_raw_ratio"]),
            status=standard["status"],
        ),
        "",
        "## Standardized Target Rows",
        "",
        "| target | raw-only MSE | learned-only MSE | raw+learned MSE | raw+learned delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["standardized_target_rows"]:
        lines.append(
            "| {target} | {raw} | {learned} | {raw_plus} | {delta} |".format(
                target=row["target"],
                raw=_fmt(row["raw_only_mse"]),
                learned=_fmt(row["learned_only_mse"]),
                raw_plus=_fmt(row["raw_plus_learned_mse"]),
                delta=_fmt(row["raw_plus_delta_vs_raw"]),
            )
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Standardization fixes IV guardrail: `{decision['standardization_fixes_iv_guardrail']}`.",
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
        description="Test standardized ridge probes for additive exact-state guardrail"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--history-len", type=int, default=30)
    parser.add_argument("--future-len", type=int, default=30)
    parser.add_argument("--max-train-windows", type=int, default=1024)
    parser.add_argument("--max-val-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--plain-guardrail-json", type=Path, default=DEFAULT_PLAIN_GUARDRAIL)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/additive_probe_standardization_head170.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head170_additive_probe_standardization.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD170: Additive Probe Standardization",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_additive_probe_standardization(args)
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
