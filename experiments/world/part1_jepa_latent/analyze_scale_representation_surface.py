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
    _constant_baselines,
    _geometry_last,
    _probe_group_targets,
    _surface_last,
    _target_groups,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
)


SCALE_CHECKPOINT = Path(
    "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_scale_head127.pt"
)


def _scaled_sequence_embeddings(
    train,
    val,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_direct_barlow_checkpoint(SCALE_CHECKPOINT, device=device)
    train_z = encode_clean_masked_windows(
        model,
        train,
        batch_size=batch_size,
        device=device,
    )
    val_z = encode_clean_masked_windows(
        model,
        val,
        batch_size=batch_size,
        device=device,
    )
    return train_z, val_z


def _feature_surfaces(encoded: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(encoded, dtype=np.float32)
    last = arr[:, -1, :]
    mean = arr.mean(axis=1)
    return {
        "scale_barlow_last": last,
        "scale_barlow_mean": mean,
        "scale_barlow_last_plus_mean": np.concatenate([last, mean], axis=1),
        "scale_barlow_flat_time": arr.reshape(arr.shape[0], -1),
    }


def analyze_representation_surface(args: argparse.Namespace) -> dict[str, Any]:
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
    train_targets = _target_groups(train)
    val_targets = _target_groups(val)
    train_z, val_z = _scaled_sequence_embeddings(
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    train_features = {
        "raw_surface_last": _surface_last(train),
        "raw_geometry_last_upper_bound": _geometry_last(train),
        **_feature_surfaces(train_z),
    }
    val_features = {
        "raw_surface_last": _surface_last(val),
        "raw_geometry_last_upper_bound": _geometry_last(val),
        **_feature_surfaces(val_z),
    }
    probes = _probe_group_targets(
        train_features,
        val_features,
        train_targets,
        val_targets,
        alpha=args.ridge_alpha,
    )
    return {
        "analysis": "world_model_scaled_representation_surface_audit",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_present_state_information",
        "checkpoint": str(SCALE_CHECKPOINT),
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "feature_shapes": {
            name: {
                "train": list(value.shape),
                "val": list(val_features[name].shape),
            }
            for name, value in train_features.items()
        },
        "probe_metrics": probes,
        "constant_baseline": _constant_baselines(train_targets, val_targets),
        "decision": _decision(probes),
    }


def _mse(probes: dict[str, Any], feature: str, target: str) -> float:
    return float(probes[feature]["targets"][target]["mse"])


def _decision(probes: dict[str, Any]) -> dict[str, Any]:
    surfaces = (
        "scale_barlow_last",
        "scale_barlow_mean",
        "scale_barlow_last_plus_mean",
        "scale_barlow_flat_time",
    )
    best_surface_by_target = {}
    for target in (
        "iv_surface",
        "vol_side_channel",
        "factor_level",
        "factor_return",
        "all_geometry",
    ):
        best = min(surfaces, key=lambda feature: _mse(probes, feature, target))
        best_surface_by_target[target] = {
            "feature": best,
            "mse": _mse(probes, best, target),
        }
    raw_iv = _mse(probes, "raw_surface_last", "iv_surface")
    best_iv = best_surface_by_target["iv_surface"]["mse"]
    return {
        "best_surface_by_target": best_surface_by_target,
        "best_surface_beats_raw_surface_on_iv": best_iv < raw_iv,
        "raw_surface_iv_mse": raw_iv,
        "best_surface_iv_mse": best_iv,
        "promotion_decision": "DO_NOT_PROMOTE",
        "interpretation": (
            "Changing the frozen representation readout surface does not fix the "
            "exact-state gap. Last-state embeddings remain the best scaled surface "
            "for IV, side-channel, factor-level, and all-geometry probes, while "
            "last+mean only improves factor returns. The blocker is therefore more "
            "likely in the learned representation/objective than in the downstream "
            "pooling choice."
        ),
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _target_metric(
    result: dict[str, Any],
    feature: str,
    target: str,
    metric: str,
) -> float:
    return float(result["probe_metrics"][feature]["targets"][target][metric])


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    features = (
        "raw_surface_last",
        "raw_geometry_last_upper_bound",
        "scale_barlow_last",
        "scale_barlow_mean",
        "scale_barlow_last_plus_mean",
        "scale_barlow_flat_time",
    )
    targets = (
        "iv_surface",
        "vol_side_channel",
        "factor_level",
        "factor_return",
        "all_geometry",
    )
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
        "`downstream_probe_present_state_information` for frozen scaled Part 1 surfaces.",
        "",
        "## Hypothesis",
        "",
        "If the exact-state gap is mainly a readout-surface problem, then mean,",
        "last+mean, or flattened per-time scaled embeddings should recover current",
        "state better than the current last-state readout and possibly approach raw",
        "last-surface baselines.",
        "",
        "## Falsifier",
        "",
        "If no frozen scaled representation surface beats the current last-state",
        "readout or raw exact-state baselines, then the blocker is likely in the",
        "learned representation/objective, not merely the probe surface.",
        "",
        "## Present-State Probe MSE",
        "",
        "| feature | dims | IV surface | side channel | factor level | factor return | all geometry |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature in features:
        dims = result["feature_shapes"][feature]["val"][1]
        lines.append(
            "| {feature} | {dims} | {iv} | {side} | {level} | {ret} | {all_geo} |".format(
                feature=feature,
                dims=dims,
                iv=_fmt(_target_metric(result, feature, "iv_surface", "mse")),
                side=_fmt(_target_metric(result, feature, "vol_side_channel", "mse")),
                level=_fmt(_target_metric(result, feature, "factor_level", "mse")),
                ret=_fmt(_target_metric(result, feature, "factor_return", "mse")),
                all_geo=_fmt(_target_metric(result, feature, "all_geometry", "mse")),
            )
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Best Scaled Surface By Target",
            "",
            "| target | feature | MSE |",
            "| --- | --- | ---: |",
        ]
    )
    for target, row in decision["best_surface_by_target"].items():
        lines.append(f"| {target} | {row['feature']} | {_fmt(row['mse'])} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best scaled surface beats raw surface on IV state: `{decision['best_surface_beats_raw_surface_on_iv']}`.",
            f"- Raw surface IV MSE: `{_fmt(decision['raw_surface_iv_mse'])}`.",
            f"- Best scaled surface IV MSE: `{_fmt(decision['best_surface_iv_mse'])}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen scaled Part 1 representation readout surfaces"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_representation_surface_head136.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head136_scale_representation_surface.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD136: Scale Representation Surface",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_representation_surface(args)
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
