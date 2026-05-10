from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.world.evaluation.context_target_jepa_data import (  # noqa: E402
    ContextTargetJepaBatch,
    build_context_target_jepa_batch,
)
from experiments.world.part1_jepa_latent.analyze_present_state_probe import (  # noqa: E402
    _constant_baselines,
    _geometry_last,
    _probe_group_targets,
    _surface_last,
    _target_groups,
)
from experiments.world.part1_jepa_latent.context_target_jepa_smoke import (  # noqa: E402
    ContextTargetJEPAConfig,
    ContextTargetJEPAModel,
    encode_clean_context,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
)


SCALE_CHECKPOINT = Path(
    "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_scale_head127.pt"
)
CONTEXT_TARGET_CHECKPOINT = Path(
    "models/world/checkpoints/part1_jepa_latent/context_target_jepa_smoke_head140.pt"
)


def _load_context_target_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> ContextTargetJEPAModel:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = ContextTargetJEPAConfig(**checkpoint["config"])
    model = ContextTargetJEPAModel(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _encode_scaled_barlow(
    train: ContextTargetJepaBatch,
    val: ContextTargetJepaBatch,
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
    return train_z[:, -1, :], val_z[:, -1, :]


def _encode_context_target(
    train: ContextTargetJepaBatch,
    val: ContextTargetJepaBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = _load_context_target_model(CONTEXT_TARGET_CHECKPOINT, device=device)
    train_z = encode_clean_context(
        model,
        train,
        batch_size=batch_size,
        device=device,
    )
    val_z = encode_clean_context(
        model,
        val,
        batch_size=batch_size,
        device=device,
    )
    return train_z[:, -1, :], val_z[:, -1, :]


def analyze_context_target_state_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    train = build_context_target_jepa_batch(
        split="train",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    val = build_context_target_jepa_batch(
        split="val",
        history_len=args.history_len,
        future_len=args.future_len,
        max_windows=args.max_val_windows,
        seed=args.seed + 1000,
    )
    train_targets = _target_groups(train)
    val_targets = _target_groups(val)
    scale_train_z, scale_val_z = _encode_scaled_barlow(
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    context_train_z, context_val_z = _encode_context_target(
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    train_features = {
        "raw_surface_last": _surface_last(train),
        "raw_geometry_last_upper_bound": _geometry_last(train),
        "scale_barlow_last": scale_train_z,
        "context_target_last": context_train_z,
    }
    val_features = {
        "raw_surface_last": _surface_last(val),
        "raw_geometry_last_upper_bound": _geometry_last(val),
        "scale_barlow_last": scale_val_z,
        "context_target_last": context_val_z,
    }
    probes = _probe_group_targets(
        train_features,
        val_features,
        train_targets,
        val_targets,
        alpha=args.ridge_alpha,
    )
    decision = _decision(probes)
    return {
        "analysis": "world_model_context_target_state_probe",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_present_state_information",
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "scale_checkpoint": str(SCALE_CHECKPOINT),
        "context_target_checkpoint": str(CONTEXT_TARGET_CHECKPOINT),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "probe_metrics": probes,
        "constant_baseline": _constant_baselines(train_targets, val_targets),
        "decision": decision,
    }


def _mse(probes: dict[str, Any], feature: str, target: str) -> float:
    return float(probes[feature]["targets"][target]["mse"])


def _rank(probes: dict[str, Any], feature: str) -> float:
    return float(probes[feature]["health"]["effective_rank"])


def _decision(probes: dict[str, Any]) -> dict[str, Any]:
    context_iv = _mse(probes, "context_target_last", "iv_surface")
    scale_iv = _mse(probes, "scale_barlow_last", "iv_surface")
    raw_iv = _mse(probes, "raw_surface_last", "iv_surface")
    context_rank = _rank(probes, "context_target_last")
    scale_rank = _rank(probes, "scale_barlow_last")
    return {
        "context_target_improves_iv_vs_scale": context_iv < scale_iv,
        "context_target_beats_raw_surface_on_iv": context_iv < raw_iv,
        "context_target_iv_mse": context_iv,
        "scale_barlow_iv_mse": scale_iv,
        "raw_surface_iv_mse": raw_iv,
        "context_target_effective_rank": context_rank,
        "scale_barlow_effective_rank": scale_rank,
        "promotion_decision": "DO_NOT_PROMOTE",
        "interpretation": (
            "The context-to-target smoke must improve exact-state probes and retain "
            "healthy rank before it can compete with scaled Barlow."
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
        "context_target_last",
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
        "`downstream_probe_present_state_information` for frozen Part 1 candidates.",
        "",
        "## Hypothesis",
        "",
        "If the context-to-target branch is the right fix for exact-state",
        "retention, its frozen context encoder should improve current-state probes",
        "against scaled Barlow without collapsing representation health.",
        "",
        "## Falsifier",
        "",
        "The context-to-target branch is not an immediate fix if its clean context",
        "embeddings are worse than scaled Barlow on current-IV state probes or have",
        "materially weaker representation rank.",
        "",
        "## Execution",
        "",
        "Loaded the HEAD140 context-to-target smoke checkpoint and the HEAD127 scaled",
        "Barlow checkpoint, encoded clean validation windows, and fit identical frozen",
        "ridge probes for present-state geometry targets.",
        "",
        "## Present-State Probe MSE",
        "",
        "| feature | rank | IV surface | side channel | factor level | factor return | all geometry |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature in features:
        lines.append(
            "| {feature} | {rank} | {iv} | {side} | {level} | {ret} | {all_geo} |".format(
                feature=feature,
                rank=_fmt(result["probe_metrics"][feature]["health"]["effective_rank"]),
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
            "## Decision",
            "",
            f"- Context-target improves IV versus scaled Barlow: `{decision['context_target_improves_iv_vs_scale']}`.",
            f"- Context-target beats raw surface on IV: `{decision['context_target_beats_raw_surface_on_iv']}`.",
            f"- Context-target IV MSE: `{_fmt(decision['context_target_iv_mse'])}`.",
            f"- Scaled Barlow IV MSE: `{_fmt(decision['scale_barlow_iv_mse'])}`.",
            f"- Raw surface IV MSE: `{_fmt(decision['raw_surface_iv_mse'])}`.",
            f"- Context-target rank: `{_fmt(decision['context_target_effective_rank'])}`.",
            f"- Scaled Barlow rank: `{_fmt(decision['scale_barlow_effective_rank'])}`.",
            f"- Promotion decision: `{decision['promotion_decision']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare context-to-target smoke with scaled Barlow on state probes"
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
        default=Path("results/world/context_target_state_probe_head141.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head141_context_target_state_probe.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD141: Context-Target State Probe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_context_target_state_probe(args)
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
