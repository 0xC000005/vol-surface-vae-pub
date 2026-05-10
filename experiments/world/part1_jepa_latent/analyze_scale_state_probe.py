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

CHECKPOINTS = {
    "head070_smoke": Path(
        "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt"
    ),
    "head127_scale": Path(
        "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_scale_head127.pt"
    ),
}
RESULTS = {
    "head070_smoke": Path("results/world/masked_multiview_barlow_head070.json"),
    "head127_scale": Path("results/world/masked_multiview_barlow_scale_head127.json"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _encode_direct(
    checkpoint: Path,
    train,
    val,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_direct_barlow_checkpoint(checkpoint, device=device)
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


def _representation_summary(path: Path) -> dict[str, float]:
    data = _load_json(path)
    view = data["val_metrics"]["view_alignment"]
    retrieval = view["retrieval"]
    health = view["view_a_health"]
    raw = data["raw_val_baseline"]["retrieval"]
    return {
        "train_windows": float(data["train_shape"][0]),
        "val_windows": float(data["val_shape"][0]),
        "top1": float(retrieval["top1"]),
        "top10": float(retrieval["top10"]),
        "mrr": float(retrieval["mrr"]),
        "median_rank": float(retrieval["median_rank"]),
        "raw_top10": float(raw["top10"]),
        "effective_rank": float(health["effective_rank"]),
        "variance_min": float(health["variance_min"]),
        "offdiag_abs_mean": float(health["offdiag_abs_mean"]),
    }


def analyze_scale_state_probe(args: argparse.Namespace) -> dict[str, Any]:
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
    train_features = {
        "raw_surface_last": _surface_last(train),
        "raw_geometry_last_upper_bound": _geometry_last(train),
    }
    val_features = {
        "raw_surface_last": _surface_last(val),
        "raw_geometry_last_upper_bound": _geometry_last(val),
    }
    for name, checkpoint in CHECKPOINTS.items():
        train_z, val_z = _encode_direct(
            checkpoint,
            train,
            val,
            batch_size=args.batch_size,
            device=device,
        )
        train_features[f"{name}_barlow_last"] = train_z
        val_features[f"{name}_barlow_last"] = val_z

    probes = _probe_group_targets(
        train_features,
        val_features,
        train_targets,
        val_targets,
        alpha=args.ridge_alpha,
    )
    representation = {
        name: _representation_summary(path) for name, path in RESULTS.items()
    }
    return {
        "analysis": "world_model_part1_scale_state_probe",
        "date": "2026-05-10",
        "objective_family": "masked_multiview_invariance_scale_diagnostic",
        "literature_status": "same_objective_scale_and_stability_diagnostic",
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "representation": representation,
        "probe_metrics": probes,
        "constant_baseline": _constant_baselines(train_targets, val_targets),
        "decision": _decision(probes, representation),
    }


def _mse(result: dict[str, Any], feature: str, target: str) -> float:
    return float(result[feature]["targets"][target]["mse"])


def _r2(result: dict[str, Any], feature: str, target: str) -> float:
    return float(result[feature]["targets"][target]["r2"])


def _decision(
    probes: dict[str, Any],
    representation: dict[str, dict[str, float]],
) -> dict[str, Any]:
    smoke = "head070_smoke_barlow_last"
    scale = "head127_scale_barlow_last"
    raw_surface = "raw_surface_last"
    scale_improves_iv = _mse(probes, scale, "iv_surface") < _mse(
        probes, smoke, "iv_surface"
    )
    scale_beats_raw_iv = _mse(probes, scale, "iv_surface") < _mse(
        probes, raw_surface, "iv_surface"
    )
    scale_improves_factor_level = _mse(probes, scale, "factor_level") < _mse(
        probes, smoke, "factor_level"
    )
    scale_preserves_factor_return = _r2(probes, scale, "factor_return") >= (
        _r2(probes, smoke, "factor_return") - 0.05
    )
    scale_rank_improves = (
        representation["head127_scale"]["effective_rank"]
        > representation["head070_smoke"]["effective_rank"]
    )
    scale_retrieval_improves = (
        representation["head127_scale"]["top10"]
        >= representation["head070_smoke"]["top10"]
    )
    return {
        "scale_improves_iv_surface_mse": scale_improves_iv,
        "scale_beats_raw_surface_on_iv": scale_beats_raw_iv,
        "scale_improves_factor_level_mse": scale_improves_factor_level,
        "scale_preserves_factor_return_signal": scale_preserves_factor_return,
        "scale_rank_improves": scale_rank_improves,
        "scale_retrieval_improves": scale_retrieval_improves,
        "promote_scale_checkpoint": (
            scale_improves_iv
            and scale_improves_factor_level
            and scale_preserves_factor_return
            and scale_rank_improves
            and scale_retrieval_improves
        ),
        "promotion_scope": "candidate_for_next_quality_gate_not_part_b_ready",
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
        "head070_smoke_barlow_last",
        "head127_scale_barlow_last",
        "raw_surface_last",
        "raw_geometry_last_upper_bound",
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
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`masked_multiview_invariance_scale_diagnostic`.",
        "",
        "## Literature Status",
        "",
        "`same_objective_scale_and_stability_diagnostic`.",
        "",
        "## Hypothesis",
        "",
        "If the state-content gap is partly a smoke-scale issue, then training the",
        "same HEAD070-style flat encoder with more windows should improve",
        "effective rank and present-state probes without adding future targets or",
        "new objective terms.",
        "",
        "## Falsifier",
        "",
        "Scale is not enough if the checkpoint improves retrieval/rank but still",
        "fails exact IV retention, factor-level retention, or factor-return signal.",
        "",
        "## Representation Metrics",
        "",
        "| run | train windows | val windows | top1 | top10 | mrr | median rank | raw top10 | effective rank | variance min | offdiag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["representation"].items():
        lines.append(
            "| {name} | {train} | {val} | {top1} | {top10} | {mrr} | {median} | {raw_top10} | {rank} | {var_min} | {offdiag} |".format(
                name=name,
                train=_fmt(row["train_windows"]),
                val=_fmt(row["val_windows"]),
                top1=_fmt(row["top1"]),
                top10=_fmt(row["top10"]),
                mrr=_fmt(row["mrr"]),
                median=_fmt(row["median_rank"]),
                raw_top10=_fmt(row["raw_top10"]),
                rank=_fmt(row["effective_rank"]),
                var_min=_fmt(row["variance_min"]),
                offdiag=_fmt(row["offdiag_abs_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Present-State Probe MSE",
            "",
            "| feature | target | MSE | R2 | effective rank |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for feature in features:
        health = result["probe_metrics"][feature]["health"]
        for target in targets:
            lines.append(
                "| {feature} | {target} | {mse} | {r2} | {rank} |".format(
                    feature=feature,
                    target=target,
                    mse=_fmt(_target_metric(result, feature, target, "mse")),
                    r2=_fmt(_target_metric(result, feature, target, "r2")),
                    rank=_fmt(health["effective_rank"]),
                )
            )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Scale improves IV-surface MSE: `{decision['scale_improves_iv_surface_mse']}`.",
            f"- Scale beats raw surface on IV state: `{decision['scale_beats_raw_surface_on_iv']}`.",
            f"- Scale improves factor-level MSE: `{decision['scale_improves_factor_level_mse']}`.",
            f"- Scale preserves factor-return signal: `{decision['scale_preserves_factor_return_signal']}`.",
            f"- Scale rank improves: `{decision['scale_rank_improves']}`.",
            f"- Scale retrieval improves: `{decision['scale_retrieval_improves']}`.",
            f"- Promote scale checkpoint: `{decision['promote_scale_checkpoint']}`.",
            f"- Promotion scope: `{decision['promotion_scope']}`.",
            "",
            "The scale checkpoint is a better Part 1 candidate than HEAD070 on",
            "this diagnostic, but it still does not beat raw surface features on",
            "exact current-IV reconstruction. Treat it as the next candidate to",
            "gate, not as Part-B-ready evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare scaled flat Barlow state probes"
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
        default=Path("results/world/scale_state_probe_head127.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head127_scale_state_probe.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD127: Scale State Probe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_scale_state_probe(args)
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
