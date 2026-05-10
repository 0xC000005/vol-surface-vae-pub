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
from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    representation_health_metrics,
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
from experiments.world.part1_jepa_latent.masked_multiview_grouped_geometry_barlow_smoke import (  # noqa: E402
    encode_clean_grouped_geometry_windows,
    load_grouped_geometry_checkpoint,
)


CHECKPOINTS = {
    "head070_flat": Path(
        "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt"
    ),
    "head126_grouped": Path(
        "models/world/checkpoints/part1_jepa_latent/masked_multiview_grouped_geometry_barlow_head126.pt"
    ),
}
RESULTS = {
    "head070_flat": Path("results/world/masked_multiview_barlow_head070.json"),
    "head126_grouped": Path(
        "results/world/masked_multiview_grouped_geometry_barlow_head126.json"
    ),
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


def _encode_grouped(
    checkpoint: Path,
    train,
    val,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_grouped_geometry_checkpoint(checkpoint, device=device)
    train_z = encode_clean_grouped_geometry_windows(
        model,
        train,
        batch_size=batch_size,
        device=device,
    )
    val_z = encode_clean_grouped_geometry_windows(
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
        "top1": float(retrieval["top1"]),
        "top10": float(retrieval["top10"]),
        "mrr": float(retrieval["mrr"]),
        "median_rank": float(retrieval["median_rank"]),
        "raw_top10": float(raw["top10"]),
        "effective_rank": float(health["effective_rank"]),
        "variance_min": float(health["variance_min"]),
        "offdiag_abs_mean": float(health["offdiag_abs_mean"]),
    }


def analyze_grouped_geometry_state_probe(args: argparse.Namespace) -> dict[str, Any]:
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
    flat_train, flat_val = _encode_direct(
        CHECKPOINTS["head070_flat"],
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    grouped_train, grouped_val = _encode_grouped(
        CHECKPOINTS["head126_grouped"],
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    train_features["head070_flat_barlow_last"] = flat_train
    val_features["head070_flat_barlow_last"] = flat_val
    train_features["head126_grouped_barlow_last"] = grouped_train
    val_features["head126_grouped_barlow_last"] = grouped_val
    probes = _probe_group_targets(
        train_features,
        val_features,
        train_targets,
        val_targets,
        alpha=args.ridge_alpha,
    )
    representation = {
        "head070_flat": _representation_summary(RESULTS["head070_flat"]),
        "head126_grouped": _representation_summary(RESULTS["head126_grouped"]),
    }
    return {
        "analysis": "world_model_part1_grouped_geometry_state_probe",
        "date": "2026-05-10",
        "objective_family": "masked_multiview_invariance_architecture_diagnostic",
        "literature_status": "supported_adjacent_direct_barlow_with_grouped_geometry_encoder",
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
    flat = "head070_flat_barlow_last"
    grouped = "head126_grouped_barlow_last"
    raw_surface = "raw_surface_last"
    grouped_improves_factor_level = _mse(probes, grouped, "factor_level") < _mse(
        probes, flat, "factor_level"
    )
    grouped_improves_iv = _mse(probes, grouped, "iv_surface") < _mse(
        probes, flat, "iv_surface"
    )
    grouped_beats_raw_iv = _mse(probes, grouped, "iv_surface") < _mse(
        probes, raw_surface, "iv_surface"
    )
    grouped_retrieval_not_worse = (
        representation["head126_grouped"]["top10"]
        >= representation["head070_flat"]["top10"]
    )
    grouped_rank_not_worse = (
        representation["head126_grouped"]["effective_rank"]
        >= representation["head070_flat"]["effective_rank"]
    )
    return {
        "grouped_improves_factor_level_mse": grouped_improves_factor_level,
        "grouped_improves_iv_surface_mse": grouped_improves_iv,
        "grouped_beats_raw_surface_on_iv": grouped_beats_raw_iv,
        "grouped_retrieval_not_worse": grouped_retrieval_not_worse,
        "grouped_rank_not_worse": grouped_rank_not_worse,
        "promote_grouped_geometry": (
            grouped_improves_factor_level
            and grouped_improves_iv
            and grouped_retrieval_not_worse
            and grouped_rank_not_worse
        ),
        "interpretation": (
            "Grouped geometry is only a promotion candidate if it improves "
            "state content without losing representation rank."
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
        "head070_flat_barlow_last",
        "head126_grouped_barlow_last",
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
        "`masked_multiview_invariance_architecture_diagnostic`.",
        "",
        "## Literature Status",
        "",
        "`supported_adjacent_direct_barlow_with_grouped_geometry_encoder`.",
        "",
        "## Hypothesis",
        "",
        "A grouped geometry encoder should improve present-state content by keeping",
        "IV surface, side channels, factor levels, and factor returns separable",
        "before temporal fusion, while retaining the same masked-multiview Barlow",
        "objective.",
        "",
        "## Falsifier",
        "",
        "The architecture is not justified if it improves retrieval by becoming",
        "lower-rank, fails to improve current-state probes, or still loses exact",
        "IV-state retention to raw surface features.",
        "",
        "## Representation Metrics",
        "",
        "| run | top1 | top10 | mrr | median rank | raw top10 | effective rank | variance min | offdiag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["representation"].items():
        lines.append(
            "| {name} | {top1} | {top10} | {mrr} | {median} | {raw_top10} | {rank} | {var_min} | {offdiag} |".format(
                name=name,
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
            f"- Grouped improves factor-level MSE: `{decision['grouped_improves_factor_level_mse']}`.",
            f"- Grouped improves IV-surface MSE: `{decision['grouped_improves_iv_surface_mse']}`.",
            f"- Grouped beats raw surface on IV state: `{decision['grouped_beats_raw_surface_on_iv']}`.",
            f"- Grouped retrieval not worse: `{decision['grouped_retrieval_not_worse']}`.",
            f"- Grouped rank not worse: `{decision['grouped_rank_not_worse']}`.",
            f"- Promote grouped geometry: `{decision['promote_grouped_geometry']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare grouped-geometry Barlow state-content probes"
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
        default=Path("results/world/grouped_geometry_state_probe_head126.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head126_grouped_geometry_state_probe.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD126: Grouped Geometry State Probe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_grouped_geometry_state_probe(args)
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
