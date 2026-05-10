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
    MaskedMultiviewBatch,
    build_masked_multiview_batch,
)
from experiments.world.evaluation.part1_metrics import (  # noqa: E402
    representation_health_metrics,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
    regression_metrics,
)


CHECKPOINTS = {
    "head070_default": Path(
        "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt"
    ),
    "head123_hard": Path(
        "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_hardmask_head123.pt"
    ),
}


def _target_groups(batch: MaskedMultiviewBatch) -> dict[str, np.ndarray]:
    last = np.asarray(batch.clean_values[:, -1, :], dtype=np.float32)
    meta = batch.token_metadata
    groups = {
        "iv_surface": meta.geometry_id == "iv_surface",
        "vol_side_channel": meta.geometry_id == "vol_side_channel",
        "factor_level": meta.geometry_id == "factor_level",
        "factor_return": meta.geometry_id == "factor_return",
        "all_geometry": np.ones(meta.n_tokens, dtype=bool),
    }
    return {name: last[:, mask].astype(np.float32) for name, mask in groups.items()}


def _surface_last(batch: MaskedMultiviewBatch) -> np.ndarray:
    meta = batch.token_metadata
    return batch.clean_values[:, -1, meta.geometry_id == "iv_surface"].astype(
        np.float32
    )


def _geometry_last(batch: MaskedMultiviewBatch) -> np.ndarray:
    return batch.clean_values[:, -1, :].astype(np.float32)


def _constant_baselines(
    train_targets: dict[str, np.ndarray],
    val_targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    out = {}
    for name, train_y in train_targets.items():
        pred = np.repeat(
            train_y.mean(axis=0, keepdims=True), val_targets[name].shape[0], axis=0
        )
        out[name] = regression_metrics(pred, val_targets[name])
    return out


def _probe_group_targets(
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


def _encode_checkpoint(
    checkpoint: Path,
    train: MaskedMultiviewBatch,
    val: MaskedMultiviewBatch,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_direct_barlow_checkpoint(checkpoint, device=device)
    train_encoded = encode_clean_masked_windows(
        model,
        train,
        batch_size=batch_size,
        device=device,
    )
    val_encoded = encode_clean_masked_windows(
        model,
        val,
        batch_size=batch_size,
        device=device,
    )
    return train_encoded[:, -1, :], val_encoded[:, -1, :]


def analyze_present_state_probe(args: argparse.Namespace) -> dict[str, Any]:
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
        train_z, val_z = _encode_checkpoint(
            checkpoint,
            train,
            val,
            batch_size=args.batch_size,
            device=device,
        )
        train_features[f"{name}_barlow_last"] = train_z
        val_features[f"{name}_barlow_last"] = val_z

    return {
        "analysis": "world_model_part1_present_state_probe",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_present_state_information",
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "target_groups": {
            name: list(value.shape) for name, value in val_targets.items()
        },
        "feature_shapes": {
            name: list(value.shape) for name, value in val_features.items()
        },
        "probe_metrics": _probe_group_targets(
            train_features,
            val_features,
            train_targets,
            val_targets,
            alpha=args.ridge_alpha,
        ),
        "constant_baseline": _constant_baselines(train_targets, val_targets),
        "decision": {
            "purpose": "Check whether frozen Part 1 embeddings retain current market-state geometry before tuning masks or architecture.",
            "promotion_signal": False,
        },
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
        "head070_default_barlow_last",
        "head123_hard_barlow_last",
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
        "`post_experiment_analysis`",
        "",
        "## Objective Family",
        "",
        "`downstream_probe_present_state_information` for frozen Part 1 embeddings.",
        "",
        "## Hypothesis",
        "",
        "A useful market-state representation should retain enough present-state",
        "geometry that linear probes can recover IV-surface, side-channel, and",
        "factor-panel values better than a constant baseline and, for non-surface",
        "targets, better than raw IV-surface-only features.",
        "",
        "## Falsifier",
        "",
        "If Barlow embeddings cannot recover factor-panel or side-channel state,",
        "then Part 1 is not yet a certified joint market-state representation,",
        "even if same-state retrieval is non-collapsed.",
        "",
        "## Present-State Probe MSE",
        "",
        "| feature | target | MSE | R2 | effective rank |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
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
    lines.extend(
        [
            "",
            "## Constant Baseline",
            "",
            "| target | MSE | R2 |",
            "| --- | ---: | ---: |",
        ]
    )
    for target in targets:
        metrics = result["constant_baseline"][target]
        lines.append(f"| {target} | {_fmt(metrics['mse'])} | {_fmt(metrics['r2'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The default Barlow embedding is not empty and is not ignoring the",
            "factor panel entirely. It is much better than raw IV-surface-only",
            "features for `factor_return`, and it is better than raw surface-only",
            "features and the constant baseline on `factor_level` and",
            "`vol_side_channel` MSE.",
            "",
            "The failure is more specific: it compresses away too much exact",
            "present-state geometry. It is worse than raw IV-surface-only features",
            "on the IV surface itself, weak on factor levels and side channels, and",
            "far from the raw full-geometry upper bound. The hard-mask checkpoint",
            "does not fix this; it generally lowers rank and worsens present-state",
            "probe quality except for a small side-channel MSE improvement.",
            "",
            "This helps explain why simple market-state baselines win persistence-like",
            "future probes. Those baselines carry exact current IV levels, while the",
            "Barlow embedding is optimized for masked-view invariance and compresses",
            "state details that those probes reward.",
            "",
            "## Decision",
            "",
            "This is a frozen present-state information audit, not a pretraining",
            "objective. Use it to decide whether the representation is actually",
            "encoding the joint market state before changing mask strength or",
            "architecture.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen present-state probe for Part 1 Barlow embeddings"
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
        default=Path("results/world/present_state_probe_head124.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head124_present_state_probe.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD124: Present-State Probe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_present_state_probe(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(
        render_markdown(result, title=args.report_title), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
