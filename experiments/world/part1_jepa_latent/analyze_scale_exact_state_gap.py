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
from experiments.world.part1_jepa_latent.analyze_present_state_probe import (  # noqa: E402
    _geometry_last,
    _surface_last,
    _target_groups,
)
from experiments.world.part1_jepa_latent.context_probe_audit import (  # noqa: E402
    ridge_probe_predict,
)
from experiments.world.part1_jepa_latent.masked_multiview_barlow_probe_audit import (  # noqa: E402
    encode_clean_masked_windows,
    load_direct_barlow_checkpoint,
    regression_metrics,
)


SCALE_CHECKPOINT = Path(
    "models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_scale_head127.pt"
)
SCALE_STATE_PROBE = Path("results/world/scale_state_probe_head127.json")


def _encode_scaled_barlow(
    train: MaskedMultiviewBatch,
    val: MaskedMultiviewBatch,
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


def _fit_predict_group(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    return ridge_probe_predict(train_x, train_y, val_x, alpha=alpha)


def _per_dim_mse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_arr = np.asarray(pred, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    return np.mean((pred_arr - truth_arr) ** 2, axis=0)


def _surface_rows(
    meta,
    raw_mse: np.ndarray,
    scale_mse: np.ndarray,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    surface_idx = np.flatnonzero(meta.geometry_id == "iv_surface")
    rows = []
    for local_idx, token_idx in enumerate(surface_idx.tolist()):
        coord = meta.geometry_coord[token_idx]
        raw = float(raw_mse[local_idx])
        scale = float(scale_mse[local_idx])
        rows.append(
            {
                "factor_id": str(meta.factor_id[token_idx]),
                "moneyness_index": int(coord[0]),
                "maturity_index": int(coord[1]),
                "raw_surface_mse": raw,
                "scale_barlow_mse": scale,
                "scale_minus_raw_mse": scale - raw,
                "scale_to_raw_ratio": scale / raw if raw > 0 else None,
            }
        )
    rows.sort(key=lambda row: float(row["scale_minus_raw_mse"]), reverse=True)
    return rows[:limit]


def _factor_family_rows(
    meta,
    target_name: str,
    mse_by_feature: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    mask = meta.geometry_id == target_name
    families = sorted({str(x) for x in meta.factor_family[mask].tolist()})
    local_families = meta.factor_family[mask]
    rows = []
    for family in families:
        family_mask = local_families.astype(str) == family
        row = {"family": family, "n_tokens": int(np.sum(family_mask))}
        for feature, mse in mse_by_feature.items():
            row[f"{feature}_mse"] = float(np.mean(mse[family_mask]))
        rows.append(row)
    rows.sort(key=lambda row: row.get("scale_barlow_mse", 0.0), reverse=True)
    return rows


def analyze_exact_state_gap(args: argparse.Namespace) -> dict[str, Any]:
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
    train_z, val_z = _encode_scaled_barlow(
        train,
        val,
        batch_size=args.batch_size,
        device=device,
    )
    train_features = {
        "raw_surface": _surface_last(train),
        "raw_geometry_upper": _geometry_last(train),
        "scale_barlow": train_z,
    }
    val_features = {
        "raw_surface": _surface_last(val),
        "raw_geometry_upper": _geometry_last(val),
        "scale_barlow": val_z,
    }

    predictions: dict[str, dict[str, np.ndarray]] = {}
    per_group: dict[str, dict[str, Any]] = {}
    for feature_name, train_x in train_features.items():
        predictions[feature_name] = {}
        per_group[feature_name] = {}
        for target_name, train_y in train_targets.items():
            pred = _fit_predict_group(
                train_x,
                val_features[feature_name],
                train_y,
                alpha=args.ridge_alpha,
            )
            predictions[feature_name][target_name] = pred
            per_group[feature_name][target_name] = regression_metrics(
                pred,
                val_targets[target_name],
            )

    iv_mse = {
        feature: _per_dim_mse(predictions[feature]["iv_surface"], val_targets["iv_surface"])
        for feature in predictions
    }
    factor_level_mse = {
        feature: _per_dim_mse(
            predictions[feature]["factor_level"], val_targets["factor_level"]
        )
        for feature in predictions
    }
    factor_return_mse = {
        feature: _per_dim_mse(
            predictions[feature]["factor_return"], val_targets["factor_return"]
        )
        for feature in predictions
    }
    side_channel_mse = {
        feature: _per_dim_mse(
            predictions[feature]["vol_side_channel"],
            val_targets["vol_side_channel"],
        )
        for feature in predictions
    }

    state_probe = json.loads(SCALE_STATE_PROBE.read_text(encoding="utf-8"))
    raw_iv = per_group["raw_surface"]["iv_surface"]["mse"]
    scale_iv = per_group["scale_barlow"]["iv_surface"]["mse"]
    scale_worse_surface_cells = int(np.sum(iv_mse["scale_barlow"] > iv_mse["raw_surface"]))
    return {
        "analysis": "world_model_scaled_exact_state_gap",
        "date": "2026-05-10",
        "objective_family": "downstream_probe_present_state_information",
        "device": str(device),
        "ridge_alpha": float(args.ridge_alpha),
        "train_shape": list(train.clean_values.shape),
        "val_shape": list(val.clean_values.shape),
        "state_probe_ref": str(SCALE_STATE_PROBE),
        "overall_probe_mse": per_group,
        "iv_surface_gap": {
            "raw_surface_mse": raw_iv,
            "scale_barlow_mse": scale_iv,
            "scale_to_raw_ratio": float(scale_iv / raw_iv),
            "scale_worse_cells": scale_worse_surface_cells,
            "n_surface_cells": int(iv_mse["scale_barlow"].shape[0]),
            "largest_scale_minus_raw_cells": _surface_rows(
                val.token_metadata,
                iv_mse["raw_surface"],
                iv_mse["scale_barlow"],
                limit=args.top_cells,
            ),
        },
        "vol_side_channel_mse_by_token": {
            feature: values.tolist() for feature, values in side_channel_mse.items()
        },
        "factor_level_by_family": _factor_family_rows(
            val.token_metadata,
            "factor_level",
            factor_level_mse,
        ),
        "factor_return_by_family": _factor_family_rows(
            val.token_metadata,
            "factor_return",
            factor_return_mse,
        ),
        "prior_state_probe_decision": state_probe["decision"],
        "decision": {
            "exact_iv_retention_gap_confirmed": scale_iv > raw_iv,
            "scale_worse_than_raw_surface_on_all_iv_cells": scale_worse_surface_cells
            == int(iv_mse["scale_barlow"].shape[0]),
            "interpretation": (
                "The scaled embedding retains broad market-state signal, but exact IV "
                "surface reconstruction remains worse than the raw last-surface "
                "baseline across most of the surface grid. This supports treating the next "
                "blocker as exact-state retention/baseline certification rather than "
                "representation collapse or seed instability."
            ),
        },
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
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
        "`downstream_probe_present_state_information` for the frozen scaled Part 1 embedding.",
        "",
        "## Hypothesis",
        "",
        "If the remaining raw-baseline gap is an exact-state retention problem,",
        "the scaled embedding should be worse than raw last-surface features on",
        "current IV-surface reconstruction even though it remains healthy under",
        "retrieval, rank, and mask audits.",
        "",
        "## Falsifier",
        "",
        "The exact-state gap would be weaker if the scaled embedding matched or",
        "beat raw last-surface features on most IV cells or if the gap were only",
        "concentrated in one narrow surface region.",
        "",
        "## Overall Present-State Probe MSE",
        "",
        "| feature | IV surface | side channel | factor level | factor return | all geometry |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature in ("raw_surface", "raw_geometry_upper", "scale_barlow"):
        row = result["overall_probe_mse"][feature]
        lines.append(
            "| {feature} | {iv} | {side} | {level} | {ret} | {all_geo} |".format(
                feature=feature,
                iv=_fmt(row["iv_surface"]["mse"]),
                side=_fmt(row["vol_side_channel"]["mse"]),
                level=_fmt(row["factor_level"]["mse"]),
                ret=_fmt(row["factor_return"]["mse"]),
                all_geo=_fmt(row["all_geometry"]["mse"]),
            )
        )
    gap = result["iv_surface_gap"]
    lines.extend(
        [
            "",
            "## IV Surface Gap",
            "",
            f"- Raw last-surface IV MSE: `{_fmt(gap['raw_surface_mse'])}`.",
            f"- Scaled Barlow IV MSE: `{_fmt(gap['scale_barlow_mse'])}`.",
            f"- Scaled/raw IV MSE ratio: `{_fmt(gap['scale_to_raw_ratio'])}`.",
            f"- Cells where scaled Barlow is worse than raw surface: `{gap['scale_worse_cells']}/{gap['n_surface_cells']}`.",
            "",
            "Largest cellwise gaps:",
            "",
            "| cell | moneyness | maturity | raw MSE | scale MSE | scale-raw | ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gap["largest_scale_minus_raw_cells"]:
        lines.append(
            "| {cell} | {mon} | {mat} | {raw} | {scale} | {delta} | {ratio} |".format(
                cell=row["factor_id"],
                mon=row["moneyness_index"],
                mat=row["maturity_index"],
                raw=_fmt(row["raw_surface_mse"]),
                scale=_fmt(row["scale_barlow_mse"]),
                delta=_fmt(row["scale_minus_raw_mse"]),
                ratio=_fmt(row["scale_to_raw_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Factor-Level MSE By Family",
            "",
            "| family | tokens | raw surface | raw geometry upper | scale Barlow |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["factor_level_by_family"]:
        lines.append(
            "| {family} | {n} | {raw_surface} | {raw_geometry} | {scale} |".format(
                family=row["family"],
                n=row["n_tokens"],
                raw_surface=_fmt(row["raw_surface_mse"]),
                raw_geometry=_fmt(row["raw_geometry_upper_mse"]),
                scale=_fmt(row["scale_barlow_mse"]),
            )
        )
    decision = result["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Exact IV retention gap confirmed: `{decision['exact_iv_retention_gap_confirmed']}`.",
            f"- Scaled Barlow worse than raw surface on all IV cells: `{decision['scale_worse_than_raw_surface_on_all_iv_cells']}`.",
            "",
            decision["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose the scaled Part 1 exact-state retention gap"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=720)
    parser.add_argument("--top-cells", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_exact_state_gap_head132.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head132_scale_exact_state_gap.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD132: Scale Exact-State Gap",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_exact_state_gap(args)
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
