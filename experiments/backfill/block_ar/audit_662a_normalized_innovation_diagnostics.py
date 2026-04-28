#!/usr/bin/env python
"""662a generated-coordinate diagnostics for normalized innovations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    clean_nonpositive_log_level_factors,
)
from experiments.backfill.block_ar.audit_627a_joint_panel_scenario_quality import (  # noqa: E402
    select_raw_state_scope,
)
from experiments.backfill.block_ar.evaluate_438a_deployable_residual_bootstrap_system import (  # noqa: E402
    set_seed,
)
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    build_increment_coordinate_block,
    reconstruct_state_from_increments,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    select_normalized_innovation_scope,
)


def quantile_abs(x: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(np.abs(np.asarray(x, dtype=np.float64)), q, axis=tuple(range(x.ndim - 1)))


def finite_ratio(x: np.ndarray) -> float:
    return float(np.isfinite(x).mean())


def top_rows(names: list[str], values: np.ndarray, *, k: int = 8) -> list[dict[str, Any]]:
    order = np.argsort(np.nan_to_num(values, nan=-np.inf, posinf=np.inf, neginf=-np.inf))[::-1]
    rows: list[dict[str, Any]] = []
    for idx in order[: min(k, len(order))]:
        rows.append({"name": names[int(idx)], "value": float(values[int(idx)])})
    return rows


def summarize_coordinate(
    names: list[str],
    gt_norm: np.ndarray,
    gen_norm: np.ndarray,
    gt_increment: np.ndarray,
    gen_increment: np.ndarray,
) -> dict[str, Any]:
    gt_norm_q99 = quantile_abs(gt_norm, 0.99)
    gen_norm_q99 = quantile_abs(gen_norm, 0.99)
    gt_inc_q99 = quantile_abs(gt_increment, 0.99)
    gen_inc_q99 = quantile_abs(gen_increment, 0.99)
    norm_ratio = gen_norm_q99 / np.maximum(gt_norm_q99, 1e-12)
    inc_ratio = gen_inc_q99 / np.maximum(gt_inc_q99, 1e-12)
    return {
        "normalized_q99_ratio_median": float(np.nanmedian(norm_ratio)),
        "normalized_q99_ratio_max": float(np.nanmax(norm_ratio)),
        "normalized_q99_pass_05_20": int(np.sum((norm_ratio >= 0.5) & (norm_ratio <= 2.0))),
        "encoded_increment_q99_ratio_median": float(np.nanmedian(inc_ratio)),
        "encoded_increment_q99_ratio_max": float(np.nanmax(inc_ratio)),
        "encoded_increment_q99_pass_05_20": int(np.sum((inc_ratio >= 0.5) & (inc_ratio <= 2.0))),
        "n_channels": int(len(names)),
        "worst_normalized_q99_ratio": top_rows(names, norm_ratio),
        "worst_encoded_increment_q99_ratio": top_rows(names, inc_ratio),
    }


def summarize_reconstruction(
    names: list[str],
    raw_history: np.ndarray,
    raw_future: np.ndarray,
    samples_raw: np.ndarray,
    oracle_recon: np.ndarray,
    *,
    iv_count: int,
) -> dict[str, Any]:
    recon_error = np.abs(oracle_recon - raw_future)
    raw_min = np.nanmin(samples_raw, axis=tuple(range(samples_raw.ndim - 1)))
    raw_max = np.nanmax(samples_raw, axis=tuple(range(samples_raw.ndim - 1)))
    out: dict[str, Any] = {
        "oracle_reconstruction_max_abs_error": float(np.nanmax(recon_error)),
        "oracle_reconstruction_mean_abs_error": float(np.nanmean(recon_error)),
        "generated_raw_finite_rate": finite_ratio(samples_raw),
        "generated_raw_min": float(np.nanmin(samples_raw)),
        "generated_raw_max": float(np.nanmax(samples_raw)),
        "worst_raw_max": top_rows(names, raw_max),
        "worst_raw_min_abs": top_rows(names, -raw_min),
    }
    if iv_count > 0:
        iv = samples_raw[..., :iv_count]
        out["iv_nonpositive_rate"] = float((iv <= 0.0).mean())
        out["iv_above_1_rate"] = float((iv >= 1.0).mean())
        out["iv_above_5_rate"] = float((iv >= 5.0).mean())
        out["iv_max"] = float(np.nanmax(iv))
    return out


def build_scope(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[Any, ...]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    positive_level_policy = payload.get(
        "positive_level_policy",
        payload.get("panel_metadata", {}).get(
            "positive_level_policy",
            getattr(args, "positive_level_policy", "reference_based"),
        ),
    )
    if args.clean_nonpositive_log_levels:
        panel, _cleaning_report = clean_nonpositive_log_level_factors(
            panel,
            columns,
            iv_count=int(args.iv_count),
            positive_level_policy=positive_level_policy,
        )
    iv_transform = payload.get(
        "iv_transform",
        payload.get("normalization", {}).get(
            "iv_transform",
            payload.get("panel_metadata", {}).get(
                "iv_transform",
                getattr(args, "iv_transform", "log_level"),
            ),
        ),
    )
    iv_lower_bound = float(
        payload.get(
            "iv_lower_bound",
            payload.get("normalization", {}).get(
                "iv_lower_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_lower_bound",
                    getattr(args, "iv_lower_bound", 1e-4),
                ),
            ),
        )
    )
    iv_upper_bound = float(
        payload.get(
            "iv_upper_bound",
            payload.get("normalization", {}).get(
                "iv_upper_bound",
                payload.get("panel_metadata", {}).get(
                    "iv_upper_bound",
                    getattr(args, "iv_upper_bound", 1.0),
                ),
            ),
        )
    )
    _train_indices, val_indices = official_train_val_indices(
        test_start=int(args.test_start),
        val_size=int(args.val_size),
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
    )
    if int(args.max_windows) > 0:
        val_indices = val_indices[: int(args.max_windows)]
    block = build_increment_coordinate_block(
        panel,
        columns,
        val_indices,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
        iv_count=int(args.iv_count),
        positive_level_policy=positive_level_policy,
        iv_transform=iv_transform,
        iv_lower_bound=iv_lower_bound,
        iv_upper_bound=iv_upper_bound,
    )
    norm_cfg = payload.get("normalization", {})
    scale_half_life = norm_cfg.get("scale_half_life", 0.0)
    if scale_half_life is not None and float(scale_half_life) <= 0.0:
        scale_half_life = None
    scope = payload.get("state_scope", args.state_scope)
    (
        history_level,
        history_norm,
        _future_level,
        future_norm,
        center,
        scale,
        _history_raw,
        specs,
    ) = select_normalized_innovation_scope(
        block,
        scope,
        int(args.iv_count),
        scale_half_life=scale_half_life,
        scale_floor=float(norm_cfg.get("scale_floor", args.scale_floor)),
    )
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    actual = [spec.name for spec in specs]
    if expected and expected != actual:
        raise RuntimeError("checkpoint state specs do not match rebuilt specs")
    raw_history, raw_future = select_raw_state_scope(block, scope, int(args.iv_count))
    return (
        scope,
        history_level.astype(np.float32),
        history_norm.astype(np.float32),
        future_norm.astype(np.float32),
        center.astype(np.float32),
        scale.astype(np.float32),
        raw_history.astype(np.float32),
        raw_future.astype(np.float32),
        specs,
    )


@torch.no_grad()
def generate_increments(
    model: Any,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    temperature: float,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, int(history_level.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(history_level.shape[0]))
        out = model.sample_batched(
            torch.from_numpy(history_level[start:end]).to(device),
            torch.from_numpy(history_norm[start:end]).to(device),
            torch.from_numpy(center[start:end]).to(device),
            torch.from_numpy(scale[start:end]).to(device),
            n_samples=int(samples),
            n_steps=int(n_steps),
            chunk_size=int(chunk_size),
            temperature=float(temperature),
        )
        chunks.append(out.detach().cpu().numpy().astype(np.float32))
        print(f"  generated windows {end}/{history_level.shape[0]}", flush=True)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state_scope", choices=["iv_only", "anchor_only", "joint38"], default="joint38")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=662)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    (
        scope,
        history_level,
        history_norm,
        future_norm,
        center,
        scale,
        raw_history,
        raw_future,
        specs,
    ) = build_scope(args, payload)
    n = min(int(args.max_windows), int(history_level.shape[0]))
    history_level = history_level[:n]
    history_norm = history_norm[:n]
    future_norm = future_norm[:n]
    center = center[:n]
    scale = scale[:n]
    raw_history = raw_history[:n]
    raw_future = raw_future[:n]

    t0 = time.time()
    gen_increment = generate_increments(
        model,
        history_level,
        history_norm,
        center,
        scale,
        samples=int(args.samples),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=device,
        temperature=float(args.sample_temperature),
    )
    safe_scale = np.maximum(scale, 1e-12)
    gen_norm = gen_increment / safe_scale[:, None, None, :]
    gt_norm = future_norm[:, : int(args.n_steps), :]
    gt_increment = gt_norm * safe_scale[:, None, :]
    samples_raw = reconstruct_state_from_increments(raw_history[:, -1, :], gen_increment, specs)
    oracle_recon = reconstruct_state_from_increments(raw_history[:, -1, :], gt_increment[:, None, :, :], specs)[:, 0]
    effective_iv_count = int(args.iv_count) if scope == "joint38" else (int(args.iv_count) if scope == "iv_only" else 0)
    names = [spec.name for spec in specs]
    result = {
        "summary": {
            "scope": scope,
            "coordinate": summarize_coordinate(
                names,
                gt_norm,
                gen_norm,
                gt_increment,
                gen_increment,
            ),
            "reconstruction": summarize_reconstruction(
                names,
                raw_history,
                raw_future[:, : int(args.n_steps), :],
                samples_raw,
                oracle_recon,
                iv_count=effective_iv_count,
            ),
        },
        "config": {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_best_val": float(payload.get("best_val", float("nan"))),
            "state_scope": scope,
            "n_windows": int(n),
            "samples": int(args.samples),
            "n_steps": int(args.n_steps),
            "sample_temperature": float(args.sample_temperature),
            "generation_time_s": float(time.time() - t0),
            "seed": int(args.seed),
        },
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(result), indent=2), encoding="utf-8")
    summary = result["summary"]
    lines = [
        "# 662a Normalized-Innovation Diagnostics",
        "",
        f"- scope: `{scope}`",
        f"- normalized q99 ratio median/max/pass: `{summary['coordinate']['normalized_q99_ratio_median']:.3f}` / `{summary['coordinate']['normalized_q99_ratio_max']:.3f}` / `{summary['coordinate']['normalized_q99_pass_05_20']}/{summary['coordinate']['n_channels']}`",
        f"- encoded-increment q99 ratio median/max/pass: `{summary['coordinate']['encoded_increment_q99_ratio_median']:.3f}` / `{summary['coordinate']['encoded_increment_q99_ratio_max']:.3f}` / `{summary['coordinate']['encoded_increment_q99_pass_05_20']}/{summary['coordinate']['n_channels']}`",
        f"- oracle reconstruction max abs error: `{summary['reconstruction']['oracle_reconstruction_max_abs_error']:.3e}`",
        f"- generated finite rate: `{summary['reconstruction']['generated_raw_finite_rate']:.4f}`",
        f"- generated raw min/max: `{summary['reconstruction']['generated_raw_min']:.4g}` / `{summary['reconstruction']['generated_raw_max']:.4g}`",
    ]
    if "iv_above_1_rate" in summary["reconstruction"]:
        lines.extend(
            [
                f"- IV >= 1 rate: `{summary['reconstruction']['iv_above_1_rate']:.4f}`",
                f"- IV >= 5 rate: `{summary['reconstruction']['iv_above_5_rate']:.4f}`",
                f"- IV max: `{summary['reconstruction']['iv_max']:.4g}`",
            ]
        )
    lines.extend(["", "**Worst Normalized q99 Ratios**"])
    for row in summary["coordinate"]["worst_normalized_q99_ratio"]:
        lines.append(f"- {row['name']}: `{row['value']:.3f}`")
    lines.extend(["", "**Worst Encoded-Increment q99 Ratios**"])
    for row in summary["coordinate"]["worst_encoded_increment_q99_ratio"]:
        lines.append(f"- {row['name']}: `{row['value']:.3f}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(make_serializable(result["summary"]), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
