#!/usr/bin/env python
"""Fit a one-parameter global fan calibration for component-mixture rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    score_sample_distribution,
    summarize_method_scores,
)
from experiments.backfill.block_ar.plot_narrative_casebook_backtest import (  # noqa: E402
    _selected_history_block,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def scale_samples_around_mean(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Scale path samples around their ensemble mean without moving the mean."""

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("samples must have shape [S,T,C]")
    mean = np.nanmean(arr, axis=0, keepdims=True)
    return (mean + float(alpha) * (arr - mean)).astype(np.float32)


def _parse_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("alpha grid must contain at least one value")
    return sorted(set(values))


def _target_delta_for_row(
    row: dict[str, Any],
    *,
    history_raw: np.ndarray,
    future_raw: np.ndarray,
) -> np.ndarray:
    start_idx = int(row.get("start_window_index", -1))
    if start_idx < 0 or start_idx >= future_raw.shape[0]:
        raise IndexError(f"start_window_index {start_idx} outside future_raw")
    start = np.asarray(history_raw[start_idx, -1, :], dtype=np.float32)
    future = np.asarray(future_raw[start_idx], dtype=np.float32)
    return (future - start[None, :]).astype(np.float32)


def _component_samples_for_row(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    artifact = row.get("artifacts", {}).get("component_prefix_mixture", {})
    arrays_path = Path(str(artifact.get("arrays", "")))
    if not arrays_path.exists():
        raise FileNotFoundError(f"component arrays not found: {arrays_path}")
    arrays = np.load(arrays_path)
    samples = np.asarray(arrays["samples"], dtype=np.float32)
    scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
    if samples.ndim != 4 or samples.shape[0] < 1:
        raise ValueError(f"{arrays_path}: expected samples shape [K,S,T,C]")
    return samples[0], scale


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    history_raw: np.ndarray,
    future_raw: np.ndarray,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        samples, delta_scale = _component_samples_for_row(row)
        target = _target_delta_for_row(
            row,
            history_raw=history_raw,
            future_raw=future_raw,
        )
        calibrated = scale_samples_around_mean(samples, alpha)
        methods = {
            "persistence": row["methods"]["persistence"],
            "averaged_prefix": row["methods"]["averaged_prefix"],
            "component_prefix_mixture": row["methods"]["component_prefix_mixture"],
            "component_calibrated_global": score_sample_distribution(
                calibrated,
                target,
                scale=delta_scale,
            ),
        }
        scored.append(
            {
                "row_no": int(row.get("row_no", len(scored))),
                "window_index": int(row.get("window_index", -1)),
                "window_id": str(row.get("window_id", "")),
                "start_window_index": int(row.get("start_window_index", -1)),
                "methods": methods,
            }
        )
    return scored


def _choose_alpha(
    rows: list[dict[str, Any]],
    *,
    alpha_grid: list[float],
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    target_coverage: float,
) -> tuple[float, list[dict[str, Any]]]:
    candidates = []
    for alpha in alpha_grid:
        scored = _score_rows(
            rows,
            alpha=float(alpha),
            history_raw=history_raw,
            future_raw=future_raw,
        )
        summary = summarize_method_scores(scored, baseline="persistence")
        method = summary["component_calibrated_global"]
        coverage = float(method.get("coverage_80_mean", 0.0))
        crps = float(method.get("ensemble_crps_z_mean", np.inf))
        energy = float(method.get("energy_score_z_mean", np.inf))
        candidates.append(
            {
                "alpha": float(alpha),
                "coverage_80_mean": coverage,
                "coverage_error": abs(coverage - float(target_coverage)),
                "ensemble_crps_z_mean": crps,
                "energy_score_z_mean": energy,
                "summary": summary,
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["coverage_error"]),
            float(item["ensemble_crps_z_mean"]),
            float(item["alpha"]),
        ),
    )
    return float(selected["alpha"]), candidates


def split_calibration_rows(
    rows: list[dict[str, Any]],
    *,
    calibration_count: int,
    split_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into calibration/evaluation partitions."""

    if len(rows) < 4:
        raise ValueError("at least four rows are required")
    mode = str(split_mode)
    if mode == "chronological":
        count = min(max(int(calibration_count), 1), len(rows) - 1)
        return rows[:count], rows[count:]
    if mode == "reverse":
        count = min(max(int(calibration_count), 1), len(rows) - 1)
        return rows[-count:], rows[:-count]
    if mode == "even_odd":
        return rows[::2], rows[1::2]
    if mode == "odd_even":
        return rows[1::2], rows[::2]
    raise ValueError(f"unknown split_mode: {split_mode!r}")


def run_global_calibration(args: argparse.Namespace) -> dict[str, Any]:
    backtest = _load_json(args.backtest_report)
    rows = backtest.get("window_scores", [])
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("backtest report needs at least four scored rows")
    device = torch.device(
        str(args.device) if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu"
    )
    block = _selected_history_block(
        checkpoint=str(backtest.get("checkpoint") or args.checkpoint),
        bridge_report=str(backtest.get("bridge_report") or args.bridge_report),
        device=device,
    )
    history_raw = np.asarray(block["history_raw"], dtype=np.float32)
    future_raw = np.asarray(block["future_raw"], dtype=np.float32)
    calibration_count = int(args.calibration_count)
    if calibration_count <= 0:
        calibration_count = max(1, len(rows) // 2)
    calibration_rows, evaluation_rows = split_calibration_rows(
        rows,
        calibration_count=calibration_count,
        split_mode=str(args.split_mode),
    )
    alpha_grid = _parse_grid(str(args.alpha_grid))
    selected_alpha, fit_candidates = _choose_alpha(
        calibration_rows,
        alpha_grid=alpha_grid,
        history_raw=history_raw,
        future_raw=future_raw,
        target_coverage=float(args.target_coverage),
    )
    eval_scores = _score_rows(
        evaluation_rows,
        alpha=selected_alpha,
        history_raw=history_raw,
        future_raw=future_raw,
    )
    eval_summary = summarize_method_scores(eval_scores, baseline="persistence")
    full_scores = _score_rows(
        rows,
        alpha=selected_alpha,
        history_raw=history_raw,
        future_raw=future_raw,
    )
    full_summary = summarize_method_scores(full_scores, baseline="persistence")
    eval_component = eval_summary["component_prefix_mixture"]
    eval_cal = eval_summary["component_calibrated_global"]
    report = {
        "status": "pass"
        if float(eval_cal.get("coverage_80_mean", 0.0))
        > float(eval_component.get("coverage_80_mean", 0.0))
        else "warning",
        "scope_note": (
            "One-parameter post-hoc fan calibration. The alpha scale is fit on "
            "the first calibration block and evaluated on the remaining held-out "
            "windows. Samples are scaled around their ensemble mean, so the mean "
            "path is unchanged."
        ),
        "backtest_report": str(args.backtest_report),
        "target_coverage": float(args.target_coverage),
        "alpha_grid": alpha_grid,
        "selected_alpha": selected_alpha,
        "split_mode": str(args.split_mode),
        "calibration_window_count": int(len(calibration_rows)),
        "evaluation_window_count": int(len(evaluation_rows)),
        "fit_candidates": fit_candidates,
        "evaluation_summary": eval_summary,
        "full_summary_at_selected_alpha": full_summary,
        "comparison": {
            "eval_calibrated_minus_component_coverage_80": float(
                eval_cal.get("coverage_80_mean", 0.0)
                - eval_component.get("coverage_80_mean", 0.0)
            ),
            "eval_calibrated_minus_component_crps": float(
                eval_cal.get("ensemble_crps_z_mean", np.nan)
                - eval_component.get("ensemble_crps_z_mean", np.nan)
            ),
            "eval_calibrated_minus_component_energy": float(
                eval_cal.get("energy_score_z_mean", np.nan)
                - eval_component.get("energy_score_z_mean", np.nan)
            ),
        },
        "interpretation": [
            "This is a calibration diagnostic, not a new narrative-conditioning method.",
            "A useful alpha should improve coverage on evaluation rows without destroying CRPS/energy.",
            "If a large alpha is required, the base generator fan is under-dispersed for realized historical paths.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest-report",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_component_backtest_heldout_29w_s96/"
            "component_backtest_report.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--bridge-report",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "manifest_bridge_eval_openai_schema_v2_representative_220/"
            "bridge_eval_report.json"
        ),
    )
    parser.add_argument(
        "--alpha-grid",
        default="1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,5.0,6.0",
    )
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--calibration-count", type=int, default=15)
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "reverse", "even_odd", "odd_even"],
        default="chronological",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = run_global_calibration(args)
    report["artifact_paths"] = {
        "report": str(output_dir / "component_global_calibration_report.json")
    }
    _write_json(report["artifact_paths"]["report"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_alpha": report["selected_alpha"],
                "comparison": report["comparison"],
                "report": report["artifact_paths"]["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
