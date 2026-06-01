#!/usr/bin/env python
"""Calibration diagnostics for held-out narrative prefix-latent backtests."""

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
from experiments.backfill.block_ar.plot_narrative_casebook_backtest import (  # noqa: E402
    _selected_history_block,
)


METHODS = ("averaged_prefix", "component_prefix_mixture")


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


def coverage_matrix(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return [T,C] mask for whether target lies inside the 10-90 sample band."""

    sample_arr = np.asarray(samples, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    if sample_arr.ndim != 3:
        raise ValueError("samples must have shape [S,T,C]")
    if target_arr.shape != sample_arr.shape[1:]:
        raise ValueError(
            f"target shape {target_arr.shape} does not match samples {sample_arr.shape}"
        )
    q10 = np.nanquantile(sample_arr, 0.10, axis=0)
    q90 = np.nanquantile(sample_arr, 0.90, axis=0)
    return ((target_arr >= q10) & (target_arr <= q90)).astype(np.float32)


def pit_matrix(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Approximate PIT/rank matrix with values in (0, 1)."""

    sample_arr = np.asarray(samples, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    if sample_arr.ndim != 3:
        raise ValueError("samples must have shape [S,T,C]")
    ranks = np.sum(sample_arr <= target_arr[None, :, :], axis=0)
    return ((ranks + 0.5) / float(sample_arr.shape[0] + 1)).astype(np.float32)


def interval_width(samples: np.ndarray) -> np.ndarray:
    sample_arr = np.asarray(samples, dtype=np.float32)
    return (
        np.nanquantile(sample_arr, 0.90, axis=0)
        - np.nanquantile(sample_arr, 0.10, axis=0)
    ).astype(np.float32)


def _histogram(values: np.ndarray, bins: int = 10) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    counts, edges = np.histogram(arr, bins=int(bins), range=(0.0, 1.0))
    freq = counts.astype(np.float64) / max(int(np.sum(counts)), 1)
    expected = 1.0 / int(bins)
    return {
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
        "frequencies": [float(x) for x in freq.tolist()],
        "mean_abs_uniform_error": float(np.mean(np.abs(freq - expected))),
    }


def _method_arrays(
    row: dict[str, Any],
    method: str,
) -> tuple[np.ndarray, Path]:
    artifact = row.get("artifacts", {}).get(method, {})
    arrays_path = Path(str(artifact.get("arrays", "")))
    if not arrays_path.exists():
        raise FileNotFoundError(f"{method} arrays not found: {arrays_path}")
    arrays = np.load(arrays_path)
    states = np.asarray(arrays["generated_states"], dtype=np.float32)
    if states.ndim != 4 or states.shape[0] < 1:
        raise ValueError(f"{arrays_path}: expected generated_states [K,S,T,C]")
    return states[0], arrays_path


def _target_for_row(row: dict[str, Any], future_raw: np.ndarray) -> np.ndarray:
    start_idx = int(row.get("start_window_index", -1))
    if start_idx < 0 or start_idx >= future_raw.shape[0]:
        raise IndexError(f"start_window_index {start_idx} outside future_raw")
    return np.asarray(future_raw[start_idx], dtype=np.float32)


def _summarize_method(
    *,
    method: str,
    rows: list[dict[str, Any]],
    future_raw: np.ndarray,
    spec_names: list[str],
) -> dict[str, Any]:
    coverages: list[np.ndarray] = []
    pits: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    miss_low: list[np.ndarray] = []
    miss_high: list[np.ndarray] = []
    sample_count = None
    for row in rows:
        samples, _arrays_path = _method_arrays(row, method)
        target = _target_for_row(row, future_raw)
        q10 = np.nanquantile(samples, 0.10, axis=0)
        q90 = np.nanquantile(samples, 0.90, axis=0)
        coverages.append(coverage_matrix(samples, target))
        pits.append(pit_matrix(samples, target))
        widths.append(interval_width(samples))
        miss_low.append((target < q10).astype(np.float32))
        miss_high.append((target > q90).astype(np.float32))
        sample_count = int(samples.shape[0])
    cov = np.stack(coverages, axis=0)
    pit = np.stack(pits, axis=0)
    width = np.stack(widths, axis=0)
    low = np.stack(miss_low, axis=0)
    high = np.stack(miss_high, axis=0)
    factor_rows = []
    terminal_rows = []
    for col, name in enumerate(spec_names):
        factor_rows.append(
            {
                "factor": str(name),
                "coverage_80": float(np.nanmean(cov[:, :, col])),
                "miss_low_rate": float(np.nanmean(low[:, :, col])),
                "miss_high_rate": float(np.nanmean(high[:, :, col])),
                "mean_10_90_width": float(np.nanmean(width[:, :, col])),
                "pit_mean": float(np.nanmean(pit[:, :, col])),
            }
        )
        terminal_rows.append(
            {
                "factor": str(name),
                "terminal_coverage_80": float(np.nanmean(cov[:, -1, col])),
                "terminal_miss_low_rate": float(np.nanmean(low[:, -1, col])),
                "terminal_miss_high_rate": float(np.nanmean(high[:, -1, col])),
                "terminal_mean_10_90_width": float(np.nanmean(width[:, -1, col])),
                "terminal_pit_mean": float(np.nanmean(pit[:, -1, col])),
            }
        )
    horizon_rows = []
    for step in range(cov.shape[1]):
        horizon_rows.append(
            {
                "forward_day": int(step + 1),
                "coverage_80": float(np.nanmean(cov[:, step, :])),
                "miss_low_rate": float(np.nanmean(low[:, step, :])),
                "miss_high_rate": float(np.nanmean(high[:, step, :])),
                "mean_10_90_width": float(np.nanmean(width[:, step, :])),
            }
        )
    factor_rows = sorted(factor_rows, key=lambda row: float(row["coverage_80"]))
    terminal_rows = sorted(
        terminal_rows, key=lambda row: float(row["terminal_coverage_80"])
    )
    return {
        "method": method,
        "window_count": int(cov.shape[0]),
        "sample_count": sample_count,
        "nominal_coverage": 0.80,
        "overall_coverage_80": float(np.nanmean(cov)),
        "terminal_coverage_80": float(np.nanmean(cov[:, -1, :])),
        "overall_miss_low_rate": float(np.nanmean(low)),
        "overall_miss_high_rate": float(np.nanmean(high)),
        "overall_mean_10_90_width": float(np.nanmean(width)),
        "pit_histogram": _histogram(pit),
        "worst_factor_coverage": factor_rows[:10],
        "worst_terminal_factor_coverage": terminal_rows[:10],
        "horizon_coverage": horizon_rows,
    }


def build_calibration_report(args: argparse.Namespace) -> dict[str, Any]:
    backtest = _load_json(args.backtest_report)
    rows = backtest.get("window_scores", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("backtest report has no window_scores")
    device = torch.device(
        str(args.device) if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu"
    )
    block = _selected_history_block(
        checkpoint=str(backtest.get("checkpoint") or args.checkpoint),
        bridge_report=str(backtest.get("bridge_report") or args.bridge_report),
        device=device,
    )
    future_raw = np.asarray(block["future_raw"], dtype=np.float32)
    spec_names = [str(x) for x in block["spec_names"].tolist()]
    method_reports = {
        method: _summarize_method(
            method=method,
            rows=rows,
            future_raw=future_raw,
            spec_names=spec_names,
        )
        for method in METHODS
    }
    component = method_reports["component_prefix_mixture"]
    averaged = method_reports["averaged_prefix"]
    coverage_gap = float(
        component["overall_coverage_80"] - averaged["overall_coverage_80"]
    )
    status = "pass" if component["overall_coverage_80"] >= averaged["overall_coverage_80"] else "warning"
    report = {
        "status": status,
        "scope_note": (
            "Held-out raw-level calibration diagnostic for the narrative "
            "prefix-latent component backtest. Coverage is computed against the "
            "realized next-30-day raw joint39 path using the generated 10-90% band."
        ),
        "backtest_report": str(args.backtest_report),
        "window_count": int(len(rows)),
        "methods": method_reports,
        "comparison": {
            "component_minus_averaged_overall_coverage_80": coverage_gap,
            "component_minus_averaged_terminal_coverage_80": float(
                component["terminal_coverage_80"] - averaged["terminal_coverage_80"]
            ),
            "component_minus_averaged_pit_uniform_error": float(
                component["pit_histogram"]["mean_abs_uniform_error"]
                - averaged["pit_histogram"]["mean_abs_uniform_error"]
            ),
        },
        "interpretation": [
            "Coverage is a calibration diagnostic, not a scenario quality score by itself.",
            "Low 80% coverage means the generated fan chart is too narrow or miscentered for realized historical paths.",
            "The component path can be distributionally better on CRPS/energy while still needing calibration work.",
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = build_calibration_report(args)
    report["artifact_paths"] = {
        "report": str(output_dir / "component_backtest_calibration_report.json")
    }
    _write_json(report["artifact_paths"]["report"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "comparison": report["comparison"],
                "component_coverage_80": report["methods"][
                    "component_prefix_mixture"
                ]["overall_coverage_80"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
