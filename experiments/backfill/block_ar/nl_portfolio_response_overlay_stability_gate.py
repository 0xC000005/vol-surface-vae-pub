#!/usr/bin/env python
"""Summarize stability of portfolio-response overlay comparisons.

This script is intentionally evaluation-only.  It reads already-computed
candidate-vs-equal comparison JSON files and decides whether a portfolio-risk
overlay is:

- clean enough to be a default candidate;
- useful as an optional portfolio overlay; or
- not currently a lever.

The gate exists to avoid promoting a support policy from a single lucky seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def build_overlay_stability_gate(
    comparison_reports: list[dict[str, Any]],
    *,
    energy_regression_tolerance: float = 0.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, report in enumerate(comparison_reports):
        scenario = report.get("scenario_delta_candidate_minus_equal", {})
        portfolio = report.get("portfolio_delta_candidate_minus_equal", {})
        row = {
            "index": idx,
            "status": str(report.get("status", "")),
            "crps_delta": float(scenario["ensemble_crps_z_mean"]),
            "energy_delta": float(scenario["energy_score_z_mean"]),
            "coverage_delta": float(scenario["coverage_80_mean"]),
            "portfolio_path_delta": float(portfolio["portfolio_reliable_path_score_z"]),
        }
        row["clean_seed"] = bool(
            row["crps_delta"] <= 0.0
            and row["energy_delta"] <= float(energy_regression_tolerance)
            and row["coverage_delta"] >= 0.0
            and row["portfolio_path_delta"] < 0.0
        )
        row["portfolio_useful_seed"] = bool(
            row["portfolio_path_delta"] < 0.0
            and row["crps_delta"] <= 0.0
            and row["coverage_delta"] >= 0.0
            and row["energy_delta"] <= max(float(energy_regression_tolerance), 5e-4)
        )
        rows.append(row)
    if not rows:
        raise ValueError("at least one comparison report is required")
    clean_count = sum(1 for row in rows if row["clean_seed"])
    portfolio_useful_count = sum(1 for row in rows if row["portfolio_useful_seed"])
    seed_count = len(rows)
    all_portfolio_useful = portfolio_useful_count == seed_count
    all_clean = clean_count == seed_count
    if all_clean:
        result_status = "overlay_default_candidate"
        benchmark_floor_status = "beats_floor"
        interpretation = (
            "All comparison seeds improve broad scenario quality, coverage, "
            "and reliable portfolio response versus equal support."
        )
    elif all_portfolio_useful:
        result_status = "overlay_portfolio_candidate"
        benchmark_floor_status = "competitive"
        interpretation = (
            "All comparison seeds improve portfolio response, CRPS, and "
            "coverage with at most a tiny energy trade-off. Keep as an "
            "optional portfolio-risk overlay, not the default broad scenario "
            "policy."
        )
    else:
        result_status = "overlay_not_current_lever"
        benchmark_floor_status = "below_floor"
        interpretation = (
            "The overlay does not consistently improve portfolio response and "
            "broad scenario diagnostics across comparison seeds."
        )
    return {
        "status": "ok",
        "result_status": result_status,
        "benchmark_floor_status": benchmark_floor_status,
        "seed_count": seed_count,
        "clean_seed_count": int(clean_count),
        "portfolio_useful_seed_count": int(portfolio_useful_count),
        "energy_regression_tolerance": float(energy_regression_tolerance),
        "summary": {
            "mean_crps_delta": _mean([row["crps_delta"] for row in rows]),
            "mean_energy_delta": _mean([row["energy_delta"] for row in rows]),
            "mean_coverage_delta": _mean([row["coverage_delta"] for row in rows]),
            "mean_portfolio_path_delta": _mean(
                [row["portfolio_path_delta"] for row in rows]
            ),
            "max_energy_delta": max(row["energy_delta"] for row in rows),
        },
        "rows": rows,
        "decision": {
            "interpretation": interpretation,
            "promote_default": bool(all_clean),
            "promote_portfolio_overlay": bool(all_portfolio_useful),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Portfolio-Response Overlay Stability Gate",
        "",
        f"Status: `{report['result_status']}`",
        f"Benchmark floor: `{report['benchmark_floor_status']}`",
        "",
        "## Summary",
        "",
        f"- seeds: `{report['seed_count']}`",
        f"- clean seeds: `{report['clean_seed_count']}`",
        f"- portfolio-useful seeds: `{report['portfolio_useful_seed_count']}`",
        f"- mean CRPS delta: `{report['summary']['mean_crps_delta']}`",
        f"- mean energy delta: `{report['summary']['mean_energy_delta']}`",
        f"- mean coverage delta: `{report['summary']['mean_coverage_delta']}`",
        f"- mean portfolio path delta: `{report['summary']['mean_portfolio_path_delta']}`",
        "",
        "## Seeds",
        "",
        "| Index | Status | CRPS | Energy | Coverage | Portfolio Path | Clean | Useful |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["index"]),
                    row["status"],
                    str(row["crps_delta"]),
                    str(row["energy_delta"]),
                    str(row["coverage_delta"]),
                    str(row["portfolio_path_delta"]),
                    str(row["clean_seed"]),
                    str(row["portfolio_useful_seed"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Decision", "", report["decision"]["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_reports", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--energy-regression-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    reports = [_load_json(path) for path in args.comparison_reports]
    gate = build_overlay_stability_gate(
        reports,
        energy_regression_tolerance=float(args.energy_regression_tolerance),
    )
    gate["artifact_paths"] = {
        "comparison_reports": [str(path) for path in args.comparison_reports],
        "report": str(args.output_dir / "portfolio_response_overlay_stability_gate.json"),
        "markdown": str(args.output_dir / "portfolio_response_overlay_stability_gate.md"),
    }
    _write_json(args.output_dir / "portfolio_response_overlay_stability_gate.json", gate)
    _write_text(args.output_dir / "portfolio_response_overlay_stability_gate.md", _markdown(gate))
    print(
        json.dumps(
            {
                "status": gate["result_status"],
                "benchmark_floor_status": gate["benchmark_floor_status"],
                "report": gate["artifact_paths"]["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
