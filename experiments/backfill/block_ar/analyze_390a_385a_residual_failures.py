#!/usr/bin/env python
"""390a: residual failure audit for the 385a active frontier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RESULT_PATH = Path("results/block_ar/385a_recent_quantiles_fm_s42/full11.json")
OUT_DIR = Path("results/block_ar/390a_385a_residual_failure_audit")
HORIZONS = ("1", "7", "14", "30")


def arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def top_cells(values: np.ndarray, largest: bool, n: int = 8) -> list[dict[str, Any]]:
    items: list[tuple[float, int, int]] = []
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            items.append((float(values[i, j]), i, j))
    items.sort(reverse=largest, key=lambda item: item[0])
    return [
        {"cell": [i, j], "value": value}
        for value, i, j in items[:n]
    ]


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coverage_by_h = {
        h: arr(result["coverage"]["per_cell_coverage"][h])
        for h in HORIZONS
    }
    cov_stack = np.stack([coverage_by_h[h] for h in HORIZONS])
    under = cov_stack < 0.70
    over = cov_stack > 0.95

    level_ks = arr(result["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    median_frac = arr(result["distributional_fidelity"]["median_bias"]["above_frac"])
    mean_bias = arr(result["distributional_fidelity"]["median_bias"]["mean_bias"])
    regime = result["regime_coverage"]

    layer2 = regime["layer2_regime_cell"]
    layer2_rows: list[dict[str, Any]] = []
    for regime_name, horizons in layer2.items():
        for h, entry in horizons.items():
            grid = arr(entry["grid"])
            layer2_rows.append(
                {
                    "regime": regime_name,
                    "horizon": h,
                    "min": float(grid.min()),
                    "max": float(grid.max()),
                    "under70": int((grid < 0.70).sum()),
                    "over95": int((grid > 0.95).sum()),
                }
            )

    payload = {
        "summary": result["summary"],
        "coverage": {
            "overall_90": result["coverage"]["overall"]["0.9"],
            "global_under70_count": int(under.sum()),
            "global_over95_count": int(over.sum()),
            "under70_by_horizon": {h: int((coverage_by_h[h] < 0.70).sum()) for h in HORIZONS},
            "over95_by_horizon": {h: int((coverage_by_h[h] > 0.95).sum()) for h in HORIZONS},
            "worst_cells_by_horizon": {
                h: top_cells(coverage_by_h[h], largest=False, n=3)
                for h in HORIZONS
            },
            "best_cells_by_horizon": {
                h: top_cells(coverage_by_h[h], largest=True, n=3)
                for h in HORIZONS
            },
            "scalar_temperature_feasibility": (
                "Not sufficient: 385a has both undercoverage and overcoverage in different "
                "cells/horizons, so scalar narrowing worsens undercovered cells and scalar "
                "widening worsens overcovered cells."
            ),
        },
        "distribution": {
            "daily_ks_pass": result["distributional_fidelity"]["ks_test"]["n_pass"],
            "level_ks_pass": result["distributional_fidelity"]["ks_level_test"]["n_pass"],
            "level_ks_median": result["distributional_fidelity"]["ks_level_test"][
                "median_stat"
            ],
            "level_ks_worst": result["distributional_fidelity"]["ks_level_test"][
                "worst_stat"
            ],
            "median_bias_pass": result["distributional_fidelity"]["median_bias"]["n_pass"],
            "bias_magnitude_pass": result["distributional_fidelity"]["median_bias"][
                "n_mag_pass"
            ],
            "largest_level_ks_cells": top_cells(level_ks, largest=True, n=8),
            "largest_median_frac_cells": top_cells(median_frac, largest=True, n=8),
            "smallest_median_frac_cells": top_cells(median_frac, largest=False, n=8),
            "largest_abs_mean_bias_cells": top_cells(np.abs(mean_bias), largest=True, n=8),
            "mean_shift_feasibility": (
                "Weak: bias magnitude already passes 25/25, so level KS is mostly a shape/"
                "occupancy problem rather than a simple mean-shift problem."
            ),
        },
        "regime_layer2": {
            "pass_count": regime["layer2_n_passing"],
            "total": regime["layer2_n_total"],
            "rows": layer2_rows,
            "read": (
                "Layer2 fails through the same cellwise width allocation issue seen globally; "
                "both under70 and over95 cells appear inside regime subsamples."
            ),
        },
        "decision": (
            "385a's residual failures are long-horizon free-running occupancy/width-allocation "
            "failures. The most principled next experiment is not scalar temperature or mean "
            "shift, but a proper-scoring-rule fine-tune on free-running 30-day paths starting "
            "from 385a, with FM retained as an anchor."
        ),
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 390a 385a Residual Failure Audit",
        "",
        f"Score: {payload['summary']['n_pass']}/11",
        f"Failed: {', '.join(payload['summary']['failed_suites'])}",
        "",
        "## Coverage",
        "",
        f"Under70 count: {payload['coverage']['global_under70_count']}",
        f"Over95 count: {payload['coverage']['global_over95_count']}",
        payload["coverage"]["scalar_temperature_feasibility"],
        "",
        "## Distribution",
        "",
        f"Daily KS pass: {payload['distribution']['daily_ks_pass']}/25",
        f"Level KS pass: {payload['distribution']['level_ks_pass']}/25",
        f"Bias magnitude pass: {payload['distribution']['bias_magnitude_pass']}/25",
        payload["distribution"]["mean_shift_feasibility"],
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
