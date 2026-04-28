#!/usr/bin/env python
"""710a: attribute regime under-inclusion to authenticity/dependence failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402


def _grid(data: Any, default: float = np.nan) -> np.ndarray:
    if data is None:
        return np.full((0, 0), default, dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def _safe_cell(grid: np.ndarray, row: int, col: int, default: float = np.nan) -> float:
    if grid.size == 0 or row >= grid.shape[0] or col >= grid.shape[1]:
        return float(default)
    return float(grid[row, col])


def analyze_regime_authenticity_failures(
    result: dict[str, Any],
    min_regime_cell: float = 0.70,
    top_k: int = 12,
) -> dict[str, Any]:
    regime_layer = result.get("regime_coverage", {}).get("layer2_regime_cell", {})
    dist = result.get("distributional_fidelity", {})
    level_ks = dist.get("ks_level_test", {})
    level_ks_grid = _grid(level_ks.get("ks_grid"))
    level_ks_gate = float(level_ks.get("ks_gate", 0.15))
    cell_mae_grid = _grid(dist.get("cell_mae", {}).get("mae_grid"))
    if cell_mae_grid.size and float(np.nanmax(cell_mae_grid)) <= 1.0:
        cell_mae_grid = cell_mae_grid * 100.0
    cointegration_grid = _grid(result.get("cointegration", {}).get("per_cell_ratio_grid"))

    failures: list[dict[str, Any]] = []
    undercovered_cells: set[tuple[int, int]] = set()
    for regime_name, horizons in regime_layer.items():
        for horizon, payload in horizons.items():
            cov_grid = _grid(payload.get("grid"))
            for row in range(cov_grid.shape[0]):
                for col in range(cov_grid.shape[1]):
                    coverage = float(cov_grid[row, col])
                    if coverage >= min_regime_cell:
                        continue
                    undercovered_cells.add((row, col))
                    ks_stat = _safe_cell(level_ks_grid, row, col)
                    coint_ratio = _safe_cell(cointegration_grid, row, col)
                    failures.append({
                        "regime": str(regime_name),
                        "horizon": int(horizon),
                        "cell": [int(row), int(col)],
                        "coverage": coverage,
                        "shortfall": float(min_regime_cell - coverage),
                        "level_ks": ks_stat,
                        "level_ks_fail": bool(np.isfinite(ks_stat) and ks_stat > level_ks_gate),
                        "cointegration_ratio": coint_ratio,
                        "cointegration_fail": bool(np.isfinite(coint_ratio) and coint_ratio < 0.25),
                        "cell_mae_iv_points": _safe_cell(cell_mae_grid, row, col),
                    })

    failures.sort(key=lambda item: item["shortfall"], reverse=True)
    cells = sorted(undercovered_cells)
    level_fails = 0
    coint_fails = 0
    for row, col in cells:
        ks_stat = _safe_cell(level_ks_grid, row, col)
        coint_ratio = _safe_cell(cointegration_grid, row, col)
        level_fails += int(np.isfinite(ks_stat) and ks_stat > level_ks_gate)
        coint_fails += int(np.isfinite(coint_ratio) and coint_ratio < 0.25)

    n_cells = len(cells)
    return {
        "min_regime_cell": float(min_regime_cell),
        "n_undercovered_slices": int(len(failures)),
        "n_unique_undercovered_cells": int(n_cells),
        "worst_undercovered_slices": failures[: int(top_k)],
        "undercovered_cell_overlap": {
            "cells": [[int(r), int(c)] for r, c in cells],
            "level_ks_fail_count": int(level_fails),
            "level_ks_fail_rate": float(level_fails / n_cells) if n_cells else 0.0,
            "cointegration_fail_count": int(coint_fails),
            "cointegration_fail_rate": float(coint_fails / n_cells) if n_cells else 0.0,
        },
    }


def write_markdown(path: Path, report: dict[str, Any], source: str) -> None:
    lines = [
        "# 710a Regime / Authenticity Failure Attribution",
        "",
        f"- source: `{source}`",
        f"- undercovered slices: `{report['n_undercovered_slices']}`",
        f"- unique undercovered cells: `{report['n_unique_undercovered_cells']}`",
        f"- level-KS fail overlap: `{report['undercovered_cell_overlap']['level_ks_fail_rate']:.3f}`",
        f"- cointegration fail overlap: `{report['undercovered_cell_overlap']['cointegration_fail_rate']:.3f}`",
        "",
        "## Worst Undercovered Slices",
        "",
        "| Regime | Horizon | Cell | Coverage | Shortfall | Level KS | Coint ratio | MAE pts |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["worst_undercovered_slices"]:
        lines.append(
            f"| `{row['regime']}` | `{row['horizon']}` | `{row['cell']}` | "
            f"`{row['coverage']:.3f}` | `{row['shortfall']:.3f}` | "
            f"`{row['level_ks']:.3f}` | `{row['cointegration_ratio']:.3f}` | "
            f"`{row['cell_mae_iv_points']:.2f}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--min_regime_cell", type=float, default=0.70)
    parser.add_argument("--top_k", type=int, default=12)
    args = parser.parse_args()

    result = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    report = analyze_regime_authenticity_failures(
        result,
        min_regime_cell=args.min_regime_cell,
        top_k=args.top_k,
    )
    report["source"] = args.input_json

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report, args.input_json)
    print(json.dumps(make_serializable({
        "undercovered_slices": report["n_undercovered_slices"],
        "unique_cells": report["n_unique_undercovered_cells"],
        "level_ks_overlap": report["undercovered_cell_overlap"]["level_ks_fail_rate"],
        "cointegration_overlap": report["undercovered_cell_overlap"]["cointegration_fail_rate"],
    }), indent=2))


if __name__ == "__main__":
    main()
