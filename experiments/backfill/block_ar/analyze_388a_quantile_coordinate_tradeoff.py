#!/usr/bin/env python
"""388a: diagnose the recent-vs-blended quantile coordinate tradeoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RESULTS = {
    "377a_checkpoint_quantiles": Path("results/block_ar/377a_recent_fm_s42/full11.json"),
    "385a_recent_quantiles": Path("results/block_ar/385a_recent_quantiles_fm_s42/full11.json"),
    "386a_recent_quantiles_w882": Path(
        "results/block_ar/386a_recent_quantiles_w882_fm_s42/full11.json"
    ),
    "387a_blend05": Path("results/block_ar/387a_quantile_blend05_fm_s42/full11.json"),
}

HORIZONS = ("1", "7", "14", "30")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def coverage_stack(result: dict[str, Any]) -> np.ndarray:
    return np.stack([arr(result["coverage"]["per_cell_coverage"][h]) for h in HORIZONS])


def coverage_counts(result: dict[str, Any]) -> dict[str, Any]:
    cov = coverage_stack(result)
    return {
        "under70": int((cov < 0.70).sum()),
        "over95": int((cov > 0.95).sum()),
        "min": float(cov.min()),
        "max": float(cov.max()),
        "horizon_under70": {
            h: int((arr(result["coverage"]["per_cell_coverage"][h]) < 0.70).sum())
            for h in HORIZONS
        },
        "horizon_over95": {
            h: int((arr(result["coverage"]["per_cell_coverage"][h]) > 0.95).sum())
            for h in HORIZONS
        },
    }


def top_delta_cells(before: np.ndarray, after: np.ndarray, n: int = 8) -> list[dict[str, Any]]:
    delta = after - before
    items: list[tuple[float, int, int]] = []
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            items.append((float(delta[i, j]), i, j))
    items.sort(reverse=True, key=lambda t: abs(t[0]))
    return [
        {"cell": [i, j], "delta": value, "before": float(before[i, j]), "after": float(after[i, j])}
        for value, i, j in items[:n]
    ]


def summarize(name: str, result: dict[str, Any]) -> dict[str, Any]:
    dist = result["distributional_fidelity"]
    coin = result["cointegration"]
    cond = result["conditionality"]
    return {
        "score": result["summary"]["n_pass"],
        "failed": result["summary"]["failed_suites"],
        "coverage": coverage_counts(result),
        "conditional_mae_reduction": cond["mae_reduction_pct"],
        "level_ks_pass": dist["ks_level_test"]["n_pass"],
        "level_ks_median": dist["ks_level_test"]["median_stat"],
        "level_ks_worst": dist["ks_level_test"]["worst_stat"],
        "median_bias_pass": dist["median_bias"]["n_pass"],
        "bias_magnitude_pass": dist["median_bias"]["n_mag_pass"],
        "cointegration_ratio": coin["gen_gt_ratio"],
        "cointegration_worst": coin["worst_cell_ratio"],
    }


def main() -> None:
    out_dir = Path("results/block_ar/388a_quantile_coordinate_tradeoff")
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = {name: load(path) for name, path in RESULTS.items()}
    summaries = {name: summarize(name, result) for name, result in loaded.items()}

    recent = loaded["385a_recent_quantiles"]
    blend = loaded["387a_blend05"]
    recent_level = arr(recent["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    blend_level = arr(blend["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    recent_bias = arr(recent["distributional_fidelity"]["median_bias"]["above_frac"])
    blend_bias = arr(blend["distributional_fidelity"]["median_bias"]["above_frac"])
    recent_cov = coverage_stack(recent)
    blend_cov = coverage_stack(blend)

    payload = {
        "summaries": summaries,
        "blend_minus_recent": {
            "coverage_under70_delta": summaries["387a_blend05"]["coverage"]["under70"]
            - summaries["385a_recent_quantiles"]["coverage"]["under70"],
            "coverage_over95_delta": summaries["387a_blend05"]["coverage"]["over95"]
            - summaries["385a_recent_quantiles"]["coverage"]["over95"],
            "level_ks_pass_delta": summaries["387a_blend05"]["level_ks_pass"]
            - summaries["385a_recent_quantiles"]["level_ks_pass"],
            "median_bias_pass_delta": summaries["387a_blend05"]["median_bias_pass"]
            - summaries["385a_recent_quantiles"]["median_bias_pass"],
            "cointegration_worst_delta": summaries["387a_blend05"]["cointegration_worst"]
            - summaries["385a_recent_quantiles"]["cointegration_worst"],
        },
        "largest_level_ks_deltas_blend_minus_recent": top_delta_cells(
            recent_level,
            blend_level,
        ),
        "largest_median_above_frac_deltas_blend_minus_recent": top_delta_cells(
            recent_bias,
            blend_bias,
        ),
        "coverage_under_mask_recent": (recent_cov < 0.70).sum(axis=0).astype(int).tolist(),
        "coverage_under_mask_blend": (blend_cov < 0.70).sum(axis=0).astype(int).tolist(),
        "coverage_over_mask_recent": (recent_cov > 0.95).sum(axis=0).astype(int).tolist(),
        "coverage_over_mask_blend": (blend_cov > 0.95).sum(axis=0).astype(int).tolist(),
        "decision": (
            "385a remains the best active coordinate. 387a removes global undercoverage "
            "but worsens level KS broadly and loses a cointegration cell, so more blend "
            "weights are not justified without a specific recent-heavy falsifier."
        ),
    }

    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 388a Quantile Coordinate Tradeoff",
        "",
        "| model | score | failed | under70 | over95 | cond MAE | level KS | median bias | coin worst |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s['score']}/11 | {', '.join(s['failed'])} | "
            f"{s['coverage']['under70']} | {s['coverage']['over95']} | "
            f"{s['conditional_mae_reduction']:.2f}% | {s['level_ks_pass']}/25 | "
            f"{s['median_bias_pass']}/25 | {s['cointegration_worst']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
