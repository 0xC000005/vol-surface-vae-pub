#!/usr/bin/env python
"""383a: compare recent-adaptation variants to isolate stable failure geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_RESULTS = {
    "340c_revised": Path("results/block_ar/340c_v0_s42_revised/full11.json"),
    "377a_full_adapt": Path("results/block_ar/377a_recent_fm_s42/full11.json"),
    "381_conditioning": Path("results/block_ar/381a_recent_fm_conditioning_s42/full11.json"),
    "381_conditioning_memory": Path(
        "results/block_ar/381a_recent_fm_conditioning_memory_proj_s42/full11.json"
    ),
    "382_anchor1": Path("results/block_ar/382a_recent_fm_anchor1p0_s42/full11.json"),
}

HORIZONS = ("1", "7", "14", "30")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def grid(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def coverage_geometry(result: dict[str, Any]) -> dict[str, Any]:
    per_h = {
        h: grid(result["coverage"]["per_cell_coverage"][h])
        for h in HORIZONS
    }
    stacked = np.stack([per_h[h] for h in HORIZONS], axis=0)
    under_mask = stacked < 0.70
    over_mask = stacked > 0.95
    return {
        "overall_90": result["coverage"]["overall"]["0.9"],
        "min_cell_horizon_90": float(stacked.min()),
        "max_cell_horizon_90": float(stacked.max()),
        "under_70_count": int(under_mask.sum()),
        "over_95_count": int(over_mask.sum()),
        "under_by_cell": under_mask.sum(axis=0).astype(int).tolist(),
        "over_by_cell": over_mask.sum(axis=0).astype(int).tolist(),
    }


def summarize_model(name: str, result: dict[str, Any]) -> dict[str, Any]:
    dist = result["distributional_fidelity"]
    cond = result["conditionality"]
    regime = result["regime_coverage"]
    coin = result["cointegration"]
    return {
        "name": name,
        "score": result["summary"]["n_pass"],
        "failed": result["summary"]["failed_suites"],
        "coverage": coverage_geometry(result),
        "conditionality": {
            "mae_reduction_pct": cond["mae_reduction_pct"],
            "h14_mae_reduction_pct": cond["per_horizon_conditionality"]["14"][
                "mae_reduction_pct"
            ],
            "h30_mae_reduction_pct": cond["per_horizon_conditionality"]["30"][
                "mae_reduction_pct"
            ],
            "width_ratio": cond["width_ratio"],
            "turb_calm_ratio": cond["turb_calm_ratio"],
        },
        "regime": {
            "layer2": f"{regime['layer2_n_passing']}/{regime['layer2_n_total']}",
            "h30_turb_calm_width": regime["width_turb_calm"]["30"][
                "width_turb_calm_ratio"
            ],
            "h30_width_vov_spearman": regime["width_vs_vov"]["30"]["spearman_rho"],
            "layer3_catastrophic_rate": regime["layer3_catastrophic_rate"],
        },
        "distribution": {
            "daily_ks_pass": dist["ks_test"]["n_pass"],
            "level_ks_pass": dist["ks_level_test"]["n_pass"],
            "level_ks_median": dist["ks_level_test"]["median_stat"],
            "level_ks_worst": dist["ks_level_test"]["worst_stat"],
            "median_bias_pass": dist["median_bias"]["n_pass"],
            "bias_magnitude_pass": dist["median_bias"]["n_mag_pass"],
        },
        "cointegration": {
            "ratio": coin["gen_gt_ratio"],
            "worst_cell_ratio": coin["worst_cell_ratio"],
        },
    }


def stable_cell_maps(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    level_fail_counts = np.zeros((5, 5), dtype=int)
    under_counts = np.zeros((5, 5), dtype=int)
    over_counts = np.zeros((5, 5), dtype=int)

    for result in results.values():
        level_ks = grid(result["distributional_fidelity"]["ks_level_test"]["ks_grid"])
        level_fail_counts += level_ks >= 0.15

        cov = coverage_geometry(result)
        under_counts += np.asarray(cov["under_by_cell"], dtype=int)
        over_counts += np.asarray(cov["over_by_cell"], dtype=int)

    return {
        "level_ks_fail_count_by_cell": level_fail_counts.tolist(),
        "coverage_under70_count_by_cell": under_counts.tolist(),
        "coverage_over95_count_by_cell": over_counts.tolist(),
        "top_level_ks_cells": top_cells(level_fail_counts, max_items=8),
        "top_undercovered_cells": top_cells(under_counts, max_items=8),
        "top_overcovered_cells": top_cells(over_counts, max_items=8),
    }


def top_cells(arr: np.ndarray, max_items: int) -> list[dict[str, Any]]:
    items: list[tuple[int, int, int]] = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            items.append((int(arr[i, j]), i, j))
    items.sort(reverse=True)
    return [
        {"cell": [i, j], "count": count}
        for count, i, j in items[:max_items]
        if count > 0
    ]


def format_grid(arr: list[list[int]]) -> str:
    return "\n".join("  " + " ".join(f"{v:2d}" for v in row) for row in arr)


def write_markdown(
    output_path: Path,
    model_summaries: list[dict[str, Any]],
    cells: dict[str, Any],
) -> None:
    lines: list[str] = [
        "# 383a Adaptation-Family Failure Geometry",
        "",
        "## Model Comparison",
        "",
        "| model | score | failed | cov90 | under70 | over95 | cond MAE red | level KS | median bias | regime L2 | coin worst |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for s in model_summaries:
        lines.append(
            "| {name} | {score}/11 | {failed} | {cov90:.3f} | {under} | {over} | "
            "{cond:.2f}% | {level}/25 | {median}/25 | {regime} | {coin:.3f} |".format(
                name=s["name"],
                score=s["score"],
                failed=", ".join(s["failed"]),
                cov90=s["coverage"]["overall_90"],
                under=s["coverage"]["under_70_count"],
                over=s["coverage"]["over_95_count"],
                cond=s["conditionality"]["mae_reduction_pct"],
                level=s["distribution"]["level_ks_pass"],
                median=s["distribution"]["median_bias_pass"],
                regime=s["regime"]["layer2"],
                coin=s["cointegration"]["worst_cell_ratio"],
            )
        )

    lines.extend(
        [
            "",
            "## Stable Cell Geometry",
            "",
            "Level-KS fail count by cell across compared models:",
            "",
            "```text",
            format_grid(cells["level_ks_fail_count_by_cell"]),
            "```",
            "",
            "Coverage under-70 count by cell across models and horizons:",
            "",
            "```text",
            format_grid(cells["coverage_under70_count_by_cell"]),
            "```",
            "",
            "Coverage over-95 count by cell across models and horizons:",
            "",
            "```text",
            format_grid(cells["coverage_over95_count_by_cell"]),
            "```",
            "",
            "## Interpretation",
            "",
            "- Daily-change KS is already stable in the adaptation family, usually `24/25` or better. The hard distribution failure is level occupancy, not one-day move realism.",
            "- The coverage failure is not a scalar-width problem: undercoverage concentrates around cells such as `(0,3)` and `(1,3)`, while overcoverage concentrates in different wing/corner cells. Scalar temperature, sample mixtures, and global anchors cannot solve that geometry.",
            "- Conditionality is fragile and mostly front-loaded: full adaptation barely passes the aggregate gate, while anchored and frozen variants lose h14/h30 MAE reduction first. This argues against further freezing/anchoring as the main path.",
            "- Regime layer2 remains `0/8` across the family even when layer1 passes. With 39 calm and 39 turbulent windows, the per-regime per-cell gate has high binomial noise, but repeated 100% overcoverage and severe specific-cell undercoverage indicate structural width allocation errors too.",
            "",
            "## Decision",
            "",
            "The next experiment should target conditional per-cell uncertainty geometry inside the model rather than adding post-hoc scalar calibration or more adaptation constraints. The cleanest candidate is a learned conditional diagonal noise-scale/readout in normal-score transition space, trained with the same FM objective and sampled by scaling the base noise before integration.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = Path("results/block_ar/383a_adaptation_family_failure_geometry")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {name: load_json(path) for name, path in MODEL_RESULTS.items()}
    model_summaries = [summarize_model(name, result) for name, result in results.items()]
    cells = stable_cell_maps(results)

    payload = {
        "models": model_summaries,
        "stable_cell_maps": cells,
        "decision": (
            "Target conditional per-cell uncertainty geometry inside the model; "
            "avoid further scalar calibration, sample mixtures, or adaptation constraints."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(output_dir / "summary.md", model_summaries, cells)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
