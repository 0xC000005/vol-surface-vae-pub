#!/usr/bin/env python
"""394a: audit the path-energy strength tradeoff around the 392a frontier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RESULTS = {
    "385a_w000_recent_fm": {
        "path": Path("results/block_ar/385a_recent_quantiles_fm_s42/full11.json"),
        "energy_weight": 0.0,
    },
    "392a_w005_energy": {
        "path": Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json"),
        "energy_weight": 0.05,
    },
    "393a_w010_energy": {
        "path": Path("results/block_ar/393a_recent_rollout_energy_w01_s42/full11.json"),
        "energy_weight": 0.1,
    },
    "391a_w020_energy": {
        "path": Path("results/block_ar/391a_recent_rollout_energy_w02_s42/full11.json"),
        "energy_weight": 0.2,
    },
}
HORIZONS = ("1", "7", "14", "30")
OUT_DIR = Path("results/block_ar/394a_energy_tradeoff")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def coverage_stack(result: dict[str, Any]) -> np.ndarray:
    return np.stack([arr(result["coverage"]["per_cell_coverage"][h]) for h in HORIZONS])


def count_by_horizon(result: dict[str, Any], op: str, threshold: float) -> dict[str, int]:
    out: dict[str, int] = {}
    for h in HORIZONS:
        grid = arr(result["coverage"]["per_cell_coverage"][h])
        if op == "lt":
            out[h] = int((grid < threshold).sum())
        elif op == "gt":
            out[h] = int((grid > threshold).sum())
        else:
            raise ValueError(op)
    return out


def layer2_summary(result: dict[str, Any]) -> dict[str, Any]:
    regime = result["regime_coverage"]
    rows: list[dict[str, Any]] = []
    for regime_name, by_horizon in regime["layer2_regime_cell"].items():
        for horizon, entry in by_horizon.items():
            grid = arr(entry["grid"])
            rows.append(
                {
                    "regime": regime_name,
                    "horizon": horizon,
                    "min": float(grid.min()),
                    "max": float(grid.max()),
                    "under70": int((grid < 0.70).sum()),
                    "over95": int((grid > 0.95).sum()),
                }
            )
    return {
        "layer2_n_passing": regime["layer2_n_passing"],
        "layer2_n_total": regime["layer2_n_total"],
        "min_cell": min(row["min"] for row in rows),
        "max_cell": max(row["max"] for row in rows),
        "under70_count": sum(row["under70"] for row in rows),
        "over95_count": sum(row["over95"] for row in rows),
        "rows": rows,
    }


def summarize(name: str, meta: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    dist = result["distributional_fidelity"]
    cov_stack = coverage_stack(result)
    return {
        "name": name,
        "energy_weight": meta["energy_weight"],
        "score": result["summary"]["n_pass"],
        "failed": result["summary"]["failed_suites"],
        "coverage_overall_90": result["coverage"]["overall"]["0.9"],
        "coverage_min": float(cov_stack.min()),
        "coverage_max": float(cov_stack.max()),
        "coverage_under70": int((cov_stack < 0.70).sum()),
        "coverage_over95": int((cov_stack > 0.95).sum()),
        "coverage_under70_by_horizon": count_by_horizon(result, "lt", 0.70),
        "coverage_over95_by_horizon": count_by_horizon(result, "gt", 0.95),
        "conditional_mae_reduction": result["conditionality"]["mae_reduction_pct"],
        "conditional_width_ratio": result["conditionality"]["width_ratio"],
        "turb_calm_ratio": result["conditionality"]["turb_calm_ratio"],
        "daily_ks_pass": dist["ks_test"]["n_pass"],
        "level_ks_pass": dist["ks_level_test"]["n_pass"],
        "level_ks_median": dist["ks_level_test"]["median_stat"],
        "level_ks_worst": dist["ks_level_test"]["worst_stat"],
        "median_bias_pass": dist["median_bias"]["n_pass"],
        "bias_magnitude_pass": dist["median_bias"]["n_mag_pass"],
        "cointegration_ratio": result["cointegration"]["gen_gt_ratio"],
        "cointegration_worst": result["cointegration"]["worst_cell_ratio"],
        "regime_layer2": layer2_summary(result),
        "max_jump_ks": result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "score",
        "coverage_overall_90",
        "coverage_under70",
        "coverage_over95",
        "conditional_mae_reduction",
        "level_ks_pass",
        "level_ks_median",
        "cointegration_worst",
    )
    return {field: right[field] - left[field] for field in fields}


def top_level_ks_deltas(
    before: dict[str, Any],
    after: dict[str, Any],
    n: int = 8,
) -> list[dict[str, Any]]:
    before_grid = arr(before["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    after_grid = arr(after["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    delta = after_grid - before_grid
    items: list[tuple[float, int, int]] = []
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            items.append((float(delta[i, j]), i, j))
    items.sort(key=lambda item: abs(item[0]), reverse=True)
    return [
        {
            "cell": [i, j],
            "delta": value,
            "before": float(before_grid[i, j]),
            "after": float(after_grid[i, j]),
        }
        for value, i, j in items[:n]
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {name: load(meta["path"]) for name, meta in RESULTS.items()}
    summaries = {
        name: summarize(name, RESULTS[name], result)
        for name, result in loaded.items()
    }

    ordered_names = sorted(summaries, key=lambda key: summaries[key]["energy_weight"])
    best_name = max(ordered_names, key=lambda key: summaries[key]["score"])
    frontier = summaries["392a_w005_energy"]
    intermediate = summaries["393a_w010_energy"]

    payload = {
        "summaries": summaries,
        "ordered_names": ordered_names,
        "best_by_score": best_name,
        "deltas": {
            "392a_minus_385a": deltas(summaries["385a_w000_recent_fm"], frontier),
            "393a_minus_392a": deltas(frontier, intermediate),
            "391a_minus_392a": deltas(frontier, summaries["391a_w020_energy"]),
        },
        "largest_level_ks_deltas_393a_minus_392a": top_level_ks_deltas(
            loaded["392a_w005_energy"],
            loaded["393a_w010_energy"],
        ),
        "mechanism_read": (
            "Path-energy strength improves level occupancy up to the intermediate run, "
            "but the gain is not free: 0.1 lowers overall coverage and conditional MAE "
            "reduction enough to lose an official suite. The clean issue is estimator/"
            "objective geometry, not architecture capacity."
        ),
        "decision": (
            "Keep 392a as the active 8/11 frontier. Do not continue scalar energy-weight "
            "sweeps. The most principled next experiment is to keep the same proper "
            "free-running energy score at weight 0.05 but reduce estimator noise with "
            "more training samples per condition before adding any new loss or model knob."
        ),
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 394a Path-Energy Tradeoff Audit",
        "",
        "| model | w | score | failed | cov90 | under70 | over95 | cond MAE | level KS | coint worst | L2 pass |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ordered_names:
        s = summaries[name]
        lines.append(
            f"| {name} | {s['energy_weight']:.2f} | {s['score']}/11 | "
            f"{', '.join(s['failed'])} | {s['coverage_overall_90']:.3f} | "
            f"{s['coverage_under70']} | {s['coverage_over95']} | "
            f"{s['conditional_mae_reduction']:.2f}% | {s['level_ks_pass']}/25 | "
            f"{s['cointegration_worst']:.3f} | "
            f"{s['regime_layer2']['layer2_n_passing']}/{s['regime_layer2']['layer2_n_total']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Read",
            "",
            payload["mechanism_read"],
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
