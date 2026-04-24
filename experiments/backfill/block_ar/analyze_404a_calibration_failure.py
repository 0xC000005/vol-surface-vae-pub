#!/usr/bin/env python
"""404a: analyze why regime marginal quantile calibration regressed 392a."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


BASE = Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json")
CAL = Path("results/block_ar/403a_regime_quantile_calibrated_392a/full11.json")
OUT_DIR = Path("results/block_ar/404a_calibration_failure")
HORIZONS = ("1", "7", "14", "30")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def coverage_stack(result: dict[str, Any]) -> np.ndarray:
    return np.stack([arr(result["coverage"]["per_cell_coverage"][h]) for h in HORIZONS])


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    cov = coverage_stack(result)
    dist = result["distributional_fidelity"]
    ts = result["time_series"]
    return {
        "score": result["summary"]["n_pass"],
        "failed": result["summary"]["failed_suites"],
        "coverage90": result["coverage"]["overall"]["0.9"],
        "under70": int((cov < 0.70).sum()),
        "over95": int((cov > 0.95).sum()),
        "conditional_mae": result["conditionality"]["mae_reduction_pct"],
        "turb_calm": result["conditionality"]["turb_calm_ratio"],
        "time_series_pass": bool(ts["overall_pass"]),
        "kurtosis_ratio": ts["kurtosis"]["kurtosis_ratio"],
        "very_small_move_ratio": ts["move_size_profile"]["very_small_moves"]["ratio"],
        "cointegration_pass": bool(result["cointegration"]["overall_pass"]),
        "cointegration_worst": result["cointegration"]["worst_cell_ratio"],
        "daily_ks": dist["ks_test"]["n_pass"],
        "level_ks": dist["ks_level_test"]["n_pass"],
        "level_ks_median": dist["ks_level_test"]["median_stat"],
        "median_bias": dist["median_bias"]["n_pass"],
        "bias_magnitude": dist["median_bias"]["n_mag_pass"],
        "regime_l2": result["regime_coverage"]["layer2_n_passing"],
        "pathwise_ks": result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def top_level_deltas(base: dict[str, Any], cal: dict[str, Any], n: int = 8) -> list[dict[str, Any]]:
    before = arr(base["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    after = arr(cal["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    delta = after - before
    items: list[tuple[float, int, int]] = []
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            items.append((float(delta[i, j]), i, j))
    items.sort(key=lambda item: abs(item[0]), reverse=True)
    return [
        {
            "cell": [i, j],
            "delta": value,
            "base": float(before[i, j]),
            "calibrated": float(after[i, j]),
        }
        for value, i, j in items[:n]
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load(BASE)
    cal = load(CAL)
    base_summary = summarize(base)
    cal_summary = summarize(cal)
    deltas = {
        key: cal_summary[key] - base_summary[key]
        for key in base_summary
        if isinstance(base_summary[key], (int, float))
    }
    payload = {
        "base_392a": base_summary,
        "calibrated_403a": cal_summary,
        "deltas_403a_minus_392a": deltas,
        "largest_level_ks_deltas": top_level_deltas(base, cal),
        "mechanism_read": (
            "The monotone marginal quantile map is too broad. It helps the worst "
            "cointegration cell, but its horizon/cell nonlinear maps alter increments "
            "and validation level occupancy. The result is lower conditionality, failed "
            "very-small-move profile, and level KS collapse from 10/25 to 3/25."
        ),
        "decision": (
            "Close marginal quantile-map calibration. If calibrated-system work continues, "
            "the next falsifier should be narrower: interval-width scaling around each "
            "sample cloud's own median, so the central path and rank/level location are "
            "less disturbed while coverage/regime intervals are adjusted."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 404a Calibration Failure Analysis",
        "",
        "| system | score | failed | cov90 | under70 | over95 | cond MAE | very-small moves | level KS | coint worst |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 392a base | {base_summary['score']}/11 | {', '.join(base_summary['failed'])} | "
            f"{base_summary['coverage90']:.3f} | {base_summary['under70']} | "
            f"{base_summary['over95']} | {base_summary['conditional_mae']:.2f}% | "
            f"{base_summary['very_small_move_ratio']:.3f} | {base_summary['level_ks']}/25 | "
            f"{base_summary['cointegration_worst']:.3f} |"
        ),
        (
            f"| 403a quantile-cal | {cal_summary['score']}/11 | {', '.join(cal_summary['failed'])} | "
            f"{cal_summary['coverage90']:.3f} | {cal_summary['under70']} | "
            f"{cal_summary['over95']} | {cal_summary['conditional_mae']:.2f}% | "
            f"{cal_summary['very_small_move_ratio']:.3f} | {cal_summary['level_ks']}/25 | "
            f"{cal_summary['cointegration_worst']:.3f} |"
        ),
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
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
