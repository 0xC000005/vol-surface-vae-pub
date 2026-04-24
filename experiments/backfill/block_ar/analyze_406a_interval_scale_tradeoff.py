#!/usr/bin/env python
"""406a: analyze interval-scale calibration tradeoffs and next policy objective."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


BASE = Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json")
INTERVAL = Path("results/block_ar/405a_interval_scale_regime_392a/full11.json")
OUT_DIR = Path("results/block_ar/406a_interval_scale_tradeoff")
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
    return {
        "score": result["summary"]["n_pass"],
        "failed": result["summary"]["failed_suites"],
        "coverage90": result["coverage"]["overall"]["0.9"],
        "under70": int((cov < 0.70).sum()),
        "over95": int((cov > 0.95).sum()),
        "conditional_mae": result["conditionality"]["mae_reduction_pct"],
        "very_small_move_ratio": result["time_series"]["move_size_profile"]["very_small_moves"]["ratio"],
        "cointegration_worst": result["cointegration"]["worst_cell_ratio"],
        "level_ks": result["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "level_ks_median": result["distributional_fidelity"]["ks_level_test"]["median_stat"],
        "regime_l2": result["regime_coverage"]["layer2_n_passing"],
        "pathwise_ks": result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
    }


def alpha_limit(base_value: float, full_value: float, gate: float, higher_is_better: bool) -> float | None:
    delta = full_value - base_value
    if abs(delta) < 1e-12:
        return None
    if higher_is_better:
        if base_value < gate:
            return 0.0
        if full_value >= gate:
            return 1.0
        return max(0.0, min(1.0, (gate - base_value) / delta))
    if base_value > gate:
        return 0.0
    if full_value <= gate:
        return 1.0
    return max(0.0, min(1.0, (gate - base_value) / delta))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = summarize(load(BASE))
    interval = summarize(load(INTERVAL))
    deltas = {
        key: interval[key] - base[key]
        for key in base
        if isinstance(base[key], (int, float))
    }
    payload = {
        "base_392a": base,
        "interval_405a": interval,
        "deltas_405a_minus_392a": deltas,
        "linear_alpha_limits_from_392a_to_405a": {
            "conditionality_gate_gt_5": alpha_limit(
                base["conditional_mae"],
                interval["conditional_mae"],
                5.0,
                higher_is_better=True,
            ),
            "very_small_move_gate_gt_0p9": alpha_limit(
                base["very_small_move_ratio"],
                interval["very_small_move_ratio"],
                0.9,
                higher_is_better=True,
            ),
        },
        "mechanism_read": (
            "Full target-90 interval scaling proves width is an effective actuator: "
            "undercoverage disappears, regime layer2 improves from 0/8 to 1/8, "
            "cointegration remains valid, and level KS improves slightly. But because "
            "the objective targets 90% everywhere, it over-widens many already-safe "
            "cells and breaks conditionality plus the small-move profile."
        ),
        "decision": (
            "Do not tune alpha as the primary next move. The principled calibrated "
            "risk objective is a deadband policy: leave cells unchanged if calibration "
            "coverage is already inside the evaluator/risk band and use the smallest "
            "scale change needed to enter the band. This is narrower than target-90 "
            "scaling and should preserve base dynamics better."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 406a Interval-Scale Tradeoff",
        "",
        "| system | score | failed | cov90 | under70 | over95 | cond MAE | very-small moves | level KS | regime L2 | coint worst |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 392a base | {base['score']}/11 | {', '.join(base['failed'])} | "
            f"{base['coverage90']:.3f} | {base['under70']} | {base['over95']} | "
            f"{base['conditional_mae']:.2f}% | {base['very_small_move_ratio']:.3f} | "
            f"{base['level_ks']}/25 | {base['regime_l2']}/8 | {base['cointegration_worst']:.3f} |"
        ),
        (
            f"| 405a target90 scale | {interval['score']}/11 | {', '.join(interval['failed'])} | "
            f"{interval['coverage90']:.3f} | {interval['under70']} | {interval['over95']} | "
            f"{interval['conditional_mae']:.2f}% | {interval['very_small_move_ratio']:.3f} | "
            f"{interval['level_ks']}/25 | {interval['regime_l2']}/8 | {interval['cointegration_worst']:.3f} |"
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
