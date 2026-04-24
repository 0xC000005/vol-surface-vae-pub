#!/usr/bin/env python
"""408a: decide whether interval calibration remains a clean primary route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RUNS = {
    "392a_base": Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json"),
    "405a_target90": Path("results/block_ar/405a_interval_scale_regime_392a/full11.json"),
    "407a_deadband": Path("results/block_ar/407a_interval_deadband_regime_392a/full11.json"),
}
OUT_DIR = Path("results/block_ar/408a_calibration_cap")
HORIZONS = ("1", "7", "14", "30")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_values(result: dict[str, Any]) -> np.ndarray:
    values = []
    for h in HORIZONS:
        values.extend(np.asarray(result["coverage"]["per_cell_coverage"][h], dtype=float).ravel())
    return np.asarray(values, dtype=float)


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    cov = coverage_values(result)
    calibration = result.get("config", {}).get("calibration", {})
    return {
        "score": int(result["summary"]["n_pass"]),
        "failed": list(result["summary"]["failed_suites"]),
        "cov90": float(result["coverage"]["overall"]["0.9"]),
        "under70": int((cov < 0.70).sum()),
        "over95": int((cov > 0.95).sum()),
        "coverage_gate_failures": int((cov < 0.70).sum() + (cov > 0.95).sum()),
        "conditional_mae_reduction_pct": float(result["conditionality"]["mae_reduction_pct"]),
        "very_small_move_ratio": float(
            result["time_series"]["move_size_profile"]["very_small_moves"]["ratio"]
        ),
        "cointegration_worst_cell_ratio": float(result["cointegration"]["worst_cell_ratio"]),
        "regime_layer2_passing": int(result["regime_coverage"]["layer2_n_passing"]),
        "level_ks_passing": int(result["distributional_fidelity"]["ks_level_test"]["n_pass"]),
        "level_ks_gap_to_gate": int(
            max(0, 15 - result["distributional_fidelity"]["ks_level_test"]["n_pass"])
        ),
        "median_fraction_passing": int(result["distributional_fidelity"]["median_bias"]["n_pass"]),
        "bias_magnitude_passing": int(result["distributional_fidelity"]["median_bias"]["n_mag_pass"]),
        "pathwise_max_jump_ks": float(
            result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]
        ),
        "scale_min": calibration.get("scale_table_min"),
        "scale_median": calibration.get("scale_median"),
        "scale_max": calibration.get("scale_max"),
    }


def markdown_table(rows: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| run | score | failed | cov90 | cov cells fail | cond MAE | small moves | level KS | regime L2 | coint worst | scale min/med/max |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in rows.items():
        scale = "n/a"
        if row["scale_min"] is not None:
            scale = f"{row['scale_min']:.2f}/{row['scale_median']:.2f}/{row['scale_max']:.2f}"
        lines.append(
            f"| {name} | {row['score']}/11 | {', '.join(row['failed'])} | "
            f"{row['cov90']:.3f} | {row['coverage_gate_failures']} | "
            f"{row['conditional_mae_reduction_pct']:.2f}% | {row['very_small_move_ratio']:.3f} | "
            f"{row['level_ks_passing']}/25 | {row['regime_layer2_passing']}/8 | "
            f"{row['cointegration_worst_cell_ratio']:.3f} | {scale} |"
        )
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {name: summarize(load(path)) for name, path in RUNS.items()}
    base = rows["392a_base"]
    deadband = rows["407a_deadband"]
    payload = {
        "runs": rows,
        "mechanism_read": (
            "The interval calibration branch is directionally meaningful but capped as a "
            "primary route. Deadband scaling improves the coverage edge count, level KS, "
            "and regime layer2 relative to the base while preserving time-series and "
            "cointegration, but the same width-only actuator lowers conditional MAE below "
            "the gate and still leaves six coverage violations, seven regime layer2 "
            "failures, and a three-cell level-KS deficit."
        ),
        "cap_evidence": {
            "does_deadband_beat_base_score": deadband["score"] > base["score"],
            "deadband_improves_level_ks_cells": deadband["level_ks_passing"]
            - base["level_ks_passing"],
            "deadband_improves_coverage_gate_failures": base["coverage_gate_failures"]
            - deadband["coverage_gate_failures"],
            "deadband_conditionality_margin_to_gate": deadband[
                "conditional_mae_reduction_pct"
            ]
            - 5.0,
            "deadband_level_ks_gap_to_gate": deadband["level_ks_gap_to_gate"],
            "deadband_regime_layer2_gap_to_gate": 8 - deadband["regime_layer2_passing"],
        },
        "decision": (
            "Close interval calibration as the primary autoresearch path. It can remain "
            "a reportable policy-calibration ablation, but continuing it would require "
            "more cell/regime-specific knobs. The next principled route should return "
            "to the learned base law and attack long-horizon level occupancy/regime "
            "allocation during training, not by post-hoc width manipulation."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 408a Calibration Cap Analysis",
        "",
        *markdown_table(rows),
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
