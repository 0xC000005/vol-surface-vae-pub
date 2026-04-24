#!/usr/bin/env python
"""411a: audit whether simple proper-score fine-tunes remain viable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RUNS = {
    "385a_recent_fm": Path("results/block_ar/385a_recent_quantiles_fm_s42/full11.json"),
    "392a_energy_w005": Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json"),
    "393a_energy_w01": Path("results/block_ar/393a_recent_rollout_energy_w01_s42/full11.json"),
    "410a_marginal_crps_w005": Path(
        "results/block_ar/410a_recent_rollout_marginal_crps_w005_s42/full11.json"
    ),
}
OUT_DIR = Path("results/block_ar/411a_proper_score_cap")
HORIZONS = ("1", "7", "14", "30")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_counts(result: dict[str, Any]) -> tuple[int, int]:
    values = []
    for h in HORIZONS:
        values.extend(np.asarray(result["coverage"]["per_cell_coverage"][h], dtype=float).ravel())
    arr = np.asarray(values)
    return int((arr < 0.70).sum()), int((arr > 0.95).sum())


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    under70, over95 = coverage_counts(result)
    return {
        "score": int(result["summary"]["n_pass"]),
        "failed": list(result["summary"]["failed_suites"]),
        "cov90": float(result["coverage"]["overall"]["0.9"]),
        "under70": under70,
        "over95": over95,
        "calibration_error": float(result["coverage"]["calibration_error"]),
        "conditional_mae": float(result["conditionality"]["mae_reduction_pct"]),
        "very_small_move_ratio": float(
            result["time_series"]["move_size_profile"]["very_small_moves"]["ratio"]
        ),
        "cointegration_worst": float(result["cointegration"]["worst_cell_ratio"]),
        "regime_layer2": int(result["regime_coverage"]["layer2_n_passing"]),
        "level_ks": int(result["distributional_fidelity"]["ks_level_test"]["n_pass"]),
        "level_ks_median": float(result["distributional_fidelity"]["ks_level_test"]["median_stat"]),
        "median_fraction": int(result["distributional_fidelity"]["median_bias"]["n_pass"]),
        "bias_magnitude": int(result["distributional_fidelity"]["median_bias"]["n_mag_pass"]),
        "pathwise_ks": float(result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
    }


def markdown_table(rows: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| run | score | failed | cov90 | under/over | cond MAE | coint worst | level KS | regime L2 | path KS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['score']}/11 | {', '.join(row['failed'])} | "
            f"{row['cov90']:.3f} | {row['under70']}/{row['over95']} | "
            f"{row['conditional_mae']:.2f}% | {row['cointegration_worst']:.3f} | "
            f"{row['level_ks']}/25 | {row['regime_layer2']}/8 | {row['pathwise_ks']:.3f} |"
        )
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {name: summarize(load(path)) for name, path in RUNS.items()}
    payload = {
        "runs": rows,
        "mechanism_read": (
            "Simple free-running proper-score fine-tuning is capped as a primary route. "
            "Weak multivariate energy gave the best score by trading part of 385a's "
            "conditionality margin for better level occupancy. Stronger energy improved "
            "level KS further but lost conditionality/coverage. Marginal CRPS preserved "
            "conditionality and improved calibration error, but did not move level KS "
            "and lost worst-cell cointegration. The common pattern is objective geometry, "
            "not architecture collapse: scalar fine-tune losses can move one failed suite "
            "but do not produce the joint conditional level/regime law needed for 11/11."
        ),
        "decision": (
            "Keep 392a as the active 8/11 frontier and close simple proper-score "
            "fine-tune losses as the primary route. The next iteration should be a "
            "paradigm shift in the base likelihood/representation, not another "
            "energy/CRPS weight or post-hoc calibration branch."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 411a Proper-Score Fine-Tune Cap",
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
