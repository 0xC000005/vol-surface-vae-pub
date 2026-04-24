#!/usr/bin/env python
"""414a: analyze complementarity between 392a AR and 413a direct path models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RUNS = {
    "392a_ar_frontier": Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json"),
    "413a_direct_path": Path("results/block_ar/413a_recent_score_path_fm_s42/full11.json"),
}
OUT_DIR = Path("results/block_ar/414a_ar_vs_direct_path_complementarity")
SUITES = [
    "surface",
    "coverage",
    "conditionality",
    "time_series",
    "block_ar",
    "cointegration",
    "regime_coverage",
    "distributional_fidelity",
    "cross_cell_correlation",
    "mean_reversion",
    "pathwise_jump_realism",
]
HORIZONS = ("1", "7", "14", "30")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cov_counts(result: dict[str, Any]) -> tuple[int, int]:
    values = []
    for h in HORIZONS:
        values.extend(np.asarray(result["coverage"]["per_cell_coverage"][h], dtype=float).ravel())
    arr = np.asarray(values)
    return int((arr < 0.70).sum()), int((arr > 0.95).sum())


def suite_passes(result: dict[str, Any]) -> dict[str, bool]:
    return {suite: bool(result[suite]["overall_pass"]) for suite in SUITES}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    under, over = cov_counts(result)
    return {
        "score": int(result["summary"]["n_pass"]),
        "failed": list(result["summary"]["failed_suites"]),
        "coverage90": float(result["coverage"]["overall"]["0.9"]),
        "under70": under,
        "over95": over,
        "conditionality": float(result["conditionality"]["mae_reduction_pct"]),
        "kurtosis_ratio": float(result["time_series"]["kurtosis"]["kurtosis_ratio"]),
        "cointegration_worst": float(result["cointegration"]["worst_cell_ratio"]),
        "regime_layer2": int(result["regime_coverage"]["layer2_n_passing"]),
        "daily_ks": int(result["distributional_fidelity"]["ks_test"]["n_pass"]),
        "level_ks": int(result["distributional_fidelity"]["ks_level_test"]["n_pass"]),
        "median_fraction": int(result["distributional_fidelity"]["median_bias"]["n_pass"]),
        "bias_magnitude": int(result["distributional_fidelity"]["median_bias"]["n_mag_pass"]),
        "corr_ratio": float(result["cross_cell_correlation"]["corr_ratio"]),
        "rank_ratio": float(result["cross_cell_correlation"]["rank_ratio"]),
        "mean_reversion_pass": bool(result["mean_reversion"]["overall_pass"]),
        "pathwise_ks": float(result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
        "suite_passes": suite_passes(result),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {name: summarize(load(path)) for name, path in RUNS.items()}
    union_passes = {
        suite: any(rows[name]["suite_passes"][suite] for name in rows)
        for suite in SUITES
    }
    payload = {
        "runs": rows,
        "union_passes": union_passes,
        "union_n_pass": int(sum(union_passes.values())),
        "mechanism_read": (
            "392a and 413a expose a genuine factorization split. The AR model owns "
            "local/structural dynamics: conditionality, time-series, cointegration, "
            "cross-cell correlation, mean reversion, and pathwise realism. The direct "
            "path model owns level occupancy: daily KS, level KS, median fraction, and "
            "bias magnitude all pass strongly. Both still fail coverage and regime "
            "layer2, but their coverage errors have opposite geometry: 392a has mostly "
            "over-95 cells, while 413a has mostly under-70 cells."
        ),
        "decision": (
            "Do one bounded diagnostic mixture, not as the final architecture but as "
            "a mechanism test. A fixed mostly-AR sample mixture can test whether the "
            "two learned laws contain complementary support that could later be "
            "distilled into one clean model. If the mixture cannot improve beyond 8/11 "
            "or damages structural passes, close mixture/ensemble work immediately."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 414a AR vs Direct Path Complementarity",
        "",
        "| run | score | failed | cov under/over | cond | coint worst | level KS | corr ratio | MR | path KS |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['score']}/11 | {', '.join(row['failed'])} | "
            f"{row['under70']}/{row['over95']} | {row['conditionality']:.2f}% | "
            f"{row['cointegration_worst']:.3f} | {row['level_ks']}/25 | "
            f"{row['corr_ratio']:.3f} | {row['mean_reversion_pass']} | {row['pathwise_ks']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Suite-union pass count: `{payload['union_n_pass']}/11`.",
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
