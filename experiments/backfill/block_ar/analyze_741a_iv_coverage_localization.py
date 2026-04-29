#!/usr/bin/env python
"""741a: localize IV coverage failures after scalar-temperature falsification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RUNS = {
    "temp1000_baseline": Path(
        "results/block_ar/734a_realvix_framework_lock_baseline/iv_val_full11_s64.json"
    ),
    "temp1025": Path(
        "results/block_ar/740a_iv_temperature_sweep/iv_val_full11_temp1025_s64.json"
    ),
    "temp1050": Path(
        "results/block_ar/740a_iv_temperature_sweep/iv_val_full11_temp105_s64.json"
    ),
}
OUT_DIR = Path("results/block_ar/741a_iv_coverage_localization")
HORIZONS = ("1", "7", "14", "30")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def flatten_cell_grid(grid: np.ndarray) -> list[tuple[float, int, int]]:
    return [(float(grid[i, j]), i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]


def coverage_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    above_frac = arr(result["distributional_fidelity"]["median_bias"]["above_frac"])
    mean_bias = arr(result["distributional_fidelity"]["median_bias"]["mean_bias"])
    for horizon in HORIZONS:
        grid = arr(result["coverage"]["per_cell_coverage"][horizon])
        for value, i, j in flatten_cell_grid(grid):
            records.append(
                {
                    "horizon": int(horizon),
                    "cell": [i, j],
                    "coverage": value,
                    "under_gap_to_70": max(0.0, 0.70 - value),
                    "over_gap_to_95": max(0.0, value - 0.95),
                    "gap_to_90": 0.90 - value,
                    "median_above_frac": float(above_frac[i, j]),
                    "mean_bias_ivpts": float(mean_bias[i, j]),
                }
            )
    return records


def regime_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer2 = result["regime_coverage"]["layer2_regime_cell"]
    for regime_name, by_horizon in layer2.items():
        for horizon, payload in by_horizon.items():
            grid = arr(payload["grid"])
            for value, i, j in flatten_cell_grid(grid):
                rows.append(
                    {
                        "regime": regime_name,
                        "horizon": int(horizon),
                        "cell": [i, j],
                        "coverage": value,
                        "under_gap_to_70": max(0.0, 0.70 - value),
                        "over_gap_to_95": max(0.0, value - 0.95),
                        "gap_to_90": 0.90 - value,
                    }
                )
    return rows


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    coverages = np.asarray([row["coverage"] for row in records], dtype=float)
    under = [row for row in records if row["coverage"] < 0.70]
    over = [row for row in records if row["coverage"] > 0.95]
    return {
        "n_points": len(records),
        "mean_coverage": float(coverages.mean()),
        "min_coverage": float(coverages.min()),
        "max_coverage": float(coverages.max()),
        "under70_count": len(under),
        "over95_count": len(over),
        "gate_fail_count": len(under) + len(over),
        "worst_under": sorted(records, key=lambda row: row["coverage"])[:10],
        "worst_over": sorted(records, key=lambda row: row["coverage"], reverse=True)[:10],
    }


def summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    cov = coverage_records(result)
    regime = regime_records(result)
    above = arr(result["distributional_fidelity"]["median_bias"]["above_frac"])
    cov_values = np.asarray([row["coverage"] for row in cov], dtype=float)
    repeated_above = np.repeat(above.reshape(-1), len(HORIZONS))
    return {
        "score": int(result["summary"]["n_pass"]),
        "failed": list(result["summary"]["failed_suites"]),
        "coverage_overall90": float(result["coverage"]["overall"]["0.9"]),
        "calibration_error": float(result["coverage"]["calibration_error"]),
        "pathwise_max_jump_ks": float(
            result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]
        ),
        "mean_reversion_full_pass": bool(result["mean_reversion"]["full_horizon"]["overall_pass"]),
        "coverage": summarize_records(cov),
        "regime_layer2": {
            **summarize_records(regime),
            "combo_passes": int(result["regime_coverage"]["layer2_n_passing"]),
            "combo_total": int(result["regime_coverage"]["layer2_n_total"]),
        },
        "coverage_vs_median_above_corr": float(np.corrcoef(cov_values, repeated_above)[0, 1]),
    }


def stable_failures(
    results: dict[str, dict[str, Any]], record_fn: Any, threshold: str
) -> list[dict[str, Any]]:
    key_fn = (
        (lambda row: (row.get("regime"), row["horizon"], tuple(row["cell"])))
        if record_fn is regime_records
        else (lambda row: (row["horizon"], tuple(row["cell"])))
    )
    sets = []
    values_by_key: dict[Any, dict[str, float]] = {}
    for name, result in results.items():
        rows = record_fn(result)
        if threshold == "under":
            failed = [row for row in rows if row["coverage"] < 0.70]
        elif threshold == "over":
            failed = [row for row in rows if row["coverage"] > 0.95]
        else:
            raise ValueError(threshold)
        keys = {key_fn(row) for row in failed}
        sets.append(keys)
        for row in failed:
            values_by_key.setdefault(key_fn(row), {})[name] = float(row["coverage"])
    common = set.intersection(*sets) if sets else set()
    rows = []
    for key in sorted(common, key=str):
        rows.append({"key": list(key), "coverages": values_by_key[key]})
    return rows


def cell_horizon_counts(records: list[dict[str, Any]], mode: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        if mode == "under" and row["coverage"] >= 0.70:
            continue
        if mode == "over" and row["coverage"] <= 0.95:
            continue
        cell = f"{row['cell'][0]},{row['cell'][1]}"
        counts[cell] = counts.get(cell, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = {name: summarize_run(result) for name, result in results.items()}
    baseline = results["temp1000_baseline"]
    baseline_cov = coverage_records(baseline)
    baseline_regime = regime_records(baseline)
    return {
        "runs": summaries,
        "stable_coverage_under70_all_temperatures": stable_failures(
            results, coverage_records, "under"
        ),
        "stable_coverage_over95_all_temperatures": stable_failures(
            results, coverage_records, "over"
        ),
        "stable_regime_under70_all_temperatures": stable_failures(
            results, regime_records, "under"
        ),
        "stable_regime_over95_all_temperatures": stable_failures(
            results, regime_records, "over"
        ),
        "baseline_under_cells_by_count": cell_horizon_counts(baseline_cov, "under"),
        "baseline_over_cells_by_count": cell_horizon_counts(baseline_cov, "over"),
        "baseline_regime_under_cells_by_count": cell_horizon_counts(baseline_regime, "under"),
        "baseline_regime_over_cells_by_count": cell_horizon_counts(baseline_regime, "over"),
        "mechanism_read": (
            "The remaining IV coverage defect is local and two-sided, not a global "
            "variance shortage. Temperature reduces aggregate calibration error but "
            "does not remove stable late-horizon undercoverage in specific cells, and "
            "it creates or preserves overcoverage in other cells. Regime layer-2 is the "
            "hardest gate because calm/turb splits expose the same local geometry with "
            "small per-regime sample sizes."
        ),
        "decision": (
            "Do not add another scalar sampler knob. The next model-side move, if any, "
            "must be a generic learned local uncertainty allocation mechanism tied to "
            "state/cell/horizon representation or a training objective that teaches "
            "local interval geometry without post-hoc calibration."
        ),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 741a IV Coverage Localization",
        "",
        "| run | score | cov90 | cal err | cov under/over | regime under/over | regime combos | path KS | MR full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in report["runs"].items():
        cov = row["coverage"]
        reg = row["regime_layer2"]
        lines.append(
            f"| {name} | {row['score']}/11 | {row['coverage_overall90']:.3f} | "
            f"{row['calibration_error']:.3f} | {cov['under70_count']}/{cov['over95_count']} | "
            f"{reg['under70_count']}/{reg['over95_count']} | "
            f"{reg['combo_passes']}/{reg['combo_total']} | "
            f"{row['pathwise_max_jump_ks']:.3f} | {row['mean_reversion_full_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Stable Failures",
            "",
            f"- stable standard undercoverage cells across temperatures: `{len(report['stable_coverage_under70_all_temperatures'])}`",
            f"- stable standard overcoverage cells across temperatures: `{len(report['stable_coverage_over95_all_temperatures'])}`",
            f"- stable regime undercoverage cells across temperatures: `{len(report['stable_regime_under70_all_temperatures'])}`",
            f"- stable regime overcoverage cells across temperatures: `{len(report['stable_regime_over95_all_temperatures'])}`",
            "",
            "## Baseline Cell Concentration",
            "",
            f"- standard undercoverage cell counts: `{report['baseline_under_cells_by_count']}`",
            f"- standard overcoverage cell counts: `{report['baseline_over_cells_by_count']}`",
            f"- regime undercoverage cell counts: `{report['baseline_regime_under_cells_by_count']}`",
            f"- regime overcoverage cell counts: `{report['baseline_regime_over_cells_by_count']}`",
            "",
            "## Worst Baseline Standard Undercoverage",
            "",
            "| horizon | cell | cov90 | median above frac | mean bias IV pts |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["runs"]["temp1000_baseline"]["coverage"]["worst_under"][:8]:
        lines.append(
            f"| {row['horizon']} | {row['cell']} | {row['coverage']:.3f} | "
            f"{row['median_above_frac']:.3f} | {row['mean_bias_ivpts']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Read",
            "",
            report["mechanism_read"],
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: load(path) for name, path in RUNS.items()}
    report = build_report(results)
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(OUT_DIR / "summary.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
