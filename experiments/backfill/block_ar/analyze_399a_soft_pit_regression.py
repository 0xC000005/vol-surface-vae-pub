#!/usr/bin/env python
"""399a: diagnose why the soft-PIT fine-tune regressed versus 392a."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


BASELINE = Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json")
SOFT_PIT = Path("results/block_ar/398a_recent_soft_pit_w10_s8_s42/full11.json")
TRAINING_HISTORY = Path("models/backfill/398a_recent_soft_pit_w10_s8_s42/training_history.json")
OUT_DIR = Path("results/block_ar/399a_soft_pit_regression")
HORIZONS = ("1", "7", "14", "30")


def load(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def coverage_stack(result: dict[str, Any]) -> np.ndarray:
    return np.stack([arr(result["coverage"]["per_cell_coverage"][h]) for h in HORIZONS])


def top_deltas(before: np.ndarray, after: np.ndarray, n: int = 8) -> list[dict[str, Any]]:
    delta = after - before
    items: list[tuple[float, int, int]] = []
    for i in range(delta.shape[-2]):
        for j in range(delta.shape[-1]):
            items.append((float(delta[..., i, j].mean()), i, j))
    items.sort(key=lambda item: abs(item[0]), reverse=True)
    return [
        {
            "cell": [i, j],
            "delta": value,
            "before": float(before[..., i, j].mean()),
            "after": float(after[..., i, j].mean()),
        }
        for value, i, j in items[:n]
    ]


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    cov = coverage_stack(result)
    dist = result["distributional_fidelity"]
    ts = result["time_series"]
    return {
        "score": result["summary"]["n_pass"],
        "failed": result["summary"]["failed_suites"],
        "coverage90": result["coverage"]["overall"]["0.9"],
        "coverage_under70": int((cov < 0.70).sum()),
        "coverage_over95": int((cov > 0.95).sum()),
        "conditional_mae": result["conditionality"]["mae_reduction_pct"],
        "turb_calm": result["conditionality"]["turb_calm_ratio"],
        "kurtosis_ratio": ts["kurtosis"]["kurtosis_ratio"],
        "cointegration_worst": result["cointegration"]["worst_cell_ratio"],
        "daily_ks_pass": dist["ks_test"]["n_pass"],
        "level_ks_pass": dist["ks_level_test"]["n_pass"],
        "level_ks_median": dist["ks_level_test"]["median_stat"],
        "median_bias_pass": dist["median_bias"]["n_pass"],
        "bias_magnitude_pass": dist["median_bias"]["n_mag_pass"],
        "pathwise_ks": result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
        "pathwise_q99_ratio": result["pathwise_jump_realism"]["pathwise_max_jump"]["q99_ratio"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load(BASELINE)
    soft_pit = load(SOFT_PIT)
    history = load(TRAINING_HISTORY)

    base_summary = summarize(baseline)
    pit_summary = summarize(soft_pit)
    deltas = {
        key: pit_summary[key] - base_summary[key]
        for key in base_summary
        if isinstance(base_summary[key], (int, float))
    }

    base_level = arr(baseline["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    pit_level = arr(soft_pit["distributional_fidelity"]["ks_level_test"]["ks_grid"])
    base_median = arr(baseline["distributional_fidelity"]["median_bias"]["above_frac"])
    pit_median = arr(soft_pit["distributional_fidelity"]["median_bias"]["above_frac"])
    base_cov = coverage_stack(baseline)
    pit_cov = coverage_stack(soft_pit)

    best_epoch = min(history, key=lambda row: row["val_total"])
    payload = {
        "baseline_392a": base_summary,
        "soft_pit_398a": pit_summary,
        "deltas_398a_minus_392a": deltas,
        "training_best_epoch": best_epoch,
        "largest_level_ks_deltas": top_deltas(base_level, pit_level),
        "largest_median_fraction_deltas": top_deltas(base_median, pit_median),
        "largest_coverage_deltas": top_deltas(base_cov, pit_cov),
        "mechanism_read": (
            "Soft-PIT moment matching moved the internal PIT mean toward 0.5, but it "
            "also shifted medians upward across many cells, reduced tail/kurtosis, and "
            "damaged worst-cell cointegration. The loss matched low-order rank moments "
            "without preserving the full level distribution."
        ),
        "decision": (
            "Close low-order PIT-moment fine-tuning. A safer calibration objective would "
            "need cell/horizon-local interval constraints plus explicit median anchoring, "
            "but that is now close to optimizing the evaluator. The next principled move "
            "is to review whether 392a's residual failures are mostly evaluator-policy "
            "constraints or require a larger base-model likelihood paradigm."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 399a Soft-PIT Regression Analysis",
        "",
        "| model | score | failed | cov90 | under70 | over95 | cond MAE | kurt ratio | level KS | median pass | coint worst |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 392a | {base_summary['score']}/11 | {', '.join(base_summary['failed'])} | "
            f"{base_summary['coverage90']:.3f} | {base_summary['coverage_under70']} | "
            f"{base_summary['coverage_over95']} | {base_summary['conditional_mae']:.2f}% | "
            f"{base_summary['kurtosis_ratio']:.3f} | {base_summary['level_ks_pass']}/25 | "
            f"{base_summary['median_bias_pass']}/25 | {base_summary['cointegration_worst']:.3f} |"
        ),
        (
            f"| 398a | {pit_summary['score']}/11 | {', '.join(pit_summary['failed'])} | "
            f"{pit_summary['coverage90']:.3f} | {pit_summary['coverage_under70']} | "
            f"{pit_summary['coverage_over95']} | {pit_summary['conditional_mae']:.2f}% | "
            f"{pit_summary['kurtosis_ratio']:.3f} | {pit_summary['level_ks_pass']}/25 | "
            f"{pit_summary['median_bias_pass']}/25 | {pit_summary['cointegration_worst']:.3f} |"
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
