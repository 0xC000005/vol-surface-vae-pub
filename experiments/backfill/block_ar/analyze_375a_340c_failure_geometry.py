#!/usr/bin/env python
"""375a: failure-geometry audit for the revised-suite 340c baseline.

This is a post-experiment analysis step. It reads the full revised 340c
evaluation artifact and the frozen-calibration ladder, then writes a compact
diagnostic summary of what is still failing and what should not be tried next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


HORIZONS = ["1", "7", "14", "30"]


def as_array(grid: Any) -> np.ndarray:
    return np.asarray(grid, dtype=np.float64)


def worst_cells(grid: Any, *, high_bad: bool = False, n: int = 5) -> list[dict[str, Any]]:
    arr = as_array(grid)
    order = np.argsort(arr.ravel())
    if high_bad:
        order = order[::-1]
    out = []
    for idx in order[:n]:
        r, c = np.unravel_index(int(idx), arr.shape)
        out.append({"cell": [int(r), int(c)], "value": float(arr[r, c])})
    return out


def count_outside(grid: Any, lo: float, hi: float) -> dict[str, int]:
    arr = as_array(grid)
    return {
        "below": int((arr < lo).sum()),
        "above": int((arr > hi).sum()),
        "inside": int(((arr >= lo) & (arr <= hi)).sum()),
        "total": int(arr.size),
    }


def coverage_audit(full: dict[str, Any]) -> dict[str, Any]:
    cov = full["coverage"]
    per_cell = cov["per_cell_coverage"]
    by_horizon = {}
    for h in HORIZONS:
        grid = per_cell[h]
        by_horizon[h] = {
            "worst": float(cov["worst_cell_per_horizon"][h]),
            "best": float(cov["best_cell_per_horizon"][h]),
            "outside_gate_counts": count_outside(grid, 0.70, 0.95),
            "most_undercovered": worst_cells(grid, n=4),
            "most_overcovered": worst_cells(grid, high_bad=True, n=4),
        }
    return {
        "overall_90": float(cov["overall"]["0.9"]),
        "overall_95": float(cov["overall"]["0.95"]),
        "calibration_error": float(cov["calibration_error"]),
        "per_horizon": by_horizon,
        "read": (
            "Aggregate coverage is close, but the per-cell gate fails because "
            "some cells are materially undercovered while others overcover."
        ),
    }


def conditionality_audit(full: dict[str, Any]) -> dict[str, Any]:
    c = full["conditionality"]
    per_h = c["per_horizon_conditionality"]
    calm = c["per_regime_conditionality"]["calm"]
    turb = c["per_regime_conditionality"]["turb"]
    return {
        "mae_reduction_pct": float(c["mae_reduction_pct"]),
        "mae_pass": bool(c["mae_pass"]),
        "width_ratio": float(c["width_ratio"]),
        "turb_calm_ratio_informational": float(c["turb_calm_ratio"]),
        "per_horizon_mae_reduction_pct": {
            h: float(per_h[h]["mae_reduction_pct"]) for h in HORIZONS
        },
        "calm_avg_mae_reduction_pct": float(calm["avg_mae_reduction_pct"]),
        "calm_worst_cell_mae_reduction_pct": float(calm["worst_cell_mae_reduction_pct"]),
        "turb_avg_mae_reduction_pct": float(turb["avg_mae_reduction_pct"]),
        "turb_worst_cell_mae_reduction_pct": float(turb["worst_cell_mae_reduction_pct"]),
        "read": (
            "Conditional signal is front-loaded at h1 and mostly gone by h14/h30; "
            "calm regimes are worse than the shuffled-history baseline."
        ),
    }


def regime_audit(full: dict[str, Any]) -> dict[str, Any]:
    r = full["regime_coverage"]
    combos = []
    for regime in ["calm", "turb"]:
        for h in HORIZONS:
            item = r["layer2_regime_cell"][regime][h]
            combos.append(
                {
                    "regime": regime,
                    "horizon": int(h),
                    "worst": float(item["worst"]),
                    "worst_cell": item["worst_cell"],
                    "best": float(item["best"]),
                    "best_cell": item["best_cell"],
                    "below_70": count_outside(item["grid"], 0.70, 0.95)["below"],
                    "above_95": count_outside(item["grid"], 0.70, 0.95)["above"],
                }
            )
    combos_by_worst = sorted(combos, key=lambda x: x["worst"])
    width_vs_vov = {
        h: {
            "spearman_rho": float(r["width_vs_vov"][h]["spearman_rho"]),
            "p90_p10_width_ratio": float(r["width_vs_vov"][h]["p90_p10_width_ratio"]),
        }
        for h in HORIZONS
    }
    return {
        "layer1_pass": bool(r["layer1_pass"]),
        "layer2_n_passing": int(r["layer2_n_passing"]),
        "layer2_n_total": int(r["layer2_n_total"]),
        "layer3_pass": bool(r["layer3_pass"]),
        "n_calm": int(r["n_calm"]),
        "n_turb": int(r["n_turb"]),
        "worst_layer2_combinations": combos_by_worst[:6],
        "width_vs_vov": width_vs_vov,
        "read": (
            "Regime aggregate coverage is not the issue; every regime/horizon "
            "combination fails only after slicing by cell."
        ),
    }


def distributional_audit(full: dict[str, Any]) -> dict[str, Any]:
    d = full["distributional_fidelity"]
    return {
        "daily_ks_n_pass": int(d["ks_test"]["n_pass"]),
        "daily_ks_worst": float(d["ks_test"]["worst_stat"]),
        "level_ks_n_pass": int(d["ks_level_test"]["n_pass"]),
        "level_ks_worst": float(d["ks_level_test"]["worst_stat"]),
        "level_ks_worst_cells": worst_cells(d["ks_level_test"]["ks_grid"], high_bad=True, n=6),
        "median_bias_n_pass": int(d["median_bias"]["n_pass"]),
        "median_bias_mag_n_pass": int(d["median_bias"]["n_mag_pass"]),
        "cell_mae_n_pass": int(d["cell_mae"]["n_pass"]),
        "cell_mae_worst": float(d["cell_mae"]["worst_mae"]),
        "read": (
            "Daily move distribution passes, but unconditional IV level law fails; "
            "the miss is mostly level/state occupancy, not local increment shape."
        ),
    }


def calibration_read(cal: dict[str, Any]) -> dict[str, Any]:
    variants = {}
    for name, item in cal["variants"].items():
        d = item["digest"]
        variants[name] = {
            "score": int(item["n_pass_proxy11"]),
            "failed": item["failed_proxy11"],
            "coverage90": float(d["coverage90"]),
            "level_ks_cells": int(d["level_ks_cells"]),
            "daily_ks_cells": int(d["daily_ks_cells"]),
            "regime_layer2": d["regime_layer2"],
            "mean_reversion_pass": bool(d["mean_reversion_pass"]),
            "cointegration_worst_cell": float(d["cointegration_worst_cell"]),
        }
    return {
        "best_variant": cal["best_variant"],
        "best_n_pass_proxy11": int(cal["best_n_pass_proxy11"]),
        "variants": variants,
        "read": (
            "The only variant that improves level KS loses core path-law tests, "
            "so post-hoc affine calibration is not the next clean knob."
        ),
    }


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    cov = summary["coverage"]
    cond = summary["conditionality"]
    reg = summary["regime"]
    dist = summary["distributional_fidelity"]
    cal = summary["calibration_ladder"]

    lines = [
        "# 375a 340c Failure Geometry",
        "",
        "This audit reads existing artifacts only. It does not change the model or test suite.",
        "",
        "## High-Level Read",
        "",
        "- The 340c backbone has learned local path mechanics: daily KS is 24/25, pathwise jump realism passes, surface/cross-cell/mean-reversion tests pass on the full artifact.",
        "- The remaining failures are conditional state occupancy and per-cell interval geometry: coverage, conditionality, regime coverage, and level KS.",
        "- Post-hoc affine calibration is a dead end: it can improve level KS, but only by degrading coverage and core path-law tests.",
        "",
        "## Coverage Geometry",
        "",
        f"- Overall 90% coverage is `{cov['overall_90']:.3f}` with calibration error `{cov['calibration_error']:.4f}`.",
        "- Per-cell gate is the failure: cells must stay inside [0.70, 0.95], but long horizons have both undercoverage and overcoverage.",
    ]
    for h in HORIZONS:
        item = cov["per_horizon"][h]
        counts = item["outside_gate_counts"]
        lines.append(
            f"- h={h}: worst `{item['worst']:.3f}`, best `{item['best']:.3f}`, "
            f"below70 `{counts['below']}`, above95 `{counts['above']}`."
        )

    lines.extend(
        [
            "",
            "## Conditionality Geometry",
            "",
            f"- MAE reduction is `{cond['mae_reduction_pct']:.2f}%`, below the `>5%` gate.",
            f"- Per-horizon MAE reductions are `{cond['per_horizon_mae_reduction_pct']}`.",
            f"- Calm-regime average MAE reduction is `{cond['calm_avg_mae_reduction_pct']:.1f}%`; turb average is `{cond['turb_avg_mae_reduction_pct']:.1f}%`.",
            "- Interpretation: the model uses history strongly for h1, weakly for h7, and barely for h14/h30. It is not just too wide or too narrow; it loses useful long-horizon conditioning.",
            "",
            "## Regime And Level Law",
            "",
            f"- Regime layer1 passes and layer3 passes, but layer2 is `{reg['layer2_n_passing']}/{reg['layer2_n_total']}`.",
            "- Worst regime/cell combinations:",
        ]
    )
    for item in reg["worst_layer2_combinations"][:5]:
        lines.append(
            f"- {item['regime']} h={item['horizon']}: worst `{item['worst']:.3f}` "
            f"at cell `{item['worst_cell']}`, best `{item['best']:.3f}`."
        )

    lines.extend(
        [
            "",
            "## Distributional Fidelity",
            "",
            f"- Daily KS passes `{dist['daily_ks_n_pass']}/25`; level KS passes only `{dist['level_ks_n_pass']}/25`.",
            f"- Median-bias cells pass `{dist['median_bias_n_pass']}/25`; cell MAE passes `{dist['cell_mae_n_pass']}/25`.",
            "- Interpretation: the generated increments are plausible, but the generated paths visit the wrong level distribution in enough cells to fail the marginal law.",
            "",
            "## Calibration Ladder Read",
            "",
            f"- Best frozen-calibration variant remains `{cal['best_variant']}` at `{cal['best_n_pass_proxy11']}/11` proxy score.",
            "- Horizon/cell affine calibration improves level KS but reduces coverage and breaks time-series or mean-reversion tests.",
            "",
            "## Decision",
            "",
            "The next clean move is not another post-hoc calibrator. The remaining pathology must be learned inside the conditional path law: long-horizon conditional state occupancy, per-cell uncertainty, and regime-sliced interval geometry. The next experiment should align training with proper distributional scoring of the future path while keeping the generative core vanilla.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full11", default="results/block_ar/340c_v0_s42_revised/full11.json")
    parser.add_argument(
        "--calibration_ladder",
        default="results/validations/2026-04-24/analysis/374a_frozen_340c_calibration_ladder/summary.json",
    )
    parser.add_argument(
        "--output_dir",
        default="results/validations/2026-04-24/analysis/375a_340c_failure_geometry",
    )
    args = parser.parse_args()

    full = json.loads(Path(args.full11).read_text(encoding="utf-8"))
    cal = json.loads(Path(args.calibration_ladder).read_text(encoding="utf-8"))

    summary = {
        "source_full11": args.full11,
        "source_calibration_ladder": args.calibration_ladder,
        "full11_score": full["summary"],
        "coverage": coverage_audit(full),
        "conditionality": conditionality_audit(full),
        "regime": regime_audit(full),
        "distributional_fidelity": distributional_audit(full),
        "calibration_ladder": calibration_read(cal),
        "decision": (
            "Stop post-hoc affine calibration. Next experiment should learn the "
            "remaining conditional occupancy/uncertainty geometry in the path law "
            "using a general distributional objective."
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md")
    print(json.dumps({"score": full["summary"], "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()
