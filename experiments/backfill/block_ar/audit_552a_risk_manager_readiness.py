#!/usr/bin/env python
"""552a: risk-manager readiness audit under stress-generator framing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402


CORE_AUTHENTICITY_SUITES = [
    "surface",
    "time_series",
    "block_ar",
    "cointegration",
    "cross_cell_correlation",
    "mean_reversion",
    "pathwise_jump_realism",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _get_float_map_values(mapping: dict[str, Any]) -> list[float]:
    return [float(value) for _key, value in sorted(mapping.items(), key=lambda item: int(item[0]))]


def lower_only_coverage_pass(
    coverage: dict[str, Any],
    min_overall_90: float = 0.85,
    min_worst_cell: float = 0.70,
) -> dict[str, Any]:
    """Stress framing: ignore high-side overcoverage but keep lower risk inclusion."""
    overall = coverage.get("overall", {})
    cov90 = float(overall.get("0.9", overall.get(0.9, 0.0)))
    horizon_pass = all(_as_bool(v) for v in coverage.get("horizon_pass", {}).values())
    worst_values = _get_float_map_values(coverage.get("worst_cell_per_horizon", {}))
    worst_cell = min(worst_values) if worst_values else 0.0
    return {
        "cov90": cov90,
        "min_overall_90": float(min_overall_90),
        "worst_cell": float(worst_cell),
        "min_worst_cell": float(min_worst_cell),
        "horizon_pass": bool(horizon_pass),
        "pass": bool(cov90 >= min_overall_90 and horizon_pass and worst_cell >= min_worst_cell),
        "ignored_original_high_side_overcoverage_cap": True,
    }


def lower_only_regime_pass(
    regime: dict[str, Any],
    min_regime_cell: float = 0.70,
) -> dict[str, Any]:
    """Stress framing: require no undercovered regime/cell, ignore overcoverage."""
    layer2 = regime.get("layer2_regime_cell", {})
    worst_values: list[float] = []
    for regime_data in layer2.values():
        for horizon_data in regime_data.values():
            if isinstance(horizon_data, dict) and "worst" in horizon_data:
                worst_values.append(float(horizon_data["worst"]))
    worst = min(worst_values) if worst_values else 0.0
    return {
        "layer1_pass": _as_bool(regime.get("layer1_pass", False)),
        "layer3_pass": _as_bool(regime.get("layer3_pass", False)),
        "worst_regime_cell": float(worst),
        "min_regime_cell": float(min_regime_cell),
        "pass": bool(
            _as_bool(regime.get("layer1_pass", False))
            and _as_bool(regime.get("layer3_pass", False))
            and worst >= min_regime_cell
        ),
        "ignored_original_high_side_overcoverage_cap": True,
    }


def distributional_authenticity(result: dict[str, Any]) -> dict[str, Any]:
    dist = result.get("distributional_fidelity", {})
    required_parts = {
        "daily_change_ks": _as_bool(dist.get("ks_test", {}).get("pass", False)),
        "median_bias": _as_bool(dist.get("median_bias", {}).get("pass", False)),
        "window_floor": _as_bool(dist.get("window_floor", {}).get("pass", False)),
        "explosion": _as_bool(dist.get("explosion", {}).get("pass", False)),
        "cell_mae": _as_bool(dist.get("cell_mae", {}).get("pass", False)),
    }
    level_ks = dist.get("ks_level_test", {})
    return {
        "required_parts": required_parts,
        "pass": bool(all(required_parts.values())),
        "level_ks_n_pass": int(level_ks.get("n_pass", 0)),
        "level_ks_pass": _as_bool(level_ks.get("pass", False)),
        "level_ks_treated_as_warning_for_stress": True,
    }


def scenario_authenticity(result: dict[str, Any]) -> dict[str, Any]:
    suite_passes = {
        name: _as_bool(result.get(name, {}).get("overall_pass", False))
        for name in CORE_AUTHENTICITY_SUITES
    }
    dist = distributional_authenticity(result)
    suite_passes["distributional_required_parts"] = dist["pass"]
    return {
        "suite_passes": suite_passes,
        "pass": bool(all(suite_passes.values())),
    }


def conditionality_read(result: dict[str, Any]) -> dict[str, Any]:
    cond = result.get("conditionality", {})
    mae_reduction = float(cond.get("mae_reduction_pct", 0.0))
    return {
        "mae_reduction_pct": mae_reduction,
        "original_pass": _as_bool(cond.get("overall_pass", False)),
        "borderline": bool(4.75 <= mae_reduction < 5.0),
        "pass": bool(_as_bool(cond.get("overall_pass", False)) or mae_reduction >= 5.0),
    }


def score_candidate(name: str, result: dict[str, Any]) -> dict[str, Any]:
    coverage = lower_only_coverage_pass(result.get("coverage", {}))
    regime = lower_only_regime_pass(result.get("regime_coverage", {}))
    authentic = scenario_authenticity(result)
    conditionality = conditionality_read(result)
    dist = distributional_authenticity(result)
    original_summary = result.get("summary", {})
    warnings: list[str] = []
    if not dist["level_ks_pass"]:
        warnings.append("level_ks_warning")
    if conditionality["borderline"]:
        warnings.append("conditionality_borderline")
    if not regime["pass"]:
        warnings.append("regime_undercoverage_warning")
    if not coverage["pass"]:
        warnings.append("coverage_underinclusion_warning")
    stress_pass = bool(
        coverage["pass"]
        and regime["pass"]
        and authentic["pass"]
        and conditionality["pass"]
    )
    return {
        "name": name,
        "original_n_pass": int(original_summary.get("n_pass", 0)),
        "original_failed_suites": list(original_summary.get("failed_suites", [])),
        "stress_pass": stress_pass,
        "stress_score": int(coverage["pass"])
        + int(regime["pass"])
        + int(authentic["pass"])
        + int(conditionality["pass"]),
        "stress_score_total": 4,
        "coverage_lower_only": coverage,
        "regime_lower_only": regime,
        "scenario_authenticity": authentic,
        "conditionality": conditionality,
        "distributional_authenticity": dist,
        "warnings": warnings,
    }


def factor_readiness(data_keys: list[str]) -> dict[str, Any]:
    non_iv = [key for key in data_keys if key != "surface"]
    return {
        "available_non_iv_factors": non_iv,
        "current_generator_scope": "iv_surface_only",
        "multifactor_ready": False,
        "recommended_factor_bridge": [
            "condition on observed return/price/level/slope/skew history",
            "generate IV scenarios jointly with factor-consistent return/level paths",
            "evaluate factor-scenario coherence before claiming portfolio stress readiness",
        ],
    }


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    ranked = report["ranked_candidates"]
    lines = [
        "# 552a Risk-Manager Readiness Audit",
        "",
        "## Framing",
        "",
        "This audit treats the generator as a risk stress scenario system, not a strictly calibrated conditional law.",
        "High-side overcoverage is not a failure here. Under-inclusion, weak conditionality, unrealistic paths, broken dependence, and missing factor linkage remain failures.",
        "",
        "## Candidate Ranking",
        "",
        "| Candidate | Original | Stress score | Stress pass | Key warnings |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in ranked:
        warnings = ", ".join(row["warnings"]) if row["warnings"] else "none"
        lines.append(
            f"| `{row['name']}` | `{row['original_n_pass']}/11` | "
            f"`{row['stress_score']}/{row['stress_score_total']}` | "
            f"`{row['stress_pass']}` | {warnings} |"
        )

    best = ranked[0]
    lines.extend(
        [
            "",
            "## Best Current Candidate",
            "",
            f"- best candidate: `{best['name']}`",
            f"- stress pass: `{best['stress_pass']}`",
            f"- lower-only coverage pass: `{best['coverage_lower_only']['pass']}` "
            f"(cov90 `{best['coverage_lower_only']['cov90']:.3f}`, worst cell `{best['coverage_lower_only']['worst_cell']:.3f}`)",
            f"- lower-only regime pass: `{best['regime_lower_only']['pass']}` "
            f"(worst regime cell `{best['regime_lower_only']['worst_regime_cell']:.3f}`)",
            f"- conditionality pass: `{best['conditionality']['pass']}` "
            f"(MAE reduction `{best['conditionality']['mae_reduction_pct']:.3f}%`)",
            f"- scenario authenticity pass: `{best['scenario_authenticity']['pass']}`",
            "",
            "## Factor Readiness",
            "",
            f"- available non-IV factors: `{', '.join(report['factor_readiness']['available_non_iv_factors'])}`",
            f"- current generator scope: `{report['factor_readiness']['current_generator_scope']}`",
            f"- multifactor ready: `{report['factor_readiness']['multifactor_ready']}`",
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("NAME", "FULL11_JSON"),
        required=True,
        help="Candidate name and full11.json path. Can be repeated.",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    candidates = [
        score_candidate(name, load_result(path))
        for name, path in args.candidate
    ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            int(item["stress_pass"]),
            int(item["stress_score"]),
            int(item["original_n_pass"]),
        ),
        reverse=True,
    )
    data = np.load(args.data_path)
    factors = factor_readiness(list(data.files))
    best = ranked[0]
    if best["stress_pass"]:
        decision = (
            "A risk-manager demo is possible now if it is framed as an IV-only stress "
            "scenario prototype with explicit warnings and separate base-law metrics. "
            "The next model research step should add factor-conditioned generation."
        )
    else:
        decision = (
            "No current candidate is fully presentable as a risk-manager stress system. "
            "The closest candidate can be shown as a prototype, but the next research step "
            "must target conditionality and regime under-inclusion, not overcoverage."
        )
    report = {
        "ranked_candidates": ranked,
        "factor_readiness": factors,
        "decision": decision,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(report), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(json.dumps(make_serializable({"best": best["name"], "stress_pass": best["stress_pass"], "decision": decision}), indent=2))


if __name__ == "__main__":
    main()
