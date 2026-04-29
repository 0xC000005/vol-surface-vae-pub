#!/usr/bin/env python
"""Diagnose the IV full-suite degradation when moving from IV-only to joint scope.

The goal is to determine whether the joint runs fail because generated paths are
unrealistic, or because the conditional envelope/level allocation is miscalibrated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RUNS = {
    "674a_iv_only": "results/validations/2026-04-27/674a_channel_level_alltrain_e3/iv_val_full11.json",
    "676a_joint38": "results/validations/2026-04-27/676a_joint38_channel_level_alltrain/val_full11.json",
    "729a_joint39_vixproxy": "results/block_ar/729a_vixproxy_scorecard/joint_iv_val_full11_s64.json",
}


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nested(data: Any, *keys: str, default: Any = None) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def values_grid(data: Any) -> list[float]:
    out: list[float] = []
    if isinstance(data, list):
        for item in data:
            out.extend(values_grid(item))
    else:
        val = finite_float(data)
        if val is not None:
            out.append(val)
    return out


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_row(name: str, path: str) -> dict[str, Any]:
    data = load_json(path)
    failed = nested(data, "summary", "failed_suites", default=[])
    coverage = data.get("coverage", {})
    dist = data.get("distributional_fidelity", {})
    mr = data.get("mean_reversion", {})
    pathwise = data.get("pathwise_jump_realism", {})
    ts = data.get("time_series", {})
    regime = data.get("regime_coverage", {})
    cond = data.get("conditionality", {})

    cov90_by_h = {
        h: finite_float(nested(coverage, "per_horizon", h, "0.9"))
        for h in ("1", "7", "14", "30")
    }
    cov95_by_h = {
        h: finite_float(nested(coverage, "per_horizon", h, "0.95"))
        for h in ("1", "7", "14", "30")
    }
    worst_cov_by_h = {
        h: finite_float(nested(coverage, "worst_cell_per_horizon", h))
        for h in ("1", "7", "14", "30")
    }
    mr_ratio_by_h = {
        h: finite_float(nested(mr, "full_horizon", "per_horizon", h, "ratio"))
        for h in ("1", "7", "14", "30")
    }
    mr_pass_by_h = {
        h: nested(mr, "full_horizon", "per_horizon", h, "aggregate_pass")
        for h in ("1", "7", "14", "30")
    }

    median_bias_above = values_grid(nested(dist, "median_bias", "above_frac", default=[]))
    return {
        "name": name,
        "path": path,
        "n_pass": nested(data, "summary", "n_pass"),
        "n_total": nested(data, "summary", "n_total"),
        "failed_suites": failed,
        "coverage90": finite_float(nested(coverage, "overall", "0.9")),
        "coverage95": finite_float(nested(coverage, "overall", "0.95")),
        "coverage_calibration_error": finite_float(nested(coverage, "calibration_error")),
        "coverage90_by_horizon": cov90_by_h,
        "coverage95_by_horizon": cov95_by_h,
        "worst_cell_coverage_by_horizon": worst_cov_by_h,
        "conditionality_width_ratio": finite_float(cond.get("width_ratio")),
        "conditionality_worst_cell_width_ratio": finite_float(cond.get("worst_cell_width_ratio")),
        "daily_ks_pass": nested(dist, "ks_test", "n_pass"),
        "daily_ks_median": finite_float(nested(dist, "ks_test", "median_stat")),
        "daily_ks_worst": finite_float(nested(dist, "ks_test", "worst_stat")),
        "level_ks_pass": nested(dist, "ks_level_test", "n_pass"),
        "level_ks_median": finite_float(nested(dist, "ks_level_test", "median_stat")),
        "level_ks_worst": finite_float(nested(dist, "ks_level_test", "worst_stat")),
        "median_bias_pass": nested(dist, "median_bias", "pass"),
        "median_above_gt_mean": mean(median_bias_above),
        "median_above_gt_min": min(median_bias_above) if median_bias_above else None,
        "median_above_gt_max": max(median_bias_above) if median_bias_above else None,
        "acf_corr": finite_float(nested(ts, "acf", "acf_correlation")),
        "kurtosis_ratio": finite_float(nested(ts, "kurtosis", "kurtosis_ratio")),
        "regime_layer1_pass": regime.get("layer1_pass"),
        "regime_layer2_pass": regime.get("layer2_pass"),
        "regime_layer2_n_passing": regime.get("layer2_n_passing"),
        "regime_layer2_n_total": regime.get("layer2_n_total"),
        "regime_layer3_pass": regime.get("layer3_pass"),
        "mean_reversion_overall_pass": mr.get("overall_pass"),
        "mean_reversion_full_overall_pass": nested(mr, "full_horizon", "overall_pass"),
        "mean_reversion_ratio_by_horizon": mr_ratio_by_h,
        "mean_reversion_pass_by_horizon": mr_pass_by_h,
        "pathwise_max_jump_pass": nested(pathwise, "pathwise_max_jump", "pass"),
        "pathwise_max_jump_ks": finite_float(nested(pathwise, "pathwise_max_jump", "ks_stat")),
        "pathwise_q90_ratio": finite_float(nested(pathwise, "pathwise_max_jump", "q90_ratio")),
        "pathwise_q99_ratio": finite_float(nested(pathwise, "pathwise_max_jump", "q99_ratio")),
        "pathwise_cell_q99_pass": nested(pathwise, "per_cell_q99", "n_pass"),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    run_paths = {
        "674a_iv_only": args.full_674_iv,
        "676a_joint38": args.full_676_joint,
        "729a_joint39_vixproxy": args.full_729_joint_vix,
    }
    rows = [run_row(name, path) for name, path in run_paths.items()]
    by_name = {row["name"]: row for row in rows}
    joint = by_name["729a_joint39_vixproxy"]
    prior_joint = by_name["676a_joint38"]
    iv_only = by_name["674a_iv_only"]

    report = {
        "analysis_id": "731a_joint_scope_iv_tradeoff",
        "inputs": run_paths,
        "rows": rows,
        "diagnosis": {
            "not_a_global_path_realism_collapse": (
                joint["daily_ks_pass"] == 24
                and (joint["acf_corr"] or 0.0) > 0.95
                and (joint["kurtosis_ratio"] or 99.0) < 1.25
            ),
            "primary_failure_class": "joint_scope_calibration_and_level_state_allocation",
            "coverage_read": (
                "All compared learned runs under-cover 90/95 percent intervals. "
                "729a is slightly better than 676a on overall 90 percent coverage, "
                "but still below a calibrated conditional envelope."
            ),
            "level_read": (
                "Daily-change KS remains strong at 24/25, while level KS falls from 15/25 "
                "in IV-only 674a to 13/25 in 676a and 12/25 in 729a. The joint penalty is "
                "mainly level-state allocation, not one-day increment shape."
            ),
            "mean_reversion_read": (
                "Mean-reversion failure is concentrated at horizon 7 oversnap-back: "
                f"674a h7 ratio {iv_only['mean_reversion_ratio_by_horizon']['7']:.3f}, "
                f"676a h7 ratio {prior_joint['mean_reversion_ratio_by_horizon']['7']:.3f}, "
                f"729a h7 ratio {joint['mean_reversion_ratio_by_horizon']['7']:.3f}; "
                "h30 remains close to 1.0."
            ),
            "pathwise_read": (
                "729a pathwise max-jump KS is a near-threshold miss at 0.507 versus gate 0.5; "
                "cell q99 tails still pass 22/25. This is a guardrail issue, not a path realism collapse."
            ),
        },
        "next_experiment_target": {
            "target": "joint_scope_conditional_envelope_and_level_state_allocation",
            "avoid": [
                "VIX-specific repair",
                "post-hoc gluing",
                "scope-specific loss weights",
                "new backend before classifying objective/data-object failure",
            ],
            "principled_options": [
                "Use a single formula for channel/group balancing so IV root variables are not diluted by the wider factor panel.",
                "Use a single multi-horizon rollout/root consistency objective across all scopes to reduce h7 oversnap-back without naming IV or factors.",
                "Audit train-tail versus validation before changing the backend, because the current failure is calibration allocation, not lack of path realism.",
            ],
            "recommended_next_step": (
                "Run a framework-lock experiment that changes only deterministic channel/group balancing "
                "or multi-horizon rollout consistency with the same formula across IV-only, anchor-only, and joint."
            ),
        },
    }
    return scrub(report)


def fmt(value: Any) -> str:
    val = finite_float(value)
    if val is None:
        return "n/a"
    return f"{val:.3f}"


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 731a Joint-Scope IV Trade-Off Diagnostic",
        "",
        "## Run Comparison",
        "",
        "| Run | Pass | Cov90 | Cov95 | Daily KS | Level KS | ACF | Kurtosis | Path KS | MR h7 | MR h30 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {name} | {n_pass}/{n_total} | {cov90} | {cov95} | {daily} | {level} | {acf} | {kurt} | {pathks} | {mr7} | {mr30} |".format(
                name=row["name"],
                n_pass=row["n_pass"],
                n_total=row["n_total"],
                cov90=fmt(row["coverage90"]),
                cov95=fmt(row["coverage95"]),
                daily=row["daily_ks_pass"],
                level=row["level_ks_pass"],
                acf=fmt(row["acf_corr"]),
                kurt=fmt(row["kurtosis_ratio"]),
                pathks=fmt(row["pathwise_max_jump_ks"]),
                mr7=fmt(row["mean_reversion_ratio_by_horizon"]["7"]),
                mr30=fmt(row["mean_reversion_ratio_by_horizon"]["30"]),
            )
        )
    lines.extend(["", "## Diagnosis", ""])
    for key, value in report["diagnosis"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Next Experiment Target", ""])
    lines.append(f"- Target: `{report['next_experiment_target']['target']}`.")
    lines.append(f"- Recommended next step: {report['next_experiment_target']['recommended_next_step']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-674-iv", default=DEFAULT_RUNS["674a_iv_only"])
    parser.add_argument("--full-676-joint", default=DEFAULT_RUNS["676a_joint38"])
    parser.add_argument("--full-729-joint-vix", default=DEFAULT_RUNS["729a_joint39_vixproxy"])
    parser.add_argument(
        "--output-json",
        default="results/block_ar/731a_joint_scope_iv_tradeoff/diagnostic.json",
    )
    parser.add_argument(
        "--output-md",
        default="results/block_ar/731a_joint_scope_iv_tradeoff/diagnostic.md",
    )
    args = parser.parse_args()

    report = build_report(args)
    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(make_markdown(report), encoding="utf-8")

    print(json.dumps(report["diagnosis"], indent=2))
    print(json.dumps(report["next_experiment_target"], indent=2))
    print(f"Wrote {output_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {output_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
