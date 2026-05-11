#!/usr/bin/env python
"""Decompose fixed-start narrative-control warnings into actionable causes."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


DEFAULT_CONTROL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_862d_full_s96_repeat/"
    "fixed_start_control_suite.json"
)
DEFAULT_OBSERVED_BAKEOFF = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_862d_full_narrative_s96/"
    "start_conditioned_bakeoff.json"
)
DEFAULT_OBSERVED_CONTRAST = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_862d_full_narrative_s96_contrast/"
    "fixed_start_narrative_contrast.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_reliability_decomposition_863a"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _group_by_start(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("start_name", "")), []).append(row)
    return grouped


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _support_indices(case_summary: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for row in _as_list(case_summary.get("top_support")):
        support = _as_dict(row)
        if support.get("window_index") is not None:
            indices.append(int(support["window_index"]))
    return indices


def _support_weights(case_summary: dict[str, Any]) -> list[float]:
    weights = [
        _as_float(_as_dict(row).get("weight"))
        for row in _as_list(case_summary.get("top_support"))
    ]
    return [weight for weight in weights if weight > 0.0]


def _effective_n(weights: list[float]) -> float:
    total = float(sum(weights))
    if total <= 0.0:
        return 0.0
    normalized = [weight / total for weight in weights]
    denom = sum(weight * weight for weight in normalized)
    return float(1.0 / denom) if denom > 0.0 else 0.0


def _support_jaccard_distance(left: list[int], right: list[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return float(1.0 - (len(left_set & right_set) / len(union)))


def support_summary(case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    support_sets = [_support_indices(row) for row in case_summaries]
    pairwise_distances = [
        _support_jaccard_distance(left, right)
        for left, right in combinations(support_sets, 2)
    ]
    top1 = [indices[0] for indices in support_sets if indices]
    top_cosines = [
        _as_float(_as_dict(_as_list(row.get("top_support"))[0]).get("memory_support_cosine"))
        for row in case_summaries
        if _as_list(row.get("top_support"))
    ]
    effective_ns = [_effective_n(_support_weights(row)) for row in case_summaries]
    support_match_rates = [
        _as_float(row.get("support_match_rate"))
        for row in case_summaries
        if row.get("support_match_rate") is not None
    ]
    return {
        "case_count": len(case_summaries),
        "median_support_jaccard_distance": _safe_median(pairwise_distances),
        "mean_support_jaccard_distance": _safe_mean(pairwise_distances),
        "top1_unique_count": len(set(top1)),
        "support_window_unique_count": len({idx for indices in support_sets for idx in indices}),
        "mean_top_support_cosine": _safe_mean(top_cosines),
        "min_top_support_cosine": min(top_cosines) if top_cosines else 0.0,
        "mean_effective_support_n_top_reported": _safe_mean(effective_ns),
        "mean_support_match_rate": _safe_mean(support_match_rates),
    }


def bakeoff_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    start_distances = [_as_float(row.get("start_distance_z")) for row in rows]
    weighted_start_distances = [
        _as_float(row.get("memory_prior_weighted_start_distance_z")) for row in rows
    ]
    memory_match_rates = [
        _as_float(row.get("memory_prior_support_weighted_match_rate")) for row in rows
    ]
    crps_improvements = [
        _as_float(_as_dict(row.get("scenario_metrics")).get("ensemble_crps_z_improvement_vs_persistence"))
        for row in rows
    ]
    energy_improvements = [
        _as_float(_as_dict(row.get("scenario_metrics")).get("energy_score_z_improvement_vs_persistence"))
        for row in rows
    ]
    operational_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("validation_operational", "unknown"))
        operational_counts[status] = operational_counts.get(status, 0) + 1
    return {
        "case_count": len(rows),
        "mean_query_to_start_distance_z": _safe_mean(start_distances),
        "mean_weighted_support_start_distance_z": _safe_mean(weighted_start_distances),
        "max_weighted_support_start_distance_z": max(weighted_start_distances)
        if weighted_start_distances
        else 0.0,
        "mean_support_weighted_match_rate": _safe_mean(memory_match_rates),
        "mean_crps_improvement_vs_persistence": _safe_mean(crps_improvements),
        "mean_energy_improvement_vs_persistence": _safe_mean(energy_improvements),
        "operational_status_counts": operational_counts,
    }


def _control_by_start(control_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("start_name")): row
        for row in _as_list(control_report.get("per_start_controls"))
        if row.get("start_name") is not None
    }


def _flags_for_start(
    *,
    control: dict[str, Any],
    support: dict[str, Any],
    bakeoff: dict[str, Any],
    max_bootstrap_ratio: float,
    max_repeat_ratio: float,
    min_top_support_cosine: float,
    max_weighted_start_distance_z: float,
) -> list[str]:
    flags: list[str] = []
    if _as_float(control.get("start_only_ratio")) > _as_float(
        control.get("max_start_only_ratio"), 0.5
    ):
        flags.append("start_geometry_leakage")
    if _as_float(control.get("repeat_ratio")) > max_repeat_ratio:
        flags.append("repeat_seed_instability")
    if _as_float(control.get("bootstrap_ratio")) > max_bootstrap_ratio:
        flags.append("rollout_sampling_noise_close_to_narrative_gap")
    if _as_float(support.get("mean_top_support_cosine")) < min_top_support_cosine:
        flags.append("weak_text_memory_support")
    if _as_float(bakeoff.get("mean_weighted_support_start_distance_z")) > max_weighted_start_distance_z:
        flags.append("support_pool_far_from_fixed_start")
    if not flags:
        flags.append("no_blocking_mechanism_detected")
    return flags


def build_reliability_decomposition(
    *,
    control_report: dict[str, Any],
    observed_bakeoff: dict[str, Any],
    observed_contrast: dict[str, Any],
    min_top_support_cosine: float = 0.78,
    max_weighted_start_distance_z: float = 12.0,
) -> dict[str, Any]:
    thresholds = _as_dict(control_report.get("thresholds"))
    max_bootstrap_ratio = _as_float(thresholds.get("max_bootstrap_ratio"), 0.75)
    max_repeat_ratio = _as_float(thresholds.get("max_repeat_ratio"), 0.75)

    controls = _control_by_start(control_report)
    rows_by_start = _group_by_start(_as_list(observed_bakeoff.get("rows")))
    summaries_by_start = _group_by_start(_as_list(observed_contrast.get("case_summaries")))
    all_starts = sorted(set(controls) | set(rows_by_start) | set(summaries_by_start))

    start_rows: list[dict[str, Any]] = []
    for start_name in all_starts:
        control = controls.get(start_name, {})
        support = support_summary(summaries_by_start.get(start_name, []))
        bakeoff = bakeoff_summary(rows_by_start.get(start_name, []))
        flags = _flags_for_start(
            control=control,
            support=support,
            bakeoff=bakeoff,
            max_bootstrap_ratio=max_bootstrap_ratio,
            max_repeat_ratio=max_repeat_ratio,
            min_top_support_cosine=min_top_support_cosine,
            max_weighted_start_distance_z=max_weighted_start_distance_z,
        )
        start_rows.append(
            {
                "start_name": start_name,
                "control_status": control.get("status", "missing"),
                "observed_median_gap": _as_float(control.get("observed_median_gap")),
                "start_only_ratio": _as_float(control.get("start_only_ratio")),
                "bootstrap_ratio": _as_float(control.get("bootstrap_ratio")),
                "repeat_ratio": _as_float(control.get("repeat_ratio")),
                "mechanism_flags": flags,
                "support_summary": support,
                "bakeoff_summary": bakeoff,
            }
        )

    flag_counts: dict[str, int] = {}
    for row in start_rows:
        for flag in row["mechanism_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    status = "pass"
    if any(row["control_status"] == "fail" for row in start_rows):
        status = "warning"
    if flag_counts.get("repeat_seed_instability", 0) or flag_counts.get(
        "rollout_sampling_noise_close_to_narrative_gap", 0
    ):
        status = "warning"
    decision = (
        "Narrative support is active because start-only ratios are zero, but "
        "production trust depends on start-level reliability. Weak starts should "
        "be warned or rerun with stronger sampling/decoder stability before any "
        "new text-to-latent architecture is introduced."
    )
    return {
        "status": status,
        "scope_note": (
            "Post-experiment decomposition of fixed-start narrative-control "
            "warnings. This is diagnostic evidence, not a product-time model."
        ),
        "thresholds": {
            "max_bootstrap_ratio": max_bootstrap_ratio,
            "max_repeat_ratio": max_repeat_ratio,
            "min_top_support_cosine": min_top_support_cosine,
            "max_weighted_start_distance_z": max_weighted_start_distance_z,
        },
        "flag_counts": flag_counts,
        "start_rows": start_rows,
        "decision": decision,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Start Reliability Decomposition",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: {report['decision']}",
        "",
        "## Mechanism Counts",
        "",
        "| Mechanism | Count |",
        "|---|---:|",
    ]
    for flag, count in sorted(report.get("flag_counts", {}).items()):
        lines.append(f"| `{flag}` | `{count}` |")
    lines.extend(
        [
            "",
            "## Per-Start Diagnosis",
            "",
            "| Start | Control | Observed Gap | Bootstrap Ratio | Repeat Ratio | Top Support Cos | Support Start Dist | Flags |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["start_rows"]:
        support = row["support_summary"]
        bakeoff = row["bakeoff_summary"]
        flags = ", ".join(f"`{flag}`" for flag in row["mechanism_flags"])
        lines.append(
            "| "
            f"`{row['start_name']}` | "
            f"`{row['control_status']}` | "
            f"`{row['observed_median_gap']:.3f}` | "
            f"`{row['bootstrap_ratio']:.3f}` | "
            f"`{row['repeat_ratio']:.3f}` | "
            f"`{support['mean_top_support_cosine']:.3f}` | "
            f"`{bakeoff['mean_weighted_support_start_distance_z']:.3f}` | "
            f"{flags} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `start_geometry_leakage` would mean the fixed-start gap can be explained without narrative; it is not observed in the current report.",
            "- `rollout_sampling_noise_close_to_narrative_gap` means the fan-chart/sample estimator is too noisy relative to the measured narrative effect.",
            "- `repeat_seed_instability` means rerunning the same narrative under a different seed moves the scenario nearly as much as changing narratives.",
            "- `support_pool_far_from_fixed_start` means the narrative-compatible support set is numerically far from the user-selected starting level.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", type=Path, default=DEFAULT_CONTROL_REPORT)
    parser.add_argument("--observed-bakeoff-report", type=Path, default=DEFAULT_OBSERVED_BAKEOFF)
    parser.add_argument("--observed-contrast-report", type=Path, default=DEFAULT_OBSERVED_CONTRAST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-top-support-cosine", type=float, default=0.78)
    parser.add_argument("--max-weighted-start-distance-z", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_reliability_decomposition(
        control_report=_load_json(args.control_report),
        observed_bakeoff=_load_json(args.observed_bakeoff_report),
        observed_contrast=_load_json(args.observed_contrast_report),
        min_top_support_cosine=args.min_top_support_cosine,
        max_weighted_start_distance_z=args.max_weighted_start_distance_z,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["artifact_paths"] = {
        "report": str(args.output_dir / "fixed_start_reliability_decomposition.json"),
        "markdown": str(args.output_dir / "fixed_start_reliability_decomposition.md"),
    }
    _write_json(report["artifact_paths"]["report"], report)
    Path(report["artifact_paths"]["markdown"]).write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
                "flag_counts": report["flag_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
