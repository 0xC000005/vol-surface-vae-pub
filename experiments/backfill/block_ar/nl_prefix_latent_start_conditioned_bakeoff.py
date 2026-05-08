#!/usr/bin/env python
"""Scenario-quality bakeoff for fixed-start narrative prefix mixtures.

This script makes no OpenAI calls. It reuses cached condition reports, fixes the
initial joint39 level first, then compares memory-prior and prefix-prior
variants through the same frozen generator rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_start_conditioned_acceptance import (  # noqa: E402
    DEFENSIVE_REPORT,
    DEFAULT_MATRIX,
    FRAGILE_REPORT,
    RATES_REPORT,
    _operational_row,
    _operational_score_row,
    _score_metrics,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    build_prefix_latent_run_args,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_conditioned_bakeoff_816a"
)

EXPANDED_MATRIX = [
    {
        "case_name": "fragile_risk_on",
        "start_name": "validated_start_18",
        "condition_report": FRAGILE_REPORT,
        "candidate_index": 18,
    },
    {
        "case_name": "fragile_risk_on",
        "start_name": "alternate_start_0",
        "condition_report": FRAGILE_REPORT,
        "candidate_index": 0,
    },
    {
        "case_name": "fragile_risk_on",
        "start_name": "balanced_policy_start_40",
        "condition_report": FRAGILE_REPORT,
        "candidate_index": 40,
    },
    {
        "case_name": "defensive_risk_off",
        "start_name": "validated_start_22",
        "condition_report": DEFENSIVE_REPORT,
        "candidate_index": 22,
    },
    {
        "case_name": "defensive_risk_off",
        "start_name": "balanced_policy_start_77",
        "condition_report": DEFENSIVE_REPORT,
        "candidate_index": 77,
    },
    {
        "case_name": "defensive_risk_off",
        "start_name": "memory_nearest_start_0",
        "condition_report": DEFENSIVE_REPORT,
        "candidate_index": 0,
    },
    {
        "case_name": "rates_selloff",
        "start_name": "validated_start_18",
        "condition_report": RATES_REPORT,
        "candidate_index": 18,
    },
    {
        "case_name": "rates_selloff",
        "start_name": "balanced_policy_start_178",
        "condition_report": RATES_REPORT,
        "candidate_index": 178,
    },
]

DEFAULT_VARIANTS = [
    {
        "variant_name": "decoder_soft_topk_combined",
        "memory_prior_mode": "soft_topk_combined",
        "prefix_prior_mode": "decoder",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 1.0,
    },
    {
        "variant_name": "decoder_soft_topk_memory",
        "memory_prior_mode": "soft_topk_memory",
        "prefix_prior_mode": "decoder",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 1.0,
    },
    {
        "variant_name": "decoder_diverse_topk_combined",
        "memory_prior_mode": "diverse_topk_combined",
        "prefix_prior_mode": "decoder",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 1.0,
    },
    {
        "variant_name": "feature_soft_topk_combined",
        "memory_prior_mode": "soft_topk_combined",
        "prefix_prior_mode": "feature_mixture",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 1.0,
    },
]

TEMPERATURE_CALIBRATION_VARIANTS = [
    {
        "variant_name": "decoder_soft_topk_combined_gen_temp_0p50",
        "memory_prior_mode": "soft_topk_combined",
        "prefix_prior_mode": "decoder",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 0.5,
    },
    {
        "variant_name": "decoder_soft_topk_combined_gen_temp_0p75",
        "memory_prior_mode": "soft_topk_combined",
        "prefix_prior_mode": "decoder",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 0.75,
    },
    {
        "variant_name": "decoder_soft_topk_combined_gen_temp_1p00",
        "memory_prior_mode": "soft_topk_combined",
        "prefix_prior_mode": "decoder",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 1.0,
    },
    {
        "variant_name": "decoder_soft_topk_combined_gen_temp_1p25",
        "memory_prior_mode": "soft_topk_combined",
        "prefix_prior_mode": "decoder",
        "top_k": 8,
        "temperature": 0.2,
        "generator_temperature": 1.25,
    },
]


VARIANT_SETS = {
    "prior": DEFAULT_VARIANTS,
    "temperature": TEMPERATURE_CALIBRATION_VARIANTS,
}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_case_spec_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{path}: expected non-empty case list")
    cases: list[dict[str, Any]] = []
    required = {"case_name", "start_name", "condition_report", "candidate_index"}
    for index, row in enumerate(raw_cases):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: case {index} is not an object")
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{path}: case {index} missing fields {missing}")
        cases.append(
            {
                "case_name": str(row["case_name"]),
                "start_name": str(row["start_name"]),
                "condition_report": str(row["condition_report"]),
                "candidate_index": int(row["candidate_index"]),
            }
        )
    return cases


def selected_historical_cases(
    case_count: int | None = None,
    *,
    case_set: str = "default",
    case_spec_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    if case_spec_json:
        rows = _load_case_spec_json(case_spec_json)
    elif str(case_set) == "default":
        rows = [case for case in DEFAULT_MATRIX if "candidate_index" in case]
    elif str(case_set) == "expanded":
        rows = list(EXPANDED_MATRIX)
    else:
        raise ValueError(f"unknown case set: {case_set!r}")
    if case_count:
        return rows[: int(case_count)]
    return rows


def selected_variants(
    variant_count: int | None = None,
    *,
    variant_set: str = "prior",
) -> list[dict[str, Any]]:
    variants = VARIANT_SETS.get(str(variant_set))
    if variants is None:
        raise ValueError(f"unknown variant set: {variant_set!r}")
    if variant_count:
        return variants[: int(variant_count)]
    return list(variants)


def _safe_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def row_from_report(
    *,
    case: dict[str, Any],
    variant: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    gate = report.get("validation_gate", {})
    operational = _operational_row(report)
    score_row = _operational_score_row(report, operational)
    metrics = _score_metrics(score_row)
    return {
        "case_name": str(case.get("case_name", "")),
        "start_name": str(case.get("start_name", "")),
        "candidate_index": int(case["candidate_index"]),
        "variant_name": str(variant["variant_name"]),
        "memory_prior_mode": str(variant["memory_prior_mode"]),
        "prefix_prior_mode": str(variant["prefix_prior_mode"]),
        "top_k": int(variant["top_k"]),
        "temperature": float(variant["temperature"]),
        "generator_temperature": float(variant.get("generator_temperature", 1.0)),
        "validation_operational": str(gate.get("operational_status", "")),
        "validation_overall": str(gate.get("overall_status", "")),
        "target_available": bool(metrics.get("target_available")),
        "scenario_metrics": metrics,
        "start_distance_z": float(operational.get("start_distance_z", 0.0) or 0.0),
        "memory_prior_weighted_start_distance_z": float(
            operational.get("memory_prior_weighted_start_distance_z", 0.0) or 0.0
        ),
        "memory_prior_analogue_count": int(
            operational.get("memory_prior_analogue_count", 0) or 0
        ),
        "run_report": str(report.get("artifact_paths", {}).get("report", "")),
    }


def summarize_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant_name"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for variant_name, variant_rows in sorted(by_variant.items()):
        target_rows = [row for row in variant_rows if bool(row.get("target_available"))]
        metrics = [
            row.get("scenario_metrics", {})
            for row in target_rows
            if isinstance(row.get("scenario_metrics"), dict)
        ]
        status_counts: dict[str, int] = {}
        for row in variant_rows:
            status = str(row.get("validation_operational", ""))
            status_counts[status] = status_counts.get(status, 0) + 1
        summary_rows.append(
            {
                "variant_name": variant_name,
                "run_count": int(len(variant_rows)),
                "target_count": int(len(target_rows)),
                "operational_status_counts": status_counts,
                "mean_energy_score_z": _safe_mean(
                    [
                        metric.get("energy_score_z")
                        for metric in metrics
                        if metric.get("energy_score_z") is not None
                    ]
                ),
                "mean_ensemble_crps_z": _safe_mean(
                    [
                        metric.get("ensemble_crps_z")
                        for metric in metrics
                        if metric.get("ensemble_crps_z") is not None
                    ]
                ),
                "mean_energy_improvement_vs_persistence": _safe_mean(
                    [
                        metric.get("energy_score_z_improvement_vs_persistence")
                        for metric in metrics
                        if metric.get("energy_score_z_improvement_vs_persistence")
                        is not None
                    ]
                ),
                "mean_crps_improvement_vs_persistence": _safe_mean(
                    [
                        metric.get("ensemble_crps_z_improvement_vs_persistence")
                        for metric in metrics
                        if metric.get("ensemble_crps_z_improvement_vs_persistence")
                        is not None
                    ]
                ),
                "mean_weighted_start_distance_z": _safe_mean(
                    [
                        row.get("memory_prior_weighted_start_distance_z")
                        for row in variant_rows
                    ]
                ),
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (
            float("inf")
            if row["mean_ensemble_crps_z"] is None
            else float(row["mean_ensemble_crps_z"])
        ),
    )


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    metrics = row.get("scenario_metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def oracle_select_best_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str = "ensemble_crps_z",
) -> list[dict[str, Any]]:
    """Select the best realized-future variant per case/start pair.

    This is an upper-bound diagnostic only. It uses realized-future metrics and
    must not be used by the live product path.
    """

    by_case: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if not bool(row.get("target_available")):
            continue
        key = (
            str(row.get("case_name", "")),
            str(row.get("start_name", "")),
            int(row.get("candidate_index", -1)),
        )
        by_case.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(by_case):
        candidates = by_case[key]
        valid = [row for row in candidates if _metric_value(row, metric) is not None]
        if not valid:
            continue
        selected.append(
            min(
                valid,
                key=lambda row: (
                    float(_metric_value(row, metric) or float("inf")),
                    str(row.get("variant_name", "")),
                ),
            )
        )
    return selected


def summarize_oracle_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = oracle_select_best_rows(rows)
    metrics = [
        row.get("scenario_metrics", {})
        for row in selected
        if isinstance(row.get("scenario_metrics"), dict)
    ]
    chosen_counts: dict[str, int] = {}
    for row in selected:
        variant = str(row.get("variant_name", ""))
        chosen_counts[variant] = chosen_counts.get(variant, 0) + 1
    return {
        "selector": "realized_future_best_crps_upper_bound",
        "scope_note": (
            "Diagnostic only. This selector uses realized-future CRPS and is not "
            "available to the live product path."
        ),
        "selected_count": int(len(selected)),
        "chosen_variant_counts": chosen_counts,
        "mean_energy_score_z": _safe_mean(
            [
                metric.get("energy_score_z")
                for metric in metrics
                if metric.get("energy_score_z") is not None
            ]
        ),
        "mean_ensemble_crps_z": _safe_mean(
            [
                metric.get("ensemble_crps_z")
                for metric in metrics
                if metric.get("ensemble_crps_z") is not None
            ]
        ),
        "mean_energy_improvement_vs_persistence": _safe_mean(
            [
                metric.get("energy_score_z_improvement_vs_persistence")
                for metric in metrics
                if metric.get("energy_score_z_improvement_vs_persistence") is not None
            ]
        ),
        "mean_crps_improvement_vs_persistence": _safe_mean(
            [
                metric.get("ensemble_crps_z_improvement_vs_persistence")
                for metric in metrics
                if metric.get("ensemble_crps_z_improvement_vs_persistence") is not None
            ]
        ),
        "selected_rows": [
            {
                "case_name": row.get("case_name"),
                "start_name": row.get("start_name"),
                "candidate_index": row.get("candidate_index"),
                "chosen_variant": row.get("variant_name"),
                "energy_score_z": _metric_value(row, "energy_score_z"),
                "ensemble_crps_z": _metric_value(row, "ensemble_crps_z"),
                "energy_score_z_improvement_vs_persistence": _metric_value(
                    row, "energy_score_z_improvement_vs_persistence"
                ),
                "ensemble_crps_z_improvement_vs_persistence": _metric_value(
                    row, "ensemble_crps_z_improvement_vs_persistence"
                ),
            }
            for row in selected
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Start Prefix-Mixture Bakeoff",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Case set: `{summary.get('case_set', 'default')}`",
        f"- Variant set: `{summary.get('variant_set', 'prior')}`",
        f"- Case count: `{summary.get('case_count')}`",
        f"- Variant count: `{summary.get('variant_count')}`",
        f"- Run count: `{summary.get('run_count')}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Runs | Targets | Status Counts | Mean Energy z | Mean CRPS z | Mean Energy Imp | Mean CRPS Imp | Mean Weighted Start z |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("variant_summary", []):
        status_counts = json.dumps(
            row.get("operational_status_counts", {}), sort_keys=True
        )
        lines.append(
            f"| `{row.get('variant_name')}` | "
            f"`{row.get('run_count')}` | "
            f"`{row.get('target_count')}` | "
            f"`{status_counts}` | "
            f"`{_format_optional(row.get('mean_energy_score_z'))}` | "
            f"`{_format_optional(row.get('mean_ensemble_crps_z'))}` | "
            f"`{_format_optional(row.get('mean_energy_improvement_vs_persistence'))}` | "
            f"`{_format_optional(row.get('mean_crps_improvement_vs_persistence'))}` | "
            f"`{_format_optional(row.get('mean_weighted_start_distance_z'))}` |"
        )
    oracle = summary.get("oracle_selection_summary", {})
    if isinstance(oracle, dict) and oracle:
        lines.extend(
            [
                "",
                "## Realized-Future Oracle Selector",
                "",
                f"- Selector: `{oracle.get('selector')}`",
                f"- Scope: {oracle.get('scope_note')}",
                f"- Selected rows: `{oracle.get('selected_count')}`",
                f"- Chosen variant counts: `{json.dumps(oracle.get('chosen_variant_counts', {}), sort_keys=True)}`",
                f"- Mean energy z: `{_format_optional(oracle.get('mean_energy_score_z'))}`",
                f"- Mean CRPS z: `{_format_optional(oracle.get('mean_ensemble_crps_z'))}`",
                f"- Mean energy improvement vs persistence: `{_format_optional(oracle.get('mean_energy_improvement_vs_persistence'))}`",
                f"- Mean CRPS improvement vs persistence: `{_format_optional(oracle.get('mean_crps_improvement_vs_persistence'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Case Rows",
            "",
            "| Case | Start | Variant | Status | Target | Energy z | CRPS z | Energy Imp | CRPS Imp | Weighted Start z | Report |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary.get("rows", []):
        metrics = row.get("scenario_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        lines.append(
            f"| `{row.get('case_name')}` | "
            f"`{row.get('start_name')}` | "
            f"`{row.get('variant_name')}` | "
            f"`{row.get('validation_operational')}` | "
            f"`{bool(row.get('target_available'))}` | "
            f"`{_format_optional(metrics.get('energy_score_z'))}` | "
            f"`{_format_optional(metrics.get('ensemble_crps_z'))}` | "
            f"`{_format_optional(metrics.get('energy_score_z_improvement_vs_persistence'))}` | "
            f"`{_format_optional(metrics.get('ensemble_crps_z_improvement_vs_persistence'))}` | "
            f"`{_format_optional(row.get('memory_prior_weighted_start_distance_z'))}` | "
            f"`{row.get('run_report')}` |"
        )
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def run_bakeoff(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = selected_historical_cases(
        args.case_count,
        case_set=args.case_set,
        case_spec_json=getattr(args, "case_spec_json", None),
    )
    variants = selected_variants(args.variant_count, variant_set=args.variant_set)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for variant in variants:
            case_dir = (
                output_dir
                / str(case["case_name"])
                / str(case["start_name"])
                / str(variant["variant_name"])
            )
            run_args = build_prefix_latent_run_args(
                start_mode="explicit_start_window",
                samples=int(args.samples),
                condition_report=str(case["condition_report"]),
                explicit_start_window_index=int(case["candidate_index"]),
                output_dir=str(case_dir),
            )
            run_args.steps = int(args.steps)
            run_args.chunk_size = int(args.chunk_size)
            run_args.device = str(args.device)
            run_args.memory_prior_mode = str(variant["memory_prior_mode"])
            run_args.memory_prior_top_k = int(variant["top_k"])
            run_args.memory_prior_temperature = float(variant["temperature"])
            run_args.prefix_prior_mode = str(variant["prefix_prior_mode"])
            run_args.temperature = float(variant.get("generator_temperature", 1.0))
            report = run_prefix_latent_story_smoke(run_args)
            rows.append(row_from_report(case=case, variant=variant, report=report))
    variant_summary = summarize_by_variant(rows)
    oracle_selection_summary = summarize_oracle_selection(rows)
    status = (
        "pass" if rows and all(row.get("target_available") for row in rows) else "fail"
    )
    output = {
        "status": status,
        "scope_note": (
            "No OpenAI calls. Compares fixed-start narrative mixture variants "
            "using realized-future metrics where historical starts are selected."
        ),
        "case_count": int(len(cases)),
        "case_set": (
            "custom" if getattr(args, "case_spec_json", None) else str(args.case_set)
        ),
        "case_spec_json": str(getattr(args, "case_spec_json", "") or ""),
        "variant_count": int(len(variants)),
        "variant_set": str(args.variant_set),
        "run_count": int(len(rows)),
        "rows": rows,
        "variant_summary": variant_summary,
        "oracle_selection_summary": oracle_selection_summary,
        "artifact_paths": {
            "report": str(output_dir / "start_conditioned_bakeoff.json"),
            "markdown": str(output_dir / "start_conditioned_bakeoff.md"),
        },
    }
    _write_json(output["artifact_paths"]["report"], output)
    Path(output["artifact_paths"]["markdown"]).write_text(
        render_markdown(output).rstrip() + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument(
        "--case-set", choices=["default", "expanded"], default="default"
    )
    parser.add_argument(
        "--case-spec-json",
        help=(
            "Optional JSON case list with case_name, start_name, condition_report, "
            "and candidate_index. When provided, it overrides --case-set."
        ),
    )
    parser.add_argument("--variant-count", type=int, default=4)
    parser.add_argument("--variant-set", choices=sorted(VARIANT_SETS), default="prior")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_bakeoff(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "case_count": summary["case_count"],
                "variant_count": summary["variant_count"],
                "run_count": summary["run_count"],
                "best_variant": (
                    summary["variant_summary"][0]["variant_name"]
                    if summary["variant_summary"]
                    else ""
                ),
                "report": summary["artifact_paths"]["report"],
                "markdown": summary["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
