#!/usr/bin/env python
"""Diagnose why some fixed starts damp narrative conditionality."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_BAKEOFF_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_matrix_858b/start_conditioned_bakeoff.json"
)
DEFAULT_CONTRAST_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_contrast_858b/"
    "fixed_start_narrative_contrast.json"
)
DEFAULT_GATE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_conditioning_gate_858b/"
    "fixed_start_conditioning_gate.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_damping_diagnostic_860a"
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


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _rank_desc(value: float, values: list[float]) -> int:
    return 1 + sum(1 for item in values if item > value)


def _extract_operational_variant(run_report: str | Path) -> dict[str, Any]:
    if not run_report:
        return {}
    path = Path(run_report)
    if not path.exists():
        return {}
    report = _load_json(path)
    for row in _as_list(report.get("variant_rows")):
        row = _as_dict(row)
        if bool(row.get("is_operational")):
            return row
    return {}


def _group_bakeoff_rows(bakeoff_report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _as_list(bakeoff_report.get("rows")):
        row = _as_dict(row)
        start_name = str(row.get("start_name", ""))
        if start_name:
            grouped[start_name].append(row)
    return dict(grouped)


def _gate_by_start(gate_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in _as_list(gate_report.get("start_block_assessments")):
        row = _as_dict(row)
        start_name = str(row.get("start_name", ""))
        if start_name:
            output[start_name] = row
    return output


def _contrast_gaps_by_start(
    contrast_report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _as_list(contrast_report.get("pairwise_contrasts")):
        row = _as_dict(row)
        start_name = str(row.get("start_name", ""))
        if start_name:
            grouped[start_name].append(row)
    return dict(grouped)


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _summarize_start(
    *,
    start_name: str,
    rows: list[dict[str, Any]],
    gate_row: dict[str, Any],
    contrast_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    op_rows = [_extract_operational_variant(row.get("run_report", "")) for row in rows]
    op_rows = [row for row in op_rows if row]
    scenario_energy = []
    scenario_crps = []
    start_distances = []
    weighted_start_distances = []
    support_cosines = []
    support_counts: Counter[int] = Counter()
    operational_statuses: Counter[str] = Counter()
    direction_statuses: Counter[str] = Counter()
    support_match_rates = []

    for row in rows:
        metrics = _as_dict(row.get("scenario_metrics"))
        scenario_energy.append(
            _as_float(metrics.get("energy_score_z_improvement_vs_persistence"))
        )
        scenario_crps.append(
            _as_float(metrics.get("ensemble_crps_z_improvement_vs_persistence"))
        )
        start_distances.append(_as_float(row.get("start_distance_z")))
        weighted_start_distances.append(
            _as_float(row.get("memory_prior_weighted_start_distance_z"))
        )
        operational_statuses[str(row.get("validation_operational", ""))] += 1
        direction_statuses[str(row.get("memory_prior_direction_status", ""))] += 1
        support_match_rates.append(
            _as_float(row.get("memory_prior_support_weighted_match_rate"))
        )

    for row in op_rows:
        support_cosines.append(_as_float(row.get("memory_support_cosine")))
        top_window = row.get("memory_prior_top_window_index")
        if top_window is not None:
            try:
                support_counts[int(top_window)] += 1
            except (TypeError, ValueError):
                pass

    contrast_gaps = [
        _as_float(row.get("standardized_l2_gap")) for row in contrast_rows
    ]
    warnings = [str(item) for item in _as_list(gate_row.get("warnings"))]
    return {
        "start_name": start_name,
        "status": str(gate_row.get("status", "")),
        "run_count": len(rows),
        "damping_warning": "start_dampens_narrative_influence" in warnings,
        "warnings": warnings,
        "failures": [str(item) for item in _as_list(gate_row.get("failures"))],
        "operational_status_counts": _counter_dict(operational_statuses),
        "direction_status_counts": _counter_dict(direction_statuses),
        "mean_start_distance_z": _mean(start_distances),
        "mean_weighted_support_start_distance_z": _mean(weighted_start_distances),
        "mean_memory_support_cosine": _mean(support_cosines),
        "min_memory_support_cosine": min(support_cosines) if support_cosines else 0.0,
        "unique_top_support_window_count": len(support_counts),
        "top_support_window_counts": _counter_dict(support_counts),
        "mean_support_match_rate": _mean(support_match_rates),
        "mean_energy_improvement_vs_persistence": _mean(scenario_energy),
        "mean_crps_improvement_vs_persistence": _mean(scenario_crps),
        "max_standardized_l2_gap": _as_float(gate_row.get("max_standardized_l2_gap")),
        "median_standardized_l2_gap": _as_float(
            gate_row.get("median_standardized_l2_gap")
        ),
        "min_standardized_l2_gap": _as_float(gate_row.get("min_standardized_l2_gap")),
        "pair_count": int(gate_row.get("pair_count", 0) or len(contrast_gaps)),
        "top_separation_markets": _as_list(gate_row.get("top_separation_markets")),
    }


def _add_relative_ranks(rows: list[dict[str, Any]]) -> None:
    metrics = {
        "start_distance_rank_high_to_low": "mean_start_distance_z",
        "weighted_support_start_distance_rank_high_to_low": (
            "mean_weighted_support_start_distance_z"
        ),
        "memory_support_cosine_rank_high_to_low": "mean_memory_support_cosine",
        "narrative_gap_rank_high_to_low": "median_standardized_l2_gap",
    }
    for rank_name, metric_name in metrics.items():
        values = [_as_float(row.get(metric_name)) for row in rows]
        for row in rows:
            row[rank_name] = _rank_desc(_as_float(row.get(metric_name)), values)


def _mechanism_notes(rows: list[dict[str, Any]]) -> None:
    passing = [row for row in rows if not row.get("damping_warning")]
    passing_gap_median = _median(
        [_as_float(row.get("median_standardized_l2_gap")) for row in passing]
    )
    passing_support_cosine = _mean(
        [_as_float(row.get("mean_memory_support_cosine")) for row in passing]
    )
    passing_weighted_start = _mean(
        [_as_float(row.get("mean_weighted_support_start_distance_z")) for row in passing]
    )
    for row in rows:
        notes: list[str] = []
        op_counts = _as_dict(row.get("operational_status_counts"))
        warning_count = int(op_counts.get("warning", 0) or 0)
        fail_count = int(op_counts.get("fail", 0) or 0)
        if warning_count or fail_count:
            notes.append(
                "start_compatibility_warning: "
                f"{warning_count} warning and {fail_count} fail rows; "
                f"mean start distance z={_as_float(row.get('mean_start_distance_z')):.3f}"
            )
        if row.get("damping_warning"):
            if _as_float(row.get("median_standardized_l2_gap")) < passing_gap_median:
                notes.append(
                    "low_narrative_separation: "
                    f"median gap={_as_float(row.get('median_standardized_l2_gap')):.3f} "
                    f"below passing-start median={passing_gap_median:.3f}"
                )
            if _as_float(row.get("mean_memory_support_cosine")) < passing_support_cosine:
                notes.append(
                    "weaker_memory_support: "
                    f"mean cosine={_as_float(row.get('mean_memory_support_cosine')):.3f} "
                    f"below passing-start mean={passing_support_cosine:.3f}"
                )
            if (
                _as_float(row.get("mean_weighted_support_start_distance_z"))
                > passing_weighted_start
            ):
                notes.append(
                    "farther_support_pool_from_start: "
                    "weighted support-start distance "
                    f"z={_as_float(row.get('mean_weighted_support_start_distance_z')):.3f} "
                    f"above passing-start mean={passing_weighted_start:.3f}"
                )
        if not notes:
            notes.append(
                "no_damping_mechanism_flagged: same-start narrative changes remain "
                "measurable and validation warnings are not concentrated here"
            )
        row["mechanism_notes"] = notes


def build_start_damping_diagnostic(
    *,
    bakeoff_report: dict[str, Any],
    contrast_report: dict[str, Any],
    gate_report: dict[str, Any],
) -> dict[str, Any]:
    """Build a no-new-knob diagnostic of fixed-start narrative damping."""
    grouped = _group_bakeoff_rows(bakeoff_report)
    gate_rows = _gate_by_start(gate_report)
    contrast_rows = _contrast_gaps_by_start(contrast_report)
    start_summaries = [
        _summarize_start(
            start_name=start_name,
            rows=rows,
            gate_row=gate_rows.get(start_name, {}),
            contrast_rows=contrast_rows.get(start_name, []),
        )
        for start_name, rows in grouped.items()
    ]
    _add_relative_ranks(start_summaries)
    _mechanism_notes(start_summaries)
    start_summaries = sorted(
        start_summaries,
        key=lambda row: (
            bool(row.get("damping_warning")),
            _as_float(row.get("median_standardized_l2_gap")),
        ),
    )
    damped = [row for row in start_summaries if row.get("damping_warning")]
    passing = [row for row in start_summaries if not row.get("damping_warning")]
    return {
        "status": "warning" if damped else "pass",
        "scope_note": (
            "Start-damping diagnostic for the fixed-start narrative matrix. This "
            "does not tune the model; it explains when the selected starting "
            "level dominates or weakens narrative conditionality."
        ),
        "headline": {
            "start_count": len(start_summaries),
            "damped_start_count": len(damped),
            "passing_start_count": len(passing),
            "run_count": sum(int(row.get("run_count", 0)) for row in start_summaries),
            "damped_starts": [row["start_name"] for row in damped],
            "passing_starts": [row["start_name"] for row in passing],
        },
        "interpretation": [
            (
                "A damped start is not a hard model failure: the same starting "
                "level is still respected and direction checks can pass, but "
                "different narratives move the generated distribution less than "
                "they do for better-supported starts."
            ),
            (
                "For production, this should become a trust warning on the chosen "
                "starting level, not a new user-facing research knob."
            ),
        ],
        "start_summaries": start_summaries,
    }


def _fmt(value: Any) -> str:
    return f"{_as_float(value):.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    headline = _as_dict(report.get("headline"))
    lines = [
        "# Fixed-Start Narrative Damping Diagnostic",
        "",
        str(report.get("scope_note", "")),
        "",
        "## Headline",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Starts: `{headline.get('start_count')}`",
        f"- Damped starts: `{', '.join(_as_list(headline.get('damped_starts')))}`",
        f"- Passing starts: `{', '.join(_as_list(headline.get('passing_starts')))}`",
        f"- Runs: `{headline.get('run_count')}`",
        "",
        "## Interpretation",
        "",
    ]
    for item in _as_list(report.get("interpretation")):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Start Diagnostics",
            "",
            "| Start | Status | Median Gap | Start Dist | Support Dist | Support Cos | Operational | Mechanism |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in _as_list(report.get("start_summaries")):
        row = _as_dict(row)
        notes = "<br>".join(str(item) for item in _as_list(row.get("mechanism_notes")))
        lines.append(
            f"| `{row.get('start_name')}` | `{row.get('status')}` | "
            f"`{_fmt(row.get('median_standardized_l2_gap'))}` | "
            f"`{_fmt(row.get('mean_start_distance_z'))}` | "
            f"`{_fmt(row.get('mean_weighted_support_start_distance_z'))}` | "
            f"`{_fmt(row.get('mean_memory_support_cosine'))}` | "
            f"`{row.get('operational_status_counts', {})}` | {notes} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bakeoff-report", type=Path, default=DEFAULT_BAKEOFF_REPORT)
    parser.add_argument("--contrast-report", type=Path, default=DEFAULT_CONTRAST_REPORT)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = build_start_damping_diagnostic(
        bakeoff_report=_load_json(args.bakeoff_report),
        contrast_report=_load_json(args.contrast_report),
        gate_report=_load_json(args.gate_report),
    )
    report["inputs"] = {
        "bakeoff_report": str(args.bakeoff_report),
        "contrast_report": str(args.contrast_report),
        "gate_report": str(args.gate_report),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "fixed_start_damping_diagnostic.json"
    markdown_path = args.output_dir / "fixed_start_damping_diagnostic.md"
    _write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "damped_starts": report["headline"]["damped_starts"],
                "report": str(json_path),
                "markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
