#!/usr/bin/env python
"""Ablate whether grounding has become a narrative bottleneck.

This script reuses cached condition-only reports and builds alternative text
condition channels:

- raw narrative only;
- implications only;
- existing grounded condition text;
- raw narrative plus explicit implications;
- full narrative plus grounding sidecar.

It then runs the existing fixed-start bakeoff with the same start grid and
calibrated rollout setting. The goal is to test whether explicit market
implications are acting as a useful guardrail or as a lossy replacement for the
full narrative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_condition_only_report import (  # noqa: E402
    QUERY_CHANNELS,
    build_condition_report,
    condition_query_text_for_channel,
    project_condition_query_text,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_conditioned_bakeoff import (  # noqa: E402
    run_bakeoff,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_sensitivity import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ADAPTER,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    load_bridge_arrays,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
)


DEFAULT_CASE_SPEC_JSON = "autoresearch-session/new_condition_fixed_start_grid_823c.json"
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_grounding_channel_ablation_845a"
)
DEFAULT_CHANNELS = (
    "grounded_condition",
    "raw_narrative",
    "implications_only",
    "narrative_plus_implications",
    "narrative_plus_grounding",
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


def _safe_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _format_optional(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def load_case_spec(path: str | Path, *, case_count: int | None = None) -> list[dict[str, Any]]:
    payload = _load_json(path)
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: expected non-empty cases list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: case {index} is not an object")
        rows.append(
            {
                "case_name": str(row["case_name"]),
                "start_name": str(row["start_name"]),
                "condition_report": str(row["condition_report"]),
                "candidate_index": int(row["candidate_index"]),
            }
        )
    if case_count is not None and int(case_count) > 0:
        return rows[: int(case_count)]
    return rows


def condition_case_from_report(report_path: str | Path) -> dict[str, Any]:
    """Convert a condition report back into a condition-grounding case payload."""

    report = _load_json(report_path)
    query = report.get("cached_query", {})
    if not isinstance(query, dict):
        raise ValueError(f"{report_path}: missing cached_query")
    grounding = query.get("grounding", {})
    if not isinstance(grounding, dict):
        raise ValueError(f"{report_path}: missing grounding")
    condition_only_grounding = grounding.get("condition_only_grounding", grounding)
    if not isinstance(condition_only_grounding, dict):
        raise ValueError(f"{report_path}: missing condition_only_grounding")
    metadata = query.get("embedding_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "case_name": str(report.get("case_name") or query.get("window_id", "")),
        "story": str(query.get("narrative_text", "")),
        "candidate_query_text": str(query.get("query_text", "")),
        "condition_only_grounding": condition_only_grounding,
        "condition_only_validation": grounding.get(
            "condition_only_validation",
            {},
        ),
        "metadata": {"model": str(metadata.get("grounding_model", ""))},
        "story_split": grounding.get("story_split", {}),
        "source_condition_report": str(report_path),
    }


def unique_source_reports(case_rows: list[dict[str, Any]]) -> dict[str, str]:
    reports: dict[str, str] = {}
    for row in case_rows:
        case_name = str(row["case_name"])
        path = str(row["condition_report"])
        prior = reports.get(case_name)
        if prior and prior != path:
            raise ValueError(f"case {case_name!r} has multiple condition reports")
        reports[case_name] = path
    return reports


def build_channel_reports(
    *,
    case_rows: list[dict[str, Any]],
    query_channels: list[str],
    output_dir: str | Path,
    bridge_arrays: str | Path,
    bridge_adapter: str | Path,
    embedding_model: str,
    dotenv: str | Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    """Build condition reports for every case/channel pair."""

    arrays = load_bridge_arrays(bridge_arrays)
    condition_dim = int(np.asarray(arrays["memory_targets"]).shape[1])
    source_reports = unique_source_reports(case_rows)
    output = Path(output_dir)
    report_paths: dict[str, dict[str, str]] = {}
    channel_records: list[dict[str, Any]] = []
    for case_name, report_path in sorted(source_reports.items()):
        case = condition_case_from_report(report_path)
        report_paths[case_name] = {}
        for channel in query_channels:
            query_text = condition_query_text_for_channel(case, query_channel=channel)
            projection = project_condition_query_text(
                query_text=query_text,
                embedding_model=str(embedding_model),
                bridge_adapter=bridge_adapter,
                condition_dim=condition_dim,
                dotenv_path=dotenv,
            )
            channel_dir = output / "condition_reports" / channel / case_name
            report = build_condition_report(
                case=case,
                projection=projection,
                output_dir=channel_dir,
                bridge_arrays=bridge_arrays,
                bridge_adapter=bridge_adapter,
                query_text=query_text,
                query_channel=channel,
            )
            report_paths[case_name][channel] = str(report["artifact_paths"]["report"])
            channel_records.append(
                {
                    "case_name": case_name,
                    "query_channel": channel,
                    "source_condition_report": str(report_path),
                    "condition_report": str(report["artifact_paths"]["report"]),
                    "query_text_length": int(len(query_text)),
                    "query_text_line_count": int(query_text.count("\n") + 1),
                    "embedding_model": str(embedding_model),
                }
            )
    return report_paths, channel_records


def write_channel_case_specs(
    *,
    case_rows: list[dict[str, Any]],
    report_paths: dict[str, dict[str, str]],
    query_channels: list[str],
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir)
    paths: dict[str, str] = {}
    for channel in query_channels:
        channel_cases: list[dict[str, Any]] = []
        for row in case_rows:
            case_name = str(row["case_name"])
            channel_cases.append(
                {
                    **row,
                    "case_name": f"{case_name}__{channel}",
                    "condition_report": report_paths[case_name][channel],
                }
            )
        path = output / "case_specs" / f"{channel}_case_spec.json"
        _write_json(path, {"cases": channel_cases})
        paths[channel] = str(path)
    return paths


def summarize_channel_bakeoff(
    *,
    channel: str,
    bakeoff: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in bakeoff.get("rows", [])
        if isinstance(row, dict) and bool(row.get("target_available"))
    ]
    metrics = [
        row.get("scenario_metrics", {})
        for row in rows
        if isinstance(row.get("scenario_metrics"), dict)
    ]
    status_counts: dict[str, int] = {}
    for row in bakeoff.get("rows", []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("validation_operational", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "query_channel": str(channel),
        "run_count": int(bakeoff.get("run_count", 0)),
        "target_count": int(len(rows)),
        "operational_status_counts": status_counts,
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
        "mean_weighted_start_distance_z": _safe_mean(
            [
                row.get("memory_prior_weighted_start_distance_z")
                for row in bakeoff.get("rows", [])
                if isinstance(row, dict)
            ]
        ),
        "bakeoff_report": str(bakeoff.get("artifact_paths", {}).get("report", "")),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Grounding Channel Ablation",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Case count: `{summary.get('case_count')}`",
        f"- Query channels: `{', '.join(summary.get('query_channels', []))}`",
        f"- Samples: `{summary.get('samples')}`",
        f"- Steps: `{summary.get('steps')}`",
        f"- Device: `{summary.get('device')}`",
        "",
        "## Channel Summary",
        "",
        "| Channel | Runs | Targets | Status Counts | Mean Energy Imp | Mean CRPS Imp | Mean Weighted Start z | Report |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in summary.get("channel_summary", []):
        lines.append(
            f"| `{row.get('query_channel')}` | "
            f"`{row.get('run_count')}` | "
            f"`{row.get('target_count')}` | "
            f"`{json.dumps(row.get('operational_status_counts', {}), sort_keys=True)}` | "
            f"`{_format_optional(row.get('mean_energy_improvement_vs_persistence'))}` | "
            f"`{_format_optional(row.get('mean_crps_improvement_vs_persistence'))}` | "
            f"`{_format_optional(row.get('mean_weighted_start_distance_z'))}` | "
            f"`{row.get('bakeoff_report')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Grounding is not treated as a replacement for the narrative. "
            "This ablation compares whether raw narrative, implication-only, "
            "existing grounded condition text, raw narrative plus explicit "
            "implications, or full narrative plus grounding sidecar gives the "
            "best fixed-start support and scenario metrics.",
        ]
    )
    return "\n".join(lines)


def run_grounding_channel_ablation(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    channels = [str(channel) for channel in args.query_channel]
    for channel in channels:
        if channel not in QUERY_CHANNELS:
            raise ValueError(f"unknown query channel: {channel}")
    case_rows = load_case_spec(args.case_spec_json, case_count=args.case_count)
    report_paths, channel_records = build_channel_reports(
        case_rows=case_rows,
        query_channels=channels,
        output_dir=output,
        bridge_arrays=args.bridge_arrays,
        bridge_adapter=args.bridge_adapter,
        embedding_model=args.embedding_model,
        dotenv=args.dotenv,
    )
    case_spec_paths = write_channel_case_specs(
        case_rows=case_rows,
        report_paths=report_paths,
        query_channels=channels,
        output_dir=output,
    )
    channel_summaries: list[dict[str, Any]] = []
    bakeoff_reports: dict[str, dict[str, Any]] = {}
    for channel in channels:
        bakeoff = run_bakeoff(
            SimpleNamespace(
                output_dir=str(output / "bakeoff" / channel),
                case_count=len(case_rows),
                case_set="default",
                case_spec_json=case_spec_paths[channel],
                variant_count=1,
                variant_set="temperature",
                samples=int(args.samples),
                steps=int(args.steps),
                chunk_size=int(args.chunk_size),
                device=str(args.device),
            )
        )
        bakeoff_reports[channel] = {
            "status": bakeoff.get("status"),
            "artifact_paths": bakeoff.get("artifact_paths", {}),
        }
        channel_summaries.append(
            summarize_channel_bakeoff(channel=channel, bakeoff=bakeoff)
        )
    channel_summaries = sorted(
        channel_summaries,
        key=lambda row: (
            row.get("mean_crps_improvement_vs_persistence") is None,
            0.0
            if row.get("mean_crps_improvement_vs_persistence") is None
            else -float(row["mean_crps_improvement_vs_persistence"]),
        ),
    )
    status = (
        "pass"
        if channel_summaries
        and all(row.get("target_count", 0) == len(case_rows) for row in channel_summaries)
        else "fail"
    )
    summary = {
        "status": status,
        "scope_note": (
            "Grounding-bottleneck ablation. Embedding calls are limited to the "
            "selected cached condition reports and query channels; no new "
            "grounding/chat calls are made."
        ),
        "case_spec_json": str(args.case_spec_json),
        "case_count": int(len(case_rows)),
        "query_channels": channels,
        "samples": int(args.samples),
        "steps": int(args.steps),
        "chunk_size": int(args.chunk_size),
        "device": str(args.device),
        "channel_summary": channel_summaries,
        "channel_records": channel_records,
        "case_spec_paths": case_spec_paths,
        "bakeoff_reports": bakeoff_reports,
        "artifact_paths": {
            "report": str(output / "grounding_channel_ablation.json"),
            "markdown": str(output / "grounding_channel_ablation.md"),
        },
    }
    _write_json(summary["artifact_paths"]["report"], summary)
    Path(summary["artifact_paths"]["markdown"]).write_text(
        render_markdown(summary).rstrip() + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-spec-json", default=DEFAULT_CASE_SPEC_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument(
        "--query-channel",
        action="append",
        choices=QUERY_CHANNELS,
        default=[],
        help="Query channel to include. Repeat to include multiple channels.",
    )
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--bridge-adapter", default=DEFAULT_BRIDGE_ADAPTER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.query_channel:
        args.query_channel = list(DEFAULT_CHANNELS)
    summary = run_grounding_channel_ablation(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "case_count": summary["case_count"],
                "query_channels": summary["query_channels"],
                "best_channel": (
                    summary["channel_summary"][0]["query_channel"]
                    if summary["channel_summary"]
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
