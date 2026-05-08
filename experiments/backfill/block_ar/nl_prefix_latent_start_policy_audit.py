"""Audit narrative-only start policies for condition-only story reports.

The production question is not whether one hand-picked historical start can
work. For narrative-only mode, the system needs a defensible default for
choosing the starting state before rolling out the frozen joint39 generator.

This harness replays the same condition-only reports through several start
selection policies and summarizes the validation gate, rollout shift, memory
support, and support-prior alignment. It intentionally reuses saved condition
reports, so no OpenAI calls are made.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STORY_SMOKE_SCRIPT = Path(
    "experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_policy_audit_810a_condition_only"
)
DEFAULT_CASES = {
    "fragile_risk_on": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_809a_fragile/condition_only_report.json"
    ),
    "defensive_risk_off": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_809b_defensive/condition_only_report.json"
    ),
    "rates_tightening": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_809c_rates/condition_only_report.json"
    ),
}
DEFAULT_START_MODES = [
    "balanced_memory_start",
    "memory_nearest_start",
    "implication_aligned_start",
]
STATUS_RANK = {"pass": 0, "warning": 1, "fail": 2}


@dataclass(frozen=True)
class AuditCase:
    name: str
    condition_report: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "case"


def parse_case(value: str) -> AuditCase:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"case must be NAME=PATH, received {value!r}"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return AuditCase(name=slugify(name), condition_report=Path(path))


def operational_case(report: dict[str, Any]) -> dict[str, Any]:
    for row in report.get("validation_gate", {}).get("cases", []):
        if bool(row.get("is_operational", False)):
            return row
    return {}


def operational_variant(report: dict[str, Any]) -> dict[str, Any]:
    for row in report.get("variant_rows", []):
        if bool(row.get("is_operational", False)):
            return row
    return {}


def support_alignment(report: dict[str, Any]) -> dict[str, Any]:
    memory_prior = report.get("cached_query", {}).get("memory_prior", {})
    alignment = memory_prior.get("support_alignment", {})
    return alignment if isinstance(alignment, dict) else {}


def summarize_report(case_name: str, start_mode: str, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    gate = report.get("validation_gate", {})
    selected = operational_case(report)
    variant = operational_variant(report)
    alignment = support_alignment(report)
    warnings = selected.get("warnings", [])
    failures = selected.get("failures", [])
    if not isinstance(warnings, list):
        warnings = []
    if not isinstance(failures, list):
        failures = []
    return {
        "case_name": case_name,
        "start_mode": start_mode,
        "report_path": str(report_path),
        "output_dir": str(report_path.parent),
        "overall_status": str(gate.get("overall_status", "unknown")),
        "operational_status": str(gate.get("operational_status", "unknown")),
        "selected_start_status": str(
            gate.get("selected_start_status", gate.get("operational_status", "unknown"))
        ),
        "warning_reasons": [str(item) for item in warnings],
        "failure_reasons": [str(item) for item in failures],
        "query_window_index": selected.get("query_window_index"),
        "start_window_index": selected.get("start_window_index"),
        "start_window_id": variant.get("start_window_id"),
        "start_source_index": variant.get("start_source_index"),
        "start_manifest_split": variant.get("start_manifest_split"),
        "start_selection_method": variant.get("start_selection_method"),
        "start_distance_z": selected.get("start_distance_z"),
        "input_memory_cosine": selected.get("input_memory_cosine"),
        "mean_abs_delta_z": selected.get("mean_abs_delta_z"),
        "terminal_mean_abs_delta_z": selected.get("terminal_mean_abs_delta_z"),
        "support_alignment_status": str(alignment.get("status", "unknown")),
        "support_checked_count": int(alignment.get("checked_count", 0) or 0),
        "support_mismatch_count": int(alignment.get("mismatch_count", 0) or 0),
        "support_match_count": int(alignment.get("match_count", 0) or 0),
    }


def _finite_float(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if result != result:
        return fallback
    return result


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row["start_mode"])].append(row)

    aggregates = []
    for mode, mode_rows in sorted(by_mode.items()):
        status_counts = Counter(str(row["selected_start_status"]) for row in mode_rows)
        warning_counts: Counter[str] = Counter()
        failure_counts: Counter[str] = Counter()
        for row in mode_rows:
            warning_counts.update(row.get("warning_reasons", []))
            failure_counts.update(row.get("failure_reasons", []))
        start_distances = [
            _finite_float(row.get("start_distance_z")) for row in mode_rows
        ]
        terminal_shifts = [
            _finite_float(row.get("terminal_mean_abs_delta_z")) for row in mode_rows
        ]
        memory_cosines = [
            _finite_float(row.get("input_memory_cosine")) for row in mode_rows
        ]
        support_mismatches = [
            int(row.get("support_mismatch_count", 0) or 0) for row in mode_rows
        ]
        checked = [int(row.get("support_checked_count", 0) or 0) for row in mode_rows]
        fail_count = status_counts.get("fail", 0) + sum(
            1 for row in mode_rows if row.get("failure_reasons")
        )
        warning_count = status_counts.get("warning", 0)
        max_terminal_shift = max(terminal_shifts) if terminal_shifts else 0.0
        max_start_distance = max(start_distances) if start_distances else 0.0
        mean = lambda values: sum(values) / len(values) if values else 0.0
        aggregates.append(
            {
                "start_mode": mode,
                "case_count": len(mode_rows),
                "status_counts": dict(sorted(status_counts.items())),
                "warning_counts": dict(sorted(warning_counts.items())),
                "failure_counts": dict(sorted(failure_counts.items())),
                "fail_count": int(fail_count),
                "warning_count": int(warning_count),
                "support_mismatch_count": int(sum(support_mismatches)),
                "support_checked_count": int(sum(checked)),
                "mean_start_distance_z": mean(start_distances),
                "max_start_distance_z": max_start_distance,
                "mean_memory_cosine": mean(memory_cosines),
                "min_memory_cosine": min(memory_cosines) if memory_cosines else 0.0,
                "mean_terminal_abs_delta_z": mean(terminal_shifts),
                "max_terminal_abs_delta_z": max_terminal_shift,
                "recommendation_score": [
                    int(fail_count),
                    int(warning_count),
                    int(sum(support_mismatches)),
                    round(float(max_terminal_shift), 6),
                    round(float(max_start_distance), 6),
                    round(float(-mean(memory_cosines)), 6),
                ],
            }
        )
    return aggregates


def choose_recommendation(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    if not aggregates:
        return {"start_mode": None, "reason": "no audit rows"}
    best = min(aggregates, key=lambda row: tuple(row["recommendation_score"]))
    reason = (
        "lowest fail/warn count, then lowest support mismatch, rollout shift, "
        "start distance, and highest memory support"
    )
    return {
        "start_mode": best["start_mode"],
        "reason": reason,
        "score": best["recommendation_score"],
    }


def story_smoke_command(
    *,
    condition_report: Path,
    output_dir: Path,
    start_mode: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(STORY_SMOKE_SCRIPT),
        "--condition-report",
        str(condition_report),
        "--output-dir",
        str(output_dir),
        "--memory-prior-mode",
        str(args.memory_prior_mode),
        "--memory-prior-top-k",
        str(args.memory_prior_top_k),
        "--memory-prior-temperature",
        str(args.memory_prior_temperature),
        "--start-mode",
        str(start_mode),
        "--implication-alignment-weight",
        str(args.implication_alignment_weight),
        "--steps",
        str(args.steps),
        "--samples",
        str(args.samples),
        "--chunk-size",
        str(args.chunk_size),
        "--device",
        str(args.device),
    ]


def run_one(case: AuditCase, start_mode: str, args: argparse.Namespace) -> dict[str, Any]:
    output_dir = (
        Path(args.output_dir)
        / f"{slugify(case.name)}__{slugify(start_mode)}"
    )
    report_path = output_dir / "prefix_latent_story_smoke_report.json"
    if not args.reports_only and not (args.skip_existing and report_path.exists()):
        command = story_smoke_command(
            condition_report=case.condition_report,
            output_dir=output_dir,
            start_mode=start_mode,
            args=args,
        )
        subprocess.run(command, cwd=Path.cwd(), check=True)
    if not report_path.exists():
        raise FileNotFoundError(
            f"missing report for {case.name}/{start_mode}: {report_path}"
        )
    return summarize_report(case.name, start_mode, report_path)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Condition-Only Start Policy Audit",
        "",
        f"Recommended start policy: `{summary['recommendation']['start_mode']}`",
        "",
        "## Aggregate Results",
        "",
        "| Start mode | Status counts | Warnings | Support mismatches | "
        "Mean start z | Max terminal shift | Mean memory cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        lines.append(
            "| {mode} | {status} | {warnings} | {mismatch}/{checked} | "
            "{mean_start:.3f} | {max_shift:.3f} | {mean_cos:.3f} |".format(
                mode=row["start_mode"],
                status=json.dumps(row["status_counts"], sort_keys=True),
                warnings=json.dumps(row["warning_counts"], sort_keys=True),
                mismatch=row["support_mismatch_count"],
                checked=row["support_checked_count"],
                mean_start=float(row["mean_start_distance_z"]),
                max_shift=float(row["max_terminal_abs_delta_z"]),
                mean_cos=float(row["mean_memory_cosine"]),
            )
        )
    lines.extend(["", "## Case Rows", ""])
    for row in summary["rows"]:
        lines.append(
            "- `{case}` / `{mode}`: status `{status}`, start `{start}`, "
            "distance z `{distance:.3f}`, terminal shift `{shift:.3f}`, "
            "memory cosine `{cosine:.3f}`, warnings `{warnings}`.".format(
                case=row["case_name"],
                mode=row["start_mode"],
                status=row["selected_start_status"],
                start=row.get("start_window_id") or row.get("start_window_index"),
                distance=_finite_float(row.get("start_distance_z")),
                shift=_finite_float(row.get("terminal_mean_abs_delta_z")),
                cosine=_finite_float(row.get("input_memory_cosine")),
                warnings=", ".join(row.get("warning_reasons", [])) or "none",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="Condition-only case as NAME=PATH. Defaults to the 3 casebook cases.",
    )
    parser.add_argument(
        "--start-mode",
        action="append",
        choices=DEFAULT_START_MODES,
        help="Start policy to audit. Can be repeated.",
    )
    parser.add_argument("--memory-prior-mode", default="soft_topk_combined")
    parser.add_argument("--memory-prior-top-k", type=int, default=8)
    parser.add_argument("--memory-prior-temperature", type=float, default=0.2)
    parser.add_argument("--implication-alignment-weight", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Summarize existing per-case reports without running story smoke.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = args.case or [
        AuditCase(name=name, condition_report=path)
        for name, path in DEFAULT_CASES.items()
    ]
    start_modes = args.start_mode or DEFAULT_START_MODES
    rows = [run_one(case, mode, args) for case in cases for mode in start_modes]
    aggregates = aggregate_rows(rows)
    summary = {
        "status": "ok",
        "case_count": len(cases),
        "start_modes": list(start_modes),
        "config": {
            "memory_prior_mode": args.memory_prior_mode,
            "memory_prior_top_k": int(args.memory_prior_top_k),
            "memory_prior_temperature": float(args.memory_prior_temperature),
            "implication_alignment_weight": float(args.implication_alignment_weight),
            "steps": int(args.steps),
            "samples": int(args.samples),
            "chunk_size": int(args.chunk_size),
            "device": str(args.device),
        },
        "recommendation": choose_recommendation(aggregates),
        "aggregates": aggregates,
        "rows": rows,
        "artifact_paths": {
            "summary_json": str(output_dir / "start_policy_audit_summary.json"),
            "summary_markdown": str(output_dir / "start_policy_audit_summary.md"),
        },
    }
    summary_path = output_dir / "start_policy_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, output_dir / "start_policy_audit_summary.md")
    print(json.dumps(summary["recommendation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
