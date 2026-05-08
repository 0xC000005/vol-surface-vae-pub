#!/usr/bin/env python
"""Offline implication-alignment evaluator for narrative-conditioned scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_gradio_live_casebook import (  # noqa: E402
    market_implication_alignment,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_implication_alignment_801a"
)
DEFAULT_INPUTS = [
    (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_gradio_live_casebook_800b_alignment/"
        "gradio_live_casebook_summary.json"
    )
]


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


def _terminal_summary_to_scenario_rows(
    terminal_summary: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in terminal_summary:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Market": item.get("market"),
                "Mean Terminal Delta": item.get("mean_terminal_delta"),
                "P10": item.get("p10"),
                "P90": item.get("p90"),
            }
        )
    return rows


def _casebook_cases(path: str | Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        alignment = case.get("market_alignment")
        if not isinstance(alignment, dict):
            alignment = {
                "status": "unknown",
                "checked_count": 0,
                "match_count": 0,
                "mismatch_count": 0,
                "skipped_count": 0,
                "mismatches": [],
                "skipped": [],
            }
        rows.append(
            {
                "source_path": str(path),
                "source_kind": "gradio_casebook_summary",
                "case_name": str(case.get("case_name", "")),
                "condition_source": str(case.get("condition_source", "")),
                "overall_status": str(case.get("overall_status", "")),
                "selected_start_status": str(case.get("selected_start_status", "")),
                "alignment": alignment,
            }
        )
    return rows


def _prefix_report_case(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("cached_query", {})
    if not isinstance(query, dict):
        query = {}
    grounding = query.get("grounding", {})
    if not isinstance(grounding, dict):
        grounding = {}
    generation = payload.get("generation", {})
    if not isinstance(generation, dict):
        generation = {}
    terminal_summary = generation.get("terminal_delta_summary", [])
    if not isinstance(terminal_summary, list):
        terminal_summary = []
    gate = payload.get("validation_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    alignment = market_implication_alignment(
        grounding=grounding,
        scenario_rows=_terminal_summary_to_scenario_rows(terminal_summary),
    )
    case_name = str(query.get("kind") or query.get("role") or Path(path).parent.name)
    return {
        "source_path": str(path),
        "source_kind": "prefix_story_smoke_report",
        "case_name": case_name,
        "condition_source": str(query.get("condition_source", "")),
        "overall_status": str(gate.get("overall_status", "")),
        "selected_start_status": str(gate.get("selected_start_status", "")),
        "alignment": alignment,
    }


def evaluate_alignment_inputs(paths: list[str | Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        if isinstance(payload.get("cases"), list):
            cases.extend(_casebook_cases(path, payload))
        elif isinstance(payload.get("cached_query"), dict):
            cases.append(_prefix_report_case(path, payload))
        else:
            raise ValueError(f"{path}: unsupported alignment input schema")
    return cases


def summarize_alignment_cases(
    cases: list[dict[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    total_checked = 0
    total_mismatches = 0
    total_skipped = 0
    for case in cases:
        alignment = case.get("alignment", {})
        if not isinstance(alignment, dict):
            alignment = {}
        status = str(alignment.get("status", "unknown"))
        source_kind = str(case.get("source_kind", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        total_checked += int(alignment.get("checked_count", 0) or 0)
        total_mismatches += int(alignment.get("mismatch_count", 0) or 0)
        total_skipped += int(alignment.get("skipped_count", 0) or 0)
    output_path = Path(output_dir) / "implication_alignment_summary.json"
    return {
        "status": "ok",
        "scope_note": (
            "Offline implication-alignment evaluator. No OpenAI calls are made; "
            "the evaluator compares grounded market implication directions with "
            "generated scenario terminal directions in existing artifacts."
        ),
        "case_count": len(cases),
        "alignment_status_counts": status_counts,
        "source_kind_counts": source_counts,
        "total_checked_implications": total_checked,
        "total_mismatches": total_mismatches,
        "total_skipped_implications": total_skipped,
        "mismatch_rate": (
            float(total_mismatches / total_checked) if total_checked else None
        ),
        "cases": cases,
        "artifact_paths": {"summary": str(output_path)},
    }


def run_alignment_evaluator(args: argparse.Namespace) -> dict[str, Any]:
    cases = evaluate_alignment_inputs([str(path) for path in args.input])
    summary = summarize_alignment_cases(cases, output_dir=args.output_dir)
    _write_json(summary["artifact_paths"]["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    if args.input is None:
        args.input = list(DEFAULT_INPUTS)
    summary = run_alignment_evaluator(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "case_count": summary["case_count"],
                "alignment_status_counts": summary["alignment_status_counts"],
                "total_checked_implications": summary["total_checked_implications"],
                "total_mismatches": summary["total_mismatches"],
                "mismatch_rate": summary["mismatch_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
