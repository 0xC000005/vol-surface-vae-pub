#!/usr/bin/env python
"""Temporal-role-aware alignment for narrative-conditioned scenario artifacts.

This evaluator makes no OpenAI calls. It audits existing grounding outputs and
separates current/regime implications from forward-looking implications before
scoring support priors and generated future paths.
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

from experiments.backfill.block_ar.nl_prefix_latent_implication_alignment import (  # noqa: E402
    _terminal_summary_to_scenario_rows,
)
from experiments.backfill.block_ar.nl_prefix_latent_market_alignment import (  # noqa: E402
    market_implication_alignment,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_temporal_role_alignment_807a"
)
DEFAULT_INPUTS = [
    (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_story_smoke_805a_mixture_one/"
        "prefix_latent_story_smoke_report.json"
    )
]

FORWARD_TERMS = {
    "could",
    "forward",
    "future",
    "forecast",
    "next",
    "risk",
    "scenario",
    "shock",
    "stress",
    "unwind",
    "unwinding",
    "reversal",
    "revert",
}


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


def _contains_forward_language(text: str) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in FORWARD_TERMS)


def temporal_role_for_implication(item: dict[str, Any]) -> str:
    """Classify an implication with the current grounding schema."""

    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [evidence]
    evidence_texts = [str(value) for value in evidence if str(value)]
    if not evidence_texts:
        return "current_regime"
    forward_flags = [_contains_forward_language(text) for text in evidence_texts]
    has_forward = any(forward_flags)
    has_current = any(not flag for flag in forward_flags)
    if has_forward and has_current:
        return "ambiguous_mixed"
    if has_forward:
        return "forward_risk"
    return "current_regime"


def split_grounding_by_temporal_role(
    grounding: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    implications = grounding.get("market_implications", [])
    if not isinstance(implications, list):
        implications = []
    result = {
        "current_regime": [],
        "forward_risk": [],
        "ambiguous_mixed": [],
    }
    for item in implications:
        if not isinstance(item, dict):
            continue
        role = temporal_role_for_implication(item)
        enriched = {**item, "temporal_role": role}
        result.setdefault(role, []).append(enriched)
    return result


def _grounding_with_implications(
    grounding: dict[str, Any],
    implications: list[dict[str, Any]],
) -> dict[str, Any]:
    return {**grounding, "market_implications": implications}


def _support_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    query = report.get("cached_query", {})
    if not isinstance(query, dict):
        return []
    prior = query.get("memory_prior", {})
    if not isinstance(prior, dict):
        return []
    rows = prior.get("terminal_rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _generated_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    generation = report.get("generation", {})
    if not isinstance(generation, dict):
        return []
    terminal = generation.get("terminal_delta_summary", [])
    if not isinstance(terminal, list):
        return []
    return _terminal_summary_to_scenario_rows(terminal)


def evaluate_temporal_report(path: str | Path) -> dict[str, Any]:
    report = _load_json(path)
    query = report.get("cached_query", {})
    if not isinstance(query, dict):
        query = {}
    grounding = query.get("grounding", {})
    if not isinstance(grounding, dict):
        grounding = {}
    roles = split_grounding_by_temporal_role(grounding)
    support_rows = _support_rows_from_report(report)
    generated_rows = _generated_rows_from_report(report)
    current_alignment = market_implication_alignment(
        grounding=_grounding_with_implications(grounding, roles["current_regime"]),
        scenario_rows=support_rows,
    )
    future_alignment = market_implication_alignment(
        grounding=_grounding_with_implications(grounding, roles["forward_risk"]),
        scenario_rows=generated_rows,
    )
    legacy_future_alignment = market_implication_alignment(
        grounding=grounding,
        scenario_rows=generated_rows,
    )
    return {
        "source_path": str(path),
        "condition_source": str(query.get("condition_source", "")),
        "memory_prior_mode": str(query.get("memory_prior_mode", "")),
        "role_counts": {role: len(rows) for role, rows in roles.items()},
        "current_support_alignment": current_alignment,
        "future_generated_alignment": future_alignment,
        "legacy_all_implications_future_alignment": legacy_future_alignment,
        "ambiguous_implications": roles["ambiguous_mixed"],
        "temporal_role_implications": roles,
    }


def _expand_inputs(paths: list[str | Path]) -> list[str]:
    expanded: list[str] = []
    for path in paths:
        payload = _load_json(path)
        if isinstance(payload.get("cases"), list):
            for case in payload["cases"]:
                if not isinstance(case, dict):
                    continue
                artifact_paths = case.get("artifact_paths", {})
                if isinstance(artifact_paths, dict) and artifact_paths.get(
                    "prefix_report"
                ):
                    expanded.append(str(artifact_paths["prefix_report"]))
        elif isinstance(payload.get("cached_query"), dict):
            expanded.append(str(path))
        else:
            raise ValueError(f"{path}: unsupported temporal alignment input")
    return expanded


def summarize_temporal_cases(
    cases: list[dict[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    totals = {
        "current_checked": 0,
        "current_mismatches": 0,
        "future_checked": 0,
        "future_mismatches": 0,
        "legacy_checked": 0,
        "legacy_mismatches": 0,
        "ambiguous_count": 0,
    }
    for case in cases:
        current = case.get("current_support_alignment", {})
        future = case.get("future_generated_alignment", {})
        legacy = case.get("legacy_all_implications_future_alignment", {})
        totals["current_checked"] += int(current.get("checked_count", 0) or 0)
        totals["current_mismatches"] += int(current.get("mismatch_count", 0) or 0)
        totals["future_checked"] += int(future.get("checked_count", 0) or 0)
        totals["future_mismatches"] += int(future.get("mismatch_count", 0) or 0)
        totals["legacy_checked"] += int(legacy.get("checked_count", 0) or 0)
        totals["legacy_mismatches"] += int(legacy.get("mismatch_count", 0) or 0)
        totals["ambiguous_count"] += len(case.get("ambiguous_implications", []))
    output_path = Path(output_dir) / "temporal_role_alignment_summary.json"
    return {
        "status": "ok",
        "scope_note": (
            "Temporal-role-aware alignment evaluator. No OpenAI calls are made. "
            "Current/regime implications are scored against support priors; "
            "forward-risk implications are scored against generated future paths; "
            "ambiguous mixed implications are reported separately."
        ),
        "case_count": len(cases),
        "totals": {
            **totals,
            "current_mismatch_rate": (
                float(totals["current_mismatches"] / totals["current_checked"])
                if totals["current_checked"]
                else None
            ),
            "future_mismatch_rate": (
                float(totals["future_mismatches"] / totals["future_checked"])
                if totals["future_checked"]
                else None
            ),
            "legacy_mismatch_rate": (
                float(totals["legacy_mismatches"] / totals["legacy_checked"])
                if totals["legacy_checked"]
                else None
            ),
        },
        "cases": cases,
        "artifact_paths": {"summary": str(output_path)},
    }


def run_temporal_alignment(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _expand_inputs([str(path) for path in args.input])
    cases = [evaluate_temporal_report(path) for path in inputs]
    summary = summarize_temporal_cases(cases, output_dir=args.output_dir)
    _write_json(summary["artifact_paths"]["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input", action="append", default=None)
    args = parser.parse_args()
    if args.input is None:
        args.input = list(DEFAULT_INPUTS)
    summary = run_temporal_alignment(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "case_count": summary["case_count"],
                "totals": summary["totals"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
