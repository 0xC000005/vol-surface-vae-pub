#!/usr/bin/env python
"""Small live narrative casebook for prefix-latent TestFlight diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_live_casebook_792a"
)
DEFAULT_SCRIPT = "experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py"


def _slug(text: str) -> str:
    keep = []
    for char in str(text).lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_") or "case"


def default_casebook_stories() -> list[dict[str, str]]:
    """Return a bounded set of story-like live TestFlight narratives."""

    return [
        {
            "name": "fragile_risk_on_rebound",
            "story": (
                "This has the shape of a fragile risk-on rebound: equities are "
                "recovering, volatility is compressing, spreads are stabilizing, "
                "and investors appear to be rotating back into carry. The forward "
                "risk is that a volatility reversal quickly unwinds the move."
            ),
        },
        {
            "name": "defensive_risk_off_shock",
            "story": (
                "This looks like a defensive risk-off shock: equities are selling "
                "off, volatility is jumping, credit spreads are widening, and "
                "investors are moving toward dollar liquidity and safe-haven "
                "assets. The next month risk is that funding stress keeps forcing "
                "de-risking across high-beta assets."
            ),
        },
        {
            "name": "rates_selloff_tightening_fear",
            "story": (
                "This is a rates-led tightening scare: Treasury yields are moving "
                "higher, equities are struggling with duration pressure, the dollar "
                "is firm, and volatility is grinding higher rather than exploding. "
                "The risk is that another leg up in yields reprices growth and "
                "keeps risk appetite fragile."
            ),
        },
    ]


def build_case_command(
    *,
    story: str,
    output_dir: str,
    args: argparse.Namespace,
) -> list[str]:
    """Build a subprocess command for one live-story prefix-latent case."""

    return [
        sys.executable,
        str(args.script),
        "--live-story",
        "--story",
        str(story),
        "--output-dir",
        str(output_dir),
        "--steps",
        str(int(args.steps)),
        "--samples",
        str(int(args.samples)),
        "--chunk-size",
        str(int(args.chunk_size)),
        "--start-mode",
        str(args.start_mode),
        "--device",
        str(args.device),
        "--grounding-model",
        str(args.grounding_model),
        "--embedding-model",
        str(args.embedding_model),
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


def summarize_case_report(
    *,
    case_name: str,
    report_path: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Extract a compact diagnostic row from one prefix-latent story report."""

    query = report.get("cached_query", {})
    gate = report.get("validation_gate", {})
    generation = report.get("generation", {})
    grounding = query.get("grounding") if isinstance(query, dict) else {}
    if not isinstance(grounding, dict):
        grounding = {}
    cases = gate.get("cases", []) if isinstance(gate, dict) else []
    operational_cases = [
        item for item in cases if isinstance(item, dict) and bool(item.get("is_operational"))
    ]
    selected_case = operational_cases[0] if operational_cases else {}
    memory_cosines = [
        float(item["input_memory_cosine"])
        for item in cases
        if isinstance(item, dict) and item.get("input_memory_cosine") is not None
    ]
    warning_counts = gate.get("warning_counts", {}) if isinstance(gate, dict) else {}
    fail_counts = gate.get("fail_counts", {}) if isinstance(gate, dict) else {}
    return {
        "case_name": str(case_name),
        "report_path": str(report_path),
        "condition_source": str(query.get("condition_source", "")),
        "overall_status": str(gate.get("overall_status", "")),
        "operational_status": str(gate.get("operational_status", "")),
        "selected_start_status": str(
            gate.get("selected_start_status", gate.get("operational_status", ""))
        ),
        "diagnostic_baseline_status": str(gate.get("diagnostic_baseline_status", "")),
        "stress_status": str(gate.get("stress_status", "")),
        "warning_counts": warning_counts if isinstance(warning_counts, dict) else {},
        "fail_counts": fail_counts if isinstance(fail_counts, dict) else {},
        "selected_start_memory_cosine": selected_case.get("input_memory_cosine"),
        "selected_start_terminal_shift_z": selected_case.get(
            "terminal_mean_abs_delta_z"
        ),
        "selected_start_warnings": selected_case.get("warnings", []),
        "min_memory_cosine": min(memory_cosines) if memory_cosines else None,
        "mean_memory_cosine": (
            sum(memory_cosines) / len(memory_cosines) if memory_cosines else None
        ),
        "grounding_frame": str(grounding.get("narrative_frame", "")),
        "grounding_warning_count": len(grounding.get("grounding_warnings", [])),
        "generated_state_shape": generation.get("generated_state_shape"),
    }


def run_casebook(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stories = default_casebook_stories()[: int(args.case_count)]
    case_rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for case_no, item in enumerate(stories, start=1):
        case_name = str(item["name"])
        case_dir = output_dir / f"{case_no:02d}_{_slug(case_name)}"
        command = build_case_command(
            story=str(item["story"]),
            output_dir=str(case_dir),
            args=args,
        )
        commands.append(command)
        subprocess.run(command, cwd=Path.cwd(), check=True)
        report_path = case_dir / "prefix_latent_story_smoke_report.json"
        report = _load_json(report_path)
        case_rows.append(
            summarize_case_report(
                case_name=case_name,
                report_path=str(report_path),
                report=report,
            )
        )
    status_counts: dict[str, int] = {}
    for row in case_rows:
        status = str(row.get("overall_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "status": "ok",
        "scope_note": (
            "Small live OpenAI prefix-latent casebook. Each case calls OpenAI "
            "for grounding and embedding, then runs the decoded-prefix frozen "
            "joint39 rollout."
        ),
        "case_count": len(case_rows),
        "status_counts": status_counts,
        "cases": case_rows,
        "commands": commands,
        "artifact_paths": {
            "summary": str(output_dir / "live_prefix_casebook_summary.json"),
        },
    }
    _write_json(output_dir / "live_prefix_casebook_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--script", default=DEFAULT_SCRIPT)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--start-mode", default="balanced_memory_start")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grounding-model", default="gpt-5.4-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()
    summary = run_casebook(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "case_count": summary["case_count"],
                "status_counts": summary["status_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
