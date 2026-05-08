#!/usr/bin/env python
"""Product-level live Gradio casebook for prefix-latent scenario diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_live_casebook import (  # noqa: E402
    default_casebook_stories,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    run_prefix_latent_for_app,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_casebook_800a"
)


def _slug(text: str) -> str:
    keep: list[str] = []
    for char in str(text).lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_") or "case"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _table_rows(value: Any) -> int:
    if hasattr(value, "shape"):
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 0


def _table_records(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        records = value.to_dict(orient="records")
        if isinstance(records, list):
            return [row for row in records if isinstance(row, dict)]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _path_labels(report: dict[str, Any]) -> list[str]:
    generation = report.get("generation", {}) if isinstance(report, dict) else {}
    path_quantiles = (
        generation.get("path_quantiles", []) if isinstance(generation, dict) else []
    )
    return sorted(
        {
            str(row.get("analogue_label", ""))
            for row in path_quantiles
            if isinstance(row, dict) and str(row.get("analogue_label", ""))
        }
    )


def _parse_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _expected_delta_sign(direction: Any) -> int | None:
    text = str(direction).strip().lower()
    positive = {
        "up",
        "higher",
        "rise",
        "rising",
        "wider",
        "widening",
        "weaker",
    }
    negative = {
        "down",
        "lower",
        "fall",
        "falling",
        "tighter",
        "tightening",
        "stronger",
    }
    if text in positive:
        return 1
    if text in negative:
        return -1
    return None


def market_implication_alignment(
    *,
    grounding: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare direct grounded implications with generated terminal directions."""

    scenario_by_market = {
        str(row.get("Market", "")).upper(): row
        for row in scenario_rows
        if isinstance(row, dict)
    }
    checked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    implications = grounding.get("market_implications", [])
    if not isinstance(implications, list):
        implications = []
    for item in implications:
        if not isinstance(item, dict):
            continue
        market = str(item.get("market", "")).upper()
        direction = str(item.get("direction", ""))
        expected = _expected_delta_sign(direction)
        scenario = scenario_by_market.get(market)
        if expected is None or scenario is None:
            skipped.append(
                {
                    "market": market,
                    "direction": direction,
                    "reason": "unsupported_direction_or_missing_scenario",
                }
            )
            continue
        observed = _parse_float(scenario.get("Mean Terminal Delta"))
        if observed is None:
            skipped.append(
                {
                    "market": market,
                    "direction": direction,
                    "reason": "non_numeric_terminal_delta",
                }
            )
            continue
        observed_sign = 0
        if observed > tolerance:
            observed_sign = 1
        elif observed < -tolerance:
            observed_sign = -1
        checked.append(
            {
                "market": market,
                "direction": direction,
                "expected_sign": expected,
                "mean_terminal_delta": observed,
                "observed_sign": observed_sign,
                "aligned": observed_sign == expected,
                "confidence": item.get("confidence"),
                "inferred": bool(item.get("inferred", False)),
            }
        )
    mismatches = [row for row in checked if not bool(row.get("aligned"))]
    return {
        "status": "pass" if not mismatches else "warning",
        "checked_count": len(checked),
        "match_count": len(checked) - len(mismatches),
        "mismatch_count": len(mismatches),
        "skipped_count": len(skipped),
        "mismatches": mismatches,
        "skipped": skipped,
    }


def _bounded_live_runner(
    *,
    output_dir: Path,
    steps: int,
    samples: int,
    chunk_size: int,
) -> Any:
    def runner(run_args: SimpleNamespace) -> dict[str, Any]:
        run_args.live_story = True
        run_args.output_dir = str(output_dir)
        run_args.steps = min(int(getattr(run_args, "steps", steps)), int(steps))
        run_args.samples = min(int(getattr(run_args, "samples", samples)), int(samples))
        run_args.chunk_size = min(
            int(getattr(run_args, "chunk_size", chunk_size)),
            int(chunk_size),
        )
        return run_prefix_latent_story_smoke(run_args)

    return runner


def summarize_gradio_case(
    *,
    case_name: str,
    story: str,
    case_dir: Path,
    final_outputs: tuple[Any, ...],
    first_outputs: tuple[Any, ...],
) -> dict[str, Any]:
    (
        markdown,
        status_markdown,
        selected_table,
        diagnostic_table,
        validation_table,
        scenario_table,
        fan_plot,
        report_json,
        report,
        analogue_update,
    ) = final_outputs
    errors: list[str] = []
    if "Prefix-latent run started" not in str(first_outputs[1]):
        errors.append("progress_status_missing")
    if not isinstance(report, dict) or report.get("status") != "ok":
        errors.append("report_not_ok")
    query = report.get("cached_query", {}) if isinstance(report, dict) else {}
    condition_source = (
        str(query.get("condition_source", "")) if isinstance(query, dict) else ""
    )
    if condition_source != "live_openai_story":
        errors.append("condition_source_mismatch")
    gate = report.get("validation_gate", {}) if isinstance(report, dict) else {}
    if not isinstance(gate, dict) or "selected_start_status" not in gate:
        errors.append("gate_selected_status_missing")
    row_counts = {
        "selected_table_rows": _table_rows(selected_table),
        "diagnostic_table_rows": _table_rows(diagnostic_table),
        "validation_table_rows": _table_rows(validation_table),
        "scenario_table_rows": _table_rows(scenario_table),
    }
    for key, value in row_counts.items():
        if int(value) < 1:
            errors.append(f"{key}_empty")
    fan_trace_count = int(len(getattr(fan_plot, "data", [])))
    if fan_trace_count < 1:
        errors.append("fan_plot_empty")
    labels = _path_labels(report if isinstance(report, dict) else {})
    if not any("Selected start:" in label for label in labels):
        errors.append("selected_start_label_missing")
    if not any("Diagnostic baseline:" in label for label in labels):
        errors.append("diagnostic_baseline_label_missing")
    grounding = query.get("grounding", {}) if isinstance(query, dict) else {}
    if not isinstance(grounding, dict):
        grounding = {}
    generation = report.get("generation", {}) if isinstance(report, dict) else {}
    if not isinstance(generation, dict):
        generation = {}
    scenario_rows = _table_records(scenario_table)
    alignment = market_implication_alignment(
        grounding=grounding,
        scenario_rows=scenario_rows,
    )
    return {
        "case_name": str(case_name),
        "story": str(story),
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "condition_source": condition_source,
        "overall_status": str(gate.get("overall_status", "")),
        "operational_status": str(gate.get("operational_status", "")),
        "selected_start_status": str(gate.get("selected_start_status", "")),
        "diagnostic_baseline_status": str(gate.get("diagnostic_baseline_status", "")),
        "stress_status": str(gate.get("stress_status", "")),
        "warning_counts": gate.get("warning_counts", {}),
        "fail_counts": gate.get("fail_counts", {}),
        "grounding_frame": str(grounding.get("narrative_frame", "")),
        "grounding_warning_count": len(grounding.get("grounding_warnings", [])),
        "market_implication_count": len(grounding.get("market_implications", [])),
        "market_alignment": alignment,
        "generated_state_shape": generation.get("generated_state_shape"),
        "path_labels": labels,
        "fan_trace_count": fan_trace_count,
        "scenario_rows": scenario_rows,
        "analogue_update": str(analogue_update),
        "markdown_length": int(len(str(markdown))),
        "status_markdown_length": int(len(str(status_markdown))),
        "json_report_length": int(len(str(report_json))),
        **row_counts,
        "artifact_paths": {
            "case_dir": str(case_dir),
            "prefix_report": str(
                case_dir / "prefix_run" / "prefix_latent_story_smoke_report.json"
            ),
            "prefix_markdown": str(
                case_dir / "prefix_run" / "prefix_latent_story_smoke_report.md"
            ),
        },
    }


def run_gradio_live_casebook(
    args: argparse.Namespace,
    *,
    app_runner: Any = run_prefix_latent_for_app,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stories = default_casebook_stories()[: int(args.case_count)]
    case_rows: list[dict[str, Any]] = []
    for case_no, item in enumerate(stories, start=1):
        case_name = str(item["name"])
        story = str(item["story"])
        case_dir = output_dir / f"{case_no:02d}_{_slug(case_name)}"
        runner = _bounded_live_runner(
            output_dir=case_dir / "prefix_run",
            steps=int(args.steps),
            samples=int(args.samples),
            chunk_size=int(args.chunk_size),
        )
        stream = app_runner(
            start_mode=str(args.start_mode),
            samples=int(args.samples),
            fan_market=str(args.fan_market),
            analogue_scope="ALL",
            live_story=True,
            story=story,
            runner=runner,
        )
        first = next(stream)
        final = list(stream)[-1]
        case_rows.append(
            summarize_gradio_case(
                case_name=case_name,
                story=story,
                case_dir=case_dir,
                first_outputs=first,
                final_outputs=final,
            )
        )
    status_counts: dict[str, int] = {}
    selected_status_counts: dict[str, int] = {}
    market_alignment_counts: dict[str, int] = {}
    harness_errors: list[str] = []
    for row in case_rows:
        status = str(row.get("overall_status", ""))
        selected_status = str(row.get("selected_start_status", ""))
        alignment = row.get("market_alignment", {})
        alignment_status = (
            str(alignment.get("status", "")) if isinstance(alignment, dict) else ""
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        selected_status_counts[selected_status] = (
            selected_status_counts.get(selected_status, 0) + 1
        )
        market_alignment_counts[alignment_status] = (
            market_alignment_counts.get(alignment_status, 0) + 1
        )
        harness_errors.extend(
            f"{row.get('case_name')}: {error}" for error in row.get("errors", [])
        )
    summary = {
        "status": "ok" if not harness_errors else "fail",
        "errors": harness_errors,
        "scope_note": (
            "Product-level live Gradio casebook. Each case calls the same "
            "run_prefix_latent_for_app wrapper used by the demo, with bounded "
            "OpenAI grounding/embedding and prefix-latent rollout controls."
        ),
        "case_count": len(case_rows),
        "overall_status_counts": status_counts,
        "selected_start_status_counts": selected_status_counts,
        "market_alignment_status_counts": market_alignment_counts,
        "cases": case_rows,
        "artifact_paths": {
            "summary": str(output_dir / "gradio_live_casebook_summary.json"),
        },
    }
    _write_json(output_dir / "gradio_live_casebook_summary.json", summary)
    if harness_errors:
        raise RuntimeError(f"Gradio live casebook failed: {harness_errors}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--start-mode", default="balanced_memory_start")
    parser.add_argument("--fan-market", default="SPX")
    args = parser.parse_args()
    summary = run_gradio_live_casebook(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "status": summary["status"],
                "case_count": summary["case_count"],
                "overall_status_counts": summary["overall_status_counts"],
                "selected_start_status_counts": summary["selected_start_status_counts"],
                "market_alignment_status_counts": summary[
                    "market_alignment_status_counts"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
