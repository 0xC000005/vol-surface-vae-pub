#!/usr/bin/env python
"""Cached/live Gradio-wrapper smoke for the prefix-latent scenario demo."""

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

from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    DEFAULT_STORY,
    cached_prefix_casebook_choices,
    cached_prefix_casebook_update,
    refresh_fan_chart,
    run_prefix_latent_for_app,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_cached_smoke_796a"
)
DEFAULT_CACHED_CASEBOOK_CHOICE = "safe_haven_gold_bid:18"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fast_prefix_runner(
    args: SimpleNamespace,
    *,
    live_story: bool,
) -> dict[str, Any]:
    """Run the real prefix smoke with bounded demo-safe controls."""

    args.live_story = bool(live_story)
    args.steps = min(int(getattr(args, "steps", 1000)), 100)
    args.samples = min(int(getattr(args, "samples", 2)), 2)
    args.chunk_size = min(int(getattr(args, "chunk_size", 2)), 2)
    args.output_dir = str(Path(args.output_dir))
    return run_prefix_latent_story_smoke(args)


def _table_rows(value: Any) -> int:
    if hasattr(value, "shape"):
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 0


def _prefix_output(final: tuple[Any, ...], index: int, default: Any = None) -> Any:
    return final[index] if index < len(final) else default


def _cached_casebook_controls(choice: str | None) -> dict[str, Any]:
    value = str(choice or "").strip()
    if not value:
        return {}
    choices = {
        str(raw_value): label for label, raw_value in cached_prefix_casebook_choices()
    }
    if value not in choices:
        raise ValueError(f"unknown cached casebook choice: {value!r}")
    (
        story,
        use_explicit_start,
        explicit_start_index,
        condition_only_story,
        live_story,
        condition_report,
        status,
    ) = cached_prefix_casebook_update(value)
    return {
        "choice": value,
        "label": choices[value],
        "story": story,
        "use_explicit_start": bool(use_explicit_start),
        "explicit_start_index": int(explicit_start_index),
        "condition_only_story": bool(condition_only_story),
        "live_story": bool(live_story),
        "condition_report": str(condition_report),
        "status_markdown": str(status),
    }


def run_gradio_cached_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_casebook = _cached_casebook_controls(
        getattr(args, "cached_casebook_choice", "")
    )
    live_story = bool(getattr(args, "live_story", False))
    if cached_casebook:
        live_story = bool(cached_casebook["live_story"])
    summary_name = (
        "gradio_live_smoke_summary.json"
        if live_story
        else (
            "gradio_cached_casebook_smoke_summary.json"
            if cached_casebook
            else "gradio_cached_smoke_summary.json"
        )
    )

    def runner(run_args: SimpleNamespace) -> dict[str, Any]:
        run_args.output_dir = str(output_dir / "prefix_run")
        return _fast_prefix_runner(run_args, live_story=live_story)

    stream = run_prefix_latent_for_app(
        start_mode=str(args.start_mode),
        samples=int(args.samples),
        fan_market=str(args.fan_market),
        analogue_scope="ALL",
        live_story=live_story,
        story=str(cached_casebook.get("story", args.story)),
        cached_condition_report=cached_casebook.get("condition_report"),
        condition_only_story=bool(cached_casebook.get("condition_only_story", False)),
        use_explicit_start=bool(cached_casebook.get("use_explicit_start", False)),
        explicit_start_window_index=cached_casebook.get("explicit_start_index"),
        runner=runner,
    )
    first = next(stream)
    final = tuple(list(stream)[-1])
    markdown = _prefix_output(final, 0, "")
    status_markdown = _prefix_output(final, 1, "")
    selected_table = _prefix_output(final, 2)
    diagnostic_table = _prefix_output(final, 3)
    validation_table = _prefix_output(final, 4)
    scenario_table = _prefix_output(final, 5)
    fan_plot = _prefix_output(final, 6)
    report_json = _prefix_output(final, 7, "")
    report = _prefix_output(final, 8, {})
    analogue_update = _prefix_output(final, 9)
    condition_table = _prefix_output(final, 10)
    warning_table = _prefix_output(final, 11)
    candidate_table = _prefix_output(final, 14)
    errors: list[str] = []
    if "Scenario Workflow Status" not in str(first[1]) or "Run started:" not in str(first[1]):
        errors.append("progress_status_missing")
    if "Story support:" not in str(status_markdown):
        errors.append("story_support_status_missing")
    if _table_rows(selected_table) < 1:
        errors.append("selected_table_empty")
    if _table_rows(diagnostic_table) < 1:
        errors.append("diagnostic_table_empty")
    if _table_rows(validation_table) < 1:
        errors.append("validation_table_empty")
    if _table_rows(scenario_table) < 1:
        errors.append("scenario_table_empty")
    if len(getattr(fan_plot, "data", [])) < 1:
        errors.append("fan_plot_empty")
    redraw_market = str(
        getattr(args, "redraw_market", getattr(args, "fan_market", "SPX"))
    )
    redraw_plot = refresh_fan_chart(
        report if isinstance(report, dict) else {}, redraw_market, "ALL"
    )
    if len(getattr(redraw_plot, "data", [])) < 1:
        errors.append("redraw_fan_plot_empty")
    if not isinstance(report, dict) or report.get("status") != "ok":
        errors.append("report_not_ok")
    gate = report.get("validation_gate", {}) if isinstance(report, dict) else {}
    if not isinstance(gate, dict) or "selected_start_status" not in gate:
        errors.append("gate_selected_status_missing")
    artifact_inputs = report.get("artifact_inputs", {}) if isinstance(report, dict) else {}
    start_reliability = (
        report.get("start_reliability_gate", {}) if isinstance(report, dict) else {}
    )
    manifest_expected = bool(
        isinstance(artifact_inputs, dict)
        and artifact_inputs.get("start_reliability_manifest")
    )
    if manifest_expected and not (
        isinstance(start_reliability, dict)
        and start_reliability.get("product_status")
    ):
        errors.append("start_reliability_status_missing")
    query = report.get("cached_query", {}) if isinstance(report, dict) else {}
    condition_source = (
        str(query.get("condition_source", "")) if isinstance(query, dict) else ""
    )
    expected_condition_source = (
        "live_openai_story" if live_story else "cached_bridge_query"
    )
    if cached_casebook:
        expected_condition_source = "external_condition_report"
    if condition_source != expected_condition_source:
        errors.append("condition_source_mismatch")
    if not str(report_json).strip().startswith("{"):
        errors.append("json_report_missing")
    generation = report.get("generation", {}) if isinstance(report, dict) else {}
    path_quantiles = (
        generation.get("path_quantiles", []) if isinstance(generation, dict) else []
    )
    path_labels = [
        str(row.get("analogue_label", ""))
        for row in path_quantiles
        if isinstance(row, dict)
    ]
    has_selected_label = any("Selected start:" in label for label in path_labels)
    has_diagnostic_label = any("Diagnostic baseline:" in label for label in path_labels)
    if not has_selected_label:
        errors.append("selected_start_label_missing")
    if not has_diagnostic_label:
        errors.append("diagnostic_baseline_label_missing")

    summary = {
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "mode": (
            "cached_casebook"
            if cached_casebook
            else "live_story" if live_story else "cached"
        ),
        "live_story": live_story,
        "cached_casebook": cached_casebook,
        "condition_source": condition_source,
        "scope_note": (
            "Live Gradio wrapper smoke. This calls the real prefix-latent Gradio "
            "wrapper path with OpenAI grounding and embedding, then verifies the "
            "product tables, labels, and prefix rollout."
            if live_story
            else (
                (
                    "Cached casebook Gradio wrapper smoke. This uses a cached "
                    "condition-only report plus a fixed historical start, makes no "
                    "OpenAI calls, and verifies post-run chart redraw."
                )
                if cached_casebook
                else (
                    "Cached Gradio wrapper smoke. This calls the real prefix-latent "
                    "Gradio wrapper path with cached text memory and makes no OpenAI calls."
                )
            )
        ),
        "start_mode": str(args.start_mode),
        "selected_table_rows": _table_rows(selected_table),
        "diagnostic_table_rows": _table_rows(diagnostic_table),
        "validation_table_rows": _table_rows(validation_table),
        "scenario_table_rows": _table_rows(scenario_table),
        "condition_table_rows": _table_rows(condition_table),
        "warning_table_rows": _table_rows(warning_table),
        "candidate_table_rows": _table_rows(candidate_table),
        "fan_trace_count": int(len(getattr(fan_plot, "data", []))),
        "redraw_market": redraw_market,
        "redraw_fan_trace_count": int(len(getattr(redraw_plot, "data", []))),
        "selected_start_status": str(gate.get("selected_start_status", "")),
        "start_reliability_status": (
            str(start_reliability.get("product_status", ""))
            if isinstance(start_reliability, dict)
            else ""
        ),
        "diagnostic_baseline_status": str(gate.get("diagnostic_baseline_status", "")),
        "research_overall_status": str(gate.get("overall_status", "")),
        "path_labels": sorted(set(path_labels)),
        "has_selected_start_label": bool(has_selected_label),
        "has_diagnostic_baseline_label": bool(has_diagnostic_label),
        "analogue_update": str(analogue_update),
        "markdown_length": int(len(str(markdown))),
        "status_markdown_length": int(len(str(status_markdown))),
        "artifact_paths": {
            "summary": str(output_dir / summary_name),
            "prefix_report": str(
                output_dir / "prefix_run" / "prefix_latent_story_smoke_report.json"
            ),
        },
    }
    _write_json(output_dir / summary_name, summary)
    if errors:
        raise RuntimeError(f"Gradio cached smoke failed: {errors}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-mode", default="balanced_memory_start")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--redraw-market", default="IV_ATM_3M")
    parser.add_argument("--story", default=DEFAULT_STORY)
    parser.add_argument(
        "--cached-casebook-choice",
        default="",
        help=(
            "Optional cached casebook value, e.g. "
            f"{DEFAULT_CACHED_CASEBOOK_CHOICE!r}. Uses no OpenAI calls."
        ),
    )
    parser.add_argument(
        "--live-story",
        action="store_true",
        help="Call OpenAI for live grounding and text embedding instead of cached text memory.",
    )
    args = parser.parse_args()
    summary = run_gradio_cached_smoke(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "status": summary["status"],
                "selected_start_status": summary["selected_start_status"],
                "fan_trace_count": summary["fan_trace_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
