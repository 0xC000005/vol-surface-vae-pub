#!/usr/bin/env python
"""Cached Gradio-wrapper smoke for the prefix-latent scenario demo."""

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
    run_prefix_latent_for_app,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_cached_smoke_796a"
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fast_cached_runner(args: SimpleNamespace) -> dict[str, Any]:
    """Run the real cached prefix smoke with bounded demo-safe controls."""

    args.live_story = False
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


def run_gradio_cached_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def runner(run_args: SimpleNamespace) -> dict[str, Any]:
        run_args.output_dir = str(output_dir / "prefix_run")
        return _fast_cached_runner(run_args)

    stream = run_prefix_latent_for_app(
        start_mode=str(args.start_mode),
        samples=int(args.samples),
        fan_market=str(args.fan_market),
        analogue_scope="ALL",
        live_story=False,
        story=str(args.story),
        runner=runner,
    )
    first = next(stream)
    final = list(stream)[-1]
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
    ) = final
    errors: list[str] = []
    if "Prefix-latent run started" not in str(first[1]):
        errors.append("progress_status_missing")
    if "Selected-start:" not in str(status_markdown):
        errors.append("selected_status_missing")
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
    if not isinstance(report, dict) or report.get("status") != "ok":
        errors.append("report_not_ok")
    gate = report.get("validation_gate", {}) if isinstance(report, dict) else {}
    if not isinstance(gate, dict) or "selected_start_status" not in gate:
        errors.append("gate_selected_status_missing")
    if not str(report_json).strip().startswith("{"):
        errors.append("json_report_missing")

    summary = {
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "scope_note": (
            "Cached Gradio wrapper smoke. This calls the real prefix-latent "
            "Gradio wrapper path with cached text memory and makes no OpenAI calls."
        ),
        "start_mode": str(args.start_mode),
        "selected_table_rows": _table_rows(selected_table),
        "diagnostic_table_rows": _table_rows(diagnostic_table),
        "validation_table_rows": _table_rows(validation_table),
        "scenario_table_rows": _table_rows(scenario_table),
        "fan_trace_count": int(len(getattr(fan_plot, "data", []))),
        "selected_start_status": str(gate.get("selected_start_status", "")),
        "diagnostic_baseline_status": str(gate.get("diagnostic_baseline_status", "")),
        "research_overall_status": str(gate.get("overall_status", "")),
        "analogue_update": str(analogue_update),
        "markdown_length": int(len(str(markdown))),
        "status_markdown_length": int(len(str(status_markdown))),
        "artifact_paths": {
            "summary": str(output_dir / "gradio_cached_smoke_summary.json"),
            "prefix_report": str(
                output_dir / "prefix_run" / "prefix_latent_story_smoke_report.json"
            ),
        },
    }
    _write_json(output_dir / "gradio_cached_smoke_summary.json", summary)
    if errors:
        raise RuntimeError(f"Gradio cached smoke failed: {errors}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-mode", default="balanced_memory_start")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--story", default=DEFAULT_STORY)
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
