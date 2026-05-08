#!/usr/bin/env python
"""API-level smoke test for the running prefix-latent Gradio demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    DEFAULT_USER_START_STATE_JSON,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_api_smoke_825a"
)
DEFAULT_URL = "http://127.0.0.1:7861"
DEFAULT_CASEBOOK_CHOICE = "safe_haven_gold_bid:18"


def _table_rows(value: Any) -> int:
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        return len(value["data"])
    if hasattr(value, "shape"):
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 0


def _plot_trace_count(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    plot_text = value.get("plot")
    if not isinstance(plot_text, str) or not plot_text.strip():
        return 0
    try:
        plot_payload = json.loads(plot_text)
    except json.JSONDecodeError:
        return 1
    data = plot_payload.get("data")
    return len(data) if isinstance(data, list) else 1


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _client_class() -> Any:
    try:
        from gradio_client import Client
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("gradio_client is required for the API smoke") from error
    return Client


def run_gradio_api_smoke(args: argparse.Namespace) -> dict[str, Any]:
    Client = _client_class()
    client = Client(str(args.url))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    casebook = client.predict(
        str(args.casebook_choice),
        api_name="/cached_prefix_casebook_update",
    )
    if len(casebook) != 7:
        raise RuntimeError(f"unexpected casebook output length: {len(casebook)}")
    (
        story,
        use_explicit_start,
        explicit_start_index,
        condition_only_story,
        live_story,
        cached_condition_report,
        casebook_status,
    ) = casebook

    run_outputs = client.predict(
        "balanced_memory_start",
        int(args.samples),
        str(args.fan_market),
        "ALL",
        bool(live_story),
        str(story),
        str(cached_condition_report),
        bool(condition_only_story),
        bool(use_explicit_start),
        int(explicit_start_index),
        False,
        str(DEFAULT_USER_START_STATE_JSON),
        api_name="/run_prefix_latent_for_app",
    )
    if len(run_outputs) != 16:
        raise RuntimeError(f"unexpected prefix run output length: {len(run_outputs)}")

    (
        report_markdown,
        status_markdown,
        selected_table,
        diagnostic_table,
        validation_table,
        scenario_table,
        fan_plot,
        report_json,
        analogue_scope_update,
        condition_table,
        warning_table,
        warning_component_table,
        shift_factor_table,
        candidate_table,
        user_start_table,
        historical_start_candidate_update,
    ) = run_outputs

    report = json.loads(str(report_json))
    redraw_plot = client.predict(
        str(args.redraw_market),
        "ALL",
        api_name="/refresh_fan_chart_2",
    )

    query = report.get("cached_query", {}) if isinstance(report, dict) else {}
    gate = report.get("validation_gate", {}) if isinstance(report, dict) else {}
    errors: list[str] = []
    if "OpenAI calls: `none" not in str(casebook_status):
        errors.append("casebook_no_openai_status_missing")
    if bool(live_story):
        errors.append("casebook_live_story_true")
    if not bool(use_explicit_start):
        errors.append("casebook_explicit_start_false")
    if int(explicit_start_index) != int(args.expected_start_index):
        errors.append("casebook_start_index_mismatch")
    if str(query.get("condition_source", "")) != "external_condition_report":
        errors.append("condition_source_mismatch")
    if report.get("status") != "ok":
        errors.append("report_not_ok")
    if str(gate.get("selected_start_status", "")) != "pass":
        errors.append("selected_start_not_pass")
    if _table_rows(selected_table) < 1:
        errors.append("selected_table_empty")
    if _table_rows(validation_table) < 1:
        errors.append("validation_table_empty")
    if _table_rows(scenario_table) < 1:
        errors.append("scenario_table_empty")
    if _table_rows(condition_table) < 1:
        errors.append("condition_table_empty")
    if _table_rows(warning_table) < 1:
        errors.append("warning_table_empty")
    if _table_rows(candidate_table) < 1:
        errors.append("candidate_table_empty")
    if _plot_trace_count(fan_plot) < 1:
        errors.append("fan_plot_empty")
    if _plot_trace_count(redraw_plot) < 1:
        errors.append("redraw_plot_empty")
    if "Selected-start:" not in str(status_markdown):
        errors.append("status_selected_start_missing")

    summary = {
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "url": str(args.url),
        "casebook_choice": str(args.casebook_choice),
        "casebook_start_index": int(explicit_start_index),
        "casebook_status_length": len(str(casebook_status)),
        "condition_source": str(query.get("condition_source", "")),
        "selected_start_status": str(gate.get("selected_start_status", "")),
        "diagnostic_baseline_status": str(gate.get("diagnostic_baseline_status", "")),
        "overall_status": str(gate.get("overall_status", "")),
        "selected_table_rows": _table_rows(selected_table),
        "diagnostic_table_rows": _table_rows(diagnostic_table),
        "validation_table_rows": _table_rows(validation_table),
        "scenario_table_rows": _table_rows(scenario_table),
        "condition_table_rows": _table_rows(condition_table),
        "warning_table_rows": _table_rows(warning_table),
        "warning_component_rows": _table_rows(warning_component_table),
        "shift_factor_rows": _table_rows(shift_factor_table),
        "candidate_table_rows": _table_rows(candidate_table),
        "user_start_table_rows": _table_rows(user_start_table),
        "fan_market": str(args.fan_market),
        "fan_trace_count": _plot_trace_count(fan_plot),
        "redraw_market": str(args.redraw_market),
        "redraw_trace_count": _plot_trace_count(redraw_plot),
        "report_markdown_length": len(str(report_markdown)),
        "status_markdown_length": len(str(status_markdown)),
        "analogue_scope_update": str(analogue_scope_update),
        "historical_start_candidate_update": str(historical_start_candidate_update),
        "artifact_paths": {
            "summary": str(output_dir / "gradio_api_smoke_summary.json"),
        },
    }
    _write_json(output_dir / "gradio_api_smoke_summary.json", summary)
    if errors:
        raise RuntimeError(f"Gradio API smoke failed: {errors}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--casebook-choice", default=DEFAULT_CASEBOOK_CHOICE)
    parser.add_argument("--expected-start-index", type=int, default=18)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--redraw-market", default="IV_ATM_3M")
    args = parser.parse_args()
    summary = run_gradio_api_smoke(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "status": summary["status"],
                "selected_start_status": summary["selected_start_status"],
                "fan_trace_count": summary["fan_trace_count"],
                "redraw_trace_count": summary["redraw_trace_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
