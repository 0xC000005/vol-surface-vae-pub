#!/usr/bin/env python
"""API-level smoke test for the running prefix-latent Gradio demo."""

from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_LIVE_STORY = (
    "This looks like a safe-haven bid with softer risk appetite: gold is "
    "rallying, Treasury yields are lower, equities are choppy, volatility "
    "remains elevated, and the dollar is not providing a clear offset. The "
    "forward risk is that safe-haven demand becomes a broader risk-off move."
)
DEFAULT_AUTH_USER_ENV = "NARRATIVE_DEMO_AUTH_USER"
DEFAULT_AUTH_PASSWORD_ENV = "NARRATIVE_DEMO_AUTH_PASSWORD"


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


def resolve_client_auth(
    args: argparse.Namespace,
    env: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    env_map = os.environ if env is None else env
    user_env = str(getattr(args, "auth_user_env", DEFAULT_AUTH_USER_ENV))
    password_env = str(getattr(args, "auth_password_env", DEFAULT_AUTH_PASSWORD_ENV))
    require_auth = bool(getattr(args, "require_auth", False))
    user = str(env_map.get(user_env, "")).strip()
    password = str(env_map.get(password_env, "")).strip()
    if user and password:
        return user, password
    if require_auth:
        missing = []
        if not user:
            missing.append(user_env)
        if not password:
            missing.append(password_env)
        raise RuntimeError(
            "missing required Gradio auth environment variable(s): "
            + ", ".join(missing)
        )
    return None


def make_client(url: str, auth: tuple[str, str] | None = None) -> Any:
    Client = _client_class()
    if auth is None:
        return Client(str(url))
    return Client(str(url), auth=auth)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _support_candidate_summary(report: dict[str, Any], *, limit: int = 8) -> list[dict]:
    query = _as_dict(report.get("cached_query"))
    memory_prior = _as_dict(query.get("memory_prior"))
    rows = _as_list(memory_prior.get("candidate_details"))
    summary: list[dict] = []
    for row in rows[:limit]:
        item = _as_dict(row)
        summary.append(
            {
                "rank": item.get("rank"),
                "window_id": item.get("window_id"),
                "window_index": item.get("window_index"),
                "weight": item.get("weight"),
                "memory_support_cosine": item.get("memory_support_cosine"),
                "start_distance_z": item.get("start_distance_z"),
                "recent_prefix_alignment_score": item.get(
                    "recent_prefix_alignment_score"
                ),
                "history_start_date": item.get("history_start_date"),
                "history_end_date": item.get("history_end_date"),
                "forecast_start_date": item.get("forecast_start_date"),
                "forecast_end_date": item.get("forecast_end_date"),
            }
        )
    return summary


def _market_implication_summary(report: dict[str, Any]) -> list[dict]:
    grounding = _as_dict(_as_dict(report.get("cached_query")).get("grounding"))
    rows = _as_list(grounding.get("market_implications"))
    return [
        {
            "market": _as_dict(row).get("market"),
            "direction": _as_dict(row).get("direction"),
            "magnitude": _as_dict(row).get("magnitude"),
            "confidence": _as_dict(row).get("confidence"),
            "horizon": _as_dict(row).get("horizon"),
            "target_use": _as_dict(row).get("target_use"),
        }
        for row in rows
    ]


def _forward_warning_summary(report: dict[str, Any]) -> list[dict]:
    grounding = _as_dict(_as_dict(report.get("cached_query")).get("grounding"))
    rows = _as_list(grounding.get("non_conditioning_forward_language"))
    return [
        {
            "phrase": _as_dict(row).get("phrase"),
            "handling": _as_dict(row).get("handling"),
            "severity": _as_dict(row).get("severity"),
            "reason": _as_dict(row).get("reason"),
        }
        for row in rows
    ]


def run_gradio_api_smoke(args: argparse.Namespace) -> dict[str, Any]:
    auth = resolve_client_auth(args)
    client = make_client(str(args.url), auth=auth)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = str(getattr(args, "mode", "cached_casebook"))
    if mode == "cached_casebook":
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
    elif mode == "live_condition_only":
        story = str(args.story)
        use_explicit_start = True
        explicit_start_index = int(args.expected_start_index)
        condition_only_story = True
        live_story = True
        cached_condition_report = ""
        casebook_status = "live condition-only OpenAI TestFlight"
    else:
        raise ValueError(f"unknown mode: {mode!r}")

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
    artifact_paths = _as_dict(report.get("artifact_paths"))
    embedding_metadata = _as_dict(_as_dict(query).get("embedding_metadata"))
    memory_prior = _as_dict(_as_dict(query).get("memory_prior"))
    support_candidates = _support_candidate_summary(report)
    support_weights = [
        row.get("weight")
        for row in support_candidates
        if isinstance(row.get("weight"), int | float)
    ]
    market_implications = _market_implication_summary(report)
    forward_warnings = _forward_warning_summary(report)
    condition_only_case = (
        report.get("condition_only_case", {}) if isinstance(report, dict) else {}
    )
    condition_only_metadata = _as_dict(
        condition_only_case.get("metadata")
        if isinstance(condition_only_case, dict)
        else {}
    )
    condition_only_validation = (
        condition_only_case.get("condition_only_validation", {})
        if isinstance(condition_only_case, dict)
        else {}
    )
    errors: list[str] = []
    if mode == "cached_casebook" and "OpenAI calls: `none" not in str(casebook_status):
        errors.append("casebook_no_openai_status_missing")
    if mode == "cached_casebook" and bool(live_story):
        errors.append("casebook_live_story_true")
    if not bool(use_explicit_start):
        errors.append("explicit_start_false")
    if int(explicit_start_index) != int(args.expected_start_index):
        errors.append("start_index_mismatch")
    if str(query.get("condition_source", "")) != "external_condition_report":
        errors.append("condition_source_mismatch")
    if mode == "live_condition_only":
        if not isinstance(condition_only_validation, dict):
            errors.append("condition_only_validation_missing")
        elif str(condition_only_validation.get("status", "")) != "pass":
            errors.append("condition_only_validation_not_pass")
        forward_warning_count = condition_only_validation.get(
            "forward_warning_count",
            0,
        )
        if int(forward_warning_count or 0) < 1:
            errors.append("forward_warning_count_missing")
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
    if len(support_candidates) < 1:
        errors.append("support_candidates_missing")
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
        "auth_used": auth is not None,
        "mode": mode,
        "casebook_choice": str(args.casebook_choice),
        "casebook_start_index": int(explicit_start_index),
        "casebook_status_length": len(str(casebook_status)),
        "condition_source": str(query.get("condition_source", "")),
        "prefix_report_path": str(artifact_paths.get("report", "")),
        "prefix_markdown_path": str(artifact_paths.get("markdown", "")),
        "prefix_arrays_path": str(artifact_paths.get("arrays", "")),
        "condition_report_path": str(embedding_metadata.get("condition_report", "")),
        "condition_arrays_path": str(embedding_metadata.get("condition_arrays", "")),
        "grounding_model": str(embedding_metadata.get("grounding_model", "")),
        "embedding_model": str(embedding_metadata.get("embedding_model", "")),
        "embedding_dim": int(embedding_metadata.get("embedding_dim", 0) or 0),
        "condition_dim": int(embedding_metadata.get("condition_dim", 0) or 0),
        "openai_response_id": str(condition_only_metadata.get("response_id", "")),
        "openai_usage": condition_only_metadata.get("usage", {}),
        "condition_only_validation_status": str(
            condition_only_validation.get("status", "")
            if isinstance(condition_only_validation, dict)
            else ""
        ),
        "condition_only_forward_warning_count": int(
            condition_only_validation.get("forward_warning_count", 0)
            if isinstance(condition_only_validation, dict)
            else 0
        ),
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
        "market_implications": market_implications,
        "forward_warnings": forward_warnings,
        "support_prior_mode": str(memory_prior.get("mode", "")),
        "support_alignment_status": str(
            _as_dict(memory_prior.get("support_alignment")).get("status", "")
        ),
        "support_candidate_count": len(support_candidates),
        "support_weight_sum": float(sum(float(weight) for weight in support_weights)),
        "support_top_candidates": support_candidates,
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
    parser.add_argument(
        "--mode",
        choices=["cached_casebook", "live_condition_only"],
        default="cached_casebook",
    )
    parser.add_argument("--casebook-choice", default=DEFAULT_CASEBOOK_CHOICE)
    parser.add_argument("--story", default=DEFAULT_LIVE_STORY)
    parser.add_argument("--expected-start-index", type=int, default=18)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--redraw-market", default="IV_ATM_3M")
    parser.add_argument("--auth-user-env", default=DEFAULT_AUTH_USER_ENV)
    parser.add_argument("--auth-password-env", default=DEFAULT_AUTH_PASSWORD_ENV)
    parser.add_argument(
        "--require-auth",
        action="store_true",
        help="fail unless Gradio auth user/password env vars are present",
    )
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
