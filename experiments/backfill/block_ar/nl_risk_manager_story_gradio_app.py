#!/usr/bin/env python
"""Local Gradio demo for narrative-conditioned scenario generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ADAPTER,
    DEFAULT_CASEBOOK,
    DEFAULT_HARD_CASE_MANIFEST,
    DEFAULT_PIPELINE_NPZ,
    DEFAULT_PIPELINE_REPORT,
    DEFAULT_STORY,
    render_story_smoke_markdown,
    run_story_smoke,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS as DEFAULT_PREFIX_BRIDGE_ARRAYS,
    DEFAULT_BRIDGE_REPORT as DEFAULT_PREFIX_BRIDGE_REPORT,
    run_prefix_latent_story_smoke,
)


DEFAULT_APP_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo"
)
DEFAULT_PREFIX_VALIDATION_GATE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_validation_gate_790b/"
    "prefix_latent_validation_gate_report.json"
)
DEFAULT_PREFIX_APP_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo/prefix_latent_live_smoke"
)
IMPLICATION_COLUMNS = [
    "Market",
    "Direction",
    "Magnitude",
    "Confidence",
    "Inferred",
    "Evidence",
]
WARNING_COLUMNS = ["Severity", "Code", "Message"]
ANALOGUE_COLUMNS = [
    "Rank",
    "Window",
    "Source",
    "History End",
    "Cosine",
    "Weight",
    "Implication Match",
    "Fit Status",
    "Split",
    "Narrative",
]
SCENARIO_COLUMNS = ["Market", "Mean Terminal Delta", "P10", "P90"]
VALIDATION_GATE_COLUMNS = [
    "Variant",
    "Query",
    "Start",
    "Status",
    "Memory Cosine",
    "Start Distance",
    "Terminal Shift",
    "Warnings",
    "Failures",
]
PREFIX_VARIANT_COLUMNS = [
    "Variant",
    "Query Window",
    "Start Window",
    "Start Distance",
    "Start Split",
]
FAN_MARKET_CHOICES = [
    ("SPX", "SPX"),
    ("VIX", "VIX"),
    ("BBB OAS", "BBB_OAS"),
    ("AAA OAS", "AAA_OAS"),
    ("US 2Y", "US2Y"),
    ("US 10Y", "US10Y"),
    ("USD/JPY", "USDJPY"),
    ("DXY", "DXY"),
    ("Gold", "GOLD"),
    ("Crude oil", "CRUDE_OIL"),
    ("IV surface average", "IV_SURFACE"),
    ("IV ATM 3M, K=1.00", "IV_ATM_3M"),
    ("IV ATM 1Y, K=1.00", "IV_ATM_1Y"),
    ("IV OTM put 1Y, K=0.70", "IV_OTM_PUT_1Y"),
    ("IV wing 6M, K=1.30", "IV_WING_6M_K130"),
]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _short(text: Any, limit: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= int(limit):
        return compact
    return compact[: int(limit) - 3].rstrip() + "..."


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _float_series(value: Any) -> list[float]:
    values = []
    for item in _as_list(value):
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def implications_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _as_dict(report.get("grounding"))
    for item in _as_list(grounding.get("market_implications")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Market": str(item.get("market", "")),
                "Direction": str(item.get("direction", "")),
                "Magnitude": str(item.get("magnitude", "")),
                "Confidence": str(item.get("confidence", "")),
                "Inferred": bool(item.get("inferred", False)),
                "Evidence": "; ".join(str(x) for x in _as_list(item.get("evidence"))),
            }
        )
    return _frame(rows, IMPLICATION_COLUMNS)


def warnings_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _as_dict(report.get("grounding"))
    for item in _as_list(grounding.get("grounding_warnings")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Severity": str(item.get("severity", "")),
                "Code": str(item.get("code", "")),
                "Message": str(item.get("message", "")),
            }
        )
    return _frame(
        rows or [{"Severity": "none", "Code": "none", "Message": "none"}],
        WARNING_COLUMNS,
    )


def analogues_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(_as_list(report.get("historical_analogues")), start=1):
        if not isinstance(item, dict):
            continue
        alignment = _as_dict(item.get("implication_alignment"))
        calendar = _as_dict(item.get("calendar"))
        rows.append(
            {
                "Rank": rank,
                "Window": str(item.get("window_id", "")),
                "Source": str(item.get("source_index", "")),
                "History End": str(calendar.get("history_end", "")),
                "Cosine": _fmt_float(item.get("cosine")),
                "Weight": _fmt_float(item.get("weight")),
                "Implication Match": _fmt_float(alignment.get("match_rate")),
                "Fit Status": str(alignment.get("status", "")),
                "Split": str(item.get("manifest_split", "")),
                "Narrative": _short(
                    item.get("casebook_narrative") or item.get("primary_narrative")
                ),
            }
        )
    return _frame(rows, ANALOGUE_COLUMNS)


def scenario_table(report: dict[str, Any]) -> pd.DataFrame:
    generation = _as_dict(report.get("generation"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(generation.get("terminal_delta_summary")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Market": str(item.get("market", "")),
                "Mean Terminal Delta": _fmt_float(item.get("mean_terminal_delta")),
                "P10": _fmt_float(item.get("p10")),
                "P90": _fmt_float(item.get("p90")),
            }
        )
    return _frame(rows, SCENARIO_COLUMNS)


def load_validation_gate_report(
    path: str | Path = DEFAULT_PREFIX_VALIDATION_GATE_REPORT,
) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def _count_items_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def validation_gate_markdown(report: dict[str, Any]) -> str:
    gate = _as_dict(report.get("gate"))
    if not gate:
        return "\n".join(
            [
                "## Latent-prefix validation",
                "",
                "- Status: `not available`",
                "- Run the cached validation gate to populate this section.",
            ]
        )
    return "\n".join(
        [
            "## Latent-prefix validation",
            "",
            f"- Operational status: `{gate.get('operational_status', 'n/a')}`",
            f"- Stress status: `{gate.get('stress_status', 'n/a')}`",
            f"- Overall diagnostic status: `{gate.get('overall_status', 'n/a')}`",
            f"- Endpoint max error: `{_fmt_float(gate.get('endpoint_max_abs_error'), 6)}`",
            f"- Warnings: `{_count_items_text(gate.get('warning_counts'))}`",
            f"- Failures: `{_count_items_text(gate.get('fail_counts'))}`",
        ]
    )


def validation_gate_table(report: dict[str, Any]) -> pd.DataFrame:
    gate = _as_dict(report.get("gate"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(gate.get("hard_cases")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Variant": str(item.get("variant", "")),
                "Query": str(item.get("query_window_index", "")),
                "Start": str(item.get("start_window_index", "")),
                "Status": str(item.get("status", "")),
                "Memory Cosine": _fmt_float(item.get("input_memory_cosine")),
                "Start Distance": _fmt_float(item.get("start_distance_z")),
                "Terminal Shift": _fmt_float(item.get("terminal_mean_abs_delta_z")),
                "Warnings": ", ".join(str(x) for x in _as_list(item.get("warnings"))),
                "Failures": ", ".join(str(x) for x in _as_list(item.get("failures"))),
            }
        )
    return _frame(rows, VALIDATION_GATE_COLUMNS)


def prefix_variant_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(report.get("variant_rows")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Variant": str(item.get("variant", "")),
                "Query Window": str(item.get("query_window_id", "")),
                "Start Window": str(item.get("start_window_id", "")),
                "Start Distance": _fmt_float(item.get("start_distance_z")),
                "Start Split": str(item.get("start_manifest_split", "")),
            }
        )
    return _frame(rows, PREFIX_VARIANT_COLUMNS)


def prefix_validation_table(report: dict[str, Any]) -> pd.DataFrame:
    gate = _as_dict(report.get("validation_gate"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(gate.get("cases")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Variant": str(item.get("variant", "")),
                "Query": str(item.get("query_window_index", "")),
                "Start": str(item.get("start_window_index", "")),
                "Status": str(item.get("status", "")),
                "Memory Cosine": _fmt_float(item.get("input_memory_cosine")),
                "Start Distance": _fmt_float(item.get("start_distance_z")),
                "Terminal Shift": _fmt_float(item.get("terminal_mean_abs_delta_z")),
                "Warnings": ", ".join(str(x) for x in _as_list(item.get("warnings"))),
                "Failures": ", ".join(str(x) for x in _as_list(item.get("failures"))),
            }
        )
    return _frame(rows, VALIDATION_GATE_COLUMNS)


def analogue_scope_choices(report: dict[str, Any]) -> list[tuple[str, str]]:
    choices = [("All retrieved analogues", "ALL")]
    added = False
    for rank, item in enumerate(_as_list(report.get("historical_analogues")), start=1):
        if not isinstance(item, dict):
            continue
        window_id = str(item.get("window_id", f"analogue_{rank}"))
        choices.append((f"Analogue {rank}: {window_id}", f"RANK_{rank}"))
        added = True
    if added:
        return choices
    seen: set[str] = {"ALL"}
    for row in _as_list(_as_dict(report.get("generation")).get("path_quantiles")):
        if not isinstance(row, dict):
            continue
        key = str(row.get("analogue_key", "ALL"))
        if key in seen or key == "ALL":
            continue
        label = str(row.get("analogue_label") or key)
        choices.append((label, key))
        seen.add(key)
    return choices


def analogue_scope_update(report: dict[str, Any]) -> Any:
    import gradio as gr

    return gr.update(choices=analogue_scope_choices(report), value="ALL")


def _path_quantile_row(
    report: dict[str, Any],
    market: str,
    analogue_scope: str = "ALL",
) -> dict[str, Any]:
    generation = _as_dict(report.get("generation"))
    requested = str(market or "SPX")
    requested_scope = str(analogue_scope or "ALL")
    rows = _as_list(generation.get("path_quantiles"))
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == requested
            and str(row.get("analogue_key", "ALL")) == requested_scope
        ):
            return row
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == requested
            and str(row.get("analogue_key", "ALL")) == "ALL"
        ):
            return row
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == "SPX"
            and str(row.get("analogue_key", "ALL")) == requested_scope
        ):
            return row
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("market")) == "SPX"
            and str(row.get("analogue_key", "ALL")) == "ALL"
        ):
            return row
    return {}


def fan_chart_figure(
    report: dict[str, Any],
    market: str,
    analogue_scope: str = "ALL",
) -> go.Figure:
    row = _path_quantile_row(report, market, analogue_scope)
    if not row:
        fig = go.Figure()
        fig.update_layout(
            title="No scenario fan data",
            xaxis_title="Forward day",
            yaxis_title="Delta from current state",
            template="plotly_white",
        )
        return fig

    display_name = str(row.get("display_name") or row.get("market") or market)
    days = _float_series(row.get("days"))
    p10 = _float_series(row.get("p10"))
    p50 = _float_series(row.get("p50"))
    p90 = _float_series(row.get("p90"))
    mean = _float_series(row.get("mean"))
    band_x = days + list(reversed(days))
    band_y = p90 + list(reversed(p10))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=band_x,
            y=band_y,
            fill="toself",
            fillcolor="rgba(33, 150, 243, 0.18)",
            line={"color": "rgba(33, 150, 243, 0)"},
            hoverinfo="skip",
            name="P10-P90 band",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p50,
            mode="lines",
            line={"color": "#1565C0", "width": 3},
            name="Median",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=mean,
            mode="lines",
            line={"color": "#455A64", "width": 2, "dash": "dash"},
            name="Mean",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p90,
            mode="lines",
            line={"color": "rgba(21, 101, 192, 0.45)", "width": 1},
            name="P90",
        )
    )
    path_colors = [
        "#2E7D32",
        "#EF6C00",
        "#6A1B9A",
        "#00838F",
        "#AD1457",
        "#5D4037",
    ]
    for idx, path in enumerate(_as_list(row.get("sample_paths"))):
        if not isinstance(path, dict):
            continue
        values = _float_series(path.get("values"))
        if len(values) != len(days):
            continue
        fig.add_trace(
            go.Scatter(
                x=days,
                y=values,
                mode="lines",
                line={
                    "color": path_colors[idx % len(path_colors)],
                    "width": 1.6,
                },
                opacity=0.78,
                name=str(path.get("label", f"Generated path {idx + 1}")),
            )
        )
    realized = _float_series(row.get("realized_path"))
    if len(realized) == len(days):
        fig.add_trace(
            go.Scatter(
                x=days,
                y=realized,
                mode="lines",
                line={"color": "#111111", "width": 3.5},
                name="Realized future",
            )
        )
    fig.update_layout(
        title=f"{display_name} 30-day scenario fan",
        xaxis_title="Forward day",
        yaxis_title="Delta from current state",
        template="plotly_white",
        margin={"l": 55, "r": 25, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    analogue_label = str(row.get("analogue_label", ""))
    if analogue_label and str(row.get("analogue_key", "ALL")) != "ALL":
        fig.add_annotation(
            text=analogue_label,
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.14,
            showarrow=False,
            font={"size": 12, "color": "#455A64"},
            xanchor="left",
        )
    cell = _as_dict(row.get("cell"))
    if cell:
        fig.add_annotation(
            text=f"{cell.get('maturity')} / K={cell.get('moneyness')}",
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.14,
            showarrow=False,
            font={"size": 12, "color": "#455A64"},
            xanchor="right",
        )
    return fig


def refresh_fan_chart(
    report: dict[str, Any] | None,
    fan_market: str,
    analogue_scope: str,
) -> go.Figure:
    return fan_chart_figure(_as_dict(report), fan_market, analogue_scope)


def status_markdown(report: dict[str, Any]) -> str:
    grounding = _as_dict(report.get("grounding"))
    condition = _as_dict(report.get("condition_diagnostics"))
    relevance = _as_dict(report.get("relevance"))
    hard_case = _as_dict(report.get("hard_case_gate"))
    generation = _as_dict(report.get("generation"))
    artifacts = _as_dict(report.get("artifact_paths"))
    return "\n".join(
        [
            "## Run Status",
            "",
            f"- Narrative frame: `{grounding.get('narrative_frame', 'n/a')}`",
            f"- Relevance: `{relevance.get('status', 'n/a')}` - {relevance.get('reason', '')}",
            f"- Hard-case gate: `{hard_case.get('status', 'n/a')}` - {hard_case.get('reason', '')}",
            f"- Top analogue cosine: `{_fmt_float(condition.get('top_cosine'))}`",
            f"- Top analogue gap: `{_fmt_float(condition.get('top_gap'))}`",
            f"- Condition dimension: `{condition.get('condition_dim', 'n/a')}`",
            f"- Generated shape: `{generation.get('generated_state_shape', 'not run')}`",
            f"- Markdown report: `{artifacts.get('markdown', 'n/a')}`",
            f"- JSON report: `{artifacts.get('json', 'n/a')}`",
        ]
    )


def prefix_latent_status_markdown(report: dict[str, Any]) -> str:
    query = _as_dict(report.get("cached_query"))
    gate = _as_dict(report.get("validation_gate"))
    generation = _as_dict(report.get("generation"))
    artifacts = _as_dict(report.get("artifact_paths"))
    return "\n".join(
        [
            "## Prefix-Latent Run Status",
            "",
            f"- Cached query: `{query.get('window_id', 'n/a')}` / `{query.get('kind', 'n/a')}`",
            f"- Condition source: `{query.get('condition_source', 'n/a')}`",
            f"- Text memory dimension: `{query.get('text_memory_dim', 'n/a')}`",
            f"- Overall: `{gate.get('overall_status', 'n/a')}`",
            f"- Operational: `{gate.get('operational_status', 'n/a')}`",
            f"- Stress: `{gate.get('stress_status', 'n/a')}`",
            f"- Endpoint max error: `{_fmt_float(gate.get('endpoint_max_abs_error'), 6)}`",
            f"- Generated shape: `{generation.get('generated_state_shape', 'not run')}`",
            f"- Markdown report: `{artifacts.get('markdown', 'n/a')}`",
            f"- JSON report: `{artifacts.get('report', 'n/a')}`",
        ]
    )


def report_json_text(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def _elapsed_text(start_time: float) -> str:
    return f"{time.monotonic() - float(start_time):.1f}s"


def _progress_status_markdown(
    *,
    start_time: float,
    samples: int,
    top_k: int,
    skip_generator: bool,
) -> str:
    generator_note = (
        "generator sampling skipped"
        if bool(skip_generator)
        else f"sampling {int(samples)} paths per analogue"
    )
    return "\n".join(
        [
            "## Run Status",
            "",
            f"- Run started: `{_elapsed_text(start_time)} ago`",
            "- Current step: `OpenAI grounding, embedding, analogue retrieval, and scenario generation`",
            f"- Requested historical analogues: `{int(top_k)}`",
            f"- Generator work: `{generator_note}`",
            "- Outputs will fill in automatically when the run completes.",
        ]
    )


def _prefix_progress_status_markdown(
    *,
    start_time: float,
    start_mode: str,
    samples: int,
    live_story: bool = False,
) -> str:
    condition_step = (
        "OpenAI grounding and embedding, start selection, prefix decoding, frozen rollout"
        if bool(live_story)
        else "cached text memory, start selection, prefix decoding, frozen rollout"
    )
    return "\n".join(
        [
            "## Prefix-Latent Run Status",
            "",
            f"- Prefix-latent run started: `{_elapsed_text(start_time)} ago`",
            f"- Current step: `{condition_step}`",
            f"- Start mode: `{start_mode}`",
            f"- Generator samples per variant: `{int(samples)}`",
            "- Outputs will fill in automatically when the run completes.",
        ]
    )


def _completed_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return status_markdown(report) + f"\n- Completed in: `{_elapsed_text(start_time)}`"


def _completed_prefix_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return prefix_latent_status_markdown(report) + f"\n- Completed in: `{_elapsed_text(start_time)}`"


def _error_status_markdown(error: BaseException, start_time: float) -> str:
    return "\n".join(
        [
            "## Run Status",
            "",
            "- Status: `error`",
            f"- Failed after: `{_elapsed_text(start_time)}`",
            f"- Error type: `{type(error).__name__}`",
            f"- Message: `{str(error)}`",
        ]
    )


def _blank_run_outputs(
    *,
    status: str,
    fan_market: str,
) -> tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    pd.DataFrame,
    go.Figure,
    str,
    dict[str, Any],
    Any,
]:
    return (
        "Run in progress. Results will appear here when complete.",
        _frame([], IMPLICATION_COLUMNS),
        _frame([], WARNING_COLUMNS),
        _frame([], ANALOGUE_COLUMNS),
        status,
        _frame([], SCENARIO_COLUMNS),
        fan_chart_figure({}, fan_market, "ALL"),
        "{}",
        {},
        analogue_scope_update({}),
    )


def _blank_prefix_outputs(
    *,
    status: str,
    fan_market: str,
) -> tuple[
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    go.Figure,
    str,
    dict[str, Any],
    Any,
]:
    return (
        "Prefix-latent run in progress. Results will appear here when complete.",
        status,
        _frame([], PREFIX_VARIANT_COLUMNS),
        _frame([], VALIDATION_GATE_COLUMNS),
        _frame([], SCENARIO_COLUMNS),
        fan_chart_figure({}, fan_market, "ALL"),
        "{}",
        {},
        analogue_scope_update({}),
    )


def build_run_args(
    *,
    story: str,
    samples: int,
    top_k: int,
    skip_generator: bool,
    output_dir: str = DEFAULT_APP_OUTPUT_DIR,
) -> SimpleNamespace:
    return SimpleNamespace(
        story=str(story or DEFAULT_STORY),
        grounding_json=None,
        grounding_model="gpt-5.4-mini",
        grounding_max_output_tokens=1200,
        embedding_model="text-embedding-3-small",
        dotenv=".env",
        pipeline_report=DEFAULT_PIPELINE_REPORT,
        pipeline_npz=DEFAULT_PIPELINE_NPZ,
        bridge_adapter=DEFAULT_BRIDGE_ADAPTER,
        casebook=DEFAULT_CASEBOOK,
        hard_case_manifest=DEFAULT_HARD_CASE_MANIFEST,
        checkpoint=None,
        top_k=int(top_k),
        ood_threshold=0.75,
        samples=int(samples),
        n_steps=30,
        chunk_size=max(8, min(32, int(samples))),
        temperature=1.0,
        device="cpu",
        skip_generator=bool(skip_generator),
        output_dir=str(output_dir),
    )


def build_prefix_latent_run_args(
    *,
    start_mode: str,
    samples: int,
    live_story: bool = False,
    story: str = DEFAULT_STORY,
    output_dir: str = DEFAULT_PREFIX_APP_OUTPUT_DIR,
) -> SimpleNamespace:
    return SimpleNamespace(
        bridge_report=DEFAULT_PREFIX_BRIDGE_REPORT,
        bridge_arrays=DEFAULT_PREFIX_BRIDGE_ARRAYS,
        pipeline_report=DEFAULT_PIPELINE_REPORT,
        checkpoint=(
            "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
            "best_model.pt"
        ),
        output_dir=str(output_dir),
        query_role="anchor",
        query_kind=None,
        query_window_id=None,
        query_index=0,
        live_story=bool(live_story),
        story=str(story or DEFAULT_STORY),
        grounding_json=None,
        grounding_model="gpt-5.4-mini",
        grounding_max_output_tokens=1200,
        embedding_model="text-embedding-3-small",
        bridge_adapter=DEFAULT_BRIDGE_ADAPTER,
        dotenv=".env",
        start_mode=str(start_mode),
        explicit_start_window_index=None,
        include_original_baseline=True,
        hidden_dim=256,
        steps=1000,
        batch_size=64,
        eval_batch_size=16,
        lr=1e-3,
        seed=791,
        device="cuda",
        skip_rollout=False,
        samples=int(samples),
        n_steps=30,
        chunk_size=max(4, min(16, int(samples))),
        temperature=1.0,
        score_scale_floor=1e-3,
        hard_case_count=8,
        max_paths=6,
        state_scope="joint38",
        eval_split="val",
        test_start=4511,
        val_size=441,
        max_windows=441,
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )


def run_story_for_app(
    story: str,
    samples: int,
    top_k: int,
    fan_market: str,
    analogue_scope: str,
    skip_generator: bool,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_story_smoke,
) -> Any:
    start_time = time.monotonic()
    running_status = _progress_status_markdown(
        start_time=start_time,
        samples=samples,
        top_k=top_k,
        skip_generator=skip_generator,
    )
    yield _blank_run_outputs(status=running_status, fan_market=fan_market)

    args = build_run_args(
        story=story,
        samples=samples,
        top_k=top_k,
        skip_generator=skip_generator,
    )
    try:
        report = runner(args)
    except Exception as error:  # pragma: no cover - defensive UI path
        error_report = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        yield (
            "The run failed before a report could be produced.",
            _frame([], IMPLICATION_COLUMNS),
            _frame([], WARNING_COLUMNS),
            _frame([], ANALOGUE_COLUMNS),
            _error_status_markdown(error, start_time),
            _frame([], SCENARIO_COLUMNS),
            fan_chart_figure({}, fan_market, "ALL"),
            report_json_text(error_report),
            error_report,
            analogue_scope_update({}),
        )
        return

    markdown_path = Path(_as_dict(report.get("artifact_paths")).get("markdown", ""))
    markdown = (
        markdown_path.read_text(encoding="utf-8")
        if str(markdown_path) and markdown_path.exists()
        else render_story_smoke_markdown(report)
    )
    yield (
        markdown,
        implications_table(report),
        warnings_table(report),
        analogues_table(report),
        _completed_status_markdown(report, start_time),
        scenario_table(report),
        fan_chart_figure(report, fan_market, "ALL"),
        report_json_text(report),
        report,
        analogue_scope_update(report),
    )


def run_prefix_latent_for_app(
    start_mode: str,
    samples: int,
    fan_market: str,
    analogue_scope: str,
    live_story: bool = False,
    story: str = DEFAULT_STORY,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_prefix_latent_story_smoke,
) -> Any:
    start_time = time.monotonic()
    running_status = _prefix_progress_status_markdown(
        start_time=start_time,
        start_mode=str(start_mode),
        samples=int(samples),
        live_story=bool(live_story),
    )
    yield _blank_prefix_outputs(status=running_status, fan_market=fan_market)

    args = build_prefix_latent_run_args(
        start_mode=str(start_mode),
        samples=int(samples),
        live_story=bool(live_story),
        story=str(story or DEFAULT_STORY),
    )
    try:
        report = runner(args)
    except Exception as error:  # pragma: no cover - defensive UI path
        error_report = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        yield (
            "The prefix-latent run failed before a report could be produced.",
            _error_status_markdown(error, start_time),
            _frame([], PREFIX_VARIANT_COLUMNS),
            _frame([], VALIDATION_GATE_COLUMNS),
            _frame([], SCENARIO_COLUMNS),
            fan_chart_figure({}, fan_market, "ALL"),
            report_json_text(error_report),
            error_report,
            analogue_scope_update({}),
        )
        return

    markdown_path = Path(_as_dict(report.get("artifact_paths")).get("markdown", ""))
    markdown = (
        markdown_path.read_text(encoding="utf-8")
        if str(markdown_path) and markdown_path.exists()
        else prefix_latent_status_markdown(report)
    )
    yield (
        markdown,
        _completed_prefix_status_markdown(report, start_time),
        prefix_variant_table(report),
        prefix_validation_table(report),
        scenario_table(report),
        fan_chart_figure(report, fan_market, "ALL"),
        report_json_text(report),
        report,
        analogue_scope_update(report),
    )


RunStoryForAppOutput = tuple[
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
    pd.DataFrame,
    go.Figure,
    str,
    dict[str, Any],
    Any,
]


def build_demo() -> Any:
    import gradio as gr

    validation_report = load_validation_gate_report()
    with gr.Blocks(title="Narrative Conditioned Scenario Demo") as demo:
        report_state = gr.State({})
        prefix_report_state = gr.State({})
        gr.Markdown(
            "# Narrative Conditioned Scenario Demo\n"
            "Read this top to bottom: story, grounded market implications, "
            "nearest historical analogues, then the generated 30-day scenario "
            "distribution. After a run, change the factor or analogue selector "
            "to redraw the fan chart without rerunning the generator."
        )
        gr.Markdown("## 1. Risk-manager story")
        story = gr.Textbox(
            label="Risk-manager narrative",
            value=DEFAULT_STORY,
            lines=6,
            max_lines=10,
            placeholder="Describe the market regime and forward risk in plain language.",
        )
        with gr.Row():
            samples = gr.Slider(
                minimum=1,
                maximum=96,
                value=24,
                step=1,
                label="Generator samples per analogue",
                info="Higher values make the fan chart smoother and run slower.",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Historical analogues",
            )
            skip_generator = gr.Checkbox(
                value=False,
                label="Skip generator",
                info="Use for a fast grounding/analogue demo.",
            )
        run_button = gr.Button("Run Scenario", variant="primary")
        status = gr.Markdown(
            "## Run Status\n\n- Waiting for a run. Click `Run Scenario` to start.",
            label="Status",
        )
        gr.Markdown("## 2. Grounded market implications")
        gr.Markdown(
            "These are the explicit market moves extracted from the story. "
            "Warnings mark wording that is interpretive or under-specified."
        )
        implications = gr.Dataframe(
            headers=IMPLICATION_COLUMNS,
            label="Extracted explicit market implications",
            interactive=False,
        )
        warnings = gr.Dataframe(
            headers=WARNING_COLUMNS,
            label="Grounding warnings",
            interactive=False,
        )
        gr.Markdown("## 3. Retrieved historical analogues")
        gr.Markdown(
            "The generator conditions on historical windows whose learned "
            "condition embeddings are nearest to the grounded story. Inspect "
            "this table before switching the fan chart to a single analogue."
        )
        analogues = gr.Dataframe(
            headers=ANALOGUE_COLUMNS,
            label="Nearest historical analogues",
            interactive=False,
        )
        gr.Markdown("## 4. Scenario distribution")
        gr.Markdown(
            "The chart shows generated deltas from the current state over the "
            "next 30 days. After generation, use these two selectors to redraw "
            "the chart without rerunning the model. A single-analogue chart "
            "also overlays representative generated paths and that analogue's "
            "realized future path."
        )
        with gr.Row():
            fan_market = gr.Dropdown(
                choices=FAN_MARKET_CHOICES,
                value="SPX",
                label="Fan chart factor",
                info="Change this after the run to switch markets or IV cells.",
            )
            analogue_scope = gr.Dropdown(
                choices=[("All retrieved analogues", "ALL")],
                value="ALL",
                label="Fan chart analogue set",
                info="Change this after the run to compare pooled vs single analogue scenarios.",
            )
        fan_plot = gr.Plot(label="30-day fan chart")
        scenario = gr.Dataframe(
            headers=SCENARIO_COLUMNS,
            label="Generated 30-day terminal delta summary",
            interactive=False,
        )
        gr.Markdown("## 5. Latent-prefix validation")
        gr.Markdown(
            "This quality gate is computed from cached held-out tests for the "
            "new text-memory-plus-start prefix decoder. It is system-level QC, "
            "not a replacement for inspecting the current story run."
        )
        gr.Markdown(validation_gate_markdown(validation_report))
        gr.Dataframe(
            value=validation_gate_table(validation_report),
            headers=VALIDATION_GATE_COLUMNS,
            label="Top validation hard cases",
            interactive=False,
        )
        gr.Markdown("## 6. Prefix-latent live smoke")
        gr.Markdown(
            "This cached smoke path uses a held-out narrative text memory plus "
            "a selected start state, decodes a recent prefix, and runs the "
            "frozen joint39 generator. It makes no OpenAI calls."
        )
        with gr.Row():
            prefix_start_mode = gr.Dropdown(
                choices=[
                    ("Original start", "original"),
                    ("Nearest train start", "nearest_train_start"),
                    ("Farthest train start", "farthest_train_start"),
                ],
                value="nearest_train_start",
                label="Start mode",
            )
            prefix_live_story = gr.Checkbox(
                value=False,
                label="Use typed story (OpenAI TestFlight)",
                info="Unchecked uses cached held-out text memory. Checked grounds and embeds the story above.",
            )
            prefix_samples = gr.Slider(
                minimum=2,
                maximum=64,
                value=16,
                step=1,
                label="Prefix-latent samples per variant",
            )
        prefix_run_button = gr.Button("Run Prefix-Latent Smoke", variant="secondary")
        prefix_status = gr.Markdown(
            "## Prefix-Latent Run Status\n\n- Waiting for a cached prefix-latent run.",
            label="Prefix-latent status",
        )
        with gr.Row():
            prefix_fan_market = gr.Dropdown(
                choices=FAN_MARKET_CHOICES,
                value="SPX",
                label="Prefix-latent fan chart factor",
            )
            prefix_analogue_scope = gr.Dropdown(
                choices=[("All retrieved analogues", "ALL")],
                value="ALL",
                label="Prefix-latent start variant",
            )
        prefix_fan_plot = gr.Plot(label="Prefix-latent 30-day fan chart")
        prefix_variants = gr.Dataframe(
            headers=PREFIX_VARIANT_COLUMNS,
            label="Prefix-latent start variants",
            interactive=False,
        )
        prefix_validation = gr.Dataframe(
            headers=VALIDATION_GATE_COLUMNS,
            label="Prefix-latent current-run validation",
            interactive=False,
        )
        prefix_scenario = gr.Dataframe(
            headers=SCENARIO_COLUMNS,
            label="Prefix-latent terminal delta summary",
            interactive=False,
        )
        with gr.Accordion("Prefix-latent Markdown report", open=False):
            prefix_report_markdown = gr.Markdown(label="Prefix-latent report")
        with gr.Accordion("Prefix-latent raw JSON report", open=False):
            prefix_report_json = gr.Code(language="json", label="Prefix-latent JSON")
        with gr.Accordion("Full Markdown report", open=False):
            report_markdown = gr.Markdown(label="Full report")
        with gr.Accordion("Raw JSON report", open=False):
            report_json = gr.Code(language="json", label="Full JSON report")
        run_button.click(
            fn=run_story_for_app,
            inputs=[story, samples, top_k, fan_market, analogue_scope, skip_generator],
            outputs=[
                report_markdown,
                implications,
                warnings,
                analogues,
                status,
                scenario,
                fan_plot,
                report_json,
                report_state,
                analogue_scope,
            ],
            show_progress="full",
            show_progress_on=status,
        )
        fan_market.change(
            fn=refresh_fan_chart,
            inputs=[report_state, fan_market, analogue_scope],
            outputs=fan_plot,
            show_progress="hidden",
        )
        analogue_scope.change(
            fn=refresh_fan_chart,
            inputs=[report_state, fan_market, analogue_scope],
            outputs=fan_plot,
            show_progress="hidden",
        )
        prefix_run_button.click(
            fn=run_prefix_latent_for_app,
            inputs=[
                prefix_start_mode,
                prefix_samples,
                prefix_fan_market,
                prefix_analogue_scope,
                prefix_live_story,
                story,
            ],
            outputs=[
                prefix_report_markdown,
                prefix_status,
                prefix_variants,
                prefix_validation,
                prefix_scenario,
                prefix_fan_plot,
                prefix_report_json,
                prefix_report_state,
                prefix_analogue_scope,
            ],
            show_progress="full",
            show_progress_on=prefix_status,
        )
        prefix_fan_market.change(
            fn=refresh_fan_chart,
            inputs=[prefix_report_state, prefix_fan_market, prefix_analogue_scope],
            outputs=prefix_fan_plot,
            show_progress="hidden",
        )
        prefix_analogue_scope.change(
            fn=refresh_fan_chart,
            inputs=[prefix_report_state, prefix_fan_market, prefix_analogue_scope],
            outputs=prefix_fan_plot,
            show_progress="hidden",
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=int(args.server_port),
        share=bool(args.share),
    )


if __name__ == "__main__":
    main()
