#!/usr/bin/env python
"""Structured display helpers for the scenario-to-narrative workbench."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (  # noqa: E402
    ScenarioSidecarV1,
    _compact,
    load_generated_deck_sidecar_from_report,
    load_historical_joint39_sidecar,
    normalize_factor_table_csv_text,
    run_workbench_packet,
    select_sidecar_negative_candidates,
)


# CLEAN stride-5 corpus (802 windows, joint39_train_0000 … joint39_train_4005, stride 5).
# Coverage caveat: covers only stride-5 train windows (rows 0–4035 of multi_factor_data.npz,
# every 5th row), 802 cards total out of 4010 possible stride-1 train positions.
# The contaminated 982g_sharded corpus (episode_card_v3_full_codex_multiformat_982g_sharded)
# MUST NOT be used — it was generated with the buggy joint39 factor-mapping (2026-06-11
# contamination; USDJPY→col 29 / AAA_OAS→col 36 instead of the canonical 27 / 34).
DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_codex_multiformat_982g_clean_stride5_20260612/"
    "multiformat_episode_cards.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "scenario_to_narrative_workbench"
)
DEFAULT_SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_SUPPORT_BANK_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
GENERATION_CANDIDATE_COUNT = 40
FACTOR_ROW_COLUMNS = [
    "Factor",
    "Start",
    "End",
    "Delta",
    "Direction",
    "Magnitude",
    "Confidence",
]
PACKET_COLUMNS = ["View", "Positive", "Negative Window", "Hard Negative"]
SOURCE_MODE_CHOICES = ["Historical Case", "Uploaded Numerical Scenario"]

# Factors quoted in percent (yields + credit OAS spreads): level shown as "%", move in bp.
_RATE_FACTORS = {"US2Y", "US5Y", "US10Y", "US30Y", "US3M", "US6M", "FED_FUNDS"}
# Confidence column: real levels pass through; the historical/generated builders overload the
# `confidence` field as a provenance tag (raw_history_path / historical_evidence /
# distribution_mean / support_row_N), so map those to an honest confidence level for display.
_CONFIDENCE_LEVELS = {"low", "medium", "high"}
_OBSERVED_PROVENANCE = {"raw_history_path", "historical_evidence", "small", "large", "flat"}
# Plain-language source label for the status panel (replaces ScenarioSidecarV1 / type-code jargon).
_SOURCE_LABELS = {
    "historical_joint39": "Historical case",
    "generated_deck": "Generated scenario deck",
    "factor_table_partial": "Uploaded numbers",
    "factor_table_full": "Uploaded numbers",
}
# Snake_case narrative view keys -> human-readable titles for the packet.
_VIEW_TITLES = {
    "sparse_user_query": "Risk-desk question",
    "weekly_risk_monitor": "Weekly risk monitor",
    "mechanism_first": "Mechanism-first read",
    "technical_factor_evidence": "Technical factor evidence",
    "factor_list_baseline": "Factor checklist",
    "institutional_risk_committee_note": "Risk committee note",
    "risk_manager_memo": "Risk-manager memo",
    "full_professional": "Full professional brief",
    "sparse_variant_tape_read": "Tape read (brief)",
    "sparse_variant_portfolio_concern": "Portfolio concern (brief)",
    "sparse_variant_macro_channel": "Macro channel (brief)",
    "sparse_variant_credit_ambiguity": "Credit ambiguity (brief)",
    "sparse_variant_rates_commodities": "Rates & commodities (brief)",
    "sparse_variant_desk_note": "Desk note (brief)",
}


def _factor_is_rate_or_spread(factor: str) -> bool:
    name = str(factor or "").strip().upper()
    return name in _RATE_FACTORS or name.endswith("_OAS") or name.endswith(" OAS")


def _factor_unit_label(factor: str) -> str:
    return "%" if _factor_is_rate_or_spread(factor) else "pts"


def _format_factor_row_values(row: ScenarioFactorRowV1) -> tuple[str, str, str]:
    """Return (start, end, delta) as clean per-factor unit strings (no float noise).

    Yields/OAS levels are in percent -> level "x.xx%", move in basis points.
    All other factors -> "pts" (0 dp for index-scale levels >= 1000, else 2 dp).
    """
    try:
        start, end, delta = float(row.start), float(row.end), float(row.delta)
    except (TypeError, ValueError):
        return str(row.start), str(row.end), str(row.delta)
    if _factor_is_rate_or_spread(row.factor):
        return f"{start:.2f}%", f"{end:.2f}%", f"{delta * 100.0:+,.0f} bp"
    decimals = 0 if max(abs(start), abs(end)) >= 1000 else 2
    return (
        f"{start:,.{decimals}f} pts",
        f"{end:,.{decimals}f} pts",
        f"{delta:+,.{decimals}f} pts",
    )


def _display_confidence(raw: object, scenario_type: str = "") -> str:
    # Disambiguate by source builder: historical rows are always realized observations and
    # generated-deck rows are always distribution means, regardless of what the (overloaded)
    # `confidence` field literally holds. This closes the "medium"-magnitude hole on the
    # support-metadata fallback path, where a magnitude string lands in the confidence field
    # and would otherwise read as a user confidence level.
    stype = str(scenario_type or "").strip()
    if stype == "historical_joint39":
        return "Observed"  # realized historical move — not a confidence judgment
    if stype == "generated_deck":
        return "Model mean"  # distribution mean, not a confidence
    text = str(raw or "").strip()
    if not text:
        return "Medium"
    lowered = text.lower()
    if lowered in _CONFIDENCE_LEVELS:
        return lowered.capitalize()  # user-supplied confidence (uploaded case)
    if lowered in _OBSERVED_PROVENANCE or lowered.startswith("support_row_"):
        return "Observed"
    if lowered == "distribution_mean":
        return "Model mean"
    return text.capitalize()


def _friendly_source_label(scenario_type: str) -> str:
    return _SOURCE_LABELS.get(str(scenario_type or "").strip(), "Scenario")


def _view_title(view_key: object) -> str:
    key = str(view_key or "").strip()
    if key in _VIEW_TITLES:
        return _VIEW_TITLES[key]
    return key.replace("_", " ").strip().title() or "Narrative"


@lru_cache(maxsize=8)
def _window_id_period_map(
    support_report_path: str = str(DEFAULT_SUPPORT_BANK_REPORT),
) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for row in _support_window_metadata(support_report_path):
        window_id = str(row.get("window_id", "")).strip()
        start_date = str(row.get("calendar_start_date", "")).strip()
        end_date = str(row.get("calendar_end_date", "")).strip()
        if window_id and start_date and end_date:
            mapping[window_id] = (start_date, end_date)
    return mapping


def _negative_window_label(window_id: object) -> str:
    """Resolve a hard-negative window id to its calendar period (hide the internal id)."""
    window_text = str(window_id or "").strip()
    if not window_text:
        return ""
    try:
        period = _window_id_period_map().get(window_text)
    except Exception:
        period = None
    if period:
        return f"Contrasting historical period: {period[0]} to {period[1]}"
    return f"Contrasting window: `{window_text}`"


def factor_rows_dataframe(sidecar: ScenarioSidecarV1) -> pd.DataFrame:
    rows = []
    for row in sidecar.factor_rows:
        start_str, end_str, delta_str = _format_factor_row_values(row)
        rows.append(
            {
                "Factor": row.factor,
                "Start": start_str,
                "End": end_str,
                "Delta": delta_str,
                "Direction": row.direction,
                "Magnitude": row.magnitude,
                "Confidence": _display_confidence(
                    row.confidence, sidecar.scenario_type
                ),
            }
        )
    return pd.DataFrame(rows, columns=FACTOR_ROW_COLUMNS)


def packet_preview_rows(packet: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "View": row.get("view_name", ""),
            "Positive": row.get("positive_text", ""),
            "Negative Window": row.get("negative_window_id", ""),
            "Hard Negative": row.get("negative_text", ""),
        }
        for row in packet.get("paired_review", [])
        if isinstance(row, dict)
    ]
    return pd.DataFrame(rows, columns=PACKET_COLUMNS)


def packet_review_markdown(packet: dict[str, Any] | None) -> str:
    pairs = []
    if isinstance(packet, dict):
        raw_pairs = packet.get("paired_review", [])
        if isinstance(raw_pairs, list):
            pairs = [row for row in raw_pairs if isinstance(row, dict)]
    if not pairs:
        return "## Generated Narratives\n\nGenerate a packet to view the narratives."

    lines = [
        "## Generated Narratives",
        "Each market scenario below is described two ways. The **Positive** narrative is a "
        "faithful plain-English reading of your scenario. The **Hard negative** is a "
        "deliberately-contradictory description used to confirm your scenario can be told apart "
        "from a near-miss — it is NOT a valid reading of your scenario and NOT a more-bearish "
        "variant.",
    ]
    for index, row in enumerate(pairs, start=1):
        view_title = _view_title(row.get("view_name"))
        negative_label = _negative_window_label(row.get("negative_window_id"))
        positive = str(row.get("positive_text", "")).strip()
        negative = str(row.get("negative_text", "")).strip()
        lines.extend(
            [
                f"### {index}. {view_title}",
                "**Positive**",
                positive or "_No positive narrative returned._",
                "**Hard negative**" + (f"  \n{negative_label}" if negative_label else ""),
                negative or "_No hard-negative narrative returned._",
            ]
        )
    return "\n\n".join(lines)


def status_cards_markdown(
    sidecar: ScenarioSidecarV1 | None,
    *,
    validation_status: str,
) -> str:
    if sidecar is None:
        return "## Scenario status\n\n- Scenario: `waiting`\n- Input check: `waiting`"
    return (
        "## Scenario status\n\n"
        "- Scenario: `loaded`\n"
        f"- Source: `{_friendly_source_label(sidecar.scenario_type)}`\n"
        f"- Factors: `{len(sidecar.factor_rows)}`\n"
        f"- Data notes: `{len(sidecar.normalization_warnings)}`\n"
        f"- Input check: `{validation_status}`"
    )


def factor_move_plot(sidecar: ScenarioSidecarV1 | None) -> go.Figure:
    if sidecar is None or not sidecar.factor_rows:
        fig = go.Figure()
        fig.update_layout(title="No scenario loaded")
        return fig

    factor_rows = list(sidecar.factor_rows)
    n_factors = len(factor_rows)
    horizon = max(1, int(sidecar.horizon_days))
    days = list(range(horizon + 1))
    # Small-multiples grid: single column when few factors, otherwise two columns so the
    # full Joint39 set (11 panels) stays a compact block instead of a long scroll.
    n_cols = 1 if n_factors <= 4 else 2
    n_rows = math.ceil(n_factors / n_cols)
    # One label+unit per panel (panel title); no duplicate y-axis title.
    subplot_titles = [
        f"{row.factor} ({_factor_unit_label(row.factor)})" for row in factor_rows
    ]
    make_kwargs: dict[str, Any] = {
        "rows": n_rows,
        "cols": n_cols,
        "shared_xaxes": n_cols == 1,
        "subplot_titles": subplot_titles,
    }
    if n_cols > 1:
        make_kwargs["horizontal_spacing"] = 0.12
    if n_rows > 1:
        make_kwargs["vertical_spacing"] = min(0.08, 0.5 / (n_rows - 1))
    fig = make_subplots(**make_kwargs)

    deepest_row_for_col: dict[int, int] = {}
    for index, row in enumerate(factor_rows):
        grid_row = index // n_cols + 1
        grid_col = index % n_cols + 1
        deepest_row_for_col[grid_col] = max(
            deepest_row_for_col.get(grid_col, 0), grid_row
        )
        if row.path_values and len(row.path_values) >= 2:
            values = [float(value) for value in row.path_values]
            x_values = list(range(len(values)))
            trace_mode = "lines+markers"
        else:
            values = [
                row.start + (row.end - row.start) * (day / horizon)
                for day in days
            ]
            x_values = days
            trace_mode = "lines"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                mode=trace_mode,
                name=row.factor,
                marker={"size": 4},
                hovertemplate=(
                    f"{row.factor}<br>"
                    "Day %{x}<br>"
                    f"Value %{{y:.4g}} {_factor_unit_label(row.factor)}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=grid_row,
            col=grid_col,
        )
    for grid_col, grid_row in deepest_row_for_col.items():
        fig.update_xaxes(title_text="Day", row=grid_row, col=grid_col)
    row_height = 150 if n_cols > 1 else (140 if n_factors <= 12 else 105)
    fig.update_layout(
        title="30-day scenario movement",
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 70, "r": 20, "t": 70, "b": 55},
        height=max(420, row_height * n_rows + 120),
    )
    return fig


def _app_warnings_for_sidecar(sidecar: ScenarioSidecarV1) -> list[str | dict[str, str]]:
    warnings = list(sidecar.normalization_warnings)
    if sidecar.scenario_type == "factor_table_partial" and not any(
        isinstance(warning, dict) and warning.get("code") == "partial_factor_coverage"
        for warning in warnings
    ):
        warnings.append(
            {
                "code": "partial_factor_coverage",
                "message": "Uploaded table does not contain the full Joint39 factor set.",
            }
        )
    return warnings


def _normalize_factor_table_sidecar_for_app(csv_text: str) -> ScenarioSidecarV1:
    sidecar = normalize_factor_table_csv_text(
        csv_text,
        scenario_id="uploaded_factor_table",
    )
    return sidecar.model_copy(
        update={"normalization_warnings": _app_warnings_for_sidecar(sidecar)}
    )


def _sidecar_outputs(
    sidecar: ScenarioSidecarV1,
) -> tuple[str, pd.DataFrame, str, str, str]:
    sidecar_json = sidecar.model_dump_json(indent=2)
    return (
        status_cards_markdown(sidecar, validation_status="normalized"),
        factor_rows_dataframe(sidecar),
        json.dumps(sidecar.normalization_warnings, indent=2, sort_keys=True),
        sidecar_json,
        sidecar_json,
    )


def normalize_factor_table_for_app(
    csv_text: str,
) -> tuple[str, pd.DataFrame, str, str, str]:
    sidecar = _normalize_factor_table_sidecar_for_app(csv_text)
    return _sidecar_outputs(sidecar)


def normalize_uploaded_scenario_for_app(
    csv_text: str,
) -> tuple[str, pd.DataFrame, str, str, str]:
    return normalize_factor_table_for_app(csv_text)


@lru_cache(maxsize=8)
def _support_window_metadata(
    support_report_path: str = str(DEFAULT_SUPPORT_BANK_REPORT),
) -> tuple[dict[str, Any], ...]:
    path = _resolve_app_path(support_report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("window_metadata", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path}: window_metadata must be a list")
    return tuple(row for row in rows if isinstance(row, dict))


def _metadata_for_calendar_end_date(
    calendar_end_date: str,
    *,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> dict[str, Any]:
    selected = str(calendar_end_date or "").strip()
    for row in _support_window_metadata(str(support_report_path)):
        if str(row.get("calendar_end_date", "")).strip() == selected:
            return dict(row)
    raise ValueError(f"no historical support window ending on {selected!r}")


def _metadata_for_calendar_start_date(
    calendar_start_date: str,
    *,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> dict[str, Any]:
    selected = str(calendar_start_date or "").strip()
    for row in _support_window_metadata(str(support_report_path)):
        if str(row.get("calendar_start_date", "")).strip() == selected:
            return dict(row)
    raise ValueError(f"no historical support window starting on {selected!r}")


@lru_cache(maxsize=8)
def _card_backed_window_ids(cards_jsonl: str = str(DEFAULT_CARDS_JSONL)) -> frozenset[str]:
    """Window ids that actually have a narrative card in the corpus the scenario loader
    reads. The support bank is stride-1 (~4010 windows) but the clean corpus is stride-5
    (~802 cards), so the date picker must offer only card-backed windows — otherwise the
    loader raises 'target window not found' for the ~80% of dates that have no card."""
    path = Path(cards_jsonl)
    if not path.exists():
        return frozenset()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                window_id = json.loads(line).get("window_id")
            except json.JSONDecodeError:
                continue
            if window_id:
                ids.add(_compact(window_id))
    return frozenset(ids)


def historical_date_choices(
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
    *,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
) -> list[tuple[str, str]]:
    card_ids = _card_backed_window_ids(str(cards_jsonl))
    choices: list[tuple[str, str]] = []
    for row in _support_window_metadata(str(support_report_path)):
        end_date = str(row.get("calendar_end_date", "")).strip()
        start_date = str(row.get("calendar_start_date", "")).strip()
        window_id = str(row.get("window_id", "")).strip()
        if not end_date or not window_id:
            continue
        if card_ids and _compact(window_id) not in card_ids:
            continue
        label = f"{end_date} | {window_id} | {start_date} to {end_date}"
        choices.append((label, end_date))
    return choices


def historical_start_date_choices(
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
    *,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
) -> list[tuple[str, str]]:
    card_ids = _card_backed_window_ids(str(cards_jsonl))
    choices: list[tuple[str, str]] = []
    for row in _support_window_metadata(str(support_report_path)):
        start_date = str(row.get("calendar_start_date", "")).strip()
        if not start_date:
            continue
        if card_ids and _compact(row.get("window_id")) not in card_ids:
            continue
        choices.append((start_date, start_date))
    return choices


def window_id_for_calendar_end_date(
    calendar_end_date: str,
    *,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> str:
    row = _metadata_for_calendar_end_date(
        calendar_end_date,
        support_report_path=support_report_path,
    )
    return str(row.get("window_id", ""))


def window_id_for_calendar_start_date(
    calendar_start_date: str,
    *,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> str:
    row = _metadata_for_calendar_start_date(
        calendar_start_date,
        support_report_path=support_report_path,
    )
    return str(row.get("window_id", ""))


def historical_window_range_markdown(
    calendar_end_date: str,
    *,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> str:
    try:
        row = _metadata_for_calendar_end_date(
            calendar_end_date,
            support_report_path=support_report_path,
        )
    except ValueError as exc:
        return f"## 30-Day Conditioning Window\n\n- Status: `{exc}`"
    start_date = row.get("calendar_start_date", "")
    end_date = row.get("calendar_end_date", "")
    forecast_start = row.get("forecast_start_date", "")
    forecast_end = row.get("forecast_end_date", "")
    return (
        "## 30-Day Conditioning Window\n\n"
        f"- Historical case: `{row.get('window_id')}`\n"
        f"- 30-day conditioning window: `{start_date} to {end_date}`\n"
        f"- Scenario horizon after condition: `{forecast_start} to {forecast_end}`"
    )


def historical_period_range_markdown(
    calendar_start_date: str,
    *,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> str:
    try:
        row = _metadata_for_calendar_start_date(
            calendar_start_date,
            support_report_path=support_report_path,
        )
    except ValueError as exc:
        return f"## Selected Historical Period\n\n- Status: `{exc}`"
    start_date = row.get("calendar_start_date", "")
    end_date = row.get("calendar_end_date", "")
    forecast_start = row.get("forecast_start_date", "")
    forecast_end = row.get("forecast_end_date", "")
    return (
        "## Selected Historical Period\n\n"
        f"- 30-day historical period: `{start_date} to {end_date}`\n"
        f"- Scenario horizon after period: `{forecast_start} to {forecast_end}`"
    )


def _normalize_historical_sidecar_for_app(
    target_window_id: str,
    *,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    support_arrays_path: str | Path | None = None,
) -> ScenarioSidecarV1:
    cards_path = Path(cards_jsonl)
    if not cards_path.is_absolute():
        cards_path = ROOT / cards_path
    sidecar, _card = load_historical_joint39_sidecar(
        cards_jsonl=cards_path,
        target_window_id=target_window_id.strip(),
        support_arrays_path=support_arrays_path,
    )
    return sidecar


def _normalize_historical_date_sidecar_for_app(
    calendar_end_date: str,
    *,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    support_arrays_path: str | Path | None = None,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> ScenarioSidecarV1:
    window_id = window_id_for_calendar_end_date(
        calendar_end_date,
        support_report_path=support_report_path,
    )
    return _normalize_historical_sidecar_for_app(
        window_id,
        cards_jsonl=cards_jsonl,
        support_arrays_path=support_arrays_path,
    )


def _normalize_historical_start_date_sidecar_for_app(
    calendar_start_date: str,
    *,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    support_arrays_path: str | Path | None = None,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> ScenarioSidecarV1:
    window_id = window_id_for_calendar_start_date(
        calendar_start_date,
        support_report_path=support_report_path,
    )
    return _normalize_historical_sidecar_for_app(
        window_id,
        cards_jsonl=cards_jsonl,
        support_arrays_path=support_arrays_path,
    )


def normalize_historical_for_app(
    target_window_id: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    support_arrays_path: str | Path | None = None,
) -> tuple[str, pd.DataFrame, str, str, str]:
    sidecar = _normalize_historical_sidecar_for_app(
        target_window_id,
        cards_jsonl=cards_jsonl,
        support_arrays_path=support_arrays_path,
    )
    return _sidecar_outputs(sidecar)


def normalize_historical_date_for_app(
    calendar_end_date: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    support_arrays_path: str | Path | None = DEFAULT_SUPPORT_ARRAYS,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> tuple[str, pd.DataFrame, str, str, str, str]:
    sidecar = _normalize_historical_date_sidecar_for_app(
        calendar_end_date,
        cards_jsonl=cards_jsonl,
        support_arrays_path=support_arrays_path,
        support_report_path=support_report_path,
    )
    status, frame, warnings_json, sidecar_json, sidecar_state = _sidecar_outputs(sidecar)
    return (
        status,
        frame,
        warnings_json,
        sidecar_json,
        sidecar_state,
        historical_window_range_markdown(
            calendar_end_date,
            support_report_path=support_report_path,
        ),
    )


def normalize_historical_start_date_for_app(
    calendar_start_date: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    support_arrays_path: str | Path | None = DEFAULT_SUPPORT_ARRAYS,
    support_report_path: str | Path = DEFAULT_SUPPORT_BANK_REPORT,
) -> tuple[str, pd.DataFrame, str, str, str, str]:
    sidecar = _normalize_historical_start_date_sidecar_for_app(
        calendar_start_date,
        cards_jsonl=cards_jsonl,
        support_arrays_path=support_arrays_path,
        support_report_path=support_report_path,
    )
    status, frame, warnings_json, sidecar_json, sidecar_state = _sidecar_outputs(sidecar)
    return (
        status,
        frame,
        warnings_json,
        sidecar_json,
        sidecar_state,
        historical_period_range_markdown(
            calendar_start_date,
            support_report_path=support_report_path,
        ),
    )


def _normalize_generated_deck_sidecar_for_app(report_path: str) -> ScenarioSidecarV1:
    return load_generated_deck_sidecar_from_report(report_path.strip())


def normalize_generated_deck_for_app(
    report_path: str,
) -> tuple[str, pd.DataFrame, str, str, str]:
    sidecar = _normalize_generated_deck_sidecar_for_app(report_path)
    return _sidecar_outputs(sidecar)


def mode_visibility_flags(source_mode: str) -> tuple[bool, bool]:
    selected = str(source_mode or "").strip()
    if selected not in SOURCE_MODE_CHOICES:
        selected = "Uploaded Numerical Scenario"
    return (
        selected == "Historical Case",
        selected == "Uploaded Numerical Scenario",
    )


def _resolve_app_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _read_jsonl_cards(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve_app_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _scenario_slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return slug.strip("_") or "scenario"


def generation_status_markdown(
    packet_path: str,
    validation_status: str,
    error_count: int,
) -> str:
    # `packet_path` is retained for call-site stability but intentionally NOT printed here:
    # the saved artifact is offered through the download button / Technical details instead of
    # leaking a full filesystem path into the primary view (D9).
    del packet_path
    return (
        "## Generation Status\n\n"
        f"- Narrative check: `{validation_status}`\n"
        f"- Errors: `{error_count}`"
    )


def generate_narrative_packet_for_app(
    sidecar_json: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dry_run: bool = False,
) -> str:
    status, _narratives = generate_narrative_packet_outputs_for_app(
        sidecar_json,
        cards_jsonl=cards_jsonl,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    return status


def generate_narrative_packet_outputs_for_app(
    sidecar_json: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dry_run: bool = False,
) -> tuple[str, str]:
    if not str(sidecar_json or "").strip():
        return (
            "## Generation Status\n\n"
            "- Narrative check: `waiting`\n"
            "- Note: `load a scenario first`",
            "## Generated Narratives\n\nload a scenario first, then generate a packet.",
        )
    sidecar = ScenarioSidecarV1.model_validate_json(sidecar_json)
    cards = _read_jsonl_cards(cards_jsonl)
    candidates = select_sidecar_negative_candidates(
        sidecar=sidecar,
        cards=cards,
        count=GENERATION_CANDIDATE_COUNT,
    )
    scenario_output_dir = _resolve_app_path(output_dir) / _scenario_slug(
        sidecar.scenario_id
    )
    packet = run_workbench_packet(
        sidecar=sidecar,
        negative_candidates=candidates,
        output_dir=scenario_output_dir,
        dry_run=bool(dry_run),
    )
    validation = packet.validation
    return (
        generation_status_markdown(
            packet.artifact_paths.get("packet", ""),
            str(validation.get("status", "unknown")),
            int(validation.get("error_count", 0) or 0),
        ),
        packet_review_markdown(packet.model_dump()),
    )


# ---------------------------------------------------------------------------
# Corpus path safety
# ---------------------------------------------------------------------------
_CONTAMINATED_PATH_FRAGMENTS: tuple[str, ...] = (
    "episode_card_v3_full_codex_multiformat_982g_sharded",
    "stride5_fourteen_view_bank_988b",
    "990f",
    "990a",
    "991a_seed",
)
_CLEAN_CORPUS_PATHS: tuple[str, ...] = (
    "episode_card_v3_codex_multiformat_982g_clean_stride5_20260612",
    "prefix_latent_support_bank_train_all_939a",
)


def assert_clean_corpus_paths(*paths: "str | Path") -> None:
    """Raise RuntimeError if any supplied path matches a known-contaminated fragment.

    Call this at module load / app startup before serving any narrative text.
    The joint39 factor-mapping contamination (2026-06-11) caused USDJPY and
    AAA_OAS narratives to be derived from the wrong data columns (col 29 / 36
    instead of canonical 27 / 34).  The contaminated corpora are:
      - episode_card_v3_full_codex_multiformat_982g_sharded
      - stride5_fourteen_view_bank_988b, 990f, 990a, 991a_seed*
    """
    for raw_path in paths:
        path_str = str(raw_path)
        for fragment in _CONTAMINATED_PATH_FRAGMENTS:
            if fragment in path_str:
                raise RuntimeError(
                    f"[contamination-gate] BLOCKED: path contains known-contaminated "
                    f"fragment '{fragment}': {path_str!r}\n"
                    "Repoint to a clean corpus before starting this app."
                )


# Startup check — fails loudly if constants have been accidentally reverted.
assert_clean_corpus_paths(DEFAULT_CARDS_JSONL, DEFAULT_SUPPORT_ARRAYS, DEFAULT_SUPPORT_BANK_REPORT)


# ---------------------------------------------------------------------------
# UI-facing error handling (keep friendly messages in the persistent panels)
# ---------------------------------------------------------------------------
def _empty_factor_frame() -> pd.DataFrame:
    return pd.DataFrame([], columns=FACTOR_ROW_COLUMNS)


def _load_error_status_markdown(error: Exception) -> str:
    message = str(error).strip() or "could not parse the scenario input"
    return (
        "## Scenario status\n\n"
        "- Scenario: `could not load`\n"
        f"- Problem: {message}\n"
        "- Input check: `failed`"
    )


def _historical_error_range_markdown(error: Exception) -> str:
    message = str(error).strip() or "could not load the selected period"
    return f"## Selected Historical Period\n\n- Status: `{message}`"


def _generation_failure_markdown(headline: str, detail: str) -> str:
    return (
        "## Generation Status\n\n"
        f"- Narrative generation: `{headline}`\n"
        f"- Detail: {detail}"
    )


def normalize_uploaded_scenario_safe_for_app(
    csv_text: str,
) -> tuple[str, pd.DataFrame, str, str, str]:
    """D5: catch parse/validation errors into the status panel (5 outputs always)."""
    try:
        return normalize_uploaded_scenario_for_app(csv_text)
    except (ValueError, ValidationError) as exc:
        return (_load_error_status_markdown(exc), _empty_factor_frame(), "[]", "", "")


def normalize_historical_start_date_safe_for_app(
    calendar_start_date: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    support_arrays_path: str | Path | None = DEFAULT_SUPPORT_ARRAYS,
) -> tuple[str, pd.DataFrame, str, str, str, str]:
    """D5: catch lookup/validation/IO errors into the status panel (6 outputs always)."""
    try:
        return normalize_historical_start_date_for_app(
            calendar_start_date,
            cards_jsonl=cards_jsonl,
            support_arrays_path=support_arrays_path,
        )
    # OSError covers a bad "Reference cards JSONL" path (FileNotFoundError); the broad fallback
    # ensures any backend failure lands in the panel rather than leaving it stuck on "waiting".
    except (ValueError, ValidationError, OSError) as exc:
        error: Exception = exc
    except Exception as exc:  # noqa: BLE001
        error = exc
    return (
        _load_error_status_markdown(error),
        _empty_factor_frame(),
        "[]",
        "",
        "",
        _historical_error_range_markdown(error),
    )


def _uploaded_plot_safe(csv_text: str) -> go.Figure:
    try:
        return factor_move_plot(_normalize_factor_table_sidecar_for_app(csv_text))
    except (ValueError, ValidationError):
        return factor_move_plot(None)


def _historical_plot_safe(
    calendar_start_date: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
) -> go.Figure:
    try:
        return factor_move_plot(
            _normalize_historical_start_date_sidecar_for_app(
                calendar_start_date,
                cards_jsonl=cards_jsonl,
                support_arrays_path=DEFAULT_SUPPORT_ARRAYS,
            )
        )
    except Exception:  # noqa: BLE001 - bad path/parse -> empty plot; message is in the status panel
        return factor_move_plot(None)


def _existing_packet_path(sidecar_json: str, output_dir: str | Path) -> str | None:
    """Locate the just-written packet JSON for the download button (D9)."""
    text = str(sidecar_json or "").strip()
    if not text:
        return None
    try:
        sidecar = ScenarioSidecarV1.model_validate_json(text)
    except (ValueError, ValidationError):
        return None
    path = (
        _resolve_app_path(output_dir)
        / _scenario_slug(sidecar.scenario_id)
        / "scenario_narrative_packet.json"
    )
    return str(path) if path.exists() else None


def generate_narrative_packet_ui(
    sidecar_json: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[str, str, str | None]:
    """D6 + D9: error-guarded generation returning (status, narratives, download_path)."""
    try:
        status, narratives = generate_narrative_packet_outputs_for_app(
            sidecar_json,
            cards_jsonl=cards_jsonl,
            output_dir=output_dir,
        )
    except subprocess.TimeoutExpired:
        return (
            _generation_failure_markdown(
                "timed out",
                "The narrative author (Codex) did not return in time. Try again, or load a "
                "smaller scenario.",
            ),
            packet_review_markdown(None),
            None,
        )
    except FileNotFoundError:
        return (
            _generation_failure_markdown(
                "backend unavailable",
                "The narrative author (Codex CLI) was not found on this machine. Install/enable "
                "it, then retry.",
            ),
            packet_review_markdown(None),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - surface any backend failure legibly in-panel
        return (
            _generation_failure_markdown("failed", f"{type(exc).__name__}: {exc}"),
            packet_review_markdown(None),
            None,
        )
    return status, narratives, _existing_packet_path(sidecar_json, output_dir)


def _generation_pending_outputs() -> tuple[str, str, None]:
    """D6: upfront latency notice shown the instant Generate is clicked."""
    return (
        "## Generation Status\n\n"
        "_Generating narrative packet — this can take several minutes while the narrative "
        "author drafts each style._",
        "## Generated Narratives\n\n_Authoring narratives…_",
        None,
    )


def build_demo() -> Any:
    import gradio as gr

    start_date_choices = historical_start_date_choices()
    choice_values = [str(value) for _label, value in start_date_choices]
    default_historical_start_date = (
        "2008-09-22"
        if "2008-09-22" in choice_values
        else (choice_values[0] if choice_values else "")
    )

    def mode_visibility_updates(source_mode: str) -> tuple[Any, Any]:
        return tuple(
            gr.update(visible=visible)
            for visible in mode_visibility_flags(source_mode)
        )

    def mode_change_outputs(source_mode: str) -> tuple[Any, ...]:
        # D10: toggle the active input group AND clear the right-panel outputs so stale data
        # from the previous mode never lingers after switching.
        historical_update, uploaded_update = mode_visibility_updates(source_mode)
        return (
            historical_update,
            uploaded_update,
            status_cards_markdown(None, validation_status="waiting"),
            _empty_factor_frame(),
            factor_move_plot(None),
            "",
            "## Generation Status\n\n- Narrative check: `waiting`",
            "## Generated Narratives\n\nGenerate a packet to view the narratives.",
            None,
            "",  # warnings_json (Technical details)
            "",  # sidecar_json (Technical details)
        )

    with gr.Blocks(title="Scenario-to-Narrative Generator") as demo:
        gr.Markdown("# Scenario-to-Narrative Generator")
        gr.Markdown(
            "Turn a market scenario — a historical episode or your own numbers — into "
            "plain-English risk narratives, each paired with a deliberately-contradictory "
            "hard-negative so you can confirm the read is distinguishable."
        )
        gr.Markdown(
            "**Two steps:** first **Visualize** a scenario on the left, then **Generate** its "
            "narrative packet on the right."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                source_mode = gr.Radio(
                    choices=SOURCE_MODE_CHOICES,
                    value="Historical Case",
                    label="Input mode",
                )
                with gr.Column(visible=True) as historical_group:
                    historical_start_date = gr.Dropdown(
                        choices=start_date_choices,
                        value=default_historical_start_date,
                        label="Select the historical period by starting date",
                        info="The workbench automatically uses the following 30 observed market days.",
                    )
                    historical_range = gr.Markdown(
                        historical_period_range_markdown(default_historical_start_date)
                    )
                    historical_button = gr.Button(
                        "Step 1 · Visualize historical scenario",
                        variant="primary",
                    )
                with gr.Column(visible=False) as uploaded_group:
                    factor_csv = gr.Textbox(
                        label="Numerical scenario CSV",
                        lines=10,
                        value=(
                            "factor,start,end,confidence\n"
                            "SPX,1294.0,1311.0,medium\n"
                            "DXY,90.3,87.2,high\n"
                            "CRUDE_OIL,62.1,70.5,medium\n"
                            "GOLD,548.0,622.5,medium\n"
                        ),
                    )
                    normalize_button = gr.Button(
                        "Step 1 · Visualize uploaded scenario",
                        variant="primary",
                    )
                with gr.Accordion("Technical paths", open=False):
                    reference_cards_jsonl = gr.Textbox(
                        label="Reference cards JSONL",
                        value=str(DEFAULT_CARDS_JSONL),
                    )
                    output_dir = gr.Textbox(
                        label="Packet output directory",
                        value=DEFAULT_OUTPUT_DIR,
                    )
            with gr.Column(scale=2):
                status = gr.Markdown(
                    status_cards_markdown(None, validation_status="waiting")
                )
                sidecar_state = gr.State("")
                factor_frame = gr.Dataframe(
                    headers=FACTOR_ROW_COLUMNS,
                    label="Numerical scenario",
                    interactive=False,
                )
                factor_plot = gr.Plot(label="30-day movement")
                generate_button = gr.Button(
                    "Step 2 · Generate + verify narrative packet"
                )
                generation_status = gr.Markdown(
                    "## Generation Status\n\n- Narrative check: `waiting`"
                )
                packet_download = gr.File(
                    label="Download narrative packet (JSON)",
                    interactive=False,
                )
                with gr.Accordion("Generated narratives", open=True):
                    narrative_review = gr.Markdown(
                        "## Generated Narratives\n\n"
                        "Generate a packet to view the narratives."
                    )
                with gr.Accordion("Technical details", open=False):
                    warnings_json = gr.Code(language="json", label="Warnings")
                    sidecar_json = gr.Code(
                        language="json", label="Scenario record (raw JSON)"
                    )

        source_mode.change(
            fn=mode_change_outputs,
            inputs=[source_mode],
            outputs=[
                historical_group,
                uploaded_group,
                status,
                factor_frame,
                factor_plot,
                sidecar_state,
                generation_status,
                narrative_review,
                packet_download,
                warnings_json,
                sidecar_json,
            ],
            show_progress="hidden",
        )
        historical_start_date.change(
            fn=historical_period_range_markdown,
            inputs=[historical_start_date],
            outputs=[historical_range],
            show_progress="hidden",
        )
        historical_button.click(
            fn=lambda calendar_start_date, cards_jsonl: normalize_historical_start_date_safe_for_app(
                calendar_start_date,
                cards_jsonl=cards_jsonl,
                support_arrays_path=DEFAULT_SUPPORT_ARRAYS,
            ),
            inputs=[historical_start_date, reference_cards_jsonl],
            outputs=[
                status,
                factor_frame,
                warnings_json,
                sidecar_json,
                sidecar_state,
                historical_range,
            ],
            show_progress="full",
        ).then(
            fn=lambda calendar_start_date, cards_jsonl: _historical_plot_safe(
                calendar_start_date,
                cards_jsonl=cards_jsonl,
            ),
            inputs=[historical_start_date, reference_cards_jsonl],
            outputs=[factor_plot],
            show_progress="hidden",
        )
        normalize_button.click(
            fn=normalize_uploaded_scenario_safe_for_app,
            inputs=[factor_csv],
            outputs=[
                status,
                factor_frame,
                warnings_json,
                sidecar_json,
                sidecar_state,
            ],
            show_progress="full",
        ).then(
            fn=_uploaded_plot_safe,
            inputs=[factor_csv],
            outputs=[factor_plot],
            show_progress="hidden",
        )
        generate_button.click(
            fn=_generation_pending_outputs,
            inputs=None,
            outputs=[generation_status, narrative_review, packet_download],
            show_progress="hidden",
        ).then(
            fn=generate_narrative_packet_ui,
            inputs=[sidecar_state, reference_cards_jsonl, output_dir],
            outputs=[generation_status, narrative_review, packet_download],
            show_progress="full",
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=int(args.server_port),
        share=bool(args.share),
        show_error=True,
    )


if __name__ == "__main__":
    main()
