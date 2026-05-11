#!/usr/bin/env python
"""Local Gradio demo for narrative-conditioned scenario generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

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
    DEFAULT_CHECKPOINT as DEFAULT_PREFIX_CHECKPOINT,
    DEFAULT_BRIDGE_REPORT as DEFAULT_PREFIX_BRIDGE_REPORT,
    run_prefix_latent_story_smoke,
    window_metadata_by_bridge_local_index,
)
from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    _spec_names,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    selected_bridge_window_indices,
)
from experiments.backfill.block_ar.nl_prefix_latent_condition_only_report import (  # noqa: E402
    run_condition_only_report,
)
from experiments.backfill.block_ar.nl_prefix_latent_live_casebook import (  # noqa: E402
    default_casebook_stories,
)
from experiments.backfill.block_ar.nl_prefix_latent_hard_case_decomposition import (  # noqa: E402
    decompose_report,
    production_decision,
)
from experiments.backfill.block_ar.nl_prefix_latent_temporal_grounding_testflight import (  # noqa: E402
    PROMPT_VERSION as CONDITION_ONLY_PROMPT_VERSION,
    condition_query_text_from_grounding,
    ground_condition_only_story_with_openai,
    split_story_for_conditioning,
    validate_condition_only_grounding_result,
)
from experiments.backfill.block_ar.nl_prefix_latent_run_record import (  # noqa: E402
    write_prefix_run_record,
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
DEFAULT_PREFIX_ROLLOUT_TEMPERATURE = 0.5
DEFAULT_USER_START_STATE_JSON = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo/prefix_latent_live_smoke/"
    "user_start_state_18.json"
)
DEFAULT_PREFIX_START_RELIABILITY_MANIFEST = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_reliability_gate_865d_full_s192_symmetric/"
    "start_reliability_gate.json"
)
DEFAULT_BOSS_DEMO_PACK_JSON = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_boss_demo_pack_829a_live_casebook/"
    "boss_demo_pack.json"
)
DEFAULT_AUTH_USER_ENV = "NARRATIVE_DEMO_AUTH_USER"
DEFAULT_AUTH_PASSWORD_ENV = "NARRATIVE_DEMO_AUTH_PASSWORD"
DEMO_TABLE_CLASS = "demo-scroll-table"
APP_CSS = """
.gradio-container {
  width: 100% !important;
  max-width: 1180px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  overflow-x: hidden;
}
.gradio-container,
.gradio-container * {
  box-sizing: border-box;
  min-width: 0;
}
.gradio-container code,
.gradio-container pre,
.gradio-container .prose,
.gradio-container .markdown {
  overflow-wrap: anywhere;
  word-break: break-word;
}
.gradio-container p,
.gradio-container li {
  max-width: 100%;
  white-space: normal;
}
.gradio-container p code,
.gradio-container li code {
  display: inline;
  white-space: normal !important;
}
.demo-scroll-table {
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100%;
  overflow-x: auto !important;
}
.demo-scroll-table > div {
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
}
.demo-scroll-table table {
  width: max-content;
  max-width: none;
}
.demo-shell {
  width: 100%;
  max-width: 1120px;
  margin-left: auto;
  margin-right: auto;
}
.demo-hero {
  margin-bottom: 0.35rem;
  text-align: center;
}
.demo-status-strip {
  padding: 0.55rem 0.75rem;
  border-left: 4px solid #2563eb;
  background: #eff6ff;
  border-radius: 6px;
  font-size: 0.95rem;
}
.demo-status-strip p {
  margin: 0;
}
.demo-responsive-row {
  gap: 0.75rem;
  align-items: stretch;
}
@media (max-width: 900px) {
  .demo-responsive-row {
    flex-direction: column !important;
  }
  .demo-responsive-row > div {
    width: 100% !important;
    min-width: 0 !important;
  }
}
@media (max-width: 640px) {
  html,
  body {
    max-width: 100vw;
    overflow-x: hidden;
  }
  .gradio-container {
    max-width: 100vw !important;
    padding-left: 14px !important;
    padding-right: 14px !important;
  }
  .gradio-container h1 {
    font-size: 1.55rem !important;
    line-height: 1.18 !important;
  }
  .gradio-container h2 {
    font-size: 1.25rem !important;
    line-height: 1.2 !important;
  }
  .gradio-container textarea,
  .gradio-container input,
  .gradio-container label {
    font-size: 0.92rem !important;
  }
  .gradio-container .wrap,
  .gradio-container .contain {
    min-width: 0 !important;
  }
  .gradio-container p,
  .gradio-container li,
  .gradio-container .prose p,
  .gradio-container .markdown p,
  .gradio-container .prose li,
  .gradio-container .markdown li {
    width: 100% !important;
    max-width: calc(100vw - 64px) !important;
  }
}
"""
CACHED_PREFIX_CASEBOOK_CONFIG = [
    (
        "commodity_inflation_pressure",
        "Commodity inflation pressure / start 18",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_822a_commodity/condition_only_report.json",
        18,
    ),
    (
        "commodity_inflation_pressure",
        "Commodity inflation pressure / balanced start 40",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_822a_commodity/condition_only_report.json",
        40,
    ),
    (
        "commodity_inflation_pressure",
        "Commodity inflation pressure / rates start 178",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_822a_commodity/condition_only_report.json",
        178,
    ),
    (
        "dollar_liquidity_squeeze",
        "Dollar liquidity squeeze / defensive start 22",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823a_dollar/condition_only_report.json",
        22,
    ),
    (
        "dollar_liquidity_squeeze",
        "Dollar liquidity squeeze / balanced start 77",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823a_dollar/condition_only_report.json",
        77,
    ),
    (
        "dollar_liquidity_squeeze",
        "Dollar liquidity squeeze / memory-nearest start 0",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823a_dollar/condition_only_report.json",
        0,
    ),
    (
        "safe_haven_gold_bid",
        "Safe-haven gold bid / start 18",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823b_safe_haven/condition_only_report.json",
        18,
    ),
    (
        "safe_haven_gold_bid",
        "Safe-haven gold bid / balanced start 77",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823b_safe_haven/condition_only_report.json",
        77,
    ),
    (
        "safe_haven_gold_bid",
        "Safe-haven gold bid / rates start 178",
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823b_safe_haven/condition_only_report.json",
        178,
    ),
]


def resolve_launch_auth(
    *,
    env: dict[str, str] | None = None,
    user_env: str = DEFAULT_AUTH_USER_ENV,
    password_env: str = DEFAULT_AUTH_PASSWORD_ENV,
    require_auth: bool = False,
) -> tuple[str, str] | None:
    env_map = os.environ if env is None else env
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


IMPLICATION_COLUMNS = [
    "Market",
    "Direction",
    "Magnitude",
    "Confidence",
    "Inferred",
    "Evidence",
]
WARNING_COLUMNS = ["Severity", "Code", "Message"]
PREFIX_CONDITION_COLUMNS = [
    "Market",
    "Direction",
    "Magnitude",
    "Confidence",
    "Horizon",
    "Evidence",
]
PREFIX_WARNING_COMPONENT_COLUMNS = ["Component", "Status", "Metric", "Value"]
PREFIX_SHIFT_FACTOR_COLUMNS = [
    "Start Mode",
    "Factor",
    "Terminal Abs Z",
    "Signed Terminal Z",
    "Path Abs Z",
]
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
SCENARIO_COLUMNS = [
    "Starting Level",
    "Start Value",
    "Market",
    "Mean Terminal Delta",
    "P10",
    "P90",
]
VALIDATION_GATE_COLUMNS = [
    "Variant",
    "Role",
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
    "Memory Support",
    "Selection",
    "Start Split",
]
PREFIX_SELECTED_START_COLUMNS = [
    "Starting Level",
    "Index",
    "Reliability",
    "Compatibility",
    "Distance",
    "Source Split",
]
PREFIX_START_CANDIDATE_COLUMNS = [
    "Rank",
    "Window",
    "Bridge Index",
    "Source",
    "History End",
    "Split",
    "Weight",
    "Memory Support",
    "Start Distance",
    "Alignment",
    "Score",
]
PREFIX_USER_START_COLUMNS = [
    "Label",
    "Source",
    "Format",
    "Dimension",
    "Nearest Train",
    "Start Distance",
    "Max Abs Z",
]
PREFIX_START_PREVIEW_COLUMNS = ["Field", "Value"]
BOSS_DEMO_CASEBOOK_COLUMNS = [
    "Case",
    "Start",
    "Status",
    "Condition",
    "Warnings",
    "Support",
    "Summary",
]
START_PREVIEW_FIELDS = [
    ("SPX", "factor:spx"),
    ("VIX", "factor:vix"),
    ("BBB OAS", "factor:bbb_oas"),
    ("AAA OAS", "factor:aaa_oas"),
    ("US 2Y", "factor:us2y"),
    ("US 10Y", "factor:us10y"),
    ("USD/JPY", "factor:usdjpy"),
    ("DXY", "factor:dxy"),
    ("Gold", "factor:gold"),
    ("Crude oil", "factor:crude_oil"),
    ("IV ATM 3M", "iv:07"),
    ("IV ATM 1Y", "iv:17"),
]
SCENARIO_MARKET_TO_START_SPEC = {
    "SPX": "factor:spx",
    "VIX": "factor:vix",
    "BBB_OAS": "factor:bbb_oas",
    "AAA_OAS": "factor:aaa_oas",
    "US2Y": "factor:us2y",
    "US10Y": "factor:us10y",
    "USDJPY": "factor:usdjpy",
    "DXY": "factor:dxy",
    "GOLD": "factor:gold",
    "CRUDE_OIL": "factor:crude_oil",
    "IV_ATM_3M": "iv:07",
    "IV_ATM_1Y": "iv:17",
}
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


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):+.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _short(text: Any, limit: int = 160) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= int(limit):
        return compact
    return compact[: int(limit) - 3].rstrip() + "..."


def _slug(text: str) -> str:
    keep: list[str] = []
    for char in str(text).lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_") or "case"


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def cached_prefix_casebook_rows() -> list[dict[str, Any]]:
    """Return cached condition-only demo rows for the prefix-latent UI."""

    story_by_name = {
        str(item["name"]): str(item["story"]) for item in default_casebook_stories()
    }
    rows: list[dict[str, Any]] = []
    for (
        case_name,
        label,
        condition_report,
        start_index,
    ) in CACHED_PREFIX_CASEBOOK_CONFIG:
        rows.append(
            {
                "label": str(label),
                "value": f"{case_name}:{int(start_index)}",
                "case_name": str(case_name),
                "story": story_by_name.get(str(case_name), DEFAULT_STORY),
                "condition_report": str(condition_report),
                "start_index": int(start_index),
            }
        )
    return rows


def cached_prefix_casebook_choices() -> list[tuple[str, str]]:
    return [("Typed story / current controls", "")] + [
        (str(row["label"]), str(row["value"])) for row in cached_prefix_casebook_rows()
    ]


def cached_prefix_casebook_update(choice: str | None) -> tuple[
    str,
    bool,
    int,
    bool,
    bool,
    str,
    str,
]:
    """Populate story/start/report controls from a saved demo setup."""

    value = str(choice or "")
    if not value:
        return (
            DEFAULT_STORY,
            False,
            22,
            True,
            False,
            "",
            "## Saved Demo Setup\n\n- Selection: `typed story / current controls`",
        )
    for row in cached_prefix_casebook_rows():
        if str(row["value"]) != value:
            continue
        report_path = str(row["condition_report"])
        exists_text = "available" if Path(report_path).exists() else "missing"
        return (
            str(row["story"]),
            True,
            int(row["start_index"]),
            False,
            False,
            report_path,
            "\n".join(
                [
                    "## Saved Demo Setup",
                    "",
                    f"- Selection: `{row['label']}`",
                    f"- Narrative family: `{row['case_name']}`",
                    f"- Fixed start index: `{row['start_index']}`",
                    f"- Cached condition report: `{exists_text}`",
                    "- OpenAI calls: `none for this cached run`",
                ]
            ),
        )
    return (
        DEFAULT_STORY,
        False,
        22,
        True,
        False,
        "",
        f"## Saved Demo Setup\n\n- Selection: `unknown ({value})`",
    )


def _operational_variant_row(report: dict[str, Any]) -> dict[str, Any]:
    for row in _as_list(report.get("variant_rows")):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return row
    for row in _as_list(report.get("variant_rows")):
        if isinstance(row, dict) and str(row.get("variant", "")) != "original":
            return row
    return {}


def _operational_score_metrics(report: dict[str, Any]) -> dict[str, Any]:
    generation = _as_dict(report.get("generation"))
    operational = _operational_variant_row(report)
    variant = str(operational.get("variant", ""))
    start_index = operational.get("start_window_index")
    for row in _as_list(generation.get("window_scores")):
        if not isinstance(row, dict):
            continue
        if variant and str(row.get("variant", "")) != variant:
            continue
        if start_index is not None and row.get("start_window_index") is not None:
            try:
                if int(row.get("start_window_index")) != int(start_index):
                    continue
            except (TypeError, ValueError):
                continue
        methods = _as_dict(row.get("methods"))
        model = _as_dict(methods.get("text_memory_plus_start_prefix_decoder"))
        if model:
            return model
    return {}


def prefix_trust_interpretation(report: dict[str, Any]) -> str:
    gate = _as_dict(report.get("validation_gate"))
    status = str(
        gate.get("selected_start_status", gate.get("operational_status", "unknown"))
    )
    metrics = _operational_score_metrics(report)
    crps_improvement = metrics.get("ensemble_crps_z_improvement_vs_persistence")
    try:
        crps_value = float(crps_improvement)
    except (TypeError, ValueError):
        crps_value = None
    if status == "pass":
        return "supported calibrated scenario"
    if status == "warning" and crps_value is not None and crps_value > 0.0:
        return (
            "usable with support/shift caveats; scenario CRPS improved vs persistence"
        )
    if status == "warning":
        return "usable only with caveats; inspect support and rollout-shift warnings"
    if status == "fail":
        return "do not use without changing the narrative or starting state"
    return "status unavailable"


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


def _prefix_grounding(report: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(_as_dict(report.get("cached_query")).get("grounding"))


def prefix_condition_implications_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _prefix_grounding(report)
    for item in _as_list(grounding.get("market_implications")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Market": str(item.get("market", "")),
                "Direction": str(item.get("direction", "")),
                "Magnitude": str(item.get("magnitude", "")),
                "Confidence": str(item.get("confidence", "")),
                "Horizon": str(item.get("horizon", "")),
                "Evidence": "; ".join(str(x) for x in _as_list(item.get("evidence"))),
            }
        )
    return _frame(rows, PREFIX_CONDITION_COLUMNS)


def prefix_condition_warnings_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grounding = _prefix_grounding(report)
    for item in _as_list(grounding.get("non_conditioning_forward_language")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Severity": str(item.get("severity", "warning")),
                "Code": "non_conditioning_forward_language",
                "Message": (
                    f"{item.get('phrase', '')} - "
                    f"{item.get('reason', item.get('handling', 'warning only'))}"
                ).strip(" -"),
            }
        )
    for item in _as_list(grounding.get("grounding_warnings")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Severity": str(item.get("severity", "warning")),
                "Code": str(item.get("code", "grounding_warning")),
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


def _selected_start_values_by_spec(report: dict[str, Any]) -> dict[str, Any]:
    selected = _as_dict(report.get("selected_start_state"))
    values = _as_dict(selected.get("values_by_name"))
    if values:
        return values
    return _as_dict(report.get("selected_start_values_by_name"))


def scenario_table(report: dict[str, Any]) -> pd.DataFrame:
    generation = _as_dict(report.get("generation"))
    selected_start = _operational_variant_row(report)
    start_label = str(selected_start.get("start_window_id") or "n/a")
    start_values = _selected_start_values_by_spec(report)
    rows: list[dict[str, Any]] = []
    for item in _as_list(generation.get("terminal_delta_summary")):
        if not isinstance(item, dict):
            continue
        market = str(item.get("market", ""))
        start_spec = SCENARIO_MARKET_TO_START_SPEC.get(market.upper(), "")
        rows.append(
            {
                "Starting Level": start_label,
                "Start Value": _fmt_float(start_values.get(start_spec)),
                "Market": market,
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


def load_boss_demo_pack(
    path: str | Path = DEFAULT_BOSS_DEMO_PACK_JSON,
) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def boss_demo_status_strip(report: dict[str, Any]) -> str:
    if not report:
        return (
            "**Validation evidence:** not available. Optional casebook details "
            "below are empty until the boss demo pack is generated."
        )
    snapshot = _as_dict(report.get("validation_snapshot"))
    live_snapshot = _as_dict(report.get("live_casebook_snapshot"))
    run_count = snapshot.get("run_count", 0)
    case_count = live_snapshot.get("case_count", 0)
    return (
        "**Validation evidence:** "
        f"offline CRPS `{snapshot.get('improved_crps_rows', 0)}/{run_count}`, "
        f"energy `{snapshot.get('improved_energy_rows', 0)}/{run_count}`, "
        f"live API casebook `{live_snapshot.get('pass_count', 0)}/{case_count}` pass. "
        "These are demo-readiness checks; full details below are optional."
    )


def boss_demo_pack_markdown(report: dict[str, Any]) -> str:
    if not report:
        return "\n".join(
            [
                "## Demo readiness evidence",
                "",
                "- Status: `not available`",
                "- Generate the boss demo pack to populate this section.",
            ]
        )
    snapshot = _as_dict(report.get("validation_snapshot"))
    live_snapshot = _as_dict(report.get("live_casebook_snapshot"))
    artifact_paths = _as_dict(report.get("artifact_paths"))
    live_models = ", ".join(_as_list(live_snapshot.get("grounding_models"))) or "n/a"
    embedding_models = (
        ", ".join(_as_list(live_snapshot.get("embedding_models"))) or "n/a"
    )
    evidence_path = str(artifact_paths.get("summary_markdown", ""))
    evidence_label = Path(evidence_path).name if evidence_path else "n/a"
    return "\n".join(
        [
            "## Demo readiness evidence",
            "",
            f"- Evidence pack: `{evidence_label}`",
            f"- Offline validation: `{snapshot.get('run_count', 0)}` runs.",
            f"- CRPS improved `{snapshot.get('improved_crps_rows', 0)}/{snapshot.get('run_count', 0)}`.",
            f"- Energy improved `{snapshot.get('improved_energy_rows', 0)}/{snapshot.get('run_count', 0)}`.",
            f"- Offline mean CRPS improvement: `{_fmt_pct(snapshot.get('mean_crps_improvement_vs_persistence'))}`.",
            f"- Live API casebook: `{live_snapshot.get('pass_count', 0)}/{live_snapshot.get('case_count', 0)}` pass.",
            f"- OpenAI tokens `{live_snapshot.get('total_openai_tokens', 0)}`.",
            f"- Min support candidates `{live_snapshot.get('min_support_candidate_count', 0)}`.",
            f"- Grounding model: `{live_models}`.",
            f"- Embedding model: `{embedding_models}`.",
            "- Contract:",
            "- Current/recent implications are conditioning inputs.",
            "- Forward-risk language is warning-only.",
        ]
    )


def boss_demo_live_casebook_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    live_snapshot = _as_dict(report.get("live_casebook_snapshot"))
    for item in _as_list(live_snapshot.get("case_rows")):
        if not isinstance(item, dict):
            continue
        summary_path = str(item.get("summary_path", ""))
        rows.append(
            {
                "Case": str(item.get("case_name", "")),
                "Start": str(item.get("expected_start_index", "")),
                "Status": str(item.get("overall_status", "")),
                "Condition": str(item.get("condition_only_validation_status", "")),
                "Warnings": str(item.get("forward_warning_count", "")),
                "Support": str(item.get("support_candidate_count", "")),
                "Summary": Path(summary_path).name if summary_path else "",
            }
        )
    return _frame(rows, BOSS_DEMO_CASEBOOK_COLUMNS)


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
                "Role": str(item.get("case_role", "")),
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
    return _prefix_variant_table_for_role(report, role=None)


def prefix_selected_start_table(report: dict[str, Any]) -> pd.DataFrame:
    row = _operational_variant_row(report)
    reliability = _as_dict(report.get("start_reliability_gate"))
    rows: list[dict[str, Any]] = []
    if row:
        rows.append(
            {
                "Starting Level": str(row.get("start_window_id", "")),
                "Index": str(row.get("start_window_index", "")),
                "Reliability": str(reliability.get("product_status", "not checked")),
                "Compatibility": _fmt_float(row.get("memory_support_cosine")),
                "Distance": _fmt_float(row.get("start_distance_z")),
                "Source Split": str(row.get("start_manifest_split", "")),
            }
        )
    return _frame(rows, PREFIX_SELECTED_START_COLUMNS)


def prefix_diagnostic_start_table(report: dict[str, Any]) -> pd.DataFrame:
    return _prefix_variant_table_for_role(report, role="diagnostic")


def _prefix_variant_table_for_role(
    report: dict[str, Any],
    *,
    role: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(report.get("variant_rows")):
        if not isinstance(item, dict):
            continue
        is_operational = bool(
            item.get(
                "is_operational",
                str(item.get("variant", "")) != "original",
            )
        )
        if role == "operational" and not is_operational:
            continue
        if role == "diagnostic" and is_operational:
            continue
        rows.append(
            {
                "Variant": str(item.get("variant", "")),
                "Query Window": str(item.get("query_window_id", "")),
                "Start Window": str(item.get("start_window_id", "")),
                "Start Distance": _fmt_float(item.get("start_distance_z")),
                "Memory Support": _fmt_float(item.get("memory_support_cosine")),
                "Selection": str(item.get("start_selection_method", "")),
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
                "Role": str(item.get("case_role", "")),
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


def prefix_start_candidates_table(report: dict[str, Any]) -> pd.DataFrame:
    query = _as_dict(report.get("cached_query"))
    memory_prior = _as_dict(query.get("memory_prior"))
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(
        _as_list(memory_prior.get("candidate_details")), start=1
    ):
        if not isinstance(item, dict):
            continue
        checked = item.get("recent_prefix_checked")
        mismatches = item.get("recent_prefix_mismatches")
        alignment_score = _fmt_float(item.get("recent_prefix_alignment_score"))
        alignment = (
            alignment_score
            if checked is None
            else f"{alignment_score} ({mismatches}/{checked} mismatches)"
        )
        rows.append(
            {
                "Rank": int(item.get("rank", rank)),
                "Window": str(item.get("window_id") or item.get("window_index", "")),
                "Bridge Index": str(
                    item.get("bridge_local_index", item.get("window_index", ""))
                ),
                "Source": str(item.get("source_index", "")),
                "History End": str(item.get("history_end_date", "")),
                "Split": str(item.get("manifest_split", "")),
                "Weight": _fmt_float(item.get("weight")),
                "Memory Support": _fmt_float(item.get("memory_support_cosine")),
                "Start Distance": _fmt_float(item.get("start_distance_z")),
                "Alignment": alignment,
                "Score": _fmt_float(item.get("combined_score")),
            }
        )
    return _frame(rows, PREFIX_START_CANDIDATE_COLUMNS)


def prefix_user_start_table(report: dict[str, Any]) -> pd.DataFrame:
    user_start = _as_dict(report.get("user_start_state"))
    if not user_start:
        return _frame([], PREFIX_USER_START_COLUMNS)
    user_row = None
    for item in _as_list(report.get("variant_rows")):
        if isinstance(item, dict) and str(item.get("variant")) == "user_start_state":
            user_row = item
            break
    nearest = (
        ""
        if user_row is None
        else str(user_row.get("nearest_train_start_window_index", ""))
    )
    distance = "" if user_row is None else _fmt_float(user_row.get("start_distance_z"))
    max_abs_z = (
        "" if user_row is None else _fmt_float(user_row.get("max_abs_user_start_z"))
    )
    return _frame(
        [
            {
                "Label": str(user_start.get("label", "")),
                "Source": str(user_start.get("source_path", "")),
                "Format": str(user_start.get("source_format", "")),
                "Dimension": str(user_start.get("dimension", "")),
                "Nearest Train": nearest,
                "Start Distance": distance,
                "Max Abs Z": max_abs_z,
            }
        ],
        PREFIX_USER_START_COLUMNS,
    )


def preview_start_state_json(path: str | None) -> tuple[str, pd.DataFrame]:
    candidate = str(path or "").strip()
    if not candidate:
        return (
            "## Start-State JSON Preview\n\n- Status: `waiting`\n- Enter a JSON path.",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    try:
        payload = json.loads(Path(candidate).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("start-state JSON must contain an object")
        label = str(payload.get("label", ""))
        coordinate = str(payload.get("coordinate", "raw_state"))
        rows = [
            {"Field": "Path", "Value": candidate},
            {"Field": "Label", "Value": label},
            {"Field": "Coordinate", "Value": coordinate},
        ]
        values = payload.get("values_by_name")
        vector = payload.get("state_vector")
        if isinstance(values, dict):
            rows.append({"Field": "Format", "Value": "values_by_name"})
            rows.append({"Field": "Field count", "Value": str(len(values))})
            for label_name, key in START_PREVIEW_FIELDS:
                if key in values:
                    rows.append({"Field": label_name, "Value": _fmt_float(values[key])})
            missing_preview = [
                label_name
                for label_name, key in START_PREVIEW_FIELDS
                if key not in values
            ]
            if missing_preview:
                rows.append(
                    {
                        "Field": "Missing preview fields",
                        "Value": ", ".join(missing_preview),
                    }
                )
        elif isinstance(vector, list):
            rows.append({"Field": "Format", "Value": "state_vector"})
            rows.append({"Field": "Vector length", "Value": str(len(vector))})
            if vector:
                rows.append({"Field": "First value", "Value": _fmt_float(vector[0])})
        else:
            raise ValueError("JSON must contain values_by_name or state_vector")
    except Exception as error:
        return (
            "## Start-State JSON Preview\n\n"
            f"- Status: `error`\n"
            f"- Error type: `{type(error).__name__}`\n"
            f"- Message: `{str(error)}`",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    return (
        "## Start-State JSON Preview\n\n"
        f"- Status: `ok`\n"
        f"- Format: `{rows[3]['Value'] if len(rows) > 3 else 'unknown'}`",
        _frame(rows, PREFIX_START_PREVIEW_COLUMNS),
    )


def build_start_state_payload(
    *,
    label: str,
    spec_names: list[str],
    raw_state: Any,
) -> dict[str, Any]:
    values = [float(value) for value in list(raw_state)]
    if len(values) != len(spec_names):
        raise ValueError("raw_state length must match spec_names")
    return {
        "label": str(label),
        "coordinate": "raw_state",
        "values_by_name": {
            str(name): float(value)
            for name, value in zip(spec_names, values, strict=True)
        },
    }


def load_joint39_start_bank_for_app() -> dict[str, Any]:
    bridge_report = json.loads(Path(DEFAULT_PREFIX_BRIDGE_REPORT).read_text())
    selected_windows = selected_bridge_window_indices(bridge_report)
    args = SimpleNamespace(
        state_scope="joint38",
        test_start=4511,
        val_size=441,
        max_windows=441,
        eval_split="val",
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
    _model, payload = load_model(DEFAULT_PREFIX_CHECKPOINT, torch.device("cpu"))
    (
        _all_history_level,
        _all_history_norm,
        _all_center,
        _all_scale,
        _all_drift,
        all_history_raw,
        specs,
        _block,
    ) = build_val_block(args, payload)
    return {
        "history_raw": all_history_raw[selected_windows],
        "spec_names": _spec_names(specs),
        "metadata": window_metadata_by_bridge_local_index(bridge_report),
    }


def export_historical_start_json_for_app(
    candidate_choice: str | int | float | None,
    explicit_start_window_index: float | int | None = None,
    *,
    output_dir: str | Path = DEFAULT_PREFIX_APP_OUTPUT_DIR,
    bank_loader: Callable[[], dict[str, Any]] = load_joint39_start_bank_for_app,
) -> tuple[str, str, pd.DataFrame]:
    start_index = historical_start_candidate_to_index(candidate_choice)
    if start_index is None and explicit_start_window_index is not None:
        start_index = historical_start_candidate_to_index(explicit_start_window_index)
    if start_index is None:
        return (
            "## Export Start JSON\n\n- Status: `error`\n- Message: `Select a historical candidate or enter an index first.`",
            "",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    try:
        bank = bank_loader()
        history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
        spec_names = list(bank["spec_names"])
        metadata = _as_dict(bank.get("metadata")).get(int(start_index), {})
        if start_index < 0 or start_index >= len(history_raw):
            raise IndexError(f"start index {start_index} outside {len(history_raw)}")
        window_id = str(
            _as_dict(metadata).get("window_id") or f"joint39_start_{start_index:04d}"
        )
        label = f"user_template_from_{window_id}"
        payload = build_start_state_payload(
            label=label,
            spec_names=spec_names,
            raw_state=history_raw[int(start_index), -1, :],
        )
        output_path = Path(output_dir) / f"user_start_template_{start_index:04d}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preview_status, preview = preview_start_state_json(str(output_path))
    except Exception as error:
        return (
            "## Export Start JSON\n\n"
            f"- Status: `error`\n"
            f"- Error type: `{type(error).__name__}`\n"
            f"- Message: `{str(error)}`",
            "",
            _frame([], PREFIX_START_PREVIEW_COLUMNS),
        )
    status = (
        "## Export Start JSON\n\n"
        f"- Status: `ok`\n"
        f"- Start index: `{start_index}`\n"
        f"- Path: `{output_path}`\n\n"
        f"{preview_status}"
    )
    return status, str(output_path), preview


def historical_start_candidate_choices(report: dict[str, Any]) -> list[tuple[str, str]]:
    query = _as_dict(report.get("cached_query"))
    memory_prior = _as_dict(query.get("memory_prior"))
    choices: list[tuple[str, str]] = []
    for item in _as_list(memory_prior.get("candidate_details")):
        if not isinstance(item, dict):
            continue
        bridge_index = item.get("bridge_local_index", item.get("window_index"))
        if bridge_index is None:
            continue
        window = str(item.get("window_id") or f"window_{bridge_index}")
        label = (
            f"{window} | idx {bridge_index} | "
            f"w {_fmt_float(item.get('weight'))} | "
            f"start {_fmt_float(item.get('start_distance_z'))}z"
        )
        choices.append((label, str(int(bridge_index))))
    return choices


def historical_start_candidate_update(report: dict[str, Any]) -> Any:
    import gradio as gr

    choices = historical_start_candidate_choices(report)
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def historical_start_candidate_to_index(choice: str | int | float | None) -> int | None:
    if choice in (None, ""):
        return None
    try:
        return int(float(choice))
    except (TypeError, ValueError):
        return None


def prefix_warning_component_table(report: dict[str, Any]) -> pd.DataFrame:
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    decompositions = _as_list(product_gate.get("decompositions"))
    selected = {}
    for item in decompositions:
        if isinstance(item, dict) and item.get("start_mode") in {
            "balanced_memory_start",
            "implication_aligned_start",
        }:
            selected = item
            break
    if not selected and decompositions and isinstance(decompositions[0], dict):
        selected = decompositions[0]
    components = _as_dict(selected.get("components")) if selected else {}
    rows: list[dict[str, Any]] = []
    for name, item in components.items():
        if not isinstance(item, dict):
            continue
        metric = ""
        value: Any = ""
        for candidate in [
            "input_memory_cosine",
            "start_distance_z",
            "endpoint_max_abs_error",
            "terminal_mean_abs_delta_z",
            "mismatch_count",
        ]:
            if candidate in item:
                metric = candidate
                value = item[candidate]
                break
        rows.append(
            {
                "Component": str(name),
                "Status": str(item.get("status", "")),
                "Metric": metric,
                "Value": _fmt_float(
                    value, 6 if metric == "endpoint_max_abs_error" else 3
                ),
            }
        )
    return _frame(rows, PREFIX_WARNING_COMPONENT_COLUMNS)


def prefix_shift_factor_table(report: dict[str, Any]) -> pd.DataFrame:
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    rows: list[dict[str, Any]] = []
    for item in _as_list(product_gate.get("decompositions")):
        if not isinstance(item, dict):
            continue
        mode = str(item.get("start_mode", ""))
        for factor in _as_list(item.get("top_rollout_shift_factors"))[:8]:
            if not isinstance(factor, dict):
                continue
            rows.append(
                {
                    "Start Mode": mode,
                    "Factor": str(factor.get("factor", "")),
                    "Terminal Abs Z": _fmt_float(factor.get("terminal_abs_shift_z")),
                    "Signed Terminal Z": _fmt_float(
                        factor.get("signed_terminal_shift_z")
                    ),
                    "Path Abs Z": _fmt_float(factor.get("mean_path_abs_shift_z")),
                }
            )
    return _frame(rows, PREFIX_SHIFT_FACTOR_COLUMNS)


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
    metrics = _operational_score_metrics(report)
    artifacts = _as_dict(report.get("artifact_paths"))
    lines = [
        "## Prefix-Latent Run Status",
        "",
        f"- Cached query: `{query.get('window_id', 'n/a')}` / `{query.get('kind', 'n/a')}`",
        f"- Condition source: `{query.get('condition_source', 'n/a')}`",
        f"- Text memory dimension: `{query.get('text_memory_dim', 'n/a')}`",
        f"- Selected-start: `{gate.get('selected_start_status', gate.get('operational_status', 'n/a'))}`",
        f"- Diagnostic baseline: `{gate.get('diagnostic_baseline_status', 'n/a')}`",
        f"- Research overall: `{gate.get('overall_status', 'n/a')}`",
        f"- Stress: `{gate.get('stress_status', 'n/a')}`",
        f"- Endpoint max error: `{_fmt_float(gate.get('endpoint_max_abs_error'), 6)}`",
        f"- Rollout temperature: `{_fmt_float(generation.get('rollout_temperature'))}`",
        f"- Scenario CRPS vs persistence: `{_fmt_pct(metrics.get('ensemble_crps_z_improvement_vs_persistence'))}`",
        f"- Scenario energy vs persistence: `{_fmt_pct(metrics.get('energy_score_z_improvement_vs_persistence'))}`",
        f"- Operational interpretation: `{prefix_trust_interpretation(report)}`",
        f"- Generated shape: `{generation.get('generated_state_shape', 'not run')}`",
    ]
    start_reliability = _as_dict(report.get("start_reliability_gate"))
    if start_reliability:
        lines.append(
            "- Start reliability: "
            f"`{start_reliability.get('product_status', 'n/a')}` - "
            f"{start_reliability.get('decision', '')}"
        )
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    decision = _as_dict(product_gate.get("production_decision"))
    if decision:
        contributors: list[str] = []
        for item in _as_list(product_gate.get("decompositions")):
            if not isinstance(item, dict):
                continue
            for factor in _as_list(item.get("top_rollout_shift_factors"))[:3]:
                if isinstance(factor, dict):
                    contributors.append(
                        f"{factor.get('factor')} ({_fmt_float(factor.get('terminal_abs_shift_z'))}z)"
                    )
            if contributors:
                break
        lines.extend(
            [
                f"- Decision code: `{decision.get('decision', 'n/a')}`",
                f"- Decision note: {decision.get('ui_guidance', decision.get('reason', ''))}",
                f"- Main note contributors: `{', '.join(contributors) or 'n/a'}`",
            ]
        )
    lines.extend(
        [
            f"- Markdown report: `{artifacts.get('markdown', 'n/a')}`",
            f"- JSON report: `{artifacts.get('report', 'n/a')}`",
            f"- Run record: `{artifacts.get('run_record', 'n/a')}`",
        ]
    )
    return "\n".join(lines)


def prefix_latent_product_status_markdown(report: dict[str, Any]) -> str:
    generation = _as_dict(report.get("generation"))
    product_gate = _as_dict(report.get("condition_only_product_gate"))
    decision = _as_dict(product_gate.get("production_decision"))
    generated_shape = generation.get("generated_state_shape")
    if generated_shape:
        status = "Scenario generation complete."
        next_step = "Review the fan chart and scenario summary."
    else:
        status = "Scenario preparation complete."
        next_step = "Generate 30-day scenarios from the selected historical start."
    result_note = (
        decision.get("ui_guidance")
        or decision.get("reason")
        or "No blocking issue was found for this narrative and selected start."
    )
    start_reliability = _as_dict(report.get("start_reliability_gate"))
    reliability_note = ""
    if start_reliability:
        reliability_note = (
            f"- Start reliability: `{start_reliability.get('product_status', 'n/a')}` - "
            f"{start_reliability.get('decision', '')}"
        )
    lines = [
        "## Scenario Workflow Status",
        "",
        f"- Status: {status}",
        f"- Story support: `{prefix_trust_interpretation(report)}`",
        f"- Result note: {result_note}",
    ]
    if reliability_note:
        lines.append(reliability_note)
    lines.append(f"- Next step: {next_step}")
    return "\n".join(
        lines
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
    temperature: float = DEFAULT_PREFIX_ROLLOUT_TEMPERATURE,
    live_story: bool = False,
    condition_only_story: bool = False,
    cached_condition_report: bool = False,
    skip_rollout: bool = False,
) -> str:
    start_label = (
        "user-selected historical start"
        if str(start_mode) == "explicit_start_window"
        else "user-supplied start state"
        if str(start_mode) == "user_start_state"
        else "start selection"
    )
    if bool(cached_condition_report):
        condition_step = (
            f"cached grounding-sidecar report, {start_label}, support mixture, "
            "prefix preparation"
        )
    elif bool(condition_only_story):
        condition_step = (
            "OpenAI grounding sidecar, text embedding, "
            f"{start_label}, support mixture, prefix preparation, warning check"
        )
    elif bool(live_story):
        condition_step = (
            f"OpenAI grounding and embedding, {start_label}, support mixture, "
            "prefix preparation"
        )
    else:
        condition_step = f"cached text memory, {start_label}, support mixture, prefix preparation"
    if bool(skip_rollout):
        condition_step = f"{condition_step}; scenario rollout skipped"
        generator_note = "not run during validation"
    else:
        condition_step = f"{condition_step}, 30-day scenario generation"
        generator_note = f"{int(samples)}"
    return "\n".join(
        [
            "## Scenario Workflow Status",
            "",
            f"- Run started: `{_elapsed_text(start_time)} ago`",
            f"- Current step: `{condition_step}`",
            f"- Scenario samples: `{generator_note}`",
            "- Outputs will fill in automatically when the run completes.",
        ]
    )


def _completed_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return status_markdown(report) + f"\n- Completed in: `{_elapsed_text(start_time)}`"


def _completed_prefix_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return (
        prefix_latent_product_status_markdown(report)
        + f"\n- Completed in: `{_elapsed_text(start_time)}`"
    )


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
    pd.DataFrame,
    go.Figure,
    str,
    dict[str, Any],
    Any,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
]:
    return (
        "Prefix-latent run in progress. Results will appear here when complete.",
        status,
        _frame([], PREFIX_SELECTED_START_COLUMNS),
        _frame([], PREFIX_VARIANT_COLUMNS),
        _frame([], VALIDATION_GATE_COLUMNS),
        _frame([], SCENARIO_COLUMNS),
        fan_chart_figure({}, fan_market, "ALL"),
        "{}",
        {},
        analogue_scope_update({}),
        _frame([], PREFIX_CONDITION_COLUMNS),
        _frame([], WARNING_COLUMNS),
        _frame([], PREFIX_WARNING_COMPONENT_COLUMNS),
        _frame([], PREFIX_SHIFT_FACTOR_COLUMNS),
        _frame([], PREFIX_START_CANDIDATE_COLUMNS),
        _frame([], PREFIX_USER_START_COLUMNS),
        historical_start_candidate_update({}),
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_condition_only_case_for_app(
    *,
    story: str,
    output_dir: str | Path,
    model: str = "gpt-5.4-mini",
    dotenv: str | Path = ".env",
    max_output_tokens: int = 1800,
    grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
) -> dict[str, Any]:
    grounding, metadata = grounder(
        str(story or DEFAULT_STORY),
        model=str(model),
        dotenv_path=dotenv,
        max_output_tokens=int(max_output_tokens),
    )
    validation = validate_condition_only_grounding_result(grounding)
    query_text = condition_query_text_from_grounding(
        str(story or DEFAULT_STORY), grounding
    )
    case_dir = Path(output_dir)
    case_payload = {
        "case_name": "gradio_live_condition_only_story",
        "story": str(story or DEFAULT_STORY),
        "story_split": split_story_for_conditioning(str(story or DEFAULT_STORY)),
        "condition_only_grounding": grounding.model_dump(),
        "condition_only_validation": validation,
        "candidate_query_text": query_text,
        "metadata": {
            **_as_dict(metadata),
            "prompt_version": CONDITION_ONLY_PROMPT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        "artifact_paths": {
            "case_json": str(case_dir / "condition_only_grounding_case.json"),
            "query_text": str(case_dir / "condition_only_query_text.txt"),
        },
    }
    _write_json(case_dir / "condition_only_grounding_case.json", case_payload)
    (case_dir / "condition_only_query_text.txt").write_text(
        query_text.rstrip() + "\n",
        encoding="utf-8",
    )
    return case_payload


def build_condition_only_report_for_app(
    *,
    story: str,
    output_dir: str | Path,
    grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> dict[str, Any]:
    output = Path(output_dir)
    case = build_condition_only_case_for_app(
        story=story,
        output_dir=output / "condition_only_grounding",
        grounder=grounder,
    )
    report = condition_report_runner(
        SimpleNamespace(
            case_json=case["artifact_paths"]["case_json"],
            summary_json=None,
            case_name=None,
            case_index=0,
            output_dir=str(output / "condition_only_report"),
            bridge_arrays=DEFAULT_PREFIX_BRIDGE_ARRAYS,
            bridge_adapter=DEFAULT_BRIDGE_ADAPTER,
            embedding_model="text-embedding-3-small",
            dotenv=".env",
        )
    )
    report["condition_only_case"] = case
    return report


def enrich_prefix_report_with_product_gate(report: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(_as_dict(report.get("artifact_paths")).get("report", ""))
    if not report_path.exists():
        return report
    decomposition = decompose_report(report_path, top_factors=8)
    product_gate = {
        "production_decision": production_decision([decomposition]),
        "decompositions": [decomposition],
    }
    enriched = {**report, "condition_only_product_gate": product_gate}
    _write_json(report_path, enriched)
    return enriched


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
    condition_report: str | None = None,
    explicit_start_window_index: int | None = None,
    start_state_json: str | None = None,
    skip_rollout: bool = False,
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
        condition_report=condition_report,
        live_story=bool(live_story),
        story=str(story or DEFAULT_STORY),
        grounding_json=None,
        grounding_model="gpt-5.4-mini",
        grounding_max_output_tokens=1200,
        embedding_model="text-embedding-3-small",
        bridge_adapter=DEFAULT_BRIDGE_ADAPTER,
        start_reliability_manifest=(
            DEFAULT_PREFIX_START_RELIABILITY_MANIFEST
            if Path(DEFAULT_PREFIX_START_RELIABILITY_MANIFEST).exists()
            else None
        ),
        dotenv=".env",
        start_mode=str(start_mode),
        explicit_start_window_index=explicit_start_window_index,
        start_state_json=start_state_json,
        start_distance_threshold_z=15.0,
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        memory_prior_mode="soft_topk_narrative_start_checked",
        memory_prior_top_k=8,
        memory_prior_temperature=0.2,
        memory_prior_diverse_max_pairwise_cosine=0.98,
        prefix_prior_mode="decoder",
        include_original_baseline=True,
        hidden_dim=256,
        steps=1000,
        batch_size=64,
        eval_batch_size=16,
        lr=1e-3,
        seed=791,
        device="cuda",
        skip_rollout=bool(skip_rollout),
        samples=int(samples),
        n_steps=30,
        chunk_size=max(4, min(16, int(samples))),
        temperature=DEFAULT_PREFIX_ROLLOUT_TEMPERATURE,
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
    cached_condition_report: str | None = None,
    condition_only_story: bool = False,
    use_explicit_start: bool = False,
    explicit_start_window_index: float | int | None = None,
    use_user_start_state: bool = False,
    start_state_json: str | None = DEFAULT_USER_START_STATE_JSON,
    approve_start: bool = True,
    skip_rollout: bool = False,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_prefix_latent_story_smoke,
    condition_grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> Any:
    start_time = time.monotonic()
    effective_start_mode = (
        "user_start_state"
        if bool(use_user_start_state)
        else "explicit_start_window" if bool(use_explicit_start) else str(start_mode)
    )
    explicit_start = (
        int(explicit_start_window_index)
        if bool(use_explicit_start)
        and not bool(use_user_start_state)
        and explicit_start_window_index is not None
        else None
    )
    user_start_path = str(start_state_json or "").strip() or None
    cached_report_path = str(cached_condition_report or "").strip()
    running_status = _prefix_progress_status_markdown(
        start_time=start_time,
        start_mode=effective_start_mode,
        samples=int(samples),
        temperature=DEFAULT_PREFIX_ROLLOUT_TEMPERATURE,
        live_story=bool(live_story),
        condition_only_story=bool(condition_only_story),
        cached_condition_report=bool(cached_report_path),
        skip_rollout=bool(skip_rollout),
    )
    yield _blank_prefix_outputs(status=running_status, fan_market=fan_market)

    try:
        condition_report_payload: dict[str, Any] | None = None
        condition_report_path: str | None = None
        output_dir = DEFAULT_PREFIX_APP_OUTPUT_DIR
        if cached_report_path:
            condition_report_path = cached_report_path
            if not Path(condition_report_path).exists():
                raise FileNotFoundError(
                    f"cached condition report not found: {condition_report_path}"
                )
            output_dir = str(
                Path(DEFAULT_PREFIX_APP_OUTPUT_DIR)
                / "cached_casebook_run"
                / _slug(Path(condition_report_path).parent.name)
            )
        elif bool(condition_only_story):
            condition_report_payload = build_condition_only_report_for_app(
                story=str(story or DEFAULT_STORY),
                output_dir=Path(DEFAULT_PREFIX_APP_OUTPUT_DIR) / "condition_only_live",
                grounder=condition_grounder,
                condition_report_runner=condition_report_runner,
            )
            condition_report_path = str(
                _as_dict(condition_report_payload.get("artifact_paths")).get("report")
            )
            output_dir = str(Path(DEFAULT_PREFIX_APP_OUTPUT_DIR) / "condition_only_run")
        args = build_prefix_latent_run_args(
            start_mode=effective_start_mode,
            samples=int(samples),
            live_story=(
                bool(live_story)
                and not bool(condition_only_story)
                and not bool(cached_report_path)
            ),
            story=str(story or DEFAULT_STORY),
            condition_report=condition_report_path,
            explicit_start_window_index=explicit_start,
            start_state_json=user_start_path if bool(use_user_start_state) else None,
            skip_rollout=bool(skip_rollout),
            output_dir=output_dir,
        )
        report = runner(args)
        if condition_report_payload is not None:
            report["condition_only_case"] = condition_report_payload.get(
                "condition_only_case"
            )
        report = enrich_prefix_report_with_product_gate(report)
        if runner is run_prefix_latent_story_smoke:
            run_record_path = Path(output_dir) / "run_record" / "prefix_latent_run_record.json"
            _as_dict(report.setdefault("artifact_paths", {}))["run_record"] = str(
                run_record_path
            )
            report_path_text = str(
                _as_dict(report.get("artifact_paths")).get("report", "")
            )
            if report_path_text:
                _write_json(Path(report_path_text), report)
            run_record = write_prefix_run_record(
                report,
                output_dir=Path(output_dir) / "run_record",
            )
            report["run_record_summary"] = {
                "record_id": run_record.get("record_id", ""),
                "status": run_record.get("status", ""),
                "artifact_paths": run_record.get("artifact_paths", {}),
            }
    except Exception as error:  # pragma: no cover - defensive UI path
        error_report = {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        yield (
            "The prefix-latent run failed before a report could be produced.",
            _error_status_markdown(error, start_time),
            _frame([], PREFIX_SELECTED_START_COLUMNS),
            _frame([], PREFIX_VARIANT_COLUMNS),
            _frame([], VALIDATION_GATE_COLUMNS),
            _frame([], SCENARIO_COLUMNS),
            fan_chart_figure({}, fan_market, "ALL"),
            report_json_text(error_report),
            error_report,
            analogue_scope_update({}),
            _frame([], PREFIX_CONDITION_COLUMNS),
            _frame([], WARNING_COLUMNS),
            _frame([], PREFIX_WARNING_COMPONENT_COLUMNS),
            _frame([], PREFIX_SHIFT_FACTOR_COLUMNS),
            _frame([], PREFIX_START_CANDIDATE_COLUMNS),
            _frame([], PREFIX_USER_START_COLUMNS),
            historical_start_candidate_update({}),
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
        prefix_selected_start_table(report),
        prefix_diagnostic_start_table(report),
        prefix_validation_table(report),
        scenario_table(report),
        fan_chart_figure(report, fan_market, "ALL"),
        report_json_text(report),
        report,
        analogue_scope_update(report),
        prefix_condition_implications_table(report),
        prefix_condition_warnings_table(report),
        prefix_warning_component_table(report),
        prefix_shift_factor_table(report),
        prefix_start_candidates_table(report),
        prefix_user_start_table(report),
        historical_start_candidate_update(report),
    )


def preview_prefix_start_for_app(
    start_mode: str,
    samples: int,
    fan_market: str,
    analogue_scope: str,
    live_story: bool = False,
    story: str = DEFAULT_STORY,
    cached_condition_report: str | None = None,
    condition_only_story: bool = False,
    use_explicit_start: bool = False,
    explicit_start_window_index: float | int | None = None,
    use_user_start_state: bool = False,
    start_state_json: str | None = DEFAULT_USER_START_STATE_JSON,
) -> Any:
    """Preview start/support diagnostics without running the final rollout."""

    yield from run_prefix_latent_for_app(
        start_mode=start_mode,
        samples=samples,
        fan_market=fan_market,
        analogue_scope=analogue_scope,
        live_story=live_story,
        story=story,
        cached_condition_report=cached_condition_report,
        condition_only_story=condition_only_story,
        use_explicit_start=use_explicit_start,
        explicit_start_window_index=explicit_start_window_index,
        use_user_start_state=use_user_start_state,
        start_state_json=start_state_json,
        approve_start=False,
        skip_rollout=True,
    )


def _manual_start_index(value: float | int | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manual_start_required_outputs(
    *, fan_market: str
) -> tuple[
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    go.Figure,
    str,
    dict[str, Any],
    Any,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
]:
    status = "\n".join(
        [
            "## Scenario Workflow Status",
            "",
            "- Historical start required.",
            "- Enter a bridge-local historical start window index before preview or generation.",
        ]
    )
    outputs = list(_blank_prefix_outputs(status=status, fan_market=fan_market))
    outputs[0] = (
        "## Historical Start Required\n\n"
        "This production workflow does not infer a starting level. Enter the "
        "historical start window index supplied by the risk manager."
    )
    return tuple(outputs)


def preview_live_openai_start_for_app(
    samples: int,
    fan_market: str,
    analogue_scope: str,
    story: str = DEFAULT_STORY,
    explicit_start_window_index: float | int | None = None,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_prefix_latent_story_smoke,
    condition_grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> Any:
    """Preview a production-style live OpenAI narrative condition and start."""

    explicit_start = _manual_start_index(explicit_start_window_index)
    if explicit_start is None:
        yield _manual_start_required_outputs(fan_market=fan_market)
        return
    yield from run_prefix_latent_for_app(
        start_mode="explicit_start_window",
        samples=samples,
        fan_market=fan_market,
        analogue_scope=analogue_scope,
        live_story=True,
        story=story,
        cached_condition_report="",
        condition_only_story=True,
        use_explicit_start=True,
        explicit_start_window_index=explicit_start,
        use_user_start_state=False,
        start_state_json=None,
        approve_start=False,
        skip_rollout=True,
        runner=runner,
        condition_grounder=condition_grounder,
        condition_report_runner=condition_report_runner,
    )


def run_live_openai_prefix_for_app(
    samples: int,
    fan_market: str,
    analogue_scope: str,
    story: str = DEFAULT_STORY,
    explicit_start_window_index: float | int | None = None,
    *,
    runner: Callable[[SimpleNamespace], dict[str, Any]] = run_prefix_latent_story_smoke,
    condition_grounder: Callable[..., Any] = ground_condition_only_story_with_openai,
    condition_report_runner: Callable[
        [SimpleNamespace],
        dict[str, Any],
    ] = run_condition_only_report,
) -> Any:
    """Run production-style live OpenAI narrative conditioning and rollout."""

    explicit_start = _manual_start_index(explicit_start_window_index)
    if explicit_start is None:
        yield _manual_start_required_outputs(fan_market=fan_market)
        return
    yield from run_prefix_latent_for_app(
        start_mode="explicit_start_window",
        samples=samples,
        fan_market=fan_market,
        analogue_scope=analogue_scope,
        live_story=True,
        story=story,
        cached_condition_report="",
        condition_only_story=True,
        use_explicit_start=True,
        explicit_start_window_index=explicit_start,
        use_user_start_state=False,
        start_state_json=None,
        approve_start=True,
        skip_rollout=False,
        runner=runner,
        condition_grounder=condition_grounder,
        condition_report_runner=condition_report_runner,
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

    with gr.Blocks(title="Narrative Conditioned Scenario Demo") as demo:
        prefix_report_state = gr.State({})
        prefix_samples = gr.State(16)
        gr.Markdown(
            "# Narrative-Conditioned Scenario Generator\n"
            "Describe the current market story, select a historical starting level, "
            "and generate a 30-day scenario distribution.",
            elem_classes=["demo-hero", "demo-shell"],
        )
        story = gr.Textbox(
            label="Risk-manager narrative",
            value=DEFAULT_STORY,
            lines=6,
            max_lines=10,
            placeholder="Describe the current/recent market state in risk-manager language.",
            elem_classes=["demo-shell"],
        )
        gr.Markdown("## Main Workflow")
        gr.Markdown(
            "Enter the historical starting level selected by the risk manager, then generate scenarios."
        )
        with gr.Accordion("How to read this screen", open=False):
            gr.Markdown(
                "- A historical start is the day-0 market level. In production, "
                "the risk manager supplies this level from today's market or a "
                "chosen historical window.\n"
                "- The narrative drives the support mixture and decoded prefix. "
                "Grounding is used as an audit check so future-looking claims are "
                "not treated as guaranteed outcomes.\n"
                "- Story support means the selected support set is compatible with "
                "the narrative and the chosen starting level.\n"
                "- Result notes are product guidance, not forecasts. The fan chart "
                "is the model's 30-day conditional distribution from the selected start."
            )
        with gr.Row(equal_height=False, elem_classes=["demo-responsive-row"]):
            with gr.Column(scale=1, min_width=280):
                prefix_explicit_start_index = gr.Number(
                    value=22,
                    precision=0,
                    label="Historical start window index",
                    info=(
                        "Bridge-local window index for the starting market level. "
                        "In production this is supplied by the risk manager."
                    ),
                )
                gr.Markdown(
                    "Reliability-checked demo starts: `0`, `18`, `22`, `40`, "
                    "`77` pass. `178` is kept as a high-instability hard case.",
                    elem_classes=["demo-hint"],
                )
            with gr.Column(scale=1, min_width=280):
                prefix_fan_market = gr.Dropdown(
                    choices=FAN_MARKET_CHOICES,
                    value="SPX",
                    label="Scenario factor",
                )
            with gr.Column(scale=1, min_width=280):
                prefix_run_button = gr.Button(
                    "Generate 30-Day Scenarios",
                    variant="primary",
                )
        prefix_status = gr.Markdown(
            "## Scenario Workflow Status\n\n- Waiting. Enter a narrative and historical start window, then generate scenarios.",
            label="Prefix-latent status",
        )
        prefix_analogue_scope = gr.Dropdown(
            choices=[("All", "ALL")],
            value="ALL",
            visible=False,
            show_label=False,
        )
        prefix_explicit_start_candidate = gr.Dropdown(
            choices=[],
            value=None,
            visible=False,
            show_label=False,
        )
        prefix_fan_plot = gr.Plot(label="30-day scenario fan chart")
        gr.Markdown(
            "The selected starting level is the day-0 market state used before the "
            "narrative-conditioned support mixture and rollout are built."
        )
        prefix_selected_start = gr.Dataframe(
            headers=PREFIX_SELECTED_START_COLUMNS,
            label="Selected starting level",
            interactive=False,
            elem_classes=[DEMO_TABLE_CLASS],
        )
        gr.Markdown(
            "Terminal deltas are changes from the selected starting level. The "
            "Starting Level column shows the historical reference used as day 0."
        )
        prefix_scenario = gr.Dataframe(
            headers=SCENARIO_COLUMNS,
            label="30-day terminal delta summary",
            interactive=False,
            elem_classes=[DEMO_TABLE_CLASS],
        )
        with gr.Accordion("Audit details", open=False):
            gr.Markdown(
                "Audit details explain why the run was accepted or flagged. "
                "Grounded implications are current/recent market claims extracted "
                "from the story; forward-looking language is kept as a warning-only "
                "sidecar; support candidates show the historical evidence pool used "
                "to construct the latent prefix."
            )
            prefix_condition_warnings = gr.Dataframe(
                headers=WARNING_COLUMNS,
                label="Warnings",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_validation = gr.Dataframe(
                headers=VALIDATION_GATE_COLUMNS,
                label="Validation",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_condition_implications = gr.Dataframe(
                headers=PREFIX_CONDITION_COLUMNS,
                label="Grounded implications",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_diagnostic_start = gr.Dataframe(
                headers=PREFIX_VARIANT_COLUMNS,
                label="Original-start comparison",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_start_candidates = gr.Dataframe(
                headers=PREFIX_START_CANDIDATE_COLUMNS,
                label="Support candidates",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_user_start = gr.Dataframe(
                headers=PREFIX_USER_START_COLUMNS,
                label="User start diagnostics",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_warning_components = gr.Dataframe(
                headers=PREFIX_WARNING_COMPONENT_COLUMNS,
                label="Warning decomposition",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            prefix_shift_factors = gr.Dataframe(
                headers=PREFIX_SHIFT_FACTOR_COLUMNS,
                label="Rollout sensitivity",
                interactive=False,
                elem_classes=[DEMO_TABLE_CLASS],
            )
            with gr.Accordion("Markdown report", open=False):
                prefix_report_markdown = gr.Markdown(label="Report")
            with gr.Accordion("Raw JSON", open=False):
                prefix_report_json = gr.Code(language="json", label="JSON")
        prefix_run_button.click(
            fn=run_live_openai_prefix_for_app,
            inputs=[
                prefix_samples,
                prefix_fan_market,
                prefix_analogue_scope,
                story,
                prefix_explicit_start_index,
            ],
            outputs=[
                prefix_report_markdown,
                prefix_status,
                prefix_selected_start,
                prefix_diagnostic_start,
                prefix_validation,
                prefix_scenario,
                prefix_fan_plot,
                prefix_report_json,
                prefix_report_state,
                prefix_analogue_scope,
                prefix_condition_implications,
                prefix_condition_warnings,
                prefix_warning_components,
                prefix_shift_factors,
                prefix_start_candidates,
                prefix_user_start,
                prefix_explicit_start_candidate,
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
    return demo


def main() -> None:
    import gradio as gr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--auth-user-env", default=DEFAULT_AUTH_USER_ENV)
    parser.add_argument("--auth-password-env", default=DEFAULT_AUTH_PASSWORD_ENV)
    parser.add_argument(
        "--require-auth",
        action="store_true",
        help="fail launch unless auth user/password env vars are present",
    )
    args = parser.parse_args()
    auth = resolve_launch_auth(
        user_env=str(args.auth_user_env),
        password_env=str(args.auth_password_env),
        require_auth=bool(args.require_auth),
    )
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=int(args.server_port),
        share=bool(args.share),
        auth=auth,
        theme=gr.themes.Origin(),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
