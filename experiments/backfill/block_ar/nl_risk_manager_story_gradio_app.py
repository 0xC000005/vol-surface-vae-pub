#!/usr/bin/env python
"""Local Gradio demo for narrative-conditioned scenario generation."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_BOSS_DEMO_PACK_JSON = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_boss_demo_pack_829a_live_casebook/"
    "boss_demo_pack.json"
)
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
SCENARIO_COLUMNS = ["Market", "Mean Terminal Delta", "P10", "P90"]
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
    """Populate story/start/report controls from a cached casebook selection."""

    value = str(choice or "")
    if not value:
        return (
            DEFAULT_STORY,
            False,
            22,
            True,
            False,
            "",
            "## Cached Casebook\n\n- Selection: `typed story / current controls`",
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
                    "## Cached Casebook",
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
        f"## Cached Casebook\n\n- Selection: `unknown ({value})`",
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


def load_boss_demo_pack(
    path: str | Path = DEFAULT_BOSS_DEMO_PACK_JSON,
) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


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
    return "\n".join(
        [
            "## Demo readiness evidence",
            "",
            f"- Evidence pack: `{artifact_paths.get('summary_markdown', '')}`",
            f"- Offline validation: `{snapshot.get('run_count', 0)}` runs; "
            f"CRPS improved `{snapshot.get('improved_crps_rows', 0)}/{snapshot.get('run_count', 0)}`; "
            f"energy improved `{snapshot.get('improved_energy_rows', 0)}/{snapshot.get('run_count', 0)}`.",
            f"- Offline mean CRPS improvement: `{_fmt_pct(snapshot.get('mean_crps_improvement_vs_persistence'))}`.",
            f"- Live API casebook: `{live_snapshot.get('pass_count', 0)}/{live_snapshot.get('case_count', 0)}` pass; "
            f"OpenAI tokens `{live_snapshot.get('total_openai_tokens', 0)}`; "
            f"min support candidates `{live_snapshot.get('min_support_candidate_count', 0)}`.",
            f"- Live models: grounding `{live_models}`; embedding `{embedding_models}`.",
            "- Contract: current/recent implications condition the generator; forward-risk language is warning-only.",
        ]
    )


def boss_demo_live_casebook_table(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    live_snapshot = _as_dict(report.get("live_casebook_snapshot"))
    for item in _as_list(live_snapshot.get("case_rows")):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Case": str(item.get("case_name", "")),
                "Start": str(item.get("expected_start_index", "")),
                "Status": str(item.get("overall_status", "")),
                "Condition": str(item.get("condition_only_validation_status", "")),
                "Warnings": str(item.get("forward_warning_count", "")),
                "Support": str(item.get("support_candidate_count", "")),
                "Summary": str(item.get("summary_path", "")),
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
    return _prefix_variant_table_for_role(report, role="operational")


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
                f"- Product decision: `{decision.get('decision', 'n/a')}`",
                f"- Product warning: {decision.get('ui_guidance', decision.get('reason', ''))}",
                f"- Main warning contributors: `{', '.join(contributors) or 'n/a'}`",
            ]
        )
    lines.extend(
        [
            f"- Markdown report: `{artifacts.get('markdown', 'n/a')}`",
            f"- JSON report: `{artifacts.get('report', 'n/a')}`",
        ]
    )
    return "\n".join(lines)


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
) -> str:
    if bool(cached_condition_report):
        condition_step = (
            "cached condition-only report, fixed start, prefix decoding, frozen rollout"
        )
    elif bool(condition_only_story):
        condition_step = (
            "condition-only OpenAI grounding, text embedding, balanced start "
            "selection, prefix decoding, frozen rollout, warning decomposition"
        )
    elif bool(live_story):
        condition_step = "OpenAI grounding and embedding, start selection, prefix decoding, frozen rollout"
    else:
        condition_step = (
            "cached text memory, start selection, prefix decoding, frozen rollout"
        )
    return "\n".join(
        [
            "## Prefix-Latent Run Status",
            "",
            f"- Prefix-latent run started: `{_elapsed_text(start_time)} ago`",
            f"- Current step: `{condition_step}`",
            f"- Start mode: `{start_mode}`",
            f"- Generator samples per variant: `{int(samples)}`",
            f"- Calibrated rollout temperature: `{float(temperature):.2f}`",
            "- Outputs will fill in automatically when the run completes.",
        ]
    )


def _completed_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return status_markdown(report) + f"\n- Completed in: `{_elapsed_text(start_time)}`"


def _completed_prefix_status_markdown(report: dict[str, Any], start_time: float) -> str:
    return (
        prefix_latent_status_markdown(report)
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
        _frame([], PREFIX_VARIANT_COLUMNS),
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
        dotenv=".env",
        start_mode=str(start_mode),
        explicit_start_window_index=explicit_start_window_index,
        start_state_json=start_state_json,
        start_distance_threshold_z=15.0,
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        memory_prior_mode="soft_topk_combined",
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
        skip_rollout=False,
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
            output_dir=output_dir,
        )
        report = runner(args)
        if condition_report_payload is not None:
            report["condition_only_case"] = condition_report_payload.get(
                "condition_only_case"
            )
        report = enrich_prefix_report_with_product_gate(report)
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
    boss_demo_pack = load_boss_demo_pack()
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
        gr.Markdown(boss_demo_pack_markdown(boss_demo_pack))
        gr.Dataframe(
            value=boss_demo_live_casebook_table(boss_demo_pack),
            headers=BOSS_DEMO_CASEBOOK_COLUMNS,
            label="Live API casebook readiness",
            interactive=False,
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
            prefix_casebook_choice = gr.Dropdown(
                choices=cached_prefix_casebook_choices(),
                value="",
                label="Cached validated casebook",
                info=(
                    "Choose a previously grounded narrative/start pair. This "
                    "fills the story, fixed start, and cached condition report."
                ),
            )
            prefix_cached_condition_report = gr.Textbox(
                value="",
                label="Cached condition report path",
                visible=False,
            )
        prefix_casebook_status = gr.Markdown(
            "## Cached Casebook\n\n- Selection: `typed story / current controls`",
            label="Cached casebook selection",
        )
        with gr.Row():
            prefix_start_mode = gr.Dropdown(
                choices=[
                    ("Implication-aligned start", "implication_aligned_start"),
                    ("Balanced memory/start support", "balanced_memory_start"),
                    ("Memory-nearest train start", "memory_nearest_start"),
                    ("Original start", "original"),
                    ("Nearest train start", "nearest_train_start"),
                    ("Farthest train start", "farthest_train_start"),
                ],
                value="balanced_memory_start",
                label="Start mode",
            )
            prefix_live_story = gr.Checkbox(
                value=False,
                label="Use typed story (OpenAI TestFlight)",
                info="Unchecked uses cached held-out text memory. Checked grounds and embeds the story above.",
            )
            prefix_condition_only_story = gr.Checkbox(
                value=True,
                label="Condition-only contract",
                info=(
                    "Use current/recent market implications only; forward-looking "
                    "phrases become warnings, not scenario targets."
                ),
            )
            prefix_use_explicit_start = gr.Checkbox(
                value=False,
                label="Use historical start",
                info=(
                    "Override model-chosen start with a bridge-local historical "
                    "window index. Use this as the first user-specified start mode."
                ),
            )
            prefix_use_user_start_state = gr.Checkbox(
                value=False,
                label="Use start JSON",
                info=(
                    "Override historical starts with a raw joint39 start-state JSON. "
                    "This is the stricter user-specified current-state mode."
                ),
            )
            prefix_explicit_start_candidate = gr.Dropdown(
                choices=[],
                value=None,
                label="Historical start candidate",
                info=(
                    "Populated after a run. Selecting a candidate writes its "
                    "index into Historical start window index."
                ),
            )
            prefix_explicit_start_index = gr.Number(
                value=22,
                precision=0,
                label="Historical start window index",
                info="Bridge-local window index; ignored unless Use historical start is checked.",
            )
            prefix_start_state_json = gr.Textbox(
                value=DEFAULT_USER_START_STATE_JSON,
                label="Start-state JSON path",
                info=(
                    "Used only when Use start JSON is checked. The file should "
                    "contain raw joint39 values_by_name or state_vector."
                ),
                lines=1,
            )
            prefix_preview_start_json = gr.Button(
                "Preview Start JSON",
                variant="secondary",
            )
            prefix_export_start_json = gr.Button(
                "Export Candidate Start JSON",
                variant="secondary",
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
        prefix_selected_start = gr.Dataframe(
            headers=PREFIX_VARIANT_COLUMNS,
            label="Proposed selected start",
            interactive=False,
        )
        prefix_diagnostic_start = gr.Dataframe(
            headers=PREFIX_VARIANT_COLUMNS,
            label="Diagnostic original-start comparison",
            interactive=False,
        )
        prefix_validation = gr.Dataframe(
            headers=VALIDATION_GATE_COLUMNS,
            label="Prefix-latent current-run validation",
            interactive=False,
        )
        prefix_condition_implications = gr.Dataframe(
            headers=PREFIX_CONDITION_COLUMNS,
            label="Condition-only implications used for support",
            interactive=False,
        )
        prefix_condition_warnings = gr.Dataframe(
            headers=WARNING_COLUMNS,
            label="Warning-only language excluded from conditioning",
            interactive=False,
        )
        prefix_warning_components = gr.Dataframe(
            headers=PREFIX_WARNING_COMPONENT_COLUMNS,
            label="Product warning decomposition",
            interactive=False,
        )
        prefix_start_candidates = gr.Dataframe(
            headers=PREFIX_START_CANDIDATE_COLUMNS,
            label="Historical start/support candidates",
            interactive=False,
        )
        prefix_user_start = gr.Dataframe(
            headers=PREFIX_USER_START_COLUMNS,
            label="User-supplied start diagnostics",
            interactive=False,
        )
        prefix_start_json_status = gr.Markdown(
            "## Start-State JSON Preview\n\n- Status: `waiting`",
            label="Start-state JSON preview status",
        )
        prefix_start_json_preview = gr.Dataframe(
            headers=PREFIX_START_PREVIEW_COLUMNS,
            label="Start-state JSON preview",
            interactive=False,
        )
        prefix_shift_factors = gr.Dataframe(
            headers=PREFIX_SHIFT_FACTOR_COLUMNS,
            label="Largest rollout-sensitivity contributors",
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
                prefix_cached_condition_report,
                prefix_condition_only_story,
                prefix_use_explicit_start,
                prefix_explicit_start_index,
                prefix_use_user_start_state,
                prefix_start_state_json,
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
        prefix_casebook_choice.change(
            fn=cached_prefix_casebook_update,
            inputs=prefix_casebook_choice,
            outputs=[
                story,
                prefix_use_explicit_start,
                prefix_explicit_start_index,
                prefix_condition_only_story,
                prefix_live_story,
                prefix_cached_condition_report,
                prefix_casebook_status,
            ],
            show_progress="hidden",
        )
        prefix_explicit_start_candidate.change(
            fn=historical_start_candidate_to_index,
            inputs=prefix_explicit_start_candidate,
            outputs=prefix_explicit_start_index,
            show_progress="hidden",
        )
        prefix_preview_start_json.click(
            fn=preview_start_state_json,
            inputs=prefix_start_state_json,
            outputs=[prefix_start_json_status, prefix_start_json_preview],
            show_progress="minimal",
        )
        prefix_export_start_json.click(
            fn=export_historical_start_json_for_app,
            inputs=[prefix_explicit_start_candidate, prefix_explicit_start_index],
            outputs=[
                prefix_start_json_status,
                prefix_start_state_json,
                prefix_start_json_preview,
            ],
            show_progress="minimal",
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
