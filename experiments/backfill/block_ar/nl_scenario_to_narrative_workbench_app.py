#!/usr/bin/env python
"""Structured display helpers for the scenario-to-narrative workbench."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (  # noqa: E402
    ScenarioSidecarV1,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "scenario_to_narrative_workbench"
)
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


def factor_rows_dataframe(sidecar: ScenarioSidecarV1) -> pd.DataFrame:
    rows = [
        {
            "Factor": row.factor,
            "Start": row.start,
            "End": row.end,
            "Delta": row.delta,
            "Direction": row.direction,
            "Magnitude": row.magnitude,
            "Confidence": row.confidence,
        }
        for row in sidecar.factor_rows
    ]
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


def status_cards_markdown(
    sidecar: ScenarioSidecarV1 | None,
    *,
    validation_status: str,
) -> str:
    if sidecar is None:
        return "## Status\n\n- Sidecar: `waiting`\n- Validation: `waiting`"
    return (
        "## Status\n\n"
        "- Sidecar: `ScenarioSidecarV1`\n"
        f"- Type: `{sidecar.scenario_type}`\n"
        f"- Factors: `{len(sidecar.factor_rows)}`\n"
        f"- Warnings: `{len(sidecar.normalization_warnings)}`\n"
        f"- Validation: `{validation_status}`"
    )


def factor_move_plot(sidecar: ScenarioSidecarV1 | None) -> go.Figure:
    fig = go.Figure()
    if sidecar is None or not sidecar.factor_rows:
        fig.update_layout(title="No scenario loaded")
        return fig

    names = [row.factor for row in sidecar.factor_rows]
    deltas = [row.delta for row in sidecar.factor_rows]
    colors = ["#0f766e" if value >= 0 else "#b42318" for value in deltas]
    fig.add_bar(x=names, y=deltas, marker_color=colors)
    fig.update_layout(
        title="Factor terminal move",
        xaxis_title="Factor",
        yaxis_title="End minus start",
        margin={"l": 40, "r": 20, "t": 45, "b": 80},
        height=360,
    )
    return fig
