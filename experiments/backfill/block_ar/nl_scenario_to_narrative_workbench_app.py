#!/usr/bin/env python
"""Structured display helpers for the scenario-to-narrative workbench."""

from __future__ import annotations

import argparse
import json
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
    load_generated_deck_sidecar_from_report,
    load_historical_joint39_sidecar,
    normalize_factor_table_csv_text,
)


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
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


def _sidecar_outputs(sidecar: ScenarioSidecarV1) -> tuple[str, pd.DataFrame, str, str]:
    return (
        status_cards_markdown(sidecar, validation_status="normalized"),
        factor_rows_dataframe(sidecar),
        json.dumps(sidecar.normalization_warnings, indent=2, sort_keys=True),
        sidecar.model_dump_json(indent=2),
    )


def normalize_factor_table_for_app(csv_text: str) -> tuple[str, pd.DataFrame, str, str]:
    sidecar = _normalize_factor_table_sidecar_for_app(csv_text)
    return _sidecar_outputs(sidecar)


def _normalize_historical_sidecar_for_app(
    target_window_id: str,
    *,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
) -> ScenarioSidecarV1:
    cards_path = Path(cards_jsonl)
    if not cards_path.is_absolute():
        cards_path = ROOT / cards_path
    sidecar, _card = load_historical_joint39_sidecar(
        cards_jsonl=cards_path,
        target_window_id=target_window_id.strip(),
    )
    return sidecar


def normalize_historical_for_app(
    target_window_id: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
) -> tuple[str, pd.DataFrame, str, str]:
    sidecar = _normalize_historical_sidecar_for_app(
        target_window_id,
        cards_jsonl=cards_jsonl,
    )
    return _sidecar_outputs(sidecar)


def _normalize_generated_deck_sidecar_for_app(report_path: str) -> ScenarioSidecarV1:
    return load_generated_deck_sidecar_from_report(report_path.strip())


def normalize_generated_deck_for_app(
    report_path: str,
) -> tuple[str, pd.DataFrame, str, str]:
    sidecar = _normalize_generated_deck_sidecar_for_app(report_path)
    return _sidecar_outputs(sidecar)


def build_demo() -> Any:
    import gradio as gr

    with gr.Blocks(title="Scenario-to-Narrative Workbench") as demo:
        gr.Markdown("# Scenario-to-Narrative Analyst Workbench")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                source_mode = gr.Radio(
                    choices=["Historical Joint39", "Generated Deck", "Factor Table"],
                    value="Factor Table",
                    label="Input mode",
                )
                historical_window_id = gr.Textbox(
                    label="Historical Joint39 window id",
                    value="joint39_train_1553",
                )
                historical_cards_jsonl = gr.Textbox(
                    label="Historical cards JSONL",
                    value=str(DEFAULT_CARDS_JSONL),
                )
                historical_button = gr.Button("Load Historical Joint39")
                deck_report_path = gr.Textbox(
                    label="Generated deck report JSON",
                    placeholder=(
                        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
                        ".../prefix_report_snapshot.json"
                    ),
                )
                deck_button = gr.Button("Load Generated Deck")
                factor_csv = gr.Textbox(
                    label="Factor table CSV",
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
                    "Normalize Factor Table",
                    variant="primary",
                )
            with gr.Column(scale=2):
                status = gr.Markdown(
                    status_cards_markdown(None, validation_status="waiting")
                )
                factor_frame = gr.Dataframe(
                    headers=FACTOR_ROW_COLUMNS,
                    label="Factor moves",
                    interactive=False,
                )
                factor_plot = gr.Plot(label="Factor terminal move")
                warnings_json = gr.Code(language="json", label="Warnings")
                sidecar_json = gr.Code(language="json", label="ScenarioSidecarV1")

        historical_button.click(
            fn=normalize_historical_for_app,
            inputs=[historical_window_id, historical_cards_jsonl],
            outputs=[status, factor_frame, warnings_json, sidecar_json],
            show_progress="full",
        ).then(
            fn=lambda target_window_id, cards_jsonl: factor_move_plot(
                _normalize_historical_sidecar_for_app(
                    target_window_id,
                    cards_jsonl=cards_jsonl,
                )
            ),
            inputs=[historical_window_id, historical_cards_jsonl],
            outputs=[factor_plot],
            show_progress="hidden",
        )
        deck_button.click(
            fn=normalize_generated_deck_for_app,
            inputs=[deck_report_path],
            outputs=[status, factor_frame, warnings_json, sidecar_json],
            show_progress="full",
        ).then(
            fn=lambda report_path: factor_move_plot(
                _normalize_generated_deck_sidecar_for_app(report_path)
            ),
            inputs=[deck_report_path],
            outputs=[factor_plot],
            show_progress="hidden",
        )
        normalize_button.click(
            fn=normalize_factor_table_for_app,
            inputs=[factor_csv],
            outputs=[status, factor_frame, warnings_json, sidecar_json],
            show_progress="full",
        ).then(
            fn=lambda text: factor_move_plot(
                _normalize_factor_table_sidecar_for_app(text)
            ),
            inputs=[factor_csv],
            outputs=[factor_plot],
            show_progress="hidden",
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
    )


if __name__ == "__main__":
    main()
