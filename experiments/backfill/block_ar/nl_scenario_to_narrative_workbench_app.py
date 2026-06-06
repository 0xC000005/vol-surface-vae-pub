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
    run_workbench_packet,
    select_sidecar_negative_candidates,
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
) -> tuple[str, pd.DataFrame, str, str, str]:
    sidecar = _normalize_historical_sidecar_for_app(
        target_window_id,
        cards_jsonl=cards_jsonl,
    )
    return _sidecar_outputs(sidecar)


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
    return (
        "## Generation Status\n\n"
        f"- Validation: `{validation_status}`\n"
        f"- Errors: `{error_count}`\n"
        f"- Packet JSON: `{packet_path}`"
    )


def generate_narrative_packet_for_app(
    sidecar_json: str,
    cards_jsonl: str | Path = DEFAULT_CARDS_JSONL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    dry_run: bool = False,
) -> str:
    if not str(sidecar_json or "").strip():
        return (
            "## Generation Status\n\n"
            "- Validation: `waiting`\n"
            "- Packet JSON: `load a scenario first`"
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
    return generation_status_markdown(
        packet.artifact_paths.get("packet", ""),
        str(validation.get("status", "unknown")),
        int(validation.get("error_count", 0) or 0),
    )


def build_demo() -> Any:
    import gradio as gr

    def mode_visibility_updates(source_mode: str) -> tuple[Any, Any]:
        return tuple(
            gr.update(visible=visible)
            for visible in mode_visibility_flags(source_mode)
        )

    with gr.Blocks(title="Scenario-to-Narrative Workbench") as demo:
        gr.Markdown("# Scenario-to-Narrative Analyst Workbench")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                source_mode = gr.Radio(
                    choices=SOURCE_MODE_CHOICES,
                    value="Uploaded Numerical Scenario",
                    label="Input mode",
                )
                with gr.Column(visible=False) as historical_group:
                    historical_window_id = gr.Textbox(
                        label="Historical window id",
                        value="joint39_train_1553",
                    )
                    historical_button = gr.Button("Visualize Historical Scenario")
                with gr.Column(visible=True) as uploaded_group:
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
                        "Visualize Uploaded Scenario",
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
                factor_plot = gr.Plot(label="Scenario move")
                generate_button = gr.Button("Generate + Verify Narrative Packet")
                generation_status = gr.Markdown(
                    "## Generation Status\n\n- Validation: `waiting`"
                )
                with gr.Accordion("Technical details", open=False):
                    warnings_json = gr.Code(language="json", label="Warnings")
                    sidecar_json = gr.Code(language="json", label="ScenarioSidecarV1")

        source_mode.change(
            fn=mode_visibility_updates,
            inputs=[source_mode],
            outputs=[historical_group, uploaded_group],
            show_progress="hidden",
        )
        historical_button.click(
            fn=normalize_historical_for_app,
            inputs=[historical_window_id, reference_cards_jsonl],
            outputs=[
                status,
                factor_frame,
                warnings_json,
                sidecar_json,
                sidecar_state,
            ],
            show_progress="full",
        ).then(
            fn=lambda target_window_id, cards_jsonl: factor_move_plot(
                _normalize_historical_sidecar_for_app(
                    target_window_id,
                    cards_jsonl=cards_jsonl,
                )
            ),
            inputs=[historical_window_id, reference_cards_jsonl],
            outputs=[factor_plot],
            show_progress="hidden",
        )
        normalize_button.click(
            fn=normalize_uploaded_scenario_for_app,
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
            fn=lambda text: factor_move_plot(
                _normalize_factor_table_sidecar_for_app(text)
            ),
            inputs=[factor_csv],
            outputs=[factor_plot],
            show_progress="hidden",
        )
        generate_button.click(
            fn=generate_narrative_packet_for_app,
            inputs=[sidecar_state, reference_cards_jsonl, output_dir],
            outputs=[generation_status],
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
    )


if __name__ == "__main__":
    main()
