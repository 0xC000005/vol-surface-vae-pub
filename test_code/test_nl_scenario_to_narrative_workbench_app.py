from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioSidecarV1,
    normalize_factor_table_csv_text,
)
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import (
    DEFAULT_CARDS_JSONL,
    FACTOR_ROW_COLUMNS,
    PACKET_COLUMNS,
    build_demo,
    factor_move_plot,
    factor_rows_dataframe,
    normalize_generated_deck_for_app,
    normalize_factor_table_for_app,
    normalize_historical_for_app,
    packet_preview_rows,
    status_cards_markdown,
)


def test_factor_rows_dataframe_has_numeric_review_columns() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n",
        scenario_id="demo_upload",
    )

    frame = factor_rows_dataframe(sidecar)

    assert list(frame.columns) == [
        "Factor",
        "Start",
        "End",
        "Delta",
        "Direction",
        "Magnitude",
        "Confidence",
    ]
    assert frame.iloc[0]["Factor"] == "SPX"
    assert frame.iloc[0]["Delta"] == 10.0


def test_packet_preview_rows_pairs_positive_and_negative_text() -> None:
    packet = {
        "paired_review": [
            {
                "view_name": "sparse_user_query",
                "positive_text": "SPX up, DXY down.",
                "negative_window_id": "joint39_train_0100",
                "negative_text": "SPX lower, DXY firmer.",
            }
        ]
    }

    frame = packet_preview_rows(packet)

    assert frame.iloc[0]["View"] == "sparse_user_query"
    assert frame.iloc[0]["Positive"] == "SPX up, DXY down."
    assert frame.iloc[0]["Hard Negative"] == "SPX lower, DXY firmer."


def test_status_cards_markdown_summarizes_sidecar() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="demo_upload",
    )

    text = status_cards_markdown(sidecar, validation_status="waiting")

    assert "ScenarioSidecarV1" in text
    assert "`factor_table_partial`" in text
    assert "`waiting`" in text


def test_empty_display_helpers_keep_stable_dataframe_columns() -> None:
    sidecar = ScenarioSidecarV1(
        scenario_id="empty_upload",
        scenario_type="factor_table_partial",
        mechanical_summary="Mechanical baseline unavailable.",
    )

    factor_frame = factor_rows_dataframe(sidecar)
    packet_frame = packet_preview_rows({})

    assert factor_frame.empty
    assert list(factor_frame.columns) == FACTOR_ROW_COLUMNS
    assert packet_frame.empty
    assert list(packet_frame.columns) == PACKET_COLUMNS


def test_factor_move_plot_handles_missing_and_empty_sidecar() -> None:
    empty_sidecar = ScenarioSidecarV1(
        scenario_id="empty_upload",
        scenario_type="factor_table_partial",
        mechanical_summary="Mechanical baseline unavailable.",
    )

    missing_figure = factor_move_plot(None)
    empty_figure = factor_move_plot(empty_sidecar)

    assert len(missing_figure.data) == 0
    assert missing_figure.layout.title.text == "No scenario loaded"
    assert len(empty_figure.data) == 0
    assert empty_figure.layout.title.text == "No scenario loaded"


def test_normalize_factor_table_for_app_returns_visual_outputs() -> None:
    csv_text = "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n"

    status, frame, warnings_json, sidecar_json = normalize_factor_table_for_app(
        csv_text
    )

    assert "ScenarioSidecarV1" in status
    assert frame.iloc[0]["Factor"] == "SPX"
    assert "partial_factor_coverage" in warnings_json
    assert '"scenario_id": "uploaded_factor_table"' in sidecar_json


def test_normalize_generated_deck_for_app_returns_visual_outputs(tmp_path) -> None:
    report_path = tmp_path / "generated_case.json"
    report_path.write_text(
        json.dumps(
            {
                "generation": {
                    "forecast_steps": 30,
                    "sample_count": 8,
                    "generated_state_shape": [1, 8, 30, 39],
                    "terminal_delta_summary": [
                        {
                            "market": "GOLD",
                            "mean_terminal_delta": 21.5,
                            "p10": 5.0,
                            "p50": 20.0,
                            "p90": 40.0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    status, frame, warnings_json, sidecar_json = normalize_generated_deck_for_app(
        str(report_path)
    )

    assert "ScenarioSidecarV1" in status
    assert "`generated_deck`" in status
    assert frame.iloc[0]["Factor"] == "GOLD"
    assert json.loads(warnings_json) == []
    assert '"scenario_id": "generated_case"' in sidecar_json
    assert '"p90_delta": 40.0' in sidecar_json


def test_normalize_historical_for_app_uses_injected_cards_jsonl(tmp_path) -> None:
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(
        json.dumps(
            {
                "window_id": "joint39_train_0010",
                "scenario_title": "defensive dollar bid",
                "archetype": "liquidity_withdrawal",
                "caption_fields": {
                    "evidence_used": ["DXY higher medium", "SPX lower small"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status, frame, warnings_json, sidecar_json = normalize_historical_for_app(
        " joint39_train_0010 ",
        cards_jsonl=cards_path,
    )

    assert "ScenarioSidecarV1" in status
    assert "`historical_joint39`" in status
    assert frame.empty
    assert json.loads(warnings_json) == []
    assert '"scenario_id": "joint39_train_0010"' in sidecar_json
    assert "DXY higher medium" in sidecar_json


def _component_by_label(demo, label: str) -> dict:
    for component in demo.config.get("components", []):
        if component.get("props", {}).get("label") == label:
            return component
    raise AssertionError(f"component not found: {label}")


def test_build_demo_exposes_historical_cards_jsonl_and_wires_callbacks() -> None:
    demo = build_demo()
    window_id = _component_by_label(demo, "Historical Joint39 window id")
    cards_jsonl = _component_by_label(demo, "Historical cards JSONL")

    assert cards_jsonl["props"]["value"] == str(DEFAULT_CARDS_JSONL)

    expected_inputs = [window_id["id"], cards_jsonl["id"]]
    matching_dependencies = [
        dependency
        for dependency in demo.config.get("dependencies", [])
        if dependency.get("inputs") == expected_inputs
    ]
    assert len(matching_dependencies) == 2


def test_build_demo_generated_deck_placeholder_targets_report_snapshot() -> None:
    demo = build_demo()
    deck_report_path = _component_by_label(demo, "Generated deck report JSON")

    placeholder = deck_report_path["props"]["placeholder"]
    assert "prefix_report_snapshot.json" in placeholder
    assert "fixed_start_live_story_deck_analysis.json" not in placeholder
