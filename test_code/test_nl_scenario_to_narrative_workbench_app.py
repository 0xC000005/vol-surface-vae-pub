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
    SOURCE_MODE_CHOICES,
    build_demo,
    factor_move_plot,
    factor_rows_dataframe,
    generate_narrative_packet_for_app,
    mode_visibility_flags,
    normalize_generated_deck_for_app,
    normalize_factor_table_for_app,
    normalize_historical_for_app,
    normalize_uploaded_scenario_for_app,
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

    status, frame, warnings_json, sidecar_json, sidecar_state = normalize_factor_table_for_app(
        csv_text
    )

    assert "ScenarioSidecarV1" in status
    assert frame.iloc[0]["Factor"] == "SPX"
    assert "partial_factor_coverage" in warnings_json
    assert '"scenario_id": "uploaded_factor_table"' in sidecar_json
    assert sidecar_state == sidecar_json


def test_normalize_uploaded_scenario_for_app_returns_visual_outputs() -> None:
    csv_text = "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n"

    status, frame, warnings_json, sidecar_json, sidecar_state = (
        normalize_uploaded_scenario_for_app(csv_text)
    )

    assert "ScenarioSidecarV1" in status
    assert frame.iloc[0]["Factor"] == "SPX"
    assert "partial_factor_coverage" in warnings_json
    assert '"scenario_id": "uploaded_factor_table"' in sidecar_json
    assert sidecar_state == sidecar_json


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

    status, frame, warnings_json, sidecar_json, sidecar_state = (
        normalize_generated_deck_for_app(str(report_path))
    )

    assert "ScenarioSidecarV1" in status
    assert "`generated_deck`" in status
    assert frame.iloc[0]["Factor"] == "GOLD"
    assert json.loads(warnings_json) == []
    assert '"scenario_id": "generated_case"' in sidecar_json
    assert '"p90_delta": 40.0' in sidecar_json
    assert sidecar_state == sidecar_json


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
                "support_metadata": {
                    "support_move_rows": [
                        {"market": "DXY", "raw_change": 2.0, "magnitude": "medium"},
                        {"market": "SPX", "raw_change": -8.0, "magnitude": "medium"},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status, frame, warnings_json, sidecar_json, sidecar_state = (
        normalize_historical_for_app(
            " joint39_train_0010 ",
            cards_jsonl=cards_path,
        )
    )

    assert "ScenarioSidecarV1" in status
    assert "`historical_joint39`" in status
    assert not frame.empty
    assert frame.iloc[0]["Factor"] == "DXY"
    assert json.loads(warnings_json) == []
    assert '"scenario_id": "joint39_train_0010"' in sidecar_json
    assert "DXY higher medium" in sidecar_json
    assert sidecar_state == sidecar_json


def _component_by_label(demo, label: str) -> dict:
    for component in demo.config.get("components", []):
        if component.get("props", {}).get("label") == label:
            return component
    raise AssertionError(f"component not found: {label}")


def test_build_demo_exposes_reference_cards_jsonl_and_wires_historical_callbacks() -> None:
    demo = build_demo()
    window_id = _component_by_label(demo, "Historical window id")
    cards_jsonl = _component_by_label(demo, "Reference cards JSONL")

    assert cards_jsonl["props"]["value"] == str(DEFAULT_CARDS_JSONL)

    expected_inputs = [window_id["id"], cards_jsonl["id"]]
    matching_dependencies = [
        dependency
        for dependency in demo.config.get("dependencies", [])
        if dependency.get("inputs") == expected_inputs
    ]
    assert len(matching_dependencies) == 2


def _components_by_id(demo) -> dict[int, dict]:
    return {
        int(component["id"]): component
        for component in demo.config.get("components", [])
    }


def test_mode_visibility_flags_select_only_active_input_group() -> None:
    assert SOURCE_MODE_CHOICES == ["Historical Case", "Uploaded Numerical Scenario"]
    assert mode_visibility_flags("Historical Case") == (True, False)
    assert mode_visibility_flags("Uploaded Numerical Scenario") == (False, True)
    assert mode_visibility_flags("unknown") == (False, True)


def test_build_demo_wires_input_mode_change_to_visibility_groups() -> None:
    demo = build_demo()
    source_mode = _component_by_label(demo, "Input mode")

    matching_dependencies = [
        dependency
        for dependency in demo.config.get("dependencies", [])
        if dependency.get("inputs") == [source_mode["id"]]
    ]

    assert len(matching_dependencies) == 1
    mode_dependency = matching_dependencies[0]
    assert mode_dependency["outputs"]
    assert len(mode_dependency["outputs"]) == 2

    components = _components_by_id(demo)
    output_components = [
        components[int(component_id)] for component_id in mode_dependency["outputs"]
    ]
    assert [component["type"] for component in output_components] == [
        "column",
        "column",
    ]
    assert [component["props"]["visible"] for component in output_components] == [
        False,
        True,
    ]


def test_build_demo_exposes_only_two_user_modes_without_narrative_table() -> None:
    demo = build_demo()
    source_mode = _component_by_label(demo, "Input mode")
    labels = {
        component.get("props", {}).get("label")
        for component in demo.config.get("components", [])
    }

    assert [tuple(choice) for choice in source_mode["props"]["choices"]] == [
        ("Historical Case", "Historical Case"),
        ("Uploaded Numerical Scenario", "Uploaded Numerical Scenario"),
    ]
    assert "Generated deck report JSON" not in labels
    assert "Factor table CSV" not in labels
    assert "Numerical scenario CSV" in labels
    assert "Positive" not in labels
    assert "Hard Negative" not in labels


def test_generate_narrative_packet_for_app_requires_loaded_scenario() -> None:
    text = generate_narrative_packet_for_app("")

    assert "Generation Status" in text
    assert "load a scenario first" in text
