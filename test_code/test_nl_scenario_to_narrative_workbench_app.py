from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioFactorRowV1,
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
    generate_narrative_packet_outputs_for_app,
    generate_narrative_packet_for_app,
    historical_date_choices,
    historical_period_range_markdown,
    historical_start_date_choices,
    historical_window_range_markdown,
    mode_visibility_flags,
    normalize_generated_deck_for_app,
    normalize_factor_table_for_app,
    normalize_historical_date_for_app,
    normalize_historical_start_date_for_app,
    normalize_historical_start_date_safe_for_app,
    normalize_historical_for_app,
    normalize_uploaded_scenario_for_app,
    packet_preview_rows,
    packet_review_markdown,
    status_cards_markdown,
    window_id_for_calendar_end_date,
    window_id_for_calendar_start_date,
)


def test_factor_rows_dataframe_has_formatted_review_columns() -> None:
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
    # D2: Start/End/Delta are clean per-factor unit strings, not raw floats.
    assert frame.iloc[0]["Start"] == "100.00 pts"
    assert frame.iloc[0]["Delta"] == "+10.00 pts"


def test_factor_rows_dataframe_formats_percent_factors_in_bp() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nUS10Y,6.39,6.01,high\nBBB_OAS,2.25,2.18,medium\n",
        scenario_id="rate_upload",
    )

    frame = factor_rows_dataframe(sidecar)

    assert frame.iloc[0]["Start"] == "6.39%"
    assert frame.iloc[0]["Delta"] == "-38 bp"
    assert frame.iloc[1]["Start"] == "2.25%"
    assert frame.iloc[1]["Delta"] == "-7 bp"


def test_factor_rows_dataframe_maps_provenance_confidence_to_level() -> None:
    # D1: historical builder overloads `confidence` as a provenance tag; the rendered
    # Confidence column must show a real level (high/medium/low), never the raw tag.
    sidecar = ScenarioSidecarV1(
        scenario_id="hist_case",
        scenario_type="historical_joint39",
        mechanical_summary="Mechanical baseline: SPX up small.",
        factor_rows=[
            ScenarioFactorRowV1(
                factor="SPX",
                start=100.0,
                end=103.0,
                delta=3.0,
                direction="up",
                magnitude="medium",
                confidence="raw_history_path",
                evidence="raw path fixture",
                path_values=[100.0, 103.0],
            )
        ],
    )

    frame = factor_rows_dataframe(sidecar)

    # Realized historical moves render as "Observed" (honest; not a confidence judgment).
    assert frame.iloc[0]["Confidence"] == "Observed"
    assert "raw_history_path" not in str(frame.iloc[0]["Confidence"])


def test_display_confidence_historical_medium_magnitude_renders_observed() -> None:
    # P2: the support-metadata fallback puts a magnitude string ("medium") in the confidence
    # field; historical rows must still render "Observed" (disambiguated by source builder),
    # not "Medium".
    sidecar = ScenarioSidecarV1(
        scenario_id="hist_fallback",
        scenario_type="historical_joint39",
        mechanical_summary="Mechanical baseline: DXY up medium.",
        factor_rows=[
            ScenarioFactorRowV1(
                factor="DXY",
                start=0.0,
                end=2.0,
                delta=2.0,
                direction="up",
                magnitude="medium",
                confidence="medium",
                evidence="support row fixture",
            )
        ],
    )

    frame = factor_rows_dataframe(sidecar)

    assert frame.iloc[0]["Confidence"] == "Observed"


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


def test_packet_review_markdown_pairs_positive_and_negative_text() -> None:
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

    text = packet_review_markdown(packet)

    assert "Generated Narratives" in text
    # D7: snake_case view key shown as a human-readable title.
    assert "Risk-desk question" in text
    assert "SPX up, DXY down." in text
    # D8: hard-negative provenance shown as a calendar period, not the internal window id.
    assert "2000-05-25 to 2000-07-07" in text
    assert "joint39" not in text
    assert "SPX lower, DXY firmer." in text


def test_status_cards_markdown_summarizes_sidecar() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="demo_upload",
    )

    text = status_cards_markdown(sidecar, validation_status="waiting")

    # D3: plain-language status, no ScenarioSidecarV1 / type-code jargon in the primary view.
    assert "Scenario status" in text
    assert "Uploaded numbers" in text
    assert "`waiting`" in text
    assert "ScenarioSidecarV1" not in text
    assert "factor_table_partial" not in text


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


def test_factor_move_plot_visualizes_thirty_day_factor_paths() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n",
        scenario_id="demo_upload",
    )

    figure = factor_move_plot(sidecar)

    assert figure.layout.title.text == "30-day scenario movement"
    x_axis_titles = [
        getattr(figure.layout, axis_name).title.text
        for axis_name in figure.layout
        if axis_name.startswith("xaxis")
    ]
    assert "Day" in x_axis_titles
    assert len(figure.data) == 2
    assert [trace.type for trace in figure.data] == ["scatter", "scatter"]
    assert [trace.mode for trace in figure.data] == ["lines", "lines"]
    assert [trace.name for trace in figure.data] == ["SPX", "DXY"]
    assert len({trace.yaxis for trace in figure.data}) == 2
    assert list(figure.data[0].x) == list(range(31))
    assert figure.data[0].y[0] == pytest.approx(100.0)
    assert figure.data[0].y[15] == pytest.approx(105.0)
    assert figure.data[0].y[-1] == pytest.approx(110.0)
    assert figure.data[1].y[0] == pytest.approx(90.0)
    assert figure.data[1].y[15] == pytest.approx(87.5)
    assert figure.data[1].y[-1] == pytest.approx(85.0)


def test_factor_move_plot_uses_raw_path_values_when_available() -> None:
    sidecar = ScenarioSidecarV1(
        scenario_id="path_case",
        scenario_type="historical_joint39",
        mechanical_summary="Mechanical baseline: SPX up small.",
        factor_rows=[
            ScenarioFactorRowV1(
                factor="SPX",
                start=100.0,
                end=103.0,
                delta=3.0,
                direction="up",
                magnitude="medium",
                confidence="raw_history_path",
                evidence="raw path fixture",
                path_values=[100.0, 104.0, 101.0, 103.0],
            )
        ],
    )

    figure = factor_move_plot(sidecar)

    assert list(figure.data[0].x) == [0, 1, 2, 3]
    assert list(figure.data[0].y) == pytest.approx([100.0, 104.0, 101.0, 103.0])


def test_normalize_factor_table_for_app_returns_visual_outputs() -> None:
    csv_text = "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n"

    status, frame, warnings_json, sidecar_json, sidecar_state = normalize_factor_table_for_app(
        csv_text
    )

    assert "Uploaded numbers" in status
    assert frame.iloc[0]["Factor"] == "SPX"
    assert "partial_factor_coverage" in warnings_json
    assert '"scenario_id": "uploaded_factor_table"' in sidecar_json
    assert sidecar_state == sidecar_json


def test_normalize_uploaded_scenario_for_app_returns_visual_outputs() -> None:
    csv_text = "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n"

    status, frame, warnings_json, sidecar_json, sidecar_state = (
        normalize_uploaded_scenario_for_app(csv_text)
    )

    assert "Uploaded numbers" in status
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

    assert "Generated scenario deck" in status
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

    assert "Historical case" in status
    assert not frame.empty
    assert frame.iloc[0]["Factor"] == "DXY"
    assert json.loads(warnings_json) == []
    assert '"scenario_id": "joint39_train_0010"' in sidecar_json
    assert "DXY higher medium" in sidecar_json
    assert sidecar_state == sidecar_json


def test_historical_date_lookup_maps_2008_crisis_date_to_window() -> None:
    choices = historical_date_choices()
    values = [value for _label, value in choices]
    labels = [label for label, _value in choices]

    assert "2008-10-31" in values
    assert any("joint39_train_2190" in label for label in labels)
    assert window_id_for_calendar_end_date("2008-10-31") == "joint39_train_2190"
    assert window_id_for_calendar_end_date(" 2008-10-31 ") == "joint39_train_2190"


def test_historical_start_date_lookup_maps_2008_crisis_period_to_window() -> None:
    choices = historical_start_date_choices()
    values = [value for _label, value in choices]
    labels = [label for label, _value in choices]

    assert "2008-09-22" in values
    assert window_id_for_calendar_start_date("2008-09-22") == "joint39_train_2190"
    assert window_id_for_calendar_start_date(" 2008-09-22 ") == "joint39_train_2190"
    assert all("joint39" not in label.lower() for label in labels)
    assert all("train" not in label.lower() for label in labels)
    assert all("turbulent" not in label.lower() for label in labels)


def test_offered_start_dates_all_have_a_card() -> None:
    """Regression: every start-date the picker offers must map to a window that has a
    stride-5 narrative card. Otherwise the scenario loader raises 'target window not
    found' (e.g. 2008-09-25 -> joint39_train_2193, which has no stride-5 card)."""
    import json as _json

    from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import (
        DEFAULT_CARDS_JSONL,
        DEFAULT_SUPPORT_BANK_REPORT,
        _support_window_metadata,
    )
    from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import _compact

    card_ids: set[str] = set()
    with open(DEFAULT_CARDS_JSONL, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                card_ids.add(_compact(_json.loads(line).get("window_id")))
    assert card_ids, "expected a non-empty stride-5 cards corpus"

    start_to_window = {
        str(row.get("calendar_start_date", "")).strip(): _compact(row.get("window_id"))
        for row in _support_window_metadata(str(DEFAULT_SUPPORT_BANK_REPORT))
    }
    offered = [value for _label, value in historical_start_date_choices()]
    uncovered = [sd for sd in offered if start_to_window.get(sd) not in card_ids]
    assert not uncovered, (
        f"{len(uncovered)} offered start-dates have no stride-5 card "
        f"(e.g. {uncovered[:3]}) -> would fail with 'target window not found'"
    )
    assert "2008-09-25" not in offered  # joint39_train_2193: no stride-5 card
    assert "2008-09-22" in offered  # joint39_train_2190: card-backed (default)


def test_historical_window_range_markdown_shows_thirty_day_condition() -> None:
    text = historical_window_range_markdown("2008-10-31")

    assert "joint39_train_2190" in text
    assert "2008-09-22 to 2008-10-31" in text
    assert "30-day conditioning window" in text


def test_historical_period_range_markdown_hides_internal_ids() -> None:
    text = historical_period_range_markdown("2008-09-22")

    assert "2008-09-22 to 2008-10-31" in text
    assert "30-day historical period" in text
    assert "joint39" not in text.lower()
    assert "train" not in text.lower()
    assert "turbulent" not in text.lower()


def test_normalize_historical_date_for_app_returns_range_markdown() -> None:
    (
        status,
        frame,
        warnings_json,
        sidecar_json,
        sidecar_state,
        range_markdown,
    ) = normalize_historical_date_for_app("2008-10-31")

    assert "Historical case" in status
    assert not frame.empty
    assert json.loads(warnings_json) == []
    assert '"scenario_id": "joint39_train_2190"' in sidecar_json
    assert sidecar_state == sidecar_json
    assert "2008-09-22 to 2008-10-31" in range_markdown


def test_normalize_historical_start_date_safe_handles_bad_cards_path() -> None:
    # MEDIUM: a bad "Reference cards JSONL" path raises FileNotFoundError from the loader;
    # the safe wrapper must land a friendly message in the status panel (6 outputs, no raise).
    (
        status,
        frame,
        warnings_json,
        sidecar_json,
        sidecar_state,
        range_markdown,
    ) = normalize_historical_start_date_safe_for_app(
        "2008-09-22",
        cards_jsonl="experiments/backfill/block_ar/nl_scenario_demo_outputs/__missing__.jsonl",
    )

    assert "could not load" in status
    assert frame.empty
    assert warnings_json == "[]"
    assert sidecar_state == ""
    assert "Selected Historical Period" in range_markdown


def test_normalize_historical_start_date_for_app_returns_range_markdown() -> None:
    (
        status,
        frame,
        warnings_json,
        sidecar_json,
        sidecar_state,
        range_markdown,
    ) = normalize_historical_start_date_for_app("2008-09-22")

    assert "Historical case" in status
    assert not frame.empty
    assert json.loads(warnings_json) == []
    assert '"scenario_id": "joint39_train_2190"' in sidecar_json
    assert sidecar_state == sidecar_json
    assert "2008-09-22 to 2008-10-31" in range_markdown
    assert "joint39" not in range_markdown.lower()


def _component_by_label(demo, label: str) -> dict:
    for component in demo.config.get("components", []):
        if component.get("props", {}).get("label") == label:
            return component
    raise AssertionError(f"component not found: {label}")


def test_build_demo_exposes_date_selector_and_wires_historical_callbacks() -> None:
    demo = build_demo()
    historical_start_date = _component_by_label(
        demo,
        "Select the historical period by starting date",
    )
    cards_jsonl = _component_by_label(demo, "Reference cards JSONL")
    labels = {
        component.get("props", {}).get("label")
        for component in demo.config.get("components", [])
    }

    assert cards_jsonl["props"]["value"] == str(DEFAULT_CARDS_JSONL)
    assert "Historical window id" not in labels
    assert "Historical ending date" not in labels
    assert "Turbulent period preset" not in labels
    assert "2008-09-22" in [
        choice[1] for choice in historical_start_date["props"]["choices"]
    ]
    assert all(
        "joint39" not in str(choice[0]).lower()
        for choice in historical_start_date["props"]["choices"]
    )
    assert all(
        "turbulent" not in str(choice[0]).lower()
        for choice in historical_start_date["props"]["choices"]
    )

    expected_inputs = [historical_start_date["id"], cards_jsonl["id"]]
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
    # D10: mode-change toggles the two visibility columns AND clears the right-panel outputs
    # (incl. both Technical-details Code blocks), so it now drives 11 components; columns first.
    assert len(mode_dependency["outputs"]) == 11

    components = _components_by_id(demo)
    output_components = [
        components[int(component_id)] for component_id in mode_dependency["outputs"]
    ]
    assert [component["type"] for component in output_components[:2]] == [
        "column",
        "column",
    ]
    # D10: default Input mode is now Historical Case (historical group visible, uploaded hidden).
    assert [component["props"]["visible"] for component in output_components[:2]] == [
        True,
        False,
    ]


def test_build_demo_exposes_only_two_user_modes_with_generated_narratives() -> None:
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
    assert "Generated narratives" in labels


def test_build_demo_wires_generate_button_to_status_and_narrative_review() -> None:
    demo = build_demo()
    sidecar_state = next(
        component
        for component in demo.config.get("components", [])
        if component.get("type") == "state"
    )
    cards_jsonl = _component_by_label(demo, "Reference cards JSONL")
    output_dir = _component_by_label(demo, "Packet output directory")

    matching_dependencies = [
        dependency
        for dependency in demo.config.get("dependencies", [])
        if dependency.get("inputs") == [
            sidecar_state["id"],
            cards_jsonl["id"],
            output_dir["id"],
        ]
    ]

    assert len(matching_dependencies) == 1
    # D9: generation now also drives the packet download component (status, narratives, file).
    assert len(matching_dependencies[0]["outputs"]) == 3


def test_generate_narrative_packet_for_app_requires_loaded_scenario() -> None:
    text = generate_narrative_packet_for_app("")

    assert "Generation Status" in text
    assert "load a scenario first" in text


def test_generate_narrative_packet_outputs_for_app_requires_loaded_scenario() -> None:
    status, narratives = generate_narrative_packet_outputs_for_app("")

    assert "Generation Status" in status
    assert "load a scenario first" in status
    assert "Generated Narratives" in narratives
    assert "load a scenario first" in narratives
