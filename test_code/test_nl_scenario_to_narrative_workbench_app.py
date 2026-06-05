from __future__ import annotations

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioSidecarV1,
    normalize_factor_table_csv_text,
)
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench_app import (
    FACTOR_ROW_COLUMNS,
    PACKET_COLUMNS,
    factor_move_plot,
    factor_rows_dataframe,
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
