from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioFactorRowV1,
    ScenarioNarrativePacketV1,
    ScenarioSidecarV1,
    normalize_factor_table_csv_text,
)


def test_factor_table_requires_numeric_start_and_end() -> None:
    csv_text = "factor,start,confidence\nSPX,1294,medium\n"

    with pytest.raises(ValueError, match="missing required column"):
        normalize_factor_table_csv_text(csv_text, scenario_id="demo_upload")


def test_factor_table_normalizes_numeric_rows() -> None:
    csv_text = (
        "factor,start,end,confidence\n"
        "SPX,1294.0,1311.0,medium\n"
        "DXY,90.3,87.2,high\n"
        "BBB_OAS,1.42,1.50,medium\n"
    )

    sidecar = normalize_factor_table_csv_text(csv_text, scenario_id="demo_upload")

    assert isinstance(sidecar, ScenarioSidecarV1)
    assert sidecar.scenario_id == "demo_upload"
    assert sidecar.scenario_type == "factor_table_partial"
    assert sidecar.horizon_days == 30
    rows = {row.factor: row for row in sidecar.factor_rows}
    assert rows["SPX"].delta == pytest.approx(17.0)
    assert rows["SPX"].direction == "up"
    assert rows["DXY"].direction == "down"
    assert rows["BBB_OAS"].direction == "wider"
    assert rows["SPX"].evidence == "start=1294; end=1311; delta=17"


def test_parsed_factor_table_sidecar_has_planned_defaults() -> None:
    csv_text = "factor,start,end\nSPX,1294,1311\n"

    sidecar = normalize_factor_table_csv_text(csv_text, scenario_id="demo_upload")

    assert sidecar.scenario_title == ""
    assert sidecar.archetype == "mixed_ambiguous"
    assert sidecar.normalization_warnings == []
    assert sidecar.summary_source == "uploaded_csv"
    assert sidecar.sample_count is None


def test_sidecar_accepts_planned_non_factor_table_types() -> None:
    for scenario_type in ("historical_joint39", "generated_deck"):
        sidecar = ScenarioSidecarV1(
            scenario_id=f"demo_{scenario_type}",
            scenario_type=scenario_type,
            mechanical_summary="Mechanical baseline:",
        )

        assert sidecar.scenario_type == scenario_type


def test_factor_row_accepts_quantile_deltas() -> None:
    row = ScenarioFactorRowV1(
        factor="SPX",
        start=1294.0,
        end=1311.0,
        delta=17.0,
        direction="up",
        magnitude="large",
        evidence="start=1294; end=1311; delta=17",
        p10_delta=9.0,
        p50_delta=17.0,
        p90_delta=25.0,
    )

    assert row.p10_delta == pytest.approx(9.0)
    assert row.p50_delta == pytest.approx(17.0)
    assert row.p90_delta == pytest.approx(25.0)


def test_narrative_packet_accepts_planned_shape() -> None:
    sidecar = ScenarioSidecarV1(
        scenario_id="demo_upload",
        scenario_type="factor_table_partial",
        mechanical_summary="Mechanical baseline:",
    )

    packet = ScenarioNarrativePacketV1(
        scenario_sidecar=sidecar,
        positive_narratives=["SPX rises while spreads widen."],
        hard_negative_narratives=["SPX falls while spreads tighten."],
        paired_review={"status": "unreviewed"},
        validation={"status": "pending"},
        artifact_paths={"sidecar": "sidecar.json"},
    )

    assert packet.scenario_sidecar == sidecar
    assert packet.positive_narratives == ["SPX rises while spreads widen."]
    assert packet.hard_negative_narratives == ["SPX falls while spreads tighten."]
    assert packet.paired_review == {"status": "unreviewed"}
    assert packet.validation == {"status": "pending"}
    assert packet.artifact_paths == {"sidecar": "sidecar.json"}
