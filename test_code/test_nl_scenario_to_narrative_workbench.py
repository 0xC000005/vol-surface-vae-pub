from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, ".")

import experiments.backfill.block_ar.nl_scenario_to_narrative_workbench as workbench_module
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioFactorRowV1,
    ScenarioNarrativePacketV1,
    ScenarioSidecarV1,
    load_historical_joint39_sidecar,
    normalize_historical_joint39_card,
    normalize_factor_table_csv_text,
    normalize_generated_deck_summary,
)


def test_workbench_core_does_not_import_private_helper_scripts() -> None:
    source = Path(workbench_module.__file__).read_text(encoding="utf-8")

    assert "nl_hard_negative_bank_regenerate" not in source
    assert "nl_sparse_variant_pilot" not in source


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


def test_factor_table_rejects_header_only_csv() -> None:
    with pytest.raises(ValueError, match="factor table contains no factor rows"):
        normalize_factor_table_csv_text("factor,start,end\n", scenario_id="empty")


def test_factor_table_rejects_duplicate_normalized_factors() -> None:
    csv_text = "factor,start,end\n" " SpX ,1294,1311\n" "spx,1290,1300\n"

    with pytest.raises(ValueError, match="duplicate factor"):
        normalize_factor_table_csv_text(csv_text, scenario_id="duplicates")


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("factor,start,end\nSPX,bad,1311\n", "start must be numeric"),
        ("factor,start,end\nSPX,1294,bad\n", "end must be numeric"),
        ("factor,start,end\nSPX,nan,1311\n", "start must be finite"),
        ("factor,start,end\nSPX,1294,nan\n", "end must be finite"),
        ("factor,start,end\nSPX,inf,1311\n", "start must be finite"),
        ("factor,start,end\nSPX,1294,inf\n", "end must be finite"),
    ],
)
def test_factor_table_rejects_invalid_numeric_values(
    csv_text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_factor_table_csv_text(csv_text, scenario_id="invalid_numeric")


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


def test_sidecar_rejects_nonpositive_horizon_days() -> None:
    with pytest.raises(ValidationError):
        ScenarioSidecarV1(
            scenario_id="zero_horizon",
            scenario_type="factor_table_partial",
            horizon_days=0,
            mechanical_summary="Mechanical baseline:",
        )


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


def test_generated_deck_summary_converts_terminal_rows() -> None:
    deck_summary = {
        "summary_source": "report_terminal_delta_summary",
        "sample_count": 16,
        "future_len": 30,
        "factor_rows": [
            {
                "factor": "SPX",
                "direction": "up",
                "magnitude": "large",
                "terminal_mean_delta": 48.0,
                "terminal_p10_delta": -74.0,
                "terminal_p50_delta": 52.0,
                "terminal_p90_delta": 110.0,
            },
            {
                "factor": "BBB_OAS",
                "direction": "tighter",
                "magnitude": "small",
                "terminal_mean_delta": -0.08,
                "terminal_p10_delta": -0.20,
                "terminal_p50_delta": -0.06,
                "terminal_p90_delta": 0.09,
            },
        ],
    }

    sidecar = normalize_generated_deck_summary(
        deck_summary,
        scenario_id="fragile_risk_on_rebound_0",
        report_path="/tmp/report.json",
        arrays_path="/tmp/arrays.npz",
    )

    assert sidecar.scenario_type == "generated_deck"
    assert sidecar.sample_count == 16
    assert sidecar.factor_rows[0].factor == "SPX"
    assert sidecar.factor_rows[0].start == 0.0
    assert sidecar.factor_rows[0].end == 48.0
    assert sidecar.factor_rows[0].p10_delta == pytest.approx(-74.0)
    assert sidecar.factor_rows[0].p50_delta == pytest.approx(52.0)
    assert sidecar.factor_rows[0].p90_delta == pytest.approx(110.0)
    assert sidecar.factor_rows[1].direction == "tighter"
    assert (
        sidecar.mechanical_summary
        == "Mechanical baseline: SPX up large; BBB_OAS tighter small"
    )
    assert sidecar.source_artifacts["report"] == "/tmp/report.json"


def test_generated_deck_summary_rejects_rows_missing_mean_terminal_delta() -> None:
    deck_summary = {
        "factor_rows": [
            {
                "factor": "SPX",
                "direction": "up",
                "magnitude": "large",
            },
            {
                "factor": "BBB_OAS",
                "direction": "tighter",
                "magnitude": "small",
            },
        ],
    }

    with pytest.raises(
        ValueError, match="generated deck summary contains no factor rows"
    ):
        normalize_generated_deck_summary(
            deck_summary,
            scenario_id="missing_mean",
            report_path="/tmp/report.json",
        )


def test_generated_deck_summary_skips_rows_missing_mean_terminal_delta() -> None:
    deck_summary = {
        "factor_rows": [
            {
                "factor": "SPX",
                "direction": "up",
                "magnitude": "large",
            },
            {
                "factor": "DXY",
                "terminal_mean_delta": -1.5,
            },
        ],
    }

    sidecar = normalize_generated_deck_summary(
        deck_summary,
        scenario_id="mixed_validity",
        report_path="/tmp/report.json",
    )

    assert len(sidecar.factor_rows) == 1
    assert sidecar.factor_rows[0].factor == "DXY"
    assert sidecar.factor_rows[0].delta == pytest.approx(-1.5)


def test_generated_deck_summary_derives_labels_from_mean_terminal_delta() -> None:
    deck_summary = {
        "factor_rows": [
            {
                "factor": "SPX",
                "direction": "up",
                "magnitude": "large",
                "terminal_mean_delta": -5.0,
            },
        ],
    }

    sidecar = normalize_generated_deck_summary(
        deck_summary,
        scenario_id="contradictory_labels",
        report_path="/tmp/report.json",
    )

    assert sidecar.factor_rows[0].direction == "down"
    assert sidecar.factor_rows[0].magnitude == "medium"


@pytest.mark.parametrize("bad_delta", ["nan", "inf"])
def test_generated_deck_summary_rejects_nonfinite_mean_terminal_delta(
    bad_delta: str,
) -> None:
    deck_summary = {
        "factor_rows": [
            {
                "factor": "SPX",
                "terminal_mean_delta": bad_delta,
            },
        ],
    }

    with pytest.raises(ValueError, match="terminal_mean_delta must be finite"):
        normalize_generated_deck_summary(
            deck_summary,
            scenario_id="nonfinite_mean",
            report_path="/tmp/report.json",
        )


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


def test_historical_joint39_card_normalizes_caption_fields(tmp_path) -> None:
    card = {
        "window_id": "joint39_train_0005",
        "scenario_title": "equity defensive pressure",
        "archetype": "liquidity_withdrawal",
        "caption_fields": {
            "evidence_used": ["SPX lower medium", "VIX higher small"],
        },
        "views": {},
    }
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")

    sidecar = normalize_historical_joint39_card(card, source_path=cards_path)

    assert sidecar.scenario_id == "joint39_train_0005"
    assert sidecar.scenario_type == "historical_joint39"
    assert sidecar.scenario_title == "equity defensive pressure"
    assert sidecar.archetype == "liquidity_withdrawal"
    assert sidecar.source_artifacts["cards_jsonl"] == str(cards_path)
    assert sidecar.normalization_warnings == []
    assert (
        sidecar.mechanical_summary
        == "Mechanical baseline: SPX lower medium; VIX higher small."
    )


def test_historical_joint39_card_warns_when_caption_evidence_missing() -> None:
    sidecar = normalize_historical_joint39_card(
        {
            "window_id": "joint39_train_0008",
            "scenario_title": "missing evidence",
            "caption_fields": {},
        },
        source_path="cards.jsonl",
    )

    assert sidecar.normalization_warnings == [
        {
            "code": "missing_caption_evidence",
            "message": "Historical card has no caption_fields.evidence_used rows.",
        }
    ]


def test_historical_joint39_card_requires_window_id() -> None:
    with pytest.raises(ValueError, match="historical card is missing window_id"):
        normalize_historical_joint39_card({}, source_path="cards.jsonl")


def test_historical_joint39_card_defaults_missing_archetype() -> None:
    card = {
        "window_id": "joint39_train_0006",
        "scenario_title": "rates pressure",
        "caption_fields": {
            "evidence_used": ["US10Y higher medium"],
        },
        "views": {},
    }

    sidecar = normalize_historical_joint39_card(card, source_path="cards.jsonl")

    assert sidecar.archetype == "mixed_ambiguous"


def test_load_historical_joint39_sidecar_accepts_positional_args(tmp_path) -> None:
    card = {
        "window_id": "joint39_train_0007",
        "scenario_title": "credit pressure",
        "archetype": "credit_stress",
        "caption_fields": {
            "evidence_used": ["BBB_OAS wider medium"],
        },
        "views": {},
    }
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")

    sidecar, raw_card = load_historical_joint39_sidecar(
        cards_path, "joint39_train_0007"
    )

    assert sidecar.scenario_id == "joint39_train_0007"
    assert sidecar.source_artifacts["cards_jsonl"] == str(cards_path)
    assert raw_card == card


def test_load_historical_joint39_sidecar_compacts_target_window_id(tmp_path) -> None:
    card = {
        "window_id": " joint39_train_0009 ",
        "scenario_title": "target id whitespace",
        "caption_fields": {
            "evidence_used": ["SPX higher small"],
        },
    }
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")

    sidecar, raw_card = load_historical_joint39_sidecar(
        cards_path, "\n joint39_train_0009  "
    )

    assert sidecar.scenario_id == "joint39_train_0009"
    assert raw_card == card
