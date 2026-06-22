from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

sys.path.insert(0, ".")

import experiments.backfill.block_ar.nl_scenario_to_narrative_workbench as workbench_module
from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
    ScenarioFactorRowV1,
    ScenarioNarrativePacketV1,
    ScenarioSidecarV1,
    build_target_payload_from_sidecar,
    load_generated_deck_sidecar_from_report,
    load_historical_joint39_sidecar,
    normalize_historical_joint39_card,
    normalize_factor_table_csv_text,
    normalize_generated_deck_summary,
    run_workbench_packet,
    select_sidecar_negative_candidates,
)


def test_workbench_core_does_not_import_private_helper_scripts() -> None:
    source = Path(workbench_module.__file__).read_text(encoding="utf-8")

    assert "nl_hard_negative_bank_regenerate" not in source
    assert "nl_sparse_variant_pilot" not in source
    assert "nl_reverse_caption_scenario_deck" not in source


def test_workbench_core_import_does_not_load_runner_stack() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import sys

sys.path.insert(0, ".")
import experiments.backfill.block_ar.nl_scenario_to_narrative_workbench  # noqa: F401

for module_name in [
    "experiments.backfill.block_ar.nl_14_view_variant_pilot",
    "experiments.backfill.block_ar.nl_codex_caption_batch",
    "experiments.backfill.block_ar.nl_sparse_variant_pilot",
    "experiments.backfill.block_ar.nl_hard_negative_bank_regenerate",
]:
    if module_name in sys.modules:
        raise SystemExit(f"imported {module_name}")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


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


def test_sidecar_negative_candidates_use_real_card_windows() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "near miss risk pressure",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower; DXY higher"},
        },
        {
            "window_id": "joint39_train_0200",
            "scenario_title": "same direction",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX higher; DXY lower"},
        },
    ]

    candidates = select_sidecar_negative_candidates(
        sidecar=sidecar,
        cards=cards,
        count=1,
    )

    assert candidates[0]["window_id"] == "joint39_train_0100"
    assert "SPX" in candidates[0]["contradiction_channels"]
    assert "DXY" in candidates[0]["contradiction_channels"]


def test_sidecar_negative_candidates_parse_comma_separated_factor_clauses() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "comma separated pressure",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower, DXY higher"},
        },
    ]

    candidates = select_sidecar_negative_candidates(sidecar, cards, 1)

    assert candidates[0]["contradiction_channels"] == ["SPX", "DXY"]


def test_sidecar_negative_candidates_preserve_caption_evidence_rows() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "near miss risk pressure",
            "archetype": "mixed_ambiguous",
            "caption_fields": {
                "mechanical_summary": "SPX lower; DXY higher",
                "evidence_used": ["SPX lower medium", "DXY higher small"],
            },
        },
    ]

    candidates = select_sidecar_negative_candidates(
        sidecar=sidecar,
        cards=cards,
        count=1,
    )

    assert candidates[0]["evidence_used"] == [
        "SPX lower medium",
        "DXY higher small",
    ]


def test_sidecar_negative_candidates_do_not_cross_talk_between_factors() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "ambiguous clauses",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX; DXY lower"},
        },
    ]

    with pytest.raises(
        ValueError, match="only found 0 sidecar negative candidates, requested 1"
    ):
        select_sidecar_negative_candidates(
            sidecar=sidecar,
            cards=cards,
            count=1,
        )


def test_sidecar_negative_candidates_accept_positional_args() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "near miss equity pressure",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower"},
        },
    ]

    candidates = select_sidecar_negative_candidates(sidecar, cards, 1)

    assert candidates[0]["window_id"] == "joint39_train_0100"


def test_sidecar_negative_candidates_skip_blank_window_ids() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": " ",
            "scenario_title": "blank id",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower"},
        },
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "valid id",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower"},
        },
    ]

    candidates = select_sidecar_negative_candidates(sidecar, cards, 1)

    assert candidates[0]["window_id"] == "joint39_train_0100"


def test_sidecar_negative_candidates_use_evidence_only_mechanical_summary() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"evidence_used": ["SPX lower medium"]},
        },
    ]

    candidates = select_sidecar_negative_candidates(sidecar, cards, 1)

    assert candidates[0]["evidence_used"] == ["SPX lower medium"]
    assert candidates[0]["mechanical_summary"] == (
        "Mechanical baseline: SPX lower medium."
    )


def test_sidecar_negative_candidates_do_not_cross_talk_between_spreads() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nBBB_OAS,1.0,1.2,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "spread ambiguity",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "BBB_OAS flat; AAA_OAS tighter"},
        },
    ]

    with pytest.raises(
        ValueError, match="only found 0 sidecar negative candidates, requested 1"
    ):
        select_sidecar_negative_candidates(sidecar, cards, 1)


def test_sidecar_negative_candidates_tie_break_by_window_id() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="upload",
    )
    cards = [
        {
            "window_id": "joint39_train_0200",
            "scenario_title": "later id",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower"},
        },
        {
            "window_id": "joint39_train_0100",
            "scenario_title": "earlier id",
            "archetype": "mixed_ambiguous",
            "caption_fields": {"mechanical_summary": "SPX lower"},
        },
    ]

    candidates = select_sidecar_negative_candidates(sidecar, cards, 1)

    assert candidates[0]["window_id"] == "joint39_train_0100"


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


def test_load_generated_deck_sidecar_from_report_uses_terminal_summary(
    tmp_path,
) -> None:
    report_path = tmp_path / "deck_report.json"
    report_path.write_text(
        json.dumps(
            {
                "generation": {
                    "forecast_steps": 30,
                    "sample_count": 16,
                    "generated_state_shape": [1, 16, 30, 39],
                    "terminal_delta_summary": [
                        {
                            "market": "SPX",
                            "mean_terminal_delta": 12.5,
                            "p10": -5.0,
                            "p50": 11.0,
                            "p90": 30.0,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    sidecar = load_generated_deck_sidecar_from_report(report_path)

    assert sidecar.scenario_id == "deck_report"
    assert sidecar.scenario_type == "generated_deck"
    assert sidecar.factor_rows[0].factor == "SPX"
    assert sidecar.factor_rows[0].p90_delta == 30.0


def test_load_generated_deck_sidecar_from_report_resolves_repo_relative_paths(
    tmp_path,
    monkeypatch,
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_path = reports_dir / "relative_deck.json"
    report_path.write_text(
        json.dumps(
            {
                "generation": {
                    "forecast_steps": 20,
                    "sample_count": 4,
                    "generated_state_shape": [1, 4, 20, 39],
                    "terminal_delta_summary": [
                        {
                            "market": "DXY",
                            "mean_terminal_delta": -1.25,
                            "p10": -2.0,
                            "p50": -1.0,
                            "p90": 0.25,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.setattr(workbench_module, "ROOT", tmp_path)
    monkeypatch.chdir(unrelated_cwd)

    sidecar = load_generated_deck_sidecar_from_report("reports/relative_deck.json")

    assert sidecar.scenario_id == "relative_deck"
    assert sidecar.horizon_days == 20
    assert sidecar.sample_count == 4
    assert sidecar.factor_rows[0].factor == "DXY"
    assert sidecar.factor_rows[0].delta == pytest.approx(-1.25)
    assert sidecar.factor_rows[0].p10_delta == pytest.approx(-2.0)
    assert sidecar.source_artifacts["report"] == str(report_path)


def test_load_generated_deck_sidecar_from_report_filters_unsupported_markets(
    tmp_path,
) -> None:
    report_path = tmp_path / "mapped_deck.json"
    report_path.write_text(
        json.dumps(
            {
                "generation": {
                    "forecast_steps": 30,
                    "sample_count": 8,
                    "terminal_delta_summary": [
                        {"market": "SPX", "mean_terminal_delta": 10.0},
                        {"market": "CRUDE_OIL", "mean_terminal_delta": 3.0},
                        {"market": "GOLD", "mean_terminal_delta": 5.0},
                        {"market": "US2Y", "mean_terminal_delta": 0.2},
                        {"market": "AAA_OAS", "mean_terminal_delta": -0.1},
                        {"market": "USDJPY", "mean_terminal_delta": 4.0},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    sidecar = load_generated_deck_sidecar_from_report(report_path)

    factors = [row.factor for row in sidecar.factor_rows]
    assert factors == ["SPX", "CRUDE", "GOLD"]
    assert "US2Y" not in factors
    assert "AAA_OAS" not in factors
    assert "USDJPY" not in factors


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


def test_build_target_payload_from_sidecar() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n",
        scenario_id="demo_upload",
    )

    target = build_target_payload_from_sidecar(sidecar)

    assert target["window_id"] == "demo_upload"
    assert target["scenario_title"] == "demo_upload"
    assert target["archetype"] == "mixed_ambiguous"
    assert "SPX up large" in target["mechanical_summary"]
    assert "SPX: start=100; end=110; delta=10" in target["evidence_used"]


def test_build_target_payload_from_generated_deck_uses_start_end_delta() -> None:
    sidecar = normalize_generated_deck_summary(
        {
            "summary_source": "report_terminal_delta_summary",
            "sample_count": 16,
            "future_len": 30,
            "factor_rows": [
                {
                    "factor": "SPX",
                    "terminal_mean_delta": 48.0,
                    "terminal_p10_delta": -74.0,
                    "terminal_p50_delta": 52.0,
                    "terminal_p90_delta": 110.0,
                },
                {
                    "factor": "DXY",
                    "terminal_mean_delta": -1.5,
                },
            ],
        },
        scenario_id="generated_demo",
        report_path="/tmp/report.json",
    )

    target = build_target_payload_from_sidecar(sidecar)

    assert "SPX: start=0; end=48; delta=48" in target["evidence_used"]
    assert "DXY: start=0; end=-1.5; delta=-1.5" in target["evidence_used"]
    assert all("mean_terminal_delta" not in row for row in target["evidence_used"])


def _packet_test_candidates() -> list[dict[str, object]]:
    return [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "scenario_title": f"candidate {idx}",
            "archetype": "mixed_ambiguous",
            "mechanical_summary": "Mechanical baseline: SPX lower; DXY higher.",
            "evidence_used": ["SPX lower", "DXY higher"],
            "contradiction_channels": ["SPX", "DXY", "GOLD"],
            "contradiction_count": 3,
            "agreement_count": 1,
        }
        for idx in range(100, 140)
    ]


def test_packet_from_report_preserves_structured_pair_rows() -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\n",
        scenario_id="demo_upload",
    )
    pairs = [
        {
            "view_name": "desk_note",
            "positive_text": "SPX is bid while the dollar softens.",
            "negative_window_id": "joint39_train_0100",
            "negative_text": "SPX is offered while the dollar firms.",
            "quality_notes": ["representative"],
        }
    ]

    packet = workbench_module._packet_from_report(
        sidecar=sidecar,
        report={
            "pairs": pairs,
            "validation": {"status": "pass"},
            "artifact_paths": {
                "report": "/tmp/fourteen_view_report.json",
                "review": "/tmp/fourteen_view_review.md",
            },
        },
    )

    assert packet.positive_narratives == [
        {"view_name": "desk_note", "text": "SPX is bid while the dollar softens."}
    ]
    assert packet.hard_negative_narratives == [
        {
            "view_name": "desk_note",
            "negative_window_id": "joint39_train_0100",
            "text": "SPX is offered while the dollar firms.",
        }
    ]
    assert packet.paired_review == pairs
    assert packet.artifact_paths["report"] == "/tmp/fourteen_view_report.json"
    assert packet.artifact_paths["review"] == "/tmp/fourteen_view_review.md"


def test_run_workbench_packet_dry_run_writes_normalized_packet_json(
    tmp_path, monkeypatch
) -> None:
    sidecar = normalize_factor_table_csv_text(
        "factor,start,end,confidence\nSPX,100,110,medium\nDXY,90,85,high\n",
        scenario_id="demo_upload",
    )
    root = Path(__file__).resolve().parents[1]
    relative_output_dir = Path(
        os.path.relpath(tmp_path / "runner_output", start=root)
    )
    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    packet = run_workbench_packet(
        sidecar=sidecar,
        negative_candidates=_packet_test_candidates(),
        output_dir=relative_output_dir,
        dry_run=True,
    )

    expected_output_dir = (root / relative_output_dir).resolve()
    expected_packet_path = expected_output_dir / "scenario_narrative_packet.json"
    assert packet.scenario_sidecar.scenario_id == "demo_upload"
    assert packet.validation["status"] == "fail"
    assert packet.artifact_paths["packet"] == str(expected_packet_path)
    assert packet.artifact_paths["report"] == str(
        expected_output_dir / "fourteen_view_report.json"
    )
    assert packet.artifact_paths["review"] == str(
        expected_output_dir / "fourteen_view_review.md"
    )

    written_packet = json.loads(expected_packet_path.read_text(encoding="utf-8"))
    assert written_packet["scenario_sidecar"]["scenario_id"] == "demo_upload"
    assert written_packet["artifact_paths"]["packet"] == str(expected_packet_path)
    assert written_packet["artifact_paths"]["report"] == str(
        expected_output_dir / "fourteen_view_report.json"
    )
    assert written_packet["artifact_paths"]["review"] == str(
        expected_output_dir / "fourteen_view_review.md"
    )


def test_narrative_packet_accepts_planned_shape() -> None:
    sidecar = ScenarioSidecarV1(
        scenario_id="demo_upload",
        scenario_type="factor_table_partial",
        mechanical_summary="Mechanical baseline:",
    )

    packet = ScenarioNarrativePacketV1(
        scenario_sidecar=sidecar,
        positive_narratives=[
            {"view_name": "desk_note", "text": "SPX rises while spreads widen."}
        ],
        hard_negative_narratives=[
            {
                "view_name": "desk_note",
                "negative_window_id": "joint39_train_0100",
                "text": "SPX falls while spreads tighten.",
            }
        ],
        paired_review=[
            {
                "view_name": "desk_note",
                "positive_text": "SPX rises while spreads widen.",
                "negative_window_id": "joint39_train_0100",
                "negative_text": "SPX falls while spreads tighten.",
            }
        ],
        validation={"status": "pending"},
        artifact_paths={"sidecar": "sidecar.json"},
    )

    assert packet.scenario_sidecar == sidecar
    assert packet.positive_narratives == [
        {"view_name": "desk_note", "text": "SPX rises while spreads widen."}
    ]
    assert packet.hard_negative_narratives == [
        {
            "view_name": "desk_note",
            "negative_window_id": "joint39_train_0100",
            "text": "SPX falls while spreads tighten.",
        }
    ]
    assert packet.paired_review == [
        {
            "view_name": "desk_note",
            "positive_text": "SPX rises while spreads widen.",
            "negative_window_id": "joint39_train_0100",
            "negative_text": "SPX falls while spreads tighten.",
        }
    ]
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


def test_historical_joint39_card_extracts_support_metadata_factor_rows(tmp_path) -> None:
    card = {
        "window_id": "joint39_train_0011",
        "scenario_title": "numeric support rows",
        "caption_fields": {
            "evidence_used": ["SPX higher small"],
        },
        "support_metadata": {
            "support_move_rows": [
                {
                    "market": "SPX",
                    "raw_change": 12.5,
                    "magnitude": "small",
                },
                {
                    "market": "BBB_OAS",
                    "raw_change": -0.2,
                    "magnitude": "small",
                },
            ],
        },
    }
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")

    sidecar = normalize_historical_joint39_card(card, source_path=cards_path)

    rows = {row.factor: row for row in sidecar.factor_rows}
    assert rows["SPX"].delta == pytest.approx(12.5)
    assert rows["SPX"].direction == "up"
    assert rows["BBB_OAS"].delta == pytest.approx(-0.2)
    assert rows["BBB_OAS"].direction == "tighter"


def test_historical_joint39_card_extracts_raw_deltas_from_evidence(tmp_path) -> None:
    card = {
        "window_id": "joint39_train_0012",
        "scenario_title": "numeric evidence rows",
        "caption_fields": {
            "evidence_used": [
                "DXY -3.09, z=-1.176326; USDJPY +0.962, z=3.618730.",
                "AAA_OAS widened small/medium: +552.89, z=0.570700.",
            ],
        },
    }
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")

    sidecar = normalize_historical_joint39_card(card, source_path=cards_path)

    rows = {row.factor: row for row in sidecar.factor_rows}
    assert rows["DXY"].delta == pytest.approx(-3.09)
    assert rows["USDJPY"].delta == pytest.approx(0.962)
    assert rows["AAA_OAS"].delta == pytest.approx(552.89)


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


def test_load_historical_joint39_sidecar_enriches_raw_history_paths(tmp_path) -> None:
    card = {
        "window_id": "joint39_train_0001",
        "scenario_title": "path-rich historical case",
        "caption_fields": {
            "evidence_used": ["SPX higher small", "DXY lower small"],
        },
    }
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")
    history_raw = np.zeros((2, 30, 39), dtype=np.float32)
    history_raw[1, :, 25] = np.asarray([100, 102, 101, 104] + [104] * 26)
    history_raw[1, :, 28] = np.linspace(90.0, 87.0, 30)
    arrays_path = tmp_path / "support_bank_arrays.npz"
    np.savez(arrays_path, history_raw=history_raw)

    sidecar, _ = load_historical_joint39_sidecar(
        cards_path,
        "joint39_train_0001",
        support_arrays_path=arrays_path,
    )

    rows = {row.factor: row for row in sidecar.factor_rows}
    assert rows["SPX"].path_values[:4] == pytest.approx([100.0, 102.0, 101.0, 104.0])
    assert rows["SPX"].start == pytest.approx(100.0)
    assert rows["SPX"].end == pytest.approx(104.0)
    assert rows["DXY"].path_values[0] == pytest.approx(90.0)
    assert rows["DXY"].path_values[-1] == pytest.approx(87.0)
    assert sidecar.source_artifacts["support_arrays"] == str(arrays_path)


def test_load_historical_joint39_sidecar_uses_correct_factor_tail_columns(tmp_path) -> None:
    card = {
        "window_id": "joint39_train_0001",
        "scenario_title": "factor tail mapping",
        "caption_fields": {
            "evidence_used": ["AAA_OAS wider small", "USDJPY lower small"],
        },
    }
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(json.dumps(card) + "\n", encoding="utf-8")
    history_raw = np.zeros((2, 30, 39), dtype=np.float32)
    # Joint39 stores 25 IV cells followed by the 14 data/multi_factor_data.npz
    # level columns:
    # spx, usdcad, usdjpy, dxy, copper, wheat, crude_oil, us2y,
    # us10y, aaa_oas, bbb_oas, nikkei, gold, vix.
    usdjpy_path = np.linspace(106.35, 98.28, 30)
    copper_path = np.linspace(3.266, 1.844, 30)
    aaa_oas_path = np.linspace(2.76, 4.07, 30)
    bbb_oas_path = np.linspace(4.02, 6.98, 30)
    nikkei_path = np.linspace(12090.59, 8576.98, 30)
    history_raw[1, :, 27] = usdjpy_path
    history_raw[1, :, 29] = copper_path
    history_raw[1, :, 34] = aaa_oas_path
    history_raw[1, :, 35] = bbb_oas_path
    history_raw[1, :, 36] = nikkei_path
    arrays_path = tmp_path / "support_bank_arrays.npz"
    np.savez(arrays_path, history_raw=history_raw)

    sidecar, _ = load_historical_joint39_sidecar(
        cards_path,
        "joint39_train_0001",
        support_arrays_path=arrays_path,
    )

    display_order = [row.factor for row in sidecar.factor_rows]
    assert display_order[:2] == ["SPX", "NIKKEI"]
    rows = {row.factor: row for row in sidecar.factor_rows}
    assert rows["USDJPY"].start == pytest.approx(usdjpy_path[0])
    assert rows["USDJPY"].end == pytest.approx(usdjpy_path[-1])
    assert rows["AAA_OAS"].start == pytest.approx(aaa_oas_path[0])
    assert rows["AAA_OAS"].end == pytest.approx(aaa_oas_path[-1])
    assert rows["BBB_OAS"].start == pytest.approx(bbb_oas_path[0])
    assert rows["BBB_OAS"].end == pytest.approx(bbb_oas_path[-1])
    assert rows["NIKKEI"].start == pytest.approx(nikkei_path[0])
    assert rows["NIKKEI"].end == pytest.approx(nikkei_path[-1])
