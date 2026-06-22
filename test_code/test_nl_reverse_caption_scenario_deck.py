import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_reverse_caption_scenario_deck as reverse_caption


def test_summarize_deck_states_uses_operational_variant_and_factor_deltas() -> None:
    states = np.zeros((2, 3, 4, 39), dtype=np.float32)
    requested_raw = np.zeros((2, 39), dtype=np.float32)
    requested_raw[1, reverse_caption.FACTOR_INDEX["SPX"]] = 100.0
    requested_raw[1, reverse_caption.FACTOR_INDEX["VIX"]] = 20.0
    requested_raw[1, reverse_caption.FACTOR_INDEX["BBB_OAS"]] = 1.5
    states[1, :, -1, reverse_caption.FACTOR_INDEX["SPX"]] = [108.0, 109.0, 110.0]
    states[1, :, -1, reverse_caption.FACTOR_INDEX["VIX"]] = [18.0, 17.0, 16.0]
    states[1, :, -1, reverse_caption.FACTOR_INDEX["BBB_OAS"]] = [1.2, 1.1, 1.0]

    summary = reverse_caption.summarize_deck_states(
        arrays={"generated_states": states, "requested_raw": requested_raw},
        operational_variant_index=1,
    )
    by_factor = {row["factor"]: row for row in summary["factor_rows"]}

    assert summary["sample_count"] == 3
    assert by_factor["SPX"]["direction"] == "up"
    assert by_factor["SPX"]["terminal_mean_delta"] == 9.0
    assert by_factor["VIX"]["direction"] == "down"
    assert by_factor["BBB_OAS"]["direction"] == "tighter"


def test_summarize_report_terminal_delta_uses_calibrated_summary() -> None:
    report = {
        "generation": {
            "sample_count": 16,
            "forecast_steps": 30,
            "generated_state_shape": [2, 16, 30, 39],
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": -22.5,
                    "p10": -80.0,
                    "p50": -20.0,
                    "p90": 15.0,
                },
                {
                    "market": "VIX",
                    "mean_terminal_delta": 4.0,
                    "p10": -1.0,
                    "p50": 3.5,
                    "p90": 11.0,
                },
            ],
        }
    }

    summary = reverse_caption.summarize_report_terminal_delta(report)
    by_factor = {row["factor"]: row for row in summary["factor_rows"]}

    assert summary["summary_source"] == "report_terminal_delta_summary"
    assert summary["sample_count"] == 16
    assert by_factor["SPX"]["direction"] == "down"
    assert by_factor["SPX"]["terminal_mean_delta"] == -22.5
    assert by_factor["VIX"]["direction"] == "up"


def test_discover_review_cases_handles_fixed_start_analysis(tmp_path: Path) -> None:
    report = tmp_path / "case" / "prefix_report_snapshot.json"
    arrays = tmp_path / "case" / "prefix_arrays_snapshot.npz"
    report.parent.mkdir()
    report.write_text(
        json.dumps(
            {
                "condition_only_case": {"story": "Source story"},
                "cached_query": {"grounding": {"cleaned_conditioning_text": "Clean story"}},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(arrays, generated_states=np.zeros((2, 30, 39), dtype=np.float32))
    analysis = tmp_path / "fixed_start_live_story_deck_analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "source_case",
                        "report_snapshot": str(report),
                        "arrays_snapshot": str(arrays),
                        "market_implications": [{"market": "SPX", "direction": "up"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = reverse_caption.discover_review_cases(analysis)

    assert len(cases) == 1
    assert cases[0]["case_name"] == "source_case"
    assert cases[0]["source_story"] == "Source story"
    assert cases[0]["cleaned_conditioning_text"] == "Clean story"
    assert cases[0]["arrays_path"] == str(arrays)


def test_build_reverse_caption_prompt_contains_source_and_deck_summary() -> None:
    prompt = reverse_caption.build_reverse_caption_prompt(
        case={
            "case_name": "dollar_liquidity",
            "source_story": "Dollar strengthens and credit spreads widen.",
            "cleaned_conditioning_text": "Dollar up; credit wider.",
            "market_implications": [{"market": "DXY", "direction": "up"}],
        },
        deck_summary={
            "factor_rows": [
                {
                    "factor": "DXY",
                    "direction": "up",
                    "terminal_mean_delta": 1.2,
                    "terminal_p10_delta": 0.4,
                    "terminal_p90_delta": 2.2,
                }
            ]
        },
    )

    assert "Dollar strengthens and credit spreads widen" in prompt
    assert "DXY" in prompt
    assert "Return only JSON" in prompt
    assert "Do not invent named real-world news events" in prompt


def test_run_reverse_caption_audit_treats_dry_run_as_preflight_pass(tmp_path: Path) -> None:
    report = tmp_path / "case" / "prefix_report_snapshot.json"
    arrays = tmp_path / "case" / "prefix_arrays_snapshot.npz"
    report.parent.mkdir()
    report.write_text(
        json.dumps({"condition_only_case": {"story": "Source story"}}),
        encoding="utf-8",
    )
    np.savez_compressed(arrays, generated_states=np.zeros((2, 30, 39), dtype=np.float32))
    analysis = tmp_path / "fixed_start_live_story_deck_analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "source_case",
                        "report_snapshot": str(report),
                        "arrays_snapshot": str(arrays),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = reverse_caption.run_reverse_caption_audit(
        SimpleNamespace(
            input=analysis,
            output_dir=tmp_path / "out",
            max_cases=1,
            model="unused",
            reasoning_effort="unused",
            timeout_seconds=1,
            dry_run=True,
        )
    )

    assert audit["status"] == "pass"
    assert audit["dry_run"] is True
    assert audit["cases"][0]["errors"][0]["code"] == "dry_run"


def test_run_reverse_caption_audit_prefers_report_summary_over_snapshot_arrays(
    tmp_path: Path,
) -> None:
    report = tmp_path / "case" / "prefix_report_snapshot.json"
    arrays = tmp_path / "case" / "prefix_arrays_snapshot.npz"
    report.parent.mkdir()
    report.write_text(
        json.dumps(
            {
                "condition_only_case": {"story": "Risk-off source story"},
                "generation": {
                    "sample_count": 16,
                    "forecast_steps": 30,
                    "terminal_delta_summary": [
                        {
                            "market": "SPX",
                            "mean_terminal_delta": -20.0,
                            "p10": -50.0,
                            "p90": 5.0,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    states = np.zeros((1, 2, 30, 39), dtype=np.float32)
    requested_raw = np.zeros((1, 39), dtype=np.float32)
    states[0, :, -1, reverse_caption.FACTOR_INDEX["SPX"]] = 100.0
    np.savez_compressed(arrays, generated_states=states, requested_raw=requested_raw)
    analysis = tmp_path / "fixed_start_live_story_deck_analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "source_case",
                        "report_snapshot": str(report),
                        "arrays_snapshot": str(arrays),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = reverse_caption.run_reverse_caption_audit(
        SimpleNamespace(
            input=analysis,
            output_dir=tmp_path / "out",
            max_cases=1,
            model="unused",
            reasoning_effort="unused",
            timeout_seconds=1,
            dry_run=True,
        )
    )
    by_factor = {row["factor"]: row for row in audit["cases"][0]["deck_summary"]["factor_rows"]}

    assert audit["cases"][0]["deck_summary"]["summary_source"] == "report_terminal_delta_summary"
    assert by_factor["SPX"]["direction"] == "down"
    assert by_factor["SPX"]["terminal_mean_delta"] == -20.0
