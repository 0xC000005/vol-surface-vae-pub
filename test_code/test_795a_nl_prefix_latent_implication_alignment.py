import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_implication_alignment import (
    evaluate_alignment_inputs,
    run_alignment_evaluator,
    summarize_alignment_cases,
)


def test_evaluate_alignment_inputs_reads_prefix_report(tmp_path) -> None:
    report_path = tmp_path / "prefix_report.json"
    report_path.write_text(
        json.dumps(
            {
                "cached_query": {
                    "condition_source": "live_openai_story",
                    "kind": "live_story",
                    "grounding": {
                        "market_implications": [
                            {"market": "SPX", "direction": "up"},
                            {"market": "VIX", "direction": "down"},
                        ]
                    },
                },
                "validation_gate": {
                    "overall_status": "warning",
                    "selected_start_status": "pass",
                },
                "generation": {
                    "terminal_delta_summary": [
                        {"market": "SPX", "mean_terminal_delta": -12.0},
                        {"market": "VIX", "mean_terminal_delta": 1.5},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    cases = evaluate_alignment_inputs([report_path])

    assert len(cases) == 1
    assert cases[0]["source_kind"] == "prefix_story_smoke_report"
    assert cases[0]["condition_source"] == "live_openai_story"
    assert cases[0]["alignment"]["mismatch_count"] == 2


def test_evaluate_alignment_inputs_reads_casebook_summary(tmp_path) -> None:
    casebook_path = tmp_path / "casebook.json"
    casebook_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "risk_on",
                        "condition_source": "live_openai_story",
                        "overall_status": "warning",
                        "selected_start_status": "pass",
                        "market_alignment": {
                            "status": "warning",
                            "checked_count": 3,
                            "match_count": 2,
                            "mismatch_count": 1,
                            "skipped_count": 0,
                            "mismatches": [{"market": "SPX"}],
                            "skipped": [],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = evaluate_alignment_inputs([casebook_path])
    summary = summarize_alignment_cases(cases, output_dir=tmp_path)

    assert cases[0]["source_kind"] == "gradio_casebook_summary"
    assert summary["alignment_status_counts"] == {"warning": 1}
    assert summary["total_checked_implications"] == 3
    assert summary["total_mismatches"] == 1
    assert summary["mismatch_rate"] == 1 / 3


def test_run_alignment_evaluator_writes_summary(tmp_path) -> None:
    casebook_path = tmp_path / "casebook.json"
    casebook_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "risk_on",
                        "market_alignment": {
                            "status": "pass",
                            "checked_count": 2,
                            "match_count": 2,
                            "mismatch_count": 0,
                            "skipped_count": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = run_alignment_evaluator(
        SimpleNamespace(input=[str(casebook_path)], output_dir=str(tmp_path / "out"))
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 1
    assert summary["alignment_status_counts"] == {"pass": 1}
    assert (tmp_path / "out" / "implication_alignment_summary.json").exists()
