import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_product_acceptance_casebook import (
    extract_case_diagnostics,
    render_markdown,
    summarize_casebook,
)


def test_summarize_casebook_aggregates_pass_fail_counts() -> None:
    summary = summarize_casebook(
        [
            {
                "name": "fragile",
                "status": "pass",
                "candidate_index": 18,
                "expected_operational_status": "pass",
                "checks": [{"name": "a", "passed": True}],
                "diagnostics": {
                    "validation_overall": "pass",
                    "validation_operational": "pass",
                    "support_candidate_count": 8,
                    "start_distance_z": 0.0,
                    "preview_field_count": 39,
                    "fan_count": 45,
                },
                "run_report": "fragile/run.json",
                "exported_start_json": "fragile/start.json",
            },
            {
                "name": "defensive",
                "status": "fail",
                "candidate_index": 22,
                "expected_operational_status": "warning",
                "checks": [{"name": "b", "passed": False}],
                "diagnostics": {
                    "validation_overall": "fail",
                    "validation_operational": "fail",
                },
                "run_report": "defensive/run.json",
                "exported_start_json": "defensive/start.json",
            },
        ]
    )

    assert summary["overall_status"] == "fail"
    assert summary["case_count"] == 2
    assert summary["pass_count"] == 1
    assert summary["fail_count"] == 1
    assert summary["expectation_fail_count"] == 1
    assert summary["cases"][1]["failed_checks"] == ["b"]
    assert summary["cases"][0]["expectation_met"] is True
    assert summary["cases"][1]["expectation_met"] is False
    assert summary["cases"][0]["support_candidate_count"] == 8
    assert summary["cases"][0]["preview_field_count"] == 39


def test_summarize_casebook_fails_when_expected_warning_is_missing() -> None:
    summary = summarize_casebook(
        [
            {
                "name": "defensive",
                "status": "pass",
                "candidate_index": 22,
                "expected_operational_status": "warning",
                "checks": [{"name": "b", "passed": True}],
                "diagnostics": {
                    "validation_overall": "pass",
                    "validation_operational": "pass",
                },
                "run_report": "defensive/run.json",
                "exported_start_json": "defensive/start.json",
            },
        ]
    )

    assert summary["overall_status"] == "fail"
    assert summary["expectation_fail_count"] == 1
    assert summary["cases"][0]["expectation_met"] is False


def test_render_markdown_lists_casebook_rows() -> None:
    summary = {
        "overall_status": "pass",
        "case_count": 1,
        "pass_count": 1,
        "fail_count": 0,
        "expectation_fail_count": 0,
        "cases": [
            {
                "name": "fragile",
                "status": "pass",
                "candidate_index": 18,
                "expected_operational_status": "pass",
                "expectation_met": True,
                "failed_checks": [],
                "validation_overall": "pass",
                "validation_operational": "pass",
                "support_candidate_count": 8,
                "start_distance_z": 0.0,
                "preview_field_count": 39,
                "fan_count": 45,
            }
        ],
    }

    text = render_markdown(summary)

    assert "Product Acceptance Casebook" in text
    assert "`fragile`" in text
    assert "`pass/ok`" in text
    assert "`pass/pass`" in text


def test_extract_case_diagnostics_reads_run_report_payload() -> None:
    smoke_summary = {
        "preview_rows": [{"Field": "Field count", "Value": "39"}],
    }
    run_report = {
        "validation_gate": {"overall_status": "pass", "operational_status": "pass"},
        "cached_query": {"memory_prior": {"candidate_details": [{}, {}]}},
        "generation": {"path_quantiles": [{}, {}, {}]},
        "variant_rows": [{"variant": "user_start_state", "start_distance_z": 1.25}],
    }

    diagnostics = extract_case_diagnostics(
        smoke_summary=smoke_summary,
        run_report=run_report,
    )

    assert diagnostics["validation_overall"] == "pass"
    assert diagnostics["support_candidate_count"] == 2
    assert diagnostics["start_distance_z"] == 1.25
    assert diagnostics["preview_field_count"] == 39
    assert diagnostics["fan_count"] == 3
