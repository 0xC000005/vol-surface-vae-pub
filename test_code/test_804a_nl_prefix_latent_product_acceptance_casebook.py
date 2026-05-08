import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_product_acceptance_casebook import (
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
                "checks": [{"name": "a", "passed": True}],
                "run_report": "fragile/run.json",
                "exported_start_json": "fragile/start.json",
            },
            {
                "name": "defensive",
                "status": "fail",
                "candidate_index": 22,
                "checks": [{"name": "b", "passed": False}],
                "run_report": "defensive/run.json",
                "exported_start_json": "defensive/start.json",
            },
        ]
    )

    assert summary["overall_status"] == "fail"
    assert summary["case_count"] == 2
    assert summary["pass_count"] == 1
    assert summary["fail_count"] == 1
    assert summary["cases"][1]["failed_checks"] == ["b"]


def test_render_markdown_lists_casebook_rows() -> None:
    summary = {
        "overall_status": "pass",
        "case_count": 1,
        "pass_count": 1,
        "fail_count": 0,
        "cases": [
            {
                "name": "fragile",
                "status": "pass",
                "candidate_index": 18,
                "failed_checks": [],
            }
        ],
    }

    text = render_markdown(summary)

    assert "Product Acceptance Casebook" in text
    assert "`fragile`" in text
