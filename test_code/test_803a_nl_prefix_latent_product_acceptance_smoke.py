import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_product_acceptance_smoke import (
    acceptance_checks,
    acceptance_status,
    render_markdown,
)


def _report() -> dict:
    return {
        "cached_query": {
            "memory_prior_contract": "per_variant_narrative_and_fixed_start",
            "memory_prior": {
                "query_start_source": "provided_start_state",
                "candidate_details": [{"window_id": "joint39_val_0036", "weight": 0.5}],
            },
        },
        "variant_rows": [
            {
                "variant": "user_start_state",
                "start_window_id": "today",
                "memory_prior_analogue_count": 8,
                "memory_prior_query_start_source": "provided_start_state",
            }
        ],
        "validation_gate": {
            "overall_status": "pass",
            "operational_status": "pass",
        },
        "generation": {
            "path_quantiles": [
                {"market": "SPX", "analogue_key": "ALL"},
                {"market": "IV_ATM_3M", "analogue_key": "ALL"},
            ]
        },
        "artifact_paths": {"report": "run.json", "arrays": "run.npz"},
    }


def test_acceptance_checks_pass_for_complete_product_report() -> None:
    preview_rows = [{"Field": "Field count", "Value": "39"}]

    checks = acceptance_checks(report=_report(), preview_rows=preview_rows)

    assert acceptance_status(checks) == "pass"
    assert {row["name"] for row in checks} >= {
        "validation_gate_not_fail",
        "user_start_variant_present",
        "support_candidates_present",
        "fixed_start_memory_prior_contract",
        "user_start_mixture_diagnostics_present",
        "start_preview_present",
        "spx_fan_available",
        "selected_iv_cell_fan_available",
    }


def test_acceptance_checks_fail_without_iv_fan() -> None:
    report = _report()
    report["generation"]["path_quantiles"] = [{"market": "SPX", "analogue_key": "ALL"}]

    checks = acceptance_checks(
        report=report,
        preview_rows=[{"Field": "Field count", "Value": "39"}],
    )

    assert acceptance_status(checks) == "fail"
    assert any(
        row["name"] == "selected_iv_cell_fan_available" and not row["passed"]
        for row in checks
    )


def test_render_markdown_lists_acceptance_checks() -> None:
    summary = {
        "status": "pass",
        "candidate_index": 18,
        "exported_start_json": "start.json",
        "run_report": "run.json",
        "checks": [
            {"name": "validation_gate_not_fail", "passed": True, "detail": "pass"}
        ],
    }

    text = render_markdown(summary)

    assert "Prefix-Latent Product Acceptance Smoke" in text
    assert "`validation_gate_not_fail`" in text
