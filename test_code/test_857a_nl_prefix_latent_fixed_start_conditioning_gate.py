import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_conditioning_gate import (
    evaluate_fixed_start_conditioning_gate,
)


def _contrast_report(*, start_diff: float = 0.0) -> dict:
    return {
        "status": "pass" if start_diff == 0.0 else "fail",
        "case_count": 6,
        "start_max_abs_diff": start_diff,
        "start_blocks": [
            {"start_name": "start_a", "case_count": 3, "start_max_abs_diff": 0.0},
            {
                "start_name": "start_b",
                "case_count": 3,
                "start_max_abs_diff": start_diff,
            },
        ],
        "case_summaries": [
            {
                "case_name": "risk_on_a",
                "start_name": "start_a",
                "direction_status": "pass",
                "support_match_rate": 1.0,
                "final_mixture_mismatch_count": 0,
            },
            {
                "case_name": "risk_off_a",
                "start_name": "start_a",
                "direction_status": "pass",
                "support_match_rate": 1.0,
                "final_mixture_mismatch_count": 0,
            },
            {
                "case_name": "risk_on_b",
                "start_name": "start_b",
                "direction_status": "pass",
                "support_match_rate": 1.0,
                "final_mixture_mismatch_count": 0,
            },
            {
                "case_name": "risk_off_b",
                "start_name": "start_b",
                "direction_status": "pass",
                "support_match_rate": 1.0,
                "final_mixture_mismatch_count": 0,
            },
        ],
        "pairwise_contrasts": [
            {
                "start_name": "start_a",
                "standardized_l2_gap": 2.0,
                "largest_abs_market_gaps": [{"market": "SPX"}],
            },
            {
                "start_name": "start_b",
                "standardized_l2_gap": 0.9,
                "largest_abs_market_gaps": [{"market": "VIX"}],
            },
        ],
    }


def _bakeoff_report(*, energy: float = 0.1, crps: float = 0.05) -> dict:
    return {
        "status": "pass",
        "variant_summary": [
            {
                "variant_name": "decoder_soft_topk_narrative_start_checked_gen_temp_0p50",
                "run_count": 6,
                "mean_energy_improvement_vs_persistence": energy,
                "mean_crps_improvement_vs_persistence": crps,
                "mean_support_weighted_match_rate": 1.0,
                "total_final_mixture_mismatches": 0,
                "operational_status_counts": {"pass": 6},
                "direction_status_counts": {"pass": 6},
            }
        ],
    }


def test_gate_passes_with_warning_for_damped_start_block() -> None:
    gate = evaluate_fixed_start_conditioning_gate(
        contrast_report=_contrast_report(),
        bakeoff_report=_bakeoff_report(),
        damping_ratio=0.75,
    )

    assert gate["overall_status"] == "warning"
    assert gate["hard_fail_count"] == 0
    assert gate["warning_count"] == 1
    assert gate["start_block_warning_count"] == 1
    assert gate["total_warning_count"] == 2
    assert any(
        block["start_name"] == "start_b" and block["status"] == "warning"
        for block in gate["start_block_assessments"]
    )
    assert {check["name"]: check["status"] for check in gate["checks"]}[
        "scenario_quality_vs_persistence"
    ] == "pass"
    assert {check["name"]: check["status"] for check in gate["checks"]}[
        "operational_validation_observation"
    ] == "pass"


def test_gate_fails_when_start_is_not_fixed() -> None:
    gate = evaluate_fixed_start_conditioning_gate(
        contrast_report=_contrast_report(start_diff=0.25),
        bakeoff_report=_bakeoff_report(),
    )

    assert gate["overall_status"] == "fail"
    assert any(
        check["name"] == "fixed_start_equality" and check["status"] == "fail"
        for check in gate["checks"]
    )


def test_gate_fails_when_distribution_quality_is_worse_than_persistence() -> None:
    gate = evaluate_fixed_start_conditioning_gate(
        contrast_report=_contrast_report(),
        bakeoff_report=_bakeoff_report(energy=-0.01, crps=-0.02),
    )

    assert gate["overall_status"] == "fail"
    assert any(
        check["name"] == "scenario_quality_vs_persistence" and check["status"] == "fail"
        for check in gate["checks"]
    )


def test_gate_surfaces_operational_validation_warnings_without_support_failure() -> (
    None
):
    bakeoff = _bakeoff_report()
    bakeoff["variant_summary"][0]["operational_status_counts"] = {
        "pass": 5,
        "warning": 1,
    }

    gate = evaluate_fixed_start_conditioning_gate(
        contrast_report=_contrast_report(),
        bakeoff_report=bakeoff,
    )

    checks = {check["name"]: check for check in gate["checks"]}
    assert checks["support_direction_consistency"]["status"] == "pass"
    assert checks["operational_validation_observation"]["status"] == "warning"
    assert "warning" in checks["operational_validation_observation"]["detail"]
