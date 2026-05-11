import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_conditioning_analysis import (
    build_conditioning_analysis,
    render_markdown,
)


def _bakeoff_report() -> dict:
    rows = []
    for start_name, energy, crps, status in [
        ("fixed_start_1", 0.10, 0.08, "pass"),
        ("fixed_start_1", 0.12, 0.09, "pass"),
        ("fixed_start_2", 0.05, 0.04, "warning"),
        ("fixed_start_2", 0.07, 0.06, "warning"),
    ]:
        rows.append(
            {
                "start_name": start_name,
                "case_name": f"case_{len(rows)}",
                "validation_operational": status,
                "memory_prior_direction_status": "pass",
                "memory_prior_support_weighted_match_rate": 1.0,
                "scenario_metrics": {
                    "energy_score_z_improvement_vs_persistence": energy,
                    "ensemble_crps_z_improvement_vs_persistence": crps,
                },
            }
        )
    return {
        "status": "pass",
        "rows": rows,
        "variant_summary": [
            {
                "mean_energy_improvement_vs_persistence": 0.085,
                "mean_crps_improvement_vs_persistence": 0.0675,
                "operational_status_counts": {"pass": 2, "warning": 2},
                "direction_status_counts": {"pass": 4},
            }
        ],
    }


def _contrast_report() -> dict:
    return {
        "case_count": 4,
        "start_blocks": [
            {"start_name": "fixed_start_1", "case_count": 2, "start_max_abs_diff": 0.0},
            {"start_name": "fixed_start_2", "case_count": 2, "start_max_abs_diff": 0.0},
        ],
        "pairwise_contrasts": [
            {
                "start_name": "fixed_start_1",
                "left_case": "risk_on",
                "right_case": "risk_off",
                "standardized_l2_gap": 2.1,
                "largest_abs_market_gaps": [
                    {"market": "SPX", "standardized_mean_gap": 1.5}
                ],
            },
            {
                "start_name": "fixed_start_2",
                "left_case": "risk_on",
                "right_case": "risk_off",
                "standardized_l2_gap": 0.8,
                "largest_abs_market_gaps": [
                    {"market": "VIX", "standardized_mean_gap": -0.7}
                ],
            },
        ],
    }


def _gate_report() -> dict:
    return {
        "overall_status": "warning",
        "hard_fail_count": 0,
        "warning_count": 2,
        "start_block_warning_count": 1,
        "total_warning_count": 3,
        "start_block_assessments": [
            {
                "start_name": "fixed_start_1",
                "status": "pass",
                "max_standardized_l2_gap": 2.1,
                "median_standardized_l2_gap": 2.1,
                "min_standardized_l2_gap": 2.1,
                "top_separation_markets": ["SPX"],
                "warnings": [],
                "failures": [],
            },
            {
                "start_name": "fixed_start_2",
                "status": "warning",
                "max_standardized_l2_gap": 0.8,
                "median_standardized_l2_gap": 0.8,
                "min_standardized_l2_gap": 0.8,
                "top_separation_markets": ["VIX"],
                "warnings": ["start_dampens_narrative_influence"],
                "failures": [],
            },
        ],
        "checks": [
            {"name": "fixed_start_equality", "status": "pass"},
            {"name": "operational_validation_observation", "status": "warning"},
        ],
    }


def test_build_conditioning_analysis_combines_gate_and_quality_metrics() -> None:
    analysis = build_conditioning_analysis(
        bakeoff_report=_bakeoff_report(),
        contrast_report=_contrast_report(),
        gate_report=_gate_report(),
    )

    assert analysis["overall_status"] == "warning"
    assert analysis["headline"]["start_count"] == 2
    assert analysis["headline"]["damped_start_count"] == 1
    start_rows = {row["start_name"]: row for row in analysis["start_summaries"]}
    assert start_rows["fixed_start_1"]["status"] == "pass"
    assert start_rows["fixed_start_2"]["status"] == "warning"
    assert start_rows["fixed_start_1"]["mean_energy_improvement_vs_persistence"] == 0.11
    assert analysis["top_pairwise_contrasts"][0]["start_name"] == "fixed_start_1"


def test_render_markdown_explains_fixed_start_conditionality() -> None:
    analysis = build_conditioning_analysis(
        bakeoff_report=_bakeoff_report(),
        contrast_report=_contrast_report(),
        gate_report=_gate_report(),
    )

    text = render_markdown(analysis)

    assert "same starting level" in text
    assert "fixed_start_2" in text
    assert "start_dampens_narrative_influence" in text
