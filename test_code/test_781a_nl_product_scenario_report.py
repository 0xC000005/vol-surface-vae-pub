import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_product_scenario_report import (
    build_product_report_package,
    render_product_report_markdown,
)


def _casebook() -> dict:
    return {
        "title": "Casebook",
        "summary": {"case_count": 1},
        "cases": [
            {
                "window_id": "w1",
                "regime_tags": ["risk_off_stress"],
                "input_narrative": "Risk-off stress with equities lower and volatility higher.",
                "market_implications": [
                    {"market": "SPX", "direction": "down", "magnitude": "large"},
                    {"market": "VIX", "direction": "up", "magnitude": "large"},
                ],
                "grounding": {
                    "external_news_used": False,
                    "validation_errors": [],
                    "validation_warnings": [],
                    "non_observed_catalysts": [
                        {
                            "label": "COVID-style panic",
                            "grounding_status": "historical_analogy",
                            "description": "Analogy only, not a confirmed cause.",
                        }
                    ],
                },
                "bridge": {
                    "target_cosine": 0.91,
                    "hard_negative_gap": 0.8,
                    "true_rank_test_pool": 4,
                },
                "historical_analogues": [
                    {
                        "rank": 1,
                        "window_id": "a1",
                        "similarity": 0.88,
                        "forecast_start_date": "2020-01-01",
                        "forecast_end_date": "2020-02-12",
                        "primary_narrative": "Prior risk-off analogue.",
                    }
                ],
                "scores": {
                    "narrative_generator_topk": {
                        "energy_score_improvement_vs_persistence": 0.2,
                        "ensemble_crps_improvement_vs_persistence": 0.1,
                        "coverage_80": 0.65,
                        "mean_path_mae_improvement_vs_persistence": -0.1,
                    },
                    "persistence": {"energy_score_z": 1.0},
                },
                "failure_flags": ["distribution_good_point_path_weak"],
            }
        ],
    }


def _acceptance_audit() -> dict:
    return {
        "case_results": [
            {
                "window_id": "w1",
                "status": "warning",
                "warning_codes": ["point_path_lags_persistence"],
                "failure_codes": [],
                "bottleneck_tags": ["product_framing"],
            }
        ]
    }


def test_build_product_report_package_declares_distributional_contract() -> None:
    package = build_product_report_package(_casebook(), _acceptance_audit())

    assert (
        package["product_contract"]["positioning"]
        == "scenario_distribution_not_point_forecast"
    )
    assert package["summary"]["case_count"] == 1
    assert package["summary"]["distributional_framing_ready_count"] == 1
    report = package["case_reports"][0]
    assert report["window_id"] == "w1"
    assert report["acceptance_status"] == "warning"
    assert report["framing_checks"] == {
        "distribution_metrics_present": True,
        "grounding_section_present": True,
        "historical_analogues_present": True,
        "not_point_forecast_statement": True,
    }
    assert "not a point forecast" in report["sections"]["positioning"].lower()


def test_render_product_report_markdown_is_risk_manager_facing() -> None:
    markdown = render_product_report_markdown(
        build_product_report_package(_casebook(), _acceptance_audit())
    )

    assert "# Product Scenario Reports" in markdown
    assert "Scenario Distribution, Not Point Forecast" in markdown
    assert "## Case 1: w1" in markdown
    assert "Grounding and Hallucination Controls" in markdown
    assert "Historical Analogues" in markdown
    assert "Distribution Metrics" in markdown
