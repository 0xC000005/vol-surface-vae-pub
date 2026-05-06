import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_risk_manager_casebook import (
    build_casebook,
    classify_regime_tags,
    metric_improvement_vs_baseline,
    render_casebook_markdown,
)


def _pipeline_report() -> dict:
    return {
        "label_backend": "openai",
        "embedding_backend": "openai",
        "selection_manifest": "manifest.json",
        "train_windows": 2,
        "rejected_label_windows": [],
        "narrative_bundles": [
            {
                "window_id": "train_a",
                "window_index": 0,
                "source_index": 100,
                "window_metadata": {
                    "manifest_split": "train",
                    "forecast_start_date": "2020-01-01",
                    "forecast_end_date": "2020-02-12",
                },
                "narratives": [{"text": "Prior risk-off analogue."}],
                "market_implications": [
                    {"market": "SPX", "direction": "down", "magnitude": "small"}
                ],
            },
            {
                "window_id": "test_a",
                "window_index": 2,
                "source_index": 200,
                "window_metadata": {
                    "manifest_split": "test",
                    "calendar_start_date": "2020-03-01",
                    "calendar_end_date": "2020-04-10",
                    "forecast_start_date": "2020-04-13",
                    "forecast_end_date": "2020-05-22",
                    "selection_reasons": ["weekly_anchor", "eventful"],
                },
                "narratives": [
                    {
                        "text": "COVID-style panic analogy with equities down and volatility up.",
                        "grounding_status": "market_fact_supported",
                        "observed_fact_tokens": "SPX: DOWN LARGE; VIX: UP LARGE; BBB_OAS: WIDER MEDIUM",
                    }
                ],
                "narrative_catalysts": [
                    {
                        "label": "COVID-style panic",
                        "grounding_status": "historical_analogy",
                        "description": "An analogy, not an asserted cause.",
                    }
                ],
                "hallucination_audit": {"validation_issues": []},
                "market_implications": [
                    {"market": "SPX", "direction": "down", "magnitude": "large"},
                    {"market": "VIX", "direction": "up", "magnitude": "large"},
                    {"market": "BBB_OAS", "direction": "wider", "magnitude": "medium"},
                ],
            },
        ],
    }


def _bridge_report() -> dict:
    return {
        "summary": {
            "heldout_mean_target_cosine": 0.86,
            "heldout_hard_negative_mean_gap": 0.88,
        },
        "evaluation": {
            "heldout_examples": [
                {
                    "window_id": "test_a",
                    "window_index": 1,
                    "role": "anchor",
                    "kind": "revised_market_description",
                    "target_cosine": 0.91,
                    "true_rank_test_pool": 3,
                    "top_train_pool": [
                        {"window_id": "train_a", "window_index": 0, "cosine": 0.84}
                    ],
                }
            ],
            "hard_negative_separation": {
                "windows": [
                    {
                        "window_id": "test_a",
                        "window_index": 1,
                        "positive_mean_cosine": 0.95,
                        "negative_mean_cosine": 0.1,
                        "negative_gap": 0.85,
                    }
                ]
            },
        },
    }


def _scenario_report() -> dict:
    return {
        "summary": {
            "narrative_generator_topk": {
                "energy_score_z_improvement_vs_persistence": 0.2,
                "ensemble_crps_z_improvement_vs_persistence": 0.1,
            }
        },
        "window_scores": [
            {
                "window_id": "test_a",
                "window_index": 1,
                "block_window_index": 2,
                "top_train_window_ids": ["train_a"],
                "top_train_cosines": [0.84],
                "methods": {
                    "persistence": {
                        "energy_score_z": 1.0,
                        "ensemble_crps_z": 0.8,
                        "mean_path_mae_z": 0.7,
                    },
                    "narrative_generator_topk": {
                        "energy_score_z": 0.75,
                        "ensemble_crps_z": 0.6,
                        "coverage_80": 0.65,
                        "mean_path_mae_z": 0.9,
                    },
                    "historical_replay_topk": {
                        "energy_score_z": 0.9,
                        "ensemble_crps_z": 0.75,
                    },
                },
            }
        ],
    }


def test_classify_regime_tags_uses_observed_market_implications() -> None:
    tags = classify_regime_tags(
        [
            {"market": "SPX", "direction": "down", "magnitude": "large"},
            {"market": "VIX", "direction": "up", "magnitude": "large"},
            {"market": "BBB_OAS", "direction": "wider", "magnitude": "medium"},
            {"market": "US10Y", "direction": "up", "magnitude": "large"},
        ]
    )

    assert "risk_off_stress" in tags
    assert "rates_shock" in tags
    assert "volatility_shock" in tags


def test_metric_improvement_vs_baseline_treats_lower_scores_as_better() -> None:
    assert metric_improvement_vs_baseline(0.75, 1.0) == 0.25
    assert metric_improvement_vs_baseline(1.25, 1.0) == -0.25
    assert metric_improvement_vs_baseline(None, 1.0) is None
    assert metric_improvement_vs_baseline(1.0, 0.0) is None


def test_build_casebook_combines_narratives_grounding_analogues_and_scores() -> None:
    casebook = build_casebook(
        _pipeline_report(),
        _bridge_report(),
        _scenario_report(),
        title="Representative run",
    )

    assert casebook["summary"]["case_count"] == 1
    assert casebook["summary"]["regime_counts"] == {
        "risk_off_stress": 1,
        "volatility_shock": 1,
    }
    case = casebook["cases"][0]
    assert case["window_id"] == "test_a"
    assert case["input_narrative"].startswith("COVID-style panic")
    assert (
        case["grounding"]["non_observed_catalysts"][0]["label"] == "COVID-style panic"
    )
    assert case["historical_analogues"][0]["window_id"] == "train_a"
    assert (
        case["historical_analogues"][0]["primary_narrative"]
        == "Prior risk-off analogue."
    )
    assert (
        case["scores"]["narrative_generator_topk"][
            "energy_score_improvement_vs_persistence"
        ]
        == 0.25
    )
    assert case["failure_flags"] == ["distribution_good_point_path_weak"]


def test_render_casebook_markdown_contains_product_facing_sections() -> None:
    markdown = render_casebook_markdown(
        build_casebook(_pipeline_report(), _bridge_report(), _scenario_report())
    )

    assert "# Risk Manager Narrative Scenario Casebook" in markdown
    assert "## Executive Summary" in markdown
    assert "## Case 1: test_a" in markdown
    assert "Input Narrative" in markdown
    assert "Historical Analogues" in markdown
    assert "Failure Flags" in markdown
