import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_bridge_failure_diagnostics import (
    build_bridge_failure_diagnostics,
    classify_bridge_failure_modes,
    render_bridge_diagnostics_markdown,
)


def _casebook() -> dict:
    return {
        "cases": [
            {
                "window_id": "mixed_fail",
                "regime_tags": ["volatility_shock"],
                "input_narrative": "Equities rally while volatility is still bid.",
                "market_implications": [
                    {"market": "SPX", "direction": "up", "magnitude": "large"},
                    {"market": "VIX", "direction": "up", "magnitude": "large"},
                    {"market": "GOLD", "direction": "up", "magnitude": "large"},
                ],
                "grounding": {
                    "validation_errors": [],
                    "validation_warnings": [],
                },
                "bridge": {
                    "target_cosine": 0.70,
                    "hard_negative_gap": 0.90,
                    "true_rank_test_pool": 28,
                },
                "historical_analogues": [
                    {
                        "window_id": "train_a",
                        "similarity": 0.94,
                        "primary_narrative": "Analogue A",
                    }
                ],
            },
            {
                "window_id": "label_borderline",
                "regime_tags": ["risk_on_recovery"],
                "input_narrative": "Risk-on with a policy analogy.",
                "market_implications": [
                    {"market": "SPX", "direction": "up", "magnitude": "large"},
                    {"market": "VIX", "direction": "down", "magnitude": "large"},
                ],
                "grounding": {
                    "validation_errors": [],
                    "validation_warnings": [
                        {"code": "external_catalyst_requires_grounding"}
                    ],
                },
                "bridge": {
                    "target_cosine": 0.82,
                    "hard_negative_gap": 0.75,
                    "true_rank_test_pool": 21,
                },
                "historical_analogues": [
                    {
                        "window_id": "train_b",
                        "similarity": 0.90,
                        "primary_narrative": "Analogue B",
                    }
                ],
            },
            {
                "window_id": "label_only",
                "regime_tags": ["risk_on_recovery"],
                "input_narrative": "Label-only issue.",
                "market_implications": [
                    {"market": "SPX", "direction": "up", "magnitude": "large"}
                ],
                "grounding": {
                    "validation_errors": [],
                    "validation_warnings": [{"code": "warning"}],
                },
                "bridge": {
                    "target_cosine": 0.90,
                    "hard_negative_gap": 0.80,
                    "true_rank_test_pool": 3,
                },
                "historical_analogues": [
                    {
                        "window_id": "train_c",
                        "similarity": 0.92,
                        "primary_narrative": "Analogue C",
                    }
                ],
            },
        ]
    }


def _acceptance_audit() -> dict:
    return {
        "case_results": [
            {
                "window_id": "mixed_fail",
                "status": "fail",
                "failure_codes": ["weak_bridge_alignment"],
                "warning_codes": [],
                "bottleneck_tags": ["bridge_alignment"],
            },
            {
                "window_id": "label_borderline",
                "status": "warning",
                "failure_codes": [],
                "warning_codes": [
                    "borderline_bridge_alignment",
                    "label_validation_warning",
                ],
                "bottleneck_tags": ["bridge_alignment", "label_quality"],
            },
            {
                "window_id": "label_only",
                "status": "warning",
                "failure_codes": [],
                "warning_codes": ["label_validation_warning"],
                "bottleneck_tags": ["label_quality"],
            },
        ]
    }


def _bridge_report() -> dict:
    return {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_id": "mixed_fail",
                    "role": "anchor",
                    "kind": "revised_market_description",
                    "target_cosine": 0.70,
                    "target_mse": 0.5,
                    "true_rank_test_pool": 28,
                    "true_rank_full_pool": 180,
                    "top_train_pool": [
                        {"window_id": "train_a", "cosine": 0.94},
                    ],
                    "top_test_pool": [
                        {"window_id": "other_test", "cosine": 0.93},
                    ],
                },
                {
                    "window_id": "label_borderline",
                    "role": "anchor",
                    "kind": "revised_market_description",
                    "target_cosine": 0.82,
                    "target_mse": 0.3,
                    "true_rank_test_pool": 21,
                    "true_rank_full_pool": 120,
                    "top_train_pool": [
                        {"window_id": "train_b", "cosine": 0.90},
                    ],
                    "top_test_pool": [
                        {"window_id": "other_test_2", "cosine": 0.88},
                    ],
                },
            ],
            "hard_negative_separation": {
                "windows": [
                    {
                        "window_id": "mixed_fail",
                        "negative_gap": 0.90,
                        "positive_mean_cosine": 0.95,
                        "negative_mean_cosine": 0.05,
                    },
                    {
                        "window_id": "label_borderline",
                        "negative_gap": 0.75,
                        "positive_mean_cosine": 0.93,
                        "negative_mean_cosine": 0.18,
                    },
                ]
            },
        }
    }


def _pipeline_report() -> dict:
    return {
        "narrative_bundles": [
            {
                "window_id": "mixed_fail",
                "narratives": [
                    {"observed_fact_tokens": "SPX: UP LARGE; VIX: UP LARGE"}
                ],
                "contrastive_narratives": [{"kind": "opposite"}, {"kind": "partial"}],
            },
            {
                "window_id": "label_borderline",
                "narratives": [
                    {"observed_fact_tokens": "SPX: UP LARGE; VIX: DOWN LARGE"}
                ],
                "contrastive_narratives": [{"kind": "opposite"}],
            },
        ]
    }


def test_classify_bridge_failure_modes_identifies_mixed_regime_hard_case() -> None:
    modes = classify_bridge_failure_modes(
        case=_casebook()["cases"][0],
        audit_row=_acceptance_audit()["case_results"][0],
        bridge_row=_bridge_report()["evaluation"]["heldout_examples"][0],
        hard_negative_row=_bridge_report()["evaluation"]["hard_negative_separation"][
            "windows"
        ][0],
        bundle=_pipeline_report()["narrative_bundles"][0],
    )

    mode_codes = {mode["code"] for mode in modes}
    assert "weak_target_alignment" in mode_codes
    assert "mixed_regime_semantic_ambiguity" in mode_codes
    assert "dense_neighbor_rank_metric_strictness" in mode_codes


def test_build_bridge_failure_diagnostics_selects_only_bridge_bottleneck_cases() -> (
    None
):
    diagnostics = build_bridge_failure_diagnostics(
        _casebook(),
        _acceptance_audit(),
        _bridge_report(),
        _pipeline_report(),
        title="Bridge diagnostics",
    )

    assert diagnostics["summary"]["bridge_problem_case_count"] == 2
    assert diagnostics["summary"]["bridge_failure_count"] == 1
    assert diagnostics["summary"]["bridge_warning_count"] == 1
    assert diagnostics["summary"]["failure_mode_counts"]["label_quality_review"] == 1
    assert (
        diagnostics["summary"]["recommended_action_counts"][
            "repair_or_regenerate_grounded_labels"
        ]
        == 1
    )
    assert [case["window_id"] for case in diagnostics["cases"]] == [
        "mixed_fail",
        "label_borderline",
    ]


def test_render_bridge_diagnostics_markdown_contains_actions_and_cases() -> None:
    markdown = render_bridge_diagnostics_markdown(
        build_bridge_failure_diagnostics(
            _casebook(),
            _acceptance_audit(),
            _bridge_report(),
            _pipeline_report(),
        )
    )

    assert "# Bridge Failure Diagnostics" in markdown
    assert "## Recommended Next Step" in markdown
    assert "mixed_fail" in markdown
    assert "dense_neighbor_rank_metric_strictness" in markdown
