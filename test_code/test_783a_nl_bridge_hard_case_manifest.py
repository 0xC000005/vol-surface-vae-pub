import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_bridge_hard_case_manifest import (
    build_bridge_hard_case_manifest,
    render_bridge_hard_case_manifest_markdown,
)


def _diagnostics() -> dict:
    return {
        "title": "Bridge Failure Diagnostics",
        "summary": {
            "recommended_next_step": "Prioritize hard-case validation.",
        },
        "cases": [
            {
                "window_id": "weak_bridge",
                "acceptance_status": "fail",
                "regime_tags": ["risk_off_stress"],
                "input_narrative": "Equities sell off while volatility rises.",
                "observed_fact_tokens": "SPX: DOWN LARGE; VIX: UP LARGE",
                "bridge": {
                    "target_cosine": 0.70,
                    "target_mse": 0.51,
                    "true_rank_test_pool": 29,
                    "true_rank_full_pool": 144,
                    "hard_negative_gap": 0.82,
                },
                "failure_modes": [
                    {"code": "weak_target_alignment", "severity": "fail"},
                    {"code": "bridge_model_hard_case", "severity": "fail"},
                ],
                "recommended_actions": [
                    "targeted_bridge_retraining_or_bakeoff_candidate"
                ],
            },
            {
                "window_id": "mixed_borderline",
                "acceptance_status": "warning",
                "regime_tags": ["mixed_regime"],
                "input_narrative": "Equities and volatility rise together.",
                "observed_fact_tokens": "SPX: UP LARGE; VIX: UP LARGE",
                "bridge": {
                    "target_cosine": 0.82,
                    "target_mse": 0.31,
                    "true_rank_test_pool": 21,
                    "true_rank_full_pool": 101,
                    "hard_negative_gap": 0.77,
                },
                "failure_modes": [
                    {"code": "borderline_target_alignment", "severity": "warning"},
                    {
                        "code": "mixed_regime_semantic_ambiguity",
                        "severity": "warning",
                    },
                    {
                        "code": "dense_neighbor_rank_metric_strictness",
                        "severity": "warning",
                    },
                ],
                "recommended_actions": [
                    "add_to_bridge_hard_case_validation_set",
                    "add_mixed_regime_contrastive_bridge_examples",
                    "review_rank_metric_against_analogue_acceptance",
                ],
            },
            {
                "window_id": "label_borderline",
                "acceptance_status": "warning",
                "regime_tags": ["risk_on_recovery"],
                "input_narrative": "Risk-on recovery with a loose analogy.",
                "observed_fact_tokens": "SPX: UP LARGE; VIX: DOWN LARGE",
                "bridge": {
                    "target_cosine": 0.84,
                    "target_mse": 0.22,
                    "true_rank_test_pool": 18,
                    "true_rank_full_pool": 99,
                    "hard_negative_gap": 0.88,
                },
                "failure_modes": [
                    {"code": "borderline_target_alignment", "severity": "warning"},
                    {"code": "label_quality_review", "severity": "warning"},
                ],
                "recommended_actions": [
                    "add_to_bridge_hard_case_validation_set",
                    "repair_or_regenerate_grounded_labels",
                ],
            },
        ],
    }


def _casebook() -> dict:
    return {
        "cases": [
            {
                "window_id": "weak_bridge",
                "manifest_window_index": 10,
                "bridge_local_window_index": 2,
                "block_window_index": 302,
                "source_index": 902,
                "calendar": {"history_end": "2020-03-02"},
            },
            {
                "window_id": "mixed_borderline",
                "manifest_window_index": 11,
                "bridge_local_window_index": 3,
                "block_window_index": 303,
                "source_index": 903,
                "calendar": {"history_end": "2020-03-09"},
            },
            {
                "window_id": "label_borderline",
                "manifest_window_index": 12,
                "bridge_local_window_index": 4,
                "block_window_index": 304,
                "source_index": 904,
                "calendar": {"history_end": "2020-03-16"},
            },
            {
                "window_id": "not_a_bridge_problem",
                "manifest_window_index": 13,
                "bridge_local_window_index": 5,
                "block_window_index": 305,
                "source_index": 905,
            },
        ]
    }


def test_build_bridge_hard_case_manifest_preserves_identity_and_subsets() -> None:
    manifest = build_bridge_hard_case_manifest(
        _diagnostics(),
        _casebook(),
        title="Bridge Hard Cases",
    )

    assert manifest["summary"]["case_count"] == 3
    assert manifest["summary"]["subset_counts"] == {
        "bridge_hard_case_validation": 2,
        "bridge_model_hard_case": 1,
        "label_repair": 1,
        "mixed_regime_contrastive": 1,
        "rank_metric_review": 1,
    }
    assert manifest["subsets"]["bridge_hard_case_validation"] == [
        "mixed_borderline",
        "label_borderline",
    ]
    assert manifest["subsets"]["mixed_regime_contrastive"] == ["mixed_borderline"]
    assert manifest["subsets"]["rank_metric_review"] == ["mixed_borderline"]
    assert manifest["subsets"]["label_repair"] == ["label_borderline"]
    assert manifest["subsets"]["bridge_model_hard_case"] == ["weak_bridge"]

    by_window = {case["window_id"]: case for case in manifest["cases"]}
    mixed_case = by_window["mixed_borderline"]
    assert mixed_case["manifest_window_index"] == 11
    assert mixed_case["bridge_local_window_index"] == 3
    assert mixed_case["block_window_index"] == 303
    assert mixed_case["source_index"] == 903
    assert mixed_case["calendar"] == {"history_end": "2020-03-09"}
    assert mixed_case["bridge"]["target_cosine"] == 0.82
    assert "mixed_regime_semantic_ambiguity" in mixed_case["failure_modes"]
    assert mixed_case["priority"] == "medium"

    assert by_window["weak_bridge"]["priority"] == "high"
    assert by_window["label_borderline"]["priority"] == "label_first"
    assert manifest["downstream_indices"]["bridge_hard_case_validation"] == {
        "window_ids": ["mixed_borderline", "label_borderline"],
        "manifest_window_indices": [11, 12],
        "bridge_local_window_indices": [3, 4],
        "block_window_indices": [303, 304],
        "source_indices": [903, 904],
    }


def test_render_bridge_hard_case_manifest_markdown_contains_subsets_and_cases() -> None:
    manifest = build_bridge_hard_case_manifest(_diagnostics(), _casebook())
    markdown = render_bridge_hard_case_manifest_markdown(manifest)

    assert "# Bridge Hard-Case Validation Manifest" in markdown
    assert "## Downstream Subsets" in markdown
    assert "mixed_borderline" in markdown
    assert "bridge_model_hard_case" in markdown
