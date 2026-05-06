import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (
    StoryGroundingResult,
    assess_hard_case_gate,
    build_story_query_text,
    build_story_smoke_report,
    render_story_smoke_markdown,
    score_implication_alignment,
)


def _grounding() -> StoryGroundingResult:
    return StoryGroundingResult.model_validate(
        {
            "narrative_frame": "fragile risk-on rebound",
            "cleaned_conditioning_text": (
                "Fragile risk-on rebound with equities recovering, volatility "
                "compressing, spreads stabilizing, and carry appetite returning."
            ),
            "market_implications": [
                {
                    "market": "SPX",
                    "direction": "up",
                    "magnitude": "medium",
                    "confidence": "high",
                    "evidence": ["equities are recovering"],
                    "inferred": False,
                },
                {
                    "market": "VIX",
                    "direction": "down",
                    "magnitude": "medium",
                    "confidence": "high",
                    "evidence": ["volatility is compressing"],
                    "inferred": False,
                },
                {
                    "market": "BBB_OAS",
                    "direction": "tighter",
                    "magnitude": "small",
                    "confidence": "medium",
                    "evidence": ["spreads are stabilizing"],
                    "inferred": True,
                },
            ],
            "grounding_warnings": [
                {
                    "code": "causal_interpretation",
                    "severity": "warning",
                    "message": "Liquidity-driven is an interpretation, not observed fact.",
                }
            ],
            "unsupported_claims": [],
            "critique": ["No cited external event is asserted."],
        }
    )


def _bundles() -> list[dict]:
    return [
        {
            "window_id": "risk_on_a",
            "source_index": 100,
            "window_index": 0,
            "manifest_split": "train",
            "calendar": {"history_end": "2020-01-01"},
            "narratives": [
                {
                    "text": (
                        "A fragile risk-on recovery with equities higher, "
                        "volatility lower, and credit stabilizing."
                    )
                }
            ],
            "market_implications": [
                {"market": "SPX", "direction": "up", "magnitude": "medium"},
                {"market": "VIX", "direction": "down", "magnitude": "medium"},
            ],
        },
        {
            "window_id": "risk_off_b",
            "source_index": 101,
            "window_index": 1,
            "manifest_split": "test",
            "narratives": [
                {
                    "text": (
                        "A de-risking window with equities falling and "
                        "volatility rising."
                    )
                }
            ],
            "market_implications": [
                {"market": "SPX", "direction": "down", "magnitude": "large"},
                {"market": "VIX", "direction": "up", "magnitude": "large"},
            ],
        },
    ]


def _casebook() -> dict:
    return {
        "cases": [
            {
                "window_id": "risk_on_a",
                "input_narrative": "Risk-on recovery casebook narrative.",
                "observed_fact_tokens": "SPX: UP MEDIUM; VIX: DOWN MEDIUM",
                "scores": {
                    "narrative_generator_topk": {
                        "energy_score_improvement_vs_persistence": 0.2,
                        "ensemble_crps_improvement_vs_persistence": 0.1,
                        "coverage_80": 0.64,
                    }
                },
            }
        ]
    }


def _hard_case_manifest() -> dict:
    return {
        "subsets": {
            "bridge_hard_case_validation": ["risk_off_b"],
            "bridge_model_hard_case": [],
            "label_repair": [],
            "mixed_regime_contrastive": [],
            "rank_metric_review": [],
        }
    }


def test_build_story_query_text_keeps_story_implications_and_warnings() -> None:
    text = build_story_query_text(
        "This has the shape of a fragile risk-on rebound.",
        _grounding(),
    )

    assert "NARRATIVE:" in text
    assert "EXPLICIT_MARKET_IMPLICATIONS:" in text
    assert "SPX: up medium" in text
    assert "GROUNDING_WARNINGS:" in text
    assert "causal_interpretation" in text


def test_build_story_smoke_report_returns_relevance_analogues_and_gate() -> None:
    query_condition = np.array([1.0, 0.0], dtype=np.float32)
    memory_targets = np.array([[0.9, 0.1], [-0.8, 0.2]], dtype=np.float32)

    report = build_story_smoke_report(
        story="This has the shape of a fragile risk-on rebound.",
        grounding=_grounding(),
        query_text="query text",
        query_condition=query_condition,
        memory_targets=memory_targets,
        bundles=_bundles(),
        casebook=_casebook(),
        hard_case_manifest=_hard_case_manifest(),
        top_k=1,
        ood_threshold=0.75,
        scenario_summary={
            "generated_state_shape": [2, 4, 30, 39],
            "finite_rate": 1.0,
            "terminal_delta_summary": [{"market": "SPX", "mean_terminal_delta": 1.2}],
        },
    )

    assert report["condition_diagnostics"]["top_cosine"] > 0.99
    assert report["relevance"]["status"] == "pass"
    assert report["historical_analogues"][0]["window_id"] == "risk_on_a"
    assert report["historical_analogues"][0]["casebook_narrative"].startswith(
        "Risk-on recovery"
    )
    assert (
        report["historical_analogues"][0]["implication_alignment"]["match_rate"] == 1.0
    )
    assert report["hard_case_gate"]["status"] == "pass"
    assert report["generation"]["terminal_delta_summary"][0]["market"] == "SPX"


def test_assess_hard_case_gate_warns_on_hard_case_overlap_and_fails_ood() -> None:
    warning = assess_hard_case_gate(
        [{"window_id": "risk_off_b", "cosine": 0.8}],
        _hard_case_manifest(),
        ood_warning=False,
    )
    assert warning["status"] == "warning"
    assert warning["subset_hits"]["bridge_hard_case_validation"] == ["risk_off_b"]

    failed = assess_hard_case_gate(
        [{"window_id": "risk_on_a", "cosine": 0.2}],
        _hard_case_manifest(),
        ood_warning=True,
    )
    assert failed["status"] == "fail"
    assert failed["reason"] == "nearest historical analogue below similarity threshold"


def test_score_implication_alignment_flags_direction_mismatches() -> None:
    aligned = score_implication_alignment(
        _grounding(),
        [
            {"market": "SPX", "direction": "up"},
            {"market": "VIX", "direction": "down"},
            {"market": "BBB_OAS", "direction": "tighter"},
        ],
    )
    assert aligned["status"] == "pass"
    assert aligned["match_rate"] == 1.0

    mismatch = score_implication_alignment(
        _grounding(),
        [
            {"market": "SPX", "direction": "down"},
            {"market": "VIX", "direction": "up"},
            {"market": "BBB_OAS", "direction": "wider"},
        ],
    )
    assert mismatch["status"] == "warning"
    assert mismatch["match_rate"] == 0.0
    assert mismatch["mismatches"][0]["market"] == "SPX"


def test_render_story_smoke_markdown_contains_boss_demo_sections() -> None:
    report = build_story_smoke_report(
        story="This has the shape of a fragile risk-on rebound.",
        grounding=_grounding(),
        query_text="query text",
        query_condition=np.array([1.0, 0.0], dtype=np.float32),
        memory_targets=np.array([[1.0, 0.0]], dtype=np.float32),
        bundles=_bundles()[:1],
        casebook=_casebook(),
        hard_case_manifest=_hard_case_manifest(),
        top_k=1,
        ood_threshold=0.75,
        scenario_summary={},
    )

    markdown = render_story_smoke_markdown(report)

    assert "# Risk Manager Story Smoke Test" in markdown
    assert "## Extracted Market Implications" in markdown
    assert "## Historical Analogues" in markdown
    assert "Implication Match" in markdown
    assert "## Scenario Summary" in markdown
