import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (
    StoryGroundingResult,
    assess_hard_case_gate,
    build_story_query_text,
    build_story_smoke_report,
    path_quantiles_for_generated_states,
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


def test_path_quantiles_for_generated_states_builds_factor_fan_chart_data() -> None:
    states = np.zeros((1, 3, 2, 4), dtype=np.float32)
    states[0, :, :, 0] = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    states[0, :, :, 3] = [[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]]
    current = np.zeros((1, 4), dtype=np.float32)
    result = path_quantiles_for_generated_states(
        states,
        current,
        ["iv:0", "iv:1", "iv:2", "factor:spx"],
    )

    spx = next(row for row in result if row["market"] == "SPX")
    assert spx["days"] == [1, 2]
    assert spx["p50"] == [20.0, 22.0]
    assert spx["p10"][0] < spx["p50"][0] < spx["p90"][0]


def test_path_quantiles_for_generated_states_includes_paper_selected_iv_cells() -> None:
    states = np.zeros((1, 3, 2, 25), dtype=np.float32)
    current = np.zeros((1, 25), dtype=np.float32)
    states[0, :, :, 7] = [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]]
    states[0, :, :, 17] = [[1.10, 1.20], [1.30, 1.40], [1.50, 1.60]]
    result = path_quantiles_for_generated_states(
        states,
        current,
        [f"iv:{idx}" for idx in range(25)],
    )

    atm_3m = next(row for row in result if row["market"] == "IV_ATM_3M")
    assert atm_3m["display_name"] == "IV ATM 3M (K=1.00)"
    assert atm_3m["cell"] == {
        "row": 1,
        "col": 2,
        "maturity": "3M",
        "moneyness": "1.00",
    }
    assert atm_3m["p50"] == [0.30000001192092896, 0.4000000059604645]

    atm_1y = next(row for row in result if row["market"] == "IV_ATM_1Y")
    assert atm_1y["p50"] == [1.2999999523162842, 1.399999976158142]


def test_path_quantiles_for_generated_states_can_split_by_analogue() -> None:
    states = np.zeros((2, 2, 2, 26), dtype=np.float32)
    states[0, :, :, 25] = [[10.0, 11.0], [12.0, 13.0]]
    states[1, :, :, 25] = [[30.0, 31.0], [32.0, 33.0]]
    current = np.zeros((2, 26), dtype=np.float32)
    result = path_quantiles_for_generated_states(
        states,
        current,
        [*(f"iv:{idx}" for idx in range(25)), "factor:spx"],
        analogues=[
            {"window_id": "joint39_val_0001"},
            {"window_id": "joint39_val_0002"},
        ],
    )

    pooled = next(
        row for row in result if row["market"] == "SPX" and row["analogue_key"] == "ALL"
    )
    first = next(
        row
        for row in result
        if row["market"] == "SPX" and row["analogue_key"] == "RANK_1"
    )
    second = next(
        row
        for row in result
        if row["market"] == "SPX" and row["analogue_key"] == "RANK_2"
    )

    assert pooled["p50"] == [21.0, 22.0]
    assert first["window_id"] == "joint39_val_0001"
    assert first["p50"] == [11.0, 12.0]
    assert second["window_id"] == "joint39_val_0002"
    assert second["p50"] == [31.0, 32.0]


def test_path_quantiles_for_generated_states_adds_paths_and_realized_analogue_future() -> (
    None
):
    states = np.zeros((1, 8, 3, 26), dtype=np.float32)
    for sample in range(8):
        states[0, sample, :, 25] = [sample, sample + 1, sample + 2]
    current = np.zeros((1, 26), dtype=np.float32)
    future = np.zeros((1, 3, 26), dtype=np.float32)
    future[0, :, 25] = [2.0, 4.0, 8.0]

    result = path_quantiles_for_generated_states(
        states,
        current,
        [*(f"iv:{idx}" for idx in range(25)), "factor:spx"],
        analogues=[{"window_id": "joint39_val_0001"}],
        future_states=future,
        max_paths=5,
    )

    spx = next(
        row
        for row in result
        if row["market"] == "SPX" and row["analogue_key"] == "RANK_1"
    )
    assert spx["realized_path"] == [2.0, 4.0, 8.0]
    assert len(spx["sample_paths"]) == 5
    assert spx["sample_paths"][0]["label"].startswith("Generated path")
    assert spx["sample_paths"][0]["values"] == [0.0, 1.0, 2.0]


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
