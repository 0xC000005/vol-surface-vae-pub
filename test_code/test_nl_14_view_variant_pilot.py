import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot


def test_validate_batch_rejects_duplicate_positive_texts():
    pairs = []
    for view_name in pilot.EXPECTED_VIEW_NAMES:
        pairs.append(
            pilot.NarrativePair(
                view_name=view_name,
                positive_text="same positive text for all views",
                negative_window_id="joint39_train_1000",
                negative_text=f"negative text for {view_name}",
                quality_notes=["test"],
            )
        )
    batch = pilot.FourteenViewPilotBatch(
        target_window_id="joint39_train_1553",
        target_title="weak-dollar commodity bid with fading credit stress",
        pairs=pairs,
    )

    validation = pilot.validate_batch(
        batch=batch,
        target={
            "window_id": "joint39_train_1553",
            "scenario_title": "weak-dollar commodity bid with fading credit stress",
        },
        negative_candidates=[{"window_id": "joint39_train_1000"}],
    )

    assert validation["status"] == "fail"
    assert any(error["code"] == "duplicate_positive_text" for error in validation["errors"])


def test_build_prompt_demands_fourteen_unique_positive_and_negative_pairs():
    prompt = pilot.build_prompt(
        target={
            "window_id": "joint39_train_1553",
            "scenario_title": "weak-dollar commodity bid with fading credit stress",
            "archetype": "weak_dollar_commodity_repricing",
            "mechanical_summary": "Mechanical baseline: DXY lower; crude higher.",
            "evidence_used": ["DXY lower", "crude higher"],
        },
        negative_candidates=[
            {
                "window_id": "joint39_train_1000",
                "scenario_title": "dollar squeeze",
                "archetype": "liquidity_withdrawal",
                "mechanical_summary": "Mechanical baseline: DXY higher; crude lower.",
                "contradiction_channels": ["DXY", "CRUDE_OIL"],
                "agreement_count": 4,
            }
        ],
    )

    assert "exactly 14 pairs" in prompt
    assert "risk_manager_memo and full_professional must not be copies" in prompt
    assert "factor_list_baseline and technical_factor_evidence must not be copies" in prompt
    assert "Do not write phrases like" in prompt
    for view_name in pilot.EXPECTED_VIEW_NAMES:
        assert view_name in prompt


def test_build_validation_retry_prompt_requests_agentic_revision_for_failed_pairs():
    batch = pilot.FourteenViewPilotBatch(
        target_window_id="joint39_train_0001",
        target_title="firm-dollar front-end tightening with muted risk stress",
        pairs=[
            pilot.NarrativePair(
                view_name="risk_manager_memo",
                positive_text=(
                    "Decision memo: classify the tape as mixed macro tightening "
                    "with limited risk damage."
                ),
                negative_window_id="joint39_train_3943",
                negative_text=(
                    "Decision memo: classify the tape as relief with defensive "
                    "participation, not a simple growth rebound."
                ),
                quality_notes=["test"],
            )
        ],
    )
    validation = {
        "status": "fail",
        "errors": [
            {
                "code": "pair_validation_failed",
                "view_name": "risk_manager_memo",
                "errors": [
                    {"code": "direct_negation_shortcut", "patterns": ["not_target_phrase"]},
                    {
                        "code": "target_or_positive_phrase_reuse",
                        "phrases": ["front-end tightening"],
                    },
                ],
            }
        ],
    }

    prompt = pilot.build_validation_retry_prompt(
        base_prompt="BASE PROMPT",
        batch=batch,
        validation=validation,
    )

    assert "BASE PROMPT" in prompt
    assert "Regenerate the full JSON" in prompt
    assert "risk_manager_memo" in prompt
    assert "direct_negation_shortcut" in prompt
    assert "target_or_positive_phrase_reuse" in prompt
    assert "Do not repair by local rules" in prompt
    assert "not a simple growth rebound" in prompt


def test_assign_negative_candidates_prefers_unique_focus_contradictions():
    candidates = []
    for idx, view_name in enumerate(pilot.EXPECTED_VIEW_NAMES):
        candidates.append(
            {
                "window_id": f"joint39_train_{2000 + idx}",
                "scenario_title": f"candidate {idx}",
                "archetype": "mixed_ambiguous",
                "mechanical_summary": "Mechanical baseline.",
                "contradiction_channels": list(pilot.VIEW_FOCUS_CHANNELS[view_name])[:2],
                "contradiction_count": 2,
                "agreement_count": 4,
            }
        )
    assignments = pilot.assign_negative_candidates_by_view(candidates)

    assigned_ids = [row["candidate"]["window_id"] for row in assignments]
    assert [row["view_name"] for row in assignments] == list(pilot.EXPECTED_VIEW_NAMES)
    assert len(set(assigned_ids)) == len(pilot.EXPECTED_VIEW_NAMES)
    for row in assignments:
        focus_hits = set(row["candidate"]["contradiction_channels"]) & set(
            pilot.VIEW_FOCUS_CHANNELS[row["view_name"]]
        )
        assert len(focus_hits) >= row["minimum_focus_contradictions"]


def test_effective_min_focus_contradictions_relaxes_impossible_focus_requirement():
    candidates = [
        {
            "window_id": "joint39_train_2000",
            "contradiction_channels": ["CRUDE_OIL", "SPX", "VIX"],
        },
        {
            "window_id": "joint39_train_2001",
            "contradiction_channels": ["CRUDE_OIL", "BBB_OAS"],
        },
    ]

    minimum = pilot.effective_min_focus_contradictions(
        "mechanism_first",
        candidates,
    )

    assert minimum == 1


def test_effective_min_focus_contradictions_uses_single_candidate_feasibility():
    candidates = [
        {
            "window_id": "joint39_train_2000",
            "contradiction_channels": ["DXY", "SPX", "VIX"],
        },
        {
            "window_id": "joint39_train_2001",
            "contradiction_channels": ["CRUDE_OIL", "BBB_OAS"],
        },
    ]

    minimum = pilot.effective_min_focus_contradictions(
        "mechanism_first",
        candidates,
    )

    assert minimum == 1


def test_validate_batch_requires_assigned_negative_window_and_focus_quality():
    target = {
        "window_id": "joint39_train_1553",
        "scenario_title": "weak-dollar commodity bid with fading credit stress",
    }
    assigned = [
        {
            "view_name": "sparse_user_query",
            "candidate": {
                "window_id": "joint39_train_2000",
                "contradiction_channels": ["DXY", "CRUDE_OIL"],
            },
            "focus_channels": ["DXY", "CRUDE_OIL", "GOLD"],
            "minimum_focus_contradictions": 2,
        }
    ]
    pairs = [
        pilot.NarrativePair(
            view_name=view_name,
            positive_text=f"distinct positive text for {view_name}",
            negative_window_id="joint39_train_2000",
            negative_text=f"distinct negative text for {view_name}",
            quality_notes=["test"],
        )
        for view_name in pilot.EXPECTED_VIEW_NAMES
    ]
    pairs[0] = pilot.NarrativePair(
        view_name="sparse_user_query",
        positive_text="DXY is softer while oil and gold are bid.",
        negative_window_id="joint39_train_2001",
        negative_text="Dollar strength is paired with energy selling.",
        quality_notes=["test"],
    )
    batch = pilot.FourteenViewPilotBatch(
        target_window_id=target["window_id"],
        target_title=target["scenario_title"],
        pairs=pairs,
    )

    validation = pilot.validate_batch(
        batch=batch,
        target=target,
        negative_candidates=[
            {"window_id": "joint39_train_2000"},
            {"window_id": "joint39_train_2001"},
        ],
        assigned_negative_candidates=assigned,
    )

    assert validation["status"] == "fail"
    sparse_errors = [
        error
        for error in validation["errors"]
        if error.get("code") == "pair_validation_failed"
        and error.get("view_name") == "sparse_user_query"
    ][0]["errors"]
    assert any(error["code"] == "negative_window_not_assigned_to_view" for error in sparse_errors)


def test_validate_pair_rejects_unexplained_large_raw_spread_magnitude():
    pair = pilot.NarrativePair(
        view_name="technical_factor_evidence",
        positive_text=(
            "Evidence read: AAA_OAS +552.89 while DXY is lower and crude is higher."
        ),
        negative_window_id="joint39_train_2000",
        negative_text="Evidence read: dollar strength and commodity selling dominate.",
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="weak-dollar commodity bid with fading credit stress",
        candidate_ids={"joint39_train_2000"},
    )

    assert any(error["code"] == "opaque_raw_spread_magnitude" for error in errors)


def test_validate_pair_rejects_assigned_prefix_meta_language():
    pair = pilot.NarrativePair(
        view_name="full_professional",
        positive_text="The current prefix is led by a weak dollar and commodity demand.",
        negative_window_id="joint39_train_2000",
        negative_text="The assigned prefix is a dollar-pressure tape with crude lower.",
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="weak-dollar commodity bid with fading credit stress",
        candidate_ids={"joint39_train_2000"},
    )

    assert any(error["code"] == "internal_training_language" for error in errors)


def test_validate_pair_allows_factor_list_shared_factor_names():
    pair = pilot.NarrativePair(
        view_name="factor_list_baseline",
        positive_text=(
            "SPX up small; VIX flat; BBB flat; AAA wider small; DXY down; "
            "USDJPY up; crude up; US2Y and US10Y up; gold up."
        ),
        negative_window_id="joint39_train_2000",
        negative_text=(
            "SPX up small; VIX lower; BBB flat; AAA wider small; DXY down; "
            "USDJPY down; crude down; US2Y and US10Y down; gold higher."
        ),
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="weak-dollar commodity bid with fading credit stress",
        candidate_ids={"joint39_train_2000"},
    )

    assert not any(error["code"] == "high_positive_negative_token_overlap" for error in errors)


def test_validate_pair_allows_factor_list_same_factor_vocabulary_with_changed_directions():
    pair = pilot.NarrativePair(
        view_name="factor_list_baseline",
        positive_text=(
            "Factors: SPX flat; VIX flat; BBB OAS flat; AAA OAS wider large; "
            "DXY down small; USDJPY up large; crude up medium; US2Y up small; "
            "US10Y up medium; gold up small."
        ),
        negative_window_id="joint39_train_2000",
        negative_text=(
            "Factors: SPX flat; VIX up small; BBB OAS wider small; AAA OAS wider small; "
            "DXY up medium; USDJPY up small; crude up small; US2Y down medium; "
            "US10Y down large; gold down small."
        ),
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="rate energy pressure with selective credit strain",
        candidate_ids={"joint39_train_2000"},
    )

    assert not any(error["code"] == "high_positive_negative_token_overlap" for error in errors)


def test_validate_pair_allows_sparse_desk_note_anchor_overlap():
    pair = pilot.NarrativePair(
        view_name="sparse_variant_desk_note",
        positive_text=(
            "Morning sheet: equities firmer, vol calmer, DXY lower, crude up a little, "
            "gold up a little."
        ),
        negative_window_id="joint39_train_2000",
        negative_text=(
            "Morning sheet: equities up, vol down, oil lower with confidence, "
            "gold lower, DXY a touch firmer."
        ),
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="post-stress reflation credit-beta relief under dollar weakness",
        candidate_ids={"joint39_train_2000"},
    )

    assert not any(error["code"] == "high_positive_negative_token_overlap" for error in errors)


def test_validate_pair_allows_generic_credit_phrase_reuse():
    pair = pilot.NarrativePair(
        view_name="mechanism_first",
        positive_text=(
            "Dollar liquidity is the transmission channel, with heavy BBB spread pressure."
        ),
        negative_window_id="joint39_train_2000",
        negative_text=(
            "Volatility pressure reaches equities and BBB credit while duration buying "
            "reinforces the defensive tone."
        ),
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title=(
            "dollar-liquidity squeeze with severe BBB credit stress and weak haven signals"
        ),
        candidate_ids={"joint39_train_2000"},
    )

    assert not any(error["code"] == "target_or_positive_phrase_reuse" for error in errors)


def test_validate_pair_allows_generic_high_grade_spread_phrase_reuse():
    pair = pilot.NarrativePair(
        view_name="sparse_user_query",
        positive_text="Rates and energy are firmer while high-grade spreads widen.",
        negative_window_id="joint39_train_2000",
        negative_text="High-grade spread pressure is visible, but rates and energy soften.",
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="rate energy pressure with isolated high-grade spread widening",
        candidate_ids={"joint39_train_2000"},
    )

    assert not any(error["code"] == "target_or_positive_phrase_reuse" for error in errors)


def test_validate_pair_allows_generic_high_grade_credit_phrase_reuse():
    pair = pilot.NarrativePair(
        view_name="institutional_risk_committee_note",
        positive_text=(
            "Committee note: severity is low-to-moderate and centered on credit repair."
        ),
        negative_window_id="joint39_train_2000",
        negative_text=(
            "Committee note: equity exposure benefits from lower volatility, while "
            "high-grade credit remains the vulnerable leg."
        ),
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="high-grade credit relief with dollar firmness",
        candidate_ids={"joint39_train_2000"},
    )

    assert not any(error["code"] == "target_or_positive_phrase_reuse" for error in errors)


def test_validate_pair_ignores_non_distinctive_two_word_title_phrases():
    pair = pilot.NarrativePair(
        view_name="risk_manager_memo",
        positive_text="Decision memo: classify the tape as reflation with dollar pressure.",
        negative_window_id="joint39_train_2000",
        negative_text=(
            "Decision memo: relief under mixed credit conditions comes from "
            "equity resilience and softer volatility."
        ),
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="post-stress reflation credit-beta relief under dollar weakness",
        candidate_ids={"joint39_train_2000"},
    )

    assert not any(error["code"] == "target_or_positive_phrase_reuse" for error in errors)


def test_validate_pair_still_rejects_distinctive_hyphenated_title_phrase():
    pair = pilot.NarrativePair(
        view_name="risk_manager_memo",
        positive_text="Decision memo: classify the tape as mixed macro tightening.",
        negative_window_id="joint39_train_2000",
        negative_text=(
            "Decision memo: portfolio sensitivity is less about front-end tightening "
            "than credit selection."
        ),
        quality_notes=["test"],
    )

    errors = pilot._pair_validation_errors(
        pair,
        target_title="firm-dollar front-end tightening with muted risk stress",
        candidate_ids={"joint39_train_2000"},
    )

    assert any(error["code"] == "target_or_positive_phrase_reuse" for error in errors)


def test_run_pilot_from_prepared_payload_dry_run_writes_prompt(tmp_path) -> None:
    target = {
        "window_id": "uploaded_factor_table",
        "scenario_title": "uploaded factor table",
        "archetype": "mixed_ambiguous",
        "mechanical_summary": "Mechanical baseline: SPX up small; DXY down medium.",
        "evidence_used": ["SPX start=100 end=110", "DXY start=90 end=85"],
    }
    negative_candidates = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "scenario_title": f"candidate {idx}",
            "archetype": "mixed_ambiguous",
            "mechanical_summary": "Mechanical baseline: SPX lower; DXY higher.",
            "evidence_used": ["SPX lower", "DXY higher"],
            "contradiction_channels": ["SPX", "DXY", "GOLD"],
            "contradiction_count": 3,
            "agreement_count": 1,
        }
        for idx in range(100, 140)
    ]
    args = SimpleNamespace(
        output_dir=tmp_path,
        support_cards_jsonl="support.jsonl",
        dry_run=True,
        model="gpt-test",
        reasoning_effort="low",
        timeout_seconds=10,
        validation_retries=0,
    )

    report = pilot.run_pilot_from_prepared_payload(
        args=args,
        target=target,
        negative_candidates=negative_candidates,
        source_paths={"cards_jsonl": "sidecar", "support_cards_jsonl": "support.jsonl"},
    )

    assert report["status"] == "fail"
    assert report["dry_run"] is True
    assert report["target"]["window_id"] == "uploaded_factor_table"
    assert (tmp_path / "fourteen_view_prompt.txt").exists()
    assert (tmp_path / "fourteen_view_schema.json").exists()
