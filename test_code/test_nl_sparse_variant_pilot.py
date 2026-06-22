import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_sparse_variant_pilot as pilot


def test_validate_pair_rejects_direct_negation_of_target_phrase():
    pair = pilot.SparsePilotPair(
        angle="tape_read",
        positive_sparse_text="The tape looks like a weak-dollar commodity bid.",
        negative_window_id="joint39_train_2204",
        negative_sparse_text=(
            "This is not a weak-dollar commodity bid; the dollar is stronger "
            "and commodities are being sold."
        ),
        quality_notes=["uses explicit negation"],
    )

    errors = pilot.validate_pair(
        pair,
        target_title="weak-dollar commodity bid with fading credit stress",
        positive_text=pair.positive_sparse_text,
    )

    assert any(error["code"] == "direct_negation_shortcut" for error in errors)
    assert any(error["code"] == "target_phrase_reuse" for error in errors)


def test_validate_pair_accepts_natural_sparse_contradiction():
    pair = pilot.SparsePilotPair(
        angle="tape_read",
        positive_sparse_text="The tape looks like a weak-dollar commodity bid.",
        negative_window_id="joint39_train_2204",
        negative_sparse_text=(
            "Dollar strength is paired with commodity selling and a defensive "
            "rates bid."
        ),
        quality_notes=["dollar and commodity channels conflict"],
    )

    errors = pilot.validate_pair(
        pair,
        target_title="weak-dollar commodity bid with fading credit stress",
        positive_text=pair.positive_sparse_text,
    )

    assert errors == []


def test_build_prompt_requests_multiple_sparse_angles_without_local_prose():
    prompt = pilot.build_sparse_pilot_prompt(
        target={
            "window_id": "joint39_train_1553",
            "scenario_title": "weak-dollar commodity bid with fading credit stress",
            "archetype": "weak_dollar_commodity_repricing",
            "mechanical_summary": "Mechanical baseline: DXY lower; crude oil higher.",
            "evidence_used": ["DXY lower", "crude oil higher"],
        },
        negative_candidates=[
            {
                "window_id": "joint39_train_2204",
                "scenario_title": "Treasury-bid risk-off",
                "archetype": "liquidity_withdrawal",
                "mechanical_summary": "Mechanical baseline: DXY higher; crude oil lower.",
                "contradiction_channels": ["DXY", "CRUDE_OIL"],
            }
        ],
        angles=["tape_read", "portfolio_concern", "macro_channel"],
    )

    assert "Return only JSON" in prompt
    assert "tape_read" in prompt
    assert "portfolio_concern" in prompt
    assert "macro_channel" in prompt
    assert "Do not use direct negation" in prompt
    assert "Do not write phrases like" in prompt
    assert "Payload" in prompt
