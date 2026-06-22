import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_card_v3_codex_testflight import (
    MultiFormatNarrative,
    _validate_multiformat,
    multiformat_narrative_to_episode_card,
)


def _narrative(**overrides):
    fields = {
        "window_id": "joint39_train_0000",
        "scenario_title": "weak-dollar commodity bid",
        "archetype": "weak_dollar_commodity_repricing",
        "archetype_confidence": "high",
        "old_factor_baseline": (
            "Legacy mechanical baseline: DXY lower; oil higher; gold higher; "
            "credit mostly stable."
        ),
        "factor_list_baseline": (
            "Factor list: broad dollar lower, oil and gold higher, credit "
            "mostly contained."
        ),
        "technical_factor_evidence": (
            "Technical evidence: DXY fell while crude and gold rose; lower-quality "
            "credit was flat."
        ),
        "sparse_user_prompt": (
            "Dollar soft, oil and gold bid, with credit mostly contained."
        ),
        "weekly_risk_monitor": (
            "The active channel is broad-dollar weakness feeding commodities; "
            "credit confirmation remains limited."
        ),
        "institutional_risk_committee_note": (
            "The committee read is moderate macro repricing rather than broad "
            "stress. Commodity and FX exposures matter most, while credit "
            "confirmation remains contained."
        ),
        "mechanism_first_memo": (
            "Mechanism first: dollar softness supports commodities, with only "
            "limited credit spillover."
        ),
        "risk_manager_memo": (
            "Risk memo: weak-dollar commodity support is the main regime. "
            "Portfolio sensitivity is concentrated in dollar and commodity "
            "exposures, with credit only a secondary concern."
        ),
        "full_risk_manager_memo": (
            "Full memo: the observed prefix shows broad-dollar weakness, higher "
            "oil and gold, and only limited credit confirmation. Transmission "
            "runs through macro and commodity exposure rather than systemic "
            "spread stress."
        ),
        "evidence_used": ["DXY lower", "oil higher", "gold higher"],
        "ambiguity_flags": ["credit confirmation is limited"],
        "no_forecast_caveat": "Current/recent prefix only; no forecast.",
        "contrastive_hard_negatives": ["Dollar squeeze with commodity liquidation."],
        "quality_self_critique": ["Views are distinct."],
    }
    fields.update(overrides)
    return MultiFormatNarrative(**fields)


def test_multiformat_card_uses_distinct_authored_old_eight_views():
    narrative = _narrative()

    card = multiformat_narrative_to_episode_card(
        narrative,
        source_path="synthetic_multiformat_narratives.jsonl",
    )
    views = card["views"]

    assert views["risk_manager_memo"] == narrative.risk_manager_memo
    assert views["full_professional"] == narrative.full_risk_manager_memo
    assert views["risk_manager_memo"] != views["full_professional"]
    assert views["factor_list_baseline"] == narrative.factor_list_baseline
    assert views["technical_factor_evidence"] == narrative.technical_factor_evidence
    assert views["factor_list_baseline"] != views["technical_factor_evidence"]


def test_validate_multiformat_rejects_duplicate_old_eight_views():
    narrative = _narrative(
        risk_manager_memo=(
            "Full memo: the observed prefix shows broad-dollar weakness, higher "
            "oil and gold, and only limited credit confirmation. Transmission "
            "runs through macro and commodity exposure rather than systemic "
            "spread stress."
        ),
        full_risk_manager_memo=(
            "Full memo: the observed prefix shows broad-dollar weakness, higher "
            "oil and gold, and only limited credit confirmation. Transmission "
            "runs through macro and commodity exposure rather than systemic "
            "spread stress."
        ),
    )

    issues = _validate_multiformat(narrative)

    assert any(
        issue["code"] == "duplicate_training_view_text"
        and issue["fields"] == ["risk_manager_memo", "full_professional"]
        for issue in issues
    )
