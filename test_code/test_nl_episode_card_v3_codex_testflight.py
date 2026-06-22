import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_card_v3_codex_testflight import (
    MultiFormatNarrative,
    _validate_multiformat,
    build_multiformat_prompt,
    multiformat_narrative_to_episode_card,
    prepare_testflight,
    render_multiformat_one_page_review,
    review_codex_outputs,
    select_cards_for_multiformat_generation,
    select_representative_cards,
    support_card_to_bundle,
)


def _support_card(window_id: str, title: str, rows: list[dict]) -> dict:
    return {
        "schema_version": "nl_episode_narrative_card_v1",
        "window_id": window_id,
        "split": "support_train",
        "scenario_title": title,
        "archetype": "mixed_ambiguous",
        "archetype_confidence": "medium",
        "caption_fields": {
            "training_caption": f"{title}: current market prefix.",
            "mechanical_summary": f"{title} mechanics.",
        },
        "support_metadata": {
            "calendar_start_date": "2020-01-01",
            "calendar_end_date": "2020-02-14",
            "window_index": int(window_id.rsplit("_", 1)[-1]),
            "support_move_rows": rows,
        },
    }


def _row(market: str, direction: str, z_change: float) -> dict:
    label = {"up": "higher", "down": "lower", "flat": "stable"}[direction]
    if market in {"BBB_OAS", "AAA_OAS"} and direction == "up":
        label = "wider"
    if market in {"BBB_OAS", "AAA_OAS"} and direction == "down":
        label = "tighter"
    return {
        "market": market,
        "direction": direction,
        "direction_label": label,
        "magnitude": "large" if abs(z_change) >= 1 else "small",
        "raw_change": z_change * 10,
        "z_change": z_change,
        "theme": market.lower(),
    }


def test_support_card_to_bundle_preserves_full_30_day_prefix_evidence() -> None:
    card = _support_card(
        "joint39_train_0001",
        "Defensive risk-off shock",
        [_row("SPX", "down", -1.2), _row("VIX", "up", 1.1)],
    )

    bundle = support_card_to_bundle(card)

    assert bundle["window_id"] == "joint39_train_0001"
    assert bundle["calendar"]["calendar_end_date"] == "2020-02-14"
    assert bundle["market_implications"][0]["market"] == "SPX"
    assert bundle["market_implications"][0]["confidence"] == "high"
    assert "support_card_972b" in bundle["narratives"][0]["kind"]


def test_prepare_testflight_writes_reviewable_codex_pipeline(tmp_path: Path) -> None:
    cards = [
        _support_card(
            "joint39_train_0001",
            "Classic safe-haven gold risk-off",
            [
                _row("GOLD", "up", 1.2),
                _row("US10Y", "down", -1.0),
                _row("VIX", "up", 1.1),
                _row("SPX", "down", -1.3),
            ],
        ),
        _support_card(
            "joint39_train_0002",
            "Commodity-inflation pressure",
            [_row("CRUDE_OIL", "up", 1.5), _row("US10Y", "up", 0.8)],
        ),
    ]
    source = tmp_path / "cards.jsonl"
    source.write_text("\n".join(json.dumps(card) for card in cards) + "\n")

    result = prepare_testflight(
        Namespace(
            source_cards_jsonl=source,
            output_dir=tmp_path / "out",
            max_cases=2,
            min_window_gap=1,
        )
    )
    pipeline = json.loads(Path(result["pipeline_report"]).read_text())

    assert result["selected_count"] == 2
    assert pipeline["schema_version"] == "episode_card_v3_codex_testflight_pipeline_v1"
    assert len(pipeline["narrative_bundles"]) == 2
    assert Path(result["selected_support_cases_markdown"]).exists()


def test_select_representative_cards_prefers_target_angle_matches() -> None:
    risk_off = _support_card(
        "joint39_train_0001",
        "Classic safe-haven gold risk-off",
        [
            _row("GOLD", "up", 1.2),
            _row("US10Y", "down", -1.0),
            _row("VIX", "up", 1.1),
            _row("SPX", "down", -1.3),
        ],
    )
    commodity = _support_card(
        "joint39_train_0002",
        "Commodity-inflation pressure",
        [_row("CRUDE_OIL", "up", 1.5), _row("US10Y", "up", 0.8)],
    )

    selected = select_representative_cards([commodity, risk_off], max_cases=2)

    assert {row["window_id"] for row in selected} == {
        "joint39_train_0001",
        "joint39_train_0002",
    }


def test_multiformat_stride_selection_uses_15_day_rich_cadence() -> None:
    cards = [
        _support_card(
            f"joint39_train_{idx:04d}",
            "Mixed cross-asset regime",
            [_row("SPX", "up", 0.3)],
        )
        for idx in range(46)
    ]

    selected = select_cards_for_multiformat_generation(
        cards,
        max_cases=0,
        min_window_gap=0,
        selection_mode="stride",
        rich_stride=15,
    )

    assert [row["window_id"] for row in selected] == [
        "joint39_train_0000",
        "joint39_train_0015",
        "joint39_train_0030",
        "joint39_train_0045",
    ]


def test_review_codex_outputs_builds_human_readable_packet(tmp_path: Path) -> None:
    caption = {
        "window_id": "joint39_train_0001",
        "schema_version": "risk_manager_caption_v2",
        "scenario_title": "Defensive Risk-Off Prefix",
        "archetype": "financial_accident",
        "archetype_confidence": "high",
        "mechanical_summary": "Equities weakened while volatility and defensive demand rose.",
        "current_market_state": "SPX is lower, VIX is higher, and gold is firmer.",
        "trigger": "The supplied market tape shows risk appetite deteriorating.",
        "transmission": "Lower equities and higher volatility raise hedge demand.",
        "cross_asset_reaction": "SPX down, VIX up, gold up, Treasury yields down.",
        "sequence": "The 30-day prefix moves from calmer risk appetite to defensive demand.",
        "portfolio_vulnerability": "Long equity and short-volatility exposure is vulnerable.",
        "risk_manager_implication": "Use this as a current-condition stress label.",
        "evidence_used": ["SPX down", "VIX up", "Gold up"],
        "ambiguity_flags": ["no external catalyst supplied"],
        "leakage_exclusions": ["realized future path"],
        "no_forecast_caveat": "This is not a forecast; it describes current conditions.",
        "training_caption": (
            "Defensive risk-off prefix: equities weakened, volatility rose, and "
            "safe-haven demand improved across the current/recent market tape."
        ),
        "contrastive_captions": [
            "Risk-on relief hard negative: equities rise and volatility falls."
        ],
        "quality_self_critique": ["Uses supplied market evidence only."],
    }
    captions_jsonl = tmp_path / "captions.jsonl"
    captions_jsonl.write_text(json.dumps(caption) + "\n")
    codex_report = tmp_path / "codex_report.json"
    codex_report.write_text(
        json.dumps(
            {
                "status": "pass",
                "requested_count": 1,
                "caption_count": 1,
                "validation_error_count": 0,
                "codex_error_count": 0,
                "prompt_version": "risk_manager_caption_v2_test",
                "codex_model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "captions": [caption],
                "validation": [{"window_id": "joint39_train_0001", "warnings": []}],
                "artifact_paths": {"captions_jsonl": str(captions_jsonl)},
            }
        )
    )

    result = review_codex_outputs(
        Namespace(output_dir=tmp_path / "out", codex_report=codex_report)
    )

    assert result["status"] == "pass"
    assert (
        Path(result["review_markdown"])
        .read_text()
        .startswith("# Codex-Authored EpisodeCardV3 Narrative Review")
    )
    assert Path(result["cards_jsonl"]).exists()


def test_multiformat_prompt_requires_distinct_codex_authored_views() -> None:
    bundle = support_card_to_bundle(
        _support_card(
            "joint39_train_0001",
            "Classic safe-haven gold risk-off",
            [
                _row("SPX", "down", -1.2),
                _row("VIX", "up", 1.4),
                _row("BBB_OAS", "up", 1.1),
                _row("GOLD", "up", 0.8),
            ],
        )
    )

    prompt = build_multiformat_prompt([bundle])

    assert "Generate genuinely distinct narrative formats" in prompt
    assert "sparse_user_prompt must mention only one or two channels" in prompt
    assert "scenario description, not an instruction or question" in prompt
    assert "institutional_risk_committee_note" in prompt
    assert "not always a stress scenario" in prompt
    assert "choose scenario_title from the evidence" in prompt
    assert "Do not derive the formats by copying the same factor list" in prompt
    assert "full_risk_manager_memo" in prompt


def test_multiformat_card_uses_direct_codex_authored_view_text() -> None:
    narrative = MultiFormatNarrative(
        window_id="joint39_train_0001",
        scenario_title="Volatility-Led Credit Stress",
        archetype="financial_accident",
        archetype_confidence="high",
        old_factor_baseline="Mechanical baseline: VIX higher and credit wider.",
        sparse_user_prompt="Volatility is driving a broader credit-risk repricing.",
        weekly_risk_monitor="Risk tone deteriorated as volatility and lower-quality credit led the move.",
        institutional_risk_committee_note="Risk committee view: this prefix would put short-volatility and credit-risk inventory under review.",
        mechanism_first_memo="The mechanism is forced de-risking through hedging demand and wider balance-sheet risk premia.",
        full_risk_manager_memo="Regime: volatility-led credit stress. The current prefix is defensive and liquidity-sensitive, not a forecast.",
        evidence_used=["VIX up", "BBB wider"],
        ambiguity_flags=["Safe-haven confirmation is incomplete."],
        no_forecast_caveat="This is current/recent conditioning context only.",
        contrastive_hard_negatives=["Calm risk-on relief with volatility lower."],
        quality_self_critique=["Sparse view intentionally names only two channels."],
    )

    card = multiformat_narrative_to_episode_card(narrative, source_path="x.jsonl")

    assert card["narrative_authoring"] == "direct_codex_multiformat"
    assert card["valid_for_training_retrieval"] is True
    assert card["views"]["sparse_user_query"] == narrative.sparse_user_prompt
    assert card["views"]["full_professional"] == narrative.full_risk_manager_memo
    assert card["views"]["weekly_risk_monitor"] == narrative.weekly_risk_monitor
    assert "Generated from" not in card["views"]["full_professional"]


def test_multiformat_review_uses_independent_generated_formats() -> None:
    case = {
        "window_id": "joint39_train_0001",
        "history_start": "2020-01-01",
        "history_end": "2020-02-14",
        "market_implications": [
            {
                "market": "SPX",
                "direction": "down",
                "magnitude": "large",
                "confidence": "high",
                "evidence": ["raw_change=-100"],
            },
            {
                "market": "VIX",
                "direction": "up",
                "magnitude": "large",
                "confidence": "high",
                "evidence": ["raw_change=10"],
            },
        ],
    }
    narrative = MultiFormatNarrative(
        window_id="joint39_train_0001",
        scenario_title="Liquidity Stress With Equity De-Risking",
        archetype="financial_accident",
        archetype_confidence="high",
        old_factor_baseline="SPX down, VIX up, credit wider.",
        sparse_user_prompt="Markets feel like a liquidity squeeze with volatility back in control.",
        weekly_risk_monitor="Risk tone deteriorated as volatility led the tape and liquidity proxies weakened.",
        institutional_risk_committee_note="Risk committee view: the current prefix would be treated as a liquidity-driven stress state with equity beta, short-volatility, and funding-sensitive exposures under review.",
        mechanism_first_memo="The main mechanism is balance-sheet pressure rather than a clean macro growth story.",
        full_risk_manager_memo="A risk manager would frame this as a liquidity stress state with equity beta and short-volatility exposure most exposed.",
        evidence_used=["SPX down", "VIX up"],
        ambiguity_flags=["No catalyst supplied."],
        no_forecast_caveat="This is not a forecast; it describes the current/recent prefix.",
        contrastive_hard_negatives=[
            "Calm risk-on relief with equities higher and volatility lower."
        ],
        quality_self_critique=[
            "Sparse prompt intentionally avoids listing every factor."
        ],
    )

    markdown = render_multiformat_one_page_review(
        narrative=narrative,
        selected_case=case,
    )

    assert "Markets feel like a liquidity squeeze" in markdown
    assert "Risk committee view" in markdown
    assert "Institutional risk-committee note" in markdown
    assert "balance-sheet pressure" in markdown
    assert "SPX down, VIX up, credit wider" in markdown
    assert markdown.index("Markets feel like") < markdown.index("A risk manager")


def test_sparse_user_prompt_rejects_instruction_like_query_text() -> None:
    narrative = MultiFormatNarrative(
        window_id="joint39_train_0001",
        scenario_title="Volatility-Led Credit Stress",
        archetype="financial_accident",
        archetype_confidence="high",
        old_factor_baseline="Mechanical factor baseline: VIX up and credit wider.",
        sparse_user_prompt=(
            "Vol is exploding and credit feels like it is starting to gap. "
            "Give me a current risk read on whether this is just equity noise."
        ),
        weekly_risk_monitor="Risk tone deteriorated as volatility led the tape.",
        institutional_risk_committee_note="Risk committee view: volatility and lower-quality credit would put equity beta and short-volatility books under review.",
        mechanism_first_memo="The mechanism is forced de-risking through volatility and credit.",
        full_risk_manager_memo=(
            "Regime: volatility-led credit stress. Transmission runs through "
            "hedging demand, lower risk appetite, and wider compensation for "
            "balance-sheet risk. This describes only the current prefix."
        ),
        evidence_used=["VIX up", "BBB wider"],
        ambiguity_flags=["Gold confirmation is limited."],
        no_forecast_caveat="Current/recent prefix description only.",
        contrastive_hard_negatives=["Calm risk-on relief with volatility lower."],
        quality_self_critique=["Sparse text is expected to be declarative."],
    )

    issues = _validate_multiformat(narrative)

    assert any(
        issue["code"] == "sparse_prompt_instruction_like"
        and issue["severity"] == "error"
        for issue in issues
    )


def test_sparse_user_prompt_rejects_internal_system_references() -> None:
    narrative = MultiFormatNarrative(
        window_id="joint39_train_0001",
        scenario_title="Equity Risk-Off",
        archetype="financial_accident",
        archetype_confidence="high",
        old_factor_baseline="Mechanical factor baseline: equity lower and volatility higher.",
        sparse_user_prompt=(
            "Equity risk has broken lower and option volatility has repriced "
            "higher. Confirmation is deliberately left to the support system."
        ),
        weekly_risk_monitor="Risk tone deteriorated as volatility led the tape.",
        institutional_risk_committee_note="Risk committee view: the current prefix is an equity-volatility stress state with hedging demand elevated.",
        mechanism_first_memo="The mechanism is forced de-risking through volatility.",
        full_risk_manager_memo=(
            "Regime: equity risk-off. Transmission runs through hedging demand "
            "and lower risk appetite. This describes only the current prefix."
        ),
        evidence_used=["SPX down", "VIX up"],
        ambiguity_flags=["Credit confirmation is mixed."],
        no_forecast_caveat="Current/recent prefix description only.",
        contrastive_hard_negatives=["Calm risk-on relief with volatility lower."],
        quality_self_critique=["Sparse text should avoid internal product language."],
    )

    issues = _validate_multiformat(narrative)

    assert any(
        issue["code"] == "sparse_prompt_internal_reference"
        and issue["severity"] == "error"
        for issue in issues
    )


def test_no_forecast_caveat_with_next_30_days_is_not_leakage() -> None:
    narrative = MultiFormatNarrative(
        window_id="joint39_train_0001",
        scenario_title="Mixed Relief",
        archetype="mixed_ambiguous",
        archetype_confidence="medium",
        old_factor_baseline="Mechanical factor baseline: equity higher and gold higher.",
        sparse_user_prompt=(
            "Equities are firmer while gold remains bid. The current tape is "
            "constructive but internally mixed."
        ),
        weekly_risk_monitor="Risk tone improved, but safe-haven demand remains visible.",
        institutional_risk_committee_note="Risk committee view: the current prefix is not a clean stress state because risk appetite and hedge demand coexist.",
        mechanism_first_memo="The mechanism is liquidity relief with unresolved hedge demand.",
        full_risk_manager_memo=(
            "Regime: mixed relief. The current prefix is internally conflicted "
            "and does not forecast the next 30 days."
        ),
        evidence_used=["SPX up", "GOLD up"],
        ambiguity_flags=["Gold confirmation complicates the risk-on read."],
        no_forecast_caveat="Current/recent prefix description only.",
        contrastive_hard_negatives=[
            "Broad liquidation with equities lower and volatility higher."
        ],
        quality_self_critique=["No-forecast caveat is explicit."],
    )

    issues = _validate_multiformat(narrative)

    assert not any(issue["code"] == "future_leakage" for issue in issues)


def test_sparse_user_prompt_warns_on_vague_ambiguity_language() -> None:
    narrative = MultiFormatNarrative(
        window_id="joint39_train_0001",
        scenario_title="Risk-Off With Uneven Confirmation",
        archetype="financial_accident",
        archetype_confidence="high",
        old_factor_baseline="Mechanical factor baseline: equity lower and volatility higher.",
        sparse_user_prompt=(
            "Equity volatility is in a stress regime and lower-quality credit "
            "is repricing wider. Not every safe-haven signal is perfectly aligned."
        ),
        weekly_risk_monitor="Risk tone deteriorated as volatility led the tape.",
        institutional_risk_committee_note="Risk committee view: volatility and credit stress would put equity beta and credit-risk exposures under review.",
        mechanism_first_memo="The mechanism is forced de-risking through volatility.",
        full_risk_manager_memo=(
            "Regime: equity risk-off. Transmission runs through hedging demand "
            "and lower risk appetite. This describes only the current prefix."
        ),
        evidence_used=["SPX down", "VIX up"],
        ambiguity_flags=["Credit confirmation is mixed."],
        no_forecast_caveat="Current/recent prefix description only.",
        contrastive_hard_negatives=["Calm risk-on relief with volatility lower."],
        quality_self_critique=["Sparse text should name ambiguity concretely."],
    )

    issues = _validate_multiformat(narrative)

    assert any(
        issue["code"] == "sparse_prompt_vague_ambiguity"
        and issue["severity"] == "warning"
        for issue in issues
    )
