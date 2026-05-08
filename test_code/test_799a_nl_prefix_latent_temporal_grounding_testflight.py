import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_temporal_grounding_testflight import (
    PROMPT_VERSION,
    ConditionOnlyGroundingResult,
    build_condition_only_grounding_messages,
    condition_query_text_from_grounding,
    run_condition_only_grounding_testflight,
    split_story_for_conditioning,
    validate_condition_only_grounding_result,
)


def _condition_grounding() -> ConditionOnlyGroundingResult:
    return ConditionOnlyGroundingResult.model_validate(
        {
            "prompt_version": PROMPT_VERSION,
            "narrative_frame": "fragile risk-on rebound",
            "current_market_state_summary": (
                "Equities are recovering, volatility is compressing, and spreads "
                "are stabilizing."
            ),
            "recent_regime_summary": (
                "The recent regime has moved from stress toward a fragile "
                "risk-on recovery."
            ),
            "cleaned_conditioning_text": (
                "Current support is a fragile risk-on rebound with equities "
                "recovering, volatility compressing, and credit spreads stabilizing."
            ),
            "current_market_state_implications": [
                {
                    "market": "SPX",
                    "direction": "up",
                    "magnitude": "medium",
                    "confidence": "high",
                    "horizon": "current_state",
                    "target_use": "support_prior",
                    "evidence": ["equities are recovering"],
                    "inferred": False,
                    "rationale": "The story states equities are recovering.",
                },
                {
                    "market": "VIX",
                    "direction": "down",
                    "magnitude": "medium",
                    "confidence": "high",
                    "horizon": "current_state",
                    "target_use": "support_prior",
                    "evidence": ["volatility is compressing"],
                    "inferred": False,
                    "rationale": "The story states volatility is compressing.",
                },
            ],
            "recent_regime_implications": [
                {
                    "market": "BBB_OAS",
                    "direction": "tighter",
                    "magnitude": "small",
                    "confidence": "medium",
                    "horizon": "recent_regime",
                    "target_use": "support_prior",
                    "evidence": ["spreads are stabilizing"],
                    "inferred": True,
                    "rationale": "Stabilizing spreads indicate a recent risk-on regime.",
                }
            ],
            "non_conditioning_forward_language": [
                {
                    "phrase": "the forward risk is that volatility reverses",
                    "reason": "This describes a possible future, not a condition.",
                    "handling": "warning_only",
                    "severity": "warning",
                }
            ],
            "unsupported_claims": [],
            "grounding_warnings": [
                {
                    "code": "FORWARD_LANGUAGE_IGNORED",
                    "severity": "warning",
                    "message": (
                        "Forward risk language is recorded but not used as a "
                        "scenario target."
                    ),
                    "target_use": "ignore",
                }
            ],
            "critique": ["No future path was converted into a generation target."],
        }
    )


def test_build_condition_only_grounding_messages_rejects_future_targets() -> None:
    messages = build_condition_only_grounding_messages("A risk story.")
    joined = "\n".join(message["content"] for message in messages)

    assert "ConditionOnlyGroundingResult" in joined
    assert "current_market_state_implications" in joined
    assert "Conditioning candidate sentences" in joined
    assert "must be exactly one of" in joined
    assert "Do not create forward_scenario_implications" in joined
    assert "Future-looking phrases" in joined


def test_split_story_for_conditioning_moves_future_sentence_to_warning_only() -> None:
    split = split_story_for_conditioning(
        "Equities are recovering. The next month risk is a volatility reversal."
    )

    assert split["conditioning_sentences"] == ["Equities are recovering."]
    assert split["non_conditioning_forward_sentences"] == [
        "The next month risk is a volatility reversal."
    ]


def test_validate_condition_only_grounding_result_accepts_conditioning_only() -> None:
    validation = validate_condition_only_grounding_result(_condition_grounding())

    assert validation["status"] == "pass"
    assert validation["condition_implication_count"] == 3
    assert validation["current_support_count"] == 2
    assert validation["recent_regime_count"] == 1
    assert validation["forward_warning_count"] == 1
    assert validation["future_target_count"] == 0
    assert validation["condition_role_error_count"] == 0
    assert validation["forward_warning_leakage_count"] == 0


def test_validate_condition_only_grounding_result_flags_forward_leakage() -> None:
    broken = _condition_grounding().model_copy(deep=True)
    broken.current_market_state_implications[0].evidence = ["SPX could rise next month"]

    validation = validate_condition_only_grounding_result(broken)

    assert validation["status"] == "warning"
    assert "condition_role_errors" in validation["reasons"]
    assert validation["condition_role_error_count"] == 1
    assert validation["condition_role_errors"][0]["field"] == "forward_language_leakage"


def test_validate_condition_only_grounding_result_flags_unsupported_market() -> None:
    broken = _condition_grounding().model_copy(deep=True)
    broken.current_market_state_implications[0].market = "credit spreads"

    validation = validate_condition_only_grounding_result(broken)

    assert validation["status"] == "warning"
    assert validation["condition_role_error_count"] == 1
    assert validation["condition_role_errors"][0]["field"] == "market"


def test_validate_condition_only_grounding_result_flags_warning_phrase_reuse() -> None:
    broken = _condition_grounding().model_copy(deep=True)
    broken.non_conditioning_forward_language[0].phrase = (
        "funding stress keeps forcing de-risking across high-beta assets"
    )
    broken.cleaned_conditioning_text += (
        " Funding stress keeps forcing de-risking across high-beta assets."
    )

    validation = validate_condition_only_grounding_result(broken)

    assert validation["status"] == "warning"
    assert "forward_warning_leakage" in validation["reasons"]
    assert validation["forward_warning_leakage_count"] == 1
    assert (
        validation["forward_warning_leakage"][0]["field"] == "cleaned_conditioning_text"
    )


def test_condition_query_text_excludes_future_target_section() -> None:
    text = condition_query_text_from_grounding("Story text.", _condition_grounding())

    assert "NARRATIVE:" not in text
    assert "CONDITIONING_FRAME:" in text
    assert "CURRENT_SUPPORT_IMPLICATIONS:" in text
    assert "RECENT_REGIME_IMPLICATIONS:" in text
    assert "NON_CONDITIONING_FORWARD_LANGUAGE:" not in text
    assert "BASE_FORWARD_IMPLICATIONS" not in text
    assert "STRESS_ALTERNATIVES" not in text
    assert "current: SPX up medium" in text


def test_run_condition_only_grounding_testflight_replays_fixture(tmp_path) -> None:
    fixture = {
        "cases": [
            {
                "case_name": "fragile_risk_on_rebound",
                "condition_only_grounding": _condition_grounding().model_dump(),
            }
        ]
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        case_count=1,
        case_name=None,
        model="fixture",
        max_output_tokens=10,
        dotenv=".env",
        grounding_json=str(fixture_path),
    )

    summary = run_condition_only_grounding_testflight(args)

    assert summary["case_count"] == 1
    assert summary["status_counts"] == {"pass": 1}
    assert summary["totals"]["future_target_count"] == 0
    assert summary["totals"]["condition_implication_count"] == 3
    assert summary["totals"]["forward_warning_leakage_count"] == 0
    assert summary["cases"][0]["story_split"]["non_conditioning_forward_sentences"]
    assert (tmp_path / "out" / "condition_only_grounding_summary.json").exists()
    assert (
        tmp_path
        / "out"
        / "01_fragile_risk_on_rebound"
        / "condition_only_query_text.txt"
    ).exists()


def test_run_condition_only_grounding_testflight_selects_named_case(tmp_path) -> None:
    fixture = {
        "cases": [
            {
                "case_name": "commodity_inflation_pressure",
                "condition_only_grounding": _condition_grounding().model_dump(),
            }
        ]
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"),
        case_count=1,
        case_name=["commodity_inflation_pressure"],
        model="fixture",
        max_output_tokens=10,
        dotenv=".env",
        grounding_json=str(fixture_path),
    )

    summary = run_condition_only_grounding_testflight(args)

    assert summary["case_count"] == 1
    assert summary["selected_case_names"] == ["commodity_inflation_pressure"]
    assert summary["cases"][0]["case_name"] == "commodity_inflation_pressure"
    assert (
        tmp_path
        / "out"
        / "01_commodity_inflation_pressure"
        / "condition_only_query_text.txt"
    ).exists()
