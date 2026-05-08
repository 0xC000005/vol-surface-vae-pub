#!/usr/bin/env python
"""OpenAI TestFlight for condition-only narrative grounding.

The production contract is intentionally conservative: a risk-manager narrative
describes the current/recent conditioning regime, not the desired future path.
Future-looking phrases are warnings only and must not become scenario targets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_live_casebook import (  # noqa: E402
    default_casebook_stories,
)
from experiments.backfill.block_ar.nl_scenario_descriptions import (  # noqa: E402
    load_dotenv_key,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_grounding_testflight_808a"
)
DEFAULT_MODEL = "gpt-5.4-mini"
PROMPT_VERSION = "condition_only_grounding_v1"
SUPPORTED_MARKETS = {
    "AAA_OAS",
    "BBB_OAS",
    "CRUDE_OIL",
    "DXY",
    "GOLD",
    "IV_SKEW",
    "IV_SURFACE",
    "SPX",
    "US10Y",
    "US2Y",
    "USDJPY",
    "VIX",
}
FORWARD_LANGUAGE_MARKERS = {
    "could",
    "forecast",
    "forward",
    "future",
    "going to",
    "may",
    "might",
    "next",
    "outlook",
    "projection",
    "risk is",
    "scenario",
    "should",
    "will",
    "would",
}
LEAKAGE_STOPWORDS = {
    "a",
    "across",
    "after",
    "and",
    "are",
    "assets",
    "be",
    "into",
    "is",
    "it",
    "keeps",
    "month",
    "risk",
    "that",
    "the",
    "this",
    "toward",
    "with",
}


class ConditionMarketImplication(BaseModel):
    """One current/recent market implication allowed into conditioning."""

    model_config = ConfigDict(extra="forbid")

    market: str = Field(min_length=1)
    direction: Literal[
        "up",
        "down",
        "flat",
        "wider",
        "tighter",
        "mixed",
        "unclear",
    ]
    magnitude: Literal["small", "medium", "large", "flat", "unclear"]
    confidence: Literal["low", "medium", "high"]
    horizon: Literal["current_state", "recent_regime"]
    target_use: Literal["support_prior"] = "support_prior"
    evidence: list[str] = Field(default_factory=list)
    inferred: bool = False
    rationale: str = Field(min_length=1)


class NonConditioningForwardLanguage(BaseModel):
    """Future-looking text that must not be used as a generation target."""

    model_config = ConfigDict(extra="forbid")

    phrase: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    handling: Literal["warning_only", "ignore_for_conditioning"] = "warning_only"
    severity: Literal["info", "warning"] = "warning"


class ConditionGroundingWarning(BaseModel):
    """Warning for unsupported or non-conditioning language."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    severity: Literal["info", "warning", "error"]
    message: str = Field(min_length=1)
    target_use: Literal["support_prior", "ignore"] = "ignore"


class ConditionOnlyGroundingResult(BaseModel):
    """Condition-only grounding result for a risk-manager narrative."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: Literal["condition_only_grounding_v1"] = PROMPT_VERSION
    narrative_frame: str = Field(min_length=1)
    current_market_state_summary: str = Field(min_length=1)
    recent_regime_summary: str = Field(min_length=1)
    cleaned_conditioning_text: str = Field(min_length=12)
    current_market_state_implications: list[ConditionMarketImplication] = Field(
        default_factory=list
    )
    recent_regime_implications: list[ConditionMarketImplication] = Field(
        default_factory=list
    )
    non_conditioning_forward_language: list[NonConditioningForwardLanguage] = Field(
        default_factory=list
    )
    unsupported_claims: list[str] = Field(default_factory=list)
    grounding_warnings: list[ConditionGroundingWarning] = Field(default_factory=list)
    critique: list[str] = Field(default_factory=list)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _slug(text: str) -> str:
    keep: list[str] = []
    for char in str(text).lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_") or "case"


def _has_forward_language(text: str) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in FORWARD_LANGUAGE_MARKERS)


def split_story_for_conditioning(story: str) -> dict[str, Any]:
    """Split a story into conditioning and warning-only sentence candidates."""

    raw_sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", str(story).strip())
        if item.strip()
    ]
    if not raw_sentences and str(story).strip():
        raw_sentences = [str(story).strip()]
    conditioning: list[str] = []
    non_conditioning: list[str] = []
    for sentence in raw_sentences:
        if _has_forward_language(sentence):
            non_conditioning.append(sentence)
        else:
            conditioning.append(sentence)
    if not conditioning and raw_sentences:
        conditioning = [
            sentence for sentence in raw_sentences if sentence not in non_conditioning
        ]
    return {
        "conditioning_sentences": conditioning,
        "non_conditioning_forward_sentences": non_conditioning,
        "sentence_count": len(raw_sentences),
    }


def _content_tokens(text: str) -> set[str]:
    token: list[str] = []
    tokens: set[str] = set()
    for char in str(text).lower():
        if char.isalnum() or char == "-":
            token.append(char)
        else:
            if token:
                value = "".join(token)
                if len(value) >= 4 and value not in LEAKAGE_STOPWORDS:
                    tokens.add(value)
                token = []
    if token:
        value = "".join(token)
        if len(value) >= 4 and value not in LEAKAGE_STOPWORDS:
            tokens.add(value)
    return tokens


def build_condition_only_grounding_messages(story: str) -> list[dict[str, str]]:
    """Build the prompt for condition-only story grounding."""

    split = split_story_for_conditioning(story)
    conditioning_text = "\n".join(
        f"- {sentence}" for sentence in split["conditioning_sentences"]
    )
    if not conditioning_text:
        conditioning_text = "- none"
    non_conditioning_text = "\n".join(
        f"- {sentence}"
        for sentence in split["non_conditioning_forward_sentences"]
    )
    if not non_conditioning_text:
        non_conditioning_text = "- none"
    system = (
        "You convert risk-manager narratives into auditable conditioning facts "
        "for a financial scenario generator. The narrative is not a requested "
        "future path. Extract only current or recent-regime market conditions. "
        "Future-looking phrases, stress forecasts, or desired outcomes must be "
        "reported as warnings and ignored for conditioning."
    )
    user = (
        "Return a ConditionOnlyGroundingResult JSON object. The market field "
        "must be exactly one of these supported names when an implication is "
        f"created: {', '.join(sorted(SUPPORTED_MARKETS))}. For generic credit "
        "spreads, prefer BBB_OAS as a broad proxy and add a proxy warning.\n\n"
        "Rules:\n"
        "1. current_market_state_implications are only for stated present "
        "conditions. Set target_use=support_prior and horizon=current_state.\n"
        "2. recent_regime_implications are only for stated recent path or "
        "regime descriptions. Set target_use=support_prior and "
        "horizon=recent_regime.\n"
        "3. Do not create forward_scenario_implications. That field does not "
        "exist. The generator, not the user narrative, determines possible "
        "future realizations.\n"
        "4. Phrases like 'the risk is', 'could', 'may', 'will', 'next month', "
        "'forward risk', or desired future outcomes are not conditioning facts. "
        "Put them in non_conditioning_forward_language and, if needed, "
        "grounding_warnings.\n"
        "5. Do not reuse forward-looking phrase content in "
        "current_market_state_summary, recent_regime_summary, "
        "cleaned_conditioning_text, or market implications unless the same "
        "condition is independently stated as current/recent outside the "
        "forward-looking phrase.\n"
        "6. Build current_market_state_summary, recent_regime_summary, and "
        "cleaned_conditioning_text only from the Conditioning candidate "
        "sentences. If no separate recent-regime path is stated there, write "
        "'No separate recent-regime description stated beyond current "
        "conditions.' Do not mention risk appetite, funding stress, de-risking, "
        "or growth repricing unless those exact ideas appear in the "
        "Conditioning candidate sentences.\n"
        "7. Direction should be up/down/flat for prices, rates, volatility, FX, "
        "and commodities, or wider/tighter/flat for credit spreads. If the "
        "direction is not current/recent enough, use a warning instead of an "
        "implication.\n"
        "8. Keep broad terms such as carry, liquidity, risk appetite, safe haven, "
        "or funding stress as frame language unless a named market implication "
        "is stated or strongly implied by current/recent language.\n\n"
        "Conditioning candidate sentences. Extract implications only from this "
        f"section:\n{conditioning_text}\n\n"
        "Warning-only forward or forecast sentences. Report these in "
        "non_conditioning_forward_language, but do not use their content in "
        "conditioning summaries or implications:\n"
        f"{non_conditioning_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ground_condition_only_story_with_openai(
    story: str,
    *,
    model: str = DEFAULT_MODEL,
    dotenv_path: str | Path = ".env",
    max_output_tokens: int = 1800,
    client: Any | None = None,
) -> tuple[ConditionOnlyGroundingResult, dict[str, Any]]:
    """Call OpenAI structured outputs for one condition-only grounding."""

    load_dotenv_key(dotenv_path)
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=build_condition_only_grounding_messages(story),
        text_format=ConditionOnlyGroundingResult,
        max_output_tokens=int(max_output_tokens),
    )
    parsed = response.output_parsed
    if not isinstance(parsed, ConditionOnlyGroundingResult):
        raise TypeError("OpenAI did not return a ConditionOnlyGroundingResult")
    usage = getattr(response, "usage", None)
    metadata = {
        "model": str(model),
        "prompt_version": PROMPT_VERSION,
        "response_id": str(getattr(response, "id", "")),
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
    }
    return parsed, metadata


def _all_condition_implications(
    result: ConditionOnlyGroundingResult,
) -> list[ConditionMarketImplication]:
    return [
        *result.current_market_state_implications,
        *result.recent_regime_implications,
    ]


def _forward_phrase_leakage_errors(
    result: ConditionOnlyGroundingResult,
) -> list[dict[str, Any]]:
    """Detect warning-only phrase content leaking back into conditioning text."""

    conditioning_fields = {
        "current_market_state_summary": result.current_market_state_summary,
        "recent_regime_summary": result.recent_regime_summary,
        "cleaned_conditioning_text": result.cleaned_conditioning_text,
    }
    for index, item in enumerate(_all_condition_implications(result)):
        conditioning_fields[f"condition_implication_{index}_evidence"] = " ".join(
            item.evidence
        )
        conditioning_fields[f"condition_implication_{index}_rationale"] = item.rationale
    errors: list[dict[str, Any]] = []
    for warning_index, warning in enumerate(result.non_conditioning_forward_language):
        phrase_tokens = _content_tokens(warning.phrase)
        if not phrase_tokens:
            continue
        for field_name, field_text in conditioning_fields.items():
            field_tokens = _content_tokens(field_text)
            overlap = sorted(phrase_tokens & field_tokens)
            threshold = 2
            if len(overlap) >= threshold:
                errors.append(
                    {
                        "warning_index": warning_index,
                        "field": field_name,
                        "overlap_tokens": overlap,
                        "phrase": warning.phrase,
                        "expected": (
                            "warning-only forward phrase should not re-enter "
                            "conditioning fields"
                        ),
                    }
                )
    return errors


def _implication_errors(
    implications: list[ConditionMarketImplication],
    *,
    allowed_horizons: set[str],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(implications):
        if item.market.upper() not in SUPPORTED_MARKETS:
            errors.append(
                {
                    "index": index,
                    "market": item.market,
                    "field": "market",
                    "value": item.market,
                    "expected": sorted(SUPPORTED_MARKETS),
                }
            )
        if item.target_use != "support_prior":
            errors.append(
                {
                    "index": index,
                    "market": item.market,
                    "field": "target_use",
                    "value": item.target_use,
                    "expected": "support_prior",
                }
            )
        if item.horizon not in allowed_horizons:
            errors.append(
                {
                    "index": index,
                    "market": item.market,
                    "field": "horizon",
                    "value": item.horizon,
                    "expected": sorted(allowed_horizons),
                }
            )
        if not item.evidence:
            errors.append(
                {
                    "index": index,
                    "market": item.market,
                    "field": "evidence",
                    "value": [],
                    "expected": "one or more current/recent evidence snippets",
                }
            )
        joined = " ".join([*item.evidence, item.rationale])
        if _has_forward_language(joined):
            errors.append(
                {
                    "index": index,
                    "market": item.market,
                    "field": "forward_language_leakage",
                    "value": joined,
                    "expected": "future-looking language should be warning-only",
                }
            )
    return errors


def validate_condition_only_grounding_result(
    result: ConditionOnlyGroundingResult,
) -> dict[str, Any]:
    """Validate whether the schema keeps conditioning separate from futures."""

    current_errors = _implication_errors(
        result.current_market_state_implications,
        allowed_horizons={"current_state"},
    )
    recent_errors = _implication_errors(
        result.recent_regime_implications,
        allowed_horizons={"recent_regime"},
    )
    condition_count = len(_all_condition_implications(result))
    forward_warning_count = len(result.non_conditioning_forward_language)
    unsupported_count = len(result.unsupported_claims)
    grounding_warning_count = len(result.grounding_warnings)
    leakage_errors = _forward_phrase_leakage_errors(result)
    errors = current_errors + recent_errors
    status = "pass"
    reasons: list[str] = []
    if condition_count == 0:
        status = "warning"
        reasons.append("no_conditioning_implications")
    if errors:
        status = "warning"
        reasons.append("condition_role_errors")
    if leakage_errors:
        status = "warning"
        reasons.append("forward_warning_leakage")
    return {
        "status": status,
        "reasons": reasons,
        "condition_implication_count": condition_count,
        "current_support_count": len(result.current_market_state_implications),
        "recent_regime_count": len(result.recent_regime_implications),
        "forward_warning_count": forward_warning_count,
        "unsupported_claim_count": unsupported_count,
        "grounding_warning_count": grounding_warning_count,
        "condition_role_error_count": len(errors),
        "condition_role_errors": errors,
        "forward_warning_leakage_count": len(leakage_errors),
        "forward_warning_leakage": leakage_errors,
        "future_target_count": 0,
    }


def condition_query_text_from_grounding(
    story: str,
    result: ConditionOnlyGroundingResult,
) -> str:
    """Build embedding candidate text using conditioning fields only."""

    def format_implication(prefix: str, item: ConditionMarketImplication) -> str:
        evidence = "; ".join(item.evidence) if item.evidence else "no direct quote"
        inferred = " inferred" if item.inferred else ""
        return (
            f"{prefix}: {item.market} {item.direction} {item.magnitude} "
            f"confidence={item.confidence} horizon={item.horizon} "
            f"target={item.target_use}{inferred} evidence={evidence}"
        )

    lines = [
        f"CONDITIONING_FRAME: {result.narrative_frame}",
        f"CURRENT_MARKET_STATE_SUMMARY: {result.current_market_state_summary}",
        f"RECENT_REGIME_SUMMARY: {result.recent_regime_summary}",
        f"CLEANED_CONDITIONING_TEXT: {result.cleaned_conditioning_text}",
        "CURRENT_SUPPORT_IMPLICATIONS:",
    ]
    lines.extend(
        format_implication("current", item)
        for item in result.current_market_state_implications
    )
    lines.append("RECENT_REGIME_IMPLICATIONS:")
    lines.extend(
        format_implication("recent", item) for item in result.recent_regime_implications
    )
    return "\n".join(lines)


def _load_fixture_results(path: str | Path) -> dict[str, ConditionOnlyGroundingResult]:
    payload = _load_json(path)
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError(f"{path}: expected cases list")
    results: dict[str, ConditionOnlyGroundingResult] = {}
    for row in raw_cases:
        if not isinstance(row, dict):
            continue
        name = str(row.get("case_name", ""))
        grounding = row.get("condition_only_grounding") or row.get(
            "temporal_grounding"
        )
        if name and isinstance(grounding, dict):
            results[name] = ConditionOnlyGroundingResult.model_validate(grounding)
    return results


def run_condition_only_grounding_testflight(
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stories = default_casebook_stories()[: int(args.case_count)]
    fixture_results = (
        _load_fixture_results(args.grounding_json) if args.grounding_json else {}
    )
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for case_no, item in enumerate(stories, start=1):
        case_name = str(item["name"])
        story = str(item["story"])
        metadata: dict[str, Any]
        if case_name in fixture_results:
            grounding = fixture_results[case_name]
            metadata = {
                "model": "fixture",
                "prompt_version": grounding.prompt_version,
                "response_id": "",
                "usage": None,
            }
        else:
            grounding, metadata = ground_condition_only_story_with_openai(
                story,
                model=str(args.model),
                dotenv_path=args.dotenv,
                max_output_tokens=int(args.max_output_tokens),
            )
        validation = validate_condition_only_grounding_result(grounding)
        query_text = condition_query_text_from_grounding(story, grounding)
        story_split = split_story_for_conditioning(story)
        case_dir = output_dir / f"{case_no:02d}_{_slug(case_name)}"
        case_payload = {
            "case_name": case_name,
            "story": story,
            "story_split": story_split,
            "condition_only_grounding": grounding.model_dump(),
            "condition_only_validation": validation,
            "candidate_query_text": query_text,
            "metadata": metadata,
            "artifact_paths": {
                "case_json": str(case_dir / "condition_only_grounding_case.json"),
                "query_text": str(case_dir / "condition_only_query_text.txt"),
            },
        }
        _write_json(case_dir / "condition_only_grounding_case.json", case_payload)
        (case_dir / "condition_only_query_text.txt").write_text(
            query_text.rstrip() + "\n",
            encoding="utf-8",
        )
        status = str(validation["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        rows.append(case_payload)
    totals = {
        "condition_implication_count": sum(
            row["condition_only_validation"]["condition_implication_count"]
            for row in rows
        ),
        "current_support_count": sum(
            row["condition_only_validation"]["current_support_count"] for row in rows
        ),
        "recent_regime_count": sum(
            row["condition_only_validation"]["recent_regime_count"] for row in rows
        ),
        "forward_warning_count": sum(
            row["condition_only_validation"]["forward_warning_count"] for row in rows
        ),
        "condition_role_error_count": sum(
            row["condition_only_validation"]["condition_role_error_count"]
            for row in rows
        ),
        "forward_warning_leakage_count": sum(
            row["condition_only_validation"]["forward_warning_leakage_count"]
            for row in rows
        ),
        "future_target_count": 0,
    }
    summary = {
        "status": "ok",
        "scope_note": (
            "OpenAI condition-only grounding TestFlight for the narrative "
            "prefix-latent program. Future-looking language is warning-only; "
            "the generated distribution is not forced toward requested futures."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": str(args.model),
        "prompt_version": PROMPT_VERSION,
        "case_count": len(rows),
        "status_counts": status_counts,
        "totals": totals,
        "cases": rows,
        "artifact_paths": {
            "summary": str(output_dir / "condition_only_grounding_summary.json"),
        },
    }
    _write_json(output_dir / "condition_only_grounding_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-count", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument(
        "--grounding-json",
        help="Replay fixture summary instead of calling OpenAI for matching case names.",
    )
    args = parser.parse_args()
    summary = run_condition_only_grounding_testflight(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "case_count": summary["case_count"],
                "status_counts": summary["status_counts"],
                "totals": summary["totals"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
