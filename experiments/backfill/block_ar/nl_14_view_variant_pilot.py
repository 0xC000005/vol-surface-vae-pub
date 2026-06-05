#!/usr/bin/env python
"""Generate a one-episode 14-positive/14-negative narrative review pilot.

This pilot fixes the old eight-view positive aliasing at the sample level and
combines those eight unique views with six sparse user-like variants. Local code
selects structured negative candidates and validates output quality; Codex/GPT
authors all positive and negative narrative text.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_codex_caption_batch import (  # noqa: E402
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
)
from experiments.backfill.block_ar.nl_sparse_variant_pilot import (  # noqa: E402
    DEFAULT_CARDS_JSONL,
    DEFAULT_OUTPUT_DIR as SPARSE_DEFAULT_OUTPUT_DIR,
    DEFAULT_SUPPORT_CARDS_JSONL,
    DIRECT_NEGATION_PATTERNS,
    INTERNAL_PATTERNS,
    LEAKAGE_PATTERNS,
    _compact,
    _evidence_used,
    _extract_json_object,
    _pattern_hits,
    _phrases,
    _read_jsonl,
    _require_all_properties,
    _resolve,
    _support_date_lookup,
    _target_payload,
    _token_jaccard,
    _word_count,
    _write_json,
    _write_text,
    select_negative_candidates,
)
from experiments.backfill.block_ar.nl_hard_negative_bank_regenerate import (  # noqa: E402
    _contradiction_channels,
    _mechanical_summary,
    _sign_vector,
    _window_number,
)


DEFAULT_OUTPUT_DIR = SPARSE_DEFAULT_OUTPUT_DIR.parent / "fourteen_view_variant_pilot_986c"
DEFAULT_TARGET_WINDOW_ID = "joint39_train_1553"
OLD_EIGHT_VIEW_NAMES = (
    "sparse_user_query",
    "weekly_risk_monitor",
    "mechanism_first",
    "technical_factor_evidence",
    "factor_list_baseline",
    "institutional_risk_committee_note",
    "risk_manager_memo",
    "full_professional",
)
SPARSE_VARIANT_VIEW_NAMES = (
    "sparse_variant_tape_read",
    "sparse_variant_portfolio_concern",
    "sparse_variant_macro_channel",
    "sparse_variant_credit_ambiguity",
    "sparse_variant_rates_commodities",
    "sparse_variant_desk_note",
)
EXPECTED_VIEW_NAMES = OLD_EIGHT_VIEW_NAMES + SPARSE_VARIANT_VIEW_NAMES
VIEW_FOCUS_CHANNELS: dict[str, tuple[str, ...]] = {
    "sparse_user_query": ("DXY", "USDJPY", "CRUDE_OIL", "GOLD"),
    "weekly_risk_monitor": ("DXY", "USDJPY", "CRUDE_OIL", "GOLD", "AAA_OAS"),
    "mechanism_first": ("DXY", "CRUDE_OIL", "GOLD"),
    "technical_factor_evidence": (
        "DXY",
        "USDJPY",
        "CRUDE_OIL",
        "GOLD",
        "US2Y",
        "US10Y",
        "AAA_OAS",
        "BBB_OAS",
    ),
    "factor_list_baseline": ("DXY", "USDJPY", "CRUDE_OIL", "GOLD", "US2Y", "US10Y"),
    "institutional_risk_committee_note": (
        "DXY",
        "USDJPY",
        "CRUDE_OIL",
        "GOLD",
        "BBB_OAS",
        "AAA_OAS",
        "US2Y",
        "US10Y",
    ),
    "risk_manager_memo": ("DXY", "CRUDE_OIL", "GOLD", "BBB_OAS", "AAA_OAS"),
    "full_professional": (
        "DXY",
        "USDJPY",
        "CRUDE_OIL",
        "GOLD",
        "BBB_OAS",
        "AAA_OAS",
        "US2Y",
        "US10Y",
    ),
    "sparse_variant_tape_read": ("DXY", "USDJPY", "CRUDE_OIL", "GOLD"),
    "sparse_variant_portfolio_concern": (
        "DXY",
        "USDJPY",
        "CRUDE_OIL",
        "GOLD",
        "US2Y",
        "US10Y",
        "SPX",
    ),
    "sparse_variant_macro_channel": ("DXY", "CRUDE_OIL", "GOLD"),
    "sparse_variant_credit_ambiguity": ("BBB_OAS", "AAA_OAS", "US2Y", "US10Y"),
    "sparse_variant_rates_commodities": ("US2Y", "US10Y", "CRUDE_OIL", "GOLD"),
    "sparse_variant_desk_note": ("DXY", "USDJPY", "CRUDE_OIL", "GOLD"),
}
VIEW_MIN_FOCUS_CONTRADICTIONS: dict[str, int] = {
    view_name: 2 for view_name in EXPECTED_VIEW_NAMES
}
VIEW_MAX_TOKEN_JACCARD: dict[str, float] = {
    "technical_factor_evidence": 0.68,
    "factor_list_baseline": 1.01,
    "sparse_variant_desk_note": 0.70,
}
GENERIC_REUSABLE_TARGET_PHRASES = {
    "bbb credit",
    "credit stress",
    "high-grade spread",
}
GENERIC_ASSET_CLASS_MODIFIERS = {
    "high-grade",
    "investment-grade",
    "lower-quality",
}
GENERIC_ASSET_CLASS_HEADS = {
    "credit",
    "spread",
    "spreads",
}
PILOT_INTERNAL_PATTERNS = INTERNAL_PATTERNS + (
    re.compile(r"\bassigned\s+(?:prefix|window|candidate)\b", re.I),
    re.compile(r"\bsupplied\s+negative\s+candidate\b", re.I),
    re.compile(r"\bchosen\s+negative\s+candidate\b", re.I),
)
OPAQUE_SPREAD_MAGNITUDE_PATTERN = re.compile(
    r"\b(?:AAA|BBB)_OAS\s*[+-]\s*\d{3,}(?:\.\d+)?\b",
    re.I,
)


class NarrativePair(BaseModel):
    """One positive narrative and its same-style hard-negative narrative."""

    model_config = ConfigDict(extra="forbid")

    view_name: str = Field(min_length=3)
    positive_text: str = Field(min_length=8)
    negative_window_id: str = Field(min_length=3)
    negative_text: str = Field(min_length=8)
    quality_notes: list[str] = Field(default_factory=list)


class FourteenViewPilotBatch(BaseModel):
    """Strict Codex output schema for the 14-view pilot."""

    model_config = ConfigDict(extra="forbid")

    target_window_id: str = Field(min_length=3)
    target_title: str = Field(min_length=3)
    pairs: list[NarrativePair] = Field(default_factory=list)


def strict_schema() -> dict[str, Any]:
    return _require_all_properties(FourteenViewPilotBatch.model_json_schema())


def select_negative_candidates_for_fourteen_view(
    *,
    cards: list[dict[str, Any]],
    target_window_id: str,
    count: int,
    min_temporal_gap: int = 30,
) -> list[dict[str, Any]]:
    """Select a broader structured candidate pool for the 14-view pilot."""

    by_id = {str(card.get("window_id", "")): pos for pos, card in enumerate(cards)}
    if target_window_id not in by_id:
        raise ValueError(f"target_window_id not found: {target_window_id}")
    target_idx = by_id[target_window_id]
    signs = np.stack([_sign_vector(card) for card in cards], axis=0)
    target = signs[target_idx]
    products = signs * target[None, :]
    contradictions = (products < 0).sum(axis=1).astype(np.float32)
    agreements = (products > 0).sum(axis=1).astype(np.float32)
    target_number = _window_number(target_window_id)
    numbers = np.array([_window_number(str(card.get("window_id", ""))) for card in cards])
    temporal_ok = np.abs(numbers - target_number) >= int(min_temporal_gap)

    # Four contradictions tends to keep the candidate close enough to be a
    # hard negative while avoiding the old "only side-channel differs" failure.
    score = -3.0 * np.abs(contradictions - 4.0) + 0.35 * agreements
    score -= 4.0 * np.maximum(0.0, contradictions - 5.0)
    score[target_idx] = -1e9
    score[~temporal_ok] = -1e9
    score[contradictions < 3] = -1e9
    order = [int(idx) for idx in np.argsort(-score) if score[int(idx)] > -1e8]
    if len(order) < count:
        fallback = select_negative_candidates(
            cards=cards,
            target_window_id=target_window_id,
            count=count,
            min_temporal_gap=min_temporal_gap,
        )
        seen = {str(cards[idx].get("window_id", "")) for idx in order}
        for row in fallback:
            window_id = str(row["window_id"])
            if window_id in seen:
                continue
            if window_id in by_id:
                order.append(by_id[window_id])
                seen.add(window_id)
            if len(order) >= count:
                break

    candidates: list[dict[str, Any]] = []
    for idx in order[:count]:
        channels = _contradiction_channels(signs[target_idx], signs[idx])
        card = cards[idx]
        candidates.append(
            {
                "window_id": str(card.get("window_id", "")),
                "scenario_title": str(card.get("scenario_title", "")),
                "archetype": str(card.get("archetype", "")),
                "mechanical_summary": _mechanical_summary(card),
                "evidence_used": _evidence_used(card)[:6],
                "contradiction_channels": channels,
                "contradiction_count": len(channels),
                "agreement_count": int(agreements[idx]),
            }
        )
    return candidates


def _candidate_score_for_view(
    candidate: dict[str, Any],
    *,
    view_name: str,
    minimum_focus_contradictions: int,
    used_window_ids: set[str],
    used_window_numbers: list[int],
) -> float:
    window_id = str(candidate.get("window_id", ""))
    channels = set(candidate.get("contradiction_channels", []))
    focus_channels = set(VIEW_FOCUS_CHANNELS[view_name])
    focus_hits = len(channels & focus_channels)
    contradiction_count = int(candidate.get("contradiction_count", len(channels)))
    agreement_count = int(candidate.get("agreement_count", 0))
    if focus_hits < int(minimum_focus_contradictions):
        return -1e9
    score = 100.0 * focus_hits
    score += 12.0 * min(contradiction_count, 5)
    score += 1.5 * agreement_count
    score -= 7.0 * abs(contradiction_count - 4)
    score -= 14.0 * max(0, contradiction_count - 5)
    if window_id in used_window_ids:
        return -1e9
    number = _window_number(window_id)
    if used_window_numbers:
        nearest_gap = min(abs(number - used) for used in used_window_numbers)
        if nearest_gap <= 2:
            score -= 200.0
        elif nearest_gap <= 5:
            score -= 90.0
        elif nearest_gap <= 10:
            score -= 35.0
    return score


def effective_min_focus_contradictions(
    view_name: str,
    negative_candidates: list[dict[str, Any]],
) -> int:
    """Return the feasible focus-channel minimum for a view.

    Some target prefixes have flat/non-informative signs in part of a view's
    nominal focus set. In those cases, a two-channel focus contradiction can be
    impossible even with a large candidate pool. Relax only to the number of
    focus channels that actually appear among structured candidate
    contradictions, keeping at least one when any focus contradiction exists.
    """

    configured = int(VIEW_MIN_FOCUS_CONTRADICTIONS[view_name])
    focus_channels = set(VIEW_FOCUS_CHANNELS[view_name])
    max_candidate_focus_hits = 0
    for candidate in negative_candidates:
        max_candidate_focus_hits = max(
            max_candidate_focus_hits,
            len(set(candidate.get("contradiction_channels", [])) & focus_channels),
        )
    if max_candidate_focus_hits <= 0:
        return 0
    return max(1, min(configured, max_candidate_focus_hits))


def assign_negative_candidates_by_view(
    negative_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign one structured negative candidate to every view."""

    def valid_count(view_name: str) -> int:
        focus_channels = set(VIEW_FOCUS_CHANNELS[view_name])
        minimum = effective_min_focus_contradictions(view_name, negative_candidates)
        return sum(
            len(set(candidate.get("contradiction_channels", [])) & focus_channels)
            >= minimum
            for candidate in negative_candidates
        )

    assignments_by_view: dict[str, dict[str, Any]] = {}
    used_window_ids: set[str] = set()
    used_window_numbers: list[int] = []
    view_order = sorted(EXPECTED_VIEW_NAMES, key=lambda name: (valid_count(name), name))
    for view_name in view_order:
        minimum_focus = effective_min_focus_contradictions(view_name, negative_candidates)
        scored = [
            (
                _candidate_score_for_view(
                    candidate,
                    view_name=view_name,
                    minimum_focus_contradictions=minimum_focus,
                    used_window_ids=used_window_ids,
                    used_window_numbers=used_window_numbers,
                ),
                candidate,
            )
            for candidate in negative_candidates
        ]
        score, candidate = max(scored, key=lambda item: item[0])
        if score <= -1e8:
            raise ValueError(f"no usable negative candidate for view {view_name}")
        window_id = str(candidate["window_id"])
        used_window_ids.add(window_id)
        used_window_numbers.append(_window_number(window_id))
        assignments_by_view[view_name] = {
            "view_name": view_name,
            "candidate": candidate,
            "focus_channels": list(VIEW_FOCUS_CHANNELS[view_name]),
            "minimum_focus_contradictions": minimum_focus,
        }
    return [assignments_by_view[view_name] for view_name in EXPECTED_VIEW_NAMES]


def build_prompt(
    *,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    assigned_negative_candidates: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "target": target,
        "negative_candidates": negative_candidates,
        "assigned_negative_candidates": assigned_negative_candidates or [],
        "expected_view_names": list(EXPECTED_VIEW_NAMES),
    }
    return (
        "You are authoring a one-episode pilot dataset for narrative-conditioned "
        "financial scenario retrieval. Return only JSON matching the supplied "
        "schema; do not wrap JSON in markdown.\n\n"
        "Task: write exactly 14 pairs for the single target historical 30-day "
        "current/recent market prefix. Each pair has one POSITIVE narrative for "
        "the target and one same-style HARD-NEGATIVE narrative for a chosen "
        "negative candidate historical prefix. All positive_text and "
        "negative_text fields must be newly authored by Codex/GPT. Do not use "
        "local templates, copied factor lists, or copied source prose.\n\n"
        "Required view contract:\n"
        "- Return exactly 14 pairs, one for every expected_view_name.\n"
        "- The eight old-style views must be unique positive narratives.\n"
        "- risk_manager_memo and full_professional must not be copies.\n"
        "- factor_list_baseline and technical_factor_evidence must not be copies.\n"
        "- The six sparse_variant_* views must be short, actual user-like "
        "inputs from different angles.\n\n"
        "Assigned hard-negative contract:\n"
        "- If assigned_negative_candidates is non-empty, every view must use "
        "its assigned candidate and negative_window_id must equal that "
        "candidate's window_id.\n"
        "- The assigned candidate includes focus_channels. In the negative_text, "
        "make the economic conflict visible through at least one of those "
        "focus channels, preferably two when the evidence supports it.\n"
        "- For sparse views, reuse the same market anchors in natural language "
        "where possible, but change their direction or mechanism according to "
        "the assigned candidate.\n\n"
        "Old-style view guide:\n"
        "- sparse_user_query: one short market-condition sentence.\n"
        "- weekly_risk_monitor: concise monitor tone with active channel and uncertainty.\n"
        "- mechanism_first: mechanism/transmission first, then a few supporting markets.\n"
        "- technical_factor_evidence: evidence-oriented, technical, raw-move or confidence language when available.\n"
        "  Do not use unexplained large raw spread magnitudes such as "
        "AAA_OAS +552.89; phrase spread evidence as tighter/wider, "
        "small/medium/large, or z-score/confidence language.\n"
        "- factor_list_baseline: compact factor-list current-prefix view, terse but readable.\n"
        "- institutional_risk_committee_note: committee-style severity, exposures, and transmission.\n"
        "- risk_manager_memo: concise decision memo with regime, portfolio sensitivity, ambiguity, no forecast.\n"
        "- full_professional: fuller professional memo with regime, trigger, transmission, confirmation, vulnerability, ambiguity, no forecast.\n\n"
        "Sparse variant guide:\n"
        "- sparse_variant_tape_read: clipped tape read.\n"
        "- sparse_variant_portfolio_concern: what exposure or hedge is bothering the user.\n"
        "- sparse_variant_macro_channel: one macro channel in plain desk language.\n"
        "- sparse_variant_credit_ambiguity: short note about credit confirmation or conflict.\n"
        "- sparse_variant_rates_commodities: rates/commodities angle.\n"
        "- sparse_variant_desk_note: fragmentary morning-sheet style.\n\n"
        "Hard-negative quality rules:\n"
        "- Every negative_text must describe only the chosen negative candidate.\n"
        "- Match the same view style as the positive_text.\n"
        "- Prefer near-miss incompatible regimes over broad all-channel opposites.\n"
        "- Avoid candidates whose mismatch is only a side-channel difference "
        "when the positive narrative is led by FX or commodities.\n"
        "- Do not use direct negation or meta-contrast shortcuts.\n"
        "- Do not write phrases like 'not X', 'opposite of X', 'inconsistent with X', "
        "'conflicts with X', or 'contradicts X'.\n"
        "- Do not reuse the target title, the positive phrase, or distinctive "
        "multi-word phrases from the positive text.\n"
        "- Do not mention hard negatives, positives, contrastive training, "
        "embeddings, retrieval, or scenario generators.\n"
        "- In positive_text and negative_text, do not write meta words like "
        "'assigned prefix', 'assigned window', 'assigned candidate', or "
        "'supplied candidate'. Narratives should read like normal market text.\n"
        "- Do not invent named real-world news events.\n"
        "- Do not mention future horizons, terminal moves, generated scenarios, VaR, ES, or P&L.\n\n"
        "Output requirements:\n"
        "- target_window_id must equal the target window id.\n"
        "- target_title must equal the target scenario title.\n"
        "- negative_window_id must be one of the supplied negative candidates.\n"
        "- When assigned_negative_candidates is provided, negative_window_id "
        "must be the assigned candidate for that view.\n"
        "- quality_notes should explain the economic mismatch without using training-language labels.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


def build_validation_retry_prompt(
    *,
    base_prompt: str,
    batch: FourteenViewPilotBatch,
    validation: dict[str, Any],
) -> str:
    """Ask Codex to regenerate a failed batch using validation feedback."""

    failed_pairs: list[dict[str, Any]] = []
    pairs_by_view = {pair.view_name: pair for pair in batch.pairs}
    for error in validation.get("errors", []):
        view_name = str(error.get("view_name", ""))
        pair = pairs_by_view.get(view_name)
        if not pair:
            failed_pairs.append(error)
            continue
        failed_pairs.append(
            {
                "view_name": view_name,
                "validation_errors": error.get("errors", []),
                "previous_positive_text": pair.positive_text,
                "previous_negative_text": pair.negative_text,
                "previous_negative_window_id": pair.negative_window_id,
            }
        )
    if not failed_pairs:
        failed_pairs = list(validation.get("errors", []))
    return (
        f"{base_prompt}\n\n"
        "Validation retry:\n"
        "Your previous JSON failed local quality validation. Regenerate the full JSON, "
        "including all 14 pairs, using the same target and assigned negative candidates. "
        "Do not repair by local rules or templates; author revised market prose from the "
        "structured evidence. Pay special attention to failed views and remove direct "
        "negation, target-title reuse, positive-phrase reuse, internal training language, "
        "and excessive positive/negative wording overlap while preserving the assigned "
        "negative_window_id for each view.\n\n"
        f"Failed validation details:\n{json.dumps(failed_pairs, indent=2, sort_keys=True)}\n"
    )


def should_retry_validation(validation: dict[str, Any]) -> bool:
    """Return whether a failed Codex batch is worth one agentic regeneration."""

    return bool(validation.get("errors"))


def is_generic_reusable_target_phrase(phrase: str) -> bool:
    """Return whether a title phrase is generic market vocabulary."""

    normalized = re.sub(r"\s+", " ", phrase).strip().lower()
    if normalized in GENERIC_REUSABLE_TARGET_PHRASES:
        return True
    tokens = normalized.split()
    return (
        len(tokens) == 2
        and tokens[0] in GENERIC_ASSET_CLASS_MODIFIERS
        and tokens[1] in GENERIC_ASSET_CLASS_HEADS
    )


def protected_target_phrases(target_title: str) -> set[str]:
    """Return title phrases distinctive enough to ban from negative prose."""

    protected: set[str] = set()
    for phrase in _phrases(target_title):
        if is_generic_reusable_target_phrase(phrase):
            continue
        tokens = phrase.split()
        has_domain_marker = any("-" in token or "_" in token for token in tokens)
        if len(tokens) >= 3 or has_domain_marker:
            protected.add(phrase)
    return protected


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _pair_validation_errors(
    pair: NarrativePair,
    *,
    target_title: str,
    candidate_ids: set[str],
    assigned_candidate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    positive = _compact(pair.positive_text)
    negative = _compact(pair.negative_text)
    if pair.negative_window_id not in candidate_ids:
        errors.append(
            {
                "code": "negative_window_not_in_candidates",
                "negative_window_id": pair.negative_window_id,
            }
        )
    if assigned_candidate is not None:
        candidate = assigned_candidate.get("candidate", {})
        expected_window_id = str(candidate.get("window_id", ""))
        if pair.negative_window_id != expected_window_id:
            errors.append(
                {
                    "code": "negative_window_not_assigned_to_view",
                    "expected": expected_window_id,
                    "actual": pair.negative_window_id,
                }
            )
        focus_channels = set(assigned_candidate.get("focus_channels", []))
        contradiction_channels = set(candidate.get("contradiction_channels", []))
        minimum = int(assigned_candidate.get("minimum_focus_contradictions", 1))
        focus_hits = sorted(focus_channels & contradiction_channels)
        if len(focus_hits) < minimum:
            errors.append(
                {
                    "code": "weak_view_focus_contradiction",
                    "focus_hits": focus_hits,
                    "minimum_focus_contradictions": minimum,
                }
            )
    if _pattern_hits(LEAKAGE_PATTERNS, positive) or _pattern_hits(LEAKAGE_PATTERNS, negative):
        errors.append({"code": "future_or_metric_leakage"})
    if pair.view_name == "technical_factor_evidence":
        opaque_fields = [
            field
            for field, text in (("positive_text", positive), ("negative_text", negative))
            if OPAQUE_SPREAD_MAGNITUDE_PATTERN.search(text)
        ]
        if opaque_fields:
            errors.append(
                {
                    "code": "opaque_raw_spread_magnitude",
                    "fields": opaque_fields,
                }
            )
    internal_hits = _pattern_hits(PILOT_INTERNAL_PATTERNS, positive) + _pattern_hits(
        PILOT_INTERNAL_PATTERNS, negative
    )
    if internal_hits:
        errors.append({"code": "internal_training_language", "patterns": internal_hits})
    negation_hits = [
        pattern
        for pattern in _pattern_hits(DIRECT_NEGATION_PATTERNS, negative)
        if pattern != r"\bnot\s+(?:a|an|the)?\s*[\w-]+"
    ]
    low_negative = negative.lower()
    target_phrases = _phrases(target_title)
    if "not " in low_negative and any(phrase in low_negative for phrase in target_phrases):
        negation_hits.append("not_target_phrase")
    if negation_hits:
        errors.append({"code": "direct_negation_shortcut", "patterns": negation_hits})
    protected_phrases = protected_target_phrases(target_title)
    reused = sorted(phrase for phrase in protected_phrases if phrase in negative.lower())
    if reused:
        errors.append({"code": "target_or_positive_phrase_reuse", "phrases": reused[:10]})
    similarity = _token_jaccard(positive, negative)
    max_similarity = VIEW_MAX_TOKEN_JACCARD.get(pair.view_name, 0.55)
    if similarity >= max_similarity:
        errors.append(
            {
                "code": "high_positive_negative_token_overlap",
                "token_jaccard": round(similarity, 4),
                "max_token_jaccard": max_similarity,
            }
        )
    if pair.view_name.startswith("sparse_variant_"):
        for field, text in (("positive_text", positive), ("negative_text", negative)):
            words = _word_count(text)
            if words < 4 or words > 34:
                errors.append(
                    {
                        "code": "sparse_variant_length_out_of_range",
                        "field": field,
                        "word_count": words,
                    }
                )
    return errors


def validate_batch(
    *,
    batch: FourteenViewPilotBatch,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    assigned_negative_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    candidate_ids = {str(row["window_id"]) for row in negative_candidates}
    if batch.target_window_id != target["window_id"]:
        errors.append(
            {
                "code": "target_window_id_mismatch",
                "expected": target["window_id"],
                "actual": batch.target_window_id,
            }
        )
    if batch.target_title != target["scenario_title"]:
        errors.append(
            {
                "code": "target_title_mismatch",
                "expected": target["scenario_title"],
                "actual": batch.target_title,
            }
        )
    view_names = [pair.view_name for pair in batch.pairs]
    missing = sorted(set(EXPECTED_VIEW_NAMES) - set(view_names))
    extra = sorted(set(view_names) - set(EXPECTED_VIEW_NAMES))
    if len(batch.pairs) != len(EXPECTED_VIEW_NAMES):
        errors.append(
            {
                "code": "pair_count_mismatch",
                "expected": len(EXPECTED_VIEW_NAMES),
                "actual": len(batch.pairs),
            }
        )
    if missing:
        errors.append({"code": "missing_views", "views": missing})
    if extra:
        errors.append({"code": "unexpected_views", "views": extra})
    duplicate_views = sorted({name for name in view_names if view_names.count(name) > 1})
    if duplicate_views:
        errors.append({"code": "duplicate_view_names", "views": duplicate_views})
    assigned_by_view = {
        str(row.get("view_name")): row for row in (assigned_negative_candidates or [])
    }
    negative_window_ids = [pair.negative_window_id for pair in batch.pairs]
    duplicate_negative_ids = sorted(
        {window_id for window_id in negative_window_ids if negative_window_ids.count(window_id) > 1}
    )
    if duplicate_negative_ids:
        errors.append(
            {
                "code": "duplicate_negative_window_id",
                "negative_window_ids": duplicate_negative_ids,
            }
        )

    seen_positive: dict[str, str] = {}
    seen_negative: dict[str, str] = {}
    for pair in batch.pairs:
        positive_key = _normalized(pair.positive_text)
        previous_positive = seen_positive.get(positive_key)
        if previous_positive is not None:
            errors.append(
                {
                    "code": "duplicate_positive_text",
                    "views": [previous_positive, pair.view_name],
                }
            )
        else:
            seen_positive[positive_key] = pair.view_name
        negative_key = _normalized(pair.negative_text)
        previous_negative = seen_negative.get(negative_key)
        if previous_negative is not None:
            errors.append(
                {
                    "code": "duplicate_negative_text",
                    "views": [previous_negative, pair.view_name],
                }
            )
        else:
            seen_negative[negative_key] = pair.view_name
        pair_errors = _pair_validation_errors(
            pair,
            target_title=target["scenario_title"],
            candidate_ids=candidate_ids,
            assigned_candidate=assigned_by_view.get(pair.view_name),
        )
        validation.append(
            {
                "view_name": pair.view_name,
                "negative_window_id": pair.negative_window_id,
                "errors": pair_errors,
            }
        )
        if pair_errors:
            errors.append(
                {
                    "code": "pair_validation_failed",
                    "view_name": pair.view_name,
                    "errors": pair_errors,
                }
            )
    return {
        "status": "pass" if not errors and batch.pairs else "fail",
        "error_count": len(errors),
        "errors": errors,
        "validation": validation,
    }


def _candidate_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["window_id"]): row for row in candidates}


def build_review_markdown(
    *,
    batch: FourteenViewPilotBatch,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    date_lookup: dict[str, str],
    validation: dict[str, Any],
    source_paths: dict[str, str],
    assigned_negative_candidates: list[dict[str, Any]] | None = None,
) -> str:
    candidates = _candidate_by_id(negative_candidates)
    assigned_by_view = {
        str(row.get("view_name")): row for row in (assigned_negative_candidates or [])
    }
    validation_by_view = {row["view_name"]: row for row in validation.get("validation", [])}
    lines: list[str] = [
        "# Fourteen-View Narrative Pilot: Positives and Hard Negatives",
        "",
        "Generated: 2026-06-03",
        "",
        "This packet reruns one selected historical episode after fixing the "
        "old positive-view duplication bug. Structured market evidence and "
        "negative candidates were computed locally; all positive and negative "
        "narrative text was authored by Codex/GPT.",
        "",
        "## Selected Episode",
        "",
        f"- Target window: `{target['window_id']}`",
        f"- Observed prefix date range: {date_lookup.get(target['window_id'], 'unavailable')}",
        f"- Scenario title: {target['scenario_title']}",
        f"- Archetype: `{target['archetype']}`",
        "",
        "Positive mechanical evidence:",
        "",
        f"> {target['mechanical_summary']}",
        "",
        "Evidence fields:",
        "",
    ]
    for item in target.get("evidence_used", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Pilot Status",
            "",
            f"- Positive narrative rows: `{len(batch.pairs)}`",
            f"- Negative narrative rows: `{len(batch.pairs)}`",
            f"- Validation status: `{validation['status']}`",
            f"- Validation error count: `{validation['error_count']}`",
            f"- View-specific assigned hard negatives: `{bool(assigned_negative_candidates)}`",
            f"- Source cards: `{source_paths['cards_jsonl']}`",
            f"- Support-date metadata: `{source_paths['support_cards_jsonl']}`",
            f"- Codex output: `{source_paths['codex_output']}`",
            "",
            "## Review Pairs",
            "",
        ]
    )
    for idx, pair in enumerate(batch.pairs, 1):
        candidate = candidates.get(pair.negative_window_id, {})
        pair_validation = validation_by_view.get(pair.view_name, {"errors": []})
        assigned = assigned_by_view.get(pair.view_name, {})
        lines.extend(
            [
                f"### {idx}. {pair.view_name}",
                "",
                "Positive narrative:",
                "",
                f"> {_compact(pair.positive_text)}",
                "",
                "Corresponding hard negative:",
                "",
                f"- Negative window: `{pair.negative_window_id}`",
                f"- Negative observed prefix date range: {date_lookup.get(pair.negative_window_id, 'unavailable')}",
                f"- Negative title: {candidate.get('scenario_title', '')}",
                f"- Negative archetype: `{candidate.get('archetype', '')}`",
                f"- Contradiction channels: {', '.join(candidate.get('contradiction_channels', []))}",
                f"- View focus channels: {', '.join(assigned.get('focus_channels', []))}",
                f"- Agreement count: `{candidate.get('agreement_count', '')}`",
                "",
                "Negative mechanical summary:",
                "",
                f"> {candidate.get('mechanical_summary', '')}",
                "",
                "Hard-negative narrative:",
                "",
                f"> {_compact(pair.negative_text)}",
                "",
                "Quality notes:",
                "",
            ]
        )
        if pair.quality_notes:
            lines.extend(f"- {_compact(note)}" for note in pair.quality_notes)
        else:
            lines.append("- None supplied.")
        lines.extend(["", "Validation:", ""])
        if pair_validation.get("errors"):
            for error in pair_validation["errors"]:
                lines.append(f"- `{error.get('code')}`: `{json.dumps(error, sort_keys=True)}`")
        else:
            lines.append("- Pass.")
        lines.append("")
    lines.extend(
        [
            "## Human Review Questions",
            "",
            "- Are the old eight positive views now genuinely distinct?",
            "- Do the six sparse variants sound like realistic user inputs?",
            "- Does each hard negative match its positive view style without saying it is a negative?",
            "- Are the negatives hard near-misses rather than trivial all-channel opposites?",
            "- Are any narratives too forecast-like or too polished for their intended style?",
            "",
        ]
    )
    return "\n".join(lines)


def _date_lookup_from_source_paths(source_paths: dict[str, str]) -> dict[str, str]:
    support_cards_jsonl = source_paths.get("support_cards_jsonl", "")
    if not support_cards_jsonl:
        return {}
    path = _resolve(Path(support_cards_jsonl))
    if not path.is_file():
        return {}
    return _support_date_lookup(Path(support_cards_jsonl))


def run_pilot_from_prepared_payload(
    args: argparse.Namespace,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    source_paths: dict[str, str],
    assigned_negative_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = _resolve(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if assigned_negative_candidates is None:
        assigned_negative_candidates = assign_negative_candidates_by_view(negative_candidates)
    prompt = build_prompt(
        target=target,
        negative_candidates=negative_candidates,
        assigned_negative_candidates=assigned_negative_candidates,
    )
    return _run_pilot_codex_loop(
        args=args,
        output_dir=output_dir,
        target=target,
        negative_candidates=negative_candidates,
        assigned_negative_candidates=assigned_negative_candidates,
        prompt=prompt,
        source_paths=source_paths,
    )


def _run_pilot_codex_loop(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    assigned_negative_candidates: list[dict[str, Any]],
    prompt: str,
    source_paths: dict[str, str],
) -> dict[str, Any]:
    schema_path = output_dir / "fourteen_view_schema.json"
    prompt_path = output_dir / "fourteen_view_prompt.txt"
    codex_output_path = output_dir / "fourteen_view_codex_output.json"
    events_path = output_dir / "fourteen_view_codex_events.jsonl"
    report_path = output_dir / "fourteen_view_report.json"
    review_path = output_dir / "fourteen_view_review.md"
    _write_json(schema_path, strict_schema())
    _write_text(prompt_path, prompt)

    started = time.time()
    errors: list[dict[str, Any]] = []
    batch: FourteenViewPilotBatch | None = None
    validation: dict[str, Any] | None = None
    attempt_records: list[dict[str, Any]] = []
    if bool(args.dry_run):
        errors.append({"code": "dry_run", "message": "Prompt/schema written; Codex not invoked."})
    else:
        max_attempts = 1 + max(0, int(args.validation_retries))
        attempt_prompt = prompt
        for attempt_index in range(max_attempts):
            attempt_number = attempt_index + 1
            attempt_output_path = output_dir / f"fourteen_view_codex_output_attempt{attempt_number}.json"
            attempt_events_path = output_dir / f"fourteen_view_codex_events_attempt{attempt_number}.jsonl"
            attempt_prompt_path = output_dir / f"fourteen_view_prompt_attempt{attempt_number}.txt"
            _write_text(attempt_prompt_path, attempt_prompt)
            cmd = [
                "codex",
                "exec",
                "--ephemeral",
                "--json",
                "--disable",
                "apps",
                "--disable",
                "image_generation",
                "-m",
                str(args.model),
                "-c",
                f"model_reasoning_effort='{args.reasoning_effort}'",
                "--sandbox",
                "read-only",
                "--cd",
                str(ROOT),
                "--output-schema",
                str(schema_path),
                "-o",
                str(attempt_output_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                input=attempt_prompt,
                capture_output=True,
                timeout=int(args.timeout_seconds),
                check=False,
            )
            _write_text(attempt_events_path, completed.stdout)
            attempt_record: dict[str, Any] = {
                "attempt": attempt_number,
                "prompt": str(attempt_prompt_path),
                "codex_output": str(attempt_output_path),
                "codex_events": str(attempt_events_path),
                "returncode": completed.returncode,
            }
            if completed.returncode != 0:
                attempt_record.update(
                    {
                        "status": "fail",
                        "code": "codex_exec_failed",
                        "stderr_tail": completed.stderr[-4000:],
                    }
                )
                attempt_records.append(attempt_record)
                errors.append(
                    {
                        "code": "codex_exec_failed",
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr[-4000:],
                        "events_path": str(attempt_events_path),
                    }
                )
                break
            try:
                attempt_batch = FourteenViewPilotBatch.model_validate_json(
                    _extract_json_object(attempt_output_path.read_text(encoding="utf-8"))
                )
            except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
                attempt_record.update(
                    {
                        "status": "fail",
                        "code": "codex_output_parse_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                attempt_records.append(attempt_record)
                errors.append(
                    {
                        "code": "codex_output_parse_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                break
            attempt_validation = validate_batch(
                batch=attempt_batch,
                target=target,
                negative_candidates=negative_candidates,
                assigned_negative_candidates=assigned_negative_candidates,
            )
            attempt_record.update(
                {
                    "status": attempt_validation["status"],
                    "validation_error_count": attempt_validation["error_count"],
                }
            )
            attempt_records.append(attempt_record)
            if (
                attempt_validation["status"] == "pass"
                or attempt_index == max_attempts - 1
                or not should_retry_validation(attempt_validation)
            ):
                batch = attempt_batch
                validation = attempt_validation
                _write_text(
                    codex_output_path,
                    attempt_output_path.read_text(encoding="utf-8"),
                )
                _write_text(events_path, completed.stdout)
                break
            attempt_prompt = build_validation_retry_prompt(
                base_prompt=prompt,
                batch=attempt_batch,
                validation=attempt_validation,
            )

    if batch is None:
        validation = {
            "status": "fail",
            "error_count": len(errors),
            "errors": list(errors),
            "validation": [],
        }
    else:
        if validation is None:
            validation = validate_batch(
                batch=batch,
                target=target,
                negative_candidates=negative_candidates,
                assigned_negative_candidates=assigned_negative_candidates,
            )
        errors.extend(validation["errors"])
    review_source_paths = {
        "cards_jsonl": source_paths.get("cards_jsonl", ""),
        "support_cards_jsonl": source_paths.get("support_cards_jsonl", ""),
        **source_paths,
    }
    review_source_paths["codex_output"] = str(codex_output_path)
    if batch is not None:
        review = build_review_markdown(
            batch=batch,
            target=target,
            negative_candidates=negative_candidates,
            date_lookup=_date_lookup_from_source_paths(source_paths),
            validation=validation,
            source_paths=review_source_paths,
            assigned_negative_candidates=assigned_negative_candidates,
        )
        _write_text(review_path, review)
    report = {
        "schema_version": "fourteen_view_variant_pilot_report_v1",
        "status": "pass" if batch is not None and not errors else "fail",
        "target_window_id": target["window_id"],
        "expected_view_names": list(EXPECTED_VIEW_NAMES),
        "negative_candidate_count": len(negative_candidates),
        "assigned_negative_candidates": assigned_negative_candidates,
        "validation": validation,
        "attempt_records": attempt_records,
        "errors": errors,
        "local_prose_generated": False,
        "codex_model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
        "dry_run": bool(args.dry_run),
        "elapsed_seconds": round(time.time() - started, 3),
        "target": target,
        "negative_candidates": negative_candidates,
        "pairs": [pair.model_dump() for pair in batch.pairs] if batch is not None else [],
        "artifact_paths": {
            "schema": str(schema_path),
            "prompt": str(prompt_path),
            "codex_output": str(codex_output_path),
            "codex_events": str(events_path),
            "report": str(report_path),
            "review": str(review_path),
        },
    }
    _write_json(report_path, report)
    return report


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    cards = _read_jsonl(Path(args.cards_jsonl))
    by_id = {str(card.get("window_id", "")): card for card in cards}
    if args.target_window_id not in by_id:
        raise ValueError(f"target window not found: {args.target_window_id}")
    target = _target_payload(by_id[args.target_window_id])
    target["evidence_used"] = _evidence_used(by_id[args.target_window_id])
    negative_candidates = select_negative_candidates_for_fourteen_view(
        cards=cards,
        target_window_id=args.target_window_id,
        count=max(int(args.negative_candidate_count), len(EXPECTED_VIEW_NAMES)),
        min_temporal_gap=int(args.min_temporal_gap),
    )
    return run_pilot_from_prepared_payload(
        args=args,
        target=target,
        negative_candidates=negative_candidates,
        source_paths={
            "cards_jsonl": str(args.cards_jsonl),
            "support_cards_jsonl": str(args.support_cards_jsonl),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-cards-jsonl", type=Path, default=DEFAULT_SUPPORT_CARDS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-window-id", default=DEFAULT_TARGET_WINDOW_ID)
    parser.add_argument("--negative-candidate-count", type=int, default=80)
    parser.add_argument("--min-temporal-gap", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--validation-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_pilot(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "target_window_id": report["target_window_id"],
                "pair_count": len(report["pairs"]),
                "error_count": len(report["errors"]),
                "report": report["artifact_paths"]["report"],
                "review": report["artifact_paths"]["review"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
