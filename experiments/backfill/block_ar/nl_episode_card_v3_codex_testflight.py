#!/usr/bin/env python
"""Small Codex-authored EpisodeCardV3 narrative regeneration TestFlight.

This script is intentionally isolated from the incumbent caption corpus. It
selects a small, diverse set of historical support windows, converts them into
the existing RiskManagerCaptionV2 prompt payload format, optionally invokes the
Codex caption runner, and builds review artifacts from the generated captions.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_codex_caption_batch import (  # noqa: E402
    DEFAULT_CODEX_MODEL,
    DEFAULT_REASONING_EFFORT,
    run_batch,
)
from experiments.backfill.block_ar.nl_episode_card_v3_testflight import (  # noqa: E402
    infer_supported_angles,
)
from experiments.backfill.block_ar.nl_episode_narrative_cards import (  # noqa: E402
    LEAKAGE_PATTERNS,
    build_cards_from_caption_jsonl,
    _snippets_for_patterns,
    view_metrics,
    write_jsonl,
)
from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (  # noqa: E402
    PROMPT_VERSION,
    validate_caption_v2,
)


DEFAULT_SOURCE_CARDS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_support_bank_cards_all_972b_tight_taxonomy/"
    "episode_narrative_support_cards.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_codex_authored_testflight"
)

TARGET_ANGLES = (
    "Classic safe-haven gold risk-off",
    "Safe-haven gold bid",
    "Gold-duration bid without risk-off confirmation",
    "Gold up in risk-on relief",
    "Fragile risk-on rebound",
    "Defensive risk-off shock",
    "Commodity-inflation pressure",
    "Dollar-liquidity squeeze",
    "Rates selloff",
    "Dollar/rates defensive regime",
)

SPARSE_INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bgive me\b",
        r"\bcan you\b",
        r"\bplease\b",
        r"\bsummarize\b",
        r"\bwrite this up\b",
        r"\bframe (?:this|it)\b",
        r"\bi need (?:a|an|the)?\s*(?:quick\s+|short\s+|current\s+)?(?:read|note|summary|context)\b",
        r"\?",
    )
)

SPARSE_INTERNAL_REFERENCE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsupport system\b",
        r"\bretrieval system\b",
        r"\bretrieval\b",
        r"\bmodel\b",
        r"\bscenario generator\b",
        r"\bconditioning\b",
        r"\btraining\b",
        r"\bembedding\b",
    )
)
SPARSE_VAGUE_AMBIGUITY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnot every\s+[\w-]+\s+signal\b",
        r"\bnot every signal\b",
        r"\bnot perfectly aligned\b",
        r"\bnot fully aligned\b",
    )
)
FACTOR_TOKEN_TERMS = (
    "SPX",
    "VIX",
    "BBB",
    "AAA",
    "DXY",
    "USDJPY",
    "GOLD",
    "CRUDE",
    "US2Y",
    "US10Y",
)


class MultiFormatNarrative(BaseModel):
    """Codex-authored independent narrative formats for one historical period."""

    model_config = ConfigDict(extra="forbid")

    window_id: str = Field(min_length=1)
    schema_version: str = "episode_card_v3_multiformat_v1"
    scenario_title: str = Field(min_length=3)
    archetype: str = Field(min_length=3)
    archetype_confidence: Literal["low", "medium", "high"] | str = "medium"
    old_factor_baseline: str = Field(min_length=20)
    factor_list_baseline: str = Field(min_length=20)
    technical_factor_evidence: str = Field(min_length=20)
    sparse_user_prompt: str = Field(min_length=20)
    weekly_risk_monitor: str = Field(min_length=20)
    institutional_risk_committee_note: str = Field(min_length=50)
    mechanism_first_memo: str = Field(min_length=20)
    risk_manager_memo: str = Field(min_length=50)
    full_risk_manager_memo: str = Field(min_length=50)
    evidence_used: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)
    no_forecast_caveat: str = Field(min_length=6)
    contrastive_hard_negatives: list[str] = Field(default_factory=list)
    quality_self_critique: list[str] = Field(default_factory=list)


class MultiFormatNarrativeBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narratives: list[MultiFormatNarrative] = Field(default_factory=list)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _support_move_rows(card: dict[str, Any]) -> list[dict[str, Any]]:
    rows = card.get("support_metadata", {}).get("support_move_rows", [])
    return [row for row in rows if isinstance(row, dict)]


def _confidence_from_z(z_change: Any) -> str:
    try:
        magnitude = abs(float(z_change))
    except (TypeError, ValueError):
        return "medium"
    if magnitude >= 1.0:
        return "high"
    if magnitude >= 0.35:
        return "medium"
    return "low"


def support_card_to_bundle(card: dict[str, Any]) -> dict[str, Any]:
    """Convert one support-bank card into a RiskManagerCaptionV2 prompt bundle."""

    metadata = card.get("support_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    market_implications = []
    for row in _support_move_rows(card):
        market = str(row.get("market", "")).strip()
        direction = str(row.get("direction", "")).strip()
        direction_label = str(row.get("direction_label", direction)).strip()
        magnitude = str(row.get("magnitude", "")).strip()
        raw = row.get("raw_change")
        z = row.get("z_change")
        if not market or not direction:
            continue
        market_implications.append(
            {
                "market": market,
                "direction": direction,
                "direction_label": direction_label,
                "magnitude": magnitude,
                "confidence": _confidence_from_z(z),
                "horizon": "30 historical trading days ending at the conditioning date",
                "evidence": [
                    f"raw_change={raw}",
                    f"z_change={z}",
                    f"theme={row.get('theme', '')}",
                ],
            }
        )
    caption_fields = card.get("caption_fields", {})
    source_text = ""
    if isinstance(caption_fields, dict):
        source_text = str(
            caption_fields.get("training_caption")
            or caption_fields.get("mechanical_summary")
            or ""
        )
    return {
        "window_id": str(card.get("window_id", "")),
        "manifest_split": str(card.get("split", "")),
        "source_index": metadata.get("window_index"),
        "window_index": metadata.get("window_index"),
        "calendar": {
            "calendar_start_date": str(metadata.get("calendar_start_date", "")),
            "calendar_end_date": str(metadata.get("calendar_end_date", "")),
            "forecast_start_date": "",
            "forecast_end_date": "",
        },
        "market_implications": market_implications,
        "narratives": [{"kind": "support_card_972b", "text": source_text}],
        "source_description_bundle": {
            "canonical_machine_text": source_text,
            "revised_description": source_text,
        },
        "support_metadata": metadata,
        "source_card": {
            "scenario_title": card.get("scenario_title", ""),
            "archetype": card.get("archetype", ""),
            "archetype_confidence": card.get("archetype_confidence", ""),
        },
    }


def _angle_names(card: dict[str, Any]) -> list[str]:
    names = [str(spec["angle_name"]) for spec in infer_supported_angles(card)]
    title = str(card.get("scenario_title", ""))
    if title and title not in names:
        names.append(title)
    return names


def _selection_score(card: dict[str, Any]) -> tuple[int, float, int]:
    rows = _support_move_rows(card)
    large_moves = 0
    z_total = 0.0
    for row in rows:
        try:
            z_abs = abs(float(row.get("z_change", 0.0)))
        except (TypeError, ValueError):
            z_abs = 0.0
        if z_abs >= 0.35:
            large_moves += 1
        z_total += z_abs
    index = int(card.get("support_metadata", {}).get("window_index") or 0)
    return large_moves, z_total, -index


def select_representative_cards(
    source_cards: list[dict[str, Any]],
    *,
    max_cases: int,
    min_window_gap: int = 180,
    target_angles: tuple[str, ...] = TARGET_ANGLES,
) -> list[dict[str, Any]]:
    """Pick diverse, evidence-rich windows for human narrative review."""

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_indices: list[int] = []

    def far_enough(card: dict[str, Any]) -> bool:
        try:
            index = int(card.get("support_metadata", {}).get("window_index"))
        except (TypeError, ValueError):
            return True
        return all(
            abs(index - prior) >= int(min_window_gap) for prior in selected_indices
        )

    def record(card: dict[str, Any]) -> None:
        selected.append(card)
        selected_ids.add(str(card.get("window_id", "")))
        try:
            selected_indices.append(
                int(card.get("support_metadata", {}).get("window_index"))
            )
        except (TypeError, ValueError):
            pass

    for target in target_angles:
        candidates = [
            card
            for card in source_cards
            if target in _angle_names(card)
            and str(card.get("window_id", "")) not in selected_ids
        ]
        if not candidates:
            continue
        candidates.sort(key=_selection_score, reverse=True)
        chosen = next((card for card in candidates if far_enough(card)), None)
        if chosen is None:
            chosen = candidates[0]
        record(chosen)
        if len(selected) >= int(max_cases):
            break
    if len(selected) < int(max_cases):
        fallback = [
            card
            for card in source_cards
            if str(card.get("window_id", "")) not in selected_ids and far_enough(card)
        ]
        fallback.sort(key=_selection_score, reverse=True)
        for card in fallback[: int(max_cases) - len(selected)]:
            record(card)
    if len(selected) < int(max_cases):
        fallback = [
            card
            for card in source_cards
            if str(card.get("window_id", "")) not in selected_ids
        ]
        fallback.sort(key=_selection_score, reverse=True)
        for card in fallback[: int(max_cases) - len(selected)]:
            record(card)
    return selected[: int(max_cases)]


def select_cards_for_multiformat_generation(
    source_cards: list[dict[str, Any]],
    *,
    max_cases: int,
    min_window_gap: int,
    selection_mode: str = "representative",
    rich_stride: int = 15,
) -> list[dict[str, Any]]:
    """Select support cards for Codex-authored multi-format generation."""

    if selection_mode == "representative":
        return select_representative_cards(
            source_cards,
            max_cases=int(max_cases),
            min_window_gap=int(min_window_gap),
        )
    if selection_mode == "all":
        selected = list(source_cards)
    elif selection_mode == "stride":
        stride = max(1, int(rich_stride))
        selected = [
            card
            for card in source_cards
            if int(card.get("support_metadata", {}).get("window_index", -1)) % stride
            == 0
        ]
    else:
        raise ValueError(f"unknown selection_mode: {selection_mode}")
    if int(max_cases) > 0:
        selected = selected[: int(max_cases)]
    return selected


def prepare_testflight(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_cards = read_jsonl(Path(args.source_cards_jsonl))
    selection_mode = str(getattr(args, "selection_mode", "representative"))
    selected_cards = select_cards_for_multiformat_generation(
        source_cards,
        max_cases=int(args.max_cases),
        min_window_gap=int(args.min_window_gap),
        selection_mode=selection_mode,
        rich_stride=int(getattr(args, "rich_stride", 15)),
    )
    bundles = [support_card_to_bundle(card) for card in selected_cards]
    selected_rows = []
    for card, bundle in zip(selected_cards, bundles, strict=True):
        selected_rows.append(
            {
                "window_id": bundle["window_id"],
                "window_index": bundle["window_index"],
                "history_start": bundle["calendar"]["calendar_start_date"],
                "history_end": bundle["calendar"]["calendar_end_date"],
                "target_angles": _angle_names(card),
                "source_title": card.get("scenario_title", ""),
                "source_archetype": card.get("archetype", ""),
                "market_implications": bundle["market_implications"],
            }
        )
    pipeline_report = {
        "schema_version": "episode_card_v3_codex_testflight_pipeline_v1",
        "scope_note": (
            "Temporary prompt bundle for a small Codex-authored narrative "
            "regeneration TestFlight. It is not a full corpus regeneration."
        ),
        "source_cards_jsonl": str(args.source_cards_jsonl),
        "selection_mode": selection_mode,
        "rich_stride": int(getattr(args, "rich_stride", 15)),
        "prompt_version": PROMPT_VERSION,
        "narrative_bundles": bundles,
    }
    pipeline_path = output_dir / "codex_testflight_pipeline_report.json"
    selection_path = output_dir / "selected_support_cases.json"
    review_path = output_dir / "selected_support_cases.md"
    write_json(pipeline_path, pipeline_report)
    write_json(selection_path, {"selected_cases": selected_rows})
    review_path.write_text(render_selection_markdown(selected_rows), encoding="utf-8")
    return {
        "status": "prepared",
        "selected_count": len(selected_rows),
        "pipeline_report": str(pipeline_path),
        "selected_support_cases": str(selection_path),
        "selected_support_cases_markdown": str(review_path),
        "selected_window_ids": [row["window_id"] for row in selected_rows],
    }


def render_selection_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Codex-Authored EpisodeCardV3 TestFlight: Selected Cases",
        "",
        "These windows are selected from the broad historical support bank for a "
        "small human-reviewable Codex narrative regeneration run.",
        "",
    ]
    for idx, row in enumerate(rows, 1):
        lines.extend(
            [
                f"## {idx}. {row['window_id']} ({row['history_start']} to {row['history_end']})",
                "",
                f"- Source title: `{row['source_title']}`",
                f"- Source archetype: `{row['source_archetype']}`",
                f"- Target angles: {', '.join(row['target_angles'])}",
                "",
                "| Market | Direction | Magnitude | Confidence | Evidence |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for item in row["market_implications"]:
            lines.append(
                "| {market} | {direction} | {magnitude} | {confidence} | {evidence} |".format(
                    market=item["market"],
                    direction=item["direction"],
                    magnitude=item["magnitude"],
                    confidence=item["confidence"],
                    evidence="; ".join(item.get("evidence", [])[:2]),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _require_all_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively make object schemas strict for Codex structured output."""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict) and properties:
                node["required"] = list(properties)
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


def strict_multiformat_batch_schema() -> dict[str, Any]:
    return _require_all_properties(MultiFormatNarrativeBatch.model_json_schema())


def _bundle_payload_for_multiformat(bundle: dict[str, Any]) -> dict[str, Any]:
    calendar = bundle.get("calendar", {})
    if not isinstance(calendar, dict):
        calendar = {}
    return {
        "window_id": bundle.get("window_id"),
        "manifest_split": bundle.get("manifest_split"),
        "window_index": bundle.get("window_index"),
        "current_recent_prefix_dates": {
            "calendar_start_date": calendar.get("calendar_start_date", ""),
            "calendar_end_date": calendar.get("calendar_end_date", ""),
        },
        "market_implications": bundle.get("market_implications", []),
        "rough_machine_caption_to_improve": (
            (bundle.get("narratives") or [{}])[0].get("text", "")
            if isinstance(bundle.get("narratives"), list)
            else ""
        ),
        "source_card": bundle.get("source_card", {}),
    }


def build_multiformat_prompt(bundles: list[dict[str, Any]]) -> str:
    """Build a prompt that asks Codex for independent narrative formats."""

    payloads = [_bundle_payload_for_multiformat(bundle) for bundle in bundles]
    return (
        "Generate genuinely distinct narrative formats for each supplied "
        "historical 30-day current/recent market prefix. Return only JSON "
        "matching the output schema. Do not wrap JSON in markdown.\n\n"
        "Critical bug-prevention rule: Do not derive the formats by copying "
        "the same factor list into every field. The point of this TestFlight "
        "is to see different writing styles for the same period.\n\n"
        "Title/taxonomy rule: choose scenario_title from the evidence, not "
        "from the rough source label. If gold is only partial/medium support, "
        "do not call the title 'Classic safe-haven gold risk-off'; use a "
        "more precise title such as 'financial-accident risk-off with partial "
        "safe-haven confirmation'. If commodity or rates evidence is only a "
        "supporting channel while credit relief dominates, use a title such "
        "as 'post-stress reflation / credit-beta relief' rather than "
        "'commodity-inflation pressure'.\n\n"
        "For each window, produce these fields:\n"
        "- old_factor_baseline: a deliberately mechanical factor baseline. "
        "It may list the main observed market moves, but keep it clearly "
        "separate from the professional prose.\n"
        "- factor_list_baseline: compact factor-list training view. Keep it "
        "terse and factor-oriented, but do not copy old_factor_baseline "
        "verbatim.\n"
        "- technical_factor_evidence: technical evidence view. Use observed "
        "move evidence, confidence, and raw/z-score style facts when supplied; "
        "do not copy factor_list_baseline or old_factor_baseline.\n"
        "- sparse_user_prompt: a sparse conditioning narrative, 1-3 "
        "declarative sentences, written in the voice of a risk manager "
        "describing the current/recent market state. The sparse_user_prompt "
        "must mention only one or two channels and should not enumerate every "
        "market. It should be economically specific without becoming a factor "
        "list: say whether the issue looks isolated or broad, and name any "
        "ambiguous confirmation channel concretely. Avoid vague phrases such "
        "as 'not every signal is aligned' unless the sentence names the "
        "channel that is not aligned. Prefer channel language such as "
        "'lower-quality credit versus high-grade credit' over raw tickers when "
        "that keeps the text natural.\n"
        "- weekly_risk_monitor: concise institutional market-monitor style. "
        "Focus on what changed, the active risk channel, and the uncertainty. "
        "Prefer channel language over raw factor lists. Use at most three "
        "explicit tickers or factor names, and do not name every confirming "
        "market.\n"
        "- institutional_risk_committee_note: institutional risk-committee "
        "note. Write as a committee summary of the observed current/recent "
        "prefix: severity, affected exposures, transmission channel, and "
        "cross-market confirmation. This field is not always a stress "
        "scenario; for relief or risk-on episodes, state that it is not an "
        "acute stress state. Do not prescribe a future stress path.\n"
        "- mechanism_first_memo: explanation-first memo. Lead with mechanism, "
        "transmission, and portfolio meaning; use only a few factors as "
        "supporting evidence.\n"
        "- risk_manager_memo: concise risk-manager memo. Include regime, "
        "current trigger evidence, portfolio sensitivity, ambiguity, and a "
        "no-forecast framing. It must be shorter and more decision-oriented "
        "than full_risk_manager_memo.\n"
        "- full_risk_manager_memo: professional risk-manager memo with regime, "
        "trigger, transmission, cross-asset confirmation, portfolio "
        "vulnerability, ambiguity, and no-forecast caveat.\n\n"
        "Rules:\n"
        "- Describe only the current/recent conditioning prefix.\n"
        "- Do not forecast what happens next.\n"
        "- Avoid future-modal phrases such as 'will widen', 'will tighten', "
        "'will rally', 'will fall', or 'the next 30 days'. If discussing "
        "portfolio vulnerability, write it as current exposure sensitivity "
        "rather than a future path assumption.\n"
        "- Do not invent named real-world news events.\n"
        "- Use the evidence table, but do not turn every narrative into a "
        "full factor list.\n"
        "- The eight training views must be separately authored. Exact copies "
        "between risk_manager_memo/full_risk_manager_memo or between "
        "factor_list_baseline/technical_factor_evidence are invalid.\n"
        "- Preserve ambiguity and conflicting signals explicitly.\n"
        "- The sparse_user_prompt must be a scenario description, not an "
        "instruction or question to an assistant. Do not use phrases such as "
        "'give me', 'can you', 'please', 'summarize', 'write this up', "
        "'frame this', 'I need a read', or question marks.\n"
        "- The sparse_user_prompt must not mention the model, retrieval, "
        "support system, embeddings, training, conditioning, or scenario "
        "generator. It should read like market commentary, not product "
        "instructions.\n"
        "- Include contrastive hard negatives that describe incompatible "
        "regimes.\n\n"
        f"Payloads:\n{json.dumps(payloads, indent=2, sort_keys=True)}\n"
    )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Codex output did not contain a JSON object")
    return stripped[start : end + 1]


def _safe_window_id(window_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in window_id)


def _load_multiformat_batch(path: Path) -> MultiFormatNarrativeBatch:
    return MultiFormatNarrativeBatch.model_validate_json(
        _extract_json_object(path.read_text(encoding="utf-8"))
    )


def _filter_negated_no_forecast_hits(
    hits: list[dict[str, str]],
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for hit in hits:
        snippet = hit.get("snippet", "").lower()
        if (
            "does not forecast" in snippet
            or "do not forecast" in snippet
            or "not forecast" in snippet
            or "no forecast" in snippet
            or "not a forecast" in snippet
            or "does not prescribe" in snippet
            or "do not prescribe" in snippet
            or "not prescribe" in snippet
            or "does not project" in snippet
            or "do not project" in snippet
            or "not project" in snippet
            or "does not imply" in snippet
            or "do not imply" in snippet
            or "not imply" in snippet
        ):
            continue
        filtered.append(hit)
    return filtered


def _count_factor_terms(text: str) -> int:
    upper = text.upper()
    return sum(token in upper for token in FACTOR_TOKEN_TERMS)


def _validate_multiformat(narrative: MultiFormatNarrative) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    fields = {
        "old_factor_baseline": narrative.old_factor_baseline,
        "factor_list_baseline": narrative.factor_list_baseline,
        "technical_factor_evidence": narrative.technical_factor_evidence,
        "sparse_user_prompt": narrative.sparse_user_prompt,
        "weekly_risk_monitor": narrative.weekly_risk_monitor,
        "institutional_risk_committee_note": (
            narrative.institutional_risk_committee_note
        ),
        "mechanism_first_memo": narrative.mechanism_first_memo,
        "risk_manager_memo": narrative.risk_manager_memo,
        "full_risk_manager_memo": narrative.full_risk_manager_memo,
    }
    for name, text in fields.items():
        hits = _filter_negated_no_forecast_hits(
            _snippets_for_patterns(text, LEAKAGE_PATTERNS)
        )
        if hits:
            issues.append(
                {
                    "code": "future_leakage",
                    "field": name,
                    "severity": "error",
                    "hits": hits[:5],
                }
            )
    sparse_factor_terms = _count_factor_terms(narrative.sparse_user_prompt)
    if sparse_factor_terms > 2:
        issues.append(
            {
                "code": "sparse_prompt_too_factor_list_like",
                "field": "sparse_user_prompt",
                "severity": "warning",
                "factor_term_count": sparse_factor_terms,
            }
        )
    weekly_factor_terms = _count_factor_terms(narrative.weekly_risk_monitor)
    if weekly_factor_terms > 4:
        issues.append(
            {
                "code": "weekly_monitor_too_factor_list_like",
                "field": "weekly_risk_monitor",
                "severity": "warning",
                "factor_term_count": weekly_factor_terms,
            }
        )
    instruction_hits = _snippets_for_patterns(
        narrative.sparse_user_prompt, SPARSE_INSTRUCTION_PATTERNS
    )
    if instruction_hits:
        issues.append(
            {
                "code": "sparse_prompt_instruction_like",
                "field": "sparse_user_prompt",
                "severity": "error",
                "hits": instruction_hits[:5],
            }
        )
    internal_hits = _snippets_for_patterns(
        narrative.sparse_user_prompt, SPARSE_INTERNAL_REFERENCE_PATTERNS
    )
    if internal_hits:
        issues.append(
            {
                "code": "sparse_prompt_internal_reference",
                "field": "sparse_user_prompt",
                "severity": "error",
                "hits": internal_hits[:5],
            }
        )
    vague_hits = _snippets_for_patterns(
        narrative.sparse_user_prompt, SPARSE_VAGUE_AMBIGUITY_PATTERNS
    )
    if vague_hits:
        issues.append(
            {
                "code": "sparse_prompt_vague_ambiguity",
                "field": "sparse_user_prompt",
                "severity": "warning",
                "hits": vague_hits[:5],
            }
        )
    if narrative.sparse_user_prompt == narrative.old_factor_baseline:
        issues.append(
            {
                "code": "duplicate_format_text",
                "field": "sparse_user_prompt",
                "severity": "error",
            }
        )
    training_views = {
        "sparse_user_query": narrative.sparse_user_prompt,
        "weekly_risk_monitor": narrative.weekly_risk_monitor,
        "mechanism_first": narrative.mechanism_first_memo,
        "technical_factor_evidence": narrative.technical_factor_evidence,
        "factor_list_baseline": narrative.factor_list_baseline,
        "institutional_risk_committee_note": narrative.institutional_risk_committee_note,
        "risk_manager_memo": narrative.risk_manager_memo,
        "full_professional": narrative.full_risk_manager_memo,
    }
    seen: dict[str, str] = {}
    for name, text in training_views.items():
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        previous = seen.get(normalized)
        if previous is not None:
            issues.append(
                {
                    "code": "duplicate_training_view_text",
                    "fields": [previous, name],
                    "severity": "error",
                }
            )
        else:
            seen[normalized] = name
    return issues


def run_multiformat_codex(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_report = Path(args.pipeline_report)
    if not pipeline_report.exists():
        prepare_testflight(
            argparse.Namespace(
                source_cards_jsonl=args.source_cards_jsonl,
                output_dir=args.output_dir,
                max_cases=args.max_cases,
                min_window_gap=args.min_window_gap,
                selection_mode=getattr(args, "selection_mode", "representative"),
                rich_stride=getattr(args, "rich_stride", 15),
            )
        )
    pipeline = json.loads(pipeline_report.read_text(encoding="utf-8"))
    bundles = pipeline.get("narrative_bundles", [])
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("pipeline report does not contain narrative_bundles")
    schema_path = output_dir / "multiformat_narrative_batch_schema.json"
    schema_path.write_text(
        json.dumps(strict_multiformat_batch_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out_dir = output_dir / "multiformat_codex_run"
    prompt_dir = out_dir / "prompts"
    batch_dir = out_dir / "batches"
    event_dir = out_dir / "codex_events"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    narratives: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))

    for batch_no, start in enumerate(range(0, len(bundles), batch_size)):
        chunk = bundles[start : start + batch_size]
        first = _safe_window_id(str(chunk[0].get("window_id", "window")))
        last = _safe_window_id(str(chunk[-1].get("window_id", "window")))
        prompt = build_multiformat_prompt(chunk)
        prompt_path = (
            prompt_dir / f"multiformat_prompt_{batch_no:06d}_{first}_to_{last}.txt"
        )
        output_path = (
            batch_dir / f"multiformat_batch_{batch_no:06d}_{first}_to_{last}.json"
        )
        events_path = (
            event_dir / f"multiformat_events_{batch_no:06d}_{first}_to_{last}.jsonl"
        )
        prompt_path.write_text(prompt, encoding="utf-8")
        requested_ids = [str(bundle.get("window_id", "")) for bundle in chunk]
        if bool(args.skip_existing) and output_path.exists():
            try:
                parsed = _load_multiformat_batch(output_path)
            except (OSError, ValidationError, ValueError, json.JSONDecodeError):
                parsed = None
            if parsed is not None:
                result_rows = parsed.narratives
            else:
                result_rows = []
        elif bool(args.dry_run):
            for window_id in requested_ids:
                error_rows.append(
                    {
                        "window_id": window_id,
                        "error_type": "DryRun",
                        "message": "Prompt/schema written; Codex was not invoked.",
                        "prompt_path": str(prompt_path),
                    }
                )
            continue
        else:
            cmd = [
                "codex",
                "exec",
                "--ephemeral",
                "--json",
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
                str(output_path),
            ]
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                input=prompt,
                capture_output=True,
                timeout=int(args.timeout_seconds),
                check=False,
            )
            events_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                for window_id in requested_ids:
                    error_rows.append(
                        {
                            "window_id": window_id,
                            "error_type": "CodexExecFailed",
                            "returncode": completed.returncode,
                            "message": completed.stderr.strip()[-4000:],
                            "events_path": str(events_path),
                        }
                    )
                if not bool(args.continue_on_error):
                    break
                continue
            try:
                result_rows = _load_multiformat_batch(output_path).narratives
            except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
                for window_id in requested_ids:
                    error_rows.append(
                        {
                            "window_id": window_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "path": str(output_path),
                        }
                    )
                if not bool(args.continue_on_error):
                    break
                continue

        by_id = {str(row.window_id): row for row in result_rows}
        for window_id in requested_ids:
            narrative = by_id.get(window_id)
            metadata_rows.append(
                {
                    "window_id": window_id,
                    "prompt_path": str(prompt_path),
                    "batch_output_path": str(output_path),
                    "events_path": str(events_path),
                    "batch_no": batch_no,
                }
            )
            if narrative is None:
                error_rows.append(
                    {
                        "window_id": window_id,
                        "error_type": "MissingNarrative",
                        "message": "Codex output did not contain requested window_id.",
                    }
                )
                continue
            issues = _validate_multiformat(narrative)
            validation_rows.append({"window_id": window_id, "issues": issues})
            narratives.append(narrative.model_dump())

        report = _multiformat_report(
            output_dir=output_dir,
            pipeline_report=pipeline_report,
            narratives=narratives,
            validation_rows=validation_rows,
            error_rows=error_rows,
            metadata_rows=metadata_rows,
            args=args,
        )
        write_multiformat_artifacts(output_dir, report)

    report = _multiformat_report(
        output_dir=output_dir,
        pipeline_report=pipeline_report,
        narratives=narratives,
        validation_rows=validation_rows,
        error_rows=error_rows,
        metadata_rows=metadata_rows,
        args=args,
    )
    write_multiformat_artifacts(output_dir, report)
    selected = json.loads((output_dir / "selected_support_cases.json").read_text())
    render_multiformat_review_files(
        output_dir=output_dir,
        report=report,
        selected_cases=selected.get("selected_cases", []),
    )
    return report


def _multiformat_report(
    *,
    output_dir: Path,
    pipeline_report: Path,
    narratives: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    error_issue_count = sum(
        1
        for row in validation_rows
        for issue in row.get("issues", [])
        if issue.get("severity") == "error"
    )
    warning_issue_count = sum(
        1
        for row in validation_rows
        for issue in row.get("issues", [])
        if issue.get("severity") != "error"
    )
    status = "needs_review"
    if not error_rows and error_issue_count == 0 and narratives:
        status = "pass_with_warnings" if warning_issue_count else "pass"
    return {
        "schema_version": "episode_card_v3_multiformat_codex_testflight_v1",
        "status": status,
        "pipeline_report": str(pipeline_report),
        "codex_model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
        "requested_count": int(
            len(json.loads(pipeline_report.read_text()).get("narrative_bundles", []))
        ),
        "narrative_count": len(narratives),
        "codex_error_count": len(error_rows),
        "validation_error_count": error_issue_count,
        "validation_warning_count": warning_issue_count,
        "narratives": narratives,
        "validation": validation_rows,
        "errors": error_rows,
        "metadata": metadata_rows,
        "artifact_paths": {
            "report": str(output_dir / "multiformat_codex_report.json"),
            "narratives_jsonl": str(output_dir / "multiformat_narratives.jsonl"),
            "cards_jsonl": str(output_dir / "multiformat_episode_cards.jsonl"),
            "review_markdown": str(output_dir / "multiformat_review_ready.md"),
            "single_period_review_markdown": str(
                output_dir / "review_ready_single_period.md"
            ),
        },
    }


def write_multiformat_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    write_json(output_dir / "multiformat_codex_report.json", report)
    write_jsonl(output_dir / "multiformat_narratives.jsonl", report["narratives"])
    cards = [
        multiformat_narrative_to_episode_card(
            MultiFormatNarrative.model_validate(row),
            source_path=str(output_dir / "multiformat_narratives.jsonl"),
        )
        for row in report["narratives"]
    ]
    write_jsonl(output_dir / "multiformat_episode_cards.jsonl", cards)


def multiformat_narrative_to_episode_card(
    narrative: MultiFormatNarrative,
    *,
    source_path: str,
) -> dict[str, Any]:
    """Map directly authored Codex/GPT fields into a retrieval card."""

    views: dict[str, Any] = {
        "old_factor_baseline": narrative.old_factor_baseline,
        "factor_list_baseline": narrative.factor_list_baseline,
        "technical_factor_evidence": narrative.technical_factor_evidence,
        "sparse_user_query": narrative.sparse_user_prompt,
        "weekly_risk_monitor": narrative.weekly_risk_monitor,
        "institutional_risk_committee_note": narrative.institutional_risk_committee_note,
        "mechanism_first": narrative.mechanism_first_memo,
        "full_professional": narrative.full_risk_manager_memo,
        "risk_manager_memo": narrative.risk_manager_memo,
        "hard_negative_views": list(narrative.contrastive_hard_negatives),
    }
    flat_texts = [
        text for text in views.values() if isinstance(text, str) and text.strip()
    ]
    flat_texts.extend(
        text
        for text in narrative.contrastive_hard_negatives
        if isinstance(text, str) and text.strip()
    )
    leakage_hits = _filter_negated_no_forecast_hits(
        _snippets_for_patterns("\n".join(flat_texts), LEAKAGE_PATTERNS)
    )
    return {
        "schema_version": "nl_episode_card_v3_codex_multiformat_card_v1",
        "source_path": str(source_path),
        "narrative_authoring": "direct_codex_multiformat",
        "valid_for_training_retrieval": not bool(leakage_hits),
        "window_id": narrative.window_id,
        "split": "",
        "scenario_title": narrative.scenario_title,
        "archetype": narrative.archetype,
        "archetype_confidence": str(narrative.archetype_confidence),
        "views": views,
        "view_metrics": {
            key: view_metrics(value)
            for key, value in views.items()
            if isinstance(value, str)
        },
        "caption_fields": {
            "training_caption": narrative.full_risk_manager_memo,
            "mechanical_summary": narrative.old_factor_baseline,
            "current_market_state": narrative.sparse_user_prompt,
            "trigger": "",
            "transmission": narrative.mechanism_first_memo,
            "cross_asset_reaction": narrative.weekly_risk_monitor,
            "sequence": "",
            "portfolio_vulnerability": narrative.institutional_risk_committee_note,
            "risk_manager_implication": narrative.risk_manager_memo,
            "evidence_used": list(narrative.evidence_used),
            "ambiguity_flags": list(narrative.ambiguity_flags),
            "leakage_exclusions": [],
            "no_forecast_caveat": narrative.no_forecast_caveat,
        },
        "codex_multiformat_fields": narrative.model_dump(),
        "leakage": {
            "has_leakage": bool(leakage_hits),
            "hit_count": len(leakage_hits),
            "hits": leakage_hits[:20],
        },
    }


def _case_by_window(selected_cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("window_id", "")): row for row in selected_cases}


def _clip(text: Any, limit: int = 360) -> str:
    compact = " ".join(str(text or "").split())
    return (
        compact if len(compact) <= limit else compact[:limit].rsplit(" ", 1)[0] + "..."
    )


def render_multiformat_one_page_review(
    *,
    narrative: MultiFormatNarrative,
    selected_case: dict[str, Any],
) -> str:
    lines = [
        "# One-Page Multi-Format Narrative Review",
        "",
        f"**Same historical period for all narratives:** `{narrative.window_id}`  ",
        f"**Dates:** `{selected_case.get('history_start', '')}` to `{selected_case.get('history_end', '')}`  ",
        f"**Codex label:** **{narrative.scenario_title}** (`{narrative.archetype}`, `{narrative.archetype_confidence}` confidence)",
        "",
        "## 1. Market Evidence",
        "",
        "| Market | Direction | Size | Confidence | Raw move |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in selected_case.get("market_implications", []):
        evidence = "; ".join(item.get("evidence", [])[:1]).replace("raw_change=", "")
        lines.append(
            f"| {item.get('market', '')} | {item.get('direction', '')} | "
            f"{item.get('magnitude', '')} | {item.get('confidence', '')} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## 2. Same Period, Actually Different Codex-Authored Formats",
            "",
            "| Format | What You Should Check | Narrative |",
            "|---|---|---|",
            f"| **Old factor baseline** | Mechanical baseline only | {_clip(narrative.old_factor_baseline, 300)} |",
            f"| **Factor list baseline** | Compact factor-list training view | {_clip(narrative.factor_list_baseline, 300)} |",
            f"| **Technical factor evidence** | Evidence-oriented technical view | {_clip(narrative.technical_factor_evidence, 360)} |",
            f"| **Sparse user prompt** | Mentions only 1-2 channels | {_clip(narrative.sparse_user_prompt, 300)} |",
            f"| **Weekly risk monitor** | Institutional monitor tone | {_clip(narrative.weekly_risk_monitor, 360)} |",
            f"| **Institutional risk-committee note** | Institutional risk-committee tone | {_clip(narrative.institutional_risk_committee_note, 420)} |",
            f"| **Mechanism-first memo** | Explains cause/transmission | {_clip(narrative.mechanism_first_memo, 420)} |",
            f"| **Risk-manager memo** | Concise decision memo | {_clip(narrative.risk_manager_memo, 420)} |",
            f"| **Full risk-manager memo** | Professional review standard | {_clip(narrative.full_risk_manager_memo, 500)} |",
            "",
            "## 3. Fast Verdict Checklist",
            "",
            "| Decision Question | Pass? |",
            "|---|---:|",
            "| Are the old eight training views meaningfully different? |  |",
            "| Does the sparse prompt avoid listing every factor? |  |",
            "| Does the institutional note sound committee-level rather than retail? |  |",
            "| Does the scenario title match the dominant evidence? |  |",
            "| Does the mechanism-first memo explain transmission, not just moves? |  |",
            "| Does the full memo sound like a professional risk narrative? |  |",
            "| Should this style be used for full regeneration? |  |",
            "",
            "## 4. Ambiguity And Guardrails",
            "",
            "**Ambiguity flags:**",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in narrative.ambiguity_flags)
    lines.extend(
        [
            "",
            "**No-forecast caveat:**",
            "",
            narrative.no_forecast_caveat,
            "",
            "<details>",
            "<summary>Open full fields and hard negatives</summary>",
            "",
            "### Full Risk-Manager Memo",
            "",
            narrative.full_risk_manager_memo,
            "",
            "### Concise Risk-Manager Memo",
            "",
            narrative.risk_manager_memo,
            "",
            "### Institutional Risk-Committee Note",
            "",
            narrative.institutional_risk_committee_note,
            "",
            "### Evidence Used",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in narrative.evidence_used)
    lines.extend(["", "### Contrastive Hard Negatives", ""])
    lines.extend(f"- {item}" for item in narrative.contrastive_hard_negatives)
    lines.extend(["", "### Self-Critique", ""])
    lines.extend(f"- {item}" for item in narrative.quality_self_critique)
    lines.extend(["", "</details>", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_multiformat_review_files(
    *,
    output_dir: Path,
    report: dict[str, Any],
    selected_cases: list[dict[str, Any]],
) -> None:
    by_case = _case_by_window(selected_cases)
    narratives = [
        MultiFormatNarrative.model_validate(row) for row in report.get("narratives", [])
    ]
    lines = [
        "# Corrected Multi-Format Codex Narrative Review",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Narratives generated: `{report.get('narrative_count')}` / `{report.get('requested_count')}`",
        f"- Validation errors: `{report.get('validation_error_count')}`",
        f"- Codex errors: `{report.get('codex_error_count')}`",
        f"- Model: `{report.get('codex_model')}`",
        f"- Reasoning effort: `{report.get('reasoning_effort')}`",
        "",
        "This corrected packet asks Codex to author each format separately. "
        "It should not reuse a locally derived factor list for every view.",
        "",
    ]
    for idx, narrative in enumerate(narratives, 1):
        case = by_case.get(narrative.window_id, {})
        lines.extend(
            [
                f"## {idx}. {narrative.window_id}: {narrative.scenario_title}",
                "",
                f"- Dates: `{case.get('history_start', '')}` to `{case.get('history_end', '')}`",
                f"- Archetype: `{narrative.archetype}` / `{narrative.archetype_confidence}`",
                "",
                "| Format | Narrative |",
                "|---|---|",
                f"| Factor list baseline | {_clip(narrative.factor_list_baseline, 260)} |",
                f"| Technical factor evidence | {_clip(narrative.technical_factor_evidence, 280)} |",
                f"| Sparse user prompt | {_clip(narrative.sparse_user_prompt, 260)} |",
                f"| Weekly risk monitor | {_clip(narrative.weekly_risk_monitor, 280)} |",
                f"| Institutional risk-committee note | {_clip(narrative.institutional_risk_committee_note, 320)} |",
                f"| Mechanism-first memo | {_clip(narrative.mechanism_first_memo, 320)} |",
                f"| Risk-manager memo | {_clip(narrative.risk_manager_memo, 320)} |",
                f"| Full risk-manager memo | {_clip(narrative.full_risk_manager_memo, 360)} |",
                "",
            ]
        )
    (output_dir / "multiformat_review_ready.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    if narratives:
        first = narratives[0]
        (output_dir / "review_ready_single_period.md").write_text(
            render_multiformat_one_page_review(
                narrative=first,
                selected_case=by_case.get(first.window_id, {}),
            ),
            encoding="utf-8",
        )


def run_codex_testflight(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_report = Path(args.pipeline_report)
    if not pipeline_report.exists():
        prep_args = argparse.Namespace(
            source_cards_jsonl=args.source_cards_jsonl,
            output_dir=args.output_dir,
            max_cases=args.max_cases,
            min_window_gap=args.min_window_gap,
            selection_mode=getattr(args, "selection_mode", "representative"),
            rich_stride=getattr(args, "rich_stride", 15),
        )
        prepare_testflight(prep_args)
    codex_dir = Path(args.output_dir) / "codex_caption_run"
    run_args = argparse.Namespace(
        pipeline_report=pipeline_report,
        output_dir=codex_dir,
        count=0,
        offset=0,
        split="all",
        selection_mode="ordered",
        window_id=[],
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        skip_existing=args.skip_existing,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )
    report = run_batch(run_args)
    review = review_codex_outputs(
        argparse.Namespace(
            output_dir=args.output_dir,
            codex_report=report["artifact_paths"]["report"],
        )
    )
    return {
        "status": report["status"],
        "codex_report": report["artifact_paths"]["report"],
        "review_report": review["review_report"],
        "review_markdown": review["review_markdown"],
        "caption_count": report["caption_count"],
        "validation_error_count": report["validation_error_count"],
        "codex_error_count": report["codex_error_count"],
    }


def _short(text: Any, limit: int = 900) -> str:
    compact = " ".join(str(text or "").split())
    return (
        compact if len(compact) <= limit else compact[:limit].rsplit(" ", 1)[0] + "..."
    )


def render_review_markdown(
    *,
    report: dict[str, Any],
    cards: list[dict[str, Any]],
    quality_report: dict[str, Any],
) -> str:
    lines = [
        "# Codex-Authored EpisodeCardV3 Narrative Review",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Captions generated: `{report.get('caption_count')}` / `{report.get('requested_count')}`",
        f"- Validation error count: `{report.get('validation_error_count')}`",
        f"- Codex error count: `{report.get('codex_error_count')}`",
        f"- Prompt version: `{report.get('prompt_version')}`",
        f"- Model: `{report.get('codex_model')}`",
        f"- Reasoning effort: `{report.get('reasoning_effort')}`",
        "",
        "## Quality Summary",
        "",
        f"- Episode cards: `{quality_report.get('card_count')}`",
        f"- Leakage card share: `{quality_report.get('leakage', {}).get('card_share_with_leakage')}`",
        f"- Full professional median word count: `{quality_report.get('views', {}).get('full_professional', {}).get('median_word_count')}`",
        f"- Sparse user-query median word count: `{quality_report.get('views', {}).get('sparse_user_query', {}).get('median_word_count')}`",
        "",
        "## Generated Narratives",
        "",
    ]
    captions = report.get("captions", [])
    by_window = {str(row.get("window_id", "")): row for row in captions}
    card_by_window = {str(card.get("window_id", "")): card for card in cards}
    for idx, row in enumerate(captions, 1):
        window_id = str(row.get("window_id", ""))
        card = card_by_window.get(window_id, {})
        fields = card.get("caption_fields", {})
        lines.extend(
            [
                f"### {idx}. {window_id}: {row.get('scenario_title', '')}",
                "",
                f"- Archetype: `{row.get('archetype', '')}` / confidence `{row.get('archetype_confidence', '')}`",
                f"- Leakage: `{card.get('leakage', {}).get('has_leakage', False)}`",
                "",
                "**Training Caption**",
                "",
                _short(row.get("training_caption", ""), 1400),
                "",
                "**Mechanism And Risk-Manager Sections**",
                "",
                f"- Mechanical summary: {_short(fields.get('mechanical_summary', ''), 650)}",
                f"- Current market state: {_short(fields.get('current_market_state', ''), 650)}",
                f"- Trigger: {_short(fields.get('trigger', ''), 650)}",
                f"- Transmission: {_short(fields.get('transmission', ''), 650)}",
                f"- Cross-asset reaction: {_short(fields.get('cross_asset_reaction', ''), 650)}",
                f"- Portfolio vulnerability: {_short(fields.get('portfolio_vulnerability', ''), 650)}",
                f"- Risk-manager implication: {_short(fields.get('risk_manager_implication', ''), 650)}",
                "",
                "**Evidence Used**",
                "",
            ]
        )
        for item in row.get("evidence_used", [])[:8]:
            lines.append(f"- {_short(item, 300)}")
        lines.extend(["", "**Contrastive Captions**", ""])
        for item in row.get("contrastive_captions", [])[:4]:
            lines.append(f"- {_short(item, 500)}")
        if by_window.get(window_id, {}).get("quality_self_critique"):
            lines.extend(["", "**Self-Critique**", ""])
            for item in row.get("quality_self_critique", [])[:4]:
                lines.append(f"- {_short(item, 400)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def review_codex_outputs(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    report = json.loads(Path(args.codex_report).read_text(encoding="utf-8"))
    captions_path = Path(report["artifact_paths"]["captions_jsonl"])
    cards, quality_report = build_cards_from_caption_jsonl(captions_path)
    for card in cards:
        card["narrative_authoring"] = "single_caption_local_view_projection"
        card["valid_for_training_retrieval"] = False
        card["invalid_for_training_retrieval_reason"] = (
            "This review path locally projects one Codex/API caption into "
            "multiple views. It is review-only; use run-multiformat and "
            "multiformat_episode_cards.jsonl for retrieval/training cards."
        )
    card_path = output_dir / "codex_authored_episode_cards.jsonl"
    quality_path = output_dir / "codex_authored_quality_report.json"
    review_report_path = output_dir / "codex_authored_review_report.json"
    review_markdown_path = output_dir / "codex_authored_narrative_review.md"
    write_jsonl(card_path, cards)
    quality_report["artifact_paths"] = {
        "cards_jsonl": str(card_path),
        "quality_report": str(quality_path),
    }
    write_json(quality_path, quality_report)
    validation_error_count = int(report.get("validation_error_count", 0))
    codex_error_count = int(report.get("codex_error_count", 0))
    leakage_count = sum(
        1 for card in cards if card.get("leakage", {}).get("has_leakage")
    )
    warnings = [
        row
        for row in report.get("validation", [])
        for _warning in row.get("warnings", [])
    ]
    review_report = {
        "schema_version": "episode_card_v3_codex_authored_testflight_review_v1",
        "status": (
            "pass"
            if validation_error_count == 0
            and codex_error_count == 0
            and leakage_count == 0
            else "needs_review"
        ),
        "codex_report": str(args.codex_report),
        "caption_count": int(report.get("caption_count", 0)),
        "requested_count": int(report.get("requested_count", 0)),
        "validation_error_count": validation_error_count,
        "codex_error_count": codex_error_count,
        "validation_warning_count": len(warnings),
        "leakage_card_count": leakage_count,
        "scenario_title_counts": dict(
            Counter(
                str(row.get("scenario_title", "")) for row in report.get("captions", [])
            )
        ),
        "archetype_counts": dict(
            Counter(str(row.get("archetype", "")) for row in report.get("captions", []))
        ),
        "artifact_paths": {
            "review_report": str(review_report_path),
            "review_markdown": str(review_markdown_path),
            "cards_jsonl": str(card_path),
            "quality_report": str(quality_path),
        },
    }
    write_json(review_report_path, review_report)
    review_markdown_path.write_text(
        render_review_markdown(
            report=report,
            cards=cards,
            quality_report=quality_report,
        ),
        encoding="utf-8",
    )
    return review_report["artifact_paths"] | {
        "status": review_report["status"],
        "caption_count": review_report["caption_count"],
        "review_report": str(review_report_path),
        "review_markdown": str(review_markdown_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--source-cards-jsonl", type=Path, default=DEFAULT_SOURCE_CARDS)
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prep.add_argument("--max-cases", type=int, default=10)
    prep.add_argument("--min-window-gap", type=int, default=180)
    prep.add_argument(
        "--selection-mode",
        choices=["representative", "stride", "all"],
        default="representative",
    )
    prep.add_argument("--rich-stride", type=int, default=15)

    run = sub.add_parser("run-codex")
    run.add_argument("--source-cards-jsonl", type=Path, default=DEFAULT_SOURCE_CARDS)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument(
        "--pipeline-report",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/codex_testflight_pipeline_report.json.",
    )
    run.add_argument("--max-cases", type=int, default=10)
    run.add_argument("--min-window-gap", type=int, default=180)
    run.add_argument(
        "--selection-mode",
        choices=["representative", "stride", "all"],
        default="representative",
    )
    run.add_argument("--rich-stride", type=int, default=15)
    run.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    run.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    run.add_argument("--timeout-seconds", type=int, default=900)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--skip-existing", action="store_true", default=True)
    run.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    review = sub.add_parser("review")
    review.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    review.add_argument(
        "--codex-report",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "codex_caption_run/codex_caption_batch_report.json",
    )

    multi = sub.add_parser("run-multiformat")
    multi.add_argument("--source-cards-jsonl", type=Path, default=DEFAULT_SOURCE_CARDS)
    multi.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    multi.add_argument(
        "--pipeline-report",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/codex_testflight_pipeline_report.json.",
    )
    multi.add_argument("--max-cases", type=int, default=10)
    multi.add_argument("--min-window-gap", type=int, default=180)
    multi.add_argument(
        "--selection-mode",
        choices=["representative", "stride", "all"],
        default="representative",
    )
    multi.add_argument("--rich-stride", type=int, default=15)
    multi.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    multi.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    multi.add_argument("--timeout-seconds", type=int, default=900)
    multi.add_argument("--batch-size", type=int, default=2)
    multi.add_argument("--skip-existing", action="store_true", default=True)
    multi.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    multi.add_argument("--continue-on-error", action="store_true")
    multi.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_testflight(args)
    elif args.command == "run-codex":
        if args.pipeline_report is None:
            args.pipeline_report = (
                Path(args.output_dir) / "codex_testflight_pipeline_report.json"
            )
        result = run_codex_testflight(args)
    elif args.command == "review":
        result = review_codex_outputs(args)
    elif args.command == "run-multiformat":
        if args.pipeline_report is None:
            args.pipeline_report = (
                Path(args.output_dir) / "codex_testflight_pipeline_report.json"
            )
        result = run_multiformat_codex(args)
    else:  # pragma: no cover
        raise ValueError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
