#!/usr/bin/env python
"""Pilot Codex-authored sparse user variants with matched hard negatives.

The script prepares structured market evidence for one historical episode,
selects incompatible historical negative candidates, asks Codex/GPT to author
multiple sparse user-like positive/negative pairs, validates shortcut failures,
and writes a human-reviewable packet. It does not author narrative prose
locally.
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
from experiments.backfill.block_ar.nl_hard_negative_bank_regenerate import (  # noqa: E402
    _contradiction_channels,
    _mechanical_summary,
    _sign_vector,
    _window_number,
)


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)
DEFAULT_SUPPORT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_support_bank_cards_all_970f/"
    "episode_narrative_support_cards.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "sparse_user_variant_pilot_986a"
)
DEFAULT_TARGET_WINDOW_ID = "joint39_train_1553"
DEFAULT_ANGLES = (
    "tape_read",
    "portfolio_concern",
    "macro_channel",
    "credit_ambiguity",
    "rates_commodities",
    "desk_note",
)
LEAKAGE_PATTERNS = (
    re.compile(r"\bnext\s+\d+\s+(day|days|trading\s+day|trading\s+days)\b", re.I),
    re.compile(r"\brealized\s+(future|post[- ]window|outcome|path)\b", re.I),
    re.compile(r"\bterminal\s+(path|value|level|return|move|distribution)\b", re.I),
    re.compile(r"\bforecast\s+horizon\b", re.I),
    re.compile(r"\bVaR\b"),
    re.compile(r"\bES\b"),
    re.compile(r"\bP&L\b", re.I),
    re.compile(r"\bwill\s+(rally|fall|rise|drop|sell off|tighten|widen)\b", re.I),
)
INTERNAL_PATTERNS = (
    re.compile(r"\bhard[- ]negative\b", re.I),
    re.compile(r"\bnegative example\b", re.I),
    re.compile(r"\bpositive example\b", re.I),
    re.compile(r"\bcontrastive\b", re.I),
    re.compile(r"\btraining data\b", re.I),
    re.compile(r"\bembedding\b", re.I),
    re.compile(r"\bretrieval\b", re.I),
    re.compile(r"\bscenario generator\b", re.I),
)
DIRECT_NEGATION_PATTERNS = (
    re.compile(r"\bnot\s+(?:a|an|the)?\s*[\w-]+", re.I),
    re.compile(r"\bopposite\s+of\b", re.I),
    re.compile(r"\binconsistent\s+with\b", re.I),
    re.compile(r"\bconflicts?\s+with\b", re.I),
    re.compile(r"\bcontradicts?\b", re.I),
)
PHRASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "is",
    "of",
    "the",
    "this",
    "to",
    "with",
}


class SparsePilotPair(BaseModel):
    """One Codex-authored sparse positive and its matched hard negative."""

    model_config = ConfigDict(extra="forbid")

    angle: str = Field(min_length=3)
    positive_sparse_text: str = Field(min_length=8)
    negative_window_id: str = Field(min_length=3)
    negative_sparse_text: str = Field(min_length=8)
    quality_notes: list[str] = Field(default_factory=list)


class SparsePilotBatch(BaseModel):
    """Strict Codex output schema for the sparse pilot."""

    model_config = ConfigDict(extra="forbid")

    target_window_id: str = Field(min_length=3)
    target_title: str = Field(min_length=3)
    pairs: list[SparsePilotPair] = Field(default_factory=list)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _require_all_properties(schema: dict[str, Any]) -> dict[str, Any]:
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


def strict_schema() -> dict[str, Any]:
    return _require_all_properties(SparsePilotBatch.model_json_schema())


def _support_date_lookup(support_cards_jsonl: Path) -> dict[str, str]:
    dates: dict[str, str] = {}
    path = _resolve(support_cards_jsonl)
    if not path.exists():
        return dates
    for row in _read_jsonl(support_cards_jsonl):
        metadata = row.get("support_metadata")
        if not isinstance(metadata, dict):
            continue
        start = metadata.get("calendar_start_date")
        end = metadata.get("calendar_end_date")
        if start and end:
            dates[str(row.get("window_id"))] = f"{start} to {end}"
    return dates


def _evidence_used(card: dict[str, Any]) -> list[str]:
    fields = card.get("caption_fields")
    if not isinstance(fields, dict):
        return []
    evidence = fields.get("evidence_used")
    if isinstance(evidence, list):
        return [_compact(item) for item in evidence if _compact(item)]
    if evidence:
        return [_compact(evidence)]
    return []


def _target_payload(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": str(card.get("window_id", "")),
        "scenario_title": str(card.get("scenario_title", "")),
        "archetype": str(card.get("archetype", "")),
        "mechanical_summary": _mechanical_summary(card),
        "evidence_used": _evidence_used(card),
    }


def select_negative_candidates(
    *,
    cards: list[dict[str, Any]],
    target_window_id: str,
    count: int,
    min_temporal_gap: int = 30,
) -> list[dict[str, Any]]:
    """Select structured near-miss incompatible windows for sparse pilots."""

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

    # Prefer near misses for sparse text: enough contradictions to matter, but
    # not the all-channel crisis opposite that creates trivial separation.
    score = -np.abs(contradictions - 3.0) + 0.18 * agreements
    score[target_idx] = -1e9
    score[~temporal_ok] = -1e9
    score[contradictions < 1] = -1e9
    primary = [int(idx) for idx in np.argsort(-score) if score[int(idx)] > -1e8]
    if len(primary) < count:
        fallback_score = contradictions - 0.05 * agreements
        fallback_score[target_idx] = -1e9
        fallback_score[~temporal_ok] = -1e9
        fallback_score[contradictions < 1] = -1e9
        for idx in np.argsort(-fallback_score):
            value = int(idx)
            if fallback_score[value] <= -1e8:
                continue
            if value not in primary:
                primary.append(value)
            if len(primary) >= count:
                break

    candidates: list[dict[str, Any]] = []
    for idx in primary[:count]:
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


def build_sparse_pilot_prompt(
    *,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    angles: list[str],
) -> str:
    """Build the Codex prompt for sparse positive/negative pair authoring."""

    payload = {
        "target": target,
        "negative_candidates": negative_candidates,
        "required_angles": angles,
    }
    return (
        "You are authoring a small pilot dataset for narrative-conditioned "
        "financial scenario retrieval. Return only JSON matching the provided "
        "schema; do not wrap JSON in markdown.\n\n"
        "Task: for the single target historical 30-day current/recent market "
        "prefix, write multiple sparse, realistic user-like POSITIVE inputs, "
        "one for each requested angle. For each positive input, choose one "
        "linked negative candidate and write a same-angle sparse hard-negative "
        "input describing that negative candidate's current/recent prefix.\n\n"
        "The positive and negative texts must both be newly authored by "
        "Codex/GPT. Do not use local templates, copied factor lists, or copied "
        "source prose. Use the mechanical summaries only as evidence.\n\n"
        "Sparse user-like style:\n"
        "- 4 to 28 words per text, usually one sentence or a clipped desk note.\n"
        "- Sound like a user describing market conditions, not a polished memo.\n"
        "- Mention one or two economically important channels, not every factor.\n"
        "- Fragmentary but interpretable language is allowed.\n"
        "- Keep it prefix/current-state only; do not ask for a forecast.\n\n"
        "Hard-negative quality rules:\n"
        "- The negative text must describe only the chosen negative candidate.\n"
        "- Match the same angle and sparse style as the positive text.\n"
        "- Do not use direct negation or meta-contrast shortcuts.\n"
        "- Do not write phrases like 'not X', 'opposite of X', 'inconsistent with X', "
        "'conflicts with X', or 'contradicts X'.\n"
        "- Do not reuse the target title, the positive phrase, or distinctive "
        "multi-word phrases from the positive text.\n"
        "- Do not mention hard negatives, positives, contrastive training, "
        "embeddings, retrieval, or scenario generators.\n"
        "- Do not invent named real-world news events.\n"
        "- Do not mention future horizons, terminal moves, generated scenarios, "
        "VaR, ES, or P&L.\n\n"
        "Output requirements:\n"
        "- target_window_id must equal the target window id.\n"
        "- target_title must equal the target scenario title.\n"
        "- Return exactly one pair per required angle.\n"
        "- Every pair must include angle, positive_sparse_text, "
        "negative_window_id, negative_sparse_text, and quality_notes.\n"
        "- negative_window_id must be one of the supplied negative candidates.\n"
        "- quality_notes should explain the economic mismatch in one or two "
        "short notes without using training-language labels.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2, sort_keys=True)}\n"
    )


def _phrases(text: str, *, max_count: int = 80) -> set[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z-]*\b", text or "")
        if token.lower() not in PHRASE_STOPWORDS
    ]
    phrases: set[str] = set()
    for size in (2, 3, 4):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[idx : idx + size])
            if len(phrase) >= 10:
                phrases.add(phrase)
            if len(phrases) >= max_count:
                return phrases
    return phrases


def _pattern_hits(patterns: tuple[re.Pattern[str], ...], text: str) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def _token_jaccard(left: str, right: str) -> float:
    ignore = {"the", "a", "an", "and", "or", "of", "to", "in", "with", "is", "are"}
    left_tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", left or "")
        if token.lower() not in ignore
    }
    right_tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z'-]+\b", right or "")
        if token.lower() not in ignore
    }
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def validate_pair(
    pair: SparsePilotPair,
    *,
    target_title: str,
    positive_text: str,
) -> list[dict[str, Any]]:
    """Validate one pair for sparse-pilot shortcut failures."""

    errors: list[dict[str, Any]] = []
    positive = _compact(positive_text)
    negative = _compact(pair.negative_sparse_text)
    positive_words = _word_count(_compact(pair.positive_sparse_text))
    negative_words = _word_count(negative)
    if positive_words < 4 or positive_words > 28:
        errors.append(
            {
                "code": "positive_sparse_length_out_of_range",
                "word_count": positive_words,
                "min_words": 4,
                "max_words": 28,
            }
        )
    if negative_words < 4 or negative_words > 32:
        errors.append(
            {
                "code": "negative_sparse_length_out_of_range",
                "word_count": negative_words,
                "min_words": 4,
                "max_words": 32,
            }
        )
    negation_hits = _pattern_hits(DIRECT_NEGATION_PATTERNS, negative)
    if negation_hits:
        errors.append({"code": "direct_negation_shortcut", "patterns": negation_hits})
    leakage_hits = _pattern_hits(LEAKAGE_PATTERNS, negative)
    if leakage_hits:
        errors.append({"code": "future_or_metric_leakage", "patterns": leakage_hits})
    internal_hits = _pattern_hits(INTERNAL_PATTERNS, negative)
    if internal_hits:
        errors.append({"code": "internal_training_language", "patterns": internal_hits})
    protected_phrases = _phrases(target_title) | _phrases(positive)
    reused = sorted(
        phrase for phrase in protected_phrases if phrase and phrase in negative.lower()
    )
    if reused:
        errors.append({"code": "target_phrase_reuse", "phrases": reused[:10]})
    similarity = _token_jaccard(positive, negative)
    if similarity >= 0.45:
        errors.append(
            {
                "code": "high_positive_negative_token_overlap",
                "token_jaccard": round(similarity, 4),
            }
        )
    return errors


def validate_batch(
    *,
    batch: SparsePilotBatch,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    angles: list[str],
) -> dict[str, Any]:
    candidate_ids = {str(row["window_id"]) for row in negative_candidates}
    errors: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
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
    seen_angles = [pair.angle for pair in batch.pairs]
    missing_angles = sorted(set(angles) - set(seen_angles))
    extra_angles = sorted(set(seen_angles) - set(angles))
    if missing_angles:
        errors.append({"code": "missing_angles", "angles": missing_angles})
    if extra_angles:
        errors.append({"code": "unexpected_angles", "angles": extra_angles})
    for pair in batch.pairs:
        pair_errors = validate_pair(
            pair,
            target_title=target["scenario_title"],
            positive_text=pair.positive_sparse_text,
        )
        if pair.negative_window_id not in candidate_ids:
            pair_errors.append(
                {
                    "code": "negative_window_not_in_candidates",
                    "negative_window_id": pair.negative_window_id,
                }
            )
        validation.append(
            {
                "angle": pair.angle,
                "negative_window_id": pair.negative_window_id,
                "errors": pair_errors,
            }
        )
        if pair_errors:
            errors.append(
                {
                    "code": "pair_validation_failed",
                    "angle": pair.angle,
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
    batch: SparsePilotBatch,
    target: dict[str, Any],
    negative_candidates: list[dict[str, Any]],
    date_lookup: dict[str, str],
    validation: dict[str, Any],
    source_paths: dict[str, str],
) -> str:
    """Build a human-reviewable sparse-pair packet."""

    candidates = _candidate_by_id(negative_candidates)
    lines: list[str] = [
        "# Sparse User Variant Pilot: Paired Positives and Hard Negatives",
        "",
        "Generated: 2026-06-03",
        "",
        "This packet is a pilot for the sparse-user narrative fix. Structured "
        "market evidence and negative-window candidates were computed locally; "
        "the positive and hard-negative narrative text was authored by Codex/GPT.",
        "",
        "## Selected Positive Episode",
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
            f"- Validation status: `{validation['status']}`",
            f"- Validation error count: `{validation['error_count']}`",
            f"- Source cards: `{source_paths['cards_jsonl']}`",
            f"- Support-date metadata: `{source_paths['support_cards_jsonl']}`",
            f"- Codex output: `{source_paths['codex_output']}`",
            "",
            "## Paired Sparse Review",
            "",
        ]
    )
    for idx, pair in enumerate(batch.pairs, 1):
        candidate = candidates.get(pair.negative_window_id, {})
        pair_validation = next(
            (
                row
                for row in validation.get("validation", [])
                if row.get("angle") == pair.angle
                and row.get("negative_window_id") == pair.negative_window_id
            ),
            {"errors": []},
        )
        lines.extend(
            [
                f"### {idx}. {pair.angle}",
                "",
                "Positive sparse input:",
                "",
                f"> {_compact(pair.positive_sparse_text)}",
                "",
                "Corresponding sparse hard negative:",
                "",
                f"- Negative window: `{pair.negative_window_id}`",
                f"- Negative observed prefix date range: {date_lookup.get(pair.negative_window_id, 'unavailable')}",
                f"- Negative title: {candidate.get('scenario_title', '')}",
                f"- Negative archetype: `{candidate.get('archetype', '')}`",
                f"- Contradiction channels: {', '.join(candidate.get('contradiction_channels', []))}",
                f"- Agreement count: `{candidate.get('agreement_count', '')}`",
                "",
                "Negative mechanical summary:",
                "",
                f"> {candidate.get('mechanical_summary', '')}",
                "",
                "Hard-negative sparse input:",
                "",
                f"> {_compact(pair.negative_sparse_text)}",
                "",
                "Quality notes:",
                "",
            ]
        )
        for note in pair.quality_notes:
            lines.append(f"- {_compact(note)}")
        if not pair.quality_notes:
            lines.append("- None supplied.")
        lines.extend(["", "Validation:", ""])
        errors = pair_validation.get("errors") or []
        if errors:
            for error in errors:
                lines.append(f"- `{error.get('code')}`: `{json.dumps(error, sort_keys=True)}`")
        else:
            lines.append("- Pass.")
        lines.append("")
    lines.extend(
        [
            "## Human Review Questions",
            "",
            "- Do the positives sound like plausible short user inputs rather than polished memos?",
            "- Does each negative match the same sparse angle without saying it is a negative?",
            "- Are the negatives near-miss enough, or still too obviously crisis/opposite?",
            "- Are any positives or negatives too short to carry the intended market condition?",
            "- Are any texts accidentally forecast-like rather than prefix/current-state descriptions?",
            "",
        ]
    )
    return "\n".join(lines)


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = _read_jsonl(Path(args.cards_jsonl))
    by_id = {str(card.get("window_id", "")): card for card in cards}
    if args.target_window_id not in by_id:
        raise ValueError(f"target window not found: {args.target_window_id}")
    target = _target_payload(by_id[args.target_window_id])
    angles = [angle.strip() for angle in str(args.angles).split(",") if angle.strip()]
    negative_candidates = select_negative_candidates(
        cards=cards,
        target_window_id=args.target_window_id,
        count=max(int(args.negative_candidate_count), len(angles)),
        min_temporal_gap=int(args.min_temporal_gap),
    )
    prompt = build_sparse_pilot_prompt(
        target=target,
        negative_candidates=negative_candidates,
        angles=angles,
    )
    schema_path = output_dir / "sparse_pilot_schema.json"
    prompt_path = output_dir / "sparse_pilot_prompt.txt"
    codex_output_path = output_dir / "sparse_pilot_codex_output.json"
    events_path = output_dir / "sparse_pilot_codex_events.jsonl"
    report_path = output_dir / "sparse_pilot_report.json"
    review_path = output_dir / "sparse_pilot_review.md"
    _write_json(schema_path, strict_schema())
    _write_text(prompt_path, prompt)

    started = time.time()
    errors: list[dict[str, Any]] = []
    batch: SparsePilotBatch | None = None
    if bool(args.dry_run):
        errors.append({"code": "dry_run", "message": "Prompt/schema written; Codex not invoked."})
    else:
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
            str(codex_output_path),
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
        _write_text(events_path, completed.stdout)
        if completed.returncode != 0:
            errors.append(
                {
                    "code": "codex_exec_failed",
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-4000:],
                    "events_path": str(events_path),
                }
            )
        else:
            try:
                batch = SparsePilotBatch.model_validate_json(
                    _extract_json_object(codex_output_path.read_text(encoding="utf-8"))
                )
            except (OSError, ValidationError, ValueError, json.JSONDecodeError) as exc:
                errors.append(
                    {
                        "code": "codex_output_parse_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    validation: dict[str, Any]
    if batch is None:
        validation = {"status": "fail", "error_count": len(errors), "errors": errors, "validation": []}
    else:
        validation = validate_batch(
            batch=batch,
            target=target,
            negative_candidates=negative_candidates,
            angles=angles,
        )
        errors.extend(validation["errors"])
    source_paths = {
        "cards_jsonl": str(args.cards_jsonl),
        "support_cards_jsonl": str(args.support_cards_jsonl),
        "codex_output": str(codex_output_path),
    }
    if batch is not None:
        review = build_review_markdown(
            batch=batch,
            target=target,
            negative_candidates=negative_candidates,
            date_lookup=_support_date_lookup(Path(args.support_cards_jsonl)),
            validation=validation,
            source_paths=source_paths,
        )
        _write_text(review_path, review)
    report = {
        "schema_version": "sparse_user_variant_pilot_report_v1",
        "status": "pass" if batch is not None and not errors else "fail",
        "target_window_id": args.target_window_id,
        "angles": angles,
        "negative_candidate_count": len(negative_candidates),
        "validation": validation,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-cards-jsonl", type=Path, default=DEFAULT_SUPPORT_CARDS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-window-id", default=DEFAULT_TARGET_WINDOW_ID)
    parser.add_argument("--angles", default=",".join(DEFAULT_ANGLES))
    parser.add_argument("--negative-candidate-count", type=int, default=12)
    parser.add_argument("--min-temporal-gap", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout-seconds", type=int, default=900)
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
