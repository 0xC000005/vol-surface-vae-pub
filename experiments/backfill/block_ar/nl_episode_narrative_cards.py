#!/usr/bin/env python
"""Build multi-view historical episode cards for narrative retrieval.

This is an isolated Phase 0 utility for the episode-level narrative retrieval
branch. It reads existing caption artifacts and writes richer retrieval cards
without modifying the incumbent caption generator or scenario pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


DEFAULT_CAPTION_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_codex_full_corpus_956d/"
    "codex_caption_batch_captions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_retrieval_phase0_cards"
)

FACTOR_TERMS = (
    "SPX",
    "VIX",
    "BBB",
    "AAA",
    "OAS",
    "US2Y",
    "US10Y",
    "DXY",
    "USDJPY",
    "Gold",
    "Crude",
    "Oil",
    "Treasury",
    "equity",
    "equities",
    "credit spread",
    "credit spreads",
    "volatility",
    "dollar",
    "yen",
)

MECHANISM_TERMS = (
    "because",
    "catalyst",
    "trigger",
    "transmission",
    "mechanism",
    "sequence",
    "driven",
    "due to",
    "risk appetite",
    "liquidity",
    "funding",
    "policy",
    "growth",
    "inflation",
    "de-risk",
    "safe-haven",
    "carry",
    "hedging",
)

LEAKAGE_PATTERNS = (
    re.compile(r"\bnext\s+\d+\s+(day|days|trading\s+day|trading\s+days)\b", re.I),
    re.compile(r"\brealized\s+(future|post-window|post window)\b", re.I),
    re.compile(r"\bgenerated\s+scenario\b", re.I),
    re.compile(r"\bterminal\s+(path|value|level|return|move)\b", re.I),
    re.compile(r"\btarget\s+p&l\b", re.I),
    re.compile(r"\bVaR\b"),
    re.compile(r"\bES\b"),
    re.compile(r"\bwill\s+(rally|fall|rise|drop|sell off|tighten|widen)\b", re.I),
)


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _join_sentences(parts: list[Any]) -> str:
    cleaned = [_compact(part) for part in parts if _compact(part)]
    text = " ".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _term_count(text: str, terms: tuple[str, ...]) -> int:
    count = 0
    for term in terms:
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text, re.I):
            count += 1
    return count


def _snippets_for_patterns(
    text: str, patterns: tuple[re.Pattern[str], ...]
) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            hits.append(
                {
                    "pattern": pattern.pattern,
                    "snippet": text[start:end].strip(),
                }
            )
    return hits


def view_metrics(text: str) -> dict[str, Any]:
    """Return retrieval-facing quality metrics for one text view."""

    leakage_hits = _snippets_for_patterns(text, LEAKAGE_PATTERNS)
    return {
        "word_count": _word_count(text),
        "factor_term_count": _term_count(text, FACTOR_TERMS),
        "mechanism_term_count": _term_count(text, MECHANISM_TERMS),
        "leakage_hit_count": len(leakage_hits),
    }


def _field(caption: dict[str, Any], name: str) -> str:
    return _compact(caption.get(name, ""))


def _list_field(caption: dict[str, Any], name: str) -> list[str]:
    value = caption.get(name, [])
    if not isinstance(value, list):
        return []
    return [_compact(item) for item in value if _compact(item)]


def _first_sentence(text: str) -> str:
    text = _compact(text)
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s+", text)
    return text[: match.start()].strip() if match else text


def _remove_factor_terms(text: str) -> str:
    """Make sparse query views less dependent on explicit factor tickers."""

    cleaned = _compact(text)
    replacements = {
        "SPX": "equity market",
        "VIX": "volatility",
        "BBB OAS": "credit spreads",
        "AAA OAS": "high-grade credit spreads",
        "US2Y": "front-end rates",
        "US10Y": "long-end rates",
        "DXY": "the broad dollar",
        "USDJPY": "yen cross",
        "Gold": "safe-haven asset",
        "gold": "safe-haven asset",
        "Crude oil": "energy prices",
        "crude oil": "energy prices",
        "Crude": "energy prices",
        "crude": "energy prices",
    }
    for old, new in replacements.items():
        cleaned = re.sub(r"\b" + re.escape(old) + r"\b", new, cleaned)
    return _compact(cleaned)


def _build_views(caption: dict[str, Any]) -> dict[str, Any]:
    title = _field(caption, "scenario_title")
    archetype = _field(caption, "archetype")
    confidence = _field(caption, "archetype_confidence") or "medium"
    ambiguity = "; ".join(_list_field(caption, "ambiguity_flags")[:2])
    evidence = "; ".join(_list_field(caption, "evidence_used")[:4])
    contrastive = _list_field(caption, "contrastive_captions")

    full_professional = _join_sentences(
        [
            f"Scenario title: {title}.",
            f"Archetype: {archetype} ({confidence} confidence).",
            f"Mechanical summary: {_field(caption, 'mechanical_summary')}",
            f"Current/recent state: {_field(caption, 'current_market_state')}",
            f"Trigger: {_field(caption, 'trigger')}",
            f"Transmission: {_field(caption, 'transmission')}",
            f"Sequence: {_field(caption, 'sequence')}",
            f"Cross-asset reaction: {_field(caption, 'cross_asset_reaction')}",
            f"Portfolio vulnerability: {_field(caption, 'portfolio_vulnerability')}",
            f"Risk-manager implication: {_field(caption, 'risk_manager_implication')}",
            f"Evidence: {evidence}",
            f"Ambiguity: {ambiguity}",
            "No-forecast caveat: this is not a forecast; it is current/recent conditioning context only.",
        ]
    )

    sparse_user_query = _join_sentences(
        [
            title,
            f"Regime: {archetype}." if archetype else "",
            _remove_factor_terms(_first_sentence(_field(caption, "trigger"))),
            _remove_factor_terms(_first_sentence(_field(caption, "transmission"))),
            _remove_factor_terms(
                _first_sentence(_field(caption, "risk_manager_implication"))
            ),
            "Current/recent setup only; not a requested future path.",
        ]
    )

    mechanism_first = _join_sentences(
        [
            f"Dominant mechanism: {_field(caption, 'transmission')}",
            f"Trigger: {_field(caption, 'trigger')}",
            f"Sequence: {_field(caption, 'sequence')}",
            f"Portfolio read-through: {_field(caption, 'portfolio_vulnerability')}",
            "This describes the conditioning prefix, not a forecast.",
        ]
    )

    one_or_two_factor_headline = _join_sentences(
        [
            title,
            _first_sentence(_field(caption, "mechanical_summary")),
            _first_sentence(_field(caption, "risk_manager_implication")),
        ]
    )

    factor_list_baseline = _join_sentences(
        [
            _field(caption, "training_caption"),
            _field(caption, "current_market_state"),
            _field(caption, "cross_asset_reaction"),
        ]
    )

    return {
        "full_professional": full_professional,
        "sparse_user_query": sparse_user_query,
        "mechanism_first": mechanism_first,
        "one_or_two_factor_headline": one_or_two_factor_headline,
        "factor_list_baseline": factor_list_baseline,
        "hard_negative_views": contrastive,
    }


def build_episode_card(caption: dict[str, Any], *, source_path: str) -> dict[str, Any]:
    """Convert one caption record into a multi-view episode retrieval card."""

    views = _build_views(caption)
    flat_texts = [text for key, text in views.items() if isinstance(text, str)]
    flat_texts.extend(
        text for text in views.get("hard_negative_views", []) if isinstance(text, str)
    )
    leakage_hits = _snippets_for_patterns("\n".join(flat_texts), LEAKAGE_PATTERNS)
    metrics = {
        key: view_metrics(value)
        for key, value in views.items()
        if isinstance(value, str)
    }
    return {
        "schema_version": "nl_episode_narrative_card_v1",
        "source_path": str(source_path),
        "window_id": _field(caption, "window_id"),
        "split": _field(caption, "manifest_split"),
        "scenario_title": _field(caption, "scenario_title"),
        "archetype": _field(caption, "archetype"),
        "archetype_confidence": _field(caption, "archetype_confidence"),
        "views": views,
        "view_metrics": metrics,
        "caption_fields": {
            "training_caption": _field(caption, "training_caption"),
            "mechanical_summary": _field(caption, "mechanical_summary"),
            "current_market_state": _field(caption, "current_market_state"),
            "trigger": _field(caption, "trigger"),
            "transmission": _field(caption, "transmission"),
            "cross_asset_reaction": _field(caption, "cross_asset_reaction"),
            "sequence": _field(caption, "sequence"),
            "portfolio_vulnerability": _field(caption, "portfolio_vulnerability"),
            "risk_manager_implication": _field(caption, "risk_manager_implication"),
            "evidence_used": _list_field(caption, "evidence_used"),
            "ambiguity_flags": _list_field(caption, "ambiguity_flags"),
            "leakage_exclusions": _list_field(caption, "leakage_exclusions"),
            "no_forecast_caveat": _field(caption, "no_forecast_caveat"),
        },
        "leakage": {
            "has_leakage": bool(leakage_hits),
            "hit_count": len(leakage_hits),
            "hits": leakage_hits[:20],
        },
    }


def _median(values: list[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: list[int | float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _view_summary(cards: list[dict[str, Any]], view_name: str) -> dict[str, Any]:
    metrics = [
        card["view_metrics"][view_name]
        for card in cards
        if view_name in card.get("view_metrics", {})
    ]
    return {
        "count": len(metrics),
        "median_word_count": _median([item["word_count"] for item in metrics]),
        "mean_word_count": _mean([item["word_count"] for item in metrics]),
        "median_factor_term_count": _median(
            [item["factor_term_count"] for item in metrics]
        ),
        "mean_factor_term_count": _mean(
            [item["factor_term_count"] for item in metrics]
        ),
        "median_mechanism_term_count": _median(
            [item["mechanism_term_count"] for item in metrics]
        ),
        "mean_mechanism_term_count": _mean(
            [item["mechanism_term_count"] for item in metrics]
        ),
        "leakage_hit_count": int(sum(item["leakage_hit_count"] for item in metrics)),
    }


def summarize_corpus_quality(
    cards: list[dict[str, Any]], *, source_path: str
) -> dict[str, Any]:
    """Summarize whether existing captions are enough for text retrieval."""

    training_summary = _view_summary(cards, "factor_list_baseline")
    sparse_summary = _view_summary(cards, "sparse_user_query")
    mechanism_summary = _view_summary(cards, "mechanism_first")
    full_summary = _view_summary(cards, "full_professional")
    leakage_count = sum(
        1 for card in cards if card.get("leakage", {}).get("has_leakage")
    )
    training_mechanism_counts = [
        _term_count(card["caption_fields"].get("training_caption", ""), MECHANISM_TERMS)
        for card in cards
    ]
    training_factor_counts = [
        _term_count(card["caption_fields"].get("training_caption", ""), FACTOR_TERMS)
        for card in cards
    ]
    with_mechanism = sum(1 for value in training_mechanism_counts if value > 0)
    factor_heavy = sum(1 for value in training_factor_counts if value >= 5)
    card_count = len(cards)

    requires_enrichment = (
        training_summary["median_factor_term_count"] >= 3
        or _median(training_mechanism_counts) < 1
        or sparse_summary["median_word_count"]
        >= full_summary["median_word_count"] * 0.8
    )
    reason_parts: list[str] = []
    if training_summary["median_factor_term_count"] >= 3:
        reason_parts.append("training captions are factor-list heavy")
    if _median(training_mechanism_counts) < 1:
        reason_parts.append("many training captions have thin mechanism language")
    if sparse_summary["median_word_count"] >= full_summary["median_word_count"] * 0.8:
        reason_parts.append("sparse views are not meaningfully shorter than full views")
    if not reason_parts:
        reason_parts.append(
            "existing captions look adequate for a local retrieval pilot"
        )

    return {
        "schema_version": "nl_episode_narrative_corpus_quality_v1",
        "source_path": str(source_path),
        "card_count": card_count,
        "training_caption": {
            "median_word_count": _median(
                [
                    _word_count(card["caption_fields"].get("training_caption", ""))
                    for card in cards
                ]
            ),
            "mean_word_count": _mean(
                [
                    _word_count(card["caption_fields"].get("training_caption", ""))
                    for card in cards
                ]
            ),
            "median_factor_term_count": _median(training_factor_counts),
            "mean_factor_term_count": _mean(training_factor_counts),
            "median_mechanism_term_count": _median(training_mechanism_counts),
            "mean_mechanism_term_count": _mean(training_mechanism_counts),
            "factor_heavy_share": factor_heavy / card_count if card_count else 0.0,
        },
        "views": {
            "full_professional": full_summary,
            "sparse_user_query": sparse_summary,
            "mechanism_first": mechanism_summary,
            "factor_list_baseline": training_summary,
        },
        "mechanism_coverage": {
            "training_caption_with_mechanism_count": with_mechanism,
            "training_caption_with_mechanism_share": (
                with_mechanism / card_count if card_count else 0.0
            ),
        },
        "leakage": {
            "card_count_with_leakage": leakage_count,
            "card_share_with_leakage": (
                leakage_count / card_count if card_count else 0.0
            ),
        },
        "recommendation": {
            "requires_multi_view_enrichment": bool(requires_enrichment),
            "reason": "; ".join(reason_parts),
        },
    }


def load_caption_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_cards_from_caption_jsonl(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    captions = load_caption_jsonl(path)
    cards = [build_episode_card(caption, source_path=str(path)) for caption in captions]
    report = summarize_corpus_quality(cards, source_path=str(path))
    return cards, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-jsonl", type=Path, default=DEFAULT_CAPTION_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    cards, report = build_cards_from_caption_jsonl(args.caption_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = args.output_dir / "episode_narrative_cards.jsonl"
    report_path = args.output_dir / "corpus_quality_report.json"
    write_jsonl(cards_path, cards)
    report["artifact_paths"] = {
        "cards_jsonl": str(cards_path),
        "report": str(report_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "ok", **report["artifact_paths"], "card_count": len(cards)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
