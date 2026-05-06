#!/usr/bin/env python
"""Web-evidence enrichment for narrative-conditioned scenario labels.

This is deliberately a separate evidence layer. It can promote some catalyst
stories to cited external events for reporting, but it does not rewrite the
market-implication text used as the primary generator condition target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_scenario_descriptions import (  # noqa: E402
    CatalystCitation,
    load_dotenv_key,
    write_jsonl,
)


class CitedCatalystEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    category: str = Field(min_length=1)
    grounding_status: Literal["cited_external_event", "historical_analogy", "unsupported"]
    description: str = Field(min_length=1)
    event_date: str = ""
    date_match: Literal["inside_window", "near_window", "outside_window", "unknown"] = "unknown"
    market_linkage: list[str] = Field(default_factory=list)
    citations: list[CatalystCitation] = Field(default_factory=list)
    relevance_score: float = Field(ge=0.0, le=1.0)


class WebEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str = Field(min_length=1)
    calendar_start_date: str = ""
    calendar_end_date: str = ""
    forecast_start_date: str = ""
    forecast_end_date: str = ""
    search_query: str = Field(min_length=1)
    evidence_summary: str = ""
    cited_events: list[CitedCatalystEvent] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    usage_policy: str = (
        "Use cited events for report explanation and audit only; keep generator "
        "conditioning anchored to explicit market_implications."
    )


MAGNITUDE_WEIGHTS = {
    "flat": 0.0,
    "none": 0.0,
    "small": 1.0,
    "medium": 2.0,
    "large": 3.0,
}

MARKET_WEIGHTS = {
    "SPX": 1.4,
    "VIX": 1.4,
    "BBB_OAS": 1.3,
    "AAA_OAS": 1.1,
    "US2Y": 1.1,
    "US10Y": 1.1,
    "CRUDE_OIL": 1.1,
    "GOLD": 1.0,
    "DXY": 1.0,
    "USDJPY": 1.0,
    "IV_SURFACE": 1.2,
    "IV_SKEW": 1.1,
}

MARKET_TAGS = {
    "SPX": "equity",
    "VIX": "vol",
    "BBB_OAS": "credit",
    "AAA_OAS": "credit",
    "US2Y": "rates",
    "US10Y": "rates",
    "CRUDE_OIL": "commodity",
    "GOLD": "safe_haven_fx",
    "DXY": "safe_haven_fx",
    "USDJPY": "safe_haven_fx",
    "IV_SURFACE": "iv",
    "IV_SKEW": "iv",
}


def _compact_bundle_for_search(bundle: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields useful for the web-evidence prompt."""

    return {
        "window_id": bundle.get("window_id"),
        "calendar": bundle.get("calendar", {}),
        "market_implications": bundle.get("market_implications", []),
        "narrative_catalysts": [
            {
                "label": item.get("label"),
                "category": item.get("category"),
                "grounding_status": item.get("grounding_status"),
                "description": item.get("description"),
                "market_linkage": item.get("market_linkage", []),
            }
            for item in bundle.get("narrative_catalysts", [])
        ],
        "primary_narrative": (
            bundle.get("narratives", [{}])[0].get("text")
            if bundle.get("narratives")
            else ""
        ),
    }


def score_bundle_salience(bundle: dict[str, Any]) -> dict[str, Any]:
    """Score how much a window is worth external evidence enrichment.

    The score intentionally uses only already-extracted market implications,
    not hard-coded macro narratives. High-scoring windows are those where the
    model label has large, multi-market moves that a risk manager would usually
    want explained by a cited report narrative.
    """

    score = 0.0
    tags: set[str] = set()
    contributions: list[dict[str, Any]] = []
    for implication in bundle.get("market_implications", []) or []:
        market = str(implication.get("market", "")).upper()
        direction = str(implication.get("direction", "")).lower()
        magnitude = str(implication.get("magnitude", "")).lower()
        magnitude_weight = MAGNITUDE_WEIGHTS.get(magnitude, 1.0 if direction != "flat" else 0.0)
        if direction == "flat":
            magnitude_weight = 0.0
        market_weight = MARKET_WEIGHTS.get(market, 1.0)
        contribution = magnitude_weight * market_weight
        if contribution <= 0.0:
            continue
        tag = MARKET_TAGS.get(market, "other")
        tags.add(tag)
        contributions.append(
            {
                "market": market,
                "direction": direction,
                "magnitude": magnitude,
                "tag": tag,
                "score": round(contribution, 6),
            }
        )
        score += contribution
    if len(tags) >= 3:
        score += 0.75 * (len(tags) - 2)
    ordered_tags = sorted(tags)
    return {
        "window_id": bundle.get("window_id", ""),
        "score": round(score, 6),
        "tags": ordered_tags,
        "contributions": contributions,
    }


def select_bundles_for_enrichment(
    bundles: list[dict[str, Any]],
    *,
    eventful_windows: int,
    calm_windows: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select eventful windows plus low-salience controls for web enrichment."""

    scored = [
        {
            "index": idx,
            **score_bundle_salience(bundle),
        }
        for idx, bundle in enumerate(bundles)
    ]
    eventful_candidates = sorted(scored, key=lambda item: (-float(item["score"]), int(item["index"])))
    eventful = [
        item
        for item in eventful_candidates
        if float(item["score"]) > 0.0
    ][: max(0, int(eventful_windows))]
    used = {int(item["index"]) for item in eventful}
    calm = [
        item
        for item in sorted(scored, key=lambda item: (float(item["score"]), int(item["index"])))
        if int(item["index"]) not in used
    ][: max(0, int(calm_windows))]
    selected_indices = [int(item["index"]) for item in eventful + calm]
    selected = [bundles[idx] for idx in selected_indices]
    selection_report = {
        "selection_mode": "salience",
        "eventful_window_ids": [str(item.get("window_id", "")) for item in eventful],
        "calm_window_ids": [str(item.get("window_id", "")) for item in calm],
        "selected_window_ids": [str(bundles[idx].get("window_id", "")) for idx in selected_indices],
        "window_scores": [
            {
                key: value
                for key, value in item.items()
                if key != "index"
            }
            for item in sorted(scored, key=lambda item: int(item["index"]))
        ],
    }
    return selected, selection_report


def build_web_evidence_messages(
    bundle: dict[str, Any],
    *,
    max_events: int = 3,
) -> list[dict[str, str]]:
    """Build the prompt for cited catalyst enrichment."""

    compact = _compact_bundle_for_search(bundle)
    calendar = compact.get("calendar", {})
    start = calendar.get("calendar_start_date", "")
    end = calendar.get("calendar_end_date", "")
    system = (
        "You are an evidence auditor for a risk-manager scenario generator. "
        "Use web search to find public, timestamped market news or market "
        "commentary near the supplied historical window. Promote a catalyst to "
        "cited_external_event only when a citation supports that event and its "
        "date is inside or near the window. Keep unsupported stories as "
        "historical_analogy or unsupported. Do not change the market implication "
        "target. Return structured JSON only."
    )
    user = (
        f"Historical window: {start} to {end}.\n"
        f"Return at most {int(max_events)} cited catalyst events. Prefer sources "
        "from reputable market commentary, official releases, major news wires, "
        "or financial institutions. Each cited event must include URL, title, "
        "published_date when available, and a short evidence sentence. If no "
        "good external event is supported, return cited_events=[] and explain in "
        "rejected_claims.\n\n"
        f"Window bundle JSON:\n{json.dumps(compact, sort_keys=True)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _response_sources(response: Any) -> list[dict[str, Any]]:
    """Extract searched/cited URLs from a Responses API object when available."""

    sources: list[dict[str, Any]] = []
    for output_item in getattr(response, "output", []) or []:
        if hasattr(output_item, "model_dump"):
            try:
                item = output_item.model_dump(warnings=False)
            except TypeError:
                item = output_item.model_dump()
        else:
            item = output_item
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if isinstance(action, dict):
            for source in action.get("sources", []) or []:
                if isinstance(source, dict):
                    sources.append(source)
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []) or []:
                if isinstance(annotation, dict) and annotation.get("type") == "url_citation":
                    sources.append(
                        {
                            "type": "url",
                            "url": annotation.get("url", ""),
                            "title": annotation.get("title", ""),
                        }
                    )
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = (str(source.get("url", "")), str(source.get("title", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def enrich_bundle_with_openai_web(
    bundle: dict[str, Any],
    *,
    model: str,
    dotenv_path: str | Path = ".env",
    max_events: int = 3,
    max_output_tokens: int = 3000,
    search_context_size: Literal["low", "medium", "high"] = "low",
) -> dict[str, Any]:
    """Run web-search evidence enrichment for one narrative bundle."""

    load_dotenv_key(dotenv_path)
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": search_context_size,
            }
        ],
        include=["web_search_call.action.sources"],
        input=build_web_evidence_messages(bundle, max_events=max_events),
        text_format=WebEvidenceReport,
        max_output_tokens=int(max_output_tokens),
        store=False,
    )
    report = response.output_parsed.model_dump()
    return {
        "window_id": bundle.get("window_id"),
        "calendar": bundle.get("calendar", {}),
        "web_evidence": report,
        "raw_sources": _response_sources(response),
        "market_implications": bundle.get("market_implications", []),
        "original_catalysts": bundle.get("narrative_catalysts", []),
    }


def load_bundles_from_pipeline_report(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bundles = payload.get("narrative_bundles", [])
    if not isinstance(bundles, list):
        raise ValueError("pipeline report does not contain narrative_bundles list")
    return bundles


def summarize_enrichment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    cited_count = 0
    source_count = 0
    for row in rows:
        evidence = row.get("web_evidence", {})
        source_count += len(row.get("raw_sources", []) or [])
        for event in evidence.get("cited_events", []) or []:
            status = str(event.get("grounding_status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "cited_external_event":
                cited_count += 1
    return {
        "window_count": len(rows),
        "cited_external_event_count": cited_count,
        "event_status_counts": status_counts,
        "raw_source_count": source_count,
    }


def _citation_domain(citation: dict[str, Any]) -> str:
    parsed = urlparse(str(citation.get("url", "")))
    return parsed.netloc.lower()


def build_evidence_review_report(
    rows: list[dict[str, Any]],
    *,
    min_relevance: float = 0.6,
    selection_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Review cited events for reporting-quality evidence constraints."""

    flag_counts: dict[str, int] = {}
    source_domain_counts: dict[str, int] = {}
    date_match_counts: dict[str, int] = {}
    event_reviews: list[dict[str, Any]] = []
    accepted_count = 0
    cited_count = 0

    for row in rows:
        window_id = str(row.get("window_id", ""))
        evidence = row.get("web_evidence", {}) or {}
        for event in evidence.get("cited_events", []) or []:
            status = str(event.get("grounding_status", ""))
            if status != "cited_external_event":
                continue
            cited_count += 1
            date_match = str(event.get("date_match", "unknown"))
            date_match_counts[date_match] = date_match_counts.get(date_match, 0) + 1
            relevance = float(event.get("relevance_score", 0.0) or 0.0)
            citations = [
                citation
                for citation in event.get("citations", []) or []
                if isinstance(citation, dict)
            ]
            flags: list[str] = []
            if relevance < float(min_relevance):
                flags.append("low_relevance")
            if not citations:
                flags.append("missing_citation")
            if date_match not in {"inside_window", "near_window"}:
                flags.append("weak_date_alignment")
            for flag in flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            domains: list[str] = []
            for citation in citations:
                domain = _citation_domain(citation)
                if not domain:
                    continue
                domains.append(domain)
                source_domain_counts[domain] = source_domain_counts.get(domain, 0) + 1
            accepted = not flags
            if accepted:
                accepted_count += 1
            event_reviews.append(
                {
                    "window_id": window_id,
                    "label": event.get("label", ""),
                    "accepted_for_report": accepted,
                    "flags": flags,
                    "date_match": date_match,
                    "relevance_score": relevance,
                    "source_domains": domains,
                }
            )

    return {
        "window_count": len(rows),
        "cited_external_event_count": cited_count,
        "accepted_cited_event_count": accepted_count,
        "flag_counts": dict(sorted(flag_counts.items())),
        "date_match_counts": dict(sorted(date_match_counts.items())),
        "source_domain_counts": dict(sorted(source_domain_counts.items())),
        "min_relevance": float(min_relevance),
        "selection": selection_report or {},
        "event_reviews": event_reviews,
    }


def _cmd_enrich(args: argparse.Namespace) -> None:
    bundles = load_bundles_from_pipeline_report(args.input_report)
    if args.selection_mode == "salience":
        selected, selection_report = select_bundles_for_enrichment(
            bundles,
            eventful_windows=int(args.eventful_windows),
            calm_windows=int(args.calm_windows),
        )
    else:
        selected = bundles[: int(args.max_windows)] if int(args.max_windows) > 0 else bundles
        selection_report = {
            "selection_mode": "first",
            "selected_window_ids": [str(bundle.get("window_id", "")) for bundle in selected],
        }
    rows = [
        enrich_bundle_with_openai_web(
            bundle,
            model=args.model,
            dotenv_path=args.dotenv,
            max_events=int(args.max_events),
            max_output_tokens=int(args.max_output_tokens),
            search_context_size=args.search_context_size,
        )
        for bundle in selected
    ]
    write_jsonl(args.output, rows)
    summary = summarize_enrichment(rows)
    summary["selection"] = selection_report
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.review_output:
        review = build_evidence_review_report(
            rows,
            min_relevance=float(args.min_relevance),
            selection_report=selection_report,
        )
        Path(args.review_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.review_output).write_text(
            json.dumps(review, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        review = None
    print(
        json.dumps(
            {
                "output": args.output,
                "summary": args.summary,
                "review_output": args.review_output,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "selection"
                },
                "selection": {
                    key: value
                    for key, value in selection_report.items()
                    if key != "window_scores"
                },
                **({"accepted_cited_event_count": review["accepted_cited_event_count"]} if review else {}),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--selection-mode", choices=["first", "salience"], default="first")
    parser.add_argument("--eventful-windows", type=int, default=5)
    parser.add_argument("--calm-windows", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--search-context-size", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--review-output", default="")
    parser.add_argument("--min-relevance", type=float, default=0.6)
    args = parser.parse_args()
    _cmd_enrich(args)


if __name__ == "__main__":
    main()
