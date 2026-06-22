#!/usr/bin/env python
"""EpisodeCardV3 structured support-angle helpers.

Local deterministic narrative rendering is intentionally disabled. This file
may infer structured angle metadata from historical support facts, but it must
not produce searchable narrative prose. Searchable training/retrieval
narratives must be directly authored by Codex/GPT and validated separately.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_cards import (  # noqa: E402
    LEAKAGE_PATTERNS,
    _snippets_for_patterns,
    view_metrics,
    write_jsonl,
)


DEFAULT_SOURCE_CARDS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_support_bank_cards_all_972b_tight_taxonomy/"
    "episode_narrative_support_cards.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_testflight"
)
DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE = (
    "Local deterministic/template EpisodeCardV3 narrative generation is disabled. "
    "It is not allowed for final artifacts, smoke tests, or mechanism tests. "
    "Use nl_episode_card_v3_codex_testflight.py run-multiformat so each "
    "searchable narrative view is Codex/GPT-authored, then validate leakage "
    "and quality before retrieval/backtesting."
)

RICH_VIEWS = (
    "risk_manager_memo",
    "fed_current_conditions",
    "institutional_risk_committee_note",
    "macro_outlook_newsletter",
    "weekly_risk_monitor",
    "technical_factor_evidence",
)
DAILY_VIEWS = ("sparse_user_query", "technical_factor_evidence")
SOURCE_STYLE = {
    "risk_manager_memo": "risk_manager_like",
    "fed_current_conditions": "fed_like",
    "institutional_risk_committee_note": "stress_test_like",
    "macro_outlook_newsletter": "macro_outlook_like",
    "weekly_risk_monitor": "weekly_monitor_like",
    "technical_factor_evidence": "technical_like",
    "sparse_user_query": "sparse_user_like",
}


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _window_index(card: dict[str, Any]) -> int:
    metadata = card.get("support_metadata", {})
    if isinstance(metadata, dict) and metadata.get("window_index") is not None:
        return int(metadata["window_index"])
    if card.get("window_index") is not None:
        return int(card["window_index"])
    match = re.search(r"_(\d+)$", str(card.get("window_id", "")))
    if match:
        return int(match.group(1))
    raise ValueError(f"card has no window index: {card.get('window_id')}")


def _row_by_market(card: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = card.get("support_metadata", {}).get("support_move_rows", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("market")): row
        for row in rows
        if isinstance(row, dict) and row.get("market")
    }


def _is(rows: dict[str, dict[str, Any]], market: str, direction: str) -> bool:
    return str(rows.get(market, {}).get("direction")) == direction


def _is_flat_or(rows: dict[str, dict[str, Any]], market: str, direction: str) -> bool:
    observed = str(rows.get(market, {}).get("direction"))
    return observed == direction or observed == "flat"


def _movement(row: dict[str, Any]) -> str:
    market = str(row.get("market", "market"))
    label = str(row.get("direction_label", row.get("direction", "moved")))
    magnitude = str(row.get("magnitude", ""))
    raw = row.get("raw_change")
    z = row.get("z_change")
    detail = ""
    if raw is not None and z is not None:
        detail = f" ({float(raw):+g}, {float(z):+g} sigma)"
    return _compact(f"{market} {label} {magnitude}{detail}")


def _evidence_for_anchors(
    card: dict[str, Any], anchors: list[str], *, limit: int = 6
) -> list[str]:
    rows_by_market = _row_by_market(card)
    evidence: list[str] = []
    seen: set[str] = set()
    for market in anchors:
        row = rows_by_market.get(market)
        if row and market not in seen:
            evidence.append(_movement(row))
            seen.add(market)
    source_evidence = card.get("caption_fields", {}).get("evidence_used", [])
    if isinstance(source_evidence, list):
        for item in source_evidence:
            text = _compact(item)
            if text and text not in evidence:
                evidence.append(text)
            if len(evidence) >= limit:
                break
    return evidence[:limit]


def _angle_spec(
    title: str,
    archetype: str,
    confidence: str,
    anchors: list[str],
    mechanism: str,
    contradiction_note: str = "",
) -> dict[str, Any]:
    return {
        "angle_name": title,
        "archetype": archetype,
        "confidence": confidence,
        "anchor_markets": anchors,
        "mechanism": mechanism,
        "contradiction_note": contradiction_note,
    }


def infer_supported_angles(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer all evidence-supported angle records for one support card."""

    rows = _row_by_market(card)
    gold_up = _is(rows, "GOLD", "up")
    us10y_down = _is(rows, "US10Y", "down")
    vix_up = _is(rows, "VIX", "up")
    vix_down = _is(rows, "VIX", "down")
    spx_up = _is(rows, "SPX", "up")
    spx_down = _is(rows, "SPX", "down")
    bbb_wider = _is(rows, "BBB_OAS", "up")
    bbb_tighter = _is(rows, "BBB_OAS", "down")
    specs: list[dict[str, Any]] = []

    if gold_up and us10y_down and vix_up and spx_down:
        specs.append(
            _angle_spec(
                "Classic safe-haven gold risk-off",
                "financial_accident",
                "high",
                ["GOLD", "US10Y", "VIX", "SPX"],
                (
                    "Gold and duration are bid while volatility rises and "
                    "equities weaken, consistent with a clean defensive "
                    "safe-haven prefix."
                ),
            )
        )
    if (
        gold_up
        and us10y_down
        and (vix_up or _is_flat_or(rows, "SPX", "down") or bbb_wider)
    ):
        specs.append(
            _angle_spec(
                "Safe-haven gold bid",
                "financial_accident",
                "medium",
                ["GOLD", "US10Y", "VIX", "SPX"],
                (
                    "Gold and duration are supported, with at least partial "
                    "stress confirmation from volatility, equities, or credit."
                ),
            )
        )
    if gold_up and us10y_down and not (vix_up and spx_down):
        specs.append(
            _angle_spec(
                "Gold-duration bid without risk-off confirmation",
                "mixed_ambiguous",
                "low",
                ["GOLD", "US10Y", "VIX", "SPX"],
                (
                    "Gold and long-duration assets are supported, but the "
                    "broader equity or volatility tape does not confirm a "
                    "clean risk-off regime."
                ),
                "Risk-off confirmation is incomplete.",
            )
        )
    if gold_up and spx_up and vix_down:
        specs.append(
            _angle_spec(
                "Gold up in risk-on relief",
                "mixed_ambiguous",
                "low",
                ["GOLD", "SPX", "VIX"],
                (
                    "Gold is bid even as equities recover and volatility "
                    "falls, so the episode reads as risk-on relief with a "
                    "Gold bid rather than classic safe-haven stress."
                ),
                "Equity and volatility confirmation conflict with classic safe-haven risk-off.",
            )
        )
    if gold_up and (vix_up or spx_down or bbb_wider):
        specs.append(
            _angle_spec(
                "Gold up mixed defensive",
                "mixed_ambiguous",
                "medium",
                ["GOLD", "VIX", "SPX", "BBB_OAS"],
                (
                    "Gold is higher and at least one defensive channel is "
                    "visible, but the full Gold-duration safe-haven pattern "
                    "is incomplete."
                ),
            )
        )
    if _is(rows, "DXY", "up") and (spx_down or vix_up or bbb_wider):
        specs.append(
            _angle_spec(
                "Dollar-liquidity squeeze",
                "liquidity_withdrawal",
                "high",
                ["DXY", "SPX", "VIX", "BBB_OAS"],
                (
                    "Dollar strength appears alongside stress in equities, "
                    "volatility, or credit, consistent with a liquidity "
                    "withdrawal channel."
                ),
            )
        )
    if _is(rows, "DXY", "up") and spx_up and vix_down:
        specs.append(
            _angle_spec(
                "Dollar strength in risk-on relief",
                "mixed_ambiguous",
                "low",
                ["DXY", "SPX", "VIX"],
                (
                    "The dollar is firmer, but equities and volatility look "
                    "more consistent with risk-on relief than funding stress."
                ),
                "Dollar strength is not confirmed by defensive risk channels.",
            )
        )
    if (_is(rows, "DXY", "up") or us10y_down) and (spx_down or vix_up or bbb_wider):
        specs.append(
            _angle_spec(
                "Dollar/rates defensive regime",
                "financial_accident",
                "medium",
                ["DXY", "US10Y", "SPX", "VIX", "BBB_OAS"],
                (
                    "Defensive pressure is visible through dollar or duration "
                    "demand, but Gold is not necessarily the dominant channel."
                ),
            )
        )
    if _is(rows, "CRUDE_OIL", "up") and (
        _is(rows, "US10Y", "up") or _is(rows, "US2Y", "up")
    ):
        specs.append(
            _angle_spec(
                "Commodity-inflation pressure",
                "inflation_supply_shock",
                "high",
                ["CRUDE_OIL", "US10Y", "US2Y"],
                (
                    "Energy prices and rates are firmer, consistent with an "
                    "inflation, real-rate, or supply-pressure channel."
                ),
            )
        )
    if _is(rows, "US10Y", "up") and (spx_down or vix_up):
        specs.append(
            _angle_spec(
                "Rates-led tightening pressure",
                "policy_overshoot",
                "medium",
                ["US10Y", "US2Y", "SPX", "VIX"],
                (
                    "Higher yields tighten financial conditions and pressure "
                    "duration-sensitive risk assets."
                ),
            )
        )
    if spx_down and (vix_up or bbb_wider):
        specs.append(
            _angle_spec(
                "Defensive risk-off shock",
                "financial_accident",
                "high" if vix_up and bbb_wider else "medium",
                ["SPX", "VIX", "BBB_OAS", "US10Y", "GOLD"],
                (
                    "Risk tolerance is being withdrawn as hedging demand rises "
                    "and equity or credit premia become less benign."
                ),
            )
        )
    if spx_up and (vix_down or bbb_tighter):
        specs.append(
            _angle_spec(
                "Fragile risk-on rebound",
                "liquidity_surge",
                "high" if vix_down and bbb_tighter else "medium",
                ["SPX", "VIX", "BBB_OAS"],
                (
                    "Risk appetite is returning as lower volatility or calmer "
                    "credit conditions support equity beta and carry."
                ),
            )
        )

    if not specs:
        specs.append(
            _angle_spec(
                "Mixed cross-asset regime",
                "mixed_ambiguous",
                "low",
                [],
                (
                    "The prefix contains market movement, but no single "
                    "dominant risk channel is cleanly isolated."
                ),
                "No high-confidence title anchor dominates the prefix.",
            )
        )

    primary = str(card.get("scenario_title", ""))
    specs.sort(key=lambda spec: 0 if spec["angle_name"] == primary else 1)
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["angle_name"])
        if name in seen:
            continue
        unique.append(spec)
        seen.add(name)
    return unique


def _technical_evidence(card: dict[str, Any], spec: dict[str, Any]) -> str:
    evidence = _evidence_for_anchors(card, list(spec["anchor_markets"]), limit=6)
    return "; ".join(evidence) if evidence else "cross-asset moves are mixed"


def _sparse_user_prompt(spec: dict[str, Any]) -> str:
    raise RuntimeError(DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE)

    title = str(spec["angle_name"])
    prompts = {
        "Fragile risk-on rebound": (
            "Risk appetite looks like it is coming back; volatility feels "
            "calmer and equities are recovering."
        ),
        "Defensive risk-off shock": (
            "Equities look fragile and hedging demand is rising."
        ),
        "Commodity-inflation pressure": (
            "Energy is firm and rates are pushing higher."
        ),
        "Dollar-liquidity squeeze": (
            "The dollar feels firm and risk assets look vulnerable."
        ),
        "Rates-led tightening pressure": (
            "Long-end rates are rising and duration-sensitive risk looks under pressure."
        ),
        "Classic safe-haven gold risk-off": (
            "Gold is bid, duration is rallying, and risk assets look under pressure."
        ),
        "Safe-haven gold bid": (
            "Gold and duration are supported, with some signs of defensive demand."
        ),
        "Gold-duration bid without risk-off confirmation": (
            "Gold and duration are bid, but the rest of the tape is not clearly risk-off."
        ),
        "Gold up in risk-on relief": (
            "Gold is catching a bid even though the broader tape looks like risk relief."
        ),
        "Gold up mixed defensive": (
            "Gold is higher and there are some defensive signals in the tape."
        ),
        "Dollar/rates defensive regime": (
            "The dollar or duration bid looks defensive, but the signal is not purely Gold-led."
        ),
        "Dollar strength in risk-on relief": (
            "The dollar is firm, but equities and volatility do not look stressed."
        ),
    }
    return prompts.get(title, "The cross-asset tape looks mixed and hard to classify.")


def _view_text(card: dict[str, Any], spec: dict[str, Any], view_name: str) -> str:
    raise RuntimeError(DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE)

    title = str(spec["angle_name"])
    confidence = str(spec["confidence"])
    mechanism = str(spec["mechanism"])
    evidence = _technical_evidence(card, spec)
    ambiguity = str(
        spec.get("contradiction_note")
        or "The exact catalyst is not supplied by the market data."
    )
    caveat = (
        "This describes the current/recent 30-day prefix only; it is not a forecast."
    )

    if view_name == "sparse_user_query":
        return f"{_sparse_user_prompt(spec)} {caveat}"
    if view_name == "technical_factor_evidence":
        return (
            f"{title}. Confidence: {confidence}. Evidence: {evidence}. "
            f"Mechanism: {mechanism} {caveat}"
        )
    if view_name == "risk_manager_memo":
        return (
            f"Scenario title: {title}. Mechanical summary: {evidence}. "
            f"Dominant mechanism: {mechanism} Portfolio implication: read the "
            f"episode through the named risk channel and its cross-asset "
            f"confirmation. Ambiguity: {ambiguity} {caveat}"
        )
    if view_name == "fed_current_conditions":
        return (
            f"Recent financial conditions are consistent with {title.lower()}. "
            f"The evidence is {evidence}. The description emphasizes current "
            f"conditions and uncertainty rather than a projected path. {caveat}"
        )
    if view_name == "institutional_risk_committee_note":
        return (
            f"Risk committee summary: the current/recent prefix is best read as "
            f"{title.lower()} with evidence in "
            f"{', '.join(spec['anchor_markets']) or 'the cross-asset tape'}. "
            f"The affected exposures depend on the named risk channel, and the "
            f"transmission channel is: {mechanism} {caveat}"
        )
    if view_name == "macro_outlook_newsletter":
        return (
            f"The macro read is {title.lower()}: {mechanism} The market evidence "
            f"is {evidence}. The key uncertainty is whether this is a durable "
            f"regime or a positioning-led episode. {caveat}"
        )
    if view_name == "weekly_risk_monitor":
        return (
            f"What changed: {title}. Signal: {evidence}. Risk channel: "
            f"{mechanism} Monitoring point: look for confirmation or reversal "
            f"in the anchor markets. {caveat}"
        )
    raise ValueError(f"unknown V3 view: {view_name}")


def _hard_negative_spec(spec: dict[str, Any]) -> dict[str, Any]:
    title = str(spec["angle_name"])
    if "risk-on" in title.lower() or "rebound" in title.lower():
        return _angle_spec(
            "Hard negative: defensive risk-off shock",
            "hard_negative",
            "reject",
            ["SPX", "VIX", "BBB_OAS"],
            "This near miss reverses the risk-on channel and should not rank as the same support.",
            "Hard negative for risk-on language.",
        )
    if "gold" in title.lower() or "safe-haven" in title.lower():
        return _angle_spec(
            "Hard negative: gold bid without safe-haven confirmation",
            "hard_negative",
            "reject",
            ["GOLD", "SPX", "VIX"],
            "This near miss mentions Gold but lacks the defensive confirmation required for classic safe-haven risk-off.",
            "Hard negative for broad safe-haven language.",
        )
    if "commodity" in title.lower() or "inflation" in title.lower():
        return _angle_spec(
            "Hard negative: commodity language without crude confirmation",
            "hard_negative",
            "reject",
            ["CRUDE_OIL", "US10Y"],
            "This near miss uses inflation language but lacks commodity/rates evidence.",
            "Hard negative for commodity-inflation language.",
        )
    if "dollar" in title.lower() or "liquidity" in title.lower():
        return _angle_spec(
            "Hard negative: dollar language without funding stress",
            "hard_negative",
            "reject",
            ["DXY", "SPX", "VIX"],
            "This near miss mentions dollar strength but lacks funding or risk-asset stress confirmation.",
            "Hard negative for dollar-liquidity language.",
        )
    return _angle_spec(
        "Hard negative: mixed regime with missing anchors",
        "hard_negative",
        "reject",
        [],
        "This near miss lacks the required anchor evidence for the positive title.",
        "Hard negative for mixed regime language.",
    )


def build_episode_card_v3(
    source_card: dict[str, Any],
    *,
    rich: bool,
    include_hard_negative: bool = True,
) -> dict[str, Any]:
    """Return one grouped EpisodeCardV3 object from a tightened source card."""

    raise RuntimeError(DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE)

    window_index = _window_index(source_card)
    stride_type = "rich_15d" if rich else "sparse_daily"
    positive_angles = infer_supported_angles(source_card)
    view_names = tuple(RICH_VIEWS) + ("sparse_user_query",) if rich else DAILY_VIEWS
    records: list[dict[str, Any]] = []

    for angle_no, spec in enumerate(positive_angles):
        for view_name in view_names:
            text = _view_text(source_card, spec, view_name)
            records.append(
                {
                    "record_id": (
                        f"{source_card.get('window_id', f'window_{window_index:04d}')}"
                        f"::{angle_no:02d}::{view_name}"
                    ),
                    "episode_id": str(
                        source_card.get("window_id", f"window_{window_index:04d}")
                    ),
                    "window_index": int(window_index),
                    "history_start": str(
                        source_card.get("support_metadata", {}).get(
                            "calendar_start_date", ""
                        )
                    ),
                    "history_end": str(
                        source_card.get("support_metadata", {}).get(
                            "calendar_end_date", ""
                        )
                    ),
                    "window_stride_type": stride_type,
                    "view_name": view_name,
                    "angle_name": str(spec["angle_name"]),
                    "archetype": str(spec["archetype"]),
                    "confidence": str(spec["confidence"]),
                    "narrative_text": text,
                    "short_query_variant": _sparse_user_prompt(spec),
                    "structured_claims": list(spec["anchor_markets"]),
                    "anchor_markets": list(spec["anchor_markets"]),
                    "supporting_evidence": _evidence_for_anchors(
                        source_card, list(spec["anchor_markets"])
                    ),
                    "contradictions": (
                        [str(spec["contradiction_note"])]
                        if spec.get("contradiction_note")
                        else []
                    ),
                    "inferred_channels": [str(spec["mechanism"])],
                    "source_style": SOURCE_STYLE[view_name],
                    "hard_negative": False,
                    "leakage_status": (
                        "pass"
                        if not _snippets_for_patterns(text, LEAKAGE_PATTERNS)
                        else "fail"
                    ),
                    "timestamp_safety_status": "local_prefix_only",
                }
            )

    if include_hard_negative and positive_angles:
        neg = _hard_negative_spec(positive_angles[0])
        text = _view_text(source_card, neg, "sparse_user_query")
        records.append(
            {
                "record_id": (
                    f"{source_card.get('window_id', f'window_{window_index:04d}')}"
                    "::hard_negative::sparse_user_query"
                ),
                "episode_id": str(
                    source_card.get("window_id", f"window_{window_index:04d}")
                ),
                "window_index": int(window_index),
                "history_start": str(
                    source_card.get("support_metadata", {}).get(
                        "calendar_start_date", ""
                    )
                ),
                "history_end": str(
                    source_card.get("support_metadata", {}).get("calendar_end_date", "")
                ),
                "window_stride_type": stride_type,
                "view_name": "sparse_user_query",
                "angle_name": str(neg["angle_name"]),
                "archetype": "hard_negative",
                "confidence": "reject",
                "narrative_text": text,
                "short_query_variant": _sparse_user_prompt(neg),
                "structured_claims": list(neg["anchor_markets"]),
                "anchor_markets": list(neg["anchor_markets"]),
                "supporting_evidence": [],
                "contradictions": [str(neg["contradiction_note"])],
                "inferred_channels": [str(neg["mechanism"])],
                "source_style": "sparse_user_like",
                "hard_negative": True,
                "leakage_status": (
                    "pass"
                    if not _snippets_for_patterns(text, LEAKAGE_PATTERNS)
                    else "fail"
                ),
                "timestamp_safety_status": "local_prefix_only",
            }
        )

    positive_records = [row for row in records if not row["hard_negative"]]
    views: dict[str, str] = {}
    for view_name in set(view_names):
        joined = " ".join(
            row["narrative_text"]
            for row in positive_records
            if row["view_name"] == view_name
        )
        if joined:
            views[view_name] = _compact(joined)
    views["full_professional"] = views.get(
        "risk_manager_memo",
        views.get("sparse_user_query", ""),
    )
    views["mechanism_first"] = _compact(
        " ".join(row["inferred_channels"][0] for row in positive_records[:3])
    )
    views["factor_list_baseline"] = views.get("technical_factor_evidence", "")
    hard_negatives = [row["narrative_text"] for row in records if row["hard_negative"]]
    views["hard_negative_views"] = hard_negatives

    flat_texts = [
        row["narrative_text"]
        for row in records
        if isinstance(row["narrative_text"], str)
    ]
    leakage_hits = _snippets_for_patterns("\n".join(flat_texts), LEAKAGE_PATTERNS)
    view_metric_map = {
        key: view_metrics(value)
        for key, value in views.items()
        if isinstance(value, str)
    }
    return {
        "schema_version": "nl_episode_card_v3",
        "source_schema_version": source_card.get("schema_version", ""),
        "source_path": source_card.get("source_path", ""),
        "window_id": str(source_card.get("window_id", f"window_{window_index:04d}")),
        "window_index": int(window_index),
        "split": str(source_card.get("split", "")),
        "scenario_title": str(positive_angles[0]["angle_name"]),
        "archetype": str(positive_angles[0]["archetype"]),
        "archetype_confidence": str(positive_angles[0]["confidence"]),
        "window_stride_type": stride_type,
        "episode_v3": {
            "positive_angle_count": len(positive_angles),
            "record_count": len(records),
            "rich_institutional": bool(rich),
        },
        "narrative_records": records,
        "views": views,
        "view_metrics": view_metric_map,
        "caption_fields": source_card.get("caption_fields", {}),
        "support_metadata": source_card.get("support_metadata", {}),
        "leakage": {
            "has_leakage": bool(leakage_hits),
            "hit_count": len(leakage_hits),
            "hits": leakage_hits[:20],
        },
    }


def summarize_episode_card_v3(cards: list[dict[str, Any]]) -> dict[str, Any]:
    raise RuntimeError(DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE)

    records = [
        row
        for card in cards
        for row in card.get("narrative_records", [])
        if isinstance(row, dict)
    ]
    view_counts = Counter(str(row.get("view_name", "")) for row in records)
    angle_counts = Counter(
        str(row.get("angle_name", ""))
        for row in records
        if not row.get("hard_negative")
    )
    confidence_counts = Counter(
        str(row.get("confidence", ""))
        for row in records
        if not row.get("hard_negative")
    )
    leakage_count = sum(1 for row in records if row.get("leakage_status") != "pass")
    rich_cards = [
        card for card in cards if card.get("window_stride_type") == "rich_15d"
    ]
    sparse_cards = [
        card for card in cards if card.get("window_stride_type") == "sparse_daily"
    ]
    return {
        "schema_version": "nl_episode_card_v3_quality_report",
        "card_count": len(cards),
        "record_count": len(records),
        "rich_15d_card_count": len(rich_cards),
        "sparse_daily_card_count": len(sparse_cards),
        "view_counts": dict(sorted(view_counts.items())),
        "positive_angle_counts": dict(sorted(angle_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "hard_negative_record_count": sum(
            1 for row in records if row.get("hard_negative")
        ),
        "leakage_record_count": leakage_count,
        "leakage_pass_share": (1.0 - leakage_count / len(records) if records else 0.0),
        "mean_positive_angles_per_card": (
            float(
                np.mean([card["episode_v3"]["positive_angle_count"] for card in cards])
            )
            if cards
            else 0.0
        ),
    }


def _selected_cards(
    source_cards: list[dict[str, Any]],
    *,
    rich_stride: int,
    max_rich_windows: int,
    max_cards: int,
) -> list[dict[str, Any]]:
    ordered = sorted(source_cards, key=_window_index)
    selected: list[dict[str, Any]] = []
    rich_count = 0
    for card in ordered:
        idx = _window_index(card)
        rich = idx % max(1, int(rich_stride)) == 0
        if rich and int(max_rich_windows) > 0 and rich_count >= int(max_rich_windows):
            rich = False
        if rich:
            rich_count += 1
        selected.append(build_episode_card_v3(card, rich=rich))
        if int(max_cards) > 0 and len(selected) >= int(max_cards):
            break
    return selected


def build_testflight_cards(
    *,
    source_cards_jsonl: Path,
    output_dir: Path,
    rich_stride: int,
    max_rich_windows: int,
    max_cards: int,
) -> dict[str, Any]:
    raise RuntimeError(DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE)

    source_cards = _read_jsonl(source_cards_jsonl)
    cards = _selected_cards(
        source_cards,
        rich_stride=rich_stride,
        max_rich_windows=max_rich_windows,
        max_cards=max_cards,
    )
    report = summarize_episode_card_v3(cards)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "schema_version": "nl_episode_card_v3_schema",
        "record_contract": [
            "episode_id",
            "window_index",
            "window_stride_type",
            "view_name",
            "angle_name",
            "confidence",
            "narrative_text",
            "structured_claims",
            "anchor_markets",
            "supporting_evidence",
            "contradictions",
            "source_style",
            "hard_negative",
            "leakage_status",
            "timestamp_safety_status",
        ],
        "rich_stride_days": int(rich_stride),
        "full_regeneration_allowed": False,
    }
    cards_path = output_dir / "episode_card_v3_testflight_cards.jsonl"
    report_path = output_dir / "episode_card_v3_quality_report.json"
    schema_path = output_dir / "episode_card_v3_schema.json"
    write_jsonl(cards_path, cards)
    report["artifact_paths"] = {
        "schema": str(schema_path),
        "cards_jsonl": str(cards_path),
        "quality_report": str(report_path),
    }
    report["source_cards_jsonl"] = str(source_cards_jsonl)
    _write_json(schema_path, schema)
    _write_json(report_path, report)
    return report


def _method_summary(report: dict[str, Any]) -> dict[str, Any]:
    block = report.get("summary", {}).get("narrative_generator_topk", {})
    return {
        "crps": block.get("ensemble_crps_z_mean"),
        "energy": block.get("energy_score_z_mean"),
        "coverage_80": block.get("coverage_80_mean"),
        "crps_improvement_vs_persistence": block.get(
            "ensemble_crps_z_improvement_vs_persistence"
        ),
        "energy_improvement_vs_persistence": block.get(
            "energy_score_z_improvement_vs_persistence"
        ),
        "window_count": block.get("window_count"),
    }


def summarize_testflight(
    *,
    quality_report: Path,
    v3_scenario_report: Path,
    start_only_scenario_report: Path,
    legacy_scenario_report: Path | None,
    conditionality_lift_report: Path,
    legacy_conditionality_lift_report: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    raise RuntimeError(
        "Local deterministic/template EpisodeCardV3 TestFlight summaries are "
        "disabled because they can be mistaken for valid narrative-regeneration "
        "evidence. Redo the assessment from Codex/GPT-authored multi-format "
        "episode cards."
    )

    quality = json.loads(quality_report.read_text(encoding="utf-8"))
    v3 = json.loads(v3_scenario_report.read_text(encoding="utf-8"))
    start = json.loads(start_only_scenario_report.read_text(encoding="utf-8"))
    lift = json.loads(conditionality_lift_report.read_text(encoding="utf-8"))
    legacy = (
        json.loads(legacy_scenario_report.read_text(encoding="utf-8"))
        if legacy_scenario_report
        else None
    )
    legacy_lift = (
        json.loads(legacy_conditionality_lift_report.read_text(encoding="utf-8"))
        if legacy_conditionality_lift_report
        else None
    )
    v3_metrics = _method_summary(v3)
    start_metrics = _method_summary(start)
    legacy_metrics = _method_summary(legacy) if legacy else None
    crps_delta = (
        None
        if v3_metrics["crps"] is None or start_metrics["crps"] is None
        else float(v3_metrics["crps"]) - float(start_metrics["crps"])
    )
    lift_verdict = str(lift.get("verdict", ""))
    leakage_ok = float(quality.get("leakage_pass_share", 0.0)) >= 0.999
    conditionality_ok = lift_verdict.startswith("conditionality_lift_detected")
    quality_ok = crps_delta is None or crps_delta <= 0.05

    legacy_tradeoff: dict[str, Any] = {}
    if legacy_metrics and legacy_metrics.get("crps") is not None:
        legacy_crps_delta = float(v3_metrics["crps"]) - float(legacy_metrics["crps"])
        legacy_energy_delta = float(v3_metrics["energy"]) - float(
            legacy_metrics["energy"]
        )
        legacy_tradeoff.update(
            {
                "crps_delta_vs_legacy_971_972": legacy_crps_delta,
                "energy_delta_vs_legacy_971_972": legacy_energy_delta,
            }
        )
        if legacy_lift:
            v3_agg = lift.get("aggregate", {})
            legacy_agg = legacy_lift.get("aggregate", {})
            legacy_tradeoff.update(
                {
                    "terminal_ks_delta_vs_legacy_971_972": (
                        float(v3_agg.get("mean_terminal_factor_ks", 0.0))
                        - float(legacy_agg.get("mean_terminal_factor_ks", 0.0))
                    ),
                    "path_energy_delta_vs_legacy_971_972": (
                        float(v3_agg.get("mean_path_energy_distance_z", 0.0))
                        - float(legacy_agg.get("mean_path_energy_distance_z", 0.0))
                    ),
                }
            )

    legacy_quality_ok = (
        not legacy_tradeoff
        or float(legacy_tradeoff.get("crps_delta_vs_legacy_971_972", 0.0)) <= 0.015
    )
    legacy_conditionality_better = (
        not legacy_tradeoff
        or float(legacy_tradeoff.get("terminal_ks_delta_vs_legacy_971_972", 0.0)) >= 0.0
        or float(legacy_tradeoff.get("path_energy_delta_vs_legacy_971_972", 0.0)) >= 0.0
    )
    if (
        leakage_ok
        and conditionality_ok
        and quality_ok
        and legacy_quality_ok
        and legacy_conditionality_better
    ):
        recommendation = "go_full_regeneration"
    elif leakage_ok and conditionality_ok:
        recommendation = "revise_schema"
    else:
        recommendation = "do_not_regenerate"
    summary = {
        "schema_version": "nl_episode_card_v3_testflight_summary",
        "status": "ok",
        "recommendation": recommendation,
        "decision_note": (
            "This is a TestFlight recommendation only. Full regeneration still "
            "requires explicit user approval."
        ),
        "quality_report": str(quality_report),
        "v3_metrics": v3_metrics,
        "start_only_metrics": start_metrics,
        "legacy_971_972_metrics": legacy_metrics,
        "legacy_971_972_conditionality_lift": (
            legacy_lift.get("aggregate", {}) if legacy_lift else None
        ),
        "crps_delta_vs_start_only": crps_delta,
        "legacy_971_972_tradeoff": legacy_tradeoff,
        "conditionality_lift_verdict": lift_verdict,
        "conditionality_lift_aggregate": lift.get("aggregate", {}),
        "quality_summary": {
            "card_count": quality.get("card_count"),
            "record_count": quality.get("record_count"),
            "rich_15d_card_count": quality.get("rich_15d_card_count"),
            "sparse_daily_card_count": quality.get("sparse_daily_card_count"),
            "mean_positive_angles_per_card": quality.get(
                "mean_positive_angles_per_card"
            ),
            "hard_negative_record_count": quality.get("hard_negative_record_count"),
            "leakage_pass_share": quality.get("leakage_pass_share"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "episode_card_v3_testflight_summary.json"
    summary["artifact_paths"] = {"summary": str(path)}
    _write_json(path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-cards",
        help="Disabled: local deterministic EpisodeCardV3 narrative cards are banned",
    )
    build.add_argument("--source-cards-jsonl", type=Path, default=DEFAULT_SOURCE_CARDS)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--rich-stride", type=int, default=15)
    build.add_argument("--max-rich-windows", type=int, default=100)
    build.add_argument(
        "--max-cards",
        type=int,
        default=0,
        help="Optional cap on total source cards. Default 0 uses all source cards.",
    )

    summarize = sub.add_parser("summarize", help="Summarize TestFlight results")
    summarize.add_argument("--quality-report", type=Path, required=True)
    summarize.add_argument("--v3-scenario-report", type=Path, required=True)
    summarize.add_argument("--start-only-scenario-report", type=Path, required=True)
    summarize.add_argument("--legacy-scenario-report", type=Path)
    summarize.add_argument("--conditionality-lift-report", type=Path, required=True)
    summarize.add_argument("--legacy-conditionality-lift-report", type=Path)
    summarize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args(argv)
    if args.command == "build-cards":
        report = build_testflight_cards(
            source_cards_jsonl=args.source_cards_jsonl,
            output_dir=args.output_dir,
            rich_stride=int(args.rich_stride),
            max_rich_windows=int(args.max_rich_windows),
            max_cards=int(args.max_cards),
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "cards_jsonl": report["artifact_paths"]["cards_jsonl"],
                    "quality_report": report["artifact_paths"]["quality_report"],
                    "card_count": report["card_count"],
                    "record_count": report["record_count"],
                    "rich_15d_card_count": report["rich_15d_card_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    summary = summarize_testflight(
        quality_report=args.quality_report,
        v3_scenario_report=args.v3_scenario_report,
        start_only_scenario_report=args.start_only_scenario_report,
        legacy_scenario_report=args.legacy_scenario_report,
        conditionality_lift_report=args.conditionality_lift_report,
        legacy_conditionality_lift_report=args.legacy_conditionality_lift_report,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "summary": summary["artifact_paths"]["summary"],
                "recommendation": summary["recommendation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
