#!/usr/bin/env python
"""Build broad support-bank episode cards from raw 30-day histories.

This is an isolated Phase 0b utility for episode-level narrative retrieval. It
creates searchable narrative cards for the broad historical support bank without
calling an LLM and without modifying the incumbent caption corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_cards import (  # noqa: E402
    build_episode_card,
    summarize_corpus_quality,
    write_jsonl,
)


DEFAULT_SUPPORT_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_support_bank_cards"
)
MARKETS = (
    {
        "name": "SPX",
        "index": 25,
        "up": "higher",
        "down": "lower",
        "theme": "equity market",
    },
    {
        "name": "VIX",
        "index": 38,
        "up": "higher",
        "down": "lower",
        "theme": "volatility",
    },
    {
        "name": "BBB_OAS",
        "index": 35,
        "up": "wider",
        "down": "tighter",
        "theme": "credit spreads",
    },
    {
        "name": "AAA_OAS",
        "index": 34,
        "up": "wider",
        "down": "tighter",
        "theme": "high-grade credit spreads",
    },
    {
        "name": "DXY",
        "index": 28,
        "up": "higher",
        "down": "lower",
        "theme": "broad dollar",
    },
    {
        "name": "USDJPY",
        "index": 27,
        "up": "higher",
        "down": "lower",
        "theme": "dollar-yen",
    },
    {
        "name": "CRUDE_OIL",
        "index": 31,
        "up": "higher",
        "down": "lower",
        "theme": "energy prices",
    },
    {
        "name": "US2Y",
        "index": 32,
        "up": "higher",
        "down": "lower",
        "theme": "front-end rates",
    },
    {
        "name": "US10Y",
        "index": 33,
        "up": "higher",
        "down": "lower",
        "theme": "long-end rates",
    },
    {
        "name": "GOLD",
        "index": 37,
        "up": "higher",
        "down": "lower",
        "theme": "safe-haven asset",
    },
)


def _round(value: float) -> float:
    return round(float(value), 6)


def _move_direction(
    delta: float, z_abs: float, spec: dict[str, Any]
) -> tuple[str, str]:
    if z_abs < 0.15:
        return "flat", "stable"
    return ("up", str(spec["up"])) if delta > 0 else ("down", str(spec["down"]))


def _magnitude(z_abs: float) -> str:
    if z_abs < 0.15:
        return "flat"
    if z_abs < 0.70:
        return "small"
    if z_abs < 1.40:
        return "medium"
    return "large"


def _move_rows(
    history_raw: np.ndarray, scales: np.ndarray, row_index: int
) -> list[dict[str, Any]]:
    history = np.asarray(history_raw, dtype=np.float32)
    start = history[row_index, 0]
    end = history[row_index, -1]
    rows: list[dict[str, Any]] = []
    for spec in MARKETS:
        idx = int(spec["index"])
        delta = float(end[idx] - start[idx])
        scale = max(float(scales[idx]), 1e-8)
        z = delta / scale
        direction, label = _move_direction(delta, abs(z), spec)
        rows.append(
            {
                "market": str(spec["name"]),
                "theme": str(spec["theme"]),
                "index": idx,
                "direction": direction,
                "direction_label": label,
                "magnitude": _magnitude(abs(z)),
                "raw_change": _round(delta),
                "z_change": _round(z),
            }
        )
    return rows


def _by_market(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["market"]): row for row in rows}


def _is(rows: dict[str, dict[str, Any]], market: str, direction: str) -> bool:
    return str(rows.get(market, {}).get("direction")) == direction


def _z_abs(rows: dict[str, dict[str, Any]], market: str) -> float:
    try:
        return abs(float(rows.get(market, {}).get("z_change", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _is_flat_or(rows: dict[str, dict[str, Any]], market: str, direction: str) -> bool:
    observed = str(rows.get(market, {}).get("direction"))
    return observed == direction or observed == "flat"


def _classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_market = _by_market(rows)
    gold_up = _is(by_market, "GOLD", "up")
    us10y_down = _is(by_market, "US10Y", "down")
    vix_up = _is(by_market, "VIX", "up")
    vix_down = _is(by_market, "VIX", "down")
    spx_up = _is(by_market, "SPX", "up")
    spx_down = _is(by_market, "SPX", "down")
    spx_not_up = _is_flat_or(by_market, "SPX", "down")
    bbb_wider = _is(by_market, "BBB_OAS", "up")
    bbb_tighter = _is(by_market, "BBB_OAS", "down")

    if gold_up and us10y_down and vix_up and spx_down:
        return {
            "title": "Classic safe-haven gold risk-off",
            "archetype": "financial_accident",
            "confidence": "high",
            "evidence_markets": ["GOLD", "US10Y", "VIX", "SPX"],
            "mechanism": (
                "Gold is higher, long yields are lower, volatility is higher, "
                "and equities are weaker, indicating a clean safe-haven "
                "Gold/duration risk-off prefix."
            ),
        }
    if _is(by_market, "DXY", "up") and (
        _is(by_market, "SPX", "down")
        or _is(by_market, "VIX", "up")
        or _is(by_market, "BBB_OAS", "up")
    ):
        confidence = "high" if not (spx_up and vix_down) else "low"
        if spx_up and vix_down:
            return {
                "title": "Dollar strength in risk-on relief",
                "archetype": "mixed_ambiguous",
                "confidence": "low",
                "evidence_markets": ["DXY", "SPX", "VIX", "BBB_OAS"],
                "mechanism": (
                    "The dollar is firmer, but equities and volatility look "
                    "more like risk-on relief than a clean liquidity squeeze."
                ),
            }
        return {
            "title": "Dollar-liquidity squeeze",
            "archetype": "liquidity_withdrawal",
            "confidence": confidence,
            "evidence_markets": ["DXY", "SPX", "VIX", "BBB_OAS"],
            "mechanism": (
                "Dollar strength and tighter liquidity conditions reduce risk "
                "appetite and raise hedging demand."
            ),
        }
    if _is(by_market, "CRUDE_OIL", "up") and (
        _is(by_market, "US10Y", "up") or _is(by_market, "US2Y", "up")
    ):
        confidence = (
            "high"
            if _z_abs(by_market, "CRUDE_OIL") >= 0.70
            and (_z_abs(by_market, "US10Y") >= 0.70 or _z_abs(by_market, "US2Y") >= 0.70)
            else "medium"
        )
        return {
            "title": "Commodity-inflation pressure",
            "archetype": "inflation_supply_shock",
            "confidence": confidence,
            "evidence_markets": ["CRUDE_OIL", "US10Y", "US2Y"],
            "mechanism": (
                "Higher energy prices and higher rates point to an inflation "
                "or real-rate pressure channel."
            ),
        }
    if gold_up and spx_up and vix_down:
        return {
            "title": "Gold up in risk-on relief",
            "archetype": "mixed_ambiguous",
            "confidence": "low",
            "evidence_markets": ["GOLD", "US10Y", "SPX", "VIX"],
            "mechanism": (
                "Gold is higher, but equities are firmer and volatility is "
                "lower, so the prefix is a risk-on relief tape with a Gold bid "
                "rather than a clean safe-haven stress regime."
            ),
        }
    if gold_up and us10y_down and (vix_up or spx_not_up or bbb_wider):
        return {
            "title": "Safe-haven gold bid",
            "archetype": "financial_accident",
            "confidence": "medium",
            "evidence_markets": ["GOLD", "US10Y", "VIX", "SPX"],
            "mechanism": (
                "Investors appear to be paying for safety, duration, or "
                "convexity rather than adding broad cyclical risk."
            ),
        }
    if gold_up and us10y_down:
        return {
            "title": "Gold-duration bid without risk-off confirmation",
            "archetype": "mixed_ambiguous",
            "confidence": "low",
            "evidence_markets": ["GOLD", "US10Y", "VIX", "SPX"],
            "mechanism": (
                "Gold and duration are supported, but equity, volatility, or "
                "credit stress does not confirm a clean risk-off safe-haven "
                "setup."
            ),
        }
    if gold_up and (vix_up or spx_down or bbb_wider):
        return {
            "title": "Gold up mixed defensive",
            "archetype": "mixed_ambiguous",
            "confidence": "medium",
            "evidence_markets": ["GOLD", "VIX", "SPX", "BBB_OAS"],
            "mechanism": (
                "Gold is higher and at least one defensive channel is present, "
                "but the full Gold-duration safe-haven pattern is incomplete."
            ),
        }
    if (_is(by_market, "DXY", "up") or us10y_down) and (
        spx_down or vix_up or bbb_wider
    ):
        return {
            "title": "Dollar/rates defensive regime",
            "archetype": "financial_accident",
            "confidence": "medium",
            "evidence_markets": ["DXY", "US10Y", "SPX", "VIX", "BBB_OAS"],
            "mechanism": (
                "Defensive pressure is visible through dollar or duration "
                "demand, but Gold is not the dominant safe-haven channel."
            ),
        }
    if _is(by_market, "US10Y", "up") and (
        _is(by_market, "SPX", "down") or _is(by_market, "VIX", "up")
    ):
        confidence = "high" if _z_abs(by_market, "US10Y") >= 0.70 else "medium"
        return {
            "title": "Rates-led tightening pressure",
            "archetype": "policy_overshoot",
            "confidence": confidence,
            "evidence_markets": ["US10Y", "US2Y", "SPX", "VIX"],
            "mechanism": (
                "Higher yields tighten financial conditions and pressure "
                "duration-sensitive risk assets."
            ),
        }
    if _is(by_market, "SPX", "down") and (
        _is(by_market, "VIX", "up") or _is(by_market, "BBB_OAS", "up")
    ):
        confidence = "high" if vix_up and bbb_wider else "medium"
        return {
            "title": "Defensive risk-off shock",
            "archetype": "financial_accident",
            "confidence": confidence,
            "evidence_markets": ["SPX", "VIX", "BBB_OAS", "US10Y", "GOLD"],
            "mechanism": (
                "Risk tolerance is being withdrawn as hedging demand rises and "
                "credit or equity risk premia become less benign."
            ),
        }
    if _is(by_market, "SPX", "up") and (
        _is(by_market, "VIX", "down") or _is(by_market, "BBB_OAS", "down")
    ):
        confidence = "high" if vix_down and bbb_tighter else "medium"
        return {
            "title": "Fragile risk-on rebound",
            "archetype": "liquidity_surge",
            "confidence": confidence,
            "evidence_markets": ["SPX", "VIX", "BBB_OAS"],
            "mechanism": (
                "Risk appetite is returning after stress as lower volatility "
                "and calmer credit conditions support equity beta and carry."
            ),
        }
    return {
        "title": "Mixed cross-asset regime",
        "archetype": "mixed_ambiguous",
        "confidence": "low",
        "evidence_markets": [],
        "mechanism": (
            "The prefix contains cross-asset movement but no single dominant "
            "risk channel is cleanly isolated."
        ),
    }


def _movement_sentence(
    rows: list[dict[str, Any]], *, include_flat: bool = False
) -> str:
    parts = []
    for row in rows:
        direction = str(row["direction"])
        if direction == "flat" and not include_flat:
            continue
        parts.append(
            f"{row['market']} {row['direction_label']} {row['magnitude']} "
            f"({row['raw_change']:+g}, {row['z_change']:+g} sigma)"
        )
    return "; ".join(parts) if parts else "cross-asset moves are mostly stable"


def _top_rows(rows: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    nonflat = [row for row in rows if row["direction"] != "flat"]
    ordered = sorted(nonflat, key=lambda row: abs(float(row["z_change"])), reverse=True)
    return ordered[: int(limit)]


def _evidence_rows(
    rows: list[dict[str, Any]],
    regime: dict[str, Any],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Return top evidence while forcing the title's anchor markets in."""

    by_market = _by_market(rows)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for market in regime.get("evidence_markets", []):
        row = by_market.get(str(market))
        if row is not None and str(row["market"]) not in seen:
            selected.append(row)
            seen.add(str(row["market"]))
    for row in _top_rows(rows, limit=len(rows)):
        market = str(row["market"])
        if market in seen:
            continue
        selected.append(row)
        seen.add(market)
        if len(selected) >= int(limit):
            break
    return selected[: int(limit)]


def _caption_from_support_row(
    *,
    history_raw: np.ndarray,
    scales: np.ndarray,
    row_index: int,
    metadata: dict[str, Any],
    source_path: str,
) -> dict[str, Any]:
    rows = _move_rows(history_raw, scales, row_index)
    regime = _classify(rows)
    top = _evidence_rows(rows, regime, limit=6)
    movement = _movement_sentence(top, include_flat=True)
    title = regime["title"]
    mechanism = regime["mechanism"]
    start = str(metadata.get("calendar_start_date", ""))
    end = str(metadata.get("calendar_end_date", ""))

    return {
        "window_id": str(metadata.get("window_id", f"support_{row_index:04d}")),
        "manifest_split": str(metadata.get("manifest_split", "")),
        "scenario_title": title,
        "archetype": regime["archetype"],
        "archetype_confidence": regime.get("confidence", "medium"),
        "mechanical_summary": (
            f"Over the observed 30-day support prefix ending {end}, {movement}."
        ),
        "current_market_state": movement,
        "trigger": (
            "No external catalyst is supplied; the support card infers the "
            "setup from observed market moves."
        ),
        "transmission": mechanism,
        "cross_asset_reaction": movement,
        "sequence": (
            "The support card summarizes the whole 30-day prefix by the net "
            "observed cross-asset movement from the beginning to the end of "
            "the conditioning window."
        ),
        "portfolio_vulnerability": (
            "Portfolio sensitivity should be read through the dominant equity, "
            "volatility, credit, rates, FX, commodity, and safe-haven channels "
            "listed in the support prefix."
        ),
        "risk_manager_implication": (
            "Use this card as a searchable historical support description; it "
            "is not an external-news explanation and not a future forecast."
        ),
        "evidence_used": [
            f"{row['market']}: {row['direction_label']} {row['magnitude']} "
            f"over {start} to {end}; raw={row['raw_change']:+g}, z={row['z_change']:+g}"
            for row in top
        ],
        "ambiguity_flags": [
            "local_support_bank_card_from_raw_history",
            "no_external_news_catalyst_used",
        ],
        "leakage_exclusions": [
            "realized post-window future",
            "generated scenario path",
            "terminal target return",
        ],
        "no_forecast_caveat": (
            "This is not a forecast; it describes the current/recent 30-day "
            "conditioning prefix only."
        ),
        "training_caption": (
            f"{title}: {movement}. Mechanism: {mechanism} This is a "
            "current/recent support-prefix description only."
        ),
        "contrastive_captions": [
            (
                "Opposite support negative: the same market channels move in "
                "the reverse direction with incompatible risk transmission."
            )
        ],
        "source_path": str(source_path),
        "support_move_rows": rows,
    }


def _scales_from_history(history_raw: np.ndarray) -> np.ndarray:
    history = np.asarray(history_raw, dtype=np.float32)
    deltas = history[:, -1, :] - history[:, 0, :]
    scale = np.nanstd(deltas, axis=0).astype(np.float32)
    fallback = np.nanmedian(np.abs(deltas), axis=0).astype(np.float32)
    scale = np.where(scale > 1e-8, scale, np.maximum(fallback, 1.0))
    return np.maximum(scale, 1e-8).astype(np.float32)


def _title_anchor_markets(title: str) -> list[str]:
    anchors = {
        "Classic safe-haven gold risk-off": ["GOLD", "US10Y", "VIX", "SPX"],
        "Safe-haven gold bid": ["GOLD", "US10Y", "VIX", "SPX"],
        "Gold-duration bid without risk-off confirmation": [
            "GOLD",
            "US10Y",
            "VIX",
            "SPX",
        ],
        "Gold up in risk-on relief": ["GOLD", "SPX", "VIX"],
        "Gold up mixed defensive": ["GOLD", "VIX", "SPX", "BBB_OAS"],
        "Dollar/rates defensive regime": ["DXY", "US10Y", "SPX", "VIX"],
        "Dollar-liquidity squeeze": ["DXY", "SPX", "VIX", "BBB_OAS"],
        "Dollar strength in risk-on relief": ["DXY", "SPX", "VIX"],
        "Commodity-inflation pressure": ["CRUDE_OIL", "US10Y", "US2Y"],
        "Rates-led tightening pressure": ["US10Y", "US2Y", "SPX", "VIX"],
        "Defensive risk-off shock": ["SPX", "VIX", "BBB_OAS"],
        "Fragile risk-on rebound": ["SPX", "VIX", "BBB_OAS"],
    }
    return anchors.get(str(title), [])


def _title_contract_pass(card: dict[str, Any]) -> bool:
    title = str(card.get("scenario_title", ""))
    rows = _by_market(
        card.get("support_metadata", {}).get("support_move_rows", [])
    )
    if title == "Classic safe-haven gold risk-off":
        return (
            _is(rows, "GOLD", "up")
            and _is(rows, "US10Y", "down")
            and _is(rows, "VIX", "up")
            and _is(rows, "SPX", "down")
        )
    if title == "Safe-haven gold bid":
        return (
            _is(rows, "GOLD", "up")
            and _is(rows, "US10Y", "down")
            and (
                _is(rows, "VIX", "up")
                or _is_flat_or(rows, "SPX", "down")
                or _is(rows, "BBB_OAS", "up")
            )
        )
    if title == "Gold-duration bid without risk-off confirmation":
        return (
            _is(rows, "GOLD", "up")
            and _is(rows, "US10Y", "down")
            and not (_is(rows, "VIX", "up") and _is(rows, "SPX", "down"))
        )
    if title == "Gold up in risk-on relief":
        return _is(rows, "GOLD", "up") and _is(rows, "SPX", "up") and _is(
            rows, "VIX", "down"
        )
    if title == "Gold up mixed defensive":
        return _is(rows, "GOLD", "up") and (
            _is(rows, "VIX", "up")
            or _is(rows, "SPX", "down")
            or _is(rows, "BBB_OAS", "up")
        )
    if title == "Dollar-liquidity squeeze":
        return _is(rows, "DXY", "up") and (
            _is(rows, "SPX", "down")
            or _is(rows, "VIX", "up")
            or _is(rows, "BBB_OAS", "up")
        )
    if title == "Dollar strength in risk-on relief":
        return _is(rows, "DXY", "up") and _is(rows, "SPX", "up") and _is(
            rows, "VIX", "down"
        )
    if title == "Commodity-inflation pressure":
        return _is(rows, "CRUDE_OIL", "up") and (
            _is(rows, "US10Y", "up") or _is(rows, "US2Y", "up")
        )
    if title == "Rates-led tightening pressure":
        return _is(rows, "US10Y", "up") and (
            _is(rows, "SPX", "down") or _is(rows, "VIX", "up")
        )
    if title == "Defensive risk-off shock":
        return _is(rows, "SPX", "down") and (
            _is(rows, "VIX", "up") or _is(rows, "BBB_OAS", "up")
        )
    if title == "Fragile risk-on rebound":
        return _is(rows, "SPX", "up") and (
            _is(rows, "VIX", "down") or _is(rows, "BBB_OAS", "down")
        )
    return True


def _support_card_contract_quality(cards: list[dict[str, Any]]) -> dict[str, Any]:
    title_counts = Counter(str(card.get("scenario_title", "")) for card in cards)
    confidence_counts = Counter(
        str(card.get("archetype_confidence", "")) for card in cards
    )
    by_title: dict[str, dict[str, Any]] = {}
    for title in sorted(title_counts):
        title_cards = [card for card in cards if card.get("scenario_title") == title]
        anchors = _title_anchor_markets(title)
        anchor_hits = []
        contract_hits = []
        for card in title_cards:
            evidence = " ".join(card.get("caption_fields", {}).get("evidence_used", []))
            if anchors:
                anchor_hits.append(any(anchor in evidence for anchor in anchors[:1]))
            else:
                anchor_hits.append(True)
            contract_hits.append(_title_contract_pass(card))
        by_title[title] = {
            "count": len(title_cards),
            "anchor_evidence_share": float(np.mean(anchor_hits))
            if anchor_hits
            else 0.0,
            "title_contract_pass_share": float(np.mean(contract_hits))
            if contract_hits
            else 0.0,
            "confidence_counts": dict(
                Counter(str(card.get("archetype_confidence", "")) for card in title_cards)
            ),
        }
    return {
        "scenario_title_counts": dict(title_counts),
        "archetype_confidence_counts": dict(confidence_counts),
        "by_title": by_title,
        "standard": {
            "source": "stress_scenario_narrative_contract",
            "description": (
                "Each title should have internally consistent market moves, "
                "forced anchor evidence, confidence, and no future leakage. "
                "This follows stress-testing practice: coherent narrative, "
                "material risk channels, severity/plausibility, and explicit "
                "scenario-variable consistency."
            ),
        },
    }


def build_support_episode_cards(
    history_raw: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    train_indices: list[int] | np.ndarray | None = None,
    source_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build episode cards for support-bank rows from raw 30-day histories."""

    history = np.asarray(history_raw, dtype=np.float32)
    if history.ndim != 3:
        raise ValueError("history_raw must have shape [N,T,C]")
    if len(metadata) < history.shape[0]:
        raise ValueError("metadata must contain at least one row per history window")
    indices = (
        [int(idx) for idx in train_indices]
        if train_indices is not None
        else list(range(int(history.shape[0])))
    )
    scales = _scales_from_history(history)
    cards: list[dict[str, Any]] = []
    archetype_counts: dict[str, int] = {}
    for idx in indices:
        caption = _caption_from_support_row(
            history_raw=history,
            scales=scales,
            row_index=idx,
            metadata=metadata[idx],
            source_path=source_path,
        )
        card = build_episode_card(caption, source_path=source_path)
        card["support_metadata"] = {
            "calendar_start_date": str(metadata[idx].get("calendar_start_date", "")),
            "calendar_end_date": str(metadata[idx].get("calendar_end_date", "")),
            "window_index": int(metadata[idx].get("window_index", idx)),
            "raw_history_card": True,
            "support_move_rows": caption["support_move_rows"],
        }
        cards.append(card)
        archetype = str(card.get("archetype", ""))
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
    report = summarize_corpus_quality(cards, source_path=source_path)
    report.update(
        {
            "source_inventory": "support_bank_raw_history",
            "selected_index_count": len(indices),
            "archetype_counts": dict(sorted(archetype_counts.items())),
            "support_card_contract_quality": _support_card_contract_quality(cards),
        }
    )
    return cards, report


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--support-arrays", type=Path, default=DEFAULT_SUPPORT_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-decoder-test", action="store_true")
    args = parser.parse_args(argv)

    report = _load_json(args.support_report)
    metadata = report.get("window_metadata", [])
    if not isinstance(metadata, list):
        raise ValueError("support report missing window_metadata")
    with np.load(args.support_arrays) as arrays:
        history_raw = arrays["history_raw"].copy()
        train_indices = (
            None if args.include_decoder_test else arrays["train_indices"].copy()
        )
    cards, card_report = build_support_episode_cards(
        history_raw,
        metadata,
        train_indices=train_indices,
        source_path=str(args.support_arrays),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = args.output_dir / "episode_narrative_support_cards.jsonl"
    report_path = args.output_dir / "support_card_quality_report.json"
    write_jsonl(cards_path, cards)
    card_report["artifact_paths"] = {
        "cards_jsonl": str(cards_path),
        "report": str(report_path),
    }
    report_path.write_text(
        json.dumps(card_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "cards_jsonl": str(cards_path),
                "report": str(report_path),
                "card_count": len(cards),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
