#!/usr/bin/env python
"""Narrative-grounded text-controlled scenario generation pilot.

This script is intentionally a pilot, but it performs the full workflow:

1. rebuild real joint39 SNI history windows and `_encode_prefix` memory targets;
2. create hallucination-aware narrative descriptions and hard negatives;
3. embed narrative text;
4. train a small text-to-memory adapter with alignment + contrastive losses;
5. retrieve nearby historical conditioning states from the learned memory vector;
6. run the frozen generator normally with those retrieved history prefixes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    cosine_similarity,
    embed_texts_with_openai,
    normalize_rows,
)
from experiments.backfill.block_ar.nl_scenario_descriptions import (  # noqa: E402
    ScenarioDescriptionBundle,
    describe_window_with_openai,
    validate_description_bundle,
)
from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)


DEFAULT_CHECKPOINT = (
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
    "best_model.pt"
)
KEY_FACTOR_NAMES = {
    "SPX": "factor:spx",
    "US2Y": "factor:us2y",
    "US10Y": "factor:us10y",
    "BBB_OAS": "factor:bbb_oas",
    "AAA_OAS": "factor:aaa_oas",
    "USDJPY": "factor:usdjpy",
    "DXY": "factor:dxy",
    "GOLD": "factor:gold",
    "CRUDE_OIL": "factor:crude_oil",
    "VIX": "factor:vix",
}


def _direction_and_magnitude(
    market: str,
    delta: float,
    *,
    small: float = 0.05,
    medium: float = 0.20,
) -> tuple[str, str]:
    abs_delta = abs(float(delta))
    if abs_delta < small:
        direction = "flat"
    elif market.endswith("_OAS"):
        direction = "wider" if delta > 0 else "tighter"
    else:
        direction = "up" if delta > 0 else "down"
    if abs_delta < small:
        magnitude = "flat"
    elif abs_delta < medium:
        magnitude = "small"
    elif abs_delta < medium * 2.5:
        magnitude = "medium"
    else:
        magnitude = "large"
    return direction, magnitude


def _fact(market: str, delta: float, evidence_name: str) -> dict[str, Any]:
    direction, magnitude = _direction_and_magnitude(market, float(delta))
    return {
        "market": market,
        "direction": direction,
        "magnitude": magnitude,
        "evidence": f"{evidence_name}={float(delta):.6g}",
        "observed": True,
    }


def key_market_summary(history_raw: np.ndarray, spec_names: list[str]) -> list[dict[str, Any]]:
    """Summarize observed 30-day history moves into key market facts."""

    history = np.asarray(history_raw, dtype=np.float32)
    if history.ndim != 2:
        raise ValueError("history_raw must have shape [history_len, n_cells]")
    index = {name: idx for idx, name in enumerate(spec_names)}
    rows: list[dict[str, Any]] = []
    iv_delta = float(np.nanmean(history[-1, :25] - history[0, :25]))
    rows.append(_fact("IV_SURFACE", iv_delta, "mean_iv_encoded_30d_change"))
    if history.shape[1] >= 25:
        front = float(np.nanmean(history[-1, :5] - history[0, :5]))
        back = float(np.nanmean(history[-1, 20:25] - history[0, 20:25]))
        rows.append(_fact("IV_SKEW", front - back, "front_minus_back_iv_change"))
    for market, spec_name in KEY_FACTOR_NAMES.items():
        if spec_name not in index:
            continue
        idx = index[spec_name]
        delta = float(history[-1, idx] - history[0, idx])
        rows.append(_fact(market, delta, f"{spec_name}_30d_change"))
    return rows


def _fact_lookup(facts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["market"]): item for item in facts}


def _is(facts: dict[str, dict[str, Any]], market: str, direction: str) -> bool:
    return facts.get(market, {}).get("direction") == direction


def _market_token_text(facts: list[dict[str, Any]]) -> str:
    pieces = [
        f"{item['market']}: {str(item['direction']).upper()} {str(item['magnitude']).upper()}"
        for item in facts
        if item.get("direction") != "flat"
    ]
    return "; ".join(pieces) if pieces else "MARKET_STATE: BROADLY FLAT"


def _classify_shock(facts: list[dict[str, Any]]) -> tuple[str, float, list[str]]:
    by_market = _fact_lookup(facts)
    if _is(by_market, "SPX", "down") and _is(by_market, "VIX", "up") and _is(by_market, "BBB_OAS", "wider"):
        if _is(by_market, "US2Y", "down") or _is(by_market, "US10Y", "down"):
            return "growth_liquidity_panic", 0.82, [
                "equity selloff",
                "volatility spike",
                "credit widening",
                "rates lower",
            ]
        if _is(by_market, "US2Y", "up") or _is(by_market, "US10Y", "up"):
            return "inflationary_risk_off_rates_shock", 0.72, [
                "equity selloff",
                "volatility spike",
                "credit widening",
                "rates higher",
            ]
        return "risk_off_stress", 0.70, [
            "equity selloff",
            "volatility spike",
            "credit widening",
        ]
    if _is(by_market, "SPX", "up") and _is(by_market, "VIX", "down"):
        return "risk_on_recovery", 0.70, ["equity rally", "volatility compression"]
    if _is(by_market, "CRUDE_OIL", "up") and (_is(by_market, "US2Y", "up") or _is(by_market, "US10Y", "up")):
        return "commodity_inflation_shock", 0.65, ["commodity pressure", "rates higher"]
    return "mixed_cross_asset_repricing", 0.45, ["mixed market signals"]


def build_scenario_narrative_bundle(
    window_id: str,
    history_raw: np.ndarray,
    spec_names: list[str],
) -> dict[str, Any]:
    """Create hallucination-aware narrative labels from observed market facts."""

    facts = key_market_summary(history_raw, spec_names)
    shock_type, confidence, drivers = _classify_shock(facts)
    token_text = _market_token_text(facts)
    unsupported = [
        "Named historical event is an analogy inferred from market pattern, not observed in the panel."
    ]
    if shock_type == "growth_liquidity_panic":
        primary = (
            "COVID-style liquidity and growth shock analogy: market panic hits "
            "high-duration and high-P/E equity risk, volatility jumps, credit "
            "spreads widen, and safe-haven rate pressure dominates."
        )
        alternative = (
            "Broad risk-manager stress narrative: equities de-rate sharply while "
            "the options surface reprices higher, credit liquidity weakens, and "
            "Treasury yields fall as investors seek safety."
        )
    elif shock_type == "inflationary_risk_off_rates_shock":
        primary = (
            "Inflationary risk-off narrative: equities fall as discount rates rise, "
            "volatility climbs, and credit spreads widen because policy relief is "
            "less available."
        )
        alternative = (
            "Rates-led valuation shock narrative: expensive growth assets are "
            "pressured by higher yields, with volatility and credit risk rising."
        )
    elif shock_type == "risk_on_recovery":
        primary = (
            "Risk-on recovery narrative: equities rally, volatility compresses, "
            "and investors move back toward growth exposure."
        )
        alternative = (
            "Relief-rally narrative: the market prices lower near-term stress and "
            "accepts more equity risk."
        )
        unsupported = []
    else:
        primary = (
            "Mixed cross-asset repricing narrative: the market state does not map "
            "cleanly to a single macro shock, so scenario conditioning should stay "
            "close to observed market facts."
        )
        alternative = (
            "Ambiguous risk narrative: some assets imply stress while others do not, "
            "so the system should report lower narrative confidence."
        )
    narratives = [
        {
            "id": "primary_macro_story",
            "text": primary,
            "shock_type": shock_type,
            "confidence": confidence,
            "grounding_status": "analogy_not_observed_fact" if unsupported else "market_fact_supported",
            "observed_fact_tokens": token_text,
            "unsupported_claims": unsupported,
            "alternative_interpretations": [alternative],
        },
        {
            "id": "risk_report_style",
            "text": alternative,
            "shock_type": shock_type,
            "confidence": max(0.35, confidence - 0.07),
            "grounding_status": "market_fact_supported",
            "observed_fact_tokens": token_text,
            "unsupported_claims": [],
            "alternative_interpretations": [primary],
        },
        {
            "id": "market_implication_style",
            "text": (
                "Market implication summary for scenario generation: "
                f"{token_text}."
            ),
            "shock_type": "explicit_market_implications",
            "confidence": 0.95,
            "grounding_status": "observed_market_facts_only",
            "observed_fact_tokens": token_text,
            "unsupported_claims": [],
            "alternative_interpretations": [],
        },
    ]
    contrastive = [
        {
            "id": "opposite_risk_sentiment",
            "kind": "opposite",
            "text": (
                "Opposite narrative: risk appetite improves, equities rally, "
                "volatility falls, credit spreads tighten, and the market prices "
                "less need for defensive positioning."
            ),
            "purpose": "hard negative, not historical target",
        },
        {
            "id": "rates_direction_flip",
            "kind": "partial",
            "text": (
                "Partial contrast: equities remain under pressure and volatility "
                "is high, but the dominant macro story is an inflation shock with "
                "rates rising rather than a flight-to-quality rate rally."
            ),
            "purpose": "hard negative, not historical target",
        },
        {
            "id": "magnitude_flip",
            "kind": "magnitude",
            "text": (
                "Magnitude contrast: the same markets move in the same broad "
                "directions, but only mildly, without panic-level volatility or "
                "material credit stress."
            ),
            "purpose": "hard negative, not historical target",
        },
    ]
    return {
        "window_id": str(window_id),
        "panel_version": "joint39",
        "grounding_policy": "market_facts_first_no_external_news",
        "observed_market_facts": facts,
        "narrative_drivers": drivers,
        "narratives": narratives,
        "contrastive_narratives": contrastive,
        "hallucination_audit": {
            "observed_facts_source": "30-day historical joint39 panel window",
            "external_news_used": False,
            "causality_claim_policy": "macro stories are marked inferred unless directly observed",
            "unsupported_claims": sorted({claim for item in narratives for claim in item["unsupported_claims"]}),
        },
    }


def build_openai_window_summary(
    window_id: str,
    history_raw: np.ndarray,
    spec_names: list[str],
    window_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact market-fact payload sent to the description model."""

    facts = key_market_summary(history_raw, spec_names)
    return {
        "window_id": str(window_id),
        "panel_version": "joint39",
        **(window_metadata or {}),
        "grounding_policy": "market_facts_first_no_external_news",
        "observed_market_facts": facts,
        "canonical_market_tokens": _market_token_text(facts),
        "summary_horizon": "30 historical trading days ending at the conditioning date",
        "external_news_available": False,
        "instruction": (
            "Create direct market-state and risk-manager narrative labels only "
            "from observed market facts and calendar dates. Put explicit "
            "directions in market_implications. Put stories like tariff shock, "
            "war-style oil shock, 2008-style credit stress, COVID-style panic, "
            "or flash-crash analogy in narrative_catalysts with grounding_status; "
            "do not assert them as confirmed historical causes without supplied "
            "citations."
        ),
    }


def description_bundle_to_narrative_bundle(
    bundle: ScenarioDescriptionBundle,
) -> dict[str, Any]:
    """Convert OpenAI structured description output into adapter training rows."""

    payload = bundle.model_dump()
    issues = validate_description_bundle(bundle)
    fact_tokens = str(payload["canonical_machine_text"])
    implication_tokens = "; ".join(
        f"{item['market']}: {str(item['direction']).upper()} {str(item['magnitude']).upper()}"
        for item in payload.get("market_implications", [])
    )
    conditioning_tokens = implication_tokens or fact_tokens
    catalysts = payload.get("narrative_catalysts", [])
    catalyst_text = "; ".join(
        f"{item['label']} [{item['grounding_status']}]: {item['description']}"
        for item in catalysts
    )
    narratives: list[dict[str, Any]] = [
        {
            "id": "revised_market_description",
            "text": str(payload["revised_description"]),
            "shock_type": "llm_grounded_market_description",
            "confidence": 0.85,
            "grounding_status": "market_fact_supported",
            "observed_fact_tokens": conditioning_tokens,
            "narrative_catalysts": catalysts,
            "unsupported_claims": [
                issue.message for issue in issues if issue.severity == "error"
            ],
            "alternative_interpretations": [],
        }
    ]
    for item in payload.get("descriptions", []):
        narratives.append(
            {
                "id": f"description_{item.get('style', 'free_form')}",
                "text": str(item["text"]),
                "shock_type": "llm_grounded_market_description",
                "confidence": 0.80,
                "grounding_status": "market_fact_supported",
                "observed_fact_tokens": conditioning_tokens,
                "narrative_catalysts": catalysts,
                "unsupported_claims": [],
                "alternative_interpretations": [str(payload["revised_description"])],
            }
        )

    contrastive_payload = payload.get("contrastive", {})
    contrastive: list[dict[str, Any]] = []
    if contrastive_payload.get("opposite"):
        contrastive.append(
            {
                "id": "opposite",
                "kind": "opposite",
                "text": str(contrastive_payload["opposite"]),
                "purpose": "hard negative, not historical target",
            }
        )
    for kind in ("partial", "magnitude"):
        for idx, text in enumerate(contrastive_payload.get(kind, [])):
            contrastive.append(
                {
                    "id": f"{kind}_{idx}",
                    "kind": kind,
                    "text": str(text),
                    "purpose": "hard negative, not historical target",
                }
            )

    return {
        "window_id": str(payload["window_id"]),
        "panel_version": str(payload["panel_version"]),
        "calendar": {
            "calendar_start_date": payload.get("calendar_start_date", ""),
            "calendar_end_date": payload.get("calendar_end_date", ""),
            "forecast_start_date": payload.get("forecast_start_date", ""),
            "forecast_end_date": payload.get("forecast_end_date", ""),
        },
        "grounding_policy": "openai_market_facts_first_no_external_news",
        "market_implications": payload.get("market_implications", []),
        "narrative_catalysts": catalysts,
        "catalyst_text": catalyst_text,
        "observed_market_facts": [
            {
                "market": item["market"],
                "direction": item["direction"],
                "magnitude": item["magnitude"],
                "confidence": item["confidence"],
                "evidence": item.get("evidence", []),
                "inferred": item.get("inferred", False),
            }
            for item in payload.get("structured_audit", [])
        ],
        "narrative_drivers": [str(item["market"]) for item in payload.get("structured_audit", [])],
        "narratives": narratives,
        "contrastive_narratives": contrastive,
        "hallucination_audit": {
            "observed_facts_source": "OpenAI label generated from supplied joint39 market summary",
            "external_news_used": False,
            "causality_claim_policy": "reject unsupported news or direct causal claims",
            "validation_issues": [issue.model_dump() for issue in issues],
            "critique": payload.get("critique", []),
        },
        "source_description_bundle": payload,
    }


def _read_label_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            window_id = str(row.get("window_id", ""))
            if not window_id:
                raise ValueError(f"{path}:{line_no}: missing window_id")
            rows[window_id] = refresh_cached_label_bundle(row)
    return rows


def refresh_cached_label_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Recompute derived narrative fields/validation for cached OpenAI labels."""

    source = bundle.get("source_description_bundle")
    if not isinstance(source, dict):
        return bundle
    try:
        description = ScenarioDescriptionBundle.model_validate(source)
    except Exception:
        return bundle
    return description_bundle_to_narrative_bundle(description)


def _append_label_cache(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(bundle, sort_keys=True, separators=(",", ":")) + "\n")


def build_window_metadata(
    block: Any,
    *,
    history_len: int,
    future_len: int,
) -> list[dict[str, Any]]:
    """Attach calendar dates to each rollout window from the aligned panel."""

    _panel, _columns, dates = load_aligned_iv_factor_panel()
    rows: list[dict[str, Any]] = []
    for idx in np.asarray(block.indices, dtype=np.int64):
        start = int(idx)
        hist_end = start + int(history_len) - 1
        fut_start = start + int(history_len)
        fut_end = start + int(history_len) + int(future_len) - 1
        rows.append(
            {
                "source_index": start,
                "calendar_start_date": str(dates[start].date()),
                "calendar_end_date": str(dates[hist_end].date()),
                "forecast_start_date": str(dates[fut_start].date()),
                "forecast_end_date": str(dates[fut_end].date()),
            }
        )
    return rows


def _load_selection_manifest_rows(path: str | Path, split: str) -> list[dict[str, Any]]:
    """Load selected manifest rows for one split or train/validation/test."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("selection manifest must contain a splits object")
    split_name = str(split)
    split_order = ["train", "validation", "test"] if split_name == "all" else [split_name]
    rows: list[dict[str, Any]] = []
    for name in split_order:
        raw_rows = splits.get(name, [])
        if not isinstance(raw_rows, list):
            raise ValueError(f"selection manifest split {name!r} must be a list")
        for row in raw_rows:
            if not isinstance(row, dict):
                raise ValueError(f"selection manifest split {name!r} contains a non-object row")
            if "window_index" not in row:
                raise ValueError(f"selection manifest split {name!r} row is missing window_index")
            rows.append({**row, "manifest_split": name})
    if not rows:
        raise ValueError(f"selection manifest has no rows for split={split_name!r}")
    return rows


def _metadata_with_manifest_row(
    base_metadata: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    """Merge data-block metadata with manifest identity fields."""

    window_index = int(row["window_index"])
    merged = dict(base_metadata)
    merged.update(
        {
            "window_index": window_index,
            "window_id": str(row.get("window_id", f"joint39_val_{window_index:04d}")),
            "source_index": int(row.get("source_index", base_metadata.get("source_index", window_index))),
            "selection_reasons": list(row.get("selection_reasons", [])),
            "manifest_split": str(row.get("manifest_split", "")),
        }
    )
    for key in (
        "calendar_start_date",
        "calendar_end_date",
        "forecast_start_date",
        "forecast_end_date",
    ):
        if key in row:
            merged[key] = row[key]
    return merged


def apply_manifest_window_selection(
    *,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    window_metadata: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Slice pipeline arrays by original manifest window indices."""

    indices = np.asarray([int(row["window_index"]) for row in manifest_rows], dtype=np.int64)
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("selection manifest contains duplicate window_index rows")
    n_available = int(history_raw.shape[0])
    bad = [int(idx) for idx in indices if idx < 0 or idx >= n_available]
    if bad:
        raise ValueError(f"selection manifest window_index out of range: {bad[:5]}")
    selected_metadata = [
        _metadata_with_manifest_row(
            window_metadata[int(idx)] if int(idx) < len(window_metadata) else {},
            row,
        )
        for idx, row in zip(indices, manifest_rows)
    ]
    return {
        "history_level": history_level[indices],
        "history_norm": history_norm[indices],
        "center": center[indices],
        "scale": scale[indices],
        "drift_feature": drift_feature[indices],
        "history_raw": history_raw[indices],
        "window_metadata": selected_metadata,
        "window_indices": indices.astype(np.int64),
        "source_indices": np.asarray(
            [int(row["source_index"]) for row in selected_metadata],
            dtype=np.int64,
        ),
    }


def _attach_window_metadata_to_bundle(
    bundle: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Attach original window identity/provenance to a narrative bundle."""

    enriched = dict(bundle)
    meta = dict(metadata)
    if "window_id" in meta:
        enriched["window_id"] = str(meta["window_id"])
    if meta:
        enriched["window_metadata"] = meta
    for key in ("window_index", "source_index", "selection_reasons", "manifest_split"):
        if key in meta:
            enriched[key] = meta[key]
    return enriched


def bundle_validation_errors(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    issues = bundle.get("hallucination_audit", {}).get("validation_issues", [])
    return [
        issue
        for issue in issues
        if str(issue.get("severity", "warning")).lower() == "error"
    ]


def build_narrative_bundles(
    history_raw: np.ndarray,
    spec_names: list[str],
    window_metadata: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Build or load narrative bundles for the selected training windows."""

    label_backend = str(args.label_backend)
    cache_path = Path(args.label_cache or Path(args.output_dir) / "narrative_label_cache.jsonl")
    cache = _read_label_cache(cache_path) if label_backend == "openai" else {}
    bundles: list[dict[str, Any] | None] = [None] * int(history_raw.shape[0])
    missing: list[tuple[int, str]] = []
    for idx in range(history_raw.shape[0]):
        metadata = window_metadata[idx] if idx < len(window_metadata) else {}
        window_id = str(metadata.get("window_id", f"joint39_val_{idx:04d}"))
        if label_backend == "rule":
            bundles[idx] = _attach_window_metadata_to_bundle(
                build_scenario_narrative_bundle(window_id, history_raw[idx], spec_names),
                metadata,
            )
            continue
        if window_id in cache:
            bundles[idx] = _attach_window_metadata_to_bundle(cache[window_id], metadata)
            continue
        missing.append((idx, window_id))

    def _label_one(item: tuple[int, str]) -> tuple[int, str, dict[str, Any]]:
        idx, window_id = item
        summary = build_openai_window_summary(
            window_id,
            history_raw[idx],
            spec_names,
            window_metadata[idx] if idx < len(window_metadata) else None,
        )
        description = describe_window_with_openai(
            summary,
            model=str(args.label_model),
            dotenv_path=args.dotenv,
            max_output_tokens=int(args.label_max_output_tokens),
        )
        metadata = window_metadata[idx] if idx < len(window_metadata) else {}
        return idx, window_id, _attach_window_metadata_to_bundle(
            description_bundle_to_narrative_bundle(description),
            metadata,
        )

    if missing:
        workers = max(1, int(args.label_concurrency))
        if workers == 1:
            for item in missing:
                idx, window_id, bundle = _label_one(item)
                bundles[idx] = bundle
                cache[window_id] = bundle
                _append_label_cache(cache_path, bundle)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_label_one, item) for item in missing]
                for future in concurrent.futures.as_completed(futures):
                    idx, window_id, bundle = future.result()
                    bundles[idx] = bundle
                    cache[window_id] = bundle
                    _append_label_cache(cache_path, bundle)
    return [bundle for bundle in bundles if bundle is not None]


def build_narrative_training_examples(
    bundle: dict[str, Any], *, target_index: int = 0
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    narratives = list(bundle.get("narratives", []))
    if not narratives:
        raise ValueError("bundle has no narratives")
    for idx, narrative in enumerate(narratives):
        role = "anchor" if idx == 0 else "positive"
        text = (
            f"NARRATIVE: {narrative['text']}\n"
            f"GROUNDING_STATUS: {narrative['grounding_status']}\n"
            f"MARKET_IMPLICATIONS: {narrative['observed_fact_tokens']}\n"
            f"NARRATIVE_CATALYSTS: {bundle.get('catalyst_text', '')}"
        )
        examples.append(
            {
                "window_id": bundle["window_id"],
                "role": role,
                "kind": narrative["id"],
                "text": text,
                "target_index": int(target_index),
            }
        )
    for negative in bundle.get("contrastive_narratives", []):
        examples.append(
            {
                "window_id": bundle["window_id"],
                "role": "negative",
                "kind": negative["kind"],
                "text": str(negative["text"]),
                "target_index": None,
            }
        )
    return examples


def hash_text_embeddings(texts: list[str], *, dim: int = 512) -> np.ndarray:
    """Deterministic local embedding fallback for tests/offline dry-runs."""

    arr = np.zeros((len(texts), int(dim)), dtype=np.float32)
    for row, text in enumerate(texts):
        for raw_token in text.lower().replace("\n", " ").split():
            token = raw_token.strip(".,;:()[]{}")
            if not token:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little", signed=False)
            col = value % int(dim)
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            arr[row, col] += sign
    return normalize_rows(arr)


class NarrativeAdapter(nn.Module):
    """Train text/narrative embeddings into the generator memory space."""

    def __init__(self, embedding_dim: int, condition_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or min(512, max(condition_dim * 2, embedding_dim // 2)))
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(condition_dim)),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def train_narrative_adapter(
    text_embeddings: np.ndarray,
    target_memory: np.ndarray,
    target_indices: np.ndarray,
    roles: list[str],
    groups: list[str],
    *,
    condition_dim: int,
    hidden_dim: int | None = None,
    steps: int = 400,
    lr: float = 1e-3,
    contrastive_weight: float = 0.25,
    contrastive_margin: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    embeddings = normalize_rows(text_embeddings)
    targets = np.asarray(target_memory, dtype=np.float32)
    target_idx = np.asarray(target_indices, dtype=np.int64)
    if embeddings.shape[0] != target_idx.shape[0]:
        raise ValueError("target_indices length must match text embeddings")
    if len(roles) != embeddings.shape[0] or len(groups) != embeddings.shape[0]:
        raise ValueError("roles/groups length must match text embeddings")
    torch.manual_seed(int(seed))
    x = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(targets).float()
    idx_t = torch.from_numpy(target_idx)
    adapter = NarrativeAdapter(
        embeddings.shape[1],
        int(condition_dim),
        hidden_dim=hidden_dim,
    )
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
    valid_mask = idx_t >= 0
    losses: list[float] = []
    group_names = sorted(set(groups))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pred = adapter(x)
        align_pred = pred[valid_mask]
        align_target = y[idx_t[valid_mask]]
        mse = F.mse_loss(align_pred, align_target)
        cosine_loss = 1.0 - F.cosine_similarity(align_pred, align_target, dim=-1).mean()
        contrast = pred.new_zeros(())
        pred_norm = F.normalize(pred, dim=-1)
        terms: list[torch.Tensor] = []
        for group in group_names:
            indices = [i for i, value in enumerate(groups) if value == group]
            anchors = [i for i in indices if roles[i] == "anchor"]
            positives = [i for i in indices if roles[i] == "positive"]
            negatives = [i for i in indices if roles[i] == "negative"]
            if not anchors or not positives or not negatives:
                continue
            anchor = pred_norm[anchors[0]]
            pos = pred_norm[positives] @ anchor
            neg = pred_norm[negatives] @ anchor
            terms.append(F.relu(float(contrastive_margin) - pos[:, None] + neg[None, :]).mean())
        if terms:
            contrast = torch.stack(terms).mean()
        loss = mse + 0.2 * cosine_loss + float(contrastive_weight) * contrast
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    adapter.eval()
    with torch.no_grad():
        condition_vectors = adapter(x).cpu().numpy().astype(np.float32)
    return {
        "adapter": adapter,
        "condition_vectors": condition_vectors,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
    }


@torch.no_grad()
def compute_memory_targets(
    model: Any,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, history_level.shape[0], int(batch_size)):
        end = min(start + int(batch_size), history_level.shape[0])
        h_level = torch.from_numpy(history_level[start:end]).to(device)
        h_norm = torch.from_numpy(history_norm[start:end]).to(device)
        ctr = torch.from_numpy(center[start:end]).to(device)
        scl = torch.from_numpy(scale[start:end]).to(device)
        drift = torch.from_numpy(drift_feature[start:end]).to(device)
        level_scores = model.level_values_to_scores(h_level)
        flow = model._to_flow_coordinate(h_norm)
        memory = model._encode_prefix(level_scores, flow, ctr, scl, drift)[:, -1]
        chunks.append(memory.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def nearest_memory_indices(query: np.ndarray, targets: np.ndarray, *, top_k: int = 3) -> list[dict[str, Any]]:
    q = np.asarray(query, dtype=np.float32)
    target_norm = normalize_rows(targets)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-8)
    sims = target_norm @ q_norm
    order = np.argsort(-sims)[: int(top_k)]
    return [{"index": int(idx), "cosine": float(sims[idx])} for idx in order]


def retrieval_weights(
    analogue_rows: list[dict[str, Any]], *, temperature: float = 0.05
) -> list[dict[str, Any]]:
    """Attach softmax weights to retrieved analogue rows."""

    if not analogue_rows:
        return []
    temp = max(float(temperature), 1e-6)
    scores = np.asarray([float(row["cosine"]) for row in analogue_rows], dtype=np.float64)
    logits = (scores - float(scores.max())) / temp
    weights = np.exp(logits)
    weights = weights / max(float(weights.sum()), 1e-12)
    rows: list[dict[str, Any]] = []
    for row, weight in zip(analogue_rows, weights):
        updated = dict(row)
        updated["weight"] = float(weight)
        rows.append(updated)
    return rows


def retrieval_diagnostics(
    analogue_rows: list[dict[str, Any]], *, ood_threshold: float = 0.75
) -> dict[str, Any]:
    """Compute simple OOD diagnostics for a query-to-history retrieval."""

    if not analogue_rows:
        return {
            "ood_warning": True,
            "reason": "no historical analogues returned",
            "top_cosine": None,
            "top_gap": None,
        }
    top = float(analogue_rows[0]["cosine"])
    second = float(analogue_rows[1]["cosine"]) if len(analogue_rows) > 1 else top
    return {
        "ood_warning": bool(top < float(ood_threshold)),
        "reason": (
            "nearest historical analogue below similarity threshold"
            if top < float(ood_threshold)
            else "nearest historical analogue inside similarity threshold"
        ),
        "top_cosine": top,
        "top_gap": float(top - second),
        "ood_threshold": float(ood_threshold),
    }


@torch.no_grad()
def sample_normal_generator_for_retrieved_analogues(
    model: Any,
    analogue_rows: list[dict[str, Any]],
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    *,
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: str | torch.device,
) -> dict[str, Any]:
    """Run the frozen generator through its normal sample_batched path."""

    if not analogue_rows:
        raise ValueError("need at least one retrieved analogue")
    indices = [int(row["index"]) for row in analogue_rows]
    dev = torch.device(device)
    sampled_increment = model.sample_batched(
        torch.from_numpy(history_level[indices]).to(dev),
        torch.from_numpy(history_norm[indices]).to(dev),
        torch.from_numpy(center[indices]).to(dev),
        torch.from_numpy(scale[indices]).to(dev),
        drift_feature=torch.from_numpy(drift_feature[indices]).to(dev),
        n_samples=int(n_samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        temperature=float(temperature),
    )
    return {
        "indices": indices,
        "increments": sampled_increment.detach().cpu().numpy().astype(np.float32),
    }


@torch.no_grad()
def sample_with_memory_condition(
    model: Any,
    condition_memory: np.ndarray,
    history_level_values: np.ndarray,
    history_normalized_innovation: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    *,
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: torch.device,
) -> np.ndarray:
    """Directly feed a learned memory vector into the frozen SNI velocity field."""

    h_level = torch.from_numpy(history_level_values[None]).to(device)
    h_norm = torch.from_numpy(history_normalized_innovation[None]).to(device)
    ctr = torch.from_numpy(center[None]).to(device)
    scl = torch.from_numpy(scale[None]).to(device)
    drift = torch.from_numpy(drift_feature[None]).to(device)
    memory_base = torch.from_numpy(condition_memory[None].astype(np.float32)).to(device)
    level_scores = model.level_values_to_scores(h_level)
    history_flow = model._to_flow_coordinate(h_norm)
    temp = float(temperature)
    dt = 1.0 / float(model.cfg.flow_steps)
    outs: list[torch.Tensor] = []
    for start in range(0, int(n_samples), int(chunk_size)):
        k = min(int(chunk_size), int(n_samples) - start)
        prefix_level_values = h_level.expand(k, model.cfg.history_len, model.cfg.n_cells).clone()
        prefix_level_scores = level_scores.expand(k, model.cfg.history_len, model.cfg.n_cells).clone()
        prefix_norm = history_flow.expand(k, model.cfg.history_len, model.cfg.n_cells).clone()
        center_rep = ctr.expand(k, model.cfg.n_cells)
        scale_rep = scl.expand(k, model.cfg.n_cells)
        drift_rep = drift.expand(k, model.cfg.n_cells)
        memory_state = memory_base.expand(k, model.cfg.memory_dim)
        base_noise = temp * model._base_noise_like(
            torch.empty(k, int(n_steps), model.cfg.n_cells, device=device, dtype=h_level.dtype)
        )
        frames: list[torch.Tensor] = []
        for step in range(int(n_steps)):
            current_level_score = prefix_level_scores[:, -1]
            x = base_noise[:, step]
            base_noise_scale = model._conditional_base_noise_scale(memory_state)
            if base_noise_scale is not None:
                x = x * base_noise_scale
            for flow_step in range(int(model.cfg.flow_steps)):
                t = torch.full(
                    (k,),
                    (flow_step + 0.5) * dt,
                    device=device,
                    dtype=h_level.dtype,
                )
                x = x + dt * model.velocity(x, current_level_score, memory_state, t)
            next_flow = x
            next_norm = model._from_flow_coordinate(next_flow)
            next_increment = next_norm * scale_rep + center_rep
            next_level_value = prefix_level_values[:, -1] + next_increment
            next_level_score = model.level_values_to_scores(next_level_value)
            frames.append(next_increment)
            prefix_level_values = torch.cat([prefix_level_values, next_level_value[:, None]], dim=1)
            prefix_level_scores = torch.cat([prefix_level_scores, next_level_score[:, None]], dim=1)
            prefix_norm = torch.cat([prefix_norm, next_flow[:, None]], dim=1)
        outs.append(torch.stack(frames, dim=1))
    return torch.cat(outs, dim=0).cpu().numpy().astype(np.float32)[None]


def summarize_generated_states(
    generated_states: np.ndarray,
    current_state: np.ndarray,
    spec_names: list[str],
) -> dict[str, Any]:
    states = np.asarray(generated_states, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    terminal_delta = states[0, :, -1, :] - current[None, :]
    rows = []
    index = {name: idx for idx, name in enumerate(spec_names)}
    iv_delta = np.nanmean(terminal_delta[:, :25], axis=1)
    rows.append(
        {
            "market": "IV_SURFACE",
            "mean_terminal_delta": float(np.nanmean(iv_delta)),
            "p10": float(np.nanquantile(iv_delta, 0.1)),
            "p90": float(np.nanquantile(iv_delta, 0.9)),
        }
    )
    for market, spec_name in KEY_FACTOR_NAMES.items():
        if spec_name not in index:
            continue
        values = terminal_delta[:, index[spec_name]]
        rows.append(
            {
                "market": market,
                "mean_terminal_delta": float(np.nanmean(values)),
                "p10": float(np.nanquantile(values, 0.1)),
                "p90": float(np.nanquantile(values, 0.9)),
            }
        )
    return {
        "generated_state_shape": list(states.shape),
        "finite_rate": float(np.isfinite(states).mean()),
        "terminal_delta_summary": rows,
    }


def summarize_retrieval_generated_states(
    generated_states: np.ndarray,
    current_states: np.ndarray,
    spec_names: list[str],
) -> dict[str, Any]:
    """Summarize normal generator samples for one or more retrieved analogues."""

    states = np.asarray(generated_states, dtype=np.float32)
    current = np.asarray(current_states, dtype=np.float32)
    if states.ndim != 4:
        raise ValueError("generated_states must have shape [K,S,T,C]")
    if current.shape != (states.shape[0], states.shape[-1]):
        raise ValueError(
            "current_states must have shape [K,C] matching generated_states"
        )
    terminal_delta = states[:, :, -1, :] - current[:, None, :]
    flat_delta = terminal_delta.reshape(-1, terminal_delta.shape[-1])
    rows = []
    index = {name: idx for idx, name in enumerate(spec_names)}
    iv_delta = np.nanmean(flat_delta[:, :25], axis=1)
    rows.append(
        {
            "market": "IV_SURFACE",
            "mean_terminal_delta": float(np.nanmean(iv_delta)),
            "p10": float(np.nanquantile(iv_delta, 0.1)),
            "p90": float(np.nanquantile(iv_delta, 0.9)),
        }
    )
    for market, spec_name in KEY_FACTOR_NAMES.items():
        if spec_name not in index:
            continue
        values = flat_delta[:, index[spec_name]]
        rows.append(
            {
                "market": market,
                "mean_terminal_delta": float(np.nanmean(values)),
                "p10": float(np.nanquantile(values, 0.1)),
                "p90": float(np.nanquantile(values, 0.9)),
            }
        )
    return {
        "generated_state_shape": list(states.shape),
        "finite_rate": float(np.isfinite(states).mean()),
        "terminal_delta_summary": rows,
    }


def _load_joint39_block(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[Any, ...]:
    from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import build_val_block

    return build_val_block(args, payload)


def _enforce_production_label_policy(args: argparse.Namespace) -> None:
    """Require model-written labels unless the run is explicitly local-only."""

    label_backend = str(getattr(args, "label_backend", "openai")).lower()
    local_test = bool(getattr(args, "local_test", False))
    if local_test or label_backend == "openai":
        return
    raise ValueError(
        "Production narrative runs require --label-backend openai. "
        "Use --local-test only for rule-label smoke tests, unit tests, or CI."
    )


def _reconstruct_states(history_last: np.ndarray, increments: np.ndarray, specs: list[Any]) -> np.ndarray:
    from experiments.backfill.block_ar.increment_coordinate_628_utils import reconstruct_state_from_increments

    return reconstruct_state_from_increments(history_last, increments, specs)


def _spec_names(specs: list[Any]) -> list[str]:
    return [str(getattr(spec, "name", spec)) for spec in specs]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    _enforce_production_label_policy(args)

    from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import load_model

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        block,
    ) = _load_joint39_block(args, payload)
    spec_names = _spec_names(specs)
    full_window_metadata = build_window_metadata(
        block,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
    )
    selection_manifest_path = getattr(args, "selection_manifest", None)
    manifest_rows: list[dict[str, Any]] = []
    if selection_manifest_path:
        manifest_rows = _load_selection_manifest_rows(
            selection_manifest_path,
            str(getattr(args, "manifest_split", "all")),
        )
        selected = apply_manifest_window_selection(
            history_level=history_level,
            history_norm=history_norm,
            center=center,
            scale=scale,
            drift_feature=drift_feature,
            history_raw=history_raw,
            window_metadata=full_window_metadata,
            manifest_rows=manifest_rows,
        )
        history_level = selected["history_level"]
        history_norm = selected["history_norm"]
        center = selected["center"]
        scale = selected["scale"]
        drift_feature = selected["drift_feature"]
        history_raw = selected["history_raw"]
        window_metadata = selected["window_metadata"]
        source_indices = selected["source_indices"]
        window_indices = selected["window_indices"]
    else:
        n = min(int(args.train_windows), int(history_level.shape[0]))
        window_indices = np.arange(n, dtype=np.int64)
        history_level = history_level[:n]
        history_norm = history_norm[:n]
        center = center[:n]
        scale = scale[:n]
        drift_feature = drift_feature[:n]
        history_raw = history_raw[:n]
        window_metadata = [
            _metadata_with_manifest_row(
                full_window_metadata[idx] if idx < len(full_window_metadata) else {},
                {
                    "window_index": idx,
                    "window_id": f"joint39_val_{idx:04d}",
                    "source_index": full_window_metadata[idx].get("source_index", idx)
                    if idx < len(full_window_metadata)
                    else idx,
                },
            )
            for idx in range(n)
        ]
        source_indices = np.asarray(
            [int(meta["source_index"]) for meta in window_metadata],
            dtype=np.int64,
        )
    memory_targets = compute_memory_targets(
        model,
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        device=device,
        batch_size=int(args.batch_size),
    )
    bundles = build_narrative_bundles(history_raw, spec_names, window_metadata, args)
    label_validation = [
        {
            "window_id": bundle["window_id"],
            "errors": bundle_validation_errors(bundle),
        }
        for bundle in bundles
    ]
    rejected_label_windows = [
        row["window_id"]
        for row in label_validation
        if row["errors"]
    ]
    if rejected_label_windows and not bool(args.allow_invalid_labels):
        keep = np.asarray(
            [idx for idx, row in enumerate(label_validation) if not row["errors"]],
            dtype=np.int64,
        )
        if keep.size == 0:
            raise ValueError("all OpenAI labels failed validation")
        history_level = history_level[keep]
        history_norm = history_norm[keep]
        center = center[keep]
        scale = scale[keep]
        drift_feature = drift_feature[keep]
        history_raw = history_raw[keep]
        memory_targets = memory_targets[keep]
        bundles = [bundles[int(idx)] for idx in keep]
        source_indices = source_indices[keep]
        window_indices = window_indices[keep]
        window_metadata = [window_metadata[int(idx)] for idx in keep]
    examples: list[dict[str, Any]] = []
    for idx, bundle in enumerate(bundles):
        examples.extend(build_narrative_training_examples(bundle, target_index=idx))
    texts = [example["text"] for example in examples]
    if args.embedding_backend == "openai":
        text_embeddings = embed_texts_with_openai(
            texts,
            model=args.embedding_model,
            dotenv_path=args.dotenv,
        )
    else:
        text_embeddings = hash_text_embeddings(texts, dim=int(args.hash_dim))
    target_indices = np.asarray(
        [-1 if example["target_index"] is None else int(example["target_index"]) for example in examples],
        dtype=np.int64,
    )
    roles = [str(example["role"]) for example in examples]
    groups = [str(example["window_id"]) for example in examples]
    train_result = train_narrative_adapter(
        text_embeddings,
        memory_targets,
        target_indices,
        roles,
        groups,
        condition_dim=int(payload["config"]["memory_dim"]),
        hidden_dim=args.hidden_dim,
        steps=int(args.adapter_steps),
        lr=float(args.adapter_lr),
        contrastive_weight=float(args.contrastive_weight),
        contrastive_margin=float(args.contrastive_margin),
        seed=int(args.seed),
    )
    adapter = train_result["adapter"]
    adapter_path = Path(args.output_dir) / "narrative_adapter.pt"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model,
            "hash_dim": int(args.hash_dim),
            "condition_dim": int(payload["config"]["memory_dim"]),
        },
        adapter_path,
    )
    query_text = args.query_text or examples[0]["text"]
    if args.embedding_backend == "openai":
        query_embedding = embed_texts_with_openai(
            [query_text],
            model=args.embedding_model,
            dotenv_path=args.dotenv,
        )
    else:
        query_embedding = hash_text_embeddings([query_text], dim=int(args.hash_dim))
    adapter.eval()
    with torch.no_grad():
        query_condition = adapter(torch.from_numpy(normalize_rows(query_embedding)).float()).cpu().numpy()[0]
    analogues = nearest_memory_indices(
        query_condition,
        memory_targets,
        top_k=int(args.top_k),
    )
    weighted_analogues = retrieval_weights(
        analogues,
        temperature=float(args.retrieval_temperature),
    )
    retrieval_report = retrieval_diagnostics(
        analogues,
        ood_threshold=float(args.ood_threshold),
    )
    sampled = sample_normal_generator_for_retrieved_analogues(
        model,
        analogues,
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        n_samples=int(args.samples),
        n_steps=int(args.n_steps),
        chunk_size=int(args.chunk_size),
        temperature=float(args.temperature),
        device=device,
    )
    retrieved_indices = sampled["indices"]
    increments = sampled["increments"]
    generated_states = _reconstruct_states(
        history_raw[retrieved_indices, -1, :],
        increments,
        specs,
    )
    scenario_summary = summarize_retrieval_generated_states(
        generated_states,
        history_raw[retrieved_indices, -1, :],
        spec_names,
    )
    report = {
        "status": "ok",
        "inference_mode": "retrieval_normal_generator",
        "scope_note": (
            "Production-style pilot: OpenAI-generated narrative labels are used "
            "for text conditioning when label_backend=openai. The learned text "
            "condition is used for historical analogue retrieval only. The frozen "
            "generator is called through its normal sample_batched path on real "
            "retrieved history prefixes, centers, and scales."
        ),
        "checkpoint": args.checkpoint,
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model if args.embedding_backend == "openai" else None,
        "label_backend": args.label_backend,
        "label_model": args.label_model if args.label_backend == "openai" else None,
        "local_test": bool(getattr(args, "local_test", False)),
        "label_cache": str(args.label_cache or Path(args.output_dir) / "narrative_label_cache.jsonl"),
        "selection_manifest": str(selection_manifest_path) if selection_manifest_path else None,
        "manifest_split": str(getattr(args, "manifest_split", "")) if selection_manifest_path else None,
        "requested_train_windows": int(len(window_metadata)),
        "window_metadata": window_metadata,
        "source_indices": source_indices.astype(int).tolist(),
        "window_indices": window_indices.astype(int).tolist(),
        "train_windows": int(len(bundles)),
        "rejected_label_windows": rejected_label_windows,
        "label_validation": label_validation,
        "example_count": len(examples),
        "memory_target_shape": list(memory_targets.shape),
        "adapter_training": {
            "loss_first": float(train_result["loss_first"]),
            "loss_last": float(train_result["loss_last"]),
            "steps": int(args.adapter_steps),
            "contrastive_weight": float(args.contrastive_weight),
        },
        "query_text": query_text,
        "query_condition_norm": float(np.linalg.norm(query_condition)),
        "retrieval_diagnostics": retrieval_report,
        "historical_analogues": [
            {
                **item,
                "weight": next(
                    weighted["weight"]
                    for weighted in weighted_analogues
                    if int(weighted["index"]) == int(item["index"])
                ),
                "source_index": int(source_indices[int(item["index"])]),
                "window_id": bundles[item["index"]]["window_id"],
                "primary_narrative": bundles[item["index"]]["narratives"][0]["text"],
            }
            for item in analogues
        ],
        "generation": scenario_summary,
        "artifact_paths": {
            "adapter": str(adapter_path),
            "npz": str(Path(args.output_dir) / "narrative_pipeline_arrays.npz"),
            "report": str(Path(args.output_dir) / "narrative_pipeline_report.json"),
        },
        "narrative_bundles": bundles,
    }
    np.savez_compressed(
        Path(args.output_dir) / "narrative_pipeline_arrays.npz",
        text_embeddings=normalize_rows(text_embeddings),
        memory_targets=memory_targets,
        query_condition=query_condition.astype(np.float32),
        generated_increments=increments.astype(np.float32),
        generated_states=generated_states.astype(np.float32),
        retrieved_indices=np.asarray(retrieved_indices, dtype=np.int64),
        source_indices=source_indices.astype(np.int64),
        window_indices=window_indices.astype(np.int64),
    )
    _write_json(Path(args.output_dir) / "narrative_pipeline_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-backend", choices=["openai", "rule"], default="openai")
    parser.add_argument("--label-model", default="gpt-5.4-mini")
    parser.add_argument("--label-cache")
    parser.add_argument(
        "--local-test",
        action="store_true",
        help="Allow rule labels for local smoke tests or CI. Do not use for reported production-style runs.",
    )
    parser.add_argument("--label-max-output-tokens", type=int, default=5000)
    parser.add_argument("--label-concurrency", type=int, default=4)
    parser.add_argument("--allow-invalid-labels", action="store_true")
    parser.add_argument("--embedding-backend", choices=["openai", "hash"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--hash-dim", type=int, default=512)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--train-windows", type=int, default=8)
    parser.add_argument("--selection-manifest")
    parser.add_argument("--manifest-split", choices=["train", "validation", "test", "all"], default="all")
    parser.add_argument("--adapter-steps", type=int, default=400)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-margin", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--query-text")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retrieval-temperature", type=float, default=0.05)
    parser.add_argument("--ood-threshold", type=float, default=0.75)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=773)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument("--eval_split", choices=["val", "train", "train_tail"], default="val")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    args = parser.parse_args()
    report = run_pipeline(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "adapter_loss_first": report["adapter_training"]["loss_first"],
                "adapter_loss_last": report["adapter_training"]["loss_last"],
                "nearest_window": report["historical_analogues"][0]["window_id"],
                "generated_state_shape": report["generation"]["generated_state_shape"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
