#!/usr/bin/env python
"""Prepare linked hard-negative generation manifests.

This script selects real incompatible historical windows and writes Codex/GPT
generation prompts. It deliberately does not author hard-negative narrative
prose locally.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "hard_negative_bank_regeneration_985b_manifest"
)

POSITIVE_TRAINING_VIEWS = (
    "sparse_user_query",
    "weekly_risk_monitor",
    "risk_manager_memo",
    "institutional_risk_committee_note",
    "technical_factor_evidence",
    "mechanism_first",
    "full_professional",
    "factor_list_baseline",
)
FACTORS = (
    "SPX",
    "VIX",
    "BBB_OAS",
    "AAA_OAS",
    "DXY",
    "USDJPY",
    "CRUDE_OIL",
    "US2Y",
    "US10Y",
    "GOLD",
)
FACTOR_ALIASES = {
    "CRUDE_OIL": ("CRUDE_OIL", "crude oil", "oil"),
    "GOLD": ("GOLD", "gold"),
    "BBB_OAS": ("BBB_OAS", "BBB OAS", "BBB spreads", "lower-quality credit"),
    "AAA_OAS": ("AAA_OAS", "AAA OAS", "AAA spreads", "high-grade credit"),
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _window_number(window_id: str) -> int:
    match = re.search(r"_(\d+)$", str(window_id))
    return int(match.group(1)) if match else -10_000_000


def _mechanical_summary(card: dict[str, Any]) -> str:
    fields = card.get("caption_fields") if isinstance(card.get("caption_fields"), dict) else {}
    views = card.get("views") if isinstance(card.get("views"), dict) else {}
    return _compact(fields.get("mechanical_summary") or views.get("factor_list_baseline"))


def _segment_for_factor(summary: str, factor: str) -> str:
    parts = re.split(r";|,|\.", summary)
    aliases = FACTOR_ALIASES.get(factor, (factor,))
    for part in parts:
        low = part.lower()
        if any(alias.lower() in low for alias in aliases):
            return part
    return ""


def _direction_from_segment(segment: str, *, is_spread: bool = False) -> int:
    low = segment.lower()
    if re.search(r"\b(flat|stable|unchanged|mixed)\b", low):
        return 0
    numeric = re.search(r"(?<![A-Za-z0-9])([+-]\d+(?:\.\d+)?)", segment)
    if numeric:
        value = float(numeric.group(1))
        if value > 0:
            return 1
        if value < 0:
            return -1
    if is_spread:
        if re.search(r"\b(wider|widening|up|higher)\b", low):
            return 1
        if re.search(r"\b(tighter|tightening|down|lower)\b", low):
            return -1
    else:
        if re.search(r"\b(up|higher|firmer|stronger|rising|rallied|bid)\b", low):
            return 1
        if re.search(r"\b(down|lower|weaker|falling|declined|sold off)\b", low):
            return -1
    return 0


def _sign_vector(card: dict[str, Any]) -> np.ndarray:
    summary = _mechanical_summary(card)
    signs = np.zeros(len(FACTORS), dtype=np.int8)
    for pos, factor in enumerate(FACTORS):
        signs[pos] = _direction_from_segment(
            _segment_for_factor(summary, factor),
            is_spread=factor.endswith("_OAS"),
        )
    return signs


def _contradiction_channels(a: np.ndarray, b: np.ndarray) -> list[str]:
    channels = []
    for factor, left, right in zip(FACTORS, a, b):
        if int(left) != 0 and int(right) != 0 and int(left) * int(right) < 0:
            channels.append(factor)
    return channels


def _select_negative_indices(
    *,
    cards: list[dict[str, Any]],
    signs: np.ndarray,
    target_idx: int,
    count: int,
    min_temporal_gap: int,
) -> list[int]:
    target = signs[target_idx]
    products = signs * target[None, :]
    contradictions = (products < 0).sum(axis=1).astype(np.float32)
    agreements = (products > 0).sum(axis=1).astype(np.float32)
    same_archetype = np.array(
        [
            str(card.get("archetype", "")) == str(cards[target_idx].get("archetype", ""))
            for card in cards
        ],
        dtype=np.float32,
    )
    target_number = _window_number(str(cards[target_idx].get("window_id", "")))
    numbers = np.array([
        _window_number(str(card.get("window_id", ""))) for card in cards
    ])
    temporal_ok = np.abs(numbers - target_number) >= int(min_temporal_gap)
    score = contradictions - 0.10 * agreements + 0.35 * (1.0 - same_archetype)
    score[target_idx] = -1e9
    score[~temporal_ok] = -1e9
    score[contradictions < 1] = -1e9
    order = np.argsort(-score)
    return [int(idx) for idx in order[:count] if score[int(idx)] > -1e8]


def _codex_prompt(
    *,
    positive_card: dict[str, Any],
    negative_card: dict[str, Any],
    positive_view: str,
    positive_text: str,
    contradiction_channels: list[str],
) -> str:
    negative_fields = negative_card.get("caption_fields")
    if not isinstance(negative_fields, dict):
        negative_fields = {}
    evidence = negative_fields.get("evidence_used", [])
    if isinstance(evidence, list):
        evidence_text = "; ".join(_compact(item) for item in evidence if _compact(item))
    else:
        evidence_text = _compact(evidence)
    return "\n".join(
        [
            "Write one hard-negative narrative for contrastive training.",
            "The text must describe the NEGATIVE historical 30-day prefix only.",
            "Do not forecast future paths, terminal moves, P&L, VaR, or realized outcomes.",
            "Do not copy the positive narrative. Do not mention that this is a negative example.",
            "Use professional risk-manager language and match the requested view style.",
            "",
            f"Requested view style: {positive_view}",
            f"Positive window id: {positive_card.get('window_id')}",
            f"Positive title: {positive_card.get('scenario_title')}",
            f"Positive view text: {_compact(positive_text)}",
            "",
            f"Negative window id: {negative_card.get('window_id')}",
            f"Negative title: {negative_card.get('scenario_title')}",
            f"Negative archetype: {negative_card.get('archetype')}",
            f"Negative mechanical summary: {_mechanical_summary(negative_card)}",
            f"Negative evidence: {evidence_text}",
            f"Contradiction channels versus positive: {', '.join(contradiction_channels)}",
            "",
            "Return only the hard-negative narrative text.",
        ]
    )


def build_manifest(
    *,
    cards: list[dict[str, Any]],
    negatives_per_window: int,
    min_temporal_gap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    signs = np.stack([_sign_vector(card) for card in cards], axis=0)
    rows: list[dict[str, Any]] = []
    missing_positive_views = 0
    insufficient_negative_windows = 0
    for idx, card in enumerate(cards):
        views = card.get("views") if isinstance(card.get("views"), dict) else {}
        selected = _select_negative_indices(
            cards=cards,
            signs=signs,
            target_idx=idx,
            count=max(int(negatives_per_window), len(POSITIVE_TRAINING_VIEWS)),
            min_temporal_gap=min_temporal_gap,
        )
        if len(selected) < len(POSITIVE_TRAINING_VIEWS):
            insufficient_negative_windows += 1
        for view_pos, view in enumerate(POSITIVE_TRAINING_VIEWS):
            positive_text = _compact(views.get(view))
            if not positive_text:
                missing_positive_views += 1
                continue
            negative_idx = selected[view_pos % max(1, len(selected))] if selected else None
            if negative_idx is None:
                continue
            negative = cards[int(negative_idx)]
            channels = _contradiction_channels(signs[idx], signs[int(negative_idx)])
            rows.append(
                {
                    "schema_version": "hard_negative_generation_manifest_v1",
                    "target_window_id": card.get("window_id"),
                    "target_window_index": idx,
                    "target_title": card.get("scenario_title"),
                    "target_archetype": card.get("archetype"),
                    "positive_view": view,
                    "positive_text": positive_text,
                    "negative_window_id": negative.get("window_id"),
                    "negative_window_index": int(negative_idx),
                    "negative_title": negative.get("scenario_title"),
                    "negative_archetype": negative.get("archetype"),
                    "contradiction_channels": channels,
                    "contradiction_count": len(channels),
                    "target_mechanical_summary": _mechanical_summary(card),
                    "negative_mechanical_summary": _mechanical_summary(negative),
                    "codex_prompt": _codex_prompt(
                        positive_card=card,
                        negative_card=negative,
                        positive_view=view,
                        positive_text=positive_text,
                        contradiction_channels=channels,
                    ),
                    "generated_negative_text": "",
                }
            )
    summary = {
        "schema_version": "hard_negative_generation_manifest_summary_v1",
        "card_count": len(cards),
        "positive_training_views": list(POSITIVE_TRAINING_VIEWS),
        "manifest_rows": len(rows),
        "target_rows": len(cards) * len(POSITIVE_TRAINING_VIEWS),
        "missing_positive_view_rows": int(missing_positive_views),
        "windows_with_insufficient_negative_candidates": int(insufficient_negative_windows),
        "min_temporal_gap": int(min_temporal_gap),
        "local_prose_generated": False,
        "interpretation": "This is a Codex/GPT generation manifest only. It links positives to incompatible historical windows and prompts Codex/GPT to author the hard-negative text.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--negatives-per-window", type=int, default=8)
    parser.add_argument("--min-temporal-gap", type=int, default=30)
    args = parser.parse_args()

    cards = _read_jsonl(args.cards_jsonl)
    rows, summary = build_manifest(
        cards=cards,
        negatives_per_window=args.negatives_per_window,
        min_temporal_gap=args.min_temporal_gap,
    )
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "hard_negative_generation_manifest.jsonl"
    summary_path = out_dir / "hard_negative_generation_manifest_summary.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok" if summary["manifest_rows"] == summary["target_rows"] else "needs_review",
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "rows": summary["manifest_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
