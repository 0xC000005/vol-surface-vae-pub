#!/usr/bin/env python
"""Build a bridge-style report from episode-level narrative retrieval.

The existing scenario-level evaluator consumes a bridge report whose held-out
query rows contain ``top_train_pool`` analogue rows. This isolated Phase 2
utility adapts episode-card retrieval into that schema without modifying the
incumbent bridge or scenario generator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    assert_cards_allowed_for_retrieval,
    build_query,
    pairwise_jaccard_summary,
    rank_episode_cards,
)


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_support_bank_cards_all_970f/episode_narrative_support_cards.jsonl"
)
DEFAULT_SUPPORT_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_bridge_report_970g"
)


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _window_index(card: dict[str, Any]) -> int:
    metadata = card.get("support_metadata", {})
    if isinstance(metadata, dict) and metadata.get("window_index") is not None:
        return int(metadata["window_index"])
    match = re.search(r"_(\d+)$", str(card.get("window_id", "")))
    if match:
        return int(match.group(1))
    raise ValueError(f"card has no window index: {card.get('window_id')}")


def _split_indices_from_support_report(
    report: dict[str, Any],
) -> tuple[list[int], list[int]]:
    rows = report.get("window_metadata", [])
    if not isinstance(rows, list):
        raise ValueError("support report missing window_metadata")
    train: list[int] = []
    test: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("window_index", len(train) + len(test)))
        split = str(row.get("manifest_split", ""))
        if split == "support_decoder_test":
            test.append(idx)
        else:
            train.append(idx)
    return train, test


def _cards_by_index(cards: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {_window_index(card): card for card in cards}


def _safe_scale(values: np.ndarray, train_indices: list[int]) -> np.ndarray:
    train = np.asarray(values, dtype=np.float32)[
        np.asarray(train_indices, dtype=np.int64)
    ]
    scale = np.nanstd(train, axis=0).astype(np.float32)
    positive = scale[np.isfinite(scale) & (scale > 1e-8)]
    fallback = float(np.nanmedian(positive)) if positive.size else 1.0
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback).astype(
        np.float32
    )


def _start_only_ranked_rows(
    *,
    query_index: int,
    history_raw: np.ndarray,
    train_indices: list[int],
    top_k: int,
    temporal_gap: int,
    cards_by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank historical supports using only accepted day-0/terminal state fit."""

    history = np.asarray(history_raw, dtype=np.float32)
    terminal = history[:, -1, :]
    scale = _safe_scale(terminal, train_indices)
    query_terminal = terminal[int(query_index)]
    ranked: list[dict[str, Any]] = []
    for candidate_index in train_indices:
        idx = int(candidate_index)
        if idx == int(query_index):
            continue
        if int(temporal_gap) > 0 and abs(idx - int(query_index)) < int(temporal_gap):
            continue
        diff = (terminal[idx] - query_terminal) / scale
        distance = float(np.sqrt(np.nanmean(np.square(diff))))
        match_score = float(1.0 / (1.0 + max(distance, 0.0)))
        card = cards_by_index.get(idx, {})
        ranked.append(
            {
                "window_index": idx,
                "window_id": str(card.get("window_id", f"window_{idx:04d}")),
                "scenario_title": str(card.get("scenario_title", "start-only support")),
                "start_distance": distance,
                "start_match_score": match_score,
            }
        )
    ranked.sort(key=lambda item: (item["start_distance"], item["window_index"]))
    selected = ranked[: max(0, int(top_k))]
    scores = [-float(item["start_distance"]) for item in selected]
    weights = _softmax_weights(scores, temperature=1.0)
    rows: list[dict[str, Any]] = []
    for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), 1):
        rows.append(
            {
                "rank": rank,
                "window_index": int(item["window_index"]),
                "window_id": str(item["window_id"]),
                "cosine": float(item["start_match_score"]),
                "weight": float(weight),
                "retrieval_score": float(item["start_match_score"]),
                "scenario_title": str(item["scenario_title"]),
                "score_components": {
                    "method": "start_only_terminal_state",
                    "start_distance": float(item["start_distance"]),
                    "start_match_score": float(item["start_match_score"]),
                },
            }
        )
    return rows


def _terminal_start_match(
    *,
    query_index: int,
    candidate_index: int,
    history_raw: np.ndarray,
    train_indices: list[int],
) -> tuple[float, float]:
    terminal = np.asarray(history_raw, dtype=np.float32)[:, -1, :]
    scale = _safe_scale(terminal, train_indices)
    diff = (terminal[int(candidate_index)] - terminal[int(query_index)]) / scale
    distance = float(np.sqrt(np.nanmean(np.square(diff))))
    return float(1.0 / (1.0 + max(distance, 0.0))), distance


def _temporal_gap_filter(
    ranked: list[dict[str, Any]],
    *,
    top_k: int,
    temporal_gap: int,
    query_index: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    gap = max(0, int(temporal_gap))
    for item in ranked:
        idx = item.get("window_index")
        # #48 causal gap: the candidate's `temporal_gap`-day window must end at/before the
        # query start, i.e. query_index - idx >= temporal_gap (also excludes idx==query and
        # near-future candidates). Backward-compatible: skipped when query_index is None.
        if (
            query_index is not None
            and gap > 0
            and idx is not None
            and int(query_index) - int(idx) < gap
        ):
            continue
        if (
            gap > 0
            and idx is not None
            and any(
                abs(int(idx) - int(row["window_index"])) < gap
                for row in selected
                if row.get("window_index") is not None
            )
        ):
            continue
        selected.append(item)
        if len(selected) >= int(top_k):
            break
    return selected


def _query_text(card: dict[str, Any]) -> str:
    views = card.get("views", {})
    if isinstance(views, dict):
        for name in ("full_professional", "mechanism_first", "sparse_user_query"):
            text = views.get(name)
            if isinstance(text, str) and text.strip():
                return text.strip()
    raise ValueError(f"card has no usable query text: {card.get('window_id')}")


def _softmax_weights(scores: list[float], *, temperature: float = 0.05) -> list[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-8)
    shifted = [(float(value) - max(scores)) / temp for value in scores]
    exps = [math.exp(value) for value in shifted]
    denom = sum(exps)
    if denom <= 0.0 or not math.isfinite(denom):
        return [1.0 / len(scores) for _ in scores]
    return [value / denom for value in exps]


def build_episode_retrieval_bridge_report(
    *,
    train_cards: list[dict[str, Any]],
    query_cards: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    method: str,
    top_k: int,
    temporal_gap: int,
    cards_path: str,
) -> dict[str, Any]:
    """Return a scenario-evaluator-compatible report for episode retrieval."""

    train_by_index = {_window_index(card): card for card in train_cards}
    ordered_train_cards = [
        train_by_index[idx] for idx in train_indices if idx in train_by_index
    ]
    if not ordered_train_cards:
        raise ValueError("no train cards available after train_indices filter")
    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    feature_cache: dict[str, dict[str, Any]] = {}

    for query_no, card in enumerate(query_cards):
        idx = _window_index(card)
        query = build_query(
            label=str(card.get("window_id", f"query_{idx:04d}")),
            text=_query_text(card),
            metadata={"window_index": idx},
        )
        ranked = rank_episode_cards(
            query,
            ordered_train_cards,
            method=method,
            top_k=top_k,
            temporal_gap=temporal_gap,
            feature_cache=feature_cache,
        )
        scores = [float(item["score"]) for item in ranked]
        weights = _softmax_weights(scores)
        top_train_pool = []
        for rank, (item, weight) in enumerate(zip(ranked, weights, strict=True), 1):
            top_train_pool.append(
                {
                    "rank": rank,
                    "window_index": int(item["window_index"]),
                    "window_id": str(item["window_id"]),
                    "cosine": float(item["score"]),
                    "weight": float(weight),
                    "retrieval_score": float(item["score"]),
                    "scenario_title": str(item.get("scenario_title", "")),
                    "score_components": item.get("score_components", {}),
                }
            )
        heldout_rows.append(
            {
                "query_id": f"episode_retrieval_{query_no:04d}_{idx}",
                "window_index": idx,
                "window_id": str(card.get("window_id", "")),
                "role": "anchor",
                "kind": "episode_narrative_retrieval",
                "query_text_source": "episode_card_full_professional",
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[str(card.get("window_id", idx))] = [
            str(item["window_id"]) for item in top_train_pool
        ]

    all_indices = sorted(set(int(idx) for idx in train_indices + test_indices))
    return {
        "schema_version": "nl_episode_narrative_bridge_report_v1",
        "status": "ok",
        "scope_note": (
            "Bridge-style report built from episode-card narrative-to-narrative "
            "retrieval. It is isolated from the incumbent text-to-memory bridge."
        ),
        "cards_path": str(cards_path),
        "retrieval_config": {
            "method": str(method),
            "top_k": int(top_k),
            "temporal_gap": int(temporal_gap),
            "support_pool": "support_train_only",
            "query_pool": "support_decoder_test",
        },
        "split": {
            "train_indices": [int(idx) for idx in train_indices],
            "test_indices": [int(idx) for idx in test_indices],
        },
        "window_indices": all_indices,
        "evaluation": {
            "heldout_examples": heldout_rows,
            "support_overlap": pairwise_jaccard_summary(result_sets),
        },
    }


def build_start_only_bridge_report(
    *,
    history_raw: np.ndarray,
    metadata: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    top_k: int,
    temporal_gap: int,
    cards_by_index: dict[int, dict[str, Any]],
    arrays_path: str,
) -> dict[str, Any]:
    """Return a bridge report whose supports are selected without narrative text."""

    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    metadata_by_index = {
        int(row.get("window_index", idx)): row for idx, row in enumerate(metadata)
    }
    for query_no, query_index in enumerate(test_indices):
        idx = int(query_index)
        top_train_pool = _start_only_ranked_rows(
            query_index=idx,
            history_raw=history_raw,
            train_indices=train_indices,
            top_k=top_k,
            temporal_gap=temporal_gap,
            cards_by_index=cards_by_index,
        )
        meta = metadata_by_index.get(idx, {})
        window_id = str(meta.get("window_id", f"window_{idx:04d}"))
        heldout_rows.append(
            {
                "query_id": f"start_only_{query_no:04d}_{idx}",
                "window_index": idx,
                "window_id": window_id,
                "role": "anchor",
                "kind": "start_only_terminal_state_retrieval",
                "query_text_source": "none_start_only_terminal_state",
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[window_id] = [str(item["window_id"]) for item in top_train_pool]

    all_indices = sorted(set(int(idx) for idx in train_indices + test_indices))
    return {
        "schema_version": "nl_episode_start_only_bridge_report_v1",
        "status": "ok",
        "scope_note": (
            "Bridge-style start-only baseline report. Supports are selected "
            "only by terminal-state compatibility with the accepted start; no "
            "narrative text or grounding claims enter the retrieval score."
        ),
        "cards_path": "",
        "arrays_path": str(arrays_path),
        "retrieval_config": {
            "method": "start_only_terminal_state",
            "top_k": int(top_k),
            "temporal_gap": int(temporal_gap),
            "support_pool": "support_train_only",
            "query_pool": "support_decoder_test",
        },
        "split": {
            "train_indices": [int(idx) for idx in train_indices],
            "test_indices": [int(idx) for idx in test_indices],
        },
        "window_indices": all_indices,
        "evaluation": {
            "heldout_examples": heldout_rows,
            "support_overlap": pairwise_jaccard_summary(result_sets),
        },
    }


def build_hybrid_start_text_bridge_report(
    *,
    train_cards: list[dict[str, Any]],
    query_cards: list[dict[str, Any]],
    history_raw: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
    method: str,
    top_k: int,
    temporal_gap: int,
    text_candidate_k: int,
    text_weight: float,
    start_weight: float,
    cards_path: str,
    arrays_path: str,
) -> dict[str, Any]:
    """Return a bridge report that reranks text candidates with start fit."""

    train_by_index = {_window_index(card): card for card in train_cards}
    ordered_train_cards = [
        train_by_index[idx] for idx in train_indices if idx in train_by_index
    ]
    if not ordered_train_cards:
        raise ValueError("no train cards available after train_indices filter")
    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    feature_cache: dict[str, dict[str, Any]] = {}
    text_w = float(text_weight)
    start_w = float(start_weight)
    denom = max(abs(text_w) + abs(start_w), 1e-8)
    text_w = text_w / denom
    start_w = start_w / denom

    for query_no, card in enumerate(query_cards):
        idx = _window_index(card)
        query = build_query(
            label=str(card.get("window_id", f"query_{idx:04d}")),
            text=_query_text(card),
            metadata={"window_index": idx},
        )
        recalled = rank_episode_cards(
            query,
            ordered_train_cards,
            method=method,
            top_k=max(int(text_candidate_k), int(top_k)),
            temporal_gap=0,
            feature_cache=feature_cache,
        )
        reranked: list[dict[str, Any]] = []
        for item in recalled:
            candidate_idx = int(item["window_index"])
            start_match, start_distance = _terminal_start_match(
                query_index=idx,
                candidate_index=candidate_idx,
                history_raw=history_raw,
                train_indices=train_indices,
            )
            text_score = float(item["score"])
            combined = text_w * text_score + start_w * start_match
            reranked.append(
                {
                    **item,
                    "score": float(combined),
                    "text_score": text_score,
                    "start_match_score": float(start_match),
                    "start_distance": float(start_distance),
                    "score_components": {
                        **dict(item.get("score_components", {})),
                        "method": "hybrid_start_text",
                        "text_score": text_score,
                        "start_match_score": float(start_match),
                        "start_distance": float(start_distance),
                        "text_weight": float(text_w),
                        "start_weight": float(start_w),
                    },
                }
            )
        reranked.sort(key=lambda item: item["score"], reverse=True)
        selected = _temporal_gap_filter(
            reranked,
            top_k=int(top_k),
            temporal_gap=int(temporal_gap),
        )
        scores = [float(item["score"]) for item in selected]
        weights = _softmax_weights(scores)
        top_train_pool = []
        for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), 1):
            top_train_pool.append(
                {
                    "rank": rank,
                    "window_index": int(item["window_index"]),
                    "window_id": str(item["window_id"]),
                    "cosine": float(item["score"]),
                    "weight": float(weight),
                    "retrieval_score": float(item["score"]),
                    "scenario_title": str(item.get("scenario_title", "")),
                    "score_components": item.get("score_components", {}),
                }
            )
        heldout_rows.append(
            {
                "query_id": f"hybrid_start_text_{query_no:04d}_{idx}",
                "window_index": idx,
                "window_id": str(card.get("window_id", "")),
                "role": "anchor",
                "kind": "hybrid_start_text_episode_retrieval",
                "query_text_source": "episode_card_full_professional",
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[str(card.get("window_id", idx))] = [
            str(item["window_id"]) for item in top_train_pool
        ]

    all_indices = sorted(set(int(idx) for idx in train_indices + test_indices))
    return {
        "schema_version": "nl_episode_hybrid_start_text_bridge_report_v1",
        "status": "ok",
        "scope_note": (
            "Bridge-style report built from local narrative-to-narrative recall "
            "followed by start-fit reranking. This tests whether narrative lift "
            "can be preserved without losing too much accepted-start fidelity."
        ),
        "cards_path": str(cards_path),
        "arrays_path": str(arrays_path),
        "retrieval_config": {
            "method": "hybrid_start_text",
            "text_method": str(method),
            "top_k": int(top_k),
            "text_candidate_k": int(text_candidate_k),
            "temporal_gap": int(temporal_gap),
            "text_weight": float(text_w),
            "start_weight": float(start_w),
            "support_pool": "support_train_only",
            "query_pool": "support_decoder_test",
        },
        "split": {
            "train_indices": [int(idx) for idx in train_indices],
            "test_indices": [int(idx) for idx in test_indices],
        },
        "window_indices": all_indices,
        "evaluation": {
            "heldout_examples": heldout_rows,
            "support_overlap": pairwise_jaccard_summary(result_sets),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector-mode",
        choices=["episode_text", "start_only", "hybrid_start_text"],
        default="episode_text",
    )
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument(
        "--support-arrays",
        type=Path,
        default=Path(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--method", choices=["lexical", "dense", "hybrid"], default="hybrid"
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--max-query-windows", type=int, default=0)
    parser.add_argument("--text-candidate-k", type=int, default=256)
    parser.add_argument("--text-weight", type=float, default=0.50)
    parser.add_argument("--start-weight", type=float, default=0.50)
    args = parser.parse_args(argv)

    cards = _read_jsonl(args.cards_jsonl)
    assert_cards_allowed_for_retrieval(cards, path=args.cards_jsonl)
    support_report = _load_json(args.support_report)
    train_indices, test_indices = _split_indices_from_support_report(support_report)
    if int(args.max_query_windows) > 0:
        test_indices = test_indices[: int(args.max_query_windows)]
    card_by_index = {_window_index(card): card for card in cards}
    if str(args.selector_mode) == "episode_text":
        train_cards = [
            card_by_index[idx] for idx in train_indices if idx in card_by_index
        ]
        query_cards = [
            card_by_index[idx] for idx in test_indices if idx in card_by_index
        ]
        report = build_episode_retrieval_bridge_report(
            train_cards=train_cards,
            query_cards=query_cards,
            train_indices=train_indices,
            test_indices=test_indices,
            method=args.method,
            top_k=int(args.top_k),
            temporal_gap=int(args.temporal_gap),
            cards_path=str(args.cards_jsonl),
        )
        query_count = len(query_cards)
        train_count = len(train_cards)
    elif str(args.selector_mode) == "start_only":
        with np.load(args.support_arrays) as payload:
            history_raw = payload["history_raw"].copy()
        metadata = support_report.get("window_metadata", [])
        if not isinstance(metadata, list):
            raise ValueError("support report missing window_metadata")
        report = build_start_only_bridge_report(
            history_raw=history_raw,
            metadata=metadata,
            train_indices=train_indices,
            test_indices=test_indices,
            top_k=int(args.top_k),
            temporal_gap=int(args.temporal_gap),
            cards_by_index=card_by_index,
            arrays_path=str(args.support_arrays),
        )
        query_count = len(report["evaluation"]["heldout_examples"])
        train_count = len(train_indices)
    else:
        with np.load(args.support_arrays) as payload:
            history_raw = payload["history_raw"].copy()
        train_cards = [
            card_by_index[idx] for idx in train_indices if idx in card_by_index
        ]
        query_cards = [
            card_by_index[idx] for idx in test_indices if idx in card_by_index
        ]
        report = build_hybrid_start_text_bridge_report(
            train_cards=train_cards,
            query_cards=query_cards,
            history_raw=history_raw,
            train_indices=train_indices,
            test_indices=test_indices,
            method=args.method,
            top_k=int(args.top_k),
            temporal_gap=int(args.temporal_gap),
            text_candidate_k=int(args.text_candidate_k),
            text_weight=float(args.text_weight),
            start_weight=float(args.start_weight),
            cards_path=str(args.cards_jsonl),
            arrays_path=str(args.support_arrays),
        )
        query_count = len(query_cards)
        train_count = len(train_cards)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_name = (
        "episode_retrieval_bridge_report.json"
        if str(args.selector_mode) == "episode_text"
        else (
            "start_only_bridge_report.json"
            if str(args.selector_mode) == "start_only"
            else "hybrid_start_text_bridge_report.json"
        )
    )
    report_path = args.output_dir / report_name
    report["artifact_paths"] = {"report": str(report_path)}
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "selector_mode": str(args.selector_mode),
                "query_count": query_count,
                "train_count": train_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
