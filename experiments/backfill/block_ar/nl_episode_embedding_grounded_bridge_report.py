#!/usr/bin/env python
"""Build a bridge report from grounded OpenAI narrative retrieval.

This is an isolated Stage-1/guardrail report: rank historical support windows by
raw OpenAI narrative embeddings, require agreement with the top current/recent
grounding claims, and emit the standard ``evaluation.heldout_examples`` schema
consumed by ``nl_scenario_level_evaluation.py``. No narratives are generated and
the frozen SNI rollout is not touched here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    _load_json,
    _read_jsonl,
    _softmax_weights,
    _split_indices_from_support_report,
    _temporal_gap_filter,
    _window_index,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    _candidate_view_rows,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    assert_cards_allowed_for_retrieval,
    pairwise_jaccard_summary,
)
from experiments.backfill.block_ar.nl_episode_text_memory_bridge_report import (  # noqa: E402
    _apply_top3_90_selection,
    _direction_check,
    _required_grounding_claims,
)


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)
DEFAULT_SUPPORT_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_EMBEDDING_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_embedding_bridge_hybrid_66q/"
    "embedding_bridge_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_embedding_grounded_top3_90_66q"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rank_grounded_by_embeddings(
    *,
    query_vector: np.ndarray,
    query_index: int,
    candidate_vectors: np.ndarray,
    candidate_rows: list[dict[str, Any]],
    cards_by_index: dict[int, dict[str, Any]],
    query_claims: list[dict[str, Any]],
    support_pool_size: int,
    temporal_gap: int,
    max_grounding_mismatches: int,
) -> list[dict[str, Any]]:
    scores = np.asarray(candidate_vectors @ query_vector, dtype=np.float32)
    ranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for pos in np.argsort(-scores):
        row = candidate_rows[int(pos)]
        idx = int(row["window_index"])
        if idx in seen:
            continue
        seen.add(idx)
        # #48 causal gap (before the support_pool_size*8 early-break): the candidate's
        # temporal_gap-day window must end at/before the query start.
        if int(query_index) - idx < int(temporal_gap):
            continue
        direction_check = _direction_check(
            query_claims=query_claims,
            candidate_card=cards_by_index.get(idx, {}),
            max_mismatches=int(max_grounding_mismatches),
        )
        if str(direction_check.get("status", "")) != "pass":
            continue
        ranked.append(
            {
                "window_index": idx,
                "window_id": str(row["window_id"]),
                "scenario_title": str(row.get("scenario_title", "")),
                "score": float(scores[int(pos)]),
                "score_components": {
                    "method": "raw_openai_embedding_grounded_candidate",
                    "embedding_score": float(scores[int(pos)]),
                    "view": str(row.get("view", "")),
                    "direction_check": direction_check,
                },
            }
        )
        if len(ranked) >= max(int(support_pool_size) * 8, int(support_pool_size)):
            break
    return _temporal_gap_filter(
        ranked,
        top_k=int(support_pool_size),
        temporal_gap=int(temporal_gap),
        query_index=int(query_index),
    )


def build_grounded_embedding_bridge_report(args: argparse.Namespace) -> dict[str, Any]:
    cards = _read_jsonl(Path(args.cards_jsonl))
    assert_cards_allowed_for_retrieval(cards, path=Path(args.cards_jsonl))
    support_report = _load_json(Path(args.support_report))
    train_indices, test_indices = _split_indices_from_support_report(support_report)
    if int(args.max_query_windows) > 0:
        test_indices = test_indices[: int(args.max_query_windows)]
    card_by_index = {_window_index(card): card for card in cards}
    train_cards = [card_by_index[idx] for idx in train_indices if idx in card_by_index]
    query_cards = [card_by_index[idx] for idx in test_indices if idx in card_by_index]
    candidate_rows, candidate_texts = _candidate_view_rows(
        train_cards, tuple(DEFAULT_VIEW_NAMES)
    )
    with np.load(Path(args.embedding_arrays)) as payload:
        embeddings = payload["text_embeddings"].astype(np.float32)
        candidate_count = int(payload["candidate_text_count"][0])
        cached_query_indices = payload["query_window_indices"].astype(np.int64)
    if candidate_count != len(candidate_rows) or candidate_count != len(candidate_texts):
        raise ValueError(
            f"candidate cache mismatch: arrays={candidate_count}, rows={len(candidate_rows)}, "
            f"texts={len(candidate_texts)}"
        )
    query_indices = np.asarray([_window_index(card) for card in query_cards], dtype=np.int64)
    if query_indices.shape[0] > cached_query_indices.shape[0] or not np.array_equal(
        query_indices, cached_query_indices[: query_indices.shape[0]]
    ):
        raise ValueError(
            "cached query embeddings do not match requested held-out query order; "
            "rerun nl_episode_narrative_embedding_bridge_report.py first"
        )
    candidate_vectors = embeddings[:candidate_count]
    query_vectors = embeddings[candidate_count : candidate_count + len(query_cards)]

    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    total_grounding_pass = 0
    for query_no, (card, query_vector) in enumerate(zip(query_cards, query_vectors, strict=True)):
        idx = _window_index(card)
        claims = _required_grounding_claims(
            card, max_claims=int(args.max_grounding_claims)
        )
        selected = _rank_grounded_by_embeddings(
            query_vector=query_vector,
            query_index=int(idx),
            candidate_vectors=candidate_vectors,
            candidate_rows=candidate_rows,
            cards_by_index=card_by_index,
            query_claims=claims,
            support_pool_size=max(int(args.support_pool_size), int(args.top_k)),
            temporal_gap=int(args.temporal_gap),
            max_grounding_mismatches=int(args.max_grounding_mismatches),
        )
        weights = _softmax_weights([float(item["score"]) for item in selected])
        candidate_pool: list[dict[str, Any]] = []
        for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), 1):
            components = dict(item.get("score_components", {}))
            components["method"] = "raw_openai_embedding_grounded_top3_90_candidate"
            candidate_pool.append(
                {
                    "rank": int(rank),
                    "window_index": int(item["window_index"]),
                    "window_id": str(item["window_id"]),
                    "cosine": float(item["score"]),
                    "weight": float(weight),
                    "retrieval_score": float(item["score"]),
                    "scenario_title": str(item.get("scenario_title", "")),
                    "score_components": components,
                }
            )
        top_train_pool = (
            _apply_top3_90_selection(candidate_pool, min_weight_mass=0.90)
            if bool(args.top3_90)
            else candidate_pool[: int(args.top_k)]
        )
        for display_rank, item in enumerate(top_train_pool, 1):
            item["rank"] = int(display_rank)
            components = dict(item.get("score_components", {}))
            components["method"] = (
                "raw_openai_embedding_grounded_top3_90"
                if bool(args.top3_90)
                else "raw_openai_embedding_grounded"
            )
            item["score_components"] = components
        total_grounding_pass += sum(
            1
            for item in top_train_pool
            if str(
                item.get("score_components", {})
                .get("direction_check", {})
                .get("status", "")
            )
            == "pass"
        )
        heldout_rows.append(
            {
                "query_id": f"embedding_grounded_{query_no:04d}_{idx}",
                "window_index": int(idx),
                "window_id": str(card.get("window_id", "")),
                "role": "anchor",
                "kind": "raw_openai_embedding_grounded_top3_90"
                if bool(args.top3_90)
                else "raw_openai_embedding_grounded",
                "query_text_source": "episode_card_full_professional",
                "required_grounding_claims": claims,
                "candidate_pool_size": int(len(candidate_pool)),
                "pre_top3_90_candidate_pool": candidate_pool,
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[str(card.get("window_id", idx))] = [
            str(item["window_id"]) for item in top_train_pool
        ]

    all_indices = sorted(set(int(idx) for idx in train_indices + test_indices))
    return {
        "schema_version": "nl_episode_embedding_grounded_bridge_report_v1",
        "status": "ok",
        "scope_note": (
            "Grounded OpenAI narrative-retrieval bridge report. It uses cached "
            "text-embedding-3-large vectors and generic current/recent grounding "
            "checks; it does not project text into SNI memory."
        ),
        "cards_path": str(args.cards_jsonl),
        "arrays_path": str(args.embedding_arrays),
        "retrieval_config": {
            "method": "raw_openai_embedding_grounded_top3_90"
            if bool(args.top3_90)
            else "raw_openai_embedding_grounded",
            "top_k": int(args.top_k),
            "support_pool_size": int(args.support_pool_size),
            "temporal_gap": int(args.temporal_gap),
            "max_grounding_claims": int(args.max_grounding_claims),
            "max_grounding_mismatches": int(args.max_grounding_mismatches),
            "view_names": list(DEFAULT_VIEW_NAMES),
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
            "grounding_pass_top_pool_count": int(total_grounding_pass),
            "grounding_top_pool_count": int(
                sum(len(row["top_train_pool"]) for row in heldout_rows)
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--embedding-arrays", type=Path, default=DEFAULT_EMBEDDING_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--support-pool-size", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--max-query-windows", type=int, default=66)
    parser.add_argument("--max-grounding-claims", type=int, default=4)
    parser.add_argument("--max-grounding-mismatches", type=int, default=0)
    parser.add_argument("--top3-90", action="store_true", default=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_grounded_embedding_bridge_report(args)
    report_path = output_dir / "embedding_grounded_bridge_report.json"
    report["artifact_paths"] = {"report": str(report_path)}
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "heldout_query_count": len(report["evaluation"]["heldout_examples"]),
                "grounding_pass_top_pool_count": report["evaluation"][
                    "grounding_pass_top_pool_count"
                ],
                "grounding_top_pool_count": report["evaluation"][
                    "grounding_top_pool_count"
                ],
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
