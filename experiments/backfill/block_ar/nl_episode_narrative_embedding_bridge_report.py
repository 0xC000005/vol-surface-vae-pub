#!/usr/bin/env python
"""Build bridge reports from true semantic episode-card retrieval.

This isolated utility mirrors ``nl_episode_narrative_bridge_report.py`` but
uses dense text embeddings instead of the local lexical/hashed-vector scorer.
The output schema is intentionally identical enough for
``nl_scenario_level_evaluation.py`` to consume without changing the incumbent
paper/demo pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    _load_json,
    _query_text,
    _read_jsonl,
    _softmax_weights,
    _split_indices_from_support_report,
    _temporal_gap_filter,
    _terminal_start_match,
    _window_index,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    assert_cards_allowed_for_retrieval,
    pairwise_jaccard_summary,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    hash_text_embeddings,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    embed_texts_with_openai,
    normalize_rows,
)


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat/multiformat_episode_cards.jsonl"
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
    "episode_card_v3_embedding_bridge"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hash_texts(texts: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=12)
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def embed_with_cache(
    texts: list[str],
    *,
    output_dir: Path,
    backend: str,
    model: str,
    dotenv_path: str,
    batch_size: int,
    hash_dim: int,
    retry_attempts: int = 6,
    retry_sleep: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return normalized embeddings, using a content-hash cache."""

    if not texts:
        raise ValueError("texts must not be empty")
    cache_dir = output_dir / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() else "_" for ch in str(model))
    cache_path = cache_dir / f"{backend}_{safe_model}_{_hash_texts(texts)}.npz"
    if cache_path.exists():
        with np.load(cache_path) as payload:
            arr = payload["embeddings"].copy()
        return arr, {
            "backend": backend,
            "model": model,
            "cache_hit": True,
            "cache_path": str(cache_path),
            "embedding_count": len(texts),
            "embedding_dim": int(arr.shape[1]),
        }
    batch_cache_hits = 0
    batch_cache_misses = 0
    if backend == "hash":
        arr = hash_text_embeddings(texts, dim=int(hash_dim))
    elif backend == "openai":
        chunks: list[np.ndarray] = []
        effective_batch_size = max(1, int(batch_size))
        for start in range(0, len(texts), effective_batch_size):
            end = min(len(texts), start + effective_batch_size)
            batch = texts[start:end]
            batch_path = (
                cache_dir
                / f"{backend}_{safe_model}_batch_{start:06d}_{end:06d}_{_hash_texts(batch)}.npz"
            )
            if batch_path.exists():
                with np.load(batch_path) as payload:
                    batch_arr = payload["embeddings"].copy()
                batch_cache_hits += 1
            else:
                last_error: Exception | None = None
                for attempt in range(max(1, int(retry_attempts))):
                    try:
                        batch_arr = embed_texts_with_openai(
                            batch,
                            model=model,
                            dotenv_path=dotenv_path,
                            batch_size=len(batch),
                        )
                        np.savez_compressed(batch_path, embeddings=batch_arr)
                        batch_cache_misses += 1
                        break
                    except Exception as exc:  # OpenAI raises typed errors from an optional dependency.
                        last_error = exc
                        if attempt + 1 >= max(1, int(retry_attempts)):
                            raise
                        sleep_seconds = float(retry_sleep) * (2**attempt)
                        print(
                            f"embedding batch {start}-{end} failed "
                            f"({type(exc).__name__}); retrying in {sleep_seconds:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(sleep_seconds)
                else:  # pragma: no cover - defensive fallback for type checkers.
                    raise RuntimeError("embedding retry loop exited unexpectedly") from last_error
            chunks.append(np.asarray(batch_arr, dtype=np.float32))
            print(
                f"embedded texts {end}/{len(texts)}",
                file=sys.stderr,
                flush=True,
            )
        arr = np.concatenate(chunks, axis=0)
        arr = normalize_rows(arr)
    else:
        raise ValueError(f"unknown embedding backend: {backend}")
    arr = np.asarray(arr, dtype=np.float32)
    np.savez_compressed(cache_path, embeddings=arr)
    return arr, {
        "backend": backend,
        "model": model,
        "cache_hit": False,
        "cache_path": str(cache_path),
        "embedding_count": len(texts),
        "embedding_dim": int(arr.shape[1]),
        "batch_cache_hits": batch_cache_hits,
        "batch_cache_misses": batch_cache_misses,
    }


def _candidate_view_rows(
    train_cards: list[dict[str, Any]], view_names: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for card in train_cards:
        views = card.get("views", {})
        if not isinstance(views, dict):
            continue
        for view_name in view_names:
            text = views.get(view_name, "")
            if not isinstance(text, str) or not text.strip():
                continue
            rows.append(
                {
                    "window_id": str(card.get("window_id", "")),
                    "window_index": _window_index(card),
                    "split": str(card.get("split", "")),
                    "scenario_title": str(card.get("scenario_title", "")),
                    "archetype": str(card.get("archetype", "")),
                    "view": view_name,
                    "text": text.strip(),
                }
            )
            texts.append(text.strip())
    if not rows:
        raise ValueError("no candidate view rows")
    return rows, texts


def _rank_by_embeddings(
    *,
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_rows: list[dict[str, Any]],
    top_k: int,
    temporal_gap: int,
) -> list[dict[str, Any]]:
    scores = np.asarray(candidate_vectors @ query_vector, dtype=np.float32)
    order = np.argsort(-scores)
    best_by_window: dict[int, dict[str, Any]] = {}
    for pos in order:
        row = candidate_rows[int(pos)]
        idx = int(row["window_index"])
        if idx in best_by_window:
            continue
        score = float(scores[int(pos)])
        best_by_window[idx] = {
            "window_index": idx,
            "window_id": str(row["window_id"]),
            "scenario_title": str(row.get("scenario_title", "")),
            "score": score,
            "score_components": {
                "method": "semantic_embedding",
                "embedding_score": score,
                "view": str(row.get("view", "")),
            },
        }
        if len(best_by_window) >= max(int(top_k) * 8, int(top_k)):
            break
    ranked = sorted(best_by_window.values(), key=lambda item: item["score"], reverse=True)
    return _temporal_gap_filter(ranked, top_k=int(top_k), temporal_gap=int(temporal_gap))


def build_embedding_bridge_report(
    *,
    cards: list[dict[str, Any]],
    support_report: dict[str, Any],
    support_arrays_path: Path,
    output_dir: Path,
    selector_mode: str,
    embedding_backend: str,
    embedding_model: str,
    dotenv_path: str,
    embedding_batch_size: int,
    hash_dim: int,
    top_k: int,
    temporal_gap: int,
    max_query_windows: int,
    text_candidate_k: int,
    text_weight: float,
    start_weight: float,
    embedding_retry_attempts: int = 6,
    embedding_retry_sleep: float = 2.0,
    view_names: tuple[str, ...] = DEFAULT_VIEW_NAMES,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_indices, test_indices = _split_indices_from_support_report(support_report)
    if int(max_query_windows) > 0:
        test_indices = test_indices[: int(max_query_windows)]
    card_by_index = {_window_index(card): card for card in cards}
    train_cards = [card_by_index[idx] for idx in train_indices if idx in card_by_index]
    query_cards = [card_by_index[idx] for idx in test_indices if idx in card_by_index]
    candidate_rows, candidate_texts = _candidate_view_rows(train_cards, view_names)
    query_texts = [_query_text(card) for card in query_cards]
    all_texts = candidate_texts + query_texts
    embeddings, embedding_meta = embed_with_cache(
        all_texts,
        output_dir=output_dir,
        backend=embedding_backend,
        model=embedding_model,
        dotenv_path=dotenv_path,
        batch_size=int(embedding_batch_size),
        hash_dim=int(hash_dim),
        retry_attempts=int(embedding_retry_attempts),
        retry_sleep=float(embedding_retry_sleep),
    )
    candidate_vectors = embeddings[: len(candidate_texts)]
    query_vectors = embeddings[len(candidate_texts) :]

    history_raw: np.ndarray | None = None
    text_w = float(text_weight)
    start_w = float(start_weight)
    denom = max(abs(text_w) + abs(start_w), 1e-8)
    text_w = text_w / denom
    start_w = start_w / denom
    if selector_mode == "hybrid_embedding_start":
        with np.load(support_arrays_path) as payload:
            history_raw = payload["history_raw"].copy()

    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    for query_no, (card, query_vector) in enumerate(zip(query_cards, query_vectors, strict=True)):
        idx = _window_index(card)
        recall_k = int(text_candidate_k) if selector_mode == "hybrid_embedding_start" else int(top_k)
        ranked = _rank_by_embeddings(
            query_vector=query_vector,
            candidate_vectors=candidate_vectors,
            candidate_rows=candidate_rows,
            top_k=max(recall_k, int(top_k)),
            temporal_gap=0 if selector_mode == "hybrid_embedding_start" else int(temporal_gap),
        )
        if selector_mode == "hybrid_embedding_start":
            assert history_raw is not None
            reranked: list[dict[str, Any]] = []
            for item in ranked:
                start_match, start_distance = _terminal_start_match(
                    query_index=idx,
                    candidate_index=int(item["window_index"]),
                    history_raw=history_raw,
                    train_indices=train_indices,
                )
                text_score = float(item["score"])
                combined = text_w * text_score + start_w * float(start_match)
                reranked.append(
                    {
                        **item,
                        "score": float(combined),
                        "score_components": {
                            **dict(item.get("score_components", {})),
                            "method": "hybrid_embedding_start",
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
                reranked, top_k=int(top_k), temporal_gap=int(temporal_gap)
            )
            kind = "hybrid_embedding_start_episode_retrieval"
        else:
            selected = ranked
            kind = "semantic_embedding_episode_retrieval"
        weights = _softmax_weights([float(item["score"]) for item in selected])
        top_train_pool = []
        for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), 1):
            top_train_pool.append(
                {
                    "rank": int(rank),
                    "window_index": int(item["window_index"]),
                    "window_id": str(item["window_id"]),
                    "cosine": float(item["score"]),
                    "weight": float(weight),
                    "retrieval_score": float(item["score"]),
                    "scenario_title": str(item.get("scenario_title", "")),
                    "score_components": dict(item.get("score_components", {})),
                }
            )
        heldout_rows.append(
            {
                "query_id": f"{selector_mode}_{query_no:04d}_{idx}",
                "window_index": int(idx),
                "window_id": str(card.get("window_id", "")),
                "role": "anchor",
                "kind": kind,
                "query_text_source": "episode_card_full_professional",
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[str(card.get("window_id", idx))] = [
            str(item["window_id"]) for item in top_train_pool
        ]

    all_indices = sorted(set(int(idx) for idx in train_indices + test_indices))
    report = {
        "schema_version": "nl_episode_embedding_bridge_report_v1",
        "status": "ok",
        "scope_note": (
            "Bridge-style report built from semantic narrative-to-narrative "
            "episode retrieval. It is isolated from the incumbent paper/demo "
            "pipeline and is evaluated downstream through the frozen SNI rollout."
        ),
        "cards_path": "",
        "arrays_path": str(support_arrays_path),
        "embedding_metadata": embedding_meta,
        "retrieval_config": {
            "method": str(selector_mode),
            "top_k": int(top_k),
            "text_candidate_k": int(text_candidate_k),
            "temporal_gap": int(temporal_gap),
            "text_weight": float(text_w),
            "start_weight": float(start_w),
            "view_names": list(view_names),
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
    arrays = {
        "text_embeddings": embeddings.astype(np.float32),
        "candidate_text_count": np.asarray([len(candidate_texts)], dtype=np.int64),
        "query_window_indices": np.asarray(
            [_window_index(card) for card in query_cards], dtype=np.int64
        ),
    }
    return report, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector-mode",
        choices=["episode_embedding", "hybrid_embedding_start"],
        default="episode_embedding",
    )
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--support-arrays", type=Path, default=DEFAULT_SUPPORT_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-backend", choices=["hash", "openai"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-retry-attempts", type=int, default=6)
    parser.add_argument("--embedding-retry-sleep", type=float, default=2.0)
    parser.add_argument("--hash-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--max-query-windows", type=int, default=0)
    parser.add_argument("--text-candidate-k", type=int, default=128)
    parser.add_argument("--text-weight", type=float, default=0.25)
    parser.add_argument("--start-weight", type=float, default=0.75)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = _read_jsonl(args.cards_jsonl)
    assert_cards_allowed_for_retrieval(cards, path=args.cards_jsonl)
    report, arrays = build_embedding_bridge_report(
        cards=cards,
        support_report=_load_json(args.support_report),
        support_arrays_path=Path(args.support_arrays),
        output_dir=output_dir,
        selector_mode=str(args.selector_mode),
        embedding_backend=str(args.embedding_backend),
        embedding_model=str(args.embedding_model),
        dotenv_path=str(args.dotenv),
        embedding_batch_size=int(args.embedding_batch_size),
        embedding_retry_attempts=int(args.embedding_retry_attempts),
        embedding_retry_sleep=float(args.embedding_retry_sleep),
        hash_dim=int(args.hash_dim),
        top_k=int(args.top_k),
        temporal_gap=int(args.temporal_gap),
        max_query_windows=int(args.max_query_windows),
        text_candidate_k=int(args.text_candidate_k),
        text_weight=float(args.text_weight),
        start_weight=float(args.start_weight),
    )
    report_name = (
        "embedding_bridge_report.json"
        if str(args.selector_mode) == "episode_embedding"
        else "hybrid_embedding_start_bridge_report.json"
    )
    arrays_name = (
        "embedding_bridge_arrays.npz"
        if str(args.selector_mode) == "episode_embedding"
        else "hybrid_embedding_start_bridge_arrays.npz"
    )
    report["cards_path"] = str(args.cards_jsonl)
    report["artifact_paths"] = {
        "report": str(output_dir / report_name),
        "arrays": str(output_dir / arrays_name),
    }
    _write_json(output_dir / report_name, report)
    np.savez_compressed(output_dir / arrays_name, **arrays)
    print(
        json.dumps(
            {
                "status": "ok",
                "selector_mode": str(args.selector_mode),
                "query_count": len(report["evaluation"]["heldout_examples"]),
                "report": str(output_dir / report_name),
                "embedding_backend": str(args.embedding_backend),
                "embedding_model": str(args.embedding_model),
                "embedding_cache_hit": bool(
                    report["embedding_metadata"].get("cache_hit", False)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
