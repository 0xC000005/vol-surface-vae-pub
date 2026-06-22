#!/usr/bin/env python
"""Local narrative-to-narrative retrieval over episode cards.

This is an isolated Phase 1 utility for the episode-level narrative retrieval
branch. It ranks historical episode cards by matching a user/professional
narrative against multi-view episode descriptions produced by
``nl_episode_narrative_cards.py``. It does not modify the incumbent
narrative-grounded scenario pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_cards import (
    MECHANISM_TERMS,
)


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_retrieval_phase0_cards_970a/episode_narrative_cards.jsonl"
)
DEFAULT_QUERY_JSON = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_qualitative_casebook_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_retrieval_phase1_local"
)
DEFAULT_VIEW_NAMES = (
    "sparse_user_query",
    "weekly_risk_monitor",
    "risk_manager_memo",
    "fed_current_conditions",
    "institutional_risk_committee_note",
    "macro_outlook_newsletter",
    "technical_factor_evidence",
    "mechanism_first",
    "full_professional",
    "factor_list_baseline",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "rather",
    "that",
    "the",
    "this",
    "to",
    "toward",
    "under",
    "while",
    "with",
}
SYNONYMS = {
    "equities": "equity",
    "stocks": "equity",
    "shares": "equity",
    "spx": "equity",
    "vix": "volatility",
    "vol": "volatility",
    "volatility": "volatility",
    "oas": "spread",
    "spreads": "spread",
    "wider": "widen",
    "widening": "widen",
    "tightened": "tighten",
    "tighter": "tighten",
    "tightening": "tighten",
    "de-risking": "derisk",
    "de-risk": "derisk",
    "derisking": "derisk",
    "defensive": "defense",
    "hedges": "hedge",
    "hedging": "hedge",
    "duration": "treasury",
    "treasuries": "treasury",
    "yields": "yield",
    "rates": "rate",
    "liquidity": "liquidity",
    "funding": "funding",
    "dollar": "dollar",
    "dxy": "dollar",
    "gold": "gold",
    "safe-haven": "safehaven",
    "safe": "safe",
    "haven": "haven",
    "oil": "oil",
    "crude": "oil",
    "commodities": "commodity",
}


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normal_token(token: str) -> str:
    token = token.lower().strip("_-")
    token = SYNONYMS.get(token, token)
    if token.endswith("ies") and len(token) > 5:
        token = token[:-3] + "y"
    elif token.endswith("ing") and len(token) > 6:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 5:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return SYNONYMS.get(token, token)


def _tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    tokens = [_normal_token(token) for token in raw_tokens]
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _lexical_score(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(right_tokens)
    recall = overlap / len(left_tokens)
    jaccard = overlap / len(left_tokens | right_tokens)
    return 0.45 * recall + 0.35 * jaccard + 0.20 * precision


def _hashed_vector(tokens: list[str], *, dims: int = 256) -> dict[int, float]:
    vector: dict[int, float] = {}
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dims
        sign = 1.0 if ((value >> 8) & 1) else -1.0
        vector[index] = vector.get(index, 0.0) + sign
    return vector


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(index, 0.0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))


def _dense_score(left: str, right: str) -> float:
    left_vec = _hashed_vector(_tokens(left))
    right_vec = _hashed_vector(_tokens(right))
    return _cosine(left_vec, right_vec)


def _mechanism_tokens(text: str) -> set[str]:
    found: set[str] = set()
    for term in MECHANISM_TERMS:
        if re.search(
            r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b", text, re.I
        ):
            found.add(_normal_token(term.replace(" ", "_")))
    token_set = _token_set(text)
    found.update(
        token
        for token in token_set
        if token
        in {
            "liquidity",
            "funding",
            "inflation",
            "policy",
            "growth",
            "derisk",
            "safehaven",
            "carry",
            "hedge",
        }
    )
    return found


def _mechanism_score(left: str, right: str) -> float:
    left_tokens = _mechanism_tokens(left)
    right_tokens = _mechanism_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _text_features(text: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = _compact(text)
    if normalized not in cache:
        tokens = _tokens(normalized)
        cache[normalized] = {
            "token_set": set(tokens),
            "vector": _hashed_vector(tokens),
            "mechanism_tokens": _mechanism_tokens(normalized),
        }
    return cache[normalized]


def _score_feature_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    method: str,
) -> dict[str, float]:
    left_tokens = left["token_set"]
    right_tokens = right["token_set"]
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens)
        precision = overlap / len(right_tokens)
        recall = overlap / len(left_tokens)
        jaccard = overlap / len(left_tokens | right_tokens)
        lexical = 0.45 * recall + 0.35 * jaccard + 0.20 * precision
    else:
        lexical = 0.0
    dense = _cosine(left["vector"], right["vector"])
    left_mechanism = left["mechanism_tokens"]
    right_mechanism = right["mechanism_tokens"]
    mechanism = (
        len(left_mechanism & right_mechanism) / len(left_mechanism | right_mechanism)
        if left_mechanism and right_mechanism
        else 0.0
    )
    if method == "lexical":
        score = lexical
    elif method == "dense":
        score = dense
    elif method == "hybrid":
        score = 0.35 * lexical + 0.45 * dense + 0.20 * mechanism
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    return {
        "score": float(score),
        "lexical": float(lexical),
        "dense": float(dense),
        "mechanism": float(mechanism),
    }


def build_query(
    *,
    label: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a retrieval query object from user/professional narrative text."""

    return {
        "schema_version": "nl_episode_narrative_query_v1",
        "label": _compact(label),
        "text": _compact(text),
        "metadata": dict(metadata or {}),
    }


def _score_text_pair(
    query_text: str, candidate_text: str, *, method: str
) -> dict[str, float]:
    lexical = _lexical_score(query_text, candidate_text)
    dense = _dense_score(query_text, candidate_text)
    mechanism = _mechanism_score(query_text, candidate_text)
    if method == "lexical":
        score = lexical
    elif method == "dense":
        score = dense
    elif method == "hybrid":
        score = 0.35 * lexical + 0.45 * dense + 0.20 * mechanism
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    return {
        "score": float(score),
        "lexical": float(lexical),
        "dense": float(dense),
        "mechanism": float(mechanism),
    }


def _card_window_index(card: dict[str, Any]) -> int | None:
    metadata = card.get("support_metadata", {})
    if isinstance(metadata, dict) and metadata.get("window_index") is not None:
        return int(metadata["window_index"])
    if card.get("window_index") is not None:
        return int(card["window_index"])
    match = re.search(r"_(\d+)$", str(card.get("window_id", "")))
    return int(match.group(1)) if match else None


def rank_episode_cards(
    query: dict[str, Any],
    cards: list[dict[str, Any]],
    *,
    method: str = "hybrid",
    top_k: int = 10,
    view_names: tuple[str, ...] | list[str] | None = None,
    feature_cache: dict[str, dict[str, Any]] | None = None,
    temporal_gap: int = 0,
) -> list[dict[str, Any]]:
    """Rank episode cards by the best matching narrative view."""

    selected_views = tuple(view_names or DEFAULT_VIEW_NAMES)
    query_text = _compact(query.get("text", ""))
    cache = feature_cache if feature_cache is not None else {}
    query_features = _text_features(query_text, cache)
    ranked: list[dict[str, Any]] = []

    for card in cards:
        best: dict[str, Any] | None = None
        views = card.get("views", {})
        for view_name in selected_views:
            candidate_text = views.get(view_name, "")
            if not isinstance(candidate_text, str) or not candidate_text.strip():
                continue
            components = _score_feature_pair(
                query_features,
                _text_features(candidate_text, cache),
                method=method,
            )
            candidate = {
                "window_id": card.get("window_id", ""),
                "window_index": _card_window_index(card),
                "split": card.get("split", ""),
                "scenario_title": card.get("scenario_title", ""),
                "archetype": card.get("archetype", ""),
                "score": components["score"],
                "score_components": {
                    "lexical": components["lexical"],
                    "dense": components["dense"],
                    "mechanism": components["mechanism"],
                    "view": view_name,
                    "method": method,
                },
                "view_text": candidate_text,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        if best is not None:
            ranked.append(best)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    gap = max(0, int(temporal_gap))
    if gap <= 0:
        return ranked[: max(0, top_k)]
    selected: list[dict[str, Any]] = []
    for item in ranked:
        idx = item.get("window_index")
        if idx is not None and any(
            abs(int(idx) - int(row["window_index"])) < gap
            for row in selected
            if row.get("window_index") is not None
        ):
            continue
        selected.append(item)
        if len(selected) >= int(top_k):
            break
    return selected


def pairwise_jaccard_summary(result_sets: dict[str, list[str]]) -> dict[str, Any]:
    """Summarize support overlap between query result sets."""

    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(result_sets), 2):
        left_set = set(result_sets[left])
        right_set = set(result_sets[right])
        union = left_set | right_set
        jaccard = len(left_set & right_set) / len(union) if union else 0.0
        pairs.append(
            {
                "left": left,
                "right": right,
                "intersection": len(left_set & right_set),
                "union": len(union),
                "jaccard": float(jaccard),
            }
        )
    values = [item["jaccard"] for item in pairs]
    return {
        "pair_count": len(pairs),
        "min_jaccard": min(values) if values else 0.0,
        "max_jaccard": max(values) if values else 0.0,
        "mean_jaccard": sum(values) / len(values) if values else 0.0,
        "pairs": pairs,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def assert_cards_allowed_for_retrieval(cards: list[dict[str, Any]], *, path: Path) -> None:
    """Reject known invalid local/template narrative card artifacts."""

    path_text = str(path)
    if "episode_card_v3_full_regen_981" in path_text:
        raise ValueError(
            f"{path}: local deterministic EpisodeCardV3 full-regeneration "
            "artifacts are invalid for retrieval/training. Regenerate direct "
            "Codex/GPT-authored multi-format cards first."
        )
    invalid_rows = [
        str(card.get("window_id", f"row_{idx}"))
        for idx, card in enumerate(cards)
        if card.get("schema_version") == "nl_episode_card_v3"
        or card.get("valid_for_training_retrieval") is False
        or str(card.get("narrative_authoring", "")).startswith(
            ("local_deterministic", "single_caption_local_view_projection")
        )
    ]
    if invalid_rows:
        preview = ", ".join(invalid_rows[:5])
        raise ValueError(
            "cards contain narrative views that are not valid for retrieval/"
            f"training ({preview}). Searchable narrative views must be directly "
            "Codex/GPT-authored; local deterministic/template views are banned."
        )


def _load_casebook_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("case_summaries", [])
    queries: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        label = case.get("label") or case.get("case_name") or f"case_{idx:02d}"
        text = case.get("narrative_text") or case.get("clean_conditioning_text") or ""
        clean_text = case.get("clean_conditioning_text", "")
        combined = _compact(f"{text} {clean_text}")
        if not combined:
            continue
        queries.append(
            build_query(
                label=label,
                text=combined,
                metadata={
                    "case_name": case.get("case_name", ""),
                    "source_path": str(path),
                    "query_index": idx,
                },
            )
        )
    return queries


def _method_report(
    *,
    method: str,
    queries: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    top_k: int,
    feature_cache: dict[str, dict[str, Any]],
    temporal_gap: int,
) -> dict[str, Any]:
    ranked_by_query: dict[str, list[dict[str, Any]]] = {}
    result_sets: dict[str, list[str]] = {}
    view_wins: dict[str, int] = {}

    for query in queries:
        ranked = rank_episode_cards(
            query,
            cards,
            method=method,
            top_k=top_k,
            feature_cache=feature_cache,
            temporal_gap=int(temporal_gap),
        )
        label = str(query["label"])
        ranked_by_query[label] = ranked
        result_sets[label] = [str(item["window_id"]) for item in ranked]
        for item in ranked:
            view_name = str(item["score_components"]["view"])
            view_wins[view_name] = view_wins.get(view_name, 0) + 1

    top_scores = [ranked[0]["score"] for ranked in ranked_by_query.values() if ranked]
    return {
        "method": method,
        "query_count": len(queries),
        "top_k": top_k,
        "temporal_gap": int(temporal_gap),
        "mean_top1_score": sum(top_scores) / len(top_scores) if top_scores else 0.0,
        "min_top1_score": min(top_scores) if top_scores else 0.0,
        "max_top1_score": max(top_scores) if top_scores else 0.0,
        "view_wins": dict(sorted(view_wins.items())),
        "pairwise_overlap": pairwise_jaccard_summary(result_sets),
        "queries": ranked_by_query,
    }


def build_retrieval_report(
    *,
    cards_path: Path,
    query_path: Path,
    methods: list[str],
    top_k: int,
    temporal_gap: int = 0,
) -> dict[str, Any]:
    cards = _read_jsonl(cards_path)
    assert_cards_allowed_for_retrieval(cards, path=cards_path)
    queries = _load_casebook_queries(query_path)
    feature_cache: dict[str, dict[str, Any]] = {}
    method_reports = [
        _method_report(
            method=method,
            queries=queries,
            cards=cards,
            top_k=top_k,
            feature_cache=feature_cache,
            temporal_gap=int(temporal_gap),
        )
        for method in methods
    ]
    return {
        "schema_version": "nl_episode_narrative_local_retrieval_report_v1",
        "cards_path": str(cards_path),
        "query_path": str(query_path),
        "card_count": len(cards),
        "query_count": len(queries),
        "temporal_gap": int(temporal_gap),
        "methods": method_reports,
        "interpretation": {
            "status": "local_retrieval_screen",
            "note": (
                "This screen tests text-to-text episode retrieval only. Scenario "
                "rollout, CRPS/energy backtests, and promotion gates are Phase 2+."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--query-json", type=Path, default=DEFAULT_QUERY_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--temporal-gap",
        type=int,
        default=0,
        help="Minimum absolute support window-index gap within each query result set.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lexical", "dense", "hybrid"],
        choices=["lexical", "dense", "hybrid"],
    )
    args = parser.parse_args(argv)

    report = build_retrieval_report(
        cards_path=args.cards_jsonl,
        query_path=args.query_json,
        methods=list(args.methods),
        top_k=args.top_k,
        temporal_gap=int(args.temporal_gap),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "local_retrieval_report.json"
    report["artifact_paths"] = {"report": str(report_path)}
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "card_count": report["card_count"],
                "query_count": report["query_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
