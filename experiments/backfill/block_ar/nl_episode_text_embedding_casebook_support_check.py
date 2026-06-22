#!/usr/bin/env python
"""Six-case support review for true text-embedding narrative retrieval.

This utility is intentionally isolated from the paper/demo default. It checks
whether raw OpenAI narrative-to-narrative retrieval over the Codex-authored
episode-card corpus selects coherent historical supports for the six public
casebook narratives. Candidate embeddings are loaded from the existing cached
``nl_episode_narrative_embedding_bridge_report.py`` arrays so the script does
not re-embed the full corpus.
"""

from __future__ import annotations

import argparse
import json
import re
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
    _window_index,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    _candidate_view_rows,
    _rank_by_embeddings,
    embed_with_cache,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    assert_cards_allowed_for_retrieval,
)
from experiments.backfill.block_ar.nl_episode_text_memory_bridge_report import (  # noqa: E402
    _direction_check,
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
DEFAULT_CASEBOOK_JSON = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_qualitative_casebook_summary.json"
)
DEFAULT_PROJECTED_REVIEW = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "six_case_projected_vs_n2n_support_review_982i/"
    "six_case_projected_vs_n2n_support_review.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "six_case_true_openai_embedding_support_check_983a"
)

MARKET_ALIASES = {
    "BBB OAS": "BBB_OAS",
    "AAA OAS": "AAA_OAS",
    "CRUDE OIL": "CRUDE_OIL",
    "US2Y": "US2Y",
    "US10Y": "US10Y",
    "DXY": "DXY",
    "GOLD": "GOLD",
    "SPX": "SPX",
    "USDJPY": "USDJPY",
    "VIX": "VIX",
}
SUPPORTED_CLAIM_MARKETS = set(MARKET_ALIASES.values())
DIRECTION_TO_SIGN = {
    "up": 1,
    "higher": 1,
    "wider": 1,
    "down": -1,
    "lower": -1,
    "tighter": -1,
    "flat": 0,
    "mixed": 0,
}
MAGNITUDE_VALUE = {"low": 1, "small": 1, "medium": 2, "high": 3, "large": 3}
GROUNDING_RE = re.compile(
    r"\b("
    + "|".join(re.escape(name) for name in MARKET_ALIASES)
    + r")\s+(up|down|higher|lower|wider|tighter|flat|mixed)\s*"
    r"\((low|small|medium|high|large)\)",
    re.IGNORECASE,
)

REVIEW_VIEW_ORDER = (
    "sparse_user_query",
    "weekly_risk_monitor",
    "institutional_risk_committee_note",
    "mechanism_first",
    "risk_manager_memo",
    "full_professional",
    "technical_factor_evidence",
    "factor_list_baseline",
    "hard_negative_views",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact(text: Any, *, limit: int | None = None) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit and len(clean) > limit:
        return clean[: max(0, limit - 1)].rstrip() + "..."
    return clean


def _parse_grounded_implications(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for order, match in enumerate(GROUNDING_RE.finditer(str(text or ""))):
        market_raw, direction_raw, magnitude_raw = match.groups()
        market = MARKET_ALIASES.get(market_raw.upper().replace("_", " "), market_raw.upper())
        if market not in SUPPORTED_CLAIM_MARKETS:
            continue
        direction = direction_raw.lower()
        sign = int(DIRECTION_TO_SIGN.get(direction, 0))
        if sign == 0:
            continue
        magnitude = magnitude_raw.lower()
        claims.append(
            {
                "market": market,
                "direction": direction,
                "sign": sign,
                "magnitude": magnitude,
                "magnitude_value": int(MAGNITUDE_VALUE.get(magnitude, 1)),
                "order": int(order),
            }
        )
    claims.sort(key=lambda row: (-int(row["magnitude_value"]), int(row["order"])))
    return claims


def _case_rows(
    casebook_payload: dict[str, Any], *, max_claims: int = 0
) -> list[dict[str, Any]]:
    rows = casebook_payload.get("case_summaries", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("casebook JSON missing non-empty case_summaries")
    cases: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("narrative_text", "")).strip()
        if not text:
            continue
        claims = _parse_grounded_implications(str(row.get("grounded_implications", "")))
        if int(max_claims) > 0:
            claims = claims[: int(max_claims)]
        cases.append(
            {
                "case_name": str(row.get("case_name", "")),
                "label": str(row.get("label", "")),
                "text": text,
                "grounded_implications": str(row.get("grounded_implications", "")),
                "claims": claims,
            }
        )
    if not cases:
        raise ValueError("casebook JSON contains no cases with narrative_text")
    return cases


def _top3_90(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    weights = _softmax_weights([float(item["score"]) for item in candidates])
    pool: list[dict[str, Any]] = []
    for rank, (item, weight) in enumerate(zip(candidates, weights, strict=True), 1):
        pool.append({**item, "rank": int(rank), "base_support_weight": float(weight)})
    selected: list[dict[str, Any]] = []
    mass = 0.0
    for item in sorted(pool, key=lambda row: float(row["base_support_weight"]), reverse=True):
        if len(selected) >= 3:
            break
        selected.append(dict(item))
        mass += float(item["base_support_weight"])
        if mass >= 0.90:
            break
    denom = sum(float(item["base_support_weight"]) for item in selected)
    if denom <= 0:
        denom = float(max(1, len(selected)))
        for item in selected:
            item["posterior_weight"] = 1.0 / denom
    else:
        for item in selected:
            item["posterior_weight"] = float(item["base_support_weight"]) / denom
    selected.sort(key=lambda row: int(row["rank"]))
    return selected


def _rank_by_embeddings_with_grounding(
    *,
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_rows: list[dict[str, Any]],
    card_by_index: dict[int, dict[str, Any]],
    claims: list[dict[str, Any]],
    top_k: int,
    temporal_gap: int,
    require_grounding_pass: bool,
) -> list[dict[str, Any]]:
    scores = np.asarray(candidate_vectors @ query_vector, dtype=np.float32)
    order = np.argsort(-scores)
    selected: list[dict[str, Any]] = []
    seen_windows: set[int] = set()
    for pos in order:
        row = candidate_rows[int(pos)]
        idx = int(row["window_index"])
        if idx in seen_windows:
            continue
        seen_windows.add(idx)
        if (
            int(temporal_gap) > 0
            and any(abs(idx - int(item["window_index"])) < int(temporal_gap) for item in selected)
        ):
            continue
        direction_check = _direction_check(
            query_claims=claims,
            candidate_card=card_by_index.get(idx, {}),
            max_mismatches=0,
        )
        if bool(require_grounding_pass) and str(direction_check.get("status", "")) != "pass":
            continue
        selected.append(
            {
                "window_index": idx,
                "window_id": str(row["window_id"]),
                "scenario_title": str(row.get("scenario_title", "")),
                "score": float(scores[int(pos)]),
                "precheck_direction_check": direction_check,
                "score_components": {
                    "method": "semantic_embedding_grounding_filtered"
                    if require_grounding_pass
                    else "semantic_embedding",
                    "embedding_score": float(scores[int(pos)]),
                    "view": str(row.get("view", "")),
                },
            }
        )
        if len(selected) >= int(top_k):
            break
    return selected


def _card_summary(
    *,
    item: dict[str, Any],
    card: dict[str, Any],
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    direction_check = _direction_check(
        query_claims=claims,
        candidate_card=card,
        max_mismatches=0,
    )
    views = card.get("views", {})
    full = ""
    if isinstance(views, dict):
        full = str(
            views.get("risk_manager_memo")
            or views.get("full_professional")
            or views.get("mechanism_first")
            or ""
        )
    return {
        "rank": int(item.get("rank", 0)),
        "window_index": int(item["window_index"]),
        "window_id": str(item["window_id"]),
        "title": str(card.get("scenario_title", item.get("scenario_title", ""))),
        "archetype": str(card.get("archetype", "")),
        "confidence": str(card.get("archetype_confidence", "")),
        "score": float(item.get("score", item.get("retrieval_score", 0.0))),
        "view": str(item.get("score_components", {}).get("view", "")),
        "base_support_weight": float(item.get("base_support_weight", 0.0)),
        "posterior_weight": float(item.get("posterior_weight", item.get("weight", 0.0))),
        "direction_check": direction_check,
        "fast_read": _compact(full, limit=260),
    }


def _support_narrative_matrix(card: dict[str, Any]) -> dict[str, Any]:
    views = card.get("views", {})
    if not isinstance(views, dict):
        return {}
    out: dict[str, Any] = {}
    for view_name in REVIEW_VIEW_ORDER:
        value = views.get(view_name)
        if isinstance(value, list):
            out[view_name] = [_compact(item) for item in value]
        elif isinstance(value, str) and value.strip():
            out[view_name] = _compact(value)
    fields = card.get("codex_multiformat_fields", {})
    if isinstance(fields, dict):
        for field_name in ("ambiguity_flags", "evidence_used", "no_forecast_caveat"):
            value = fields.get(field_name)
            if value and field_name not in out:
                out[field_name] = value
    return out


def _index_projected_review(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", [])
    indexed: dict[str, dict[str, Any]] = {}
    if isinstance(queries, list):
        for row in queries:
            if isinstance(row, dict):
                indexed[str(row.get("case_name", ""))] = row
    return indexed


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# True OpenAI Text-Embedding Support Review")
    lines.append("")
    lines.append(
        "Purpose: compare raw `text-embedding-3-large` narrative-to-narrative "
        "support retrieval against the existing projected-memory plus grounding "
        "selector for the six public casebook narratives."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Case | OpenAI N2N pass | OpenAI high-conf | Projected pass | Projected high-conf | Initial read |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in report["summary"]["cases"]:
        lines.append(
            "| {label} | {openai_pass}/3 | {openai_high}/3 | {proj_pass}/3 | "
            "{proj_high}/3 | {read} |".format(
                label=row["label"],
                openai_pass=row["openai_direction_pass_count"],
                openai_high=row["openai_high_confidence_count"],
                proj_pass=row["projected_direction_pass_count"],
                proj_high=row["projected_high_confidence_count"],
                read=row["initial_read"],
            )
        )
    lines.append("")
    lines.append("## Case Details")
    for case in report["queries"]:
        lines.append("")
        lines.append(f"### {case['label']}")
        lines.append("")
        lines.append(f"Grounded implications: {case['grounded_implications']}")
        lines.append("")
        lines.append("#### True OpenAI N2N Top3/90")
        lines.append("")
        lines.append("| Rank | Support | Title | Archetype | Confidence | Pass | View | Score |")
        lines.append("|---:|---|---|---|---|---|---|---:|")
        for item in case["openai_top3_90"]:
            lines.append(
                "| {rank} | {wid} | {title} | {arch} | {conf} | {status} | {view} | {score:.4f} |".format(
                    rank=item["rank"],
                    wid=item["window_id"],
                    title=item["title"],
                    arch=item["archetype"],
                    conf=item["confidence"],
                    status=item["direction_check"]["status"],
                    view=item["view"],
                    score=item["score"],
                )
            )
        lines.append("")
        lines.append("#### Projected-Memory + Grounding Top3/90")
        lines.append("")
        lines.append("| Rank | Support | Title | Archetype | Confidence | Pass | Score |")
        lines.append("|---:|---|---|---|---|---|---:|")
        for item in case["projected_top3_90"]:
            lines.append(
                "| {rank} | {wid} | {title} | {arch} | {conf} | {status} | {score:.4f} |".format(
                    rank=item["rank"],
                    wid=item["window_id"],
                    title=item["title"],
                    arch=item["archetype"],
                    conf=item["confidence"],
                    status=item["direction_check"]["status"],
                    score=item["score"],
                )
            )
        lines.append("")
        lines.append("#### OpenAI N2N Narrative Matrix")
        for item in case["openai_top3_90"]:
            lines.append("")
            lines.append(f"##### {item['rank']}. {item['window_id']}: {item['title']}")
            matrix = case["support_narratives"].get(item["window_id"], {})
            for name in REVIEW_VIEW_ORDER:
                value = matrix.get(name)
                if value is None:
                    continue
                lines.append("")
                lines.append(f"**{name.replace('_', ' ').title()}**")
                if isinstance(value, list):
                    for sub in value:
                        lines.append(f"- {sub}")
                else:
                    lines.append(str(value))
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cards = _read_jsonl(Path(args.cards_jsonl))
    assert_cards_allowed_for_retrieval(cards, path=Path(args.cards_jsonl))
    support_report = _load_json(Path(args.support_report))
    train_indices, _ = _split_indices_from_support_report(support_report)
    card_by_index = {_window_index(card): card for card in cards}
    train_cards = [card_by_index[idx] for idx in train_indices if idx in card_by_index]
    candidate_rows, candidate_texts = _candidate_view_rows(
        train_cards, tuple(DEFAULT_VIEW_NAMES)
    )

    with np.load(Path(args.embedding_arrays)) as payload:
        text_embeddings = payload["text_embeddings"].astype(np.float32)
        candidate_count = int(payload["candidate_text_count"][0])
    if candidate_count != len(candidate_rows):
        raise ValueError(
            f"candidate embedding count mismatch: arrays={candidate_count}, "
            f"rebuilt_rows={len(candidate_rows)}"
        )
    if candidate_count != len(candidate_texts):
        raise ValueError(
            f"candidate text count mismatch: arrays={candidate_count}, "
            f"rebuilt_texts={len(candidate_texts)}"
        )
    candidate_vectors = text_embeddings[:candidate_count]

    cases = _case_rows(
        _load_json(Path(args.casebook_json)),
        max_claims=int(args.max_grounding_claims),
    )
    query_embeddings, query_embedding_meta = embed_with_cache(
        [case["text"] for case in cases],
        output_dir=Path(args.output_dir) / "query_embeddings",
        backend="openai",
        model=str(args.embedding_model),
        dotenv_path=str(args.dotenv),
        batch_size=int(args.embedding_batch_size),
        hash_dim=512,
        retry_attempts=int(args.embedding_retry_attempts),
        retry_sleep=float(args.embedding_retry_sleep),
    )
    projected_by_case = _index_projected_review(Path(args.projected_review))

    query_reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for case, query_vector in zip(cases, query_embeddings, strict=True):
        if bool(args.require_grounding_pass):
            ranked = _rank_by_embeddings_with_grounding(
                query_vector=query_vector,
                candidate_vectors=candidate_vectors,
                candidate_rows=candidate_rows,
                card_by_index=card_by_index,
                claims=list(case["claims"]),
                top_k=int(args.support_pool_size),
                temporal_gap=int(args.temporal_gap),
                require_grounding_pass=True,
            )
        else:
            ranked = _rank_by_embeddings(
                query_vector=query_vector,
                candidate_vectors=candidate_vectors,
                candidate_rows=candidate_rows,
                top_k=int(args.support_pool_size),
                temporal_gap=int(args.temporal_gap),
            )
        selected = _top3_90(ranked)
        openai_top3 = [
            _card_summary(
                item=item,
                card=card_by_index[int(item["window_index"])],
                claims=list(case["claims"]),
            )
            for item in selected
        ]
        projected_row = projected_by_case.get(str(case["case_name"]), {})
        projected_top3_raw = projected_row.get("projected_top3_90", [])
        projected_top3: list[dict[str, Any]] = []
        if isinstance(projected_top3_raw, list):
            for item in projected_top3_raw[:3]:
                if not isinstance(item, dict):
                    continue
                card = card_by_index.get(int(item.get("window_index", -1)), {})
                projected_top3.append(
                    {
                        "rank": int(item.get("rank", len(projected_top3) + 1)),
                        "window_index": int(item.get("window_index", -1)),
                        "window_id": str(item.get("window_id", "")),
                        "title": str(item.get("title") or card.get("scenario_title", "")),
                        "archetype": str(item.get("archetype") or card.get("archetype", "")),
                        "confidence": str(
                            item.get("confidence") or card.get("archetype_confidence", "")
                        ),
                        "score": float(item.get("score", item.get("memory_score", 0.0))),
                        "direction_check": dict(item.get("direction_check", {})),
                    }
                )
        support_narratives = {
            item["window_id"]: _support_narrative_matrix(
                card_by_index[int(item["window_index"])]
            )
            for item in openai_top3
        }
        openai_pass = sum(
            1
            for item in openai_top3
            if str(item["direction_check"].get("status", "")) == "pass"
        )
        projected_pass = sum(
            1
            for item in projected_top3
            if str(item["direction_check"].get("status", "")) == "pass"
        )
        openai_high = sum(1 for item in openai_top3 if item["confidence"] == "high")
        projected_high = sum(1 for item in projected_top3 if item["confidence"] == "high")
        read = (
            "projected-memory more coherent"
            if projected_pass > openai_pass or projected_high > openai_high
            else "raw OpenAI N2N competitive"
            if projected_pass == openai_pass
            else "raw OpenAI N2N stronger"
        )
        summary_rows.append(
            {
                "case_name": case["case_name"],
                "label": case["label"],
                "openai_direction_pass_count": int(openai_pass),
                "openai_high_confidence_count": int(openai_high),
                "projected_direction_pass_count": int(projected_pass),
                "projected_high_confidence_count": int(projected_high),
                "initial_read": read,
                "openai_titles": [item["title"] for item in openai_top3],
                "projected_titles": [item["title"] for item in projected_top3],
            }
        )
        query_reports.append(
            {
                **case,
                "openai_top3_90": openai_top3,
                "projected_top3_90": projected_top3,
                "support_narratives": support_narratives,
            }
        )
    return {
        "schema_version": "nl_episode_text_embedding_casebook_support_check_v1",
        "status": "ok",
        "scope_note": (
            "Isolated support-coherence review for the two-stage text-space "
            "retrieval branch. It reuses existing Codex-authored narratives and "
            "cached OpenAI candidate embeddings; no narratives are generated."
        ),
        "inputs": {
            "cards_jsonl": str(args.cards_jsonl),
            "support_report": str(args.support_report),
            "embedding_arrays": str(args.embedding_arrays),
            "casebook_json": str(args.casebook_json),
            "projected_review": str(args.projected_review),
        },
        "embedding_meta": {
            **query_embedding_meta,
            "candidate_embedding_count": int(candidate_count),
            "candidate_view_names": list(DEFAULT_VIEW_NAMES),
        },
        "retrieval_config": {
            "method": "raw_text_embedding_3_large_grounding_filtered_top3_90"
            if bool(args.require_grounding_pass)
            else "raw_text_embedding_3_large_narrative_to_narrative_top3_90",
            "support_pool_size": int(args.support_pool_size),
            "temporal_gap": int(args.temporal_gap),
            "query_source": "paper_casebook_narrative_text",
            "require_grounding_pass": bool(args.require_grounding_pass),
            "max_grounding_claims": int(args.max_grounding_claims),
        },
        "queries": query_reports,
        "summary": {
            "case_count": int(len(query_reports)),
            "cases": summary_rows,
            "openai_total_direction_pass": int(
                sum(row["openai_direction_pass_count"] for row in summary_rows)
            ),
            "projected_total_direction_pass": int(
                sum(row["projected_direction_pass_count"] for row in summary_rows)
            ),
            "openai_total_high_confidence": int(
                sum(row["openai_high_confidence_count"] for row in summary_rows)
            ),
            "projected_total_high_confidence": int(
                sum(row["projected_high_confidence_count"] for row in summary_rows)
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--embedding-arrays", type=Path, default=DEFAULT_EMBEDDING_ARRAYS)
    parser.add_argument("--casebook-json", type=Path, default=DEFAULT_CASEBOOK_JSON)
    parser.add_argument("--projected-review", type=Path, default=DEFAULT_PROJECTED_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-retry-attempts", type=int, default=6)
    parser.add_argument("--embedding-retry-sleep", type=float, default=2.0)
    parser.add_argument("--support-pool-size", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--require-grounding-pass", action="store_true")
    parser.add_argument("--max-grounding-claims", type=int, default=0)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    report_path = output_dir / "true_openai_embedding_casebook_support_check.json"
    markdown_path = output_dir / "true_openai_embedding_casebook_support_check.md"
    report["artifact_paths"] = {
        "report": str(report_path),
        "markdown": str(markdown_path),
    }
    _write_json(report_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "case_count": int(report["summary"]["case_count"]),
                "openai_total_direction_pass": int(
                    report["summary"]["openai_total_direction_pass"]
                ),
                "projected_total_direction_pass": int(
                    report["summary"]["projected_total_direction_pass"]
                ),
                "report": str(report_path),
                "markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
