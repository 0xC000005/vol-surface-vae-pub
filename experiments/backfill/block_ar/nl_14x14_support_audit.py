#!/usr/bin/env python
"""Emit support audits and bridge reports from the 14+14 retrieval arrays.

This utility is intentionally post-training only. It does not call OpenAI and
does not generate narrative text. It consumes the saved 14+14 manifest
embeddings/condition vectors, produces support-level audit tables, and writes
bridge-report-shaped rows that the frozen SNI scenario evaluator can consume.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    _load_json,
    _softmax_weights,
    _split_indices_from_support_report,
    _temporal_gap_filter,
)
DEFAULT_EXAMPLES_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_self_supervised_training_manifest_990a/training_examples.jsonl"
)
DEFAULT_TRAINING_RUN_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_retrieval_training_openai_full_990e/training_run_report.json"
)
DEFAULT_TEXT_SPACE_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_retrieval_training_openai_full_990e/text_space/"
    "text_space_training_arrays.npz"
)
DEFAULT_PROJECTED_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_retrieval_training_openai_full_990e/projected_memory/"
    "projected_memory_training_arrays.npz"
)
DEFAULT_SUPPORT_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_support_audit_990f"
)
DEFAULT_BASELINE_BRIDGE_REPORTS = {
    "baseline_start_only_981d_top3": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_start_only_bridge_981d_gap30_full66/"
        "start_only_bridge_report.json"
    ),
    "baseline_projected_memory_981t_top3": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_projected_memory_bridge_981t_adapter_gap30_full66/"
        "projected_memory_bridge_report.json"
    ),
    "baseline_raw_openai_grounded_982g_top3_90": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_codex_982g_embedding_grounded_top3_90_66q/"
        "embedding_grounded_bridge_report.json"
    ),
    "current_episode_grounded_text_preference_984a_top3_90": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_codex_982g_grounded_text_preference_reranker_984a/"
        "grounded_text_preference_bridge_report.json"
    ),
}
PUBLIC_PAPER_DEMO_POSTERIOR_ARTIFACTS = {
    "posterior_selection": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_962c_selection_confirmation/"
        "posterior_ensemble_selection_report.json"
    ),
    "multistart_confirmation": Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "posterior_ensemble_candidate_963c_multistart_confirmation/"
        "posterior_ensemble_multistart_confirmation.json"
    ),
}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = _resolve(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_no}: expected JSON object")
            row = dict(row)
            row["embedding_index"] = int(len(rows))
            row["label_window_index"] = int(row["label_window_index"])
            row["target_window_index"] = int(row["target_window_index"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{input_path}: no JSONL rows")
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    input_path = _resolve(path)
    with np.load(input_path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _window_id(index: int) -> str:
    return f"joint39_train_{int(index):04d}"


def _metadata_by_index(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for row in report.get("window_metadata", []):
        if isinstance(row, dict) and row.get("window_index") is not None:
            metadata[int(row["window_index"])] = dict(row)
    return metadata


def _safe_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _query_examples(
    examples: list[dict[str, Any]],
    *,
    query_window_indices: list[int],
    query_view: str,
) -> list[dict[str, Any]]:
    allowed = {int(idx) for idx in query_window_indices}
    rows = [
        row
        for row in examples
        if str(row.get("role", "")) == "positive"
        and str(row.get("view_name", "")) == str(query_view)
        and int(row.get("target_window_index", -1)) in allowed
    ]
    rows.sort(key=lambda row: int(row["target_window_index"]))
    missing = sorted(allowed.difference(int(row["target_window_index"]) for row in rows))
    if missing:
        raise ValueError(
            "missing positive query examples for windows: "
            + ", ".join(str(idx) for idx in missing[:10])
        )
    return rows


def _default_query_window_indices(
    examples: list[dict[str, Any]],
    support_report: dict[str, Any],
    *,
    query_view: str,
) -> list[int]:
    _train_indices, test_indices = _split_indices_from_support_report(support_report)
    available = {
        int(row["target_window_index"])
        for row in examples
        if str(row.get("role", "")) == "positive"
        and str(row.get("view_name", "")) == str(query_view)
    }
    return sorted(int(idx) for idx in test_indices if int(idx) in available)


def _candidate_positions(
    examples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    vector_count: int,
    roles: tuple[str, ...],
) -> list[int]:
    train_set = {int(idx) for idx in train_indices}
    allowed_roles = {str(role) for role in roles}
    positions: list[int] = []
    for row in examples:
        embedding_index = int(row["embedding_index"])
        if embedding_index >= int(vector_count):
            continue
        if str(row.get("role", "")) not in allowed_roles:
            continue
        if int(row.get("label_window_index", -1)) not in train_set:
            continue
        positions.append(embedding_index)
    if not positions:
        raise ValueError("no candidate examples remain after train/role filtering")
    return positions


def _support_item(
    *,
    window_index: int,
    score: float,
    method: str,
    metadata_by_index: dict[int, dict[str, Any]],
    source_role: str | None = None,
    source_view: str | None = None,
    source_example_id: str | None = None,
) -> dict[str, Any]:
    metadata = metadata_by_index.get(int(window_index), {})
    components: dict[str, Any] = {
        "method": str(method),
        "score": float(score),
    }
    if source_role is not None:
        components["source_role"] = str(source_role)
    if source_view is not None:
        components["source_view"] = str(source_view)
    if source_example_id is not None:
        components["source_example_id"] = str(source_example_id)
    return {
        "window_index": int(window_index),
        "window_id": str(metadata.get("window_id", _window_id(window_index))),
        "scenario_title": str(metadata.get("scenario_title", "")),
        "score": float(score),
        "score_components": components,
    }


def _rank_by_example_vectors(
    *,
    query_vector: np.ndarray,
    vectors: np.ndarray,
    examples: list[dict[str, Any]],
    candidate_positions: list[int],
    metadata_by_index: dict[int, dict[str, Any]],
    method: str,
    support_pool_size: int,
    temporal_gap: int,
    query_index: int,
) -> list[dict[str, Any]]:
    values = _safe_normalize(vectors)
    query = _safe_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    positions = np.asarray(candidate_positions, dtype=np.int64)
    scores = values[positions] @ query
    order = np.argsort(-scores)
    best_by_label: dict[int, dict[str, Any]] = {}
    for local_position in order:
        example_position = int(positions[int(local_position)])
        row = examples[example_position]
        label = int(row["label_window_index"])
        # Causal gap: the candidate window spans [label, label+horizon]; require it to end
        # at/before the query window's start, i.e. query_index - label >= temporal_gap. Also
        # excludes candidates at/after the query. Applied BEFORE the support_pool_size*8 break
        # so invalid high-scorers don't crowd out valid candidates. Distinct from the
        # mutual-diversity _temporal_gap_filter applied afterward.
        if int(query_index) - label < int(temporal_gap):
            continue
        if label in best_by_label:
            continue
        score = float(scores[int(local_position)])
        best_by_label[label] = _support_item(
            window_index=label,
            score=score,
            method=method,
            metadata_by_index=metadata_by_index,
            source_role=str(row.get("role", "")),
            source_view=str(row.get("view_name", "")),
            source_example_id=str(row.get("example_id", "")),
        )
        if len(best_by_label) >= max(int(support_pool_size) * 8, int(support_pool_size)):
            break
    ranked = sorted(best_by_label.values(), key=lambda item: item["score"], reverse=True)
    return _temporal_gap_filter(
        ranked, top_k=int(support_pool_size), temporal_gap=int(temporal_gap)
    )


def _rank_projected_memory(
    *,
    query_vector: np.ndarray,
    memory_targets: np.ndarray,
    train_indices: list[int],
    query_index: int,
    metadata_by_index: dict[int, dict[str, Any]],
    method: str,
    support_pool_size: int,
    temporal_gap: int,
) -> list[dict[str, Any]]:
    memory = _safe_normalize(memory_targets)
    query = _safe_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    train = np.asarray(train_indices, dtype=np.int64)
    scores = memory[train] @ query
    ranked: list[dict[str, Any]] = []
    for pos in np.argsort(-scores):
        idx = int(train[int(pos)])
        # Causal gap (subsumes idx == query_index): require query_index - idx >= temporal_gap
        # so the candidate window ends at/before the query start. Before the *8 early-break.
        if int(query_index) - idx < int(temporal_gap):
            continue
        ranked.append(
            _support_item(
                window_index=idx,
                score=float(scores[int(pos)]),
                method=method,
                metadata_by_index=metadata_by_index,
            )
        )
        if len(ranked) >= max(int(support_pool_size) * 8, int(support_pool_size)):
            break
    return _temporal_gap_filter(
        ranked, top_k=int(support_pool_size), temporal_gap=int(temporal_gap)
    )


def _apply_top3_90(
    candidates: list[dict[str, Any]], *, top_k: int = 3, min_mass: float = 0.90
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not candidates:
        return [], []
    weights = _softmax_weights([float(item["score"]) for item in candidates])
    candidate_pool: list[dict[str, Any]] = []
    for rank, (item, weight) in enumerate(zip(candidates, weights, strict=True), 1):
        row = copy.deepcopy(item)
        row["rank"] = int(rank)
        row["cosine"] = float(item["score"])
        row["retrieval_score"] = float(item["score"])
        row["base_support_weight"] = float(weight)
        row["weight"] = float(weight)
        candidate_pool.append(row)
    selected: list[dict[str, Any]] = []
    mass = 0.0
    for item in sorted(
        candidate_pool, key=lambda row: float(row.get("base_support_weight", 0.0)), reverse=True
    ):
        if len(selected) >= min(3, int(top_k)):
            break
        selected.append(copy.deepcopy(item))
        mass += float(item.get("base_support_weight", 0.0))
        if mass >= float(min_mass):
            break
    denom = sum(max(float(item.get("base_support_weight", 0.0)), 0.0) for item in selected)
    if denom <= 0:
        denom = float(max(len(selected), 1))
        for item in selected:
            item["weight"] = 1.0 / denom
            item["posterior_weight"] = float(item["weight"])
    else:
        for item in selected:
            item["weight"] = max(float(item.get("base_support_weight", 0.0)), 0.0) / denom
            item["posterior_weight"] = float(item["weight"])
    selected.sort(key=lambda row: int(row["rank"]))
    for rank, item in enumerate(selected, 1):
        item["rank"] = int(rank)
        item["posterior_role"] = "top3_90_selected"
    return selected, candidate_pool


def _bridge_row(
    *,
    method: str,
    query_no: int,
    query_example: dict[str, Any],
    selected: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    query_index = int(query_example["target_window_index"])
    return {
        "query_id": f"{method}_{query_no:04d}_{query_index}",
        "window_index": query_index,
        "window_id": str(query_example.get("target_window_id", _window_id(query_index))),
        "role": "anchor",
        "kind": str(method),
        "query_text_source": str(query_example.get("view_name", "")),
        "query_text": str(query_example.get("text", "")),
        "candidate_pool_size": int(len(candidate_pool)),
        "pre_top3_90_candidate_pool": candidate_pool,
        "top_train_pool": selected,
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "query_count": 0,
            "unique_support_count": 0,
            "mean_top1_score": math.nan,
            "mean_selected_support_count": math.nan,
            "mean_weight_entropy": math.nan,
        }
    unique_supports = {
        int(item["window_index"])
        for row in rows
        for item in row.get("top_train_pool", [])
    }
    top_scores = [
        float(row["top_train_pool"][0].get("cosine", 0.0))
        for row in rows
        if row.get("top_train_pool")
    ]
    support_counts = [len(row.get("top_train_pool", [])) for row in rows]
    entropies: list[float] = []
    for row in rows:
        weights = np.asarray(
            [float(item.get("weight", 0.0)) for item in row.get("top_train_pool", [])],
            dtype=np.float64,
        )
        weights = weights[weights > 0.0]
        if weights.size:
            entropies.append(float(-np.sum(weights * np.log(weights))))
    return {
        "query_count": int(len(rows)),
        "unique_support_count": int(len(unique_supports)),
        "mean_top1_score": float(np.mean(top_scores)) if top_scores else math.nan,
        "mean_selected_support_count": float(np.mean(support_counts)) if support_counts else math.nan,
        "mean_weight_entropy": float(np.mean(entropies)) if entropies else math.nan,
    }


def _support_sets(rows: list[dict[str, Any]]) -> dict[int, set[int]]:
    return {
        int(row["window_index"]): {
            int(item["window_index"]) for item in row.get("top_train_pool", [])
        }
        for row in rows
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    denom = len(left | right)
    if denom == 0:
        return 0.0
    return float(len(left & right) / denom)


def _method_overlap(method_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    names = sorted(method_rows)
    sets_by_method = {name: _support_sets(method_rows[name]) for name in names}
    rows: list[dict[str, Any]] = []
    for left_pos, left in enumerate(names):
        for right in names[left_pos + 1 :]:
            shared_queries = sorted(set(sets_by_method[left]) & set(sets_by_method[right]))
            scores = [
                _jaccard(sets_by_method[left][idx], sets_by_method[right][idx])
                for idx in shared_queries
            ]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "shared_query_count": int(len(shared_queries)),
                    "mean_top3_jaccard": float(np.mean(scores)) if scores else math.nan,
                }
            )
    return rows


def _normalize_existing_top_pool(
    pool: list[dict[str, Any]], *, method: str, top_k: int
) -> list[dict[str, Any]]:
    selected = [copy.deepcopy(item) for item in pool[: int(top_k)]]
    if not selected:
        return []
    raw = np.asarray(
        [float(item.get("weight", item.get("posterior_weight", 0.0)) or 0.0) for item in selected],
        dtype=np.float64,
    )
    if float(np.sum(np.maximum(raw, 0.0))) <= 0.0:
        raw = np.ones(len(selected), dtype=np.float64)
    weights = np.maximum(raw, 0.0)
    weights = weights / float(weights.sum())
    for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), 1):
        item["rank"] = int(rank)
        item["weight"] = float(weight)
        item["posterior_weight"] = float(weight)
        item["posterior_role"] = "top3_90_selected"
        item["cosine"] = float(item.get("cosine", item.get("retrieval_score", 0.0)) or 0.0)
        components = dict(item.get("score_components", {}))
        components["matched_eval_source_method"] = str(method)
        item["score_components"] = components
    return selected


def _baseline_rows(
    *,
    method: str,
    report_path: Path,
    query_window_indices: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []
    report = _load_json(report_path)
    allowed = {int(idx) for idx in query_window_indices}
    rows: list[dict[str, Any]] = []
    for row in report.get("evaluation", {}).get("heldout_examples", []):
        if int(row.get("window_index", -1)) not in allowed:
            continue
        cloned = copy.deepcopy(row)
        cloned["query_id"] = f"{method}_{cloned.get('query_id', cloned.get('window_index'))}"
        cloned["kind"] = str(method)
        cloned["role"] = "anchor"
        cloned["matched_source_bridge_report"] = str(report_path)
        cloned["top_train_pool"] = _normalize_existing_top_pool(
            list(cloned.get("top_train_pool", [])),
            method=method,
            top_k=int(top_k),
        )
        rows.append(cloned)
    rows.sort(key=lambda item: int(item["window_index"]))
    return rows


def _write_support_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 14+14 Support-Level Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Query windows: `{len(report['query_window_indices'])}`",
        f"- Top selection: `top-{report['top_k']}/90`",
        "",
        "## Method Summary",
        "",
        "| Method | Queries | Unique supports | Mean top1 score | Mean selected supports |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, summary in sorted(report["method_summaries"].items()):
        lines.append(
            "| {method} | {queries} | {unique} | {score:.4f} | {count:.2f} |".format(
                method=method,
                queries=summary["query_count"],
                unique=summary["unique_support_count"],
                score=float(summary["mean_top1_score"]),
                count=float(summary["mean_selected_support_count"]),
            )
        )
    lines.extend(["", "## Matched Query Support Sets", ""])
    for query in report["query_reviews"]:
        lines.extend(
            [
                f"### {query['window_id']}",
                "",
                f"Query: {query['query_text']}",
                "",
                "| Method | Selected supports |",
                "|---|---|",
            ]
        )
        for method, supports in query["supports_by_method"].items():
            support_text = ", ".join(
                f"{item['window_id']} ({item['weight']:.2f})" for item in supports
            )
            lines.append(f"| {method} | {support_text} |")
        lines.append("")
    lines.extend(
        [
            "## Existing Public Candidate Note",
            "",
            report["public_paper_demo_candidate_note"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_14x14_support_audit(
    *,
    examples_jsonl: str | Path,
    support_report_path: str | Path,
    text_space_arrays_path: str | Path,
    projected_arrays_path: str | Path,
    output_dir: str | Path,
    training_run_report_path: str | Path | None = None,
    query_window_indices: list[int] | None = None,
    query_view: str = "full_professional",
    top_k: int = 3,
    support_pool_size: int = 8,
    temporal_gap: int = 30,
    baseline_bridge_reports: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    output = _resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = _read_jsonl(examples_jsonl)
    support_report = _load_json(_resolve(support_report_path))
    train_indices, test_indices = _split_indices_from_support_report(support_report)
    metadata = _metadata_by_index(support_report)
    queries = query_window_indices or _default_query_window_indices(
        examples, support_report, query_view=str(query_view)
    )
    queries = sorted(int(idx) for idx in queries)
    if not queries:
        raise ValueError("no matched query windows found")
    query_rows = _query_examples(
        examples,
        query_window_indices=queries,
        query_view=str(query_view),
    )
    text_arrays = _load_npz(text_space_arrays_path)
    projected_arrays = _load_npz(projected_arrays_path)
    raw_vectors = np.asarray(text_arrays["embeddings"], dtype=np.float32)
    adapted_vectors = np.asarray(text_arrays["adapted_embeddings"], dtype=np.float32)
    condition_vectors = np.asarray(projected_arrays["condition_vectors"], dtype=np.float32)
    memory_targets = np.asarray(projected_arrays["memory_targets"], dtype=np.float32)
    candidate_positions = _candidate_positions(
        examples,
        train_indices=train_indices,
        vector_count=raw_vectors.shape[0],
        roles=("positive", "hard_negative"),
    )

    method_rows: dict[str, list[dict[str, Any]]] = {
        "raw_openai_14x14_top3_90": [],
        "text_space_contrastive_14x14_top3_90": [],
        "projected_memory_14x14_top3_90": [],
    }
    for query_no, query in enumerate(query_rows):
        query_embedding_index = int(query["embedding_index"])
        query_index = int(query["target_window_index"])
        raw_ranked = _rank_by_example_vectors(
            query_vector=raw_vectors[query_embedding_index],
            vectors=raw_vectors,
            examples=examples,
            candidate_positions=candidate_positions,
            metadata_by_index=metadata,
            method="raw_openai_14x14_top3_90_candidate",
            support_pool_size=int(support_pool_size),
            temporal_gap=int(temporal_gap),
            query_index=query_index,
        )
        adapted_ranked = _rank_by_example_vectors(
            query_vector=adapted_vectors[query_embedding_index],
            vectors=adapted_vectors,
            examples=examples,
            candidate_positions=candidate_positions,
            metadata_by_index=metadata,
            method="text_space_contrastive_14x14_top3_90_candidate",
            support_pool_size=int(support_pool_size),
            temporal_gap=int(temporal_gap),
            query_index=query_index,
        )
        projected_ranked = _rank_projected_memory(
            query_vector=condition_vectors[query_embedding_index],
            memory_targets=memory_targets,
            train_indices=train_indices,
            query_index=query_index,
            metadata_by_index=metadata,
            method="projected_memory_14x14_top3_90_candidate",
            support_pool_size=int(support_pool_size),
            temporal_gap=int(temporal_gap),
        )
        for method, ranked in [
            ("raw_openai_14x14_top3_90", raw_ranked),
            ("text_space_contrastive_14x14_top3_90", adapted_ranked),
            ("projected_memory_14x14_top3_90", projected_ranked),
        ]:
            selected, candidate_pool = _apply_top3_90(
                ranked, top_k=int(top_k), min_mass=0.90
            )
            for item in selected:
                components = dict(item.get("score_components", {}))
                components["method"] = method
                item["score_components"] = components
            method_rows[method].append(
                _bridge_row(
                    method=method,
                    query_no=query_no,
                    query_example=query,
                    selected=selected,
                    candidate_pool=candidate_pool,
                )
            )

    baselines: dict[str, list[dict[str, Any]]] = {}
    for name, path_value in (baseline_bridge_reports or {}).items():
        path = _resolve(path_value)
        rows = _baseline_rows(
            method=str(name),
            report_path=path,
            query_window_indices=queries,
            top_k=int(top_k),
        )
        if rows:
            baselines[str(name)] = rows
    all_method_rows = {**method_rows, **baselines}

    bridge_reports: dict[str, str] = {}
    for method, rows in method_rows.items():
        report_path = output / f"{method}_bridge_report.json"
        bridge = {
            "schema_version": "nl_14x14_support_bridge_report_v1",
            "status": "ok",
            "method": method,
            "scope_note": (
                "Bridge report emitted from saved 14+14 retrieval arrays. "
                "No OpenAI calls and no narrative generation were performed."
            ),
            "source_examples_jsonl": str(_resolve(examples_jsonl)),
            "source_training_run_report": str(_resolve(training_run_report_path))
            if training_run_report_path
            else "",
            "retrieval_config": {
                "method": method,
                "query_view": str(query_view),
                "candidate_roles": ["positive", "hard_negative"],
                "top_k": int(top_k),
                "support_pool_size": int(support_pool_size),
                "temporal_gap": int(temporal_gap),
                "top3_90": True,
            },
            "split": {
                "train_indices": [int(idx) for idx in train_indices],
                "test_indices": [int(idx) for idx in queries],
            },
            "window_indices": [
                int(idx)
                for idx in sorted(
                    set(train_indices).union(test_indices).union(queries)
                )
            ],
            "evaluation": {"heldout_examples": rows},
        }
        bridge["artifact_paths"] = {"report": str(report_path)}
        _write_json(report_path, bridge)
        bridge_reports[method] = str(report_path)

    combined_rows: list[dict[str, Any]] = []
    for method in sorted(all_method_rows):
        combined_rows.extend(all_method_rows[method])
    combined_report_path = output / "matched_top3_90_bridge_report.json"
    combined_report = {
        "schema_version": "nl_14x14_matched_top3_90_bridge_report_v1",
        "status": "ok",
        "scope_note": (
            "Matched bridge report for frozen-SNI scenario evaluation. Rows are "
            "duplicated across candidate methods for the same query windows; run "
            "nl_scenario_level_evaluation.py with --allow-duplicate-query-windows "
            "and common random numbers for within-window comparisons."
        ),
        "source_examples_jsonl": str(_resolve(examples_jsonl)),
        "source_training_run_report": str(_resolve(training_run_report_path))
        if training_run_report_path
        else "",
        "matched_query_window_indices": [int(idx) for idx in queries],
        "methods": sorted(all_method_rows),
        "split": {
            "train_indices": [int(idx) for idx in train_indices],
            "test_indices": [int(idx) for idx in queries],
        },
        "window_indices": [
            int(idx) for idx in sorted(set(train_indices).union(test_indices).union(queries))
        ],
        "evaluation": {
            "heldout_examples": combined_rows,
            "support_overlap": _method_overlap(all_method_rows),
        },
    }
    _write_json(combined_report_path, combined_report)

    query_reviews: list[dict[str, Any]] = []
    query_by_window = {int(row["target_window_index"]): row for row in query_rows}
    for idx in queries:
        method_supports: dict[str, list[dict[str, Any]]] = {}
        for method, rows in sorted(all_method_rows.items()):
            matched = [row for row in rows if int(row["window_index"]) == int(idx)]
            if not matched:
                continue
            method_supports[method] = [
                {
                    "window_index": int(item["window_index"]),
                    "window_id": str(item.get("window_id", _window_id(item["window_index"]))),
                    "weight": float(item.get("weight", 0.0)),
                    "score": float(item.get("cosine", item.get("retrieval_score", 0.0)) or 0.0),
                }
                for item in matched[0].get("top_train_pool", [])
            ]
        query_example = query_by_window[int(idx)]
        query_reviews.append(
            {
                "window_index": int(idx),
                "window_id": str(query_example.get("target_window_id", _window_id(idx))),
                "query_text": str(query_example.get("text", "")),
                "supports_by_method": method_supports,
            }
        )

    audit_report_path = output / "support_level_audit_report.json"
    markdown_path = output / "support_level_audit_report.md"
    audit_report = {
        "schema_version": "nl_14x14_support_level_audit_v1",
        "status": "pass",
        "scope_note": (
            "Support-level audit over the stride-5 14+14 corpus on the frozen "
            "support-decoder-test overlap. This is a support-selection audit, "
            "not a narrative-generation run."
        ),
        "query_window_indices": [int(idx) for idx in queries],
        "query_view": str(query_view),
        "top_k": int(top_k),
        "support_pool_size": int(support_pool_size),
        "temporal_gap": int(temporal_gap),
        "method_summaries": {
            method: _summarize_rows(rows) for method, rows in sorted(all_method_rows.items())
        },
        "support_overlap": _method_overlap(all_method_rows),
        "bridge_reports": bridge_reports,
        "baseline_bridge_reports": {
            name: str(_resolve(path)) for name, path in (baseline_bridge_reports or {}).items()
        },
        "public_paper_demo_candidate_note": (
            "The verified public paper/demo candidate is the nearest-similar "
            "main-regime top3/90 posterior-ensemble workflow, not a direct "
            "bridge-report artifact. This matched bridge report includes the "
            "bridge-shaped episode grounded text-preference candidate when its "
            "artifact is available; the public posterior artifacts are recorded "
            "separately for provenance."
        ),
        "public_paper_demo_posterior_artifacts": {
            name: str(_resolve(path)) for name, path in PUBLIC_PAPER_DEMO_POSTERIOR_ARTIFACTS.items()
        },
        "query_reviews": query_reviews,
        "artifact_paths": {
            "report": str(audit_report_path),
            "markdown": str(markdown_path),
            "matched_bridge_report": str(combined_report_path),
            "bridge_reports": bridge_reports,
        },
    }
    _write_json(audit_report_path, audit_report)
    _write_support_markdown(markdown_path, audit_report)
    return audit_report


def _parse_query_windows(raw: str) -> list[int] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES_JSONL)
    parser.add_argument("--training-run-report", type=Path, default=DEFAULT_TRAINING_RUN_REPORT)
    parser.add_argument("--text-space-arrays", type=Path, default=DEFAULT_TEXT_SPACE_ARRAYS)
    parser.add_argument("--projected-arrays", type=Path, default=DEFAULT_PROJECTED_ARRAYS)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-windows", default="")
    parser.add_argument("--query-view", default="full_professional")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--support-pool-size", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--skip-default-baselines", action="store_true")
    args = parser.parse_args(argv)

    baselines: dict[str, Path] = (
        {} if bool(args.skip_default_baselines) else dict(DEFAULT_BASELINE_BRIDGE_REPORTS)
    )
    report = build_14x14_support_audit(
        examples_jsonl=args.examples_jsonl,
        support_report_path=args.support_report,
        text_space_arrays_path=args.text_space_arrays,
        projected_arrays_path=args.projected_arrays,
        output_dir=args.output_dir,
        training_run_report_path=args.training_run_report,
        query_window_indices=_parse_query_windows(str(args.query_windows)),
        query_view=str(args.query_view),
        top_k=int(args.top_k),
        support_pool_size=int(args.support_pool_size),
        temporal_gap=int(args.temporal_gap),
        baseline_bridge_reports=baselines,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "query_count": len(report["query_window_indices"]),
                "methods": sorted(report["method_summaries"]),
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
                "matched_bridge_report": report["artifact_paths"]["matched_bridge_report"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
