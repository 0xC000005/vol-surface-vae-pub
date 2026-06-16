#!/usr/bin/env python
"""Stage-2 preference reranker for grounded text-retrieval supports.

The Stage-1 grounded OpenAI text retriever can find directionally coherent
historical support prefixes, but support coherence alone did not beat the
projected-memory selector on frozen-SNI scenario quality. This isolated Stage-2
utility keeps that grounded text candidate pool and learns a small historical
backtest preference model over support candidates. It does not generate
narratives, project text to SNI memory, or alter the paper/demo defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_embedding_grounded_bridge_report import (  # noqa: E402
    DEFAULT_CARDS_JSONL,
    DEFAULT_EMBEDDING_ARRAYS,
    DEFAULT_SUPPORT_REPORT,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    _load_json,
    _read_jsonl,
    _softmax_weights,
    _split_indices_from_support_report,
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
    _direction_check,
    _required_grounding_claims,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    score_sample_distribution,
)


DEFAULT_BRIDGE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_embedding_grounded_top3_90_66q/"
    "embedding_grounded_bridge_report.json"
)
DEFAULT_SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_grounded_text_preference_reranker_984a"
)
VIEW_FEATURE_NAMES = tuple(f"view_{name}" for name in DEFAULT_VIEW_NAMES)
FEATURE_NAMES = [
    "embedding_score",
    "support_weight",
    "rank_inverse",
    "start_distance_z",
    "start_similarity",
    "recent_delta_distance_z",
    "candidate_recent_delta_norm_z",
    "query_recent_delta_norm_z",
    "grounding_checked_count",
    *VIEW_FEATURE_NAMES,
]


@dataclass(frozen=True)
class PreferenceTrainingTable:
    features: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, Any]]
    feature_names: list[str]


@dataclass(frozen=True)
class TextSupportPreferenceModel:
    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: list[str]
    ridge_alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = (x - self.feature_mean[None, :]) / self.feature_std[None, :]
        return (z @ self.coefficients + float(self.intercept)).astype(np.float64)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _safe_scale(values: np.ndarray, fit_indices: np.ndarray, *, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    mean = np.nanmean(arr[fit], axis=0, keepdims=True)
    std = np.nanstd(arr[fit], axis=0, keepdims=True)
    std = np.where(np.isfinite(std) & (std > floor), std, floor)
    return mean.astype(np.float32), std.astype(np.float32)


def build_delta_scale(
    future_delta: np.ndarray, train_indices: np.ndarray, *, floor: float = 1e-3
) -> np.ndarray:
    train = np.asarray(future_delta, dtype=np.float32)[np.asarray(train_indices, dtype=np.int64)]
    scale = np.nanstd(train, axis=0).astype(np.float32)
    positive = scale[np.isfinite(scale) & (scale > float(floor))]
    fallback = float(np.nanmedian(positive)) if positive.size else float(floor)
    return np.where(np.isfinite(scale) & (scale > float(floor)), scale, fallback).astype(np.float32)


def replay_loss_z(
    *,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    query_window_index: int,
    candidate_window_index: int,
) -> float:
    future = np.asarray(future_delta, dtype=np.float32)
    scale = np.maximum(np.asarray(delta_scale, dtype=np.float32), 1e-8)
    query = future[int(query_window_index)] / scale
    candidate = future[int(candidate_window_index)] / scale
    return float(np.mean(np.abs(query - candidate)))


def _feature_stats(history_raw: np.ndarray, fit_indices: np.ndarray) -> dict[str, np.ndarray]:
    history = np.asarray(history_raw, dtype=np.float32)
    terminal = history[:, -1, :]
    recent_delta = history[:, -1, :] - history[:, 0, :]
    start_mean, start_std = _safe_scale(terminal, fit_indices)
    recent_mean, recent_std = _safe_scale(recent_delta, fit_indices)
    start_z = (terminal - start_mean) / start_std
    recent_z = (recent_delta - recent_mean) / recent_std
    recent_norm = np.linalg.norm(recent_z, axis=1)
    return {
        "start_z": start_z.astype(np.float32),
        "recent_z": recent_z.astype(np.float32),
        "recent_norm": recent_norm.astype(np.float32),
    }


def _candidate_features(
    *,
    query_window_index: int,
    candidate: dict[str, Any],
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    query_idx = int(query_window_index)
    candidate_idx = int(candidate["window_index"])
    start_z = np.asarray(stats["start_z"], dtype=np.float32)
    recent_z = np.asarray(stats["recent_z"], dtype=np.float32)
    recent_norm = np.asarray(stats["recent_norm"], dtype=np.float32)
    start_distance = float(
        np.linalg.norm(start_z[candidate_idx] - start_z[query_idx])
        / math.sqrt(max(start_z.shape[1], 1))
    )
    recent_distance = float(
        np.linalg.norm(recent_z[candidate_idx] - recent_z[query_idx])
        / math.sqrt(max(recent_z.shape[1], 1))
    )
    components = candidate.get("score_components", {})
    if not isinstance(components, dict):
        components = {}
    direction_check = components.get("direction_check", {})
    if not isinstance(direction_check, dict):
        direction_check = {}
    view = str(components.get("view", ""))
    view_flags = [1.0 if view == name else 0.0 for name in DEFAULT_VIEW_NAMES]
    rank = max(float(candidate.get("rank", 0.0) or 0.0), 1.0)
    embedding_score = float(
        components.get(
            "embedding_score",
            candidate.get("retrieval_score", candidate.get("cosine", 0.0)),
        )
        or 0.0
    )
    values = [
        embedding_score,
        float(candidate.get("weight", candidate.get("base_support_weight", 0.0)) or 0.0),
        1.0 / rank,
        start_distance,
        math.exp(-start_distance),
        recent_distance,
        float(recent_norm[candidate_idx]),
        float(recent_norm[query_idx]),
        float(direction_check.get("checked_count", 0.0) or 0.0),
        *view_flags,
    ]
    return np.asarray(values, dtype=np.float32)


def build_training_table_from_pools(
    *,
    query_pools: list[dict[str, Any]],
    history_raw: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    fit_indices: np.ndarray | None = None,
) -> PreferenceTrainingTable:
    if fit_indices is None:
        fit_indices = np.arange(np.asarray(history_raw).shape[0], dtype=np.int64)
    stats = _feature_stats(history_raw, np.asarray(fit_indices, dtype=np.int64))
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    labels: list[float] = []
    for pool in query_pools:
        query_idx = int(pool["query_window_index"])
        candidates = pool.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_idx = int(candidate["window_index"])
            loss = replay_loss_z(
                future_delta=future_delta,
                delta_scale=delta_scale,
                query_window_index=query_idx,
                candidate_window_index=candidate_idx,
            )
            features.append(
                _candidate_features(
                    query_window_index=query_idx,
                    candidate=candidate,
                    stats=stats,
                )
            )
            labels.append(-float(loss))
            rows.append(
                {
                    "query_window_index": query_idx,
                    "candidate_window_index": candidate_idx,
                    "candidate_rank": int(candidate.get("rank", len(rows) + 1)),
                    "true_replay_loss_z": float(loss),
                }
            )
    if not features:
        raise ValueError("no training rows were built")
    return PreferenceTrainingTable(
        features=np.stack(features, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        rows=rows,
        feature_names=list(FEATURE_NAMES),
    )


def fit_text_support_preference_reranker(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
) -> TextSupportPreferenceModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("features and labels must have compatible shapes")
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-8)
    z = (x - mean[None, :]) / std[None, :]
    design = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return TextSupportPreferenceModel(
        coefficients=coef[1:].astype(np.float64),
        intercept=float(coef[0]),
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
        feature_names=list(FEATURE_NAMES),
        ridge_alpha=float(ridge_alpha),
    )


def _model_payload(model: TextSupportPreferenceModel) -> dict[str, Any]:
    return {
        "feature_names": list(model.feature_names),
        "coefficients": [float(value) for value in model.coefficients],
        "intercept": float(model.intercept),
        "feature_mean": [float(value) for value in model.feature_mean],
        "feature_std": [float(value) for value in model.feature_std],
        "ridge_alpha": float(model.ridge_alpha),
    }


def _score_candidates(
    *,
    query_window_index: int,
    candidates: list[dict[str, Any]],
    stats: dict[str, np.ndarray],
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    model: TextSupportPreferenceModel,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    features = np.stack(
        [
            _candidate_features(
                query_window_index=query_window_index,
                candidate=candidate,
                stats=stats,
            )
            for candidate in candidates
        ],
        axis=0,
    )
    scores = model.predict(features)
    rows: list[dict[str, Any]] = []
    for candidate, score, feature_row in zip(candidates, scores, features, strict=True):
        item = json.loads(json.dumps(candidate))
        item["learned_support_score"] = float(score)
        item["true_replay_loss_z"] = replay_loss_z(
            future_delta=future_delta,
            delta_scale=delta_scale,
            query_window_index=int(query_window_index),
            candidate_window_index=int(candidate["window_index"]),
        )
        item["preference_features"] = {
            name: float(value)
            for name, value in zip(model.feature_names, feature_row, strict=False)
        }
        rows.append(item)
    rows.sort(key=lambda row: float(row["learned_support_score"]), reverse=True)
    return rows


def _apply_learned_top3_90_selection(
    scored_candidates: list[dict[str, Any]],
    *,
    weight_temperature: float = 1.0,
) -> list[dict[str, Any]]:
    if not scored_candidates:
        return []
    scores = [float(item["learned_support_score"]) for item in scored_candidates]
    weights = _softmax_weights(scores, temperature=float(weight_temperature))
    weighted: list[dict[str, Any]] = []
    for rank, (item, weight) in enumerate(zip(scored_candidates, weights, strict=True), 1):
        row = dict(item)
        row["rank"] = int(rank)
        row["base_support_weight"] = float(weight)
        weighted.append(row)
    selected: list[dict[str, Any]] = []
    mass = 0.0
    for item in weighted:
        if len(selected) >= 3:
            break
        selected.append(dict(item))
        mass += float(item["base_support_weight"])
        if mass >= 0.90:
            break
    denom = sum(float(item["base_support_weight"]) for item in selected)
    if denom <= 0.0:
        denom = float(max(len(selected), 1))
        for item in selected:
            item["weight"] = 1.0 / denom
    else:
        for item in selected:
            item["weight"] = float(item["base_support_weight"]) / denom
    for display_rank, item in enumerate(selected, 1):
        item["rank"] = int(display_rank)
        item["posterior_weight"] = float(item["weight"])
        item["posterior_role"] = "grounded_text_preference_top3_90_selected"
        components = dict(item.get("score_components", {}))
        components["method"] = "grounded_text_preference_top3_90"
        components["learned_support_score"] = float(item["learned_support_score"])
        item["score_components"] = components
    return selected


def rerank_bridge_report_with_model(
    *,
    bridge_report: dict[str, Any],
    history_raw: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    model: TextSupportPreferenceModel,
    fit_indices: np.ndarray | None = None,
    weight_temperature: float = 1.0,
) -> dict[str, Any]:
    output = json.loads(json.dumps(bridge_report))
    if fit_indices is None:
        split_train = output.get("split", {}).get("train_indices", [])
        fit_indices = (
            np.asarray(split_train, dtype=np.int64)
            if split_train
            else np.arange(np.asarray(history_raw).shape[0], dtype=np.int64)
        )
    stats = _feature_stats(history_raw, np.asarray(fit_indices, dtype=np.int64))
    rows = output.get("evaluation", {}).get("heldout_examples", [])
    reranked_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_pool = row.get("pre_top3_90_candidate_pool") or row.get("top_train_pool") or []
        if not isinstance(source_pool, list) or not source_pool:
            continue
        scored = _score_candidates(
            query_window_index=int(row["window_index"]),
            candidates=[dict(item) for item in source_pool if isinstance(item, dict)],
            stats=stats,
            future_delta=future_delta,
            delta_scale=delta_scale,
            model=model,
        )
        top_pool = _apply_learned_top3_90_selection(
            scored, weight_temperature=float(weight_temperature)
        )
        row["preference_reranked_candidate_pool"] = scored
        row["top_train_pool"] = top_pool
        row["kind"] = "grounded_text_preference_top3_90"
        reranked_count += 1
    result_sets = {
        str(row.get("window_id", row.get("window_index", ""))): [
            str(item.get("window_id", item.get("window_index", "")))
            for item in row.get("top_train_pool", [])
        ]
        for row in rows
        if isinstance(row, dict)
    }
    output["support_policy"] = {
        "name": "grounded_text_preference_reranker",
        "research_lane": "stage2_text_space_backtest_preference",
        "candidate_rows_reranked": int(reranked_count),
        "feature_names": list(model.feature_names),
        "ridge_alpha": float(model.ridge_alpha),
        "method": (
            "Ridge preference model trained on grounded text-retrieval support "
            "candidates. Labels are negative historical replay loss against "
            "realized next-30-day deltas. The model reranks the existing "
            "candidate pool and then applies the same top3/90 support-posterior "
            "assembly before frozen-SNI rollout."
        ),
    }
    output.setdefault("evaluation", {})["support_overlap"] = pairwise_jaccard_summary(result_sets)
    return output


def _score_topk_replay(
    *,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    query_window_index: int,
    candidate_indices: list[int],
    top_k: int,
) -> dict[str, Any]:
    selected = [int(idx) for idx in candidate_indices[: max(int(top_k), 1)]]
    samples = np.asarray(
        future_delta[np.asarray(selected, dtype=np.int64)], dtype=np.float32
    )
    target = np.asarray(future_delta[int(query_window_index)], dtype=np.float32)
    return score_sample_distribution(samples, target, scale=delta_scale)


def _mean_metric(rows: list[dict[str, Any]], method: str, metric: str) -> float | None:
    values = [
        float(row["methods"][method][metric])
        for row in rows
        if row.get("methods", {}).get(method, {}).get(metric) is not None
    ]
    return float(np.mean(values)) if values else None


def _metric_delta(candidate: float, baseline: float, metric: str) -> float:
    if metric == "coverage_80":
        return candidate - baseline
    return baseline - candidate


def _summarize_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("ensemble_crps_z", "energy_score_z", "coverage_80")
    methods = ("learned_rerank_topk_replay", "oracle_pool_topk_replay")
    summary: dict[str, Any] = {"baseline": "original_topk_replay", "window_count": len(rows)}
    for method in methods:
        block: dict[str, Any] = {}
        for metric in metrics:
            baseline_mean = _mean_metric(rows, "original_topk_replay", metric)
            method_mean = _mean_metric(rows, method, metric)
            if baseline_mean is None or method_mean is None:
                continue
            deltas = [
                _metric_delta(
                    float(row["methods"][method][metric]),
                    float(row["methods"]["original_topk_replay"][metric]),
                    metric,
                )
                for row in rows
            ]
            block[metric] = {
                "baseline_mean": round(float(baseline_mean), 12),
                "candidate_mean": round(float(method_mean), 12),
                "mean_delta_positive_is_better": round(float(np.mean(deltas)), 12),
                "win_rate": round(float(np.mean(np.asarray(deltas) > 0.0)), 12),
            }
        summary[method] = block
    return summary


def evaluate_preference_rerank_replay(
    *,
    bridge_report: dict[str, Any],
    reranked_bridge_report: dict[str, Any],
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    top_k: int,
    role: str = "anchor",
) -> dict[str, Any]:
    original_rows = {
        int(row["window_index"]): row
        for row in bridge_report.get("evaluation", {}).get("heldout_examples", [])
        if isinstance(row, dict) and str(row.get("role", "")) == str(role)
    }
    reranked_rows = {
        int(row["window_index"]): row
        for row in reranked_bridge_report.get("evaluation", {}).get(
            "heldout_examples", []
        )
        if isinstance(row, dict) and str(row.get("role", "")) == str(role)
    }
    window_rows: list[dict[str, Any]] = []
    for window_index in sorted(set(original_rows) & set(reranked_rows)):
        original_source = original_rows[window_index].get("pre_top3_90_candidate_pool")
        if not original_source:
            original_source = original_rows[window_index].get("top_train_pool", [])
        original_pool = [int(item["window_index"]) for item in original_source]
        learned_pool = [
            int(item["window_index"])
            for item in reranked_rows[window_index].get("top_train_pool", [])
        ]
        oracle_pool = sorted(
            original_pool,
            key=lambda idx: replay_loss_z(
                future_delta=future_delta,
                delta_scale=delta_scale,
                query_window_index=window_index,
                candidate_window_index=int(idx),
            ),
        )
        methods = {
            "original_topk_replay": _score_topk_replay(
                future_delta=future_delta,
                delta_scale=delta_scale,
                query_window_index=window_index,
                candidate_indices=original_pool,
                top_k=top_k,
            ),
            "learned_rerank_topk_replay": _score_topk_replay(
                future_delta=future_delta,
                delta_scale=delta_scale,
                query_window_index=window_index,
                candidate_indices=learned_pool,
                top_k=top_k,
            ),
            "oracle_pool_topk_replay": _score_topk_replay(
                future_delta=future_delta,
                delta_scale=delta_scale,
                query_window_index=window_index,
                candidate_indices=oracle_pool,
                top_k=top_k,
            ),
        }
        window_rows.append(
            {
                "window_index": int(window_index),
                "window_id": str(original_rows[window_index].get("window_id", "")),
                "original_pool": original_pool,
                "learned_pool": learned_pool,
                "oracle_pool": oracle_pool,
                "methods": methods,
            }
        )
    return {
        "status": "ok",
        "window_count": len(window_rows),
        "top_k": int(top_k),
        "summary": _summarize_replay(window_rows),
        "window_scores": window_rows,
    }


def _rank_grounded_text_pool(
    *,
    query_vector: np.ndarray,
    query_index: int,
    query_claims: list[dict[str, Any]],
    candidate_vectors: np.ndarray,
    candidate_rows: list[dict[str, Any]],
    cards_by_index: dict[int, dict[str, Any]],
    support_pool_size: int,
    temporal_gap: int,
    max_grounding_mismatches: int,
    initial_scan: int,
) -> list[dict[str, Any]]:
    scores = np.asarray(candidate_vectors @ query_vector, dtype=np.float32)
    n = scores.shape[0]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    scan_sizes = []
    size = min(max(int(initial_scan), int(support_pool_size) * 8), n)
    while size < n:
        scan_sizes.append(size)
        size = min(size * 4, n)
    scan_sizes.append(n)
    for scan_size in scan_sizes:
        if scan_size >= n:
            order = np.argsort(-scores)
        else:
            rough = np.argpartition(-scores, scan_size - 1)[:scan_size]
            order = rough[np.argsort(-scores[rough])]
        selected = []
        seen = set()
        for pos in order:
            row = candidate_rows[int(pos)]
            idx = int(row["window_index"])
            if idx == int(query_index) or idx in seen:
                continue
            seen.add(idx)
            # #48 causal gap: candidate's temporal_gap-day window must end at/before query start.
            if int(query_index) - idx < int(temporal_gap):
                continue
            if (
                int(temporal_gap) > 0
                and any(abs(idx - int(item["window_index"])) < int(temporal_gap) for item in selected)
            ):
                continue
            direction_check = _direction_check(
                query_claims=query_claims,
                candidate_card=cards_by_index.get(idx, {}),
                max_mismatches=int(max_grounding_mismatches),
            )
            if str(direction_check.get("status", "")) != "pass":
                continue
            selected.append(
                {
                    "rank": len(selected) + 1,
                    "window_index": idx,
                    "window_id": str(row["window_id"]),
                    "cosine": float(scores[int(pos)]),
                    "retrieval_score": float(scores[int(pos)]),
                    "scenario_title": str(row.get("scenario_title", "")),
                    "score_components": {
                        "method": "raw_openai_embedding_grounded_preference_candidate",
                        "embedding_score": float(scores[int(pos)]),
                        "view": str(row.get("view", "")),
                        "direction_check": direction_check,
                    },
                }
            )
            if len(selected) >= int(support_pool_size):
                break
        if len(selected) >= int(support_pool_size) or scan_size >= n:
            break
    weights = _softmax_weights([float(item["cosine"]) for item in selected])
    for item, weight in zip(selected, weights, strict=False):
        item["weight"] = float(weight)
    return selected


def build_train_query_pools(
    *,
    cards: list[dict[str, Any]],
    support_report: dict[str, Any],
    embedding_arrays: dict[str, np.ndarray],
    query_view: str,
    support_pool_size: int,
    temporal_gap: int,
    max_grounding_claims: int,
    max_grounding_mismatches: int,
    max_train_queries: int,
    initial_scan: int,
) -> list[dict[str, Any]]:
    train_indices, _test_indices = _split_indices_from_support_report(support_report)
    if int(max_train_queries) > 0:
        train_indices = train_indices[: int(max_train_queries)]
    cards_by_index = {_window_index(card): card for card in cards}
    train_cards_all = [
        cards_by_index[idx]
        for idx in _split_indices_from_support_report(support_report)[0]
        if idx in cards_by_index
    ]
    candidate_rows, candidate_texts = _candidate_view_rows(
        train_cards_all, tuple(DEFAULT_VIEW_NAMES)
    )
    candidate_count = int(np.asarray(embedding_arrays["candidate_text_count"])[0])
    if candidate_count != len(candidate_rows) or candidate_count != len(candidate_texts):
        raise ValueError("embedding arrays do not match rebuilt candidate rows")
    vectors = np.asarray(embedding_arrays["text_embeddings"], dtype=np.float32)[:candidate_count]
    query_vector_by_index: dict[int, np.ndarray] = {}
    fallback_by_index: dict[int, np.ndarray] = {}
    for row, vector in zip(candidate_rows, vectors, strict=True):
        idx = int(row["window_index"])
        fallback_by_index.setdefault(idx, vector)
        if str(row.get("view", "")) == str(query_view):
            query_vector_by_index[idx] = vector
    pools: list[dict[str, Any]] = []
    for count, idx in enumerate(train_indices, 1):
        card = cards_by_index.get(int(idx))
        if card is None:
            continue
        query_vector = query_vector_by_index.get(int(idx), fallback_by_index.get(int(idx)))
        if query_vector is None:
            continue
        claims = _required_grounding_claims(card, max_claims=int(max_grounding_claims))
        candidates = _rank_grounded_text_pool(
            query_vector=query_vector,
            query_index=int(idx),
            query_claims=claims,
            candidate_vectors=vectors,
            candidate_rows=candidate_rows,
            cards_by_index=cards_by_index,
            support_pool_size=int(support_pool_size),
            temporal_gap=int(temporal_gap),
            max_grounding_mismatches=int(max_grounding_mismatches),
            initial_scan=int(initial_scan),
        )
        if candidates:
            pools.append({"query_window_index": int(idx), "candidates": candidates})
        if count % 250 == 0:
            print(f"  built train query pools {count}/{len(train_indices)}", flush=True)
    return pools


def run_preference_reranker(args: argparse.Namespace) -> dict[str, Any]:
    bridge_report = _load_json(args.bridge_report)
    cards = _read_jsonl(args.cards_jsonl)
    assert_cards_allowed_for_retrieval(cards, path=Path(args.cards_jsonl))
    support_report = _load_json(args.support_report)
    embedding_arrays = _load_npz(args.embedding_arrays)
    support_arrays = _load_npz(args.support_arrays)
    history_raw = np.asarray(support_arrays["history_raw"], dtype=np.float32)
    future_delta = np.asarray(support_arrays["future_delta"], dtype=np.float32)
    train_indices = np.asarray(support_arrays["train_indices"], dtype=np.int64)
    delta_scale = build_delta_scale(
        future_delta, train_indices, floor=float(args.score_scale_floor)
    )
    train_pools = build_train_query_pools(
        cards=cards,
        support_report=support_report,
        embedding_arrays=embedding_arrays,
        query_view=str(args.query_view),
        support_pool_size=int(args.support_pool_size),
        temporal_gap=int(args.temporal_gap),
        max_grounding_claims=int(args.max_grounding_claims),
        max_grounding_mismatches=int(args.max_grounding_mismatches),
        max_train_queries=int(args.max_train_queries),
        initial_scan=int(args.initial_scan),
    )
    table = build_training_table_from_pools(
        query_pools=train_pools,
        history_raw=history_raw,
        future_delta=future_delta,
        delta_scale=delta_scale,
        fit_indices=train_indices,
    )
    model = fit_text_support_preference_reranker(
        table.features, table.labels, ridge_alpha=float(args.ridge_alpha)
    )
    reranked_bridge = rerank_bridge_report_with_model(
        bridge_report=bridge_report,
        history_raw=history_raw,
        future_delta=future_delta,
        delta_scale=delta_scale,
        model=model,
        fit_indices=train_indices,
        weight_temperature=float(args.weight_temperature),
    )
    replay_eval = evaluate_preference_rerank_replay(
        bridge_report=bridge_report,
        reranked_bridge_report=reranked_bridge,
        future_delta=future_delta,
        delta_scale=delta_scale,
        top_k=int(args.top_k),
        role=str(args.query_role),
    )
    learned_summary = replay_eval["summary"]["learned_rerank_topk_replay"]
    learned_crps_delta = learned_summary["ensemble_crps_z"]["mean_delta_positive_is_better"]
    learned_energy_delta = learned_summary["energy_score_z"]["mean_delta_positive_is_better"]
    result_status = (
        "mechanism_found"
        if float(learned_crps_delta) > 0.0 or float(learned_energy_delta) > 0.0
        else "dead_end"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "grounded_text_preference_reranker_report.json"
    bridge_path = output_dir / "grounded_text_preference_bridge_report.json"
    model_path = output_dir / "grounded_text_preference_model.json"
    reranked_bridge["artifact_paths"] = {
        **dict(reranked_bridge.get("artifact_paths", {})),
        "report": str(bridge_path),
    }
    report = {
        "status": "ok",
        "research_lane": "stage2_text_space_backtest_preference",
        "result_status": result_status,
        "scope_note": (
            "Stage-2 grounded text-support preference reranker. Training labels "
            "come from historical replay closeness over support-bank future "
            "deltas. Promotion still requires frozen-SNI scenario-level "
            "evaluation and comparison against projected-memory/hybrid-start baselines."
        ),
        "inputs": {
            "bridge_report": str(args.bridge_report),
            "cards_jsonl": str(args.cards_jsonl),
            "support_report": str(args.support_report),
            "embedding_arrays": str(args.embedding_arrays),
            "support_arrays": str(args.support_arrays),
        },
        "config": {
            "query_view": str(args.query_view),
            "support_pool_size": int(args.support_pool_size),
            "temporal_gap": int(args.temporal_gap),
            "max_grounding_claims": int(args.max_grounding_claims),
            "max_grounding_mismatches": int(args.max_grounding_mismatches),
            "max_train_queries": int(args.max_train_queries),
            "ridge_alpha": float(args.ridge_alpha),
            "weight_temperature": float(args.weight_temperature),
            "top_k": int(args.top_k),
        },
        "training": {
            "query_pool_count": int(len(train_pools)),
            "training_row_count": int(table.features.shape[0]),
            "feature_names": list(table.feature_names),
            "label_mean": float(np.mean(table.labels)),
            "label_std": float(np.std(table.labels)),
        },
        "model": _model_payload(model),
        "replay_eval": replay_eval,
        "artifact_paths": {
            "report": str(report_path),
            "reranked_bridge_report": str(bridge_path),
            "model": str(model_path),
        },
        "decision_hint": (
            "Run nl_scenario_level_evaluation.py on reranked_bridge_report if "
            "result_status is mechanism_found; do not promote without CRPS/Energy "
            "and conditionality comparison against current baselines."
        ),
    }
    _write_json(model_path, _model_payload(model))
    _write_json(bridge_path, reranked_bridge)
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", type=Path, default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--embedding-arrays", type=Path, default=DEFAULT_EMBEDDING_ARRAYS)
    parser.add_argument("--support-arrays", type=Path, default=DEFAULT_SUPPORT_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-view", default="full_professional")
    parser.add_argument("--support-pool-size", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--max-grounding-claims", type=int, default=4)
    parser.add_argument("--max-grounding-mismatches", type=int, default=0)
    parser.add_argument("--max-train-queries", type=int, default=1024)
    parser.add_argument("--initial-scan", type=int, default=512)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--weight-temperature", type=float, default=1.0)
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--query-role", default="anchor")
    args = parser.parse_args(argv)
    report = run_preference_reranker(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "result_status": report["result_status"],
                "report": report["artifact_paths"]["report"],
                "reranked_bridge_report": report["artifact_paths"]["reranked_bridge_report"],
                "training_query_pool_count": report["training"]["query_pool_count"],
                "training_row_count": report["training"]["training_row_count"],
                "learned_replay_summary": report["replay_eval"]["summary"][
                    "learned_rerank_topk_replay"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
