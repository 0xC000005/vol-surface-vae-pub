#!/usr/bin/env python
"""Learned support-reranker TestFlight for narrative-conditioned mixtures.

This is an exploration-lane experiment. It keeps the historical support mixture
as the backbone, then asks whether simple candidate-level features can learn to
rerank support windows toward better historical replay quality. No OpenAI calls
are made.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    summarize_bridge_metrics,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    score_sample_distribution,
)


DEFAULT_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_ORACLE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_oracle_fullheldout_786c/prefix_latent_oracle_arrays.npz"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_learned_support_reranker_879a_testflight"
)


@dataclass(frozen=True)
class PairwiseTrainingTable:
    features: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, Any]]
    feature_names: list[str]


@dataclass(frozen=True)
class LinearSupportReranker:
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


FEATURE_NAMES = [
    "memory_cosine",
    "start_distance_z",
    "start_similarity",
    "memory_minus_start_distance",
    "candidate_recent_delta_norm",
    "candidate_start_norm",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
    return arr / denom


def _cosine_to_query(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    t = np.asarray(targets, dtype=np.float32)
    if t.ndim != 2 or q.shape[1] != t.shape[1]:
        raise ValueError("query and targets must be compatible 2-D memory arrays")
    return (_normalize_rows(t) @ _normalize_rows(q)[0]).astype(np.float32)


def _standardized_start_states(
    history_level: np.ndarray, fit_indices: np.ndarray
) -> np.ndarray:
    history = np.asarray(history_level, dtype=np.float32)
    if history.ndim != 3:
        raise ValueError("history_level must have shape [N,T,C]")
    starts = history[:, -1, :]
    fit = np.asarray(fit_indices, dtype=np.int64)
    mean = starts[fit].mean(axis=0, keepdims=True)
    std = np.maximum(starts[fit].std(axis=0, keepdims=True), 1e-6)
    return ((starts - mean) / std).astype(np.float32)


def candidate_pool_indices(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_pool_size: int,
    exclude_index: int | None = None,
) -> np.ndarray:
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("candidate_indices must be non-empty")
    cosines = _cosine_to_query(query_memory, memory_targets)[candidates]
    order = np.argsort(-cosines)
    selected: list[int] = []
    for pos in order:
        idx = int(candidates[int(pos)])
        if exclude_index is not None and idx == int(exclude_index):
            continue
        selected.append(idx)
        if len(selected) >= int(candidate_pool_size):
            break
    if not selected:
        raise ValueError("candidate pool is empty after exclusions")
    return np.asarray(selected, dtype=np.int64)


def _candidate_feature_matrix(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    start_z: np.ndarray,
    query_window_index: int,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    memory = np.asarray(memory_targets, dtype=np.float32)
    history = np.asarray(history_level, dtype=np.float32)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    query_idx = int(query_window_index)
    cosines = _cosine_to_query(query_memory, memory)[candidates]
    start_distance = np.linalg.norm(
        start_z[candidates] - start_z[query_idx][None, :], axis=1
    )
    recent_delta = history[candidates, -1, :] - history[candidates, 0, :]
    recent_norm = np.linalg.norm(recent_delta, axis=1)
    candidate_start_norm = np.linalg.norm(start_z[candidates], axis=1)
    features = np.stack(
        [
            cosines,
            start_distance,
            np.exp(-start_distance),
            cosines - start_distance,
            recent_norm,
            candidate_start_norm,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def replay_loss_z(
    *,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    query_window_index: int,
    candidate_window_index: int,
) -> float:
    future = np.asarray(future_delta, dtype=np.float32)
    scale = np.maximum(np.asarray(delta_scale, dtype=np.float32), 1e-8)
    q = future[int(query_window_index)] / scale
    c = future[int(candidate_window_index)] / scale
    return float(np.mean(np.abs(q - c)))


def build_pairwise_training_table(
    *,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    train_indices: np.ndarray,
    candidate_pool_size: int = 16,
) -> PairwiseTrainingTable:
    """Build query-candidate rows labeled by future replay closeness."""

    train = np.asarray(train_indices, dtype=np.int64)
    memory = np.asarray(memory_targets, dtype=np.float32)
    start_z = _standardized_start_states(history_level, train)
    rows: list[dict[str, Any]] = []
    feature_blocks: list[np.ndarray] = []
    labels: list[float] = []
    for query_idx in train:
        pool = candidate_pool_indices(
            query_memory=memory[int(query_idx)],
            memory_targets=memory,
            candidate_indices=train,
            candidate_pool_size=int(candidate_pool_size),
            exclude_index=int(query_idx),
        )
        features = _candidate_feature_matrix(
            query_memory=memory[int(query_idx)],
            memory_targets=memory,
            history_level=history_level,
            start_z=start_z,
            query_window_index=int(query_idx),
            candidate_indices=pool,
        )
        feature_blocks.append(features)
        for local_pos, candidate_idx in enumerate(pool):
            loss = replay_loss_z(
                future_delta=future_delta,
                delta_scale=delta_scale,
                query_window_index=int(query_idx),
                candidate_window_index=int(candidate_idx),
            )
            labels.append(-loss)
            rows.append(
                {
                    "query_window_index": int(query_idx),
                    "candidate_window_index": int(candidate_idx),
                    "candidate_rank": int(local_pos + 1),
                    "true_replay_loss_z": float(loss),
                }
            )
    if not feature_blocks:
        raise ValueError("no pairwise training rows built")
    return PairwiseTrainingTable(
        features=np.concatenate(feature_blocks, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        rows=rows,
        feature_names=list(FEATURE_NAMES),
    )


def fit_linear_support_reranker(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
) -> LinearSupportReranker:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have compatible shapes")
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-8)
    z = (x - mean[None, :]) / std[None, :]
    design = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return LinearSupportReranker(
        coefficients=coef[1:].astype(np.float64),
        intercept=float(coef[0]),
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
        feature_names=list(FEATURE_NAMES),
        ridge_alpha=float(ridge_alpha),
    )


def score_candidate_pool(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    candidate_indices: np.ndarray,
    query_window_index: int,
    model: LinearSupportReranker,
) -> list[dict[str, Any]]:
    train_like_indices = np.arange(np.asarray(history_level).shape[0], dtype=np.int64)
    start_z = _standardized_start_states(history_level, train_like_indices)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    features = _candidate_feature_matrix(
        query_memory=query_memory,
        memory_targets=memory_targets,
        history_level=history_level,
        start_z=start_z,
        query_window_index=int(query_window_index),
        candidate_indices=candidates,
    )
    predictions = model.predict(features)
    cosines = _cosine_to_query(query_memory, memory_targets)[candidates]
    rows: list[dict[str, Any]] = []
    for pos, candidate_idx in enumerate(candidates):
        loss = replay_loss_z(
            future_delta=future_delta,
            delta_scale=delta_scale,
            query_window_index=int(query_window_index),
            candidate_window_index=int(candidate_idx),
        )
        rows.append(
            {
                "window_index": int(candidate_idx),
                "original_support_rank": int(pos + 1),
                "cosine": float(cosines[pos]),
                "learned_support_score": float(predictions[pos]),
                "true_replay_loss_z": float(loss),
                "features": {
                    name: float(value)
                    for name, value in zip(
                        model.feature_names, features[pos], strict=False
                    )
                },
            }
        )
    rows.sort(key=lambda row: float(row["learned_support_score"]), reverse=True)
    return rows


def rerank_bridge_report_with_model(
    *,
    bridge_report: dict[str, Any],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    model: LinearSupportReranker,
) -> dict[str, Any]:
    output = json.loads(json.dumps(bridge_report))
    rows = output.get("evaluation", {}).get("heldout_examples", [])
    reranked_rows = 0
    for row in rows:
        if not isinstance(row, dict) or not row.get("top_train_pool"):
            continue
        emb_idx = int(row.get("embedding_index", row["window_index"]))
        query_memory = np.asarray(condition_vectors[emb_idx], dtype=np.float32)
        pool = row.get("top_train_pool", [])
        candidate_indices = np.asarray(
            [int(item["window_index"]) for item in pool], dtype=np.int64
        )
        scored = score_candidate_pool(
            query_memory=query_memory,
            memory_targets=memory_targets,
            history_level=history_level,
            future_delta=future_delta,
            delta_scale=delta_scale,
            candidate_indices=candidate_indices,
            query_window_index=int(row["window_index"]),
            model=model,
        )
        by_idx = {int(item["window_index"]): dict(item) for item in pool}
        new_pool: list[dict[str, Any]] = []
        for scored_row in scored:
            base = by_idx[int(scored_row["window_index"])]
            base.update(scored_row)
            new_pool.append(base)
        row["top_train_pool"] = new_pool
        reranked_rows += 1
    try:
        output["summary"] = summarize_bridge_metrics(output["evaluation"])
    except KeyError:
        output["summary"] = {
            "heldout_example_count": len(rows),
            "heldout_window_count": len(
                {
                    int(row["window_index"])
                    for row in rows
                    if isinstance(row, dict) and row.get("window_index") is not None
                }
            ),
        }
    output["support_policy"] = {
        "name": "learned_support_reranker_testflight",
        "research_lane": "exploration",
        "candidate_rows_reranked": int(reranked_rows),
        "feature_names": list(model.feature_names),
        "ridge_alpha": float(model.ridge_alpha),
        "method": (
            "Linear ridge reranker trained on train-window query/candidate pairs. "
            "The label is negative standardized replay loss between historical "
            "future paths. The model reranks, but does not replace, the support pool."
        ),
    }
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


def evaluate_reranker_replay(
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
        if isinstance(row, dict) and str(row.get("role")) == str(role)
    }
    reranked_rows = {
        int(row["window_index"]): row
        for row in reranked_bridge_report.get("evaluation", {}).get(
            "heldout_examples", []
        )
        if isinstance(row, dict) and str(row.get("role")) == str(role)
    }
    common = sorted(set(original_rows) & set(reranked_rows))
    window_rows: list[dict[str, Any]] = []
    for window_index in common:
        original_pool = [
            int(item["window_index"])
            for item in original_rows[window_index].get("top_train_pool", [])
        ]
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
        "summary": _summarize_replay_eval(window_rows),
        "window_scores": window_rows,
    }


def _mean_metric(rows: list[dict[str, Any]], method: str, metric: str) -> float | None:
    values = [
        float(row["methods"][method][metric])
        for row in rows
        if row.get("methods", {}).get(method, {}).get(metric) is not None
    ]
    if not values:
        return None
    return float(np.mean(values))


def _delta(candidate: float, baseline: float, metric: str) -> float:
    if metric == "coverage_80":
        return candidate - baseline
    return baseline - candidate


def _summarize_replay_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("ensemble_crps_z", "energy_score_z", "coverage_80")
    methods = ("learned_rerank_topk_replay", "oracle_pool_topk_replay")
    summary: dict[str, Any] = {
        "baseline": "original_topk_replay",
        "window_count": len(rows),
    }
    for method in methods:
        method_summary: dict[str, Any] = {}
        for metric in metrics:
            base_mean = _mean_metric(rows, "original_topk_replay", metric)
            method_mean = _mean_metric(rows, method, metric)
            if base_mean is None or method_mean is None:
                continue
            deltas = [
                _delta(
                    float(row["methods"][method][metric]),
                    float(row["methods"]["original_topk_replay"][metric]),
                    metric,
                )
                for row in rows
                if row["methods"][method].get(metric) is not None
                and row["methods"]["original_topk_replay"].get(metric) is not None
            ]
            method_summary[metric] = {
                "baseline_mean": round(float(base_mean), 12),
                "candidate_mean": round(float(method_mean), 12),
                "mean_delta_positive_is_better": round(float(np.mean(deltas)), 12),
                "win_rate": round(float(np.mean(np.asarray(deltas) > 0.0)), 12),
            }
        summary[method] = method_summary
    return summary


def _model_payload(model: LinearSupportReranker) -> dict[str, Any]:
    return {
        "feature_names": model.feature_names,
        "coefficients": [float(value) for value in model.coefficients],
        "intercept": float(model.intercept),
        "feature_mean": [float(value) for value in model.feature_mean],
        "feature_std": [float(value) for value in model.feature_std],
        "ridge_alpha": float(model.ridge_alpha),
    }


def run_testflight(args: argparse.Namespace) -> dict[str, Any]:
    bridge_report = _load_json(args.bridge_report)
    bridge_arrays_path = Path(
        str(bridge_report.get("artifact_paths", {}).get("arrays", args.bridge_arrays))
    )
    if args.bridge_arrays:
        bridge_arrays_path = Path(args.bridge_arrays)
    bridge_arrays = _load_npz(bridge_arrays_path)
    oracle_arrays = _load_npz(args.oracle_arrays)
    memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    condition_vectors = np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32)
    history_level = np.asarray(oracle_arrays["history_level"], dtype=np.float32)
    future_delta = np.asarray(oracle_arrays["future_delta"], dtype=np.float32)
    delta_scale = np.asarray(oracle_arrays["delta_scale"], dtype=np.float32)
    train_indices = np.asarray(bridge_report["split"]["train_indices"], dtype=np.int64)

    table = build_pairwise_training_table(
        memory_targets=memory_targets,
        history_level=history_level,
        future_delta=future_delta,
        delta_scale=delta_scale,
        train_indices=train_indices,
        candidate_pool_size=int(args.candidate_pool_size),
    )
    model = fit_linear_support_reranker(
        table.features,
        table.labels,
        ridge_alpha=float(args.ridge_alpha),
    )
    reranked = rerank_bridge_report_with_model(
        bridge_report=bridge_report,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        history_level=history_level,
        future_delta=future_delta,
        delta_scale=delta_scale,
        model=model,
    )
    replay_eval = evaluate_reranker_replay(
        bridge_report=bridge_report,
        reranked_bridge_report=reranked,
        future_delta=future_delta,
        delta_scale=delta_scale,
        top_k=int(args.top_k),
        role=str(args.query_role),
    )
    learned_crps_delta = replay_eval["summary"]["learned_rerank_topk_replay"][
        "ensemble_crps_z"
    ]["mean_delta_positive_is_better"]
    learned_energy_delta = replay_eval["summary"]["learned_rerank_topk_replay"][
        "energy_score_z"
    ]["mean_delta_positive_is_better"]
    if learned_crps_delta > 0.0 or learned_energy_delta > 0.0:
        status = "mechanism_found"
    else:
        status = "dead_end"
    output_dir = Path(args.output_dir)
    report_path = output_dir / "learned_support_reranker_report.json"
    bridge_path = output_dir / "learned_support_reranked_bridge_report.json"
    model_path = output_dir / "learned_support_reranker_model.json"
    report = {
        "status": "ok",
        "research_lane": "exploration",
        "result_status": status,
        "benchmark_floor_status": "not_tested",
        "scope_note": (
            "Exploration-lane TestFlight. The learned model reranks historical "
            "support candidates but does not replace the support mixture. This "
            "report evaluates replay-quality signal only; scenario-level frozen "
            "generator evaluation is the next gate if the mechanism is positive."
        ),
        "bridge_report": str(args.bridge_report),
        "bridge_arrays": str(bridge_arrays_path),
        "oracle_arrays": str(args.oracle_arrays),
        "candidate_pool_size": int(args.candidate_pool_size),
        "top_k": int(args.top_k),
        "training_pair_count": int(table.features.shape[0]),
        "training_query_count": int(
            len(set(row["query_window_index"] for row in table.rows))
        ),
        "model": _model_payload(model),
        "replay_eval": replay_eval,
        "artifact_paths": {
            "report": str(report_path),
            "reranked_bridge_report": str(bridge_path),
            "model": str(model_path),
        },
        "decision_hint": (
            "Run scenario-level evaluation on the reranked bridge only if "
            "result_status is mechanism_found."
        ),
    }
    reranked["artifact_paths"] = {
        **reranked.get("artifact_paths", {}),
        "report": str(bridge_path),
    }
    _write_json(model_path, _model_payload(model))
    _write_json(bridge_path, reranked)
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays")
    parser.add_argument("--oracle-arrays", default=DEFAULT_ORACLE_ARRAYS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-pool-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--query-role", default="anchor")
    args = parser.parse_args()
    report = run_testflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "result_status": report["result_status"],
                "report": report["artifact_paths"]["report"],
                "reranked_bridge_report": report["artifact_paths"][
                    "reranked_bridge_report"
                ],
                "summary": report["replay_eval"]["summary"][
                    "learned_rerank_topk_replay"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
