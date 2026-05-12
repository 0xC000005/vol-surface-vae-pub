#!/usr/bin/env python
"""Learn a generator-response mixture policy for NL scenario support.

This is an exploration-lane TestFlight. It keeps the historical support mixture
and learns only how to choose among candidate support mixtures using labels from
the frozen scenario generator's own rollout scores. No OpenAI calls are made.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BASE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_CANDIDATE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_train_mixture_labels_885a_8q/mixture_label_bridge_report.json"
)
DEFAULT_SCENARIO_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_train_mixture_labels_885a_8q/scenario_eval/"
    "scenario_level_eval_report.json"
)
DEFAULT_BRIDGE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_arrays.npz"
)
DEFAULT_ORACLE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_oracle_fullheldout_786c/prefix_latent_oracle_arrays.npz"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_learned_mixture_policy_885b_testflight"
)

FEATURE_NAMES = [
    "support_cosine_mean",
    "support_cosine_min",
    "support_cosine_max",
    "support_cosine_std",
    "mixture_memory_cosine",
    "support_memory_diversity",
    "position_mean",
    "position_max",
    "start_distance_mean",
    "start_distance_min",
    "start_distance_max",
    "recent_delta_norm_mean",
    "recent_delta_norm_std",
]

ITEM_FEATURE_NAMES = [
    "support_cosine",
    "support_position",
    "query_support_memory_cosine",
    "query_support_memory_distance",
    "support_start_distance",
    "support_recent_delta_norm",
    "support_query_delta_distance",
]


@dataclass(frozen=True)
class MixturePolicyTrainingTable:
    features: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, Any]]
    feature_names: list[str]


@dataclass(frozen=True)
class SupportSetTrainingTable:
    item_features: np.ndarray
    labels: np.ndarray
    rows: list[dict[str, Any]]
    item_feature_names: list[str]


@dataclass(frozen=True)
class LinearMixturePolicy:
    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: list[str]
    ridge_alpha: float
    policy_kind: str = "linear_ridge"

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = (x - self.feature_mean[None, :]) / self.feature_std[None, :]
        return (z @ self.coefficients + float(self.intercept)).astype(np.float64)


@dataclass(frozen=True)
class PairwiseMixtureRanker:
    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: list[str]
    learning_rate: float
    epochs: int
    l2: float
    pair_count: int
    policy_kind: str = "pairwise_ranker"

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = (x - self.feature_mean[None, :]) / self.feature_std[None, :]
        return (z @ self.coefficients + float(self.intercept)).astype(np.float64)


@dataclass(frozen=True)
class SupportSetItemRanker:
    item_w1: np.ndarray
    item_b1: np.ndarray
    out_w: np.ndarray
    out_b: float
    item_feature_mean: np.ndarray
    item_feature_std: np.ndarray
    item_feature_names: list[str]
    hidden_dim: int
    learning_rate: float
    epochs: int
    l2: float
    pair_count: int
    policy_kind: str = "support_set_item_ranker"

    @property
    def feature_names(self) -> list[str]:
        return list(self.item_feature_names)

    def predict(self, item_features: np.ndarray) -> np.ndarray:
        x = np.asarray(item_features, dtype=np.float64)
        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        z = (x - self.item_feature_mean[None, None, :]) / self.item_feature_std[
            None, None, :
        ]
        hidden = np.maximum(
            0.0, np.einsum("nkd,dh->nkh", z, self.item_w1) + self.item_b1
        )
        pooled = hidden.mean(axis=1)
        return (pooled @ self.out_w + float(self.out_b)).reshape(-1).astype(np.float64)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
    return arr / denom


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), 1e-8)
    return float(np.dot(aa, bb) / denom)


def _pairwise_memory_diversity(memory_rows: np.ndarray) -> float:
    rows = _normalize_rows(np.asarray(memory_rows, dtype=np.float32))
    if rows.shape[0] <= 1:
        return 0.0
    values: list[float] = []
    for i in range(rows.shape[0]):
        for j in range(i + 1, rows.shape[0]):
            values.append(float(np.linalg.norm(rows[i] - rows[j])))
    return float(np.mean(values)) if values else 0.0


def _support_indices(row: dict[str, Any]) -> list[int]:
    raw = row.get("candidate_support_window_indices")
    if isinstance(raw, list) and raw:
        return [int(idx) for idx in raw]
    return [
        int(item["window_index"])
        for item in row.get("top_train_pool", [])
        if isinstance(item, dict)
    ]


def _support_positions(row: dict[str, Any]) -> list[int]:
    raw = row.get("candidate_mixture_positions")
    if isinstance(raw, list) and raw:
        return [int(pos) for pos in raw]
    return list(range(1, len(_support_indices(row)) + 1))


def _support_cosines(row: dict[str, Any]) -> np.ndarray:
    values = [
        float(item.get("cosine", 0.0))
        for item in row.get("top_train_pool", [])
        if isinstance(item, dict)
    ]
    if not values:
        values = [0.0]
    return np.asarray(values, dtype=np.float32)


def mixture_feature_vector(
    row: dict[str, Any],
    *,
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
) -> np.ndarray:
    """Build inference-available features for one candidate support mixture."""

    query_memory = np.asarray(
        condition_vectors[int(row["embedding_index"])],
        dtype=np.float32,
    )
    support = np.asarray(_support_indices(row), dtype=np.int64)
    if support.size == 0:
        raise ValueError("candidate row has no support indices")
    memory = np.asarray(memory_targets, dtype=np.float32)
    history = np.asarray(history_level, dtype=np.float32)
    support_memory = memory[support]
    support_starts = history[support, -1, :]
    query_start = history[int(row["window_index"]), -1, :]
    start_distances = np.linalg.norm(support_starts - query_start[None, :], axis=1)
    recent_delta = history[support, -1, :] - history[support, 0, :]
    recent_norm = np.linalg.norm(recent_delta, axis=1)
    cosines = _support_cosines(row)
    positions = np.asarray(_support_positions(row), dtype=np.float32)
    mixture_memory = np.mean(support_memory, axis=0)
    values = [
        float(np.mean(cosines)),
        float(np.min(cosines)),
        float(np.max(cosines)),
        float(np.std(cosines)),
        _cosine(query_memory, mixture_memory),
        _pairwise_memory_diversity(support_memory),
        float(np.mean(positions)),
        float(np.max(positions)),
        float(np.mean(start_distances)),
        float(np.min(start_distances)),
        float(np.max(start_distances)),
        float(np.mean(recent_norm)),
        float(np.std(recent_norm)),
    ]
    return np.asarray(values, dtype=np.float32)


def support_item_feature_matrix(
    row: dict[str, Any],
    *,
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
) -> np.ndarray:
    """Build per-support item features for one candidate support mixture."""

    query_memory = np.asarray(
        condition_vectors[int(row["embedding_index"])],
        dtype=np.float32,
    )
    support = np.asarray(_support_indices(row), dtype=np.int64)
    positions = np.asarray(_support_positions(row), dtype=np.float32)
    cosines = _support_cosines(row).astype(np.float32)
    if support.size == 0:
        raise ValueError("candidate row has no support indices")
    if positions.size != support.size:
        positions = np.arange(1, support.size + 1, dtype=np.float32)
    if cosines.size != support.size:
        cosines = np.resize(cosines, support.size).astype(np.float32)
    memory = np.asarray(memory_targets, dtype=np.float32)
    history = np.asarray(history_level, dtype=np.float32)
    support_memory = memory[support]
    support_starts = history[support, -1, :]
    query_history = history[int(row["window_index"])]
    query_start = query_history[-1, :]
    query_recent_delta = query_history[-1, :] - query_history[0, :]
    support_recent_delta = history[support, -1, :] - history[support, 0, :]
    memory_cosines = np.asarray(
        [_cosine(query_memory, item) for item in support_memory],
        dtype=np.float32,
    )
    memory_distances = np.linalg.norm(
        support_memory - query_memory[None, :],
        axis=1,
    ).astype(np.float32)
    start_distances = np.linalg.norm(
        support_starts - query_start[None, :],
        axis=1,
    ).astype(np.float32)
    recent_norm = np.linalg.norm(support_recent_delta, axis=1).astype(np.float32)
    delta_distance = np.linalg.norm(
        support_recent_delta - query_recent_delta[None, :],
        axis=1,
    ).astype(np.float32)
    return np.stack(
        [
            cosines,
            positions,
            memory_cosines,
            memory_distances,
            start_distances,
            recent_norm,
            delta_distance,
        ],
        axis=1,
    ).astype(np.float32)


def _score_lookup(scenario_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in scenario_report.get("window_scores", []):
        if isinstance(row, dict) and row.get("query_id"):
            lookup[str(row["query_id"])] = row
    return lookup


def _metric(row: dict[str, Any], method: str, metric: str) -> float | None:
    value = row.get("methods", {}).get(method, {}).get(metric)
    if value is None:
        return None
    raw = float(value)
    return raw if np.isfinite(raw) else None


def build_mixture_policy_training_table(
    *,
    candidate_bridge: dict[str, Any],
    scenario_report: dict[str, Any],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    method: str = "narrative_generator_topk",
    metric: str = "energy_score_z",
) -> MixturePolicyTrainingTable:
    """Build candidate-mixture rows labeled by frozen-generator rollout score."""

    scores = _score_lookup(scenario_report)
    features: list[np.ndarray] = []
    labels: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in candidate_bridge.get("evaluation", {}).get("heldout_examples", []):
        if not isinstance(row, dict) or not row.get("query_id"):
            continue
        score = scores.get(str(row["query_id"]))
        if score is None:
            continue
        value = _metric(score, method, metric)
        if value is None:
            continue
        feature = mixture_feature_vector(
            row,
            condition_vectors=condition_vectors,
            memory_targets=memory_targets,
            history_level=history_level,
        )
        features.append(feature)
        labels.append(-float(value))
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "window_index": int(row["window_index"]),
                "support_window_indices": _support_indices(row),
                f"generator_{metric}": float(value),
            }
        )
    if not features:
        raise ValueError("no labeled candidate-mixture rows built")
    return MixturePolicyTrainingTable(
        features=np.stack(features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        rows=rows,
        feature_names=list(FEATURE_NAMES),
    )


def build_support_set_training_table(
    *,
    candidate_bridge: dict[str, Any],
    scenario_report: dict[str, Any],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    method: str = "narrative_generator_topk",
    metric: str = "energy_score_z",
) -> SupportSetTrainingTable:
    """Build candidate-mixture rows as per-support-item feature sets."""

    scores = _score_lookup(scenario_report)
    item_features: list[np.ndarray] = []
    labels: list[float] = []
    rows: list[dict[str, Any]] = []
    support_count: int | None = None
    for row in candidate_bridge.get("evaluation", {}).get("heldout_examples", []):
        if not isinstance(row, dict) or not row.get("query_id"):
            continue
        score = scores.get(str(row["query_id"]))
        if score is None:
            continue
        value = _metric(score, method, metric)
        if value is None:
            continue
        feature = support_item_feature_matrix(
            row,
            condition_vectors=condition_vectors,
            memory_targets=memory_targets,
            history_level=history_level,
        )
        if support_count is None:
            support_count = int(feature.shape[0])
        if int(feature.shape[0]) != support_count:
            raise ValueError("support-set ranker requires fixed support count per run")
        item_features.append(feature)
        labels.append(-float(value))
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "window_index": int(row["window_index"]),
                "support_window_indices": _support_indices(row),
                f"generator_{metric}": float(value),
            }
        )
    if not item_features:
        raise ValueError("no labeled support-set rows built")
    return SupportSetTrainingTable(
        item_features=np.stack(item_features).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        rows=rows,
        item_feature_names=list(ITEM_FEATURE_NAMES),
    )


def fit_linear_mixture_policy(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
) -> LinearMixturePolicy:
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
    return LinearMixturePolicy(
        coefficients=coef[1:].astype(np.float64),
        intercept=float(coef[0]),
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
        feature_names=list(FEATURE_NAMES),
        ridge_alpha=float(ridge_alpha),
    )


def _pairwise_differences(
    features: np.ndarray,
    labels: np.ndarray,
    query_ids: list[Any],
    *,
    min_label_gap: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] != len(query_ids):
        raise ValueError("features, labels, and query_ids must have compatible shapes")
    groups: dict[str, list[int]] = {}
    for idx, query_id in enumerate(query_ids):
        groups.setdefault(str(query_id), []).append(idx)
    diffs: list[np.ndarray] = []
    for indices in groups.values():
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = indices[left_pos]
                right = indices[right_pos]
                gap = float(y[left] - y[right])
                if abs(gap) <= float(min_label_gap):
                    continue
                if gap > 0:
                    diffs.append(x[left] - x[right])
                else:
                    diffs.append(x[right] - x[left])
    if not diffs:
        raise ValueError("no within-query label preferences found")
    return np.stack(diffs).astype(np.float64)


def fit_pairwise_mixture_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    query_ids: list[Any],
    feature_names: list[str] | None = None,
    learning_rate: float = 0.05,
    epochs: int = 500,
    l2: float = 1e-3,
    min_label_gap: float = 1e-8,
) -> PairwiseMixtureRanker:
    """Fit a within-query pairwise logistic ranker for candidate mixtures."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must have compatible shapes")
    if x.shape[0] != len(query_ids):
        raise ValueError("query_ids length must match feature rows")
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-8)
    z = (x - mean[None, :]) / std[None, :]
    diffs = _pairwise_differences(
        z,
        y,
        query_ids,
        min_label_gap=float(min_label_gap),
    )
    weights = np.zeros(z.shape[1], dtype=np.float64)
    lr = float(learning_rate)
    for _ in range(int(epochs)):
        margins = diffs @ weights
        margins = np.clip(margins, -60.0, 60.0)
        preference_error = 1.0 / (1.0 + np.exp(margins))
        grad = -(preference_error[:, None] * diffs).mean(axis=0)
        grad += float(l2) * weights
        weights -= lr * grad
    names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
    return PairwiseMixtureRanker(
        coefficients=weights.astype(np.float64),
        intercept=0.0,
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
        feature_names=names,
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        l2=float(l2),
        pair_count=int(diffs.shape[0]),
    )


def _preference_pairs(
    labels: np.ndarray,
    query_ids: list[Any],
    *,
    min_label_gap: float = 1e-8,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if y.shape[0] != len(query_ids):
        raise ValueError("labels and query_ids must have compatible shapes")
    groups: dict[str, list[int]] = {}
    for idx, query_id in enumerate(query_ids):
        groups.setdefault(str(query_id), []).append(idx)
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = indices[left_pos]
                right = indices[right_pos]
                gap = float(y[left] - y[right])
                if abs(gap) <= float(min_label_gap):
                    continue
                if gap > 0:
                    pairs.append((left, right))
                else:
                    pairs.append((right, left))
    if not pairs:
        raise ValueError("no within-query label preferences found")
    return np.asarray(pairs, dtype=np.int64)


def fit_support_set_item_ranker(
    item_features: np.ndarray,
    labels: np.ndarray,
    *,
    query_ids: list[Any],
    item_feature_names: list[str] | None = None,
    hidden_dim: int = 8,
    learning_rate: float = 0.01,
    epochs: int = 500,
    l2: float = 1e-4,
    min_label_gap: float = 1e-8,
    seed: int = 0,
) -> SupportSetItemRanker:
    """Fit a small DeepSets-style scorer over candidate support items."""

    x = np.asarray(item_features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 3:
        raise ValueError("item_features must have shape [rows, support, features]")
    if x.shape[0] != y.shape[0] or x.shape[0] != len(query_ids):
        raise ValueError("item_features, labels, and query_ids must match")
    if int(hidden_dim) <= 0:
        raise ValueError("hidden_dim must be positive")
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-8)
    z = ((x - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    pairs = _preference_pairs(
        y,
        query_ids,
        min_label_gap=float(min_label_gap),
    )

    torch.manual_seed(int(seed))
    features_t = torch.as_tensor(z, dtype=torch.float32)
    preferred_t = torch.as_tensor(pairs[:, 0], dtype=torch.long)
    worse_t = torch.as_tensor(pairs[:, 1], dtype=torch.long)
    item_layer = torch.nn.Linear(int(x.shape[-1]), int(hidden_dim))
    out_layer = torch.nn.Linear(int(hidden_dim), 1)
    optimizer = torch.optim.Adam(
        [*item_layer.parameters(), *out_layer.parameters()],
        lr=float(learning_rate),
    )
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.relu(item_layer(features_t))
        pooled = hidden.mean(dim=1)
        scores = out_layer(pooled).squeeze(-1)
        margin = scores[preferred_t] - scores[worse_t]
        rank_loss = torch.nn.functional.softplus(-margin).mean()
        penalty = torch.zeros((), dtype=torch.float32)
        for param in [*item_layer.parameters(), *out_layer.parameters()]:
            penalty = penalty + torch.sum(param * param)
        loss = rank_loss + float(l2) * penalty
        loss.backward()
        optimizer.step()

    names = (
        list(item_feature_names)
        if item_feature_names is not None
        else list(ITEM_FEATURE_NAMES)
    )
    return SupportSetItemRanker(
        item_w1=item_layer.weight.detach().cpu().numpy().T.astype(np.float64),
        item_b1=item_layer.bias.detach().cpu().numpy().astype(np.float64),
        out_w=out_layer.weight.detach().cpu().numpy().reshape(-1).astype(np.float64),
        out_b=float(out_layer.bias.detach().cpu().numpy().reshape(-1)[0]),
        item_feature_mean=mean.astype(np.float64),
        item_feature_std=std.astype(np.float64),
        item_feature_names=names,
        hidden_dim=int(hidden_dim),
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        l2=float(l2),
        pair_count=int(pairs.shape[0]),
    )


def _group_candidate_rows(
    candidate_bridge: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_bridge.get("evaluation", {}).get("heldout_examples", []):
        if isinstance(row, dict) and row.get("window_index") is not None:
            groups.setdefault(int(row["window_index"]), []).append(row)
    return groups


def rerank_bridge_report_with_mixture_policy(
    *,
    bridge_report: dict[str, Any],
    candidate_bridge: dict[str, Any],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    model: LinearMixturePolicy | PairwiseMixtureRanker | SupportSetItemRanker,
) -> dict[str, Any]:
    """Replace each query's support pool with the best predicted support mixture."""

    output = json.loads(json.dumps(bridge_report))
    groups = _group_candidate_rows(candidate_bridge)
    rows = output.get("evaluation", {}).get("heldout_examples", [])
    changed = 0
    for row in rows:
        if not isinstance(row, dict) or int(row.get("window_index", -1)) not in groups:
            continue
        candidates = groups[int(row["window_index"])]
        scored: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            if getattr(model, "policy_kind", "") == "support_set_item_ranker":
                feature = support_item_feature_matrix(
                    candidate,
                    condition_vectors=condition_vectors,
                    memory_targets=memory_targets,
                    history_level=history_level,
                )
            else:
                feature = mixture_feature_vector(
                    candidate,
                    condition_vectors=condition_vectors,
                    memory_targets=memory_targets,
                    history_level=history_level,
                )
            scored.append((float(model.predict(feature)[0]), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        score, best = scored[0]
        row["top_train_pool"] = json.loads(json.dumps(best["top_train_pool"]))
        row["support_policy"] = {
            "name": "learned_generator_response_mixture_policy",
            "policy_kind": str(getattr(model, "policy_kind", "unknown")),
            "selected_query_id": str(best.get("query_id", "")),
            "selected_score": float(score),
            "candidate_count": len(candidates),
            "selected_support_positions": _support_positions(best),
            "selected_support_window_indices": _support_indices(best),
        }
        changed += 1
    output["mixture_policy"] = {
        "name": "learned_generator_response_mixture_policy",
        "research_lane": "exploration",
        "candidate_rows_reranked": int(changed),
        "feature_names": list(model.feature_names),
        "policy_kind": str(getattr(model, "policy_kind", "unknown")),
        "ridge_alpha": (
            None
            if getattr(model, "ridge_alpha", None) is None
            else float(getattr(model, "ridge_alpha"))
        ),
        "method": (
            "Policy trained on candidate support mixtures labeled by "
            "frozen-generator rollout score. It selects a support mixture; "
            "it does not replace the historical support store."
        ),
    }
    return output


def _model_payload(
    model: LinearMixturePolicy | PairwiseMixtureRanker | SupportSetItemRanker,
) -> dict[str, Any]:
    if isinstance(model, SupportSetItemRanker):
        payload = {
            "policy_kind": model.policy_kind,
            "feature_names": model.feature_names,
            "item_feature_names": model.item_feature_names,
            "item_w1": model.item_w1.astype(float).tolist(),
            "item_b1": [float(value) for value in model.item_b1],
            "out_w": [float(value) for value in model.out_w],
            "out_b": float(model.out_b),
            "item_feature_mean": [float(value) for value in model.item_feature_mean],
            "item_feature_std": [float(value) for value in model.item_feature_std],
            "hidden_dim": int(model.hidden_dim),
            "learning_rate": float(model.learning_rate),
            "epochs": int(model.epochs),
            "l2": float(model.l2),
            "pair_count": int(model.pair_count),
        }
        return payload
    payload = {
        "policy_kind": str(getattr(model, "policy_kind", "unknown")),
        "feature_names": model.feature_names,
        "coefficients": [float(value) for value in model.coefficients],
        "intercept": float(model.intercept),
        "feature_mean": [float(value) for value in model.feature_mean],
        "feature_std": [float(value) for value in model.feature_std],
    }
    if getattr(model, "ridge_alpha", None) is not None:
        payload["ridge_alpha"] = float(getattr(model, "ridge_alpha"))
    if getattr(model, "learning_rate", None) is not None:
        payload["learning_rate"] = float(getattr(model, "learning_rate"))
    if getattr(model, "epochs", None) is not None:
        payload["epochs"] = int(getattr(model, "epochs"))
    if getattr(model, "l2", None) is not None:
        payload["l2"] = float(getattr(model, "l2"))
    if getattr(model, "pair_count", None) is not None:
        payload["pair_count"] = int(getattr(model, "pair_count"))
    return payload


def run_testflight(args: argparse.Namespace) -> dict[str, Any]:
    base_bridge = _load_json(args.base_bridge_report)
    train_candidate_bridge = _load_json(args.candidate_bridge_report)
    rerank_candidate_bridge = _load_json(
        args.rerank_candidate_bridge_report or args.candidate_bridge_report
    )
    scenario_report = _load_json(args.scenario_report)
    bridge_arrays = _load_npz(args.bridge_arrays)
    oracle_arrays = _load_npz(args.oracle_arrays)
    condition_vectors = np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32)
    memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    history_level = np.asarray(oracle_arrays["history_level"], dtype=np.float32)
    if str(args.policy_kind) == "linear":
        table = build_mixture_policy_training_table(
            candidate_bridge=train_candidate_bridge,
            scenario_report=scenario_report,
            condition_vectors=condition_vectors,
            memory_targets=memory_targets,
            history_level=history_level,
            method=str(args.method),
            metric=str(args.metric),
        )
        model: LinearMixturePolicy | PairwiseMixtureRanker = fit_linear_mixture_policy(
            table.features,
            table.labels,
            ridge_alpha=float(args.ridge_alpha),
        )
        training_row_count = int(table.features.shape[0])
        training_rows = table.rows
        training_labels = table.labels
    elif str(args.policy_kind) == "pairwise":
        table = build_mixture_policy_training_table(
            candidate_bridge=train_candidate_bridge,
            scenario_report=scenario_report,
            condition_vectors=condition_vectors,
            memory_targets=memory_targets,
            history_level=history_level,
            method=str(args.method),
            metric=str(args.metric),
        )
        model = fit_pairwise_mixture_ranker(
            table.features,
            table.labels,
            query_ids=[row["window_index"] for row in table.rows],
            feature_names=table.feature_names,
            learning_rate=float(args.pairwise_learning_rate),
            epochs=int(args.pairwise_epochs),
            l2=float(args.pairwise_l2),
            min_label_gap=float(args.pairwise_min_label_gap),
        )
        training_row_count = int(table.features.shape[0])
        training_rows = table.rows
        training_labels = table.labels
    elif str(args.policy_kind) == "support_set":
        support_table = build_support_set_training_table(
            candidate_bridge=train_candidate_bridge,
            scenario_report=scenario_report,
            condition_vectors=condition_vectors,
            memory_targets=memory_targets,
            history_level=history_level,
            method=str(args.method),
            metric=str(args.metric),
        )
        model = fit_support_set_item_ranker(
            support_table.item_features,
            support_table.labels,
            query_ids=[row["window_index"] for row in support_table.rows],
            item_feature_names=support_table.item_feature_names,
            hidden_dim=int(args.set_hidden_dim),
            learning_rate=float(args.set_learning_rate),
            epochs=int(args.set_epochs),
            l2=float(args.set_l2),
            min_label_gap=float(args.set_min_label_gap),
            seed=int(args.set_seed),
        )
        training_row_count = int(support_table.item_features.shape[0])
        training_rows = support_table.rows
        training_labels = support_table.labels
    else:
        raise ValueError(f"unsupported policy kind: {args.policy_kind}")
    reranked = rerank_bridge_report_with_mixture_policy(
        bridge_report=base_bridge,
        candidate_bridge=rerank_candidate_bridge,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        history_level=history_level,
        model=model,
    )
    output_dir = Path(args.output_dir)
    report_path = output_dir / "learned_mixture_policy_report.json"
    bridge_path = output_dir / "learned_mixture_policy_bridge_report.json"
    model_path = output_dir / "learned_mixture_policy_model.json"
    reranked["artifact_paths"] = {
        **reranked.get("artifact_paths", {}),
        "report": str(bridge_path),
    }
    report = {
        "status": "ok",
        "research_lane": "exploration",
        "result_status": "candidate_policy_built",
        "benchmark_floor_status": "not_tested",
        "scope_note": (
            "Exploration-lane mixture policy trained on frozen-generator "
            "rollout-response labels from train-window candidate mixtures. "
            "Scenario-level held-out evaluation is required before promotion."
        ),
        "base_bridge_report": str(args.base_bridge_report),
        "candidate_bridge_report": str(args.candidate_bridge_report),
        "rerank_candidate_bridge_report": str(
            args.rerank_candidate_bridge_report or args.candidate_bridge_report
        ),
        "scenario_report": str(args.scenario_report),
        "bridge_arrays": str(args.bridge_arrays),
        "oracle_arrays": str(args.oracle_arrays),
        "method": str(args.method),
        "metric": str(args.metric),
        "policy_kind": str(args.policy_kind),
        "training_row_count": int(training_row_count),
        "training_query_count": int(
            len({int(row["window_index"]) for row in training_rows})
        ),
        "label_mean": float(np.mean(training_labels)),
        "label_std": float(np.std(training_labels)),
        "model": _model_payload(model),
        "artifact_paths": {
            "report": str(report_path),
            "reranked_bridge_report": str(bridge_path),
            "model": str(model_path),
        },
    }
    _write_json(model_path, _model_payload(model))
    _write_json(bridge_path, reranked)
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bridge-report", default=DEFAULT_BASE_BRIDGE_REPORT)
    parser.add_argument(
        "--candidate-bridge-report",
        default=DEFAULT_CANDIDATE_BRIDGE_REPORT,
    )
    parser.add_argument("--rerank-candidate-bridge-report")
    parser.add_argument("--scenario-report", default=DEFAULT_SCENARIO_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--oracle-arrays", default=DEFAULT_ORACLE_ARRAYS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", default="narrative_generator_topk")
    parser.add_argument("--metric", default="energy_score_z")
    parser.add_argument(
        "--policy-kind",
        choices=["linear", "pairwise", "support_set"],
        default="linear",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--pairwise-learning-rate", type=float, default=0.05)
    parser.add_argument("--pairwise-epochs", type=int, default=500)
    parser.add_argument("--pairwise-l2", type=float, default=1e-3)
    parser.add_argument("--pairwise-min-label-gap", type=float, default=1e-8)
    parser.add_argument("--set-hidden-dim", type=int, default=8)
    parser.add_argument("--set-learning-rate", type=float, default=0.01)
    parser.add_argument("--set-epochs", type=int, default=500)
    parser.add_argument("--set-l2", type=float, default=1e-4)
    parser.add_argument("--set-min-label-gap", type=float, default=1e-8)
    parser.add_argument("--set-seed", type=int, default=0)
    args = parser.parse_args()
    report = run_testflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "reranked_bridge_report": report["artifact_paths"][
                    "reranked_bridge_report"
                ],
                "training_row_count": report["training_row_count"],
                "training_query_count": report["training_query_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
