#!/usr/bin/env python
"""Offline analogue-mixture prior evaluator for narrative scenario support.

This script makes no OpenAI calls. It tests the new main direction before
training a residual bridge: retrieve several historical analogue prefixes,
build soft mixtures, and ask whether the mixture prior is more aligned with the
grounded story implications than top-1 analogue or generated-rollout baselines.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KEY_FACTOR_NAMES,
)
from experiments.backfill.block_ar.nl_prefix_latent_market_alignment import (  # noqa: E402
    market_implication_alignment,
)


DEFAULT_CASEBOOK_SUMMARY = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_casebook_800b_alignment/"
    "gradio_live_casebook_summary.json"
)
DEFAULT_ORACLE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_oracle_fullheldout_786c/"
    "prefix_latent_oracle_arrays.npz"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_analogue_mixture_prior_804b"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_float_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != int(ndim):
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr.shape}")
    return arr


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    score = np.asarray(values, dtype=np.float64)
    temp = max(float(temperature), 1e-6)
    shifted = (score - float(np.max(score))) / temp
    weight = np.exp(shifted)
    denom = float(np.sum(weight))
    if not math.isfinite(denom) or denom <= 0.0:
        return np.ones(score.shape[0], dtype=np.float32) / float(score.shape[0])
    return (weight / denom).astype(np.float32)


def _cosine_to_query(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    t = _as_float_array(targets, name="targets", ndim=2)
    if q.shape[1] != t.shape[1]:
        raise ValueError(
            f"query dim {q.shape[1]} does not match target dim {t.shape[1]}"
        )
    denom = np.maximum(np.linalg.norm(t, axis=1) * np.linalg.norm(q), 1e-8)
    return (np.sum(t * q, axis=1) / denom).astype(np.float32)


def _safe_start_z(start_state: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    start = _as_float_array(start_state, name="start_state", ndim=2)
    fit = np.asarray(fit_indices, dtype=np.int64)
    mean = start[fit].mean(axis=0, keepdims=True)
    std = np.maximum(start[fit].std(axis=0, keepdims=True), 1e-6)
    return ((start - mean) / std).astype(np.float32)


def start_distances_to_query_start(
    *,
    start_state: np.ndarray,
    train_indices: np.ndarray,
    query_window_index: int,
    query_start_state: np.ndarray | None = None,
) -> np.ndarray:
    """Measure train-start distances to the fixed level used for conditioning."""

    start = _as_float_array(start_state, name="start_state", ndim=2)
    train = np.asarray(train_indices, dtype=np.int64)
    if train.size == 0:
        raise ValueError("train_indices must be non-empty")
    mean = start[train].mean(axis=0, keepdims=True)
    std = np.maximum(start[train].std(axis=0, keepdims=True), 1e-6)
    train_z = (start[train] - mean) / std
    if query_start_state is None:
        query_idx = int(query_window_index)
        if query_idx < 0 or query_idx >= start.shape[0]:
            raise IndexError(f"query_window_index {query_idx} outside {start.shape[0]}")
        query = start[query_idx]
    else:
        query = np.asarray(query_start_state, dtype=np.float32).reshape(-1)
        if query.shape != (start.shape[1],):
            raise ValueError(f"query_start_state must have shape [{start.shape[1]}]")
    query_z = (query[None, :] - mean) / std
    return np.linalg.norm(train_z - query_z, axis=1).astype(np.float32)


def load_state_spec_names(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> list[str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    specs = payload.get("state_specs", [])
    names = [str(item.get("name", "")) for item in specs if isinstance(item, dict)]
    if len(names) < 1:
        raise ValueError(f"{checkpoint}: missing state_specs")
    return names


def prefix_terminal_rows(
    *,
    history_level: np.ndarray,
    window_index: int,
    spec_names: list[str],
) -> list[dict[str, Any]]:
    history = _as_float_array(history_level, name="history_level", ndim=3)
    idx = int(window_index)
    if idx < 0 or idx >= history.shape[0]:
        raise IndexError(f"window_index {idx} outside {history.shape[0]}")
    terminal_delta = history[idx, -1, :] - history[idx, 0, :]
    rows = [
        {
            "Market": "IV_SURFACE",
            "Mean Terminal Delta": float(np.nanmean(terminal_delta[:25])),
        }
    ]
    index = {name: col for col, name in enumerate(spec_names)}
    for market, spec_name in KEY_FACTOR_NAMES.items():
        col = index.get(spec_name)
        if col is None:
            continue
        rows.append(
            {
                "Market": market,
                "Mean Terminal Delta": float(terminal_delta[col]),
            }
        )
    return rows


def weighted_prefix_terminal_rows(
    *,
    history_level: np.ndarray,
    window_indices: np.ndarray,
    weights: np.ndarray,
    spec_names: list[str],
) -> list[dict[str, Any]]:
    history = _as_float_array(history_level, name="history_level", ndim=3)
    idx = np.asarray(window_indices, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if idx.size == 0:
        raise ValueError("window_indices must be non-empty")
    if idx.shape[0] != w.shape[0]:
        raise ValueError("weights must match window_indices")
    w = w / max(float(np.sum(w)), 1e-8)
    terminal_delta = history[idx, -1, :] - history[idx, 0, :]
    mixed_delta = np.sum(terminal_delta * w[:, None], axis=0)
    rows = [
        {
            "Market": "IV_SURFACE",
            "Mean Terminal Delta": float(np.nanmean(mixed_delta[:25])),
        }
    ]
    index = {name: col for col, name in enumerate(spec_names)}
    for market, spec_name in KEY_FACTOR_NAMES.items():
        col = index.get(spec_name)
        if col is None:
            continue
        rows.append(
            {
                "Market": market,
                "Mean Terminal Delta": float(mixed_delta[col]),
            }
        )
    return rows


def _alignment_score(alignment: dict[str, Any]) -> float:
    checked = int(alignment.get("checked_count", 0) or 0)
    if checked <= 0:
        return 0.0
    matches = float(alignment.get("match_count", 0) or 0)
    mismatches = float(alignment.get("mismatch_count", 0) or 0)
    return float((matches - mismatches) / checked)


def candidate_support_table(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    train_indices: np.ndarray,
    query_window_index: int,
    query_start_state: np.ndarray | None = None,
    grounding: dict[str, Any],
    spec_names: list[str],
    start_distance_threshold_z: float,
    start_distance_penalty: float,
    implication_alignment_weight: float,
) -> list[dict[str, Any]]:
    train = np.asarray(train_indices, dtype=np.int64)
    if train.size == 0:
        raise ValueError("train_indices must be non-empty")
    memory = _as_float_array(memory_targets, name="memory_targets", ndim=2)
    start_state = _as_float_array(history_level, name="history_level", ndim=3)[:, -1, :]
    cosines = _cosine_to_query(query_memory, memory)[train]
    distances = start_distances_to_query_start(
        start_state=start_state,
        train_indices=train,
        query_window_index=int(query_window_index),
        query_start_state=query_start_state,
    )
    rows: list[dict[str, Any]] = []
    for pos, window_idx in enumerate(train):
        terminal_rows = prefix_terminal_rows(
            history_level=history_level,
            window_index=int(window_idx),
            spec_names=spec_names,
        )
        alignment = market_implication_alignment(
            grounding=grounding,
            scenario_rows=terminal_rows,
        )
        recent_score = _alignment_score(alignment)
        excess_distance = max(
            float(distances[pos]) - float(start_distance_threshold_z),
            0.0,
        )
        start_distance_cost = float(start_distance_penalty) * float(distances[pos])
        excess_distance_cost = float(start_distance_penalty) * excess_distance
        narrative_start_score = (
            float(cosines[pos]) - start_distance_cost - excess_distance_cost
        )
        combined_score = (
            narrative_start_score + float(implication_alignment_weight) * recent_score
        )
        rows.append(
            {
                "window_index": int(window_idx),
                "memory_support_cosine": float(cosines[pos]),
                "start_distance_z": float(distances[pos]),
                "start_only_score": float(-distances[pos]),
                "start_distance_cost": float(start_distance_cost),
                "excess_start_distance_cost": float(excess_distance_cost),
                "narrative_start_score": float(narrative_start_score),
                "recent_prefix_alignment_score": float(recent_score),
                "recent_prefix_checked": int(alignment.get("checked_count", 0) or 0),
                "recent_prefix_match_count": int(alignment.get("match_count", 0) or 0),
                "recent_prefix_mismatches": int(
                    alignment.get("mismatch_count", 0) or 0
                ),
                "recent_prefix_match_rate": (
                    float(alignment.get("match_count", 0) or 0)
                    / float(alignment.get("checked_count", 1) or 1)
                ),
                "recent_prefix_alignment_status": str(alignment.get("status", "")),
                "recent_prefix_alignment": alignment,
                "combined_score": float(combined_score),
            }
        )
    return rows


def _top_indices(rows: list[dict[str, Any]], key: str, k: int) -> np.ndarray:
    ordered = sorted(
        range(len(rows)),
        key=lambda pos: float(rows[pos].get(key, -1e9)),
        reverse=True,
    )
    return np.asarray(
        [int(rows[pos]["window_index"]) for pos in ordered[: max(int(k), 1)]],
        dtype=np.int64,
    )


def _scores_for_indices(
    rows: list[dict[str, Any]],
    window_indices: np.ndarray,
    key: str,
) -> np.ndarray:
    by_idx = {int(row["window_index"]): row for row in rows}
    return np.asarray(
        [float(by_idx[int(idx)].get(key, 0.0)) for idx in window_indices],
        dtype=np.float32,
    )


def direction_passing_top_indices(
    rows: list[dict[str, Any]],
    *,
    key: str,
    k: int,
) -> np.ndarray:
    """Select by narrative/start score after applying a hard direction gate."""

    ordered = sorted(
        rows,
        key=lambda row: float(row.get(key, -1e9)),
        reverse=True,
    )
    passing = [
        row
        for row in ordered
        if int(row.get("recent_prefix_checked", 0) or 0) > 0
        and int(row.get("recent_prefix_mismatches", 0) or 0) == 0
    ]
    source = passing if passing else ordered
    return np.asarray(
        [int(row["window_index"]) for row in source[: max(int(k), 1)]],
        dtype=np.int64,
    )


def _direction_checked_source(
    rows: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: float(row.get(key, -1e9)),
        reverse=True,
    )
    passing = [
        row
        for row in ordered
        if int(row.get("recent_prefix_checked", 0) or 0) > 0
        and int(row.get("recent_prefix_mismatches", 0) or 0) == 0
    ]
    return passing if passing else ordered


def cohesive_top_indices(
    rows: list[dict[str, Any]],
    *,
    memory_targets: np.ndarray,
    key: str,
    k: int,
    min_index_gap: int = 0,
) -> np.ndarray:
    """Select the top narrative hit plus its closest latent family members."""

    source = _direction_checked_source(rows, key=key)
    if not source:
        return np.asarray([], dtype=np.int64)
    memory = _as_float_array(memory_targets, name="memory_targets", ndim=2)
    anchor_idx = int(source[0]["window_index"])
    selected = [anchor_idx]
    remaining = [row for row in source[1:] if int(row["window_index"]) != anchor_idx]
    if remaining:
        remaining_indices = np.asarray(
            [int(row["window_index"]) for row in remaining],
            dtype=np.int64,
        )
        anchor_cosine = _cosine_to_query(memory[anchor_idx], memory[remaining_indices])
        scored_remaining = sorted(
            zip(remaining, anchor_cosine, strict=True),
            key=lambda item: (
                float(item[1]),
                float(item[0].get(key, -1e9)),
            ),
            reverse=True,
        )
        for row, _cosine in scored_remaining:
            idx = int(row["window_index"])
            if any(
                abs(idx - selected_idx) < int(min_index_gap)
                for selected_idx in selected
            ):
                continue
            selected.append(idx)
            if len(selected) >= int(k):
                break
    return np.asarray(selected[: max(int(k), 1)], dtype=np.int64)


def latent_family_top_indices(
    rows: list[dict[str, Any]],
    *,
    memory_targets: np.ndarray,
    key: str,
    k: int,
    min_family_cosine: float,
    min_index_gap: int = 0,
) -> np.ndarray:
    """Select the highest scoring coherent latent family, not isolated hits."""

    source = _direction_checked_source(rows, key=key)
    if not source:
        return np.asarray([], dtype=np.int64)
    memory = _as_float_array(memory_targets, name="memory_targets", ndim=2)
    threshold = float(np.clip(min_family_cosine, -1.0, 1.0))
    best_family: list[dict[str, Any]] = []
    best_key: tuple[float, int, float] | None = None
    for anchor in source:
        anchor_idx = int(anchor["window_index"])
        source_indices = np.asarray(
            [int(row["window_index"]) for row in source],
            dtype=np.int64,
        )
        cosines = _cosine_to_query(memory[anchor_idx], memory[source_indices])
        family_candidates = [
            row
            for row, cosine in zip(source, cosines, strict=True)
            if float(cosine) >= threshold
        ]
        family_ranked = sorted(
            family_candidates,
            key=lambda row: float(row.get(key, -1e9)),
            reverse=True,
        )
        family: list[dict[str, Any]] = []
        for row in family_ranked:
            idx = int(row["window_index"])
            if any(
                abs(idx - int(selected["window_index"])) < int(min_index_gap)
                for selected in family
            ):
                continue
            family.append(row)
            if len(family) >= int(k):
                break
        if not family:
            continue
        score_sum = float(sum(float(row.get(key, -1e9)) for row in family))
        top_score = float(family[0].get(key, -1e9))
        family_key = (score_sum, len(family), top_score)
        if best_key is None or family_key > best_key:
            best_key = family_key
            best_family = family
    return np.asarray(
        [int(row["window_index"]) for row in best_family[: max(int(k), 1)]],
        dtype=np.int64,
    )


def diverse_top_indices(
    rows: list[dict[str, Any]],
    *,
    memory_targets: np.ndarray,
    key: str,
    k: int,
    max_pairwise_cosine: float,
    min_index_gap: int = 0,
) -> np.ndarray:
    ordered = sorted(
        rows,
        key=lambda row: float(row.get(key, -1e9)),
        reverse=True,
    )
    memory = _as_float_array(memory_targets, name="memory_targets", ndim=2)
    selected: list[int] = []
    for row in ordered:
        idx = int(row["window_index"])
        if any(
            abs(idx - selected_idx) < int(min_index_gap) for selected_idx in selected
        ):
            continue
        if not selected:
            selected.append(idx)
        else:
            selected_memory = memory[np.asarray(selected, dtype=np.int64)]
            cosine = _cosine_to_query(memory[idx], selected_memory)
            if float(np.max(cosine)) <= float(max_pairwise_cosine):
                selected.append(idx)
        if len(selected) >= int(k):
            break
    if len(selected) < int(k):
        for row in ordered:
            idx = int(row["window_index"])
            if idx not in selected and not any(
                abs(idx - selected_idx) < int(min_index_gap)
                for selected_idx in selected
            ):
                selected.append(idx)
            if len(selected) >= int(k):
                break
    return np.asarray(selected[: max(int(k), 1)], dtype=np.int64)


def direction_passing_diverse_top_indices(
    rows: list[dict[str, Any]],
    *,
    memory_targets: np.ndarray,
    key: str,
    k: int,
    max_pairwise_cosine: float,
    min_index_gap: int = 0,
) -> np.ndarray:
    """Select diverse support after first applying the grounding direction gate."""

    ordered = sorted(
        rows,
        key=lambda row: float(row.get(key, -1e9)),
        reverse=True,
    )
    passing = [
        row
        for row in ordered
        if int(row.get("recent_prefix_checked", 0) or 0) > 0
        and int(row.get("recent_prefix_mismatches", 0) or 0) == 0
    ]
    source = passing if passing else ordered
    return diverse_top_indices(
        source,
        memory_targets=memory_targets,
        key=key,
        k=k,
        max_pairwise_cosine=max_pairwise_cosine,
        min_index_gap=int(min_index_gap),
    )


def evaluate_mixture_variant(
    *,
    name: str,
    history_level: np.ndarray,
    window_indices: np.ndarray,
    scores: np.ndarray,
    grounding: dict[str, Any],
    spec_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    idx = np.asarray(window_indices, dtype=np.int64)
    if idx.size == 1:
        weights = np.ones(1, dtype=np.float32)
    else:
        weights = _softmax(scores, temperature=temperature)
    rows = weighted_prefix_terminal_rows(
        history_level=history_level,
        window_indices=idx,
        weights=weights,
        spec_names=spec_names,
    )
    alignment = market_implication_alignment(grounding=grounding, scenario_rows=rows)
    return {
        "variant": str(name),
        "status": str(alignment.get("status", "")),
        "checked_count": int(alignment.get("checked_count", 0) or 0),
        "mismatch_count": int(alignment.get("mismatch_count", 0) or 0),
        "mismatch_rate": (
            float(alignment.get("mismatch_count", 0) or 0)
            / float(alignment.get("checked_count", 1) or 1)
        ),
        "analogue_count": int(idx.size),
        "window_indices": [int(value) for value in idx],
        "weights": [float(value) for value in weights],
        "alignment": alignment,
        "terminal_rows": rows,
    }


def direction_check_for_mixture(
    *,
    candidate_details: list[dict[str, Any]],
    weights: list[float] | np.ndarray,
    final_mixture_alignment: dict[str, Any],
    min_support_match_rate: float = 0.60,
) -> dict[str, Any]:
    """Audit grounding directions against selected support and mixed prefix.

    This is deliberately not a scoring function. The narrative/start scorer can
    choose support, then this check decides whether the support and final mixed
    prefix are directionally consistent with the grounded claims.
    """

    selected = [row for row in candidate_details if isinstance(row, dict)]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if selected and w.shape[0] != len(selected):
        raise ValueError("weights must match candidate_details")
    if selected:
        denom = float(np.sum(w))
        if not np.isfinite(denom) or denom <= 0.0:
            w = np.ones(len(selected), dtype=np.float64) / float(len(selected))
        else:
            w = w / denom
    support_checked_weight = 0.0
    weighted_match_rate = 0.0
    weighted_mismatch_rate = 0.0
    candidate_rows: list[dict[str, Any]] = []
    for candidate, weight in zip(selected, w, strict=False):
        checked = int(candidate.get("recent_prefix_checked", 0) or 0)
        matches = int(candidate.get("recent_prefix_match_count", 0) or 0)
        mismatches = int(candidate.get("recent_prefix_mismatches", 0) or 0)
        match_rate = float(matches / checked) if checked else None
        mismatch_rate = float(mismatches / checked) if checked else None
        if checked:
            support_checked_weight += float(weight)
            weighted_match_rate += float(weight) * float(match_rate or 0.0)
            weighted_mismatch_rate += float(weight) * float(mismatch_rate or 0.0)
        candidate_rows.append(
            {
                "window_index": int(candidate.get("window_index", -1)),
                "weight": float(weight),
                "checked_count": checked,
                "match_count": matches,
                "mismatch_count": mismatches,
                "match_rate": match_rate,
                "mismatch_rate": mismatch_rate,
                "status": str(candidate.get("recent_prefix_alignment_status", "")),
            }
        )
    if support_checked_weight > 0.0:
        weighted_match_rate = float(weighted_match_rate / support_checked_weight)
        weighted_mismatch_rate = float(weighted_mismatch_rate / support_checked_weight)
    else:
        weighted_match_rate = None
        weighted_mismatch_rate = None
    final_checked = int(final_mixture_alignment.get("checked_count", 0) or 0)
    final_mismatches = int(final_mixture_alignment.get("mismatch_count", 0) or 0)
    if final_checked > 0 and final_mismatches > 0:
        status = "reject"
        reason = "final_mixed_prefix_direction_mismatch"
    elif weighted_match_rate is None:
        status = "warning"
        reason = "no_checkable_grounding_direction"
    elif weighted_match_rate < float(min_support_match_rate):
        status = "warning"
        reason = "selected_support_direction_weak"
    else:
        status = "pass"
        reason = "selected_support_and_mixed_prefix_directionally_consistent"
    return {
        "status": status,
        "reason": reason,
        "support_checked_weight": float(support_checked_weight),
        "support_weighted_match_rate": weighted_match_rate,
        "support_weighted_mismatch_rate": weighted_mismatch_rate,
        "min_support_match_rate": float(min_support_match_rate),
        "candidate_direction_rows": candidate_rows,
        "final_mixture_alignment": final_mixture_alignment,
        "final_mixture_checked_count": final_checked,
        "final_mixture_mismatch_count": final_mismatches,
    }


def _support_item_from_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_index": int(row["window_index"]),
        "window_id": str(
            row.get("window_id", f"joint39_val_{int(row['window_index']):04d}")
        ),
        "cosine": float(row.get("memory_support_cosine", 0.0)),
        "score": float(row.get("narrative_start_score", 0.0)),
        "start_distance_z": float(row.get("start_distance_z", 0.0)),
        "recent_prefix_alignment_status": str(
            row.get("recent_prefix_alignment_status", "")
        ),
    }


def _portfolio_quality_guard_candidate_mixtures(
    *,
    candidates: list[dict[str, Any]],
    memory_targets: np.ndarray,
    top_k: int,
    candidate_pool_size: int,
    mixture_size: int,
    max_mixtures: int,
    diverse_max_pairwise_cosine: float,
    diverse_min_index_gap: int,
) -> list[dict[str, Any]]:
    """Build live candidate-mixture rows from the same support table."""

    pool_indices = direction_passing_diverse_top_indices(
        candidates,
        memory_targets=memory_targets,
        key="narrative_start_score",
        k=max(int(candidate_pool_size), int(top_k), int(mixture_size)),
        max_pairwise_cosine=float(diverse_max_pairwise_cosine),
        min_index_gap=int(diverse_min_index_gap),
    )
    by_idx = {int(row["window_index"]): row for row in candidates}
    pool_rows = [by_idx[int(idx)] for idx in pool_indices if int(idx) in by_idx]
    if len(pool_rows) < int(mixture_size):
        pool_rows = sorted(
            candidates,
            key=lambda row: float(row.get("narrative_start_score", -1e9)),
            reverse=True,
        )[: max(int(candidate_pool_size), int(mixture_size))]
    pool_rows = pool_rows[: max(int(candidate_pool_size), int(mixture_size))]
    mixture_size = min(int(mixture_size), len(pool_rows))
    if mixture_size <= 0:
        raise ValueError("portfolio quality guard candidate pool is empty")
    combos = list(itertools.combinations(range(len(pool_rows)), mixture_size))
    if int(max_mixtures) > 0:
        combos = combos[: int(max_mixtures)]
    rows: list[dict[str, Any]] = []
    for rank, combo in enumerate(combos, start=1):
        support_rows = [pool_rows[pos] for pos in combo]
        support_items = [_support_item_from_candidate(row) for row in support_rows]
        rows.append(
            {
                "query_id": (
                    "live_portfolio_quality_guard__mixture_"
                    f"{rank:03d}__"
                    + "-".join(str(item["window_index"]) for item in support_items)
                ),
                "candidate_mixture_rank": int(rank),
                "candidate_mixture_positions": [int(pos + 1) for pos in combo],
                "candidate_support_window_indices": [
                    int(item["window_index"]) for item in support_items
                ],
                "candidate_support_window_ids": [
                    str(item["window_id"]) for item in support_items
                ],
                "top_train_pool": support_items,
            }
        )
    return rows


def _candidate_mixture_selection_order(
    candidate_mixtures: list[dict[str, Any]],
    support_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return candidate mixtures in response-score order when available."""

    by_id = {str(row.get("query_id", "")): row for row in candidate_mixtures}
    probabilities = support_policy.get("candidate_probabilities", [])
    if isinstance(probabilities, list) and probabilities:
        ranked = sorted(
            [row for row in probabilities if isinstance(row, dict)],
            key=lambda row: float(row.get("score", row.get("probability", 0.0)) or 0.0),
            reverse=True,
        )
        ordered = [
            by_id[str(row.get("query_id", ""))]
            for row in ranked
            if str(row.get("query_id", "")) in by_id
        ]
        if ordered:
            return ordered
    return list(candidate_mixtures)


def _candidate_mixture_to_prior_components(
    *,
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    grounding: dict[str, Any],
    spec_names: list[str],
) -> dict[str, Any] | None:
    """Build prior components for a single candidate mixture if direction-safe."""

    support_items = [
        item
        for item in candidate.get("top_train_pool", [])
        if isinstance(item, dict) and item.get("window_index") is not None
    ]
    if not support_items:
        return None
    indices = np.asarray(
        [int(item["window_index"]) for item in support_items],
        dtype=np.int64,
    )
    raw_weights = np.asarray(
        [float(item.get("weight", 1.0) or 0.0) for item in support_items],
        dtype=np.float32,
    )
    if raw_weights.shape[0] != indices.shape[0] or float(np.sum(raw_weights)) <= 0.0:
        weights = np.full(
            indices.shape[0],
            1.0 / float(indices.shape[0]),
            dtype=np.float32,
        )
    else:
        weights = (raw_weights / float(np.sum(raw_weights))).astype(np.float32)
    terminal_rows = weighted_prefix_terminal_rows(
        history_level=history_level,
        window_indices=indices,
        weights=weights,
        spec_names=spec_names,
    )
    support_alignment = market_implication_alignment(
        grounding=grounding,
        scenario_rows=terminal_rows,
    )
    by_idx = {int(row["window_index"]): row for row in candidates}
    candidate_details = [by_idx[int(idx)] for idx in indices if int(idx) in by_idx]
    direction_check = direction_check_for_mixture(
        candidate_details=candidate_details,
        weights=weights,
        final_mixture_alignment=support_alignment,
    )
    if str(direction_check.get("status", "")) == "reject":
        return None
    mixture_memory = np.sum(
        memory_targets[indices] * weights[:, None],
        axis=0,
    ).astype(np.float32)
    return {
        "memory": mixture_memory,
        "indices": indices,
        "weights": weights,
        "support_alignment": support_alignment,
        "direction_check": direction_check,
        "terminal_rows": terminal_rows,
        "candidate_details": candidate_details,
        "selected_candidate_query_id": str(candidate.get("query_id", "")),
        "selected_candidate_rank": int(candidate.get("candidate_mixture_rank", 0) or 0),
    }


def _prior_from_candidate_components(
    *,
    prior_mode: str,
    components: dict[str, Any],
    top_k: int,
    diverse_max_pairwise_cosine: float,
    diverse_min_index_gap: int,
    candidate_mixture_count: int,
    min_candidate_mixtures: int,
    support_policy: dict[str, Any],
    query_start_state: np.ndarray | None,
    support_policy_name: str = "portfolio_response_quality_guard_924e",
    policy_output_key: str = "portfolio_quality_guard_policy",
    extra_support_diversity: dict[str, Any] | None = None,
    extra_policy_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a memory prior from a pre-vetted candidate-mixture component set."""

    indices = np.asarray(components["indices"], dtype=np.int64)
    weights = np.asarray(components["weights"], dtype=np.float32)
    support_diversity = {
        "policy": str(support_policy_name),
        "requested_top_k": int(top_k),
        "selected_count": int(indices.size),
        "direction_gate": True,
        "latent_max_pairwise_cosine": float(diverse_max_pairwise_cosine),
        "temporal_min_index_gap": int(diverse_min_index_gap),
        "temporal_non_overlap_enforced": bool(int(diverse_min_index_gap) > 0),
        "padding_with_temporal_overlaps": False,
        "portfolio_quality_guard_active": True,
        "portfolio_quality_guard_fallback": False,
        "quality_guard_candidate_count": int(candidate_mixture_count),
        "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
    }
    if extra_support_diversity:
        support_diversity.update(extra_support_diversity)
    policy = {
        **support_policy,
        "fallback_to_equal_support": False,
        "selected_candidate_query_id": components["selected_candidate_query_id"],
        "selected_candidate_rank": components["selected_candidate_rank"],
    }
    if extra_policy_fields:
        policy.update(extra_policy_fields)
    result = {
        "mode": str(prior_mode),
        "memory": components["memory"],
        "analogue_count": int(indices.size),
        "window_indices": [int(idx) for idx in indices.tolist()],
        "weights": [float(weight) for weight in weights.tolist()],
        "support_alignment": components["support_alignment"],
        "direction_check": components["direction_check"],
        "support_diversity_policy": support_diversity,
        "terminal_rows": components["terminal_rows"],
        "candidate_details": components["candidate_details"],
        "query_start_source": (
            "provided_start_state"
            if query_start_state is not None
            else "query_window_index"
        ),
    }
    result[str(policy_output_key)] = policy
    return result


def build_mixture_memory_prior(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    history_level: np.ndarray,
    train_indices: np.ndarray,
    query_window_index: int,
    query_start_state: np.ndarray | None = None,
    grounding: dict[str, Any],
    spec_names: list[str],
    mode: str,
    top_k: int,
    temperature: float,
    start_distance_threshold_z: float,
    start_distance_penalty: float,
    implication_alignment_weight: float,
    diverse_max_pairwise_cosine: float,
    diverse_min_index_gap: int = 0,
    quality_guard_context: dict[str, Any] | None = None,
    quality_guard_candidate_pool_size: int = 5,
    quality_guard_mixture_size: int = 3,
    quality_guard_max_mixtures: int = 0,
    quality_guard_min_candidate_mixtures: int = 1,
    quality_guard_max_candidate_entropy_quantile: float | None = None,
    quality_guard_min_support_weight_max_quantile: float | None = 0.25,
    quality_guard_probability_temperature: float | None = None,
) -> dict[str, Any]:
    """Build a query memory or analogue-mixture memory prior."""

    prior_mode = str(mode)
    query = np.asarray(query_memory, dtype=np.float32).reshape(-1)
    memory = _as_float_array(memory_targets, name="memory_targets", ndim=2)
    if prior_mode == "query_memory":
        return {
            "mode": prior_mode,
            "memory": query.astype(np.float32),
            "analogue_count": 0,
            "window_indices": [],
            "weights": [],
            "support_alignment": {},
            "candidate_details": [],
            "query_start_source": (
                "provided_start_state"
                if query_start_state is not None
                else "query_window_index"
            ),
        }
    candidates = candidate_support_table(
        query_memory=query,
        memory_targets=memory,
        history_level=history_level,
        train_indices=train_indices,
        query_window_index=int(query_window_index),
        query_start_state=query_start_state,
        grounding=grounding,
        spec_names=spec_names,
        start_distance_threshold_z=float(start_distance_threshold_z),
        start_distance_penalty=float(start_distance_penalty),
        implication_alignment_weight=float(implication_alignment_weight),
    )
    if prior_mode in {
        "portfolio_quality_guard_924e",
        "portfolio_direction_first_quality_guard_938a",
        "broad_replay_response_guard_940a",
    }:
        from experiments.backfill.block_ar.nl_portfolio_response_quality_guard_policy import (
            build_quality_guard_policy_context,
            select_quality_guard_support,
        )

        if prior_mode == "broad_replay_response_guard_940a":
            from experiments.backfill.block_ar.nl_broad_support_response_utility import (
                load_broad_response_utility_context,
                select_feature_quality_guard_support,
            )

            direction_first_selection = True
            support_policy_name = "broad_replay_response_guard_940a"
            support_policy_kind = "broad_replay_response_quality_guard_direction_first"
        else:
            direction_first_selection = (
                prior_mode == "portfolio_direction_first_quality_guard_938a"
            )
            support_policy_name = "portfolio_direction_first_quality_guard_938a"
            support_policy_kind = (
                "train_only_portfolio_plus_quality_guard_direction_first"
            )
        base_prior = build_mixture_memory_prior(
            query_memory=query,
            memory_targets=memory,
            history_level=history_level,
            train_indices=train_indices,
            query_window_index=int(query_window_index),
            query_start_state=query_start_state,
            grounding=grounding,
            spec_names=spec_names,
            mode="diverse_topk_narrative_start_checked",
            top_k=int(top_k),
            temperature=float(temperature),
            start_distance_threshold_z=float(start_distance_threshold_z),
            start_distance_penalty=float(start_distance_penalty),
            implication_alignment_weight=float(implication_alignment_weight),
            diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            diverse_min_index_gap=int(diverse_min_index_gap),
            quality_guard_context=quality_guard_context,
            quality_guard_candidate_pool_size=int(quality_guard_candidate_pool_size),
            quality_guard_mixture_size=int(quality_guard_mixture_size),
            quality_guard_max_mixtures=int(quality_guard_max_mixtures),
            quality_guard_min_candidate_mixtures=int(
                quality_guard_min_candidate_mixtures
            ),
        )
        context = (
            quality_guard_context
            if quality_guard_context is not None
            else (
                load_broad_response_utility_context()
                if prior_mode == "broad_replay_response_guard_940a"
                else build_quality_guard_policy_context(
                    max_candidate_entropy_quantile=(
                        quality_guard_max_candidate_entropy_quantile
                    ),
                    min_support_weight_max_quantile=(
                        quality_guard_min_support_weight_max_quantile
                    ),
                )
            )
        )
        if quality_guard_probability_temperature is not None:
            context = dict(context)
            context["probability_temperature"] = float(
                quality_guard_probability_temperature
            )
        candidate_mixtures = _portfolio_quality_guard_candidate_mixtures(
            candidates=candidates,
            memory_targets=memory,
            top_k=int(top_k),
            candidate_pool_size=int(quality_guard_candidate_pool_size),
            mixture_size=int(quality_guard_mixture_size),
            max_mixtures=int(quality_guard_max_mixtures),
            diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            diverse_min_index_gap=int(diverse_min_index_gap),
        )
        min_candidate_mixtures = max(1, int(quality_guard_min_candidate_mixtures))
        if len(candidate_mixtures) < min_candidate_mixtures:
            base_prior["mode"] = prior_mode
            base_prior["portfolio_quality_guard_policy"] = {
                "name": "portfolio_response_quality_guard_prior",
                "policy_kind": "train_only_portfolio_plus_quality_guard",
                "fallback_to_equal_support": True,
                "fallback_reason": "insufficient_live_candidate_mixtures",
                "candidate_count": int(len(candidate_mixtures)),
                "min_candidate_mixtures": int(min_candidate_mixtures),
            }
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "portfolio_quality_guard_active": False,
                "portfolio_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior
        if direction_first_selection:
            direction_safe_components: dict[str, dict[str, Any]] = {}
            direction_rejected = 0
            for candidate in candidate_mixtures:
                components = _candidate_mixture_to_prior_components(
                    candidate=candidate,
                    candidates=candidates,
                    memory_targets=memory,
                    history_level=history_level,
                    grounding=grounding,
                    spec_names=spec_names,
                )
                if components is None:
                    direction_rejected += 1
                    continue
                direction_safe_components[str(candidate.get("query_id", ""))] = (
                    components
                )
            direction_safe_mixtures = [
                candidate
                for candidate in candidate_mixtures
                if str(candidate.get("query_id", "")) in direction_safe_components
            ]
            if len(direction_safe_mixtures) < min_candidate_mixtures:
                base_prior["mode"] = prior_mode
                base_prior["portfolio_quality_guard_policy"] = {
                    "name": "portfolio_response_quality_guard_prior",
                    "policy_kind": support_policy_kind,
                    "fallback_to_equal_support": True,
                    "fallback_reason": "insufficient_direction_safe_candidate_mixtures",
                    "candidate_count": int(len(candidate_mixtures)),
                    "direction_safe_candidate_count": int(len(direction_safe_mixtures)),
                    "direction_rejected_candidate_count": int(direction_rejected),
                    "min_candidate_mixtures": int(min_candidate_mixtures),
                }
                base_prior["support_diversity_policy"] = {
                    **base_prior.get("support_diversity_policy", {}),
                    "portfolio_quality_guard_active": False,
                    "portfolio_quality_guard_fallback": True,
                    "portfolio_quality_guard_direction_first": True,
                    "quality_guard_candidate_count": int(len(candidate_mixtures)),
                    "quality_guard_direction_safe_candidate_count": int(
                        len(direction_safe_mixtures)
                    ),
                    "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
                }
                return base_prior
            use_feature_selector = (
                prior_mode == "broad_replay_response_guard_940a"
                and str(context.get("preferred_selector", "support_prior"))
                == "feature_model"
            )
            if use_feature_selector:
                selection = select_feature_quality_guard_support(
                    direction_safe_mixtures,
                    context=context,
                    history_level=history_level,
                    query_start_state=query_start_state,
                    grounding=grounding,
                )
            else:
                selection = select_quality_guard_support(
                    direction_safe_mixtures,
                    portfolio_prior=context["portfolio_prior"],
                    crps_prior=context["crps_prior"],
                    energy_prior=context["energy_prior"],
                    probability_temperature=float(
                        context.get("probability_temperature", 0.25)
                    ),
                    max_candidate_entropy_threshold=context.get(
                        "max_candidate_entropy_threshold"
                    ),
                    max_candidate_entropy_quantile=context.get(
                        "max_candidate_entropy_quantile"
                    ),
                    min_support_weight_max_threshold=None,
                    min_support_weight_max_quantile=None,
                )
            if bool(selection["fallback"]):
                base_prior["mode"] = prior_mode
                base_prior["portfolio_quality_guard_policy"] = {
                    **selection["support_policy"],
                    "fallback_to_equal_support": True,
                    "fallback_reason": "direction_first_quality_guard_fallback",
                    "policy_kind": support_policy_kind,
                    "direction_safe_candidate_count": int(len(direction_safe_mixtures)),
                    "direction_rejected_candidate_count": int(direction_rejected),
                }
                base_prior["support_diversity_policy"] = {
                    **base_prior.get("support_diversity_policy", {}),
                    "portfolio_quality_guard_active": False,
                    "portfolio_quality_guard_fallback": True,
                    "portfolio_quality_guard_direction_first": True,
                    "quality_guard_candidate_count": int(len(candidate_mixtures)),
                    "quality_guard_direction_safe_candidate_count": int(
                        len(direction_safe_mixtures)
                    ),
                    "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
                }
                return base_prior
            if prior_mode == "broad_replay_response_guard_940a":
                selected = [
                    item
                    for item in selection.get("selected", [])
                    if isinstance(item, dict)
                ]
                aggregate_candidate = {
                    "query_id": "broad_replay_response_guard__weighted_selection",
                    "candidate_mixture_rank": 0,
                    "top_train_pool": selected,
                }
                components = _candidate_mixture_to_prior_components(
                    candidate=aggregate_candidate,
                    candidates=candidates,
                    memory_targets=memory,
                    history_level=history_level,
                    grounding=grounding,
                    spec_names=spec_names,
                )
                if components is None:
                    base_prior["mode"] = prior_mode
                    base_prior["portfolio_quality_guard_policy"] = {
                        **selection["support_policy"],
                        "fallback_to_equal_support": True,
                        "fallback_reason": "weighted_direction_safe_aggregate_rejected",
                        "policy_kind": support_policy_kind,
                        "direction_safe_candidate_count": int(
                            len(direction_safe_mixtures)
                        ),
                        "direction_rejected_candidate_count": int(direction_rejected),
                    }
                    base_prior["support_diversity_policy"] = {
                        **base_prior.get("support_diversity_policy", {}),
                        "portfolio_quality_guard_active": False,
                        "portfolio_quality_guard_fallback": True,
                        "portfolio_quality_guard_direction_first": True,
                        "quality_guard_candidate_count": int(len(candidate_mixtures)),
                        "quality_guard_direction_safe_candidate_count": int(
                            len(direction_safe_mixtures)
                        ),
                        "quality_guard_min_candidate_mixtures": int(
                            min_candidate_mixtures
                        ),
                    }
                    return base_prior
                return _prior_from_candidate_components(
                    prior_mode=prior_mode,
                    components=components,
                    top_k=int(top_k),
                    diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
                    diverse_min_index_gap=int(diverse_min_index_gap),
                    candidate_mixture_count=int(len(candidate_mixtures)),
                    min_candidate_mixtures=int(min_candidate_mixtures),
                    support_policy={
                        **selection["support_policy"],
                        "policy_kind": support_policy_kind,
                        "direction_first_candidate_selection": True,
                        "weighted_candidate_aggregation": True,
                        "direction_safe_candidate_count": int(
                            len(direction_safe_mixtures)
                        ),
                        "direction_rejected_candidate_count": int(direction_rejected),
                    },
                    query_start_state=query_start_state,
                    support_policy_name=support_policy_name,
                    extra_support_diversity={
                        "policy": support_policy_name,
                        "portfolio_quality_guard_direction_first": True,
                        "weighted_candidate_aggregation": True,
                        "quality_guard_direction_safe_candidate_count": int(
                            len(direction_safe_mixtures)
                        ),
                    },
                )
            candidate_order = _candidate_mixture_selection_order(
                direction_safe_mixtures,
                selection["support_policy"],
            )
            selected_candidate = candidate_order[0]
            selected_query_id = str(selected_candidate.get("query_id", ""))
            return _prior_from_candidate_components(
                prior_mode=prior_mode,
                components=direction_safe_components[selected_query_id],
                top_k=int(top_k),
                diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
                diverse_min_index_gap=int(diverse_min_index_gap),
                candidate_mixture_count=int(len(candidate_mixtures)),
                min_candidate_mixtures=int(min_candidate_mixtures),
                support_policy={
                    **selection["support_policy"],
                    "policy_kind": support_policy_kind,
                    "direction_first_candidate_selection": True,
                    "direction_safe_candidate_count": int(len(direction_safe_mixtures)),
                    "direction_rejected_candidate_count": int(direction_rejected),
                },
                query_start_state=query_start_state,
                support_policy_name=support_policy_name,
                extra_support_diversity={
                    "policy": support_policy_name,
                    "portfolio_quality_guard_direction_first": True,
                    "quality_guard_direction_safe_candidate_count": int(
                        len(direction_safe_mixtures)
                    ),
                },
            )
        selection = select_quality_guard_support(
            candidate_mixtures,
            portfolio_prior=context["portfolio_prior"],
            crps_prior=context["crps_prior"],
            energy_prior=context["energy_prior"],
            probability_temperature=float(context.get("probability_temperature", 0.25)),
            max_candidate_entropy_threshold=context.get(
                "max_candidate_entropy_threshold"
            ),
            max_candidate_entropy_quantile=context.get(
                "max_candidate_entropy_quantile"
            ),
            min_support_weight_max_threshold=context.get(
                "min_support_weight_max_threshold"
            ),
            min_support_weight_max_quantile=context.get(
                "min_support_weight_max_quantile"
            ),
        )
        if bool(selection["fallback"]):
            base_prior["mode"] = prior_mode
            base_prior["portfolio_quality_guard_policy"] = selection["support_policy"]
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "portfolio_quality_guard_active": False,
                "portfolio_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior

        selected = [
            item for item in selection.get("selected", []) if isinstance(item, dict)
        ]
        indices = np.asarray(
            [int(item["window_index"]) for item in selected],
            dtype=np.int64,
        )
        weights = np.asarray(
            [float(item.get("weight", 0.0) or 0.0) for item in selected],
            dtype=np.float32,
        )
        if indices.size == 0:
            base_prior["mode"] = prior_mode
            base_prior["portfolio_quality_guard_policy"] = {
                **selection["support_policy"],
                "fallback_to_equal_support": True,
                "fallback_reason": "empty_selected_support",
            }
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "portfolio_quality_guard_active": False,
                "portfolio_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior
        weights = np.maximum(weights, 0.0)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(indices.shape[0], dtype=np.float32)
        weights = (weights / float(weights.sum())).astype(np.float32)
        mixture_memory = np.sum(memory[indices] * weights[:, None], axis=0).astype(
            np.float32
        )
        terminal_rows = weighted_prefix_terminal_rows(
            history_level=history_level,
            window_indices=indices,
            weights=weights,
            spec_names=spec_names,
        )
        support_alignment = market_implication_alignment(
            grounding=grounding,
            scenario_rows=terminal_rows,
        )
        by_idx = {int(row["window_index"]): row for row in candidates}
        candidate_details = [by_idx[int(idx)] for idx in indices if int(idx) in by_idx]
        direction_check = direction_check_for_mixture(
            candidate_details=candidate_details,
            weights=weights,
            final_mixture_alignment=support_alignment,
        )
        if str(direction_check.get("status", "")) == "reject":
            replacement = None
            for candidate in _candidate_mixture_selection_order(
                candidate_mixtures,
                selection["support_policy"],
            ):
                replacement = _candidate_mixture_to_prior_components(
                    candidate=candidate,
                    candidates=candidates,
                    memory_targets=memory,
                    history_level=history_level,
                    grounding=grounding,
                    spec_names=spec_names,
                )
                if replacement is not None:
                    break
            if replacement is not None:
                return {
                    "mode": prior_mode,
                    "memory": replacement["memory"],
                    "analogue_count": int(replacement["indices"].size),
                    "window_indices": [
                        int(idx) for idx in replacement["indices"].tolist()
                    ],
                    "weights": [
                        float(weight) for weight in replacement["weights"].tolist()
                    ],
                    "support_alignment": replacement["support_alignment"],
                    "direction_check": replacement["direction_check"],
                    "support_diversity_policy": {
                        "policy": "portfolio_response_quality_guard_924e",
                        "requested_top_k": int(top_k),
                        "selected_count": int(replacement["indices"].size),
                        "direction_gate": True,
                        "latent_max_pairwise_cosine": float(
                            diverse_max_pairwise_cosine
                        ),
                        "temporal_min_index_gap": int(diverse_min_index_gap),
                        "temporal_non_overlap_enforced": bool(
                            int(diverse_min_index_gap) > 0
                        ),
                        "padding_with_temporal_overlaps": False,
                        "portfolio_quality_guard_active": True,
                        "portfolio_quality_guard_fallback": False,
                        "portfolio_quality_guard_direction_safe_candidate_fallback": True,
                        "quality_guard_candidate_count": int(len(candidate_mixtures)),
                        "quality_guard_min_candidate_mixtures": int(
                            min_candidate_mixtures
                        ),
                    },
                    "portfolio_quality_guard_policy": {
                        **selection["support_policy"],
                        "fallback_to_equal_support": False,
                        "direction_safe_candidate_fallback": True,
                        "initial_marginal_direction_check": direction_check,
                        "selected_candidate_query_id": replacement[
                            "selected_candidate_query_id"
                        ],
                        "selected_candidate_rank": replacement[
                            "selected_candidate_rank"
                        ],
                    },
                    "terminal_rows": replacement["terminal_rows"],
                    "candidate_details": replacement["candidate_details"],
                    "query_start_source": (
                        "provided_start_state"
                        if query_start_state is not None
                        else "query_window_index"
                    ),
                }
            base_prior["mode"] = prior_mode
            base_prior["portfolio_quality_guard_policy"] = {
                **selection["support_policy"],
                "fallback_to_equal_support": True,
                "fallback_reason": str(
                    direction_check.get(
                        "reason", "quality_guard_direction_check_reject"
                    )
                ),
                "direction_check": direction_check,
            }
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "portfolio_quality_guard_active": False,
                "portfolio_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior
        return {
            "mode": prior_mode,
            "memory": mixture_memory,
            "analogue_count": int(indices.size),
            "window_indices": [int(idx) for idx in indices],
            "weights": [float(weight) for weight in weights],
            "support_alignment": support_alignment,
            "direction_check": direction_check,
            "support_diversity_policy": {
                "policy": "portfolio_response_quality_guard_924e",
                "requested_top_k": int(top_k),
                "selected_count": int(indices.size),
                "direction_gate": True,
                "latent_max_pairwise_cosine": float(diverse_max_pairwise_cosine),
                "temporal_min_index_gap": int(diverse_min_index_gap),
                "temporal_non_overlap_enforced": bool(int(diverse_min_index_gap) > 0),
                "padding_with_temporal_overlaps": False,
                "portfolio_quality_guard_active": True,
                "portfolio_quality_guard_fallback": False,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            },
            "portfolio_quality_guard_policy": selection["support_policy"],
            "terminal_rows": terminal_rows,
            "candidate_details": candidate_details,
            "query_start_source": (
                "provided_start_state"
                if query_start_state is not None
                else "query_window_index"
            ),
        }
    if prior_mode in {
        "narrative_book_quality_guard_926b",
        "narrative_book_direction_first_quality_guard_938c",
    }:
        from experiments.backfill.block_ar.nl_narrative_book_conditioned_quality_guard import (
            blend_book_priors,
            build_narrative_book_guard_policy_context,
            narrative_book_relevance_weights,
        )
        from experiments.backfill.block_ar.nl_portfolio_response_quality_guard_policy import (
            select_quality_guard_support,
        )

        narrative_book_direction_first = (
            prior_mode == "narrative_book_direction_first_quality_guard_938c"
        )
        base_prior = build_mixture_memory_prior(
            query_memory=query,
            memory_targets=memory,
            history_level=history_level,
            train_indices=train_indices,
            query_window_index=int(query_window_index),
            query_start_state=query_start_state,
            grounding=grounding,
            spec_names=spec_names,
            mode="diverse_topk_narrative_start_checked",
            top_k=int(top_k),
            temperature=float(temperature),
            start_distance_threshold_z=float(start_distance_threshold_z),
            start_distance_penalty=float(start_distance_penalty),
            implication_alignment_weight=float(implication_alignment_weight),
            diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            diverse_min_index_gap=int(diverse_min_index_gap),
            quality_guard_context=quality_guard_context,
            quality_guard_candidate_pool_size=int(quality_guard_candidate_pool_size),
            quality_guard_mixture_size=int(quality_guard_mixture_size),
            quality_guard_max_mixtures=int(quality_guard_max_mixtures),
            quality_guard_min_candidate_mixtures=int(
                quality_guard_min_candidate_mixtures
            ),
        )
        context = (
            quality_guard_context
            if quality_guard_context is not None
            and "priors_by_book" in quality_guard_context
            else build_narrative_book_guard_policy_context(
                min_support_weight_max_quantile=(
                    quality_guard_min_support_weight_max_quantile
                )
            )
        )
        book_weights = narrative_book_relevance_weights(grounding)
        portfolio_prior = blend_book_priors(context["priors_by_book"], book_weights)
        candidate_mixtures = _portfolio_quality_guard_candidate_mixtures(
            candidates=candidates,
            memory_targets=memory,
            top_k=int(top_k),
            candidate_pool_size=int(quality_guard_candidate_pool_size),
            mixture_size=int(quality_guard_mixture_size),
            max_mixtures=int(quality_guard_max_mixtures),
            diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            diverse_min_index_gap=int(diverse_min_index_gap),
        )
        min_candidate_mixtures = max(1, int(quality_guard_min_candidate_mixtures))
        if len(candidate_mixtures) < min_candidate_mixtures:
            base_prior["mode"] = prior_mode
            base_prior["narrative_book_response_policy"] = {
                "name": "narrative_book_response_quality_guard_prior",
                "policy_kind": "train_only_narrative_book_portfolio_plus_quality_guard",
                "fallback_to_equal_support": True,
                "fallback_reason": "insufficient_live_candidate_mixtures",
                "candidate_count": int(len(candidate_mixtures)),
                "min_candidate_mixtures": int(min_candidate_mixtures),
                "book_weights": book_weights,
            }
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "narrative_book_quality_guard_active": False,
                "narrative_book_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior
        if narrative_book_direction_first:
            direction_safe_components: dict[str, dict[str, Any]] = {}
            direction_rejected = 0
            for candidate in candidate_mixtures:
                components = _candidate_mixture_to_prior_components(
                    candidate=candidate,
                    candidates=candidates,
                    memory_targets=memory,
                    history_level=history_level,
                    grounding=grounding,
                    spec_names=spec_names,
                )
                if components is None:
                    direction_rejected += 1
                    continue
                direction_safe_components[str(candidate.get("query_id", ""))] = (
                    components
                )
            direction_safe_mixtures = [
                candidate
                for candidate in candidate_mixtures
                if str(candidate.get("query_id", "")) in direction_safe_components
            ]
            if len(direction_safe_mixtures) < min_candidate_mixtures:
                base_prior["mode"] = prior_mode
                base_prior["narrative_book_response_policy"] = {
                    "name": "narrative_book_response_quality_guard_prior",
                    "policy_kind": (
                        "train_only_narrative_book_direction_first_quality_guard"
                    ),
                    "fallback_to_equal_support": True,
                    "fallback_reason": "insufficient_direction_safe_candidate_mixtures",
                    "candidate_count": int(len(candidate_mixtures)),
                    "direction_safe_candidate_count": int(len(direction_safe_mixtures)),
                    "direction_rejected_candidate_count": int(direction_rejected),
                    "min_candidate_mixtures": int(min_candidate_mixtures),
                    "book_weights": book_weights,
                }
                base_prior["support_diversity_policy"] = {
                    **base_prior.get("support_diversity_policy", {}),
                    "narrative_book_quality_guard_active": False,
                    "narrative_book_quality_guard_fallback": True,
                    "narrative_book_quality_guard_direction_first": True,
                    "quality_guard_candidate_count": int(len(candidate_mixtures)),
                    "quality_guard_direction_safe_candidate_count": int(
                        len(direction_safe_mixtures)
                    ),
                    "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
                }
                return base_prior
            selection = select_quality_guard_support(
                direction_safe_mixtures,
                portfolio_prior=portfolio_prior,
                crps_prior=context["crps_prior"],
                energy_prior=context["energy_prior"],
                probability_temperature=float(
                    context.get("probability_temperature", 0.25)
                ),
                max_candidate_entropy_threshold=context.get(
                    "max_candidate_entropy_threshold"
                ),
                max_candidate_entropy_quantile=context.get(
                    "max_candidate_entropy_quantile"
                ),
                min_support_weight_max_threshold=None,
                min_support_weight_max_quantile=None,
            )
            if bool(selection["fallback"]):
                base_prior["mode"] = prior_mode
                base_prior["narrative_book_response_policy"] = {
                    **selection["support_policy"],
                    "name": "narrative_book_response_quality_guard_prior",
                    "fallback_to_equal_support": True,
                    "fallback_reason": "narrative_book_direction_first_fallback",
                    "book_weights": book_weights,
                    "direction_safe_candidate_count": int(len(direction_safe_mixtures)),
                    "direction_rejected_candidate_count": int(direction_rejected),
                    "source_artifacts": context.get("source_artifacts", {}),
                }
                base_prior["support_diversity_policy"] = {
                    **base_prior.get("support_diversity_policy", {}),
                    "narrative_book_quality_guard_active": False,
                    "narrative_book_quality_guard_fallback": True,
                    "narrative_book_quality_guard_direction_first": True,
                    "quality_guard_candidate_count": int(len(candidate_mixtures)),
                    "quality_guard_direction_safe_candidate_count": int(
                        len(direction_safe_mixtures)
                    ),
                    "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
                }
                return base_prior
            candidate_order = _candidate_mixture_selection_order(
                direction_safe_mixtures,
                selection["support_policy"],
            )
            selected_candidate = candidate_order[0]
            selected_query_id = str(selected_candidate.get("query_id", ""))
            return _prior_from_candidate_components(
                prior_mode=prior_mode,
                components=direction_safe_components[selected_query_id],
                top_k=int(top_k),
                diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
                diverse_min_index_gap=int(diverse_min_index_gap),
                candidate_mixture_count=int(len(candidate_mixtures)),
                min_candidate_mixtures=int(min_candidate_mixtures),
                support_policy={
                    **selection["support_policy"],
                    "name": "narrative_book_response_quality_guard_prior",
                    "policy_kind": (
                        "train_only_narrative_book_direction_first_quality_guard"
                    ),
                    "direction_first_candidate_selection": True,
                    "book_weights": book_weights,
                    "direction_safe_candidate_count": int(len(direction_safe_mixtures)),
                    "direction_rejected_candidate_count": int(direction_rejected),
                    "source_artifacts": context.get("source_artifacts", {}),
                },
                query_start_state=query_start_state,
                support_policy_name="narrative_book_direction_first_quality_guard_938c",
                policy_output_key="narrative_book_response_policy",
                extra_support_diversity={
                    "policy": "narrative_book_direction_first_quality_guard_938c",
                    "portfolio_quality_guard_active": False,
                    "portfolio_quality_guard_fallback": False,
                    "narrative_book_quality_guard_active": True,
                    "narrative_book_quality_guard_fallback": False,
                    "narrative_book_quality_guard_direction_first": True,
                    "quality_guard_direction_safe_candidate_count": int(
                        len(direction_safe_mixtures)
                    ),
                },
            )
        selection = select_quality_guard_support(
            candidate_mixtures,
            portfolio_prior=portfolio_prior,
            crps_prior=context["crps_prior"],
            energy_prior=context["energy_prior"],
            probability_temperature=float(context.get("probability_temperature", 0.25)),
            max_candidate_entropy_threshold=context.get(
                "max_candidate_entropy_threshold"
            ),
            max_candidate_entropy_quantile=context.get(
                "max_candidate_entropy_quantile"
            ),
            min_support_weight_max_threshold=context.get(
                "min_support_weight_max_threshold"
            ),
            min_support_weight_max_quantile=context.get(
                "min_support_weight_max_quantile"
            ),
        )
        if bool(selection["fallback"]):
            base_prior["mode"] = prior_mode
            base_prior["narrative_book_response_policy"] = {
                **selection["support_policy"],
                "name": "narrative_book_response_quality_guard_prior",
                "book_weights": book_weights,
            }
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "narrative_book_quality_guard_active": False,
                "narrative_book_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior

        selected = [
            item for item in selection.get("selected", []) if isinstance(item, dict)
        ]
        indices = np.asarray(
            [int(item["window_index"]) for item in selected],
            dtype=np.int64,
        )
        weights = np.asarray(
            [float(item.get("weight", 0.0) or 0.0) for item in selected],
            dtype=np.float32,
        )
        if indices.size == 0:
            base_prior["mode"] = prior_mode
            base_prior["narrative_book_response_policy"] = {
                **selection["support_policy"],
                "name": "narrative_book_response_quality_guard_prior",
                "fallback_to_equal_support": True,
                "fallback_reason": "empty_selected_support",
                "book_weights": book_weights,
            }
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "narrative_book_quality_guard_active": False,
                "narrative_book_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior
        weights = np.maximum(weights, 0.0)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(indices.shape[0], dtype=np.float32)
        weights = (weights / float(weights.sum())).astype(np.float32)
        mixture_memory = np.sum(memory[indices] * weights[:, None], axis=0).astype(
            np.float32
        )
        terminal_rows = weighted_prefix_terminal_rows(
            history_level=history_level,
            window_indices=indices,
            weights=weights,
            spec_names=spec_names,
        )
        support_alignment = market_implication_alignment(
            grounding=grounding,
            scenario_rows=terminal_rows,
        )
        by_idx = {int(row["window_index"]): row for row in candidates}
        candidate_details = [by_idx[int(idx)] for idx in indices if int(idx) in by_idx]
        direction_check = direction_check_for_mixture(
            candidate_details=candidate_details,
            weights=weights,
            final_mixture_alignment=support_alignment,
        )
        if str(direction_check.get("status", "")) == "reject":
            base_prior["mode"] = prior_mode
            base_prior["narrative_book_response_policy"] = {
                **selection["support_policy"],
                "name": "narrative_book_response_quality_guard_prior",
                "policy_kind": "train_only_narrative_book_portfolio_plus_quality_guard",
                "book_weights": book_weights,
                "fallback_to_equal_support": True,
                "fallback_reason": str(
                    direction_check.get(
                        "reason", "narrative_book_direction_check_reject"
                    )
                ),
                "direction_check": direction_check,
                "source_artifacts": context.get("source_artifacts", {}),
            }
            base_prior["support_diversity_policy"] = {
                **base_prior.get("support_diversity_policy", {}),
                "narrative_book_quality_guard_active": False,
                "narrative_book_quality_guard_fallback": True,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            }
            return base_prior
        return {
            "mode": prior_mode,
            "memory": mixture_memory,
            "analogue_count": int(indices.size),
            "window_indices": [int(idx) for idx in indices],
            "weights": [float(weight) for weight in weights],
            "support_alignment": support_alignment,
            "direction_check": direction_check,
            "support_diversity_policy": {
                "policy": "narrative_book_response_quality_guard",
                "requested_top_k": int(top_k),
                "selected_count": int(indices.size),
                "direction_gate": True,
                "latent_max_pairwise_cosine": float(diverse_max_pairwise_cosine),
                "temporal_min_index_gap": int(diverse_min_index_gap),
                "temporal_non_overlap_enforced": bool(int(diverse_min_index_gap) > 0),
                "padding_with_temporal_overlaps": False,
                "narrative_book_quality_guard_active": True,
                "narrative_book_quality_guard_fallback": False,
                "quality_guard_candidate_count": int(len(candidate_mixtures)),
                "quality_guard_min_candidate_mixtures": int(min_candidate_mixtures),
            },
            "narrative_book_response_policy": {
                **selection["support_policy"],
                "name": "narrative_book_response_quality_guard_prior",
                "policy_kind": "train_only_narrative_book_portfolio_plus_quality_guard",
                "book_weights": book_weights,
                "source_artifacts": context.get("source_artifacts", {}),
            },
            "terminal_rows": terminal_rows,
            "candidate_details": candidate_details,
            "query_start_source": (
                "provided_start_state"
                if query_start_state is not None
                else "query_window_index"
            ),
        }
    if prior_mode == "soft_topk_memory":
        indices = _top_indices(candidates, "memory_support_cosine", top_k)
        scores = _scores_for_indices(candidates, indices, "memory_support_cosine")
    elif prior_mode == "soft_topk_narrative_start":
        indices = _top_indices(candidates, "narrative_start_score", top_k)
        scores = _scores_for_indices(candidates, indices, "narrative_start_score")
    elif prior_mode == "soft_topk_narrative_start_checked":
        indices = direction_passing_top_indices(
            candidates,
            key="narrative_start_score",
            k=top_k,
        )
        scores = _scores_for_indices(candidates, indices, "narrative_start_score")
    elif prior_mode == "diverse_topk_narrative_start_checked":
        indices = direction_passing_diverse_top_indices(
            candidates,
            memory_targets=memory,
            key="narrative_start_score",
            k=top_k,
            max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            min_index_gap=int(diverse_min_index_gap),
        )
        scores = _scores_for_indices(candidates, indices, "narrative_start_score")
    elif prior_mode == "cohesive_topk_narrative_start_checked":
        indices = cohesive_top_indices(
            candidates,
            memory_targets=memory,
            key="narrative_start_score",
            k=top_k,
            min_index_gap=int(diverse_min_index_gap),
        )
        scores = _scores_for_indices(candidates, indices, "narrative_start_score")
    elif prior_mode == "cluster_family_narrative_start_checked":
        indices = latent_family_top_indices(
            candidates,
            memory_targets=memory,
            key="narrative_start_score",
            k=top_k,
            min_family_cosine=float(diverse_max_pairwise_cosine),
            min_index_gap=int(diverse_min_index_gap),
        )
        scores = _scores_for_indices(candidates, indices, "narrative_start_score")
    elif prior_mode == "kernel_topk_narrative_start_checked":
        indices = direction_passing_top_indices(
            candidates,
            key="narrative_start_score",
            k=top_k,
        )
        scores = _scores_for_indices(candidates, indices, "memory_support_cosine")
    elif prior_mode == "soft_topk_start_only":
        indices = _top_indices(candidates, "start_only_score", top_k)
        scores = _scores_for_indices(candidates, indices, "start_only_score")
    elif prior_mode == "soft_topk_combined":
        indices = _top_indices(candidates, "combined_score", top_k)
        scores = _scores_for_indices(candidates, indices, "combined_score")
    elif prior_mode == "diverse_topk_narrative_start":
        indices = diverse_top_indices(
            candidates,
            memory_targets=memory,
            key="narrative_start_score",
            k=top_k,
            max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            min_index_gap=int(diverse_min_index_gap),
        )
        scores = _scores_for_indices(candidates, indices, "narrative_start_score")
    elif prior_mode == "diverse_topk_combined":
        indices = diverse_top_indices(
            candidates,
            memory_targets=memory,
            key="combined_score",
            k=top_k,
            max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            min_index_gap=int(diverse_min_index_gap),
        )
        scores = _scores_for_indices(candidates, indices, "combined_score")
    else:
        raise ValueError(f"unknown memory prior mode: {mode!r}")
    weights = _softmax(scores, temperature=temperature)
    mixture_memory = np.sum(memory[indices] * weights[:, None], axis=0).astype(
        np.float32
    )
    terminal_rows = weighted_prefix_terminal_rows(
        history_level=history_level,
        window_indices=indices,
        weights=weights,
        spec_names=spec_names,
    )
    support_alignment = market_implication_alignment(
        grounding=grounding,
        scenario_rows=terminal_rows,
    )
    by_idx = {int(row["window_index"]): row for row in candidates}
    candidate_details = [by_idx[int(idx)] for idx in indices]
    direction_check = direction_check_for_mixture(
        candidate_details=candidate_details,
        weights=weights,
        final_mixture_alignment=support_alignment,
    )
    support_policy_name = (
        "direction_checked_anchor_cohesive_support"
        if prior_mode == "cohesive_topk_narrative_start_checked"
        else (
            "direction_checked_latent_family_support"
            if prior_mode == "cluster_family_narrative_start_checked"
            else (
                "direction_checked_low_temperature_similarity_kernel_support"
                if prior_mode == "kernel_topk_narrative_start_checked"
                else (
                    "direction_checked_latent_temporal_diverse_support"
                    if "diverse" in prior_mode
                    else "score_ranked_support"
                )
            )
        )
    )
    support_diversity_policy = {
        "policy": support_policy_name,
        "requested_top_k": int(top_k),
        "selected_count": int(indices.size),
        "direction_gate": prior_mode
        in {
            "soft_topk_narrative_start_checked",
            "diverse_topk_narrative_start_checked",
            "cohesive_topk_narrative_start_checked",
            "cluster_family_narrative_start_checked",
            "kernel_topk_narrative_start_checked",
        },
        "latent_max_pairwise_cosine": (
            float(diverse_max_pairwise_cosine) if "diverse" in prior_mode else None
        ),
        "temporal_min_index_gap": (
            int(diverse_min_index_gap) if "diverse" in prior_mode else 0
        ),
        "temporal_non_overlap_enforced": bool(
            "diverse" in prior_mode and int(diverse_min_index_gap) > 0
        ),
        "padding_with_temporal_overlaps": False,
    }
    if prior_mode == "cohesive_topk_narrative_start_checked" and indices.size:
        support_diversity_policy["anchor_window_index"] = int(indices[0])
        support_diversity_policy["temporal_min_index_gap"] = int(diverse_min_index_gap)
        support_diversity_policy["temporal_non_overlap_enforced"] = bool(
            int(diverse_min_index_gap) > 0
        )
    if prior_mode == "cluster_family_narrative_start_checked":
        support_diversity_policy["family_size"] = int(indices.size)
        support_diversity_policy["family_min_pairwise_cosine"] = float(
            diverse_max_pairwise_cosine
        )
        support_diversity_policy["temporal_min_index_gap"] = int(diverse_min_index_gap)
        support_diversity_policy["temporal_non_overlap_enforced"] = bool(
            int(diverse_min_index_gap) > 0
        )
    if prior_mode == "kernel_topk_narrative_start_checked":
        support_diversity_policy["kernel_score"] = "memory_support_cosine"
        support_diversity_policy["kernel_temperature"] = float(temperature)
    return {
        "mode": prior_mode,
        "memory": mixture_memory,
        "analogue_count": int(indices.size),
        "window_indices": [int(idx) for idx in indices],
        "weights": [float(weight) for weight in weights],
        "support_alignment": support_alignment,
        "direction_check": direction_check,
        "support_diversity_policy": support_diversity_policy,
        "terminal_rows": terminal_rows,
        "candidate_details": candidate_details,
        "query_start_source": (
            "provided_start_state"
            if query_start_state is not None
            else "query_window_index"
        ),
    }


def _case_inputs_from_casebook(casebook_summary: str | Path) -> list[dict[str, Any]]:
    payload = _load_json(casebook_summary)
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{casebook_summary}: missing cases")
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        artifact_paths = case.get("artifact_paths", {})
        if not isinstance(artifact_paths, dict):
            artifact_paths = {}
        prefix_report = Path(str(artifact_paths.get("prefix_report", "")))
        if not prefix_report.exists():
            raise FileNotFoundError(
                f"missing prefix report for {case.get('case_name')}"
            )
        rows.append(
            {
                "case_name": str(case.get("case_name", "")),
                "story": str(case.get("story", "")),
                "prefix_report": str(prefix_report),
                "prefix_arrays": str(
                    prefix_report.with_name("prefix_latent_story_smoke_arrays.npz")
                ),
                "generated_alignment": case.get("market_alignment", {}),
                "selected_start_status": str(case.get("selected_start_status", "")),
                "source_casebook": str(casebook_summary),
            }
        )
    return rows


def evaluate_case(
    *,
    case_input: dict[str, Any],
    oracle_arrays: dict[str, np.ndarray],
    spec_names: list[str],
    top_k: int,
    temperature: float,
    start_distance_threshold_z: float,
    start_distance_penalty: float,
    implication_alignment_weight: float,
    diverse_max_pairwise_cosine: float,
) -> dict[str, Any]:
    report = _load_json(case_input["prefix_report"])
    arrays = np.load(case_input["prefix_arrays"])
    query_memory = np.asarray(arrays["text_memory"][0], dtype=np.float32)
    grounding = report.get("cached_query", {}).get("grounding", {})
    if not isinstance(grounding, dict):
        grounding = {}
    history_level = np.asarray(oracle_arrays["history_level"], dtype=np.float32)
    memory_targets = np.asarray(oracle_arrays["true_memory"], dtype=np.float32)
    train_indices = np.asarray(oracle_arrays["train_indices"], dtype=np.int64)
    query_window_index = int(report.get("cached_query", {}).get("window_index", 0))
    candidates = candidate_support_table(
        query_memory=query_memory,
        memory_targets=memory_targets,
        history_level=history_level,
        train_indices=train_indices,
        query_window_index=query_window_index,
        grounding=grounding,
        spec_names=spec_names,
        start_distance_threshold_z=float(start_distance_threshold_z),
        start_distance_penalty=float(start_distance_penalty),
        implication_alignment_weight=float(implication_alignment_weight),
    )
    memory_top = _top_indices(candidates, "memory_support_cosine", top_k)
    narrative_start_top = _top_indices(candidates, "narrative_start_score", top_k)
    narrative_start_checked_top = direction_passing_top_indices(
        candidates,
        key="narrative_start_score",
        k=top_k,
    )
    combined_top = _top_indices(candidates, "combined_score", top_k)
    diverse_narrative_start_top = diverse_top_indices(
        candidates,
        memory_targets=memory_targets,
        key="narrative_start_score",
        k=top_k,
        max_pairwise_cosine=float(diverse_max_pairwise_cosine),
    )
    diverse_top = diverse_top_indices(
        candidates,
        memory_targets=memory_targets,
        key="combined_score",
        k=top_k,
        max_pairwise_cosine=float(diverse_max_pairwise_cosine),
    )
    variants = [
        evaluate_mixture_variant(
            name="top1_memory",
            history_level=history_level,
            window_indices=memory_top[:1],
            scores=_scores_for_indices(
                candidates, memory_top[:1], "memory_support_cosine"
            ),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
        evaluate_mixture_variant(
            name="top1_combined",
            history_level=history_level,
            window_indices=combined_top[:1],
            scores=_scores_for_indices(candidates, combined_top[:1], "combined_score"),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
        evaluate_mixture_variant(
            name="soft_topk_memory",
            history_level=history_level,
            window_indices=memory_top,
            scores=_scores_for_indices(candidates, memory_top, "memory_support_cosine"),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
        evaluate_mixture_variant(
            name="soft_topk_narrative_start",
            history_level=history_level,
            window_indices=narrative_start_top,
            scores=_scores_for_indices(
                candidates,
                narrative_start_top,
                "narrative_start_score",
            ),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
        evaluate_mixture_variant(
            name="soft_topk_narrative_start_checked",
            history_level=history_level,
            window_indices=narrative_start_checked_top,
            scores=_scores_for_indices(
                candidates,
                narrative_start_checked_top,
                "narrative_start_score",
            ),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
        evaluate_mixture_variant(
            name="soft_topk_combined",
            history_level=history_level,
            window_indices=combined_top,
            scores=_scores_for_indices(candidates, combined_top, "combined_score"),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
        evaluate_mixture_variant(
            name="diverse_topk_narrative_start",
            history_level=history_level,
            window_indices=diverse_narrative_start_top,
            scores=_scores_for_indices(
                candidates,
                diverse_narrative_start_top,
                "narrative_start_score",
            ),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
        evaluate_mixture_variant(
            name="diverse_topk_combined",
            history_level=history_level,
            window_indices=diverse_top,
            scores=_scores_for_indices(candidates, diverse_top, "combined_score"),
            grounding=grounding,
            spec_names=spec_names,
            temperature=temperature,
        ),
    ]
    candidate_by_idx = {int(row["window_index"]): row for row in candidates}
    for variant in variants:
        weighted_support = 0.0
        weighted_distance = 0.0
        for idx, weight in zip(
            variant["window_indices"], variant["weights"], strict=True
        ):
            row = candidate_by_idx[int(idx)]
            weighted_support += float(weight) * float(row["memory_support_cosine"])
            weighted_distance += float(weight) * float(row["start_distance_z"])
        selected_details = [
            candidate_by_idx[int(idx)] for idx in variant["window_indices"]
        ]
        variant["weighted_memory_support_cosine"] = float(weighted_support)
        variant["weighted_start_distance_z"] = float(weighted_distance)
        variant["candidate_details"] = selected_details
        variant["direction_check"] = direction_check_for_mixture(
            candidate_details=selected_details,
            weights=variant["weights"],
            final_mixture_alignment=variant["alignment"],
        )
    generated_alignment = case_input.get("generated_alignment", {})
    if not isinstance(generated_alignment, dict):
        generated_alignment = {}
    return {
        "case_name": str(case_input.get("case_name", "")),
        "story": str(case_input.get("story", "")),
        "query_window_index": query_window_index,
        "selected_start_status": str(case_input.get("selected_start_status", "")),
        "generated_rollout_alignment": generated_alignment,
        "variants": variants,
        "candidate_count": len(candidates),
        "artifact_paths": {
            "prefix_report": str(case_input["prefix_report"]),
            "prefix_arrays": str(case_input["prefix_arrays"]),
        },
    }


def summarize_cases(
    cases: list[dict[str, Any]],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    variant_totals: dict[str, dict[str, Any]] = {}
    generated_checked = 0
    generated_mismatches = 0
    for case in cases:
        generated = case.get("generated_rollout_alignment", {})
        if isinstance(generated, dict):
            generated_checked += int(generated.get("checked_count", 0) or 0)
            generated_mismatches += int(generated.get("mismatch_count", 0) or 0)
        for variant in case.get("variants", []):
            if not isinstance(variant, dict):
                continue
            name = str(variant.get("variant", ""))
            total = variant_totals.setdefault(
                name,
                {
                    "case_count": 0,
                    "checked_count": 0,
                    "mismatch_count": 0,
                    "status_counts": {},
                    "mean_weighted_memory_support_cosine": 0.0,
                    "mean_weighted_start_distance_z": 0.0,
                    "direction_status_counts": {},
                },
            )
            total["case_count"] += 1
            total["checked_count"] += int(variant.get("checked_count", 0) or 0)
            total["mismatch_count"] += int(variant.get("mismatch_count", 0) or 0)
            status = str(variant.get("status", "unknown"))
            total["status_counts"][status] = total["status_counts"].get(status, 0) + 1
            direction_check = variant.get("direction_check", {})
            if isinstance(direction_check, dict):
                direction_status = str(direction_check.get("status", "unknown"))
                total["direction_status_counts"][direction_status] = (
                    total["direction_status_counts"].get(direction_status, 0) + 1
                )
            total["mean_weighted_memory_support_cosine"] += float(
                variant.get("weighted_memory_support_cosine", 0.0) or 0.0
            )
            total["mean_weighted_start_distance_z"] += float(
                variant.get("weighted_start_distance_z", 0.0) or 0.0
            )
    for total in variant_totals.values():
        case_count = max(int(total["case_count"]), 1)
        checked = int(total["checked_count"])
        mismatches = int(total["mismatch_count"])
        total["mismatch_rate"] = float(mismatches / checked) if checked else None
        total["mean_weighted_memory_support_cosine"] = float(
            total["mean_weighted_memory_support_cosine"] / case_count
        )
        total["mean_weighted_start_distance_z"] = float(
            total["mean_weighted_start_distance_z"] / case_count
        )
    output_path = Path(output_dir) / "analogue_mixture_prior_summary.json"
    return {
        "status": "ok",
        "scope_note": (
            "Offline analogue-mixture prior evaluator. No OpenAI calls are made. "
            "It evaluates whether soft historical support mixtures align with "
            "grounded market implications before residual bridge training."
        ),
        "case_count": len(cases),
        "generated_rollout_baseline": {
            "checked_count": generated_checked,
            "mismatch_count": generated_mismatches,
            "mismatch_rate": (
                float(generated_mismatches / generated_checked)
                if generated_checked
                else None
            ),
        },
        "variant_totals": variant_totals,
        "cases": cases,
        "artifact_paths": {"summary": str(output_path)},
    }


def run_analogue_mixture_prior(args: argparse.Namespace) -> dict[str, Any]:
    case_inputs = _case_inputs_from_casebook(args.casebook_summary)
    oracle_arrays = dict(np.load(args.oracle_arrays))
    spec_names = load_state_spec_names(args.checkpoint)
    cases = [
        evaluate_case(
            case_input=case_input,
            oracle_arrays=oracle_arrays,
            spec_names=spec_names,
            top_k=int(args.top_k),
            temperature=float(args.temperature),
            start_distance_threshold_z=float(args.start_distance_threshold_z),
            start_distance_penalty=float(args.start_distance_penalty),
            implication_alignment_weight=float(args.implication_alignment_weight),
            diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
        )
        for case_input in case_inputs
    ]
    summary = summarize_cases(cases, output_dir=args.output_dir)
    _write_json(summary["artifact_paths"]["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook-summary", default=DEFAULT_CASEBOOK_SUMMARY)
    parser.add_argument("--oracle-arrays", default=DEFAULT_ORACLE_ARRAYS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--start-distance-threshold-z", type=float, default=15.0)
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument("--implication-alignment-weight", type=float, default=0.25)
    parser.add_argument("--diverse-max-pairwise-cosine", type=float, default=0.98)
    args = parser.parse_args()
    summary = run_analogue_mixture_prior(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "case_count": summary["case_count"],
                "generated_rollout_baseline": summary["generated_rollout_baseline"],
                "variant_totals": {
                    name: {
                        "checked_count": row["checked_count"],
                        "mismatch_count": row["mismatch_count"],
                        "mismatch_rate": row["mismatch_rate"],
                        "status_counts": row["status_counts"],
                    }
                    for name, row in summary["variant_totals"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
