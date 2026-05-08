#!/usr/bin/env python
"""Offline analogue-mixture prior evaluator for narrative scenario support.

This script makes no OpenAI calls. It tests the new main direction before
training a residual bridge: retrieve several historical analogue prefixes,
build soft mixtures, and ask whether the mixture prior is more aligned with the
grounded story implications than top-1 analogue or generated-rollout baselines.
"""

from __future__ import annotations

import argparse
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
        combined_score = (
            float(cosines[pos])
            + float(implication_alignment_weight) * recent_score
            - start_distance_cost
            - excess_distance_cost
        )
        rows.append(
            {
                "window_index": int(window_idx),
                "memory_support_cosine": float(cosines[pos]),
                "start_distance_z": float(distances[pos]),
                "start_distance_cost": float(start_distance_cost),
                "excess_start_distance_cost": float(excess_distance_cost),
                "recent_prefix_alignment_score": float(recent_score),
                "recent_prefix_checked": int(alignment.get("checked_count", 0) or 0),
                "recent_prefix_mismatches": int(
                    alignment.get("mismatch_count", 0) or 0
                ),
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


def diverse_top_indices(
    rows: list[dict[str, Any]],
    *,
    memory_targets: np.ndarray,
    key: str,
    k: int,
    max_pairwise_cosine: float,
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
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= int(k):
                break
    return np.asarray(selected[: max(int(k), 1)], dtype=np.int64)


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
    if prior_mode == "soft_topk_memory":
        indices = _top_indices(candidates, "memory_support_cosine", top_k)
        scores = _scores_for_indices(candidates, indices, "memory_support_cosine")
    elif prior_mode == "soft_topk_combined":
        indices = _top_indices(candidates, "combined_score", top_k)
        scores = _scores_for_indices(candidates, indices, "combined_score")
    elif prior_mode == "diverse_topk_combined":
        indices = diverse_top_indices(
            candidates,
            memory_targets=memory,
            key="combined_score",
            k=top_k,
            max_pairwise_cosine=float(diverse_max_pairwise_cosine),
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
    return {
        "mode": prior_mode,
        "memory": mixture_memory,
        "analogue_count": int(indices.size),
        "window_indices": [int(idx) for idx in indices],
        "weights": [float(weight) for weight in weights],
        "support_alignment": support_alignment,
        "terminal_rows": terminal_rows,
        "candidate_details": [by_idx[int(idx)] for idx in indices],
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
    combined_top = _top_indices(candidates, "combined_score", top_k)
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
            name="soft_topk_combined",
            history_level=history_level,
            window_indices=combined_top,
            scores=_scores_for_indices(candidates, combined_top, "combined_score"),
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
        variant["weighted_memory_support_cosine"] = float(weighted_support)
        variant["weighted_start_distance_z"] = float(weighted_distance)
        variant["candidate_details"] = [
            candidate_by_idx[int(idx)] for idx in variant["window_indices"]
        ]
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
                },
            )
            total["case_count"] += 1
            total["checked_count"] += int(variant.get("checked_count", 0) or 0)
            total["mismatch_count"] += int(variant.get("mismatch_count", 0) or 0)
            status = str(variant.get("status", "unknown"))
            total["status_counts"][status] = total["status_counts"].get(status, 0) + 1
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
