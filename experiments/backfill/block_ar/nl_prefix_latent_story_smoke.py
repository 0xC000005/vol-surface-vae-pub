#!/usr/bin/env python
"""Story smoke for the narrative prefix-latent workflow.

By default this script makes no OpenAI calls: it uses a cached held-out
narrative/text memory from the representative bridge run, combines it with an
original or selected explicit starting state, decodes a synthetic recent prefix,
and runs the frozen joint39 SNI generator through its native rollout. With
``--live-story`` it grounds and embeds the typed story before the same
prefix-decoder and frozen-rollout path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    _spec_names,
    compute_memory_targets,
    summarize_retrieval_generated_states,
)
from experiments.backfill.block_ar.nl_prefix_latent_memory_decoder import (  # noqa: E402
    _decode_features,
    _future_raw_from_block,
    apply_memory_start_stats,
    build_memory_start_input_matrix,
    train_memory_start_prefix_decoder,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    build_prefix_feature_matrix,
    sample_prefix_generator_deltas,
    selected_bridge_window_indices,
    split_indices_from_bridge_report,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_sensitivity import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
    endpoint_alignment_summary,
)
from experiments.backfill.block_ar.nl_prefix_latent_validation_gate import (  # noqa: E402
    DEFAULT_GATE_THRESHOLDS,
    case_rollout_shift_rows,
    evaluate_start_case_gates,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ADAPTER,
    DEFAULT_PIPELINE_REPORT,
    DEFAULT_STORY,
    StoryGroundingResult,
    _load_bridge_adapter,
    _load_story_grounding,
    build_story_query_text,
    path_quantiles_for_generated_states,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    build_delta_scale,
    future_delta_paths,
    load_bridge_arrays,
    score_sample_distribution,
    summarize_method_scores,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    embed_texts_with_openai,
    normalize_rows,
)


StartMode = Literal[
    "original",
    "nearest_train_start",
    "farthest_train_start",
    "memory_nearest_start",
    "balanced_memory_start",
    "explicit_start_window",
]

DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_story_smoke_791a"
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


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def select_cached_story_query(
    bridge_report: dict[str, Any],
    *,
    role: str = "anchor",
    kind: str | None = None,
    window_id: str | None = None,
    query_index: int = 0,
) -> dict[str, Any]:
    """Select a cached held-out text-memory query from a bridge report."""

    rows = bridge_report.get("evaluation", {}).get("heldout_examples", [])
    if not isinstance(rows, list):
        raise ValueError("bridge report must contain evaluation.heldout_examples")
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("role", "")) != str(role):
            continue
        if kind is not None and str(row.get("kind", "")) != str(kind):
            continue
        if window_id is not None and str(row.get("window_id", "")) != str(window_id):
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError(
            f"no cached story query found for role={role!r}, kind={kind!r}, "
            f"window_id={window_id!r}"
        )
    idx = int(query_index)
    if idx < 0 or idx >= len(filtered):
        raise IndexError(f"query_index {idx} outside {len(filtered)} matching rows")
    return dict(filtered[idx])


def narrative_text_for_query(
    pipeline_report: dict[str, Any],
    query_row: dict[str, Any],
) -> str:
    """Return the cached narrative text corresponding to a bridge query row."""

    window_id = str(query_row.get("window_id", ""))
    kind = str(query_row.get("kind", ""))
    bundles = pipeline_report.get("narrative_bundles", [])
    if not isinstance(bundles, list):
        return ""
    fallback = ""
    for bundle in bundles:
        if not isinstance(bundle, dict) or str(bundle.get("window_id", "")) != window_id:
            continue
        narratives = bundle.get("narratives", [])
        if not isinstance(narratives, list):
            return ""
        for narrative in narratives:
            if not isinstance(narrative, dict):
                continue
            text = str(narrative.get("text", ""))
            if not fallback and text:
                fallback = text
            if str(narrative.get("id", "")) == kind and text:
                return text
    return fallback


def window_metadata_by_bridge_local_index(
    bridge_report: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Return metadata keyed by bridge-local selected-window index."""

    rows: dict[int, dict[str, Any]] = {}
    metadata = bridge_report.get("window_metadata", [])
    if not isinstance(metadata, list):
        return rows
    for local_idx, row in enumerate(metadata):
        if not isinstance(row, dict):
            continue
        rows[int(local_idx)] = {
            "window_id": str(row.get("window_id", f"window_{local_idx}")),
            "window_index": row.get("window_index"),
            "source_index": row.get("source_index"),
            "manifest_split": row.get("manifest_split"),
            "calendar": {
                "calendar_start_date": row.get("calendar_start_date"),
                "calendar_end_date": row.get("calendar_end_date"),
                "forecast_start_date": row.get("forecast_start_date"),
                "forecast_end_date": row.get("forecast_end_date"),
            },
            "selection_reasons": row.get("selection_reasons", []),
        }
    return rows


def _safe_start_z(start_state: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    start = np.asarray(start_state, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("train_indices must be non-empty")
    mean = start[fit].mean(axis=0, keepdims=True)
    std = np.maximum(start[fit].std(axis=0, keepdims=True), 1e-6)
    return ((start - mean) / std).astype(np.float32)


def _start_distance(
    *,
    query_window_index: int,
    start_window_index: int,
    start_state: np.ndarray,
    train_indices: np.ndarray,
) -> float:
    start_z = _safe_start_z(start_state, train_indices)
    return float(
        np.linalg.norm(start_z[int(start_window_index)] - start_z[int(query_window_index)])
    )


def _memory_support_cosine(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    window_index: int,
) -> float:
    query = np.asarray(query_memory, dtype=np.float32).reshape(-1)
    targets = np.asarray(memory_targets, dtype=np.float32)
    idx = int(window_index)
    if targets.ndim != 2:
        raise ValueError("memory_targets must have shape [N,D]")
    if query.shape[0] != targets.shape[1]:
        raise ValueError("query_memory dim must match memory_targets")
    if idx < 0 or idx >= targets.shape[0]:
        raise IndexError(f"window_index {idx} outside {targets.shape[0]}")
    target = targets[idx]
    denom = max(float(np.linalg.norm(query) * np.linalg.norm(target)), 1e-8)
    return float(np.dot(query, target) / denom)


def _nearest_memory_support_index(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    candidate_indices: np.ndarray,
) -> tuple[int, float]:
    query = np.asarray(query_memory, dtype=np.float32).reshape(1, -1)
    targets = np.asarray(memory_targets, dtype=np.float32)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("candidate_indices must be non-empty")
    if targets.ndim != 2:
        raise ValueError("memory_targets must have shape [N,D]")
    if query.shape[1] != targets.shape[1]:
        raise ValueError("query_memory dim must match memory_targets")
    if np.any(candidates < 0) or np.any(candidates >= targets.shape[0]):
        raise IndexError("candidate_indices outside memory_targets")
    candidate_targets = targets[candidates]
    denom = np.maximum(
        np.linalg.norm(candidate_targets, axis=1) * np.linalg.norm(query, axis=1)[0],
        1e-8,
    )
    cosine = np.sum(candidate_targets * query, axis=1) / denom
    best_pos = int(np.argmax(cosine))
    return int(candidates[best_pos]), float(cosine[best_pos])


def _start_distances_to_query(
    *,
    query_window_index: int,
    start_state: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    starts = np.asarray(start_state, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    start_z = _safe_start_z(starts, train)
    return np.linalg.norm(start_z[train] - start_z[int(query_window_index)], axis=1)


def _candidate_memory_support_cosines(
    *,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_memory, dtype=np.float32).reshape(1, -1)
    targets = np.asarray(memory_targets, dtype=np.float32)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("candidate_indices must be non-empty")
    if targets.ndim != 2:
        raise ValueError("memory_targets must have shape [N,D]")
    if query.shape[1] != targets.shape[1]:
        raise ValueError("query_memory dim must match memory_targets")
    if np.any(candidates < 0) or np.any(candidates >= targets.shape[0]):
        raise IndexError("candidate_indices outside memory_targets")
    candidate_targets = targets[candidates]
    denom = np.maximum(
        np.linalg.norm(candidate_targets, axis=1) * np.linalg.norm(query, axis=1)[0],
        1e-8,
    )
    return (np.sum(candidate_targets * query, axis=1) / denom).astype(np.float32)


def _balanced_memory_support_start(
    *,
    query_window_index: int,
    start_state: np.ndarray,
    train_indices: np.ndarray,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    start_distance_threshold_z: float,
    start_distance_penalty: float,
) -> dict[str, Any]:
    train = np.asarray(train_indices, dtype=np.int64)
    cosines = _candidate_memory_support_cosines(
        query_memory=query_memory,
        memory_targets=memory_targets,
        candidate_indices=train,
    )
    distances = _start_distances_to_query(
        query_window_index=int(query_window_index),
        start_state=start_state,
        train_indices=train,
    )
    threshold = float(start_distance_threshold_z)
    penalty = float(start_distance_penalty)
    inside = distances <= threshold
    score = cosines - penalty * np.maximum(distances - threshold, 0.0)
    if bool(np.any(inside)):
        candidate_positions = np.flatnonzero(inside)
        best_pos = int(candidate_positions[int(np.argmax(cosines[inside]))])
        method = "max_memory_inside_start_threshold"
    else:
        best_pos = int(np.argmax(score))
        method = "penalized_memory_no_candidate_inside_threshold"
    chosen_cosine = float(cosines[best_pos])
    chosen_distance = float(distances[best_pos])
    return {
        "start_window_index": int(train[best_pos]),
        "memory_support_cosine": chosen_cosine,
        "start_selection_score": float(score[best_pos]),
        "start_selection_method": method,
        "start_distance_threshold_z": threshold,
        "start_distance_penalty": penalty,
        "memory_support_rank": int(1 + np.sum(cosines > chosen_cosine + 1e-8)),
        "start_distance_rank": int(1 + np.sum(distances < chosen_distance - 1e-8)),
        "candidate_count": int(train.size),
        "candidate_count_inside_distance": int(np.sum(inside)),
    }


def resolve_start_window_index(
    *,
    query_window_index: int,
    start_state: np.ndarray,
    train_indices: np.ndarray,
    start_mode: StartMode | str,
    explicit_start_window_index: int | None = None,
    query_memory: np.ndarray | None = None,
    memory_targets: np.ndarray | None = None,
    start_distance_threshold_z: float = float(
        DEFAULT_GATE_THRESHOLDS["start_distance_warn"]
    ),
    start_distance_penalty: float = 0.02,
) -> dict[str, Any]:
    """Resolve a product start-selection mode to one bridge-local window index."""

    starts = np.asarray(start_state, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    query_idx = int(query_window_index)
    mode = str(start_mode)
    if query_idx < 0 or query_idx >= starts.shape[0]:
        raise IndexError(f"query_window_index {query_idx} outside {starts.shape[0]}")
    if mode == "original":
        start_idx = query_idx
        support_cosine = None
        proposal: dict[str, Any] = {}
    elif mode == "explicit_start_window":
        if explicit_start_window_index is None:
            raise ValueError("explicit_start_window_index is required")
        start_idx = int(explicit_start_window_index)
        support_cosine = None
        proposal = {}
    elif mode == "memory_nearest_start":
        if query_memory is None or memory_targets is None:
            raise ValueError(
                "query_memory and memory_targets are required for memory_nearest_start"
            )
        start_idx, support_cosine = _nearest_memory_support_index(
            query_memory=query_memory,
            memory_targets=memory_targets,
            candidate_indices=train,
        )
        proposal = {
            "start_selection_score": float(support_cosine),
            "start_selection_method": "max_memory_support",
            "memory_support_rank": 1,
            "candidate_count": int(train.size),
        }
    elif mode == "balanced_memory_start":
        if query_memory is None or memory_targets is None:
            raise ValueError(
                "query_memory and memory_targets are required for balanced_memory_start"
            )
        proposal = _balanced_memory_support_start(
            query_window_index=query_idx,
            start_state=starts,
            train_indices=train,
            query_memory=query_memory,
            memory_targets=memory_targets,
            start_distance_threshold_z=float(start_distance_threshold_z),
            start_distance_penalty=float(start_distance_penalty),
        )
        start_idx = int(proposal["start_window_index"])
        support_cosine = float(proposal["memory_support_cosine"])
    else:
        start_z = _safe_start_z(starts, train)
        distances = np.linalg.norm(start_z[train] - start_z[query_idx], axis=1)
        if mode == "nearest_train_start":
            start_idx = int(train[int(np.argmin(distances))])
        elif mode == "farthest_train_start":
            start_idx = int(train[int(np.argmax(distances))])
        else:
            raise ValueError(f"unknown start_mode: {start_mode!r}")
        support_cosine = None
        proposal = {}
    if start_idx < 0 or start_idx >= starts.shape[0]:
        raise IndexError(f"start_window_index {start_idx} outside {starts.shape[0]}")
    if support_cosine is None and query_memory is not None and memory_targets is not None:
        support_cosine = _memory_support_cosine(
            query_memory=query_memory,
            memory_targets=memory_targets,
            window_index=int(start_idx),
        )
    result = {
        "query_window_index": query_idx,
        "start_window_index": int(start_idx),
        "variant": mode,
        "start_distance_z": _start_distance(
            query_window_index=query_idx,
            start_window_index=int(start_idx),
            start_state=starts,
            train_indices=train,
        ),
    }
    if support_cosine is not None:
        result["memory_support_cosine"] = float(support_cosine)
    result.update(proposal)
    return result


def build_live_story_variant_rows(
    *,
    query_row: dict[str, Any],
    start_state: np.ndarray,
    train_indices: np.ndarray,
    start_mode: StartMode | str,
    explicit_start_window_index: int | None = None,
    include_original_baseline: bool = True,
    query_memory: np.ndarray | None = None,
    memory_targets: np.ndarray | None = None,
    start_distance_threshold_z: float = float(
        DEFAULT_GATE_THRESHOLDS["start_distance_warn"]
    ),
    start_distance_penalty: float = 0.02,
) -> list[dict[str, Any]]:
    """Build original and selected-start rows for one live cached story query."""

    query_idx = int(query_row["window_index"])
    original = resolve_start_window_index(
        query_window_index=query_idx,
        start_state=start_state,
        train_indices=train_indices,
        start_mode="original",
        query_memory=query_memory,
        memory_targets=memory_targets,
        start_distance_threshold_z=float(start_distance_threshold_z),
        start_distance_penalty=float(start_distance_penalty),
    )
    selected = resolve_start_window_index(
        query_window_index=query_idx,
        start_state=start_state,
        train_indices=train_indices,
        start_mode=start_mode,
        explicit_start_window_index=explicit_start_window_index,
        query_memory=query_memory,
        memory_targets=memory_targets,
        start_distance_threshold_z=float(start_distance_threshold_z),
        start_distance_penalty=float(start_distance_penalty),
    )
    original.update(
            {
                "query_window_id": str(query_row.get("window_id", "")),
                "embedding_index": int(query_row.get("embedding_index", -1)),
                "kind": str(query_row.get("kind", "")),
                "role": str(query_row.get("role", "")),
            }
    )
    selected.update(
            {
                "query_window_id": str(query_row.get("window_id", "")),
                "embedding_index": int(query_row.get("embedding_index", -1)),
                "kind": str(query_row.get("kind", "")),
                "role": str(query_row.get("role", "")),
            }
    )
    if not include_original_baseline or selected["start_window_index"] == query_idx:
        return [selected if not include_original_baseline else original]
    return [original, selected]


def generated_delta_samples_to_states(
    samples: np.ndarray,
    current_states: np.ndarray,
) -> np.ndarray:
    """Convert generated raw-delta samples to generated raw state paths."""

    delta = np.asarray(samples, dtype=np.float32)
    current = np.asarray(current_states, dtype=np.float32)
    if delta.ndim != 4:
        raise ValueError("samples must have shape [K,S,T,C]")
    if current.shape != (delta.shape[0], delta.shape[-1]):
        raise ValueError("current_states must have shape [K,C]")
    return (current[:, None, None, :] + delta).astype(np.float32)


def build_live_story_condition_memory(
    *,
    story: str,
    grounding: StoryGroundingResult,
    embedding_model: str,
    bridge_adapter: str | Path,
    condition_dim: int,
    dotenv_path: str | Path,
    embedder: Any = embed_texts_with_openai,
    adapter_loader: Any = _load_bridge_adapter,
) -> dict[str, Any]:
    """Project a grounded live story into generator condition-memory space."""

    query_text = build_story_query_text(story, grounding)
    query_embedding = np.asarray(
        embedder(
            [query_text],
            model=str(embedding_model),
            dotenv_path=dotenv_path,
            batch_size=1,
        ),
        dtype=np.float32,
    )
    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError("embedder must return one 2-D embedding row")
    adapter = adapter_loader(
        bridge_adapter,
        embedding_dim=int(query_embedding.shape[1]),
        condition_dim=int(condition_dim),
    )
    with torch.no_grad():
        query_condition = (
            adapter(torch.from_numpy(normalize_rows(query_embedding)).float())
            .detach()
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )
    return {
        "query_text": query_text,
        "query_condition": query_condition,
        "embedding_metadata": {
            "embedding_model": str(embedding_model),
            "embedding_dim": int(query_embedding.shape[1]),
            "condition_dim": int(condition_dim),
            "query_text_length": int(len(query_text)),
            "query_text_line_count": int(query_text.count("\n") + 1),
        },
    }


def _score_live_rollouts(
    *,
    samples: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    variant_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_no, row in enumerate(variant_rows):
        start_idx = int(row["start_window_index"])
        target = np.asarray(future_delta[start_idx], dtype=np.float32)
        rows.append(
            {
                "case_index": int(row_no),
                "query_window_index": int(row["query_window_index"]),
                "start_window_index": start_idx,
                "variant": str(row["variant"]),
                "methods": {
                    "persistence": score_sample_distribution(
                        np.zeros((1, *target.shape), dtype=np.float32),
                        target,
                        scale=delta_scale,
                    ),
                    "text_memory_plus_start_prefix_decoder": score_sample_distribution(
                        samples[row_no],
                        target,
                        scale=delta_scale,
                    ),
                },
            }
        )
    return rows, summarize_method_scores(rows, baseline="persistence")


def _variant_path_labels(
    variant_rows: list[dict[str, Any]],
    window_metadata: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for row in variant_rows:
        start_idx = int(row["start_window_index"])
        info = window_metadata.get(start_idx, {})
        labels.append(
            {
                "window_id": str(info.get("window_id", f"window_{start_idx}")),
                "variant": str(row.get("variant", "")),
                "start_window_index": start_idx,
            }
        )
    return labels


def _enrich_variant_rows(
    rows: list[dict[str, Any]],
    window_metadata: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        start_info = window_metadata.get(int(row["start_window_index"]), {})
        query_info = window_metadata.get(int(row["query_window_index"]), {})
        enriched.append(
            {
                **row,
                "query_window_id": str(
                    query_info.get("window_id", row.get("query_window_id", ""))
                ),
                "start_window_id": str(start_info.get("window_id", "")),
                "start_source_index": start_info.get("source_index"),
                "start_manifest_split": start_info.get("manifest_split"),
                "start_calendar": start_info.get("calendar", {}),
            }
        )
    return enriched


def _render_markdown(report: dict[str, Any]) -> str:
    query = report.get("cached_query", {})
    gate = report.get("validation_gate", {})
    generation = report.get("generation", {})
    decoder = report.get("decoder", {})
    live_story = str(query.get("condition_source", "")) == "live_openai_story"
    narrative_heading = "Live Narrative" if live_story else "Cached Narrative"
    condition_line = (
        "Live OpenAI-grounded text memory is used as the narrative condition."
        if live_story
        else "Cached text memory is used as the narrative condition."
    )
    lines = [
        "# Prefix-Latent Story Smoke",
        "",
        f"## {narrative_heading}",
        "",
        str(query.get("narrative_text", "")),
        "",
        "## Product Contract",
        "",
        f"- {condition_line}",
        "- The requested start state is pinned as the final prefix level.",
        "- A learned memory+start decoder reconstructs a recent prefix object.",
        "- The frozen joint39 SNI generator performs the 30-day rollout.",
        "",
        "## Start Selection",
        "",
        "| Variant | Query Window | Start Window | Start Distance z | Memory Support Cosine |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in report.get("variant_rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("variant", "")),
                    str(row.get("query_window_id", "")),
                    str(row.get("start_window_id", "")),
                    f"{float(row.get('start_distance_z', 0.0)):.3f}",
                    (
                        f"{float(row['memory_support_cosine']):.3f}"
                        if row.get("memory_support_cosine") is not None
                        else "n/a"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Overall: `{gate.get('overall_status')}`",
            f"- Operational: `{gate.get('operational_status')}`",
            f"- Stress: `{gate.get('stress_status')}`",
            f"- Endpoint max error: {gate.get('endpoint_max_abs_error')}",
            f"- Warning counts: {gate.get('warning_counts')}",
            f"- Failure counts: {gate.get('fail_counts')}",
            "",
            "## Decoder",
            "",
            f"- Train MSE: {decoder.get('train_mse')}",
            f"- Test MSE: {decoder.get('test_mse')}",
            f"- Loss first/last: {decoder.get('loss_first')} / {decoder.get('loss_last')}",
            "",
            "## Scenario",
            "",
            f"- Generated shape: {generation.get('generated_state_shape')}",
            f"- Finite rate: {generation.get('finite_rate')}",
            f"- Rollout scores: {generation.get('rollout_summary')}",
        ]
    )
    return "\n".join(lines)


def run_prefix_latent_story_smoke(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    bridge_report = _load_json(args.bridge_report)
    pipeline_report = _load_json(args.pipeline_report)
    selected_windows = selected_bridge_window_indices(bridge_report)
    train_indices, test_indices = split_indices_from_bridge_report(bridge_report)
    bridge_arrays = load_bridge_arrays(args.bridge_arrays)
    true_memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    device = torch.device(
        args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu"
    )
    model, payload = load_model(args.checkpoint, device)
    (
        all_history_level,
        all_history_norm,
        all_center,
        all_scale,
        all_drift_feature,
        all_history_raw,
        specs,
        block,
    ) = build_val_block(args, payload)
    history_level = all_history_level[selected_windows]
    history_norm = all_history_norm[selected_windows]
    center = all_center[selected_windows]
    scale = all_scale[selected_windows]
    drift_feature = all_drift_feature[selected_windows]
    history_raw = all_history_raw[selected_windows]
    future_raw_all = _future_raw_from_block(
        block,
        int(all_history_raw.shape[0]),
        int(all_history_raw.shape[-1]),
    )
    future_raw = future_raw_all[selected_windows]
    future_delta = future_delta_paths(all_history_raw, future_raw_all)[selected_windows]
    features, layout = build_prefix_feature_matrix(
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
    )
    inputs, input_stats = build_memory_start_input_matrix(
        true_memory_targets,
        history_level[:, -1, :],
        fit_indices=train_indices,
    )
    decoder_result = train_memory_start_prefix_decoder(
        inputs,
        features,
        train_indices=train_indices,
        test_indices=test_indices,
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=device,
    )
    query_row = select_cached_story_query(
        bridge_report,
        role=str(args.query_role),
        kind=args.query_kind,
        window_id=args.query_window_id,
        query_index=int(args.query_index),
    )
    grounding_payload: dict[str, Any] | None = None
    query_text: str | None = None
    embedding_metadata: dict[str, Any] = {}
    condition_source = "cached_bridge_query"
    if bool(getattr(args, "live_story", False)):
        grounding = _load_story_grounding(
            story=str(args.story),
            grounding_json=args.grounding_json,
            model=str(args.grounding_model),
            dotenv_path=args.dotenv,
            max_output_tokens=int(args.grounding_max_output_tokens),
        )
        live_condition = build_live_story_condition_memory(
            story=str(args.story),
            grounding=grounding,
            embedding_model=str(args.embedding_model),
            bridge_adapter=args.bridge_adapter,
            condition_dim=int(true_memory_targets.shape[1]),
            dotenv_path=args.dotenv,
        )
        query_memory = np.asarray(live_condition["query_condition"], dtype=np.float32)
        query_text = str(live_condition["query_text"])
        embedding_metadata = dict(live_condition["embedding_metadata"])
        embedding_metadata["grounding_model"] = str(args.grounding_model)
        grounding_payload = grounding.model_dump()
        condition_source = "live_openai_story"
        query_row = {
            **query_row,
            "role": "live_story",
            "kind": "live_story",
            "embedding_index": -1,
        }
    else:
        query_memory = np.asarray(
            bridge_arrays["condition_vectors"][int(query_row["embedding_index"])],
            dtype=np.float32,
        )
        embedding_metadata = {
            "condition_dim": int(query_memory.shape[0]),
            "embedding_index": int(query_row["embedding_index"]),
        }
    variant_rows = build_live_story_variant_rows(
        query_row=query_row,
        start_state=history_level[:, -1, :],
        train_indices=train_indices,
        start_mode=str(args.start_mode),
        explicit_start_window_index=args.explicit_start_window_index,
        include_original_baseline=bool(args.include_original_baseline),
        query_memory=query_memory,
        memory_targets=true_memory_targets,
        start_distance_threshold_z=float(args.start_distance_threshold_z),
        start_distance_penalty=float(args.start_distance_penalty),
    )
    window_metadata = window_metadata_by_bridge_local_index(bridge_report)
    variant_rows = _enrich_variant_rows(variant_rows, window_metadata)
    text_memory = np.repeat(query_memory[None, :], repeats=len(variant_rows), axis=0)
    start_indices = np.asarray(
        [int(row["start_window_index"]) for row in variant_rows],
        dtype=np.int64,
    )
    requested_start = history_level[start_indices, -1, :]
    variant_inputs = apply_memory_start_stats(text_memory, requested_start, input_stats)
    decoded_prefix = _decode_features(
        decoder_result["model"],
        decoder_result["target_mean"],
        decoder_result["target_std"],
        variant_inputs,
        start_state=requested_start,
        layout=layout,
        device=device,
    )
    endpoint = endpoint_alignment_summary(
        decoded_prefix["history_level"],
        requested_start,
    )
    decoded_memory = compute_memory_targets(
        model,
        decoded_prefix["history_level"],
        decoded_prefix["history_norm"],
        decoded_prefix["center"],
        decoded_prefix["scale"],
        decoded_prefix["drift_feature"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    train_delta = future_delta[train_indices]
    delta_scale = build_delta_scale(train_delta, floor=float(args.score_scale_floor))
    samples = np.empty((0,), dtype=np.float32)
    generated_states = np.empty((0,), dtype=np.float32)
    window_scores: list[dict[str, Any]] = []
    rollout_summary: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    if not bool(args.skip_rollout):
        samples = sample_prefix_generator_deltas(
            model,
            indices=np.arange(len(variant_rows), dtype=np.int64),
            history_level=decoded_prefix["history_level"],
            history_norm=decoded_prefix["history_norm"],
            center=decoded_prefix["center"],
            scale=decoded_prefix["scale"],
            drift_feature=decoded_prefix["drift_feature"],
            history_raw=history_raw[start_indices],
            specs=specs,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        current_raw = history_raw[start_indices, -1, :]
        generated_states = generated_delta_samples_to_states(samples, current_raw)
        window_scores, rollout_summary = _score_live_rollouts(
            samples=samples,
            future_delta=future_delta,
            delta_scale=delta_scale,
            variant_rows=variant_rows,
        )
        generation = summarize_retrieval_generated_states(
            generated_states,
            current_raw,
            _spec_names(specs),
        )
        generation["path_quantiles"] = path_quantiles_for_generated_states(
            generated_states,
            current_raw,
            _spec_names(specs),
            analogues=_variant_path_labels(variant_rows, window_metadata),
            future_states=future_raw[start_indices],
            max_paths=int(args.max_paths),
        )
        generation["window_scores"] = window_scores
        generation["rollout_summary"] = rollout_summary
    rollout_shifts = case_rollout_shift_rows(
        samples=samples,
        variant_rows=variant_rows,
        scale=delta_scale,
    )
    validation_gate = evaluate_start_case_gates(
        variant_rows=variant_rows,
        decoded_memory=decoded_memory,
        text_memory=text_memory,
        rollout_shifts=rollout_shifts,
        endpoint_max_abs_error=float(endpoint["max_abs_error"]),
        hard_case_count=int(args.hard_case_count),
    )
    narrative_text = (
        str(args.story)
        if bool(getattr(args, "live_story", False))
        else narrative_text_for_query(pipeline_report, query_row)
    )
    live_story_mode = bool(getattr(args, "live_story", False))
    report = {
        "status": "ok",
        "scope_note": (
            "Live OpenAI-grounded prefix-latent story smoke. OpenAI is called for "
            "story grounding and text embedding, then the projected memory is "
            "combined with an explicit start state, decoded into a recent prefix, "
            "and rolled out through the frozen joint39 SNI generator."
            if live_story_mode
            else (
                "Cached live prefix-latent story smoke. No OpenAI API calls are made. "
                "A cached text-predicted generator memory is combined with an explicit "
                "start state, decoded into a recent prefix, and rolled out through the "
                "frozen joint39 SNI generator."
            )
        ),
        "cached_query": {
            **query_row,
            "condition_source": condition_source,
            "narrative_text": narrative_text,
            "query_text": query_text,
            "grounding": grounding_payload,
            "embedding_metadata": embedding_metadata,
            "text_memory_dim": int(query_memory.shape[0]),
            "query_memory_norm": float(np.linalg.norm(query_memory)),
        },
        "artifact_inputs": {
            "bridge_report": str(args.bridge_report),
            "bridge_arrays": str(args.bridge_arrays),
            "pipeline_report": str(args.pipeline_report),
            "checkpoint": str(args.checkpoint),
        },
        "selected_window_count": int(selected_windows.size),
        "train_window_count": int(train_indices.size),
        "test_window_count": int(test_indices.size),
        "device": str(device),
        "decoder": {
            "loss_first": float(decoder_result["loss_first"]),
            "loss_last": float(decoder_result["loss_last"]),
            "train_mse": float(decoder_result["train_mse"]),
            "test_mse": float(decoder_result["test_mse"]),
        },
        "endpoint_alignment": endpoint,
        "variant_rows": variant_rows,
        "rollout_shifts": rollout_shifts,
        "validation_gate": validation_gate,
        "generation": generation,
        "artifact_paths": {
            "report": str(Path(args.output_dir) / "prefix_latent_story_smoke_report.json"),
            "markdown": str(Path(args.output_dir) / "prefix_latent_story_smoke_report.md"),
            "arrays": str(Path(args.output_dir) / "prefix_latent_story_smoke_arrays.npz"),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "prefix_latent_story_smoke_arrays.npz",
        selected_window_indices=selected_windows.astype(np.int64),
        train_indices=train_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        start_indices=start_indices.astype(np.int64),
        requested_start=requested_start.astype(np.float32),
        text_memory=text_memory.astype(np.float32),
        decoded_memory=decoded_memory.astype(np.float32),
        decoded_history_level=decoded_prefix["history_level"].astype(np.float32),
        decoded_history_norm=decoded_prefix["history_norm"].astype(np.float32),
        decoded_center=decoded_prefix["center"].astype(np.float32),
        decoded_scale=decoded_prefix["scale"].astype(np.float32),
        decoded_drift_feature=decoded_prefix["drift_feature"].astype(np.float32),
        delta_scale=delta_scale.astype(np.float32),
        samples=samples.astype(np.float32),
        generated_states=generated_states.astype(np.float32),
    )
    _write_json(output_dir / "prefix_latent_story_smoke_report.json", report)
    _write_text(output_dir / "prefix_latent_story_smoke_report.md", _render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-role", default="anchor")
    parser.add_argument("--query-kind")
    parser.add_argument("--query-window-id")
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--live-story", action="store_true")
    parser.add_argument("--story", default=DEFAULT_STORY)
    parser.add_argument("--grounding-json")
    parser.add_argument("--grounding-model", default="gpt-5.4-mini")
    parser.add_argument("--grounding-max-output-tokens", type=int, default=1200)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--bridge-adapter", default=DEFAULT_BRIDGE_ADAPTER)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument(
        "--start-mode",
        choices=[
            "original",
            "nearest_train_start",
            "farthest_train_start",
            "memory_nearest_start",
            "balanced_memory_start",
            "explicit_start_window",
        ],
        default="balanced_memory_start",
    )
    parser.add_argument("--explicit-start-window-index", type=int)
    parser.add_argument(
        "--start-distance-threshold-z",
        type=float,
        default=float(DEFAULT_GATE_THRESHOLDS["start_distance_warn"]),
    )
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument("--include-original-baseline", action="store_true", default=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=791)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
    parser.add_argument("--hard-case-count", type=int, default=8)
    parser.add_argument("--max-paths", type=int, default=6)
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument("--eval_split", choices=["val", "train", "train_tail"], default="val")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    args = parser.parse_args()
    report = run_prefix_latent_story_smoke(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
                "arrays": report["artifact_paths"]["arrays"],
                "device": report["device"],
                "query_window": report["cached_query"]["window_id"],
                "condition_source": report["cached_query"]["condition_source"],
                "start_mode": args.start_mode,
                "variant_count": len(report["variant_rows"]),
                "validation_status": report["validation_gate"]["overall_status"],
                "operational_status": report["validation_gate"]["operational_status"],
                "generated_shape": report["generation"].get("generated_state_shape"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
