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
    KEY_FACTOR_NAMES,
    _reconstruct_states,
    _spec_names,
    build_window_metadata,
    compute_memory_targets,
    summarize_retrieval_generated_states,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    encode_state,
)
from experiments.backfill.block_ar.nl_prefix_latent_market_alignment import (  # noqa: E402
    market_implication_alignment,
)
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (  # noqa: E402
    build_mixture_memory_prior,
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
    reconstruct_prefix_from_features,
    selected_bridge_window_indices,
    split_indices_from_bridge_report,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_sensitivity import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
    endpoint_alignment_summary,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_reliability_gate import (  # noqa: E402
    evaluate_start_reliability,
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
    "implication_aligned_start",
    "explicit_start_window",
    "user_start_state",
]
MemoryPriorMode = Literal[
    "query_memory",
    "soft_topk_memory",
    "soft_topk_narrative_start",
    "soft_topk_narrative_start_checked",
    "diverse_topk_narrative_start_checked",
    "cohesive_topk_narrative_start_checked",
    "cluster_family_narrative_start_checked",
    "kernel_topk_narrative_start_checked",
    "portfolio_quality_guard_924e",
    "portfolio_direction_first_quality_guard_938a",
    "broad_replay_response_guard_940a",
    "narrative_book_quality_guard_926b",
    "narrative_book_direction_first_quality_guard_938c",
    "soft_topk_start_only",
    "soft_topk_combined",
    "diverse_topk_narrative_start",
    "diverse_topk_combined",
]

RESPONSE_DIRECTION_SIGN = {
    "up": 1.0,
    "higher": 1.0,
    "wider": 1.0,
    "steeper": 1.0,
    "down": -1.0,
    "lower": -1.0,
    "tighter": -1.0,
    "narrower": -1.0,
    "flatter": -1.0,
    "flat": 0.0,
    "mixed": 0.0,
    "unchanged": 0.0,
}
RESPONSE_CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
RESPONSE_MAGNITUDE_WEIGHT = {"large": 1.2, "medium": 1.0, "small": 0.75}
RESPONSE_MARKET_ALIASES = {
    "CRUDE": "CRUDE_OIL",
    "CRUDE OIL": "CRUDE_OIL",
    "OIL": "CRUDE_OIL",
    "RATES": "US10Y",
    "TREASURY_10Y": "US10Y",
    "UST10Y": "US10Y",
    "CREDIT_SPREADS": "BBB_OAS",
    "SPREADS": "BBB_OAS",
}
PrefixPriorMode = Literal[
    "decoder",
    "feature_mixture",
]
RolloutMixtureMode = Literal[
    "averaged_prefix",
    "component_prefix_mixture",
    "response_preview_component_mixture",
]
ResponsePreviewObjective = Literal[
    "narrative_channels",
    "factor_portfolio",
    "channel_portfolio",
]

RESPONSE_PORTFOLIO_EXPOSURES = (
    {"market": "SPX", "sensitivity": 1.00},
    {"market": "VIX", "sensitivity": -0.55},
    {"market": "BBB_OAS", "sensitivity": -0.45},
    {"market": "US10Y", "sensitivity": -0.35},
    {"market": "DXY", "sensitivity": -0.25},
    {"market": "CRUDE_OIL", "sensitivity": 0.20},
    {"market": "GOLD", "sensitivity": 0.15},
)

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


def _story_smoke_start_reliability(
    *,
    manifest_path: str | None,
    operational_row: dict[str, Any],
) -> dict[str, Any] | None:
    if not manifest_path:
        return None
    manifest = _load_json(manifest_path)
    start_window_index = operational_row.get("start_window_index")
    if start_window_index is None:
        return {
            "start_name": "unknown",
            "product_status": "warn_needs_stronger_evidence",
            "decision": "No fixed-start reliability evidence is available.",
            "warnings": ["selected_start_index_missing"],
            "failures": [],
            "source": "fallback",
            "manifest_path": str(manifest_path),
        }
    reliability = evaluate_start_reliability(
        manifest,
        f"fixed_start_{int(start_window_index)}",
    )
    reliability["manifest_path"] = str(manifest_path)
    return reliability


def _condition_arrays_path_from_report(
    report_path: str | Path,
    report: dict[str, Any],
) -> Path:
    artifact_paths = report.get("artifact_paths", {})
    if isinstance(artifact_paths, dict) and artifact_paths.get("arrays"):
        return Path(str(artifact_paths["arrays"]))
    return Path(report_path).with_name("prefix_latent_story_smoke_arrays.npz")


def _external_condition_from_report(report_path: str | Path) -> dict[str, Any]:
    report = _load_json(report_path)
    query = report.get("cached_query", {})
    if not isinstance(query, dict):
        raise ValueError(f"{report_path}: missing cached_query")
    arrays_path = _condition_arrays_path_from_report(report_path, report)
    arrays = np.load(arrays_path)
    if "text_memory" not in arrays:
        raise ValueError(f"{arrays_path}: missing text_memory")
    text_memory = np.asarray(arrays["text_memory"], dtype=np.float32)
    if text_memory.ndim != 2 or text_memory.shape[0] < 1:
        raise ValueError(f"{arrays_path}: text_memory must have shape [N,D]")
    grounding = query.get("grounding")
    if not isinstance(grounding, dict):
        grounding = {}
    return {
        "query": query,
        "query_memory": text_memory[0].astype(np.float32),
        "query_text": query.get("query_text"),
        "grounding": grounding,
        "embedding_metadata": query.get("embedding_metadata", {}),
        "arrays_path": str(arrays_path),
    }


def load_user_start_state(
    start_state_json: str | Path,
    specs: list[Any],
) -> dict[str, Any]:
    """Load a user-supplied raw joint state and encode it for the generator."""

    payload = _load_json(start_state_json)
    names = _spec_names(specs)
    coordinate = str(payload.get("coordinate", "raw_state"))
    if coordinate != "raw_state":
        raise ValueError(
            "start-state JSON currently supports coordinate='raw_state' only"
        )
    if "state_vector" in payload:
        raw = np.asarray(payload["state_vector"], dtype=np.float32)
        source_format = "state_vector"
    elif "values_by_name" in payload:
        values = payload["values_by_name"]
        if not isinstance(values, dict):
            raise ValueError("values_by_name must be a JSON object")
        missing = [name for name in names if name not in values]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"values_by_name missing {len(missing)} fields: {preview}")
        raw = np.asarray([values[name] for name in names], dtype=np.float32)
        source_format = "values_by_name"
    else:
        raise ValueError("start-state JSON must contain state_vector or values_by_name")
    if raw.shape != (len(specs),):
        raise ValueError(f"start state must have shape [{len(specs)}], got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("start state contains non-finite values")
    encoded = encode_state(raw[None, :], specs)[0].astype(np.float32)
    if not np.all(np.isfinite(encoded)):
        raise ValueError("encoded start state contains non-finite values")
    label = str(payload.get("label") or "user_supplied_start")
    return {
        "label": label,
        "source_path": str(start_state_json),
        "source_format": source_format,
        "coordinate": coordinate,
        "raw_state": raw.astype(np.float32),
        "encoded_state": encoded,
        "spec_names": names,
    }


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
        if (
            not isinstance(bundle, dict)
            or str(bundle.get("window_id", "")) != window_id
        ):
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


def window_metadata_by_local_index(
    metadata: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Return metadata keyed by local row index for a support/start bank."""

    rows: dict[int, dict[str, Any]] = {}
    for local_idx, row in enumerate(metadata):
        if not isinstance(row, dict):
            continue
        rows[int(local_idx)] = {
            "window_id": str(row.get("window_id", f"window_{local_idx}")),
            "window_index": row.get("window_index", local_idx),
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


def _load_support_bank(
    *,
    report_path: str | Path,
    arrays_path: str | Path,
) -> dict[str, Any]:
    """Load a broad support bank built by nl_prefix_latent_support_bank.py."""

    report = _load_json(report_path)
    arrays = np.load(arrays_path)
    required = {
        "memory_targets",
        "history_level",
        "history_norm",
        "center",
        "scale",
        "drift_feature",
        "history_raw",
        "future_raw",
        "future_delta",
        "train_indices",
        "test_indices",
    }
    missing = sorted(required.difference(arrays.files))
    if missing:
        raise ValueError(f"{arrays_path}: missing support-bank arrays {missing}")
    metadata = report.get("window_metadata", [])
    if not isinstance(metadata, list):
        raise ValueError(f"{report_path}: window_metadata must be a list")
    n = int(np.asarray(arrays["memory_targets"]).shape[0])
    if len(metadata) != n:
        raise ValueError(
            f"{report_path}: metadata rows {len(metadata)} != support rows {n}"
        )
    return {
        "report": report,
        "memory_targets": np.asarray(arrays["memory_targets"], dtype=np.float32),
        "history_level": np.asarray(arrays["history_level"], dtype=np.float32),
        "history_norm": np.asarray(arrays["history_norm"], dtype=np.float32),
        "center": np.asarray(arrays["center"], dtype=np.float32),
        "scale": np.asarray(arrays["scale"], dtype=np.float32),
        "drift_feature": np.asarray(arrays["drift_feature"], dtype=np.float32),
        "history_raw": np.asarray(arrays["history_raw"], dtype=np.float32),
        "future_raw": np.asarray(arrays["future_raw"], dtype=np.float32),
        "future_delta": np.asarray(arrays["future_delta"], dtype=np.float32),
        "train_indices": np.asarray(arrays["train_indices"], dtype=np.int64),
        "test_indices": np.asarray(arrays["test_indices"], dtype=np.int64),
        "support_indices": np.asarray(
            (
                arrays["support_indices"]
                if "support_indices" in arrays.files
                else arrays["train_indices"]
            ),
            dtype=np.int64,
        ),
        "metadata": metadata,
    }


def _safe_start_z(start_state: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    start = np.asarray(start_state, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("train_indices must be non-empty")
    mean, std = _fit_start_stats(start, fit)
    return ((start - mean) / std).astype(np.float32)


def _fit_start_stats(
    start_state: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_state, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if start.ndim != 2:
        raise ValueError("start_state must have shape [N,C]")
    if fit.size == 0:
        raise ValueError("fit_indices must be non-empty")
    mean = start[fit].mean(axis=0, keepdims=True)
    std = np.maximum(start[fit].std(axis=0, keepdims=True), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def user_start_support_summary(
    *,
    encoded_start: np.ndarray,
    history_level: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    """Score a supplied start state against the training start-state manifold."""

    starts = np.asarray(history_level, dtype=np.float32)[:, -1, :]
    train = np.asarray(train_indices, dtype=np.int64)
    encoded = np.asarray(encoded_start, dtype=np.float32)
    if encoded.shape != (starts.shape[1],):
        raise ValueError(f"encoded_start must have shape [{starts.shape[1]}]")
    mean, std = _fit_start_stats(starts, train)
    train_z = (starts[train] - mean) / std
    user_z = (encoded[None, :] - mean) / std
    distance = np.linalg.norm(train_z - user_z, axis=1) / np.sqrt(starts.shape[1])
    best_pos = int(np.argmin(distance))
    return {
        "nearest_train_start_window_index": int(train[best_pos]),
        "nearest_train_start_distance_z": float(distance[best_pos]),
        "mean_train_start_distance_z": float(np.mean(distance)),
        "max_abs_user_start_z": float(np.max(np.abs(user_z))),
    }


def build_user_start_variant_row(
    *,
    query_row: dict[str, Any],
    user_start: dict[str, Any],
    history_level: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, Any]:
    support = user_start_support_summary(
        encoded_start=np.asarray(user_start["encoded_state"], dtype=np.float32),
        history_level=history_level,
        train_indices=train_indices,
    )
    return {
        "variant": "user_start_state",
        "query_window_index": int(query_row["window_index"]),
        "start_window_index": -1,
        "start_window_id": str(user_start.get("label") or "user_supplied_start"),
        "start_source_index": None,
        "start_manifest_split": "user_supplied",
        "start_selection_method": "user_supplied_joint39_state",
        "case_role": "operational_selected_start",
        "is_operational": True,
        "start_distance_z": float(support["nearest_train_start_distance_z"]),
        "nearest_train_start_window_index": int(
            support["nearest_train_start_window_index"]
        ),
        "nearest_train_start_distance_z": float(
            support["nearest_train_start_distance_z"]
        ),
        "mean_train_start_distance_z": float(support["mean_train_start_distance_z"]),
        "max_abs_user_start_z": float(support["max_abs_user_start_z"]),
    }


def _start_distance(
    *,
    query_window_index: int,
    start_window_index: int,
    start_state: np.ndarray,
    train_indices: np.ndarray,
) -> float:
    start_z = _safe_start_z(start_state, train_indices)
    return float(
        np.linalg.norm(
            start_z[int(start_window_index)] - start_z[int(query_window_index)]
        )
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


def _recent_prefix_terminal_rows(
    *,
    history_raw: np.ndarray,
    window_index: int,
    spec_names: list[str],
) -> list[dict[str, Any]]:
    history = np.asarray(history_raw, dtype=np.float32)
    idx = int(window_index)
    if history.ndim != 3:
        raise ValueError("history_raw must have shape [N,T,C]")
    if idx < 0 or idx >= history.shape[0]:
        raise IndexError(f"window_index {idx} outside history_raw")
    terminal_delta = history[idx, -1, :] - history[idx, 0, :]
    rows = [
        {
            "Market": "IV_SURFACE",
            "Mean Terminal Delta": float(np.nanmean(terminal_delta[:25])),
        }
    ]
    index = {name: col for col, name in enumerate(spec_names)}
    for market, spec_name in KEY_FACTOR_NAMES.items():
        if spec_name not in index:
            continue
        rows.append(
            {
                "Market": market,
                "Mean Terminal Delta": float(terminal_delta[index[spec_name]]),
            }
        )
    return rows


def _alignment_score_for_candidate(
    *,
    grounding: dict[str, Any],
    history_raw: np.ndarray,
    window_index: int,
    spec_names: list[str],
) -> tuple[float, dict[str, Any]]:
    alignment = market_implication_alignment(
        grounding=grounding,
        scenario_rows=_recent_prefix_terminal_rows(
            history_raw=history_raw,
            window_index=int(window_index),
            spec_names=spec_names,
        ),
    )
    checked = int(alignment.get("checked_count", 0) or 0)
    if checked <= 0:
        return 0.0, alignment
    matches = float(alignment.get("match_count", 0) or 0)
    mismatches = float(alignment.get("mismatch_count", 0) or 0)
    return float((matches - mismatches) / checked), alignment


def _implication_aligned_start(
    *,
    query_window_index: int,
    start_state: np.ndarray,
    train_indices: np.ndarray,
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    grounding: dict[str, Any],
    history_raw: np.ndarray,
    spec_names: list[str],
    start_distance_threshold_z: float,
    start_distance_penalty: float,
    implication_alignment_weight: float,
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
    alignment_scores: list[float] = []
    alignments: list[dict[str, Any]] = []
    for candidate in train:
        score, alignment = _alignment_score_for_candidate(
            grounding=grounding,
            history_raw=history_raw,
            window_index=int(candidate),
            spec_names=spec_names,
        )
        alignment_scores.append(score)
        alignments.append(alignment)
    alignment_array = np.asarray(alignment_scores, dtype=np.float32)
    threshold = float(start_distance_threshold_z)
    inside = distances <= threshold
    score = (
        cosines
        + float(implication_alignment_weight) * alignment_array
        - float(start_distance_penalty) * np.maximum(distances - threshold, 0.0)
    )
    if bool(np.any(inside)):
        candidate_positions = np.flatnonzero(inside)
        best_pos = int(candidate_positions[int(np.argmax(score[inside]))])
        method = "max_memory_plus_recent_implication_inside_start_threshold"
    else:
        best_pos = int(np.argmax(score))
        method = (
            "penalized_memory_plus_recent_implication_no_candidate_inside_threshold"
        )
    chosen_cosine = float(cosines[best_pos])
    chosen_distance = float(distances[best_pos])
    chosen_alignment = alignments[best_pos]
    return {
        "start_window_index": int(train[best_pos]),
        "memory_support_cosine": chosen_cosine,
        "start_selection_score": float(score[best_pos]),
        "start_selection_method": method,
        "start_distance_threshold_z": threshold,
        "start_distance_penalty": float(start_distance_penalty),
        "implication_alignment_weight": float(implication_alignment_weight),
        "recent_prefix_alignment_score": float(alignment_array[best_pos]),
        "recent_prefix_alignment_status": str(chosen_alignment.get("status", "")),
        "recent_prefix_alignment_checked": int(
            chosen_alignment.get("checked_count", 0) or 0
        ),
        "recent_prefix_alignment_mismatches": int(
            chosen_alignment.get("mismatch_count", 0) or 0
        ),
        "memory_support_rank": int(1 + np.sum(cosines > chosen_cosine + 1e-8)),
        "start_distance_rank": int(1 + np.sum(distances < chosen_distance - 1e-8)),
        "implication_alignment_rank": int(
            1 + np.sum(alignment_array > alignment_array[best_pos] + 1e-8)
        ),
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
    grounding: dict[str, Any] | None = None,
    history_raw: np.ndarray | None = None,
    spec_names: list[str] | None = None,
    start_distance_threshold_z: float = float(
        DEFAULT_GATE_THRESHOLDS["start_distance_warn"]
    ),
    start_distance_penalty: float = 0.02,
    implication_alignment_weight: float = 0.25,
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
    elif mode == "implication_aligned_start":
        if query_memory is None or memory_targets is None:
            raise ValueError(
                "query_memory and memory_targets are required for implication_aligned_start"
            )
        if grounding is None or history_raw is None or spec_names is None:
            raise ValueError(
                "grounding, history_raw, and spec_names are required for "
                "implication_aligned_start"
            )
        proposal = _implication_aligned_start(
            query_window_index=query_idx,
            start_state=starts,
            train_indices=train,
            query_memory=query_memory,
            memory_targets=memory_targets,
            grounding=grounding,
            history_raw=history_raw,
            spec_names=spec_names,
            start_distance_threshold_z=float(start_distance_threshold_z),
            start_distance_penalty=float(start_distance_penalty),
            implication_alignment_weight=float(implication_alignment_weight),
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
    if (
        support_cosine is None
        and query_memory is not None
        and memory_targets is not None
    ):
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
    grounding: dict[str, Any] | None = None,
    history_raw: np.ndarray | None = None,
    spec_names: list[str] | None = None,
    start_distance_threshold_z: float = float(
        DEFAULT_GATE_THRESHOLDS["start_distance_warn"]
    ),
    start_distance_penalty: float = 0.02,
    implication_alignment_weight: float = 0.25,
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
        implication_alignment_weight=float(implication_alignment_weight),
    )
    selected = resolve_start_window_index(
        query_window_index=query_idx,
        start_state=start_state,
        train_indices=train_indices,
        start_mode=start_mode,
        explicit_start_window_index=explicit_start_window_index,
        query_memory=query_memory,
        memory_targets=memory_targets,
        grounding=grounding,
        history_raw=history_raw,
        spec_names=spec_names,
        start_distance_threshold_z=float(start_distance_threshold_z),
        start_distance_penalty=float(start_distance_penalty),
        implication_alignment_weight=float(implication_alignment_weight),
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
            "case_role": "operational_selected_start",
            "is_operational": True,
        }
    )
    if not include_original_baseline or selected["start_window_index"] == query_idx:
        row = selected if not include_original_baseline else original
        row["case_role"] = "operational_selected_start"
        row["is_operational"] = True
        return [row]
    original["case_role"] = "diagnostic_original_start"
    original["is_operational"] = False
    return [original, selected]


def query_start_state_for_variant(
    *,
    row: dict[str, Any],
    history_level: np.ndarray,
    user_start: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return the fixed initial level that conditions one variant's mixture."""

    start_idx = int(row["start_window_index"])
    if start_idx >= 0:
        return np.asarray(history_level[start_idx, -1, :], dtype=np.float32)
    if user_start is None:
        raise ValueError("user start row requires loaded user_start")
    return np.asarray(user_start["encoded_state"], dtype=np.float32)


def annotate_variant_with_memory_prior(
    row: dict[str, Any],
    memory_prior: dict[str, Any],
) -> dict[str, Any]:
    """Attach compact start-conditioned mixture diagnostics to a variant row."""

    annotated = dict(row)
    candidate_details = memory_prior.get("candidate_details", [])
    if not isinstance(candidate_details, list):
        candidate_details = []
    weighted_start_distance = 0.0
    for candidate, weight in zip(
        candidate_details,
        memory_prior.get("weights", []),
        strict=False,
    ):
        if not isinstance(candidate, dict):
            continue
        weighted_start_distance += float(weight) * float(
            candidate.get("start_distance_z", 0.0) or 0.0
        )
    annotated.update(
        {
            "memory_prior_mode": str(memory_prior.get("mode", "")),
            "memory_prior_query_start_source": str(
                memory_prior.get("query_start_source", "")
            ),
            "memory_prior_analogue_count": int(
                memory_prior.get("analogue_count", 0) or 0
            ),
            "memory_prior_weighted_start_distance_z": float(weighted_start_distance),
            "memory_prior_top_window_index": (
                int(candidate_details[0]["window_index"])
                if candidate_details
                and isinstance(candidate_details[0], dict)
                and "window_index" in candidate_details[0]
                else None
            ),
        }
    )
    direction_check = memory_prior.get("direction_check", {})
    if isinstance(direction_check, dict):
        annotated.update(
            {
                "memory_prior_direction_status": str(direction_check.get("status", "")),
                "memory_prior_direction_reason": str(direction_check.get("reason", "")),
                "memory_prior_support_weighted_match_rate": (
                    None
                    if direction_check.get("support_weighted_match_rate") is None
                    else float(direction_check.get("support_weighted_match_rate", 0.0))
                ),
                "memory_prior_final_mixture_mismatch_count": int(
                    direction_check.get("final_mixture_mismatch_count", 0) or 0
                ),
            }
        )
    support_policy = memory_prior.get("support_diversity_policy", {})
    if isinstance(support_policy, dict):
        annotated["memory_prior_support_diversity_policy"] = dict(support_policy)
    quality_guard_policy = memory_prior.get("portfolio_quality_guard_policy", {})
    if isinstance(quality_guard_policy, dict):
        annotated["portfolio_quality_guard_policy"] = dict(quality_guard_policy)
    return annotated


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


def scale_delta_samples_around_mean(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Scale generated delta-path samples around each variant ensemble mean."""

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("samples must have shape [K,S,T,C]")
    mean = np.nanmean(arr, axis=1, keepdims=True)
    return (mean + float(alpha) * (arr - mean)).astype(np.float32)


def sample_prefix_generator_deltas_from_start_raw(
    model: Any,
    *,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    start_raw: np.ndarray,
    specs: list[Any],
    samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: torch.device,
) -> np.ndarray:
    """Sample future deltas from decoded prefixes pinned to supplied raw starts."""

    sampled_increment = model.sample_batched(
        torch.from_numpy(np.asarray(history_level, dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(history_norm, dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(center, dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(scale, dtype=np.float32)).to(device),
        drift_feature=torch.from_numpy(np.asarray(drift_feature, dtype=np.float32)).to(
            device
        ),
        n_samples=int(samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        temperature=float(temperature),
    )
    current = np.asarray(start_raw, dtype=np.float32)
    generated_states = _reconstruct_states(
        current,
        sampled_increment.detach().cpu().numpy(),
        specs,
    )
    return (generated_states - current[:, None, None, :]).astype(np.float32)


def _allocate_weighted_sample_counts(
    weights: np.ndarray,
    total_samples: int,
) -> np.ndarray:
    """Deterministically allocate total samples across weighted components."""

    total = int(total_samples)
    if total < 1:
        raise ValueError("total_samples must be positive")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0.0:
        w = np.ones_like(w, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w = w / max(float(np.sum(w)), 1e-12)
    expected = w * float(total)
    counts = np.floor(expected).astype(np.int64)
    remainder = total - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(expected - counts))
        for pos in order[:remainder]:
            counts[int(pos)] += 1
    return counts


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    temp = max(float(temperature), 1.0e-6)
    scaled = arr / temp
    scaled = scaled - float(np.max(scaled))
    exp = np.exp(scaled)
    total = float(np.sum(exp))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(arr.size, dtype=np.float64) / float(arr.size)
    return exp / total


def _response_implication_rows(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    """Return current/recent market implications across live/cached schemas."""

    payload = grounding if isinstance(grounding, dict) else {}
    nested = payload.get("condition_only_grounding")
    if isinstance(nested, dict):
        payload = nested
    rows: list[dict[str, Any]] = []
    for key in ("current_market_state_implications", "market_implications"):
        raw = payload.get(key, [])
        if isinstance(raw, list):
            rows.extend(row for row in raw if isinstance(row, dict))
    return rows


def _normal_response_market(value: Any) -> str:
    text = str(value or "").strip().upper().replace("_", " ")
    return RESPONSE_MARKET_ALIASES.get(text, text.replace(" ", "_"))


def response_channels_from_grounding(
    *,
    grounding: dict[str, Any],
    spec_names: list[str],
) -> list[dict[str, Any]]:
    """Map grounded current/recent claims to raw-state response channels."""

    spec_index = {str(name): idx for idx, name in enumerate(spec_names)}
    channels: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _response_implication_rows(grounding):
        market = _normal_response_market(row.get("market"))
        spec_name = KEY_FACTOR_NAMES.get(market)
        if spec_name is None or spec_name not in spec_index:
            continue
        direction = str(row.get("direction", "")).strip().lower()
        confidence = str(row.get("confidence", "medium")).strip().lower()
        magnitude = str(row.get("magnitude", "medium")).strip().lower()
        channels.append(
            {
                "market": market,
                "factor": spec_name,
                "index": int(spec_index[spec_name]),
                "sign": float(RESPONSE_DIRECTION_SIGN.get(direction, 0.0)),
                "direction": direction,
                "weight": float(
                    RESPONSE_CONFIDENCE_WEIGHT.get(confidence, 0.6)
                    * RESPONSE_MAGNITUDE_WEIGHT.get(magnitude, 1.0)
                ),
                "confidence": confidence,
                "magnitude": magnitude,
                "evidence": row.get("evidence"),
            }
        )
        seen.add(spec_name)
    for market in ("SPX", "VIX", "BBB_OAS", "US10Y", "DXY", "CRUDE_OIL", "GOLD"):
        spec_name = KEY_FACTOR_NAMES.get(market)
        if spec_name and spec_name in spec_index and spec_name not in seen:
            channels.append(
                {
                    "market": market,
                    "factor": spec_name,
                    "index": int(spec_index[spec_name]),
                    "sign": 0.0,
                    "direction": "activation",
                    "weight": 0.25,
                    "confidence": "fallback",
                    "magnitude": "fallback",
                    "evidence": "fallback narrative-relevant risk channel",
                }
            )
    return channels


def _component_response_score(
    *,
    component_states: np.ndarray,
    start_raw: np.ndarray,
    channels: list[dict[str, Any]],
    objective: ResponsePreviewObjective = "narrative_channels",
    spec_names: list[str] | None = None,
) -> float:
    """Score preview paths in the channels implied by the narrative."""

    states = np.asarray(component_states, dtype=np.float64)
    start = np.asarray(start_raw, dtype=np.float64).reshape(-1)
    if states.ndim != 3:
        raise ValueError("component_states must have shape [S,T,C]")
    pieces: list[float] = []
    weights: list[float] = []
    for channel in channels:
        idx = int(channel["index"])
        scale = max(abs(float(start[idx])), 1.0)
        terminal = (states[:, -1, idx] - float(start[idx])) / scale
        path = (states[:, :, idx] - float(start[idx])) / scale
        sign = float(channel.get("sign", 0.0))
        signed = sign * float(np.median(terminal)) if sign != 0.0 else 0.0
        activation = abs(float(np.median(terminal)))
        width = float(np.percentile(path, 90) - np.percentile(path, 10))
        pieces.append(0.55 * signed + 0.25 * activation + 0.20 * width)
        weights.append(float(channel.get("weight", 1.0)))
    if not pieces:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(pieces, dtype=np.float64)
    if float(w.sum()) <= 0.0:
        channel_score = float(np.mean(v))
    else:
        channel_score = float(np.sum(v * w) / np.sum(w))
    if objective == "narrative_channels":
        return channel_score
    if objective == "channel_portfolio":
        channel_portfolio_score = _component_channel_portfolio_response_score(
            component_states=states,
            start_raw=start,
            channels=channels,
        )
        return float(0.55 * channel_score + 0.45 * channel_portfolio_score)
    if objective != "factor_portfolio":
        raise ValueError(f"unknown response preview objective: {objective}")
    portfolio_score = _component_portfolio_response_score(
        component_states=states,
        start_raw=start,
        spec_names=spec_names or [],
    )
    return float(0.65 * channel_score + 0.35 * portfolio_score)


def _component_channel_portfolio_response_score(
    *,
    component_states: np.ndarray,
    start_raw: np.ndarray,
    channels: list[dict[str, Any]],
) -> float:
    """Score joint movement in the signed factors named by the narrative."""

    states = np.asarray(component_states, dtype=np.float64)
    start = np.asarray(start_raw, dtype=np.float64).reshape(-1)
    terms: list[np.ndarray] = []
    for channel in channels:
        sign = float(channel.get("sign", 0.0))
        if sign == 0.0:
            continue
        idx = int(channel["index"])
        scale = max(abs(float(start[idx])), 1.0)
        weight = float(channel.get("weight", 1.0))
        terms.append(sign * weight * (states[:, :, idx] - float(start[idx])) / scale)
    if not terms:
        return 0.0
    channel_pnl = np.sum(np.stack(terms, axis=-1), axis=-1)
    terminal = channel_pnl[:, -1]
    signed = float(np.median(terminal))
    activation = abs(signed)
    width = float(np.percentile(channel_pnl, 90) - np.percentile(channel_pnl, 10))
    loss_tail = max(0.0, -float(np.percentile(terminal, 5)))
    return float(0.45 * signed + 0.20 * activation + 0.25 * width - 0.10 * loss_tail)


def _component_portfolio_response_score(
    *,
    component_states: np.ndarray,
    start_raw: np.ndarray,
    spec_names: list[str],
) -> float:
    """Direction-agnostic portfolio-tail response score for preview rollouts."""

    states = np.asarray(component_states, dtype=np.float64)
    start = np.asarray(start_raw, dtype=np.float64).reshape(-1)
    spec_index = {str(name): int(pos) for pos, name in enumerate(spec_names)}
    terms: list[np.ndarray] = []
    for exposure in RESPONSE_PORTFOLIO_EXPOSURES:
        spec_name = KEY_FACTOR_NAMES.get(str(exposure["market"]))
        if spec_name not in spec_index:
            continue
        idx = int(spec_index[spec_name])
        scale = max(abs(float(start[idx])), 1.0)
        terms.append(
            (states[:, :, idx] - float(start[idx]))
            / scale
            * float(exposure["sensitivity"])
        )
    if not terms:
        return 0.0
    pnl = np.sum(np.stack(terms, axis=-1), axis=-1)
    terminal = pnl[:, -1]
    activation = abs(float(np.median(terminal)))
    width = float(np.percentile(pnl, 90) - np.percentile(pnl, 10))
    loss_tail = max(0.0, -float(np.percentile(terminal, 5)))
    return float(0.25 * activation + 0.45 * width + 0.30 * loss_tail)


def _response_preview_weights(
    *,
    base_weights: np.ndarray,
    response_scores: np.ndarray,
    alpha: float,
    temperature: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.maximum(np.asarray(base_weights, dtype=np.float64).reshape(-1), 1.0e-8)
    base = base / max(float(base.sum()), 1.0e-8)
    scores = np.asarray(response_scores, dtype=np.float64).reshape(-1)
    if base.size != scores.size:
        raise ValueError("base_weights and response_scores must have the same size")
    if scores.size > 1 and float(np.std(scores)) > 1.0e-9:
        scaled_scores = (scores - float(np.mean(scores))) / float(np.std(scores))
    else:
        scaled_scores = np.zeros_like(scores)
    logits = np.log(base) + float(alpha) * scaled_scores
    response = _softmax(logits, temperature=float(temperature))
    mix = min(max(float(blend), 0.0), 1.0)
    bounded = (1.0 - mix) * base + mix * response
    bounded = bounded / max(float(bounded.sum()), 1.0e-8)
    return bounded, scaled_scores


def _component_memory_rows_for_prior(
    *,
    prior: dict[str, Any],
    fallback_memory: np.ndarray,
    memory_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return component memories, support indices, and mixture weights."""

    indices = np.asarray(prior.get("window_indices", []), dtype=np.int64)
    weights = np.asarray(prior.get("weights", []), dtype=np.float32)
    targets = np.asarray(memory_targets, dtype=np.float32)
    if indices.size == 0 or weights.size != indices.size:
        return (
            np.asarray(fallback_memory, dtype=np.float32).reshape(1, -1),
            np.asarray([-1], dtype=np.int64),
            np.asarray([1.0], dtype=np.float32),
        )
    if np.any(indices < 0) or np.any(indices >= targets.shape[0]):
        raise IndexError("memory-prior window_indices outside memory_targets")
    weights = np.maximum(weights.astype(np.float32), 0.0)
    weights = weights / max(float(np.sum(weights)), 1e-8)
    return targets[indices].astype(np.float32), indices, weights.astype(np.float32)


def _decode_prefix_for_component_memory(
    *,
    prefix_prior_mode: str,
    component_memory: np.ndarray,
    component_index: int,
    requested_start: np.ndarray,
    decoder_result: dict[str, Any],
    input_stats: dict[str, np.ndarray],
    features: np.ndarray,
    layout: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Decode one support component into a start-pinned recent-prefix object."""

    start = np.asarray(requested_start, dtype=np.float32).reshape(1, -1)
    if prefix_prior_mode == "feature_mixture" and int(component_index) >= 0:
        component_features = np.asarray(
            features[int(component_index)], dtype=np.float32
        )[None, :]
        return reconstruct_prefix_from_features(
            component_features,
            start_state=start,
            layout=layout,
        )
    inputs = apply_memory_start_stats(
        np.asarray(component_memory, dtype=np.float32).reshape(1, -1),
        start,
        input_stats,
    )
    return _decode_features(
        decoder_result["model"],
        decoder_result["target_mean"],
        decoder_result["target_std"],
        inputs,
        start_state=start,
        layout=layout,
        device=device,
    )


def response_preview_reweight_memory_priors(
    model: Any,
    *,
    memory_priors: list[dict[str, Any]],
    fallback_memories: np.ndarray,
    memory_targets: np.ndarray,
    prefix_prior_mode: str,
    decoder_result: dict[str, Any],
    input_stats: dict[str, np.ndarray],
    features: np.ndarray,
    layout: dict[str, Any],
    requested_start: np.ndarray,
    requested_raw: np.ndarray,
    specs: list[Any],
    grounding: dict[str, Any],
    spec_names: list[str],
    preview_samples_per_component: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    response_alpha: float,
    response_temperature: float,
    response_blend: float,
    response_objective: ResponsePreviewObjective,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run small per-component previews and update support weights."""

    preview_count = max(int(preview_samples_per_component), 1)
    channels = response_channels_from_grounding(
        grounding=grounding if isinstance(grounding, dict) else {},
        spec_names=spec_names,
    )
    updated_priors: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    for variant_idx, prior in enumerate(memory_priors):
        component_memories, component_indices, component_weights = (
            _component_memory_rows_for_prior(
                prior=prior,
                fallback_memory=np.asarray(fallback_memories[variant_idx]),
                memory_targets=memory_targets,
            )
        )
        scores: list[float] = []
        component_details: list[dict[str, Any]] = []
        for component_pos, component_memory in enumerate(component_memories):
            decoded = _decode_prefix_for_component_memory(
                prefix_prior_mode=str(prefix_prior_mode),
                component_memory=component_memory,
                component_index=int(component_indices[component_pos]),
                requested_start=np.asarray(requested_start[variant_idx]),
                decoder_result=decoder_result,
                input_stats=input_stats,
                features=features,
                layout=layout,
                device=device,
            )
            preview_delta = sample_prefix_generator_deltas_from_start_raw(
                model,
                history_level=decoded["history_level"],
                history_norm=decoded["history_norm"],
                center=decoded["center"],
                scale=decoded["scale"],
                drift_feature=decoded["drift_feature"],
                start_raw=np.asarray(requested_raw[variant_idx], dtype=np.float32)[
                    None, :
                ],
                specs=specs,
                samples=preview_count,
                n_steps=int(n_steps),
                chunk_size=min(int(chunk_size), preview_count),
                temperature=float(temperature),
                device=device,
            )[0]
            preview_states = (
                np.asarray(requested_raw[variant_idx], dtype=np.float32)[None, None, :]
                + preview_delta
            ).astype(np.float32)
            score = _component_response_score(
                component_states=preview_states,
                start_raw=np.asarray(requested_raw[variant_idx], dtype=np.float32),
                channels=channels,
                objective=response_objective,
                spec_names=spec_names,
            )
            scores.append(float(score))
            component_details.append(
                {
                    "variant_index": int(variant_idx),
                    "component_index": int(component_pos),
                    "support_window_index": int(component_indices[component_pos]),
                    "base_weight": float(component_weights[component_pos]),
                    "response_score": float(score),
                    "preview_sample_count": int(preview_count),
                }
            )
        response_weights, scaled_scores = _response_preview_weights(
            base_weights=component_weights,
            response_scores=np.asarray(scores, dtype=np.float64),
            alpha=float(response_alpha),
            temperature=float(response_temperature),
            blend=float(response_blend),
        )
        updated = dict(prior)
        updated["weights"] = [float(value) for value in response_weights]
        updated["memory"] = (
            np.sum(component_memories * response_weights[:, None], axis=0)
            .astype(np.float32)
            .tolist()
        )
        updated_policy = dict(updated.get("response_preview_support_weighting", {}))
        updated_policy.update(
            {
                "policy": "response_preview_component_mixture",
                "preview_samples_per_component": int(preview_count),
                "response_alpha": float(response_alpha),
                "response_temperature": float(response_temperature),
                "response_blend": float(response_blend),
                "response_objective": str(response_objective),
                "channel_count": int(len(channels)),
            }
        )
        updated["response_preview_support_weighting"] = updated_policy
        old_details = [
            dict(row)
            for row in updated.get("candidate_details", [])
            if isinstance(row, dict)
        ]
        by_index = {int(row.get("window_index", -1)): row for row in old_details}
        new_details: list[dict[str, Any]] = []
        for detail, weight, scaled in zip(
            component_details,
            response_weights,
            scaled_scores,
            strict=True,
        ):
            row = by_index.get(int(detail["support_window_index"]), {})
            row.update(
                {
                    **detail,
                    "window_index": int(detail["support_window_index"]),
                    "base_weight": float(detail["base_weight"]),
                    "weight": float(weight),
                    "response_weight": float(weight),
                    "response_score_z": float(scaled),
                }
            )
            new_details.append(row)
        updated["candidate_details"] = new_details
        updated_priors.append(updated)
        preview_rows.append(
            {
                "variant_index": int(variant_idx),
                "support_count": int(len(new_details)),
                "channels": channels,
                "components": new_details,
                "base_weights": [float(value) for value in component_weights],
                "response_weights": [float(value) for value in response_weights],
                "weight_l1_delta": float(
                    np.sum(np.abs(response_weights - component_weights))
                ),
                "effective_support_count": float(
                    1.0 / max(float(np.sum(response_weights**2)), 1.0e-12)
                ),
                "response_objective": str(response_objective),
            }
        )
    return updated_priors, preview_rows


def sample_component_prefix_mixture_from_start_raw(
    model: Any,
    *,
    memory_priors: list[dict[str, Any]],
    fallback_memories: np.ndarray,
    memory_targets: np.ndarray,
    prefix_prior_mode: str,
    decoder_result: dict[str, Any],
    input_stats: dict[str, np.ndarray],
    features: np.ndarray,
    layout: dict[str, Any],
    requested_start: np.ndarray,
    requested_raw: np.ndarray,
    specs: list[Any],
    samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Roll out each support component separately and pool weighted samples.

    The previous default averaged support memories before decoding one prefix.
    That is useful as a smooth baseline, but it can erase regime-mixture shape.
    This path preserves the support mixture through the frozen rollout.
    """

    variant_samples: list[np.ndarray] = []
    variant_states: list[np.ndarray] = []
    component_rows: list[dict[str, Any]] = []
    for variant_idx, prior in enumerate(memory_priors):
        component_memories, component_indices, component_weights = (
            _component_memory_rows_for_prior(
                prior=prior,
                fallback_memory=np.asarray(fallback_memories[variant_idx]),
                memory_targets=memory_targets,
            )
        )
        counts = _allocate_weighted_sample_counts(component_weights, int(samples))
        pieces: list[np.ndarray] = []
        for component_pos, sample_count in enumerate(counts):
            if int(sample_count) <= 0:
                component_rows.append(
                    {
                        "variant_index": int(variant_idx),
                        "component_index": int(component_pos),
                        "support_window_index": int(component_indices[component_pos]),
                        "weight": float(component_weights[component_pos]),
                        "sample_count": 0,
                        "skipped": True,
                    }
                )
                continue
            decoded = _decode_prefix_for_component_memory(
                prefix_prior_mode=str(prefix_prior_mode),
                component_memory=component_memories[component_pos],
                component_index=int(component_indices[component_pos]),
                requested_start=np.asarray(requested_start[variant_idx]),
                decoder_result=decoder_result,
                input_stats=input_stats,
                features=features,
                layout=layout,
                device=device,
            )
            component_delta = sample_prefix_generator_deltas_from_start_raw(
                model,
                history_level=decoded["history_level"],
                history_norm=decoded["history_norm"],
                center=decoded["center"],
                scale=decoded["scale"],
                drift_feature=decoded["drift_feature"],
                start_raw=np.asarray(requested_raw[variant_idx], dtype=np.float32)[
                    None, :
                ],
                specs=specs,
                samples=int(sample_count),
                n_steps=int(n_steps),
                chunk_size=min(int(chunk_size), int(sample_count)),
                temperature=float(temperature),
                device=device,
            )[0]
            pieces.append(component_delta.astype(np.float32))
            component_rows.append(
                {
                    "variant_index": int(variant_idx),
                    "component_index": int(component_pos),
                    "support_window_index": int(component_indices[component_pos]),
                    "weight": float(component_weights[component_pos]),
                    "sample_count": int(sample_count),
                    "skipped": False,
                }
            )
        if not pieces:
            raise RuntimeError("component mixture produced no rollout samples")
        pooled_delta = np.concatenate(pieces, axis=0).astype(np.float32)
        if pooled_delta.shape[0] != int(samples):
            raise RuntimeError(
                f"expected {samples} pooled samples, got {pooled_delta.shape[0]}"
            )
        variant_samples.append(pooled_delta)
        current = np.asarray(requested_raw[variant_idx], dtype=np.float32)
        variant_states.append(
            (current[None, None, :] + pooled_delta).astype(np.float32)
        )
    return (
        np.stack(variant_samples, axis=0).astype(np.float32),
        np.stack(variant_states, axis=0).astype(np.float32),
        component_rows,
    )


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
        if start_idx < 0 or start_idx >= future_delta.shape[0]:
            rows.append(
                {
                    "case_index": int(row_no),
                    "query_window_index": int(row["query_window_index"]),
                    "start_window_index": start_idx,
                    "variant": str(row["variant"]),
                    "target_available": False,
                    "target_note": "No realized future is available for a user-supplied start state.",
                    "methods": {},
                }
            )
            continue
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
        window_id = str(
            row.get("start_window_id") or info.get("window_id", f"window_{start_idx}")
        )
        is_operational = bool(row.get("is_operational", True))
        label_prefix = (
            "User supplied start"
            if start_idx < 0
            else "Selected start" if is_operational else "Diagnostic baseline"
        )
        labels.append(
            {
                "window_id": window_id,
                "variant": str(row.get("variant", "")),
                "case_role": str(row.get("case_role", "")),
                "is_operational": is_operational,
                "analogue_label": f"{label_prefix}: {window_id}",
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
                "start_window_id": str(
                    start_info.get("window_id", row.get("start_window_id", ""))
                ),
                "start_source_index": start_info.get(
                    "source_index", row.get("start_source_index")
                ),
                "start_manifest_split": start_info.get(
                    "manifest_split", row.get("start_manifest_split")
                ),
                "start_calendar": start_info.get("calendar", {}),
            }
        )
    return enriched


def enrich_memory_prior_candidate_metadata(
    memory_prior: dict[str, Any],
    window_metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Attach human-readable window metadata to memory-prior candidates."""

    enriched_prior = dict(memory_prior)
    weights_by_index = {
        int(idx): float(weight)
        for idx, weight in zip(
            memory_prior.get("window_indices", []),
            memory_prior.get("weights", []),
            strict=False,
        )
    }
    candidate_details: list[dict[str, Any]] = []
    for rank, item in enumerate(memory_prior.get("candidate_details", []), start=1):
        if not isinstance(item, dict):
            continue
        bridge_index = int(item.get("window_index", -1))
        info = window_metadata.get(bridge_index, {})
        calendar = info.get("calendar", {}) if isinstance(info, dict) else {}
        row = {
            **item,
            "rank": int(rank),
            "bridge_local_index": bridge_index,
            "weight": weights_by_index.get(bridge_index),
            "window_id": str(info.get("window_id", "")) if info else "",
            "source_index": info.get("source_index") if info else None,
            "manifest_split": str(info.get("manifest_split", "")) if info else "",
            "history_start_date": (
                str(calendar.get("calendar_start_date", ""))
                if isinstance(calendar, dict)
                else ""
            ),
            "history_end_date": (
                str(calendar.get("calendar_end_date", ""))
                if isinstance(calendar, dict)
                else ""
            ),
            "forecast_start_date": (
                str(calendar.get("forecast_start_date", ""))
                if isinstance(calendar, dict)
                else ""
            ),
            "forecast_end_date": (
                str(calendar.get("forecast_end_date", ""))
                if isinstance(calendar, dict)
                else ""
            ),
        }
        candidate_details.append(row)
    if candidate_details:
        enriched_prior["candidate_details"] = candidate_details
    return enriched_prior


def _render_markdown(report: dict[str, Any]) -> str:
    query = report.get("cached_query", {})
    gate = report.get("validation_gate", {})
    generation = report.get("generation", {})
    decoder = report.get("decoder", {})
    memory_prior = query.get("memory_prior", {})
    if not isinstance(memory_prior, dict):
        memory_prior = {}
    support_policy = memory_prior.get("support_diversity_policy", {})
    if not isinstance(support_policy, dict):
        support_policy = {}
    quality_guard_policy = memory_prior.get("portfolio_quality_guard_policy", {})
    if not isinstance(quality_guard_policy, dict):
        quality_guard_policy = {}
    response_preview_policy = query.get("response_preview_support_weighting", {})
    if not isinstance(response_preview_policy, dict):
        response_preview_policy = {}
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
        "- Historical support is selected after the start is fixed; support "
        "selection is part of conditioning, not a display-only explanation.",
        "- A learned memory+start decoder reconstructs a recent prefix object.",
        "- The frozen joint39 SNI generator performs the 30-day rollout.",
    ]
    if support_policy:
        lines.extend(
            [
                "- Support policy: " f"`{support_policy.get('policy', 'unknown')}`.",
                "- Support count: "
                f"`{support_policy.get('selected_count')}/"
                f"{support_policy.get('requested_top_k')}` selected; "
                f"direction gate `{support_policy.get('direction_gate')}`, "
                "temporal non-overlap "
                f"`{support_policy.get('temporal_non_overlap_enforced')}` "
                f"with minimum index gap "
                f"`{support_policy.get('temporal_min_index_gap')}`.",
                "- Overlap padding: "
                f"`{support_policy.get('padding_with_temporal_overlaps')}`.",
            ]
        )
    if quality_guard_policy:
        lines.extend(
            [
                "- Portfolio quality guard: "
                f"`{quality_guard_policy.get('policy_kind', 'unknown')}`; "
                "fallback "
                f"`{quality_guard_policy.get('fallback_to_equal_support')}`; "
                "reason "
                f"`{quality_guard_policy.get('fallback_reason', 'n/a')}`; "
                "candidates "
                f"`{quality_guard_policy.get('candidate_count', 'n/a')}`; "
                "minimum candidates "
                f"`{quality_guard_policy.get('min_candidate_mixtures', 'n/a')}`; "
                "support max "
                f"`{quality_guard_policy.get('support_weight_max', 'n/a')}`.",
            ]
        )
    if bool(response_preview_policy.get("applied")):
        lines.extend(
            [
                "- Response preview weighting: applied with "
                f"`{response_preview_policy.get('preview_samples_per_component')}` "
                "preview samples/component; alpha "
                f"`{response_preview_policy.get('response_alpha')}`; temperature "
                f"`{response_preview_policy.get('response_temperature')}`; blend "
                f"`{response_preview_policy.get('response_blend')}`; objective "
                f"`{response_preview_policy.get('response_objective', 'narrative_channels')}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Start Selection",
            "",
            "| Variant | Query Window | Start Window | Start Distance z | Memory Support Cosine |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
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
            f"- Research overall: `{gate.get('overall_status')}`",
            f"- Selected-start: `{gate.get('selected_start_status', gate.get('operational_status'))}`",
            f"- Diagnostic baseline: `{gate.get('diagnostic_baseline_status')}`",
            f"- Stress: `{gate.get('stress_status')}`",
            f"- Endpoint max error: {gate.get('endpoint_max_abs_error')}",
            f"- Warning counts: {gate.get('warning_counts')}",
            f"- Failure counts: {gate.get('fail_counts')}",
        ]
    )
    reliability = report.get("start_reliability_gate")
    if isinstance(reliability, dict):
        lines.extend(
            [
                "",
                "## Start Reliability Gate",
                "",
                f"- Product status: `{reliability.get('product_status')}`",
                f"- Decision: {reliability.get('decision')}",
                f"- Warnings: {reliability.get('warnings')}",
                f"- Failures: {reliability.get('failures')}",
                f"- Source: `{reliability.get('source')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Decoder",
            "",
            f"- Train MSE: {decoder.get('train_mse')}",
            f"- Test MSE: {decoder.get('test_mse')}",
            f"- Loss first/last: {decoder.get('loss_first')} / {decoder.get('loss_last')}",
            "",
            "## Scenario",
            "",
            f"- Rollout temperature: {generation.get('rollout_temperature')}",
            f"- Rollout fan scale: {generation.get('rollout_fan_scale')}",
            f"- Sample count: {generation.get('sample_count')}",
            f"- Generated shape: {generation.get('generated_state_shape')}",
            f"- Finite rate: {generation.get('finite_rate')}",
            f"- Rollout scores: {generation.get('rollout_summary')}",
        ]
    )
    return "\n".join(lines)


def run_prefix_latent_story_smoke(args: argparse.Namespace) -> dict[str, Any]:
    base_seed = int(args.seed)
    raw_decoder_seed = getattr(args, "decoder_seed", None)
    raw_rollout_seed = getattr(args, "rollout_seed", None)
    decoder_seed = base_seed if raw_decoder_seed is None else int(raw_decoder_seed)
    rollout_seed = base_seed if raw_rollout_seed is None else int(raw_rollout_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    bridge_report = _load_json(args.bridge_report)
    pipeline_report = _load_json(args.pipeline_report)
    selected_windows = selected_bridge_window_indices(bridge_report)
    bridge_train_indices, bridge_test_indices = split_indices_from_bridge_report(
        bridge_report
    )
    bridge_arrays = load_bridge_arrays(args.bridge_arrays)
    bridge_memory_targets = np.asarray(
        bridge_arrays["memory_targets"], dtype=np.float32
    )
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
    start_history_level = all_history_level[selected_windows]
    start_history_norm = all_history_norm[selected_windows]
    start_center = all_center[selected_windows]
    start_scale = all_scale[selected_windows]
    start_drift_feature = all_drift_feature[selected_windows]
    start_history_raw = all_history_raw[selected_windows]
    future_raw_all = _future_raw_from_block(
        block,
        int(all_history_raw.shape[0]),
        int(all_history_raw.shape[-1]),
    )
    start_future_raw = future_raw_all[selected_windows]
    start_future_delta = future_delta_paths(all_history_raw, future_raw_all)[
        selected_windows
    ]
    start_window_metadata = window_metadata_by_bridge_local_index(bridge_report)
    if bool(getattr(args, "support_bank_report", None)) or bool(
        getattr(args, "support_bank_arrays", None)
    ):
        if not args.support_bank_report or not args.support_bank_arrays:
            raise ValueError(
                "--support-bank-report and --support-bank-arrays must be provided "
                "together"
            )
        support_bank = _load_support_bank(
            report_path=args.support_bank_report,
            arrays_path=args.support_bank_arrays,
        )
        support_memory_targets = support_bank["memory_targets"]
        support_history_level = support_bank["history_level"]
        support_history_norm = support_bank["history_norm"]
        support_center = support_bank["center"]
        support_scale = support_bank["scale"]
        support_drift_feature = support_bank["drift_feature"]
        support_history_raw = support_bank["history_raw"]
        support_future_delta = support_bank["future_delta"]
        support_train_indices = support_bank["train_indices"]
        support_test_indices = support_bank["test_indices"]
        support_candidate_indices = support_bank["support_indices"]
        support_window_metadata = window_metadata_by_local_index(
            support_bank["metadata"]
        )
        support_bank_summary = {
            "source": "external_support_bank",
            "report": str(args.support_bank_report),
            "arrays": str(args.support_bank_arrays),
            "support_window_count": int(support_memory_targets.shape[0]),
            "support_candidate_count": int(support_candidate_indices.size),
            "train_window_count": int(support_train_indices.size),
            "test_window_count": int(support_test_indices.size),
            "calendar_end_date_range": support_bank["report"].get(
                "calendar_end_date_range", {}
            ),
            "train_calendar_end_date_range": support_bank["report"].get(
                "train_calendar_end_date_range", {}
            ),
        }
    else:
        support_memory_targets = bridge_memory_targets
        support_history_level = start_history_level
        support_history_norm = start_history_norm
        support_center = start_center
        support_scale = start_scale
        support_drift_feature = start_drift_feature
        support_history_raw = start_history_raw
        support_future_delta = start_future_delta
        support_train_indices = bridge_train_indices
        support_test_indices = bridge_test_indices
        support_candidate_indices = bridge_train_indices
        support_window_metadata = start_window_metadata
        support_bank_summary = {
            "source": "bridge_labeled_windows",
            "support_window_count": int(support_memory_targets.shape[0]),
            "support_candidate_count": int(support_candidate_indices.size),
            "train_window_count": int(support_train_indices.size),
            "test_window_count": int(support_test_indices.size),
        }
    features, layout = build_prefix_feature_matrix(
        support_history_level,
        support_history_norm,
        support_center,
        support_scale,
        support_drift_feature,
    )
    user_start: dict[str, Any] | None = None
    if getattr(args, "start_state_json", None):
        user_start = load_user_start_state(args.start_state_json, specs)
    inputs, input_stats = build_memory_start_input_matrix(
        support_memory_targets,
        support_history_level[:, -1, :],
        fit_indices=support_train_indices,
    )
    decoder_result = train_memory_start_prefix_decoder(
        inputs,
        features,
        train_indices=support_train_indices,
        test_indices=support_test_indices,
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=decoder_seed,
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
    condition_report_narrative_text: str | None = None
    embedding_metadata: dict[str, Any] = {}
    condition_source = "cached_bridge_query"
    if getattr(args, "condition_report", None):
        external_condition = _external_condition_from_report(args.condition_report)
        external_query = external_condition["query"]
        query_memory = np.asarray(
            external_condition["query_memory"],
            dtype=np.float32,
        )
        grounding_payload = dict(external_condition["grounding"])
        query_text = (
            str(external_condition["query_text"])
            if external_condition.get("query_text") is not None
            else None
        )
        if external_query.get("narrative_text") is not None:
            condition_report_narrative_text = str(external_query["narrative_text"])
        raw_metadata = external_condition.get("embedding_metadata", {})
        embedding_metadata = (
            dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        )
        embedding_metadata["condition_report"] = str(args.condition_report)
        embedding_metadata["condition_arrays"] = str(external_condition["arrays_path"])
        condition_source = "external_condition_report"
        query_row = {
            **query_row,
            "role": str(external_query.get("role", "external_condition")),
            "kind": str(external_query.get("kind", "external_condition")),
            "window_id": str(
                external_query.get("window_id", query_row.get("window_id", ""))
            ),
            "window_index": int(
                external_query.get("window_index", query_row["window_index"])
            ),
            "embedding_index": int(external_query.get("embedding_index", -1)),
        }
    elif bool(getattr(args, "live_story", False)):
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
            condition_dim=int(support_memory_targets.shape[1]),
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
    spec_names = _spec_names(specs)
    memory_prior_mode = str(getattr(args, "memory_prior_mode", "query_memory"))
    if str(args.start_mode) == "user_start_state":
        if user_start is None:
            raise ValueError("--start-state-json is required for user_start_state")
        variant_rows = []
        if bool(args.include_original_baseline):
            original = resolve_start_window_index(
                query_window_index=int(query_row["window_index"]),
                start_state=start_history_level[:, -1, :],
                train_indices=bridge_train_indices,
                start_mode="original",
                query_memory=query_memory,
                memory_targets=bridge_memory_targets,
            )
            original["case_role"] = "diagnostic_original_start"
            original["is_operational"] = False
            variant_rows.append(original)
        variant_rows.append(
            build_user_start_variant_row(
                query_row=query_row,
                user_start=user_start,
                history_level=support_history_level,
                train_indices=support_train_indices,
            )
        )
    else:
        variant_rows = build_live_story_variant_rows(
            query_row=query_row,
            start_state=start_history_level[:, -1, :],
            train_indices=bridge_train_indices,
            start_mode=str(args.start_mode),
            explicit_start_window_index=args.explicit_start_window_index,
            include_original_baseline=bool(args.include_original_baseline),
            query_memory=query_memory,
            memory_targets=bridge_memory_targets,
            grounding=grounding_payload,
            history_raw=start_history_raw,
            spec_names=spec_names,
            start_distance_threshold_z=float(args.start_distance_threshold_z),
            start_distance_penalty=float(args.start_distance_penalty),
            implication_alignment_weight=float(args.implication_alignment_weight),
        )
    memory_priors = []
    conditioning_rows = []
    for row in variant_rows:
        query_start_state = query_start_state_for_variant(
            row=row,
            history_level=start_history_level,
            user_start=user_start,
        )
        prior = build_mixture_memory_prior(
            query_memory=query_memory,
            memory_targets=support_memory_targets,
            history_level=support_history_level,
            train_indices=support_candidate_indices,
            query_window_index=int(query_row["window_index"]),
            query_start_state=query_start_state,
            grounding=grounding_payload if isinstance(grounding_payload, dict) else {},
            spec_names=spec_names,
            mode=memory_prior_mode,
            top_k=int(getattr(args, "memory_prior_top_k", 8)),
            temperature=float(getattr(args, "memory_prior_temperature", 0.2)),
            start_distance_threshold_z=float(args.start_distance_threshold_z),
            start_distance_penalty=float(args.start_distance_penalty),
            implication_alignment_weight=float(args.implication_alignment_weight),
            diverse_max_pairwise_cosine=float(
                getattr(args, "memory_prior_diverse_max_pairwise_cosine", 0.98)
            ),
            diverse_min_index_gap=int(
                getattr(args, "memory_prior_diverse_min_index_gap", 0)
            ),
            quality_guard_candidate_pool_size=int(
                getattr(args, "memory_prior_quality_guard_candidate_pool_size", 5)
            ),
            quality_guard_mixture_size=int(
                getattr(args, "memory_prior_quality_guard_mixture_size", 3)
            ),
            quality_guard_max_mixtures=int(
                getattr(args, "memory_prior_quality_guard_max_mixtures", 0)
            ),
            quality_guard_min_candidate_mixtures=int(
                getattr(args, "memory_prior_quality_guard_min_candidate_mixtures", 1)
            ),
            quality_guard_max_candidate_entropy_quantile=(
                None
                if float(
                    getattr(
                        args,
                        "memory_prior_quality_guard_max_candidate_entropy_quantile",
                        -1.0,
                    )
                )
                < 0.0
                else float(
                    getattr(
                        args,
                        "memory_prior_quality_guard_max_candidate_entropy_quantile",
                        -1.0,
                    )
                )
            ),
            quality_guard_min_support_weight_max_quantile=(
                None
                if float(
                    getattr(
                        args,
                        "memory_prior_quality_guard_min_support_weight_max_quantile",
                        0.25,
                    )
                )
                < 0.0
                else float(
                    getattr(
                        args,
                        "memory_prior_quality_guard_min_support_weight_max_quantile",
                        0.25,
                    )
                )
            ),
            quality_guard_probability_temperature=(
                None
                if float(
                    getattr(
                        args,
                        "memory_prior_quality_guard_probability_temperature",
                        -1.0,
                    )
                )
                < 0.0
                else float(
                    getattr(
                        args,
                        "memory_prior_quality_guard_probability_temperature",
                        -1.0,
                    )
                )
            ),
        )
        memory_priors.append(prior)
        conditioning_rows.append(np.asarray(prior["memory"], dtype=np.float32))
    operational_prior_pos = next(
        (
            pos
            for pos, row in enumerate(variant_rows)
            if bool(row.get("is_operational"))
        ),
        max(len(variant_rows) - 1, 0),
    )
    conditioning_memory = np.asarray(
        conditioning_rows[operational_prior_pos],
        dtype=np.float32,
    )
    memory_prior = enrich_memory_prior_candidate_metadata(
        memory_priors[operational_prior_pos],
        support_window_metadata,
    )
    variant_rows = [
        annotate_variant_with_memory_prior(row, prior)
        for row, prior in zip(variant_rows, memory_priors, strict=True)
    ]
    variant_rows = _enrich_variant_rows(variant_rows, start_window_metadata)
    text_memory = np.stack(conditioning_rows, axis=0).astype(np.float32)
    start_indices = np.asarray(
        [int(row["start_window_index"]) for row in variant_rows],
        dtype=np.int64,
    )
    requested_start_rows: list[np.ndarray] = []
    requested_raw_rows: list[np.ndarray] = []
    future_state_rows: list[np.ndarray] = []
    all_future_targets_available = True
    for row in variant_rows:
        start_idx = int(row["start_window_index"])
        if start_idx >= 0:
            requested_start_rows.append(start_history_level[start_idx, -1, :])
            requested_raw_rows.append(start_history_raw[start_idx, -1, :])
            future_state_rows.append(start_future_raw[start_idx])
        else:
            if user_start is None:
                raise ValueError("user start row requires loaded user_start")
            requested_start_rows.append(
                np.asarray(user_start["encoded_state"], dtype=np.float32)
            )
            requested_raw_rows.append(
                np.asarray(user_start["raw_state"], dtype=np.float32)
            )
            all_future_targets_available = False
    requested_start = np.stack(requested_start_rows, axis=0).astype(np.float32)
    requested_raw = np.stack(requested_raw_rows, axis=0).astype(np.float32)
    future_states_for_paths = (
        np.stack(future_state_rows, axis=0).astype(np.float32)
        if all_future_targets_available
        else None
    )
    prefix_prior_mode = str(getattr(args, "prefix_prior_mode", "decoder"))
    rollout_mixture_mode = str(
        getattr(args, "rollout_mixture_mode", "component_prefix_mixture")
    )
    response_preview_rows: list[dict[str, Any]] = []
    if (
        not bool(args.skip_rollout)
        and rollout_mixture_mode == "response_preview_component_mixture"
    ):
        np.random.seed(rollout_seed + 10007)
        torch.manual_seed(rollout_seed + 10007)
        memory_priors, response_preview_rows = response_preview_reweight_memory_priors(
            model,
            memory_priors=memory_priors,
            fallback_memories=text_memory,
            memory_targets=support_memory_targets,
            prefix_prior_mode=prefix_prior_mode,
            decoder_result=decoder_result,
            input_stats=input_stats,
            features=features,
            layout=layout,
            requested_start=requested_start,
            requested_raw=requested_raw,
            specs=specs,
            grounding=grounding_payload if isinstance(grounding_payload, dict) else {},
            spec_names=spec_names,
            preview_samples_per_component=int(
                getattr(args, "response_preview_samples_per_component", 8)
            ),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            response_alpha=float(getattr(args, "response_preview_alpha", 0.75)),
            response_temperature=float(
                getattr(args, "response_preview_temperature", 1.0)
            ),
            response_blend=float(getattr(args, "response_preview_blend", 1.0)),
            response_objective=str(
                getattr(args, "response_preview_objective", "narrative_channels")
            ),
            device=device,
        )
        conditioning_rows = [
            np.asarray(prior["memory"], dtype=np.float32) for prior in memory_priors
        ]
        conditioning_memory = np.asarray(
            conditioning_rows[operational_prior_pos],
            dtype=np.float32,
        )
        memory_prior = enrich_memory_prior_candidate_metadata(
            memory_priors[operational_prior_pos],
            support_window_metadata,
        )
        variant_rows = [
            annotate_variant_with_memory_prior(row, prior)
            for row, prior in zip(variant_rows, memory_priors, strict=True)
        ]
        variant_rows = _enrich_variant_rows(variant_rows, start_window_metadata)
        text_memory = np.stack(conditioning_rows, axis=0).astype(np.float32)
    variant_inputs = apply_memory_start_stats(text_memory, requested_start, input_stats)
    if prefix_prior_mode == "feature_mixture":
        mixed_feature_rows = []
        for prior in memory_priors:
            prior_indices = np.asarray(prior.get("window_indices", []), dtype=np.int64)
            prior_weights = np.asarray(prior.get("weights", []), dtype=np.float32)
            if prior_indices.size == 0 or prior_weights.size != prior_indices.size:
                raise ValueError(
                    "feature_mixture prefix prior requires an analogue mixture "
                    "memory prior with non-empty window_indices and weights"
                )
            prior_weights = prior_weights / max(float(np.sum(prior_weights)), 1e-8)
            mixed_feature_rows.append(
                np.sum(features[prior_indices] * prior_weights[:, None], axis=0)
            )
        mixed_features = np.stack(mixed_feature_rows, axis=0).astype(np.float32)
        decoded_prefix = reconstruct_prefix_from_features(
            mixed_features,
            start_state=requested_start,
            layout=layout,
        )
    elif prefix_prior_mode == "decoder":
        decoded_prefix = _decode_features(
            decoder_result["model"],
            decoder_result["target_mean"],
            decoder_result["target_std"],
            variant_inputs,
            start_state=requested_start,
            layout=layout,
            device=device,
        )
    else:
        raise ValueError(f"unknown prefix_prior_mode: {prefix_prior_mode!r}")
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
    train_delta = support_future_delta[support_train_indices]
    delta_scale = build_delta_scale(train_delta, floor=float(args.score_scale_floor))
    samples = np.empty((0,), dtype=np.float32)
    uncalibrated_samples = np.empty((0,), dtype=np.float32)
    generated_states = np.empty((0,), dtype=np.float32)
    window_scores: list[dict[str, Any]] = []
    rollout_summary: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    rollout_component_rows: list[dict[str, Any]] = []
    if not bool(args.skip_rollout):
        np.random.seed(rollout_seed)
        torch.manual_seed(rollout_seed)
        if rollout_mixture_mode in {
            "component_prefix_mixture",
            "response_preview_component_mixture",
        }:
            samples, generated_states, rollout_component_rows = (
                sample_component_prefix_mixture_from_start_raw(
                    model,
                    memory_priors=memory_priors,
                    fallback_memories=text_memory,
                    memory_targets=support_memory_targets,
                    prefix_prior_mode=prefix_prior_mode,
                    decoder_result=decoder_result,
                    input_stats=input_stats,
                    features=features,
                    layout=layout,
                    requested_start=requested_start,
                    requested_raw=requested_raw,
                    specs=specs,
                    samples=int(args.samples),
                    n_steps=int(args.n_steps),
                    chunk_size=int(args.chunk_size),
                    temperature=float(args.temperature),
                    device=device,
                )
            )
        elif rollout_mixture_mode == "averaged_prefix":
            samples = sample_prefix_generator_deltas_from_start_raw(
                model,
                history_level=decoded_prefix["history_level"],
                history_norm=decoded_prefix["history_norm"],
                center=decoded_prefix["center"],
                scale=decoded_prefix["scale"],
                drift_feature=decoded_prefix["drift_feature"],
                start_raw=requested_raw,
                specs=specs,
                samples=int(args.samples),
                n_steps=int(args.n_steps),
                chunk_size=int(args.chunk_size),
                temperature=float(args.temperature),
                device=device,
            )
            generated_states = generated_delta_samples_to_states(samples, requested_raw)
        else:
            raise ValueError(f"unknown rollout_mixture_mode: {rollout_mixture_mode!r}")
        current_raw = requested_raw
        rollout_fan_scale = float(getattr(args, "rollout_fan_scale", 1.0))
        uncalibrated_samples = samples.astype(np.float32, copy=True)
        if abs(rollout_fan_scale - 1.0) > 1e-8:
            samples = scale_delta_samples_around_mean(samples, rollout_fan_scale)
            generated_states = generated_delta_samples_to_states(samples, requested_raw)
        window_scores, rollout_summary = _score_live_rollouts(
            samples=samples,
            future_delta=start_future_delta,
            delta_scale=delta_scale,
            variant_rows=variant_rows,
        )
        generation = summarize_retrieval_generated_states(
            generated_states,
            current_raw,
            _spec_names(specs),
        )
        path_quantiles = path_quantiles_for_generated_states(
            generated_states,
            current_raw,
            _spec_names(specs),
            analogues=_variant_path_labels(variant_rows, start_window_metadata),
            future_states=future_states_for_paths,
            max_paths=int(args.max_paths),
        )
        for item in path_quantiles:
            if isinstance(item, dict) and str(item.get("analogue_key", "ALL")) == "ALL":
                item["analogue_label"] = "All start variants"
        generation["rollout_temperature"] = float(args.temperature)
        generation["sample_count"] = int(args.samples)
        generation["forecast_steps"] = int(args.n_steps)
        generation["solver_steps"] = int(args.steps)
        generation["chunk_size"] = int(args.chunk_size)
        generation["rollout_mixture_mode"] = rollout_mixture_mode
        generation["rollout_fan_scale"] = float(rollout_fan_scale)
        generation["rollout_fan_calibration"] = {
            "mode": "global_delta_scale_around_mean",
            "alpha": float(rollout_fan_scale),
            "mean_path_preserved": True,
            "applied": bool(abs(rollout_fan_scale - 1.0) > 1e-8),
        }
        generation["sample_delta_shape"] = [int(dim) for dim in samples.shape]
        generation["generated_state_shape"] = [
            int(dim) for dim in generated_states.shape
        ]
        generation["rollout_component_count"] = int(
            sum(1 for row in rollout_component_rows if not bool(row.get("skipped")))
        )
        generation["rollout_component_count_operational"] = int(
            sum(
                1
                for row in rollout_component_rows
                if int(row.get("variant_index", -1)) == int(operational_prior_pos)
                and not bool(row.get("skipped"))
            )
        )
        generation["rollout_component_sample_count_operational"] = int(
            sum(
                int(row.get("sample_count", 0))
                for row in rollout_component_rows
                if int(row.get("variant_index", -1)) == int(operational_prior_pos)
            )
        )
        generation["response_preview_support_weighting"] = {
            "applied": bool(
                rollout_mixture_mode == "response_preview_component_mixture"
            ),
            "rows": response_preview_rows,
            "response_objective": str(
                getattr(args, "response_preview_objective", "narrative_channels")
            ),
        }
        generation["rollout_component_rows"] = rollout_component_rows
        generation["path_quantiles"] = path_quantiles
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
    start_reliability_gate = _story_smoke_start_reliability(
        manifest_path=getattr(args, "start_reliability_manifest", None),
        operational_row=variant_rows[operational_prior_pos],
    )
    if condition_report_narrative_text is not None:
        narrative_text = condition_report_narrative_text
    elif bool(getattr(args, "live_story", False)):
        narrative_text = str(args.story)
    else:
        narrative_text = narrative_text_for_query(pipeline_report, query_row)
    live_story_mode = bool(getattr(args, "live_story", False))
    condition_report_mode = bool(getattr(args, "condition_report", None))
    report = {
        "status": "ok",
        "scope_note": (
            "Live OpenAI-grounded prefix-latent story smoke. OpenAI is called for "
            "story grounding and text embedding, then the projected memory is "
            "combined with an explicit start state, decoded into a recent prefix, "
            "and rolled out through the frozen joint39 SNI generator."
            if live_story_mode
            else (
                (
                    "Replayed live prefix-latent story smoke. No OpenAI API calls are made. "
                    "A saved live-story text memory and grounding report are reused, "
                    "optionally converted into an analogue-mixture memory prior, then "
                    "decoded with an explicit start state and rolled out through the "
                    "frozen joint39 SNI generator."
                )
                if condition_report_mode
                else (
                    "Cached live prefix-latent story smoke. No OpenAI API calls are made. "
                    "A cached text-predicted generator memory is combined with an explicit "
                    "start state, decoded into a recent prefix, and rolled out through the "
                    "frozen joint39 SNI generator."
                )
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
            "conditioning_memory_norm": float(np.linalg.norm(conditioning_memory)),
            "memory_prior_mode": memory_prior_mode,
            "memory_prior_contract": "per_variant_narrative_and_fixed_start",
            "rollout_mixture_mode": rollout_mixture_mode,
            "operational_memory_prior_variant_index": int(operational_prior_pos),
            "response_preview_support_weighting": {
                "applied": bool(
                    rollout_mixture_mode == "response_preview_component_mixture"
                    and len(response_preview_rows) > 0
                ),
                "preview_samples_per_component": int(
                    getattr(args, "response_preview_samples_per_component", 8)
                ),
                "response_alpha": float(getattr(args, "response_preview_alpha", 0.75)),
                "response_temperature": float(
                    getattr(args, "response_preview_temperature", 1.0)
                ),
                "response_blend": float(getattr(args, "response_preview_blend", 1.0)),
            },
            "memory_prior": {
                key: value for key, value in memory_prior.items() if key != "memory"
            },
        },
        "artifact_inputs": {
            "bridge_report": str(args.bridge_report),
            "bridge_arrays": str(args.bridge_arrays),
            "pipeline_report": str(args.pipeline_report),
            "checkpoint": str(args.checkpoint),
            "start_reliability_manifest": (
                str(args.start_reliability_manifest)
                if getattr(args, "start_reliability_manifest", None)
                else None
            ),
            "start_state_json": (
                str(args.start_state_json)
                if getattr(args, "start_state_json", None)
                else None
            ),
        },
        "user_start_state": (
            {
                "label": str(user_start.get("label", "")),
                "source_path": str(user_start.get("source_path", "")),
                "source_format": str(user_start.get("source_format", "")),
                "coordinate": str(user_start.get("coordinate", "")),
                "dimension": int(np.asarray(user_start["raw_state"]).shape[0]),
                "spec_names": list(user_start.get("spec_names", [])),
            }
            if user_start is not None
            else None
        ),
        "selected_start_state": {
            "variant_index": int(operational_prior_pos),
            "values_by_name": {
                str(name): float(requested_raw[operational_prior_pos, idx])
                for idx, name in enumerate(spec_names)
            },
        },
        "selected_window_count": int(selected_windows.size),
        "train_window_count": int(support_train_indices.size),
        "test_window_count": int(support_test_indices.size),
        "bridge_labeled_window_count": int(selected_windows.size),
        "bridge_train_window_count": int(bridge_train_indices.size),
        "bridge_test_window_count": int(bridge_test_indices.size),
        "support_bank": support_bank_summary,
        "device": str(device),
        "seeds": {
            "base_seed": int(base_seed),
            "decoder_seed": int(decoder_seed),
            "rollout_seed": int(rollout_seed),
        },
        "decoder": {
            "prefix_prior_mode": prefix_prior_mode,
            "rollout_mixture_mode": rollout_mixture_mode,
            "loss_first": float(decoder_result["loss_first"]),
            "loss_last": float(decoder_result["loss_last"]),
            "train_mse": float(decoder_result["train_mse"]),
            "test_mse": float(decoder_result["test_mse"]),
        },
        "endpoint_alignment": endpoint,
        "variant_rows": variant_rows,
        "rollout_shifts": rollout_shifts,
        "validation_gate": validation_gate,
        "start_reliability_gate": start_reliability_gate,
        "generation": generation,
        "artifact_paths": {
            "report": str(
                Path(args.output_dir) / "prefix_latent_story_smoke_report.json"
            ),
            "markdown": str(
                Path(args.output_dir) / "prefix_latent_story_smoke_report.md"
            ),
            "arrays": str(
                Path(args.output_dir) / "prefix_latent_story_smoke_arrays.npz"
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "prefix_latent_story_smoke_arrays.npz",
        selected_window_indices=selected_windows.astype(np.int64),
        train_indices=support_train_indices.astype(np.int64),
        test_indices=support_test_indices.astype(np.int64),
        support_candidate_indices=support_candidate_indices.astype(np.int64),
        bridge_train_indices=bridge_train_indices.astype(np.int64),
        bridge_test_indices=bridge_test_indices.astype(np.int64),
        start_indices=start_indices.astype(np.int64),
        requested_start=requested_start.astype(np.float32),
        requested_raw=requested_raw.astype(np.float32),
        text_memory=text_memory.astype(np.float32),
        decoded_memory=decoded_memory.astype(np.float32),
        decoded_history_level=decoded_prefix["history_level"].astype(np.float32),
        decoded_history_norm=decoded_prefix["history_norm"].astype(np.float32),
        decoded_center=decoded_prefix["center"].astype(np.float32),
        decoded_scale=decoded_prefix["scale"].astype(np.float32),
        decoded_drift_feature=decoded_prefix["drift_feature"].astype(np.float32),
        uncalibrated_samples=uncalibrated_samples.astype(np.float32),
        rollout_component_variant_index=np.asarray(
            [int(row["variant_index"]) for row in rollout_component_rows],
            dtype=np.int64,
        ),
        rollout_component_window_index=np.asarray(
            [int(row["support_window_index"]) for row in rollout_component_rows],
            dtype=np.int64,
        ),
        rollout_component_weight=np.asarray(
            [float(row["weight"]) for row in rollout_component_rows],
            dtype=np.float32,
        ),
        rollout_component_sample_count=np.asarray(
            [int(row["sample_count"]) for row in rollout_component_rows],
            dtype=np.int64,
        ),
        delta_scale=delta_scale.astype(np.float32),
        samples=samples.astype(np.float32),
        generated_states=generated_states.astype(np.float32),
    )
    _write_json(output_dir / "prefix_latent_story_smoke_report.json", report)
    _write_text(
        output_dir / "prefix_latent_story_smoke_report.md", _render_markdown(report)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument(
        "--support-bank-report",
        help=(
            "Optional broad SNI support-bank report. When provided with "
            "--support-bank-arrays, historical support search and prefix-decoder "
            "training use this bank instead of the labeled bridge-window slice."
        ),
    )
    parser.add_argument(
        "--support-bank-arrays",
        help="NPZ arrays path paired with --support-bank-report.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-role", default="anchor")
    parser.add_argument("--query-kind")
    parser.add_argument("--query-window-id")
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument(
        "--condition-report",
        help=(
            "Replay a saved prefix story-smoke report and its arrays as the "
            "condition source. This avoids new OpenAI calls while preserving a "
            "live-story text memory and grounding payload."
        ),
    )
    parser.add_argument("--live-story", action="store_true")
    parser.add_argument("--story", default=DEFAULT_STORY)
    parser.add_argument("--grounding-json")
    parser.add_argument("--grounding-model", default="gpt-5.4-mini")
    parser.add_argument("--grounding-max-output-tokens", type=int, default=1200)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--bridge-adapter", default=DEFAULT_BRIDGE_ADAPTER)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument(
        "--start-reliability-manifest",
        help=(
            "Optional start reliability manifest. When provided, the selected "
            "historical start receives a product-facing pass/warn diagnostic."
        ),
    )
    parser.add_argument(
        "--start-mode",
        choices=[
            "original",
            "nearest_train_start",
            "farthest_train_start",
            "memory_nearest_start",
            "balanced_memory_start",
            "implication_aligned_start",
            "explicit_start_window",
            "user_start_state",
        ],
        default="balanced_memory_start",
    )
    parser.add_argument("--explicit-start-window-index", type=int)
    parser.add_argument(
        "--start-state-json",
        help=(
            "Path to a raw joint39 start-state JSON object. Use with "
            "--start-mode user_start_state. The JSON may contain state_vector "
            "or values_by_name keyed by generator spec names."
        ),
    )
    parser.add_argument(
        "--start-distance-threshold-z",
        type=float,
        default=float(DEFAULT_GATE_THRESHOLDS["start_distance_warn"]),
    )
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument("--implication-alignment-weight", type=float, default=0.25)
    parser.add_argument(
        "--memory-prior-mode",
        choices=[
            "query_memory",
            "soft_topk_memory",
            "soft_topk_narrative_start",
            "soft_topk_narrative_start_checked",
            "diverse_topk_narrative_start_checked",
            "cohesive_topk_narrative_start_checked",
            "cluster_family_narrative_start_checked",
            "kernel_topk_narrative_start_checked",
            "portfolio_quality_guard_924e",
            "portfolio_direction_first_quality_guard_938a",
            "broad_replay_response_guard_940a",
            "narrative_book_quality_guard_926b",
            "narrative_book_direction_first_quality_guard_938c",
            "soft_topk_start_only",
            "soft_topk_combined",
            "diverse_topk_narrative_start",
            "diverse_topk_combined",
        ],
        default="query_memory",
    )
    parser.add_argument("--memory-prior-top-k", type=int, default=8)
    parser.add_argument("--memory-prior-temperature", type=float, default=0.2)
    parser.add_argument(
        "--memory-prior-diverse-max-pairwise-cosine",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--memory-prior-diverse-min-index-gap",
        type=int,
        default=0,
        help=(
            "Minimum bridge-local window-index separation between diverse "
            "support components. Use a positive value to avoid overlapping "
            "historical prefixes in demo/product support tables."
        ),
    )
    parser.add_argument(
        "--memory-prior-quality-guard-candidate-pool-size",
        type=int,
        default=5,
        help=(
            "For portfolio_quality_guard_924e and "
            "narrative_book_quality_guard_926b, number of diverse support "
            "components to consider before forming candidate mixtures."
        ),
    )
    parser.add_argument(
        "--memory-prior-quality-guard-mixture-size",
        type=int,
        default=3,
        help=(
            "For portfolio_quality_guard_924e and "
            "narrative_book_quality_guard_926b, number of support components "
            "inside each live candidate mixture."
        ),
    )
    parser.add_argument(
        "--memory-prior-quality-guard-max-mixtures",
        type=int,
        default=0,
        help=(
            "For portfolio_quality_guard_924e and "
            "narrative_book_quality_guard_926b, cap candidate mixtures. "
            "Use 0 to evaluate all combinations from the candidate pool."
        ),
    )
    parser.add_argument(
        "--memory-prior-quality-guard-min-candidate-mixtures",
        type=int,
        default=1,
        help=(
            "For portfolio_quality_guard_924e and "
            "narrative_book_quality_guard_926b, fall back to the base support "
            "prior unless at least this many live candidate mixtures exist."
        ),
    )
    parser.add_argument(
        "--memory-prior-quality-guard-max-candidate-entropy-quantile",
        type=float,
        default=-1.0,
        help=(
            "Optional train-set candidate-entropy quantile gate. Negative "
            "disables the entropy gate."
        ),
    )
    parser.add_argument(
        "--memory-prior-quality-guard-min-support-weight-max-quantile",
        type=float,
        default=0.25,
        help=(
            "Train-set support-concentration quantile gate. Use a negative "
            "value to disable the support-max fallback for research probes."
        ),
    )
    parser.add_argument(
        "--memory-prior-quality-guard-probability-temperature",
        type=float,
        default=-1.0,
        help=(
            "Optional support-policy probability temperature override. "
            "Negative uses the policy context default."
        ),
    )
    parser.add_argument(
        "--prefix-prior-mode",
        choices=["decoder", "feature_mixture"],
        default="decoder",
        help=(
            "decoder uses the learned memory+start prefix decoder. "
            "feature_mixture bypasses that decoder and reconstructs a "
            "start-pinned prefix directly from weighted analogue prefix features."
        ),
    )
    parser.add_argument(
        "--rollout-mixture-mode",
        choices=[
            "averaged_prefix",
            "component_prefix_mixture",
            "response_preview_component_mixture",
        ],
        default="component_prefix_mixture",
        help=(
            "averaged_prefix decodes one weighted-average support memory. "
            "component_prefix_mixture decodes and rolls out each support component "
            "separately, then pools weighted samples so regime-mixture shape is "
            "not averaged away before generation. response_preview_component_mixture "
            "first runs small per-component preview rollouts to reweight support "
            "in narrative-relevant risk channels before final pooling."
        ),
    )
    parser.add_argument(
        "--response-preview-samples-per-component",
        type=int,
        default=8,
        help=(
            "For response_preview_component_mixture, number of small preview "
            "rollout samples used to score each support component before final "
            "component-preserving generation."
        ),
    )
    parser.add_argument("--response-preview-alpha", type=float, default=0.75)
    parser.add_argument("--response-preview-temperature", type=float, default=1.0)
    parser.add_argument(
        "--response-preview-blend",
        type=float,
        default=1.0,
        help=(
            "Convex blend between original support weights and response-preview "
            "weights. 1.0 fully applies preview weights; smaller values bound "
            "the response update around the historical support prior."
        ),
    )
    parser.add_argument(
        "--response-preview-objective",
        choices=["narrative_channels", "factor_portfolio", "channel_portfolio"],
        default="narrative_channels",
        help=(
            "Preview scoring objective. narrative_channels uses only the "
            "grounded narrative risk channels. factor_portfolio keeps that "
            "score and adds a fixed portfolio-tail response term so support "
            "weights target the same product gate used in stress tests. "
            "channel_portfolio scores the joint signed risk-channel movement "
            "implied by the narrative."
        ),
    )
    parser.add_argument(
        "--include-original-baseline", action="store_true", default=True
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=791)
    parser.add_argument(
        "--decoder-seed",
        type=int,
        default=None,
        help=(
            "Optional seed for the memory+start prefix decoder. If omitted, "
            "--seed is used for backward-compatible behavior."
        ),
    )
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=None,
        help=(
            "Optional seed reset immediately before frozen-generator rollout. "
            "If omitted, --seed is used for backward-compatible behavior."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--rollout-fan-scale",
        type=float,
        default=1.0,
        help=(
            "Post-hoc fan calibration alpha. Values other than 1.0 scale "
            "generated delta samples around each ensemble mean, preserving the "
            "mean path while changing fan width."
        ),
    )
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
    parser.add_argument("--hard-case-count", type=int, default=8)
    parser.add_argument("--max-paths", type=int, default=6)
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument(
        "--eval_split", choices=["val", "train", "train_tail"], default="val"
    )
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument(
        "--positive_level_policy",
        choices=["reference_based", "observed_positive"],
        default="reference_based",
    )
    parser.add_argument(
        "--iv_transform", choices=["log_level", "bounded_logit"], default="log_level"
    )
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument(
        "--drift_feature_mode", choices=["none", "ewma_mean"], default="none"
    )
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
