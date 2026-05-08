#!/usr/bin/env python
"""Live cached story smoke for the narrative prefix-latent workflow.

This script makes no OpenAI calls. It uses a cached held-out narrative/text
memory from the representative OpenAI bridge run, combines it with an original
or selected explicit starting state, decodes a synthetic recent prefix, and
runs the frozen joint39 SNI generator through its native rollout.
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
    case_rollout_shift_rows,
    evaluate_start_case_gates,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    DEFAULT_PIPELINE_REPORT,
    path_quantiles_for_generated_states,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    build_delta_scale,
    future_delta_paths,
    load_bridge_arrays,
    score_sample_distribution,
    summarize_method_scores,
)


StartMode = Literal[
    "original",
    "nearest_train_start",
    "farthest_train_start",
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


def resolve_start_window_index(
    *,
    query_window_index: int,
    start_state: np.ndarray,
    train_indices: np.ndarray,
    start_mode: StartMode | str,
    explicit_start_window_index: int | None = None,
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
    elif mode == "explicit_start_window":
        if explicit_start_window_index is None:
            raise ValueError("explicit_start_window_index is required")
        start_idx = int(explicit_start_window_index)
    else:
        start_z = _safe_start_z(starts, train)
        distances = np.linalg.norm(start_z[train] - start_z[query_idx], axis=1)
        if mode == "nearest_train_start":
            start_idx = int(train[int(np.argmin(distances))])
        elif mode == "farthest_train_start":
            start_idx = int(train[int(np.argmax(distances))])
        else:
            raise ValueError(f"unknown start_mode: {start_mode!r}")
    if start_idx < 0 or start_idx >= starts.shape[0]:
        raise IndexError(f"start_window_index {start_idx} outside {starts.shape[0]}")
    return {
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


def build_live_story_variant_rows(
    *,
    query_row: dict[str, Any],
    start_state: np.ndarray,
    train_indices: np.ndarray,
    start_mode: StartMode | str,
    explicit_start_window_index: int | None = None,
    include_original_baseline: bool = True,
) -> list[dict[str, Any]]:
    """Build original and selected-start rows for one live cached story query."""

    query_idx = int(query_row["window_index"])
    original = resolve_start_window_index(
        query_window_index=query_idx,
        start_state=start_state,
        train_indices=train_indices,
        start_mode="original",
    )
    selected = resolve_start_window_index(
        query_window_index=query_idx,
        start_state=start_state,
        train_indices=train_indices,
        start_mode=start_mode,
        explicit_start_window_index=explicit_start_window_index,
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
    lines = [
        "# Prefix-Latent Story Smoke",
        "",
        "## Cached Narrative",
        "",
        str(query.get("narrative_text", "")),
        "",
        "## Product Contract",
        "",
        "- Cached text memory is used as the narrative condition.",
        "- The requested start state is pinned as the final prefix level.",
        "- A learned memory+start decoder reconstructs a recent prefix object.",
        "- The frozen joint39 SNI generator performs the 30-day rollout.",
        "",
        "## Start Selection",
        "",
        "| Variant | Query Window | Start Window | Start Distance z |",
        "| --- | --- | --- | ---: |",
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
    variant_rows = build_live_story_variant_rows(
        query_row=query_row,
        start_state=history_level[:, -1, :],
        train_indices=train_indices,
        start_mode=str(args.start_mode),
        explicit_start_window_index=args.explicit_start_window_index,
        include_original_baseline=bool(args.include_original_baseline),
    )
    window_metadata = window_metadata_by_bridge_local_index(bridge_report)
    variant_rows = _enrich_variant_rows(variant_rows, window_metadata)
    query_memory = np.asarray(
        bridge_arrays["condition_vectors"][int(query_row["embedding_index"])],
        dtype=np.float32,
    )
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
    narrative_text = narrative_text_for_query(pipeline_report, query_row)
    report = {
        "status": "ok",
        "scope_note": (
            "Cached live prefix-latent story smoke. No OpenAI API calls are made. "
            "A cached text-predicted generator memory is combined with an explicit "
            "start state, decoded into a recent prefix, and rolled out through the "
            "frozen joint39 SNI generator."
        ),
        "cached_query": {
            **query_row,
            "narrative_text": narrative_text,
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
    parser.add_argument(
        "--start-mode",
        choices=[
            "original",
            "nearest_train_start",
            "farthest_train_start",
            "explicit_start_window",
        ],
        default="original",
    )
    parser.add_argument("--explicit-start-window-index", type=int)
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
