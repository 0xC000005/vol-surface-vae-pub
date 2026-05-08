#!/usr/bin/env python
"""Start-state sensitivity gate for the narrative prefix-latent decoder.

This script makes no OpenAI calls. It tests the production contract where the
same narrative-derived generator memory is held fixed while a risk manager
changes the explicit starting market state.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    compute_memory_targets,
)
from experiments.backfill.block_ar.nl_prefix_latent_memory_decoder import (  # noqa: E402
    _decode_features,
    _future_raw_from_block,
    apply_memory_start_stats,
    build_memory_start_input_matrix,
    cosine_summary,
    train_memory_start_prefix_decoder,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    build_prefix_feature_matrix,
    sample_prefix_generator_deltas,
    selected_bridge_window_indices,
    split_indices_from_bridge_report,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    build_delta_scale,
    future_delta_paths,
    load_bridge_arrays,
    score_sample_distribution,
    select_heldout_query_rows,
    summarize_method_scores,
)


DEFAULT_BRIDGE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/"
    "bridge_eval_arrays.npz"
)


def _safe_start_z(start_state: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    start = np.asarray(start_state, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    mean = start[fit].mean(axis=0, keepdims=True)
    std = np.maximum(start[fit].std(axis=0, keepdims=True), 1e-6)
    return ((start - mean) / std).astype(np.float32)


def build_start_variant_rows(
    *,
    query_window_indices: np.ndarray,
    train_indices: np.ndarray,
    start_state: np.ndarray,
) -> list[dict[str, Any]]:
    """Build original, nearest-train-start, and farthest-train-start variants."""

    queries = np.asarray(query_window_indices, dtype=np.int64)
    train = np.asarray(train_indices, dtype=np.int64)
    start_z = _safe_start_z(start_state, train)
    if queries.size == 0 or train.size == 0:
        raise ValueError("query_window_indices and train_indices must be non-empty")
    rows: list[dict[str, Any]] = []
    for query_idx in queries:
        distances = np.linalg.norm(start_z[train] - start_z[int(query_idx)], axis=1)
        nearest = int(train[int(np.argmin(distances))])
        farthest = int(train[int(np.argmax(distances))])
        rows.extend(
            [
                {
                    "query_window_index": int(query_idx),
                    "start_window_index": int(query_idx),
                    "variant": "original",
                    "start_distance_z": 0.0,
                },
                {
                    "query_window_index": int(query_idx),
                    "start_window_index": nearest,
                    "variant": "nearest_train_start",
                    "start_distance_z": float(np.min(distances)),
                },
                {
                    "query_window_index": int(query_idx),
                    "start_window_index": farthest,
                    "variant": "farthest_train_start",
                    "start_distance_z": float(np.max(distances)),
                },
            ]
        )
    return rows


def endpoint_alignment_summary(
    history_level: np.ndarray,
    requested_start: np.ndarray,
) -> dict[str, float]:
    """Measure whether decoded prefixes end at the requested starting state."""

    history = np.asarray(history_level, dtype=np.float32)
    start = np.asarray(requested_start, dtype=np.float32)
    if history.ndim != 3 or start.shape != (history.shape[0], history.shape[2]):
        raise ValueError("history_level must be [N,T,C] and requested_start [N,C]")
    err = np.abs(history[:, -1, :] - start)
    return {
        "max_abs_error": float(np.max(err)),
        "mean_abs_error": float(np.mean(err)),
    }


def _cosine_by_variant(
    left: np.ndarray,
    right: np.ndarray,
    variant_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(variant_rows):
        buckets[str(row["variant"])].append(idx)
    return {
        name: cosine_summary(left[np.asarray(indices)], right[np.asarray(indices)])
        for name, indices in buckets.items()
    }


def rollout_sensitivity_summary(
    *,
    samples: np.ndarray,
    variant_rows: list[dict[str, Any]],
    scale: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compare variant sample means against each query's original-start sample mean."""

    sample_arr = np.asarray(samples, dtype=np.float32)
    scale_arr = np.maximum(np.asarray(scale, dtype=np.float32), 1e-8)
    if sample_arr.ndim != 4:
        raise ValueError("samples must have shape [K,S,T,C]")
    if scale_arr.shape != sample_arr.shape[2:]:
        raise ValueError("scale must have shape [T,C]")
    original_by_query: dict[int, np.ndarray] = {}
    sample_z = sample_arr / scale_arr[None, None, :, :]
    for idx, row in enumerate(variant_rows):
        if str(row["variant"]) == "original":
            original_by_query[int(row["query_window_index"])] = sample_z[idx].mean(axis=0)
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for idx, row in enumerate(variant_rows):
        variant = str(row["variant"])
        if variant == "original":
            continue
        query_idx = int(row["query_window_index"])
        if query_idx not in original_by_query:
            continue
        diff = sample_z[idx].mean(axis=0) - original_by_query[query_idx]
        buckets[variant].append(
            {
                "mean_abs_delta_z": float(np.mean(np.abs(diff))),
                "terminal_mean_abs_delta_z": float(np.mean(np.abs(diff[-1]))),
            }
        )
    summary: dict[str, dict[str, float]] = {}
    for variant, values in buckets.items():
        summary[variant] = {
            "case_count": float(len(values)),
            "mean_abs_delta_z": float(np.mean([v["mean_abs_delta_z"] for v in values])),
            "terminal_mean_abs_delta_z": float(
                np.mean([v["terminal_mean_abs_delta_z"] for v in values])
            ),
        }
    return summary


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _score_original_rollouts(
    *,
    query_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    samples: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_by_window = {int(row["window_index"]): row for row in query_rows}
    window_scores: list[dict[str, Any]] = []
    for variant_idx, variant_row in enumerate(variant_rows):
        if str(variant_row["variant"]) != "original":
            continue
        window_index = int(variant_row["query_window_index"])
        if window_index not in query_by_window:
            continue
        target = future_delta[window_index]
        window_scores.append(
            {
                "window_index": window_index,
                "window_id": str(query_by_window[window_index].get("window_id", "")),
                "methods": {
                    "persistence": score_sample_distribution(
                        np.zeros((1, *target.shape), dtype=np.float32),
                        target,
                        scale=delta_scale,
                    ),
                    "original_start_text_memory_prefix_decoder": score_sample_distribution(
                        samples[variant_idx],
                        target,
                        scale=delta_scale,
                    ),
                },
            }
        )
    return window_scores, summarize_method_scores(window_scores, baseline="persistence")


def run_start_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    bridge_report = _load_json(args.bridge_report)
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
    query_rows = select_heldout_query_rows(
        bridge_report,
        role=str(args.query_role),
        max_windows=int(args.max_eval_windows),
    )
    query_indices = np.asarray(
        [int(row["window_index"]) for row in query_rows],
        dtype=np.int64,
    )
    variant_rows = build_start_variant_rows(
        query_window_indices=query_indices,
        train_indices=train_indices,
        start_state=history_level[:, -1, :],
    )
    query_embedding_index = {
        int(row["window_index"]): int(row["embedding_index"]) for row in query_rows
    }
    variant_query_indices = np.asarray(
        [int(row["query_window_index"]) for row in variant_rows],
        dtype=np.int64,
    )
    variant_start_indices = np.asarray(
        [int(row["start_window_index"]) for row in variant_rows],
        dtype=np.int64,
    )
    text_memory = np.asarray(
        [
            bridge_arrays["condition_vectors"][query_embedding_index[int(query_idx)]]
            for query_idx in variant_query_indices
        ],
        dtype=np.float32,
    )
    requested_start = history_level[variant_start_indices, -1, :]
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
    endpoint_summary = endpoint_alignment_summary(
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
    input_memory_cosine_by_variant = _cosine_by_variant(
        decoded_memory,
        text_memory,
        variant_rows,
    )
    train_delta = future_delta[train_indices]
    delta_scale = build_delta_scale(train_delta, floor=float(args.score_scale_floor))
    samples = np.empty((0,), dtype=np.float32)
    rollout_sensitivity: dict[str, Any] | None = None
    original_window_scores: list[dict[str, Any]] = []
    original_rollout_summary: dict[str, Any] | None = None
    if bool(args.run_rollout):
        samples = sample_prefix_generator_deltas(
            model,
            indices=np.arange(len(variant_rows), dtype=np.int64),
            history_level=decoded_prefix["history_level"],
            history_norm=decoded_prefix["history_norm"],
            center=decoded_prefix["center"],
            scale=decoded_prefix["scale"],
            drift_feature=decoded_prefix["drift_feature"],
            history_raw=history_raw[variant_start_indices],
            specs=specs,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        rollout_sensitivity = rollout_sensitivity_summary(
            samples=samples,
            variant_rows=variant_rows,
            scale=delta_scale,
        )
        original_window_scores, original_rollout_summary = _score_original_rollouts(
            query_rows=query_rows,
            variant_rows=variant_rows,
            samples=samples,
            future_delta=future_delta,
            delta_scale=delta_scale,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "scope_note": (
            "Cached start-state sensitivity gate. Same text-predicted generator "
            "memory is held fixed while explicit start states are changed. No OpenAI "
            "API calls are made."
        ),
        "bridge_report": str(args.bridge_report),
        "bridge_arrays": str(args.bridge_arrays),
        "selected_window_count": int(selected_windows.size),
        "train_window_count": int(train_indices.size),
        "test_window_count": int(test_indices.size),
        "query_role": str(args.query_role),
        "query_window_count": int(query_indices.size),
        "variant_count": int(len(variant_rows)),
        "decoder": {
            "loss_first": float(decoder_result["loss_first"]),
            "loss_last": float(decoder_result["loss_last"]),
            "train_mse": float(decoder_result["train_mse"]),
            "test_mse": float(decoder_result["test_mse"]),
        },
        "endpoint_alignment": endpoint_summary,
        "input_memory_cosine_by_variant": input_memory_cosine_by_variant,
        "rollout_sensitivity": rollout_sensitivity,
        "original_rollout_summary": original_rollout_summary,
        "variant_rows": variant_rows,
        "artifact_paths": {
            "report": str(output_dir / "prefix_latent_start_sensitivity_report.json"),
            "arrays": str(output_dir / "prefix_latent_start_sensitivity_arrays.npz"),
        },
    }
    np.savez_compressed(
        output_dir / "prefix_latent_start_sensitivity_arrays.npz",
        selected_window_indices=selected_windows.astype(np.int64),
        train_indices=train_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        variant_query_indices=variant_query_indices.astype(np.int64),
        variant_start_indices=variant_start_indices.astype(np.int64),
        requested_start=requested_start.astype(np.float32),
        decoded_memory=decoded_memory.astype(np.float32),
        text_memory=text_memory.astype(np.float32),
        delta_scale=delta_scale.astype(np.float32),
        samples=samples.astype(np.float32),
    )
    _write_json(output_dir / "prefix_latent_start_sensitivity_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--query-role", default="anchor")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-rollout", action="store_true")
    parser.add_argument("--max-eval-windows", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
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
    report = run_start_sensitivity(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "endpoint_alignment": report["endpoint_alignment"],
                "input_memory_cosine_by_variant": report["input_memory_cosine_by_variant"],
                "rollout_sensitivity": report["rollout_sensitivity"],
                "original_rollout_summary": report["original_rollout_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
