#!/usr/bin/env python
"""Scenario-level evaluation for narrative-conditioned generator outputs.

The bridge evaluation answers whether language maps into the generator memory
space. This script asks the next product question: when that language condition
retrieves historical analogues and drives the frozen generator, are the resulting
30-day scenario paths closer to realized held-out futures than simple baselines?
"""

from __future__ import annotations

import argparse
import json
import sys
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
    _reconstruct_states,
    sample_normal_generator_for_retrieved_analogues,
    sample_with_memory_condition,
    sample_with_memory_residual_condition,
)


DEFAULT_CHECKPOINT = (
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
    "best_model.pt"
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finite_float(value: float | np.floating[Any]) -> float | None:
    raw = float(value)
    return raw if np.isfinite(raw) else None


def _round(value: float | np.floating[Any]) -> float | None:
    raw = _finite_float(value)
    return None if raw is None else round(raw, 12)


def load_bridge_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bridge_report_arrays_path(
    bridge_report: dict[str, Any],
    *,
    explicit_path: str | Path | None,
) -> Path:
    """Return the bridge arrays path used for direct-memory ablations."""

    if explicit_path:
        return Path(explicit_path)
    artifact_paths = bridge_report.get("artifact_paths", {})
    if isinstance(artifact_paths, dict) and artifact_paths.get("arrays"):
        return Path(str(artifact_paths["arrays"]))
    raise ValueError(
        "bridge arrays path is required for direct-memory evaluation; pass "
        "--bridge-arrays or include artifact_paths.arrays in the bridge report"
    )


def load_bridge_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load bridge-evaluation arrays needed for direct memory injection."""

    with np.load(path) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    required = {"condition_vectors", "memory_targets"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"bridge arrays missing required keys: {missing}")
    return arrays


def direct_memory_condition_for_query(
    query_row: dict[str, Any],
    condition_vectors: np.ndarray,
) -> np.ndarray:
    """Return the text-predicted condition-memory vector for a held-out query."""

    if "embedding_index" not in query_row:
        raise ValueError("held-out query row does not contain embedding_index")
    embedding_index = int(query_row["embedding_index"])
    vectors = np.asarray(condition_vectors, dtype=np.float32)
    if embedding_index < 0 or embedding_index >= vectors.shape[0]:
        raise IndexError(
            f"embedding_index {embedding_index} outside condition_vectors with "
            f"{vectors.shape[0]} rows"
        )
    return vectors[embedding_index].astype(np.float32, copy=True)


def true_memory_condition_for_window(
    window_index: int,
    memory_targets: np.ndarray,
) -> np.ndarray:
    """Return the frozen generator memory target for an original bridge window."""

    targets = np.asarray(memory_targets, dtype=np.float32)
    idx = int(window_index)
    if idx < 0 or idx >= targets.shape[0]:
        raise IndexError(
            f"window_index {idx} outside memory_targets with {targets.shape[0]} rows"
        )
    return targets[idx].astype(np.float32, copy=True)


def parse_memory_residual_alphas(raw: str) -> list[float]:
    """Parse comma-separated residual strengths for prompt-conditioning sweeps."""

    values: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if value < 0.0 or value > 1.0:
            raise ValueError("memory residual alphas must be in [0, 1]")
        values.append(value)
    if not values:
        raise ValueError("at least one memory residual alpha is required")
    return values


def memory_residual_method_name(prefix: str, alpha: float) -> str:
    """Build stable report keys for alpha-grid residual methods."""

    return f"{prefix}_a{int(round(float(alpha) * 100)):03d}"


def select_heldout_query_rows(
    bridge_report: dict[str, Any],
    *,
    role: str = "anchor",
    max_windows: int = 0,
) -> list[dict[str, Any]]:
    """Select one held-out query row per window from a bridge-eval report."""

    rows = bridge_report.get("evaluation", {}).get("heldout_examples", [])
    if not isinstance(rows, list):
        raise ValueError("bridge report does not contain evaluation.heldout_examples")
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if str(row.get("role", "")) != str(role):
            continue
        window_index = int(row["window_index"])
        if window_index in seen:
            continue
        seen.add(window_index)
        selected.append(row)
        if int(max_windows) > 0 and len(selected) >= int(max_windows):
            break
    if not selected:
        raise ValueError(f"no held-out rows found for role={role!r}")
    return selected


def bridge_local_to_block_indices(bridge_report: dict[str, Any], n_windows: int) -> np.ndarray:
    """Map bridge-local row indices back to original validation-block indices."""

    raw = bridge_report.get("window_indices")
    if isinstance(raw, list) and len(raw) >= int(n_windows):
        return np.asarray([int(idx) for idx in raw[: int(n_windows)]], dtype=np.int64)
    metadata = bridge_report.get("window_metadata", [])
    if isinstance(metadata, list) and len(metadata) >= int(n_windows):
        rows: list[int] = []
        for idx, row in enumerate(metadata[: int(n_windows)]):
            if isinstance(row, dict) and "window_index" in row:
                rows.append(int(row["window_index"]))
            else:
                rows.append(idx)
        return np.asarray(rows, dtype=np.int64)
    return np.arange(int(n_windows), dtype=np.int64)


def future_delta_paths(history_raw: np.ndarray, future_raw: np.ndarray) -> np.ndarray:
    """Convert future state paths to deltas from the last history state."""

    history = np.asarray(history_raw, dtype=np.float32)
    future = np.asarray(future_raw, dtype=np.float32)
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history_raw and future_raw must both have shape [N,T,C]")
    if history.shape[0] != future.shape[0] or history.shape[2] != future.shape[2]:
        raise ValueError("history_raw and future_raw dimensions are inconsistent")
    return future - history[:, -1:, :]


def build_delta_scale(train_delta: np.ndarray, *, floor: float = 1e-3) -> np.ndarray:
    """Build per-horizon/per-channel scale from train future deltas."""

    delta = np.asarray(train_delta, dtype=np.float32)
    if delta.ndim != 3:
        raise ValueError("train_delta must have shape [N,T,C]")
    scale = np.nanstd(delta, axis=0).astype(np.float32)
    floor_value = float(floor)
    if floor_value <= 0.0:
        raise ValueError("floor must be positive")
    return np.maximum(scale, floor_value).astype(np.float32)


def _pairwise_l1_term(samples: np.ndarray) -> float:
    if samples.shape[0] <= 1:
        return 0.0
    return float(np.mean(np.abs(samples[:, None, :, :] - samples[None, :, :, :])))


def _energy_score(samples_flat: np.ndarray, target_flat: np.ndarray) -> float:
    term_1 = float(np.linalg.norm(samples_flat - target_flat[None, :], axis=1).mean())
    if samples_flat.shape[0] <= 1:
        term_2 = 0.0
    else:
        distances = np.linalg.norm(samples_flat[:, None, :] - samples_flat[None, :, :], axis=-1)
        term_2 = 0.5 * float(distances.mean())
    return (term_1 - term_2) / np.sqrt(float(target_flat.size))


def score_sample_distribution(
    samples: np.ndarray,
    target: np.ndarray,
    *,
    scale: np.ndarray,
) -> dict[str, Any]:
    """Score a sample distribution against one realized future-delta path."""

    sample_arr = np.asarray(samples, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    scale_arr = np.asarray(scale, dtype=np.float32)
    if sample_arr.ndim != 3:
        raise ValueError("samples must have shape [S,T,C]")
    if target_arr.shape != sample_arr.shape[1:]:
        raise ValueError("target shape must match sample path shape [T,C]")
    if scale_arr.shape != target_arr.shape:
        raise ValueError("scale shape must match target shape")
    safe_scale = np.maximum(scale_arr, 1e-8)
    sample_z = sample_arr / safe_scale[None, :, :]
    target_z = target_arr / safe_scale
    mean_path = sample_z.mean(axis=0)
    abs_error = np.abs(mean_path - target_z)
    sq_error = (mean_path - target_z) ** 2
    term_1 = float(np.mean(np.abs(sample_z - target_z[None, :, :])))
    crps = term_1 - 0.5 * _pairwise_l1_term(sample_z)
    coverage_80: float | None
    if sample_z.shape[0] >= 2:
        lo = np.quantile(sample_z, 0.10, axis=0)
        hi = np.quantile(sample_z, 0.90, axis=0)
        coverage_80 = float(np.mean((target_z >= lo) & (target_z <= hi)))
    else:
        coverage_80 = None
    return {
        "sample_count": int(sample_z.shape[0]),
        "mean_path_mae_z": _round(float(abs_error.mean())),
        "mean_path_rmse_z": _round(float(np.sqrt(sq_error.mean()))),
        "terminal_mae_z": _round(float(np.abs(mean_path[-1] - target_z[-1]).mean())),
        "ensemble_crps_z": _round(crps),
        "energy_score_z": _round(_energy_score(sample_z.reshape(sample_z.shape[0], -1), target_z.reshape(-1))),
        "coverage_80": _round(coverage_80) if coverage_80 is not None else None,
    }


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        float(row[metric])
        for row in rows
        if row.get(metric) is not None and np.isfinite(float(row[metric]))
    ]
    if not values:
        return None
    return float(np.mean(values))


def summarize_method_scores(
    window_scores: list[dict[str, Any]],
    *,
    baseline: str = "persistence",
) -> dict[str, Any]:
    """Aggregate per-window method metrics and improvement vs a baseline."""

    methods = sorted({name for row in window_scores for name in row.get("methods", {})})
    metrics = [
        "mean_path_mae_z",
        "mean_path_rmse_z",
        "terminal_mae_z",
        "ensemble_crps_z",
        "energy_score_z",
        "coverage_80",
    ]
    summary: dict[str, Any] = {}
    baseline_rows = [
        row["methods"][baseline]
        for row in window_scores
        if baseline in row.get("methods", {})
    ]
    for method in methods:
        method_rows = [
            row["methods"][method]
            for row in window_scores
            if method in row.get("methods", {})
        ]
        block: dict[str, Any] = {"window_count": len(method_rows)}
        for metric in metrics:
            mean_value = _mean_metric(method_rows, metric)
            if mean_value is None:
                continue
            block[f"{metric}_mean"] = _round(mean_value)
            if metric != "coverage_80" and baseline_rows and method != baseline:
                baseline_value = _mean_metric(baseline_rows, metric)
                if baseline_value and baseline_value > 0.0:
                    block[f"{metric}_improvement_vs_{baseline}"] = _round(
                        (baseline_value - mean_value) / baseline_value
                    )
        summary[method] = block
    return summary


def _future_raw_from_block(block: Any, n_windows: int, n_cells: int) -> np.ndarray:
    future = np.asarray(block.future_state[:n_windows, :, :n_cells], dtype=np.float32)
    if future.ndim != 3:
        raise ValueError("block.future_state must have shape [N,T,C]")
    return future


def _actual_future_replay_samples(
    future_delta: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    return np.asarray(future_delta[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


def _median_baseline_samples(train_delta: np.ndarray) -> np.ndarray:
    return np.nanmedian(train_delta, axis=0, keepdims=True).astype(np.float32)


def _zero_baseline_samples(target_delta: np.ndarray) -> np.ndarray:
    return np.zeros((1, *target_delta.shape), dtype=np.float32)


def _states_to_deltas(states: np.ndarray, current_states: np.ndarray) -> np.ndarray:
    state_arr = np.asarray(states, dtype=np.float32)
    current = np.asarray(current_states, dtype=np.float32)
    if state_arr.ndim != 4:
        raise ValueError("states must have shape [K,S,T,C]")
    if current.shape != (state_arr.shape[0], state_arr.shape[-1]):
        raise ValueError("current_states must have shape [K,C]")
    return (state_arr - current[:, None, None, :]).reshape(-1, state_arr.shape[2], state_arr.shape[3])


def _sample_direct_memory_deltas(
    model: Any,
    condition_memory: np.ndarray,
    *,
    block_index: int,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    specs: list[Any],
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: torch.device,
) -> np.ndarray:
    """Sample held-out history paths while directly injecting condition memory."""

    idx = int(block_index)
    sampled_increments = sample_with_memory_condition(
        model,
        np.asarray(condition_memory, dtype=np.float32),
        history_level[idx],
        history_norm[idx],
        center[idx],
        scale[idx],
        drift_feature[idx],
        n_samples=int(n_samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        temperature=float(temperature),
        device=device,
    )
    generated_states = _reconstruct_states(
        history_raw[[idx], -1, :],
        sampled_increments,
        specs,
    )
    return _states_to_deltas(generated_states, history_raw[[idx], -1, :])


def _sample_memory_residual_deltas(
    model: Any,
    condition_memory: np.ndarray,
    base_memory: np.ndarray,
    *,
    block_indices: list[int],
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    specs: list[Any],
    alpha: float,
    n_samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: torch.device,
) -> np.ndarray:
    """Sample paths with dynamic market memory plus a text-memory residual."""

    indices = [int(idx) for idx in block_indices]
    sampled_increments = sample_with_memory_residual_condition(
        model,
        np.asarray(condition_memory, dtype=np.float32),
        np.asarray(base_memory, dtype=np.float32),
        history_level[indices],
        history_norm[indices],
        center[indices],
        scale[indices],
        drift_feature[indices],
        alpha=float(alpha),
        n_samples=int(n_samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        temperature=float(temperature),
        device=device,
    )
    generated_states = _reconstruct_states(
        history_raw[indices, -1, :],
        sampled_increments,
        specs,
    )
    return _states_to_deltas(generated_states, history_raw[indices, -1, :])


@torch.no_grad()
def run_scenario_level_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    bridge_report = load_bridge_report(args.bridge_report)
    query_rows = select_heldout_query_rows(
        bridge_report,
        role=args.query_role,
        max_windows=int(args.max_windows_eval),
    )
    bridge_arrays: dict[str, np.ndarray] | None = None
    bridge_arrays_source: str | None = None
    residual_alphas = (
        parse_memory_residual_alphas(args.memory_residual_alphas)
        if bool(args.include_memory_residual_generator)
        else []
    )
    needs_bridge_arrays = bool(args.include_direct_memory_generator) or bool(
        args.include_memory_residual_generator
    )
    if needs_bridge_arrays:
        bridge_arrays_path_value = bridge_report_arrays_path(
            bridge_report,
            explicit_path=args.bridge_arrays,
        )
        bridge_arrays = load_bridge_arrays(bridge_arrays_path_value)
        bridge_arrays_source = str(bridge_arrays_path_value)
    all_window_count = max(
        int(max(row["window_index"] for row in query_rows)) + 1,
        int(max(idx for row in query_rows for idx in [item["window_index"] for item in row["top_train_pool"]])) + 1,
        int(max(bridge_report.get("split", {}).get("train_indices", [0]))) + 1,
    )
    local_to_block = bridge_local_to_block_indices(bridge_report, n_windows=all_window_count)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    (
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        history_raw,
        specs,
        block,
    ) = build_val_block(args, payload)
    n_needed = int(np.max(local_to_block[:all_window_count])) + 1
    if history_raw.shape[0] < n_needed:
        raise ValueError(f"rebuilt validation block has {history_raw.shape[0]} windows, need {n_needed}")
    n_cells = int(history_raw.shape[-1])
    future_raw = _future_raw_from_block(block, int(history_raw.shape[0]), n_cells)
    future_delta = future_delta_paths(history_raw, future_raw)
    train_indices = [
        int(idx)
        for idx in bridge_report.get("split", {}).get("train_indices", [])
        if int(idx) < local_to_block.shape[0] and int(local_to_block[int(idx)]) < future_delta.shape[0]
    ]
    if not train_indices:
        raise ValueError("bridge report split has no train_indices")
    train_block_indices = local_to_block[np.asarray(train_indices, dtype=np.int64)]
    train_delta = future_delta[train_block_indices]
    delta_scale = build_delta_scale(train_delta, floor=float(args.score_scale_floor))

    window_scores: list[dict[str, Any]] = []
    generated_chunks: dict[str, np.ndarray] = {}
    for row_no, query in enumerate(query_rows):
        window_index = int(query["window_index"])
        target_block_index = int(local_to_block[window_index])
        target = future_delta[target_block_index]
        top_train_items = query.get("top_train_pool", [])[: int(args.top_k)]
        top_train_local = [int(item["window_index"]) for item in top_train_items]
        top_train = [int(local_to_block[idx]) for idx in top_train_local]
        analogue_rows = [
            {"index": idx, "cosine": float(item.get("cosine", 0.0))}
            for idx, item in zip(top_train, top_train_items)
        ]
        sampled = sample_normal_generator_for_retrieved_analogues(
            model,
            analogue_rows,
            history_level,
            history_norm,
            center,
            scale,
            drift_feature,
            n_samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        retrieved_indices = sampled["indices"]
        generated_states = _reconstruct_states(
            history_raw[retrieved_indices, -1, :],
            sampled["increments"],
            specs,
        )
        narrative_samples = _states_to_deltas(
            generated_states,
            history_raw[retrieved_indices, -1, :],
        )
        replay_samples = _actual_future_replay_samples(future_delta, retrieved_indices)
        methods = {
            "persistence": score_sample_distribution(
                _zero_baseline_samples(target),
                target,
                scale=delta_scale,
            ),
            "train_median_delta": score_sample_distribution(
                _median_baseline_samples(train_delta),
                target,
                scale=delta_scale,
            ),
            "historical_replay_topk": score_sample_distribution(
                replay_samples,
                target,
                scale=delta_scale,
            ),
            "narrative_generator_topk": score_sample_distribution(
                narrative_samples,
                target,
                scale=delta_scale,
            ),
        }
        if bridge_arrays is not None:
            direct_condition = direct_memory_condition_for_query(
                query,
                bridge_arrays["condition_vectors"],
            )
            direct_samples = _sample_direct_memory_deltas(
                model,
                direct_condition,
                block_index=target_block_index,
                history_level=history_level,
                history_norm=history_norm,
                center=center,
                scale=scale,
                drift_feature=drift_feature,
                history_raw=history_raw,
                specs=specs,
                n_samples=int(args.samples),
                n_steps=int(args.n_steps),
                chunk_size=int(args.chunk_size),
                temperature=float(args.temperature),
                device=device,
            )
            methods["narrative_direct_memory"] = score_sample_distribution(
                direct_samples,
                target,
                scale=delta_scale,
            )
            generated_chunks[f"direct_memory_{window_index}"] = direct_samples.astype(
                np.float32
            )
            if residual_alphas:
                base_condition = true_memory_condition_for_window(
                    window_index,
                    bridge_arrays["memory_targets"],
                )
                for alpha in residual_alphas:
                    method_name = memory_residual_method_name(
                        "narrative_residual_memory",
                        alpha,
                    )
                    residual_samples = _sample_memory_residual_deltas(
                        model,
                        direct_condition,
                        base_condition,
                        block_indices=[target_block_index],
                        history_level=history_level,
                        history_norm=history_norm,
                        center=center,
                        scale=scale,
                        drift_feature=drift_feature,
                        history_raw=history_raw,
                        specs=specs,
                        alpha=float(alpha),
                        n_samples=int(args.samples),
                        n_steps=int(args.n_steps),
                        chunk_size=int(args.chunk_size),
                        temperature=float(args.temperature),
                        device=device,
                    )
                    methods[method_name] = score_sample_distribution(
                        residual_samples,
                        target,
                        scale=delta_scale,
                    )
                    generated_chunks[
                        f"{method_name}_{window_index}"
                    ] = residual_samples.astype(np.float32)
                if bool(args.include_memory_residual_topk_generator):
                    top_base_conditions = np.asarray(
                        bridge_arrays["memory_targets"][np.asarray(top_train_local)],
                        dtype=np.float32,
                    )
                    top_text_conditions = np.repeat(
                        direct_condition[None],
                        len(top_train_local),
                        axis=0,
                    )
                    for alpha in residual_alphas:
                        method_name = memory_residual_method_name(
                            "narrative_residual_topk",
                            alpha,
                        )
                        topk_residual_samples = _sample_memory_residual_deltas(
                            model,
                            top_text_conditions,
                            top_base_conditions,
                            block_indices=top_train,
                            history_level=history_level,
                            history_norm=history_norm,
                            center=center,
                            scale=scale,
                            drift_feature=drift_feature,
                            history_raw=history_raw,
                            specs=specs,
                            alpha=float(alpha),
                            n_samples=int(args.samples),
                            n_steps=int(args.n_steps),
                            chunk_size=int(args.chunk_size),
                            temperature=float(args.temperature),
                            device=device,
                        )
                        methods[method_name] = score_sample_distribution(
                            topk_residual_samples,
                            target,
                            scale=delta_scale,
                        )
                        generated_chunks[
                            f"{method_name}_{window_index}"
                        ] = topk_residual_samples.astype(np.float32)
            if bool(args.include_oracle_generator):
                true_condition = true_memory_condition_for_window(
                    window_index,
                    bridge_arrays["memory_targets"],
                )
                oracle_direct_samples = _sample_direct_memory_deltas(
                    model,
                    true_condition,
                    block_index=target_block_index,
                    history_level=history_level,
                    history_norm=history_norm,
                    center=center,
                    scale=scale,
                    drift_feature=drift_feature,
                    history_raw=history_raw,
                    specs=specs,
                    n_samples=int(args.samples),
                    n_steps=int(args.n_steps),
                    chunk_size=int(args.chunk_size),
                    temperature=float(args.temperature),
                    device=device,
                )
                methods["oracle_direct_memory_true_history"] = score_sample_distribution(
                    oracle_direct_samples,
                    target,
                    scale=delta_scale,
                )
                generated_chunks[
                    f"oracle_direct_memory_{window_index}"
                ] = oracle_direct_samples.astype(np.float32)
        if bool(args.include_oracle_generator):
            oracle_sampled = sample_normal_generator_for_retrieved_analogues(
                model,
                [{"index": target_block_index, "cosine": 1.0}],
                history_level,
                history_norm,
                center,
                scale,
                drift_feature,
                n_samples=int(args.samples),
                n_steps=int(args.n_steps),
                chunk_size=int(args.chunk_size),
                temperature=float(args.temperature),
                device=device,
            )
            oracle_states = _reconstruct_states(
                history_raw[[target_block_index], -1, :],
                oracle_sampled["increments"],
                specs,
            )
            oracle_samples = _states_to_deltas(oracle_states, history_raw[[target_block_index], -1, :])
            methods["oracle_generator_true_history"] = score_sample_distribution(
                oracle_samples,
                target,
                scale=delta_scale,
            )
            generated_chunks[f"oracle_{window_index}"] = oracle_samples.astype(np.float32)
        generated_chunks[f"narrative_{window_index}"] = narrative_samples.astype(np.float32)
        generated_chunks[f"replay_{window_index}"] = replay_samples.astype(np.float32)
        window_scores.append(
            {
                "row_no": row_no,
                "window_index": window_index,
                "block_window_index": target_block_index,
                "window_id": query.get("window_id", ""),
                "query_role": query.get("role", ""),
                "query_kind": query.get("kind", ""),
                "top_train_indices": retrieved_indices,
                "top_train_window_ids": [item.get("window_id", "") for item in query.get("top_train_pool", [])[: int(args.top_k)]],
                "top_train_cosines": [float(item.get("cosine", 0.0)) for item in query.get("top_train_pool", [])[: int(args.top_k)]],
                "methods": methods,
            }
        )
        print(f"  scored held-out scenario {row_no + 1}/{len(query_rows)}", flush=True)

    summary = summarize_method_scores(window_scores, baseline="persistence")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "scope_note": (
            "Scenario-level held-out evaluation. Scores standardized future "
            "delta paths, so methods are compared on scenario movement shape "
            "rather than absolute level matching. No OpenAI API calls are made."
        ),
        "bridge_report": str(args.bridge_report),
        "checkpoint": str(args.checkpoint),
        "query_role": str(args.query_role),
        "heldout_window_count": len(window_scores),
        "top_k": int(args.top_k),
        "samples": int(args.samples),
        "n_steps": int(args.n_steps),
        "seed": int(args.seed),
        "direct_memory": {
            "enabled": bool(args.include_direct_memory_generator),
            "bridge_arrays": bridge_arrays_source,
            "method": (
                "For narrative_direct_memory, the held-out window's own "
                "current history/normalization state is used, while the "
                "frozen generator memory state is replaced by the "
                "text-predicted bridge condition vector. When oracle output "
                "is enabled, oracle_direct_memory_true_history uses the true "
                "encoded memory target for the same window."
            ),
        },
        "memory_residual": {
            "enabled": bool(args.include_memory_residual_generator),
            "topk_enabled": bool(args.include_memory_residual_topk_generator),
            "alphas": residual_alphas,
            "method": (
                "For residual methods, the generator keeps re-encoding its "
                "evolving market prefix. Text conditioning enters only as "
                "alpha * (text_predicted_memory - base_history_memory). "
                "narrative_residual_memory uses the held-out window history; "
                "narrative_residual_topk applies the same text target as a "
                "residual over each retrieved analogue history."
            ),
        },
        "summary": summary,
        "window_scores": window_scores,
        "artifact_paths": {
            "report": str(output_dir / "scenario_level_eval_report.json"),
            "arrays": str(output_dir / "scenario_level_eval_arrays.npz"),
        },
    }
    array_payload = {
        "future_delta": future_delta.astype(np.float32),
        "delta_scale": delta_scale.astype(np.float32),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "train_block_indices": np.asarray(train_block_indices, dtype=np.int64),
        "local_to_block_indices": local_to_block.astype(np.int64),
        "evaluated_indices": np.asarray([row["window_index"] for row in window_scores], dtype=np.int64),
        "evaluated_block_indices": np.asarray([row["block_window_index"] for row in window_scores], dtype=np.int64),
    }
    array_payload.update(generated_chunks)
    np.savez_compressed(output_dir / "scenario_level_eval_arrays.npz", **array_payload)
    _write_json(output_dir / "scenario_level_eval_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bridge-arrays")
    parser.add_argument("--query-role", default="anchor")
    parser.add_argument("--max-windows-eval", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-direct-memory-generator", action="store_true")
    parser.add_argument("--include-memory-residual-generator", action="store_true")
    parser.add_argument("--include-memory-residual-topk-generator", action="store_true")
    parser.add_argument("--memory-residual-alphas", default="0.10,0.25,0.50")
    parser.add_argument("--include-oracle-generator", action="store_true")
    parser.add_argument("--score-scale-floor", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=776)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument("--eval_split", choices=["val", "train", "train_tail"], default="val")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=50)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    args = parser.parse_args()
    report = run_scenario_level_evaluation(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "heldout_window_count": report["heldout_window_count"],
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
