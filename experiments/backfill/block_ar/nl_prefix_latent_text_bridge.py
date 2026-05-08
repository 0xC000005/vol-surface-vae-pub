#!/usr/bin/env python
"""Cached text-plus-start to prefix-latent bridge experiment.

This script makes no OpenAI calls. It reuses the existing representative
OpenAI-generated narrative embeddings and tests whether text plus an explicit
starting state can predict the oracle prefix latent from
`nl_prefix_latent_oracle_autoencoder.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    compute_memory_targets,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    build_prefix_feature_matrix,
    reconstruct_prefix_from_features,
    run_oracle_prefix_autoencoder,
    sample_prefix_generator_deltas,
    selected_bridge_window_indices,
    split_indices_from_bridge_report,
    train_prefix_autoencoder,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    build_delta_scale,
    future_delta_paths,
    score_sample_distribution,
    summarize_method_scores,
)
from experiments.backfill.block_ar.nl_text_conditioning import normalize_rows  # noqa: E402


DEFAULT_PIPELINE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_schema_v2_representative_220/narrative_pipeline_report.json"
)
DEFAULT_PIPELINE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_schema_v2_representative_220/narrative_pipeline_arrays.npz"
)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True)
    return np.maximum(std, 1e-6).astype(np.float32)


def build_text_start_input_matrix(
    text_embeddings: np.ndarray,
    start_state: np.ndarray,
    target_indices: np.ndarray,
    *,
    fit_window_indices: np.ndarray,
    input_mode: str = "text_start",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Concatenate normalized text embeddings with standardized start state."""

    text = normalize_rows(np.asarray(text_embeddings, dtype=np.float32))
    starts = np.asarray(start_state, dtype=np.float32)
    targets = np.asarray(target_indices, dtype=np.int64)
    if text.ndim != 2:
        raise ValueError("text_embeddings must be 2-D")
    if starts.ndim != 2:
        raise ValueError("start_state must be 2-D")
    if targets.shape != (text.shape[0],):
        raise ValueError("target_indices must have one row per text embedding")
    if np.any(targets < 0) or np.any(targets >= starts.shape[0]):
        raise ValueError("target_indices must point to valid start_state rows")
    fit = np.asarray(fit_window_indices, dtype=np.int64)
    if fit.size == 0:
        raise ValueError("fit_window_indices must be non-empty")
    start_mean = starts[fit].mean(axis=0, keepdims=True).astype(np.float32)
    start_std = _safe_std(starts[fit])
    standardized_start = ((starts[targets] - start_mean) / start_std).astype(np.float32)
    mode = str(input_mode)
    if mode == "text_start":
        inputs = np.concatenate([text, standardized_start], axis=1).astype(np.float32)
    elif mode == "text_only":
        inputs = text.astype(np.float32)
    elif mode == "start_only":
        inputs = standardized_start.astype(np.float32)
    else:
        raise ValueError("input_mode must be text_start, text_only, or start_only")
    return inputs, {"start_mean": start_mean, "start_std": start_std}


class TextStartPrefixBridge(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mse(a: np.ndarray, b: np.ndarray, indices: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    return float(np.mean((np.asarray(a)[idx] - np.asarray(b)[idx]) ** 2))


def train_text_start_prefix_bridge(
    inputs: np.ndarray,
    target_latents: np.ndarray,
    *,
    train_example_indices: np.ndarray,
    test_example_indices: np.ndarray,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Train a small supervised bridge from text+start to prefix latent."""

    x_arr = np.asarray(inputs, dtype=np.float32)
    y_arr = np.asarray(target_latents, dtype=np.float32)
    if x_arr.ndim != 2 or y_arr.ndim != 2 or x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("inputs and target_latents must be 2-D with matching rows")
    train_idx = np.asarray(train_example_indices, dtype=np.int64)
    test_idx = np.asarray(test_example_indices, dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("train and test example indices must be non-empty")
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    target_mean = y_arr[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    target_std = _safe_std(y_arr[train_idx])
    y_train_space = ((y_arr - target_mean) / target_std).astype(np.float32)
    model = TextStartPrefixBridge(
        input_dim=int(x_arr.shape[1]),
        output_dim=int(y_arr.shape[1]),
        hidden_dim=int(hidden_dim),
    ).to(dev)
    x = torch.from_numpy(x_arr).to(dev)
    y = torch.from_numpy(y_train_space).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    batch = max(1, min(int(batch_size), int(train_idx.size)))
    with torch.no_grad():
        loss_first = float(torch.mean((model(x[train_idx]) - y[train_idx]) ** 2).cpu())
    losses: list[float] = []
    model.train()
    for _step in range(int(steps)):
        choice = rng.choice(train_idx, size=batch, replace=train_idx.size < batch)
        pred = model(x[choice])
        loss = torch.mean((pred - y[choice]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        pred_all_std = model(x).detach().cpu().numpy().astype(np.float32)
    pred_all = (pred_all_std * target_std + target_mean).astype(np.float32)
    return {
        "model": model,
        "predicted_latents": pred_all,
        "target_mean": target_mean,
        "target_std": target_std,
        "loss_first": loss_first,
        "loss_last": losses[-1] if losses else loss_first,
        "losses": losses,
        "train_mse": _mse(y_arr, pred_all, train_idx),
        "test_mse": _mse(y_arr, pred_all, test_idx),
    }


def cosine_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    cos = np.sum(a * b, axis=1) / denom
    return {
        "mean": float(np.mean(cos)),
        "median": float(np.median(cos)),
        "min": float(np.min(cos)),
        "p10": float(np.quantile(cos, 0.1)),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _future_raw_from_block(block: Any, n_windows: int, n_cells: int) -> np.ndarray:
    future = np.asarray(block.future_state[:n_windows, :, :n_cells], dtype=np.float32)
    if future.ndim != 3:
        raise ValueError("block.future_state must have shape [N,T,C]")
    return future


def _supervised_example_indices(
    examples: list[dict[str, Any]],
    *,
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    eval_role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_set = set(int(idx) for idx in np.asarray(train_windows, dtype=np.int64))
    test_set = set(int(idx) for idx in np.asarray(test_windows, dtype=np.int64))
    train_rows: list[int] = []
    test_rows: list[int] = []
    rollout_rows: list[int] = []
    seen_rollout_windows: set[int] = set()
    target_indices: list[int] = []
    for row_no, example in enumerate(examples):
        target = example.get("target_index")
        if target is None:
            target_indices.append(-1)
            continue
        idx = int(target)
        target_indices.append(idx)
        if str(example.get("role", "")) == "negative":
            continue
        if idx in train_set:
            train_rows.append(row_no)
        elif idx in test_set:
            test_rows.append(row_no)
            if (
                str(example.get("role", "")) == str(eval_role)
                and idx not in seen_rollout_windows
            ):
                rollout_rows.append(row_no)
                seen_rollout_windows.add(idx)
    return (
        np.asarray(train_rows, dtype=np.int64),
        np.asarray(test_rows, dtype=np.int64),
        np.asarray(rollout_rows, dtype=np.int64),
        np.asarray(target_indices, dtype=np.int64),
    )


def _decode_latents_to_prefix(
    prefix_model: nn.Module,
    feature_stats: Any,
    latents: np.ndarray,
    *,
    start_state: np.ndarray,
    layout: Any,
    device: torch.device,
) -> dict[str, np.ndarray]:
    prefix_model.eval()
    with torch.no_grad():
        decoded_std = prefix_model.decoder(
            torch.from_numpy(np.asarray(latents, dtype=np.float32)).to(device)
        )
    decoded_features = feature_stats.inverse(decoded_std.detach().cpu().numpy())
    return reconstruct_prefix_from_features(
        decoded_features,
        start_state=start_state,
        layout=layout,
    )


def _score_rollout_rows(
    *,
    rollout_windows: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    predicted_samples: np.ndarray,
    true_samples: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_no, window_idx in enumerate(np.asarray(rollout_windows, dtype=np.int64)):
        target = future_delta[int(window_idx)]
        rows.append(
            {
                "window_index": int(window_idx),
                "methods": {
                    "persistence": score_sample_distribution(
                        np.zeros((1, *target.shape), dtype=np.float32),
                        target,
                        scale=delta_scale,
                    ),
                    "true_prefix_oracle_generator": score_sample_distribution(
                        true_samples[row_no],
                        target,
                        scale=delta_scale,
                    ),
                    "text_start_prefix_generator": score_sample_distribution(
                        predicted_samples[row_no],
                        target,
                        scale=delta_scale,
                    ),
                },
            }
        )
    return rows, summarize_method_scores(rows, baseline="persistence")


def run_text_start_prefix_bridge(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    pipeline_report = _load_json(args.pipeline_report)
    bridge_report = _load_json(args.bridge_report)
    examples = build_bridge_examples(pipeline_report)
    with np.load(args.pipeline_arrays) as payload:
        text_embeddings = payload["text_embeddings"].astype(np.float32)
    if text_embeddings.shape[0] != len(examples):
        raise ValueError("pipeline text_embeddings rows do not match rebuilt examples")
    selected_windows = selected_bridge_window_indices(bridge_report)
    train_windows, test_windows = split_indices_from_bridge_report(bridge_report)
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
    prefix_result = train_prefix_autoencoder(
        features,
        train_indices=train_windows,
        test_indices=test_windows,
        latent_dim=int(args.prefix_latent_dim),
        hidden_dim=int(args.prefix_hidden_dim),
        steps=int(args.prefix_steps),
        batch_size=int(args.prefix_batch_size),
        lr=float(args.prefix_lr),
        seed=int(args.seed),
        device=device,
    )
    prefix_latents = prefix_result["encoded_latents"]
    train_examples, test_examples, rollout_examples, target_indices = _supervised_example_indices(
        examples,
        train_windows=train_windows,
        test_windows=test_windows,
        eval_role=str(args.eval_role),
    )
    valid = target_indices >= 0
    inputs, input_stats = build_text_start_input_matrix(
        text_embeddings[valid],
        history_level[:, -1, :],
        target_indices[valid],
        fit_window_indices=train_windows,
        input_mode=str(args.input_mode),
    )
    valid_to_original = np.flatnonzero(valid)
    original_to_valid = {int(orig): pos for pos, orig in enumerate(valid_to_original)}
    train_valid = np.asarray(
        [original_to_valid[int(idx)] for idx in train_examples if int(idx) in original_to_valid],
        dtype=np.int64,
    )
    test_valid = np.asarray(
        [original_to_valid[int(idx)] for idx in test_examples if int(idx) in original_to_valid],
        dtype=np.int64,
    )
    target_latents = prefix_latents[target_indices[valid]]
    bridge_result = train_text_start_prefix_bridge(
        inputs,
        target_latents,
        train_example_indices=train_valid,
        test_example_indices=test_valid,
        hidden_dim=int(args.bridge_hidden_dim),
        steps=int(args.bridge_steps),
        batch_size=int(args.bridge_batch_size),
        lr=float(args.bridge_lr),
        seed=int(args.seed),
        device=device,
    )
    predicted_latents_valid = bridge_result["predicted_latents"]
    target_latents_valid = target_latents
    latent_cosine_test = cosine_summary(
        predicted_latents_valid[test_valid],
        target_latents_valid[test_valid],
    )
    rollout_valid = np.asarray(
        [original_to_valid[int(idx)] for idx in rollout_examples if int(idx) in original_to_valid],
        dtype=np.int64,
    )
    rollout_target_windows = target_indices[valid][rollout_valid]
    if int(args.max_eval_windows) > 0:
        rollout_valid = rollout_valid[: int(args.max_eval_windows)]
        rollout_target_windows = rollout_target_windows[: int(args.max_eval_windows)]
    predicted_prefix = _decode_latents_to_prefix(
        prefix_result["model"],
        prefix_result["feature_stats"],
        predicted_latents_valid[rollout_valid],
        start_state=history_level[rollout_target_windows, -1, :],
        layout=layout,
        device=device,
    )
    predicted_memory = compute_memory_targets(
        model,
        predicted_prefix["history_level"],
        predicted_prefix["history_norm"],
        predicted_prefix["center"],
        predicted_prefix["scale"],
        predicted_prefix["drift_feature"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    true_memory = compute_memory_targets(
        model,
        history_level[rollout_target_windows],
        history_norm[rollout_target_windows],
        center[rollout_target_windows],
        scale[rollout_target_windows],
        drift_feature[rollout_target_windows],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    memory_cosine_rollout = cosine_summary(predicted_memory, true_memory)
    rollout_summary: dict[str, Any] | None = None
    window_scores: list[dict[str, Any]] = []
    predicted_samples = np.empty((0,), dtype=np.float32)
    true_samples = np.empty((0,), dtype=np.float32)
    delta_scale = build_delta_scale(future_delta[train_windows], floor=float(args.score_scale_floor))
    if bool(args.run_rollout):
        predicted_samples = sample_prefix_generator_deltas(
            model,
            indices=np.arange(len(rollout_target_windows)),
            history_level=predicted_prefix["history_level"],
            history_norm=predicted_prefix["history_norm"],
            center=predicted_prefix["center"],
            scale=predicted_prefix["scale"],
            drift_feature=predicted_prefix["drift_feature"],
            history_raw=history_raw[rollout_target_windows],
            specs=specs,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        true_samples = sample_prefix_generator_deltas(
            model,
            indices=rollout_target_windows,
            history_level=history_level,
            history_norm=history_norm,
            center=center,
            scale=scale,
            drift_feature=drift_feature,
            history_raw=history_raw,
            specs=specs,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        window_scores, rollout_summary = _score_rollout_rows(
            rollout_windows=rollout_target_windows,
            future_delta=future_delta,
            delta_scale=delta_scale,
            predicted_samples=predicted_samples,
            true_samples=true_samples,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "scope_note": (
            "Cached text-plus-start to prefix-latent bridge. No OpenAI API calls "
            "are made; text embeddings are loaded from the representative pipeline artifact."
        ),
        "pipeline_report": str(args.pipeline_report),
        "pipeline_arrays": str(args.pipeline_arrays),
        "bridge_report": str(args.bridge_report),
        "selected_window_count": int(selected_windows.size),
        "train_window_count": int(train_windows.size),
        "test_window_count": int(test_windows.size),
        "train_example_count": int(train_valid.size),
        "test_example_count": int(test_valid.size),
        "rollout_eval_window_count": int(rollout_target_windows.size),
        "prefix_autoencoder": {
            "latent_dim": int(args.prefix_latent_dim),
            "train_mse": float(prefix_result["train_mse"]),
            "test_mse": float(prefix_result["test_mse"]),
            "loss_first": float(prefix_result["loss_first"]),
            "loss_last": float(prefix_result["loss_last"]),
        },
        "text_start_bridge": {
            "input_mode": str(args.input_mode),
            "loss_first": float(bridge_result["loss_first"]),
            "loss_last": float(bridge_result["loss_last"]),
            "train_mse": float(bridge_result["train_mse"]),
            "test_mse": float(bridge_result["test_mse"]),
            "latent_cosine_test": latent_cosine_test,
            "memory_cosine_rollout": memory_cosine_rollout,
        },
        "rollout_summary": rollout_summary,
        "window_scores": window_scores,
        "artifact_paths": {
            "report": str(output_dir / "prefix_latent_text_bridge_report.json"),
            "arrays": str(output_dir / "prefix_latent_text_bridge_arrays.npz"),
        },
    }
    np.savez_compressed(
        output_dir / "prefix_latent_text_bridge_arrays.npz",
        selected_window_indices=selected_windows.astype(np.int64),
        train_windows=train_windows.astype(np.int64),
        test_windows=test_windows.astype(np.int64),
        train_example_indices=train_valid.astype(np.int64),
        test_example_indices=test_valid.astype(np.int64),
        rollout_target_windows=rollout_target_windows.astype(np.int64),
        text_start_inputs=inputs.astype(np.float32),
        target_latents=target_latents_valid.astype(np.float32),
        predicted_latents=predicted_latents_valid.astype(np.float32),
        predicted_memory=predicted_memory.astype(np.float32),
        true_memory=true_memory.astype(np.float32),
        predicted_prefix_samples=predicted_samples.astype(np.float32),
        true_prefix_samples=true_samples.astype(np.float32),
        input_start_mean=input_stats["start_mean"].astype(np.float32),
        input_start_std=input_stats["start_std"].astype(np.float32),
    )
    _write_json(output_dir / "prefix_latent_text_bridge_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--pipeline-arrays", default=DEFAULT_PIPELINE_ARRAYS)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-latent-dim", type=int, default=32)
    parser.add_argument("--prefix-hidden-dim", type=int, default=256)
    parser.add_argument("--prefix-steps", type=int, default=1000)
    parser.add_argument("--prefix-batch-size", type=int, default=64)
    parser.add_argument("--prefix-lr", type=float, default=1e-3)
    parser.add_argument("--bridge-hidden-dim", type=int, default=256)
    parser.add_argument("--bridge-steps", type=int, default=1200)
    parser.add_argument("--bridge-batch-size", type=int, default=128)
    parser.add_argument("--bridge-lr", type=float, default=1e-3)
    parser.add_argument(
        "--input-mode",
        choices=["text_start", "text_only", "start_only"],
        default="text_start",
    )
    parser.add_argument("--eval-role", default="anchor")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=787)
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
    report = run_text_start_prefix_bridge(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "train_example_count": report["train_example_count"],
                "test_example_count": report["test_example_count"],
                "text_start_bridge": report["text_start_bridge"],
                "rollout_summary": report["rollout_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
