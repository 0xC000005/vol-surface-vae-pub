#!/usr/bin/env python
"""Generator-memory to prefix decoder for narrative prefix-latent research.

This script makes no OpenAI calls. It tests whether the frozen SNI generator's
own final memory is a more stable latent target than an arbitrary autoencoder
coordinate. The decoder maps:

    final memory + explicit starting state -> recent-prefix feature object

It can then replace true memory with cached text-predicted memory from the
existing bridge arrays.
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
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    compute_memory_targets,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    build_prefix_feature_matrix,
    reconstruct_prefix_from_features,
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


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True)
    return np.maximum(std, 1e-6).astype(np.float32)


def build_memory_start_input_matrix(
    memory: np.ndarray,
    start_state: np.ndarray,
    *,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Standardize and concatenate generator memory with explicit start state."""

    mem = np.asarray(memory, dtype=np.float32)
    start = np.asarray(start_state, dtype=np.float32)
    fit = np.asarray(fit_indices, dtype=np.int64)
    if mem.ndim != 2 or start.ndim != 2 or mem.shape[0] != start.shape[0]:
        raise ValueError("memory and start_state must be 2-D with matching rows")
    if fit.size == 0:
        raise ValueError("fit_indices must be non-empty")
    mem_mean = mem[fit].mean(axis=0, keepdims=True).astype(np.float32)
    mem_std = _safe_std(mem[fit])
    start_mean = start[fit].mean(axis=0, keepdims=True).astype(np.float32)
    start_std = _safe_std(start[fit])
    inputs = np.concatenate(
        [(mem - mem_mean) / mem_std, (start - start_mean) / start_std],
        axis=1,
    ).astype(np.float32)
    return inputs, {
        "memory_mean": mem_mean,
        "memory_std": mem_std,
        "start_mean": start_mean,
        "start_std": start_std,
    }


def apply_memory_start_stats(
    memory: np.ndarray,
    start_state: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    mem = np.asarray(memory, dtype=np.float32)
    start = np.asarray(start_state, dtype=np.float32)
    return np.concatenate(
        [
            (mem - stats["memory_mean"]) / stats["memory_std"],
            (start - stats["start_mean"]) / stats["start_std"],
        ],
        axis=1,
    ).astype(np.float32)


class MemoryStartPrefixDecoder(nn.Module):
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


def train_memory_start_prefix_decoder(
    inputs: np.ndarray,
    features: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Train a supervised decoder from memory+start inputs to prefix features."""

    x_arr = np.asarray(inputs, dtype=np.float32)
    y_arr = np.asarray(features, dtype=np.float32)
    train_idx = np.asarray(train_indices, dtype=np.int64)
    test_idx = np.asarray(test_indices, dtype=np.int64)
    if x_arr.ndim != 2 or y_arr.ndim != 2 or x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("inputs and features must be 2-D with matching rows")
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("train_indices and test_indices must be non-empty")
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    target_mean = y_arr[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    target_std = _safe_std(y_arr[train_idx])
    y_std = ((y_arr - target_mean) / target_std).astype(np.float32)
    dev = torch.device(device)
    model = MemoryStartPrefixDecoder(
        input_dim=int(x_arr.shape[1]),
        output_dim=int(y_arr.shape[1]),
        hidden_dim=int(hidden_dim),
    ).to(dev)
    x = torch.from_numpy(x_arr).to(dev)
    y = torch.from_numpy(y_std).to(dev)
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
        pred_std = model(x).detach().cpu().numpy().astype(np.float32)
    predicted = (pred_std * target_std + target_mean).astype(np.float32)
    return {
        "model": model,
        "target_mean": target_mean,
        "target_std": target_std,
        "predicted_features": predicted,
        "loss_first": loss_first,
        "loss_last": losses[-1] if losses else loss_first,
        "losses": losses,
        "train_mse": _mse(y_arr, predicted, train_idx),
        "test_mse": _mse(y_arr, predicted, test_idx),
    }


def _decode_features(
    decoder: nn.Module,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    inputs: np.ndarray,
    *,
    start_state: np.ndarray,
    layout: Any,
    device: torch.device,
) -> dict[str, np.ndarray]:
    decoder.eval()
    with torch.no_grad():
        pred_std = decoder(torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(device))
    features = pred_std.detach().cpu().numpy().astype(np.float32) * target_std + target_mean
    return reconstruct_prefix_from_features(features, start_state=start_state, layout=layout)


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


def _score_rollouts(
    *,
    rows: list[dict[str, Any]],
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    true_samples: np.ndarray,
    oracle_memory_samples: np.ndarray,
    text_memory_samples: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row_no, row in enumerate(rows):
        idx = int(row["window_index"])
        target = future_delta[idx]
        scored.append(
            {
                "window_index": idx,
                "window_id": str(row.get("window_id", "")),
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
                    "oracle_memory_prefix_decoder": score_sample_distribution(
                        oracle_memory_samples[row_no],
                        target,
                        scale=delta_scale,
                    ),
                    "text_memory_prefix_decoder": score_sample_distribution(
                        text_memory_samples[row_no],
                        target,
                        scale=delta_scale,
                    ),
                },
            }
        )
    return scored, summarize_method_scores(scored, baseline="persistence")


def run_memory_decoder(args: argparse.Namespace) -> dict[str, Any]:
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
    future_raw_all = _future_raw_from_block(block, int(all_history_raw.shape[0]), int(all_history_raw.shape[-1]))
    future_delta = future_delta_paths(all_history_raw, future_raw_all)[selected_windows]
    features, layout = build_prefix_feature_matrix(history_level, history_norm, center, scale, drift_feature)
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
    oracle_prefix = reconstruct_prefix_from_features(
        decoder_result["predicted_features"],
        start_state=history_level[:, -1, :],
        layout=layout,
    )
    oracle_decoded_memory = compute_memory_targets(
        model,
        oracle_prefix["history_level"],
        oracle_prefix["history_norm"],
        oracle_prefix["center"],
        oracle_prefix["scale"],
        oracle_prefix["drift_feature"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    oracle_memory_cosine_test = cosine_summary(
        oracle_decoded_memory[test_indices],
        true_memory_targets[test_indices],
    )
    query_rows = select_heldout_query_rows(
        bridge_report,
        role=str(args.query_role),
        max_windows=int(args.max_eval_windows),
    )
    row_indices = np.asarray([int(row["window_index"]) for row in query_rows], dtype=np.int64)
    text_memory = np.asarray(
        [bridge_arrays["condition_vectors"][int(row["embedding_index"])] for row in query_rows],
        dtype=np.float32,
    )
    text_inputs = apply_memory_start_stats(
        text_memory,
        history_level[row_indices, -1, :],
        input_stats,
    )
    text_prefix = _decode_features(
        decoder_result["model"],
        decoder_result["target_mean"],
        decoder_result["target_std"],
        text_inputs,
        start_state=history_level[row_indices, -1, :],
        layout=layout,
        device=device,
    )
    text_decoded_memory = compute_memory_targets(
        model,
        text_prefix["history_level"],
        text_prefix["history_norm"],
        text_prefix["center"],
        text_prefix["scale"],
        text_prefix["drift_feature"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    text_memory_cosine = cosine_summary(text_decoded_memory, true_memory_targets[row_indices])
    train_delta = future_delta[train_indices]
    delta_scale = build_delta_scale(train_delta, floor=float(args.score_scale_floor))
    rollout_summary: dict[str, Any] | None = None
    window_scores: list[dict[str, Any]] = []
    true_samples = np.empty((0,), dtype=np.float32)
    oracle_memory_samples = np.empty((0,), dtype=np.float32)
    text_memory_samples = np.empty((0,), dtype=np.float32)
    if bool(args.run_rollout):
        true_samples = sample_prefix_generator_deltas(
            model,
            indices=row_indices,
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
        oracle_memory_samples = sample_prefix_generator_deltas(
            model,
            indices=row_indices,
            history_level=oracle_prefix["history_level"],
            history_norm=oracle_prefix["history_norm"],
            center=oracle_prefix["center"],
            scale=oracle_prefix["scale"],
            drift_feature=oracle_prefix["drift_feature"],
            history_raw=history_raw,
            specs=specs,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        text_memory_samples = sample_prefix_generator_deltas(
            model,
            indices=np.arange(len(row_indices), dtype=np.int64),
            history_level=text_prefix["history_level"],
            history_norm=text_prefix["history_norm"],
            center=text_prefix["center"],
            scale=text_prefix["scale"],
            drift_feature=text_prefix["drift_feature"],
            history_raw=history_raw[row_indices],
            specs=specs,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        window_scores, rollout_summary = _score_rollouts(
            rows=query_rows,
            future_delta=future_delta,
            delta_scale=delta_scale,
            true_samples=true_samples,
            oracle_memory_samples=oracle_memory_samples,
            text_memory_samples=text_memory_samples,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "scope_note": (
            "Cached generator-memory prefix decoder. No OpenAI API calls are made. "
            "The decoder maps final SNI memory plus explicit start state to a recent-prefix object."
        ),
        "bridge_report": str(args.bridge_report),
        "bridge_arrays": str(args.bridge_arrays),
        "selected_window_count": int(selected_windows.size),
        "train_window_count": int(train_indices.size),
        "test_window_count": int(test_indices.size),
        "query_role": str(args.query_role),
        "rollout_eval_window_count": int(row_indices.size),
        "decoder": {
            "loss_first": float(decoder_result["loss_first"]),
            "loss_last": float(decoder_result["loss_last"]),
            "train_mse": float(decoder_result["train_mse"]),
            "test_mse": float(decoder_result["test_mse"]),
            "oracle_memory_cosine_test": oracle_memory_cosine_test,
            "text_memory_cosine": text_memory_cosine,
        },
        "rollout_summary": rollout_summary,
        "window_scores": window_scores,
        "artifact_paths": {
            "report": str(output_dir / "prefix_latent_memory_decoder_report.json"),
            "arrays": str(output_dir / "prefix_latent_memory_decoder_arrays.npz"),
        },
    }
    np.savez_compressed(
        output_dir / "prefix_latent_memory_decoder_arrays.npz",
        selected_window_indices=selected_windows.astype(np.int64),
        train_indices=train_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        row_indices=row_indices.astype(np.int64),
        true_memory_targets=true_memory_targets.astype(np.float32),
        oracle_decoded_memory=oracle_decoded_memory.astype(np.float32),
        text_memory=text_memory.astype(np.float32),
        text_decoded_memory=text_decoded_memory.astype(np.float32),
        true_samples=true_samples.astype(np.float32),
        oracle_memory_samples=oracle_memory_samples.astype(np.float32),
        text_memory_samples=text_memory_samples.astype(np.float32),
    )
    _write_json(output_dir / "prefix_latent_memory_decoder_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument(
        "--bridge-arrays",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "manifest_bridge_eval_openai_schema_v2_representative_220/"
            "bridge_eval_arrays.npz"
        ),
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--query-role", default="anchor")
    parser.add_argument("--seed", type=int, default=788)
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
    report = run_memory_decoder(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "decoder": report["decoder"],
                "rollout_summary": report["rollout_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
