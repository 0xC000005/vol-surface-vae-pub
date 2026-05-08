#!/usr/bin/env python
"""Oracle prefix-latent gate for narrative-conditioned scenario generation.

This script makes no OpenAI calls. It tests the first local prerequisite for a
true text-plus-start prompt-conditioned generator: whether a compact latent can
reconstruct enough of the recent joint39 prefix object for the frozen SNI
generator to preserve its encoded memory and rollout behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
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
    _reconstruct_states,
    compute_memory_targets,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    build_delta_scale,
    future_delta_paths,
    score_sample_distribution,
    summarize_method_scores,
)


DEFAULT_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/"
    "bridge_eval_report.json"
)


@dataclass(frozen=True)
class PrefixFeatureLayout:
    history_len: int
    n_cells: int
    level_size: int
    norm_size: int
    center_size: int
    log_scale_size: int
    drift_size: int

    @property
    def input_dim(self) -> int:
        return (
            self.level_size
            + self.norm_size
            + self.center_size
            + self.log_scale_size
            + self.drift_size
        )


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")


def build_prefix_feature_matrix(
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
) -> tuple[np.ndarray, PrefixFeatureLayout]:
    """Flatten the SNI prefix object while anchoring levels to final state."""

    level = np.asarray(history_level, dtype=np.float32)
    norm = np.asarray(history_norm, dtype=np.float32)
    ctr = np.asarray(center, dtype=np.float32)
    scl = np.asarray(scale, dtype=np.float32)
    drift = np.asarray(drift_feature, dtype=np.float32)
    if level.ndim != 3:
        raise ValueError("history_level must have shape [N,T,C]")
    n, t, c = level.shape
    _require_shape("history_norm", norm, (n, t, c))
    _require_shape("center", ctr, (n, c))
    _require_shape("scale", scl, (n, c))
    _require_shape("drift_feature", drift, (n, c))
    if np.any(~np.isfinite(scl)) or np.any(scl <= 0.0):
        raise ValueError("scale must be finite and positive")
    level_delta = level - level[:, -1:, :]
    log_scale = np.log(np.maximum(scl, 1e-8))
    feature = np.concatenate(
        [
            level_delta.reshape(n, -1),
            norm.reshape(n, -1),
            ctr,
            log_scale,
            drift,
        ],
        axis=1,
    ).astype(np.float32)
    layout = PrefixFeatureLayout(
        history_len=int(t),
        n_cells=int(c),
        level_size=int(t * c),
        norm_size=int(t * c),
        center_size=int(c),
        log_scale_size=int(c),
        drift_size=int(c),
    )
    return feature, layout


def reconstruct_prefix_from_features(
    features: np.ndarray,
    *,
    start_state: np.ndarray,
    layout: PrefixFeatureLayout,
) -> dict[str, np.ndarray]:
    """Invert `build_prefix_feature_matrix` and pin the final prefix level."""

    x = np.asarray(features, dtype=np.float32)
    start = np.asarray(start_state, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != layout.input_dim:
        raise ValueError(
            f"features must have shape [N,{layout.input_dim}], got {x.shape}"
        )
    if start.shape != (x.shape[0], layout.n_cells):
        raise ValueError(
            f"start_state must have shape [{x.shape[0]},{layout.n_cells}], got {start.shape}"
        )
    cursor = 0
    level_delta = x[:, cursor : cursor + layout.level_size].reshape(
        x.shape[0], layout.history_len, layout.n_cells
    )
    cursor += layout.level_size
    history_norm = x[:, cursor : cursor + layout.norm_size].reshape(
        x.shape[0], layout.history_len, layout.n_cells
    )
    cursor += layout.norm_size
    center = x[:, cursor : cursor + layout.center_size]
    cursor += layout.center_size
    log_scale = x[:, cursor : cursor + layout.log_scale_size]
    cursor += layout.log_scale_size
    drift = x[:, cursor : cursor + layout.drift_size]
    level_delta = level_delta.copy()
    level_delta[:, -1, :] = 0.0
    return {
        "history_level": (start[:, None, :] + level_delta).astype(np.float32),
        "history_norm": history_norm.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": np.exp(log_scale).astype(np.float32),
        "drift_feature": drift.astype(np.float32),
    }


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.std).astype(
            np.float32
        )

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.std + self.mean).astype(
            np.float32
        )


class PrefixAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def _fit_feature_stats(features: np.ndarray, train_indices: np.ndarray) -> FeatureStats:
    train = np.asarray(features, dtype=np.float32)[np.asarray(train_indices, dtype=np.int64)]
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-4).astype(np.float32)
    return FeatureStats(mean=mean, std=std)


def _mse(a: np.ndarray, b: np.ndarray, indices: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    return float(np.mean((np.asarray(a)[idx] - np.asarray(b)[idx]) ** 2))


def train_prefix_autoencoder(
    features: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Train a compact MLP autoencoder over standardized prefix features."""

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    feature_arr = np.asarray(features, dtype=np.float32)
    train_idx = np.asarray(train_indices, dtype=np.int64)
    test_idx = np.asarray(test_indices, dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("train_indices and test_indices must both be non-empty")
    stats = _fit_feature_stats(feature_arr, train_idx)
    standardized = stats.transform(feature_arr)
    dev = torch.device(device)
    model = PrefixAutoencoder(
        input_dim=int(feature_arr.shape[1]),
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
    ).to(dev)
    x = torch.from_numpy(standardized).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    batch = max(1, min(int(batch_size), int(train_idx.size)))
    with torch.no_grad():
        pred0, _z0 = model(x[train_idx])
        loss_first = float(torch.mean((pred0 - x[train_idx]) ** 2).detach().cpu())
    losses: list[float] = []
    model.train()
    for _step in range(int(steps)):
        choice = rng.choice(train_idx, size=batch, replace=train_idx.size < batch)
        xb = x[choice]
        pred, _z = model(xb)
        loss = torch.mean((pred - xb) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        reconstructed_std, latents = model(x)
    reconstructed = stats.inverse(reconstructed_std.detach().cpu().numpy())
    encoded = latents.detach().cpu().numpy().astype(np.float32)
    return {
        "model": model,
        "feature_stats": stats,
        "encoded_latents": encoded,
        "reconstructed_features": reconstructed.astype(np.float32),
        "loss_first": loss_first,
        "loss_last": losses[-1] if losses else loss_first,
        "losses": losses,
        "train_mse": _mse(feature_arr, reconstructed, train_idx),
        "test_mse": _mse(feature_arr, reconstructed, test_idx),
    }


def load_bridge_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_bridge_window_indices(bridge_report: dict[str, Any]) -> np.ndarray:
    raw = bridge_report.get("window_indices")
    if not isinstance(raw, list) or not raw:
        raise ValueError("bridge report must contain non-empty window_indices")
    return np.asarray([int(idx) for idx in raw], dtype=np.int64)


def split_indices_from_bridge_report(bridge_report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    split = bridge_report.get("split", {})
    train = np.asarray([int(idx) for idx in split.get("train_indices", [])], dtype=np.int64)
    test = np.asarray([int(idx) for idx in split.get("test_indices", [])], dtype=np.int64)
    if train.size == 0 or test.size == 0:
        raise ValueError("bridge report split must contain train_indices and test_indices")
    return train, test


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


def _future_raw_from_block(block: Any, n_windows: int, n_cells: int) -> np.ndarray:
    future = np.asarray(block.future_state[:n_windows, :, :n_cells], dtype=np.float32)
    if future.ndim != 3:
        raise ValueError("block.future_state must have shape [N,T,C]")
    return future


@torch.no_grad()
def sample_prefix_generator_deltas(
    model: Any,
    *,
    indices: np.ndarray,
    history_level: np.ndarray,
    history_norm: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    drift_feature: np.ndarray,
    history_raw: np.ndarray,
    specs: list[Any],
    samples: int,
    n_steps: int,
    chunk_size: int,
    temperature: float,
    device: torch.device,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    sampled_increment = model.sample_batched(
        torch.from_numpy(history_level[idx]).to(device),
        torch.from_numpy(history_norm[idx]).to(device),
        torch.from_numpy(center[idx]).to(device),
        torch.from_numpy(scale[idx]).to(device),
        drift_feature=torch.from_numpy(drift_feature[idx]).to(device),
        n_samples=int(samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
        temperature=float(temperature),
    )
    generated_states = _reconstruct_states(
        history_raw[idx, -1, :],
        sampled_increment.detach().cpu().numpy(),
        specs,
    )
    return (generated_states - history_raw[idx, None, None, -1, :]).astype(np.float32)


def _score_prefix_rollouts(
    *,
    test_indices: np.ndarray,
    future_delta: np.ndarray,
    delta_scale: np.ndarray,
    true_samples: np.ndarray,
    decoded_samples: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window_scores: list[dict[str, Any]] = []
    for row_no, idx in enumerate(np.asarray(test_indices, dtype=np.int64)):
        target = future_delta[int(idx)]
        methods = {
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
            "decoded_prefix_generator": score_sample_distribution(
                decoded_samples[row_no],
                target,
                scale=delta_scale,
            ),
        }
        window_scores.append({"window_index": int(idx), "methods": methods})
    return window_scores, summarize_method_scores(window_scores, baseline="persistence")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_oracle_prefix_autoencoder(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    bridge_report = load_bridge_report(args.bridge_report)
    selected_windows = selected_bridge_window_indices(bridge_report)
    train_indices, test_indices = split_indices_from_bridge_report(bridge_report)
    if int(args.max_eval_windows) > 0:
        eval_test_indices = test_indices[: int(args.max_eval_windows)]
    else:
        eval_test_indices = test_indices
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
    max_selected = int(np.max(selected_windows))
    if all_history_level.shape[0] <= max_selected:
        raise ValueError(
            f"rebuilt block has {all_history_level.shape[0]} windows, need {max_selected + 1}"
        )
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
    future_delta_all = future_delta_paths(all_history_raw, future_raw_all)
    future_delta = future_delta_all[selected_windows]
    feature_matrix, layout = build_prefix_feature_matrix(
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
    )
    train_result = train_prefix_autoencoder(
        feature_matrix,
        train_indices=train_indices,
        test_indices=test_indices,
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=device,
    )
    decoded = reconstruct_prefix_from_features(
        train_result["reconstructed_features"],
        start_state=history_level[:, -1, :],
        layout=layout,
    )
    true_memory = compute_memory_targets(
        model,
        history_level,
        history_norm,
        center,
        scale,
        drift_feature,
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    decoded_memory = compute_memory_targets(
        model,
        decoded["history_level"],
        decoded["history_norm"],
        decoded["center"],
        decoded["scale"],
        decoded["drift_feature"],
        device=device,
        batch_size=int(args.eval_batch_size),
    )
    memory_cosine_all = cosine_summary(true_memory, decoded_memory)
    memory_cosine_test = cosine_summary(
        true_memory[test_indices],
        decoded_memory[test_indices],
    )
    train_delta = future_delta[train_indices]
    delta_scale = build_delta_scale(train_delta, floor=float(args.score_scale_floor))
    rollout_summary: dict[str, Any] | None = None
    window_scores: list[dict[str, Any]] = []
    true_samples = np.empty((0,), dtype=np.float32)
    decoded_samples = np.empty((0,), dtype=np.float32)
    if bool(args.run_rollout):
        true_samples = sample_prefix_generator_deltas(
            model,
            indices=eval_test_indices,
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
        decoded_samples = sample_prefix_generator_deltas(
            model,
            indices=eval_test_indices,
            history_level=decoded["history_level"],
            history_norm=decoded["history_norm"],
            center=decoded["center"],
            scale=decoded["scale"],
            drift_feature=decoded["drift_feature"],
            history_raw=history_raw,
            specs=specs,
            samples=int(args.samples),
            n_steps=int(args.n_steps),
            chunk_size=int(args.chunk_size),
            temperature=float(args.temperature),
            device=device,
        )
        window_scores, rollout_summary = _score_prefix_rollouts(
            test_indices=eval_test_indices,
            future_delta=future_delta,
            delta_scale=delta_scale,
            true_samples=true_samples,
            decoded_samples=decoded_samples,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "scope_note": (
            "Local oracle prefix-latent gate. This run does not call OpenAI. "
            "It tests whether a compact latent can reconstruct the SNI recent-prefix "
            "object well enough to preserve frozen-generator memory and rollout behavior."
        ),
        "checkpoint": str(args.checkpoint),
        "bridge_report": str(args.bridge_report),
        "selected_window_count": int(selected_windows.size),
        "train_window_count": int(train_indices.size),
        "test_window_count": int(test_indices.size),
        "rollout_eval_window_count": int(eval_test_indices.size),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "layout": asdict(layout),
        "autoencoder": {
            "loss_first": float(train_result["loss_first"]),
            "loss_last": float(train_result["loss_last"]),
            "train_mse": float(train_result["train_mse"]),
            "test_mse": float(train_result["test_mse"]),
        },
        "memory_cosine_all": memory_cosine_all,
        "memory_cosine_test": memory_cosine_test,
        "rollout_summary": rollout_summary,
        "window_scores": window_scores,
        "artifact_paths": {
            "report": str(output_dir / "prefix_latent_oracle_report.json"),
            "arrays": str(output_dir / "prefix_latent_oracle_arrays.npz"),
        },
    }
    np.savez_compressed(
        output_dir / "prefix_latent_oracle_arrays.npz",
        selected_window_indices=selected_windows.astype(np.int64),
        train_indices=train_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        feature_matrix=feature_matrix.astype(np.float32),
        reconstructed_features=train_result["reconstructed_features"].astype(np.float32),
        encoded_latents=train_result["encoded_latents"].astype(np.float32),
        true_memory=true_memory.astype(np.float32),
        decoded_memory=decoded_memory.astype(np.float32),
        history_level=history_level.astype(np.float32),
        decoded_history_level=decoded["history_level"].astype(np.float32),
        future_delta=future_delta.astype(np.float32),
        delta_scale=delta_scale.astype(np.float32),
        true_prefix_samples=true_samples.astype(np.float32),
        decoded_prefix_samples=decoded_samples.astype(np.float32),
    )
    _write_json(output_dir / "prefix_latent_oracle_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=786)
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
    report = run_oracle_prefix_autoencoder(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "selected_window_count": report["selected_window_count"],
                "autoencoder": report["autoencoder"],
                "memory_cosine_test": report["memory_cosine_test"],
                "rollout_summary": report["rollout_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
