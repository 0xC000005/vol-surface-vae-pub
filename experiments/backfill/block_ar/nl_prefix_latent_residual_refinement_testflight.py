#!/usr/bin/env python
"""Offline residual-refinement TestFlight for narrative/start memory support.

This is a bounded check of the next prefix-latent direction:

    text-predicted memory + fixed start -> support-mixture memory
    support-mixture memory + text memory + fixed start -> bounded residual

The script makes no OpenAI calls. It reuses saved OpenAI embeddings, the current
text-to-memory bridge outputs, and frozen-generator memory targets. The goal is
to test whether a learned residual on top of the auditable support mixture has
signal before adding a larger latent-prior architecture.
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

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
    evaluate_condition_bridge,
    summarize_bridge_metrics,
)
from experiments.backfill.block_ar.nl_text_conditioning import (
    normalize_rows,
)  # noqa: E402


DEFAULT_PIPELINE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_schema_v2_representative_220/narrative_pipeline_report.json"
)
DEFAULT_BRIDGE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_arrays.npz"
)
DEFAULT_ORACLE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_oracle_fullheldout_786c/prefix_latent_oracle_arrays.npz"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_residual_refinement_testflight_870a"
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


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    raw = float(value)
    return round(raw, 12) if np.isfinite(raw) else None


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32).std(axis=0, keepdims=True)
    return np.maximum(std, 1e-6).astype(np.float32)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    shifted = (raw - float(np.max(raw))) / max(float(temperature), 1e-6)
    weights = np.exp(shifted)
    denom = float(np.sum(weights))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.ones(raw.shape[0], dtype=np.float32) / float(raw.shape[0])
    return (weights / denom).astype(np.float32)


def _cosine_to_rows(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    t = np.asarray(targets, dtype=np.float32)
    denom = np.maximum(np.linalg.norm(t, axis=1) * np.linalg.norm(q), 1e-8)
    return (np.sum(t * q, axis=1) / denom).astype(np.float32)


def _standardize_starts(
    start_state: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    starts = np.asarray(start_state, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    mean = starts[train].mean(axis=0, keepdims=True).astype(np.float32)
    std = _safe_std(starts[train])
    return ((starts - mean) / std).astype(np.float32), {
        "mean_shape": list(mean.shape),
        "std_min": _round(float(np.min(std))),
        "std_max": _round(float(np.max(std))),
    }


def build_support_mixture_memory(
    *,
    examples: list[dict[str, Any]],
    query_memory: np.ndarray,
    memory_targets: np.ndarray,
    start_state: np.ndarray,
    train_indices: np.ndarray,
    top_k: int,
    temperature: float,
    start_distance_penalty: float,
    exclude_self: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build soft top-k narrative/start support-memory for every example."""

    cond = normalize_rows(np.asarray(query_memory, dtype=np.float32))
    memory = np.asarray(memory_targets, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    start_z, start_stats = _standardize_starts(start_state, train)
    mixtures = np.zeros_like(cond, dtype=np.float32)
    support_rows: list[dict[str, Any]] = []
    for row_idx, example in enumerate(examples):
        window_index = int(example["window_index"])
        bank = train
        if exclude_self:
            filtered = bank[bank != window_index]
            if filtered.size:
                bank = filtered
        cosine = _cosine_to_rows(cond[row_idx], memory[bank])
        distance = np.linalg.norm(start_z[bank] - start_z[window_index], axis=1)
        score = cosine - float(start_distance_penalty) * distance.astype(np.float32)
        order = np.argsort(-score)[: max(1, int(top_k))]
        selected = bank[order]
        weights = _softmax(score[order], float(temperature))
        mixtures[row_idx] = np.sum(memory[selected] * weights[:, None], axis=0)
        if str(example.get("role")) == "anchor" and len(support_rows) < 10:
            support_rows.append(
                {
                    "embedding_index": int(example["embedding_index"]),
                    "window_index": window_index,
                    "selected_windows": [int(value) for value in selected],
                    "weights": [_round(float(value)) for value in weights],
                    "best_score": _round(float(score[order[0]])),
                    "best_cosine": _round(float(cosine[order[0]])),
                    "best_start_distance_z": _round(float(distance[order[0]])),
                }
            )
    return mixtures, {
        "top_k": int(top_k),
        "temperature": float(temperature),
        "start_distance_penalty": float(start_distance_penalty),
        "exclude_self": bool(exclude_self),
        "start_standardization": start_stats,
        "sample_support_rows": support_rows,
    }


class ResidualRefiner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _group_hard_negative_loss(
    refined: torch.Tensor,
    *,
    roles: list[str],
    groups: list[str],
    margin: float,
) -> torch.Tensor:
    """Keep anchor/positive refinements separated from hard-negative text."""

    if refined.numel() == 0:
        return refined.new_zeros(())
    normed = torch.nn.functional.normalize(refined, dim=-1)
    terms: list[torch.Tensor] = []
    for group in sorted(set(groups)):
        indices = [idx for idx, value in enumerate(groups) if value == group]
        anchors = [idx for idx in indices if roles[idx] == "anchor"]
        positives = [idx for idx in indices if roles[idx] == "positive"]
        negatives = [idx for idx in indices if roles[idx] == "negative"]
        if not anchors or not negatives:
            continue
        anchor = normed[anchors[0]]
        pos = (
            normed[positives] @ anchor
            if positives
            else torch.ones(1, device=refined.device)
        )
        neg = normed[negatives] @ anchor
        terms.append(torch.relu(float(margin) - pos[:, None] + neg[None, :]).mean())
    if not terms:
        return refined.new_zeros(())
    return torch.stack(terms).mean()


def build_residual_inputs(
    *,
    query_memory: np.ndarray,
    mixture_memory: np.ndarray,
    start_state: np.ndarray,
    examples: list[dict[str, Any]],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build residual-refiner inputs from text memory, start, and support prior."""

    cond = normalize_rows(np.asarray(query_memory, dtype=np.float32))
    mixture = normalize_rows(np.asarray(mixture_memory, dtype=np.float32))
    start_z, start_stats = _standardize_starts(start_state, train_indices)
    example_starts = np.asarray(
        [start_z[int(example["window_index"])] for example in examples],
        dtype=np.float32,
    )
    diff = cond - mixture
    inputs = np.concatenate([cond, mixture, diff, example_starts], axis=1).astype(
        np.float32
    )
    return inputs, {
        "input_dim": int(inputs.shape[1]),
        "channels": [
            "text_memory",
            "support_mixture_memory",
            "memory_difference",
            "start_z",
        ],
        "start_standardization": start_stats,
    }


def train_residual_refiner(
    *,
    inputs: np.ndarray,
    mixture_memory: np.ndarray,
    memory_targets: np.ndarray,
    examples: list[dict[str, Any]],
    train_indices: np.ndarray,
    hidden_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.25,
) -> dict[str, Any]:
    """Train a bounded residual model around the support-mixture memory."""

    train_set = set(int(value) for value in train_indices)
    train_rows = [
        row_idx
        for row_idx, example in enumerate(examples)
        if int(example["window_index"]) in train_set
        and example.get("target_index") is not None
    ]
    group_rows = [
        row_idx
        for row_idx, example in enumerate(examples)
        if int(example["window_index"]) in train_set
    ]
    if not train_rows:
        raise ValueError("no target-bearing train rows for residual refiner")
    row_idx = np.asarray(train_rows, dtype=np.int64)
    target_indices = np.asarray(
        [int(examples[idx]["target_index"]) for idx in row_idx],
        dtype=np.int64,
    )
    residual_target = (
        np.asarray(memory_targets, dtype=np.float32)[target_indices]
        - np.asarray(mixture_memory, dtype=np.float32)[row_idx]
    )
    residual_mean = residual_target.mean(axis=0, keepdims=True).astype(np.float32)
    residual_std = _safe_std(residual_target)
    residual_train = ((residual_target - residual_mean) / residual_std).astype(
        np.float32
    )
    residual_norm_cap = float(
        np.quantile(np.linalg.norm(residual_target, axis=1), 0.95)
    )
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    x = torch.from_numpy(np.asarray(inputs, dtype=np.float32)).to(dev)
    mixture_t = torch.from_numpy(np.asarray(mixture_memory, dtype=np.float32)).to(dev)
    residual_mean_t = torch.from_numpy(residual_mean).to(dev)
    residual_std_t = torch.from_numpy(residual_std).to(dev)
    y = torch.from_numpy(residual_train).to(dev)
    train_t = torch.from_numpy(row_idx).long().to(dev)
    group_t = torch.from_numpy(np.asarray(group_rows, dtype=np.int64)).long().to(dev)
    group_roles = [str(examples[idx]["role"]) for idx in group_rows]
    group_ids = [str(examples[idx]["window_id"]) for idx in group_rows]
    model = ResidualRefiner(
        input_dim=int(inputs.shape[1]),
        output_dim=int(memory_targets.shape[1]),
        hidden_dim=int(hidden_dim),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    batch = max(1, min(int(batch_size), int(row_idx.size)))
    with torch.no_grad():
        initial = model(x[train_t])
        loss_first = float(torch.mean((initial - y) ** 2).cpu())
    losses: list[float] = []
    hard_losses: list[float] = []
    for _ in range(int(steps)):
        choice = rng.choice(
            np.arange(row_idx.size), size=batch, replace=row_idx.size < batch
        )
        choice_t = torch.from_numpy(choice).long().to(dev)
        pred = model(x[train_t[choice_t]])
        mse_loss = torch.mean((pred - y[choice_t]) ** 2)
        if float(hard_negative_weight) > 0.0:
            group_pred_std = model(x[group_t])
            group_residual = group_pred_std * residual_std_t + residual_mean_t
            group_refined = mixture_t[group_t] + group_residual
            hard_loss = _group_hard_negative_loss(
                group_refined,
                roles=group_roles,
                groups=group_ids,
                margin=float(hard_negative_margin),
            )
        else:
            hard_loss = mse_loss.new_zeros(())
        loss = mse_loss + float(hard_negative_weight) * hard_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        hard_losses.append(float(hard_loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        pred_std = model(x).cpu().numpy().astype(np.float32)
    pred_residual = (pred_std * residual_std + residual_mean).astype(np.float32)
    norms = np.linalg.norm(pred_residual, axis=1)
    scale = np.minimum(1.0, residual_norm_cap / np.maximum(norms, 1e-8))
    bounded_residual = (pred_residual * scale[:, None]).astype(np.float32)
    refined = (np.asarray(mixture_memory, dtype=np.float32) + bounded_residual).astype(
        np.float32
    )
    return {
        "condition_vectors": refined,
        "loss_first": float(loss_first),
        "loss_last": float(losses[-1] if losses else loss_first),
        "train_row_count": int(row_idx.size),
        "group_row_count": int(len(group_rows)),
        "hard_negative_weight": float(hard_negative_weight),
        "hard_negative_margin": float(hard_negative_margin),
        "hard_negative_loss_last": float(hard_losses[-1] if hard_losses else 0.0),
        "residual_norm_cap_p95": residual_norm_cap,
        "predicted_residual_norm_mean": float(np.mean(norms)),
        "bounded_residual_norm_mean": float(
            np.mean(np.linalg.norm(bounded_residual, axis=1))
        ),
    }


def _evaluate(
    *,
    examples: list[dict[str, Any]],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    evaluation = evaluate_condition_bridge(
        examples,
        condition_vectors,
        memory_targets,
        train_indices=[int(value) for value in train_indices],
        test_indices=[int(value) for value in test_indices],
        top_k=int(top_k),
    )
    return {
        "summary": summarize_bridge_metrics(evaluation),
        "evaluation": evaluation,
    }


def _delta_summary(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | None]:
    metrics = [
        "heldout_mean_target_cosine",
        "heldout_hard_negative_mean_gap",
        "heldout_hard_negative_mean_margin",
        "heldout_recall_at_1_test_pool",
        "heldout_recall_at_3_test_pool",
    ]
    out: dict[str, float | None] = {}
    for metric in metrics:
        left = candidate["summary"].get(metric)
        right = baseline["summary"].get(metric)
        out[metric] = (
            None
            if left is None or right is None
            else _round(float(left) - float(right))
        )
    return out


def resolve_torch_device(requested: str) -> str:
    value = str(requested).lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return value


def run_testflight(args: argparse.Namespace) -> dict[str, Any]:
    report = _load_json(args.pipeline_report)
    examples = build_bridge_examples(report)
    with np.load(args.bridge_arrays) as bridge_payload:
        bridge_arrays = {
            name: bridge_payload[name].copy() for name in bridge_payload.files
        }
    with np.load(args.oracle_arrays) as oracle_payload:
        oracle_arrays = {
            name: oracle_payload[name].copy() for name in oracle_payload.files
        }
    query_memory = np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32)
    memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    train_indices = np.asarray(bridge_arrays["train_indices"], dtype=np.int64)
    test_indices = np.asarray(bridge_arrays["test_indices"], dtype=np.int64)
    history_level = np.asarray(oracle_arrays["history_level"], dtype=np.float32)
    if query_memory.shape[0] != len(examples):
        raise ValueError("condition_vectors rows do not match rebuilt examples")
    if history_level.shape[0] != memory_targets.shape[0]:
        raise ValueError("oracle history rows do not match memory targets")
    start_state = history_level[:, -1, :]
    mixture_memory, mixture_details = build_support_mixture_memory(
        examples=examples,
        query_memory=query_memory,
        memory_targets=memory_targets,
        start_state=start_state,
        train_indices=train_indices,
        top_k=int(args.top_k),
        temperature=float(args.temperature),
        start_distance_penalty=float(args.start_distance_penalty),
        exclude_self=bool(args.exclude_self),
    )
    residual_inputs, input_details = build_residual_inputs(
        query_memory=query_memory,
        mixture_memory=mixture_memory,
        start_state=start_state,
        examples=examples,
        train_indices=train_indices,
    )
    device = resolve_torch_device(args.device)
    residual = train_residual_refiner(
        inputs=residual_inputs,
        mixture_memory=mixture_memory,
        memory_targets=memory_targets,
        examples=examples,
        train_indices=train_indices,
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        device=device,
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_margin=float(args.hard_negative_margin),
    )
    baselines = {
        "text_memory_incumbent": _evaluate(
            examples=examples,
            condition_vectors=query_memory,
            memory_targets=memory_targets,
            train_indices=train_indices,
            test_indices=test_indices,
            top_k=int(args.eval_top_k),
        ),
        "support_mixture_memory": _evaluate(
            examples=examples,
            condition_vectors=mixture_memory,
            memory_targets=memory_targets,
            train_indices=train_indices,
            test_indices=test_indices,
            top_k=int(args.eval_top_k),
        ),
        "residual_refined_memory": _evaluate(
            examples=examples,
            condition_vectors=residual["condition_vectors"],
            memory_targets=memory_targets,
            train_indices=train_indices,
            test_indices=test_indices,
            top_k=int(args.eval_top_k),
        ),
    }
    refined = baselines["residual_refined_memory"]
    text = baselines["text_memory_incumbent"]
    mixture = baselines["support_mixture_memory"]
    target_gain = _delta_summary(refined, text)["heldout_mean_target_cosine"]
    mixture_gain = _delta_summary(refined, mixture)["heldout_mean_target_cosine"]
    gap_delta = _delta_summary(refined, text)["heldout_hard_negative_mean_gap"]
    status = "diagnostic_only"
    if (
        target_gain is not None
        and mixture_gain is not None
        and gap_delta is not None
        and target_gain >= float(args.min_target_cosine_gain)
        and mixture_gain >= float(args.min_mixture_cosine_gain)
        and gap_delta >= -float(args.max_gap_loss)
    ):
        status = "testflight_pass"
    output_dir = Path(args.output_dir)
    output = {
        "status": status,
        "scope_note": (
            "Offline residual-refinement TestFlight. No OpenAI API calls; no "
            "generator rollout. This tests memory-space signal before larger "
            "prefix-latent architecture changes."
        ),
        "device": device,
        "config": {
            "top_k": int(args.top_k),
            "temperature": float(args.temperature),
            "start_distance_penalty": float(args.start_distance_penalty),
            "hidden_dim": int(args.hidden_dim),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "hard_negative_weight": float(args.hard_negative_weight),
            "hard_negative_margin": float(args.hard_negative_margin),
            "min_target_cosine_gain": float(args.min_target_cosine_gain),
            "min_mixture_cosine_gain": float(args.min_mixture_cosine_gain),
            "max_gap_loss": float(args.max_gap_loss),
        },
        "inputs": input_details,
        "mixture_support": mixture_details,
        "residual_training": {
            key: _round(value) if isinstance(value, float) else value
            for key, value in residual.items()
            if key != "condition_vectors"
        },
        "metrics": {name: payload["summary"] for name, payload in baselines.items()},
        "deltas": {
            "residual_minus_text_memory": _delta_summary(refined, text),
            "residual_minus_support_mixture": _delta_summary(refined, mixture),
            "support_mixture_minus_text_memory": _delta_summary(mixture, text),
        },
        "decision": {
            "status": status,
            "promote": False,
            "reason": (
                "Promote only after scenario-level rollout and independent verification."
                if status == "testflight_pass"
                else "Do not promote: memory-space residual did not clear all TestFlight gates."
            ),
        },
        "artifact_paths": {
            "report": str(output_dir / "residual_refinement_testflight.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "residual_refinement_testflight.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "report": output["artifact_paths"]["report"],
                "metrics": output["metrics"],
                "deltas": output["deltas"],
            },
            sort_keys=True,
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--oracle-arrays", default=DEFAULT_ORACLE_ARRAYS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument(
        "--exclude-self", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=870)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.25)
    parser.add_argument("--min-target-cosine-gain", type=float, default=0.005)
    parser.add_argument("--min-mixture-cosine-gain", type=float, default=0.005)
    parser.add_argument("--max-gap-loss", type=float, default=0.01)
    args = parser.parse_args()
    run_testflight(args)


if __name__ == "__main__":
    main()
