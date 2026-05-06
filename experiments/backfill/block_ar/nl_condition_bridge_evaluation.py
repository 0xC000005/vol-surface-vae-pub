#!/usr/bin/env python
"""Held-out bridge evaluation for narrative-conditioned scenario control.

This script evaluates the part that has to work before a production demo is
credible:

    narrative text -> adapter -> real generator condition memory

It is intentionally artifact-based. Given a prior narrative pipeline report and
its saved embeddings/memory targets, it trains the adapter on one subset of
windows and evaluates condition alignment/retrieval on held-out windows without
making new OpenAI calls.
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

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    build_narrative_training_examples,
    train_narrative_adapter,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    cosine_similarity,
    normalize_rows,
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _round_float(value: float) -> float:
    return round(float(value), 12)


def select_train_test_windows(
    n_windows: int,
    *,
    train_windows: int,
    test_windows: int,
) -> dict[str, list[int]]:
    """Use the first train slice and the next held-out slice."""

    n = int(n_windows)
    train_n = min(max(0, int(train_windows)), n)
    test_n = min(max(0, int(test_windows)), max(0, n - train_n))
    train_indices = list(range(train_n))
    test_indices = list(range(train_n, train_n + test_n))
    excluded_indices = list(range(train_n + test_n, n))
    if not train_indices:
        raise ValueError("need at least one training window")
    if not test_indices:
        raise ValueError("need at least one held-out test window")
    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "excluded_indices": excluded_indices,
    }


def select_train_test_windows_from_report(
    report: dict[str, Any],
    n_windows: int,
    *,
    train_windows: int,
    test_windows: int,
) -> dict[str, Any]:
    """Prefer manifest train/test labels when pipeline artifacts include them."""

    metadata = report.get("window_metadata", [])
    if isinstance(metadata, list) and len(metadata) >= int(n_windows):
        train_indices = [
            idx
            for idx, row in enumerate(metadata[: int(n_windows)])
            if isinstance(row, dict) and str(row.get("manifest_split", "")) == "train"
        ]
        test_indices = [
            idx
            for idx, row in enumerate(metadata[: int(n_windows)])
            if isinstance(row, dict) and str(row.get("manifest_split", "")) == "test"
        ]
        if train_indices and test_indices:
            used = set(train_indices) | set(test_indices)
            return {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "excluded_indices": [idx for idx in range(int(n_windows)) if idx not in used],
                "source": "manifest",
            }
    split = select_train_test_windows(
        n_windows,
        train_windows=int(train_windows),
        test_windows=int(test_windows),
    )
    split["source"] = "sequential"
    return split


def build_bridge_examples(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the exact example order used by the narrative pipeline."""

    bundles = report.get("narrative_bundles", [])
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("report must contain nonempty narrative_bundles")
    examples: list[dict[str, Any]] = []
    for window_index, bundle in enumerate(bundles):
        rows = build_narrative_training_examples(bundle, target_index=window_index)
        for row in rows:
            enriched = dict(row)
            enriched["window_index"] = int(window_index)
            enriched["embedding_index"] = len(examples)
            examples.append(enriched)
    return examples


def _top_rows(
    query: np.ndarray,
    memory_targets: np.ndarray,
    candidate_indices: list[int],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    target_norm = normalize_rows(memory_targets[np.asarray(candidate_indices, dtype=np.int64)])
    q = np.asarray(query, dtype=np.float32)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-8)
    sims = target_norm @ q_norm
    order = np.argsort(-sims)[: max(1, int(top_k))]
    rows: list[dict[str, Any]] = []
    for local_idx in order:
        window_index = int(candidate_indices[int(local_idx)])
        rows.append(
            {
                "window_index": window_index,
                "cosine": _round_float(float(sims[int(local_idx)])),
            }
        )
    return rows


def _true_rank(
    query: np.ndarray,
    memory_targets: np.ndarray,
    true_index: int,
    *,
    candidate_indices: list[int] | None = None,
) -> int:
    all_indices = (
        list(range(int(memory_targets.shape[0])))
        if candidate_indices is None
        else [int(idx) for idx in candidate_indices]
    )
    rows = _top_rows(query, memory_targets, all_indices, top_k=len(all_indices))
    ordered = [row["window_index"] for row in rows]
    return int(ordered.index(int(true_index)) + 1)


def _window_id_lookup(examples: list[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for example in examples:
        lookup[int(example["window_index"])] = str(example["window_id"])
    return lookup


def _annotate_window_ids(
    rows: list[dict[str, Any]],
    window_ids: dict[int, str],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "window_id": window_ids.get(int(row["window_index"]), ""),
        }
        for row in rows
    ]


def evaluate_condition_bridge(
    examples: list[dict[str, Any]],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    *,
    train_indices: list[int],
    test_indices: list[int],
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate held-out condition alignment and retrieval behavior."""

    cond = np.asarray(condition_vectors, dtype=np.float32)
    targets = np.asarray(memory_targets, dtype=np.float32)
    if cond.shape[0] != len(examples):
        raise ValueError("condition_vectors rows must match examples")
    if targets.ndim != 2:
        raise ValueError("memory_targets must be 2-D")
    test_set = set(int(idx) for idx in test_indices)
    train_pool = [int(idx) for idx in train_indices]
    full_pool = list(range(int(targets.shape[0])))
    window_ids = _window_id_lookup(examples)
    heldout_rows: list[dict[str, Any]] = []
    for example in examples:
        window_index = int(example["window_index"])
        if window_index not in test_set or example["role"] == "negative":
            continue
        emb_idx = int(example["embedding_index"])
        query = cond[emb_idx]
        target = targets[window_index]
        target_cosine = cosine_similarity(query, target)
        top_full = _top_rows(query, targets, full_pool, top_k=top_k)
        top_test = _top_rows(query, targets, list(test_indices), top_k=top_k)
        top_train = _top_rows(query, targets, train_pool, top_k=top_k)
        heldout_rows.append(
            {
                "window_index": window_index,
                "window_id": str(example["window_id"]),
                "embedding_index": emb_idx,
                "role": str(example["role"]),
                "kind": str(example.get("kind", "")),
                "target_cosine": _round_float(target_cosine),
                "target_mse": _round_float(float(np.mean((query - target) ** 2))),
                "true_rank_full_pool": _true_rank(query, targets, window_index),
                "true_rank_test_pool": _true_rank(
                    query,
                    targets,
                    window_index,
                    candidate_indices=list(test_indices),
                ),
                "top_full_pool": _annotate_window_ids(top_full, window_ids),
                "top_test_pool": _annotate_window_ids(top_test, window_ids),
                "top_train_pool": _annotate_window_ids(top_train, window_ids),
            }
        )

    separation_rows: list[dict[str, Any]] = []
    for window_index in test_indices:
        group = [item for item in examples if int(item["window_index"]) == int(window_index)]
        anchors = [item for item in group if item["role"] == "anchor"]
        negatives = [item for item in group if item["role"] == "negative"]
        positives = [item for item in group if item["role"] == "positive"]
        if not anchors or not negatives:
            continue
        anchor_vec = cond[int(anchors[0]["embedding_index"])]
        if positives:
            positive_cosines = [
                cosine_similarity(anchor_vec, cond[int(item["embedding_index"])])
                for item in positives
            ]
        else:
            positive_cosines = [
                cosine_similarity(anchor_vec, targets[int(window_index)])
            ]
        negative_cosines = [
            cosine_similarity(anchor_vec, cond[int(item["embedding_index"])])
            for item in negatives
        ]
        hard_margin = float(min(positive_cosines) - max(negative_cosines))
        negative_gap = float(np.mean(positive_cosines) - np.mean(negative_cosines))
        separation_rows.append(
            {
                "window_index": int(window_index),
                "window_id": window_ids.get(int(window_index), ""),
                "positive_mean_cosine": _round_float(float(np.mean(positive_cosines))),
                "negative_mean_cosine": _round_float(float(np.mean(negative_cosines))),
                "hard_margin": _round_float(hard_margin),
                "negative_gap": _round_float(negative_gap),
            }
        )
    hard_margins = [float(row["hard_margin"]) for row in separation_rows]
    negative_gaps = [float(row["negative_gap"]) for row in separation_rows]
    return {
        "heldout_window_count": len(set(row["window_index"] for row in heldout_rows)),
        "heldout_example_count": len(heldout_rows),
        "heldout_examples": heldout_rows,
        "hard_negative_separation": {
            "window_count": len(separation_rows),
            "mean_hard_margin": _round_float(float(np.mean(hard_margins))) if hard_margins else None,
            "mean_negative_gap": _round_float(float(np.mean(negative_gaps))) if negative_gaps else None,
            "windows": separation_rows,
        },
    }


def summarize_bridge_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    rows = evaluation.get("heldout_examples", []) or []
    if not rows:
        return {
            "heldout_example_count": 0,
            "heldout_window_count": 0,
        }
    target_cos = np.asarray([float(row["target_cosine"]) for row in rows], dtype=np.float64)
    ranks = np.asarray([int(row["true_rank_full_pool"]) for row in rows], dtype=np.int64)
    test_ranks = np.asarray([int(row["true_rank_test_pool"]) for row in rows], dtype=np.int64)
    top_train = np.asarray(
        [float(row["top_train_pool"][0]["cosine"]) for row in rows if row.get("top_train_pool")],
        dtype=np.float64,
    )
    hard = evaluation.get("hard_negative_separation", {}) or {}
    return {
        "heldout_example_count": len(rows),
        "heldout_window_count": int(evaluation.get("heldout_window_count", 0)),
        "heldout_mean_target_cosine": _round_float(float(target_cos.mean())),
        "heldout_median_target_cosine": _round_float(float(np.median(target_cos))),
        "heldout_recall_at_1_full_pool": _round_float(float(np.mean(ranks <= 1))),
        "heldout_recall_at_3_full_pool": _round_float(float(np.mean(ranks <= 3))),
        "heldout_median_true_rank_full_pool": _round_float(float(np.median(ranks))),
        "heldout_recall_at_1_test_pool": _round_float(float(np.mean(test_ranks <= 1))),
        "heldout_recall_at_3_test_pool": _round_float(float(np.mean(test_ranks <= 3))),
        "heldout_median_true_rank_test_pool": _round_float(float(np.median(test_ranks))),
        "heldout_mean_top_train_cosine": (
            _round_float(float(top_train.mean())) if top_train.size else None
        ),
        "heldout_hard_negative_window_count": int(hard.get("window_count", 0) or 0),
        "heldout_hard_negative_mean_margin": hard.get("mean_hard_margin"),
        "heldout_hard_negative_mean_gap": hard.get("mean_negative_gap"),
    }


def load_pipeline_artifacts(
    report_path: str | Path,
    arrays_path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    with np.load(arrays_path) as data:
        arrays = {name: data[name] for name in data.files}
    return report, arrays


def run_bridge_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    report, arrays = load_pipeline_artifacts(args.input_report, args.input_npz)
    examples = build_bridge_examples(report)
    text_embeddings = np.asarray(arrays["text_embeddings"], dtype=np.float32)
    memory_targets = np.asarray(arrays["memory_targets"], dtype=np.float32)
    if text_embeddings.shape[0] != len(examples):
        raise ValueError(
            f"text_embeddings rows ({text_embeddings.shape[0]}) do not match "
            f"rebuilt examples ({len(examples)})"
        )
    split = select_train_test_windows_from_report(
        report,
        int(memory_targets.shape[0]),
        train_windows=int(args.train_windows),
        test_windows=int(args.test_windows),
    )
    train_set = set(split["train_indices"])
    train_example_indices = [
        idx
        for idx, example in enumerate(examples)
        if int(example["window_index"]) in train_set
    ]
    train_embeddings = text_embeddings[np.asarray(train_example_indices, dtype=np.int64)]
    train_examples = [examples[idx] for idx in train_example_indices]
    target_indices = np.asarray(
        [
            -1 if example["target_index"] is None else int(example["target_index"])
            for example in train_examples
        ],
        dtype=np.int64,
    )
    roles = [str(example["role"]) for example in train_examples]
    groups = [str(example["window_id"]) for example in train_examples]
    train_result = train_narrative_adapter(
        train_embeddings,
        memory_targets,
        target_indices,
        roles,
        groups,
        condition_dim=int(memory_targets.shape[1]),
        hidden_dim=args.hidden_dim,
        steps=int(args.adapter_steps),
        lr=float(args.adapter_lr),
        contrastive_weight=float(args.contrastive_weight),
        contrastive_margin=float(args.contrastive_margin),
        seed=int(args.seed),
    )
    adapter = train_result["adapter"]
    adapter.eval()
    with torch.no_grad():
        all_conditions = (
            adapter(torch.from_numpy(normalize_rows(text_embeddings)).float())
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    evaluation = evaluate_condition_bridge(
        examples,
        all_conditions,
        memory_targets,
        train_indices=split["train_indices"],
        test_indices=split["test_indices"],
        top_k=int(args.top_k),
    )
    summary = summarize_bridge_metrics(evaluation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "bridge_adapter.pt"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "condition_dim": int(memory_targets.shape[1]),
            "embedding_dim": int(text_embeddings.shape[1]),
            "source_report": str(args.input_report),
            "source_npz": str(args.input_npz),
        },
        adapter_path,
    )
    np.savez_compressed(
        output_dir / "bridge_eval_arrays.npz",
        condition_vectors=all_conditions,
        memory_targets=memory_targets,
        text_embeddings=normalize_rows(text_embeddings),
        train_indices=np.asarray(split["train_indices"], dtype=np.int64),
        test_indices=np.asarray(split["test_indices"], dtype=np.int64),
    )
    output_report = {
        "status": "ok",
        "scope_note": (
            "Held-out bridge evaluation only. It trains the text-to-condition "
            "adapter on training-window narrative embeddings and evaluates "
            "against held-out generator memory targets. No OpenAI API calls are "
            "made by this script."
        ),
        "input_report": str(args.input_report),
        "input_npz": str(args.input_npz),
        "embedding_backend": report.get("embedding_backend"),
        "embedding_model": report.get("embedding_model"),
        "window_metadata": report.get("window_metadata", []),
        "source_indices": report.get("source_indices", []),
        "window_indices": report.get("window_indices", list(range(int(memory_targets.shape[0])))),
        "split": split,
        "train_example_count": len(train_examples),
        "adapter_training": {
            "loss_first": _round_float(float(train_result["loss_first"])),
            "loss_last": _round_float(float(train_result["loss_last"])),
            "steps": int(args.adapter_steps),
            "contrastive_weight": float(args.contrastive_weight),
            "contrastive_margin": float(args.contrastive_margin),
        },
        "summary": summary,
        "evaluation": evaluation,
        "artifact_paths": {
            "adapter": str(adapter_path),
            "arrays": str(output_dir / "bridge_eval_arrays.npz"),
            "report": str(output_dir / "bridge_eval_report.json"),
        },
    }
    _write_json(output_dir / "bridge_eval_report.json", output_report)
    return output_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-windows", type=int, default=40)
    parser.add_argument("--test-windows", type=int, default=10)
    parser.add_argument("--adapter-steps", type=int, default=700)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-margin", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=775)
    args = parser.parse_args()
    report = run_bridge_evaluation(args)
    print(
        json.dumps(
            {
                "report": report["artifact_paths"]["report"],
                "adapter_loss_first": report["adapter_training"]["loss_first"],
                "adapter_loss_last": report["adapter_training"]["loss_last"],
                **report["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
