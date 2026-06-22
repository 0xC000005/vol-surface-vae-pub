#!/usr/bin/env python
"""Train manifest-aware 14+14 narrative retrieval pilots.

This module consumes the validated stride-5 self-supervised manifest. It does
not generate narrative text. It provides two smoke/full-run trainers:

1. text-space contrastive retrieval over narrative embeddings;
2. projected-memory bridge training with explicit paired hard-negative margins.

991a additions (reporting contract, not research axes): window-level holdout
with purge gap, view-family holdout for sparse-query/view-robustness
diagnostics, checkpoint saving, and validation-coupled early stopping. All
loss terms, weights, margins, temperatures, and architectures are unchanged
from the 990e pilot.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    embed_with_cache,
)
from experiments.backfill.block_ar.nl_text_conditioning import normalize_rows  # noqa: E402

# Locality-soft (Track A T3) reuses the EXACT cosine / mask / retrieval primitives
# and the binding purge constants from the T2 fit-gate calibration so the held-out
# locality-recall@K eval is byte-identical in definition to the pre-registered gate.
from experiments.backfill.block_ar.nl_locality_soft_fit_gate_calibration_t2 import (  # noqa: E402
    PURGE_GAP as CAL_PURGE_GAP,
    RETRIEVAL_TOP_K as CAL_RETRIEVAL_TOP_K,
    VAL_WINDOW_RANGES as CAL_VAL_WINDOW_RANGES,
    build_masks as cal_build_masks,
    l2_normalize as cal_l2_normalize,
    topk_in_pool as cal_topk_in_pool,
)


DEFAULT_EXAMPLES_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_self_supervised_training_manifest_990a/training_examples.jsonl"
)
DEFAULT_PAIRS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_self_supervised_training_manifest_990a/training_pairs.jsonl"
)
DEFAULT_SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_retrieval_training_smoke_990b"
)

LOSS_TRACE_EVERY = 50


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = _resolve(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{input_path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    output = _resolve(path)
    lines = [
        "# NL 14+14 Manifest Retrieval Training",
        "",
        f"- Status: `{report['status']}`",
        f"- Methods: `{', '.join(report['methods'])}`",
    ]
    for key in ("text_space", "projected_memory"):
        method = report.get(key)
        if not isinstance(method, dict):
            continue
        lines.extend(
            [
                "",
                f"## {method['method']}",
                "",
                f"- Status: `{method['status']}`",
                f"- Examples: `{method['training']['example_count']}`",
                f"- Pair rows consumed: `{method['training']['pair_rows_consumed']}`",
            ]
        )
        for metric, value in sorted(method.get("evaluation", {}).items()):
            if isinstance(value, (float, int)):
                lines.append(f"- {metric}: `{value}`")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _safe_normalize_rows(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, float(eps))


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_manifest_training_data(
    *,
    examples_jsonl: str | Path,
    pairs_jsonl: str | Path,
    max_targets: int | None = None,
) -> dict[str, Any]:
    """Load examples and explicit pairs from the 14+14 training manifest."""

    examples = _read_jsonl(examples_jsonl)
    pairs = _read_jsonl(pairs_jsonl)
    if max_targets is not None and int(max_targets) > 0:
        target_ids = sorted(
            {
                str(pair.get("target_window_id", ""))
                for pair in pairs
                if str(pair.get("target_window_id", "")).strip()
            },
            key=lambda value: int(value.rsplit("_", 1)[-1]),
        )[: int(max_targets)]
        allowed_targets = set(target_ids)
        examples = [
            row
            for row in examples
            if str(row.get("target_window_id", "")) in allowed_targets
        ]
        pairs = [
            row
            for row in pairs
            if str(row.get("target_window_id", "")) in allowed_targets
        ]

    cleaned_examples: list[dict[str, Any]] = []
    for embedding_index, row in enumerate(examples):
        text = str(row.get("text", "")).strip()
        example_id = str(row.get("example_id", "")).strip()
        if not text or not example_id:
            continue
        cleaned = dict(row)
        cleaned["embedding_index"] = int(embedding_index)
        cleaned["text"] = text
        cleaned["example_id"] = example_id
        cleaned["label_window_index"] = int(cleaned["label_window_index"])
        cleaned["target_window_index"] = int(cleaned["target_window_index"])
        cleaned_examples.append(cleaned)

    by_id = {str(row["example_id"]): row for row in cleaned_examples}
    cleaned_pairs: list[dict[str, Any]] = []
    for row in pairs:
        pos_id = str(row.get("positive_example_id", "")).strip()
        neg_id = str(row.get("negative_example_id", "")).strip()
        if pos_id not in by_id or neg_id not in by_id:
            continue
        cleaned = dict(row)
        cleaned["positive_embedding_index"] = int(by_id[pos_id]["embedding_index"])
        cleaned["negative_embedding_index"] = int(by_id[neg_id]["embedding_index"])
        cleaned["target_window_index"] = int(cleaned["target_window_index"])
        cleaned["negative_window_index"] = int(cleaned["negative_window_index"])
        cleaned_pairs.append(cleaned)

    if not cleaned_examples:
        raise ValueError("manifest contains no usable examples")
    if not cleaned_pairs:
        raise ValueError("manifest contains no usable explicit pair rows")
    return {
        "examples": cleaned_examples,
        "pairs": cleaned_pairs,
        "texts": [str(row["text"]) for row in cleaned_examples],
        "labels": np.asarray(
            [int(row["label_window_index"]) for row in cleaned_examples],
            dtype=np.int64,
        ),
        "views": [str(row.get("view_name", "")) for row in cleaned_examples],
    }


def _parse_window_ranges(spec: str | None) -> list[tuple[int, int]]:
    if spec is None or not str(spec).strip():
        return []
    ranges: list[tuple[int, int]] = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lo_text, hi_text = chunk.split(":")
        lo, hi = int(lo_text), int(hi_text)
        if hi <= lo:
            raise ValueError(f"invalid window range {chunk!r}: hi must exceed lo")
        ranges.append((lo, hi))
    return ranges


def _build_holdout_split(
    examples: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    val_window_ranges: str | None = None,
    purge_gap: int = 0,
    holdout_view_families: str | None = None,
) -> dict[str, Any]:
    """Mask-level split. The embedded text list is never filtered (cache safety).

    Examples: label in val ranges -> val (Tier A); label in purge zone ->
    dropped from both sides; held-out view family on a train window -> Tier-B
    query (never sampled in training); else train.
    Pairs: train only if BOTH endpoint windows are outside val+purge and the
    view family is not held out; pairs targeting a val window are reserved as
    held-out margin-evaluation pairs.
    """

    ranges = _parse_window_ranges(val_window_ranges)
    gap = max(0, int(purge_gap))
    val_windows: set[int] = set()
    for lo, hi in ranges:
        val_windows.update(range(lo, hi))
    purge_windows: set[int] = set()
    for lo, hi in ranges:
        purge_windows.update(range(max(0, lo - gap), lo))
        purge_windows.update(range(hi, hi + gap))
    purge_windows -= val_windows
    holdout_views: set[str] = set()
    if holdout_view_families:
        holdout_views = {
            part.strip()
            for part in str(holdout_view_families).split(",")
            if part.strip()
        }

    train_idx: list[int] = []
    val_idx: list[int] = []
    tierb_idx: list[int] = []
    purged = 0
    for row in examples:
        window = int(row["label_window_index"])
        view = str(row.get("view_name", ""))
        emb_idx = int(row["embedding_index"])
        if window in val_windows:
            val_idx.append(emb_idx)
        elif window in purge_windows:
            purged += 1
        elif view in holdout_views:
            tierb_idx.append(emb_idx)
        else:
            train_idx.append(emb_idx)

    excluded = val_windows | purge_windows
    train_pair_positions: list[int] = []
    val_pair_positions: list[int] = []
    for position, row in enumerate(pairs):
        target = int(row["target_window_index"])
        negative = int(row["negative_window_index"])
        view = str(row.get("view_name", ""))
        if target in val_windows:
            val_pair_positions.append(position)
        if (
            target not in excluded
            and negative not in excluded
            and view not in holdout_views
        ):
            train_pair_positions.append(position)

    return {
        "enabled": bool(ranges or holdout_views),
        "val_window_ranges": str(val_window_ranges) if val_window_ranges else "",
        "purge_gap": gap,
        "holdout_view_families": sorted(holdout_views),
        "val_window_count": len(val_windows),
        "purge_window_count": len(purge_windows),
        "train_example_idx": np.asarray(train_idx, dtype=np.int64),
        "val_example_idx": np.asarray(val_idx, dtype=np.int64),
        "tierb_query_idx": np.asarray(tierb_idx, dtype=np.int64),
        "purged_example_count": int(purged),
        "train_pair_positions": np.asarray(train_pair_positions, dtype=np.int64),
        "val_pair_positions": np.asarray(val_pair_positions, dtype=np.int64),
    }


def _holdout_split_payload(split: dict[str, Any], *, total_examples: int, total_pairs: int) -> dict[str, Any]:
    return {
        "schema_version": "nl_14x14_holdout_split_v1",
        "enabled": bool(split["enabled"]),
        "val_window_ranges": split["val_window_ranges"],
        "purge_gap": int(split["purge_gap"]),
        "holdout_view_families": list(split["holdout_view_families"]),
        "val_window_count": int(split["val_window_count"]),
        "purge_window_count": int(split["purge_window_count"]),
        "total_example_count": int(total_examples),
        "train_example_count": int(split["train_example_idx"].shape[0]),
        "val_example_count": int(split["val_example_idx"].shape[0]),
        "tierb_query_count": int(split["tierb_query_idx"].shape[0]),
        "purged_example_count": int(split["purged_example_count"]),
        "total_pair_count": int(total_pairs),
        "train_pair_count": int(split["train_pair_positions"].shape[0]),
        "val_pair_count": int(split["val_pair_positions"].shape[0]),
    }


def _embed_manifest_texts(
    *,
    texts: list[str],
    output_dir: Path,
    embedding_backend: str,
    embedding_model: str,
    dotenv_path: str,
    embedding_batch_size: int,
    hash_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    embeddings, meta = embed_with_cache(
        texts,
        output_dir=output_dir,
        backend=str(embedding_backend),
        model=str(embedding_model),
        dotenv_path=str(dotenv_path),
        batch_size=int(embedding_batch_size),
        hash_dim=int(hash_dim),
    )
    return normalize_rows(np.asarray(embeddings, dtype=np.float32)), meta


def _pair_arrays(
    pairs: list[dict[str, Any]], positions: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    if positions is None:
        rows = pairs
    else:
        rows = [pairs[int(position)] for position in positions]
    return {
        "positive": np.asarray(
            [int(row["positive_embedding_index"]) for row in rows], dtype=np.int64
        ),
        "negative": np.asarray(
            [int(row["negative_embedding_index"]) for row in rows], dtype=np.int64
        ),
        "target": np.asarray(
            [int(row["target_window_index"]) for row in rows], dtype=np.int64
        ),
        "negative_window": np.asarray(
            [int(row["negative_window_index"]) for row in rows], dtype=np.int64
        ),
    }


def _positive_mate_indices(
    examples: list[dict[str, Any]],
    labels: np.ndarray,
    pairs: list[dict[str, Any]],
    *,
    positions: np.ndarray | None = None,
    allowed: set[int] | None = None,
) -> np.ndarray:
    """Mate = another positive view of the same window. When `allowed` is
    given (training mode), mates are restricted to that example set so
    held-out windows/views never receive gradient."""

    positive_by_label: dict[int, list[int]] = defaultdict(list)
    for row in examples:
        if str(row.get("role", "")) == "positive":
            emb_idx = int(row["embedding_index"])
            if allowed is not None and emb_idx not in allowed:
                continue
            positive_by_label[int(row["label_window_index"])].append(emb_idx)
    if positions is None:
        rows = pairs
    else:
        rows = [pairs[int(position)] for position in positions]
    mates: list[int] = []
    for pair in rows:
        pos_idx = int(pair["positive_embedding_index"])
        label = int(labels[pos_idx])
        candidates = positive_by_label.get(label, [pos_idx])
        mate = next((idx for idx in candidates if idx != pos_idx), candidates[0])
        mates.append(int(mate))
    return np.asarray(mates, dtype=np.int64)


class TextSpaceAdapter(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(values), dim=-1)


class TextMemoryBridge(nn.Module):
    def __init__(self, embedding_dim: int, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(condition_dim)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _supervised_contrastive_loss(
    z: torch.Tensor, labels: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    sim = z @ z.T / max(float(temperature), 1e-6)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return torch.zeros((), dtype=z.dtype, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    log_denom = torch.logsumexp(sim, dim=1)
    log_pos = torch.logsumexp(sim.masked_fill(~positive, -1e9), dim=1)
    return -(log_pos[valid] - log_denom[valid]).mean()


def _module_outputs(
    module: nn.Module, x_all: torch.Tensor, example_count: int
) -> np.ndarray:
    was_training = module.training
    module.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, example_count, 2048):
            chunks.append(module(x_all[start : start + 2048]).detach().cpu().numpy())
    if was_training:
        module.train()
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _same_label_recall_at_1(
    vectors: np.ndarray, labels: np.ndarray, *, max_examples: int = 1500
) -> float:
    values = _safe_normalize_rows(vectors)
    labels = np.asarray(labels, dtype=np.int64)
    if values.shape[0] <= 1:
        return math.nan
    count = min(int(max_examples), values.shape[0])
    query_positions = np.linspace(0, values.shape[0] - 1, count, dtype=np.int64)
    hits = 0
    for idx in query_positions:
        scores = values @ values[int(idx)]
        scores[int(idx)] = -np.inf
        nearest = int(np.argmax(scores))
        hits += int(labels[nearest] == labels[int(idx)])
    return float(hits / max(len(query_positions), 1))


def _same_label_recall_at_k(
    vectors: np.ndarray,
    labels: np.ndarray,
    query_idx: np.ndarray,
    gallery_idx: np.ndarray,
    *,
    k: int,
    max_queries: int = 2000,
) -> float:
    if query_idx.shape[0] == 0 or gallery_idx.shape[0] == 0:
        return math.nan
    values = _safe_normalize_rows(vectors)
    labels = np.asarray(labels, dtype=np.int64)
    gallery = values[gallery_idx]
    gallery_labels = labels[gallery_idx]
    gallery_position = {int(emb): pos for pos, emb in enumerate(gallery_idx)}
    count = min(int(max_queries), query_idx.shape[0])
    chosen = np.linspace(0, query_idx.shape[0] - 1, count, dtype=np.int64)
    kk = min(int(k), gallery.shape[0])
    hits = 0
    for ci in chosen:
        emb = int(query_idx[int(ci)])
        scores = gallery @ values[emb]
        self_pos = gallery_position.get(emb)
        if self_pos is not None:
            scores[self_pos] = -np.inf
        if kk >= scores.shape[0]:
            top = np.arange(scores.shape[0])
        else:
            top = np.argpartition(scores, -kk)[-kk:]
        hits += int(np.any(gallery_labels[top] == labels[emb]))
    return float(hits / max(count, 1))


def _median_topk_temporal_distance(
    vectors: np.ndarray,
    labels: np.ndarray,
    query_idx: np.ndarray,
    gallery_idx: np.ndarray,
    *,
    k: int = 10,
    max_queries: int = 1000,
) -> float:
    if query_idx.shape[0] == 0 or gallery_idx.shape[0] == 0:
        return math.nan
    values = _safe_normalize_rows(vectors)
    labels = np.asarray(labels, dtype=np.int64)
    gallery = values[gallery_idx]
    gallery_labels = labels[gallery_idx]
    count = min(int(max_queries), query_idx.shape[0])
    chosen = np.linspace(0, query_idx.shape[0] - 1, count, dtype=np.int64)
    kk = min(int(k), gallery.shape[0])
    per_query: list[float] = []
    for ci in chosen:
        emb = int(query_idx[int(ci)])
        scores = gallery @ values[emb]
        if kk >= scores.shape[0]:
            top = np.arange(scores.shape[0])
        else:
            top = np.argpartition(scores, -kk)[-kk:]
        per_query.append(
            float(np.median(np.abs(gallery_labels[top] - labels[emb])))
        )
    return float(np.median(np.asarray(per_query, dtype=np.float64)))


def _text_pair_margin(
    vectors: np.ndarray,
    *,
    pair_indices: dict[str, np.ndarray],
    mate_indices: np.ndarray,
    max_pairs: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    values = _safe_normalize_rows(vectors)
    pair_count = int(pair_indices["positive"].shape[0])
    if pair_count == 0:
        return {"mean_pair_margin": math.nan, "pair_margin_positive_rate": math.nan}
    take = min(pair_count, int(max_pairs))
    if rng is not None and pair_count > take:
        chosen = rng.choice(pair_count, size=take, replace=False)
    else:
        chosen = np.arange(take, dtype=np.int64)
    pos = pair_indices["positive"][chosen]
    neg = pair_indices["negative"][chosen]
    mate = mate_indices[chosen]
    pos_sim = np.sum(values[pos] * values[mate], axis=1)
    neg_sim = np.sum(values[pos] * values[neg], axis=1)
    margins = pos_sim - neg_sim
    return {
        "mean_pair_margin": float(np.mean(margins)),
        "pair_margin_positive_rate": float(np.mean(margins > 0.0)),
    }


def _load_memory_targets(support_arrays_path: str | Path) -> np.ndarray:
    path = _resolve(support_arrays_path)
    with np.load(path) as payload:
        if "memory_targets" not in payload.files:
            raise ValueError(f"{path}: missing memory_targets")
        return np.asarray(payload["memory_targets"], dtype=np.float32)


def _memory_pair_margins(
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    pair_idx: dict[str, np.ndarray],
    *,
    max_pairs: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    pair_count = int(pair_idx["positive"].shape[0])
    if pair_count == 0:
        return {
            "source_hard_negative_margin_mean": math.nan,
            "source_hard_negative_margin_positive_rate": math.nan,
            "reciprocal_hard_negative_margin_mean": math.nan,
            "reciprocal_hard_negative_margin_positive_rate": math.nan,
        }
    pred = _safe_normalize_rows(condition_vectors)
    memory = _safe_normalize_rows(memory_targets)
    take = min(pair_count, int(max_pairs))
    if rng is not None and pair_count > take:
        chosen = rng.choice(pair_count, size=take, replace=False)
    else:
        chosen = np.arange(take, dtype=np.int64)
    pos = pair_idx["positive"][chosen]
    neg = pair_idx["negative"][chosen]
    target = pair_idx["target"][chosen]
    negative_window = pair_idx["negative_window"][chosen]
    source_target = np.sum(pred[pos] * memory[target], axis=1)
    source_negative = np.sum(pred[pos] * memory[negative_window], axis=1)
    reciprocal_target = np.sum(pred[neg] * memory[negative_window], axis=1)
    reciprocal_source = np.sum(pred[neg] * memory[target], axis=1)
    source_margin = source_target - source_negative
    reciprocal_margin = reciprocal_target - reciprocal_source
    return {
        "source_hard_negative_margin_mean": float(np.mean(source_margin)),
        "source_hard_negative_margin_positive_rate": float(np.mean(source_margin > 0.0)),
        "reciprocal_hard_negative_margin_mean": float(np.mean(reciprocal_margin)),
        "reciprocal_hard_negative_margin_positive_rate": float(
            np.mean(reciprocal_margin > 0.0)
        ),
    }


def _target_cosine_and_rank(
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    labels: np.ndarray,
    *,
    query_idx: np.ndarray | None = None,
    max_examples: int = 1000,
) -> dict[str, float]:
    pred = _safe_normalize_rows(condition_vectors)
    memory = _safe_normalize_rows(memory_targets)
    labels = np.asarray(labels, dtype=np.int64)
    if query_idx is None:
        pool = np.arange(pred.shape[0], dtype=np.int64)
    else:
        pool = np.asarray(query_idx, dtype=np.int64)
    if pool.shape[0] == 0:
        return {
            "target_cosine_mean": math.nan,
            "target_cosine_median": math.nan,
            "true_memory_rank_median": math.nan,
            "recall_at_10_true_memory": math.nan,
        }
    count = min(int(max_examples), pool.shape[0])
    chosen = pool[np.linspace(0, pool.shape[0] - 1, count, dtype=np.int64)]
    target_cos = np.sum(pred[chosen] * memory[labels[chosen]], axis=1)
    ranks: list[int] = []
    for idx in chosen:
        scores = memory @ pred[int(idx)]
        true_score = float(scores[int(labels[int(idx)])])
        ranks.append(int(np.sum(scores > true_score) + 1))
    return {
        "target_cosine_mean": float(np.mean(target_cos)),
        "target_cosine_median": float(np.median(target_cos)),
        "true_memory_rank_median": float(np.median(np.asarray(ranks, dtype=np.float32))),
        "recall_at_10_true_memory": float(np.mean(np.asarray(ranks) <= 10)),
    }


# ----------------------------------------------------------------------------
# Locality-soft (Track A T3) helpers — module level so the unit tests can call
# them in isolation. See docs/superpowers/plans/2026-06-17-locality-soft-retriever.md
# sections 1 (loss form), 3 (NV-filter decision), 4 (fit-gate metric).
# ----------------------------------------------------------------------------

def _assert_calibration_purge(val_window_ranges: str | None, purge_gap: int) -> None:
    """Guard: locality-soft must use the calibration purge or the pre-registered
    fit-gate X (T2) is meaningless. The binding spec ties X to exactly
    CAL_VAL_WINDOW_RANGES + CAL_PURGE_GAP. Fail loudly otherwise so a purge
    ablation cannot silently invalidate the gate comparison.
    """
    ranges = _parse_window_ranges(val_window_ranges)
    if ranges != list(CAL_VAL_WINDOW_RANGES) or int(purge_gap) != CAL_PURGE_GAP:
        raise ValueError(
            "locality-soft gate requires the calibration purge "
            f"(--val-window-ranges == {CAL_VAL_WINDOW_RANGES}, --purge-gap == "
            f"{CAL_PURGE_GAP}); got ranges={ranges}, purge_gap={purge_gap}. The "
            "pre-registered fit-gate X is only valid at the calibration purge."
        )


def _load_neighbor_cache(neighbors_npz: str | Path) -> dict[str, np.ndarray]:
    """Load memory_knn_neighbors.npz (per-window top-50 memory-KNN cache).

    Used ONLY to build the loss positives P(w) on TRAIN windows. The eval-side
    P(w) (fit gate) is recomputed by exact ranking over the train pool, never
    truncated from this full-bank/no-purge cache.
    """
    path = _resolve(neighbors_npz)
    with np.load(path) as payload:
        if "neighbor_indices" not in payload.files:
            raise ValueError(f"{path}: missing neighbor_indices")
        indices = np.asarray(payload["neighbor_indices"], dtype=np.int64)
        cosines = np.asarray(payload["neighbor_cosines"], dtype=np.float32)
    return {"neighbor_indices": indices, "neighbor_cosines": cosines}


def _neighborhood_positives(
    window: int,
    neighbor_cache: dict[str, np.ndarray],
    *,
    locality_k: int,
    allowed_windows: set[int] | None = None,
) -> np.ndarray:
    """Return P(w) = top-`locality_k` memory-KNN neighbors of `window`.

    Self is already excluded in the cache. When `allowed_windows` is given,
    neighbors outside that set (e.g. held-out windows for the loss path) are
    skipped before truncating to K, so a train-side positive never points at a
    held-out window.
    """
    nbr = neighbor_cache["neighbor_indices"][int(window)]
    if allowed_windows is not None:
        nbr = np.asarray([int(j) for j in nbr if int(j) in allowed_windows], dtype=np.int64)
    return np.asarray(nbr[: int(locality_k)], dtype=np.int64)


def _neighborhood_infonce_loss(
    pred: torch.Tensor,
    batch_windows: torch.Tensor,
    memory_norm: torch.Tensor,
    neighbor_cache: dict[str, np.ndarray],
    *,
    locality_k: int,
    tau: float,
    tau_loc: float,
    rho: float,
    allowed_windows: set[int] | None,
    device: torch.device,
) -> torch.Tensor:
    """Neighborhood-InfoNCE for the projected-memory bridge (plan section 1).

        L_nce = - log [  sum_{k in P(w)} pi_k * exp(s(c, m_k)/tau)
                        / ( sum_{k in P(w)} exp(s(c, m_k)/tau)
                            + sum_{n in N_rand'} exp(s(c, m_n)/tau) ) ]

    where pi_k = softmax(cos(m_w, m_k)/tau_loc) over P(w) (numerator only),
    P(w) = top-K memory-KNN of w, and N_rand' = the rest of the in-batch windows
    minus NV-filtered false negatives minus any n in P(w). `pred` is the bridge
    output for each batch row; positives/negatives are looked up against the
    (L2-normalized) memory bank `memory_norm`.
    """
    pred_norm = F.normalize(pred, dim=-1)
    windows = [int(w) for w in batch_windows.detach().cpu().tolist()]
    per_row: list[torch.Tensor] = []
    for row, w in enumerate(windows):
        pw = _neighborhood_positives(
            w, neighbor_cache, locality_k=locality_k, allowed_windows=allowed_windows
        )
        if pw.shape[0] == 0:
            continue
        c = pred_norm[row]
        pw_t = torch.from_numpy(pw).long().to(device)
        m_pos = memory_norm[pw_t]                      # (P, D), unit rows
        m_w = memory_norm[int(w)]                      # (D,)
        # pi_k = softmax over locality cos(m_w, m_k) / tau_loc (numerator weights)
        loc_cos = m_pos @ m_w                          # (P,)
        log_pi = F.log_softmax(loc_cos / max(float(tau_loc), 1e-6), dim=0)
        pos_logits = (m_pos @ c) / max(float(tau), 1e-6)  # (P,)
        # NV-Retriever false-negative threshold on in-batch random negatives only.
        max_pos_loc = float(loc_cos.max().item())
        # Exclude windows in P(w) and apply the NV false-negative mask.
        pw_window_set = set(int(x) for x in pw.tolist())
        neg_logits_list: list[torch.Tensor] = []
        for j, wj in enumerate(windows):
            if j == row or wj in pw_window_set:
                continue
            cos_nw = float((memory_norm[int(wj)] @ m_w).item())
            if cos_nw > float(rho) * max_pos_loc:
                continue  # false negative: effectively inside the neighborhood
            neg_logits_list.append((memory_norm[int(wj)] @ c) / max(float(tau), 1e-6))
        # numerator = sum_k pi_k * exp(pos_logit_k)  -> logsumexp(log_pi + pos_logits)
        log_num = torch.logsumexp(log_pi + pos_logits, dim=0)
        denom_terms = [pos_logits]
        if neg_logits_list:
            denom_terms.append(torch.stack(neg_logits_list))
        log_denom = torch.logsumexp(torch.cat(denom_terms), dim=0)
        per_row.append(-(log_num - log_denom))
    if not per_row:
        return torch.zeros((), dtype=pred.dtype, device=device)
    return torch.stack(per_row).mean()


def _neighborhood_supcon_loss(
    z: torch.Tensor,
    batch_windows: torch.Tensor,
    neighbor_cache: dict[str, np.ndarray],
    memory_norm: torch.Tensor,
    *,
    locality_k: int,
    tau: float,
    tau_loc: float,
    allowed_windows: set[int] | None,
    device: torch.device,
) -> torch.Tensor:
    """Soft-label SupCon for the text-space adapter (plan section 1, text variant).

    Positives for anchor i = batch rows whose window is the anchor's window OR a
    window in P(anchor_window); soft-weighted by pi_k = softmax(cos(m_w, m_k)/tau_loc)
    over P(w) (self gets the max neighborhood weight). `z` rows are unit vectors.
    """
    windows = [int(w) for w in batch_windows.detach().cpu().tolist()]
    n = len(windows)
    sim = z @ z.T / max(float(tau), 1e-6)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    sim = sim.masked_fill(eye, -1e9)
    log_denom = torch.logsumexp(sim, dim=1)
    per_row: list[torch.Tensor] = []
    for i, wi in enumerate(windows):
        pw = _neighborhood_positives(
            wi, neighbor_cache, locality_k=locality_k, allowed_windows=allowed_windows
        )
        m_wi = memory_norm[int(wi)]
        # weight map: window -> pi weight. self-window uses the max neighborhood weight.
        weight_by_window: dict[int, float] = {}
        if pw.shape[0] > 0:
            loc_cos = (memory_norm[torch.from_numpy(pw).long().to(device)] @ m_wi)
            pi = F.softmax(loc_cos / max(float(tau_loc), 1e-6), dim=0)
            pi_max = float(pi.max().item())
            for k_idx, wk in enumerate(pw.tolist()):
                weight_by_window[int(wk)] = float(pi[k_idx].item())
        else:
            pi_max = 1.0
        weight_by_window[int(wi)] = pi_max  # self is a guaranteed positive
        weights = torch.zeros(n, dtype=z.dtype, device=device)
        for j, wj in enumerate(windows):
            if j == i:
                continue
            if wj in weight_by_window:
                weights[j] = weight_by_window[wj]
        total = float(weights.sum().item())
        if total <= 0.0:
            continue
        weights = weights / total
        log_pos = torch.logsumexp(sim[i] + torch.log(weights + 1e-12), dim=0)
        per_row.append(-(log_pos - log_denom[i]))
    if not per_row:
        return torch.zeros((), dtype=z.dtype, device=device)
    return torch.stack(per_row).mean()


def _locality_recall_at_k(
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    labels: np.ndarray,
    query_idx: np.ndarray,
    *,
    locality_k: int,
    retrieval_top_k: int = CAL_RETRIEVAL_TOP_K,
    bank_size: int | None = None,
    val_window_ranges: list[tuple[int, int]] | None = None,
    purge_gap: int = CAL_PURGE_GAP,
) -> dict[str, float]:
    """locality-recall@K — byte-identical in definition to the T2 calibration.

    For each held-out query example at row r (window w = labels[r]):
      P(w) = top-`locality_k` memory-KNN of w within the TRAIN POOL (exact
             ranking, self excluded);
      retrieved = top-`retrieval_top_k` (FIXED at 10) train-pool windows by
             cosine to the bridge prediction c;
      hit = |retrieved INTERSECT P(w)| >= 1; also mean Jaccard.
    Train pool / val / purge masks come from build_masks over the full memory
    bank (4010 windows), using the SAME purge constants as the calibration.
    """
    memory = np.asarray(memory_targets, dtype=np.float32)
    normed = cal_l2_normalize(memory)                          # (N, D), unit rows
    pred = _safe_normalize_rows(np.asarray(condition_vectors, dtype=np.float32))
    labels = np.asarray(labels, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)
    n_bank = int(bank_size if bank_size is not None else memory.shape[0])
    ranges = (
        [tuple(r) for r in val_window_ranges]
        if val_window_ranges is not None
        else list(CAL_VAL_WINDOW_RANGES)
    )
    # Build train/val/purge masks identically to the calibration (build_masks
    # reads the module constants; when the caller overrides ranges/gap for a
    # synthetic test we construct the masks explicitly with the same semantics).
    if ranges == list(CAL_VAL_WINDOW_RANGES) and int(purge_gap) == CAL_PURGE_GAP:
        masks = cal_build_masks(n_bank)
    else:
        val = np.zeros(n_bank, dtype=bool)
        for lo, hi in ranges:
            val[lo:hi] = True
        purge = np.zeros(n_bank, dtype=bool)
        for lo, hi in ranges:
            purge[max(0, lo - int(purge_gap)) : min(n_bank, hi + int(purge_gap))] = True
        purge = purge & ~val
        masks = {"val": val, "purge": purge, "train": ~val & ~purge}
    train_idx = np.where(masks["train"])[0].astype(np.int64)

    pw_cache: dict[int, np.ndarray] = {}
    hits: list[float] = []
    jaccards: list[float] = []
    for r in query_idx.tolist():
        w = int(labels[int(r)])
        if w not in pw_cache:
            pw_cache[w] = cal_topk_in_pool(
                normed[w], normed, train_idx, int(locality_k), exclude=w
            )
        pw = pw_cache[w]
        if pw.shape[0] == 0:
            continue
        retrieved = cal_topk_in_pool(pred[int(r)], normed, train_idx, int(retrieval_top_k))
        inter = np.intersect1d(retrieved, pw, assume_unique=False).shape[0]
        union = np.union1d(retrieved, pw).shape[0]
        hits.append(1.0 if inter >= 1 else 0.0)
        jaccards.append(float(inter) / float(union) if union > 0 else 0.0)
    n = len(hits)
    return {
        "locality_recall_at_K": float(np.mean(hits)) if n else float("nan"),
        "locality_mean_jaccard": float(np.mean(jaccards)) if n else float("nan"),
        "locality_n_eval": int(n),
        "locality_k": int(locality_k),
        "retrieval_top_k": int(retrieval_top_k),
    }


def _save_checkpoint(
    state_dict: dict[str, torch.Tensor],
    path: Path,
    *,
    config: dict[str, Any],
    seed: int,
    lr: float,
    argv: list[str] | None,
    best_step: int | None,
) -> None:
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
            "config": config,
            "seed": int(seed),
            "lr": float(lr),
            "argv": list(argv) if argv else [],
            "best_step": best_step,
            "git_sha": _git_sha(),
        },
        path,
    )


def _run_training_loop(
    *,
    module: nn.Module,
    opt: torch.optim.Optimizer,
    steps: int,
    step_fn: Callable[[int], dict[str, float]],
    selection_fn: Callable[[], float] | None,
    higher_is_better: bool,
    eval_every: int,
    patience: int,
) -> dict[str, Any]:
    """Shared fixed-step / validation-coupled loop. step_fn performs one
    optimizer step and returns the loss components to trace."""

    losses: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    best_metric: float | None = None
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_evals = 0
    early_stopped = False
    stopped_step = 0
    use_early_stop = (
        selection_fn is not None and int(eval_every) > 0 and int(patience) > 0
    )
    for step in range(int(steps)):
        components = step_fn(step)
        stopped_step = step + 1
        if step == 0 or (step + 1) % LOSS_TRACE_EVERY == 0 or step == int(steps) - 1:
            losses.append({"step": int(step + 1), **components})
        if use_early_stop and (step + 1) % int(eval_every) == 0:
            metric = float(selection_fn())
            eval_history.append({"step": int(step + 1), "metric": metric})
            improved = best_metric is None or (
                metric > best_metric if higher_is_better else metric < best_metric
            )
            if improved:
                best_metric = metric
                best_step = int(step + 1)
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in module.state_dict().items()}
                )
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= int(patience):
                    early_stopped = True
                    if losses and losses[-1]["step"] != stopped_step:
                        losses.append({"step": int(stopped_step), **components})
                    break
    final_state = copy.deepcopy(
        {k: v.detach().cpu() for k, v in module.state_dict().items()}
    )
    if best_state is None:
        best_state = final_state
        best_step = stopped_step if stopped_step else None
    return {
        "losses": losses,
        "eval_history": eval_history,
        "early_stopped": early_stopped,
        "stopped_step": int(stopped_step),
        "best_step": best_step,
        "best_metric": best_metric,
        "best_state": best_state,
        "final_state": final_state,
    }


def train_text_space_from_manifest(
    *,
    examples_jsonl: str | Path,
    pairs_jsonl: str | Path,
    output_dir: str | Path,
    embedding_cache_dir: str | Path | None = None,
    embedding_backend: str = "hash",
    embedding_model: str = "text-embedding-3-large",
    dotenv_path: str = ".env",
    embedding_batch_size: int = 128,
    hash_dim: int = 256,
    steps: int = 100,
    batch_size: int = 256,
    adapter_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.07,
    pair_margin: float = 0.15,
    pair_weight: float = 0.5,
    seed: int = 0,
    device: str | torch.device | None = None,
    max_targets: int | None = None,
    val_window_ranges: str | None = None,
    purge_gap: int = 0,
    holdout_view_families: str | None = None,
    eval_every: int = 0,
    patience: int = 0,
    locality_soft: bool = False,
    locality_k: int = 5,
    tau_loc: float = 0.10,
    nv_false_neg_rho: float = 0.95,
    neighbors_npz: str | Path | None = None,
    support_arrays_path: str | Path | None = None,
    argv_record: list[str] | None = None,
) -> dict[str, Any]:
    output = _resolve(output_dir)
    embedding_output = _resolve(embedding_cache_dir) if embedding_cache_dir else output
    output.mkdir(parents=True, exist_ok=True)
    if locality_soft:
        if not neighbors_npz:
            raise ValueError("--locality-soft requires --neighbors-npz")
        if not support_arrays_path:
            raise ValueError(
                "text-space locality-soft requires support_arrays_path for memory geometry"
            )
        _assert_calibration_purge(val_window_ranges, purge_gap)
    data = load_manifest_training_data(
        examples_jsonl=examples_jsonl,
        pairs_jsonl=pairs_jsonl,
        max_targets=max_targets,
    )
    examples = data["examples"]
    pairs = data["pairs"]
    labels_np = np.asarray(data["labels"], dtype=np.int64)
    views = list(data["views"])
    split = _build_holdout_split(
        examples,
        pairs,
        val_window_ranges=val_window_ranges,
        purge_gap=purge_gap,
        holdout_view_families=holdout_view_families,
    )
    _write_json(
        output / "holdout_split.json",
        _holdout_split_payload(split, total_examples=len(examples), total_pairs=len(pairs)),
    )
    train_idx = split["train_example_idx"]
    val_idx = split["val_example_idx"]
    tierb_idx = split["tierb_query_idx"]
    train_pair_positions = split["train_pair_positions"]
    val_pair_positions = split["val_pair_positions"]
    if split["enabled"] and train_idx.shape[0] < 2:
        raise ValueError("holdout split leaves fewer than 2 training examples")
    if split["enabled"] and train_pair_positions.shape[0] < 1:
        raise ValueError("holdout split leaves no training pair rows")

    embeddings, embedding_meta = _embed_manifest_texts(
        texts=data["texts"],
        output_dir=embedding_output,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        dotenv_path=dotenv_path,
        embedding_batch_size=embedding_batch_size,
        hash_dim=hash_dim,
    )
    train_allowed = {int(idx) for idx in train_idx}
    train_pair_idx = _pair_arrays(pairs, train_pair_positions)
    train_mate_idx = _positive_mate_indices(
        examples, labels_np, pairs, positions=train_pair_positions, allowed=train_allowed
    )
    val_pair_idx = _pair_arrays(pairs, val_pair_positions)
    val_mate_idx = _positive_mate_indices(
        examples, labels_np, pairs, positions=val_pair_positions
    )
    rng = np.random.default_rng(int(seed))
    eval_rng = np.random.default_rng(int(seed) + 99991)
    torch.manual_seed(int(seed))
    device_t = torch.device(device or "cpu")
    x_all = torch.from_numpy(embeddings).float().to(device_t)
    labels_t_all = torch.from_numpy(labels_np).long().to(device_t)
    adapter = TextSpaceAdapter(
        embedding_dim=embeddings.shape[1],
        output_dim=int(adapter_dim),
        hidden_dim=int(hidden_dim),
    ).to(device_t)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
    example_count = embeddings.shape[0]
    train_count = int(train_idx.shape[0])
    train_pair_count = int(train_pair_positions.shape[0])
    batch_n = min(max(2, int(batch_size)), train_count)
    pair_batch_n = min(max(1, int(batch_size) // 2), train_pair_count)
    all_idx = np.arange(example_count, dtype=np.int64)

    # Locality-soft (T3, text variant): neighbor cache + memory geometry for the
    # soft-label SupCon. Neighborhood P(w) is defined in 734a memory space, over
    # the SAME full build_masks(4010) train pool as the fit gate (see the
    # projected-memory note above) — not the manifest-window subset.
    neighbor_cache: dict[str, np.ndarray] | None = None
    allowed_train_windows: set[int] | None = None
    memory_norm_t: torch.Tensor | None = None
    if locality_soft:
        neighbor_cache = _load_neighbor_cache(neighbors_npz)
        memory_targets_ts = _load_memory_targets(support_arrays_path)
        gate_train_mask = cal_build_masks(memory_targets_ts.shape[0])["train"]
        allowed_train_windows = {int(i) for i in np.where(gate_train_mask)[0]}
        memory_norm_t = F.normalize(
            torch.from_numpy(memory_targets_ts).float().to(device_t), dim=-1
        )

    def step_fn(step: int) -> dict[str, float]:
        chosen_pos = rng.choice(train_count, size=batch_n, replace=batch_n > train_count)
        chosen = train_idx[chosen_pos]
        chosen_t = torch.from_numpy(chosen.astype(np.int64)).long().to(device_t)
        z = adapter(x_all[chosen_t])
        if locality_soft:
            supcon = _neighborhood_supcon_loss(
                z,
                labels_t_all[chosen_t],
                neighbor_cache,
                memory_norm_t,
                locality_k=int(locality_k),
                tau=float(temperature),
                tau_loc=float(tau_loc),
                allowed_windows=allowed_train_windows,
                device=device_t,
            )
        else:
            supcon = _supervised_contrastive_loss(
                z, labels_t_all[chosen_t], temperature=float(temperature)
            )
        pair_choice = rng.choice(
            train_pair_count, size=pair_batch_n, replace=pair_batch_n > train_pair_count
        )
        pos_t = torch.from_numpy(train_pair_idx["positive"][pair_choice]).long().to(device_t)
        neg_t = torch.from_numpy(train_pair_idx["negative"][pair_choice]).long().to(device_t)
        mate_t = torch.from_numpy(train_mate_idx[pair_choice]).long().to(device_t)
        pos_z = adapter(x_all[pos_t])
        neg_z = adapter(x_all[neg_t])
        mate_z = adapter(x_all[mate_t])
        pos_sim = torch.sum(pos_z * mate_z, dim=-1)
        neg_sim = torch.sum(pos_z * neg_z, dim=-1)
        margin_loss = F.relu(float(pair_margin) + neg_sim - pos_sim).mean()
        loss = supcon + float(pair_weight) * margin_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
        opt.step()
        return {
            "loss": float(loss.detach().cpu()),
            "supcon": float(supcon.detach().cpu()),
            "pair_margin": float(margin_loss.detach().cpu()),
        }

    selection_fn: Callable[[], float] | None = None
    if tierb_idx.shape[0] > 0 and train_idx.shape[0] > 0:
        def selection_fn() -> float:  # noqa: F811
            adapted_now = _module_outputs(adapter, x_all, example_count)
            return _same_label_recall_at_k(
                adapted_now, labels_np, tierb_idx, train_idx, k=10
            )
    elif val_idx.shape[0] > 0:
        def selection_fn() -> float:  # noqa: F811
            adapted_now = _module_outputs(adapter, x_all, example_count)
            return _same_label_recall_at_k(
                adapted_now, labels_np, val_idx, all_idx, k=10
            )

    loop = _run_training_loop(
        module=adapter,
        opt=opt,
        steps=int(steps),
        step_fn=step_fn,
        selection_fn=selection_fn,
        higher_is_better=True,
        eval_every=int(eval_every),
        patience=int(patience),
    )

    checkpoint_config = {
        "method": "text_space_contrastive_14x14",
        "embedding_dim": int(embeddings.shape[1]),
        "adapter_dim": int(adapter_dim),
        "hidden_dim": int(hidden_dim),
        "temperature": float(temperature),
        "pair_margin": float(pair_margin),
        "pair_weight": float(pair_weight),
        "val_window_ranges": split["val_window_ranges"],
        "purge_gap": int(split["purge_gap"]),
        "holdout_view_families": list(split["holdout_view_families"]),
        "embedding_backend": str(embedding_backend),
        "embedding_model": str(embedding_model),
        "locality_soft": bool(locality_soft),
        "locality_k": int(locality_k),
        "tau_loc": float(tau_loc),
        "nv_false_neg_rho": float(nv_false_neg_rho),
    }
    _save_checkpoint(
        loop["final_state"],
        output / "text_space_adapter_final.pt",
        config=checkpoint_config,
        seed=int(seed),
        lr=float(lr),
        argv=argv_record,
        best_step=loop["best_step"],
    )
    _save_checkpoint(
        loop["best_state"],
        output / "text_space_adapter_best.pt",
        config=checkpoint_config,
        seed=int(seed),
        lr=float(lr),
        argv=argv_record,
        best_step=loop["best_step"],
    )

    def _evaluate(state: dict[str, torch.Tensor]) -> dict[str, Any]:
        adapter.load_state_dict(state)
        adapted_now = _module_outputs(adapter, x_all, example_count)
        evaluation: dict[str, Any] = {
            "raw_same_label_recall_at_1": _same_label_recall_at_1(embeddings, labels_np),
            "adapted_same_label_recall_at_1": _same_label_recall_at_1(
                adapted_now, labels_np
            ),
            **_text_pair_margin(
                adapted_now,
                pair_indices=train_pair_idx,
                mate_indices=train_mate_idx,
                rng=np.random.default_rng(int(seed) + 424243),
            ),
        }
        if val_idx.shape[0] > 0:
            evaluation["heldout_same_label_recall_at_10"] = _same_label_recall_at_k(
                adapted_now, labels_np, val_idx, all_idx, k=10
            )
            evaluation["heldout_same_label_recall_at_1"] = _same_label_recall_at_k(
                adapted_now, labels_np, val_idx, all_idx, k=1
            )
            evaluation["heldout_median_top10_temporal_distance_raw"] = (
                _median_topk_temporal_distance(
                    embeddings, labels_np, val_idx, train_idx, k=10
                )
            )
            evaluation["heldout_median_top10_temporal_distance_adapted"] = (
                _median_topk_temporal_distance(
                    adapted_now, labels_np, val_idx, train_idx, k=10
                )
            )
        if val_pair_positions.shape[0] > 0:
            heldout_margins = _text_pair_margin(
                adapted_now, pair_indices=val_pair_idx, mate_indices=val_mate_idx
            )
            evaluation["heldout_mean_pair_margin"] = heldout_margins["mean_pair_margin"]
            evaluation["heldout_pair_margin_positive_rate"] = heldout_margins[
                "pair_margin_positive_rate"
            ]
        if tierb_idx.shape[0] > 0 and train_idx.shape[0] > 0:
            evaluation["heldout_view_same_label_recall_at_10"] = _same_label_recall_at_k(
                adapted_now, labels_np, tierb_idx, train_idx, k=10
            )
            per_family: dict[str, dict[str, float]] = {}
            tierb_set = set(int(idx) for idx in tierb_idx)
            family_members: dict[str, list[int]] = defaultdict(list)
            for row in examples:
                emb_idx = int(row["embedding_index"])
                if emb_idx in tierb_set:
                    family_members[str(row.get("view_name", ""))].append(emb_idx)
            for family, members in sorted(family_members.items()):
                member_idx = np.asarray(members, dtype=np.int64)
                per_family[family] = {
                    "recall_at_1": _same_label_recall_at_k(
                        adapted_now, labels_np, member_idx, train_idx, k=1
                    ),
                    "recall_at_10": _same_label_recall_at_k(
                        adapted_now, labels_np, member_idx, train_idx, k=10
                    ),
                    "query_count": int(member_idx.shape[0]),
                }
            evaluation["heldout_view_recall"] = per_family
        return evaluation, adapted_now

    evaluation_final, _ = _evaluate(loop["final_state"])
    evaluation_best, adapted = _evaluate(loop["best_state"])

    report = {
        "schema_version": "nl_14x14_text_space_contrastive_report_v2",
        "status": "pass",
        "method": "text_space_contrastive_14x14",
        "argv": list(argv_record) if argv_record else [],
        "training": {
            "example_count": int(train_count),
            "total_example_count": int(example_count),
            "pair_rows_consumed": int(train_pair_count),
            "steps": int(steps),
            "stopped_step": loop["stopped_step"],
            "early_stopped": loop["early_stopped"],
            "best_step": loop["best_step"],
            "best_metric": loop["best_metric"],
            "eval_history": loop["eval_history"],
            "eval_every": int(eval_every),
            "patience": int(patience),
            "batch_size": int(batch_n),
            "pair_batch_size": int(pair_batch_n),
            "adapter_dim": int(adapter_dim),
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "seed": int(seed),
            "loss_trace": loop["losses"],
        },
        "holdout": _holdout_split_payload(
            split, total_examples=len(examples), total_pairs=len(pairs)
        ),
        "embedding": embedding_meta,
        "evaluation": evaluation_best,
        "evaluation_final": evaluation_final,
        "artifact_paths": {
            "report": str(output / "text_space_training_report.json"),
            "arrays": str(output / "text_space_training_arrays.npz"),
            "checkpoint_best": str(output / "text_space_adapter_best.pt"),
            "checkpoint_final": str(output / "text_space_adapter_final.pt"),
            "holdout_split": str(output / "holdout_split.json"),
        },
    }
    np.savez_compressed(
        output / "text_space_training_arrays.npz",
        embeddings=embeddings,
        adapted_embeddings=adapted,
        labels=labels_np,
        pair_positive_indices=train_pair_idx["positive"],
        pair_negative_indices=train_pair_idx["negative"],
        train_example_idx=train_idx,
        val_example_idx=val_idx,
        tierb_query_idx=tierb_idx,
    )
    _write_json(output / "text_space_training_report.json", report)
    return report


def train_projected_memory_from_manifest(
    *,
    examples_jsonl: str | Path,
    pairs_jsonl: str | Path,
    support_arrays_path: str | Path,
    output_dir: str | Path,
    embedding_cache_dir: str | Path | None = None,
    embedding_backend: str = "hash",
    embedding_model: str = "text-embedding-3-large",
    dotenv_path: str = ".env",
    embedding_batch_size: int = 128,
    hash_dim: int = 256,
    steps: int = 100,
    batch_size: int = 256,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.2,
    contrastive_weight: float = 0.2,
    pair_weight: float = 0.5,
    reciprocal_pair_weight: float = 0.25,
    pair_margin: float = 0.15,
    contrastive_temperature: float = 0.07,
    seed: int = 0,
    device: str | torch.device | None = None,
    max_targets: int | None = None,
    val_window_ranges: str | None = None,
    purge_gap: int = 0,
    holdout_view_families: str | None = None,
    eval_every: int = 0,
    patience: int = 0,
    locality_soft: bool = False,
    locality_k: int = 5,
    tau_loc: float = 0.10,
    nv_false_neg_rho: float = 0.95,
    neighbors_npz: str | Path | None = None,
    locality_mse_weight: float = 0.1,
    argv_record: list[str] | None = None,
) -> dict[str, Any]:
    output = _resolve(output_dir)
    embedding_output = _resolve(embedding_cache_dir) if embedding_cache_dir else output
    output.mkdir(parents=True, exist_ok=True)
    if locality_soft:
        if not neighbors_npz:
            raise ValueError("--locality-soft requires --neighbors-npz")
        _assert_calibration_purge(val_window_ranges, purge_gap)
    data = load_manifest_training_data(
        examples_jsonl=examples_jsonl,
        pairs_jsonl=pairs_jsonl,
        max_targets=max_targets,
    )
    examples = data["examples"]
    pairs = data["pairs"]
    labels_np = np.asarray(data["labels"], dtype=np.int64)
    memory_targets = _load_memory_targets(support_arrays_path)
    if int(np.max(labels_np)) >= memory_targets.shape[0]:
        raise ValueError("manifest label_window_index exceeds memory_targets rows")
    split = _build_holdout_split(
        examples,
        pairs,
        val_window_ranges=val_window_ranges,
        purge_gap=purge_gap,
        holdout_view_families=holdout_view_families,
    )
    _write_json(
        output / "holdout_split.json",
        _holdout_split_payload(split, total_examples=len(examples), total_pairs=len(pairs)),
    )
    train_idx = split["train_example_idx"]
    val_idx = split["val_example_idx"]
    tierb_idx = split["tierb_query_idx"]
    train_pair_positions = split["train_pair_positions"]
    val_pair_positions = split["val_pair_positions"]
    if split["enabled"] and train_idx.shape[0] < 2:
        raise ValueError("holdout split leaves fewer than 2 training examples")
    if split["enabled"] and train_pair_positions.shape[0] < 1:
        raise ValueError("holdout split leaves no training pair rows")

    embeddings, embedding_meta = _embed_manifest_texts(
        texts=data["texts"],
        output_dir=embedding_output,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        dotenv_path=dotenv_path,
        embedding_batch_size=embedding_batch_size,
        hash_dim=hash_dim,
    )
    train_pair_idx = _pair_arrays(pairs, train_pair_positions)
    val_pair_idx = _pair_arrays(pairs, val_pair_positions)
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    device_t = torch.device(device or "cpu")
    x_all = torch.from_numpy(embeddings).float().to(device_t)
    y_all = torch.from_numpy(memory_targets).float().to(device_t)
    labels_t_all = torch.from_numpy(labels_np).long().to(device_t)
    bridge = TextMemoryBridge(
        embedding_dim=embeddings.shape[1],
        condition_dim=memory_targets.shape[1],
        hidden_dim=int(hidden_dim),
    ).to(device_t)
    opt = torch.optim.AdamW(bridge.parameters(), lr=float(lr), weight_decay=1e-4)
    example_count = embeddings.shape[0]
    train_count = int(train_idx.shape[0])
    train_pair_count = int(train_pair_positions.shape[0])
    batch_n = min(max(2, int(batch_size)), train_count)
    pair_batch_n = min(max(1, int(batch_size) // 2), train_pair_count)

    # Locality-soft (T3): neighbor cache for P(w) loss positives + the set of
    # train windows the loss may point a positive at. CRITICAL: this pool MUST
    # equal the fit-gate P(w) pool (the full build_masks(4010) train mask, ~3165
    # windows) so T5 optimizes the SAME neighborhood T6 gates on. It is NOT the
    # manifest-window subset (memory vectors exist for all 4010 windows; restricting
    # to windows-with-narratives would train toward a different, thinner pool than
    # the gate scores). Plan section 1 -> "train pool (purge-respecting; Section 4)".
    neighbor_cache: dict[str, np.ndarray] | None = None
    allowed_train_windows: set[int] | None = None
    memory_norm_t: torch.Tensor | None = None
    if locality_soft:
        neighbor_cache = _load_neighbor_cache(neighbors_npz)
        gate_train_mask = cal_build_masks(memory_targets.shape[0])["train"]
        allowed_train_windows = {
            int(i) for i in np.where(gate_train_mask)[0]
        }
        memory_norm_t = F.normalize(y_all, dim=-1)

    def step_fn(step: int) -> dict[str, float]:
        chosen_pos = rng.choice(train_count, size=batch_n, replace=batch_n > train_count)
        chosen = train_idx[chosen_pos]
        chosen_t = torch.from_numpy(chosen.astype(np.int64)).long().to(device_t)
        batch_labels = labels_t_all[chosen_t]
        pred = bridge(x_all[chosen_t])
        target = y_all[batch_labels]

        pair_choice = rng.choice(
            train_pair_count, size=pair_batch_n, replace=pair_batch_n > train_pair_count
        )
        pos_t = torch.from_numpy(train_pair_idx["positive"][pair_choice]).long().to(device_t)
        neg_t = torch.from_numpy(train_pair_idx["negative"][pair_choice]).long().to(device_t)
        target_t = torch.from_numpy(train_pair_idx["target"][pair_choice]).long().to(device_t)
        neg_window_t = (
            torch.from_numpy(train_pair_idx["negative_window"][pair_choice])
            .long()
            .to(device_t)
        )
        pred_pos = bridge(x_all[pos_t])
        pred_neg = bridge(x_all[neg_t])
        m_i = y_all[target_t]
        m_j = y_all[neg_window_t]
        # Curated mechanism margins — UNCHANGED in both objectives.
        source_margin_loss = F.relu(
            float(pair_margin)
            + F.cosine_similarity(pred_pos, m_j, dim=-1)
            - F.cosine_similarity(pred_pos, m_i, dim=-1)
        ).mean()
        reciprocal_margin_loss = F.relu(
            float(pair_margin)
            + F.cosine_similarity(pred_neg, m_i, dim=-1)
            - F.cosine_similarity(pred_neg, m_j, dim=-1)
        ).mean()

        if locality_soft:
            # Total = L_nce + pair_w*L_src + recip_w*L_rec + locality_mse_w*L_mse.
            # The exact-window InfoNCE + cosine + heavy MSE are REPLACED by the
            # neighborhood-InfoNCE; MSE becomes a light anchor to memory_w.
            nce = _neighborhood_infonce_loss(
                pred,
                batch_labels,
                memory_norm_t,
                neighbor_cache,
                locality_k=int(locality_k),
                tau=float(contrastive_temperature),
                tau_loc=float(tau_loc),
                rho=float(nv_false_neg_rho),
                allowed_windows=allowed_train_windows,
                device=device_t,
            )
            mse = F.mse_loss(pred, target)
            loss = (
                nce
                + float(pair_weight) * source_margin_loss
                + float(reciprocal_pair_weight) * reciprocal_margin_loss
                + float(locality_mse_weight) * mse
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=5.0)
            opt.step()
            return {
                "loss": float(loss.detach().cpu()),
                "neighborhood_infonce": float(nce.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "source_margin": float(source_margin_loss.detach().cpu()),
                "reciprocal_margin": float(reciprocal_margin_loss.detach().cpu()),
            }

        mse = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        unique_targets, inverse = torch.unique(
            batch_labels, sorted=True, return_inverse=True
        )
        logits = (
            F.normalize(pred, dim=-1) @ F.normalize(y_all[unique_targets], dim=-1).T
        ) / max(float(contrastive_temperature), 1e-6)
        contrastive = F.cross_entropy(logits, inverse)
        loss = (
            float(mse_weight) * mse
            + float(cosine_weight) * cosine
            + float(contrastive_weight) * contrastive
            + float(pair_weight) * source_margin_loss
            + float(reciprocal_pair_weight) * reciprocal_margin_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=5.0)
        opt.step()
        return {
            "loss": float(loss.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "cosine": float(cosine.detach().cpu()),
            "contrastive": float(contrastive.detach().cpu()),
            "source_margin": float(source_margin_loss.detach().cpu()),
            "reciprocal_margin": float(reciprocal_margin_loss.detach().cpu()),
        }

    selection_fn: Callable[[], float] | None = None
    selection_higher_is_better = False
    if locality_soft and val_idx.shape[0] > 0:
        # T5: early-stop on held-out locality-recall@K (higher is better),
        # computed identically to the T2 fit gate.
        def selection_fn() -> float:  # noqa: F811
            condition_now = _module_outputs(bridge, x_all, example_count)
            stats = _locality_recall_at_k(
                condition_now,
                memory_targets,
                labels_np,
                val_idx,
                locality_k=int(locality_k),
            )
            return stats["locality_recall_at_K"]
        selection_higher_is_better = True
    elif val_idx.shape[0] > 0:
        def selection_fn() -> float:  # noqa: F811
            condition_now = _module_outputs(bridge, x_all, example_count)
            stats = _target_cosine_and_rank(
                condition_now,
                memory_targets,
                labels_np,
                query_idx=val_idx,
                max_examples=1500,
            )
            return stats["true_memory_rank_median"]

    loop = _run_training_loop(
        module=bridge,
        opt=opt,
        steps=int(steps),
        step_fn=step_fn,
        selection_fn=selection_fn,
        higher_is_better=selection_higher_is_better,
        eval_every=int(eval_every),
        patience=int(patience),
    )

    checkpoint_config = {
        "method": "projected_memory_14x14",
        "embedding_dim": int(embeddings.shape[1]),
        "condition_dim": int(memory_targets.shape[1]),
        "hidden_dim": int(hidden_dim),
        "mse_weight": float(mse_weight),
        "cosine_weight": float(cosine_weight),
        "contrastive_weight": float(contrastive_weight),
        "pair_weight": float(pair_weight),
        "reciprocal_pair_weight": float(reciprocal_pair_weight),
        "pair_margin": float(pair_margin),
        "contrastive_temperature": float(contrastive_temperature),
        "val_window_ranges": split["val_window_ranges"],
        "purge_gap": int(split["purge_gap"]),
        "holdout_view_families": list(split["holdout_view_families"]),
        "embedding_backend": str(embedding_backend),
        "embedding_model": str(embedding_model),
        "locality_soft": bool(locality_soft),
        "locality_k": int(locality_k),
        "tau_loc": float(tau_loc),
        "nv_false_neg_rho": float(nv_false_neg_rho),
        "locality_mse_weight": float(locality_mse_weight),
    }
    _save_checkpoint(
        loop["final_state"],
        output / "projected_memory_bridge_final.pt",
        config=checkpoint_config,
        seed=int(seed),
        lr=float(lr),
        argv=argv_record,
        best_step=loop["best_step"],
    )
    _save_checkpoint(
        loop["best_state"],
        output / "projected_memory_bridge_best.pt",
        config=checkpoint_config,
        seed=int(seed),
        lr=float(lr),
        argv=argv_record,
        best_step=loop["best_step"],
    )

    def _evaluate(state: dict[str, torch.Tensor]) -> tuple[dict[str, Any], np.ndarray]:
        bridge.load_state_dict(state)
        condition_now = _module_outputs(bridge, x_all, example_count)
        evaluation: dict[str, Any] = {
            **_target_cosine_and_rank(condition_now, memory_targets, labels_np),
            **_memory_pair_margins(
                condition_now,
                memory_targets,
                train_pair_idx,
                rng=np.random.default_rng(int(seed) + 424243),
            ),
        }
        if val_idx.shape[0] > 0:
            heldout_stats = _target_cosine_and_rank(
                condition_now,
                memory_targets,
                labels_np,
                query_idx=val_idx,
                max_examples=1500,
            )
            evaluation["heldout_target_cosine_mean"] = heldout_stats["target_cosine_mean"]
            evaluation["heldout_target_cosine_median"] = heldout_stats[
                "target_cosine_median"
            ]
            evaluation["heldout_true_memory_rank_median"] = heldout_stats[
                "true_memory_rank_median"
            ]
            evaluation["heldout_recall_at_10_true_memory"] = heldout_stats[
                "recall_at_10_true_memory"
            ]
            # Locality-recall@K fit-gate metric (reported alongside the exact-window
            # rank/recall). Computed for BOTH objectives so the clean restamped
            # baseline and the locality-soft candidate are gate-comparable, but it
            # only requires neighbors when locality-soft is on. Definition is
            # byte-identical to the T2 calibration.
            if locality_soft:
                loc_stats = _locality_recall_at_k(
                    condition_now,
                    memory_targets,
                    labels_np,
                    val_idx,
                    locality_k=int(locality_k),
                )
                evaluation["heldout_locality_recall_at_K"] = loc_stats[
                    "locality_recall_at_K"
                ]
                evaluation["heldout_locality_mean_jaccard"] = loc_stats[
                    "locality_mean_jaccard"
                ]
                evaluation["heldout_locality_n_eval"] = loc_stats["locality_n_eval"]
                evaluation["heldout_locality_k"] = loc_stats["locality_k"]
                evaluation["heldout_locality_retrieval_top_k"] = loc_stats[
                    "retrieval_top_k"
                ]
        if val_pair_positions.shape[0] > 0:
            heldout_margins = _memory_pair_margins(
                condition_now, memory_targets, val_pair_idx
            )
            for key, value in heldout_margins.items():
                evaluation[f"heldout_{key}"] = value
        if tierb_idx.shape[0] > 0:
            tierb_stats = _target_cosine_and_rank(
                condition_now,
                memory_targets,
                labels_np,
                query_idx=tierb_idx,
                max_examples=1500,
            )
            evaluation["heldout_view_true_memory_rank_median"] = tierb_stats[
                "true_memory_rank_median"
            ]
            evaluation["heldout_view_recall_at_10_true_memory"] = tierb_stats[
                "recall_at_10_true_memory"
            ]
        return evaluation, condition_now

    evaluation_final, _ = _evaluate(loop["final_state"])
    evaluation_best, condition_vectors = _evaluate(loop["best_state"])

    report = {
        "schema_version": "nl_14x14_projected_memory_report_v2",
        "status": "pass",
        "method": "projected_memory_14x14",
        "argv": list(argv_record) if argv_record else [],
        "training": {
            "example_count": int(train_count),
            "total_example_count": int(example_count),
            "pair_rows_consumed": int(train_pair_count),
            "steps": int(steps),
            "stopped_step": loop["stopped_step"],
            "early_stopped": loop["early_stopped"],
            "best_step": loop["best_step"],
            "best_metric": loop["best_metric"],
            "eval_history": loop["eval_history"],
            "eval_every": int(eval_every),
            "patience": int(patience),
            "batch_size": int(batch_n),
            "pair_batch_size": int(pair_batch_n),
            "condition_dim": int(memory_targets.shape[1]),
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "seed": int(seed),
            "loss_trace": loop["losses"],
            "loss_terms": {
                "mse_weight": float(mse_weight),
                "cosine_weight": float(cosine_weight),
                "contrastive_weight": float(contrastive_weight),
                "pair_weight": float(pair_weight),
                "reciprocal_pair_weight": float(reciprocal_pair_weight),
                "pair_margin": float(pair_margin),
            },
        },
        "holdout": _holdout_split_payload(
            split, total_examples=len(examples), total_pairs=len(pairs)
        ),
        "embedding": embedding_meta,
        "evaluation": evaluation_best,
        "evaluation_final": evaluation_final,
        "artifact_paths": {
            "report": str(output / "projected_memory_training_report.json"),
            "arrays": str(output / "projected_memory_training_arrays.npz"),
            "checkpoint_best": str(output / "projected_memory_bridge_best.pt"),
            "checkpoint_final": str(output / "projected_memory_bridge_final.pt"),
            "holdout_split": str(output / "holdout_split.json"),
        },
    }
    np.savez_compressed(
        output / "projected_memory_training_arrays.npz",
        embeddings=embeddings,
        condition_vectors=condition_vectors,
        labels=labels_np,
        memory_targets=memory_targets,
        pair_positive_indices=train_pair_idx["positive"],
        pair_negative_indices=train_pair_idx["negative"],
        pair_target_indices=train_pair_idx["target"],
        pair_negative_window_indices=train_pair_idx["negative_window"],
        train_example_idx=train_idx,
        val_example_idx=val_idx,
        tierb_query_idx=tierb_idx,
    )
    _write_json(output / "projected_memory_training_report.json", report)
    return report


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    methods: list[str] = []
    argv_record = list(getattr(args, "argv_record", None) or [])
    report: dict[str, Any] = {
        "schema_version": "nl_14x14_manifest_retrieval_training_run_v2",
        "status": "pass",
        "methods": methods,
        "argv": argv_record,
        "artifact_paths": {
            "report": str(output / "training_run_report.json"),
            "markdown": str(output / "training_run_report.md"),
        },
    }
    common = {
        "examples_jsonl": args.examples_jsonl,
        "pairs_jsonl": args.pairs_jsonl,
        "embedding_backend": str(args.embedding_backend),
        "embedding_model": str(args.embedding_model),
        "dotenv_path": str(args.dotenv_path),
        "embedding_batch_size": int(args.embedding_batch_size),
        "hash_dim": int(args.hash_dim),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "lr": float(args.lr),
        "pair_margin": float(args.pair_margin),
        "seed": int(args.seed),
        "device": str(args.device),
        "max_targets": args.max_targets,
        "val_window_ranges": getattr(args, "val_window_ranges", None),
        "purge_gap": int(getattr(args, "purge_gap", 0) or 0),
        "holdout_view_families": getattr(args, "holdout_view_families", None),
        "eval_every": int(getattr(args, "eval_every", 0) or 0),
        "patience": int(getattr(args, "patience", 0) or 0),
        "locality_soft": bool(getattr(args, "locality_soft", False)),
        "locality_k": int(getattr(args, "locality_k", 5)),
        "tau_loc": float(getattr(args, "tau_loc", 0.10)),
        "nv_false_neg_rho": float(getattr(args, "nv_false_neg_rho", 0.95)),
        "neighbors_npz": getattr(args, "neighbors_npz", None),
        "argv_record": argv_record,
    }
    embedding_cache_dir = getattr(args, "embedding_cache_dir", None)
    shared_embedding_cache = (
        _resolve(embedding_cache_dir)
        if embedding_cache_dir
        else output / "shared_embedding_cache"
    )
    if args.method in {"both", "text-space"}:
        methods.append("text-space")
        report["text_space"] = train_text_space_from_manifest(
            output_dir=output / "text_space",
            embedding_cache_dir=shared_embedding_cache,
            adapter_dim=int(args.adapter_dim),
            support_arrays_path=args.support_arrays,
            **common,
        )
    if args.method in {"both", "projected-memory"}:
        methods.append("projected-memory")
        report["projected_memory"] = train_projected_memory_from_manifest(
            support_arrays_path=args.support_arrays,
            output_dir=output / "projected_memory",
            embedding_cache_dir=shared_embedding_cache,
            mse_weight=float(getattr(args, "mse_weight", 1.0)),
            contrastive_weight=float(getattr(args, "contrastive_weight", 0.2)),
            **common,
        )
    _write_json(output / "training_run_report.json", report)
    _write_markdown(output / "training_run_report.md", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-jsonl", type=Path, default=DEFAULT_EXAMPLES_JSONL)
    parser.add_argument("--pairs-jsonl", type=Path, default=DEFAULT_PAIRS_JSONL)
    parser.add_argument("--support-arrays", type=Path, default=DEFAULT_SUPPORT_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--method",
        choices=("both", "text-space", "projected-memory"),
        default="both",
    )
    parser.add_argument("--embedding-backend", choices=("hash", "openai"), default="hash")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pair-margin", type=float, default=0.15)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--val-window-ranges", type=str, default=None)
    parser.add_argument("--purge-gap", type=int, default=0)
    parser.add_argument("--holdout-view-families", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    # Locality-soft retriever objective (Track A T3). Flag OFF -> byte-identical
    # to the exact-window objective. See docs/superpowers/plans/2026-06-17-locality-soft-retriever.md
    parser.add_argument(
        "--locality-soft",
        action="store_true",
        help="Use the neighborhood (locality-soft) objective instead of exact-window.",
    )
    parser.add_argument("--locality-k", type=int, default=5, help="P(w) neighborhood size K.")
    parser.add_argument("--tau-loc", type=float, default=0.10, help="Locality softmax temperature.")
    parser.add_argument(
        "--nv-false-neg-rho",
        type=float,
        default=0.95,
        help="NV-Retriever false-negative threshold rho (random/in-batch negs only).",
    )
    parser.add_argument(
        "--neighbors-npz",
        type=Path,
        default=None,
        help="Path to memory_knn_neighbors.npz (required when --locality-soft).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.argv_record = list(argv) if argv is not None else list(sys.argv[1:])
    report = run_training(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
