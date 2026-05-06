#!/usr/bin/env python
"""Text embeddings and demo condition vectors for natural-language scenarios.

The generated text embedding is real supervision input. The 128-dimensional
condition vector produced here is only an untrained adapter demo; it has the
same shape as the SNI prefix memory state but is not yet a learned replacement
for ``model._encode_prefix(... )[:, -1]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_scenario_descriptions import load_dotenv_key


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CONDITION_DIM = 128


def load_description_records(path: str | Path) -> list[dict[str, Any]]:
    """Load generated scenario-description records from JSON or JSONL."""

    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{input_path}:{line_no}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{input_path}:{line_no}: expected JSON object")
                rows.append(row)
        return rows

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError(f"{input_path}: expected JSON object, list of objects, or JSONL")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_conditioning_text(
    record: dict[str, Any], *, include_free_form: bool = False
) -> str:
    """Build the text string used for embedding one description record.

    The canonical machine-readable line is intentionally first. It carries
    explicit direction and magnitude tokens, which helps avoid relying only on
    semantically broad natural-language words such as "rates" or "volatility".
    """

    parts = [
        f"WINDOW_ID: {_as_text(record.get('window_id'))}",
        f"PANEL: {_as_text(record.get('panel_version'))}",
        f"CANONICAL: {_as_text(record.get('canonical_machine_text'))}",
        f"SUMMARY: {_as_text(record.get('revised_description'))}",
    ]
    if include_free_form:
        for item in record.get("descriptions", []):
            if not isinstance(item, dict):
                continue
            style = _as_text(item.get("style")) or "unknown"
            text = _as_text(item.get("text"))
            if text:
                parts.append(f"FREE_FORM {style}: {text}")
    return "\n".join(part for part in parts if part and not part.endswith(": "))


def build_contrast_texts(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return hard negative texts associated with a description record."""

    contrastive = record.get("contrastive", {})
    if not isinstance(contrastive, dict):
        return []

    rows: list[dict[str, str]] = []
    opposite = _as_text(contrastive.get("opposite"))
    if opposite:
        rows.append({"kind": "opposite", "text": opposite})
    for kind in ("partial", "magnitude"):
        values = contrastive.get(kind, [])
        if not isinstance(values, list):
            continue
        for value in values:
            text = _as_text(value)
            if text:
                rows.append({"kind": kind, "text": text})
    return rows


def build_contrastive_examples(record: dict[str, Any]) -> list[dict[str, str]]:
    """Build anchor/positive/negative texts for a small contrastive pilot."""

    examples = [
        {
            "role": "anchor",
            "kind": "canonical_summary",
            "text": build_conditioning_text(record, include_free_form=False),
        }
    ]
    for item in record.get("descriptions", []):
        if not isinstance(item, dict):
            continue
        text = _as_text(item.get("text"))
        if not text:
            continue
        examples.append(
            {
                "role": "positive",
                "kind": _as_text(item.get("style")) or "free_form",
                "text": text,
            }
        )
    for contrast in build_contrast_texts(record):
        examples.append(
            {
                "role": "negative",
                "kind": contrast["kind"],
                "text": contrast["text"],
            }
        )
    return examples


def normalize_rows(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 2-D array row-wise."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool(np.any(norms <= eps)):
        raise ValueError("cannot normalize zero-norm rows")
    return array / norms


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity for two one-dimensional vectors."""

    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"shape mismatch: {left_arr.shape} vs {right_arr.shape}")
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom <= 1e-12:
        raise ValueError("cannot compare zero-norm vectors")
    score = float(np.dot(left_arr, right_arr) / denom)
    return float(np.clip(score, -1.0, 1.0))


def anchor_similarity_metrics(
    vectors: np.ndarray, roles: list[str], *, anchor_index: int = 0
) -> dict[str, Any]:
    """Summarize anchor-positive and anchor-negative cosine separation."""

    normalized = normalize_rows(vectors)
    if len(roles) != normalized.shape[0]:
        raise ValueError("roles length must match vector rows")
    if roles[anchor_index] != "anchor":
        raise ValueError("anchor_index must point to an anchor role")
    positives = [
        cosine_similarity(normalized[anchor_index], normalized[idx])
        for idx, role in enumerate(roles)
        if role == "positive"
    ]
    negatives = [
        cosine_similarity(normalized[anchor_index], normalized[idx])
        for idx, role in enumerate(roles)
        if role == "negative"
    ]
    if not positives:
        raise ValueError("need at least one positive example")
    if not negatives:
        raise ValueError("need at least one negative example")

    pos = np.asarray(positives, dtype=np.float32)
    neg = np.asarray(negatives, dtype=np.float32)
    return {
        "positive_mean_cosine": float(pos.mean()),
        "positive_min_cosine": float(pos.min()),
        "negative_mean_cosine": float(neg.mean()),
        "negative_max_cosine": float(neg.max()),
        "separation_mean": float(pos.mean() - neg.mean()),
        "hard_margin": float(pos.min() - neg.max()),
        "positive_cosines": [float(x) for x in pos],
        "negative_cosines": [float(x) for x in neg],
    }


class TextConditionAdapter(nn.Module):
    """Small projector from frozen text embeddings into SNI memory space."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        condition_dim: int = DEFAULT_CONDITION_DIM,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        hidden = int(hidden_dim or min(512, max(condition_dim * 2, embedding_dim // 2)))
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(condition_dim)),
            nn.LayerNorm(int(condition_dim)),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


class ContrastiveProjectionHead(nn.Module):
    """Trainable metric projection for text-embedding contrastive pilots."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        metric_dim: int = 64,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden = int(hidden_dim or min(512, max(metric_dim * 2, embedding_dim // 2)))
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(metric_dim)),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(embeddings), dim=-1)


def train_contrastive_projection(
    embeddings: np.ndarray,
    roles: list[str],
    *,
    metric_dim: int = 64,
    hidden_dim: int | None = None,
    steps: int = 500,
    lr: float = 1e-3,
    margin: float = 0.25,
    seed: int = 0,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Train a tiny projection to separate positives from hard negatives.

    This deliberately trains only a metric projection over frozen embeddings.
    It is a pre-batch diagnostic, not a replacement for the later adapter
    alignment to the generator's historical SNI memory vectors.
    """

    normalized = normalize_rows(embeddings)
    if len(roles) != normalized.shape[0]:
        raise ValueError("roles length must match embedding rows")
    anchor_indices = [idx for idx, role in enumerate(roles) if role == "anchor"]
    positive_indices = [idx for idx, role in enumerate(roles) if role == "positive"]
    negative_indices = [idx for idx, role in enumerate(roles) if role == "negative"]
    if len(anchor_indices) != 1:
        raise ValueError("need exactly one anchor example")
    if not positive_indices:
        raise ValueError("need at least one positive example")
    if not negative_indices:
        raise ValueError("need at least one negative example")

    torch.manual_seed(int(seed))
    device = torch.device("cpu")
    x = torch.from_numpy(normalized).float().to(device)
    projection = ContrastiveProjectionHead(
        embedding_dim=normalized.shape[1],
        metric_dim=int(metric_dim),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    anchor_index = int(anchor_indices[0])
    pos_idx = torch.tensor(positive_indices, dtype=torch.long, device=device)
    neg_idx = torch.tensor(negative_indices, dtype=torch.long, device=device)
    losses: list[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        projected = projection(x)
        anchor = projected[anchor_index]
        pos_sim = projected[pos_idx] @ anchor
        neg_sim = projected[neg_idx] @ anchor
        ranking = F.relu(float(margin) - pos_sim[:, None] + neg_sim[None, :]).mean()
        positive_alignment = (1.0 - pos_sim).mean()
        loss = ranking + 0.1 * positive_alignment
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    projection.eval()
    with torch.no_grad():
        projected_np = projection(x).cpu().numpy().astype(np.float32)

    return {
        "projected": projected_np,
        "raw_metrics": anchor_similarity_metrics(normalized, roles),
        "projected_metrics": anchor_similarity_metrics(projected_np, roles),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
    }


def project_embeddings_with_adapter(
    embeddings: np.ndarray,
    *,
    condition_dim: int = DEFAULT_CONDITION_DIM,
    hidden_dim: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Project text embeddings with a deterministic untrained adapter."""

    normalized = normalize_rows(embeddings)
    torch.manual_seed(int(seed))
    adapter = TextConditionAdapter(
        embedding_dim=normalized.shape[1],
        condition_dim=int(condition_dim),
        hidden_dim=hidden_dim,
    )
    adapter.eval()
    with torch.no_grad():
        condition = adapter(torch.from_numpy(normalized).float()).cpu().numpy()
    return condition.astype(np.float32)


def embed_texts_with_openai(
    texts: list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dotenv_path: str | Path = ".env",
) -> np.ndarray:
    """Call the OpenAI Embeddings API for a list of texts."""

    if not texts:
        raise ValueError("texts must not be empty")
    load_dotenv_key(dotenv_path)
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = [item.embedding for item in ordered]
    return np.asarray(vectors, dtype=np.float32)


def _summarize_vector(vector: np.ndarray, *, n: int = 8) -> dict[str, Any]:
    return {
        "dim": int(vector.shape[0]),
        "l2_norm": float(np.linalg.norm(vector)),
        "preview": [float(x) for x in vector[:n]],
    }


def _record_id(record: dict[str, Any], index: int) -> str:
    return _as_text(record.get("window_id")) or f"record_{index:06d}"


def build_demo_report(
    records: list[dict[str, Any]],
    *,
    embeddings: np.ndarray,
    condition_vectors: np.ndarray,
    texts: list[str],
    text_kinds: list[str],
    embedding_model: str,
    condition_dim: int,
    adapter_seed: int,
) -> dict[str, Any]:
    normalized_embeddings = normalize_rows(embeddings)
    primary_indices = [idx for idx, kind in enumerate(text_kinds) if kind == "primary"]
    contrast_rows: list[dict[str, Any]] = []
    if primary_indices:
        primary_index = primary_indices[0]
        for idx, kind in enumerate(text_kinds):
            if idx == primary_index:
                continue
            contrast_rows.append(
                {
                    "kind": kind,
                    "embedding_cosine_vs_primary": cosine_similarity(
                        normalized_embeddings[primary_index],
                        normalized_embeddings[idx],
                    ),
                    "condition_cosine_vs_primary": cosine_similarity(
                        condition_vectors[primary_index],
                        condition_vectors[idx],
                    ),
                }
            )

    return {
        "adapter_status": "untrained_demo_projection",
        "adapter_note": (
            "Train this adapter against historical _encode_prefix(... )[:, -1] "
            "targets before using condition vectors for generation."
        ),
        "embedding_model": embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "condition_dim": int(condition_dim),
        "adapter_seed": int(adapter_seed),
        "record_count": len(records),
        "text_count": len(texts),
        "primary_text": texts[0] if texts else "",
        "primary_embedding": _summarize_vector(normalized_embeddings[0]),
        "primary_condition_vector": _summarize_vector(condition_vectors[0]),
        "contrast_diagnostics": contrast_rows,
        "records": [
            {
                "index": index,
                "window_id": _record_id(record, index),
                "panel_version": _as_text(record.get("panel_version")),
            }
            for index, record in enumerate(records)
        ],
    }


def _prepare_texts(
    records: list[dict[str, Any]],
    *,
    include_free_form: bool,
    include_contrasts_for_first: bool,
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    kinds: list[str] = []
    for index, record in enumerate(records):
        texts.append(
            build_conditioning_text(record, include_free_form=include_free_form)
        )
        kinds.append("primary")
        if index == 0 and include_contrasts_for_first:
            for contrast in build_contrast_texts(record):
                texts.append(contrast["text"])
                kinds.append(str(contrast["kind"]))
    return texts, kinds


def _cmd_demo(args: argparse.Namespace) -> None:
    records = load_description_records(args.input)
    if args.limit is not None:
        records = records[: int(args.limit)]
    texts, kinds = _prepare_texts(
        records,
        include_free_form=bool(args.include_free_form),
        include_contrasts_for_first=not bool(args.no_contrasts),
    )
    embeddings = embed_texts_with_openai(
        texts,
        model=args.embedding_model,
        dotenv_path=args.dotenv,
    )
    condition_vectors = project_embeddings_with_adapter(
        embeddings,
        condition_dim=int(args.condition_dim),
        hidden_dim=args.hidden_dim,
        seed=int(args.seed),
    )
    report = build_demo_report(
        records,
        embeddings=embeddings,
        condition_vectors=condition_vectors,
        texts=texts,
        text_kinds=kinds,
        embedding_model=args.embedding_model,
        condition_dim=int(args.condition_dim),
        adapter_seed=int(args.seed),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            text_embeddings=normalize_rows(embeddings),
            condition_vectors=condition_vectors,
        )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_npz": str(args.output_npz) if args.output_npz else None,
                "embedding_dim": int(embeddings.shape[1]),
                "condition_dim": int(args.condition_dim),
                "text_count": len(texts),
            },
            sort_keys=True,
        )
    )


def _cmd_embed(args: argparse.Namespace) -> None:
    records = load_description_records(args.input)
    texts, kinds = _prepare_texts(
        records,
        include_free_form=bool(args.include_free_form),
        include_contrasts_for_first=not bool(args.no_contrasts),
    )
    embeddings = normalize_rows(
        embed_texts_with_openai(
            texts,
            model=args.embedding_model,
            dotenv_path=args.dotenv,
        )
    )
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, text_embeddings=embeddings)
    metadata = {
        "embedding_model": args.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "text_count": len(texts),
        "text_kinds": kinds,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _cmd_contrastive_pilot(args: argparse.Namespace) -> None:
    records = load_description_records(args.input)
    record_index = int(args.record_index)
    if record_index < 0 or record_index >= len(records):
        raise ValueError(
            f"record-index {record_index} is outside 0..{len(records) - 1}"
        )
    record = records[record_index]
    examples = build_contrastive_examples(record)
    texts = [example["text"] for example in examples]
    roles = [example["role"] for example in examples]
    embeddings = embed_texts_with_openai(
        texts,
        model=args.embedding_model,
        dotenv_path=args.dotenv,
    )
    result = train_contrastive_projection(
        embeddings,
        roles,
        metric_dim=int(args.metric_dim),
        hidden_dim=args.hidden_dim,
        steps=int(args.steps),
        lr=float(args.lr),
        margin=float(args.margin),
        seed=int(args.seed),
        weight_decay=float(args.weight_decay),
    )
    raw_metrics = result["raw_metrics"]
    projected_metrics = result["projected_metrics"]
    report = {
        "status": "contrastive_projection_pilot",
        "scope_note": (
            "This tests whether hard negatives can improve text-space directional "
            "separation. It does not prove generator-ready conditioning until the "
            "projection is also aligned to historical _encode_prefix memory targets."
        ),
        "window_id": _record_id(record, record_index),
        "panel_version": _as_text(record.get("panel_version")),
        "embedding_model": args.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "metric_dim": int(args.metric_dim),
        "example_count": len(examples),
        "raw_metrics": raw_metrics,
        "projected_metrics": projected_metrics,
        "improvement": {
            "separation_mean_delta": float(
                projected_metrics["separation_mean"] - raw_metrics["separation_mean"]
            ),
            "hard_margin_delta": float(
                projected_metrics["hard_margin"] - raw_metrics["hard_margin"]
            ),
            "negative_mean_cosine_delta": float(
                projected_metrics["negative_mean_cosine"]
                - raw_metrics["negative_mean_cosine"]
            ),
        },
        "training": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "margin": float(args.margin),
            "seed": int(args.seed),
            "loss_first": float(result["loss_first"]),
            "loss_last": float(result["loss_last"]),
        },
        "examples": examples,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_npz:
        output_npz = Path(args.output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            text_embeddings=normalize_rows(embeddings),
            projected_embeddings=result["projected"],
        )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_npz": str(args.output_npz) if args.output_npz else None,
                "raw_separation_mean": raw_metrics["separation_mean"],
                "projected_separation_mean": projected_metrics["separation_mean"],
                "raw_hard_margin": raw_metrics["hard_margin"],
                "projected_hard_margin": projected_metrics["hard_margin"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "demo",
        help="Embed generated descriptions and produce untrained 128-d demo conditions",
    )
    demo.add_argument("--input", required=True)
    demo.add_argument("--output-json", required=True)
    demo.add_argument("--output-npz")
    demo.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    demo.add_argument("--condition-dim", type=int, default=DEFAULT_CONDITION_DIM)
    demo.add_argument("--hidden-dim", type=int)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--dotenv", default=".env")
    demo.add_argument("--limit", type=int)
    demo.add_argument("--include-free-form", action="store_true")
    demo.add_argument("--no-contrasts", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    embed = sub.add_parser("embed", help="Only generate normalized text embeddings")
    embed.add_argument("--input", required=True)
    embed.add_argument("--output-json", required=True)
    embed.add_argument("--output-npz", required=True)
    embed.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    embed.add_argument("--dotenv", default=".env")
    embed.add_argument("--include-free-form", action="store_true")
    embed.add_argument("--no-contrasts", action="store_true")
    embed.set_defaults(func=_cmd_embed)

    contrastive = sub.add_parser(
        "contrastive-pilot",
        help="Train a small contrastive projection on one labeled scenario record",
    )
    contrastive.add_argument("--input", required=True)
    contrastive.add_argument("--output-json", required=True)
    contrastive.add_argument("--output-npz")
    contrastive.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    contrastive.add_argument("--metric-dim", type=int, default=DEFAULT_CONDITION_DIM)
    contrastive.add_argument("--hidden-dim", type=int)
    contrastive.add_argument("--steps", type=int, default=500)
    contrastive.add_argument("--lr", type=float, default=1e-3)
    contrastive.add_argument("--margin", type=float, default=0.25)
    contrastive.add_argument("--seed", type=int, default=0)
    contrastive.add_argument("--weight-decay", type=float, default=1e-4)
    contrastive.add_argument("--dotenv", default=".env")
    contrastive.add_argument("--record-index", type=int, default=0)
    contrastive.set_defaults(func=_cmd_contrastive_pilot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
