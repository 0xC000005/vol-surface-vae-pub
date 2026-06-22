#!/usr/bin/env python
"""Train a text-embedding to SNI-memory bridge on episode-card narratives.

This is an isolated experiment for the old hidden-space / CLIP-style path:
professional narrative text -> text embedding -> learned bridge -> frozen SNI
128-dim final encoder memory. It consumes only saved Codex/GPT-authored
episode-card narratives and does not alter the incumbent paper/demo pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    _load_json,
    _query_text,
    _read_jsonl,
    _softmax_weights,
    _split_indices_from_support_report,
    _temporal_gap_filter,
    _terminal_start_match,
    _window_index,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    embed_with_cache,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    assert_cards_allowed_for_retrieval,
    pairwise_jaccard_summary,
)
from experiments.backfill.block_ar.nl_text_conditioning import normalize_rows  # noqa: E402


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)
DEFAULT_SUPPORT_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_text_memory_bridge"
)
FACTOR_MARKETS = (
    "IV_SURFACE",
    "SPX",
    "US2Y",
    "US10Y",
    "BBB_OAS",
    "AAA_OAS",
    "USDJPY",
    "DXY",
    "GOLD",
    "CRUDE_OIL",
    "VIX",
)
FACTOR_DIRECTION_RE = re.compile(
    r"\b("
    + "|".join(re.escape(name) for name in FACTOR_MARKETS)
    + r")\s+(up|down|wider|tighter|flat|higher|lower)\b"
    r"(?:\s+(small|medium|large))?",
    re.IGNORECASE,
)
DIRECTION_TO_SIGN = {
    "up": 1,
    "higher": 1,
    "wider": 1,
    "down": -1,
    "lower": -1,
    "tighter": -1,
    "flat": 0,
}
MAGNITUDE_VALUE = {"small": 1, "medium": 2, "large": 3}


class TextMemoryBridge(nn.Module):
    """Small bridge from normalized text embeddings to SNI encoder memory."""

    def __init__(
        self, embedding_dim: int, condition_dim: int, hidden_dim: int | None = None
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or min(768, max(condition_dim * 4, embedding_dim // 2)))
        self.net = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(condition_dim)),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _card_by_index(cards: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {_window_index(card): card for card in cards}


def _view_rows(
    cards: list[dict[str, Any]],
    *,
    allowed_indices: set[int],
    view_names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for card in cards:
        window_index = _window_index(card)
        if window_index not in allowed_indices:
            continue
        views = card.get("views", {})
        if not isinstance(views, dict):
            continue
        for view_name in view_names:
            text = views.get(view_name)
            if not isinstance(text, str) or not text.strip():
                continue
            clean = text.strip()
            rows.append(
                {
                    "embedding_index": len(rows),
                    "window_index": int(window_index),
                    "window_id": str(card.get("window_id", "")),
                    "split": str(card.get("split", "")),
                    "scenario_title": str(card.get("scenario_title", "")),
                    "archetype": str(card.get("archetype", "")),
                    "view": str(view_name),
                    "role": "anchor" if view_name == "full_professional" else "positive",
                    "text": clean,
                }
            )
            texts.append(clean)
    if not rows:
        raise ValueError("no usable narrative views found")
    return rows, texts


def _query_embedding_index(
    rows: list[dict[str, Any]],
    *,
    window_index: int,
    preferred_view: str,
) -> int:
    matches = [
        row
        for row in rows
        if int(row["window_index"]) == int(window_index)
        and str(row.get("view", "")) == str(preferred_view)
    ]
    if matches:
        return int(matches[0]["embedding_index"])
    fallback = [
        row for row in rows if int(row["window_index"]) == int(window_index)
    ]
    if fallback:
        return int(fallback[0]["embedding_index"])
    raise ValueError(f"no embedding row for window_index={window_index}")


def _factor_grounding_text(card: dict[str, Any]) -> str:
    views = card.get("views", {})
    if isinstance(views, dict):
        for key in ("factor_list_baseline", "technical_factor_evidence", "old_factor_baseline"):
            text = views.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    fields = card.get("codex_multiformat_fields", {})
    if isinstance(fields, dict):
        text = fields.get("old_factor_baseline")
        if isinstance(text, str) and text.strip():
            return text.strip()
    caption_fields = card.get("caption_fields", {})
    if isinstance(caption_fields, dict):
        text = caption_fields.get("mechanical_summary")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _parse_factor_grounding(card: dict[str, Any]) -> list[dict[str, Any]]:
    text = _factor_grounding_text(card)
    rows: list[dict[str, Any]] = []
    for pos, match in enumerate(FACTOR_DIRECTION_RE.finditer(text)):
        market, direction, magnitude = match.groups()
        clean_direction = str(direction).lower()
        sign = int(DIRECTION_TO_SIGN.get(clean_direction, 0))
        mag = int(MAGNITUDE_VALUE.get(str(magnitude or "large").lower(), 3))
        rows.append(
            {
                "market": str(market).upper(),
                "direction": clean_direction,
                "sign": sign,
                "magnitude": str(magnitude or "large").lower(),
                "magnitude_value": mag,
                "order": int(pos),
            }
        )
    return rows


def _required_grounding_claims(
    card: dict[str, Any], *, max_claims: int
) -> list[dict[str, Any]]:
    rows = [row for row in _parse_factor_grounding(card) if int(row["sign"]) != 0]
    rows.sort(key=lambda row: (-int(row["magnitude_value"]), int(row["order"])))
    if int(max_claims) > 0:
        rows = rows[: int(max_claims)]
    return rows


def _direction_check(
    *,
    query_claims: list[dict[str, Any]],
    candidate_card: dict[str, Any],
    max_mismatches: int,
) -> dict[str, Any]:
    candidate_rows = {
        str(row["market"]).upper(): row for row in _parse_factor_grounding(candidate_card)
    }
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for claim in query_claims:
        market = str(claim["market"]).upper()
        expected = int(claim["sign"])
        if expected == 0:
            continue
        checked += 1
        observed_row = candidate_rows.get(market)
        observed = int(observed_row["sign"]) if observed_row else 0
        if observed != expected:
            mismatches.append(
                {
                    "market": market,
                    "expected_direction": str(claim["direction"]),
                    "observed_direction": str(observed_row.get("direction", "missing"))
                    if observed_row
                    else "missing",
                }
            )
    status = "pass" if len(mismatches) <= int(max_mismatches) else "reject"
    return {
        "status": status,
        "checked_count": int(checked),
        "mismatch_count": int(len(mismatches)),
        "max_mismatches": int(max_mismatches),
        "mismatches": mismatches,
    }


def _train_bridge(
    *,
    text_embeddings: np.ndarray,
    target_memory: np.ndarray,
    train_example_indices: np.ndarray,
    target_indices: np.ndarray,
    condition_dim: int,
    hidden_dim: int | None,
    steps: int,
    batch_size: int,
    lr: float,
    mse_weight: float,
    cosine_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    seed: int,
    device: str | torch.device | None,
) -> tuple[TextMemoryBridge, np.ndarray, dict[str, Any]]:
    embeddings = normalize_rows(np.asarray(text_embeddings, dtype=np.float32))
    targets = np.asarray(target_memory, dtype=np.float32)
    example_idx = np.asarray(train_example_indices, dtype=np.int64)
    target_idx = np.asarray(target_indices, dtype=np.int64)
    if example_idx.size == 0:
        raise ValueError("train_example_indices must not be empty")
    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))
    device_t = torch.device(device or "cpu")
    x_all = torch.from_numpy(embeddings).float().to(device_t)
    y_all = torch.from_numpy(targets).float().to(device_t)
    train_x = torch.from_numpy(example_idx).long().to(device_t)
    train_target = torch.from_numpy(target_idx).long().to(device_t)
    bridge = TextMemoryBridge(
        embeddings.shape[1],
        int(condition_dim),
        hidden_dim=hidden_dim,
    ).to(device_t)
    opt = torch.optim.AdamW(bridge.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[dict[str, float]] = []
    batch_n = min(max(1, int(batch_size)), int(example_idx.size))
    for step in range(int(steps)):
        choice = np_rng.choice(int(example_idx.size), size=batch_n, replace=batch_n > example_idx.size)
        choice_t = torch.from_numpy(choice.astype(np.int64)).to(device_t)
        batch_rows = train_x[choice_t]
        batch_targets = train_target[choice_t]
        pred = bridge(x_all[batch_rows])
        target = y_all[batch_targets]
        mse = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        unique_targets, inverse = torch.unique(
            batch_targets, sorted=True, return_inverse=True
        )
        logits = (
            F.normalize(pred, dim=-1)
            @ F.normalize(y_all[unique_targets], dim=-1).T
        ) / max(float(contrastive_temperature), 1e-6)
        contrastive = F.cross_entropy(logits, inverse)
        loss = (
            float(mse_weight) * mse
            + float(cosine_weight) * cosine
            + float(contrastive_weight) * contrastive
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=5.0)
        opt.step()
        if step in {0, int(steps) - 1} or (step + 1) % max(1, int(steps) // 10) == 0:
            losses.append(
                {
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "cosine": float(cosine.detach().cpu()),
                    "contrastive": float(contrastive.detach().cpu()),
                }
            )
    bridge.eval()
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], 2048):
            pred = bridge(x_all[start : start + 2048]).detach().cpu().numpy()
            vectors.append(pred.astype(np.float32))
    condition_vectors = np.concatenate(vectors, axis=0)
    meta = {
        "train_example_count": int(example_idx.size),
        "steps": int(steps),
        "batch_size": int(batch_n),
        "lr": float(lr),
        "mse_weight": float(mse_weight),
        "cosine_weight": float(cosine_weight),
        "contrastive_weight": float(contrastive_weight),
        "contrastive_temperature": float(contrastive_temperature),
        "seed": int(seed),
        "device": str(device_t),
        "hidden_dim": int(hidden_dim or min(768, max(condition_dim * 4, embeddings.shape[1] // 2))),
        "loss_trace": losses,
    }
    return bridge, condition_vectors.astype(np.float32), meta


def _rank_supports(
    *,
    query_vector: np.ndarray,
    memory_targets: np.ndarray,
    train_indices: list[int],
    query_index: int,
    history_raw: np.ndarray | None,
    top_k: int,
    temporal_gap: int,
    cards_by_index: dict[int, dict[str, Any]],
    start_weight: float,
    support_pool_size: int,
    grounding_gate: bool,
    query_claims: list[dict[str, Any]] | None,
    max_grounding_mismatches: int,
) -> list[dict[str, Any]]:
    memory = _safe_normalize_rows(np.asarray(memory_targets, dtype=np.float32))
    query = _safe_normalize_rows(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    scores = memory[np.asarray(train_indices, dtype=np.int64)] @ query
    ranked: list[dict[str, Any]] = []
    for local_pos in np.argsort(-scores):
        idx = int(train_indices[int(local_pos)])
        if idx == int(query_index):
            continue
        memory_score = float(scores[int(local_pos)])
        start_match = 0.0
        start_distance = math.nan
        combined = memory_score
        if history_raw is not None and float(start_weight) != 0.0:
            start_match, start_distance = _terminal_start_match(
                query_index=int(query_index),
                candidate_index=idx,
                history_raw=history_raw,
                train_indices=train_indices,
            )
            combined = memory_score + float(start_weight) * float(start_match)
        card = cards_by_index.get(idx, {})
        direction_check = {
            "status": "not_checked",
            "checked_count": 0,
            "mismatch_count": 0,
            "max_mismatches": int(max_grounding_mismatches),
            "mismatches": [],
        }
        if bool(grounding_gate):
            direction_check = _direction_check(
                query_claims=list(query_claims or []),
                candidate_card=card,
                max_mismatches=int(max_grounding_mismatches),
            )
            if str(direction_check["status"]) != "pass":
                continue
        ranked.append(
            {
                "window_index": idx,
                "window_id": str(card.get("window_id", f"window_{idx:04d}")),
                "scenario_title": str(card.get("scenario_title", "")),
                "score": float(combined),
                "memory_score": memory_score,
                "start_match_score": float(start_match),
                "start_distance": float(start_distance),
                "score_components": {
                    "method": "text_memory_bridge_grounded_candidate"
                    if grounding_gate
                    else "text_memory_bridge",
                    "memory_score": memory_score,
                    "start_match_score": float(start_match),
                    "start_distance": float(start_distance),
                    "start_weight": float(start_weight),
                    "direction_check": direction_check,
                },
            }
        )
        if len(ranked) >= max(int(support_pool_size) * 16, int(support_pool_size)):
            break
    return _temporal_gap_filter(
        ranked, top_k=int(support_pool_size), temporal_gap=int(temporal_gap)
    )


def _apply_top3_90_selection(
    candidates: list[dict[str, Any]], *, min_weight_mass: float = 0.90
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    sorted_candidates = sorted(
        candidates, key=lambda item: float(item.get("weight", 0.0)), reverse=True
    )
    selected: list[dict[str, Any]] = []
    cumulative = 0.0
    for item in sorted_candidates:
        if len(selected) >= 3:
            break
        selected.append(dict(item))
        cumulative += max(float(item.get("weight", 0.0)), 0.0)
        if cumulative >= float(min_weight_mass):
            break
    total = sum(max(float(item.get("weight", 0.0)), 0.0) for item in selected)
    if total <= 0.0:
        total = float(max(len(selected), 1))
        for item in selected:
            item["weight"] = 1.0 / total
    else:
        for item in selected:
            item["base_support_weight"] = float(item.get("weight", 0.0))
            item["weight"] = max(float(item.get("weight", 0.0)), 0.0) / total
            item["posterior_weight"] = float(item["weight"])
            item["posterior_role"] = "top3_90_selected"
    selected.sort(key=lambda item: int(item.get("rank", 0)))
    return selected


def _rank_true_target(
    *, query_index: int, query_vector: np.ndarray, memory_targets: np.ndarray
) -> int:
    memory = _safe_normalize_rows(np.asarray(memory_targets, dtype=np.float32))
    query = _safe_normalize_rows(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
    scores = memory @ query
    true_score = float(scores[int(query_index)])
    return int(np.sum(scores > true_score) + 1)


def _safe_normalize_rows(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, float(eps))


def _summarize_heldout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.asarray([row["true_memory_rank"] for row in rows], dtype=np.float32)
    cosines = np.asarray([row["true_memory_cosine"] for row in rows], dtype=np.float32)
    top_scores = [
        float(row["top_train_pool"][0]["score_components"]["memory_score"])
        for row in rows
        if row.get("top_train_pool")
    ]
    return {
        "heldout_query_count": int(len(rows)),
        "mean_true_memory_cosine": float(np.mean(cosines)) if cosines.size else math.nan,
        "median_true_memory_rank": float(np.median(ranks)) if ranks.size else math.nan,
        "recall_at_10_true_memory": float(np.mean(ranks <= 10.0)) if ranks.size else math.nan,
        "mean_top_train_memory_score": float(np.mean(top_scores)) if top_scores else math.nan,
    }


def build_text_memory_bridge_report(
    *,
    cards: list[dict[str, Any]],
    support_report: dict[str, Any],
    support_arrays_path: Path,
    output_dir: Path,
    embedding_backend: str,
    embedding_model: str,
    dotenv_path: str,
    embedding_batch_size: int,
    hash_dim: int,
    adapter_steps: int,
    adapter_batch_size: int,
    top_k: int,
    temporal_gap: int,
    max_query_windows: int,
    device: str | torch.device | None,
    hidden_dim: int | None = None,
    adapter_lr: float = 1e-3,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.2,
    contrastive_weight: float = 0.2,
    contrastive_temperature: float = 0.07,
    seed: int = 0,
    query_view: str = "full_professional",
    start_weight: float = 0.0,
    support_pool_size: int = 8,
    grounding_gate: bool = False,
    max_grounding_claims: int = 4,
    max_grounding_mismatches: int = 0,
    top3_90: bool = False,
    embedding_retry_attempts: int = 6,
    embedding_retry_sleep: float = 2.0,
    view_names: tuple[str, ...] = DEFAULT_VIEW_NAMES,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_indices, test_indices = _split_indices_from_support_report(support_report)
    if int(max_query_windows) > 0:
        test_indices = test_indices[: int(max_query_windows)]
    cards_by_index = _card_by_index(cards)
    all_indices = set(int(idx) for idx in train_indices + test_indices)
    rows, texts = _view_rows(cards, allowed_indices=all_indices, view_names=view_names)
    embeddings, embedding_meta = embed_with_cache(
        texts,
        output_dir=Path(output_dir),
        backend=str(embedding_backend),
        model=str(embedding_model),
        dotenv_path=str(dotenv_path),
        batch_size=int(embedding_batch_size),
        hash_dim=int(hash_dim),
        retry_attempts=int(embedding_retry_attempts),
        retry_sleep=float(embedding_retry_sleep),
    )
    with np.load(support_arrays_path) as payload:
        memory_targets = payload["memory_targets"].astype(np.float32)
        history_raw = (
            payload["history_raw"].astype(np.float32)
            if "history_raw" in payload.files
            else None
        )
    target_indices = np.asarray([int(row["window_index"]) for row in rows], dtype=np.int64)
    train_index_set = set(int(idx) for idx in train_indices)
    train_example_indices = np.asarray(
        [
            int(row["embedding_index"])
            for row in rows
            if int(row["window_index"]) in train_index_set
        ],
        dtype=np.int64,
    )
    bridge, condition_vectors, training_meta = _train_bridge(
        text_embeddings=embeddings,
        target_memory=memory_targets,
        train_example_indices=train_example_indices,
        target_indices=target_indices[train_example_indices],
        condition_dim=memory_targets.shape[1],
        hidden_dim=hidden_dim,
        steps=int(adapter_steps),
        batch_size=int(adapter_batch_size),
        lr=float(adapter_lr),
        mse_weight=float(mse_weight),
        cosine_weight=float(cosine_weight),
        contrastive_weight=float(contrastive_weight),
        contrastive_temperature=float(contrastive_temperature),
        seed=int(seed),
        device=device,
    )
    heldout_rows: list[dict[str, Any]] = []
    result_sets: dict[str, list[str]] = {}
    for query_no, query_index in enumerate(test_indices):
        card = cards_by_index.get(int(query_index))
        if card is None:
            continue
        embedding_index = _query_embedding_index(
            rows, window_index=int(query_index), preferred_view=str(query_view)
        )
        query_vector = condition_vectors[int(embedding_index)]
        query_claims = _required_grounding_claims(
            card, max_claims=int(max_grounding_claims)
        )
        selected = _rank_supports(
            query_vector=query_vector,
            memory_targets=memory_targets,
            train_indices=train_indices,
            query_index=int(query_index),
            history_raw=history_raw,
            top_k=int(top_k),
            temporal_gap=int(temporal_gap),
            cards_by_index=cards_by_index,
            start_weight=float(start_weight),
            support_pool_size=max(int(support_pool_size), int(top_k)),
            grounding_gate=bool(grounding_gate),
            query_claims=query_claims,
            max_grounding_mismatches=int(max_grounding_mismatches),
        )
        weights = _softmax_weights([float(item["score"]) for item in selected])
        candidate_pool: list[dict[str, Any]] = []
        for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), 1):
            candidate_pool.append(
                {
                    "rank": int(rank),
                    "window_index": int(item["window_index"]),
                    "window_id": str(item["window_id"]),
                    "cosine": float(item["score"]),
                    "weight": float(weight),
                    "retrieval_score": float(item["score"]),
                    "scenario_title": str(item.get("scenario_title", "")),
                    "score_components": dict(item.get("score_components", {})),
                }
            )
        top_train_pool = (
            _apply_top3_90_selection(candidate_pool, min_weight_mass=0.90)
            if bool(top3_90)
            else candidate_pool[: int(top_k)]
        )
        method_name = (
            "text_memory_bridge_grounded_top3_90"
            if bool(grounding_gate) and bool(top3_90)
            else "text_memory_bridge_grounded"
            if bool(grounding_gate)
            else "text_memory_bridge"
        )
        for display_rank, item in enumerate(top_train_pool, 1):
            item["rank"] = int(display_rank)
            components = dict(item.get("score_components", {}))
            components["method"] = method_name
            item["score_components"] = components
        true_cosine = float(
            normalize_rows(query_vector.reshape(1, -1))[0]
            @ normalize_rows(memory_targets[int(query_index)].reshape(1, -1))[0]
        )
        heldout_rows.append(
            {
                "query_id": f"text_memory_{query_no:04d}_{int(query_index)}",
                "window_index": int(query_index),
                "window_id": str(card.get("window_id", "")),
                "role": "anchor",
                "kind": method_name,
                "embedding_index": int(embedding_index),
                "query_text_source": str(query_view),
                "query_text": _query_text(card),
                "required_grounding_claims": query_claims,
                "candidate_pool_size": int(len(candidate_pool)),
                "pre_top3_90_candidate_pool": candidate_pool,
                "true_memory_cosine": true_cosine,
                "true_memory_rank": _rank_true_target(
                    query_index=int(query_index),
                    query_vector=query_vector,
                    memory_targets=memory_targets,
                ),
                "top_train_pool": top_train_pool,
            }
        )
        result_sets[str(card.get("window_id", query_index))] = [
            str(item["window_id"]) for item in top_train_pool
        ]
    example_rows = [
        {
            key: row[key]
            for key in (
                "embedding_index",
                "window_index",
                "window_id",
                "split",
                "scenario_title",
                "archetype",
                "view",
                "role",
            )
        }
        for row in rows
    ]
    report = {
        "schema_version": "nl_episode_text_memory_bridge_report_v1",
        "status": "ok",
        "scope_note": (
            "Isolated full-corpus bridge experiment. Direct Codex/GPT-authored "
            "episode narratives are embedded and projected into the frozen SNI "
            "final encoder-memory space; no narrative text is generated here."
        ),
        "cards_path": "",
        "arrays_path": str(support_arrays_path),
        "embedding_metadata": embedding_meta,
        "adapter_training": training_meta,
        "retrieval_config": {
            "method": "text_embedding_to_sni_final_memory",
            "top_k": int(top_k),
            "temporal_gap": int(temporal_gap),
            "query_view": str(query_view),
            "view_names": list(view_names),
            "support_pool": "support_train_only",
            "query_pool": "support_decoder_test",
            "start_weight": float(start_weight),
            "support_pool_size": int(support_pool_size),
            "grounding_gate": bool(grounding_gate),
            "max_grounding_claims": int(max_grounding_claims),
            "max_grounding_mismatches": int(max_grounding_mismatches),
            "top3_90": bool(top3_90),
        },
        "split": {
            "train_indices": [int(idx) for idx in train_indices],
            "test_indices": [int(idx) for idx in test_indices],
        },
        "window_indices": [int(idx) for idx in sorted(all_indices)],
        "example_rows": example_rows,
        "summary": _summarize_heldout(heldout_rows),
        "evaluation": {
            "heldout_examples": heldout_rows,
            "support_overlap": pairwise_jaccard_summary(result_sets),
        },
    }
    arrays = {
        "text_embeddings": np.asarray(embeddings, dtype=np.float32),
        "condition_vectors": np.asarray(condition_vectors, dtype=np.float32),
        "memory_targets": np.asarray(memory_targets, dtype=np.float32),
        "example_window_indices": target_indices.astype(np.int64),
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "test_indices": np.asarray(test_indices, dtype=np.int64),
    }
    adapter_path = Path(output_dir) / "text_memory_bridge_adapter.pt"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": bridge.state_dict(),
            "embedding_dim": int(embeddings.shape[1]),
            "condition_dim": int(memory_targets.shape[1]),
            "training": training_meta,
        },
        adapter_path,
    )
    return report, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--support-arrays", type=Path, default=DEFAULT_SUPPORT_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-backend", choices=["hash", "openai"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-retry-attempts", type=int, default=6)
    parser.add_argument("--embedding-retry-sleep", type=float, default=2.0)
    parser.add_argument("--hash-dim", type=int, default=512)
    parser.add_argument("--adapter-steps", type=int, default=1000)
    parser.add_argument("--adapter-batch-size", type=int, default=1024)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--max-query-windows", type=int, default=0)
    parser.add_argument("--query-view", default="full_professional")
    parser.add_argument("--start-weight", type=float, default=0.0)
    parser.add_argument("--support-pool-size", type=int, default=8)
    parser.add_argument("--grounding-gate", action="store_true")
    parser.add_argument("--max-grounding-claims", type=int, default=4)
    parser.add_argument("--max-grounding-mismatches", type=int, default=0)
    parser.add_argument("--top3-90", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = _read_jsonl(args.cards_jsonl)
    assert_cards_allowed_for_retrieval(cards, path=args.cards_jsonl)
    device = args.device
    if str(device) == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    report, arrays = build_text_memory_bridge_report(
        cards=cards,
        support_report=_load_json(args.support_report),
        support_arrays_path=Path(args.support_arrays),
        output_dir=output_dir,
        embedding_backend=str(args.embedding_backend),
        embedding_model=str(args.embedding_model),
        dotenv_path=str(args.dotenv),
        embedding_batch_size=int(args.embedding_batch_size),
        embedding_retry_attempts=int(args.embedding_retry_attempts),
        embedding_retry_sleep=float(args.embedding_retry_sleep),
        hash_dim=int(args.hash_dim),
        adapter_steps=int(args.adapter_steps),
        adapter_batch_size=int(args.adapter_batch_size),
        adapter_lr=float(args.adapter_lr),
        hidden_dim=int(args.hidden_dim) if int(args.hidden_dim) > 0 else None,
        mse_weight=float(args.mse_weight),
        cosine_weight=float(args.cosine_weight),
        contrastive_weight=float(args.contrastive_weight),
        contrastive_temperature=float(args.contrastive_temperature),
        seed=int(args.seed),
        device=device,
        top_k=int(args.top_k),
        temporal_gap=int(args.temporal_gap),
        max_query_windows=int(args.max_query_windows),
        query_view=str(args.query_view),
        start_weight=float(args.start_weight),
        support_pool_size=int(args.support_pool_size),
        grounding_gate=bool(args.grounding_gate),
        max_grounding_claims=int(args.max_grounding_claims),
        max_grounding_mismatches=int(args.max_grounding_mismatches),
        top3_90=bool(args.top3_90),
    )
    report_path = output_dir / "text_memory_bridge_report.json"
    arrays_path = output_dir / "text_memory_bridge_arrays.npz"
    model_path = output_dir / "text_memory_bridge_adapter.pt"
    report["cards_path"] = str(args.cards_jsonl)
    report["artifact_paths"] = {
        "report": str(report_path),
        "arrays": str(arrays_path),
        "adapter": str(model_path),
    }
    _write_json(report_path, report)
    np.savez_compressed(arrays_path, **arrays)
    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(report_path),
                "arrays": str(arrays_path),
                "embedding_backend": str(args.embedding_backend),
                "embedding_model": str(args.embedding_model),
                "embedding_count": int(arrays["text_embeddings"].shape[0]),
                "heldout_query_count": len(report["evaluation"]["heldout_examples"]),
                "mean_true_memory_cosine": report["summary"]["mean_true_memory_cosine"],
                "median_true_memory_rank": report["summary"]["median_true_memory_rank"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
