#!/usr/bin/env python
"""Train/evaluate a text-space contrastive narrative retriever.

Stage 1 of the current NL autoresearch branch stays in OpenAI embedding space:
same historical prefix across multiple Codex-authored narrative views is treated
as a positive, while directionally incompatible prefixes are pushed apart. The
output is a support-selection report only; frozen SNI rollout and top3/90
assembly are unchanged and evaluated downstream.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
    _read_jsonl,
    _split_indices_from_support_report,
    _window_index,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    _candidate_view_rows,
    _rank_by_embeddings,
    embed_with_cache,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    assert_cards_allowed_for_retrieval,
)
from experiments.backfill.block_ar.nl_episode_text_embedding_casebook_support_check import (  # noqa: E402
    DEFAULT_CASEBOOK_JSON,
    DEFAULT_CARDS_JSONL,
    DEFAULT_EMBEDDING_ARRAYS,
    DEFAULT_PROJECTED_REVIEW,
    DEFAULT_SUPPORT_REPORT,
    _card_summary,
    _case_rows,
    _index_projected_review,
    _top3_90,
    _write_json,
)
from experiments.backfill.block_ar.nl_episode_text_memory_bridge_report import (  # noqa: E402
    FACTOR_MARKETS,
    _parse_factor_grounding,
)
from experiments.backfill.block_ar.nl_text_conditioning import normalize_rows  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_text_space_contrastive_983b"
)


class TextSpaceAdapter(nn.Module):
    """Small metric adapter over normalized OpenAI text embeddings."""

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


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Text-Space Contrastive Retriever Review",
        "",
        "Purpose: evaluate Stage 1 retrieval that stays in OpenAI embedding space "
        "but learns a financial-condition metric from multi-view positives and "
        "directional hard negatives.",
        "",
        "## Summary",
        "",
        "| Case | Contrastive pass | Raw OpenAI pass | Projected pass | Initial read |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["summary"]["cases"]:
        lines.append(
            "| {label} | {contrastive}/3 | {raw}/3 | {projected}/3 | {read} |".format(
                label=row["label"],
                contrastive=row["contrastive_direction_pass_count"],
                raw=row["raw_openai_direction_pass_count"],
                projected=row["projected_direction_pass_count"],
                read=row["initial_read"],
            )
        )
    lines.extend(
        [
            "",
            "## Training Diagnostics",
            "",
            f"- raw same-window recall@1: {report['training_diagnostics']['raw_same_window_recall_at_1']:.4f}",
            f"- adapted same-window recall@1: {report['training_diagnostics']['adapted_same_window_recall_at_1']:.4f}",
            f"- final loss: {report['training']['loss_trace'][-1]['loss']:.4f}",
            "",
            "## Case Details",
        ]
    )
    for case in report["queries"]:
        lines.extend(
            [
                "",
                f"### {case['label']}",
                "",
                f"Grounded implications: {case['grounded_implications']}",
                "",
                "| Rank | Support | Title | Archetype | Confidence | Pass | Score |",
                "|---:|---|---|---|---|---|---:|",
            ]
        )
        for item in case["contrastive_top3_90"]:
            lines.append(
                "| {rank} | {wid} | {title} | {arch} | {conf} | {status} | {score:.4f} |".format(
                    rank=item["rank"],
                    wid=item["window_id"],
                    title=item["title"],
                    arch=item["archetype"],
                    conf=item["confidence"],
                    status=item["direction_check"]["status"],
                    score=item["score"],
                )
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _card_claim_signs(card: dict[str, Any]) -> np.ndarray:
    vector = np.zeros(len(FACTOR_MARKETS), dtype=np.float32)
    market_to_pos = {name: pos for pos, name in enumerate(FACTOR_MARKETS)}
    for row in _parse_factor_grounding(card):
        market = str(row["market"]).upper()
        if market in market_to_pos:
            vector[market_to_pos[market]] = float(row["sign"])
    return vector


def _candidate_claim_matrix(
    candidate_rows: list[dict[str, Any]], card_by_index: dict[int, dict[str, Any]]
) -> np.ndarray:
    signs = np.zeros((len(candidate_rows), len(FACTOR_MARKETS)), dtype=np.float32)
    cache: dict[int, np.ndarray] = {}
    for pos, row in enumerate(candidate_rows):
        idx = int(row["window_index"])
        if idx not in cache:
            cache[idx] = _card_claim_signs(card_by_index.get(idx, {}))
        signs[pos] = cache[idx]
    return signs


def _incompatible_mask(signs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    products = signs[:, None, :] * signs[None, :, :]
    incompatible = (products < 0).any(dim=-1)
    not_same = labels[:, None] != labels[None, :]
    return incompatible & not_same


def _supervised_contrastive_loss(
    z: torch.Tensor, labels: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    sim = z @ z.T / max(float(temperature), 1e-6)
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    denom_mask = ~eye
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return torch.zeros((), dtype=z.dtype, device=z.device)
    sim = sim.masked_fill(~denom_mask, -1e9)
    log_denom = torch.logsumexp(sim, dim=1)
    log_pos = torch.logsumexp(sim.masked_fill(~positive, -1e9), dim=1)
    return -(log_pos[valid] - log_denom[valid]).mean()


def _train_adapter(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    signs: np.ndarray,
    output_dim: int,
    hidden_dim: int,
    steps: int,
    windows_per_batch: int,
    lr: float,
    temperature: float,
    incompatible_weight: float,
    margin: float,
    seed: int,
    device: str,
) -> tuple[TextSpaceAdapter, dict[str, Any]]:
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    device_t = torch.device(device)
    x_all = torch.from_numpy(normalize_rows(embeddings).astype(np.float32)).to(device_t)
    labels_np = np.asarray(labels, dtype=np.int64)
    signs_all = torch.from_numpy(np.asarray(signs, dtype=np.float32)).to(device_t)
    by_label: dict[int, list[int]] = defaultdict(list)
    for pos, label in enumerate(labels_np):
        by_label[int(label)].append(int(pos))
    eligible = np.asarray(
        [label for label, rows in by_label.items() if len(rows) >= 2],
        dtype=np.int64,
    )
    if eligible.size == 0:
        raise ValueError("need at least one label with two narrative views")
    adapter = TextSpaceAdapter(
        embedding_dim=x_all.shape[1],
        output_dim=int(output_dim),
        hidden_dim=int(hidden_dim),
    ).to(device_t)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=1e-4)
    trace: list[dict[str, float]] = []
    batch_windows = min(max(1, int(windows_per_batch)), int(eligible.size))
    for step in range(int(steps)):
        chosen = rng.choice(eligible, size=batch_windows, replace=batch_windows > eligible.size)
        batch_indices: list[int] = []
        for label in chosen:
            rows = by_label[int(label)]
            picked = rng.choice(rows, size=2, replace=len(rows) < 2)
            batch_indices.extend(int(pos) for pos in picked)
        batch_np = np.asarray(batch_indices, dtype=np.int64)
        batch = torch.from_numpy(batch_np).long().to(device_t)
        batch_labels = torch.from_numpy(labels_np[batch_np]).long().to(device_t)
        z = adapter(x_all[batch])
        supcon = _supervised_contrastive_loss(
            z, batch_labels, temperature=float(temperature)
        )
        incompat = _incompatible_mask(signs_all[batch], batch_labels)
        if bool(incompat.any()):
            pair_sim = z @ z.T
            margin_loss = F.relu(pair_sim[incompat] - float(margin)).mean()
        else:
            margin_loss = torch.zeros((), dtype=z.dtype, device=z.device)
        loss = supcon + float(incompatible_weight) * margin_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
        opt.step()
        if step in {0, int(steps) - 1} or (step + 1) % max(1, int(steps) // 10) == 0:
            trace.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "supcon": float(supcon.detach().cpu()),
                    "direction_margin": float(margin_loss.detach().cpu()),
                }
            )
    return adapter, {
        "steps": int(steps),
        "windows_per_batch": int(batch_windows),
        "lr": float(lr),
        "temperature": float(temperature),
        "incompatible_weight": float(incompatible_weight),
        "margin": float(margin),
        "seed": int(seed),
        "device": str(device_t),
        "loss_trace": trace,
    }


def _encode_adapter(
    adapter: TextSpaceAdapter,
    embeddings: np.ndarray,
    *,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    device_t = torch.device(device)
    x = torch.from_numpy(normalize_rows(embeddings).astype(np.float32)).to(device_t)
    out: list[np.ndarray] = []
    adapter.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], int(batch_size)):
            z = adapter(x[start : start + int(batch_size)]).detach().cpu().numpy()
            out.append(z.astype(np.float32))
    return np.concatenate(out, axis=0)


def _same_window_recall_at_1(
    *,
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(int(seed))
    labels_np = np.asarray(labels, dtype=np.int64)
    eligible = np.flatnonzero(
        np.asarray(
            [np.sum(labels_np == label) >= 2 for label in labels_np],
            dtype=bool,
        )
    )
    if eligible.size == 0:
        return math.nan
    sample = rng.choice(
        eligible,
        size=min(int(sample_size), int(eligible.size)),
        replace=False,
    )
    normalized = normalize_rows(vectors.astype(np.float32))
    hits = 0
    for pos in sample:
        scores = normalized @ normalized[int(pos)]
        scores[int(pos)] = -np.inf
        best = int(np.argmax(scores))
        if int(labels_np[best]) == int(labels_np[int(pos)]):
            hits += 1
    return float(hits / max(1, len(sample)))


def build_text_space_contrastive_report(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cards = _read_jsonl(Path(args.cards_jsonl))
    assert_cards_allowed_for_retrieval(cards, path=Path(args.cards_jsonl))
    support_report = _load_json(Path(args.support_report))
    train_indices, _test_indices = _split_indices_from_support_report(support_report)
    card_by_index = {_window_index(card): card for card in cards}
    train_cards = [card_by_index[idx] for idx in train_indices if idx in card_by_index]
    candidate_rows, candidate_texts = _candidate_view_rows(
        train_cards, tuple(DEFAULT_VIEW_NAMES)
    )

    with np.load(Path(args.embedding_arrays)) as payload:
        text_embeddings = payload["text_embeddings"].astype(np.float32)
        candidate_count = int(payload["candidate_text_count"][0])
    if candidate_count != len(candidate_rows) or candidate_count != len(candidate_texts):
        raise ValueError(
            f"candidate cache mismatch: arrays={candidate_count}, rows={len(candidate_rows)}, "
            f"texts={len(candidate_texts)}"
        )
    candidate_embeddings = text_embeddings[:candidate_count]
    labels = np.asarray([int(row["window_index"]) for row in candidate_rows], dtype=np.int64)
    signs = _candidate_claim_matrix(candidate_rows, card_by_index)

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    adapter, training_meta = _train_adapter(
        embeddings=candidate_embeddings,
        labels=labels,
        signs=signs,
        output_dim=int(args.output_dim),
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        windows_per_batch=int(args.windows_per_batch),
        lr=float(args.lr),
        temperature=float(args.temperature),
        incompatible_weight=float(args.incompatible_weight),
        margin=float(args.margin),
        seed=int(args.seed),
        device=device,
    )
    adapted_candidates = _encode_adapter(
        adapter, candidate_embeddings, device=device, batch_size=int(args.encode_batch_size)
    )

    cases = _case_rows(_load_json(Path(args.casebook_json)))
    query_embeddings, query_meta = embed_with_cache(
        [case["text"] for case in cases],
        output_dir=Path(args.output_dir) / "query_embeddings",
        backend="openai",
        model=str(args.embedding_model),
        dotenv_path=str(args.dotenv),
        batch_size=int(args.embedding_batch_size),
        hash_dim=512,
        retry_attempts=int(args.embedding_retry_attempts),
        retry_sleep=float(args.embedding_retry_sleep),
    )
    adapted_queries = _encode_adapter(
        adapter, query_embeddings, device=device, batch_size=int(args.encode_batch_size)
    )
    raw_report = _load_json(Path(args.raw_openai_report))
    raw_by_case = {
        str(row.get("case_name", "")): row
        for row in raw_report.get("summary", {}).get("cases", [])
        if isinstance(row, dict)
    }
    projected_by_case = _index_projected_review(Path(args.projected_review))

    query_reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for case, query_vector in zip(cases, adapted_queries, strict=True):
        ranked = _rank_by_embeddings(
            query_vector=query_vector,
            candidate_vectors=adapted_candidates,
            candidate_rows=candidate_rows,
            top_k=int(args.support_pool_size),
            temporal_gap=int(args.temporal_gap),
        )
        selected = _top3_90(ranked)
        contrastive_top3 = [
            _card_summary(
                item=item,
                card=card_by_index[int(item["window_index"])],
                claims=list(case["claims"]),
            )
            for item in selected
        ]
        projected_items = projected_by_case.get(str(case["case_name"]), {}).get(
            "projected_top3_90", []
        )
        projected_pass = sum(
            1
            for item in projected_items[:3]
            if isinstance(item, dict)
            and str(item.get("direction_check", {}).get("status", "")) == "pass"
        )
        raw_pass = int(
            raw_by_case.get(str(case["case_name"]), {}).get(
                "openai_direction_pass_count", 0
            )
        )
        contrastive_pass = sum(
            1
            for item in contrastive_top3
            if str(item["direction_check"].get("status", "")) == "pass"
        )
        read = (
            "improves raw but still trails projected"
            if contrastive_pass > raw_pass and contrastive_pass < projected_pass
            else "matches projected on direction checks"
            if contrastive_pass >= projected_pass
            else "does not improve raw direction checks"
        )
        summary_rows.append(
            {
                "case_name": case["case_name"],
                "label": case["label"],
                "contrastive_direction_pass_count": int(contrastive_pass),
                "raw_openai_direction_pass_count": int(raw_pass),
                "projected_direction_pass_count": int(projected_pass),
                "initial_read": read,
                "contrastive_titles": [item["title"] for item in contrastive_top3],
            }
        )
        query_reports.append(
            {
                **case,
                "contrastive_top3_90": contrastive_top3,
            }
        )

    raw_recall = _same_window_recall_at_1(
        vectors=candidate_embeddings,
        labels=labels,
        sample_size=int(args.recall_sample_size),
        seed=int(args.seed),
    )
    adapted_recall = _same_window_recall_at_1(
        vectors=adapted_candidates,
        labels=labels,
        sample_size=int(args.recall_sample_size),
        seed=int(args.seed),
    )
    report = {
        "schema_version": "nl_episode_text_space_contrastive_retriever_v1",
        "status": "ok",
        "scope_note": (
            "Stage 1 isolated text-space metric experiment. It consumes saved "
            "Codex-authored narratives and cached OpenAI embeddings; it does not "
            "regenerate narratives, project to SNI memory, or alter top3/90 assembly."
        ),
        "inputs": {
            "cards_jsonl": str(args.cards_jsonl),
            "support_report": str(args.support_report),
            "embedding_arrays": str(args.embedding_arrays),
            "casebook_json": str(args.casebook_json),
            "raw_openai_report": str(args.raw_openai_report),
            "projected_review": str(args.projected_review),
        },
        "embedding_meta": {
            **query_meta,
            "candidate_embedding_count": int(candidate_count),
            "candidate_view_names": list(DEFAULT_VIEW_NAMES),
        },
        "training": training_meta,
        "training_diagnostics": {
            "raw_same_window_recall_at_1": float(raw_recall),
            "adapted_same_window_recall_at_1": float(adapted_recall),
        },
        "retrieval_config": {
            "method": "text_space_contrastive_top3_90",
            "support_pool_size": int(args.support_pool_size),
            "temporal_gap": int(args.temporal_gap),
            "output_dim": int(args.output_dim),
            "hidden_dim": int(args.hidden_dim),
        },
        "queries": query_reports,
        "summary": {
            "case_count": int(len(summary_rows)),
            "cases": summary_rows,
            "contrastive_total_direction_pass": int(
                sum(row["contrastive_direction_pass_count"] for row in summary_rows)
            ),
            "raw_openai_total_direction_pass": int(
                sum(row["raw_openai_direction_pass_count"] for row in summary_rows)
            ),
            "projected_total_direction_pass": int(
                sum(row["projected_direction_pass_count"] for row in summary_rows)
            ),
        },
    }
    arrays = {
        "candidate_embeddings": candidate_embeddings.astype(np.float32),
        "adapted_candidate_embeddings": adapted_candidates.astype(np.float32),
        "query_embeddings": query_embeddings.astype(np.float32),
        "adapted_query_embeddings": adapted_queries.astype(np.float32),
        "candidate_window_indices": labels.astype(np.int64),
        "candidate_claim_signs": signs.astype(np.float32),
    }
    return report, arrays, adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--embedding-arrays", type=Path, default=DEFAULT_EMBEDDING_ARRAYS)
    parser.add_argument("--casebook-json", type=Path, default=DEFAULT_CASEBOOK_JSON)
    parser.add_argument("--projected-review", type=Path, default=DEFAULT_PROJECTED_REVIEW)
    parser.add_argument(
        "--raw-openai-report",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR.parent
            / "six_case_true_openai_embedding_support_check_983a"
            / "true_openai_embedding_casebook_support_check.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-retry-attempts", type=int, default=6)
    parser.add_argument("--embedding-retry-sleep", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--windows-per-batch", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--incompatible-weight", type=float, default=0.15)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--recall-sample-size", type=int, default=2000)
    parser.add_argument("--support-pool-size", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report, arrays, adapter = build_text_space_contrastive_report(args)
    report_path = output_dir / "text_space_contrastive_retriever_report.json"
    arrays_path = output_dir / "text_space_contrastive_retriever_arrays.npz"
    adapter_path = output_dir / "text_space_contrastive_adapter.pt"
    markdown_path = output_dir / "text_space_contrastive_retriever_report.md"
    report["artifact_paths"] = {
        "report": str(report_path),
        "arrays": str(arrays_path),
        "adapter": str(adapter_path),
        "markdown": str(markdown_path),
    }
    _write_json(report_path, report)
    np.savez_compressed(arrays_path, **arrays)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "retrieval_config": report["retrieval_config"],
            "training": report["training"],
        },
        adapter_path,
    )
    _write_markdown(markdown_path, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "case_count": int(report["summary"]["case_count"]),
                "contrastive_total_direction_pass": int(
                    report["summary"]["contrastive_total_direction_pass"]
                ),
                "raw_openai_total_direction_pass": int(
                    report["summary"]["raw_openai_total_direction_pass"]
                ),
                "projected_total_direction_pass": int(
                    report["summary"]["projected_total_direction_pass"]
                ),
                "raw_same_window_recall_at_1": float(
                    report["training_diagnostics"]["raw_same_window_recall_at_1"]
                ),
                "adapted_same_window_recall_at_1": float(
                    report["training_diagnostics"]["adapted_same_window_recall_at_1"]
                ),
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
