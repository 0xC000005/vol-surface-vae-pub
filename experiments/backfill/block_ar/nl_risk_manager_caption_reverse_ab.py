#!/usr/bin/env python
"""Reverse-direction TestFlight for risk-manager captions.

This script tests whether richer risk-manager-grade scenario-to-text captions
help the reverse path:

    caption text -> text embedding -> SNI terminal memory -> support mixture

It deliberately focuses on bridge/support evidence first. It writes a bridge-
style report that `nl_scenario_level_evaluation.py` can consume for a small
rollout A/B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    NarrativeAdapter,
    hash_text_embeddings,
    train_narrative_adapter,
)
from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (  # noqa: E402
    RiskManagerCaptionV2,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    _load_bridge_adapter,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    cosine_similarity,
    embed_texts_with_openai,
    normalize_rows,
)


DEFAULT_PIPELINE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_full_906b_all_windows/narrative_pipeline_report.json"
)
DEFAULT_BRIDGE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_report.json"
)
DEFAULT_BRIDGE_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_arrays.npz"
)
DEFAULT_BRIDGE_ADAPTER = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_adapter.pt"
)
DEFAULT_API_CAPTION_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_testflight_916a_openai/"
    "risk_manager_caption_v2_testflight_report.json"
)
DEFAULT_CODEX_CAPTION_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_v2_codex_probe_916c"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_caption_reverse_ab_916e"
)
DEFAULT_SIMPLE_DEMO_NARRATIVE = (
    "This has the shape of a fragile risk-on rebound: equities are recovering, "
    "volatility is compressing, spreads are stabilizing, and investors appear "
    "to be rotating back into carry. The forward risk is that a volatility "
    "reversal quickly unwinds the move."
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


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _slug(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "p")
    )


def _hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _caption_report_by_window(path: str | Path) -> dict[str, RiskManagerCaptionV2]:
    report = _load_json(path)
    captions: dict[str, RiskManagerCaptionV2] = {}
    for row in report.get("captions", []):
        cap = RiskManagerCaptionV2.model_validate(row)
        captions[cap.window_id] = cap
    return captions


def _codex_caption_paths(codex_dir: str | Path) -> list[Path]:
    base = Path(codex_dir)
    candidates = list(base.glob("codex_gpt55_caption*.json"))
    if (base / "captions").is_dir():
        candidates.extend((base / "captions").glob("codex_gpt55_caption*.json"))
    paths = []
    seen: set[Path] = set()
    for path in sorted(candidates):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if "failed_validator" in path.name:
            continue
        if path.name.endswith("_pass.json") or "joint39_val_" in path.name:
            paths.append(path)
    return paths


def _codex_captions_by_window(codex_dir: str | Path) -> dict[str, RiskManagerCaptionV2]:
    captions: dict[str, RiskManagerCaptionV2] = {}
    for path in _codex_caption_paths(codex_dir):
        cap = RiskManagerCaptionV2.model_validate_json(path.read_text(encoding="utf-8"))
        captions[cap.window_id] = cap
    return captions


def _matched_caption_window_ids(
    *,
    api_captions: dict[str, RiskManagerCaptionV2],
    codex_captions: dict[str, RiskManagerCaptionV2],
    bundles: dict[str, dict[str, Any]],
    window_index_by_id: dict[str, int],
) -> list[str]:
    """Use any captioned window, not only the older API-caption subset."""

    caption_window_ids = set(api_captions) | set(codex_captions)
    return [
        window_id
        for window_id in sorted(caption_window_ids)
        if window_id in bundles and window_id in window_index_by_id
    ]


def _first_narrative_text(bundle: dict[str, Any], preferred_id: str) -> str:
    for row in bundle.get("narratives", []):
        if isinstance(row, dict) and str(row.get("id")) == preferred_id:
            return str(row.get("text", "")).strip()
    for row in bundle.get("narratives", []):
        if isinstance(row, dict) and row.get("text"):
            return str(row["text"]).strip()
    return ""


def _observed_fact_text(bundle: dict[str, Any]) -> str:
    for row in bundle.get("narratives", []):
        if isinstance(row, dict) and row.get("observed_fact_tokens"):
            return str(row["observed_fact_tokens"]).strip()
    implications = []
    for item in bundle.get("market_implications", []):
        if not isinstance(item, dict):
            continue
        market = str(item.get("market", "")).strip()
        direction = str(item.get("direction", "")).strip()
        magnitude = str(item.get("magnitude", "")).strip()
        if market and direction:
            implications.append(f"{market}: {direction} {magnitude}".strip())
    return "; ".join(implications)


def _structured_caption_text(caption: RiskManagerCaptionV2) -> str:
    parts = [
        f"TITLE: {caption.scenario_title}",
        f"ARCHETYPE: {caption.archetype}",
        f"CURRENT_STATE: {caption.current_market_state}",
        f"MECHANICS: {caption.mechanical_summary}",
        f"TRIGGER: {caption.trigger}",
        f"TRANSMISSION: {caption.transmission}",
        f"CROSS_ASSET: {caption.cross_asset_reaction}",
        f"SEQUENCE: {caption.sequence}",
        f"PORTFOLIO_VULNERABILITY: {caption.portfolio_vulnerability}",
        f"RISK_MANAGER_IMPLICATION: {caption.risk_manager_implication}",
        f"TRAINING_CAPTION: {caption.training_caption}",
    ]
    if caption.evidence_used:
        parts.append("EVIDENCE: " + " | ".join(caption.evidence_used))
    if caption.ambiguity_flags:
        parts.append("AMBIGUITY: " + " | ".join(caption.ambiguity_flags))
    return "\n".join(parts)


def _fused_fact_caption_text(*, fact_text: str, caption_text: str) -> str:
    """Combine explicit direction facts with richer professional caption text."""

    fact = str(fact_text).strip()
    caption = str(caption_text).strip()
    if not fact:
        return caption
    if not caption:
        return f"FACT_TOKENS: {fact}"
    return f"FACT_TOKENS: {fact}\n\nPROFESSIONAL_NARRATIVE:\n{caption}"


def build_text_variants(
    *,
    window_id: str,
    bundle: dict[str, Any],
    api_caption: RiskManagerCaptionV2 | None,
    codex_caption: RiskManagerCaptionV2 | None,
    include_generic_demo: bool,
) -> list[dict[str, Any]]:
    """Build simple and risk-manager-grade text variants for one window."""

    variants: list[dict[str, Any]] = []
    legacy_anchor = _first_narrative_text(bundle, "revised_market_description")
    legacy_terse = _first_narrative_text(bundle, "description_terse_trader")
    fact_text = _observed_fact_text(bundle)
    if legacy_anchor:
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "legacy_anchor",
                "variant_group": "simple",
                "provider": "legacy_pipeline",
                "text": legacy_anchor,
            }
        )
    if legacy_terse and legacy_terse != legacy_anchor:
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "legacy_terse",
                "variant_group": "simple",
                "provider": "legacy_pipeline",
                "text": legacy_terse,
            }
        )
    if fact_text:
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "simple_fact_tokens",
                "variant_group": "simple",
                "provider": "legacy_pipeline",
                "text": fact_text,
            }
        )
    if include_generic_demo:
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "generic_old_demo_story",
                "variant_group": "generic_demo",
                "provider": "manual_demo",
                "text": DEFAULT_SIMPLE_DEMO_NARRATIVE,
            }
        )
    if api_caption is not None:
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "api_v2_training_caption",
                "variant_group": "rich_api",
                "provider": "gpt-5.4-mini-api",
                "text": api_caption.training_caption,
            }
        )
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "api_v2_structured_caption",
                "variant_group": "rich_api",
                "provider": "gpt-5.4-mini-api",
                "text": _structured_caption_text(api_caption),
            }
        )
        if fact_text:
            variants.append(
                {
                    "window_id": window_id,
                    "variant_id": "api_v2_fused_fact_training_caption",
                    "variant_group": "fused_api",
                    "provider": "gpt-5.4-mini-api",
                    "text": _fused_fact_caption_text(
                        fact_text=fact_text,
                        caption_text=api_caption.training_caption,
                    ),
                }
            )
            variants.append(
                {
                    "window_id": window_id,
                    "variant_id": "api_v2_fused_fact_structured_caption",
                    "variant_group": "fused_api",
                    "provider": "gpt-5.4-mini-api",
                    "text": _fused_fact_caption_text(
                        fact_text=fact_text,
                        caption_text=_structured_caption_text(api_caption),
                    ),
                }
            )
    if codex_caption is not None:
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "codex_v2_training_caption",
                "variant_group": "rich_codex",
                "provider": "codex-gpt-5.5-xhigh",
                "text": codex_caption.training_caption,
            }
        )
        variants.append(
            {
                "window_id": window_id,
                "variant_id": "codex_v2_structured_caption",
                "variant_group": "rich_codex",
                "provider": "codex-gpt-5.5-xhigh",
                "text": _structured_caption_text(codex_caption),
            }
        )
        if fact_text:
            variants.append(
                {
                    "window_id": window_id,
                    "variant_id": "codex_v2_fused_fact_training_caption",
                    "variant_group": "fused_codex",
                    "provider": "codex-gpt-5.5-xhigh",
                    "text": _fused_fact_caption_text(
                        fact_text=fact_text,
                        caption_text=codex_caption.training_caption,
                    ),
                }
            )
            variants.append(
                {
                    "window_id": window_id,
                    "variant_id": "codex_v2_fused_fact_structured_caption",
                    "variant_group": "fused_codex",
                    "provider": "codex-gpt-5.5-xhigh",
                    "text": _fused_fact_caption_text(
                        fact_text=fact_text,
                        caption_text=_structured_caption_text(codex_caption),
                    ),
                }
            )
    return variants


def select_diverse_support_rows(
    query: np.ndarray,
    memory_targets: np.ndarray,
    *,
    candidate_indices: list[int],
    top_k: int,
    temporal_gap: int,
    exclude_window_index: int | None = None,
    window_ids: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Select top-k memory supports while avoiding near-duplicate windows."""

    candidates = [int(idx) for idx in candidate_indices]
    if exclude_window_index is not None:
        candidates = [idx for idx in candidates if idx != int(exclude_window_index)]
    if not candidates:
        return []
    target_norm = normalize_rows(np.asarray(memory_targets, dtype=np.float32)[candidates])
    q = np.asarray(query, dtype=np.float32)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-8)
    sims = target_norm @ q_norm
    order = np.argsort(-sims)
    selected: list[dict[str, Any]] = []
    gap = max(0, int(temporal_gap))
    for local_idx in order:
        window_index = int(candidates[int(local_idx)])
        if gap > 0 and any(
            abs(window_index - int(row["window_index"])) < gap for row in selected
        ):
            continue
        selected.append(
            {
                "rank": len(selected) + 1,
                "window_index": window_index,
                "window_id": (window_ids or {}).get(window_index, ""),
                "cosine": _round_float(float(sims[int(local_idx)])),
            }
        )
        if len(selected) >= int(top_k):
            break
    return selected


def _rank_full_pool(query: np.ndarray, memory_targets: np.ndarray, target_index: int) -> int:
    targets = normalize_rows(np.asarray(memory_targets, dtype=np.float32))
    q = np.asarray(query, dtype=np.float32)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-8)
    sims = targets @ q_norm
    ordered = list(np.argsort(-sims))
    return int(ordered.index(int(target_index)) + 1)


def _split_name(index: int, train_indices: set[int], test_indices: set[int]) -> str:
    if int(index) in train_indices:
        return "train"
    if int(index) in test_indices:
        return "test"
    return "excluded"


def _mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return None
    return _round_float(float(np.mean(finite)))


def compare_variant_groups(
    rows: list[dict[str, Any]],
    *,
    baseline_group: str = "simple",
) -> dict[str, Any]:
    """Aggregate variant rows and compute deltas versus a baseline group."""

    groups = sorted({str(row["variant_group"]) for row in rows})
    group_summary: dict[str, Any] = {}
    for group in groups:
        group_rows = [row for row in rows if str(row["variant_group"]) == group]
        group_summary[group] = {
            "count": len(group_rows),
            "mean_target_cosine": _mean([float(row["target_cosine"]) for row in group_rows]),
            "mean_true_rank_full_pool": _mean(
                [float(row["true_rank_full_pool"]) for row in group_rows]
            ),
            "top_support_hit_rate": _mean(
                [1.0 if bool(row.get("top_support_hit")) else 0.0 for row in group_rows]
            ),
            "mean_nearest_support_abs_window_distance": _mean(
                [
                    float(row["nearest_support_abs_window_distance"])
                    for row in group_rows
                    if row.get("nearest_support_abs_window_distance") is not None
                ]
            ),
            "mean_support_cosine": _mean(
                [
                    float(row["mean_support_cosine"])
                    for row in group_rows
                    if row.get("mean_support_cosine") is not None
                ]
            ),
        }
    baseline = group_summary.get(baseline_group)
    deltas: dict[str, Any] = {}
    if baseline:
        for group, summary in group_summary.items():
            if group == baseline_group:
                continue
            deltas[group] = {}
            for key in (
                "mean_target_cosine",
                "mean_true_rank_full_pool",
                "top_support_hit_rate",
                "mean_nearest_support_abs_window_distance",
                "mean_support_cosine",
            ):
                left = summary.get(key)
                right = baseline.get(key)
                deltas[group][f"{key}_delta"] = (
                    None
                    if left is None or right is None
                    else _round_float(float(left) - float(right))
                )
    return {
        "baseline_group": baseline_group,
        "groups": group_summary,
        "deltas_vs_simple": deltas,
    }


def _embed_with_cache(
    texts: list[str],
    *,
    model: str,
    backend: str,
    cache_dir: Path,
    dotenv_path: str,
    batch_size: int,
    hash_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"embeddings_{_slug(model)}_{_hash_texts(texts)}.npz"
    if cache_path.exists():
        with np.load(cache_path) as payload:
            arr = payload["embeddings"].copy()
        return arr, {
            "backend": backend,
            "model": model,
            "cache_path": str(cache_path),
            "cache_hit": True,
            "embedding_count": len(texts),
            "embedding_dim": int(arr.shape[1]),
        }
    if backend == "hash":
        arr = hash_text_embeddings(texts, dim=int(hash_dim))
    elif backend == "openai":
        arr = embed_texts_with_openai(
            texts,
            model=model,
            dotenv_path=dotenv_path,
            batch_size=int(batch_size),
        )
    else:
        raise ValueError(f"unsupported embedding backend: {backend}")
    np.savez_compressed(cache_path, embeddings=np.asarray(arr, dtype=np.float32))
    return np.asarray(arr, dtype=np.float32), {
        "backend": backend,
        "model": model,
        "cache_path": str(cache_path),
        "cache_hit": False,
        "embedding_count": len(texts),
        "embedding_dim": int(arr.shape[1]),
    }


def _load_or_train_adapter(
    *,
    model: str,
    embeddings: np.ndarray,
    examples: list[dict[str, Any]],
    memory_targets: np.ndarray,
    train_indices: list[int],
    saved_adapter: Path,
    use_saved_adapter: bool,
    steps: int,
    seed: int,
    device: str,
) -> tuple[NarrativeAdapter, dict[str, Any]]:
    if (
        use_saved_adapter
        and saved_adapter.exists()
        and int(embeddings.shape[1]) == 1536
        and "text-embedding-3-small" in str(model)
    ):
        adapter = _load_bridge_adapter(
            saved_adapter,
            embedding_dim=int(embeddings.shape[1]),
            condition_dim=int(memory_targets.shape[1]),
        )
        return adapter, {
            "source": "saved_adapter",
            "path": str(saved_adapter),
            "steps": 0,
        }

    train_set = set(int(idx) for idx in train_indices)
    train_example_indices = [
        idx
        for idx, example in enumerate(examples)
        if int(example["window_index"]) in train_set
    ]
    train_examples = [examples[idx] for idx in train_example_indices]
    target_indices = np.asarray(
        [
            -1 if example["target_index"] is None else int(example["target_index"])
            for example in train_examples
        ],
        dtype=np.int64,
    )
    train_result = train_narrative_adapter(
        embeddings[np.asarray(train_example_indices, dtype=np.int64)],
        memory_targets,
        target_indices,
        [str(example["role"]) for example in train_examples],
        [str(example["window_id"]) for example in train_examples],
        condition_dim=int(memory_targets.shape[1]),
        steps=int(steps),
        seed=int(seed),
        device=device if torch.cuda.is_available() or device == "cpu" else "cpu",
    )
    return train_result["adapter"], {
        "source": "trained_for_embedding_model",
        "steps": int(steps),
        "train_example_count": len(train_examples),
        "loss_first": _round_float(float(train_result["loss_first"])),
        "loss_last": _round_float(float(train_result["loss_last"])),
    }


def run_reverse_ab(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_report = _load_json(args.pipeline_report)
    bridge_report = _load_json(args.bridge_report)
    with np.load(args.bridge_arrays) as payload:
        memory_targets = np.asarray(payload["memory_targets"], dtype=np.float32)
        train_indices = [int(idx) for idx in payload["train_indices"]]
        test_indices = [int(idx) for idx in payload["test_indices"]]
        cached_small_embeddings = np.asarray(payload["text_embeddings"], dtype=np.float32)
    train_set = set(train_indices)
    test_set = set(test_indices)
    examples = build_bridge_examples(pipeline_report)
    bundles = {
        str(bundle["window_id"]): bundle
        for bundle in pipeline_report.get("narrative_bundles", [])
        if isinstance(bundle, dict) and bundle.get("window_id")
    }
    window_id_by_index = {
        idx: str(row.get("window_id", ""))
        for idx, row in enumerate(bridge_report.get("window_metadata", []))
        if isinstance(row, dict)
    }
    window_index_by_id = {value: key for key, value in window_id_by_index.items()}
    api_captions = _caption_report_by_window(args.api_caption_report)
    codex_captions = _codex_captions_by_window(args.codex_caption_dir)
    window_ids = _matched_caption_window_ids(
        api_captions=api_captions,
        codex_captions=codex_captions,
        bundles=bundles,
        window_index_by_id=window_index_by_id,
    )
    if int(args.max_windows) > 0:
        window_ids = window_ids[: int(args.max_windows)]
    if not window_ids:
        raise ValueError("no matched API caption windows found")

    variants: list[dict[str, Any]] = []
    for window_id in window_ids:
        variants.extend(
            build_text_variants(
                window_id=window_id,
                bundle=bundles[window_id],
                api_caption=api_captions.get(window_id),
                codex_caption=codex_captions.get(window_id),
                include_generic_demo=bool(args.include_generic_demo),
            )
        )
    if not variants:
        raise ValueError("no text variants built")

    model_reports: list[dict[str, Any]] = []
    bridge_like_reports: dict[str, str] = {}
    for model in [item.strip() for item in str(args.embedding_models).split(",") if item.strip()]:
        all_example_embeddings = cached_small_embeddings
        corpus_meta: dict[str, Any] = {
            "backend": "cache",
            "model": "text-embedding-3-small",
            "cache_hit": True,
            "embedding_count": int(cached_small_embeddings.shape[0]),
            "embedding_dim": int(cached_small_embeddings.shape[1]),
        }
        if model != "text-embedding-3-small" or bool(args.force_reembed_corpus):
            corpus_texts = [str(example["text"]) for example in examples]
            all_example_embeddings, corpus_meta = _embed_with_cache(
                corpus_texts,
                model=model,
                backend=str(args.embedding_backend),
                cache_dir=output_dir / "embedding_cache",
                dotenv_path=str(args.dotenv),
                batch_size=int(args.embedding_batch_size),
                hash_dim=int(args.hash_dim),
            )
        adapter, adapter_meta = _load_or_train_adapter(
            model=model,
            embeddings=all_example_embeddings,
            examples=examples,
            memory_targets=memory_targets,
            train_indices=train_indices,
            saved_adapter=Path(args.bridge_adapter),
            use_saved_adapter=not bool(args.force_train_adapter),
            steps=int(args.adapter_steps),
            seed=int(args.seed),
            device=str(args.device),
        )
        variant_embeddings, variant_embedding_meta = _embed_with_cache(
            [str(row["text"]) for row in variants],
            model=model,
            backend=str(args.embedding_backend),
            cache_dir=output_dir / "embedding_cache",
            dotenv_path=str(args.dotenv),
            batch_size=int(args.embedding_batch_size),
            hash_dim=int(args.hash_dim),
        )
        adapter.eval()
        adapter_device = next(adapter.parameters()).device
        with torch.no_grad():
            condition_vectors = (
                adapter(
                    torch.from_numpy(normalize_rows(variant_embeddings))
                    .float()
                    .to(adapter_device)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        rows: list[dict[str, Any]] = []
        bridge_rows: list[dict[str, Any]] = []
        for emb_idx, (variant, condition) in enumerate(
            zip(variants, condition_vectors, strict=True)
        ):
            window_id = str(variant["window_id"])
            window_index = int(window_index_by_id[window_id])
            target = memory_targets[window_index]
            top_support = select_diverse_support_rows(
                condition,
                memory_targets,
                candidate_indices=train_indices,
                top_k=int(args.top_k),
                temporal_gap=int(args.temporal_gap),
                exclude_window_index=window_index,
                window_ids=window_id_by_index,
            )
            support_distances = [
                abs(int(row["window_index"]) - window_index) for row in top_support
            ]
            row = {
                "window_id": window_id,
                "window_index": window_index,
                "split": _split_name(window_index, train_set, test_set),
                "embedding_model": model,
                "embedding_index": emb_idx,
                "variant_id": str(variant["variant_id"]),
                "variant_group": str(variant["variant_group"]),
                "provider": str(variant["provider"]),
                "text_word_count": len(str(variant["text"]).split()),
                "target_cosine": _round_float(cosine_similarity(condition, target)),
                "target_mse": _round_float(float(np.mean((condition - target) ** 2))),
                "true_rank_full_pool": _rank_full_pool(
                    condition,
                    memory_targets,
                    window_index,
                ),
                "top_support_hit": any(
                    int(row["window_index"]) == window_index for row in top_support
                ),
                "nearest_support_abs_window_distance": (
                    None if not support_distances else int(min(support_distances))
                ),
                "mean_support_abs_window_distance": (
                    None if not support_distances else _round_float(float(np.mean(support_distances)))
                ),
                "mean_support_cosine": _mean(
                    [float(row["cosine"]) for row in top_support]
                ),
                "support_rows": top_support,
            }
            rows.append(row)
            bridge_rows.append(
                {
                    "window_index": window_index,
                    "window_id": window_id,
                    "embedding_index": emb_idx,
                    "role": "anchor",
                    "kind": str(variant["variant_id"]),
                    "query_id": f"{window_id}::{variant['variant_id']}::{model}",
                    "target_cosine": row["target_cosine"],
                    "true_rank_full_pool": row["true_rank_full_pool"],
                    "top_full_pool": top_support,
                    "top_test_pool": [],
                    "top_train_pool": top_support,
                }
            )

        summary_all = compare_variant_groups(rows, baseline_group="simple")
        non_train_rows = [row for row in rows if str(row["split"]) != "train"]
        summary_non_train = compare_variant_groups(non_train_rows, baseline_group="simple")
        model_slug = _slug(model)
        arrays_path = output_dir / f"reverse_ab_bridge_arrays_{model_slug}.npz"
        np.savez_compressed(
            arrays_path,
            condition_vectors=condition_vectors.astype(np.float32),
            memory_targets=memory_targets.astype(np.float32),
            variant_embeddings=variant_embeddings.astype(np.float32),
            train_indices=np.asarray(train_indices, dtype=np.int64),
            test_indices=np.asarray(test_indices, dtype=np.int64),
        )
        bridge_like = {
            "status": "ok",
            "scope_note": (
                "Bridge-style report for reverse-direction risk-manager-caption "
                "A/B. Query rows are text variants for matched historical windows."
            ),
            "embedding_backend": str(args.embedding_backend),
            "embedding_model": model,
            "input_report": str(args.pipeline_report),
            "split": {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "excluded_indices": [
                    idx
                    for idx in range(int(memory_targets.shape[0]))
                    if idx not in train_set and idx not in test_set
                ],
            },
            "window_metadata": bridge_report.get("window_metadata", []),
            "window_indices": bridge_report.get(
                "window_indices", list(range(int(memory_targets.shape[0])))
            ),
            "summary": summary_all,
            "evaluation": {
                "heldout_examples": bridge_rows,
                "heldout_example_count": len(bridge_rows),
                "heldout_window_count": len(set(row["window_index"] for row in bridge_rows)),
            },
            "artifact_paths": {
                "arrays": str(arrays_path),
                "report": str(output_dir / f"reverse_ab_bridge_report_{model_slug}.json"),
            },
        }
        bridge_report_path = output_dir / f"reverse_ab_bridge_report_{model_slug}.json"
        _write_json(bridge_report_path, bridge_like)
        bridge_like_reports[model] = str(bridge_report_path)
        model_reports.append(
            {
                "embedding_model": model,
                "corpus_embedding": corpus_meta,
                "variant_embedding": variant_embedding_meta,
                "adapter": adapter_meta,
                "variant_count": len(rows),
                "window_count": len(window_ids),
                "summary_all": summary_all,
                "summary_non_train": summary_non_train,
                "rows": rows,
                "artifact_paths": {
                    "bridge_report": str(bridge_report_path),
                    "arrays": str(arrays_path),
                },
            }
        )

    report = {
        "status": "ok",
        "scope_note": (
            "Reverse-direction TestFlight comparing simple legacy/demo text "
            "against risk-manager-grade V2 captions before any full relabeling."
        ),
        "pipeline_report": str(args.pipeline_report),
        "bridge_report": str(args.bridge_report),
        "api_caption_report": str(args.api_caption_report),
        "codex_caption_dir": str(args.codex_caption_dir),
        "window_ids": window_ids,
        "variant_count": len(variants),
        "embedding_models": [row["embedding_model"] for row in model_reports],
        "bridge_like_reports": bridge_like_reports,
        "model_reports": model_reports,
        "artifact_paths": {
            "report": str(output_dir / "caption_reverse_ab_report.json"),
        },
    }
    _write_json(output_dir / "caption_reverse_ab_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", default=str(DEFAULT_PIPELINE_REPORT))
    parser.add_argument("--bridge-report", default=str(DEFAULT_BRIDGE_REPORT))
    parser.add_argument("--bridge-arrays", default=str(DEFAULT_BRIDGE_ARRAYS))
    parser.add_argument("--bridge-adapter", default=str(DEFAULT_BRIDGE_ADAPTER))
    parser.add_argument("--api-caption-report", default=str(DEFAULT_API_CAPTION_REPORT))
    parser.add_argument("--codex-caption-dir", default=str(DEFAULT_CODEX_CAPTION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embedding-models", default="text-embedding-3-small")
    parser.add_argument("--embedding-backend", choices=["openai", "hash"], default="openai")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--hash-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--max-windows", type=int, default=10)
    parser.add_argument("--adapter-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-reembed-corpus", action="store_true")
    parser.add_argument("--force-train-adapter", action="store_true")
    parser.add_argument("--include-generic-demo", action="store_true")
    args = parser.parse_args()
    report = run_reverse_ab(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "window_count": len(report["window_ids"]),
                "variant_count": report["variant_count"],
                "embedding_models": report["embedding_models"],
                "report": report["artifact_paths"]["report"],
                "bridge_like_reports": report["bridge_like_reports"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
