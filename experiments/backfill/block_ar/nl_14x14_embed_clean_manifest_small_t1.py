#!/usr/bin/env python3
"""Track A T1: embed the CLEAN 14x14 training manifest at text-embedding-3-small (1536-d).

Reuses the EXISTING embedding utilities (no new OpenAI client):
  * ``load_manifest_training_data`` -> identical text list / ordering / hash as the
    retrieval-training harness, so the produced cache is a guaranteed cache-HIT for T5.
  * ``embed_with_cache`` / ``embed_texts_with_openai`` -> the same content-addressed
    cache the harness consumes from ``<cache-dir>/embedding_cache/``.

Policy: PILOT (5 examples, validate dim=1536 + finite + ~unit norms) BEFORE SCALE.
The API key is read from .env by the underlying utility; it is NEVER printed/handled here.
Token counting is offline (tiktoken cl100k_base); cost uses the published
text-embedding-3-small rate ($0.02 / 1M tokens).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.backfill.block_ar.nl_14x14_manifest_retrieval_training import (  # noqa: E402
    load_manifest_training_data,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    embed_with_cache,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    embed_texts_with_openai,
)

EMBEDDING_MODEL = "text-embedding-3-small"
EXPECTED_DIM = 1536
USD_PER_MILLION_TOKENS = 0.02  # text-embedding-3-small published rate


def _count_tokens(texts: list[str]) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return int(sum(len(enc.encode(t)) for t in texts))
    except Exception as exc:  # pragma: no cover - offline fallback
        print(f"tiktoken unavailable ({exc!r}); using ~4-chars/token estimate", file=sys.stderr)
        return int(sum(max(1, len(t) // 4) for t in texts))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--examples-jsonl",
        default="experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "stride5_self_supervised_training_manifest_clean_20260615/training_examples.jsonl",
    )
    p.add_argument(
        "--pairs-jsonl",
        default="experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "stride5_self_supervised_training_manifest_clean_20260615/training_pairs.jsonl",
    )
    p.add_argument(
        "--cache-dir",
        default="experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "stride5_14x14_shared_embedding_cache_small_20260617",
    )
    p.add_argument("--dotenv-path", default=".env")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--mode", choices=("pilot", "scale"), required=True)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = load_manifest_training_data(
        examples_jsonl=args.examples_jsonl,
        pairs_jsonl=args.pairs_jsonl,
        max_targets=None,
    )
    texts: list[str] = data["texts"]
    print(f"[T1] loaded {len(texts)} manifest texts", file=sys.stderr)

    if args.mode == "pilot":
        sample = texts[:5]
        n_tok = _count_tokens(sample)
        print(f"[T1][PILOT] embedding {len(sample)} texts at {EMBEDDING_MODEL}", file=sys.stderr)
        arr = embed_texts_with_openai(
            sample, model=EMBEDDING_MODEL, dotenv_path=args.dotenv_path, batch_size=len(sample)
        )
        norms = np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1)
        report = {
            "mode": "pilot",
            "model": EMBEDDING_MODEL,
            "n_examples": int(arr.shape[0]),
            "embedding_dim": int(arr.shape[1]),
            "expected_dim": EXPECTED_DIM,
            "dim_ok": bool(arr.shape[1] == EXPECTED_DIM),
            "all_finite": bool(np.isfinite(arr).all()),
            "raw_norm_min": float(norms.min()),
            "raw_norm_max": float(norms.max()),
            "raw_norm_mean": float(norms.mean()),
            "pilot_tokens_5ex": int(n_tok),
            "total_manifest_texts": int(len(texts)),
        }
        out = cache_dir / "t1_pilot_report.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        ok = report["dim_ok"] and report["all_finite"]
        print(f"[T1][PILOT] {'PASS' if ok else 'FAIL'} -> {out}", file=sys.stderr)
        return 0 if ok else 1

    # SCALE
    total_tokens = _count_tokens(texts)
    est_cost = total_tokens / 1_000_000.0 * USD_PER_MILLION_TOKENS
    print(
        f"[T1][SCALE] embedding {len(texts)} texts at {EMBEDDING_MODEL}; "
        f"~{total_tokens} tokens, est ${est_cost:.4f}",
        file=sys.stderr,
    )
    t0 = time.time()
    arr, meta = embed_with_cache(
        texts,
        output_dir=cache_dir,
        backend="openai",
        model=EMBEDDING_MODEL,
        dotenv_path=args.dotenv_path,
        batch_size=int(args.batch_size),
        hash_dim=256,  # unused for openai backend
    )
    elapsed = time.time() - t0
    report = {
        "mode": "scale",
        "model": EMBEDDING_MODEL,
        "backend": "openai",
        "count": int(arr.shape[0]),
        "embedding_dim": int(arr.shape[1]),
        "expected_dim": EXPECTED_DIM,
        "dim_ok": bool(arr.shape[1] == EXPECTED_DIM),
        "all_finite": bool(np.isfinite(arr).all()),
        "total_tokens_tiktoken": int(total_tokens),
        "usd_per_million_tokens": USD_PER_MILLION_TOKENS,
        "approx_cost_usd": float(est_cost),
        "elapsed_seconds": float(elapsed),
        "cache_dir": str(cache_dir),
        "aggregate_cache_path": meta.get("cache_path"),
        "cache_hit": meta.get("cache_hit"),
        "batch_cache_hits": meta.get("batch_cache_hits"),
        "batch_cache_misses": meta.get("batch_cache_misses"),
    }
    out = cache_dir / "t1_scale_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    ok = report["dim_ok"] and report["all_finite"] and report["count"] == len(texts)
    print(f"[T1][SCALE] {'PASS' if ok else 'FAIL'} -> {out}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
