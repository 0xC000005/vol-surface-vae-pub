#!/usr/bin/env python
"""Memory-space locality oracle audit (992b).

Measures the information ceiling of exact-window retrieval in the frozen SNI
memory space (939a bank): if even the memory of a temporally adjacent window
cannot rank the true memory highly, then no text-conditioned predictor can,
and an exact-window-rank fit gate is set above the achievable ceiling.

Oracles:
  A. query = memory[w + k] (k in 5, 10, 30): rank of memory[w], recall@10/@100;
  B. top-10 memory-space neighbors of memory[w]: median temporal distance;
  C. cosine(memory[w], memory[w+5]) vs cosine of random pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_14x14_manifest_retrieval_training import (  # noqa: E402
    _write_json,
)

SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_memory_locality_oracle_992b"
)
SEED = 0
SAMPLE_WINDOWS = 600
NEIGHBOR_OFFSETS = (5, 10, 30)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> int:
    with np.load(_resolve(SUPPORT_ARRAYS)) as payload:
        memory = np.asarray(payload["memory_targets"], dtype=np.float32)
    m = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-12)
    bank_size = m.shape[0]
    rng = np.random.default_rng(SEED)
    margin = max(NEIGHBOR_OFFSETS)
    windows = rng.choice(
        np.arange(margin, bank_size - margin), size=SAMPLE_WINDOWS, replace=False
    )

    oracle_a: dict[str, dict[str, float]] = {}
    for offset in NEIGHBOR_OFFSETS:
        ranks = []
        for w in windows:
            scores = m @ m[int(w) + int(offset)]
            ranks.append(int(np.sum(scores > float(scores[int(w)])) + 1))
        ranks_arr = np.asarray(ranks)
        oracle_a[f"offset_{offset}"] = {
            "rank_median": float(np.median(ranks_arr)),
            "recall_at_10": float(np.mean(ranks_arr <= 10)),
            "recall_at_100": float(np.mean(ranks_arr <= 100)),
        }

    top10_distances = []
    for w in windows[:300]:
        scores = m @ m[int(w)]
        scores[int(w)] = -np.inf
        top = np.argpartition(scores, -10)[-10:]
        top10_distances.append(float(np.median(np.abs(top - int(w)))))

    adjacent_cos = np.asarray([float(m[int(w)] @ m[int(w) + 5]) for w in windows])
    random_pairs = rng.choice(bank_size, size=(SAMPLE_WINDOWS, 2))
    random_cos = np.asarray([float(m[int(i)] @ m[int(j)]) for i, j in random_pairs])

    report = {
        "schema_version": "nl_14x14_memory_locality_oracle_v1",
        "support_arrays": str(SUPPORT_ARRAYS),
        "bank_size": int(bank_size),
        "sample_windows": int(SAMPLE_WINDOWS),
        "seed": SEED,
        "oracle_a_adjacent_memory_query": oracle_a,
        "oracle_b_top10_memory_neighbor_median_temporal_distance_days": float(
            np.median(np.asarray(top10_distances))
        ),
        "oracle_c_cosine": {
            "adjacent_offset5_median": float(np.median(adjacent_cos)),
            "random_pair_median": float(np.median(random_cos)),
        },
        "interpretation": (
            "If oracle A at offset 5 cannot reach the fit-gate thresholds "
            "(rank<=400, recall@10>=0.10) decisively, exact-window retrieval in "
            "this memory space is information-limited for ANY predictor whose "
            "input lacks the realized path; the gate measures the space, not "
            "the bridge."
        ),
    }
    output = _resolve(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "memory_locality_oracle_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
