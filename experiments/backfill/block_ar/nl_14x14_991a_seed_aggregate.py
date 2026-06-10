#!/usr/bin/env python
"""Aggregate the 991a 3-seed stride-5 14+14 retraining and adjudicate the
pre-registered fit gate.

Reproducible diagnostic for the 991a iteration (plan:
docs/research_protocols/nl_prefix_latent_991a_train_fit_minimal_fix_plan.md).

Adds the two controls the gate needs that the per-seed reports do not carry:
1. raw-embedding Tier-B / Tier-A recall controls for the text-space arm;
2. bridge held-out rank vs distance-to-nearest-train-window (window-identity
   generalization mechanism check).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_14x14_manifest_retrieval_training import (  # noqa: E402
    _safe_normalize_rows,
    _same_label_recall_at_k,
    _write_json,
)

SEED_DIR_TEMPLATE = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_retrieval_training_openai_holdout_991a_seed{seed}"
)
OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "stride5_14x14_retrieval_training_openai_holdout_991a_aggregate"
)
SEEDS = (0, 1, 2)

# Pre-registered fit gate (plan section 3).
BRIDGE_RANK_GATE = 400.0
BRIDGE_RECALL_GATE = 0.10
TEXT_RECALL_RATIO_GATE = 2.0
TEXT_TEMPORAL_REDUCTION_GATE = 0.30
SEEDS_REQUIRED = 2


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _median_min_max(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _rank_of_true_memory(
    condition_vectors: np.ndarray, memory_targets: np.ndarray, labels: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    pred = _safe_normalize_rows(condition_vectors)
    memory = _safe_normalize_rows(memory_targets)
    ranks = []
    for i in idx:
        scores = memory @ pred[int(i)]
        true_score = float(scores[int(labels[int(i)])])
        ranks.append(int(np.sum(scores > true_score) + 1))
    return np.asarray(ranks, dtype=np.int64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(values.shape[0], dtype=np.float64)
        return ranks

    rx, ry = rank(np.asarray(x, dtype=np.float64)), rank(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.sum(rx**2) * np.sum(ry**2)))
    return float(np.sum(rx * ry) / denom) if denom > 0 else float("nan")


def main() -> int:
    per_seed: dict[str, Any] = {}
    collected: dict[str, dict[str, list[float]]] = {"text_space": {}, "projected_memory": {}}
    gate_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        seed_dir = _resolve(SEED_DIR_TEMPLATE.format(seed=seed))
        run_report = json.loads((seed_dir / "training_run_report.json").read_text())
        seed_entry: dict[str, Any] = {}

        # --- text-space: raw controls from arrays ---
        with np.load(seed_dir / "text_space" / "text_space_training_arrays.npz") as arrays:
            raw = arrays["embeddings"]
            labels = arrays["labels"]
            train_idx = arrays["train_example_idx"]
            val_idx = arrays["val_example_idx"]
            tierb_idx = arrays["tierb_query_idx"]
        all_idx = np.arange(raw.shape[0], dtype=np.int64)
        raw_tierb_recall10 = _same_label_recall_at_k(
            raw, labels, tierb_idx, train_idx, k=10
        )
        raw_tiera_recall10 = _same_label_recall_at_k(raw, labels, val_idx, all_idx, k=10)

        text_eval = run_report["text_space"]["evaluation"]
        text_train = run_report["text_space"]["training"]
        adapted_tierb = float(text_eval["heldout_view_same_label_recall_at_10"])
        dist_raw = float(text_eval["heldout_median_top10_temporal_distance_raw"])
        dist_adapted = float(text_eval["heldout_median_top10_temporal_distance_adapted"])
        temporal_reduction = (dist_raw - dist_adapted) / dist_raw if dist_raw else float("nan")
        recall_ratio = adapted_tierb / raw_tierb_recall10 if raw_tierb_recall10 else float("inf")

        # --- bridge: rank vs distance-to-nearest-train-window ---
        with np.load(
            seed_dir / "projected_memory" / "projected_memory_training_arrays.npz"
        ) as arrays:
            condition_vectors = arrays["condition_vectors"]
            memory_targets = arrays["memory_targets"]
            bridge_labels = arrays["labels"]
            bridge_train_idx = arrays["train_example_idx"]
            bridge_val_idx = arrays["val_example_idx"]
        take = min(1500, bridge_val_idx.shape[0])
        chosen = bridge_val_idx[
            np.linspace(0, bridge_val_idx.shape[0] - 1, take, dtype=np.int64)
        ]
        val_ranks = _rank_of_true_memory(
            condition_vectors, memory_targets, bridge_labels, chosen
        )
        train_windows = np.unique(bridge_labels[bridge_train_idx])
        val_windows = bridge_labels[chosen]
        distance_to_train = np.min(
            np.abs(val_windows[:, None] - train_windows[None, :]), axis=1
        )
        buckets = {
            "31_60": (distance_to_train >= 31) & (distance_to_train <= 60),
            "61_90": (distance_to_train >= 61) & (distance_to_train <= 90),
            "91_plus": distance_to_train >= 91,
        }
        rank_by_bucket = {
            name: (float(np.median(val_ranks[mask])) if int(mask.sum()) else float("nan"))
            for name, mask in buckets.items()
        }
        bucket_counts = {name: int(mask.sum()) for name, mask in buckets.items()}

        bridge_eval = run_report["projected_memory"]["evaluation"]
        bridge_train = run_report["projected_memory"]["training"]
        bridge_rank = float(bridge_eval["heldout_true_memory_rank_median"])
        bridge_recall = float(bridge_eval["heldout_recall_at_10_true_memory"])

        # --- gate adjudication per seed ---
        bridge_pass = bridge_rank <= BRIDGE_RANK_GATE and bridge_recall >= BRIDGE_RECALL_GATE
        text_pass = (
            recall_ratio >= TEXT_RECALL_RATIO_GATE
            and temporal_reduction >= TEXT_TEMPORAL_REDUCTION_GATE
        )
        gate_rows.append(
            {
                "seed": seed,
                "bridge_pass": bool(bridge_pass),
                "text_space_pass": bool(text_pass),
            }
        )

        seed_entry["text_space"] = {
            "early_stopped": bool(text_train["early_stopped"]),
            "stopped_step": int(text_train["stopped_step"]),
            "best_step": text_train["best_step"],
            "heldout_view_same_label_recall_at_10": adapted_tierb,
            "raw_tierb_same_label_recall_at_10_control": float(raw_tierb_recall10),
            "tierb_recall_ratio_adapted_over_raw": float(recall_ratio),
            "heldout_same_label_recall_at_10": float(
                text_eval["heldout_same_label_recall_at_10"]
            ),
            "raw_tiera_same_label_recall_at_10_control": float(raw_tiera_recall10),
            "heldout_median_top10_temporal_distance_raw": dist_raw,
            "heldout_median_top10_temporal_distance_adapted": dist_adapted,
            "temporal_distance_reduction": float(temporal_reduction),
            "in_sample_adapted_recall_at_1": float(
                text_eval["adapted_same_label_recall_at_1"]
            ),
            "in_sample_raw_recall_at_1": float(text_eval["raw_same_label_recall_at_1"]),
        }
        seed_entry["projected_memory"] = {
            "early_stopped": bool(bridge_train["early_stopped"]),
            "stopped_step": int(bridge_train["stopped_step"]),
            "best_step": bridge_train["best_step"],
            "heldout_true_memory_rank_median": bridge_rank,
            "heldout_recall_at_10_true_memory": bridge_recall,
            "heldout_target_cosine_mean": float(bridge_eval["heldout_target_cosine_mean"]),
            "in_sample_true_memory_rank_median": float(
                bridge_eval["true_memory_rank_median"]
            ),
            "in_sample_recall_at_10_true_memory": float(
                bridge_eval["recall_at_10_true_memory"]
            ),
            "heldout_view_true_memory_rank_median": float(
                bridge_eval["heldout_view_true_memory_rank_median"]
            ),
            "heldout_rank_by_distance_to_train_bucket": rank_by_bucket,
            "heldout_bucket_counts": bucket_counts,
            "heldout_rank_vs_distance_spearman": _spearman(
                distance_to_train.astype(np.float64), val_ranks.astype(np.float64)
            ),
            "heldout_source_hard_negative_margin_positive_rate": float(
                bridge_eval["heldout_source_hard_negative_margin_positive_rate"]
            ),
        }
        per_seed[f"seed_{seed}"] = seed_entry

        for method in ("text_space", "projected_memory"):
            for key, value in seed_entry[method].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    collected[method].setdefault(key, []).append(float(value))

    aggregate = {
        method: {key: _median_min_max(values) for key, values in metrics.items()}
        for method, metrics in collected.items()
    }

    bridge_passes = sum(1 for row in gate_rows if row["bridge_pass"])
    text_passes = sum(1 for row in gate_rows if row["text_space_pass"])
    fit_gate = {
        "pre_registered_thresholds": {
            "bridge_heldout_rank_median_max": BRIDGE_RANK_GATE,
            "bridge_heldout_recall_at_10_min": BRIDGE_RECALL_GATE,
            "text_tierb_recall_ratio_min": TEXT_RECALL_RATIO_GATE,
            "text_temporal_distance_reduction_min": TEXT_TEMPORAL_REDUCTION_GATE,
            "seeds_required": SEEDS_REQUIRED,
        },
        "per_seed": gate_rows,
        "bridge_seeds_passing": bridge_passes,
        "text_space_seeds_passing": text_passes,
        "bridge_gate_pass": bridge_passes >= SEEDS_REQUIRED,
        "text_space_gate_pass": text_passes >= SEEDS_REQUIRED,
        "overall_pass": bridge_passes >= SEEDS_REQUIRED and text_passes >= SEEDS_REQUIRED,
        "consequence_on_fail": (
            "train_fit falsified as primary at convergence; do NOT run 991b-991e; "
            "reclassify to objective-geometry (backend analogue) and run 992a "
            "single-knob mse_weight 1.0->0.2 / contrastive_weight 0.2->1.0"
        ),
    }

    report = {
        "schema_version": "nl_14x14_991a_seed_aggregate_v1",
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "fit_gate": fit_gate,
    }
    output = _resolve(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "seed_aggregate_report.json", report)
    print(json.dumps({"fit_gate": fit_gate, "aggregate_keys": sorted(aggregate)}, indent=2))
    print(f"written: {output / 'seed_aggregate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
