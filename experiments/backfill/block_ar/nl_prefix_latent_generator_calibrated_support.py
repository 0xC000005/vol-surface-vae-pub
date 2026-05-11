#!/usr/bin/env python
"""Generator-calibrated support reranking for narrative bridge reports.

The support-quality decomposition showed that better historical replay support
does not necessarily imply better frozen-generator rollout support. This script
keeps calibration separate from evaluation:

1. Build a self-calibration bridge report over train support windows.
2. Run the existing scenario evaluator on that calibration report.
3. Rerank a candidate test bridge report using the train-window generator
   calibration scores.

No OpenAI calls are made here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (
    summarize_bridge_metrics,
)


DEFAULT_SOURCE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_CANDIDATE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_memory_blend_pareto_871b_alpha025_artifacts/"
    "bridge_eval_report_blend_alpha250.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_generator_calibrated_support_873a"
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


def _window_ids(report: dict[str, Any]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for row in report.get("window_metadata", []):
        if not isinstance(row, dict):
            continue
        if row.get("window_index") is None:
            continue
        lookup[int(row["window_index"])] = str(row.get("window_id", ""))
    if lookup:
        return lookup
    for row in report.get("evaluation", {}).get("heldout_examples", []):
        if isinstance(row, dict) and row.get("window_index") is not None:
            lookup[int(row["window_index"])] = str(row.get("window_id", ""))
    return lookup


def train_indices_from_report(report: dict[str, Any]) -> list[int]:
    split = report.get("split", {})
    if isinstance(split, dict) and isinstance(split.get("train_indices"), list):
        return [int(value) for value in split["train_indices"]]
    rows = report.get("window_metadata", [])
    train_indices = [
        int(idx)
        for idx, row in enumerate(rows)
        if isinstance(row, dict) and str(row.get("manifest_split", "")) == "train"
    ]
    if train_indices:
        return train_indices
    raise ValueError("source bridge report does not contain train indices")


def build_self_calibration_bridge_report(
    source_bridge_report: dict[str, Any],
    *,
    max_windows: int = 0,
) -> dict[str, Any]:
    """Build a report whose top support for each train query is itself."""

    train_indices = train_indices_from_report(source_bridge_report)
    if int(max_windows) > 0:
        train_indices = train_indices[: int(max_windows)]
    if not train_indices:
        raise ValueError("no train windows selected for self calibration")
    window_ids = _window_ids(source_bridge_report)
    heldout_examples: list[dict[str, Any]] = []
    for row_no, window_index in enumerate(train_indices):
        window_id = window_ids.get(int(window_index), f"window_{window_index:04d}")
        heldout_examples.append(
            {
                "window_index": int(window_index),
                "window_id": window_id,
                "embedding_index": int(row_no),
                "role": "anchor",
                "kind": "generator_self_calibration",
                "target_cosine": 1.0,
                "target_mse": 0.0,
                "true_rank_full_pool": 1,
                "true_rank_test_pool": 1,
                "top_full_pool": [
                    {
                        "window_index": int(window_index),
                        "window_id": window_id,
                        "cosine": 1.0,
                    }
                ],
                "top_test_pool": [
                    {
                        "window_index": int(window_index),
                        "window_id": window_id,
                        "cosine": 1.0,
                    }
                ],
                "top_train_pool": [
                    {
                        "window_index": int(window_index),
                        "window_id": window_id,
                        "cosine": 1.0,
                    }
                ],
            }
        )
    evaluation = {
        "heldout_window_count": len(train_indices),
        "heldout_example_count": len(heldout_examples),
        "heldout_examples": heldout_examples,
        "hard_negative_separation": {
            "window_count": 0,
            "mean_hard_margin": None,
            "mean_negative_gap": None,
            "windows": [],
        },
    }
    return {
        "status": "ok",
        "scope_note": (
            "Self-calibration bridge report over train support windows. Each "
            "query retrieves itself so the scenario evaluator can estimate "
            "frozen-generator support quality without using held-out test futures."
        ),
        "source_bridge_report": source_bridge_report.get("artifact_paths", {}).get(
            "report",
            "",
        ),
        "window_metadata": source_bridge_report.get("window_metadata", []),
        "source_indices": source_bridge_report.get("source_indices", []),
        "window_indices": source_bridge_report.get("window_indices", []),
        "split": {
            "train_indices": train_indices,
            "test_indices": train_indices,
            "excluded_indices": [],
            "source": "train_self_calibration",
        },
        "summary": summarize_bridge_metrics(evaluation),
        "evaluation": evaluation,
    }


def extract_generator_quality(
    calibration_scenario_report: dict[str, Any],
    *,
    method: str = "narrative_generator_topk",
    metric: str = "energy_score_z",
) -> dict[int, dict[str, Any]]:
    """Extract lower-is-better generator quality by support window index."""

    quality: dict[int, dict[str, Any]] = {}
    for row in calibration_scenario_report.get("window_scores", []):
        if not isinstance(row, dict):
            continue
        window_index = int(row["window_index"])
        method_row = row.get("methods", {}).get(method, {})
        if not isinstance(method_row, dict) or method_row.get(metric) is None:
            continue
        quality[window_index] = {
            "window_index": window_index,
            "window_id": str(row.get("window_id", "")),
            "method": method,
            "metric": metric,
            "quality_score": float(method_row[metric]),
            "coverage_80": method_row.get("coverage_80"),
            "ensemble_crps_z": method_row.get("ensemble_crps_z"),
            "energy_score_z": method_row.get("energy_score_z"),
        }
    if not quality:
        raise ValueError("no generator quality rows extracted")
    return quality


def rerank_pool_by_quality(
    pool: list[dict[str, Any]],
    quality_by_window: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rerank measured candidates by generator quality; preserve unknown order."""

    indexed: list[tuple[int, dict[str, Any]]] = [
        (position, dict(item)) for position, item in enumerate(pool)
    ]

    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        position, row = item
        window_index = int(row["window_index"])
        quality = quality_by_window.get(window_index)
        if quality is None:
            return (1, 0.0, position)
        return (0, float(quality["quality_score"]), position)

    reranked: list[dict[str, Any]] = []
    for position, row in sorted(indexed, key=key):
        window_index = int(row["window_index"])
        quality = quality_by_window.get(window_index)
        updated = dict(row)
        updated["original_support_rank"] = int(position + 1)
        if quality is not None:
            updated["generator_calibration_score"] = float(quality["quality_score"])
            updated["generator_calibration_metric"] = str(quality["metric"])
        else:
            updated["generator_calibration_score"] = None
            updated["generator_calibration_metric"] = ""
        reranked.append(updated)
    return reranked


def rerank_bridge_report_by_quality(
    candidate_bridge_report: dict[str, Any],
    quality_by_window: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Return a bridge report with top_train_pool reranked by support quality."""

    output = json.loads(json.dumps(candidate_bridge_report))
    rows = output.get("evaluation", {}).get("heldout_examples", [])
    reranked_count = 0
    measured_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        pool = row.get("top_train_pool", [])
        if not isinstance(pool, list) or not pool:
            continue
        measured = [
            item
            for item in pool
            if isinstance(item, dict) and int(item["window_index"]) in quality_by_window
        ]
        if measured:
            measured_count += len(measured)
            reranked_count += 1
        row["top_train_pool"] = rerank_pool_by_quality(pool, quality_by_window)
    output["summary"] = summarize_bridge_metrics(output["evaluation"])
    output["support_policy"] = {
        "name": "generator_calibrated_support_rerank",
        "quality_source": "train_self_calibration_scenario_report",
        "candidate_rows_reranked": reranked_count,
        "measured_candidate_occurrences": measured_count,
        "method": (
            "Within each precomputed top_train_pool, candidates with train "
            "self-calibration generator quality are sorted by lower energy "
            "score; unmeasured candidates keep original order after measured candidates."
        ),
    }
    return output


def run_build_calibration(args: argparse.Namespace) -> dict[str, Any]:
    source = _load_json(args.source_bridge_report)
    report = build_self_calibration_bridge_report(
        source,
        max_windows=int(args.max_calibration_windows),
    )
    output_dir = Path(args.output_dir)
    path = output_dir / "generator_self_calibration_bridge_report.json"
    report["artifact_paths"] = {"report": str(path)}
    _write_json(path, report)
    return report


def run_rerank(args: argparse.Namespace) -> dict[str, Any]:
    candidate = _load_json(args.candidate_bridge_report)
    quality = extract_generator_quality(
        _load_json(args.calibration_scenario_report),
        method=str(args.method),
        metric=str(args.metric),
    )
    report = rerank_bridge_report_by_quality(candidate, quality)
    output_dir = Path(args.output_dir)
    quality_path = output_dir / "generator_calibration_quality.json"
    report_path = output_dir / "generator_calibrated_bridge_report.json"
    report["artifact_paths"] = {
        "report": str(report_path),
        "quality": str(quality_path),
    }
    _write_json(quality_path, {"quality_by_window": quality})
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    build = subparsers.add_parser("build-calibration-report")
    build.add_argument("--source-bridge-report", default=DEFAULT_SOURCE_BRIDGE_REPORT)
    build.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--max-calibration-windows", type=int, default=0)

    rerank = subparsers.add_parser("rerank")
    rerank.add_argument(
        "--candidate-bridge-report", default=DEFAULT_CANDIDATE_BRIDGE_REPORT
    )
    rerank.add_argument("--calibration-scenario-report", required=True)
    rerank.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    rerank.add_argument("--method", default="narrative_generator_topk")
    rerank.add_argument("--metric", default="energy_score_z")

    args = parser.parse_args()
    if args.mode == "build-calibration-report":
        report = run_build_calibration(args)
        payload = {
            "status": report["status"],
            "report": report["artifact_paths"]["report"],
            "heldout_window_count": report["evaluation"]["heldout_window_count"],
        }
    else:
        report = run_rerank(args)
        payload = {
            "status": report["status"],
            "report": report["artifact_paths"]["report"],
            "support_policy": report["support_policy"],
            "summary": report["summary"],
        }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
