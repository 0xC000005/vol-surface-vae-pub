#!/usr/bin/env python
"""Build and summarize rollout-response labels for support candidates.

The learned support reranker showed that replay and self-calibration proxies can
look good while actual frozen-generator rollout regresses. This TestFlight
creates candidate-specific duplicate query rows so the existing scenario
evaluator can score multiple support candidates for the same narrative query.

No OpenAI calls are made here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    _annotate_window_ids,
    _top_rows,
    _window_id_lookup,
    build_bridge_examples,
    load_pipeline_artifacts,
)
from experiments.backfill.block_ar.nl_text_conditioning import (  # noqa: E402
    cosine_similarity,
)


DEFAULT_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_label_testflight_881a"
)
QUERY_SPLIT_CHOICES = ("train", "test", "excluded", "all")


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _anchor_rows(
    bridge_report: dict[str, Any], *, max_query_windows: int
) -> list[dict[str, Any]]:
    rows = bridge_report.get("evaluation", {}).get("heldout_examples", [])
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or str(row.get("role", "")) != "anchor":
            continue
        window_index = int(row["window_index"])
        if window_index in seen:
            continue
        seen.add(window_index)
        selected.append(row)
        if int(max_query_windows) > 0 and len(selected) >= int(max_query_windows):
            break
    if not selected:
        raise ValueError("no anchor rows selected")
    return selected


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _anchor_by_window(examples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    anchors: dict[int, dict[str, Any]] = {}
    for example in examples:
        if str(example.get("role", "")) != "anchor":
            continue
        window_index = int(example["window_index"])
        anchors.setdefault(window_index, example)
    return anchors


def build_query_bridge_from_examples(
    *,
    examples: list[dict[str, Any]],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    query_indices: list[int],
    support_indices: list[int],
    candidate_pool_size: int = 5,
    exclude_query_from_support: bool = True,
) -> dict[str, Any]:
    """Build query rows from cached condition vectors for arbitrary windows."""

    cond = np.asarray(condition_vectors, dtype=np.float32)
    targets = np.asarray(memory_targets, dtype=np.float32)
    anchors = _anchor_by_window(examples)
    window_ids = _window_id_lookup(examples)
    support_pool_all = [int(idx) for idx in support_indices]
    rows: list[dict[str, Any]] = []
    for window_index in [int(idx) for idx in query_indices]:
        example = anchors.get(window_index)
        if example is None:
            continue
        emb_idx = int(example["embedding_index"])
        support_pool = [
            int(idx)
            for idx in support_pool_all
            if not bool(exclude_query_from_support) or int(idx) != window_index
        ]
        if not support_pool:
            continue
        query = cond[emb_idx]
        target = targets[window_index]
        row = {
            "window_index": int(window_index),
            "window_id": str(example.get("window_id", f"window_{window_index:04d}")),
            "embedding_index": emb_idx,
            "role": str(example.get("role", "")),
            "kind": str(example.get("kind", "")),
            "target_cosine": float(cosine_similarity(query, target)),
            "target_mse": float(np.mean((query - target) ** 2)),
            "top_train_pool": _annotate_window_ids(
                _top_rows(query, targets, support_pool, top_k=int(candidate_pool_size)),
                window_ids,
            ),
        }
        rows.append(row)
    if not rows:
        raise ValueError("no query rows built from cached examples")
    return {
        "status": "ok",
        "purpose": "cached_query_bridge",
        "scope_note": (
            "Query bridge built from cached condition vectors. It can be used "
            "to create non-heldout rollout-response labels without OpenAI calls."
        ),
        "split": {
            "train_indices": support_pool_all,
            "test_indices": [int(idx) for idx in query_indices],
            "source": "cached_query_bridge",
        },
        "evaluation": {
            "heldout_window_count": len({row["window_index"] for row in rows}),
            "heldout_example_count": len(rows),
            "heldout_examples": rows,
        },
    }


def _split_indices(
    bridge_report: dict[str, Any],
    *,
    split_name: str,
    n_windows: int,
) -> list[int]:
    name = str(split_name)
    if name == "all":
        return list(range(int(n_windows)))
    key = f"{name}_indices"
    raw = bridge_report.get("split", {}).get(key)
    if not isinstance(raw, list):
        raise ValueError(f"bridge report split has no {key}")
    return [int(idx) for idx in raw]


def _first_n(indices: list[int], *, max_count: int) -> list[int]:
    values = [int(idx) for idx in indices]
    if int(max_count) <= 0:
        return values
    return values[: int(max_count)]


def _parse_index_list(raw: str | None) -> list[int] | None:
    if raw is None or not str(raw).strip():
        return None
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def build_cached_query_bridge_from_bridge_artifacts(
    *,
    bridge_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    pipeline_report: dict[str, Any],
    query_split: str = "train",
    support_split: str = "train",
    query_indices: list[int] | None = None,
    support_indices: list[int] | None = None,
    max_query_windows: int = 8,
    candidate_pool_size: int = 5,
    exclude_query_from_support: bool = True,
) -> dict[str, Any]:
    """Build arbitrary query rows from cached bridge/pipeline artifacts."""

    condition_vectors = np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32)
    memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    examples = build_bridge_examples(pipeline_report)
    if condition_vectors.shape[0] != len(examples):
        raise ValueError(
            f"condition_vectors rows ({condition_vectors.shape[0]}) do not match "
            f"rebuilt examples ({len(examples)})"
        )
    n_windows = int(memory_targets.shape[0])
    selected_query = (
        [int(idx) for idx in query_indices]
        if query_indices is not None
        else _split_indices(bridge_report, split_name=query_split, n_windows=n_windows)
    )
    selected_support = (
        [int(idx) for idx in support_indices]
        if support_indices is not None
        else _split_indices(bridge_report, split_name=support_split, n_windows=n_windows)
    )
    output = build_query_bridge_from_examples(
        examples=examples,
        condition_vectors=condition_vectors,
        memory_targets=memory_targets,
        query_indices=_first_n(selected_query, max_count=int(max_query_windows)),
        support_indices=selected_support,
        candidate_pool_size=int(candidate_pool_size),
        exclude_query_from_support=bool(exclude_query_from_support),
    )
    output["query_bridge_config"] = {
        "query_split": str(query_split),
        "support_split": str(support_split),
        "query_indices": _first_n(selected_query, max_count=int(max_query_windows)),
        "support_indices": selected_support,
        "max_query_windows": int(max_query_windows),
        "candidate_pool_size": int(candidate_pool_size),
        "exclude_query_from_support": bool(exclude_query_from_support),
    }
    for key in (
        "window_metadata",
        "source_indices",
        "window_indices",
        "embedding_backend",
        "embedding_model",
    ):
        if key in bridge_report:
            output[key] = bridge_report[key]
        elif key in pipeline_report:
            output[key] = pipeline_report[key]
    return output


def build_candidate_label_bridge(
    bridge_report: dict[str, Any],
    *,
    max_query_windows: int = 4,
    candidate_pool_size: int = 3,
) -> dict[str, Any]:
    """Return a bridge report with duplicate rows, one support candidate each."""

    selected = _anchor_rows(bridge_report, max_query_windows=int(max_query_windows))
    candidate_rows: list[dict[str, Any]] = []
    for query in selected:
        pool = query.get("top_train_pool", [])
        if not isinstance(pool, list) or not pool:
            continue
        query_id = str(query.get("window_id", f"window_{query['window_index']}"))
        for rank, candidate in enumerate(pool[: int(candidate_pool_size)], start=1):
            if not isinstance(candidate, dict):
                continue
            support_id = str(
                candidate.get("window_id", f"support_{candidate['window_index']}")
            )
            row = json.loads(json.dumps(query))
            row["kind"] = "rollout_response_candidate_label"
            row["query_id"] = (
                f"{_safe_id(query_id)}__candidate_{rank:03d}__{_safe_id(support_id)}"
            )
            row["candidate_support_rank"] = int(rank)
            row["candidate_support_window_index"] = int(candidate["window_index"])
            row["candidate_support_window_id"] = support_id
            row["top_train_pool"] = [candidate]
            candidate_rows.append(row)
    if not candidate_rows:
        raise ValueError("no candidate-specific rows built")

    output = json.loads(json.dumps(bridge_report))
    output["purpose"] = "rollout_response_candidate_labels"
    output["scope_note"] = (
        "Candidate-specific duplicate query bridge. Use the scenario evaluator "
        "with --allow-duplicate-query-windows and --top-k 1."
    )
    output["candidate_label_config"] = {
        "max_query_windows": int(max_query_windows),
        "candidate_pool_size": int(candidate_pool_size),
        "candidate_row_count": len(candidate_rows),
    }
    output["evaluation"]["heldout_examples"] = candidate_rows
    return output


def build_mixture_label_bridge(
    bridge_report: dict[str, Any],
    *,
    max_query_windows: int = 4,
    candidate_pool_size: int = 5,
    mixture_size: int = 3,
    max_mixtures_per_query: int = 0,
) -> dict[str, Any]:
    """Return duplicate rows for candidate support subsets."""

    selected = _anchor_rows(bridge_report, max_query_windows=int(max_query_windows))
    mixture_rows: list[dict[str, Any]] = []
    for query in selected:
        pool = query.get("top_train_pool", [])
        if not isinstance(pool, list) or len(pool) < int(mixture_size):
            continue
        query_id = str(query.get("window_id", f"window_{query['window_index']}"))
        pool_slice = [
            item for item in pool[: int(candidate_pool_size)] if isinstance(item, dict)
        ]
        combos = list(itertools.combinations(range(len(pool_slice)), int(mixture_size)))
        if int(max_mixtures_per_query) > 0:
            combos = combos[: int(max_mixtures_per_query)]
        for rank, combo in enumerate(combos, start=1):
            support_items = [json.loads(json.dumps(pool_slice[pos])) for pos in combo]
            support_ids = [
                str(item.get("window_id", f"support_{item['window_index']}"))
                for item in support_items
            ]
            row = json.loads(json.dumps(query))
            row["kind"] = "rollout_response_mixture_label"
            row["query_id"] = (
                f"{_safe_id(query_id)}__mixture_{rank:03d}__"
                f"{_safe_id('-'.join(support_ids))}"
            )
            row["candidate_mixture_rank"] = int(rank)
            row["candidate_mixture_positions"] = [int(pos + 1) for pos in combo]
            row["candidate_support_window_indices"] = [
                int(item["window_index"]) for item in support_items
            ]
            row["candidate_support_window_ids"] = support_ids
            row["top_train_pool"] = support_items
            mixture_rows.append(row)
    if not mixture_rows:
        raise ValueError("no mixture-specific rows built")

    output = json.loads(json.dumps(bridge_report))
    output["purpose"] = "rollout_response_mixture_labels"
    output["scope_note"] = (
        "Mixture-specific duplicate query bridge. Use the scenario evaluator "
        "with --allow-duplicate-query-windows and --top-k equal to mixture_size."
    )
    output["candidate_label_config"] = {
        "max_query_windows": int(max_query_windows),
        "candidate_pool_size": int(candidate_pool_size),
        "mixture_size": int(mixture_size),
        "max_mixtures_per_query": int(max_mixtures_per_query),
        "candidate_row_count": len(mixture_rows),
    }
    output["evaluation"]["heldout_examples"] = mixture_rows
    return output


def _metric(row: dict[str, Any], method: str, metric: str) -> float | None:
    raw = row.get("methods", {}).get(method, {}).get(metric)
    if raw is None:
        return None
    value = float(raw)
    return value if np.isfinite(value) else None


def _candidate_lookup(candidate_bridge: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = candidate_bridge.get("evaluation", {}).get("heldout_examples", [])
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("query_id"):
            lookup[str(row["query_id"])] = row
    if not lookup:
        raise ValueError("candidate bridge has no query_id rows")
    return lookup


def _support_payload(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("candidate_support_window_indices") is not None:
        return {
            "candidate_rank": int(source.get("candidate_mixture_rank", 0)),
            "support_window_indices": [
                int(value) for value in source["candidate_support_window_indices"]
            ],
            "support_window_ids": [
                str(value) for value in source.get("candidate_support_window_ids", [])
            ],
            "support_positions": [
                int(value) for value in source.get("candidate_mixture_positions", [])
            ],
            "support_cosines": [
                float(item.get("cosine", 0.0))
                for item in source.get("top_train_pool", [])
                if isinstance(item, dict)
            ],
        }
    return {
        "candidate_rank": int(source["candidate_support_rank"]),
        "support_window_indices": [int(source["candidate_support_window_index"])],
        "support_window_ids": [str(source.get("candidate_support_window_id", ""))],
        "support_positions": [int(source["candidate_support_rank"])],
        "support_cosines": [
            (
                float(source["top_train_pool"][0].get("cosine", 0.0))
                if source.get("top_train_pool")
                else 0.0
            )
        ],
    }


def summarize_rollout_response_labels(
    candidate_bridge: dict[str, Any],
    scenario_report: dict[str, Any],
    baseline_scenario_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize candidate-specific generator labels from scenario evaluation."""

    lookup = _candidate_lookup(candidate_bridge)
    rows: list[dict[str, Any]] = []
    for score in scenario_report.get("window_scores", []):
        if not isinstance(score, dict) or not score.get("query_id"):
            continue
        query_id = str(score["query_id"])
        source = lookup.get(query_id)
        if source is None:
            continue
        support = _support_payload(source)
        rows.append(
            {
                "query_id": query_id,
                "window_index": int(source["window_index"]),
                "window_id": str(source.get("window_id", "")),
                "candidate_rank": int(support["candidate_rank"]),
                "support_window_indices": support["support_window_indices"],
                "support_window_ids": support["support_window_ids"],
                "support_positions": support["support_positions"],
                "support_cosines": support["support_cosines"],
                "support_cosine": float(np.mean(support["support_cosines"])),
                "generator_energy_score_z": _metric(
                    score,
                    "narrative_generator_topk",
                    "energy_score_z",
                ),
                "generator_ensemble_crps_z": _metric(
                    score,
                    "narrative_generator_topk",
                    "ensemble_crps_z",
                ),
                "generator_coverage_80": _metric(
                    score,
                    "narrative_generator_topk",
                    "coverage_80",
                ),
                "replay_energy_score_z": _metric(
                    score,
                    "historical_replay_topk",
                    "energy_score_z",
                ),
                "replay_ensemble_crps_z": _metric(
                    score,
                    "historical_replay_topk",
                    "ensemble_crps_z",
                ),
            }
        )
    groups: list[dict[str, Any]] = []
    for window_index in sorted({row["window_index"] for row in rows}):
        group_rows = [row for row in rows if row["window_index"] == window_index]
        valid = [
            row for row in group_rows if row.get("generator_energy_score_z") is not None
        ]
        if not valid:
            continue
        best = min(valid, key=lambda row: float(row["generator_energy_score_z"]))
        top1 = min(group_rows, key=lambda row: int(row["candidate_rank"]))
        groups.append(
            {
                "window_index": int(window_index),
                "window_id": str(group_rows[0].get("window_id", "")),
                "candidate_count": len(group_rows),
                "top1_support_window_indices": top1["support_window_indices"],
                "top1_support_positions": top1["support_positions"],
                "top1_generator_energy_score_z": top1.get("generator_energy_score_z"),
                "top1_generator_ensemble_crps_z": top1.get("generator_ensemble_crps_z"),
                "best_generator_support_window_indices": best["support_window_indices"],
                "best_generator_support_positions": best["support_positions"],
                "best_generator_support_rank": int(best["candidate_rank"]),
                "best_generator_energy_score_z": best.get("generator_energy_score_z"),
                "best_generator_ensemble_crps_z": best.get("generator_ensemble_crps_z"),
                "best_minus_top1_energy_score_z": (
                    None
                    if top1.get("generator_energy_score_z") is None
                    else float(best["generator_energy_score_z"])
                    - float(top1["generator_energy_score_z"])
                ),
                "best_minus_top1_ensemble_crps_z": (
                    None
                    if top1.get("generator_ensemble_crps_z") is None
                    else float(best["generator_ensemble_crps_z"])
                    - float(top1["generator_ensemble_crps_z"])
                ),
            }
        )
    best_not_top1 = sum(
        1 for group in groups if int(group["best_generator_support_rank"]) != 1
    )
    energy_gain_values = [
        float(group["best_minus_top1_energy_score_z"])
        for group in groups
        if group.get("best_minus_top1_energy_score_z") is not None
    ]
    crps_gain_values = [
        float(group["best_minus_top1_ensemble_crps_z"])
        for group in groups
        if group.get("best_minus_top1_ensemble_crps_z") is not None
    ]
    best_energy_values = [
        float(group["best_generator_energy_score_z"])
        for group in groups
        if group.get("best_generator_energy_score_z") is not None
    ]
    top1_energy_values = [
        float(group["top1_generator_energy_score_z"])
        for group in groups
        if group.get("top1_generator_energy_score_z") is not None
    ]
    best_crps_values = [
        float(group["best_generator_ensemble_crps_z"])
        for group in groups
        if group.get("best_generator_ensemble_crps_z") is not None
    ]
    top1_crps_values = [
        float(group["top1_generator_ensemble_crps_z"])
        for group in groups
        if group.get("top1_generator_ensemble_crps_z") is not None
    ]
    summary = {
        "query_count": len(groups),
        "candidate_row_count": len(rows),
        "best_generator_not_top1_count": int(best_not_top1),
        "best_generator_not_top1_fraction": (
            None if not groups else float(best_not_top1 / len(groups))
        ),
        "top1_generator_energy_score_z_mean": (
            None if not top1_energy_values else float(np.mean(top1_energy_values))
        ),
        "best_generator_energy_score_z_mean": (
            None if not best_energy_values else float(np.mean(best_energy_values))
        ),
        "mean_best_minus_top1_energy_score_z": (
            None if not energy_gain_values else float(np.mean(energy_gain_values))
        ),
        "top1_generator_ensemble_crps_z_mean": (
            None if not top1_crps_values else float(np.mean(top1_crps_values))
        ),
        "best_generator_ensemble_crps_z_mean": (
            None if not best_crps_values else float(np.mean(best_crps_values))
        ),
        "mean_best_minus_top1_ensemble_crps_z": (
            None if not crps_gain_values else float(np.mean(crps_gain_values))
        ),
    }
    if baseline_scenario_report is not None:
        baseline = baseline_scenario_report.get("summary", {}).get(
            "narrative_generator_topk", {}
        )
        baseline_energy = baseline.get("energy_score_z_mean")
        baseline_crps = baseline.get("ensemble_crps_z_mean")
        summary["baseline_topk_energy_score_z_mean"] = baseline_energy
        summary["baseline_topk_ensemble_crps_z_mean"] = baseline_crps
        if (
            baseline_energy is not None
            and summary["best_generator_energy_score_z_mean"] is not None
        ):
            energy_delta = float(
                summary["best_generator_energy_score_z_mean"] - float(baseline_energy)
            )
            summary["best_candidate_minus_baseline_topk_energy_score_z"] = energy_delta
            summary["best_single_minus_baseline_topk_energy_score_z"] = energy_delta
        if (
            baseline_crps is not None
            and summary["best_generator_ensemble_crps_z_mean"] is not None
        ):
            crps_delta = float(
                summary["best_generator_ensemble_crps_z_mean"] - float(baseline_crps)
            )
            summary["best_candidate_minus_baseline_topk_ensemble_crps_z"] = crps_delta
            summary["best_single_minus_baseline_topk_ensemble_crps_z"] = crps_delta
    return {
        "status": "ok",
        "scope_note": (
            "Candidate-specific rollout-response labels. Lower generator energy "
            "and CRPS are better."
        ),
        "summary": summary,
        "groups": groups,
        "rows": rows,
    }


def build_command(args: argparse.Namespace) -> None:
    bridge = _load_json(args.bridge_report)
    output = build_candidate_label_bridge(
        bridge,
        max_query_windows=int(args.max_query_windows),
        candidate_pool_size=int(args.candidate_pool_size),
    )
    output_dir = Path(args.output_dir)
    report_path = output_dir / "candidate_label_bridge_report.json"
    output["artifact_paths"] = {"report": str(report_path)}
    _write_json(report_path, output)
    print(f"wrote {report_path} rows={len(output['evaluation']['heldout_examples'])}")


def build_mixture_command(args: argparse.Namespace) -> None:
    bridge = _load_json(args.bridge_report)
    output = build_mixture_label_bridge(
        bridge,
        max_query_windows=int(args.max_query_windows),
        candidate_pool_size=int(args.candidate_pool_size),
        mixture_size=int(args.mixture_size),
        max_mixtures_per_query=int(args.max_mixtures_per_query),
    )
    output_dir = Path(args.output_dir)
    report_path = output_dir / "mixture_label_bridge_report.json"
    output["artifact_paths"] = {"report": str(report_path)}
    _write_json(report_path, output)
    print(f"wrote {report_path} rows={len(output['evaluation']['heldout_examples'])}")


def _bridge_arrays_path(
    bridge_report: dict[str, Any],
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    artifact_paths = bridge_report.get("artifact_paths", {})
    if isinstance(artifact_paths, dict) and artifact_paths.get("arrays"):
        return Path(str(artifact_paths["arrays"]))
    raise ValueError("pass --bridge-arrays or use a bridge report with artifact_paths.arrays")


def _pipeline_report_path(
    bridge_report: dict[str, Any],
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if bridge_report.get("input_report"):
        return Path(str(bridge_report["input_report"]))
    raise ValueError("pass --pipeline-report or use a bridge report with input_report")


def build_query_command(args: argparse.Namespace) -> None:
    bridge = _load_json(args.bridge_report)
    pipeline = _load_json(_pipeline_report_path(bridge, args.pipeline_report))
    _, bridge_arrays = load_pipeline_artifacts(
        args.bridge_report,
        _bridge_arrays_path(bridge, args.bridge_arrays),
    )
    output = build_cached_query_bridge_from_bridge_artifacts(
        bridge_report=bridge,
        bridge_arrays=bridge_arrays,
        pipeline_report=pipeline,
        query_split=str(args.query_split),
        support_split=str(args.support_split),
        query_indices=_parse_index_list(args.query_indices),
        support_indices=_parse_index_list(args.support_indices),
        max_query_windows=int(args.max_query_windows),
        candidate_pool_size=int(args.candidate_pool_size),
        exclude_query_from_support=not bool(args.include_self_support),
    )
    output_dir = Path(args.output_dir)
    report_path = output_dir / "query_bridge_report.json"
    output["artifact_paths"] = {"report": str(report_path)}
    _write_json(report_path, output)
    print(f"wrote {report_path} rows={len(output['evaluation']['heldout_examples'])}")


def summarize_command(args: argparse.Namespace) -> None:
    candidate_bridge = _load_json(args.candidate_bridge_report)
    scenario_report = _load_json(args.scenario_report)
    baseline_report = (
        _load_json(args.baseline_scenario_report)
        if args.baseline_scenario_report
        else None
    )
    summary = summarize_rollout_response_labels(
        candidate_bridge,
        scenario_report,
        baseline_scenario_report=baseline_report,
    )
    output_dir = Path(args.output_dir)
    report_path = output_dir / "rollout_response_label_summary.json"
    summary["artifact_paths"] = {
        "candidate_bridge_report": str(args.candidate_bridge_report),
        "scenario_report": str(args.scenario_report),
        "baseline_scenario_report": str(args.baseline_scenario_report),
        "report": str(report_path),
    }
    _write_json(report_path, summary)
    print(
        f"wrote {report_path} "
        f"queries={summary['summary']['query_count']} "
        f"best_not_top1={summary['summary']['best_generator_not_top1_count']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-bridge")
    build.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    build.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--max-query-windows", type=int, default=4)
    build.add_argument("--candidate-pool-size", type=int, default=3)
    build.set_defaults(func=build_command)

    build_mixture = subparsers.add_parser("build-mixture-bridge")
    build_mixture.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    build_mixture.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    build_mixture.add_argument("--max-query-windows", type=int, default=4)
    build_mixture.add_argument("--candidate-pool-size", type=int, default=5)
    build_mixture.add_argument("--mixture-size", type=int, default=3)
    build_mixture.add_argument("--max-mixtures-per-query", type=int, default=0)
    build_mixture.set_defaults(func=build_mixture_command)

    build_query = subparsers.add_parser("build-query-bridge")
    build_query.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    build_query.add_argument("--bridge-arrays")
    build_query.add_argument("--pipeline-report")
    build_query.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    build_query.add_argument("--query-split", choices=QUERY_SPLIT_CHOICES, default="train")
    build_query.add_argument("--support-split", choices=QUERY_SPLIT_CHOICES, default="train")
    build_query.add_argument("--query-indices")
    build_query.add_argument("--support-indices")
    build_query.add_argument("--max-query-windows", type=int, default=8)
    build_query.add_argument("--candidate-pool-size", type=int, default=5)
    build_query.add_argument("--include-self-support", action="store_true")
    build_query.set_defaults(func=build_query_command)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--candidate-bridge-report", required=True)
    summarize.add_argument("--scenario-report", required=True)
    summarize.add_argument("--baseline-scenario-report")
    summarize.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    summarize.set_defaults(func=summarize_command)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
