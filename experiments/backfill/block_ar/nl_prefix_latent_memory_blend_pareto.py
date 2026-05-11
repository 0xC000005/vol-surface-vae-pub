#!/usr/bin/env python
"""Pareto diagnostic for text memory versus support-mixture memory.

The residual-refinement TestFlights showed a clear trade-off:

- text memory preserves hard-negative directionality;
- support-mixture memory improves target placement but collapses directionality.

This script measures that trade-off directly with a small fixed interpolation
grid. It is not a hyperparameter search and it does not call OpenAI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
    evaluate_condition_bridge,
    summarize_bridge_metrics,
)
from experiments.backfill.block_ar.nl_prefix_latent_residual_refinement_testflight import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
    DEFAULT_ORACLE_ARRAYS,
    DEFAULT_PIPELINE_REPORT,
    build_support_mixture_memory,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_memory_blend_pareto_871a"
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


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    raw = float(value)
    return round(raw, 12) if np.isfinite(raw) else None


def parse_alpha_grid(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if value < 0.0 or value > 1.0:
            raise ValueError("blend alphas must be in [0, 1]")
        values.append(value)
    if not values:
        raise ValueError("at least one blend alpha is required")
    return values


def blend_condition_memory(
    text_memory: np.ndarray,
    support_memory: np.ndarray,
    *,
    support_alpha: float,
) -> np.ndarray:
    """Convexly blend text memory and support-mixture memory."""

    text = np.asarray(text_memory, dtype=np.float32)
    support = np.asarray(support_memory, dtype=np.float32)
    if text.shape != support.shape:
        raise ValueError("text_memory and support_memory shapes must match")
    alpha = float(support_alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("support_alpha must be in [0, 1]")
    return ((1.0 - alpha) * text + alpha * support).astype(np.float32)


def _evaluate_summary(
    *,
    examples: list[dict[str, Any]],
    condition_vectors: np.ndarray,
    memory_targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    evaluation = evaluate_condition_bridge(
        examples,
        condition_vectors,
        memory_targets,
        train_indices=[int(value) for value in train_indices],
        test_indices=[int(value) for value in test_indices],
        top_k=int(top_k),
    )
    return summarize_bridge_metrics(evaluation)


def build_pareto_report(
    *,
    examples: list[dict[str, Any]],
    text_memory: np.ndarray,
    support_memory: np.ndarray,
    memory_targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alphas: list[float],
    eval_top_k: int,
    min_target_gain: float,
    min_gap_fraction: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    text_summary = _evaluate_summary(
        examples=examples,
        condition_vectors=text_memory,
        memory_targets=memory_targets,
        train_indices=train_indices,
        test_indices=test_indices,
        top_k=eval_top_k,
    )
    text_target = float(text_summary["heldout_mean_target_cosine"])
    text_gap = float(text_summary["heldout_hard_negative_mean_gap"])
    gap_floor = float(min_gap_fraction) * text_gap
    for alpha in alphas:
        blended = blend_condition_memory(
            text_memory,
            support_memory,
            support_alpha=float(alpha),
        )
        summary = _evaluate_summary(
            examples=examples,
            condition_vectors=blended,
            memory_targets=memory_targets,
            train_indices=train_indices,
            test_indices=test_indices,
            top_k=eval_top_k,
        )
        target = float(summary["heldout_mean_target_cosine"])
        gap = float(summary["heldout_hard_negative_mean_gap"])
        rows.append(
            {
                "support_alpha": float(alpha),
                "heldout_mean_target_cosine": _round(target),
                "heldout_hard_negative_mean_gap": _round(gap),
                "heldout_hard_negative_mean_margin": summary.get(
                    "heldout_hard_negative_mean_margin"
                ),
                "heldout_recall_at_3_test_pool": summary.get(
                    "heldout_recall_at_3_test_pool"
                ),
                "target_gain_vs_text": _round(target - text_target),
                "gap_retained_fraction": _round(gap / text_gap if text_gap else 0.0),
                "passes_gate": bool(
                    target >= text_target + float(min_target_gain) and gap >= gap_floor
                ),
            }
        )
    passing = [row for row in rows if row["passes_gate"]]
    best_passing = None
    if passing:
        best_passing = max(
            passing,
            key=lambda row: (
                float(row["heldout_mean_target_cosine"]),
                float(row["heldout_hard_negative_mean_gap"]),
            ),
        )
    best_target = max(rows, key=lambda row: float(row["heldout_mean_target_cosine"]))
    best_gap = max(rows, key=lambda row: float(row["heldout_hard_negative_mean_gap"]))
    return {
        "status": "testflight_pass" if best_passing is not None else "diagnostic_only",
        "text_baseline": text_summary,
        "gap_floor": _round(gap_floor),
        "min_target_gain": float(min_target_gain),
        "min_gap_fraction": float(min_gap_fraction),
        "rows": rows,
        "best_passing": best_passing,
        "best_by_target": best_target,
        "best_by_gap": best_gap,
    }


def run_pareto(args: argparse.Namespace) -> dict[str, Any]:
    report = _load_json(args.pipeline_report)
    examples = build_bridge_examples(report)
    with np.load(args.bridge_arrays) as bridge_payload:
        bridge_arrays = {
            name: bridge_payload[name].copy() for name in bridge_payload.files
        }
    with np.load(args.oracle_arrays) as oracle_payload:
        oracle_arrays = {
            name: oracle_payload[name].copy() for name in oracle_payload.files
        }
    text_memory = np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32)
    memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    train_indices = np.asarray(bridge_arrays["train_indices"], dtype=np.int64)
    test_indices = np.asarray(bridge_arrays["test_indices"], dtype=np.int64)
    history_level = np.asarray(oracle_arrays["history_level"], dtype=np.float32)
    support_memory, mixture_details = build_support_mixture_memory(
        examples=examples,
        query_memory=text_memory,
        memory_targets=memory_targets,
        start_state=history_level[:, -1, :],
        train_indices=train_indices,
        top_k=int(args.top_k),
        temperature=float(args.temperature),
        start_distance_penalty=float(args.start_distance_penalty),
        exclude_self=bool(args.exclude_self),
    )
    alphas = parse_alpha_grid(args.alphas)
    full_alphas = sorted(set([0.0, 1.0, *alphas]))
    pareto = build_pareto_report(
        examples=examples,
        text_memory=text_memory,
        support_memory=support_memory,
        memory_targets=memory_targets,
        train_indices=train_indices,
        test_indices=test_indices,
        alphas=full_alphas,
        eval_top_k=int(args.eval_top_k),
        min_target_gain=float(args.min_target_gain),
        min_gap_fraction=float(args.min_gap_fraction),
    )
    output_dir = Path(args.output_dir)
    output = {
        "status": pareto["status"],
        "scope_note": (
            "Offline text-memory/support-mixture Pareto diagnostic. No OpenAI "
            "calls and no generator rollout."
        ),
        "config": {
            "interior_alphas": alphas,
            "evaluated_alphas": full_alphas,
            "top_k": int(args.top_k),
            "temperature": float(args.temperature),
            "start_distance_penalty": float(args.start_distance_penalty),
            "min_target_gain": float(args.min_target_gain),
            "min_gap_fraction": float(args.min_gap_fraction),
        },
        "mixture_support": mixture_details,
        "pareto": pareto,
        "decision": {
            "promote": False,
            "reason": (
                "Diagnostic pass: an explicit blend preserves enough text direction "
                "while improving target placement; scenario rollout and verifier are "
                "still required."
                if pareto["status"] == "testflight_pass"
                else "Diagnostic only: no blend cleared both target-placement and direction-retention gates."
            ),
        },
        "artifact_paths": {
            "report": str(output_dir / "memory_blend_pareto.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "memory_blend_pareto.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "report": output["artifact_paths"]["report"],
                "rows": pareto["rows"],
                "best_passing": pareto["best_passing"],
            },
            sort_keys=True,
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--oracle-arrays", default=DEFAULT_ORACLE_ARRAYS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alphas",
        default="0.25,0.50,0.75",
        help="Interior support-mixture blend weights; endpoints are added automatically.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--eval-top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument(
        "--exclude-self",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-target-gain", type=float, default=0.005)
    parser.add_argument("--min-gap-fraction", type=float, default=0.50)
    args = parser.parse_args()
    run_pareto(args)


if __name__ == "__main__":
    main()
