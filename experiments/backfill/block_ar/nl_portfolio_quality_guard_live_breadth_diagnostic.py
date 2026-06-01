#!/usr/bin/env python
"""Diagnose live candidate breadth for the portfolio quality guard.

This is a cheap, rollout-free check for the opt-in
``portfolio_quality_guard_924e`` story-smoke memory prior.  It compares the
current base diverse support prior against the quality-guard prior across cached
condition reports and fixed starting windows, then reports whether the guard has
enough live candidate mixtures to be a meaningful product lever.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    _spec_names,
)
from experiments.backfill.block_ar.nl_portfolio_response_quality_guard_policy import (  # noqa: E402
    build_quality_guard_policy_context,
)
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (  # noqa: E402
    build_mixture_memory_prior,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    selected_bridge_window_indices,
    split_indices_from_bridge_report,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_sensitivity import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    _external_condition_from_report,
    _load_json,
    _write_json,
    _write_text,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    load_bridge_arrays,
)


DEFAULT_CONDITION_REPORTS = [
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_full906b_914a/fragile_risk_on_start18/"
        "condition_only_report.json"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_full906b_914a/defensive_risk_off_start18/"
        "condition_only_report.json"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_full906b_914a/commodity_inflation_start18/"
        "condition_only_report.json"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_full906b_914a/dollar_liquidity_start18/"
        "condition_only_report.json"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_full906b_914a/safe_haven_gold_start18/"
        "condition_only_report.json"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_full906b_914a/rates_selloff_start18/"
        "condition_only_report.json"
    ),
]
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_quality_guard_live_breadth_925e"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _support_jaccard(left: list[int], right: list[int]) -> float:
    a = set(int(item) for item in left)
    b = set(int(item) for item in right)
    if not a and not b:
        return 1.0
    union = a | b
    return float(len(a & b) / max(len(union), 1))


def _case_name(path: Path, report: dict[str, Any]) -> str:
    raw = report.get("case_name")
    if raw:
        return str(raw)
    return path.parent.name


def _base_block_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        state_scope="joint38",
        eval_split="val",
        test_start=4511,
        val_size=441,
        max_windows=441,
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )


def load_support_context(args: argparse.Namespace) -> dict[str, Any]:
    bridge_report = _load_json(args.bridge_report)
    selected_windows = selected_bridge_window_indices(bridge_report)
    train_indices, _test_indices = split_indices_from_bridge_report(bridge_report)
    bridge_arrays = load_bridge_arrays(args.bridge_arrays)
    memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    device = torch.device(
        args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu"
    )
    _model, payload = load_model(args.checkpoint, device)
    (
        all_history_level,
        _all_history_norm,
        _all_center,
        _all_scale,
        _all_drift_feature,
        _all_history_raw,
        specs,
        _block,
    ) = build_val_block(_base_block_args(args), payload)
    history_level = all_history_level[selected_windows]
    return {
        "bridge_report": bridge_report,
        "history_level": np.asarray(history_level, dtype=np.float32),
        "memory_targets": memory_targets,
        "train_indices": np.asarray(train_indices, dtype=np.int64),
        "spec_names": _spec_names(specs),
        "quality_guard_context": build_quality_guard_policy_context(
            max_candidate_entropy_quantile=(
                None
                if float(args.quality_guard_max_candidate_entropy_quantile) < 0.0
                else float(args.quality_guard_max_candidate_entropy_quantile)
            ),
            min_support_weight_max_quantile=(
                None
                if float(args.quality_guard_min_support_weight_max_quantile) < 0.0
                else float(args.quality_guard_min_support_weight_max_quantile)
            ),
        ),
    }


def compare_live_priors(
    *,
    condition_report_paths: list[Path],
    start_window_indices: list[int],
    context: dict[str, Any],
    query_window_index: int,
    top_k: int,
    temperature: float,
    diverse_max_pairwise_cosine: float,
    diverse_min_index_gap: int,
    quality_guard_candidate_pool_size: int,
    quality_guard_mixture_size: int,
    quality_guard_max_mixtures: int,
    min_candidate_mixtures: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    history_level = np.asarray(context["history_level"], dtype=np.float32)
    memory_targets = np.asarray(context["memory_targets"], dtype=np.float32)
    train_indices = np.asarray(context["train_indices"], dtype=np.int64)
    spec_names = list(context["spec_names"])
    quality_guard_context = context["quality_guard_context"]
    for condition_path in condition_report_paths:
        condition_path = Path(condition_path)
        report = _load_json(condition_path)
        condition = _external_condition_from_report(condition_path)
        query_memory = np.asarray(condition["query_memory"], dtype=np.float32)
        grounding = (
            condition["grounding"] if isinstance(condition["grounding"], dict) else {}
        )
        case_name = _case_name(condition_path, report)
        for start_idx in start_window_indices:
            if start_idx < 0 or start_idx >= history_level.shape[0]:
                raise IndexError(
                    f"start index {start_idx} outside history_level size "
                    f"{history_level.shape[0]}"
                )
            query_start_state = np.asarray(
                history_level[start_idx, -1, :], dtype=np.float32
            )
            base = build_mixture_memory_prior(
                query_memory=query_memory,
                memory_targets=memory_targets,
                history_level=history_level,
                train_indices=train_indices,
                query_window_index=int(query_window_index),
                query_start_state=query_start_state,
                grounding=grounding,
                spec_names=spec_names,
                mode="diverse_topk_narrative_start_checked",
                top_k=int(top_k),
                temperature=float(temperature),
                start_distance_threshold_z=15.0,
                start_distance_penalty=0.02,
                implication_alignment_weight=0.25,
                diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
                diverse_min_index_gap=int(diverse_min_index_gap),
            )
            guarded = build_mixture_memory_prior(
                query_memory=query_memory,
                memory_targets=memory_targets,
                history_level=history_level,
                train_indices=train_indices,
                query_window_index=int(query_window_index),
                query_start_state=query_start_state,
                grounding=grounding,
                spec_names=spec_names,
                mode="portfolio_quality_guard_924e",
                top_k=int(top_k),
                temperature=float(temperature),
                start_distance_threshold_z=15.0,
                start_distance_penalty=0.02,
                implication_alignment_weight=0.25,
                diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
                diverse_min_index_gap=int(diverse_min_index_gap),
                quality_guard_context=quality_guard_context,
                quality_guard_candidate_pool_size=int(
                    quality_guard_candidate_pool_size
                ),
                quality_guard_mixture_size=int(quality_guard_mixture_size),
                quality_guard_max_mixtures=int(quality_guard_max_mixtures),
                quality_guard_min_candidate_mixtures=int(min_candidate_mixtures),
            )
            support_policy = guarded.get("portfolio_quality_guard_policy", {})
            if not isinstance(support_policy, dict):
                support_policy = {}
            diversity_policy = guarded.get("support_diversity_policy", {})
            if not isinstance(diversity_policy, dict):
                diversity_policy = {}
            base_indices = [int(item) for item in base.get("window_indices", [])]
            guarded_indices = [int(item) for item in guarded.get("window_indices", [])]
            candidate_count = int(
                support_policy.get(
                    "candidate_count",
                    diversity_policy.get("quality_guard_candidate_count", 0),
                )
                or 0
            )
            fallback = bool(support_policy.get("fallback_to_equal_support", False))
            active_low_breadth = bool(
                candidate_count < int(min_candidate_mixtures) and not fallback
            )
            rows.append(
                {
                    "case_name": case_name,
                    "condition_report": str(condition_path),
                    "start_window_index": int(start_idx),
                    "base_window_indices": base_indices,
                    "quality_guard_window_indices": guarded_indices,
                    "base_analogue_count": int(base.get("analogue_count", 0) or 0),
                    "quality_guard_analogue_count": int(
                        guarded.get("analogue_count", 0) or 0
                    ),
                    "support_jaccard": _support_jaccard(base_indices, guarded_indices),
                    "quality_guard_candidate_count": candidate_count,
                    "quality_guard_low_breadth": bool(
                        candidate_count < int(min_candidate_mixtures)
                    ),
                    "quality_guard_active_low_breadth": active_low_breadth,
                    "quality_guard_fallback": fallback,
                    "quality_guard_fallback_reason": str(
                        support_policy.get("fallback_reason", "")
                    ),
                    "quality_guard_active": bool(
                        diversity_policy.get("portfolio_quality_guard_active", False)
                    ),
                    "quality_guard_direction_status": str(
                        (
                            guarded.get("direction_check", {})
                            if isinstance(guarded.get("direction_check", {}), dict)
                            else {}
                        ).get("status", "")
                    ),
                    "quality_guard_support_weight_max": (
                        None
                        if support_policy.get("support_weight_max") is None
                        else float(support_policy.get("support_weight_max"))
                    ),
                    "quality_guard_support_weight_max_threshold": (
                        None
                        if support_policy.get("min_support_weight_max_threshold")
                        is None
                        else float(
                            support_policy.get("min_support_weight_max_threshold")
                        )
                    ),
                    "quality_guard_selected_score": (
                        None
                        if support_policy.get("selected_score") is None
                        else float(support_policy.get("selected_score"))
                    ),
                }
            )
    return summarize_rows(
        rows,
        min_candidate_mixtures=int(min_candidate_mixtures),
        condition_reports=[str(path) for path in condition_report_paths],
        start_window_indices=start_window_indices,
        top_k=int(top_k),
        quality_guard_candidate_pool_size=int(quality_guard_candidate_pool_size),
        quality_guard_mixture_size=int(quality_guard_mixture_size),
        quality_guard_max_mixtures=int(quality_guard_max_mixtures),
        diverse_min_index_gap=int(diverse_min_index_gap),
        diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
    )


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    min_candidate_mixtures: int,
    condition_reports: list[str] | None = None,
    start_window_indices: list[int] | None = None,
    top_k: int = 8,
    quality_guard_candidate_pool_size: int = 12,
    quality_guard_mixture_size: int = 3,
    quality_guard_max_mixtures: int = 64,
    diverse_min_index_gap: int = 30,
    diverse_max_pairwise_cosine: float = 0.95,
) -> dict[str, Any]:
    candidate_counts = [
        int(row.get("quality_guard_candidate_count", 0) or 0) for row in rows
    ]
    jaccards = [float(row.get("support_jaccard", 0.0) or 0.0) for row in rows]
    fallback_count = sum(1 for row in rows if bool(row.get("quality_guard_fallback")))
    low_breadth_count = sum(
        1 for row in rows if bool(row.get("quality_guard_low_breadth"))
    )
    active_low_breadth_count = sum(
        1
        for row in rows
        if bool(
            row.get(
                "quality_guard_active_low_breadth",
                bool(row.get("quality_guard_low_breadth"))
                and not bool(row.get("quality_guard_fallback")),
            )
        )
    )
    changed_support_count = sum(
        1 for row in rows if float(row.get("support_jaccard", 1.0) or 1.0) < 1.0
    )
    row_count = len(rows)
    if row_count == 0:
        status = "empty"
    elif active_low_breadth_count > 0:
        status = "candidate_breadth_warning"
    elif fallback_count > 0:
        status = "fallback_warning"
    elif changed_support_count == 0:
        status = "no_live_support_change"
    else:
        status = "candidate_breadth_ok"
    return {
        "status": status,
        "row_count": int(row_count),
        "condition_reports": list(condition_reports or []),
        "start_window_indices": [int(item) for item in (start_window_indices or [])],
        "config": {
            "top_k": int(top_k),
            "quality_guard_candidate_pool_size": int(quality_guard_candidate_pool_size),
            "quality_guard_mixture_size": int(quality_guard_mixture_size),
            "quality_guard_max_mixtures": int(quality_guard_max_mixtures),
            "min_candidate_mixtures": int(min_candidate_mixtures),
            "diverse_min_index_gap": int(diverse_min_index_gap),
            "diverse_max_pairwise_cosine": float(diverse_max_pairwise_cosine),
        },
        "summary": {
            "quality_guard_fallback_count": int(fallback_count),
            "quality_guard_low_breadth_count": int(low_breadth_count),
            "quality_guard_active_low_breadth_count": int(active_low_breadth_count),
            "changed_support_count": int(changed_support_count),
            "mean_support_jaccard": (
                None if not jaccards else float(np.mean(np.asarray(jaccards)))
            ),
            "min_quality_guard_candidate_count": (
                None if not candidate_counts else int(min(candidate_counts))
            ),
            "mean_quality_guard_candidate_count": (
                None
                if not candidate_counts
                else float(np.mean(np.asarray(candidate_counts, dtype=np.float64)))
            ),
            "median_quality_guard_candidate_count": (
                None
                if not candidate_counts
                else float(np.median(np.asarray(candidate_counts, dtype=np.float64)))
            ),
        },
        "decision": {
            "promote_live_default": False,
            "interpretation": _interpret_status(
                status,
                fallback_count=fallback_count,
                low_breadth_count=low_breadth_count,
                active_low_breadth_count=active_low_breadth_count,
                row_count=row_count,
            ),
        },
        "rows": rows,
    }


def _interpret_status(
    status: str,
    *,
    fallback_count: int,
    low_breadth_count: int,
    active_low_breadth_count: int,
    row_count: int,
) -> str:
    if status == "candidate_breadth_warning":
        return (
            "The live quality guard still lacks enough candidate-mixture breadth "
            f"on {active_low_breadth_count}/{row_count} active rows. Keep it "
            "opt-in and fix "
            "candidate generation before any demo/default promotion."
        )
    if status == "fallback_warning":
        return (
            "The active guard no longer uses low-breadth candidate sets, but it "
            f"falls back on {fallback_count}/{row_count} rows, including "
            f"{low_breadth_count} low-breadth rows. Treat it as an optional "
            "overlay."
        )
    if status == "no_live_support_change":
        return (
            "The guard has breadth but does not change live support selection. "
            "It is not yet a useful product lever."
        )
    if status == "candidate_breadth_ok":
        return (
            "The guard has enough live candidate breadth and changes support on "
            "some rows. A paired rollout comparison is now justified."
        )
    return "No rows were evaluated."


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Portfolio Quality Guard Live Breadth Diagnostic",
        "",
        f"Status: `{report['status']}`",
        f"Rows: `{report['row_count']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(report["decision"]["interpretation"]),
            "",
            "## Rows",
            "",
            (
                "| Case | Start | Base support | Guard support | Candidates | "
                "Fallback | Jaccard | Active |"
            ),
            "|---|---:|---|---|---:|---|---:|---|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_name"]),
                    str(row["start_window_index"]),
                    str(row["base_window_indices"]),
                    str(row["quality_guard_window_indices"]),
                    str(row["quality_guard_candidate_count"]),
                    str(row["quality_guard_fallback"])
                    + (
                        ""
                        if not row.get("quality_guard_fallback_reason")
                        else f" ({row['quality_guard_fallback_reason']})"
                    ),
                    f"{float(row['support_jaccard']):.3f}",
                    str(row["quality_guard_active"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", type=Path, default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", type=Path, default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--condition-reports",
        type=Path,
        nargs="*",
        default=DEFAULT_CONDITION_REPORTS,
    )
    parser.add_argument("--start-window-indices", default="0,18,22,40,77,178")
    parser.add_argument("--query-window-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--diverse-max-pairwise-cosine", type=float, default=0.95)
    parser.add_argument("--diverse-min-index-gap", type=int, default=30)
    parser.add_argument("--quality-guard-candidate-pool-size", type=int, default=12)
    parser.add_argument("--quality-guard-mixture-size", type=int, default=3)
    parser.add_argument("--quality-guard-max-mixtures", type=int, default=64)
    parser.add_argument("--min-candidate-mixtures", type=int, default=4)
    parser.add_argument(
        "--quality-guard-max-candidate-entropy-quantile",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--quality-guard-min-support-weight-max-quantile",
        type=float,
        default=0.25,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    context = load_support_context(args)
    report = compare_live_priors(
        condition_report_paths=[Path(path) for path in args.condition_reports],
        start_window_indices=_parse_ints(args.start_window_indices),
        context=context,
        query_window_index=int(args.query_window_index),
        top_k=int(args.top_k),
        temperature=float(args.temperature),
        diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
        diverse_min_index_gap=int(args.diverse_min_index_gap),
        quality_guard_candidate_pool_size=int(args.quality_guard_candidate_pool_size),
        quality_guard_mixture_size=int(args.quality_guard_mixture_size),
        quality_guard_max_mixtures=int(args.quality_guard_max_mixtures),
        min_candidate_mixtures=int(args.min_candidate_mixtures),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "portfolio_quality_guard_live_breadth.json"
    markdown_path = args.output_dir / "portfolio_quality_guard_live_breadth.md"
    report["artifact_paths"] = {
        "report": str(json_path),
        "markdown": str(markdown_path),
    }
    _write_json(json_path, _jsonable(report))
    _write_text(markdown_path, _markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "row_count": report["row_count"],
                "summary": report["summary"],
                "report": str(json_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
