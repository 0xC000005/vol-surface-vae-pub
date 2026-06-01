#!/usr/bin/env python
"""Backtest component-preserving narrative prefix rollouts on held-out windows.

The standard scenario-level evaluator scores the older analogue-generator path.
This script runs the actual prefix-latent story-smoke workflow on held-out cached
queries and compares the old averaged-prefix rollout against the newer
component-preserving support-mixture rollout.
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

from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
    DEFAULT_BRIDGE_REPORT,
    DEFAULT_CHECKPOINT,
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    DEFAULT_PIPELINE_REPORT,
    build_prefix_latent_run_args,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    load_bridge_report,
    select_heldout_query_rows,
    summarize_method_scores,
)


METHOD_KEY = "text_memory_plus_start_prefix_decoder"
DEFAULT_ROLLOUT_MODES = ("averaged_prefix", "component_prefix_mixture")
ALLOWED_ROLLOUT_MODES = (
    "averaged_prefix",
    "component_prefix_mixture",
    "response_preview_component_mixture",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_name(value: Any, fallback: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return text.strip("_") or fallback


def _operational_window_score(report: dict[str, Any]) -> dict[str, Any]:
    generation = report.get("generation") or {}
    rows = generation.get("window_scores") or []
    if not rows:
        raise ValueError("story-smoke report has no generation.window_scores")
    selected = report.get("selected_start_state") or {}
    variant_index = selected.get("variant_index")
    if variant_index is not None:
        for row in rows:
            if int(row.get("case_index", -1)) == int(variant_index):
                return row
    for row in rows:
        if (row.get("methods") or {}).get(METHOD_KEY):
            return row
    raise ValueError("story-smoke report has no scored operational rollout")


def _method_summary_row(
    *,
    query_no: int,
    query: dict[str, Any],
    mode_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    support: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    row_ref: dict[str, Any] | None = None
    for mode, report in mode_reports.items():
        row = _operational_window_score(report)
        row_ref = row
        if "persistence" in row.get("methods", {}):
            methods.setdefault("persistence", row["methods"]["persistence"])
        if METHOD_KEY not in row.get("methods", {}):
            raise ValueError(f"{mode} report missing {METHOD_KEY}")
        methods[mode] = row["methods"][METHOD_KEY]
        generation = report.get("generation") or {}
        cached = report.get("cached_query") or {}
        support[mode] = {
            "rollout_component_count": generation.get("rollout_component_count"),
            "rollout_mixture_mode": generation.get("rollout_mixture_mode"),
            "memory_prior_mode": cached.get("memory_prior_mode"),
            "top_support_windows": [
                item.get("window_id")
                for item in (cached.get("memory_prior") or {}).get("support_items", [])[:8]
                if isinstance(item, dict)
            ],
        }
        artifacts[mode] = report.get("artifact_paths", {})
    if row_ref is None:
        raise ValueError("no mode reports were supplied")
    return {
        "row_no": int(query_no),
        "window_index": int(query.get("window_index", row_ref.get("query_window_index", -1))),
        "window_id": str(query.get("window_id", "")),
        "query_role": str(query.get("role", "")),
        "query_kind": str(query.get("kind", "")),
        "query_id": str(query.get("query_id", "")),
        "start_window_index": int(row_ref.get("start_window_index", -1)),
        "block_window_index": int(row_ref.get("start_window_index", -1)),
        "methods": methods,
        "support": support,
        "artifacts": artifacts,
    }


def _build_story_args(
    *,
    query: dict[str, Any],
    mode: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> argparse.Namespace:
    story_args = build_prefix_latent_run_args(
        start_mode="original",
        samples=int(args.samples),
        live_story=False,
        condition_report=None,
        explicit_start_window_index=None,
        start_state_json=None,
        skip_rollout=False,
        output_dir=str(output_dir),
    )
    story_args.bridge_report = str(args.bridge_report)
    story_args.bridge_arrays = str(args.bridge_arrays)
    story_args.pipeline_report = str(args.pipeline_report)
    story_args.support_bank_report = (
        str(args.support_bank_report) if args.support_bank_report else None
    )
    story_args.support_bank_arrays = (
        str(args.support_bank_arrays) if args.support_bank_arrays else None
    )
    story_args.checkpoint = str(args.checkpoint)
    story_args.query_role = str(args.query_role)
    story_args.query_kind = str(query.get("kind", "")) or None
    story_args.query_window_id = str(query.get("window_id", "")) or None
    story_args.query_index = 0
    story_args.rollout_mixture_mode = str(mode)
    story_args.include_original_baseline = False
    story_args.memory_prior_mode = str(args.memory_prior_mode)
    story_args.memory_prior_top_k = int(args.memory_prior_top_k)
    story_args.memory_prior_temperature = float(args.memory_prior_temperature)
    story_args.memory_prior_diverse_max_pairwise_cosine = float(
        args.memory_prior_diverse_max_pairwise_cosine
    )
    story_args.memory_prior_diverse_min_index_gap = int(
        args.memory_prior_diverse_min_index_gap
    )
    story_args.memory_prior_quality_guard_candidate_pool_size = int(
        args.memory_prior_quality_guard_candidate_pool_size
    )
    story_args.memory_prior_quality_guard_mixture_size = int(
        args.memory_prior_quality_guard_mixture_size
    )
    story_args.memory_prior_quality_guard_max_mixtures = int(
        args.memory_prior_quality_guard_max_mixtures
    )
    story_args.memory_prior_quality_guard_min_candidate_mixtures = int(
        args.memory_prior_quality_guard_min_candidate_mixtures
    )
    story_args.memory_prior_quality_guard_max_candidate_entropy_quantile = float(
        args.memory_prior_quality_guard_max_candidate_entropy_quantile
    )
    story_args.memory_prior_quality_guard_min_support_weight_max_quantile = float(
        args.memory_prior_quality_guard_min_support_weight_max_quantile
    )
    story_args.memory_prior_quality_guard_probability_temperature = float(
        args.memory_prior_quality_guard_probability_temperature
    )
    story_args.start_distance_penalty = float(args.start_distance_penalty)
    story_args.implication_alignment_weight = float(args.implication_alignment_weight)
    story_args.steps = int(args.decoder_steps)
    story_args.batch_size = int(args.batch_size)
    story_args.eval_batch_size = int(args.eval_batch_size)
    story_args.samples = int(args.samples)
    story_args.n_steps = int(args.n_steps)
    story_args.chunk_size = int(args.chunk_size)
    story_args.temperature = float(args.temperature)
    story_args.seed = int(args.seed) + int(args.seed_stride) * int(query.get("window_index", 0))
    story_args.device = str(args.device)
    story_args.max_paths = int(args.max_paths)
    story_args.response_preview_samples_per_component = int(
        args.response_preview_samples_per_component
    )
    story_args.response_preview_alpha = float(args.response_preview_alpha)
    story_args.response_preview_temperature = float(args.response_preview_temperature)
    story_args.response_preview_blend = float(args.response_preview_blend)
    story_args.response_preview_objective = str(args.response_preview_objective)
    return story_args


def run_component_backtest(args: argparse.Namespace) -> dict[str, Any]:
    bridge_report = load_bridge_report(str(args.bridge_report))
    query_rows = select_heldout_query_rows(
        bridge_report,
        role=str(args.query_role),
        max_windows=int(args.max_windows),
        allow_duplicate_windows=bool(args.allow_duplicate_query_windows),
    )
    if not query_rows:
        raise ValueError("no held-out query rows selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_modes = tuple(str(mode) for mode in args.rollout_mode)
    window_scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row_no, query in enumerate(query_rows, start=1):
        window_label = _safe_name(query.get("window_id"), f"query_{row_no:03d}")
        mode_reports: dict[str, dict[str, Any]] = {}
        for mode in rollout_modes:
            run_dir = output_dir / "runs" / f"{row_no:03d}_{window_label}" / mode
            story_args = _build_story_args(
                query=query,
                mode=mode,
                output_dir=run_dir,
                args=args,
            )
            print(
                f"held-out {row_no}/{len(query_rows)} {window_label}: {mode}",
                flush=True,
            )
            try:
                mode_reports[mode] = run_prefix_latent_story_smoke(story_args)
            except Exception as exc:  # pragma: no cover - diagnostic script path
                failures.append(
                    {
                        "row_no": row_no,
                        "window_id": str(query.get("window_id", "")),
                        "mode": mode,
                        "error": repr(exc),
                    }
                )
                if not bool(args.keep_going):
                    raise
        if set(mode_reports) == set(rollout_modes):
            window_scores.append(
                _method_summary_row(
                    query_no=row_no - 1,
                    query=query,
                    mode_reports=mode_reports,
                )
            )

    summary = summarize_method_scores(window_scores, baseline="persistence")
    component = summary.get("component_prefix_mixture", {})
    averaged = summary.get("averaged_prefix", {})
    response_preview = summary.get("response_preview_component_mixture", {})
    comparison = {
        "component_minus_averaged_mean_crps": (
            float(component.get("ensemble_crps_z_mean", np.nan))
            - float(averaged.get("ensemble_crps_z_mean", np.nan))
        ),
        "component_minus_averaged_mean_energy": (
            float(component.get("energy_score_z_mean", np.nan))
            - float(averaged.get("energy_score_z_mean", np.nan))
        ),
        "component_crps_improvement_vs_persistence": component.get(
            "ensemble_crps_z_improvement_vs_persistence"
        ),
        "component_energy_improvement_vs_persistence": component.get(
            "energy_score_z_improvement_vs_persistence"
        ),
        "averaged_crps_improvement_vs_persistence": averaged.get(
            "ensemble_crps_z_improvement_vs_persistence"
        ),
        "averaged_energy_improvement_vs_persistence": averaged.get(
            "energy_score_z_improvement_vs_persistence"
        ),
        "response_preview_crps_improvement_vs_persistence": response_preview.get(
            "ensemble_crps_z_improvement_vs_persistence"
        ),
        "response_preview_energy_improvement_vs_persistence": response_preview.get(
            "energy_score_z_improvement_vs_persistence"
        ),
        "response_preview_minus_component_mean_crps": (
            float(response_preview.get("ensemble_crps_z_mean", np.nan))
            - float(component.get("ensemble_crps_z_mean", np.nan))
        ),
        "response_preview_minus_component_mean_energy": (
            float(response_preview.get("energy_score_z_mean", np.nan))
            - float(component.get("energy_score_z_mean", np.nan))
        ),
    }
    report = {
        "status": "ok" if not failures else "partial",
        "scope_note": (
            "Held-out cached narrative backtest for the prefix-latent workflow. "
            "No OpenAI API calls are made. Each historical query uses its own "
            "observed starting level and realized next-30-day path."
        ),
        "research_lane": "candidate",
        "result_status": "mechanism_found",
        "bridge_report": str(args.bridge_report),
        "bridge_arrays": str(args.bridge_arrays),
        "pipeline_report": str(args.pipeline_report),
        "support_bank_report": str(args.support_bank_report or ""),
        "support_bank_arrays": str(args.support_bank_arrays or ""),
        "checkpoint": str(args.checkpoint),
        "query_role": str(args.query_role),
        "heldout_window_count_requested": int(args.max_windows),
        "heldout_window_count_scored": len(window_scores),
        "failures": failures,
        "rollout_modes": list(rollout_modes),
        "memory_prior_mode": str(args.memory_prior_mode),
        "memory_prior_top_k": int(args.memory_prior_top_k),
        "samples": int(args.samples),
        "n_steps": int(args.n_steps),
        "decoder_steps": int(args.decoder_steps),
        "temperature": float(args.temperature),
        "response_preview_samples_per_component": int(
            args.response_preview_samples_per_component
        ),
        "response_preview_alpha": float(args.response_preview_alpha),
        "response_preview_temperature": float(args.response_preview_temperature),
        "response_preview_blend": float(args.response_preview_blend),
        "response_preview_objective": str(args.response_preview_objective),
        "device": str(args.device),
        "summary": summary,
        "mode_comparison": comparison,
        "window_scores": window_scores,
        "artifact_paths": {
            "report": str(output_dir / "component_backtest_report.json"),
        },
    }
    _write_json(output_dir / "component_backtest_report.json", report)
    print(json.dumps(report["mode_comparison"], indent=2, sort_keys=True))
    print(f"wrote {output_dir / 'component_backtest_report.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--pipeline-report", default=DEFAULT_PIPELINE_REPORT)
    parser.add_argument("--support-bank-report")
    parser.add_argument("--support-bank-arrays")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_component_backtest_heldout"
        ),
    )
    parser.add_argument("--query-role", default="anchor")
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--allow-duplicate-query-windows", action="store_true")
    parser.add_argument(
        "--rollout-mode",
        action="append",
        choices=list(ALLOWED_ROLLOUT_MODES),
        default=None,
        help=(
            "Rollout mode to backtest. Repeat to compare modes. Defaults to "
            "averaged_prefix and component_prefix_mixture."
        ),
    )
    parser.add_argument(
        "--memory-prior-mode",
        default="diverse_topk_narrative_start_checked",
        choices=[
            "query_memory",
            "soft_topk_memory",
            "soft_topk_narrative_start",
            "soft_topk_narrative_start_checked",
            "diverse_topk_narrative_start_checked",
            "cohesive_topk_narrative_start_checked",
            "cluster_family_narrative_start_checked",
            "kernel_topk_narrative_start_checked",
            "portfolio_quality_guard_924e",
            "portfolio_direction_first_quality_guard_938a",
            "broad_replay_response_guard_940a",
            "narrative_book_quality_guard_926b",
            "narrative_book_direction_first_quality_guard_938c",
            "soft_topk_start_only",
            "soft_topk_combined",
            "diverse_topk_narrative_start",
            "diverse_topk_combined",
        ],
    )
    parser.add_argument("--memory-prior-top-k", type=int, default=8)
    parser.add_argument("--memory-prior-temperature", type=float, default=0.2)
    parser.add_argument("--memory-prior-diverse-max-pairwise-cosine", type=float, default=0.98)
    parser.add_argument("--memory-prior-diverse-min-index-gap", type=int, default=30)
    parser.add_argument("--memory-prior-quality-guard-candidate-pool-size", type=int, default=12)
    parser.add_argument("--memory-prior-quality-guard-mixture-size", type=int, default=3)
    parser.add_argument("--memory-prior-quality-guard-max-mixtures", type=int, default=64)
    parser.add_argument("--memory-prior-quality-guard-min-candidate-mixtures", type=int, default=4)
    parser.add_argument(
        "--memory-prior-quality-guard-max-candidate-entropy-quantile",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--memory-prior-quality-guard-min-support-weight-max-quantile",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--memory-prior-quality-guard-probability-temperature",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument("--implication-alignment-weight", type=float, default=0.25)
    parser.add_argument("--decoder-steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=910)
    parser.add_argument("--seed-stride", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-paths", type=int, default=3)
    parser.add_argument("--response-preview-samples-per-component", type=int, default=8)
    parser.add_argument("--response-preview-alpha", type=float, default=0.75)
    parser.add_argument("--response-preview-temperature", type=float, default=1.0)
    parser.add_argument("--response-preview-blend", type=float, default=1.0)
    parser.add_argument(
        "--response-preview-objective",
        choices=["narrative_channels", "factor_portfolio", "channel_portfolio"],
        default="narrative_channels",
    )
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    if args.rollout_mode is None:
        args.rollout_mode = list(DEFAULT_ROLLOUT_MODES)
    run_component_backtest(args)


if __name__ == "__main__":
    main()
