#!/usr/bin/env python
"""Run sequential paired rollouts for active quality-guard rows."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)
from experiments.backfill.block_ar.nl_portfolio_quality_guard_paired_rollout_summary import (  # noqa: E402
    build_pair_summary,
    load_story_smoke_case,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_sensitivity import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIPELINE_REPORT,
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_prefix_latent_validation_gate import (  # noqa: E402
    DEFAULT_GATE_THRESHOLDS,
)
from experiments.backfill.block_ar.nl_risk_manager_story_smoke import (  # noqa: E402
    DEFAULT_BRIDGE_ADAPTER,
    DEFAULT_STORY,
)


DEFAULT_BREADTH_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_quality_guard_live_breadth_925f_minbreadth/"
    "portfolio_quality_guard_live_breadth.json"
)
DEFAULT_RUNNER_OUTPUT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_quality_guard_paired_rollout_runner_925j"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "case"


def select_active_breadth_rows(
    breadth_report: dict[str, Any],
    *,
    max_pairs: int,
    require_changed_support: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for row in breadth_report.get("rows", []):
        if not isinstance(row, dict):
            continue
        if bool(row.get("quality_guard_fallback")):
            continue
        if bool(row.get("quality_guard_low_breadth")):
            continue
        if not bool(row.get("quality_guard_active")):
            continue
        if require_changed_support and float(row.get("support_jaccard", 1.0)) >= 1.0:
            continue
        rows.append(dict(row))
    rows.sort(
        key=lambda item: (
            float(item.get("support_jaccard", 1.0)),
            -int(item.get("quality_guard_candidate_count", 0) or 0),
            str(item.get("case_name", "")),
            int(item.get("start_window_index", 0) or 0),
        )
    )
    if int(max_pairs) > 0:
        rows = rows[: int(max_pairs)]
    return rows


def _story_args(
    *,
    condition_report: str,
    output_dir: str,
    start_window_index: int,
    memory_prior_mode: str,
    samples: int,
    steps: int,
    seed: int,
    device: str,
    quality_guard_candidate_pool_size: int,
    quality_guard_mixture_size: int,
    quality_guard_max_mixtures: int,
    quality_guard_min_candidate_mixtures: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        bridge_report=DEFAULT_BRIDGE_REPORT,
        bridge_arrays=DEFAULT_BRIDGE_ARRAYS,
        pipeline_report=DEFAULT_PIPELINE_REPORT,
        checkpoint=DEFAULT_CHECKPOINT,
        output_dir=str(output_dir or DEFAULT_OUTPUT_DIR),
        query_role="anchor",
        query_kind=None,
        query_window_id=None,
        query_index=0,
        condition_report=str(condition_report),
        live_story=False,
        story=DEFAULT_STORY,
        grounding_json=None,
        grounding_model="gpt-5.4-mini",
        grounding_max_output_tokens=1200,
        embedding_model="text-embedding-3-small",
        bridge_adapter=DEFAULT_BRIDGE_ADAPTER,
        start_reliability_manifest=None,
        dotenv=".env",
        start_mode="explicit_start_window",
        explicit_start_window_index=int(start_window_index),
        start_state_json=None,
        start_distance_threshold_z=float(DEFAULT_GATE_THRESHOLDS["start_distance_warn"]),
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        memory_prior_mode=str(memory_prior_mode),
        memory_prior_top_k=8,
        memory_prior_temperature=0.2,
        memory_prior_diverse_max_pairwise_cosine=0.95,
        memory_prior_diverse_min_index_gap=30,
        memory_prior_quality_guard_candidate_pool_size=int(
            quality_guard_candidate_pool_size
        ),
        memory_prior_quality_guard_mixture_size=int(quality_guard_mixture_size),
        memory_prior_quality_guard_max_mixtures=int(quality_guard_max_mixtures),
        memory_prior_quality_guard_min_candidate_mixtures=int(
            quality_guard_min_candidate_mixtures
        ),
        prefix_prior_mode="decoder",
        rollout_mixture_mode="component_prefix_mixture",
        include_original_baseline=True,
        hidden_dim=256,
        steps=int(steps),
        batch_size=64,
        eval_batch_size=16,
        lr=1e-3,
        seed=int(seed),
        device=str(device),
        skip_rollout=False,
        samples=int(samples),
        n_steps=30,
        chunk_size=max(4, min(16, int(samples))),
        temperature=0.50,
        rollout_fan_scale=1.0,
        score_scale_floor=1e-3,
        hard_case_count=8,
        max_paths=6,
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


def _write_pair_summary_artifacts(
    *,
    pair_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "portfolio_quality_guard_paired_rollout_summary.json"
    markdown_path = output_dir / "portfolio_quality_guard_paired_rollout_summary.md"
    pair_summary["artifact_paths"] = {
        "report": str(json_path),
        "markdown": str(markdown_path),
    }
    _write_json(json_path, pair_summary)
    _write_text(markdown_path, _pair_markdown(pair_summary))
    return dict(pair_summary["artifact_paths"])


def _pair_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Paired Rollout: {report['case_name']}",
        "",
        f"Support Jaccard: `{float(report['support_jaccard']):.3f}`",
        f"Base support: `{report['base_support_indices']}`",
        f"Quality guard support: `{report['quality_guard_support_indices']}`",
        "",
        "## Factors",
        "",
        "| Factor | Mean Delta | Width Delta |",
        "|---|---:|---:|",
    ]
    for row in report["factor_terminal_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["factor"]),
                    f"{float(row['terminal_mean_delta_qg_minus_base']):.4f}",
                    f"{float(row['width_delta_qg_minus_base']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Portfolios", "", "| Book | PnL Delta | P05 Delta | Path Loss Delta |", "|---|---:|---:|---:|"])
    for row in report["portfolio_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["book"]),
                    f"{float(row['terminal_mean_pnl_delta_qg_minus_base']):.4f}",
                    f"{float(row['terminal_p05_delta_qg_minus_base']):.4f}",
                    f"{float(row['path_loss_mean_delta_qg_minus_base']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _aggregate(pair_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not pair_reports:
        return {"status": "empty", "case_count": 0, "cases": []}
    cases = []
    for report in pair_reports:
        factors = {row["factor"]: row for row in report["factor_terminal_rows"]}
        equity = next(
            row for row in report["portfolio_rows"] if row["book"] == "equity_beta_carry"
        )
        cases.append(
            {
                "case_name": report["case_name"],
                "support_jaccard": float(report["support_jaccard"]),
                "spx_mean_delta": float(
                    factors["SPX"]["terminal_mean_delta_qg_minus_base"]
                ),
                "vix_mean_delta": float(
                    factors["VIX"]["terminal_mean_delta_qg_minus_base"]
                ),
                "vix_width_delta": float(factors["VIX"]["width_delta_qg_minus_base"]),
                "equity_beta_path_loss_delta": float(
                    equity["path_loss_mean_delta_qg_minus_base"]
                ),
            }
        )
    return {
        "status": "pilot_completed",
        "case_count": len(pair_reports),
        "mean_support_jaccard": float(np.mean([c["support_jaccard"] for c in cases])),
        "mean_abs_spx_delta": float(np.mean([abs(c["spx_mean_delta"]) for c in cases])),
        "mean_vix_width_delta": float(np.mean([c["vix_width_delta"] for c in cases])),
        "mean_equity_beta_path_loss_delta": float(
            np.mean([c["equity_beta_path_loss_delta"] for c in cases])
        ),
        "cases": cases,
    }


def run_paired_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    breadth = _load_json(args.breadth_report)
    selected_rows = select_active_breadth_rows(
        breadth,
        max_pairs=int(args.max_pairs),
        require_changed_support=not bool(args.include_unchanged_support),
    )
    pair_reports: list[dict[str, Any]] = []
    for row in selected_rows:
        case_name = str(row["case_name"])
        start_idx = int(row["start_window_index"])
        condition_report = str(row["condition_report"])
        case_slug = f"{_slug(case_name)}_start{start_idx}"
        base_dir = Path(args.output_dir) / case_slug / "base"
        guarded_dir = Path(args.output_dir) / case_slug / "quality_guard"
        run_prefix_latent_story_smoke(
            _story_args(
                condition_report=condition_report,
                output_dir=str(base_dir),
                start_window_index=start_idx,
                memory_prior_mode="diverse_topk_narrative_start_checked",
                samples=int(args.samples),
                steps=int(args.steps),
                seed=int(args.seed),
                device=str(args.device),
                quality_guard_candidate_pool_size=int(args.quality_guard_candidate_pool_size),
                quality_guard_mixture_size=int(args.quality_guard_mixture_size),
                quality_guard_max_mixtures=int(args.quality_guard_max_mixtures),
                quality_guard_min_candidate_mixtures=int(args.quality_guard_min_candidate_mixtures),
            )
        )
        run_prefix_latent_story_smoke(
            _story_args(
                condition_report=condition_report,
                output_dir=str(guarded_dir),
                start_window_index=start_idx,
                memory_prior_mode="portfolio_quality_guard_924e",
                samples=int(args.samples),
                steps=int(args.steps),
                seed=int(args.seed),
                device=str(args.device),
                quality_guard_candidate_pool_size=int(args.quality_guard_candidate_pool_size),
                quality_guard_mixture_size=int(args.quality_guard_mixture_size),
                quality_guard_max_mixtures=int(args.quality_guard_max_mixtures),
                quality_guard_min_candidate_mixtures=int(args.quality_guard_min_candidate_mixtures),
            )
        )
        pair_summary = build_pair_summary(
            case_name=f"{case_name}_start{start_idx}",
            base=load_story_smoke_case(base_dir),
            guarded=load_story_smoke_case(guarded_dir),
        )
        _write_pair_summary_artifacts(
            pair_summary=pair_summary,
            output_dir=Path(args.output_dir) / case_slug / "summary",
        )
        pair_reports.append(pair_summary)
    aggregate = _aggregate(pair_reports)
    aggregate["selected_row_count"] = len(selected_rows)
    aggregate["source_breadth_report"] = str(args.breadth_report)
    aggregate["artifact_paths"] = {
        "report": str(Path(args.output_dir) / "paired_rollout_runner_summary.json"),
        "markdown": str(Path(args.output_dir) / "paired_rollout_runner_summary.md"),
    }
    _write_json(aggregate["artifact_paths"]["report"], aggregate)
    _write_text(aggregate["artifact_paths"]["markdown"], _aggregate_markdown(aggregate))
    return aggregate


def _aggregate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Quality-Guard Paired Rollout Runner",
        "",
        f"Status: `{report['status']}`",
        f"Cases: `{report['case_count']}`",
        f"Mean support Jaccard: `{report.get('mean_support_jaccard')}`",
        "",
        "| Case | Jaccard | SPX Delta | VIX Delta | VIX Width Delta | Equity Path Loss Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("cases", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_name"]),
                    f"{float(row['support_jaccard']):.3f}",
                    f"{float(row['spx_mean_delta']):.4f}",
                    f"{float(row['vix_mean_delta']):.4f}",
                    f"{float(row['vix_width_delta']):.4f}",
                    f"{float(row['equity_beta_path_loss_delta']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breadth-report", type=Path, default=DEFAULT_BREADTH_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNNER_OUTPUT)
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--include-unchanged-support", action="store_true")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=791)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quality-guard-candidate-pool-size", type=int, default=12)
    parser.add_argument("--quality-guard-mixture-size", type=int, default=3)
    parser.add_argument("--quality-guard-max-mixtures", type=int, default=64)
    parser.add_argument("--quality-guard-min-candidate-mixtures", type=int, default=4)
    args = parser.parse_args()
    report = run_paired_rollouts(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": report["case_count"],
                "report": report["artifact_paths"]["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
