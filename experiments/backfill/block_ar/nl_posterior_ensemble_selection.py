#!/usr/bin/env python
"""Select a production ensemble view for narrative-conditioned scenarios.

This artifact-only evaluator reuses saved component-prefix backtest rollouts.
It does not call OpenAI and does not rerun the frozen SNI generator. For each
saved backtest report, it rescales the generated samples by component posterior
view:

* all selected regimes;
* strongest regime only;
* main-regime view, top two or 80 percent cumulative weight;
* main-regime view, top three or 90 percent cumulative weight.

The output combines held-out scenario metrics with fixed-start conditionality
metrics from the support-coherence component-posterior bakeoff.
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
from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (  # noqa: E402
    component_slices_for_variant,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)
from experiments.backfill.block_ar.nl_prefix_latent_memory_decoder import (  # noqa: E402
    _future_raw_from_block,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    future_delta_paths,
    score_sample_distribution,
    summarize_method_scores,
)
from experiments.backfill.block_ar.nl_sparse_component_family_view import (  # noqa: E402
    select_sparse_components,
)
from experiments.backfill.block_ar.nl_support_component_posterior_bakeoff import (  # noqa: E402
    POSTERIOR_MODES,
    PLOT_POLICY_LABELS,
    PLOT_POSTERIOR_LABELS,
)


DEFAULT_BACKTEST_REPORTS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "support_cohesion_component_posterior_bakeoff_960b_guardrail_current_24w_s24/"
    "component_backtest_report.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "support_cohesion_component_posterior_bakeoff_960b_guardrail_cohesive_24w_s24/"
    "component_backtest_report.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "support_cohesion_component_posterior_bakeoff_960b_guardrail_cluster_24w_s24/"
    "component_backtest_report.json",
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "support_cohesion_component_posterior_bakeoff_960b_guardrail_kernel_24w_s24/"
    "component_backtest_report.json",
)
DEFAULT_CONDITIONALITY_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "support_cohesion_component_posterior_bakeoff_960a_smoke_s8_start22/"
    "component_posterior_bakeoff.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "posterior_ensemble_selection_961a"
)

POLICY_LABEL_BY_MEMORY_MODE = {
    "diverse_topk_narrative_start_checked": "Diverse historical regimes",
    "cohesive_topk_narrative_start_checked": "Nearest similar regimes",
    "cluster_family_narrative_start_checked": "One similar-regime family",
    "kernel_topk_narrative_start_checked": "Similarity-weighted regimes",
}
POLICY_KEY_BY_MEMORY_MODE = {
    "diverse_topk_narrative_start_checked": "current_start_checked_gap30",
    "cohesive_topk_narrative_start_checked": "cohesive_support_gap30",
    "cluster_family_narrative_start_checked": "cluster_family_support_gap30",
    "kernel_topk_narrative_start_checked": "kernel_similarity_support_gap30",
}


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
        return value if math.isfinite(value) else None
    return value


def _val_future_delta(*, checkpoint: str) -> np.ndarray:
    args = SimpleNamespace(
        data_path="data/vol_surface_with_ret.npz",
        state_scope="joint38",
        eval_split="val",
        test_start=4511,
        val_size=441,
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
        max_windows=441,
    )
    _model, payload = load_model(checkpoint, torch.device("cpu"))
    *_, all_history_raw, _specs, block = build_val_block(args, payload)
    future_raw = _future_raw_from_block(
        block,
        int(all_history_raw.shape[0]),
        int(all_history_raw.shape[-1]),
    )
    return future_delta_paths(all_history_raw, future_raw)


def _component_rows(arrays: Any, *, variant: int, sample_count: int) -> list[dict[str, Any]]:
    rows = component_slices_for_variant(
        variant_index=int(variant),
        component_variant_index=np.asarray(
            arrays["rollout_component_variant_index"], dtype=np.int64
        ),
        component_window_index=np.asarray(
            arrays["rollout_component_window_index"], dtype=np.int64
        ),
        component_weight=np.asarray(arrays["rollout_component_weight"], dtype=np.float64),
        component_sample_count=np.asarray(
            arrays["rollout_component_sample_count"], dtype=np.int64
        ),
        sample_count=int(sample_count),
    )
    return [{**row, "component_no": int(i)} for i, row in enumerate(rows)]


def _select_sample_indices(
    components: list[dict[str, Any]],
    *,
    posterior_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if posterior_mode == "full":
        selected = [dict(item) for item in components]
        total = sum(max(float(item.get("weight", 0.0)), 0.0) for item in selected)
        for item in selected:
            item["sparse_weight"] = (
                max(float(item.get("weight", 0.0)), 0.0) / total
                if total > 0.0
                else 1.0 / max(len(selected), 1)
            )
    else:
        spec = POSTERIOR_MODES[posterior_mode]
        selected = select_sparse_components(
            components,
            max_components=int(spec["max_components"]),
            min_cumulative_weight=float(spec["min_cumulative_weight"]),
        )
    indices: list[int] = []
    for component in selected:
        start, stop = component["sample_slice"]
        indices.extend(range(int(start), int(stop)))
    return np.asarray(indices, dtype=np.int64), selected


def _score_run_artifact(
    *,
    report_path: Path,
    arrays_path: Path,
    future_delta: np.ndarray,
    posterior_modes: tuple[str, ...],
) -> dict[str, Any]:
    report = _load_json(report_path)
    row = (report.get("generation") or {}).get("window_scores", [{}])[0]
    with np.load(arrays_path, allow_pickle=True) as arrays:
        variant = int((report.get("selected_start_state") or {}).get("variant_index", 0))
        delta_samples = np.asarray(arrays["samples"], dtype=np.float32)[variant]
        selected_windows = np.asarray(arrays["selected_window_indices"], dtype=np.int64)
        scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
        start_idx = int(row.get("start_window_index", int(np.asarray(arrays["start_indices"])[0])))
        source_idx = int(selected_windows[start_idx])
        target = np.asarray(future_delta[source_idx], dtype=np.float32)
        components = _component_rows(
            arrays,
            variant=variant,
            sample_count=int(delta_samples.shape[0]),
        )
        posterior_scores = {}
        for mode in posterior_modes:
            sample_indices, selected = _select_sample_indices(components, posterior_mode=mode)
            samples = delta_samples[sample_indices]
            posterior_scores[mode] = {
                "score": score_sample_distribution(samples, target, scale=scale),
                "component_count": int(len(selected)),
                "sample_count": int(samples.shape[0]),
                "component_window_indices": [
                    int(item["window_index"]) for item in selected
                ],
                "component_weight_sum": float(
                    sum(float(item.get("weight", 0.0)) for item in selected)
                ),
            }
    return {
        "query_window_index": int(row.get("query_window_index", -1)),
        "start_window_index": int(row.get("start_window_index", -1)),
        "source_index": int(source_idx),
        "persistence": row.get("methods", {}).get("persistence"),
        "posterior_scores": posterior_scores,
        "report_path": str(report_path),
        "arrays_path": str(arrays_path),
    }


def _score_backtest_report(
    *,
    backtest_report: str | Path,
    future_delta: np.ndarray,
    posterior_modes: tuple[str, ...],
) -> dict[str, Any]:
    report_path = Path(backtest_report)
    report = _load_json(report_path)
    memory_mode = str(report.get("memory_prior_mode", "unknown"))
    policy_key = POLICY_KEY_BY_MEMORY_MODE.get(memory_mode, memory_mode)
    policy_label = POLICY_LABEL_BY_MEMORY_MODE.get(
        memory_mode,
        PLOT_POLICY_LABELS.get(policy_key, policy_key),
    )
    rows_by_posterior = {mode: [] for mode in posterior_modes}
    run_rows = []
    for row in report.get("window_scores", []):
        artifact = ((row.get("artifacts") or {}).get("component_prefix_mixture") or {})
        component_report = artifact.get("report")
        arrays_path = artifact.get("arrays")
        if not component_report or not arrays_path:
            continue
        scored = _score_run_artifact(
            report_path=Path(component_report),
            arrays_path=Path(arrays_path),
            future_delta=future_delta,
            posterior_modes=posterior_modes,
        )
        run_rows.append(scored)
        for mode in posterior_modes:
            rows_by_posterior[mode].append(
                {
                    "window_index": scored["query_window_index"],
                    "block_window_index": scored["source_index"],
                    "methods": {
                        "persistence": scored["persistence"],
                        f"{policy_key}::{mode}": scored["posterior_scores"][mode][
                            "score"
                        ],
                    },
                }
            )
    posterior_summaries = {}
    for mode, rows in rows_by_posterior.items():
        method_name = f"{policy_key}::{mode}"
        summary = summarize_method_scores(rows, baseline="persistence")
        posterior_summaries[mode] = {
            "method_name": method_name,
            "label": f"{policy_label} / {PLOT_POSTERIOR_LABELS.get(mode, mode)}",
            "window_count": int(len(rows)),
            "summary": summary.get(method_name, {}),
            "persistence": summary.get("persistence", {}),
            "sample_count_mean": float(
                np.mean(
                    [
                        scored["posterior_scores"][mode]["sample_count"]
                        for scored in run_rows
                    ]
                )
            )
            if run_rows
            else None,
            "component_count_mean": float(
                np.mean(
                    [
                        scored["posterior_scores"][mode]["component_count"]
                        for scored in run_rows
                    ]
                )
            )
            if run_rows
            else None,
            "component_weight_sum_mean": float(
                np.mean(
                    [
                        scored["posterior_scores"][mode]["component_weight_sum"]
                        for scored in run_rows
                    ]
                )
            )
            if run_rows
            else None,
        }
    return {
        "backtest_report": str(report_path),
        "policy_key": policy_key,
        "policy_label": policy_label,
        "memory_prior_mode": memory_mode,
        "heldout_window_count": int(len(run_rows)),
        "posterior_summaries": posterior_summaries,
    }


def _conditionality_lookup(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path:
        return {}
    report_path = Path(path)
    if not report_path.exists():
        return {}
    report = _load_json(report_path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for policy in report.get("policies", []):
        policy_key = str(policy.get("policy", ""))
        for posterior in policy.get("posterior_modes", []):
            mode = str(posterior.get("posterior_mode", ""))
            out[(policy_key, mode)] = {
                "pairwise": posterior.get("pairwise", {}),
                "metrics": posterior.get("metrics", {}),
                "case_count": posterior.get("case_count"),
                "sample_count_min": posterior.get("sample_count_min"),
                "component_count_mean": posterior.get("component_count_mean"),
            }
    return out


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    crps_values = [
        float(c["backtest"].get("ensemble_crps_z_improvement_vs_persistence", 0.0))
        for c in candidates
    ]
    energy_values = [
        float(c["backtest"].get("energy_score_z_improvement_vs_persistence", 0.0))
        for c in candidates
    ]
    factor_values = [
        float((c.get("conditionality") or {}).get("metrics", {}).get("mean_factor_terminal_ks", 0.0))
        for c in candidates
    ]
    portfolio_values = [
        float((c.get("conditionality") or {}).get("metrics", {}).get("mean_portfolio_terminal_ks", 0.0))
        for c in candidates
    ]

    def norm(value: float, values: list[float]) -> float:
        lo = min(values)
        hi = max(values)
        if hi - lo <= 1e-12:
            return 0.5
        return (value - lo) / (hi - lo)

    ranked = []
    for candidate in candidates:
        backtest = candidate["backtest"]
        coverage = backtest.get("coverage_80_mean")
        coverage_penalty = (
            min(abs(float(coverage) - 0.80) / 0.20, 1.0)
            if coverage is not None
            else 1.0
        )
        factor_ks = float(
            (candidate.get("conditionality") or {})
            .get("metrics", {})
            .get("mean_factor_terminal_ks", 0.0)
        )
        portfolio_ks = float(
            (candidate.get("conditionality") or {})
            .get("metrics", {})
            .get("mean_portfolio_terminal_ks", 0.0)
        )
        crps = float(backtest.get("ensemble_crps_z_improvement_vs_persistence", 0.0))
        energy = float(backtest.get("energy_score_z_improvement_vs_persistence", 0.0))
        score = (
            0.25 * norm(crps, crps_values)
            + 0.25 * norm(energy, energy_values)
            + 0.20 * norm(factor_ks, factor_values)
            + 0.20 * norm(portfolio_ks, portfolio_values)
            + 0.10 * (1.0 - coverage_penalty)
        )
        ranked.append(
            {
                **candidate,
                "selection_score": float(score),
                "selection_score_note": (
                    "Weighted rank score: CRPS 25%, energy 25%, factor KS 20%, "
                    "portfolio KS 20%, coverage closeness to 0.80 10%."
                ),
            }
        )
    return sorted(ranked, key=lambda item: float(item["selection_score"]), reverse=True)


def build_selection_report(args: argparse.Namespace) -> dict[str, Any]:
    posterior_modes = tuple(args.posterior_mode or ("full", "top2_80", "top3_90"))
    future_delta = _val_future_delta(checkpoint=str(args.checkpoint))
    condition_lookup = _conditionality_lookup(args.conditionality_report)
    support_reports = [
        _score_backtest_report(
            backtest_report=path,
            future_delta=future_delta,
            posterior_modes=posterior_modes,
        )
        for path in args.backtest_report
    ]
    candidates = []
    for support in support_reports:
        for posterior_mode, posterior in support["posterior_summaries"].items():
            conditionality = condition_lookup.get(
                (str(support["policy_key"]), str(posterior_mode)),
                {},
            )
            candidates.append(
                {
                    "policy_key": support["policy_key"],
                    "policy_label": support["policy_label"],
                    "posterior_mode": posterior_mode,
                    "posterior_label": PLOT_POSTERIOR_LABELS.get(
                        posterior_mode,
                        posterior_mode,
                    ),
                    "candidate_label": posterior["label"],
                    "window_count": posterior["window_count"],
                    "sample_count_mean": posterior["sample_count_mean"],
                    "component_count_mean": posterior["component_count_mean"],
                    "component_weight_sum_mean": posterior[
                        "component_weight_sum_mean"
                    ],
                    "backtest": posterior["summary"],
                    "conditionality": conditionality,
                }
            )
    ranked = _rank_candidates(candidates)
    return {
        "status": "ok",
        "scope_note": (
            "Artifact-only production ensemble selection. Backtest metrics are "
            "rescored from saved component-prefix generated samples; fixed-start "
            "conditionality metrics are read from the saved component-posterior "
            "bakeoff."
        ),
        "checkpoint": str(args.checkpoint),
        "backtest_reports": list(args.backtest_report),
        "conditionality_report": str(args.conditionality_report),
        "posterior_modes": list(posterior_modes),
        "support_reports": support_reports,
        "ranked_candidates": ranked,
        "recommended_candidate": ranked[0] if ranked else None,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Narrative Ensemble Selection",
        "",
        report["scope_note"],
        "",
        "| Rank | Candidate | Windows | Samples | Components | CRPS impr. | Energy impr. | Cov80 | Factor KS | Portfolio KS | Score |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, candidate in enumerate(report["ranked_candidates"], start=1):
        backtest = candidate["backtest"]
        conditionality = candidate.get("conditionality") or {}
        metrics = conditionality.get("metrics", {})
        lines.append(
            "| {rank} | {label} | {windows} | {samples:.1f} | {components:.1f} | "
            "{crps:.3f} | {energy:.3f} | {cov:.3f} | {factor:.3f} | "
            "{portfolio:.3f} | {score:.3f} |".format(
                rank=rank,
                label=candidate["candidate_label"],
                windows=int(candidate["window_count"]),
                samples=float(candidate["sample_count_mean"] or 0.0),
                components=float(candidate["component_count_mean"] or 0.0),
                crps=float(
                    backtest.get("ensemble_crps_z_improvement_vs_persistence", float("nan"))
                ),
                energy=float(
                    backtest.get("energy_score_z_improvement_vs_persistence", float("nan"))
                ),
                cov=float(backtest.get("coverage_80_mean", float("nan"))),
                factor=float(metrics.get("mean_factor_terminal_ks", float("nan"))),
                portfolio=float(metrics.get("mean_portfolio_terminal_ks", float("nan"))),
                score=float(candidate["selection_score"]),
            )
        )
    best = report.get("recommended_candidate") or {}
    if best:
        lines.extend(
            [
                "",
                "## Recommended Candidate",
                "",
                f"**{best['candidate_label']}**",
                "",
                best["selection_score_note"],
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-report", action="append", default=[])
    parser.add_argument(
        "--conditionality-report",
        default=DEFAULT_CONDITIONALITY_REPORT,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--posterior-mode",
        action="append",
        choices=tuple(POSTERIOR_MODES),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.backtest_report:
        args.backtest_report = list(DEFAULT_BACKTEST_REPORTS)
    output_dir = Path(args.output_dir)
    report = build_selection_report(args)
    report["artifact_paths"] = {
        "json": str(output_dir / "posterior_ensemble_selection_report.json"),
        "markdown": str(output_dir / "posterior_ensemble_selection_report.md"),
    }
    _write_json(report["artifact_paths"]["json"], report)
    _write_text(report["artifact_paths"]["markdown"], _render_markdown(report))
    compact = {
        "status": report["status"],
        "recommended_candidate": (
            None
            if report.get("recommended_candidate") is None
            else {
                "candidate_label": report["recommended_candidate"]["candidate_label"],
                "selection_score": report["recommended_candidate"][
                    "selection_score"
                ],
                "backtest": report["recommended_candidate"]["backtest"],
                "conditionality_metrics": (
                    report["recommended_candidate"].get("conditionality") or {}
                ).get("metrics", {}),
            }
        ),
        "artifact_paths": report["artifact_paths"],
    }
    print(json.dumps(_jsonable(compact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
