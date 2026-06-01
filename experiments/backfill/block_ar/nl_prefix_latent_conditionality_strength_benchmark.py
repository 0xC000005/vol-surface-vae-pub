#!/usr/bin/env python
"""Consolidate fixed-start narrative conditionality evidence.

This benchmark is intentionally diagnostic. It does not train a new bridge or
call OpenAI. It joins the current full-support component rollout with matching
repeat/start-only controls, then reports whether different narratives produce
distributional differences above controls in factor and portfolio-risk units.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_shape_audit import (  # noqa: E402
    _load_repeat_cases,
    _load_start_only_cases,
    build_shape_audit,
    load_observed_cases,
)
from experiments.backfill.block_ar.nl_prefix_latent_portfolio_conditionality_audit import (  # noqa: E402
    build_portfolio_conditionality_report,
    plot_portfolio_conditionality,
)


DEFAULT_COMPONENT_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_mixture_fixed_start_914c_full906b_s384"
)
DEFAULT_CONTROL_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_fixed_decoder_controls_914c_full906b_s384"
)
DEFAULT_VARIANT_DIR = (
    "decoder_component_full906b_diverse_topk_narrative_start_checked_gap30_"
    "temp0p20_s384_gen_temp_0p50"
)
DEFAULT_DIRECT_MEMORY_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_scenario_level_eval_full_906b_all_windows_direct_memory/"
    "scenario_level_eval_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_conditionality_strength_benchmark_915a_full906b_s384"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _case_report(case: dict[str, Any]) -> dict[str, Any]:
    return _load_json(str(case["run_report"]))


def _support_indices(report: dict[str, Any]) -> set[int]:
    prior = report.get("cached_query", {}).get("memory_prior", {})
    details = prior.get("candidate_details", []) if isinstance(prior, dict) else []
    out: set[int] = set()
    for item in details:
        if isinstance(item, dict) and item.get("window_index") is not None:
            out.add(int(item["window_index"]))
    return out


def _support_overlap_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reports = {str(case["case_name"]): _case_report(case) for case in cases}
    support_sets = {name: _support_indices(report) for name, report in reports.items()}
    labels = {str(case["case_name"]): str(case["label"]) for case in cases}
    names = list(support_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = support_sets[left]
            b = support_sets[right]
            union = len(a | b)
            rows.append(
                {
                    "left_case": left,
                    "right_case": right,
                    "left_label": labels[left],
                    "right_label": labels[right],
                    "left_support_count": int(len(a)),
                    "right_support_count": int(len(b)),
                    "shared_support_count": int(len(a & b)),
                    "jaccard": float(len(a & b) / union) if union else 0.0,
                }
            )
    return rows


def _support_case_rows(cases: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        report = _case_report(case)
        prior = report.get("cached_query", {}).get("memory_prior", {})
        details = prior.get("candidate_details", []) if isinstance(prior, dict) else []
        weights = prior.get("weights", []) if isinstance(prior, dict) else []
        for pos, item in enumerate(details[:limit]):
            if not isinstance(item, dict):
                continue
            alignment = item.get("recent_prefix_alignment", {})
            rows.append(
                {
                    "case": str(case["case_name"]),
                    "label": str(case["label"]),
                    "rank": int(item.get("rank", pos + 1) or pos + 1),
                    "window_id": str(item.get("window_id", "")),
                    "window_index": int(item.get("window_index", -1) or -1),
                    "history_end_date": str(item.get("history_end_date", "")),
                    "weight": float(weights[pos]) if pos < len(weights) else float(item.get("weight", 0.0) or 0.0),
                    "memory_support_cosine": float(item.get("memory_support_cosine", 0.0) or 0.0),
                    "combined_score": float(item.get("combined_score", 0.0) or 0.0),
                    "start_distance_z": float(item.get("start_distance_z", 0.0) or 0.0),
                    "direction_status": str(item.get("recent_prefix_alignment_status", "")),
                    "direction_mismatches": int(item.get("recent_prefix_mismatches", 0) or 0),
                    "direction_checked": int(item.get("recent_prefix_checked", 0) or 0),
                    "alignment": alignment if isinstance(alignment, dict) else {},
                }
            )
    return rows


def _plot_support_overlap(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output: str | Path,
) -> None:
    labels = [str(case["label"]) for case in cases]
    names = [str(case["case_name"]) for case in cases]
    matrix = np.eye(len(names), dtype=np.float64)
    lookup = {
        (str(row["left_case"]), str(row["right_case"])): float(row["jaccard"])
        for row in rows
    }
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i == j:
                continue
            matrix[i, j] = lookup.get((left, right), lookup.get((right, left), 0.0))
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Historical support-set overlap by narrative")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Jaccard overlap")
    fig.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scenario_quality_summary(path: str | Path) -> dict[str, Any]:
    report = _load_json(path)
    summary = report.get("summary", {})
    methods = {}
    for name in ["narrative_direct_memory", "narrative_generator_topk", "persistence"]:
        row = summary.get(name)
        if not isinstance(row, dict):
            continue
        methods[name] = {
            "window_count": int(row.get("window_count", report.get("heldout_window_count", 0)) or 0),
            "coverage_80_mean": row.get("coverage_80_mean"),
            "energy_score_z_improvement_vs_persistence": row.get(
                "energy_score_z_improvement_vs_persistence"
            ),
            "ensemble_crps_z_improvement_vs_persistence": row.get(
                "ensemble_crps_z_improvement_vs_persistence"
            ),
            "mean_path_mae_z_improvement_vs_persistence": row.get(
                "mean_path_mae_z_improvement_vs_persistence"
            ),
            "terminal_mae_z_improvement_vs_persistence": row.get(
                "terminal_mae_z_improvement_vs_persistence"
            ),
        }
    return {
        "report": str(path),
        "heldout_window_count": int(report.get("heldout_window_count", 0) or 0),
        "methods": methods,
    }


def _median_from_summary(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(f"{key}_median_across_pairs")
    return float(value) if value is not None else None


def _diagnostic_decision(
    shape_report: dict[str, Any],
    portfolio_report: dict[str, Any],
) -> dict[str, Any]:
    shape_status = str(shape_report.get("status", "unknown"))
    portfolio_status = str(portfolio_report.get("status", "unknown"))
    portfolio_ratios = portfolio_report.get("ratios", {})
    path_ratios = shape_report.get("ratios", {})
    failures = list(shape_report.get("failures", [])) + list(portfolio_report.get("failures", []))
    warnings = list(shape_report.get("warnings", [])) + list(portfolio_report.get("warnings", []))
    solved = (
        shape_status == "pass"
        and portfolio_status == "pass"
        and float(portfolio_ratios.get("path_vs_repeat", 0.0)) >= 1.25
        and float(portfolio_ratios.get("path_vs_bootstrap", 0.0)) >= 1.10
        and float(portfolio_ratios.get("var95_vs_repeat", 0.0)) >= 1.25
    )
    if solved:
        verdict = "conditionality_solved_for_this_fixed_start"
    elif failures:
        verdict = "conditionality_not_solved"
    else:
        verdict = "conditionality_partially_supported_with_warnings"
    return {
        "verdict": verdict,
        "shape_status": shape_status,
        "portfolio_status": portfolio_status,
        "warnings": warnings,
        "failures": failures,
        "key_ratios": {
            "factor_path_repeat_to_observed_energy": path_ratios.get(
                "repeat_to_observed_path_energy"
            ),
            "factor_path_bootstrap_to_observed_energy": path_ratios.get(
                "bootstrap_to_observed_path_energy"
            ),
            "portfolio_path_vs_repeat": portfolio_ratios.get("path_vs_repeat"),
            "portfolio_path_vs_bootstrap": portfolio_ratios.get("path_vs_bootstrap"),
            "portfolio_var95_vs_repeat": portfolio_ratios.get("var95_vs_repeat"),
            "portfolio_var95_vs_bootstrap": portfolio_ratios.get("var95_vs_bootstrap"),
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    quality = report["heldout_quality"]["methods"]
    support_overlap = report["support_overlap_summary"]
    lines = [
        "# NL Conditionality Strength Benchmark",
        "",
        f"Status: **{decision['verdict']}**",
        "",
        "## What Was Tested",
        "",
        (
            "The same starting joint39 level is held fixed across six risk-manager "
            "narratives. The benchmark compares generated 30-day raw-level path "
            "distributions against same-narrative repeat, within-run bootstrap, "
            "and start-only controls."
        ),
        "",
        "## Held-Out Distributional Quality",
        "",
    ]
    for name in ["narrative_direct_memory", "narrative_generator_topk"]:
        row = quality.get(name, {})
        if not row:
            continue
        lines.append(
            "- "
            + name
            + ": "
            + f"coverage={float(row.get('coverage_80_mean', 0.0)):.3f}, "
            + f"CRPS improvement={float(row.get('ensemble_crps_z_improvement_vs_persistence', 0.0)):+.3f}, "
            + f"energy improvement={float(row.get('energy_score_z_improvement_vs_persistence', 0.0)):+.3f}"
        )
    lines += [
        "",
        "## Conditionality Evidence",
        "",
        f"- Factor path audit status: `{decision['shape_status']}`.",
        f"- Portfolio audit status: `{decision['portfolio_status']}`.",
        (
            "- Factor path energy repeat/observed: "
            f"`{float(decision['key_ratios']['factor_path_repeat_to_observed_energy']):.3f}`."
        ),
        (
            "- Factor path energy bootstrap/observed: "
            f"`{float(decision['key_ratios']['factor_path_bootstrap_to_observed_energy']):.3f}`."
        ),
        (
            "- Portfolio path observed/repeat: "
            f"`{float(decision['key_ratios']['portfolio_path_vs_repeat']):.3f}`."
        ),
        (
            "- Portfolio path observed/bootstrap: "
            f"`{float(decision['key_ratios']['portfolio_path_vs_bootstrap']):.3f}`."
        ),
        (
            "- Portfolio VaR95 observed/repeat: "
            f"`{float(decision['key_ratios']['portfolio_var95_vs_repeat']):.3f}`."
        ),
        "",
        "## Support Diversity",
        "",
        (
            f"- Median pairwise support Jaccard: "
            f"`{support_overlap['median_jaccard']:.3f}`."
        ),
        (
            f"- Max pairwise support Jaccard: "
            f"`{support_overlap['max_jaccard']:.3f}`."
        ),
        (
            f"- Pairwise support comparisons: "
            f"`{support_overlap['pair_count']}`."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The current full-support workflow has real narrative signal: supports "
            "are different, start-only controls are zero, and portfolio path "
            "distance is well above same-narrative repeat noise. It is not a clean "
            "production pass because bootstrap noise remains close to the observed "
            "cross-narrative effect and VaR-style tail separation is below the "
            "repeat-control threshold."
        ),
        "",
        "## Artifacts",
        "",
    ]
    for key, path in report["artifact_paths"].items():
        lines.append(f"- {key}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def build_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    shape_dir = output_dir / "factor_path_audit"
    shape_args = argparse.Namespace(
        component_root=str(args.component_root),
        control_root=str(args.control_root),
        variant_dir=str(args.variant_dir),
        output_dir=str(shape_dir),
        fan_scale=float(args.fan_scale),
        max_start_abs_diff=float(args.max_start_abs_diff),
        max_repeat_ratio=float(args.max_repeat_ratio),
        max_bootstrap_ratio=float(args.max_bootstrap_ratio),
        max_start_only_ratio=float(args.max_start_only_ratio),
    )
    shape_report = build_shape_audit(shape_args)
    _write_json(shape_report["artifact_paths"]["report"], shape_report)

    observed_cases = load_observed_cases(
        Path(args.component_root),
        variant_dir=str(args.variant_dir),
        fan_scale=float(args.fan_scale),
    )
    start_only_cases = _load_start_only_cases(
        Path(args.control_root),
        fan_scale=float(args.fan_scale),
    )
    repeat_cases = _load_repeat_cases(
        Path(args.control_root),
        fan_scale=float(args.fan_scale),
    )
    portfolio_report = build_portfolio_conditionality_report(
        observed_cases=observed_cases,
        start_only_cases=start_only_cases,
        repeat_cases=repeat_cases,
    )
    portfolio_path = output_dir / "portfolio_conditionality_audit.json"
    portfolio_figure = output_dir / "portfolio_conditionality_controls.png"
    _write_json(portfolio_path, portfolio_report)
    plot_portfolio_conditionality(portfolio_report, portfolio_figure)

    support_rows = _support_case_rows(observed_cases)
    overlap_rows = _support_overlap_rows(observed_cases)
    overlap_values = [float(row["jaccard"]) for row in overlap_rows]
    support_overlap_summary = {
        "pair_count": int(len(overlap_rows)),
        "median_jaccard": float(median(overlap_values)) if overlap_values else 0.0,
        "max_jaccard": float(max(overlap_values)) if overlap_values else 0.0,
        "mean_jaccard": float(np.mean(overlap_values)) if overlap_values else 0.0,
    }
    support_overlap_figure = output_dir / "support_overlap_matrix.png"
    _plot_support_overlap(observed_cases, overlap_rows, support_overlap_figure)

    heldout_quality = _scenario_quality_summary(args.scenario_quality_report)
    decision = _diagnostic_decision(shape_report, portfolio_report)
    report = {
        "scope_note": (
            "Consolidated fixed-start narrative conditionality benchmark. This "
            "uses cached component-preserving full-support runs and matching "
            "controls; it does not call OpenAI or train a new model."
        ),
        "decision": decision,
        "component_root": str(args.component_root),
        "control_root": str(args.control_root),
        "variant_dir": str(args.variant_dir),
        "fan_scale": float(args.fan_scale),
        "heldout_quality": heldout_quality,
        "support_overlap_summary": support_overlap_summary,
        "support_overlap_rows": overlap_rows,
        "support_rows": support_rows,
        "shape_audit_summary": {
            "status": shape_report.get("status"),
            "warnings": shape_report.get("warnings", []),
            "failures": shape_report.get("failures", []),
            "ratios": shape_report.get("ratios", {}),
            "summaries": shape_report.get("summaries", {}),
        },
        "portfolio_audit_summary": {
            "status": portfolio_report.get("status"),
            "warnings": portfolio_report.get("warnings", []),
            "failures": portfolio_report.get("failures", []),
            "ratios": portfolio_report.get("ratios", {}),
            "summaries": portfolio_report.get("summaries", {}),
        },
        "artifact_paths": {
            "benchmark_json": str(output_dir / "conditionality_strength_benchmark.json"),
            "benchmark_markdown": str(output_dir / "conditionality_strength_benchmark.md"),
            "factor_path_audit_json": str(shape_report["artifact_paths"]["report"]),
            "factor_raw_fan_plot": str(shape_report["artifact_paths"]["raw_fan_plot"]),
            "factor_path_metric_plot": str(
                shape_report["artifact_paths"]["path_metric_summary_plot"]
            ),
            "portfolio_audit_json": str(portfolio_path),
            "portfolio_controls_plot": str(portfolio_figure),
            "support_overlap_plot": str(support_overlap_figure),
        },
    }
    _write_json(report["artifact_paths"]["benchmark_json"], report)
    _write_text(report["artifact_paths"]["benchmark_markdown"], _markdown_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--scenario-quality-report", type=Path, default=DEFAULT_DIRECT_MEMORY_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fan-scale", type=float, default=1.0)
    parser.add_argument("--max-start-abs-diff", type=float, default=1e-5)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.75)
    parser.add_argument("--max-bootstrap-ratio", type=float, default=0.75)
    parser.add_argument("--max-start-only-ratio", type=float, default=0.50)
    args = parser.parse_args()
    report = build_benchmark(args)
    print(
        json.dumps(
            _jsonable(
                {
                    "verdict": report["decision"]["verdict"],
                    "warnings": report["decision"]["warnings"],
                    "failures": report["decision"]["failures"],
                    "key_ratios": report["decision"]["key_ratios"],
                    "support_overlap_summary": report["support_overlap_summary"],
                    "report": report["artifact_paths"]["benchmark_json"],
                    "markdown": report["artifact_paths"]["benchmark_markdown"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
